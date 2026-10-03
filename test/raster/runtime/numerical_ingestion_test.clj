(ns raster.runtime.numerical-ingestion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.numerical-state :as state]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign Arena MemorySegment]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- fixture [submit! release! & [placement!]]
  (let [description (content/provider-description
                     {:id :ingestion :capabilities #{:ingest}
                      :tiers [(content/storage-tier {:id :local :kind :memory :locality :node
                                                     :durability :cached :capabilities #{:scoped-segment}})]})
        event (content/storage-event {:provider-id :ingestion :id :write :operation :ingest})
        saved (atom nil)
        submitted-address (atom nil)
        submissions (atom 0)
        provider (reify content/ContentProvider
                   (-provider-descriptor [_] description)
                   (-release-storage-event! [_ event] (release! event))
                   (-await-storage-event! [_ _]
                     ((or placement! identity)
                      (content/content-placement {:provider-id :ingestion :tier-id :local
                                                  :content @submitted-address})))
                   content/ContentIngestor
                   (-submit-ingestion! [_ address tier n read! opts]
                     (swap! submissions inc)
                     (reset! saved read!)
                     (reset! submitted-address address)
                     (submit! {:address address :tier tier :n n :read! read! :opts opts :event event})))]
    {:provider provider :saved saved :submissions submissions :event event}))

(defn- reader [^MemorySegment source calls]
  (fn [offset ^MemorySegment destination]
    (swap! calls conj [offset (.byteSize destination)])
    (MemorySegment/copy source offset destination 0 (.byteSize destination))
    (.byteSize destination)))

(deftest bounded-ingestion-consumes-once-and-expires-source-borrow
  (doseq [n [0 1 65536 131079]]
    (let [bytes (byte-array (map #(unchecked-byte (mod % 251)) (range n)))
          source (MemorySegment/ofArray bytes)
          address (content/content-address-of source)
          calls (atom []) released (atom [])
          stored (byte-array n)
          {:keys [provider saved event]}
          (fixture (fn [{:keys [n read! event tier opts]}]
                     (is (= :local tier)) (is (= {:tag :test} opts))
                     (loop [offset 0]
                       (when (< offset n)
                         (let [width (min 65536 (- n offset))
                               destination (.asSlice (MemorySegment/ofArray stored) offset width)]
                           (is (= width (read! offset destination)))
                           (recur (+ offset width)))))
                     event)
                   #(swap! released conj %))]
      (is (identical? event (content/submit-ingestion! provider address :local n
                                                       (reader source calls) {:tag :test})))
      (is (= (seq bytes) (seq stored)))
      (is (= n (reduce + 0 (map second @calls))))
      (is (every? #(<= 1 (second %) 65536) @calls))
      (let [before @calls]
        (is (= :numerical-content-ingestion-expired
               (:reason (ex-data (error-of #(@saved 0 (MemorySegment/ofArray (byte-array 1))))))))
        (is (= before @calls)))
      (is (empty? @released))
      (content/release-storage-event! provider event)
      (is (= [event] @released)))))

(deftest invalid-ingestion-preflight-never-submits-or-reads
  (let [source (MemorySegment/ofArray (byte-array 1))
        address (content/content-address-of source)
        calls (atom [])
        {:keys [provider submissions]} (fixture (fn [_] (throw (ex-info "unexpected submit" {}))) identity)]
    (doseq [[a tier n opts] [[address :missing 1 {}] [address :local -1 {}]
                             [address :local (inc' Long/MAX_VALUE) {}]
                             [address :local 1 nil]
                             [(state/content-address :other "opaque") :local 1 {}]]]
      (is (some? (error-of #(content/submit-ingestion! provider a tier n (reader source calls) opts)))))
    (is (zero? @submissions))
    (is (empty? @calls))))

(deftest rejected-ingestion-handoffs-drain-and-preserve-reader-failure
  (doseq [mode [:incomplete :digest :foreign-event :wrong-operation :short-read :swallowed-error]]
    (let [source (MemorySegment/ofArray (byte-array [1 2 3 4]))
          address (content/content-address-of source)
          primary (ex-info "reader failed" {})
          drain (ex-info "drain uncertain" {})
          releases (atom []) calls (atom [])
          {:keys [provider saved event]}
          (fixture (fn [{:keys [read! event]}]
                     (let [destination (MemorySegment/ofArray (byte-array 4))]
                       (when-not (= mode :incomplete)
                         (if (contains? #{:swallowed-error :short-read :digest} mode)
                           (try (read! 0 destination) (catch Throwable _))
                           (read! 0 destination)))
                       (case mode
                         :foreign-event (assoc event :provider-id :foreign)
                         :wrong-operation (assoc event :operation :promote)
                         event)))
                   #(do (swap! releases conj %) (throw drain)))
          read! (case mode
                  :short-read (fn [_ _] 3)
                  :swallowed-error (fn [_ _] (throw primary))
                  (reader source calls))
          declared (if (= mode :digest)
                     (content/content-address-of (MemorySegment/ofArray (byte-array [4 3 2 1]))) address)
          failure (error-of #(content/submit-ingestion! provider declared :local 4 read!))]
      (is (some? failure))
      (when (= mode :swallowed-error) (is (identical? primary failure)))
      (is (= 1 (count @releases)))
      (is (identical? drain (first (.getSuppressed ^Throwable failure))))
      (is (= :numerical-content-ingestion-expired
             (:reason (ex-data (error-of #(@saved 0 (MemorySegment/ofArray (byte-array 4)))))))))))

(deftest source-submission-failure-expires-reader-without-guessed-event-release
  (let [source (MemorySegment/ofArray (byte-array 1))
        primary (ex-info "submission failed before handoff" {}) releases (atom [])
        {:keys [provider saved]} (fixture (fn [_] (throw primary)) #(swap! releases conj %))]
    (is (identical? primary
                    (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                          :local 1 (reader source (atom []))))))
    (is (empty? @releases))
    (is (= :numerical-content-ingestion-expired
           (:reason (ex-data (error-of #(@saved 0 (MemorySegment/ofArray (byte-array 1))))))))))

(deftest invalid-provider-windows-cannot-touch-source
  (doseq [mode [:readonly :oversized :empty :out-of-range :unordered :cross-thread :closed]]
    (with-open [arena (Arena/ofShared)]
      (let [source (MemorySegment/ofArray (byte-array 4)) calls (atom []) releases (atom [])
            {:keys [provider]}
            (fixture (fn [{:keys [read! event]}]
                       (let [destination (case mode
                                           :readonly (.asReadOnly (MemorySegment/ofArray (byte-array 4)))
                                           :oversized (MemorySegment/ofArray (byte-array 65537))
                                           :empty (MemorySegment/ofArray (byte-array 0))
                                           :out-of-range (MemorySegment/ofArray (byte-array 5))
                                           :closed (let [a (Arena/ofShared) s (.allocate a 4)] (.close a) s)
                                           (.allocate arena 4))]
                         (if (= mode :cross-thread)
                           @(future (error-of #(read! 0 destination)))
                           (error-of #(read! (if (= mode :unordered) 1 0) destination))))
                       event)
                     #(swap! releases conj %))]
        (is (some? (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                         :local 4 (reader source calls)))))
        (is (empty? @calls))
        (is (= 1 (count @releases)))))))

(deftest callback-failure-cannot-be-caught-and-retried-into-success
  (let [source (MemorySegment/ofArray (byte-array [1 2 3 4]))
        calls (atom []) releases (atom []) observed (atom [])
        {:keys [provider saved]}
        (fixture (fn [{:keys [read! event]}]
                   (let [destination (MemorySegment/ofArray (byte-array 4))]
                     (swap! observed conj (error-of #(read! 1 destination)))
                     (swap! observed conj (error-of #(read! 0 destination))))
                   event)
                 #(swap! releases conj %))
        error (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                    :local 4 (reader source calls)))]
    (is (identical? error (first @observed)))
    (is (identical? error (second @observed)))
    (is (empty? @calls))
    (is (= 1 (count @releases)))))

(deftest final-window-digest-failure-precedes-provider-commit
  (let [source (MemorySegment/ofArray (byte-array [1 2]))
        wrong (content/content-address-of (MemorySegment/ofArray (byte-array [2 1])))
        committed (atom false) releases (atom [])
        {:keys [provider]}
        (fixture (fn [{:keys [read! event]}]
                   (read! 0 (MemorySegment/ofArray (byte-array 2)))
                   (reset! committed true)
                   event)
                 #(swap! releases conj %))]
    (is (= :numerical-content-ingestion-digest
           (:reason (ex-data (error-of #(content/submit-ingestion! provider wrong :local 2
                                                                   (reader source (atom []))))))))
    (is (false? @committed))
    (is (empty? @releases) "no event was handed off; submission cleanup remains provider-owned")))

(deftest source-can-close-before-provider-placement-await
  (let [arena (Arena/ofShared) source (.allocate arena 4)
        address (content/content-address-of source) releases (atom [])
        {:keys [provider saved]}
        (fixture (fn [{:keys [read! event]}]
                   (read! 0 (MemorySegment/ofArray (byte-array 4))) event)
                 #(swap! releases conj %))]
    (try
      (let [event (content/submit-ingestion! provider address :local 4 (reader source (atom [])))]
        (.close arena)
        (is (= address (:content (content/await-storage-event! provider event))))
        (content/release-storage-event! provider event)
        (is (= :numerical-content-ingestion-expired
               (:reason (ex-data (error-of #(@saved 0 (MemorySegment/ofArray (byte-array 4))))))))
        (is (= 1 (count @releases))))
      (finally (when (.isAlive (.scope source)) (.close arena))))))

(deftest synchronous-ingestion-checks-exact-placement-and-always-drains
  (doseq [mutation [identity #(assoc % :provider-id :foreign)
                    #(assoc % :tier-id :other)
                    #(assoc % :content (state/content-address :sha-256 (apply str (repeat 64 "f"))))]]
    (let [source (MemorySegment/ofArray (byte-array [1]))
          address (content/content-address-of source) releases (atom [])
          {:keys [provider]}
          (fixture (fn [{:keys [read! event]}]
                     (read! 0 (MemorySegment/ofArray (byte-array 1))) event)
                   #(swap! releases conj %) mutation)]
      (let [result (try (content/ingest-content! provider address :local 1 (reader source (atom [])))
                        (catch Throwable error error))]
        (if (identical? mutation identity)
          (is (= address (:content result)))
          (is (instance? clojure.lang.ExceptionInfo result))))
      (is (= 1 (count @releases))))))

(deftest optional-ingestion-does-not-change-non-ingesting-provider-contract
  (doseq [capabilities [#{} #{:ingest}]]
    (let [description (content/provider-description
                       {:id :legacy :capabilities capabilities
                        :tiers [(content/storage-tier {:id :local :kind :memory :locality :node
                                                       :durability :cached})]})
          provider (reify content/ContentProvider (-provider-descriptor [_] description))
          source (MemorySegment/ofArray (byte-array 1)) calls (atom [])
          failure (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                        :local 1 (reader source calls)))]
      (is (= description (content/provider-descriptor provider)))
      (is (= (if (empty? capabilities) :numerical-content-provider-capability
                 :numerical-content-ingestion-provider) (:reason (ex-data failure))))
      (is (empty? @calls)))))

(deftest partial-submission-failure-ends-source-access-without-guessing-event-ownership
  (let [source (MemorySegment/ofArray (byte-array [1 2]))
        primary (ex-info "provider failed after prefix" {}) releases (atom []) calls (atom [])
        {:keys [provider saved]}
        (fixture (fn [{:keys [read!]}]
                   (read! 0 (MemorySegment/ofArray (byte-array 1)))
                   (throw primary)) #(swap! releases conj %))]
    (is (identical? primary
                    (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                          :local 2 (reader source calls)))))
    (is (= [[0 1]] @calls))
    (is (empty? @releases))
    (is (= :numerical-content-ingestion-expired
           (:reason (ex-data (error-of #(@saved 1 (MemorySegment/ofArray (byte-array 1))))))))))

(deftest caught-reentrant-reader-fault-poisons-outer-read-before-success
  (let [source (MemorySegment/ofArray (byte-array 1)) releases (atom [])
        observed (atom nil) callback (atom nil)
        {:keys [provider]}
        (fixture (fn [{:keys [read! event]}]
                   (reset! callback read!)
                   (reset! observed (error-of #(read! 0 (MemorySegment/ofArray (byte-array 1)))))
                   event) #(swap! releases conj %))
        read! (fn [offset destination]
                (error-of #(@callback offset destination))
                (MemorySegment/copy source offset destination 0 1)
                1)
        failure (error-of #(content/submit-ingestion! provider (content/content-address-of source)
                                                      :local 1 read!))]
    (is (= :numerical-content-ingestion-reentrant (:reason (ex-data failure))))
    (is (identical? failure @observed))
    (is (= 1 (count @releases)))))

(deftest provider-submission-error-cannot-replace-caught-source-or-digest-fault
  (doseq [mode [:source :digest]]
    (let [source (MemorySegment/ofArray (byte-array [1]))
          primary (ex-info "source fault retains authority" {})
          secondary (ex-info "provider submission also failed" {})
          releases (atom []) observed (atom nil) calls (atom [])
          {:keys [provider saved]}
          (fixture (fn [{:keys [read!]}]
                     (reset! observed (error-of #(read! 0 (MemorySegment/ofArray (byte-array 1)))))
                     (throw secondary)) #(swap! releases conj %))
          address (if (= mode :digest)
                    (content/content-address-of (MemorySegment/ofArray (byte-array [2])))
                    (content/content-address-of source))
          read! (if (= mode :source)
                  (fn [_ _] (swap! calls conj :read) (throw primary))
                  (reader source calls))
          failure (error-of #(content/submit-ingestion! provider address :local 1 read!))]
      (is (identical? @observed failure))
      (when (= mode :source) (is (identical? primary failure)))
      (when (= mode :digest) (is (= :numerical-content-ingestion-digest (:reason (ex-data failure)))))
      (is (= [secondary] (vec (.getSuppressed ^Throwable failure))))
      (is (empty? @releases))
      (let [before @calls]
        (is (= :numerical-content-ingestion-expired
               (:reason (ex-data (error-of #(@saved 0 (MemorySegment/ofArray (byte-array 1))))))))
        (is (= before @calls))))))

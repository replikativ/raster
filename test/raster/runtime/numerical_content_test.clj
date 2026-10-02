(ns raster.runtime.numerical-content-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as buffer-view]
            [raster.compiler.ir.numerical-state :as numerical-state]
            [raster.gpu.core :as gpu]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.security MessageDigest]
           [java.util HexFormat]))

(defn- address
  [n]
  (numerical-state/content-address :sha-256 (format "%064x" n)))

(defn- fake-provider
  [provider-id]
  (let [events (atom {})
        releases (atom 0)
        local-tier (content/storage-tier
                    {:id :local
                     :kind :file
                     :locality :node
                     :durability :cached
                     :capabilities #{:scoped-segment}})
        durable-tier (content/storage-tier
                      {:id :durable
                       :kind :object-store
                       :locality :site
                       :durability :durable
                       :capabilities #{:durable-receipt :range-read :multipart-write}})
        description (content/provider-description
                     {:id provider-id
                      :tiers [local-tier durable-tier]
                      :capabilities #{:promote :localize :scoped-segment}
                      :attributes {:implementation :fake}})
        submit (fn [operation value]
                 (let [event (content/storage-event
                              {:provider-id provider-id
                               :id (random-uuid)
                               :operation operation})]
                   (swap! events assoc (:id event)
                          {:status :pending
                           :value value
                           :measurement {:timing-source :fake
                                         :bytes 16
                                         :elapsed-ns 20}})
                   event))
        provider
        (reify content/ContentProvider
          (-provider-descriptor [_] description)
          (-submit-promotion! [_ chunk target-tier _opts]
            (submit :promote
                    (content/content-placement
                     {:provider-id provider-id :tier-id target-tier :content chunk})))
          (-submit-localization! [_ chunk opts]
            (submit :localize
                    (content/content-placement
                     {:provider-id provider-id :tier-id (or (:tier opts) :local)
                      :content chunk})))
          (-open-local-content! [_ chunk _opts]
            (let [bytes (byte-array (map byte (range 16)))
                  placement (content/content-placement
                             {:provider-id provider-id :tier-id :local :content chunk})]
              (content/local-content-lease
               {:content chunk
                :placement placement
                :segment (MemorySegment/ofArray bytes)
                :byte-offset 4
                :byte-length 8
                :release-fn #(swap! releases inc)})))
          (-storage-event-complete? [_ event]
            (= :complete (get-in @events [(:id event) :status])))
          (-await-storage-event! [_ event]
            (swap! events assoc-in [(:id event) :status] :complete)
            (get-in @events [(:id event) :value]))
          (-storage-event-measurement [_ event]
            (when (= :complete (get-in @events [(:id event) :status]))
              (get-in @events [(:id event) :measurement])))
          (-release-storage-event! [_ event]
            (swap! events dissoc (:id event))))]
    {:provider provider :events events :releases releases :description description}))

(deftest provider-capabilities-govern-promotion-and-localization
  (let [{:keys [provider events description]} (fake-provider :tiered-store)
        chunk (address 1)
        promotion (content/submit-promotion! provider chunk :durable)
        localization (content/submit-localization! provider chunk {:tier :local})]
    (is (= description (content/provider-descriptor provider)))
    (is (= :storage (get-in promotion [:queue :class])))
    (is (= :promote (:operation promotion)))
    (is (= :localize (:operation localization)))
    (is (false? (content/storage-event-complete? provider promotion)))
    (is (= :durable (:tier-id (content/await-storage-event! provider promotion))))
    (is (content/storage-event-complete? provider promotion))
    (is (= {:timing-source :fake :bytes 16 :elapsed-ns 20}
           (content/storage-event-measurement provider promotion)))
    (is (= :local (:tier-id (content/await-storage-event! provider localization))))
    (content/release-storage-event! provider promotion)
    (content/release-storage-event! provider localization)
    (is (empty? @events))))

(deftest events-and-target-tiers-cannot-cross-provider-boundaries
  (let [{left :provider} (fake-provider :left)
        {right :provider} (fake-provider :right)
        event (content/submit-localization! left (address 2))]
    (is (= :numerical-content-event-provider-mismatch
           (:reason
            (try
              (content/await-storage-event! right event)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))
    (is (= :numerical-content-provider-tier
           (:reason
            (try
              (content/submit-promotion! left (address 2) :unknown)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))
    (content/release-storage-event! left event)))

(deftest local-content-is-scoped-and-range-limited
  (let [{:keys [provider releases]} (fake-provider :local-provider)
        chunk (address 3)
        observed
        (content/with-local-content
          provider chunk
          (fn [lease]
            (let [segment (content/lease-segment lease)]
              (is (= 8 (.byteSize ^MemorySegment segment)))
              (is (= 4 (.get ^MemorySegment segment ValueLayout/JAVA_BYTE 0)))
              lease)))]
    (is (content/lease-closed? observed))
    (is (= 1 @releases))
    (.close ^AutoCloseable observed)
    (is (= 1 @releases) "lease release is idempotent")
    (is (= :numerical-content-lease-closed
           (:reason
            (try
              (content/lease-segment observed)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))))

(defn- raw-chunk [address bytes]
  (numerical-state/chunk
   {:id :sample :offsets [0] :shape [bytes]
    :logical-byte-length bytes :stored-byte-length bytes
    :content address :storage {:format :raw-array :byte-order :little-endian}}))

(defn- chunk-lease [bytes address release-count]
  (content/local-content-lease
   {:content address
    :placement (content/content-placement
                {:provider-id :local-test :tier-id :file :content address})
    :segment (MemorySegment/ofArray bytes) :byte-offset 4 :byte-length 8
    :release-fn #(swap! release-count inc)}))

(deftest rejected-provider-leases-release-on-every-failed-handoff
  (let [expected (address 1)
        description (content/provider-description
                     {:id :local-test
                      :tiers [(content/storage-tier {:id :file :kind :file :locality :node
                                                     :durability :cached})]
                      :capabilities #{:scoped-segment}})]
    (doseq [[alter-lease reason] [[#(assoc-in % [:placement :tier-id] :unknown)
                                  :numerical-content-provider-tier]
                                 [#(assoc-in % [:placement :provider-id] :foreign)
                                  :numerical-content-placement-provider-mismatch]
                                 [#(assoc % :content (address 2)) :numerical-content-open-mismatch]
                                 [#(assoc % :byte-length 100) nil]]]
      (let [releases (atom 0)
            lease (alter-lease (chunk-lease (byte-array 16) expected releases))
            provider (reify content/ContentProvider
                       (-provider-descriptor [_] description)
                       (-open-local-content! [_ _ _] lease))
            failure (try (content/open-local-content! provider expected) nil
                         (catch Throwable error error))]
        (is (some? failure))
        (when reason (is (= reason (:reason (ex-data failure)))))
        (is (= 1 @releases))
        (is (content/lease-closed? lease))
        (.close ^AutoCloseable lease)
        (is (= 1 @releases))))
    (let [releases (atom 0)
          cleanup (ex-info "provider cleanup failed" {})
          lease (-> (chunk-lease (byte-array 16) expected releases)
                    (assoc-in [:placement :provider-id] :foreign)
                    (assoc :release-fn #(do (swap! releases inc) (throw cleanup))))
          provider (reify content/ContentProvider
                     (-provider-descriptor [_] description)
                     (-open-local-content! [_ _ _] lease))
          failure (try (content/open-local-content! provider expected) nil
                       (catch Throwable error error))]
      (is (= :numerical-content-placement-provider-mismatch (:reason (ex-data failure))))
      (is (= [cleanup] (vec (.getSuppressed ^Throwable failure))))
      (is (= 1 @releases)))))

(deftest content-address-streams-across-staging-blocks
  (let [bytes (byte-array (map unchecked-byte (range 131073)))
        expected (.formatHex (HexFormat/of)
                             (.digest (MessageDigest/getInstance "SHA-256") bytes))]
    (is (= (numerical-state/content-address :sha-256 expected)
           (content/content-address-of (MemorySegment/ofArray bytes))))))

(deftest bounded-reader-hashes-one-byte-stream-not-a-tree-of-chunk-digests
  (let [bytes (byte-array (map unchecked-byte (range 131073)))
        source (MemorySegment/ofArray bytes)
        requests (atom [])
        actual (content/content-address-from-reader
                (alength bytes)
                (fn [offset ^MemorySegment destination]
                  (let [n (.byteSize destination)]
                    (swap! requests conj [offset n])
                    (MemorySegment/copy source offset destination 0 n)
                    n)))]
    (is (= (content/content-address-of source) actual))
    (is (= [[0 65536] [65536 65536] [131072 1]] @requests))
    (is (= (content/content-address-of (MemorySegment/ofArray (byte-array 0)))
           (content/content-address-from-reader 0 #(throw (AssertionError. "empty read")))))))

(deftest reader-contract-failures-do-not-produce-content-evidence
  (let [reads (atom 0)
        read! (fn [& _] (swap! reads inc) 0)
        reason (fn [f] (try (f) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (doseq [n [-1 0.5 (inc (bigint Long/MAX_VALUE))]]
      (is (= :numerical-content-hash-reader
             (reason #(content/content-address-from-reader n read!)))))
    (is (zero? @reads))
    (is (= :numerical-content-hash-reader
           (reason #(content/content-address-from-reader 1 nil))))
    (is (= :numerical-content-hash-short-read
           (reason #(content/content-address-from-reader 8 read!))))
    (is (= 1 @reads))))

(deftest localized-chunks-are-verified-over-the-leased-byte-range
  (let [bytes (byte-array (map byte (range 16)))
        expected-address (content/content-address-of
                          (.asSlice (MemorySegment/ofArray bytes) 4 8))
        chunk (raw-chunk expected-address 8)
        releases (atom 0)]
    (with-open [lease (chunk-lease bytes expected-address releases)]
      (is (identical? lease (content/verify-chunk-lease! chunk lease)))
      (is (= 0 @releases) "verification does not consume the provider lease")
      (aset-byte bytes 7 (byte 99))
      (is (= :numerical-content-chunk-digest
             (:reason (try (content/verify-chunk-lease! chunk lease) nil
                           (catch clojure.lang.ExceptionInfo error (ex-data error))))))
      (is (false? (content/lease-closed? lease))))
    (is (= 1 @releases))
    (with-open [lease (chunk-lease bytes expected-address releases)]
      (is (= :numerical-content-chunk-extent
             (:reason (try (content/verify-chunk-lease! (assoc chunk :stored-byte-length 7)
                                                           lease)
                           nil
                           (catch clojure.lang.ExceptionInfo error (ex-data error))))))
      (is (= :numerical-content-chunk-identity
             (:reason (try (content/verify-chunk-lease!
                            (assoc chunk :content (address 1)) lease)
                           nil
                           (catch clojure.lang.ExceptionInfo error (ex-data error))))))
      (is (false? (content/lease-closed? lease))))
    (let [unsupported (numerical-state/content-address :unsupported "digest")]
      (with-open [lease (chunk-lease bytes unsupported releases)]
        (is (= :numerical-content-chunk-algorithm
               (:reason (try (content/verify-chunk-lease!
                              (assoc chunk :content unsupported) lease)
                             nil
                             (catch clojure.lang.ExceptionInfo error (ex-data error))))))))
    (is (= 3 @releases))))

(defn- codec-fixture [bytes element-dtype order]
  (let [segment (MemorySegment/ofArray bytes)
        address (content/content-address-of segment)
        chunk (numerical-state/chunk
               {:id :codec :offsets [0] :shape [(quot (alength bytes) (dtype/bytes-of element-dtype))]
                :logical-byte-length (alength bytes) :stored-byte-length (alength bytes)
                :content address :storage {:format :raw-array :byte-order order}})
        lease (content/local-content-lease
               {:content address
                :placement (content/content-placement
                            {:provider-id :codec-test :tier-id :memory :content address})
                :segment segment :byte-length (alength bytes) :release-fn (fn [])})]
    {:chunk chunk :lease lease :source segment}))

(deftest raw-array-codec-preserves-all-element-bits-in-both-orders
  ;; Arbitrary bits include payload NaNs, sign bits and half patterns; never host FP decoding.
  (doseq [element-dtype [:byte :half :int :long :float :double :f32]
          source-order [:little-endian :big-endian]
          target-order [:little-endian :big-endian]]
    (let [width (dtype/bytes-of element-dtype)
          ;; Little-endian quiet-NaN payload, negative zero and all-one payload per width.
          words ({1 [0 -128 -1]
                  2 [1 126 0 -128 -1 -1]
                  4 [1 0 -64 127 0 0 0 -128 -1 -1 -1 -1]
                  8 [1 0 0 0 0 0 -8 127 0 0 0 0 0 0 0 -128 -1 -1 -1 -1 -1 -1 -1 -1]}
                 width)
          bytes (byte-array (if (= :little-endian source-order)
                              words (mapcat reverse (partition width words))))
          {:keys [chunk lease source]} (codec-fixture bytes element-dtype source-order)
          target (byte-array (alength bytes))
          expected (if (= source-order target-order)
                     (vec bytes)
                     (vec (mapcat reverse (partition width bytes))))]
      (with-open [lease lease]
        (let [destination (MemorySegment/ofArray target)]
          (is (identical? destination
                          (content/decode-raw-array-chunk! chunk lease element-dtype destination target-order))))
        (is (= expected (vec target)))
        (is (= (vec bytes) (vec (.toArray source ValueLayout/JAVA_BYTE))))
        (is (false? (content/lease-closed? lease)))))))

(deftest raw-array-codec-bounds-staging-and-validates-before-any-write
  (let [bytes (byte-array (map unchecked-byte (range 131080)))
        {:keys [chunk lease source]} (codec-fixture bytes :double :little-endian)
        target (byte-array (alength bytes))
        destination (MemorySegment/ofArray target)
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (with-open [lease lease]
      (is (identical? destination
                      (content/decode-raw-array-chunk! chunk lease :double destination :big-endian)))
      (is (= (vec (mapcat reverse (partition 8 bytes))) (vec target)))
      (java.util.Arrays/fill target (byte 99))
      (let [unchanged (vec target)]
        (doseq [[bad-chunk bad-lease dt dst order expected]
                [[(assoc-in chunk [:storage :format] :compressed) lease :double destination :big-endian
                  :numerical-content-codec-format]
                 [(assoc-in chunk [:storage :compression] :none) lease :double destination :big-endian
                  :numerical-content-codec-format]
                 [chunk lease :double destination :native :numerical-content-codec-byte-order]
                 [chunk lease :float destination :big-endian :numerical-content-codec-extent]
                 [chunk lease :double (.asSlice destination 0 8) :big-endian :numerical-content-codec-destination]
                 [chunk lease :double (.asReadOnly destination) :big-endian :numerical-content-codec-destination]
                 [chunk lease :double source :big-endian :numerical-content-codec-alias]
                 [(assoc chunk :shape [Long/MAX_VALUE]) lease :double destination :big-endian
                  :numerical-content-codec-extent]
                 [chunk lease :unknown destination :big-endian :unknown-dtype]
                 [chunk lease :double nil :big-endian :numerical-content-codec-destination]]]
          (is (= expected (reason #(content/decode-raw-array-chunk! bad-chunk bad-lease dt dst order))))
          (is (= unchanged (vec target))))
        (aset-byte bytes 2 (byte 77))
        (is (= :numerical-content-chunk-digest
               (reason #(content/decode-raw-array-chunk! chunk lease :double destination :big-endian))))
        (is (= unchanged (vec target)))))))

(deftest raw-array-codec-rejects-partial-alias-and-closed-destinations
  (let [bytes (byte-array (range 16))
        base (MemorySegment/ofArray bytes)
        address (content/content-address-of (.asSlice base 4 8))
        chunk (raw-chunk address 8)
        reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (with-open [lease (chunk-lease bytes address (atom 0))]
      (is (= :numerical-content-codec-alias
             (reason #(content/decode-raw-array-chunk! chunk lease :byte (.asSlice base 8 8) :little-endian))))
      (is (= (vec (range 16)) (vec bytes)))
      (let [arena (Arena/ofConfined)
            destination (.allocate arena 8)]
        (.close arena)
        (is (= :numerical-content-codec-destination
               (reason #(content/decode-raw-array-chunk! chunk lease :byte destination :little-endian))))
        (is (false? (content/lease-closed? lease)))))))

(defn- fake-transfer-session
  []
  (let [buffer {:dtype :byte :n-elements 8 :byte-size 8}
        allocation (buffer-view/allocation
                    {:id :resident-allocation
                     :byte-size 8
                     :memory-space :device
                     :device :ze:0
                     :coherence :host-coherent
                     :ownership :owned})]
    (atom {:device-id :ze:0
           :session-id :retained-transfer-session
           :buffers {:buffer buffer}
           :allocations {:buffer allocation}
           :kernel-graphs {}
           :events {}
           :closed? false})))

(deftest polling-and-cancellation-release-local-content-only-at-the-transfer-boundary
  (let [{:keys [provider releases]} (fake-provider :transfer-provider)
        lease (content/open-local-content! provider (address 4))
        session (fake-transfer-session)
        backend-token ::backend-transfer
        backend-complete? (atom false)
        resolver
        (fn [_ name]
          (case name
            "plan-range" (fn [_ host spec direction]
                           {:host-segment host :spec spec :direction direction :n-bytes 8})
            "submit-range-batch!" (fn [_ direction]
                                    (is (= :upload direction))
                                    backend-token)
            "await-event!" (fn [token]
                             (is (= backend-token token))
                             {:timing-source :device-event
                              :elapsed-ns 10 :bytes 8 :commands 1
                              :direction :upload :asynchronous? true})
            "release-event!" (fn [token] (is (= backend-token token)))
            "event-complete?" (fn [token]
                                (is (= backend-token token))
                                @backend-complete?)
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (let [event (gpu/submit-upload-ranges-retained!
                     session
                     [[:buffer (content/lease-segment lease) {:elements 8}]]
                     [lease])]
          (is (false? (content/lease-closed? lease)))
          (is (= 0 @releases))
          (is (false? (gpu/event-complete? session event)))
          ;; A cancellation request may stop scheduling new work, but it cannot close the mapped
          ;; arena. Even a positive device poll is only an observation: release-event! establishes
          ;; the host-visible transfer boundary and consumes both event and lease ownership.
          (reset! backend-complete? true)
          (is (gpu/event-complete? session event))
          (is (false? (content/lease-closed? lease)))
          (gpu/release-event! session event)
          (is (content/lease-closed? lease))
          (is (= 1 @releases))
          (is (empty? (:events @session))))))))

(deftest failed-transfer-submission-does-not-take-lease-ownership
  (let [{:keys [provider releases]} (fake-provider :failing-provider)
        lease (content/open-local-content! provider (address 5))
        session (fake-transfer-session)]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "plan-range" (fn [& _] (throw (ex-info "invalid range" {})))
           (throw (ex-info "unexpected mocked runtime function" {:name name}))))}
      (fn []
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid range"
                              (gpu/submit-upload-ranges-retained!
                               session
                               [[:buffer (content/lease-segment lease) {:elements 8}]]
                               [lease])))
        (is (false? (content/lease-closed? lease)))
        (is (= 0 @releases))))
    (.close ^AutoCloseable lease)
    (is (= 1 @releases))))

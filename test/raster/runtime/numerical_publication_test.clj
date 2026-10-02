(ns raster.runtime.numerical-publication-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-state :as state]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign MemorySegment]))

(defn- fixture-state []
  (let [payloads (mapv #(byte-array (range % (+ % 8))) [0 8])
        addresses (mapv #(content/content-address-of (MemorySegment/ofArray %)) payloads)]
    {:payloads (zipmap addresses payloads)
     :state (state/certify
             (state/manifest
              {:id :step-2 :parents [:step-1] :logical-coordinate {:step 2}
               :fields [(state/field
                         {:id :values :value (av/tensor {:dtype :byte :shape [16]})
                          :chunk-shape [8] :coordinate-space {:axis :samples}
                          :chunks (mapv (fn [i address]
                                          (state/chunk
                                           {:id i :offsets [(* 8 i)] :shape [8]
                                            :logical-byte-length 8 :stored-byte-length 8
                                            :content address
                                            :storage {:format :raw-array :byte-order :little-endian}}))
                                        (range 2) addresses)})]
               :numerical-contract {:mode :integer :determinism :bitwise :compatibility-id "byte-test-v1"}
               :provenance {:program-fingerprint "declared-test-producer-not-authenticated"}}))}))

(defn- publication-provider [payloads intercept!]
  (let [description (atom (content/provider-description
                           {:id :publication-test :capabilities #{:localize :scoped-segment :promote}
                            :tiers [(content/storage-tier
                                     {:id :local :kind :memory :locality :node :durability :cached
                                      :capabilities #{:scoped-segment}})
                                    (content/storage-tier
                                     {:id :durable :kind :object-store :locality :site :durability :durable
                                      :capabilities #{:durable-receipt}})]}))
        trace (atom []) events (atom {}) leases (atom #{}) promoted (atom [])
        counts (atom {:event-open 0 :event-release 0 :lease-open 0 :lease-release 0})
        max-live (atom 0)
        observe! #(swap! max-live max (+ (count @events) (count @leases)))
        step! (fn [stage value] (swap! trace conj stage) (intercept! stage value))
        submit! (fn [operation address tier]
                  (step! (keyword (str "submit-" (name operation))) address)
                  (let [event (content/storage-event {:provider-id :publication-test :id (random-uuid)
                                                      :operation operation})
                        placement (content/content-placement
                                   {:provider-id :publication-test :tier-id tier :content address})]
                    (swap! counts update :event-open inc)
                    (swap! events assoc (:id event) {:operation operation :placement placement})
                    (observe!)
                    event))
        provider (reify content/ContentProvider
                   (-provider-descriptor [_] (intercept! :descriptor @description))
                   (-submit-localization! [_ address opts] (submit! :localize address (:tier opts)))
                   (-submit-promotion! [_ address tier _] (submit! :promote address tier))
                   (-await-storage-event! [_ event]
                     (let [{:keys [operation placement]} (get @events (:id event))
                           result (step! (keyword (str "await-" (name operation))) placement)]
                       (when (= :promote operation) (swap! promoted conj (:content placement)))
                       result))
                   (-release-storage-event! [_ event]
                     (let [operation (get-in @events [(:id event) :operation])]
                       (swap! counts update :event-release inc)
                       (swap! events dissoc (:id event))
                       (step! (keyword (str "release-" (name operation))) nil)))
                   (-open-local-content! [_ address opts]
                     (let [bytes (step! :open (get payloads address))
                           id (random-uuid)
                           placement (step! :lease-placement
                                            (content/content-placement
                                             {:provider-id :publication-test :tier-id (:tier opts)
                                              :content address}))]
                       (swap! counts update :lease-open inc)
                       (swap! leases conj id)
                       (observe!)
                       (content/local-content-lease
                        {:content address :placement placement
                         :segment (.asReadOnly (MemorySegment/ofArray bytes)) :byte-length (alength bytes)
                         :release-fn #(do (swap! counts update :lease-release inc)
                                          (swap! leases disj id) (step! :close nil))}))))]
    {:provider provider :description description :trace trace :events events :leases leases
     :counts counts :promoted promoted :max-live max-live}))

(defn- assert-released! [{:keys [events leases counts max-live]}]
  (is (empty? @events))
  (is (empty? @leases))
  (is (= (:event-open @counts) (:event-release @counts)))
  (is (= (:lease-open @counts) (:lease-release @counts)))
  (is (<= @max-live 1)))

(deftest metadata-is-published-only-after-ordered-verified-durable-receipts
  (let [{:keys [payloads state]} (fixture-state)
        {:keys [provider trace promoted] :as runtime} (publication-provider payloads (fn [_ value] value))
        publications (atom [])
        result (content/finalize-state-availability!
                provider state :durable
                (fn [manifest]
                  (is (= 2 (count @promoted)))
                  (assert-released! runtime)
                  (swap! trace conj :publish)
                  (swap! publications conj manifest)
                  :ack))]
    (is (= :ack (:publication result)))
    (is (identical? state (:certified-state result)))
    (is (identical? (:manifest state) (first @publications)))
    (is (= [:values :values] (mapv :field-id (:placements result))))
    (is (= [0 1] (mapv :chunk-id (:placements result))))
    (is (= (mapv :content (get-in state [:manifest :fields 0 :chunks]))
           (mapv #(get-in % [:placement :content]) (:placements result))))
    (is (= (vec (concat (mapcat identity (repeat 2 [:submit-localize :await-localize :release-localize
                                                  :open :lease-placement :close
                                                  :submit-promote :await-promote :release-promote]))
                        [:publish])) @trace))
    (is (not (contains? (:manifest state) :placements)))
    (assert-released! runtime)))

(deftest failed-stages-never-publish-and-release-every-acquired-resource
  (doseq [stage [:submit-localize :await-localize :release-localize :open :close
                :submit-promote :await-promote :release-promote]]
    (let [{:keys [payloads state]} (fixture-state)
          failure (ex-info (str "injected " stage) {})
          {:keys [provider] :as runtime}
          (publication-provider payloads (fn [current value] (if (= stage current) (throw failure) value)))
          publications (atom 0)
          result (try (content/finalize-state-availability! provider state :durable
                                                          (fn [_] (swap! publications inc)))
                      (catch Throwable error error))]
      (is (identical? failure result) (name stage))
      (is (zero? @publications))
      (assert-released! runtime))))

(deftest repeated-content-retains-field-order-without-runtime-metadata-in-the-manifest
  (let [{:keys [payloads state]} (fixture-state)
        first-field (first (get-in state [:manifest :fields]))
        candidate (state/certify (assoc (:manifest state) :fields
                                       [first-field (assoc first-field :id :copies)]))
        opaque (Object.)
        {:keys [provider promoted] :as runtime}
        (publication-provider payloads (fn [stage value]
                                        (if (= :await-promote stage)
                                          (assoc value :attributes {:provider-receipt opaque}) value)))
        publications (atom 0)
        result (content/finalize-state-availability!
                provider candidate :durable
                (fn [manifest]
                  (is (= (:manifest candidate) manifest))
                  (swap! publications inc)))]
    (is (= [[:values 0] [:values 1] [:copies 0] [:copies 1]]
           (mapv (juxt :field-id :chunk-id) (:placements result))))
    (is (= 4 (count @promoted)) "each declared chunk is independently verified, not hash-cache trusted")
    (is (= 1 @publications))
    (is (every? #(identical? opaque (get-in % [:placement :attributes :provider-receipt]))
                (:placements result)))
    (is (identical? candidate (state/verify! (:certified-state result))))
    (assert-released! runtime)))

(deftest mismatched-placements-corrupt-bytes-and-cleanup-failures-stay-visible
  (doseq [[stage alter reason]
          [[:await-localize (constantly nil) :numerical-content-completion-placement]
           [:await-localize #(assoc % :tier-id :durable) :numerical-content-completion-mismatch]
           [:await-promote #(assoc % :tier-id :local) :numerical-content-completion-mismatch]
           [:await-promote #(assoc % :provider-id :foreign) :numerical-content-completion-mismatch]
           [:await-promote #(assoc % :content (state/content-address :sha-256 "wrong"))
            :numerical-content-completion-mismatch]
           [:lease-placement #(assoc % :tier-id :durable) :numerical-content-completion-mismatch]
           [:open #(byte-array (repeat (alength %) 99)) :numerical-content-chunk-digest]
           [:open #(byte-array (dec (alength %))) :numerical-content-chunk-extent]]]
    (let [{:keys [payloads state]} (fixture-state)
          {:keys [provider] :as runtime}
          (publication-provider payloads (fn [current value] (if (= stage current) (alter value) value)))
          publications (atom 0)
          failure (try (content/finalize-state-availability! provider state :durable
                                                           (fn [_] (swap! publications inc)))
                       (catch Throwable error error))]
      (is (= reason (:reason (ex-data failure))))
      (is (zero? @publications))
      (assert-released! runtime)))
  (let [{:keys [payloads state]} (fixture-state)
        primary (ex-info "await failed" {}) cleanup (ex-info "drain failed" {})
        {:keys [provider] :as runtime}
        (publication-provider payloads (fn [stage value]
                                        (case stage :await-localize (throw primary)
                                          :release-localize (throw cleanup) value)))
        result (try (content/finalize-state-availability! provider state :durable
                                                        (fn [_] (throw (AssertionError. "early publication"))))
                    (catch Throwable error error))]
    (is (identical? primary result))
    (is (= [cleanup] (vec (.getSuppressed ^Throwable result))))
    (assert-released! runtime)))

(deftest accepted-events-drain-even-when-descriptor-revalidation-fails
  (doseq [cleanup-fails? [false true]]
    (let [{:keys [payloads state]} (fixture-state)
          descriptors (atom 0)
          primary (ex-info "descriptor unavailable after submission" {})
          cleanup (ex-info "safe drain failed" {})
          {:keys [provider] :as runtime}
          (publication-provider payloads
                                (fn [stage value]
                                  (cond
                                    (and (= :descriptor stage) (= 3 (swap! descriptors inc)))
                                    (throw primary)
                                    (and cleanup-fails? (= :release-localize stage)) (throw cleanup)
                                    :else value)))
          publications (atom 0)
          result (try (content/finalize-state-availability! provider state :durable
                                                          (fn [_] (swap! publications inc)))
                      (catch Throwable error error))]
      (is (identical? primary result))
      (is (= (if cleanup-fails? [cleanup] []) (vec (.getSuppressed ^Throwable result))))
      (is (zero? @publications))
      (assert-released! runtime)))
  (let [{:keys [payloads state]} (fixture-state)
        current (atom nil)
        {:keys [provider description] :as runtime}
        (publication-provider payloads (fn [stage value]
                                        (when (= :await-localize stage) (reset! @current nil))
                                        value))
        _ (reset! current description)
        publications (atom 0)
        result (try (content/finalize-state-availability! provider state :durable
                                                        (fn [_] (swap! publications inc)))
                    (catch Throwable error error))]
    (is (= :numerical-content-provider-descriptor-type (:reason (ex-data result))))
    (is (zero? @publications))
    (assert-released! runtime)))

(deftest complete-preflight-precedes-every-provider-submission
  (doseq [scenario [:certificate :algorithm :callback :capability :local-tier :target-tier :durability]]
    (let [{:keys [payloads state]} (fixture-state)
          {:keys [provider description trace] :as runtime}
          (publication-provider payloads (fn [_ value] value))
          candidate (case scenario
                      :certificate (assoc-in state [:certificate :stored-byte-length] 17)
                      :algorithm (state/certify
                                  (assoc-in (:manifest state) [:fields 0 :chunks 1 :content]
                                            (state/content-address :unsupported "second-chunk")))
                      state)
          publications (atom 0)
          _ (case scenario
              :capability (swap! description update :capabilities disj :promote)
              :local-tier (swap! description update :tiers
                                 #(mapv (fn [tier] (update tier :capabilities disj :scoped-segment)) %))
              :durability (swap! description update-in [:tiers 1 :capabilities] disj :durable-receipt)
              nil)
          failure (try (content/finalize-state-availability!
                        provider candidate (if (= :target-tier scenario) :absent :durable)
                        (when-not (= :callback scenario) (fn [_] (swap! publications inc))))
                       (catch Throwable error error))]
      (is (= (case scenario
               :certificate :numerical-state-certificate
               :algorithm :numerical-content-chunk-algorithm
               :callback :numerical-content-publication-callback
               :capability :numerical-content-provider-capability
               :local-tier :numerical-content-localization-tier
               :target-tier :numerical-content-provider-tier
               :durability :numerical-content-promotion-durability)
             (:reason (ex-data failure))))
      (is (zero? @publications))
      (is (empty? @trace) "including an unsupported algorithm in the second chunk")
      (assert-released! runtime))))

(deftest later-failure-leaves-honest-orphan-blobs-and-no-early-metadata
  (let [{:keys [payloads state]} (fixture-state)
        opens (atom 0)
        {:keys [provider promoted] :as runtime}
        (publication-provider payloads (fn [stage value]
                                        (if (and (= :open stage) (= 2 (swap! opens inc)))
                                          (byte-array (repeat 8 99)) value)))
        publications (atom 0)
        failure (try (content/finalize-state-availability! provider state :durable
                                                         (fn [_] (swap! publications inc)))
                     (catch Throwable error error))]
    (is (= :numerical-content-chunk-digest (:reason (ex-data failure))))
    (is (zero? @publications))
    (is (= [(get-in state [:manifest :fields 0 :chunks 0 :content])] @promoted))
    (assert-released! runtime))
  (let [{:keys [payloads state]} (fixture-state)
        {:keys [provider promoted] :as runtime} (publication-provider payloads (fn [_ value] value))
        failure (ex-info "publication acknowledgment failed" {})
        publications (atom 0)
        result (try (content/finalize-state-availability!
                     provider state :durable (fn [_] (swap! publications inc) (throw failure)))
                    (catch Throwable error error))]
    (is (identical? failure result))
    (is (= 1 @publications))
    (is (= 2 (count @promoted)) "the caller may already have published metadata; no rollback claim")
    (assert-released! runtime)))

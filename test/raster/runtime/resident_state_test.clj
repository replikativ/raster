(ns raster.runtime.resident-state-test
  "Hardware-free transfer doubles; native tests independently exercise original receipt evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.test-lifecycle :as lifecycle]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident])
  (:import [java.lang.foreign MemorySegment]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- with-fixture [n f]
  (let [payloads {:a (MemorySegment/ofArray (float-array (map float (range n))))
                  :b (MemorySegment/ofArray (float-array (repeat n -2.0)))}
        nodes (into {} (map (fn [[id _]]
                              [id {:view (view/view
                                          (view/allocation {:id id :byte-size (* 4 n)
                                                            :memory-space :device :ownership :owned})
                                          {:id id :dtype :float :shape [n]})}]) payloads))
        session (atom {})
        executable (lifecycle/linked-executable
                    {:plan {:id :unit :outputs [:a :b] :nodes nodes}
                     :session session :owns-session? true :node-views {:a :a :b :b}
                     :output-leases (atom 0) :output-ready? (atom true)
                     :completed-replays (atom 1)})
        leaf (fn [id] (assoc (select-keys (:view (nodes id)) [:dtype :shape :strides :byte-length])
                             :content (content/content-address-of (payloads id))
                             :representation :device-native
                             :storage {:format :raw-array :byte-order :little-endian}))
        description {:program-fingerprint "unit-producer" :completed-fingerprint "unit-completed"
                     :representations {:float {:kind :raster.compiled/resident-representation-v1
                                               :dtype :float :byte-order :little-endian
                                               :program-fingerprint "unit-producer"
                                               :probe-fingerprint "unit-probe"}}
                     :outputs {:a (leaf :a) :b (leaf :b)} :post-state {:b (leaf :b)}}
        reads (atom [])]
    (with-redefs [link/output-values (constantly {})
                  compiled/completed-storage-description
                  (fn [receipt _]
                    (when-not (compiled/completed-evidence? receipt) (throw (ex-info "forged" {})))
                    @(:lease receipt)
                    description)
                  gpu/download-range!
                  (fn [_ id ^MemorySegment destination {:keys [src-element elements]}]
                    (swap! reads conj [id src-element elements])
                    (MemorySegment/copy (payloads id) (* 4 src-element) destination 0 (* 4 elements))
                    destination)]
      (with-open [receipt (#'compiled/seal-artifact
                           (compiled/->CompletedEvidence
                            {:bound-schedules [{:strategy :unit}] :parent "execution-parent"}
                            (link/output-lease! executable) executable 0 {} (atom false) nil))]
        (f {:receipt receipt :description description :executable executable :payloads payloads
            :reads reads
            :opts {:id :step-2 :parents [:state-1] :logical-coordinate {:step 2}
                   :numerical-contract {:mode :fp32 :determinism :toleranced :compatibility-id "unit"}
                   :fields [{:id :field-a :node :a :source :outputs
                             :value (av/tensor {:dtype :float :shape [n]})}
                            {:id :field-b :node :b :source :post-state
                             :value (av/tensor {:dtype :float :shape [n]})}]}})))))

(defn- provider [intercept!]
  (let [description (content/provider-description
                     {:id :unit-store :capabilities #{:ingest :scoped-segment :localize :promote}
                      :tiers [(content/storage-tier {:id :local :kind :memory :locality :node
                                                     :durability :cached :capabilities #{:scoped-segment}})
                              (content/storage-tier {:id :durable :kind :object-store :locality :site
                                                     :durability :durable :capabilities #{:durable-receipt}})]})
        blobs (atom {}) events (atom {}) submissions (atom []) releases (atom [])
        submit! (fn [operation address tier]
                  (let [event (content/storage-event {:provider-id :unit-store :id (random-uuid)
                                                      :operation operation})]
                    (swap! events assoc (:id event) (content/content-placement
                                                     {:provider-id :unit-store :tier-id tier :content address}))
                    event))
        p (reify content/ContentProvider
            (-provider-descriptor [_] description)
            (-submit-localization! [_ address opts] (submit! :localize address (:tier opts)))
            (-submit-promotion! [_ address tier _] (submit! :promote address tier))
            (-await-storage-event! [_ event] (intercept! :await event) (get @events (:id event)))
            (-release-storage-event! [_ event]
              (swap! releases conj (:id event)) (swap! events dissoc (:id event)))
            (-open-local-content! [_ address opts]
              (let [bytes (get @blobs address)]
                (content/local-content-lease
                 {:content address :placement (content/content-placement
                                               {:provider-id :unit-store :tier-id (:tier opts) :content address})
                  :segment (MemorySegment/ofArray bytes) :byte-length (alength bytes) :release-fn (fn [])})))
            content/ContentIngestor
            (-submit-ingestion! [_ address tier n read! _]
              (swap! submissions conj address)
              (intercept! :submit address)
              (let [bytes (byte-array n)]
                (loop [offset 0 windows (cycle [1 7 65531 3 11])]
                  (when (< offset n)
                    (let [width (min (first windows) (- n offset))]
                      (read! offset (.asSlice (MemorySegment/ofArray bytes) offset width))
                      (recur (+ offset width) (next windows)))))
                (swap! blobs assoc address bytes))
              (submit! :ingest address tier)))]
    {:provider p :blobs blobs :events events :submissions submissions :releases releases}))

(deftest capture-streams-unaligned-windows-and-binds-field-producers
  (with-fixture 40000
    (fn [{:keys [receipt opts reads payloads executable]}]
      (let [{:keys [provider blobs events]} (provider (fn [& _]))
            {:keys [state placements]} (resident/capture! receipt {} provider :local opts)
            manifest (:manifest state)]
        (is (state/verify! state))
        (is (= [:field-a :field-b] (mapv :field-id placements)))
        (is (= [:state-1] (:parents manifest)))
        (is (= "execution-parent" (get-in manifest [:provenance :parent-replay])))
        (is (= [{:strategy :unit}] (get-in manifest [:provenance :bound-schedules])))
        (is (= [{:field-id :field-a :source :outputs :node :a
                 :content (content/content-address-of (:a payloads))}
                {:field-id :field-b :source :post-state :node :b
                 :content (content/content-address-of (:b payloads))}]
               (get-in manifest [:provenance :field-producers])))
        (is (= 2 (count @blobs)))
        (doseq [field (:fields manifest)]
          (let [chunk (first (:chunks field)) bytes (@blobs (:content chunk))]
            (is (= (:content chunk) (content/content-address-of (MemorySegment/ofArray bytes))))))
        (is (every? #(<= (* 4 (last %)) 65542) @reads))
        (is (= 1 @(:output-leases executable)) "private capture lease is gone, caller lease remains")
        (is (empty? @events))
        (is (= :published (:publication (content/finalize-state-availability!
                                         provider state :durable (fn [_] :published)))))
        (is (empty? @events))))))

(deftest malformed-declared-fields-fail-before-any-provider-writes
  (with-fixture 4
    (fn [{:keys [receipt opts executable reads]}]
      (doseq [alter [#(assoc % :unknown true) #(assoc % :fields [])
                     #(assoc-in % [:fields 1 :source] :inputs)
                     #(assoc-in % [:fields 1 :node] :absent)
                     #(assoc-in % [:fields 1 :unexpected] true)
                     #(assoc-in % [:fields 1 :value] (av/tensor {:dtype :double :shape [4]}))
                     #(assoc-in % [:fields 1 :value] (av/tensor {:dtype :float :shape [3]}))
                     #(assoc-in % [:fields 1 :value :logical-layout] {:kind :strided})
                     #(assoc-in % [:fields 1 :value :representation] {:kind :quantized})
                     #(assoc-in % [:fields 1 :value :placement] {:device :unit})
                     #(assoc-in % [:fields 1 :id] :field-a)
                     #(assoc-in % [:fields 1 :attributes] {:opaque (Object.)})]]
        (let [{:keys [provider submissions]} (provider (fn [& _]))]
          (is (some? (error-of #(resident/capture! receipt {} provider :local (alter opts)))))
          (is (empty? @submissions))
          (is (empty? @reads))
          (is (= 1 @(:output-leases executable)))))
      (.close ^java.io.Closeable receipt)
      (let [{:keys [provider submissions]} (provider (fn [& _]))]
        (is (some? (error-of #(resident/capture! receipt {} provider :local opts))))
        (is (empty? @submissions))))))

(deftest later-field-failure-releases-private-lease-and-does-not-return-state
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [attempts (atom 0) failure (ex-info "second field ingestion failed" {})
            {:keys [provider blobs events submissions]}
            (provider (fn [stage _]
                        (when (= :submit stage)
                          (when (= 2 (swap! attempts inc)) (throw failure)))))]
        (is (identical? failure (error-of #(resident/capture! receipt {} provider :local opts))))
        (is (= 2 (count @submissions)))
        (is (= 1 (count @blobs)) "first verified blob is explicitly nontransactional orphan content")
        (is (empty? @events))
        (is (= 1 @(:output-leases executable)))))))

(deftest private-lease-protects-capture-after-reentrant-caller-receipt-close
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [observed (atom [])
            {:keys [provider]}
            (provider (fn [stage _]
                        (when (contains? #{:submit :await} stage)
                          (.close ^java.io.Closeable receipt)
                          (swap! observed conj (:reason (ex-data (error-of #(link/run! executable)))))
                          (swap! observed conj (:reason (ex-data (error-of #(link/close! executable))))))))]
        (is (state/verify! (:state (resident/capture! receipt {} provider :local opts))))
        (is (= (vec (repeat 8 :link-output-lease-active)) @observed))
        (is (zero? @(:output-leases executable)))))))

(deftest download-failure-does-not-publish-or-retain-the-private-lease
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [failure (ex-info "download failed" {})
            {:keys [provider blobs events]} (provider (fn [& _]))]
        (with-redefs [gpu/download-range! (fn [& _] (throw failure))]
          (is (identical? failure (error-of #(resident/capture! receipt {} provider :local opts)))))
        (is (empty? @blobs))
        (is (empty? @events))
        (is (= 1 @(:output-leases executable)))))))

(deftest provider-worker-observes-private-lease-without-monitor-deadlock
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [observed (atom [])
            {:keys [provider]}
            (provider (fn [stage _]
                        (when (contains? #{:submit :await} stage)
                          (.close ^java.io.Closeable receipt)
                          (let [worker (future
                                         [(:reason (ex-data (error-of #(link/run! executable))))
                                          (:reason (ex-data (error-of #(link/close! executable))))])
                                result (deref worker 2000 ::timeout)]
                            (when (= ::timeout result)
                              (future-cancel worker)
                              (throw (ex-info "provider worker blocked on capture monitor" {})))
                            (swap! observed into result)))))]
        (is (state/verify! (:state (resident/capture! receipt {} provider :local opts))))
        (is (= (vec (repeat 8 :link-output-lease-active)) @observed))
        (is (zero? @(:output-leases executable)))))))

(deftest later-download-failure-preserves-the-primary-and-unpins
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [download gpu/download-range! attempts (atom 0)
            failure (ex-info "download failed after an unaligned window" {})
            {:keys [provider blobs events]} (provider (fn [& _]))]
        (with-redefs [gpu/download-range! (fn [& args]
                                            (if (= 2 (swap! attempts inc))
                                              (throw failure)
                                              (apply download args)))]
          (is (identical? failure (error-of #(resident/capture! receipt {} provider :local opts)))))
        (is (= 2 @attempts))
        (is (empty? @blobs))
        (is (empty? @events))
        (is (= 1 @(:output-leases executable)))))))

(deftest uncertain-private-lease-cleanup-retains-explicit-authority
  (with-fixture 4
    (fn [{:keys [receipt opts executable]}]
      (let [acquire link/output-lease!
            held (atom nil)
            failure (ex-info "lease close outcome is unknown" {})
            {:keys [provider events]} (provider (fn [& _]))]
        (try
          (with-redefs [link/output-lease!
                        (fn [executable]
                          (reset! held (acquire executable))
                          (reify java.io.Closeable (close [_] (throw failure))))]
            (let [error (error-of #(resident/capture! receipt {} provider :local opts))]
              (is (identical? failure (.getCause ^Throwable error)))
              (is (some? (::cleanup/unresolved (ex-data error))))
              (is (= 2 @(:output-leases executable)))
              (is (empty? @events))))
          ;; The test knows its synthetic close did nothing; production cannot guess that.
          (finally (when-let [lease @held] (.close ^java.io.Closeable lease))))
        (is (= 1 @(:output-leases executable)))))))

(deftest captured-restore-checks-independent-semantics-producer-and-full-field-content
  (with-fixture 4
    (fn [{:keys [receipt opts reads]}]
      (let [{:keys [provider]} (provider (fn [& _]))
            captured (:state (resident/capture! receipt {} provider :local opts))
            semantics {:fields [{:id :field-a :value (av/tensor {:dtype :float :shape [4]}) :coordinate-space {}}
                                {:id :field-b :value (av/tensor {:dtype :float :shape [4]}) :coordinate-space {}}]
                       :logical-coordinate {:step 2} :numerical-contract (:numerical-contract opts)}
            specs [{:id :field-a :source :outputs :key :a-key}
                   {:id :field-b :source :post-state :key :b-key}]
            ;; Explicit compiler-interface double; actual sealed Prepared/target/composition
            ;; admission is independently tested in equation-artifact and native restore tests.
            interface {:kind :raster.compiled/producer-interface-v1
                       :program {:scope :exact-bound-program :fingerprint "unit-producer"}
                       :outputs [{:key :a-key :node :a :dtype :float :shape [4]}]
                       :post-state [{:key :b-key :node :b :dtype :float :shape [4]}]}
            source ::synthetic-prepared
            error #(error-of (fn [] (resident/verify-restore! %1 %2 source %3)))]
        (reset! reads [])
        (with-redefs [compiled/producer-interface (constantly interface)]
          (is (identical? captured (resident/verify-restore! captured semantics source specs)))
          (doseq [bad [(assoc semantics :logical-coordinate {:step 3})
                       (assoc-in semantics [:fields 0 :coordinate-space] {:grid :wrong})
                       (assoc-in semantics [:fields 0 :value] (av/tensor {:dtype :double :shape [4]}))
                       (assoc-in semantics [:numerical-contract :compatibility-id] "wrong-policy")
                       (assoc semantics :provenance (:provenance (:manifest captured)))]]
            (is (some? (error captured bad specs))))
          (doseq [bad [(vec (reverse specs))
                       (assoc-in specs [0 :key] :absent)
                       (assoc-in specs [0 :source] :inputs)
                       (assoc-in specs [0 :node] :a)
                       (assoc-in specs [1 :id] :field-a)]]
            (is (some? (error captured semantics bad))))
          (doseq [alter [#(assoc-in % [:provenance :field-producers 0 :content]
                                    (get-in % [:fields 1 :chunks 0 :content]))
                         #(assoc-in % [:provenance :field-producers 0 :node] :b)
                         #(assoc-in % [:provenance :field-producers 0 :unexpected] true)
                         #(assoc-in % [:provenance :scope] :declared-not-captured)
                         #(assoc-in % [:provenance :unexpected] true)
                         #(assoc-in % [:provenance :representations] {})
                         #(assoc-in % [:provenance :representations :float :dtype] :double)
                         #(assoc-in % [:provenance :representations :float :kind] :unknown)
                         #(assoc-in % [:provenance :representations :float :program-fingerprint] "foreign")
                         #(assoc-in % [:provenance :representations :float :byte-order] :big-endian)
                         #(assoc-in % [:fields 0 :chunks 0 :id] :not-v1)]]
            (is (some? (error (state/certify (alter (:manifest captured))) semantics specs))))
          ;; Exact-provenance restoration has not become a subset/source-program matcher.
          (let [strict (assoc semantics :provenance (:provenance (:manifest captured)))
                changed (state/certify (assoc-in (:manifest captured)
                                                 [:provenance :completed-fingerprint] "different-replay"))]
            (is (identical? captured (state/verify-restore! captured strict)))
            (is (= :provenance (:facet (ex-data (error-of #(state/verify-restore! changed strict))))))
            (is (identical? changed (resident/verify-restore! changed semantics source specs))
                "dynamic replay audit data is not a target-predicted compatibility value")))
        (doseq [bad [(assoc-in interface [:program :fingerprint] "foreign-program")
                     (assoc-in interface [:program :scope] :not-exact)
                     (assoc interface :kind :future-unknown-schema)
                     (update interface :outputs conj
                             {:key :alias-key :node :a :dtype :float :shape [4]})
                     (assoc-in interface [:outputs 0 :node] :b)
                     (assoc-in interface [:outputs 0 :shape] [3])]]
          (with-redefs [compiled/producer-interface (constantly bad)]
            (is (some? (error captured semantics specs)))))
        (is (empty? @reads) "verification never opens provider bytes or downloads resident storage")))))

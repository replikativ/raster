(ns raster.ode.amr-cycle-state-test
  "Hardware-free capture fault controls; compiler/device authority is tested independently."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.distributed-plan :as plan]
            [raster.compiler.ir.numerical-state :as state]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.distributed :as runtime]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.ode.amr-cycle-state :as capture]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident]
            [raster.runtime.resident-state-test :as provider-fixture])
  (:import [java.lang.foreign MemorySegment]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- fixture [f]
  (let [payloads {:coarse (MemorySegment/ofArray (double-array [1 2 3 4]))
                  :fine (MemorySegment/ofArray (double-array [5 6 7 8]))}
        fields (mapv (fn [id]
                       (state/field {:id id :value (av/tensor {:dtype :double :shape [2 2]
                                                             :sharding {:kind :partitioned :axis 0
                                                                        :devices [:worker]}})
                                     :chunk-shape [2 2] :coordinate-space {:patch id}
                                     :chunks [(state/chunk {:id 0 :offsets [0 0] :shape [2 2]
                                                            :logical-byte-length 32 :stored-byte-length 32
                                                            :content (content/content-address-of (payloads id))
                                                            :storage {:format :raw-array :byte-order :little-endian}})]}))
                     [:coarse :fine])
        input (state/manifest {:id :input :parents [] :logical-coordinate {:step 0 :phase :synchronized}
                               :fields fields :provenance {:program-fingerprint "unit-input"}
                               :numerical-contract {:mode :ieee-fp64 :determinism :toleranced
                                                                   :compatibility-id "unit"}})
        nodes (into {} (for [id [:coarse :fine]]
                         [id {:view (view/view (view/allocation {:id id :byte-size 32
                                                                 :memory-space :device :ownership :owned})
                                              {:id id :dtype :double :shape [4]})}]))
        local {:nodes nodes :outputs [:coarse :fine]
               :values (into {} (for [id [:coarse :fine]] [id {:leaves [{:node id}]}]))}
        interface {:outputs (mapv (fn [id] {:key id :node id :dtype :double :shape [4]})
                                 [:coarse :fine])}
        bindings {:done {:entry :cycle :link-plan local}}
        plan {:steps [{:id :done :kind :compute :device :worker}] :outputs [:done]
              :device-plans {:worker {:target :unit :entries {:cycle {:link-plan local}}
                                     :steps {:done {:entry :cycle}}}}}
        certified {:workload {:plan {:distributed-plan {:plan plan} :state {:manifest input}}}
                   :prepared {:lowering {:plan local}}
                   :certificate {:completion :done :entry :cycle :program {:fingerprint "unit"}
                                 :fields (into {} (for [id [:coarse :fine]]
                                                   [id {:field id :value id :node id :domain {:shape [2 2]}}]))}}
        owner (#'runtime/seal-runtime-value
               (runtime/map->DistributedExecutable {:plan plan :sessions {:unit (atom {})}
                                                     :bindings bindings
                                                     :state (atom :complete)}))
        reads (atom [])]
    (with-redefs [execution/verify! identity
                  compiled/plan (fn [prepared] (get-in prepared [:lowering :plan]))
                  compiled/prepared? (constantly true)
                  compiled/execution-identity (constantly {:scope :unit :fingerprint "unit"})
                  compiled/producer-interface (constantly interface)
                  plan/check-retained-output-readiness identity
                  plan/compute-bindings (constantly {:bindings bindings :unbound []})
                  runtime/output-values (constantly {:done {:coarse :coarse :fine :fine}})
                  runtime/storage-representation-description (fn [& _] {:dtype :double :byte-order :little-endian})
                  gpu/download-range!
                  (fn [_ id destination {:keys [src-element elements]}]
                    (swap! reads conj [id src-element elements])
                    (MemorySegment/copy (payloads id) (* 8 src-element) destination 0 (* 8 elements)))]
      (f certified owner input payloads reads))))

(deftest whole-cycle-capture-reuses-manifest-and-provider-contracts
  (fixture
   (fn [certified owner input payloads reads]
     (let [{:keys [provider blobs events]} (#'provider-fixture/provider
                                          (fn [& _]
                                            (is (= :reading-outputs @(:state owner)))
                                            (is (= :distributed-runtime-output-scope-active
                                                   (reason #(runtime/close! owner))))))
           result (capture/capture! certified owner {} provider :local
                                    {:id :next :logical-coordinate {:step 1 :phase :synchronized}})
           manifest (get-in result [:state :manifest])]
       (is (state/verify! (:state result)))
       (is (= [:input] (:parents manifest)))
       (is (= (:numerical-contract input) (:numerical-contract manifest)))
       (is (= :complete @(:state owner)))
       (is (empty? @events))
       (is (= 2 (count @blobs)))
       (is (seq @reads))
       (doseq [field (:fields manifest)]
         (let [source (some #(when (= (:id field) (:id %)) %) (:fields input))]
           (is (= (select-keys source [:value :coordinate-space :attributes])
                  (select-keys field [:value :coordinate-space :attributes]))))
         (is (= (content/content-address-of (payloads (:id field)))
                (get-in field [:chunks 0 :content]))))))))

(deftest invalid-owner-coordinate-and-provider-failure-return-no-state
  (fixture
   (fn [certified owner _ _ reads]
     (let [{:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))
           opts {:id :next :logical-coordinate {:step 1 :phase :synchronized}}]
       (is (= :amr-cycle-capture-owner
              (reason #(capture/capture! certified (assoc owner :plan (:plan owner)) {} provider :local opts))))
       (is (= :amr-cycle-capture-coordinate
              (reason #(capture/capture! certified owner {} provider :local (assoc opts :id :input)))))
       (is (empty? @reads)) (is (empty? @submissions))
       (let [error (ex-info "provider failed" {:reason :unit-provider})
             failed (#'provider-fixture/provider (fn [& _] (throw error)))]
         (is (= :unit-provider
                (reason #(capture/capture! certified owner {} (:provider failed) :local opts))))
         (is (= :complete @(:state owner))))))))

(deftest provider-callback-cannot-change-storage-and-return-a-successful-capture
  (fixture
   (fn [certified owner _ _ _]
     (let [changed? (atom false)
           {:keys [provider]} (#'provider-fixture/provider (fn [& _] (reset! changed? true)))]
       (with-redefs [runtime/storage-representation-description
                     (fn [& _]
                       (when @changed?
                         (throw (ex-info "changed snapshot" {:reason :distributed-representation-mismatch})))
                       {:dtype :double :byte-order :little-endian})]
         (is (= :distributed-representation-mismatch
                (reason #(capture/capture! certified owner {} provider :local
                                           {:id :next :logical-coordinate {:step 1 :phase :synchronized}}))))
         (is (= :complete @(:state owner))))))))

(deftest coordinates-cannot-add-drop-or-rewrite-unproved-facets
  (fixture
   (fn [certified owner _ _ reads]
     (let [{:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))
           opts {:id :next :logical-coordinate {:step 1 :phase :synchronized}}]
       (doseq [[input output] [[{:step 0 :phase :synchronized} {:step 1 :phase :synchronized :time 999}]
                               [{:step 0 :phase :synchronized :time 0} {:step 1 :phase :synchronized}]
                               [{:step 0 :phase :synchronized :time 0} {:step 1 :phase :synchronized :time 999}]
                               [{:step -1 :phase :synchronized} {:step 0 :phase :synchronized}]]]
         (is (= :amr-cycle-capture-coordinate
                (reason #(capture/capture! (assoc-in certified [:workload :plan :state :manifest :logical-coordinate] input)
                                           owner {} provider :local (assoc opts :logical-coordinate output))))))
       (is (empty? @reads)) (is (empty? @submissions))))))

(deftest malformed-field-and-late-provider-errors-never-return-a-manifest
  (fixture
   (fn [certified owner _ _ reads]
     (let [{:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))
           opts {:id :next :logical-coordinate {:step 1 :phase :synchronized}}
           other (#'runtime/seal-runtime-value
                  (runtime/map->DistributedExecutable (assoc (into {} owner) :plan {:id :other})))]
       (is (= :amr-cycle-capture-owner
              (reason #(capture/capture! certified other {} provider :local opts))))
       (is (= :amr-cycle-capture-field
              (reason #(capture/capture! (assoc-in certified [:certificate :fields :coarse :domain :shape] [1 4])
                                         owner {} provider :local opts))))
       (is (empty? @reads)) (is (empty? @submissions))
       (with-redefs [runtime/storage-representation-description
                     (fn [& _] (throw (ex-info "missing fact" {:reason :distributed-representation-mismatch})))]
         (is (= :distributed-representation-mismatch
                (reason #(capture/capture! certified owner {} provider :local opts)))))
       (let [awaits (atom 0)
             {:keys [provider blobs events]}
             (#'provider-fixture/provider
              (fn [& _]
                (when (= 2 (swap! awaits inc))
                  (throw (ex-info "second field failed" {:reason :unit-second-field})))))]
         (is (= :unit-second-field
                (reason #(capture/capture! certified owner {} provider :local opts))))
         (is (pos? (count @blobs)) "orphan content is permitted, but no manifest is returned")
         (is (empty? @events))
         (is (= :complete @(:state owner))))))))

(defn- capture-options [input]
  {:id :next :parents [:input] :logical-coordinate {:step 1 :phase :synchronized}
   :numerical-contract (:numerical-contract input)
   :fields (mapv (fn [field]
                   {:id (:id field) :step :done :key (:id field)
                    :value (:value field) :coordinate-space (:coordinate-space field)})
                 (:fields input))})

(deftest generic-adapter-still-rejects-specialized-sharded-schema
  (fixture
   (fn [certified owner input _ reads]
     (let [{:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))]
       (is (= :resident-state-distributed-field
              (reason #(resident/capture-distributed!
                        owner {[:worker :cycle] (:prepared certified)} {} provider :local
                        (capture-options input)))))
       (is (empty? @reads))
       (is (empty? @submissions))))))

(deftest shared-engine-certifies-final-manifest-before-provider-writes
  (fixture
   (fn [certified owner input _ _]
     (let [opts (capture-options input)
           failure (ex-info "builder failed" {:reason :unit-builder})]
       (doseq [[expected builder]
               [[:unit-builder (fn [_] (throw failure))]
                [:resident-state-manifest-fields
                 (fn [{:keys [fields provenance]}]
                   (state/manifest (assoc (dissoc opts :fields) :fields (vec (reverse fields))
                                          :provenance provenance)))]]]
         (let [{:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))
               error (try
                       (resident/capture-distributed-with-manifest!
                        owner {[:worker :cycle] (:prepared certified)} {} provider :local opts
                        (fn [metadata]
                          (is (= #{:fields :provenance} (set (keys metadata))))
                          (builder metadata)))
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
           (is (= expected (:reason (ex-data error))))
           (when (= :unit-builder expected) (is (identical? failure error)))
           (is (empty? @submissions))
           (is (= :complete @(:state owner)))))))))

(deftest shared-engine-revalidates-storage-after-manifest-builder
  (fixture
   (fn [certified owner input _ _]
     (let [changed? (atom false)
           opts (capture-options input)
           {:keys [provider submissions]} (#'provider-fixture/provider (fn [& _]))]
       (with-redefs [runtime/storage-representation-description
                     (fn [& _]
                       (when @changed?
                         (throw (ex-info "changed snapshot" {:reason :distributed-representation-mismatch})))
                       {:dtype :double :byte-order :little-endian})]
         (is (= :distributed-representation-mismatch
                (reason #(resident/capture-distributed-with-manifest!
                          owner {[:worker :cycle] (:prepared certified)} {} provider :local opts
                          (fn [{:keys [fields provenance]}]
                            (reset! changed? true)
                            (state/manifest (assoc (dissoc opts :fields) :fields fields :provenance provenance)))))))
         (is (empty? @submissions))
         (is (= :complete @(:state owner))))))))

(deftest amr-stops-before-next-field-after-provider-await-changes-storage
  (fixture
   (fn [certified owner _ _ _]
     (let [changed? (atom false)
           {:keys [provider submissions blobs events]}
           (#'provider-fixture/provider (fn [stage _] (when (= :await stage) (reset! changed? true))))]
       (with-redefs [runtime/storage-representation-description
                     (fn [& _]
                       (when @changed?
                         (throw (ex-info "changed snapshot" {:reason :distributed-representation-mismatch})))
                       {:dtype :double :byte-order :little-endian})]
         (is (= :distributed-representation-mismatch
                (reason #(capture/capture! certified owner {} provider :local
                                           {:id :next :logical-coordinate {:step 1 :phase :synchronized}}))))
         (is (= 1 (count @submissions)))
         (is (= 1 (count @blobs)))
         (is (empty? @events))
         (is (= :complete @(:state owner))))))))

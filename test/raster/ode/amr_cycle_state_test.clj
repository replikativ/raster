(ns raster.ode.amr-cycle-state-test
  "Hardware-free capture fault controls; compiler/device authority is tested independently."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.distributed :as runtime]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.ode.amr-cycle-state :as capture]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state-test :as provider-fixture])
  (:import [java.lang.foreign MemorySegment]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- fixture [f]
  (let [payloads {:coarse (MemorySegment/ofArray (double-array [1 2 3 4]))
                  :fine (MemorySegment/ofArray (double-array [5 6 7 8]))}
        fields (mapv (fn [id]
                       (state/field {:id id :value (av/tensor {:dtype :double :shape [2 2]})
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
        plan {:steps [{:id :done :device :worker}] :device-plans {:worker {:target :unit}}}
        nodes (into {} (for [id [:coarse :fine]]
                         [id {:view (view/view (view/allocation {:id id :byte-size 32
                                                                 :memory-space :device :ownership :owned})
                                              {:id id :dtype :double :shape [4]})}]))
        certified {:workload {:plan {:distributed-plan {:plan plan} :state {:manifest input}}}
                   :prepared {:lowering {:plan {:nodes nodes}}}
                   :certificate {:completion :done :program {:fingerprint "unit"}
                                 :fields (into {} (for [id [:coarse :fine]]
                                                   [id {:field id :value id :node id :domain {:shape [2 2]}}]))}}
        owner (#'runtime/seal-runtime-value
               (runtime/map->DistributedExecutable {:plan plan :sessions {:unit (atom {})}
                                                     :state (atom :complete)}))
        reads (atom [])]
    (with-redefs [execution/verify! identity
                  compiled/plan (fn [prepared] (get-in prepared [:lowering :plan]))
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

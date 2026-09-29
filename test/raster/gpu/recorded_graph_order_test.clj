(ns raster.gpu.recorded-graph-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]))

(deftest emitted-graph-order-reads-the-bound-recording
  (let [handle (gpu/->KernelGraphHandle :emitted)
        sess (atom {:closed? false :kernel-graphs
                    {:emitted {:prepareds [{:phase :producer} {:phase :consumer}]}}})]
    (is (= {:record-time-prologue []
            :per-replay [{:kernel-phase :producer} {:kernel-phase :consumer}]
            :completion :unproven}
           (gpu/kernel-graph-execution-order sess handle)))
    (swap! sess assoc :closed? true)
    (is (= :gpu-execution-order-closed
           (try (gpu/kernel-graph-execution-order sess handle)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest graph-order-separates-record-time-prologue-from-replay
  (let [calls (atom [])
        sess (atom {:device-id :ocl:0 :prepared
                    {:phase-a {:phase :a}
                     :phase-b {:phase :b :const-prologue? true}
                     :phase-c {:phase :c}}
                    :graphs {} :closed? false})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & _]
                                              (let [graph {:prepareds prepareds}]
                                                (swap! calls conj [:record (mapv :phase prepareds)])
                                                graph))
                            "replay-graph!" (fn [graph]
                                              (swap! calls conj [:replay
                                                                 (mapv :phase (:prepareds graph))]))
                            (throw (ex-info "unexpected runtime call" {:name name}))))
        resolve-soft (fn [_device-id name]
                       (case name
                         "destroy-graph!" (fn [graph]
                                            (swap! calls conj [:destroy
                                                               (mapv :phase (:prepareds graph))]))
                         nil))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolve-soft}
      (fn []
        (gpu/record-graph! sess [:phase-a :phase-b :phase-c] :graph)
        (is (= [[:record [:b]] [:replay [:b]] [:record [:a :c]]] @calls))
        (is (= {:record-time-prologue [{:phase :phase-b :kernel-phase :b}]
                :per-replay [{:phase :phase-a :kernel-phase :a}
                             {:phase :phase-c :kernel-phase :c}]
                :completion :unproven}
               (gpu/graph-execution-order sess :graph)))
        (let [linked (link/map->LinkedExecutable
                      {:plan {:id :model-plan :target :ocl:0
                              :instances [{:id :model
                                           :descriptor {:steps [{} {} {}]}}]}
                       :session sess :graph-key :graph :phases [:phase-a :phase-b :phase-c]
                       :pending-inputs (atom #{}) :closed? (atom false)})]
          (is (= [{:phase :phase-b :kernel-phase :b
                   :source {:instance :model :step 1}}]
                 (:record-time-prologue (link/execution-order linked))))
          (is (= [{:instance :model :step 0} {:instance :model :step 2}]
                 (mapv :source (:per-replay (link/execution-order linked))))))
        (gpu/replay! sess :graph)
        (is (= [:replay [:a :c]] (last @calls)))
        (gpu/release-recorded-graph! sess :graph)
        (is (= [[:destroy [:a :c]] [:destroy [:b]]]
               (take-last 2 @calls)))
        (is (= :gpu-execution-order-missing
               (try (gpu/graph-execution-order sess :graph)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest one-semantic-phase-can-split-between-prologue-and-replay
  (let [sess (atom {:device-id :ocl:0
                    :prepared {:phase (gpu/->BoundExecutableStep
                                       [{:phase :constant-transform :const-prologue? true}
                                        {:phase :value-kernel}]
                                       {} [])}
                    :graphs {} :closed? false})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & _] {:prepareds prepareds})
                            "replay-graph!" (fn [_])
                            (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) (fn [& _] nil)}
      (fn []
        (gpu/record-graph! sess [:phase] :graph)
        (is (= {:record-time-prologue
                [{:phase :phase :kernel-phase :constant-transform}]
                :per-replay [{:phase :phase :kernel-phase :value-kernel}]
               :completion :unproven}
               (gpu/graph-execution-order sess :graph)))))))

(deftest bound-phases-and-emitted-graphs-record-in-one-order
  (let [calls (atom [])
        handle (gpu/->KernelGraphHandle :emitted)
        sess (atom {:device-id :ocl:0 :closed? false
                    :prepared {:before {:phase :before}
                               :constant {:phase :constant :const-prologue? true}
                               :after {:phase :after}}
                    :kernel-graphs {:emitted {:prepareds [{:phase :producer}
                                                          {:phase :consumer}]}}
                    :graphs {}})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & [options]]
                                              (swap! calls conj [:record (mapv :phase prepareds) options])
                                              {:prepareds prepareds})
                            "replay-graph!" (fn [graph]
                                              (swap! calls conj [:replay
                                                                 (mapv :phase (:prepareds graph))]))
                            (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
       (fn [_device-id name]
         (when (= name "destroy-graph!")
           (fn [graph]
             (swap! calls conj [:destroy (mapv :phase (:prepareds graph))]))))}
      (fn []
        (gpu/record-bound-sequence!
         sess [{:kind :phase :phase :before}
               {:kind :graph :handle handle}
               {:kind :phase :phase :constant}
               {:kind :phase :phase :after}]
         :mixed {:profile? true})
        (is (= [[:record [:before :producer :consumer :constant :after]
                 {:barriers? true :profile? true}]]
               @calls))
        (is (= {:record-time-prologue []
                :per-replay [{:phase :before :kernel-phase :before}
                             {:phase handle :kernel-phase :producer}
                             {:phase handle :kernel-phase :consumer}
                             {:phase :constant :kernel-phase :constant}
                             {:phase :after :kernel-phase :after}]
                :completion :unproven}
               (gpu/graph-execution-order sess :mixed)))
        (let [before @calls]
          (is (= :gpu-recording-invalid-source
                 (try (gpu/record-bound-sequence!
                       sess [{:kind :graph :phase :before}] :mixed)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= before @calls))
          (is (= :gpu-recording-unbound-phase
                 (try (gpu/record-bound-sequence!
                       sess [{:kind :phase :phase :missing}] :mixed)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= before @calls)))
        (gpu/release-recorded-graph! sess :mixed)
        (is (= [:destroy [:before :producer :consumer :constant :after]]
               (last @calls)))))))

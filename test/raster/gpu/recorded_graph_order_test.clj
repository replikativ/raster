(ns raster.gpu.recorded-graph-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]))

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
                      {:session sess :graph-key :graph :phases [:phase-a :phase-b :phase-c]
                       :pending-inputs (atom #{}) :closed? (atom false)})]
          (is (= (gpu/graph-execution-order sess :graph)
                 (link/execution-order linked))))
        (gpu/replay! sess :graph)
        (is (= [:replay [:a :c]] (last @calls)))
        (gpu/release-recorded-graph! sess :graph)
        (is (= [[:destroy [:a :c]] [:destroy [:b]]]
               (take-last 2 @calls)))
        (is (= :gpu-execution-order-missing
               (try (gpu/graph-execution-order sess :graph)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

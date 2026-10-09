(ns raster.gpu.distributed-storage-admission-test
  "Hardware-free admission checks; synthetic kernels are never submitted."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.distributed-storage-plan-test :as fixture]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]))

(defn- scratch-plan []
  (let [local (assoc-in (#'fixture/local-plan) [:nodes :x-node :source] (float-array 6))
        kernel (get-in local [:instances 0 :descriptor :steps 0 :artifact])
        executable
        (graph/make
         {:inputs [(graph/buffer 'x :float 6 :device :input)
                   (graph/buffer 'weights :float 6 :device :input)]
          :outputs [(graph/buffer 'y :float 6 :device :output)]
          :temporaries [(graph/buffer 'scratch :float 24 :device :temporary)]
          :scalars [(graph/scalar 'n :long)]
          :nodes [(graph/->ScheduledKernel
                   :copy kernel [(graph/->ValueUse 'x :read)
                                 (graph/->ValueUse 'weights :read)
                                 (graph/->ValueUse 'y :write)] #{'n} [])]
          :abi (:abi kernel) :arguments (:arguments kernel)})]
    (#'fixture/one-call
     (assoc-in local [:instances 0 :descriptor :steps 0 :artifact] executable))))

(deftest rejected-storage-budgets-acquire-no-runtime-resources
  (let [plan (scratch-plan)]
    (doseq [[options expected]
            [[{:device-capacities {:gpu-0 71}} :distributed-runtime-memory]
             [{:device-capacities {:gpu-0 167} :include-graph-temporaries? true}
              :distributed-runtime-memory]
             [{:include-graph-temporaries? :unknown} :distributed-runtime-physical-budget]]]
      (let [effects (atom [])
            unexpected (fn [operation]
                         (fn [& _]
                           (swap! effects conj operation)
                           (throw (ex-info "admission acquired a runtime resource"
                                           {:reason :unexpected-runtime-resource}))))]
        (with-redefs [gpu/make-session (unexpected :session)
                      gpu/alloc! (unexpected :allocation)
                      gpu/upload-range! (unexpected :upload)]
          (let [error (try (runtime/instantiate! plan options)
                           nil
                           (catch clojure.lang.ExceptionInfo e (ex-data e)))]
            (is (= expected (:reason error)) (str options))
            (when (= 167 (get-in options [:device-capacities :gpu-0]))
              (is (= 168 (:bytes error))))
            (is (empty? @effects))))))))

(deftest accepted-storage-budgets-reach-the-session-boundary
  ;; Stop at the first resource operation. This proves admission, not successful
  ;; device execution or a total-memory bound. Native training tests cover execution.
  (let [plan (scratch-plan)]
    (doseq [options [{:device-capacities {:gpu-0 72}}
                     {:device-capacities {:gpu-0 168} :include-graph-temporaries? true}]]
      (let [opened (atom [])]
        (with-redefs [gpu/make-session
                      (fn [target]
                        (swap! opened conj target)
                        (throw (ex-info "stop before acquiring a session"
                                        {:reason :test-session-boundary})))]
          (is (= :test-session-boundary
                 (try (runtime/instantiate! plan options)
                      nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= [:gpu-0] @opened)))))))

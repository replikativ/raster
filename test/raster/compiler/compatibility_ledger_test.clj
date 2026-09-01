(ns raster.compiler.compatibility-ledger-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is use-fixtures]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.attention :as attention]
            [raster.linalg.contract :as contract]
            [raster.nn :as nn]
            [raster.ode.pde :as pde]
            [raster.quant.kernels-k :as qk]
            [raster.runtime.hardware :as hardware]))

(def ^:private ledger-path "test/raster/compiler/compatibility_ledger.edn")
(def ^:private target :ze:compatibility-ledger)

(use-fixtures
  :once
  (fn [f]
    (hardware/register-target-device!
     target
     {:name "Synthetic Intel GPU for compiler compatibility ledger"
      :capabilities {:total-eus 64
                     :threads-per-eu 8
                     :simd-width 16
                     :subgroup-sizes [16 32]
                     :max-workgroup-size 1024
                     :shared-local-memory 131072}})
    (f)))

(defn- frequencies-by
  [key-fn xs]
  (frequencies (map key-fn xs)))

(defn- signature
  [report]
  {:route (-> (:route report)
              (update :declines
                      #(frequencies-by (juxt :stage :kind :reason) %)))
   :fusion (:fusion report)
   :lowering (:lowering report)
   :emission (-> (:emission report)
                 (update :declines #(frequencies-by (juxt :leaf :reason) %)))})

(defn- equation-first-signature
  [compilation plan]
  {:route {:semantic-dialect (get-in compilation [:semantic :dialect])
           :scheduled-dialect (get-in compilation [:scheduled :dialect])
           :emitted-dialect (get-in compilation [:emitted :dialect])
           :fallback (get-in compilation [:stats :fallback])}
   :lowering {:nodes (count (:nodes plan))
              :values (count (:values plan))
              :instances (count (:instances plan))
              :program-instances (count (filter link-plan/program-link-instance?
                                                (:instances plan)))
              :outputs (count (:outputs plan))
              :driver-allocations (get-in plan [:attributes :driver-allocations])}
   :emission (get-in compilation [:stats :emission])})

(defn- compile-workload
  [id]
  (case id
    :dense-relu-jvm
    (pipeline/compile-report #'nn/predict-fn)

    :symbolic-dense-contraction-gpu
    (pipeline/compile-report #'contract/contract-mm :target-device target :dtype :float)

    :q4k-dp4a-rows-gpu
    (pipeline/compile-report #'qk/qmatmul-q4k-dp4a-rows!
                             :target-device target :dtype :float)

    :gqa-causal-mha-gpu
    (let [compilation (equation-first/compile
                       #'attention/gqa-causal-mha {:target target :dtype :float})
          plan (equation-first/lower
                compilation
                [(float-array [1 0 0 1 1 1 1 -1])
                 (float-array [1 0 0 1])
                 (float-array [1 2 3 4])
                 1 2 2 1 2])]
      (equation-first-signature compilation plan))

    :heat-rhs-1d-jvm
    (pipeline/compile-report #'pde/heat-rhs-1d! :dtype :double)

    :heat-loss-rk4-gpu
    (let [compilation (equation-first/compile
                       #'pde/heat-loss-rk4 {:target target :dtype :double})
          plan (equation-first/lower
                compilation [(double-array 8) (double-array 8) 0.1 1.0 0.01 3])]
      (equation-first-signature compilation plan))

    (throw (ex-info "compatibility ledger has no workload compiler"
                    {:workload id}))))

(deftest representative-workloads-have-exact-compiler-signatures
  (let [{:keys [schema-version workloads]} (edn/read-string (slurp ledger-path))]
    (is (= 1 schema-version))
    (is (= #{:dense-relu-jvm
             :symbolic-dense-contraction-gpu
             :q4k-dp4a-rows-gpu
             :gqa-causal-mha-gpu
             :heat-rhs-1d-jvm
             :heat-loss-rk4-gpu}
           (set (keys workloads))))
    (doseq [[id expected] workloads]
      (let [report (compile-workload id)
            actual (cond
                     (:declined report) report
                     (contains? #{:gqa-causal-mha-gpu :heat-loss-rk4-gpu} id) report
                     :else (signature report))]
        (is (= expected actual)
            (str "compiler compatibility changed for " id
                 "; improve the ledger downward or explain the new debt"))
        (when (= :heat-rhs-1d-jvm id)
          (is (= 1 (get-in report [:pass-stats :backend-applied-stats :simd-stencils]))
              "the scientific stencil must consume its scheduled SegStencil exactly once"))))))

(ns raster.compiler.compatibility-ledger-test
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is use-fixtures]]
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
    (pipeline/compile-report #'attention/gqa-causal-mha
                             :target-device target :dtype :float)

    :heat-rhs-1d-jvm
    (pipeline/compile-report #'pde/heat-rhs-1d! :dtype :double)

    (throw (ex-info "compatibility ledger has no workload compiler"
                    {:workload id}))))

(deftest representative-workloads-have-exact-compiler-signatures
  (let [{:keys [schema-version workloads]} (edn/read-string (slurp ledger-path))]
    (is (= 1 schema-version))
    (is (= #{:dense-relu-jvm
             :symbolic-dense-contraction-gpu
             :q4k-dp4a-rows-gpu
             :gqa-causal-mha-gpu
             :heat-rhs-1d-jvm}
           (set (keys workloads))))
    (doseq [[id expected] workloads]
      (is (= expected (signature (compile-workload id)))
          (str "compiler compatibility changed for " id
               "; improve the ledger downward or explain the new debt")))))

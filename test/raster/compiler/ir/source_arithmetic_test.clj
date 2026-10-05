(ns raster.compiler.ir.source-arithmetic-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.edn :as edn]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.contract-lower :as lower]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.typed-contraction-context :as context]))

(deftest source-arithmetic-is-closed-and-does-not-grant-refinement-consent
  (is (numerics/source-arithmetic? numerics/retained-source-arithmetic))
  (doseq [dtype [:float :double]]
    (let [contract (numerics/blas-source-arithmetic dtype)]
      (is (= contract (edn/read-string (pr-str contract))))
      (is (= contract (numerics/validate-source-arithmetic! contract)))
      (doseq [forged [(assoc contract :permitted-modes #{:reassociated})
                      (assoc contract :precision :mixed-f16-f32)
                      (assoc-in contract [:operands :conversion] :narrow)
                      (assoc-in contract [:accumulation :minimum-dtype] :half)
                      (assoc-in contract [:accumulation :scope] :whole-operation)
                      (assoc-in contract [:result-transform] :implementation-defined)
                      (update contract :operands dissoc :conversion)]]
        (is (not (numerics/source-arithmetic? forged)))
        (is (thrown? clojure.lang.ExceptionInfo
                     (numerics/validate-source-arithmetic! forged))))))
  (doseq [bad [nil {} {:kind :retained-typed-ssa :minimum-dtype :float}]]
    (is (not (numerics/source-arithmetic? bad))))
  (is (thrown? clojure.lang.ExceptionInfo (numerics/blas-source-arithmetic :half))))

(deftest contraction-provenance-survives-the-existing-projection-spine
  (doseq [dtype [:float :double]]
    (let [components {:out 'out :free-axes [['i 2]] :contract-axes [['k 3]]
                      :dtype dtype :body '(aget a (+ (* i 3) k))
                      :local-identities {:accumulator 'acc}}
          ordered (facts/from-components components)
          contract (numerics/blas-source-arithmetic dtype)
          projected (facts/from-components (assoc components :source-arithmetic contract))
          round-trip (facts/contraction-facts (facts/surface-form projected) :dtype dtype)]
      (is (= numerics/retained-source-arithmetic (:source-arithmetic ordered)))
      (is (= contract (:source-arithmetic projected) (:source-arithmetic round-trip)))
      (is (not (fingerprint/equivalent? ordered projected)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (facts/from-components
                    (assoc components :source-arithmetic
                           (numerics/blas-source-arithmetic (if (= :float dtype) :double :float)))))))))

(deftest source-arithmetic-is-retained-through-the-scheduled-artifact-certificate
  (let [product (with-meta
                  '(raster.par/contract C [[i 2] [j 4]] [[k 3]]
                     (* (aget A (+ (* i 3) k)) (aget B (+ (* k 4) j))))
                  {:raster.source/arithmetic :abstract-blas-product
                   :raster.type/elem-type :float})
        source (list 'let* ['result product] 'result)
        {:keys [form]} (pipeline/schedule-parallel-form
                        source {:dtype :float :target-device :ocl:0
                                :array-types {'A :float 'B :float 'C :float}})
        equation (first (:equations form))
        graph (:graph (equation-graph/make-for-equation form equation))
        node (first (:nodes graph))
        algorithm (:algorithm equation)
        verified (:facts (context/validate! algorithm (:operation node)))
        refinement (schedule/schedule-portable-for-node node graph verified {} {})
        emitted (target/emit-artifact "source_arithmetic" refinement :opencl-portable)
        contract (numerics/blas-source-arithmetic :float)]
    (is (= contract
           (get-in (dialect/operation-parts (first (dialect/equations algorithm)))
                   [:attributes :source-arithmetic])
           (:source-arithmetic verified)
           (:source-arithmetic (:operation node))
           (:source-arithmetic (lower/contraction-facts->segred verified))
           (get-in refinement [:numerics :source-arithmetic])
           (get-in emitted [:provenance :scheduled-operation :numerics :source-arithmetic])))
    (is (= emitted (scheduled/validate-artifact-projection! refinement emitted)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (scheduled/validate-artifact-projection!
                  refinement
                  (assoc-in emitted [:provenance :scheduled-operation :numerics :source-arithmetic]
                            numerics/retained-source-arithmetic))))))

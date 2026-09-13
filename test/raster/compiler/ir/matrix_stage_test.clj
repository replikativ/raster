(ns raster.compiler.ir.matrix-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]))

(defn- cast-region []
  (body/->ScalarSSARegion
   ['loaded] [] [] :float
   [(body/->ScalarCompute (body/value 'converted :half)
                          (body/cast-expression 'loaded :half :nearest-even :ieee))]
   'converted :half))

(defn- stage-spec []
  {:id [:matrix-input-stage :contract] :lhs 'A :rhs 'B :result 'C
   :dimensions [13 32 32] :result-shape [13 32]
   :reduction {:kind :full :range [0 32]}
   :schedule {:kind :matrix-instruction-tiling
              :tile {:block-m 16 :block-n 32 :sg-m 8 :sg-n 16 :block-k 32 :num-stages 1
                     :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}}})

(deftest matrix-stage-keeps-fragment-types-separate-from-physical-inputs
  (let [plain (matrix-stage/make (stage-spec))
        region (cast-region)
        converted (matrix-stage/make (assoc (stage-spec) :input-value-regions {'A region}))]
    (is (= {} (:input-value-regions plain)))
    (is (= 'A (:rhs (matrix-stage/make (assoc (stage-spec) :rhs 'A)))))
    (is (= :half (:operand-dtype converted)))
    (is (= :float (get-in converted [:input-value-regions 'A :accumulator-dtype])))
    (is (identical? region (get-in converted [:input-value-regions 'A])))
    (is (= region (body/validate-input-value-region! region :float :half)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (matrix-stage/make (assoc (stage-spec) :input-value-regions {'missing region}))))
    (doseq [bad [(assoc region :result-dtype :float)
                 (assoc region :parameters ['loaded 'capture])
                 (assoc region :indices ['i])
                 (assoc region :result 'missing)
                 (assoc-in region [:operations 0 :expression :arguments] ['outside])
                 (assoc-in region [:operations 0 :expression :options :rounding] nil)
                 (assoc region :operations
                        [(body/->ScalarLoad (body/value 'converted :half) 'B [0 0]
                                            nil nil :cached)])]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (matrix-stage/make (assoc (stage-spec) :input-value-regions {'A bad})))))))

(deftest transformed-matrix-stage-reaches-the-certified-body-and-physical-abi
  (let [stage (matrix-stage/make (assoc (stage-spec) :input-value-regions {'A (cast-region)}))
        uses [(graph/->ValueUse 'A :read) (graph/->ValueUse 'B :read) (graph/->ValueUse 'C :write)]
        original (graph/make
                  {:inputs [(graph/buffer 'A :float 416 :device :input)
                            (graph/buffer 'B :half 1024 :device :input)]
                   :outputs [(graph/buffer 'C :float 416 :device :output)]
                   :temporaries [] :scalars []
                   :nodes [(graph/->ScheduledKernel (:id stage) stage uses #{} [])]
                   :abi [(abi/slot 'A :input :float) (abi/slot 'B :input :half)
                         (abi/slot 'C :output :float)]
                   :arguments '[A B C]
                   :effects {:reads ['A 'B] :writes ['C]}})
        emitted (gemm/emit-scheduled-stage-graph original
                                                {:prefix "matrix_input_stage" :target-dialect :opencl-intel})
        node (first (:nodes emitted))
        a (:operation node)
        scheduled (artifact/attribute a :scheduled-kernel-body)
        kernel (:body scheduled)]
    (is (= (graph/boundary-contract original) (graph/boundary-contract emitted)))
    (is (= (graph/dataflow-contract original) (graph/dataflow-contract emitted)))
    (is (identical? stage (:source scheduled)))
    (is (= [:float :half :float] (mapv :dtype (take 3 (:abi a)))))
    (is (= :no-write-alias (:aliasing (first (:abi a)))))
    (is (= :float (get-in kernel [:parameters 0 :dtype])))
    (is (re-find #"convert_half_rte" (:source a)))
    (is (= scheduled (scheduled-body/validate-against-node! scheduled (first (:nodes original)) original)))
    (is (= a (scheduled-body/validate-artifact-projection! scheduled a)))
    (is (= [] (:temporaries emitted)))))

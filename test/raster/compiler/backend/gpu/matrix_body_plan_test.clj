(ns raster.compiler.backend.gpu.matrix-body-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-opencl :as opencl]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(defn- matrix-body
  [family]
  (let [matrix (case family
                 :mma {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
                 {:family family :m 8 :n 16 :k 16 :subgroup 16})]
    (schedule/matrix-body
     {:id [:matrix-plan-test family]
      :row 'a :col 'b :out 'c
      :dimensions [64 64 64]
      :tile (assoc (hardware/derive-gemm-tile {}) :matrix matrix)})))

(deftest matrix-targets-cannot-ignore-a-typed-input-transformation
  (let [region (body/->ScalarSSARegion
                ['element] [] [] :half
                [(body/->ScalarCompute (body/value 'twice :half)
                                       (body/scalar-expression :+ :half ['element 'element]))]
                'twice :half)
        kernel (walk/postwalk
                (fn [node]
                  (if (instance? raster.compiler.ir.kernel_body.TileLoad node)
                    (assoc node :value-region region) node))
                (matrix-body :dpas))]
    (is (= kernel (body/validate! kernel)))
    (is (nil? (opencl/lower-uniform-store-region kernel {} :cuda))
        "an input transformation is not a store epilogue")
    (doseq [target [:opencl-intel :cuda :hip]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"transformed tile input"
                            (matrix-target/emit-matrix-kernel "transformed_input" kernel target))))))

(deftest matrix-plan-is-neutral-over-instruction-families
  (let [plan (matrix-plan/analyze (matrix-body :mma))]
    (is (= {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
           (:instruction plan)))
    (is (= [128 128 32 32]
           ((juxt :block-m :block-n :sg-m :sg-n) plan)))
    (is (= [16 16 16 32]
           ((juxt :mi :ni :ki :subgroup) plan)))))

(deftest matrix-k-arithmetic-requires-widening-before-computation
  (let [kernel (matrix-body :dpas)
        narrow (walk/postwalk
                (fn [node]
                  (cond
                    (instance? raster.compiler.ir.kernel_body.IndexCast node) (:argument node)
                    (instance? raster.compiler.ir.kernel_body.ForLoop node)
                    (assoc-in node [:index :type] :int)
                    :else node)) kernel)
        late (update-in kernel [:operations 0 :operations]
                        (fn [operations]
                          (mapv (fn [op]
                                  (if (instance? raster.compiler.ir.kernel_body.ForLoop op)
                                    (assoc op :lower
                                           (body/index-cast (body/expression :add 0 0) :long :exact))
                                    op)) operations)))]
    (is (= :long (:index-dtype (matrix-plan/analyze kernel))))
    (is (body/kernel-body? (body/validate! narrow)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires long induction"
                          (matrix-plan/analyze narrow)))
    (is (body/kernel-body? (body/validate! late)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"widen leaves before arithmetic"
                          (matrix-plan/analyze late)))))

(deftest matrix-plan-rejects-structure-hidden-by-set-comparisons
  (let [kernel (matrix-body :dpas)]
    (testing "duplicate matrix operations cannot replace a missing pair"
      (let [bad
            (update-in
             kernel [:operations 0 :operations]
             (fn [operations]
               (mapv
                (fn [operation]
                  (if (instance? raster.compiler.ir.kernel_body.ForLoop operation)
                    (update-in
                     operation [:operations 0 :operations]
                     (fn [inner]
                       (let [mad (first (filter #(instance?
                                                 raster.compiler.ir.kernel_body.MatrixMad %)
                                               inner))]
                         (conj (pop inner) mad (peek inner)))))
                    operation))
                operations)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment product"
                              (matrix-plan/analyze bad)))))
    (testing "launch geometry is a checked consequence of the body topology"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"launch does not match"
                            (matrix-plan/analyze
                             (assoc-in kernel [:launch :workgroup-size 0] 1)))))
    (testing "an emitter cannot assume row-major storage from a stale permutation"
      (let [shape (get-in kernel [:parameters 0 :shape])]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exact row-major"
                              (matrix-plan/analyze
                               (assoc-in kernel [:parameters 0 :layout]
                                         (layout/col-major shape :half)))))))
    (testing "runtime dimension identities are specialized to the storage contract"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dimension specializations"
                            (matrix-plan/analyze
                             (assoc-in kernel [:attributes :dimension-values 'M] 63)))))
    (testing "restrict-qualified matrix inputs require stable read contracts"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stable no-write-alias"
                            (matrix-plan/analyze (assoc kernel :stable-reads [])))))))

(deftest intel-opencl-retains-its-own-instruction-legality
  (testing "neutral MMA analysis does not make an Intel DPAS emitter accept MMA"
    (try
      (opencl/emit-matrix-kernel "mma_is_not_dpas" (matrix-body :mma))
      (is false "Intel OpenCL lowering unexpectedly accepted an MMA instruction")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-body-opencl-unimplemented (:reason (ex-data exception))))
        (is (= :mma (get-in (ex-data exception) [:instruction :family])))))))

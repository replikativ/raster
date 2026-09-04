(ns raster.compiler.backend.gpu.matrix-body-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as opencl]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.layout :as layout]
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

(deftest matrix-plan-is-neutral-over-instruction-families
  (let [plan (matrix-plan/analyze (matrix-body :mma))]
    (is (= {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
           (:instruction plan)))
    (is (= [128 128 32 32]
           ((juxt :block-m :block-n :sg-m :sg-n) plan)))
    (is (= [16 16 16 32]
           ((juxt :mi :ni :ki :subgroup) plan)))))

(deftest matrix-plan-rejects-structure-hidden-by-set-comparisons
  (let [kernel (matrix-body :dpas)]
    (testing "duplicate matrix operations cannot replace a missing pair"
      (let [bad
            (update-in
             kernel [:operations 0 :operations]
             (fn [operations]
               (mapv
                (fn [operation]
                  (if (instance? raster.compiler.ir.kernel_body.Loop operation)
                    (update-in
                     operation [:operations 0 :operations]
                     (fn [inner]
                       (conj inner (first (filter #(instance?
                                                   raster.compiler.ir.kernel_body.MatrixMad %)
                                                 inner)))))
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
                             (assoc-in kernel [:attributes :dimension-values 'M] 63)))))))

(deftest intel-opencl-retains-its-own-instruction-legality
  (testing "neutral MMA analysis does not make an Intel DPAS emitter accept MMA"
    (try
      (opencl/emit-matrix-kernel "mma_is_not_dpas" (matrix-body :mma))
      (is false "Intel OpenCL lowering unexpectedly accepted an MMA instruction")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-body-opencl-unimplemented (:reason (ex-data exception))))
        (is (= :mma (get-in (ex-data exception) [:instruction :family])))))))

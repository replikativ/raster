(ns raster.compiler.passes.parallel.gpu-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.gpu-plan :as gpu-plan]
            [raster.compiler.passes.parallel.patterns :as patterns]))

(deftest detect-do-wrapped-transpose-pattern
  (testing "Shared pattern matcher detects do-wrapped transpose loops"
    (let [form '(do "transpose"
                    (dotimes [i rows]
                      (dotimes [j cols]
                        (clojure.core/aset out
                                           (+ (* j rows) i)
                                           (clojure.core/aget src (+ (* i cols) j)))))
                    out)
          match (patterns/match-do-wrapped-transpose form)]
      (is (= {:src 'src
              :out 'out
              :out-buf 'out
              :rows 'rows
              :cols 'cols}
             match)))))

(deftest rewrite-do-wrapped-transpose-to-gpu-kernel
  (testing "gpu-plan rewrites transpose loops to registered transpose kernels"
    (let [form '(let* [xt (do "transpose"
                              (dotimes [i rows]
                                (dotimes [j cols]
                                  (clojure.core/aset out
                                                     (+ (* j rows) i)
                                                     (clojure.core/aget src (+ (* i cols) j)))))
                              out)]
                      xt)
          {:keys [form kernels stats]} (gpu-plan/gpu-plan-pass form :dtype :float)
          bindings (second form)
          rewritten-expr (second bindings)]
      (is (= 1 (:transpose-rewrites stats)))
      (is (= 1 (count kernels)))
      (is (= :transpose (:type (first kernels))))
      (is (seq? rewritten-expr))
      (is (= 'raster.gpu.ze-runtime/invoke-registered-transpose!
             (first rewritten-expr)))
      (is (= 'src (nth rewritten-expr 2)))
      (is (= 'out (nth rewritten-expr 3)))
      (is (= 'rows (nth rewritten-expr 4)))
      (is (= 'cols (nth rewritten-expr 5))))))
;; ── #38: a GEMM redomap (matmul in the raster language, no BLAS call) is rewritten
;; to the SAME registered-GEMM kernel path as a blas/dgemm! call — the two front doors
;; converge. Device-free: gpu-plan-pass PLANS kernels (form→form), it does not execute.
(deftest rewrite-gemm-redomap-to-gpu-kernel
  (testing "a nested-dotimes matmul redomap is recognized and offloaded to invoke-registered-gemm!"
    (let [form '(let* [C   (clojure.core/double-array (clojure.core/* m n))
                       out (do (dotimes [i m]
                                 (dotimes [j n]
                                   (aset C (+ (* i n) j)
                                         (loop [acc 0.0 p 0]
                                           (if (< p k)
                                             (recur (+ acc (* (aget A (+ (* i k) p)) (aget B (+ (* p n) j)))) (inc p))
                                             acc)))))
                               C)]
                  out)
          {:keys [form kernels stats]} (gpu-plan/gpu-plan-pass form :dtype :float)]
      (is (= 1 (:gemm-rewrites stats)) "the redomap counts as a GEMM rewrite, like a BLAS call")
      (is (= 1 (count kernels)))
      (is (re-find #"invoke-registered-gemm!" (pr-str form))
          "emits the same registered-GEMM invocation the BLAS path uses"))))

(deftest non-gemm-redomap-is-not-offloaded
  (testing "a reduction that is NOT a GEMM (single-array sum) is left for the SegOp backend"
    (let [form '(let* [C   (clojure.core/double-array (clojure.core/* m n))
                       out (do (dotimes [i m]
                                 (dotimes [j n]
                                   (aset C (+ (* i n) j)
                                         (loop [acc 0.0 p 0]
                                           (if (< p k)
                                             (recur (+ acc (aget A (+ (* i k) p))) (inc p))
                                             acc)))))
                               C)]
                  out)
          {:keys [stats]} (gpu-plan/gpu-plan-pass form :dtype :float)]
      (is (= 0 (:gemm-rewrites stats)) "not a matmul → no GEMM offload (sound)"))))

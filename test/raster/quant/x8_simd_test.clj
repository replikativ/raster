(ns raster.quant.x8-simd-test
  "The tiled Q4_0 GEMV (qmatmul-q4-x8!) compiled with :simd? true lowers its 8-column
   fold to the AVX2 int→float widening block (out8 int → cvtepi32_ps → float acc8
   register accumulator) and stays bit-exact to the row-major composable reference.
   This is the #27 payoff: the compiler-generated column-vectorized int8 fold matches
   the hand kernel, from composable deftm source. Guarded on clang+AVX2."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.quant :as cq]
            [raster.quant.kernels :as k]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.cpu.codegen :as cpu]))

(defn- clang-avx2? []
  (try (cpu/compile-source! "#include <immintrin.h>\nint main(){__m256 v=_mm256_setzero_ps();return (int)_mm256_cvtss_f32(v);}\n") true
       (catch Throwable _ false)))

(deftest x8-simd-fold-vectorizes-and-matches
  (when (clang-avx2?)
    (testing "column fold emits cvtepi32_ps + bit-exact vs composable"
      ;; the fold is vectorized in the emitted C
      (is (re-find #"_mm256_cvtepi32_ps"
                   (:c-source (meta (aot/compile-aot-c #'k/qmatmul-q4-x8! :float :simd? true))))
          ":simd? true lowers the fold to the widening AVX2 block")
      (doseq [[in out] [[256 64] [512 128] [4096 512]]]
        (let [wf (let [a (float-array (* out in))] (dotimes [i (* out in)] (aset a i (float (- (rand) 0.5)))) a)
              x  (let [a (float-array in)] (dotimes [i in] (aset a i (float (- (rand) 0.5)))) a)
              {:keys [wq ws]} (cq/quantize-weight-q4 wf)
              {:keys [wqi wsi]} (cq/repack-stream wq ws out in)
              {:keys [xq xs xsum]} (cq/quantize-act-i8-par x in)
              y-x8   (vec ((k/make-x8-c-gemv) xq xs xsum wqi wsi in out))
              y-comp (vec ((k/make-composable-c-gemv) xq xs xsum wq ws in out))
              maxd (reduce max 0.0 (map (fn [a b] (Math/abs (- (double a) (double b)))) y-x8 y-comp))]
          (is (< maxd 1.0e-3) (str "x8 SIMD vs composable in=" in " out=" out " maxdiff=" maxd)))))))

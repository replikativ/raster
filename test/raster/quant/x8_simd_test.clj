(ns raster.quant.x8-simd-test
  "The tiled Q4_0 GEMV retains its explicit Double scale arithmetic even with SIMD
   requested. Float output storage is not permission to narrow intermediate products.
   The integer-dot target override remains available; the mixed-precision fold needs
   a precision-preserving vector schedule. Guarded on clang+AVX2."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.quant :as cq]
            [raster.quant.kernels :as k]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.cpu.codegen :as cpu]))

(defn- clang-avx2? []
  (try (cpu/compile-source! "#include <immintrin.h>\nint main(){__m256 v=_mm256_setzero_ps();return (int)_mm256_cvtss_f32(v);}\n") true
       (catch Throwable _ false)))

(deftest registered-x8-helper-override-does-not-emit-its-reference-body
  (when (clang-avx2?)
    (with-redefs [c-emit/generate-c-helper
                  (fn [_]
                    (throw (ex-info "target override emitted the discarded reference body" {})))]
      (is (re-find #"gpufn_wi8_dot_q4_x8"
                   (:c-source
                    (meta (aot/compile-aot-c #'k/qmatmul-q4-x8! :float :simd? true))))))))

(deftest x8-fold-keeps-double-arithmetic-and-matches
  (when (clang-avx2?)
    (testing "explicit Double fold must not become an all-Float vector fold"
      (is (not (re-find #"_mm256_cvtepi32_ps"
                        (:c-source (meta (aot/compile-aot-c #'k/qmatmul-q4-x8! :float :simd? true)))))
          "mixed precision declines rather than silently narrowing; homogeneous Float widening is tested in csimd-test")
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

(deftest x8-double-scale-rounding-is-independent-of-simd-request
  (when (clang-avx2?)
    (let [native-simd (aot/compile-aot-c #'k/qmatmul-q4-x8! :float :simd? true)
          native-scalar (aot/compile-aot-c #'k/qmatmul-q4-x8! :float :simd? false)
          xq (byte-array 32)
          ;; Each output dot is 9*3, with zero-point correction 8*3: folded=3.
          wqi (byte-array (repeat 128 (unchecked-byte 0x99)))
          xsum (int-array [3])]
      (aset-byte xq 0 (byte 3))
      (doseq [[a b] [[(float 1.0000001) (float 0.3)]
                    [(float 1.0000001) (float 2.1)]
                    [(float 0.00003) (float 0.3)]]]
        (let [xs (float-array [a])
              wsi (float-array (repeat 8 b))
              expected (float (* (* (double a) (double b)) 3.0))
              narrowed (float (* (float (* a b)) (float 3)))
              bits #(Float/floatToRawIntBits (float %))]
          (is (not= (bits expected) (bits narrowed))
              "fixture distinguishes retained Double multiplication from early Float rounding")
          (doseq [kernel [k/qmatmul-q4-x8! native-scalar native-simd]]
            (let [y (float-array 8)]
              (kernel xq xs xsum wqi wsi y 32 8 0 1)
              (is (= (vec (repeat 8 (bits expected))) (mapv bits y))
                  "JVM, scalar native and SIMD-requested native retain the same Double fold"))))))))

(ns raster.compiler.csimd-test
  "The explicit CPU-C SIMD emitter (compile-segred-c) turns a SegRed reduction
   into AVX2 __m256 intrinsic C that matches the scalar reduction bit-for-bit
   across dtypes and non-multiple-of-stride sizes. This is the reusable core of
   #27 (the C analog of jvm/segop_simd), validated in isolation before the
   pipeline wiring. Guarded on clang + a machine that runs AVX2."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.csimd :as cs]
            [raster.compiler.backend.cpu.codegen :as cpu]
            [raster.compiler.backend.gpu.c-emit :as ce]))

(defn- clang-avx2? []
  (try
    (let [src (str cs/simd-includes "int main(){__m256d v=_mm256_setzero_pd();return (int)_mm256_cvtsd_f64(v);}\n")]
      (cpu/compile-source! src) true)
    (catch Throwable _ false)))

(defn- ssq-segred [dt]
  {:space {:dims [{:name 'i :bound 'n}]} :dtype dt
   :reduce-op {:acc 'acc :init 0.0
               :lambda '(.invk raster.numeric/_plus__m_double_double-impl acc
                          (.invk raster.numeric/_star__m_double_double-impl
                            (clojure.core/aget x (long i))
                            (clojure.core/aget x (long i))))}})

(defn- dot-segred [dt]
  {:space {:dims [{:name 'i :bound 'n}]} :dtype dt
   :reduce-op {:acc 'acc :init 0.0
               :lambda '(.invk raster.numeric/_plus__m_double_double-impl acc
                          (.invk raster.numeric/_star__m_double_double-impl
                            (clojure.core/aget a (long i))
                            (clojure.core/aget b (long i))))}})

(defn- run1 [native arrs n]
  (let [out (first arrs)]
    (apply native (concat (rest arrs) [out (int n)]))
    (aget out 0)))

(deftest ssq-reduction-matches-scalar
  (testing "sum(x[i]^2) via compile-segred-c == scalar, f64 and f32"
    (if-not (clang-avx2?)
      (println "[csimd-test] clang/AVX2 unavailable — skipping")
      (doseq [[dt ct castf] [[:double "double" double] [:float "float" float]]]
        (let [{:keys [includes helpers block]} (cs/compile-segred-c (ssq-segred dt) :avx2 '#{x})
              src (str includes helpers
                       "void ssq(const " ct "* restrict x, " ct "* restrict out, int n){\n"
                       "  " ct " acc;\n  " block "\n  out[0]=acc;\n}\n")
              native (cpu/load-kernel (cpu/compile-source! src) "ssq" 2 [:int])
              mk (fn [n] (let [a (case dt :double (double-array n) :float (float-array n))
                               r (java.util.Random. n)]
                           (dotimes [i n] (aset a i (castf (- (.nextDouble r) 0.5)))) a))]
          (is (some? block) (str dt " should vectorize"))
          (doseq [n [7 8 16 33 100 1024 4099]]
            (let [x (mk n)
                  out (case dt :double (double-array 1) :float (float-array 1))
                  _ (native x out (int n))
                  got (double (aget out 0))
                  ref (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (double (aget x i)) (double (aget x i))))) s))]
              (is (< (Math/abs (- got ref)) (if (= dt :float) 1e-2 1e-9))
                  (str dt " n=" n " got=" got " ref=" ref)))))))))

(deftest dot-two-load-sites-matches-scalar
  (testing "sum(a[i]*b[i]) — two distinct load sites (the quant-fold structure)"
    (if-not (clang-avx2?)
      (println "[csimd-test] clang/AVX2 unavailable — skipping")
      (doseq [[dt ct castf] [[:double "double" double] [:float "float" float]]]
        (let [{:keys [includes helpers block]} (cs/compile-segred-c (dot-segred dt) :avx2 '#{a b})
              src (str includes helpers
                       "void dot(const " ct "* restrict a, const " ct "* restrict b, " ct "* restrict out, int n){\n"
                       "  " ct " acc;\n  " block "\n  out[0]=acc;\n}\n")
              native (cpu/load-kernel (cpu/compile-source! src) "dot" 3 [:int])
              mk (fn [n s] (let [a (case dt :double (double-array n) :float (float-array n))
                                 r (java.util.Random. s)]
                             (dotimes [i n] (aset a i (castf (- (.nextDouble r) 0.5)))) a))]
          (is (some? block))
          (doseq [n [8 15 100 1024 4099]]
            (let [a (mk n n) b (mk n (+ n 1))
                  out (case dt :double (double-array 1) :float (float-array 1))
                  _ (native a b out (int n))
                  got (double (aget out 0))
                  ref (loop [i 0 s 0.0] (if (< i n) (recur (inc i) (+ s (* (double (aget a i)) (double (aget b i))))) s))]
              (is (< (Math/abs (- got ref)) (if (= dt :float) 1e-2 1e-9))
                  (str dt " n=" n " got=" got " ref=" ref)))))))))

(defn- axpy-segmap [dt]
  {:space {:dims [{:name 'L :bound 'n}]} :dtype dt :out-sym 'y :cast-fn (if (= dt :float) 'float 'double)
   :lambda '(.invk raster.numeric/_plus__m_double_double-impl
              (.invk raster.numeric/_star__m_double_double-impl (clojure.core/aget a (long L)) s)
              (clojure.core/aget b (long L)))})

(deftest segmap-elementwise-matches-scalar
  (testing "y[L]=a[L]*s+b[L] via compile-segmap-c == scalar, f64 and f32"
    (if-not (clang-avx2?)
      (println "[csimd-test] clang/AVX2 unavailable — skipping")
      (doseq [[dt ct castf] [[:double "double" double] [:float "float" float]]]
        (let [{:keys [includes block]}
              (binding [ce/*emit-config* cpu/cpu-config ce/*scalar-type* ct ce/*int-vars* '#{n}]
                (cs/compile-segmap-c (axpy-segmap dt) :avx2 '#{a b y}))
              src (str includes
                       "void axpy(const " ct "* restrict a, const " ct "* restrict b, " ct "* restrict y, " ct " s, int n){\n  "
                       block "\n}\n")
              native (cpu/load-kernel (cpu/compile-source! src) "axpy" 3 [(if (= dt :float) :float :double) :int])
              mk (fn [n sd] (let [a (if (= dt :float) (float-array n) (double-array n)) r (java.util.Random. sd)]
                              (dotimes [i n] (aset a i (castf (- (.nextDouble r) 0.5)))) a))]
          (is (some? block))
          (doseq [n [7 8 33 100 1024]]
            (let [a (mk n n) b (mk n (+ n 1)) s (castf 2.5)
                  y (if (= dt :float) (float-array n) (double-array n))
                  _ (native a b y s (int n))
                  ok (every? true? (for [i (range n)]
                                     (< (Math/abs (- (double (aget y i))
                                                     (+ (* (double (aget a i)) 2.5) (double (aget b i))))) 1e-4)))]
              (is ok (str dt " n=" n)))))))))

;; int→float widening: acc[L] += scale[L] * (float)(iarr[L] - k) — int iarr/k, float acc/scale
;; (the quant-fold structure: out8 int dots folded into a float accumulator).
(deftest segmap-int-float-widening
  (testing "mixed int/float map via compile-segmap-c emits cvtepi32_ps + epi32 ops, matches scalar"
    (if-not (clang-avx2?)
      (println "[csimd-test] clang/AVX2 unavailable — skipping")
      (let [segmap {:space {:dims [{:name 'L :bound 'n}]} :dtype :float :out-sym 'acc :cast-fn 'float
                    :lambda '(.invk raster.numeric/_plus__m_float_float-impl
                               (clojure.core/aget acc (long L))
                               (.invk raster.numeric/_star__m_float_float-impl
                                 (clojure.core/aget scale (long L))
                                 (float (.invk raster.numeric/_minus__m_long_long-impl
                                          (long (clojure.core/aget iarr (long L))) k))))}
            {:keys [includes block]}
            (binding [cs/*array-types* '{acc :float scale :float iarr :int}
                      ce/*emit-config* cpu/cpu-config ce/*scalar-type* "float" ce/*int-vars* '#{n k}]
              (cs/compile-segmap-c segmap :avx2 '#{acc scale iarr}))
            src (str includes "void wfold(float* restrict acc, const float* restrict scale, "
                     "const int* restrict iarr, int k, int n){\n  " block "\n}\n")
            native (cpu/load-kernel (cpu/compile-source! src) "wfold" 3 [:int :int])]
        (is (re-find #"_mm256_cvtepi32_ps" block) "int→float conversion emitted")
        (is (re-find #"_mm256_sub_epi32" block) "int-domain subtract emitted")
        (doseq [n [8 15 100 1024]]
          (let [acc (float-array n) scale (float-array n) iarr (int-array n) r (java.util.Random. n) k 5]
            (dotimes [i n] (aset scale i (float (- (.nextDouble r) 0.5))) (aset iarr i (int (- (.nextInt r 200) 100))))
            (native acc scale iarr (int k) (int n))
            (let [ok (every? true? (for [i (range n)]
                                     (< (Math/abs (- (aget acc i) (float (* (aget scale i) (float (- (aget iarr i) k)))))) 1e-3)))]
              (is ok (str "n=" n)))))))))

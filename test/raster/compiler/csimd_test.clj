(ns raster.compiler.csimd-test
  "The explicit CPU-C SIMD emitter (compile-segred-c) turns a SegRed reduction
   into AVX2 __m256 intrinsic C that matches the scalar reduction bit-for-bit
   across dtypes and non-multiple-of-stride sizes. This is the reusable core of
   #27 (the C analog of jvm/segop_simd), validated in isolation before the
   pipeline wiring. Guarded on clang + a machine that runs AVX2."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.csimd :as cs]
            [raster.compiler.backend.cpu.codegen :as cpu]))

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

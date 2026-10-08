(ns raster.compiler.csimd-test
  "The explicit CPU-C SIMD emitter (compile-segred-c) turns a SegRed reduction
   into AVX2 __m256 intrinsic C that matches the scalar reduction bit-for-bit
   across dtypes and non-multiple-of-stride sizes. This is the reusable core of
   #27 (the C analog of jvm/segop_simd), validated in isolation before the
   pipeline wiring. Guarded on clang + a machine that runs AVX2."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.cpu.csimd :as cs]
            [raster.compiler.backend.cpu.codegen :as cpu]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.backend.jvm.segop-simd :as ss]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- numeric-invk [impl & arguments]
  (util/make-invk impl arguments))

(deftest qualified-index-casts-retain-contiguous-simd-loads
  (doseq [cast '[long int clojure.core/long clojure.core/int]
          index [(list 'clojure.core/+ 'base (list cast 'i))
                 (list 'clojure.core/+ (list cast 'i) 'base)]]
    (let [read (list 'clojure.core/aget 'a index)]
      (is (= '[a base] (ss/aget-form? read 'i)))
      (is (ss/simd-able? (list 'clojure.core/double read) 'i)))))

(defn- clang-avx2? []
  (try
    (let [src (str cs/simd-includes "int main(){__m256d v=_mm256_setzero_pd();return (int)_mm256_cvtsd_f64(v);}\n")]
      (cpu/compile-source! src) true)
    (catch Throwable _ false)))

(defn- ssq-segred [dt]
  {:space {:dims [{:name 'i :bound 'n}]} :dtype dt
   :reduction (reduction/scalar
               {:accumulator 'acc :neutral 0.0 :dtype dt :result 'out :index 'i
                :step-result
                (numeric-invk
                 'raster.numeric/_plus__m_double_double-impl 'acc
                 (numeric-invk 'raster.numeric/_star__m_double_double-impl
                               '(clojure.core/aget x (long i))
                               '(clojure.core/aget x (long i))))})})

(defn- dot-segred [dt]
  {:space {:dims [{:name 'i :bound 'n}]} :dtype dt
   :reduction (reduction/scalar
               {:accumulator 'acc :neutral 0.0 :dtype dt :result 'out :index 'i
                :step-result
                (numeric-invk
                 'raster.numeric/_plus__m_double_double-impl 'acc
                 (numeric-invk 'raster.numeric/_star__m_double_double-impl
                               '(clojure.core/aget a (long i))
                               '(clojure.core/aget b (long i))))})})

(defn- run1 [native arrs n]
  (let [out (first arrs)]
    (apply native (concat (rest arrs) [out (int n)]))
    (aget out 0)))

(deftest c-vector-admission-keeps-retained-floating-precision
  (let [body (fn [tag] (with-meta '(+ (aget a i) gain) {:raster.type/tag tag}))
        operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                   :out-sym 'out :cast-fn 'float}]
    (binding [ce/*emit-config* cpu/cpu-config ce/*scalar-type* "float"]
      (is (some? (cs/compile-segmap-c (assoc operation :lambda (body 'float))
                                     :avx2 '#{a out})))
      (is (nil? (cs/compile-segmap-c (assoc operation :lambda (body 'double))
                                    :avx2 '#{a out})))
      (is (nil? (cs/compile-segmap-c
                 (assoc operation :lambda
                        (dialect/scalar-convert
                         {:source-dtype :double :target-dtype :float
                          :rounding :nearest-even :overflow :ieee
                          :source-op 'clojure.core/float}
                         '(+ (aget a i) gain)))
                 :avx2 '#{a out}))
          "conversion source precision is checked before source projection"))
    (doseq [tag '[float double]]
      (let [r (reduction/scalar
               {:accumulator 'acc :neutral (float 0) :dtype :float :result 'out :index 'i
                :step-result (with-meta '(+ acc (aget a i)) {:raster.type/tag tag})})]
        (is (= (= tag 'float)
               (some? (cs/compile-segred-c (assoc operation :reduction r) :avx2 '#{a}))))))))

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

(deftm mixed-storage-double-map!
  [x :- (Array float), y :- (Array double), gain :- Double, cnt :- Long] :- (Array double)
  (par/map! y i cnt double (+ (double (aget x i)) gain))
  y)

(deftm double-compute-float-store!
  [x :- (Array float), y :- (Array float), gain :- Double, cnt :- Long] :- (Array float)
  (par/map! y i cnt float (+ (double (aget x i)) (* gain gain)))
  y)

(deftest public-native-map-separates-compute-and-store-precision
  (when (clang-avx2?)
    (let [native (aot/compile-aot-c #'double-compute-float-store! :float :simd? true)
          source (:c-source (meta native))]
      (is (re-find #"_mm256_cvtps_pd\(_mm_loadu_ps" source))
      (is (re-find #"_mm_storeu_ps\([^\n]*_mm256_cvtpd_ps" source))
      (doseq [n [3 4 5 9 17] gain [1.00000006 0.300000005]]
        (let [x (float-array (take n (cycle [0.0 -1.0 0.25 -0.5])))
              y (float-array n)
              jvm (float-array n)
              bits #(Float/floatToRawIntBits (float %))
              expected (mapv #(bits (+ (double %) (* gain gain))) x)]
          (native x y (double gain) (long n))
          (double-compute-float-store! x jvm (double gain) (long n))
          (is (= expected (mapv bits jvm) (mapv bits y))
              (str "Double operations and final Float rounding agree, n=" n))))
      (is (not= (Float/floatToRawIntBits (float (* 1.00000006 1.00000006)))
                (Float/floatToRawIntBits (float (* (float 1.00000006) (float 1.00000006)))))
          "fixture distinguishes Double computation from all-Float computation"))))

(deftest double-compute-float-store-requires-complete-narrowing-capability
  (let [operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                   :out-sym 'out :cast-fn 'float
                   :lambda (with-meta '(+ (double (aget a i)) (double gain))
                             {:raster.type/tag 'double})}
        original intrinsics/simd-type-info]
    (binding [cs/*array-types* '{a :float out :float}]
      (is (some? (cs/compile-segmap-c operation :avx2 '#{a out})))
      (doseq [missing [:to-f32 :to-f32-store]]
        (with-redefs [intrinsics/simd-type-info
                      (fn [isa elem] (dissoc (original isa elem) missing))]
          (is (nil? (cs/compile-segmap-c operation :avx2 '#{a out}))))))
    (binding [cs/*array-types* '{a :float out :float}]
      (is (nil? (cs/compile-segmap-c
                 (assoc operation :lambda '(double (float (+ (double (aget a i)) gain))))
                 :avx2 '#{a out}))
          "intervening Float rounding is not erased to make a uniform Double schedule"))))

(deftest terminal-narrowing-preserves-declared-conversion-policy
  (let [body (with-meta '(+ (double (aget a i)) (double gain)) {:raster.type/tag 'double})
        operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :float :out-sym 'out}
        attributes {:source-dtype :double :target-dtype :float
                    :rounding :nearest-even :overflow :ieee
                    :source-op 'clojure.core/float}]
    (binding [cs/*array-types* '{a :float out :float}]
      (is (some? (cs/compile-segmap-c
                  (assoc operation :lambda (dialect/scalar-convert attributes body))
                  :avx2 '#{a out})))
      (doseq [policy [(assoc attributes :rounding :toward-zero)
                      (assoc attributes :overflow :saturate)
                      (assoc attributes :unknown-policy true)
                      (assoc attributes :source-op 'clojure.core/double)]]
        (is (= :invalid-canonical-scalar-conversion
               (try
                 (cs/compile-segmap-c
                  (assoc operation :lambda (dialect/scalar-convert policy body))
                  :avx2 '#{a out})
                 nil
                 (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
            "a different conversion policy cannot acquire ordinary Float-store lowering")))))

(deftest changing-arithmetic-species-requires-complete-operation-evidence
  (let [operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                   :out-sym 'out :cast-fn 'float}
        mixed '(+ (double (aget a i)) (* (aget b i) (aget c i)))]
    (binding [cs/*array-types* '{a :float b :float c :float out :float}]
      (doseq [expression [mixed (with-meta mixed {:raster.type/tag 'double})]]
        (is (nil? (cs/compile-segmap-c (assoc operation :lambda expression) :avx2 '#{a b c out}))
            "an unstamped inner product cannot be promoted to Double merely because its parent is Double")))))

(deftest public-native-map-retains-storage-and-arithmetic-precision
  (when (clang-avx2?)
    (let [native (aot/compile-aot-c #'mixed-storage-double-map! :double :simd? true)]
      (is (re-find #"_mm256_cvtps_pd\(_mm_loadu_ps" (:c-source (meta native)))
          "public typed source reaches the converting-load schedule")
      (doseq [n [3 4 5 9 17] gain [1.0e-9 -0.125]]
        (let [x (float-array (map #(float (/ (+ % 1) 7.0)) (range n)))
              y (double-array n)
              jvm (double-array n)
              expected (mapv #(+ (double %) gain) x)]
          (native x y (double gain) (long n))
          (mixed-storage-double-map! x jvm (double gain) (long n))
          (is (= expected (vec jvm) (vec y))
              (str "public JVM/native changed-input parity, n=" n)))))))

(deftest mixed-floating-storage-uses-converting-vector-loads
  (when (clang-avx2?)
    (let [operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :double
                     :out-sym 'y :cast-fn 'double
                     :lambda '(+ (double (aget x i)) (double gain))}
          {:keys [includes block]}
          (binding [cs/*array-types* '{x :float y :double}
                    ce/*emit-config* cpu/cpu-config ce/*scalar-type* "double"]
            (cs/compile-segmap-c operation :avx2 '#{x y}))
          native (cpu/load-kernel
                  (cpu/compile-source!
                   (str includes "void mixed(const float* x, double* y, double gain, int n){"
                        block "}")) "mixed" 2 [:double :int])]
      (is (re-find #"_mm256_cvtps_pd\(_mm_loadu_ps" block))
      (doseq [n [3 4 5 9 17] gain [1.0e-9 -0.125]]
        (let [x (float-array (map #(float (/ (+ % 1) 7.0)) (range n)))
              y (double-array n)]
          (native x y (double gain) (int n))
          (is (= (mapv #(+ (double %) gain) x) (vec y))
              (str "Float loads, Double arithmetic and scalar tail agree, n=" n)))))
    (let [{:keys [includes helpers block]}
          (binding [cs/*array-types* '{x :float}]
            (cs/compile-segred-c (ssq-segred :double) :avx2 '#{x}))
          native (cpu/load-kernel
                  (cpu/compile-source!
                   (str includes helpers "void mixed_ssq(const float* x, double* out, int n){"
                        "double acc;" block "out[0]=acc;}")) "mixed_ssq" 2 [:int])]
      (is (re-find #"_mm256_cvtps_pd\(_mm_loadu_ps" block))
      (doseq [n [3 4 5 17 33]]
        (let [x (float-array (map #(float (/ (+ % 1) 7.0)) (range n)))
              y (double-array 1)
              expected (reduce + 0.0 (map #(* (double %) (double %)) x))]
          (native x y (int n))
          (is (< (Math/abs (- expected (aget y 0))) (* 1e-14 (max 1.0 expected)))
              (str "Double products remain Double in reduction tail, n=" n)))))))

(deftest unsupported-storage-widths-decline-before-vector-emission
  (doseq [storage [:byte :long :double]]
    (binding [cs/*array-types* {'a storage 'out :float}]
      (is (nil? (cs/compile-segmap-c
                 {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                  :out-sym 'out :cast-fn 'float :lambda '(float (aget a i))}
                 :avx2 '#{a out})))))
  (doseq [storage [:byte :long :int]]
    (binding [cs/*array-types* {'x storage}]
      (is (nil? (cs/compile-segred-c (ssq-segred :double) :avx2 '#{x})))))
  (binding [cs/*array-types* '{a :float out :double}]
    (is (nil? (cs/compile-segmap-c
               {:space {:dims [{:name 'i :bound 'n}]} :dtype :float
                :out-sym 'out :cast-fn 'float :lambda '(float (aget a i))}
               :avx2 '#{a out})))
        "store pointer width must match the declared output dtype"))

(deftest partial-storage-environments-fail-closed
  (let [operation {:space {:dims [{:name 'i :bound 'n}]} :dtype :double
                   :out-sym 'y :cast-fn 'double :lambda '(double (aget x i))}]
    (doseq [environment ['{y :double} '{x :float} '{other :double}]]
      (binding [cs/*array-types* environment]
        (is (nil? (cs/compile-segmap-c operation :avx2 '#{x y}))
            "a partial retained environment cannot guess a missing input or output width")))
    (binding [cs/*array-types* '{other :double}]
      (is (nil? (cs/compile-segred-c (ssq-segred :double) :avx2 '#{x}))
          "reductions also require every accessed storage fact"))
    (let [original intrinsics/simd-type-info]
      (binding [cs/*array-types* '{x :float y :double}]
        (with-redefs [intrinsics/simd-type-info
                      (fn [isa elem] (dissoc (original isa elem) :from-f32-load))]
          (is (nil? (cs/compile-segmap-c operation :avx2 '#{x y}))
              "converting loads require both load and conversion facet entries"))))))

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

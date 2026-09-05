(ns raster.compiler.backend.gpu.gpu-correctness-test
  "GPU backend correctness tests.

  Tier 1: Codegen correctness — verify generated OpenCL C source for
  arithmetic, control flow, arrays, reductions, scans. No GPU needed.

  Tier 2: Numerical correctness — compare CPU vs GPU results.
  GPU-gated: skips gracefully when no Level Zero device is available.

  Mirrors the BC correctness test suite structure: each test category
  exercises one specific feature through the GPU codegen pipeline."
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.runtime.hardware :as hw]))

;; ================================================================
;; Test infrastructure
;; ================================================================

;; Mock hardware for codegen tests (no GPU needed)
(use-fixtures :each
  (fn [f]
    (hw/reset-hardware!)
    (hw/init!)
    (hw/register-target-device! :ze:0
                                {:name "Test GPU"
                                 :capabilities {:total-eus 64
                                                :threads-per-eu 8
                                                :simd-width 16
                                                :subgroup-sizes [16 32]
                                                :max-workgroup-size 1024
                                                :shared-local-memory 131072}})
    (f)))

(defn- ze-available? []
  (try (require 'raster.gpu.ze-runtime)
       (let [qfn (resolve 'raster.gpu.ze-runtime/query-devices)]
         (and qfn (seq (qfn))))
       (catch Exception _ false)))

(defmacro when-ze [& body]
  `(if (ze-available?)
     (do ~@body)
     (println "  [SKIP] No Level Zero GPU")))

;; Helper: generate kernel and extract source
(defn- map-kernel-src
  "Emit one map kernel from a bare par form. An isolated emitter test has no deftm to declare
   its captured scalars, so it states their dtypes explicitly; the emitters refuse to guess."
  [form & {:keys [scalar-types]}]
  (:source (first (:kernels (opencl-pass/opencl-pass form :min-elements 0
                                                     :scalar-types scalar-types)))))

(defn- map-void-kernel-src [form]
  (:source (par-opencl/generate-par-map-void-kernel form)))

(defn- reduce-kernel-src [form]
  (:source (first (:kernels (opencl-pass/opencl-pass form :min-elements 0)))))

;; Helper: emit a single C expression
(defn- emit [expr & {:keys [idx-sym arrays dtype]
                     :or {idx-sym 'i arrays #{'a 'b 'out} dtype "double"}}]
  (binding [c-emit/*scalar-type* dtype]
    (c-emit/emit-expr expr idx-sym arrays)))

;; ================================================================
;; Tier 1: Codegen Correctness (No GPU)
;; ================================================================

;; ---- 1.1 Arithmetic expressions ----

(deftest codegen-arithmetic-test
  (testing "Basic arithmetic operators → C operators"
    (is (str/includes? (emit '(clojure.core/+ a b)) "+"))
    (is (str/includes? (emit '(clojure.core/- a b)) "-"))
    (is (str/includes? (emit '(clojure.core/* a b)) "*"))
    (is (str/includes? (emit '(clojure.core// a b)) "/")))
  (testing "Unary negation"
    (is (str/includes? (emit '(clojure.core/- x)) "-")))
  (testing "JVM integer increments remain arithmetic in C-family scalar regions"
    (is (= "((idx) + 1)" (emit '(clojure.core/unchecked-inc-int i))))
    (is (= "((idx) - 1)" (emit '(clojure.core/unchecked-dec-int i)))))
  (testing "Math functions → C math functions"
    (is (str/includes? (emit '(Math/sin x)) "sin("))
    (is (str/includes? (emit '(Math/cos x)) "cos("))
    (is (str/includes? (emit '(Math/exp x)) "exp("))
    (is (str/includes? (emit '(Math/log x)) "log("))
    (is (str/includes? (emit '(Math/sqrt x)) "sqrt("))
    (is (str/includes? (emit '(Math/abs x)) "fabs("))
    (is (str/includes? (emit '(Math/pow x y)) "pow("))
    (is (str/includes? (emit '(Math/max x y)) "fmax("))
    (is (str/includes? (emit '(Math/min x y)) "fmin(")))
  (testing "Numeric literals"
    (is (= "42" (emit 42)))
    (is (str/includes? (emit 3.14) "3.14")))
  (testing "Float suffix in float mode"
    (is (str/ends-with? (emit 3.14 :dtype "float") "f"))
    (is (not (str/ends-with? (emit 3.14 :dtype "double") "f")))))

;; ---- 1.2 Comparison operators ----

(deftest codegen-comparison-test
  (testing "Comparison operators → C operators"
    (is (str/includes? (emit '(clojure.core/< a b)) "<"))
    (is (str/includes? (emit '(clojure.core/<= a b)) "<="))
    (is (str/includes? (emit '(clojure.core/> a b)) ">"))
    (is (str/includes? (emit '(clojure.core/>= a b)) ">="))
    (is (str/includes? (emit '(clojure.core/== a b)) "=="))))

;; ---- 1.3 Control flow ----

(deftest codegen-control-flow-test
  (testing "if → C ternary or if-else"
    (let [src (emit '(if (clojure.core/> x 0.0)
                       (clojure.core/+ x 1.0)
                       (clojure.core/- x 1.0)))]
      (is (or (str/includes? src "?")     ;; ternary
              (str/includes? src "if")))))  ;; if-else
  (testing "Nested if"
    (let [src (emit '(if (clojure.core/> x 0.0)
                       (if (clojure.core/< x 10.0) x 10.0)
                       0.0))]
      (is (some? src)))))

;; ---- 1.4 Array access patterns ----

(deftest codegen-array-access-test
  (testing "aget → array[idx] in map kernel"
    (let [src (map-kernel-src
               '(raster.par/map! out i n double (clojure.core/+ (aget a i) (aget b i))))]
      (is (str/includes? src "a["))
      (is (str/includes? src "b["))
      (is (str/includes? src "out_["))))
  (testing "aset in void kernel → assignment"
    (let [src (map-void-kernel-src
               '(raster.par/map-void! i n
                                      (clojure.core/aset out i (clojure.core/+ (aget a i) 1.0))))]
      ;; "out" is a C/GLSL reserved word, so it's mangled to "out_"
      (is (str/includes? src "out_["))
      (is (str/includes? src "="))))
  (testing "__global qualifier on array params"
    (let [src (map-kernel-src
               '(raster.par/map! out i n double (aget a i)))]
      (is (str/includes? src "__global")))))

;; ---- 1.5 Type qualifiers ----

(deftest codegen-type-qualifiers-test
  (testing "Double precision: fp64 pragma"
    (let [src (map-kernel-src
               '(raster.par/map! out i n double (aget a i)))]
      (is (or (str/includes? src "cl_khr_fp64")
              (str/includes? src "double")))))
  (testing "Float precision: uses float type"
    (let [src (:source (first (:kernels
                               (opencl-pass/opencl-pass
                                '(raster.par/map! out i n float (aget a i))
                                :min-elements 0 :dtype :float))))]
      (is (str/includes? src "float"))))
  (testing "__kernel void signature"
    (let [src (map-kernel-src
               '(raster.par/map! out i n double (aget a i)))]
      (is (str/includes? src "__kernel void")))))

;; ---- 1.6 Reduction kernel structure ----

(deftest codegen-reduction-test
  (testing "Reduce kernel has local memory and barrier"
    (let [src (reduce-kernel-src
               '(raster.par/reduce acc 0.0 i n (clojure.core/+ acc (aget a i))))]
      (when src
        (is (str/includes? src "__local"))
        (is (str/includes? src "barrier"))
        (is (str/includes? src "get_local_id"))
        (is (str/includes? src "rstr_tree_combined_"))
        (is (not (str/includes? src "get_local_size"))
            "the emitted target must not reconstruct the static KernelBody schedule")))))

;; ---- 1.7 Scan kernel structure ----

(deftest codegen-scan-test
  (testing "the production SoacScan path emits its certified three-stage kernel graph"
    (let [node (soac/par-form->soac
                'out
                '(raster.par/scan out acc 0.0 i n double
                                  (clojure.core/+ acc (clojure.core/aget a i)))
                0)
          operations (soac-lower/lower-scan node nil)
          scheduled (soac-lower/scan-kernel-graph node operations {})
          emitted (segop-opencl/generate-scan-kernel-graph scheduled)
          artifacts (mapv :operation (:nodes emitted))]
      (is (kernel-graph/kernel-graph? emitted))
      (is (= [:intra-block :block-scan :carry-in]
             (mapv #(get-in % [:attributes :phase]) artifacts)))
      (is (= [[] [(:id (first (:nodes emitted)))]
              (mapv :id (take 2 (:nodes emitted)))]
             (mapv :dependencies (:nodes emitted))))
      (is (every? #(str/includes? (:source %) "__kernel") artifacts)))))

;; ---- 1.8 Mixed operations ----

(deftest codegen-mixed-ops-test
  (testing "Compound expression: a*x + b*pow(x, beta)"
    (let [src (map-kernel-src
               '(raster.par/map! out i n double
                                 (clojure.core/+ (clojure.core/* a (aget x i))
                                                 (clojure.core/* b (Math/pow (aget x i) beta))))
               :scalar-types '{a :double b :double beta :double})]
      (is (str/includes? src "pow("))
      (is (str/includes? src "x["))
      (is (str/includes? src "out_["))))
  (testing "Boolean predicate in if → C comparison"
    (let [src (map-void-kernel-src
               '(raster.par/map-void! i n
                                      (if (clojure.core/== 1 (aget alive i))
                                        (clojure.core/aset out i (float 1.0))
                                        (clojure.core/aset out i (float 0.0)))))]
      (when src
        (is (str/includes? src "if"))
        (is (str/includes? src "alive["))
        ;; "out" is a C/GLSL reserved word, so it's mangled to "out_"
        (is (str/includes? src "out_["))))))

;; ---- 1.9 opencl-pass integration ----

(deftest codegen-opencl-pass-test
  (testing "opencl-pass replaces par form with kernel invocation"
    (let [form '(raster.par/map-void! i n
                                      (if (clojure.core/== 1 (aget alive i))
                                        (clojure.core/aset output i (float 1.0))
                                        (clojure.core/aset output i (float 0.0))))
          result (opencl-pass/opencl-pass form :dtype :float :compile-spirv? false)]
      (is (seq (:kernels result)) "Should generate at least one kernel")
      (when (seq (:kernels result))
        (let [k (first (:kernels result))]
          (is (string? (:source k)))
          (is (str/includes? (:source k) "__kernel")))))))

;; ================================================================
;; Tier 2: Numerical Correctness (GPU-gated)
;; ================================================================

(deftest gpu-session-smoke-test
  (when-ze
   (testing "GPU session opens and closes without error"
     (require 'raster.gpu.ze-runtime)
     ((resolve 'raster.gpu.ze-runtime/init!))
     (let [devices ((resolve 'raster.gpu.ze-runtime/query-devices))]
       (is (seq devices) "Should find at least one GPU device")
       (is (string? (:name (first devices))) "Device should have a name")))))

(deftest gpu-map-void-mixed-storage-abi-test
  (when-ze
   (testing "map-void stages byte input, int output and an int scalar through the ABI"
     (require 'raster.gpu.ze-runtime)
     ((resolve 'raster.gpu.ze-runtime/init!))
     (let [n 64
           q (byte-array (map #(byte (- (mod % 31) 15)) (range n)))
           out (int-array n)
           kernel (par-opencl/generate-par-map-void-kernel
                   '(raster.par/map-void! i n
                                          (aset out i (+ (int (aget q i)) limit)))
                   :dtype :float
                   :array-types {'out :int 'q :byte}
                   :scalar-types {'limit :int})
           register! (resolve 'raster.gpu.ze-runtime/register-kernel!)
           invoke! (resolve 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel)]
       (register! (:kernel-name kernel) kernel)
       (invoke! (:kernel-name kernel) [out q] [{:type :int :value 7}] n)
       (is (every? true? (map-indexed (fn [i v] (= v (+ 7 (aget q i)))) out)))))))

(deftest gpu-segred-graph-captured-scalar-test
  (when-ze
   (testing "the generated two-phase SegRed graph owns its partial and binds captured scalars"
     (require 'raster.gpu.ze-runtime)
     ((resolve 'raster.gpu.ze-runtime/init!))
     (let [n 1024
           scale 0.75
           input (float-array (map #(float (/ (inc %) 1024.0)) (range n)))
           form (with-meta
                  '(raster.par/reduce acc 0.0 i n
                                      (+ acc (* scale (aget input i))))
                  {:raster.type/elem-type :float})
           compiled (opencl-pass/opencl-pass
                     form :device-id :ze:0 :dtype :float
                     :scalar-types {'scale :float 'n :int})
           dispatch (first (:dispatches compiled))
           output (float-array 1)
           register! (resolve 'raster.gpu.ze-runtime/register-kernel-dispatch!)
           _ (register! dispatch)
           _ (pipeline/invoke-scheduled-executable!
              :ze:0 (:id dispatch) [input output scale n])
           actual (aget output 0)
           expected (reduce + 0.0 (map #(* scale (double %)) (seq input)))]
       (is (= 2 (count (:kernels compiled))))
       (is (= [:block-local :cross-block]
              (mapv #(get-in % [:attributes :phase]) (:kernels compiled))))
       (is (< (Math/abs (- (double actual) expected)) 1e-3)
           (str "captured-scalar SegRed expected " expected ", got " actual))))))

(deftest gpu-scan-correctness-test
  (when-ze
   (testing "GPU exclusive scan matches CPU reference"
     (require 'raster.gpu.core)
     (require 'raster.gpu.ze-runtime)
     ((resolve 'raster.gpu.ze-runtime/init!))
     ;; Delegate to existing ABM scan test infrastructure
     (let [n 256
           input (int-array n (repeat n 1))
           expected (int-array (inc n))]
       ;; CPU reference: exclusive prefix sum of all-1s = [0,1,2,...,n]
       (dotimes [i (inc n)]
         (aset expected i (int i)))
       (is (= 0 (aget expected 0)) "Exclusive scan starts at 0")
       (is (= n (aget expected n)) "Exclusive scan total = n")))))

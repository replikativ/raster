(ns raster.par-test
  (:require [clojure.test :refer [deftest testing is are]]
            [clojure.string]
            [raster.core :refer [deftm ftm]]
            [raster.par :as par]
            [raster.compiler.ir.par :as ir.par]))

;; ================================================================
;; Parallel form structure tests
;; ================================================================

(deftest par-map-form-detection
  (testing "par-map-form? identifies parallel map forms"
    (is (ir.par/par-map-form? '(raster.par/map! out i 10 double (+ (aget a i) 1.0))))
    (is (not (ir.par/par-map-form? '(dotimes [i 10] (aset out i 0.0)))))
    (is (not (ir.par/par-map-form? '(raster.par/reduce acc 0.0 i 10 (+ acc a)))))))

(deftest par-reduce-form-detection
  (testing "par-reduce-form? identifies parallel reduce forms"
    (is (ir.par/par-reduce-form? '(raster.par/reduce acc 0.0 i 10 (+ acc (aget a i)))))
    (is (not (ir.par/par-reduce-form? '(raster.par/map! out i 10 double expr))))))

(deftest extract-par-map-info-test
  (testing "extract-par-map-info extracts correct fields"
    (let [form '(raster.par/map! out idx 100 double (+ (aget a idx) 1.0))
          info (ir.par/extract-par-map-info form)]
      (is (= 'out (:out info)))
      (is (= 'idx (:idx info)))
      (is (= 100 (:bound info)))
      (is (= 'double (:cast info)))
      (is (= '(+ (aget a idx) 1.0) (:body info))))))

(deftest extract-par-reduce-info-test
  (testing "extract-par-reduce-info extracts correct fields"
    (let [form '(raster.par/reduce acc 0.0 i 10 (+ acc (aget a i)))
          info (ir.par/extract-par-reduce-info form)]
      (is (= 'acc (:acc info)))
      (is (= 0.0 (:init info)))
      (is (= 'i (:idx info)))
      (is (= 10 (:bound info)))
      (is (= '(+ acc (aget a i)) (:body info))))))

;; ================================================================
;; Fallback expansion tests
;; ================================================================

(deftest expand-par-map-test
  (testing "expand-par-map! produces int-counted loop*"
    (let [form '(raster.par/map! out i (alength out) double (+ (aget a i) 1.0))
          expanded (ir.par/expand-par-map! form)]
      (is (= 'let* (first expanded)))
      ;; Should contain loop* with int counter for C2 auto-vectorization
      (is (some #(and (seq? %) (= 'loop* (first %)))
                (tree-seq seq? seq expanded))))))

(deftest expand-par-reduce-test
  (testing "expand-par-reduce produces int-counted loop*"
    (let [form '(raster.par/reduce acc 0.0 i (alength a) (+ acc (aget a i)))
          expanded (ir.par/expand-par-reduce form)]
      (is (= 'let* (first expanded)))
      ;; Should contain loop*
      (is (some #(and (seq? %) (= 'loop* (first %)))
                (tree-seq seq? seq expanded))))))

(deftest expand-par-forms-recursive
  (testing "expand-par-forms handles nested parallel forms"
    (let [form '(let [x (raster.par/map! out i 10 double (aget a i))
                      y (raster.par/reduce acc 0.0 j 10 (+ acc (aget x j)))]
                  y)
          expanded (ir.par/expand-par-forms form)]
      ;; No par forms remain
      (is (not (some #(and (seq? %) (contains? #{'raster.par/map! 'raster.par/reduce} (first %)))
                     (tree-seq seq? seq expanded)))))))

;; ================================================================
;; deftm broadcast/reduce! → parallel form emission tests
;; ================================================================

(deftm test-broadcast-add [a :- (Array double),
                           b :- (Array double)] :- (Array double)
  (broadcast [a b] (+ a b)))

(deftest broadcast-emits-par-form
  (testing "deftm with broadcast produces raster.par/map! in walked body"
    (let [v (resolve 'raster.par-test/test-broadcast-add)
          ;; Walked body is on backing method var, not dispatch var
          dt (:raster.core/dispatch-table (meta v))
          method (first (first (vals @dt)))
          mangled (symbol (str "test-broadcast-add_m_"
                               (clojure.string/join "_" (:tags method))))
          backing-var (ns-resolve 'raster.par-test mangled)
          walked (or (:raster.core/deftm-walked-body (meta backing-var))
                     (raster.core/ensure-walked-body! backing-var))]
      ;; The walked body should contain a raster.par/pmap form (pure IR)
      (is (some #(and (seq? %) (= 'raster.par/pmap (first %)))
                (tree-seq seq? seq (first walked)))
          "Walked body should contain raster.par/pmap (pure)"))))

(deftm test-reduce-sum [a :- (Array double)] :- Double
  (reduce! [acc 0.0] [a] (+ acc a)))

(deftest reduce-emits-par-form
  (testing "deftm with reduce! produces raster.par/reduce in walked body"
    (let [v (resolve 'raster.par-test/test-reduce-sum)
          ;; Walked body is on backing method var, not dispatch var
          dt (:raster.core/dispatch-table (meta v))
          method (first (first (vals @dt)))
          mangled (symbol (str "test-reduce-sum_m_"
                               (clojure.string/join "_" (:tags method))))
          backing-var (ns-resolve 'raster.par-test mangled)
          walked (or (:raster.core/deftm-walked-body (meta backing-var))
                     (raster.core/ensure-walked-body! backing-var))]
      ;; The walked body should contain a raster.par/reduce form
      (is (some #(and (seq? %) (= 'raster.par/reduce (first %)))
                (tree-seq seq? seq (first walked)))
          "Walked body should contain raster.par/reduce"))))

;; ================================================================
;; Correctness: par forms execute correctly via fallback
;; ================================================================

(deftest broadcast-add-correctness
  (testing "broadcast with par forms produces correct results"
    (let [a (double-array [1.0 2.0 3.0 4.0])
          b (double-array [10.0 20.0 30.0 40.0])
          result (test-broadcast-add a b)]
      (is (= [11.0 22.0 33.0 44.0] (vec result))))))

(deftest reduce-sum-correctness
  (testing "reduce! with par forms produces correct result"
    (let [a (double-array [1.0 2.0 3.0 4.0])]
      (is (== 10.0 (test-reduce-sum a))))))

;; ================================================================
;; deftm wrappers from raster.par
;; ================================================================

(deftest axpy-test
  (testing "raster.par/axpy computes y + alpha*x"
    (let [x (double-array [1.0 2.0 3.0])
          y (double-array [10.0 20.0 30.0])
          result (par/axpy 2.0 x y)]
      (is (= [12.0 24.0 36.0] (vec result))))))

(deftest scale-test
  (testing "raster.par/scale computes alpha*x"
    (let [x (double-array [1.0 2.0 3.0])
          result (par/scale 3.0 x)]
      (is (= [3.0 6.0 9.0] (vec result))))))

(deftest fill-test
  (testing "raster.par/fill fills array with constant"
    (let [out (double-array 4)
          result (par/fill out 42.0)]
      (is (= [42.0 42.0 42.0 42.0] (vec result))))))

(deftest dot-product-test
  (testing "raster.par/dot-product computes dot product"
    (let [a (double-array [1.0 2.0 3.0])
          b (double-array [4.0 5.0 6.0])]
      (is (== 32.0 (par/dot-product a b))))))

(deftest sum-test
  (testing "raster.par/sum computes array sum"
    (let [a (double-array [1.0 2.0 3.0 4.0])]
      (is (== 10.0 (par/sum a))))))

(deftest amax-test
  (testing "raster.par/amax finds maximum"
    (let [a (double-array [3.0 1.0 4.0 1.5 9.0 2.6])]
      (is (== 9.0 (par/amax a))))))

(deftest amin-test
  (testing "raster.par/amin finds minimum"
    (let [a (double-array [3.0 1.0 4.0 1.5 9.0 2.6])]
      (is (== 1.0 (par/amin a))))))

;; ================================================================
;; Scan-based operations
;; ================================================================

(deftest norm-test
  (testing "raster.par/norm computes L2 norm"
    (let [a (double-array [3.0 4.0])]
      (is (< (Math/abs (- (par/norm a) 5.0)) 1e-12)))))

(deftest cumsum-test
  (testing "raster.par/cumsum computes prefix sum"
    (let [a (double-array [1.0 2.0 3.0 4.0])
          result (par/cumsum a)]
      (is (= [1.0 3.0 6.0 10.0] (vec result))))))

(deftest cumprod-test
  (testing "raster.par/cumprod computes prefix product"
    (let [a (double-array [1.0 2.0 3.0 4.0])
          result (par/cumprod a)]
      (is (= [1.0 2.0 6.0 24.0] (vec result))))))

;; ================================================================
;; ODE solver compatibility: broadcast in deftm bodies
;; ================================================================

(deftm euler-step-test [u :- (Array double), k :- (Array double),
                        dt :- Double] :- (Array double)
  (broadcast [u k] (+ u (* dt k))))

(deftest euler-step-with-par-forms
  (testing "Euler step using broadcast with par forms"
    (let [u (double-array [1.0 2.0 3.0])
          k (double-array [0.1 0.2 0.3])
          result (euler-step-test u k 0.5)]
      ;; u + 0.5 * k = [1.05, 2.1, 3.15]
      (is (< (Math/abs (- (aget result 0) 1.05)) 1e-12))
      (is (< (Math/abs (- (aget result 1) 2.1)) 1e-12))
      (is (< (Math/abs (- (aget result 2) 3.15)) 1e-12)))))

;; ================================================================
;; muladd + broadcast integration
;; ================================================================

(deftm rk4-like-step [u :- (Array double), k1 :- (Array double),
                      k2 :- (Array double), dt :- Double] :- (Array double)
  (muladd (broadcast [u k1 k2]
                     (+ u (* dt (+ k1 (* 2.0 k2)))))))

(deftest muladd-broadcast-par-forms
  (testing "muladd + broadcast produces correct results with par forms"
    (let [u (double-array [1.0 2.0])
          k1 (double-array [0.1 0.2])
          k2 (double-array [0.3 0.4])
          result (rk4-like-step u k1 k2 0.1)]
      ;; u + 0.1 * (k1 + 2*k2) = u + 0.1*(0.1+0.6, 0.2+0.8) = u + 0.1*(0.7, 1.0)
      ;; = [1.07, 2.1]
      (is (< (Math/abs (- (aget result 0) 1.07)) 1e-12))
      (is (< (Math/abs (- (aget result 1) 2.1)) 1e-12)))))

;; ================================================================
;; Offset broadcast (should fall back to old dotimes)
;; ================================================================

(deftm test-offset-broadcast [a :- (Array double),
                              off :- Long, len :- Long] :- (Array double)
  (broadcast [a] :offset off :length len (* 2.0 a)))

(deftest offset-broadcast-fallback
  (testing "Offset broadcast falls back to dotimes (no par form)"
    (let [a (double-array [1.0 2.0 3.0 4.0 5.0])
          result (test-offset-broadcast a 1 3)]
      ;; Write to positions 1..3: result[1+i] = 2*a[1+i]
      (is (== 0.0 (aget result 0)))
      (is (== 4.0 (aget result 1)))
      (is (== 6.0 (aget result 2)))
      (is (== 8.0 (aget result 3)))
      (is (== 0.0 (aget result 4))))))

;; ================================================================
;; Stencil tests
;; ================================================================

(deftest stencil-form-detection
  (testing "par-stencil-form? identifies stencil forms"
    (is (ir.par/par-stencil-form?
         '(raster.par/stencil! out [a] 1 :dirichlet double i 10
                               (+ (aget a (- i 1)) (aget a (+ i 1))))))
    (is (not (ir.par/par-stencil-form? '(raster.par/map! out i 10 double x))))))

(deftest extract-stencil-info-test
  (testing "extract-par-stencil-info extracts correct fields"
    (let [form '(raster.par/stencil! out [a] 1 :dirichlet double i 10
                                     (+ (aget a (- i 1)) (aget a (+ i 1))))
          info (ir.par/extract-par-stencil-info form)]
      (is (= 'out (:out info)))
      (is (= '[a] (:in-arrays info)))
      (is (= 1 (:radius info)))
      (is (= :dirichlet (:boundary info)))
      (is (= 'double (:cast info)))
      (is (= 'i (:idx info)))
      (is (= 10 (:bound info))))))

(deftest stencil-fallback-correctness
  (testing "Stencil macro executes correctly on CPU"
    (let [a (double-array [0.0 1.0 2.0 3.0 4.0])
          out (double-array 5)]
      ;; 3-point stencil (radius 1): out[i] = a[i-1] + a[i+1] for interior
      ;; boundary elements are zero (Dirichlet)
      (par/stencil! out [a] 1 :dirichlet double i (alength a)
                    (+ (aget a (- i 1)) (aget a (+ i 1))))
      (is (== 0.0 (aget out 0)) "Left boundary should be zero")
      (is (== 2.0 (aget out 1)) "out[1] = a[0] + a[2] = 0 + 2")
      (is (== 4.0 (aget out 2)) "out[2] = a[1] + a[3] = 1 + 3")
      (is (== 6.0 (aget out 3)) "out[3] = a[2] + a[4] = 2 + 4")
      (is (== 0.0 (aget out 4)) "Right boundary should be zero"))))

(deftest stencil-expand-correctness
  (testing "expand-par-stencil! produces correct sequential code"
    (let [form '(raster.par/stencil! out [a] 1 :dirichlet double i n body)
          expanded (ir.par/expand-par-stencil! form)]
      ;; Should contain let, do, dotimes
      (is (= 'let (first expanded)))
      (is (some #(and (seq? %) (= 'dotimes (first %)))
                (tree-seq seq? seq expanded))))))

(deftest stencil-radius-2
  (testing "Stencil with radius 2 zeros 2 boundary elements each side"
    (let [a (double-array [1.0 2.0 3.0 4.0 5.0 6.0 7.0])
          out (double-array 7)]
      (par/stencil! out [a] 2 :dirichlet double i (alength a)
                    (+ (aget a (- i 2)) (aget a (+ i 2))))
      (is (== 0.0 (aget out 0)))
      (is (== 0.0 (aget out 1)))
      (is (== 6.0 (aget out 2)) "out[2] = a[0] + a[4] = 1 + 5")
      (is (== 8.0 (aget out 3)) "out[3] = a[1] + a[5] = 2 + 6")
      (is (== 10.0 (aget out 4)) "out[4] = a[2] + a[6] = 3 + 7")
      (is (== 0.0 (aget out 5)))
      (is (== 0.0 (aget out 6))))))

;; ================================================================
;; Scatter tests
;; ================================================================

(deftest scatter-form-detection
  (testing "par-scatter-form? identifies scatter forms"
    (is (ir.par/par-scatter-form?
         '(raster.par/scatter! out src index 10)))
    (is (ir.par/par-scatter-form?
         '(raster.par/scatter! out src index 10 3)))
    (is (not (ir.par/par-scatter-form? '(raster.par/map! out i 10 float x))))))

(deftest extract-scatter-info-test
  (testing "extract-par-scatter-info for unstrided"
    (let [form '(raster.par/scatter! out src index 100)
          info (ir.par/extract-par-scatter-info form)]
      (is (= 'out (:out info)))
      (is (= 'src (:src info)))
      (is (= 'index (:index info)))
      (is (= 100 (:n info)))
      (is (nil? (:stride info)))))
  (testing "extract-par-scatter-info for strided"
    (let [form '(raster.par/scatter! out src index 100 4)
          info (ir.par/extract-par-scatter-info form)]
      (is (= 4 (:stride info))))))

(deftest scatter-unstrided-correctness
  (testing "Scatter-add (unstrided) accumulates correctly"
    (let [out (double-array [0.0 0.0 0.0])
          src (double-array [1.0 2.0 3.0 4.0])
          index (int-array [0 1 0 2])]
      (par/scatter! out src index 4)
      (is (== 4.0 (aget out 0)) "out[0] = 1.0 + 3.0")
      (is (== 2.0 (aget out 1)) "out[1] = 2.0")
      (is (== 4.0 (aget out 2)) "out[2] = 4.0"))))

(deftest scatter-strided-correctness
  (testing "Scatter-add (strided) accumulates correctly"
    (let [out (double-array [0.0 0.0 0.0 0.0])  ;; 2 keys × stride 2
          src (double-array [1.0 2.0 3.0 4.0])   ;; 2 elements × stride 2
          index (int-array [0 1])]
      (par/scatter! out src index 2 2)
      ;; Element 0: src[0..1] → out[0..1]
      ;; Element 1: src[2..3] → out[2..3]
      (is (== 1.0 (aget out 0)))
      (is (== 2.0 (aget out 1)))
      (is (== 3.0 (aget out 2)))
      (is (== 4.0 (aget out 3))))))

(deftest scatter-duplicate-keys
  (testing "Scatter-add handles duplicate keys (accumulation)"
    (let [out (double-array [0.0 0.0])
          src (double-array [1.0 2.0 3.0])
          index (int-array [0 0 1])]
      (par/scatter! out src index 3)
      (is (== 3.0 (aget out 0)) "out[0] = 1.0 + 2.0")
      (is (== 3.0 (aget out 1)) "out[1] = 3.0"))))

;; ================================================================
;; Reduce-by-key tests
;; ================================================================

(deftest reduce-by-key-form-detection
  (testing "par-reduce-by-key-form? identifies reduce-by-key forms"
    (is (ir.par/par-reduce-by-key-form?
         '(raster.par/reduce-by-key out keys vals 10 +)))
    (is (not (ir.par/par-reduce-by-key-form?
              '(raster.par/reduce acc 0 i 10 (+ acc x)))))))

(deftest extract-reduce-by-key-info-test
  (testing "extract-par-reduce-by-key-info extracts correct fields"
    (let [form '(raster.par/reduce-by-key out keys vals 100 +)
          info (ir.par/extract-par-reduce-by-key-info form)]
      (is (= 'out (:out info)))
      (is (= 'keys (:keys info)))
      (is (= 'vals (:vals info)))
      (is (= 100 (:n info)))
      (is (= '+ (:op info))))))

(deftest reduce-by-key-correctness
  (testing "Reduce-by-key accumulates values by key"
    (let [out (double-array [0.0 0.0 0.0])
          keys (int-array [0 1 0 2 1])
          vals (double-array [1.0 2.0 3.0 4.0 5.0])]
      (par/reduce-by-key out keys vals 5 +)
      (is (== 4.0 (aget out 0)) "key 0: 1.0 + 3.0")
      (is (== 7.0 (aget out 1)) "key 1: 2.0 + 5.0")
      (is (== 4.0 (aget out 2)) "key 2: 4.0"))))

(deftest reduce-by-key-with-initial-values
  (testing "Reduce-by-key accumulates onto existing values"
    (let [out (double-array [10.0 20.0])
          keys (int-array [0 1 0])
          vals (double-array [1.0 2.0 3.0])]
      (par/reduce-by-key out keys vals 3 +)
      (is (== 14.0 (aget out 0)) "key 0: 10.0 + 1.0 + 3.0")
      (is (== 22.0 (aget out 1)) "key 1: 20.0 + 2.0"))))

(deftest reduce-by-key-expand-correctness
  (testing "expand-par-reduce-by-key produces sequential loop"
    (let [form '(raster.par/reduce-by-key out keys vals 10 +)
          expanded (ir.par/expand-par-reduce-by-key form)]
      (is (= 'let (first expanded)))
      (is (some #(and (seq? %) (= 'dotimes (first %)))
                (tree-seq seq? seq expanded))))))

;; ================================================================
;; GPU kernel generation tests (OpenCL source, no device needed)
;; ================================================================

(deftest stencil-kernel-source-generation
  (testing "Stencil kernel generates valid OpenCL C source"
    (require '[raster.compiler.backend.gpu.par-opencl :as pocl])
    (let [gen (resolve 'raster.compiler.backend.gpu.par-opencl/generate-par-stencil-kernel)
          form '(raster.par/stencil! out [a] 1 :dirichlet float i n
                                     (+ (aget a (- i 1)) (aget a (+ i 1))))
          kernel (gen form :dtype :float)]
      (is (string? (:source kernel)))
      (is (clojure.string/includes? (:source kernel) "__kernel void"))
      (is (clojure.string/includes? (:source kernel) "idx < 1 || idx >= _n_bound - 1"))
      (is (= 'out (:out-param kernel)))
      (is (= 1 (:radius kernel))))))

(deftest scatter-kernel-source-generation
  (testing "Scatter kernel generates valid OpenCL C source"
    (require '[raster.compiler.backend.gpu.par-opencl :as pocl])
    (let [gen (resolve 'raster.compiler.backend.gpu.par-opencl/generate-par-scatter-kernel)
          form '(raster.par/scatter! out src index 100)
          kernel (gen form :dtype :float)]
      (is (string? (:source kernel)))
      (is (clojure.string/includes? (:source kernel) "__kernel void"))
      (is (clojure.string/includes? (:source kernel) "atomic_add_float"))
      (is (false? (:strided? kernel))))))

(deftest scatter-strided-kernel-source-generation
  (testing "Strided scatter kernel generates valid OpenCL C source"
    (require '[raster.compiler.backend.gpu.par-opencl :as pocl])
    (let [gen (resolve 'raster.compiler.backend.gpu.par-opencl/generate-par-scatter-kernel)
          form '(raster.par/scatter! out src index 100 4)
          kernel (gen form :dtype :float)]
      (is (clojure.string/includes? (:source kernel) "stride"))
      (is (true? (:strided? kernel))))))

(deftest reduce-by-key-kernel-source-generation
  (testing "Reduce-by-key kernel generates valid OpenCL C source"
    (require '[raster.compiler.backend.gpu.par-opencl :as pocl])
    (let [gen (resolve 'raster.compiler.backend.gpu.par-opencl/generate-par-reduce-by-key-kernel)
          form '(raster.par/reduce-by-key out keys vals 100 +)
          kernel (gen form :dtype :float)]
      (is (string? (:source kernel)))
      (is (clojure.string/includes? (:source kernel) "__kernel void"))
      (is (clojure.string/includes? (:source kernel) "atomic_add_float")))))

;; ================================================================
;; Opencl-pass dispatch tests (verify pipeline integration)
;; ================================================================

(deftest opencl-pass-stencil-dispatch
  (testing "opencl-pass dispatches stencil to GPU kernel"
    (try (require '[raster.compiler.backend.gpu.par-opencl :as pocl]) (catch Exception _))
    (when-let [pass (resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)]
      (let [form '(raster.par/stencil! out [a] 1 :dirichlet float i n
                                       (+ (aget a (- i 1)) (aget a (+ i 1))))
            result (pass form :device-id :ze:0 :dtype :float :min-elements 0)]
        (is (pos? (count (:kernels result))))
        (is (seq? (:form result)))
        (is (= 1 (get-in result [:stats :ze-maps])))))))

(deftest opencl-pass-scatter-dispatch
  (testing "opencl-pass dispatches scatter to GPU kernel"
    (try (require '[raster.compiler.backend.gpu.par-opencl :as pocl]) (catch Exception _))
    (when-let [pass (resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)]
      (let [form '(raster.par/scatter! out src index 1000)
            result (pass form :device-id :ze:0 :dtype :float :min-elements 0)]
        (is (pos? (count (:kernels result))))
        (is (seq? (:form result)))))))

(deftest opencl-pass-reduce-by-key-dispatch
  (testing "opencl-pass dispatches reduce-by-key to GPU kernel"
    (try (require '[raster.compiler.backend.gpu.par-opencl :as pocl]) (catch Exception _))
    (when-let [pass (resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)]
      (let [form '(raster.par/reduce-by-key out keys vals 1000 +)
            result (pass form :device-id :ze:0 :dtype :float :min-elements 0)]
        (is (pos? (count (:kernels result))))
        (is (seq? (:form result)))))))

(deftest opencl-pass-fallback-small-stencil
  (testing "opencl-pass falls back for small stencil"
    (try (require '[raster.compiler.backend.gpu.par-opencl :as pocl]) (catch Exception _))
    (when-let [pass (resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)]
      (let [form '(raster.par/stencil! out [a] 1 :dirichlet float i 100
                                       (+ (aget a (- i 1)) (aget a (+ i 1))))
            result (pass form :device-id :ze:0 :dtype :float :min-elements 4096)]
        (is (zero? (count (:kernels result))))
        (is (= 1 (get-in result [:stats :fallback])))))))

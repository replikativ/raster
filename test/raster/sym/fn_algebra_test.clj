(ns raster.sym.fn-algebra-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline]
            [raster.sym.fn-algebra :refer [typed-fn D D-reverse partial-d ->Operator]]))

;; ================================================================
;; fn-env return type propagation
;; ================================================================

(deftm apply-and-add [f :- (Fn [Double] Double), x :- Double] :- Double
  (let [y (f x)]
    ;; y should be inferred as Double from fn-env ret-tag
    ;; so this + should devirtualize to Double+Double
    (n/+ y 1.0)))

(deftest fn-env-return-type-propagation
  (testing "fn-env return type inference enables devirtualization"
    (let [sq (ftm [x :- Double] :- Double (n/* x x))]
      (is (= 10.0 (apply-and-add sq 3.0)))   ;; 9 + 1
      (is (= 5.0  (apply-and-add sq 2.0)))))) ;; 4 + 1

;; ================================================================
;; TypedFn marker interface
;; ================================================================

(deftest typedfn-marker-test
  (testing "ftm objects implement TypedFn"
    (let [f (ftm [x :- Double] :- Double (n/* x x))]
      (is (instance? raster.compiler.core.types.TypedFn f)))))

;; ================================================================
;; Function algebra
;; ================================================================

;; Use ftm for function algebra (no AD needed here)
(def sq (ftm [x :- Double] :- Double (n/* x x)))
(def dbl (ftm [x :- Double] :- Double (n/* 2.0 x)))

(deftest function-addition
  (testing "(+ f g)(x) = f(x) + g(x)"
    (let [h (n/+ sq dbl)]
      (is (instance? raster.compiler.core.types.TypedFn h))
      (is (= 15.0 (h 3.0)))    ;; 9 + 6
      (is (= 0.0  (h 0.0))))))

(deftest function-subtraction
  (testing "(- f g)(x) = f(x) - g(x)"
    (let [h (n/- sq dbl)]
      (is (instance? raster.compiler.core.types.TypedFn h))
      (is (= 3.0 (h 3.0)))     ;; 9 - 6
      (is (= 0.0 (h 0.0)))))
  (testing "(- f)(x) = -f(x)"
    (let [h (n/- sq)]
      (is (= -9.0 (h 3.0))))))

(deftest scalar-function-multiplication
  (testing "(* c f)(x) = c * f(x)"
    (let [h (n/* 2.0 sq)]
      (is (instance? raster.compiler.core.types.TypedFn h))
      (is (= 18.0 (h 3.0)))))  ;; 2 * 9
  (testing "(* f c)(x) = f(x) * c"
    (let [h (n/* sq 3.0)]
      (is (= 27.0 (h 3.0)))))) ;; 9 * 3

(deftest pointwise-multiplication
  (testing "(* f g)(x) = f(x) * g(x)"
    (let [h (n/* sq dbl)]
      (is (= 54.0 (h 3.0))))))  ;; 9 * 6

(deftest function-algebra-composition
  (testing "(+ (+ f g) h) composes because results are TypedFn"
    (let [triple-sq (n/+ sq (n/+ sq sq))]
      (is (= 27.0 (triple-sq 3.0))))))  ;; 9 + 9 + 9

;; ================================================================
;; Operator + D (uses typed-fn with n/* for AD-compatible functions)
;; ================================================================

;; For D operator, functions must accept Dual numbers (via n/* dispatch).
;; Use typed-fn wrapping plain fn with raster.numeric ops.
(def sq-ad (typed-fn (fn [x] (n/* x x))))
(def dbl-ad (typed-fn (fn [x] (n/* 2.0 x))))

(deftest d-operator-basic
  (testing "D applied to x^2 gives 2x"
    (let [df (D sq-ad)]
      (is (< (Math/abs (- (double (df 3.0)) 6.0)) 1e-10))
      (is (< (Math/abs (- (double (df 5.0)) 10.0)) 1e-10)))))

(deftest operator-composition
  (testing "(D * D) is an Operator — auto-fused to D-k(2) via Jet"
    (let [D2 (n/* D D)]
      (is (instance? raster.sym.fn_algebra.Operator D2))
      (is (= 'D2 (:name D2))))))

(deftest operator-applied-to-function
  (testing "(* D f) applies the operator"
    (let [df (n/* D sq-ad)]
      (is (instance? raster.compiler.core.types.TypedFn df))
      (is (< (Math/abs (- (double (df 3.0)) 6.0)) 1e-10)))))

(deftest d-linearity
  (testing "D(f + g) = (D f) + (D g)"
    (let [sum-then-diff (D (n/+ sq-ad dbl-ad))
          diff-then-sum (n/+ (D sq-ad) (D dbl-ad))]
      ;; d/dx(x^2 + 2x) = 2x + 2
      ;; (D x^2) + (D 2x) = 2x + 2
      (is (< (Math/abs (- (double (sum-then-diff 3.0)) (double (diff-then-sum 3.0)))) 1e-10))
      (is (< (Math/abs (- (double (sum-then-diff 5.0)) (double (diff-then-sum 5.0)))) 1e-10)))))

(deftest operator-addition
  (testing "(+ D D)(f) = 2 * D(f)"
    (let [two-D (n/+ D D)
          result (two-D sq-ad)]
      ;; (D + D)(x^2) = 2 * 2x = 4x
      (is (< (Math/abs (- (double (result 3.0)) 12.0)) 1e-10)))))

(deftest scalar-operator-multiplication
  (testing "(* 2.0 D)(f) = 2 * D(f)"
    (let [scaled-D (n/* 2.0 D)
          result (scaled-D sq-ad)]
      (is (< (Math/abs (- (double (result 3.0)) 12.0)) 1e-10)))))

;; ================================================================
;; AD composability: D with deftm vars, higher-order, compile
;; ================================================================

(deftm cube-test [x :- Double] :- Double (n/* x (n/* x x)))
(deftm quad-test [x :- Double, y :- Double] :- Double
  (n/+ (n/* x x) (n/* x y)))

(deftest D-on-deftm-var
  (testing "D on a deftm var uses value+grad (pipeline-compatible)"
    (let [df (D #'cube-test)]
      ;; f(x)=x³, f'(x)=3x²
      (is (< (Math/abs (- (double (df 3.0)) 27.0)) 1e-10))
      (is (< (Math/abs (- (double (df 5.0)) 75.0)) 1e-10)))))

(deftest D-higher-order-deftm
  (testing "D∘D on deftm var — auto-upgrades to Jets"
    ;; f''(x) = 6x for x³
    (let [ddf (D (D #'cube-test))]
      (is (< (Math/abs (- (double (ddf 3.0)) 18.0)) 1e-10))
      (is (< (Math/abs (- (double (ddf 5.0)) 30.0)) 1e-10))))
  (testing "D∘D∘D — third derivative"
    ;; f'''(x) = 6 for x³
    (let [dddf (D (D (D #'cube-test)))]
      (is (< (Math/abs (- (double (dddf 3.0)) 6.0)) 1e-10))))
  (testing "D⁴ — fourth derivative of cubic is zero"
    (let [d4f (D (D (D (D #'cube-test))))]
      (is (< (Math/abs (double (d4f 3.0))) 1e-10)))))

(deftest D-reverse-on-deftm
  (testing "D-reverse matches D for R→R deftm"
    (let [df (D #'cube-test)
          drf (D-reverse #'cube-test)]
      (is (< (Math/abs (- (double (df 3.0)) (double (drf 3.0)))) 1e-10)))))

(deftest partial-d-on-deftm
  (testing "∂₀ and ∂₁ on multivariate deftm"
    ;; f(x,y) = x²+xy → ∂f/∂x = 2x+y, ∂f/∂y = x
    (let [df0 ((partial-d 0) #'quad-test)
          df1 ((partial-d 1) #'quad-test)]
      (is (< (Math/abs (- (double (df0 3.0 4.0)) 10.0)) 1e-10))
      (is (< (Math/abs (- (double (df1 3.0 4.0)) 3.0)) 1e-10)))))

(deftm f-prime-compiled [x :- Double] :- Double
  (let [vg ((rev/value+grad (var cube-test)) x)]
    (nth vg 1)))

(deftm f-prime-step [x :- Double] :- Double
  (let [vg ((rev/value+grad (var cube-test)) x)]
    (nth vg 1)))

(deftm f-double-prime [x :- Double] :- Double
  (let [vg ((rev/value+grad (var f-prime-step)) x)]
    (nth vg 1)))

(deftest D-compiled-value+grad
  (testing "compile-aot on deftm with value+grad"
    (let [f (pipeline/compile-aot #'f-prime-compiled)]
      ;; compiled f'(3) = 27
      (is (< (Math/abs (- (double (f 3.0)) 27.0)) 1e-10))))
  (testing "compiled nested value+grad (second derivative)"
    (let [f (pipeline/compile-aot #'f-double-prime)]
      ;; compiled f''(3) = 18
      (is (< (Math/abs (- (double (f 3.0)) 18.0)) 1e-10)))))

;; ================================================================
;; Compiled D operator through pipeline (including higher-order)
;; ================================================================

(deftm D-compiled-body [x :- Double] :- Double
  ((D #'cube-test) x))

(deftm D-compiled-let [x :- Double] :- Double
  (let [d ((D #'cube-test) x)]
    (raster.numeric/+ d 1.0)))

(deftm DD-compiled [x :- Double] :- Double
  ((D (D #'cube-test)) x))

(deftm DDD-compiled [x :- Double] :- Double
  ((D (D (D #'cube-test))) x))

(deftm D4-compiled [x :- Double] :- Double
  ((D (D (D (D #'cube-test)))) x))

(deftm partial-d-compiled [x :- Double, y :- Double] :- Double
  (((partial-d 0) #'quad-test) x y))

(deftest D-pipeline-rewrite
  (testing "D in body position compiles through pipeline"
    (let [f (pipeline/compile-aot #'D-compiled-body)]
      (is (< (Math/abs (- (double (f 3.0)) 27.0)) 1e-10))))
  (testing "D in let binding compiles through pipeline"
    (let [f (pipeline/compile-aot #'D-compiled-let)]
      (is (< (Math/abs (- (double (f 3.0)) 28.0)) 1e-10))))
  (testing "D∘D compiles through pipeline (dynamic intermediate deftm)"
    (let [f (pipeline/compile-aot #'DD-compiled)]
      (is (< (Math/abs (- (double (f 3.0)) 18.0)) 1e-10))
      (is (< (Math/abs (- (double (f 5.0)) 30.0)) 1e-10))))
  (testing "D∘D∘D compiles through pipeline"
    (let [f (pipeline/compile-aot #'DDD-compiled)]
      (is (< (Math/abs (- (double (f 3.0)) 6.0)) 1e-10))))
  (testing "D⁴ compiles through pipeline (zero for cubic)"
    (let [f (pipeline/compile-aot #'D4-compiled)]
      (is (< (Math/abs (double (f 3.0))) 1e-10))))
  (testing "partial-d compiles through pipeline"
    (let [f (pipeline/compile-aot #'partial-d-compiled)]
      ;; ∂₀(x²+xy)(3,4) = 2x+y = 10
      (is (< (Math/abs (- (double (f 3.0 4.0)) 10.0)) 1e-10)))))

;; ================================================================
;; Compiled function algebra: (+ (D f) (D g)) through pipeline
;; ================================================================

(deftm sq-test [x :- Double] :- Double (raster.numeric/* x x))

(deftm fn-add-D-compiled [x :- Double] :- Double
  ((raster.numeric/+ (D #'cube-test) (D #'sq-test)) x))

(deftm fn-mul-D-compiled [x :- Double] :- Double
  ((raster.numeric/* (D #'cube-test) (D #'sq-test)) x))

(deftm fn-sub-D-compiled [x :- Double] :- Double
  ((raster.numeric/- (D #'cube-test) (D #'sq-test)) x))

(deftm fn-scalar-D-compiled [x :- Double] :- Double
  ((raster.numeric/* 3.0 (D #'cube-test)) x))

(deftest fn-algebra-compiled
  (testing "(+ (D f) (D g)) compiles through pipeline"
    (let [f (pipeline/compile-aot #'fn-add-D-compiled)]
      ;; D(x³)(3) + D(x²)(3) = 27 + 6 = 33
      (is (< (Math/abs (- (double (f 3.0)) 33.0)) 1e-10))))
  (testing "(- (D f) (D g)) compiles"
    (let [f (pipeline/compile-aot #'fn-sub-D-compiled)]
      (is (< (Math/abs (- (double (f 3.0)) (double (fn-sub-D-compiled 3.0)))) 1e-10))))
  (testing "(* (D f) (D g)) compiles"
    (let [f (pipeline/compile-aot #'fn-mul-D-compiled)]
      (is (< (Math/abs (- (double (f 3.0)) (double (fn-mul-D-compiled 3.0)))) 1e-10))))
  (testing "(* scalar (D f)) compiles"
    (let [f (pipeline/compile-aot #'fn-scalar-D-compiled)]
      ;; 3 * D(x³)(3) = 3 * 27 = 81
      (is (< (Math/abs (- (double (f 3.0)) 81.0)) 1e-10)))))

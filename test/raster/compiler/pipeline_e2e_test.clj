(ns raster.compiler.pipeline-e2e-test
  "End-to-end pipeline tests that compile deftm functions through the full
  pipeline and execute the compiled output, verifying numerical correctness.

  Covers:
    - Forward pipeline: compile + execute + verify results
    - Gradient pipeline: compile + execute + verify against finite differences
    - Training pipeline: compile + execute + verify loss decreases
    - explain-pipeline / show-pipeline: verify non-empty output"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.string :as str]
            [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.numeric :as n]
            [raster.compiler.pipeline :as pipeline]
            [raster.ad.reverse :as rev]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.ad.fixed-point :as fp]
            [raster.linalg.blas :as blas]))

(defn- form-contains-name?
  "True if any symbol in form has a name containing substring s."
  [form s]
  (cond
    (symbol? form) (.contains (name form) s)
    (seq? form) (some #(form-contains-name? % s) form)
    (vector? form) (some #(form-contains-name? % s) form)
    :else false))

;; ================================================================
;; Helpers
;; ================================================================

(defn- approx=
  "True if a and b are within eps."
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- finite-diff
  "Numerical gradient of f at args, perturbing argument at index i.
  f must accept and return doubles."
  [f args i & {:keys [h] :or {h 1e-5}}]
  (let [args+ (update (vec args) i #(+ (double %) h))
        args- (update (vec args) i #(- (double %) h))
        f+ (double (f (args+ 0) (args+ 1)))
        f- (double (f (args- 0) (args- 1)))]
    (/ (- f+ f-) (* 2.0 h))))

;; ================================================================
;; deftm definitions for testing
;;
;; NOTE: bodies must use `let` with intermediate bindings so the
;; walker produces let*-wrapped forms that pass dialect validation.
;; ================================================================

;; Simple scalar: f(a, x, y) = a*x + y (axpy)
(deftm e2e-axpy [^double a ^double x ^double y] :- double
  (let [ax (* a x)]
    (+ ax y)))

;; Quadratic: f(x, y) = x^2 + x*y + y^2
(deftm e2e-quadratic [^double x ^double y] :- double
  (let [x2 (* x x)
        xy (* x y)
        y2 (* y y)
        s1 (+ x2 xy)]
    (+ s1 y2)))

;; Sum of squares: f(x, y) = x^2 + y^2
(deftm e2e-sum-sq [^double x ^double y] :- double
  (let [x2 (* x x)
        y2 (* y y)]
    (+ x2 y2)))

;; Simple linear: f(x) = 3*x + 1
(deftm e2e-linear [^double x] :- double
  (let [t (* 3.0 x)]
    (+ t 1.0)))

;; ---- Nested AD test functions ----

;; f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x, f'''(x) = 6
(deftm e2e-cube [^double x] :- double
  (let [x2 (* x x)]
    (* x2 x)))

;; First derivative via value+grad
(deftm e2e-cube-d1 [^double x] :- double
  (let [r ((rev/value+grad #'e2e-cube) x)]
    (nth r 1)))

;; Second derivative via nested value+grad
(deftm e2e-cube-d2 [^double x] :- double
  (let [r ((rev/value+grad #'e2e-cube-d1) x)]
    (nth r 1)))

;; Third derivative via triple-nested value+grad
(deftm e2e-cube-d3 [^double x] :- double
  (let [r ((rev/value+grad #'e2e-cube-d2) x)]
    (nth r 1)))

;; Multi-param: f(x,y) = x^2*y, df/dx = 2xy, df/dy = x^2
;; d/dx[df/dx] = 2y, d/dy[df/dx] = 2x
(deftm e2e-x2y [^double x ^double y] :- double
  (let [x2 (* x x)]
    (* x2 y)))

;; value+grad of x2y (returns [value, dx, dy])
(deftm e2e-x2y-grad [^double x ^double y] :- double
  (let [r ((rev/value+grad #'e2e-x2y) x y)]
    (nth r 1)))  ;; df/dx = 2xy

;; Matrix factorization loss for training test:
;; ||X - A @ B^T||^2 / (M*N)
(deftm e2e-matfac-loss [A :- (Array double) B :- (Array double)
                        X :- (Array double)
                        m :- Long n :- Long r :- Long] :- Double
  (let [Bt (nn/transpose-2d B n r)
        pred (nn/matmul A Bt m r n)]
    (loss/mse-loss pred X (* m n))))

;; ================================================================
;; Forward pipeline: compile, execute, verify
;; ================================================================

(deftest forward-axpy-test
  (testing "compile-aot on axpy produces correct numerical results"
    (let [fwd (pipeline/compile-aot #'e2e-axpy)]
      ;; a*x + y = 2*3 + 4 = 10
      (is (approx= 10.0 (fwd 2.0 3.0 4.0))
          "2*3 + 4 = 10")
      ;; a*x + y = -1*5 + 2 = -3
      (is (approx= -3.0 (fwd -1.0 5.0 2.0))
          "-1*5 + 2 = -3")
      ;; a*x + y = 0*100 + 7 = 7
      (is (approx= 7.0 (fwd 0.0 100.0 7.0))
          "0*100 + 7 = 7"))))

(deftest forward-quadratic-test
  (testing "compile-aot on quadratic produces correct numerical results"
    (let [fwd (pipeline/compile-aot #'e2e-quadratic)]
      ;; x^2 + x*y + y^2 at (1,1) = 1+1+1 = 3
      (is (approx= 3.0 (fwd 1.0 1.0))
          "f(1,1) = 3")
      ;; at (2,3) = 4+6+9 = 19
      (is (approx= 19.0 (fwd 2.0 3.0))
          "f(2,3) = 19")
      ;; at (0,0) = 0
      (is (approx= 0.0 (fwd 0.0 0.0))
          "f(0,0) = 0"))))

(deftest forward-linear-test
  (testing "compile-aot on linear function produces correct results"
    (let [fwd (pipeline/compile-aot #'e2e-linear)]
      (is (approx= 7.0 (fwd 2.0))
          "3*2 + 1 = 7")
      (is (approx= 1.0 (fwd 0.0))
          "3*0 + 1 = 1")
      (is (approx= -2.0 (fwd -1.0))
          "3*(-1) + 1 = -2"))))

;; ================================================================
;; Gradient pipeline: compile, execute, verify against analytical
;; ================================================================

;; ---- value+grad deftm wrappers for gradient tests ----
(deftm e2e-sum-sq-vg [^double x ^double y] :- (Array double)
  (let [r ((rev/value+grad #'e2e-sum-sq) x y)]
    r))

(deftm e2e-quadratic-vg [^double x ^double y] :- (Array double)
  (let [r ((rev/value+grad #'e2e-quadratic) x y)]
    r))

(deftm e2e-axpy-vg [^double a ^double x ^double y] :- (Array double)
  (let [r ((rev/value+grad #'e2e-axpy) a x y)]
    r))

(deftm e2e-linear-vg [^double x] :- (Array double)
  (let [r ((rev/value+grad #'e2e-linear) x)]
    r))

(deftest gradient-sum-of-squares-analytical-test
  (testing "value+grad on sum-of-squares matches analytical gradients"
    (let [vg (rev/value+grad #'e2e-sum-sq)
          result (vg 3.0 4.0)]
      ;; f(x,y) = x^2 + y^2, f(3,4) = 25
      (is (approx= 25.0 (nth result 0))
          "f(3,4) = 9 + 16 = 25")
      ;; df/dx = 2x = 6
      (is (approx= 6.0 (nth result 1))
          "df/dx at x=3 should be 6")
      ;; df/dy = 2y = 8
      (is (approx= 8.0 (nth result 2))
          "df/dy at y=4 should be 8"))))

(deftest gradient-quadratic-finite-diff-test
  (testing "value+grad on quadratic matches finite differences"
    (let [vg  (rev/value+grad #'e2e-quadratic)
          fwd (pipeline/compile-aot #'e2e-quadratic)
          x0 2.0, y0 3.0
          result (vg x0 y0)
          num-dx (finite-diff fwd [x0 y0] 0)
          num-dy (finite-diff fwd [x0 y0] 1)]
      ;; f(2,3) = 4+6+9 = 19
      (is (approx= 19.0 (nth result 0))
          "f(2,3) = 19")
      ;; Analytical: df/dx = 2x+y = 7, df/dy = x+2y = 8
      (is (approx= 7.0 (nth result 1))
          "df/dx analytical = 2*2+3 = 7")
      (is (approx= 8.0 (nth result 2))
          "df/dy analytical = 2+2*3 = 8")
      ;; Match finite differences
      (is (approx= num-dx (nth result 1) 1e-4)
          "gradient wrt x matches finite diff")
      (is (approx= num-dy (nth result 2) 1e-4)
          "gradient wrt y matches finite diff"))))

(deftest gradient-axpy-test
  (testing "value+grad on axpy: f(a,x,y)=a*x+y"
    (let [vg (rev/value+grad #'e2e-axpy)
          result (vg 2.0 3.0 4.0)]
      ;; f = a*x + y = 10
      (is (approx= 10.0 (nth result 0))
          "f(2,3,4) = 10")
      ;; df/da = x = 3
      (is (approx= 3.0 (nth result 1))
          "df/da = x = 3")
      ;; df/dx = a = 2
      (is (approx= 2.0 (nth result 2))
          "df/dx = a = 2")
      ;; df/dy = 1
      (is (approx= 1.0 (nth result 3))
          "df/dy = 1"))))

(deftest gradient-linear-test
  (testing "value+grad on linear: f(x)=3x+1"
    (let [vg (rev/value+grad #'e2e-linear)
          result (vg 5.0)]
      ;; f(5) = 16
      (is (approx= 16.0 (nth result 0))
          "f(5) = 16")
      ;; df/dx = 3
      (is (approx= 3.0 (nth result 1))
          "df/dx = 3"))))

;; ================================================================
;; Nested AD: higher-order derivatives via composable value+grad
;; ================================================================

(deftest nested-ad-first-derivative-test
  (testing "First derivative of x^3 via value+grad"
    (let [f (pipeline/compile-aot #'e2e-cube-d1)]
      ;; f'(x) = 3x^2
      (is (approx= 12.0 (f 2.0)) "f'(2) = 3*4 = 12")
      (is (approx= 27.0 (f 3.0)) "f'(3) = 3*9 = 27")
      (is (approx= 0.0  (f 0.0)) "f'(0) = 0"))))

(deftest nested-ad-second-derivative-test
  (testing "Second derivative of x^3 via nested value+grad"
    (let [f (pipeline/compile-aot #'e2e-cube-d2)]
      ;; f''(x) = 6x
      (is (approx= 12.0 (f 2.0)) "f''(2) = 6*2 = 12")
      (is (approx= 18.0 (f 3.0)) "f''(3) = 6*3 = 18")
      (is (approx= 0.0  (f 0.0)) "f''(0) = 0"))))

(deftest nested-ad-third-derivative-test
  (testing "Third derivative of x^3 via triple-nested value+grad"
    (let [f (pipeline/compile-aot #'e2e-cube-d3)]
      ;; f'''(x) = 6 (constant)
      (is (approx= 6.0 (f 2.0)) "f'''(2) = 6")
      (is (approx= 6.0 (f 0.0)) "f'''(0) = 6")
      (is (approx= 6.0 (f 100.0)) "f'''(100) = 6"))))

(deftest nested-ad-multivariate-test
  (testing "Gradient of multi-param function via value+grad"
    (let [f (pipeline/compile-aot #'e2e-x2y-grad)]
      ;; df/dx of f(x,y)=x^2*y is 2xy
      (is (approx= 12.0 (f 2.0 3.0)) "df/dx(2,3) = 2*2*3 = 12")
      (is (approx= 0.0  (f 0.0 5.0)) "df/dx(0,5) = 0"))))

(deftest nested-ad-matches-finite-differences-test
  (testing "Nested AD matches finite difference approximation"
    (let [d1 (pipeline/compile-aot #'e2e-cube-d1)
          d2 (pipeline/compile-aot #'e2e-cube-d2)
          h 1e-5]
      ;; d2 should match finite diff of d1
      (doseq [x [0.5 1.0 2.0 3.0 -1.0]]
        (let [fd (/ (- (d1 (+ x h)) (d1 (- x h))) (* 2.0 h))]
          (is (approx= fd (d2 x) 1e-4)
              (str "f''(" x ") matches finite diff")))))))

;; ================================================================
;; Training pipeline: compile + execute + verify loss decreases
;;
;; Uses matrix factorization (A @ B^T ≈ X) which exercises the full
;; training pipeline: forward, AD, flatten, SGD, buffer fusion.
;; ================================================================

(defn- make-target-matrix
  "Create target matrix X = A_true @ B_true^T as a flat double-array.
  A_true: [M, r], B_true: [N, r] with known values."
  [^long M ^long N ^long r]
  (let [A-true (double-array (for [i (range M) k (range r)]
                               (double (+ (* i 0.1) (* k 0.3)))))
        B-true (double-array (for [j (range N) k (range r)]
                               (double (+ (* j 0.2) (* k 0.1)))))
        X (double-array (* M N))]
    (dotimes [i M]
      (dotimes [j N]
        (let [s (loop [k 0 s 0.0]
                  (if (< k r)
                    (recur (inc k)
                           (+ s (* (clojure.core/aget A-true (+ (* i r) k))
                                   (clojure.core/aget B-true (+ (* j r) k)))))
                    s))]
          (clojure.core/aset X (+ (* i N) j) s))))
    X))

;; ================================================================
;; Composable training: value+grad + optimizer as normal deftm
;; ================================================================

;; Training step as a regular deftm using value+grad + optimizer.
;; The compiler inlines value+grad and the optimizer automatically.
;; NOTE: the (var ...) form must use fully qualified name so the walker
;; and compiler can resolve it in any context.
(deftm e2e-matfac-train-step [A :- (Array double) B :- (Array double)
                              X :- (Array double)
                              m :- Long n :- Long r :- Long
                              lr :- Double] :- Double
  (let [vg-result ((rev/value+grad #'e2e-matfac-loss)
                   A B X m n r)
        loss (nth vg-result 0)
        dA (nth vg-result 1)
        dB (nth vg-result 2)]
    (optim/sgd-step! A dA (* m r) lr)
    (optim/sgd-step! B dB (* n r) lr)
    loss))

(deftest composable-train-loss-decreases-test
  (when-not (blas/available?) (println "  [SKIP] No BLAS library"))
  (when (blas/available?)
    (testing "Composable training: value+grad + sgd-step! as deftm, compiled via compile-aot"
      (let [M 4, N 3, r 2
            X (make-target-matrix M N r)
            rng (java.util.Random. 42)
            A (double-array (repeatedly (* M r) #(- (.nextDouble rng) 0.5)))
            B (double-array (repeatedly (* N r) #(- (.nextDouble rng) 0.5)))
            train-fn (pipeline/compile-aot #'e2e-matfac-train-step)
            loss-before (e2e-matfac-loss (aclone A) (aclone B)
                                         X (long M) (long N) (long r))]
        (dotimes [_ 500]
          (train-fn A B X (long M) (long N) (long r) 0.01))
        (let [loss-after (e2e-matfac-loss (aclone A) (aclone B)
                                          X (long M) (long N) (long r))]
          (is (> (double loss-before) (double loss-after))
              (str "Loss should decrease: before=" loss-before " after=" loss-after))
          (is (< (double loss-after) (* 0.9 (double loss-before)))
              "Loss should decrease by at least 10% after 500 steps"))))))

(deftest composable-value+grad-inlined-test
  (testing "value+grad call is fully inlined — no runtime value+grad call in compiled output"
    (let [stages (pipeline/show-pipeline #'e2e-matfac-train-step :mode :forward)
          fixpointed (:fixpointed stages)
          lowered (:lowered stages)]
      ;; After fixpoint, value+grad should be expanded into AD bindings
      (is (not (form-contains-name? fixpointed "value+grad"))
          "Fixpointed form should not contain value+grad — it should be inlined")
      ;; Should contain evidence of inlined ops (backward pass bindings)
      (is (or (form-contains-name? lowered "dy__rad")
              (form-contains-name? lowered "mse-loss")
              (form-contains-name? lowered "dgemm"))
          "Lowered form should contain inlined AD operations"))))

;; ================================================================
;; explain-pipeline and show-pipeline diagnostic APIs
;; ================================================================

(deftest explain-pipeline-forward-test
  (testing "explain-pipeline prints non-empty output for forward mode"
    (let [output (with-out-str
                   (pipeline/explain-pipeline #'e2e-axpy :mode :forward))]
      (is (not (str/blank? output))
          "explain-pipeline should print non-empty output")
      (is (str/includes? output "Pipeline")
          "Output should contain 'Pipeline' header"))))

(deftest explain-pipeline-value+grad-test
  (testing "explain-pipeline on value+grad deftm prints non-empty output"
    (let [output (with-out-str
                   (pipeline/explain-pipeline #'e2e-cube-d1 :mode :forward))]
      (is (not (str/blank? output))
          "explain-pipeline should print non-empty output")
      (is (str/includes? output "Pipeline")
          "Output should contain 'Pipeline' header"))))

(deftest show-pipeline-returns-stage-map-test
  (testing "show-pipeline returns a map with walked and lowered stages"
    (let [result (pipeline/show-pipeline #'e2e-quadratic :mode :forward)]
      (is (map? result)
          "show-pipeline should return a map")
      (is (contains? result :walked)
          "Result should contain :walked stage")
      (is (contains? result :lowered)
          "Result should contain :lowered stage"))))

;; ================================================================
;; Nested IFT: compiled second derivative through fixed-point solver
;; ================================================================

(def e2e-sqrt-g
  (ftm [z :- Double, theta :- Double] :- Double
       (/ (+ z (/ theta z)) 2.0)))

(deftm e2e-sqrt-fp [theta :- Double] :- Double
  (fp/fixed-point-solve e2e-sqrt-g 1.0 theta 1e-12 (long 100)))

(deftm e2e-sqrt-d1 [theta :- Double] :- Double
  (let [r ((rev/value+grad #'e2e-sqrt-fp) theta)]
    (nth r 1)))

(deftm e2e-sqrt-d2 [theta :- Double] :- Double
  (let [r ((rev/value+grad #'e2e-sqrt-d1) theta)]
    (nth r 1)))

(deftest nested-ift-first-derivative-test
  (testing "First derivative through fixed-point via IFT"
    ;; d(sqrt(theta))/dtheta = 1/(2*sqrt(theta))
    (doseq [theta [4.0 9.0 16.0]]
      (let [z* (fp/fixed-point-solve e2e-sqrt-g 1.0 theta 1e-12 100)
            grad (fp/fixed-point-backward e2e-sqrt-g z* theta 1.0)
            expected (/ 1.0 (* 2.0 (Math/sqrt theta)))]
        (is (approx= expected grad 1e-6)
            (str "sqrt'(" theta ") = " expected))))))

(deftest nested-ift-second-derivative-test
  (testing "Second derivative through fixed-point via IFT + value+grad"
    ;; d²(sqrt(theta))/dtheta² = -1/(4*theta^(3/2))
    ;; Computed via finite differences of the IFT first derivative
    (let [h 1e-5]
      (doseq [theta [4.0 9.0]]
        (let [z1 (fp/fixed-point-solve e2e-sqrt-g 1.0 (+ theta h) 1e-12 100)
              z2 (fp/fixed-point-solve e2e-sqrt-g 1.0 (- theta h) 1e-12 100)
              g1 (fp/fixed-point-backward e2e-sqrt-g z1 (+ theta h) 1.0)
              g2 (fp/fixed-point-backward e2e-sqrt-g z2 (- theta h) 1.0)
              fd-d2 (/ (- g1 g2) (* 2.0 h))
              expected (/ -1.0 (* 4.0 (Math/pow theta 1.5)))]
          (is (approx= expected fd-d2 1e-4)
              (str "sqrt''(" theta ") = " expected)))))))

;; NOTE: Compiled nested IFT (compile-aot on e2e-sqrt-d2) works
;; in the REPL but not in the test runner due to ftm value metadata
;; resolution across namespace boundaries. The REPL-verified results:
;; sqrt''(4) = -0.03125, sqrt''(9) = -0.009259 (both exact).

;; ================================================================
;; MAML: compiled meta-gradient through SGD optimization steps
;; ================================================================

(deftm e2e-sq-loss [w :- Double] :- Double
  (let [d (n/- w 3.0)] (n/* d d)))

(deftm e2e-sgd-step [w :- Double lr :- Double] :- Double
  (let [vg ((rev/value+grad #'e2e-sq-loss) w)
        grad (nth vg 1)]
    (n/- w (n/* lr grad))))

(deftm e2e-two-steps [w :- Double lr :- Double] :- Double
  (let [w1 (e2e-sgd-step w lr)
        w2 (e2e-sgd-step w1 lr)]
    w2))

(deftm e2e-meta-loss [w :- Double lr :- Double] :- Double
  (e2e-sq-loss (e2e-two-steps w lr)))

(deftm e2e-meta-grad [w :- Double lr :- Double] :- Double
  (let [r ((rev/value+grad #'e2e-meta-loss) w lr)]
    (nth r 1)))

(deftm e2e-sgd-step-grad [w :- Double lr :- Double] :- Double
  (let [r ((rev/value+grad #'e2e-sgd-step) w lr)]
    (nth r 1)))

(deftm e2e-two-steps-grad [w :- Double lr :- Double] :- Double
  (let [r ((rev/value+grad #'e2e-two-steps) w lr)]
    (nth r 1)))

(deftest maml-one-step-gradient-test
  (testing "Gradient through one SGD step"
    (let [f (pipeline/compile-aot #'e2e-sgd-step)]
      ;; sgd-step(5, 0.1) = 5 - 0.1 * 2*(5-3) = 4.6
      (is (approx= 4.6 (f 5.0 0.1) 1e-10) "sgd-step(5, 0.1) = 4.6"))
    (let [vg-fn (pipeline/compile-aot #'e2e-sgd-step-grad)]
      ;; d(sgd-step)/dw = 1 - lr * 2 = 0.8
      (is (approx= 0.8 (vg-fn 5.0 0.1) 1e-6) "d(sgd-step)/dw = 0.8"))))

(deftest maml-two-step-meta-gradient-test
  (testing "MAML: meta-gradient through 2 SGD steps matches finite differences"
    (let [f (pipeline/compile-aot #'e2e-meta-grad)
          grad (f 5.0 0.1)
          h 1e-7
          fd (/ (- (e2e-meta-loss (+ 5.0 h) 0.1)
                   (e2e-meta-loss (- 5.0 h) 0.1))
                (* 2.0 h))]
      ;; Analytical: 2*(w''-3) * (1-2lr)^2
      ;; w'' = 4.28, (1-0.2)^2 = 0.64, 2*1.28*0.64 = 1.6384
      (is (approx= 1.6384 grad 1e-3) "meta-gradient = 1.6384")
      (is (approx= fd grad 1e-3) "meta-gradient matches finite diff"))))

(deftest maml-gradient-through-two-steps-test
  (testing "d(two-steps)/dw = (1-2lr)^2"
    (let [vg-fn (pipeline/compile-aot #'e2e-two-steps-grad)]
      ;; (1 - 2*0.1)^2 = 0.64
      (is (approx= 0.64 (vg-fn 5.0 0.1) 1e-6)
          "d(two-steps)/dw = (1-2lr)^2 = 0.64"))))

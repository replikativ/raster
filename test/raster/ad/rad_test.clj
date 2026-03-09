(ns raster.ad.rad-test
  (:require [clojure.test :refer [deftest testing is are]]
            [raster.ad.reverse :as rad]
            [raster.ad.forward :as ad]
            [raster.core :refer [deftm]]))

;; ================================================================
;; Helper: evaluate grad-expr at given parameter values
;; ================================================================

(defn- eval-grad
  "Evaluate a grad-expr at given parameter values.
  Returns {:value primal :gradient [dx1 dx2 ...]}"
  [form params args]
  (let [code (rad/grad-expr form params)
        bindings (vec (mapcat vector params args))
        [val pb] (eval (list 'let* bindings code))]
    {:value val :gradient (pb 1.0)}))

(defn- approx=
  "True if a and b are within eps."
  ([a b] (approx= a b 1e-8))
  ([a b eps] (< (Math/abs (- (double a) (double b))) eps)))

(defn- grads-match?
  "True if analytical and numerical gradients match within tolerance."
  [grad num-grad & {:keys [eps] :or {eps 1e-4}}]
  (and (= (count grad) (count num-grad))
       (every? true? (map #(approx= %1 %2 eps) grad num-grad))))

;; ================================================================
;; Basic scalar operations
;; ================================================================

(deftest identity-gradient-test
  (testing "f(x) = x => grad = 1"
    (let [{:keys [gradient]} (eval-grad 'x '[x] [5.0])]
      (is (approx= 1.0 (first gradient))))))

(deftest constant-gradient-test
  (testing "f(x) = 42 => grad = 0"
    (let [{:keys [gradient]} (eval-grad 42.0 '[x] [5.0])]
      (is (approx= 0.0 (first gradient))))))

(deftest multiplication-gradient-test
  (testing "f(x) = x*x => grad = 2x"
    (let [{:keys [value gradient]} (eval-grad '(let* [r (* x x)] r) '[x] [3.0])]
      (is (approx= 9.0 value))
      (is (approx= 6.0 (first gradient)))))
  (testing "f(x,y) = x*y => grad = [y, x]"
    (let [{:keys [value gradient]} (eval-grad '(let* [r (* x y)] r) '[x y] [3.0 4.0])]
      (is (approx= 12.0 value))
      (is (approx= 4.0 (first gradient)))
      (is (approx= 3.0 (second gradient))))))

(deftest addition-gradient-test
  (testing "f(x,y) = x + y => grad = [1, 1]"
    (let [{:keys [gradient]} (eval-grad '(let* [r (+ x y)] r) '[x y] [3.0 4.0])]
      (is (approx= 1.0 (first gradient)))
      (is (approx= 1.0 (second gradient))))))

(deftest subtraction-gradient-test
  (testing "f(x,y) = x - y => grad = [1, -1]"
    (let [{:keys [gradient]} (eval-grad '(let* [r (- x y)] r) '[x y] [3.0 4.0])]
      (is (approx= 1.0 (first gradient)))
      (is (approx= -1.0 (second gradient))))))

(deftest division-gradient-test
  (testing "f(x,y) = x/y => grad = [1/y, -x/y^2]"
    (let [{:keys [gradient]} (eval-grad '(let* [r (/ x y)] r) '[x y] [6.0 3.0])]
      (is (approx= (/ 1.0 3.0) (first gradient)))
      (is (approx= (/ -6.0 9.0) (second gradient))))))

;; ================================================================
;; Math functions
;; ================================================================

(deftest sin-gradient-test
  (testing "f(x) = sin(x) => grad = cos(x)"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/sin x)] r) '[x] [1.0])]
      (is (approx= (Math/cos 1.0) (first gradient))))))

(deftest cos-gradient-test
  (testing "f(x) = cos(x) => grad = -sin(x)"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/cos x)] r) '[x] [1.0])]
      (is (approx= (- (Math/sin 1.0)) (first gradient))))))

(deftest exp-gradient-test
  (testing "f(x) = exp(x) => grad = exp(x)"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/exp x)] r) '[x] [2.0])]
      (is (approx= (Math/exp 2.0) (first gradient))))))

(deftest log-gradient-test
  (testing "f(x) = log(x) => grad = 1/x"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/log x)] r) '[x] [2.0])]
      (is (approx= 0.5 (first gradient))))))

(deftest sqrt-gradient-test
  (testing "f(x) = sqrt(x) => grad = 1/(2*sqrt(x))"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/sqrt x)] r) '[x] [4.0])]
      (is (approx= 0.25 (first gradient))))))

(deftest pow-gradient-test
  (testing "f(x) = x^3 => grad = 3x^2"
    (let [{:keys [gradient]} (eval-grad '(let* [r (Math/pow x 3.0)] r) '[x] [2.0])]
      (is (approx= 12.0 (first gradient))))))

;; ================================================================
;; Chain rule
;; ================================================================

(deftest chain-rule-test
  (testing "f(x) = sin(x^2) => grad = cos(x^2) * 2x"
    (let [{:keys [gradient]} (eval-grad '(let* [a (* x x) r (Math/sin a)] r) '[x] [2.0])
          expected (* (Math/cos 4.0) 4.0)]
      (is (approx= expected (first gradient)))))
  (testing "f(x) = exp(sin(x)) => grad = exp(sin(x)) * cos(x)"
    (let [{:keys [gradient]}
          (eval-grad '(let* [a (Math/sin x) r (Math/exp a)] r) '[x] [1.0])
          expected (* (Math/exp (Math/sin 1.0)) (Math/cos 1.0))]
      (is (approx= expected (first gradient))))))

;; ================================================================
;; Multi-binding expressions
;; ================================================================

(deftest polynomial-gradient-test
  (testing "f(x) = 3x^2 + 2x => grad = 6x + 2"
    (let [{:keys [value gradient]}
          (eval-grad '(let* [x2 (* x x)
                             ax2 (* 3.0 x2)
                             bx (* 2.0 x)
                             result (+ ax2 bx)]
                            result)
                     '[x] [2.0])]
      (is (approx= 16.0 value))
      (is (approx= 14.0 (first gradient))))))

(deftest rosenbrock-gradient-test
  (testing "Rosenbrock at minimum (1,1): grad = [0, 0]"
    (let [{:keys [value gradient]}
          (eval-grad '(let* [a (- 1.0 x)
                             a2 (* a a)
                             x2 (* x x)
                             b (- y x2)
                             b2 (* b b)
                             c (* 100.0 b2)
                             result (+ a2 c)]
                            result)
                     '[x y] [1.0 1.0])]
      (is (approx= 0.0 value))
      (is (approx= 0.0 (first gradient) 1e-12))
      (is (approx= 0.0 (second gradient) 1e-12))))
  (testing "Rosenbrock at (2,3): grad = [802, -200]"
    (let [{:keys [gradient]}
          (eval-grad '(let* [a (- 1.0 x)
                             a2 (* a a)
                             x2 (* x x)
                             b (- y x2)
                             b2 (* b b)
                             c (* 100.0 b2)
                             result (+ a2 c)]
                            result)
                     '[x y] [2.0 3.0])]
      (is (approx= 802.0 (first gradient)))
      (is (approx= -200.0 (second gradient))))))

;; ================================================================
;; Bare expressions (no let wrapper)
;; ================================================================

(deftest bare-expression-test
  (testing "Bare symbol: f(x) = x"
    (let [{:keys [gradient]} (eval-grad 'x '[x] [5.0])]
      (is (approx= 1.0 (first gradient)))))
  (testing "Bare call: f(x,y) = (* x y)"
    (let [{:keys [gradient]} (eval-grad '(* x y) '[x y] [3.0 4.0])]
      (is (approx= 4.0 (first gradient)))
      (is (approx= 3.0 (second gradient))))))

;; ================================================================
;; Numerical gradient comparison
;; ================================================================

(deftest numerical-gradient-test
  (testing "numerical-gradient utility"
    (let [grad (rad/numerical-gradient
                (fn [x y] (+ (* x x) (* y y)))
                [3.0 4.0])]
      ;; f = x^2 + y^2, grad = [2x, 2y] = [6, 8]
      (is (approx= 6.0 (first grad) 1e-4))
      (is (approx= 8.0 (second grad) 1e-4)))))

(deftest ad-vs-numerical-test
  (testing "AD gradient matches numerical for complex expression"
    (let [{:keys [gradient]}
          (eval-grad '(let* [s (Math/sin x)
                             c (Math/cos y)
                             p (* s c)
                             e (Math/exp p)]
                            e)
                     '[x y] [1.0 2.0])
          num (rad/numerical-gradient
               (fn [x y] (Math/exp (* (Math/sin x) (Math/cos y))))
               [1.0 2.0])]
      (is (grads-match? gradient num)))))

;; ================================================================
;; Edge cases
;; ================================================================

(deftest constant-binding-test
  (testing "Binding with only constants — no active path"
    (let [{:keys [gradient]}
          (eval-grad '(let* [a (* 2.0 3.0)
                             r (+ a x)]
                            r)
                     '[x] [5.0])]
      ;; f(x) = 6 + x, grad = 1
      (is (approx= 1.0 (first gradient))))))

(deftest mixed-active-const-test
  (testing "Some params active, some effectively constant"
    (let [{:keys [gradient]}
          (eval-grad '(let* [a (* x 2.0)
                             b (* y 3.0)
                             r (+ a b)]
                            r)
                     '[x y] [1.0 2.0])]
      ;; f(x,y) = 2x + 3y, grad = [2, 3]
      (is (approx= 2.0 (first gradient)))
      (is (approx= 3.0 (second gradient))))))

;; ================================================================
;; If / branching
;; ================================================================

(deftest if-body-gradient-test
  (testing "if in let body — then branch (a > threshold)"
    (let [{:keys [value gradient]}
          (eval-grad '(let* [a (* x x)]
                            (if (> a 4) a (* 2.0 a)))
                     '[x] [3.0])]
      ;; x=3, a=9>4, f=a=9, df/dx=2x=6
      (is (approx= 9.0 value))
      (is (approx= 6.0 (first gradient)))))
  (testing "if in let body — else branch (a <= threshold)"
    (let [{:keys [value gradient]}
          (eval-grad '(let* [a (* x x)]
                            (if (> a 4) a (* 2.0 a)))
                     '[x] [1.0])]
      ;; x=1, a=1<=4, f=2a=2, df/dx=2*2x=4
      (is (approx= 2.0 value))
      (is (approx= 4.0 (first gradient))))))

(deftest relu-gradient-test
  (testing "ReLU: max(0, x) — positive"
    (let [{:keys [value gradient]}
          (eval-grad '(if (> x 0) x 0.0) '[x] [3.0])]
      (is (approx= 3.0 value))
      (is (approx= 1.0 (first gradient)))))
  (testing "ReLU: max(0, x) — negative"
    (let [{:keys [value gradient]}
          (eval-grad '(if (> x 0) x 0.0) '[x] [-2.0])]
      (is (approx= 0.0 value))
      (is (approx= 0.0 (first gradient))))))

(deftest abs-gradient-test
  (testing "abs(x) = if x>0 then x else -x"
    (let [{:keys [gradient]}
          (eval-grad '(if (> x 0) x (- 0.0 x)) '[x] [3.0])]
      (is (approx= 1.0 (first gradient))))
    (let [{:keys [gradient]}
          (eval-grad '(if (> x 0) x (- 0.0 x)) '[x] [-3.0])]
      (is (approx= -1.0 (first gradient))))))

(deftest piecewise-gradient-test
  (testing "Piecewise: f(x) = x^2 if x>0, else -x^2"
    (let [form '(let* [x2 (* x x)]
                      (if (> x 0) x2 (- 0.0 x2)))]
      ;; x=2: f=4, grad=4
      (let [{:keys [value gradient]} (eval-grad form '[x] [2.0])]
        (is (approx= 4.0 value))
        (is (approx= 4.0 (first gradient))))
      ;; x=-2: f=-4, grad=4 (since d(-x^2)/dx = -2x = -2*(-2) = 4)
      (let [{:keys [value gradient]} (eval-grad form '[x] [-2.0])]
        (is (approx= -4.0 value))
        (is (approx= 4.0 (first gradient)))))))

;; ================================================================
;; ANF normalization (nested sub-expressions)
;; ================================================================

(deftest anf-nested-call-test
  (testing "Nested calls: f(x) = sin(double(x)) via ANF lifting"
    (let [{:keys [gradient]}
          (eval-grad '(let* [r (Math/sin (double x))] r) '[x] [1.0])]
      (is (approx= (Math/cos 1.0) (first gradient)))))
  (testing "Double-nested: f(x) = exp(sin(double(x)))"
    (let [{:keys [gradient]}
          (eval-grad '(let* [r (Math/exp (Math/sin (double x)))] r)
                     '[x] [1.0])]
      (is (approx= (* (Math/exp (Math/sin 1.0)) (Math/cos 1.0))
                   (first gradient)))))
  (testing "Multiple nested args: f(x,y) = pow(double(x), double(y))"
    (let [{:keys [gradient]}
          (eval-grad '(let* [r (Math/pow (double x) (double y))] r)
                     '[x y] [2.0 3.0])]
      ;; d/dx x^y = y*x^(y-1) = 3*4 = 12
      (is (approx= 12.0 (first gradient)))
      ;; d/dy x^y = x^y*ln(x) = 8*ln(2)
      (is (approx= (* 8.0 (Math/log 2.0)) (second gradient))))))

;; ================================================================
;; Numeric cast rrules
;; ================================================================

(deftest double-cast-gradient-test
  (testing "double cast is identity for gradients"
    (let [{:keys [gradient]} (eval-grad '(double x) '[x] [3.0])]
      (is (approx= 1.0 (first gradient)))))
  (testing "Nested double cast in expression"
    (let [{:keys [gradient]}
          (eval-grad '(let* [a (double x) r (* a a)] r) '[x] [3.0])]
      (is (approx= 6.0 (first gradient))))))

;; ================================================================
;; Loop/recur support
;; ================================================================

(deftest loop-power-test
  (testing "f(x) = x^3 via loop => grad = 3x^2"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 acc 1.0]
                        (if (< i 3)
                          (recur (+ i 1) (* acc x))
                          acc))
                     '[x] [2.0])]
      (is (approx= 8.0 value))
      (is (approx= 12.0 (first gradient))))))

(deftest loop-sum-test
  (testing "f(x) = 3*x via loop accumulation => grad = 3"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 acc 0.0]
                        (if (< i 3)
                          (recur (+ i 1) (+ acc x))
                          acc))
                     '[x] [5.0])]
      (is (approx= 15.0 value))
      (is (approx= 3.0 (first gradient))))))

(deftest loop-polynomial-test
  (testing "f(x) = x + x^2 + x^3 via Horner-like loop"
    ;; (loop [i 0 acc 0.0] ... (recur (+ i 1) (+ (* acc x) x)) ...)
    ;; iter 0: acc=0, new_acc = 0*x + x = x
    ;; iter 1: acc=x, new_acc = x*x + x = x^2 + x
    ;; iter 2: acc=x^2+x, new_acc = (x^2+x)*x + x = x^3+x^2+x
    ;; f(x) = x^3 + x^2 + x, f'(x) = 3x^2 + 2x + 1
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 acc 0.0]
                        (if (< i 3)
                          (recur (+ i 1) (+ (* acc x) x))
                          acc))
                     '[x] [2.0])]
      (is (approx= 14.0 value))       ;; 8+4+2
      (is (approx= 17.0 (first gradient)))))) ;; 12+4+1

(deftest loop-two-params-test
  (testing "f(x,y) = x*y + x*y + x*y = 3*x*y via loop"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 acc 0.0]
                        (if (< i 3)
                          (recur (+ i 1) (+ acc (* x y)))
                          acc))
                     '[x y] [2.0 3.0])]
      (is (approx= 18.0 value))       ;; 3*2*3
      (is (approx= 9.0 (first gradient)))  ;; d/dx = 3*y = 9
      (is (approx= 6.0 (second gradient)))))) ;; d/dy = 3*x = 6

(deftest loop-with-let-test
  (testing "let bindings before loop: f(x) = 3*x^2"
    (let [{:keys [value gradient]}
          (eval-grad '(let* [x2 (* x x)]
                            (loop [i 0 acc 0.0]
                              (if (< i 3)
                                (recur (+ i 1) (+ acc x2))
                                acc)))
                     '[x] [2.0])]
      (is (approx= 12.0 value))
      (is (approx= 12.0 (first gradient))))))

(deftest loop-numerical-test
  (testing "Loop gradient matches numerical gradient"
    (let [{:keys [gradient]}
          (eval-grad '(loop [i 0 acc 0.0]
                        (if (< i 5)
                          (recur (+ i 1) (+ acc (* x x)))
                          acc))
                     '[x] [3.0])
          ;; f(x) = 5*x^2, f'(x) = 10*x = 30
          num (rad/numerical-gradient
               (fn [x] (* 5.0 (* x x)))
               [3.0])]
      (is (grads-match? gradient num)))))

;; ================================================================
;; Multi-var loop/recur support
;; ================================================================

(deftest multi-var-loop-independent-test
  (testing "Two independent accumulators: f(x) = 5*x^2 (via a), b unused"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 a 0.0 b 0.0]
                        (if (< i 5)
                          (recur (+ i 1) (+ a (* x x)) (+ b x))
                          a))
                     '[x] [3.0])]
      ;; f(x) = 5*x^2, f'(x) = 10x = 30
      (is (approx= 45.0 value))
      (is (approx= 30.0 (first gradient))))))

(deftest multi-var-loop-cross-dependent-test
  (testing "Cross-dependent vars: a=a*b, b=a+b, init a=x, b=x"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 a x b x]
                        (if (< i 3)
                          (recur (+ i 1) (* a b) (+ a b))
                          a))
                     '[x] [2.0])]
      ;; At x=2: iter0: a=4,b=4; iter1: a=16,b=8; iter2: a=128,b=24
      ;; f(x) = 2x^5 + 4x^4, f'(x) = 10x^4 + 16x^3
      ;; f(2) = 64+64=128, f'(2) = 160+128=288
      (is (approx= 128.0 value))
      (is (approx= 288.0 (first gradient))))))

(deftest multi-var-loop-numerical-test
  (testing "Multi-var loop gradient matches numerical gradient"
    (let [{:keys [gradient]}
          (eval-grad '(loop [i 0 a 0.0 b 0.0]
                        (if (< i 5)
                          (recur (+ i 1) (+ a (* x x)) (+ b x))
                          a))
                     '[x] [3.0])
          num (rad/numerical-gradient
               (fn [x] (* 5.0 (* x x)))
               [3.0])]
      (is (grads-match? gradient num)))))

(deftest multi-var-loop-cross-numerical-test
  (testing "Cross-dependent loop gradient matches numerical"
    (let [{:keys [gradient]}
          (eval-grad '(loop [i 0 a x b x]
                        (if (< i 3)
                          (recur (+ i 1) (* a b) (+ a b))
                          a))
                     '[x] [2.0])
          num (rad/numerical-gradient
               (fn [x]
                 (loop [i 0 a x b x]
                   (if (< i 3)
                     (recur (inc i) (* a b) (+ a b))
                     a)))
               [2.0])]
      (is (grads-match? gradient num)))))

(deftest multi-var-loop-two-params-test
  (testing "Multi-var loop with two free params"
    (let [{:keys [value gradient]}
          (eval-grad '(loop [i 0 a 0.0 b 0.0]
                        (if (< i 3)
                          (recur (+ i 1) (+ a (* x y)) (+ b x))
                          a))
                     '[x y] [2.0 3.0])]
      ;; f(x,y) = 3*x*y, df/dx = 3y = 9, df/dy = 3x = 6
      (is (approx= 18.0 value))
      (is (approx= 9.0 (first gradient)))
      (is (approx= 6.0 (second gradient))))))

;; ================================================================
;; Forward-over-reverse (Hessians / second derivatives)
;; ================================================================

(defn- make-grad-fn
  "Build a scalar function x -> f'(x) via reverse AD on a quoted form."
  [form]
  (let [transformed (rad/grad-expr form '[x])
        wrapper (list 'fn '[x]
                      (list 'let ['vpb transformed]
                            (list 'first (list (list 'nth 'vpb 1) 1.0))))]
    (eval wrapper)))

(deftest forward-over-reverse-polynomial-test
  (testing "f(x) = x^3, f''(x) = 6x"
    (let [grad-fn (make-grad-fn '(let* [x2 (* x x) r (* x x2)] r))]
      (is (approx= 12.0 (ad/derivative grad-fn 2.0)))
      (is (approx= 18.0 (ad/derivative grad-fn 3.0)))
      (is (approx= 0.0 (ad/derivative grad-fn 0.0)))))
  (testing "f(x) = x^4, f''(x) = 12x^2"
    (let [grad-fn (make-grad-fn '(let* [x2 (* x x) x4 (* x2 x2)] x4))]
      (is (approx= 48.0 (ad/derivative grad-fn 2.0)))
      (is (approx= 108.0 (ad/derivative grad-fn 3.0))))))

(deftest forward-over-reverse-transcendental-test
  (testing "f(x) = sin(x), f''(x) = -sin(x)"
    (let [grad-fn (make-grad-fn '(let* [r (Math/sin x)] r))]
      (is (approx= (- (Math/sin 1.0)) (ad/derivative grad-fn 1.0)))
      (is (approx= (- (Math/sin 0.5)) (ad/derivative grad-fn 0.5)))))
  (testing "f(x) = exp(x), f''(x) = exp(x)"
    (let [grad-fn (make-grad-fn '(let* [r (Math/exp x)] r))]
      (is (approx= (Math/exp 1.0) (ad/derivative grad-fn 1.0)))
      (is (approx= (Math/exp 2.0) (ad/derivative grad-fn 2.0)))))
  (testing "f(x) = log(x), f''(x) = -1/x^2"
    (let [grad-fn (make-grad-fn '(let* [r (Math/log x)] r))]
      (is (approx= -0.25 (ad/derivative grad-fn 2.0)))))
  (testing "f(x) = sqrt(x), f''(x) = -1/(4*x^(3/2))"
    (let [grad-fn (make-grad-fn '(let* [r (Math/sqrt x)] r))]
      (is (approx= (- (/ 1.0 (* 4.0 (Math/pow 4.0 1.5))))
                   (ad/derivative grad-fn 4.0))))))

(deftest forward-over-reverse-composition-test
  (testing "f(x) = sin(exp(x))"
    (let [grad-fn (make-grad-fn '(let* [e (Math/exp x) r (Math/sin e)] r))
          x 0.5
          ex (Math/exp x)
          expected (+ (- (* (Math/sin ex) ex ex)) (* (Math/cos ex) ex))]
      (is (approx= expected (ad/derivative grad-fn x)))))
  (testing "f(x) = cos(x^2)"
    (let [grad-fn (make-grad-fn '(let* [x2 (* x x) r (Math/cos x2)] r))
          x 1.0
          ;; f'(x) = -2x*sin(x^2), f''(x) = -2*sin(x^2) - 4x^2*cos(x^2)
          expected (- (- (* 2.0 (Math/sin (* x x))))
                      (* 4.0 x x (Math/cos (* x x))))]
      (is (approx= expected (ad/derivative grad-fn x))))))

;; ================================================================
;; Reverse-over-reverse: Hessian-vector product & full Hessian
;; ================================================================

(deftm ^:private hvp-quadratic [^double x ^double y] :- double
  (+ (* x x) (* x y)))

(deftm ^:private hvp-rosenbrock [^double x ^double y] :- double
  (+ (* (- 1.0 x) (- 1.0 x))
     (* 100.0 (* (- y (* x x)) (- y (* x x))))))

(deftm ^:private hvp-sin-sq [^double x] :- double
  (* (Math/sin x) (Math/sin x)))

(deftest compile-hvp-fn-test
  (testing "HVP of x^2+xy: Hessian = [[2,1],[1,0]]"
    (let [hvp-fn (rad/compile-hvp-fn #'hvp-quadratic)]
      (is (= [2.0 1.0] (hvp-fn [3.0 2.0] [1.0 0.0])))
      (is (= [1.0 0.0] (hvp-fn [3.0 2.0] [0.0 1.0])))))

  (testing "HVP is linear in v"
    (let [hvp-fn (rad/compile-hvp-fn #'hvp-quadratic)
          hv1 (hvp-fn [3.0 2.0] [1.0 0.0])
          hv2 (hvp-fn [3.0 2.0] [0.0 1.0])
          hv12 (hvp-fn [3.0 2.0] [1.0 1.0])]
      (is (= (mapv + hv1 hv2) hv12)))))

(deftest compile-hessian-fn-test
  (testing "Hessian of x^2+xy"
    (let [h-fn (rad/compile-hessian-fn #'hvp-quadratic)]
      (is (= [[2.0 1.0] [1.0 0.0]] (h-fn [3.0 2.0])))))

  (testing "Hessian of Rosenbrock at (1,1)"
    (let [h-fn (rad/compile-hessian-fn #'hvp-rosenbrock)]
      (is (= [[802.0 -400.0] [-400.0 200.0]] (h-fn [1.0 1.0])))))

  (testing "Hessian of Rosenbrock at (0,0)"
    (let [h-fn (rad/compile-hessian-fn #'hvp-rosenbrock)]
      (let [h (h-fn [0.0 0.0])]
        (is (= 2.0 (ffirst h)))
        (is (< (Math/abs (double (second (first h)))) 1e-10))
        (is (< (Math/abs (double (first (second h)))) 1e-10))
        (is (= 200.0 (second (second h)))))))

  (testing "Hessian of sin^2(x)"
    (let [h-fn (rad/compile-hessian-fn #'hvp-sin-sq)]
      (is (= [[2.0]] (h-fn [0.0])))
      (is (= [[-2.0]] (h-fn [(/ Math/PI 2)]))))))

;; ================================================================
;; Dotimes AD support
;; ================================================================

(deftest dotimes-parse-test
  (testing "parse-dotimes-form extracts components"
    (let [form '(dotimes [i 5] (aset out i (* (aget x i) 2.0)))
          parsed (#'rad/parse-dotimes-form form)]
      (is (= 'i (:counter-sym parsed)))
      (is (= 5 (:bound-expr parsed)))
      (is (seq? (:body parsed))))))

(deftest dotimes-analyze-body-test
  (testing "analyze-dotimes-body finds aset/aget operations"
    (let [body '(let* [v (aget x i)] (aset out i (* v 2.0)))
          analysis (#'rad/analyze-dotimes-body body 'i)]
      (is (= 1 (count (:asets analysis))))
      (is (= 'out (:arr (first (:asets analysis)))))
      (is (= 1 (count (:agets analysis))))
      (is (= 'x (:arr (first (:agets analysis)))))))
  (testing "bare aset without let* wrapper"
    (let [body '(aset out i (* (aget x i) 2.0))
          analysis (#'rad/analyze-dotimes-body body 'i)]
      (is (= 1 (count (:asets analysis))))
      (is (= 'out (:arr (first (:asets analysis))))))))

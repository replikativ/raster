(ns raster.sym.diff
  "Symbolic differentiation of S-expressions.

   Differentiates an S-expression w.r.t. a named variable using
   standard calculus rules (sum, product, quotient, chain).

  Output is a raw S-expression. Compose with raster.compiler.passes.scalar.simplify/simplify-derivative
   for cleanup.

   Usage:
     (diff '(* x x) 'x)            ;=> (+ (* 1 x) (* x 1))
     (differentiate '(* x x) 'x)   ;=> simplified form"
  (:require [raster.compiler.passes.scalar.simplify :as simplify]))

(defn depends-on?
  "True if expr contains var-sym as a free variable."
  [expr var-sym]
  (cond
    (= expr var-sym) true
    (symbol? expr) false
    (number? expr) false
    (nil? expr) false
    (boolean? expr) false
    (string? expr) false
    (keyword? expr) false
    (seq? expr) (boolean (some #(depends-on? % var-sym) (rest expr)))
    (vector? expr) (boolean (some #(depends-on? % var-sym) expr))
    :else false))

(declare diff)

(defn diff
  "Differentiate S-expression expr w.r.t. variable var-sym.
   Returns an unsimplified S-expression."
  [expr var-sym]
  (cond
    ;; The variable itself
    (= expr var-sym) 1

    ;; Other atom (symbol, number, nil, etc.)
    (not (seq? expr)) 0

    :else
    (let [[op & args] expr]
      (cond
        ;; --- Addition ---
        (contains? #{'+ 'raster.numeric/+ 'clojure.core/+} op)
        (let [[a b] args]
          (list '+ (diff a var-sym) (diff b var-sym)))

        ;; --- Subtraction ---
        (contains? #{'- 'raster.numeric/- 'clojure.core/-} op)
        (if (= 1 (count args))
          (list '- (diff (first args) var-sym))
          (let [[a b] args]
            (list '- (diff a var-sym) (diff b var-sym))))

        ;; --- Multiplication (product rule) ---
        (contains? #{'* 'raster.numeric/* 'clojure.core/*} op)
        (let [[a b] args]
          (list '+ (list '* (diff a var-sym) b)
                (list '* a (diff b var-sym))))

        ;; --- Division (quotient rule) ---
        (contains? #{'/ 'raster.numeric// 'clojure.core//} op)
        (let [[a b] args]
          (list '/ (list '- (list '* (diff a var-sym) b)
                         (list '* a (diff b var-sym)))
                (list '* b b)))

        ;; --- Power ---
        (contains? #{'pow 'raster.numeric/pow 'Math/pow} op)
        (let [[base exp-] args]
          (cond
            ;; Constant exponent: n * base^(n-1) * d(base)
            (not (depends-on? exp- var-sym))
            (list '* (list '* exp- (list 'pow base (list '- exp- 1)))
                  (diff base var-sym))
            ;; Constant base: base^exp * ln(base) * d(exp)
            (not (depends-on? base var-sym))
            (list '* (list '* expr (list 'log base))
                  (diff exp- var-sym))
            ;; General: f^g * (g'*ln(f) + g*f'/f)
            :else
            (list '* expr
                  (list '+ (list '* (diff exp- var-sym) (list 'log base))
                        (list '* exp- (list '/ (diff base var-sym) base))))))

        ;; --- Sqrt ---
        (contains? #{'sqrt 'raster.numeric/sqrt 'Math/sqrt} op)
        (let [[a] args]
          (list '* (list '/ 1 (list '* 2 (list 'sqrt a)))
                (diff a var-sym)))

        ;; --- Abs ---
        (contains? #{'abs 'raster.numeric/abs 'Math/abs} op)
        (let [[a] args]
          (list '* (list 'sign a) (diff a var-sym)))

        ;; --- Trig ---
        (contains? #{'sin 'raster.math/sin 'Math/sin} op)
        (let [[a] args]
          (list '* (list 'cos a) (diff a var-sym)))

        (contains? #{'cos 'raster.math/cos 'Math/cos} op)
        (let [[a] args]
          (list '* (list '- (list 'sin a)) (diff a var-sym)))

        (contains? #{'tan 'raster.math/tan 'Math/tan} op)
        (let [[a] args]
          (list '* (list '+ 1 (list '* (list 'tan a) (list 'tan a)))
                (diff a var-sym)))

        ;; --- Inverse trig ---
        (contains? #{'asin 'raster.math/asin 'Math/asin} op)
        (let [[a] args]
          (list '* (list '/ 1 (list 'sqrt (list '- 1 (list '* a a))))
                (diff a var-sym)))

        (contains? #{'acos 'raster.math/acos 'Math/acos} op)
        (let [[a] args]
          (list '* (list '- (list '/ 1 (list 'sqrt (list '- 1 (list '* a a)))))
                (diff a var-sym)))

        (contains? #{'atan 'raster.math/atan 'Math/atan} op)
        (let [[a] args]
          (list '* (list '/ 1 (list '+ 1 (list '* a a)))
                (diff a var-sym)))

        (contains? #{'atan2 'raster.math/atan2 'Math/atan2} op)
        (let [[y x] args]
          (list '/ (list '- (list '* x (diff y var-sym))
                         (list '* y (diff x var-sym)))
                (list '+ (list '* x x) (list '* y y))))

        ;; --- Hyperbolic ---
        (contains? #{'sinh 'raster.math/sinh} op)
        (let [[a] args]
          (list '* (list 'cosh a) (diff a var-sym)))

        (contains? #{'cosh 'raster.math/cosh} op)
        (let [[a] args]
          (list '* (list 'sinh a) (diff a var-sym)))

        (contains? #{'tanh 'raster.math/tanh} op)
        (let [[a] args]
          (list '* (list '- 1 (list '* (list 'tanh a) (list 'tanh a)))
                (diff a var-sym)))

        ;; --- Exp/Log ---
        (contains? #{'exp 'raster.math/exp 'Math/exp} op)
        (let [[a] args]
          (list '* (list 'exp a) (diff a var-sym)))

        (contains? #{'log 'raster.math/log 'Math/log} op)
        (let [[a] args]
          (list '* (list '/ 1 a) (diff a var-sym)))

        (contains? #{'log2 'raster.math/log2} op)
        (let [[a] args]
          (list '* (list '/ 1 (list '* a (list 'log 2))) (diff a var-sym)))

        (contains? #{'log10 'raster.math/log10 'Math/log10} op)
        (let [[a] args]
          (list '* (list '/ 1 (list '* a (list 'log 10))) (diff a var-sym)))

        ;; --- Cbrt ---
        (contains? #{'cbrt 'raster.math/cbrt 'Math/cbrt} op)
        (let [[a] args]
          (list '* (list '/ 1 (list '* 3 (list 'pow a (list '/ 2 3))))
                (diff a var-sym)))

        ;; --- Hypot ---
        (contains? #{'hypot 'raster.math/hypot 'Math/hypot} op)
        (let [[a b] args
              h (list 'hypot a b)]
          (list '+ (list '* (list '/ a h) (diff a var-sym))
                (list '* (list '/ b h) (diff b var-sym))))

        ;; --- Min/Max (subgradient) ---
        (contains? #{'min 'raster.numeric/min} op)
        (let [[a b] args]
          (list 'if (list '< a b) (diff a var-sym) (diff b var-sym)))

        (contains? #{'max 'raster.numeric/max} op)
        (let [[a b] args]
          (list 'if (list '> a b) (diff a var-sym) (diff b var-sym)))

        ;; --- Sign (derivative is 0 a.e.) ---
        (contains? #{'sign 'raster.numeric/sign} op)
        0

        ;; --- Conditional ---
        (= 'if op)
        (let [[test then else] args]
          (list 'if test (diff then var-sym) (when else (diff else var-sym))))

        ;; --- FMA: fma(a,b,c) = a*b + c ---
        (contains? #{'fma 'raster.math/fma 'Math/fma} op)
        (let [[a b c] args]
          (list '+ (list '+ (list '* (diff a var-sym) b)
                         (list '* a (diff b var-sym)))
                (diff c var-sym)))

        ;; --- let* ---
        (contains? #{'let 'let* 'clojure.core/let} op)
        (let [[bindings & body] args
              pairs (partition 2 bindings)
              diff-pairs (mapcat (fn [[sym init]]
                                   [sym init
                                    (symbol (str "d_" (name sym))) (diff init var-sym)])
                                 pairs)
              ;; For the body, we need to apply chain rule through intermediate vars.
              ;; The simple approach: treat body as depending on both original and d_ vars.
              diff-body (diff (last body) var-sym)]
          (list 'let* (vec diff-pairs) diff-body))

        ;; --- Default: unknown function ---
        :else
        (throw (ex-info (str "sym-diff: unknown operation " op)
                        {:op op :expr expr}))))))

(defn differentiate
  "Differentiate and simplify a symbolic expression.
   Uses raster.compiler.passes.scalar.simplify/simplify-derivative for cleanup."
  ([expr var-sym]
   (differentiate expr var-sym 5))
  ([expr var-sym max-rounds]
   (simplify/simplify-derivative (diff expr var-sym) max-rounds)))

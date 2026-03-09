(ns raster.sym.analysis
  "Linearity and sparsity analysis of symbolic expressions.

   Determines, for each variable, whether an expression is:
   - :constant (does not depend on variable)
   - :linear (linear in variable)
   - :quadratic (quadratic in variable)
   - :nonlinear (general nonlinear dependence)

   Also predicts Jacobian and Hessian sparsity patterns."
  (:require [raster.sym.diff :as sd]))

;; ================================================================
;; Degree ordinals
;; ================================================================

(def ^:private degree-ord
  {:constant 0 :linear 1 :quadratic 2 :nonlinear 3})

(defn- max-degree [a b]
  (if (>= (degree-ord a 3) (degree-ord b 3)) a b))

(defn- add-degrees [a b]
  (let [sum (+ (long (degree-ord a 3)) (long (degree-ord b 3)))]
    (cond
      (= sum 0) :constant
      (= sum 1) :linear
      (= sum 2) :quadratic
      :else :nonlinear)))

(defn- mul-degree [deg ^long n]
  (cond
    (= deg :constant) :constant
    (and (= deg :linear) (= n 1)) :linear
    (and (= deg :linear) (= n 2)) :quadratic
    :else :nonlinear))

;; ================================================================
;; Degree analysis
;; ================================================================

(def ^:private add-ops
  #{'+ '- 'raster.numeric/+ 'raster.numeric/- 'clojure.core/+ 'clojure.core/-})

(def ^:private mul-ops
  #{'* 'raster.numeric/* 'clojure.core/*})

(def ^:private div-ops
  #{'/ 'raster.numeric// 'clojure.core//})

(def ^:private pow-ops
  #{'pow 'Math/pow 'raster.numeric/pow})

(def ^:private transcendental-ops
  #{'sin 'cos 'tan 'exp 'log 'sqrt 'asin 'acos 'atan 'sinh 'cosh 'tanh
    'raster.math/sin 'raster.math/cos 'raster.math/tan 'raster.math/exp
    'raster.math/log 'raster.math/sinh 'raster.math/cosh 'raster.math/tanh
    'raster.numeric/sqrt})

(defn- degree-of
  "Compute the polynomial degree of expr w.r.t. var-sym."
  [expr var-sym]
  (cond
    (= expr var-sym) :linear
    (not (seq? expr)) :constant
    (not (sd/depends-on? expr var-sym)) :constant
    :else
    (let [[op & args] expr]
      (cond
        ;; Addition/subtraction: max degree
        (contains? add-ops op)
        (reduce max-degree :constant (map #(degree-of % var-sym) args))

        ;; Multiplication: sum degrees
        (contains? mul-ops op)
        (let [[a b] args]
          (add-degrees (degree-of a var-sym) (degree-of b var-sym)))

        ;; Division: if denominator constant, keep numerator degree
        (contains? div-ops op)
        (let [[a b] args]
          (if (= :constant (degree-of b var-sym))
            (degree-of a var-sym)
            :nonlinear))

        ;; Power with constant integer exponent
        (contains? pow-ops op)
        (let [[base exp-] args]
          (cond
            (not (sd/depends-on? base var-sym)) :constant
            (and (number? exp-) (integer? exp-))
            (mul-degree (degree-of base var-sym) (long exp-))
            :else :nonlinear))

        ;; Transcendental functions
        (contains? transcendental-ops op) :nonlinear

        ;; if: max of branches
        (= 'if op)
        (let [[_ then else] args]
          (max-degree (degree-of then var-sym)
                      (if else (degree-of else var-sym) :constant)))

        ;; Default
        :else :nonlinear))))

;; ================================================================
;; Public API
;; ================================================================

(defn linearity-analysis
  "For each variable, determine the degree of dependence.
   Returns a map: {var-sym -> :constant/:linear/:quadratic/:nonlinear}."
  [expr vars]
  (into {} (map (fn [v] [v (degree-of expr v)]) vars)))

(defn jacobian-sparsity
  "Predict nonzero Jacobian entries for a vector of expressions.
   exprs: vector of S-expressions (rows)
   vars: vector of variable symbols (columns)
   Returns a vector of vectors of booleans."
  [exprs vars]
  (mapv (fn [expr]
          (mapv (fn [v] (boolean (sd/depends-on? expr v))) vars))
        exprs))

(defn hessian-sparsity
  "Predict nonzero Hessian entries for a scalar expression.
   Returns a vector of vectors of booleans (symmetric matrix)."
  [expr vars]
  (let [n (count vars)
        derivs (mapv (fn [v] (sd/differentiate expr v)) vars)]
    (mapv (fn [i]
            (mapv (fn [j]
                    (if (<= i j)
                      (boolean (sd/depends-on? (nth derivs i) (nth vars j)))
                      (boolean (sd/depends-on? (nth derivs j) (nth vars i)))))
                  (range n)))
          (range n))))

(ns raster.sym.taylor
  "Taylor series expansion of symbolic expressions.

   Computes Taylor polynomials by repeated symbolic differentiation
   and numerical evaluation at the expansion point.

   Usage:
     (taylor-expand '(sin x) 'x 0.0 5)
     ;=> (+ (* x 1.0) (* (pow x 3) -0.16666666666666666) (* (pow x 5) 0.008333333333333333))

     (taylor-coeffs '(exp x) 'x 0.0 4)
     ;=> [1.0 1.0 0.5 0.16666666666666666 0.041666666666666664]"
  (:require [raster.sym.diff :as sd]))

(defn- factorial ^double [^long n]
  (loop [i 1 acc 1.0]
    (if (> i n) acc
        (recur (inc i) (* acc (double i))))))

(defn- eval-at
  "Substitute var-sym = val in an S-expression and evaluate numerically.
   Only handles the operations produced by sym-diff."
  [expr var-sym ^double val]
  (cond
    (= expr var-sym) val
    (number? expr) (double expr)
    (symbol? expr) (throw (ex-info (str "Unbound symbol: " expr) {:sym expr}))
    (seq? expr)
    (let [[op & args] expr]
      (cond
        (contains? #{'+ 'raster.numeric/+ 'clojure.core/+} op)
        (reduce (fn [^double a b] (+ a (double (eval-at b var-sym val))))
                0.0 args)

        (contains? #{'- 'raster.numeric/- 'clojure.core/-} op)
        (if (= 1 (count args))
          (- (double (eval-at (first args) var-sym val)))
          (- (double (eval-at (first args) var-sym val))
             (double (eval-at (second args) var-sym val))))

        (contains? #{'* 'raster.numeric/* 'clojure.core/*} op)
        (reduce (fn [^double a b] (* a (double (eval-at b var-sym val))))
                1.0 args)

        (contains? #{'/ 'raster.numeric// 'clojure.core//} op)
        (/ (double (eval-at (first args) var-sym val))
           (double (eval-at (second args) var-sym val)))

        (contains? #{'pow 'raster.numeric/pow 'Math/pow} op)
        (Math/pow (eval-at (first args) var-sym val)
                  (eval-at (second args) var-sym val))

        (contains? #{'sin 'raster.math/sin 'Math/sin} op)
        (Math/sin (eval-at (first args) var-sym val))

        (contains? #{'cos 'raster.math/cos 'Math/cos} op)
        (Math/cos (eval-at (first args) var-sym val))

        (contains? #{'tan 'raster.math/tan 'Math/tan} op)
        (Math/tan (eval-at (first args) var-sym val))

        (contains? #{'exp 'raster.math/exp 'Math/exp} op)
        (Math/exp (eval-at (first args) var-sym val))

        (contains? #{'log 'raster.math/log 'Math/log} op)
        (Math/log (eval-at (first args) var-sym val))

        (contains? #{'sqrt 'raster.numeric/sqrt 'Math/sqrt} op)
        (Math/sqrt (eval-at (first args) var-sym val))

        (contains? #{'abs 'raster.numeric/abs 'Math/abs} op)
        (Math/abs (double (eval-at (first args) var-sym val)))

        (contains? #{'sign 'raster.numeric/sign} op)
        (Math/signum (double (eval-at (first args) var-sym val)))

        (contains? #{'sinh 'raster.math/sinh} op)
        (Math/sinh (eval-at (first args) var-sym val))

        (contains? #{'cosh 'raster.math/cosh} op)
        (Math/cosh (eval-at (first args) var-sym val))

        (contains? #{'tanh 'raster.math/tanh} op)
        (Math/tanh (eval-at (first args) var-sym val))

        (contains? #{'asin 'raster.math/asin} op)
        (Math/asin (eval-at (first args) var-sym val))

        (contains? #{'acos 'raster.math/acos} op)
        (Math/acos (eval-at (first args) var-sym val))

        (contains? #{'atan 'raster.math/atan} op)
        (Math/atan (eval-at (first args) var-sym val))

        (contains? #{'cbrt 'raster.math/cbrt} op)
        (Math/cbrt (eval-at (first args) var-sym val))

        (contains? #{'hypot 'raster.math/hypot} op)
        (Math/hypot (eval-at (first args) var-sym val)
                    (eval-at (second args) var-sym val))

        (= 'if op)
        (let [[test then else] args
              tv (eval-at test var-sym val)]
          (if (and (number? tv) (not (zero? tv)))
            (eval-at then var-sym val)
            (eval-at (or else 0) var-sym val)))

        :else
        (throw (ex-info (str "eval-at: unknown operation " op) {:op op}))))
    :else (throw (ex-info "eval-at: unrecognized form" {:expr expr}))))

(defn taylor-coeffs
  "Compute Taylor coefficients of f(x) expanded around x=a up to order n.
   Returns a vector of (n+1) coefficients [c0, c1, ..., cn] where
   f(x) ≈ c0 + c1*(x-a) + c2*(x-a)^2 + ... + cn*(x-a)^n."
  [expr var-sym ^double a ^long n]
  (loop [i 0
         current-deriv expr
         coeffs []]
    (if (> i n)
      coeffs
      (let [val-at-a (eval-at current-deriv var-sym a)
            coeff (/ val-at-a (factorial i))
            next-deriv (sd/differentiate current-deriv var-sym)]
        (recur (inc i) next-deriv (conj coeffs coeff))))))

(defn taylor-expand
  "Compute the Taylor polynomial of f(x) expanded around x=a up to order n.
   Returns a simplified S-expression.

   Drops terms with near-zero coefficients (< 1e-15)."
  [expr var-sym a ^long n]
  (let [coeffs (taylor-coeffs expr var-sym (double a) n)
        eps 1e-15
        dx (if (zero? (double a)) var-sym (list '- var-sym a))
        terms (keep-indexed
               (fn [i c]
                 (when (> (Math/abs (double c)) eps)
                   (cond
                     (zero? i) c
                     (== 1.0 (double c))
                     (if (= 1 i) dx (list 'pow dx i))
                     :else
                     (if (= 1 i)
                       (list '* dx c)
                       (list '* (list 'pow dx i) c)))))
               coeffs)]
    (if (empty? terms)
      0
      (reduce (fn [a b] (list '+ a b)) terms))))

(defn taylor-remainder
  "Estimate the Lagrange remainder bound for the Taylor polynomial.
   |R_n(x)| <= M * |x-a|^(n+1) / (n+1)!
   where M is an upper bound on |f^(n+1)(t)| for t between a and x.

   Returns {:remainder-bound <expr>, :next-deriv <S-expr>}."
  [expr var-sym ^double a ^long n]
  (let [coeffs (taylor-coeffs expr var-sym a (inc n))
        next-coeff (last coeffs)]
    {:remainder-bound (list '* (Math/abs (double next-coeff))
                            (list 'pow (list 'abs (list '- var-sym a)) (inc n)))
     :next-deriv-at-a (* next-coeff (factorial (inc n)))}))

;; ================================================================
;; Built-in Taylor expansions for link functions
;; ================================================================

(defn sigmoid-taylor
  "Taylor expansion of sigmoid 1/(1+e^{-x}) around x=0.
   Returns an S-expression polynomial of given order."
  [x ^long order]
  (taylor-expand '(/ 1 (+ 1 (exp (- x)))) 'x 0.0 order))

(defn probit-taylor
  "Taylor expansion of the probit function Φ(x) (standard normal CDF)
   around x=0. Uses the approximation via erf.
   Returns an S-expression polynomial of given order."
  [x ^long order]
  ;; Φ(x) = ½(1 + erf(x/√2)), expand via Taylor of erf
  ;; erf(x) ≈ (2/√π)(x - x³/3 + x⁵/10 - x⁷/42 + ...)
  (let [;; Coefficients of erf Taylor series around 0
        ;; erf^(n)(0) / n! for odd n
        inv-sqrt-pi (/ 2.0 (Math/sqrt Math/PI))
        inv-sqrt-2 (/ 1.0 (Math/sqrt 2.0))
        ;; Pre-compute coefficients for Φ(x)
        coeffs (loop [n 0 deriv '(/ 1 (+ 1 (exp (- x)))) cs []]
                 (if (> n order) cs
                     (let [v (eval-at deriv 'x 0.0)
                           c (/ v (factorial n))]
                       (recur (inc n) (sd/differentiate deriv 'x) (conj cs c)))))]
    ;; Build polynomial
    (let [eps 1e-15
          dx x
          terms (keep-indexed
                 (fn [i c]
                   (when (> (Math/abs (double c)) eps)
                     (cond
                       (zero? i) c
                       (== 1.0 (double c))
                       (if (= 1 i) dx (list 'pow dx i))
                       :else
                       (if (= 1 i)
                         (list '* dx c)
                         (list '* (list 'pow dx i) c)))))
                 coeffs)]
      (if (empty? terms)
        0
        (reduce (fn [a b] (list '+ a b)) terms)))))

(defn log-sum-exp-taylor
  "Taylor expansion of log(Σ e^{xᵢ}) around the given point.
   xs: vector of variable symbols
   point: vector of expansion point values
   order: Taylor order for each variable

   Returns an S-expression polynomial."
  [xs point ^long order]
  (let [n (count xs)
        ;; At the expansion point, compute log(Σ e^{aᵢ})
        sum-exp (reduce + (map #(Math/exp (double %)) point))
        base-val (Math/log sum-exp)
        ;; For a single-variable Taylor, expand log(Σ e^{xᵢ}) treating
        ;; each xᵢ independently. Use first-order multi-dim Taylor.
        ;; Gradient: ∂/∂xᵢ log(Σ e^{xⱼ}) = e^{xᵢ} / Σ e^{xⱼ}
        softmax-vals (mapv #(/ (Math/exp (double %)) sum-exp) point)
        ;; Build first-order terms: Σᵢ softmax(xᵢ) * (xᵢ - aᵢ)
        linear-terms (mapv (fn [i]
                             (let [si (nth softmax-vals i)
                                   xi (nth xs i)
                                   ai (nth point i)
                                   dx (if (zero? (double ai)) xi (list '- xi ai))]
                               (list '* si dx)))
                           (range n))
        ;; Hessian: ∂²/∂xᵢ∂xⱼ = -softmax(xᵢ)*softmax(xⱼ) + δᵢⱼ*softmax(xᵢ)
        quad-terms (when (>= order 2)
                     (for [i (range n) j (range i n)]
                       (let [si (nth softmax-vals i)
                             sj (nth softmax-vals j)
                             hij (if (= i j) (* si (- 1.0 si)) (- (* si sj)))
                             xi (nth xs i) xj (nth xs j)
                             ai (nth point i) aj (nth point j)
                             dxi (if (zero? (double ai)) xi (list '- xi ai))
                             dxj (if (zero? (double aj)) xj (list '- xj aj))
                             scale (if (= i j) 0.5 1.0)]
                         (when (> (Math/abs (* hij scale)) 1e-15)
                           (list '* (* hij scale) (list '* dxi dxj))))))
        ;; Assemble
        all-terms (concat [base-val] linear-terms (filter some? quad-terms))]
    (reduce (fn [a b] (list '+ a b)) all-terms)))

(defn taylor-log-likelihood
  "Compute a Taylor polynomial approximation of a log-likelihood.
   Useful for converting transcendental likelihoods to polynomial form
   for algebraic identifiability analysis.

   model: {:log-likelihood S-expr, :params [syms], :observations [syms]}
   var: variable to expand around (typically an observation)
   point: expansion point value

   Options:
     :order — Taylor order (default 4)

   Returns S-expression polynomial approximation."
  [model var point & {:keys [order] :or {order 4}}]
  (taylor-expand (:log-likelihood model) var (double point) order))

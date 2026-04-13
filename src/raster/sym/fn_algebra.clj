(ns raster.sym.fn-algebra
  "Function algebra and Operator type for composable mathematics.

  Provides:
  - Arithmetic on typed functions: (+ f g), (- f g), (* c f), (* f g)
  - Operator type for function-to-function maps (D, composition)
  - D operator for forward-mode differentiation
  - D-k operator for k-th order derivatives via Jet
  - D-reverse operator for reverse-mode differentiation

  All results implement TypedFn, so they compose freely:
    (D (+ f g)) = (+ (D f) (D g))"
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.ad.forward :as ad]
            [raster.ad.reverse :as rev]
            [raster.ad.jet :as jet]
            [raster.compiler.core.types :as types]))

(import 'raster.compiler.core.types.TypedFn)

;; ================================================================
;; Helper: wrap a plain fn as TypedFn + IFn
;; ================================================================

(defn typed-fn
  "Wrap a plain Clojure function as a TypedFn-implementing object.
  The result implements IFn__double for typed dispatch, IObj for metadata,
  and participates in function algebra dispatch."
  ([f] (typed-fn f nil))
  ([f m]
   (let [meta-atom (atom m)]
     (proxy [clojure.lang.AFn raster.fn.IFn__double TypedFn clojure.lang.IObj] []
       (invk [a] (f a))
       (invoke
         ([a] (f a))
         ([a b] (f a b))
         ([a b c] (f a b c))
         ([a b c d] (f a b c d)))
       (applyTo [args] (clojure.lang.AFn/applyToHelper f args))
       (meta [] @meta-atom)
       (withMeta [new-meta] (typed-fn f new-meta))))))

;; ================================================================
;; Operator type: function → function maps
;; ================================================================

(defrecord Operator [f name arity]
  TypedFn
  clojure.lang.IFn
  (invoke [_ x] (f x))
  (invoke [_ x y] (f x y))
  (invoke [_ x y z] (f x y z))
  (applyTo [_ args] (clojure.lang.AFn/applyToHelper f args)))

;; ================================================================
;; Function addition: (+ f g) => fn(x) = f(x) + g(x)
;; Typed overloads return ftm (pipeline-compilable).
;; No TypedFn fallback — fail loudly if types don't match.
;; ================================================================

;; R→R — ^:inline enables pipeline beta reduction even for multi-use
(deftm raster.numeric/+ [f :- (Fn [Double] Double), g :- (Fn [Double] Double)]
  ^:inline (ftm [a :- Double] :- Double (n/+ (f a) (g a))))

;; R²→R
(deftm raster.numeric/+ [f :- (Fn [Double Double] Double), g :- (Fn [Double Double] Double)]
  ^:inline (ftm [a :- Double, b :- Double] :- Double (n/+ (f a b) (g a b))))

;; Operator + Operator: (+ A B)(f) = (A f) + (B f)
(deftm raster.numeric/+ [a :- Operator, b :- Operator] :- Operator
  (->Operator (fn [f] (n/+ ((:f a) f) ((:f b) f)))
              (symbol (str "(" (:name a) "+" (:name b) ")"))
              (:arity a)))

;; ================================================================
;; Function subtraction: (- f g) => fn(x) = f(x) - g(x)
;; ================================================================

;; R→R
(deftm raster.numeric/- [f :- (Fn [Double] Double), g :- (Fn [Double] Double)]
  ^:inline (ftm [a :- Double] :- Double (n/- (f a) (g a))))

(deftm raster.numeric/- [f :- (Fn [Double] Double)]
  ^:inline (ftm [a :- Double] :- Double (n/- (f a))))

;; ================================================================
;; Function multiplication
;; ================================================================

;; Scalar * function (R→R)
(deftm raster.numeric/* [c :- Double, f :- (Fn [Double] Double)]
  ^:inline (ftm [a :- Double] :- Double (n/* c (f a))))

(deftm raster.numeric/* [f :- (Fn [Double] Double), c :- Double]
  ^:inline (ftm [a :- Double] :- Double (n/* (f a) c)))

;; Pointwise multiplication (R→R)
(deftm raster.numeric/* [f :- (Fn [Double] Double), g :- (Fn [Double] Double)]
  ^:inline (ftm [a :- Double] :- Double (n/* (f a) (g a))))

;; Derivative order detection for automatic D^k fusion
(defn- derivative-order
  "Return the derivative order of an operator if it's a D or D-k operator, nil otherwise.
   D → 1, D2 → 2, D3 → 3, etc."
  [^Operator op]
  (let [n (str (:name op))]
    (cond
      (= n "D") 1
      (and (.startsWith n "D") (every? #(Character/isDigit ^char %) (subs n 1)))
      (Long/parseLong (subs n 1))
      :else nil)))

;; Operator composition: (A * B)(x) = A(B(x))
;; Detects D^k chains and upgrades to Jet-based D-k for efficiency
;; (avoids perturbation confusion from nested Dual-over-Dual)
(deftm raster.numeric/* [a :- Operator, b :- Operator] :- Operator
  (let [a-order (derivative-order a)
        b-order (derivative-order b)]
    (if (and a-order b-order)
      ;; Both are derivative operators — use Jet for combined order
      (let [k (long (+ (long a-order) (long b-order)))]
        (->Operator
         (fn [f]
           (typed-fn
            (fn [x]
              (jet/jet-derivative f (double x) k))))
         (symbol (str "D" k)) 1))
      ;; General operator composition
      (->Operator (comp (:f a) (:f b))
                  (symbol (str (:name a) "∘" (:name b)))
                  (:arity b)))))

;; Operator applied to function: (* D f) = (D f)
(deftm raster.numeric/* [op :- Operator, f :- TypedFn]
  ((:f op) f))

;; Scalar * Operator
(deftm raster.numeric/* [c :- Double, op :- Operator] :- Operator
  (->Operator (fn [f] (n/* c ((:f op) f)))
              (symbol (str c "*" (:name op)))
              (:arity op)))

;; ================================================================
;; D operator: derivative (pipeline-compatible via value+grad)
;; ================================================================

(defn- deftm-var?
  "True if f is a var with deftm metadata (can use value+grad)."
  [f]
  (and (var? f)
       (or (:raster.core/deftm (meta f))
           (:raster.core/dispatch-table (meta f)))))

(defn- make-derivative-fn
  "Create a derivative function for f. Uses value+grad for deftm vars
   (pipeline-compatible), falls back to forward-mode for plain fns.

   For D∘D composition: tagged derivative functions are detected and
   higher orders are computed via Jets (avoids perturbation confusion
   from nested Dual numbers and devirtualized .invk incompatibility)."
  [f]
  (cond
    ;; Nested D: f is already a derivative — upgrade to Jet-based higher-order.
    ;; Uses the original function (not walked body) so Jets propagate
    ;; through raster.numeric dispatch.
    (:derivative-of (meta f))
    (let [{:keys [derivative-of order unwound-fn]} (meta f)
          k (inc order)
          base-fn (or unwound-fn derivative-of)]
      (with-meta
        (typed-fn (fn [x] (jet/jet-derivative base-fn (double x) k)))
        {:derivative-of derivative-of :order k :unwound-fn base-fn}))

    ;; deftm var: use value+grad (pipeline-compatible)
    (deftm-var? f)
    (let [vg (rev/value+grad f)
          ;; Build an un-walked fn from the source body for Jet/Dual
          ;; propagation in D∘D. The source body uses raster.numeric
          ;; ops (generic dispatch) which accept Jets/Duals.
          ;; Resolve to the mangled deftm var to get source body
          resolved (or (when (:raster.core/deftm-source-body (meta f)) f)
                       (when-let [dt (:raster.core/dispatch-table (meta f))]
                         (let [entry (first (mapcat val @dt))
                               mn (symbol (str (types/mangle (:name (meta f))
                                                             (:tags entry))))]
                           (ns-resolve (:ns (meta f)) mn))))
          src-body (:raster.core/deftm-source-body (meta resolved))
          params (:raster.core/deftm-params (meta resolved))
          src-ns (or (:ns (meta resolved)) *ns*)
          unwound (when (and src-body params)
                    (binding [*ns* src-ns]
                      (eval (list 'fn (vec params) (first src-body)))))]
      (with-meta
        (typed-fn (fn [x] (nth (vg (double x)) 1)))
        {:derivative-of f :order 1 :unwound-fn (or unwound f)}))

    ;; Plain IFn: forward-mode Dual numbers.
    ;; The function's invoke(Object) path uses generic raster.numeric dispatch
    ;; (not devirtualized .invk), so Dual propagates correctly through AD.
    :else
    (let [f-for-ad f]
      (with-meta
        (typed-fn (fn [x] (ad/derivative f-for-ad (double x))))
        {:derivative-of f :order 1 :unwound-fn f}))))

(def D
  "Derivative operator. (D f) returns the derivative of f as a TypedFn.
  For deftm vars, uses reverse-mode value+grad (pipeline-compatible).
  For plain functions, falls back to forward-mode Dual numbers."
  (->Operator make-derivative-fn 'D 1))

;; ================================================================
;; D-k operator: k-th order derivative via Jet
;; ================================================================

(defn D-k
  "k-th derivative operator via Jet propagation.
   (D-k 1) ≡ D, but (D-k 4) computes the 4th derivative
   in a single pass (O(k²)) rather than k nested passes (O(2^k))."
  [^long k]
  (->Operator
   (fn [f]
     (typed-fn
      (fn [x]
        (jet/jet-derivative f (double x) k))))
   (symbol (str "D" k)) 1))

;; ================================================================
;; D-reverse operator: reverse-mode derivative
;; ================================================================

(def D-reverse
  "Derivative operator using reverse-mode AD.
   For deftm vars, uses value+grad (pipeline-compatible).
   For plain functions, falls back to forward-mode (equivalent for R→R)."
  (->Operator make-derivative-fn 'Dr 1))

;; ================================================================
;; Partial derivative operator: (∂ i) for mechanics workflows
;; ================================================================

(defn partial-d
  "Partial derivative operator. ((∂ i) f) returns ∂f/∂x_i as a TypedFn.
  For deftm vars, uses value+grad (pipeline-compatible).
  For plain functions, uses forward-mode Dual numbers.

  Usage:
    ((∂ 0) L)          ;; ∂L/∂q
    ((∂ 1) L)          ;; ∂L/∂qdot
    (((∂ 0) L) 1.0 0.0) ;; evaluate at q=1, qdot=0"
  [^long i]
  (->Operator
   (fn [f]
     (if (deftm-var? f)
       ;; value+grad path: pipeline-compatible
       (let [vg (rev/value+grad f)]
         (typed-fn
          (fn
            ([a] (nth (vg (double a)) (inc i)))
            ([a b] (nth (vg (double a) (double b)) (inc i)))
            ([a b c] (nth (vg (double a) (double b) (double c)) (inc i))))))
       ;; Forward-mode fallback for plain IFn
       (typed-fn
        (fn
          ([a]
           (let [d (ad/dual (double a) (if (== i 0) 1.0 0.0))
                 result (f d)]
             (n/derivative-value result)))
          ([a b]
           (let [da (ad/dual (double a) (if (== i 0) 1.0 0.0))
                 db (ad/dual (double b) (if (== i 1) 1.0 0.0))
                 result (f da db)]
             (n/derivative-value result)))
          ([a b c]
           (let [da (ad/dual (double a) (if (== i 0) 1.0 0.0))
                 db (ad/dual (double b) (if (== i 1) 1.0 0.0))
                 dc (ad/dual (double c) (if (== i 2) 1.0 0.0))
                 result (f da db dc)]
             (n/derivative-value result)))))))
   (symbol (str "∂_" i)) :variadic))

(def ∂
  "Alias for partial-d. Partial derivative operator.
  ((∂ 1) L) extracts ∂L/∂qdot from Lagrangian L."
  partial-d)

(ns raster.sym.core
  "Symbolic expression type for raster.

   (defabstract AbstractSymbol) — base type for all symbolic types
   (defvalue Sym [form] :implements AbstractSymbol) — wraps an S-expression.

   All raster.numeric and raster.math operations are extended to handle
   AbstractSymbol, building expression trees automatically via deftm dispatch.

   Usage:
     (require '[raster.sym.core :refer [->Sym sym]])
     (let [x (sym 'x)]
       (raster.numeric/+ x 2))
     ;=> Sym((+ x 2))"
  (:refer-clojure :exclude [+ - * / abs < <= > >= ==])
  (:require [raster.core :refer [deftm defvalue defval defabstract]]
            [raster.numeric]
            [raster.math]
            [raster.types.promote :as promote]))

;; ================================================================
;; Abstract base type for all symbolic types
;; ================================================================

(defabstract AbstractSymbol)

;; ================================================================
;; Sym value type
;; ================================================================

(defvalue Sym [form] :implements AbstractSymbol)

(defn sym
  "Create a symbolic variable. If given a symbol, wraps it.
   If given an S-expression, wraps it directly."
  [x]
  (->Sym x))

(defn sym?
  "True if x is an AbstractSymbol."
  [x]
  (instance? AbstractSymbol x))

(defn unwrap
  "Extract the S-expression from a Sym. Returns x unchanged if not a Sym."
  [x]
  (if (instance? Sym x) (.form ^Sym x) x))

;; ================================================================
;; Promotion: AbstractSymbol + Number → Sym
;; ================================================================

(defval TSym)
(promote/register-type-tag! Sym (->TSym))

(promote/register-promote-rule! Sym Long Sym)
(promote/register-promote-rule! Sym Double Sym)
(promote/register-promote-rule! Sym Float Sym)
(promote/register-promote-rule! Sym Integer Sym)
(promote/register-promote-rule! Long Sym Sym)
(promote/register-promote-rule! Double Sym Sym)
(promote/register-promote-rule! Float Sym Sym)
(promote/register-promote-rule! Integer Sym Sym)

;; convert: wrap non-Sym values as Sym containing the literal
(deftm raster.types.promote/convert [t :- TSym, x :- Sym] :- Sym x)
(deftm raster.types.promote/convert [t :- TSym, x :- Long] :- Sym (->Sym x))
(deftm raster.types.promote/convert [t :- TSym, x :- Double] :- Sym (->Sym x))
(deftm raster.types.promote/convert [t :- TSym, x :- Number] :- Sym (->Sym (.doubleValue ^Number x)))

;; ================================================================
;; Arithmetic: raster.numeric ops on AbstractSymbol
;; ================================================================

;; --- Addition ---
(deftm raster.numeric/+ [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '+ (unwrap x) (unwrap y))))
(deftm raster.numeric/+ [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '+ (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/+ [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '+ (.doubleValue ^Number x) (unwrap y))))

;; --- Subtraction ---
(deftm raster.numeric/- [x :- AbstractSymbol] :- Sym
  (->Sym (list '- (unwrap x))))
(deftm raster.numeric/- [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '- (unwrap x) (unwrap y))))
(deftm raster.numeric/- [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '- (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/- [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '- (.doubleValue ^Number x) (unwrap y))))

;; --- Multiplication ---
(deftm raster.numeric/* [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '* (unwrap x) (unwrap y))))
(deftm raster.numeric/* [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '* (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/* [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '* (.doubleValue ^Number x) (unwrap y))))

;; --- Division ---
(deftm raster.numeric// [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '/ (unwrap x) (unwrap y))))
(deftm raster.numeric// [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '/ (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric// [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '/ (.doubleValue ^Number x) (unwrap y))))

;; --- Power ---
(deftm raster.numeric/pow [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list 'pow (unwrap x) (unwrap y))))
(deftm raster.numeric/pow [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list 'pow (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/pow [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list 'pow (.doubleValue ^Number x) (unwrap y))))

;; --- Sqrt, Abs, Sign ---
(deftm raster.numeric/sqrt [x :- AbstractSymbol] :- Sym
  (->Sym (list 'sqrt (unwrap x))))
(deftm raster.numeric/abs [x :- AbstractSymbol] :- Sym
  (->Sym (list 'abs (unwrap x))))
(deftm raster.numeric/sign [x :- AbstractSymbol] :- Sym
  (->Sym (list 'sign (unwrap x))))

;; --- Min, Max ---
(deftm raster.numeric/min [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list 'min (unwrap x) (unwrap y))))
(deftm raster.numeric/min [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list 'min (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/min [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list 'min (.doubleValue ^Number x) (unwrap y))))

(deftm raster.numeric/max [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list 'max (unwrap x) (unwrap y))))
(deftm raster.numeric/max [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list 'max (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/max [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list 'max (.doubleValue ^Number x) (unwrap y))))

;; --- Comparisons (return Sym, symbolic predicates) ---
(deftm raster.numeric/< [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '< (unwrap x) (unwrap y))))
(deftm raster.numeric/< [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '< (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/< [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '< (.doubleValue ^Number x) (unwrap y))))

(deftm raster.numeric/<= [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '<= (unwrap x) (unwrap y))))
(deftm raster.numeric/<= [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '<= (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/<= [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '<= (.doubleValue ^Number x) (unwrap y))))

(deftm raster.numeric/> [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '> (unwrap x) (unwrap y))))
(deftm raster.numeric/> [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '> (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/> [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '> (.doubleValue ^Number x) (unwrap y))))

(deftm raster.numeric/>= [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '>= (unwrap x) (unwrap y))))
(deftm raster.numeric/>= [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '>= (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/>= [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '>= (.doubleValue ^Number x) (unwrap y))))

(deftm raster.numeric/== [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list '== (unwrap x) (unwrap y))))
(deftm raster.numeric/== [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list '== (unwrap x) (.doubleValue ^Number y))))
(deftm raster.numeric/== [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list '== (.doubleValue ^Number x) (unwrap y))))

;; --- Predicates ---
(deftm raster.numeric/zero? [x :- AbstractSymbol]
  (->Sym (list 'zero? (unwrap x))))
(deftm raster.numeric/pos? [x :- AbstractSymbol]
  (->Sym (list 'pos? (unwrap x))))
(deftm raster.numeric/neg? [x :- AbstractSymbol]
  (->Sym (list 'neg? (unwrap x))))

;; --- real-value: error on symbolic ---
(deftm raster.numeric/real-value [x :- AbstractSymbol]
  (throw (ex-info "Cannot extract real-value from symbolic expression"
                  {:sym x})))

;; --- zero/one constructors ---
(deftm raster.numeric/zero [x :- AbstractSymbol] :- Sym (->Sym 0))
(deftm raster.numeric/one [x :- AbstractSymbol] :- Sym (->Sym 1))

;; ================================================================
;; Math functions: raster.math ops on AbstractSymbol
;; ================================================================

(deftm raster.math/sin [x :- AbstractSymbol] :- Sym
  (->Sym (list 'sin (unwrap x))))
(deftm raster.math/cos [x :- AbstractSymbol] :- Sym
  (->Sym (list 'cos (unwrap x))))
(deftm raster.math/tan [x :- AbstractSymbol] :- Sym
  (->Sym (list 'tan (unwrap x))))
(deftm raster.math/asin [x :- AbstractSymbol] :- Sym
  (->Sym (list 'asin (unwrap x))))
(deftm raster.math/acos [x :- AbstractSymbol] :- Sym
  (->Sym (list 'acos (unwrap x))))
(deftm raster.math/atan [x :- AbstractSymbol] :- Sym
  (->Sym (list 'atan (unwrap x))))
(deftm raster.math/atan2 [y :- AbstractSymbol, x :- AbstractSymbol] :- Sym
  (->Sym (list 'atan2 (unwrap y) (unwrap x))))
(deftm raster.math/atan2 [y :- AbstractSymbol, x :- Number] :- Sym
  (->Sym (list 'atan2 (unwrap y) (.doubleValue ^Number x))))
(deftm raster.math/atan2 [y :- Number, x :- AbstractSymbol] :- Sym
  (->Sym (list 'atan2 (.doubleValue ^Number y) (unwrap x))))

(deftm raster.math/sinh [x :- AbstractSymbol] :- Sym
  (->Sym (list 'sinh (unwrap x))))
(deftm raster.math/cosh [x :- AbstractSymbol] :- Sym
  (->Sym (list 'cosh (unwrap x))))
(deftm raster.math/tanh [x :- AbstractSymbol] :- Sym
  (->Sym (list 'tanh (unwrap x))))

(deftm raster.math/exp [x :- AbstractSymbol] :- Sym
  (->Sym (list 'exp (unwrap x))))
(deftm raster.math/log [x :- AbstractSymbol] :- Sym
  (->Sym (list 'log (unwrap x))))
(deftm raster.math/log2 [x :- AbstractSymbol] :- Sym
  (->Sym (list 'log2 (unwrap x))))
(deftm raster.math/log10 [x :- AbstractSymbol] :- Sym
  (->Sym (list 'log10 (unwrap x))))

(deftm raster.math/ceil [x :- AbstractSymbol] :- Sym
  (->Sym (list 'ceil (unwrap x))))
(deftm raster.math/floor [x :- AbstractSymbol] :- Sym
  (->Sym (list 'floor (unwrap x))))
(deftm raster.math/cbrt [x :- AbstractSymbol] :- Sym
  (->Sym (list 'cbrt (unwrap x))))

(deftm raster.math/fma [x :- AbstractSymbol, y :- AbstractSymbol, z :- AbstractSymbol] :- Sym
  (->Sym (list 'fma (unwrap x) (unwrap y) (unwrap z))))

(deftm raster.math/hypot [x :- AbstractSymbol, y :- AbstractSymbol] :- Sym
  (->Sym (list 'hypot (unwrap x) (unwrap y))))
(deftm raster.math/hypot [x :- AbstractSymbol, y :- Number] :- Sym
  (->Sym (list 'hypot (unwrap x) (.doubleValue ^Number y))))
(deftm raster.math/hypot [x :- Number, y :- AbstractSymbol] :- Sym
  (->Sym (list 'hypot (.doubleValue ^Number x) (unwrap y))))

;; ================================================================
;; Abstract interpreter for Sym tracing
;; ================================================================

(declare sym-interpret)

(def ^:private demangle-map
  {"_plus_" "+", "_minus_" "-", "_star_" "*", "_div_" "/"
   "_eq__eq_" "==", "_lt_" "<", "_gt_" ">", "_lt__eq_" "<="
   "_gt__eq_" ">=", "_bang_" "!"})

(defn- impl-sym->dispatch-fn
  "Resolve an impl var symbol like raster.numeric/_star__m_double_double-impl
   back to its dispatch function for Sym-compatible calling."
  [impl-sym]
  (when-let [ns-str (namespace impl-sym)]
    (let [n (name impl-sym)
          ;; Strip -impl suffix
          base (if (.endsWith n "-impl")
                 (subs n 0 (clojure.core/- (count n) 5))
                 n)
          ;; Strip _m_... type tag suffix
          m-idx (.indexOf base "_m_")]
      (when (pos? m-idx)
        (let [mangled (subs base 0 m-idx)
              ;; Demangle: _star_ → *, _plus_ → +, etc.
              demangled (reduce (fn [s [from to]]
                                  (.replace ^String s ^String from ^String to))
                                mangled demangle-map)
              dispatch-sym (symbol ns-str demangled)]
          (when-let [v (resolve dispatch-sym)]
            @v))))))

(defn- sym-interpret-call
  "Interpret a function call with Sym-aware dispatch."
  [form env]
  (let [[head & args] form
        evaled-args (mapv #(sym-interpret % env) args)]
    (cond
      ;; .invk devirtualized call — resolve back to dispatch function for Sym compat
      (= '.invk head)
      (let [[impl & call-args] args
            evaled-call-args (mapv #(sym-interpret % env) call-args)]
        (if-let [dispatch-fn (when (symbol? impl) (impl-sym->dispatch-fn impl))]
          (apply dispatch-fn evaled-call-args)
          (->Sym (list* impl (map unwrap evaled-call-args)))))

      ;; Qualified symbol — resolve and call
      (and (symbol? head) (namespace head))
      (if-let [f (resolve head)]
        (apply @f evaled-args)
        (->Sym (list* head (map unwrap evaled-args))))

      ;; Unqualified symbol in env (local fn)
      (and (symbol? head) (contains? env head))
      (let [f (get env head)]
        (if (fn? f)
          (apply f evaled-args)
          (->Sym (list* (unwrap f) (map unwrap evaled-args)))))

      ;; Fallback: build symbolic call
      :else
      (->Sym (list* head (map unwrap evaled-args))))))

(defn sym-interpret
  "Interpret a walked S-expression with Sym-aware evaluation.
   env: map of symbol → value (Sym or concrete)."
  [form env]
  (cond
    ;; Symbol lookup
    (symbol? form)
    (if (contains? env form)
      (get env form)
      (if (namespace form)
        (if-let [v (resolve form)]
          (if (fn? @v) @v (deref v))
          form)
        form))

    ;; Literal
    (or (number? form) (nil? form) (boolean? form) (string? form) (keyword? form))
    form

    ;; let* / let
    (and (seq? form) (contains? #{'let 'let* 'clojure.core/let} (first form)))
    (let [[_ bindings & body] form
          pairs (partition 2 bindings)
          env' (reduce (fn [e [sym init]]
                         (assoc e sym (sym-interpret init e)))
                       env pairs)]
      (last (map #(sym-interpret % env') body)))

    ;; do
    (and (seq? form) (= 'do (first form)))
    (last (map #(sym-interpret % env) (rest form)))

    ;; if
    (and (seq? form) (= 'if (first form)))
    (let [[_ test then else] form
          test-val (sym-interpret test env)]
      (cond
        ;; Symbolic test: explore both branches
        (sym? test-val)
        (let [then-val (sym-interpret then env)
              else-val (when else (sym-interpret else env))]
          (->Sym (list 'if (unwrap test-val)
                       (unwrap then-val)
                       (if else-val (unwrap else-val) nil))))
        ;; Concrete test
        test-val (sym-interpret then env)
        :else    (when else (sym-interpret else env))))

    ;; loop/recur — preserve symbolically
    (and (seq? form) (= 'loop (first form)))
    (let [[_ bindings & body] form
          pairs (partition 2 bindings)
          init-vals (mapv (fn [[_ init]] (sym-interpret init env)) pairs)
          loop-syms (mapv first pairs)]
      (->Sym (list* 'loop
                    (vec (interleave loop-syms (map unwrap init-vals)))
                    body)))

    ;; quote
    (and (seq? form) (= 'quote (first form)))
    (second form)

    ;; throw — preserve symbolically
    (and (seq? form) (= 'throw (first form)))
    (->Sym (list 'throw (unwrap (sym-interpret (second form) env))))

    ;; case* — preserve symbolically
    (and (seq? form) (= 'case* (first form)))
    (->Sym form)

    ;; dotimes — preserve symbolically
    (and (seq? form) (= 'dotimes (first form)))
    (->Sym form)

    ;; Function call
    (seq? form)
    (sym-interpret-call form env)

    ;; Vector
    (vector? form)
    (mapv #(sym-interpret % env) form)

    ;; Default
    :else form))

;; ================================================================
;; Public API: trace-symbolic
;; ================================================================

(defn trace-symbolic
  "Trace a deftm var symbolically, producing a Sym expression tree.

   f-var: a deftm var (e.g. #'my-fn)
   sym-bindings: map of param-name (symbol) → value (Sym or number)

   Returns a Sym whose .form is the symbolic expression."
  [f-var sym-bindings]
  (let [meta-map (meta f-var)
        ;; Resolve to the mangled var with walked body
        resolved (or (when (:raster.core/deftm meta-map) f-var)
                     (when-let [dt (:raster.core/dispatch-table meta-map)]
                       (let [entries (vals @dt)
                             method (first (first entries))
                             ns-obj (:ns meta-map)
                             mangled (symbol (str (:name meta-map) "_m_"
                                                  (clojure.string/join "_" (:tags method))))]
                         (ns-resolve ns-obj mangled))))
        _ (when-not resolved
            (throw (ex-info "Var has no deftm walked body" {:var f-var})))
        resolved-meta (meta resolved)
        walked-body (or (:raster.core/deftm-walked-body resolved-meta)
                        ((requiring-resolve 'raster.core/ensure-walked-body!) resolved))
        params (:raster.core/deftm-params resolved-meta)
        ;; Build env from param→sym-binding mapping
        env (reduce (fn [e p]
                      (if-let [v (get sym-bindings p)]
                        (assoc e p v)
                        e))
                    {} params)
        result (if (= 1 (count walked-body))
                 (sym-interpret (first walked-body) env)
                 (last (map #(sym-interpret % env) walked-body)))]
    (if (sym? result) result (->Sym result))))

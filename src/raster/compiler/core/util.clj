(ns raster.compiler.core.util
  "Shared utilities for the compiler pipeline."
  (:require [clojure.set :as set]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]))

;; ================================================================
;; Free variable analysis
;; ================================================================

(defn free-syms-flat
  "Collect unqualified symbols from a FLAT S-expression (no scoping).
   Only correct for already-flattened let* forms where all bindings are
   at top level. For nested forms, use free-syms instead."
  [e]
  (cond (and (symbol? e) (not (namespace e))) #{e}
        (seq? e)    (apply set/union #{} (map free-syms-flat e))
        (vector? e) (apply set/union #{} (map free-syms-flat e))
        :else       #{}))

(defn free-syms
  "Collect unqualified free symbols from a (possibly nested) S-expression,
   correctly respecting lexical scoping from all binding forms.

   Uses form/form-info for form classification — adding a new binding form
   to form.clj is sufficient; no changes needed here.

   Handles: let/let*, loop/loop*, dotimes, fn/fn*, letfn*, try/catch."
  ([e] (free-syms e #{}))
  ([e bound]
   (cond
     ;; Symbol: free if unqualified, not bound, not a method call (.foo),
     ;; and not a clojure.core var (cast intrinsics like double, long, int
     ;; appear bare in walked IR but are never local variables)
     (symbol? e)
     (let [n (name e)]
       (if (and (not (namespace e))
                (not (contains? bound e))
                (not (.startsWith n "."))
                (not (when-let [v (try (ns-resolve 'clojure.core e) (catch Exception _ nil))]
                       (var? v))))
         #{e}
         #{}))

     ;; Sequence forms — dispatch on kind
     (seq? e)
     (let [info (form/form-info e)
           kind (:kind info)]
       (case kind
         ;; let/let* — thread bindings sequentially
         :binding
         (let [[_ bindings & body] e
               pairs (partition 2 bindings)
               [bound' binding-frees]
               (reduce (fn [[b frees] [sym init]]
                         [(conj b sym)
                          (set/union frees (free-syms init b))])
                       [bound #{}] pairs)
               body-frees (apply set/union #{} (map #(free-syms % bound') body))]
           (set/union binding-frees body-frees))

         ;; loop/loop*/dotimes — thread bindings, scope covers body
         :scope
         (let [head (first e)]
           (if (= 'dotimes head)
             (let [[_ [sym bound-expr] & body] e
                   bound' (conj bound sym)]
               (set/union (free-syms bound-expr bound)
                          (apply set/union #{} (map #(free-syms % bound') body))))
             ;; loop/loop*
             (let [[_ bindings & body] e
                   pairs (partition 2 bindings)
                   [bound' binding-frees]
                   (reduce (fn [[b frees] [sym init]]
                             [(conj b sym)
                              (set/union frees (free-syms init b))])
                           [bound #{}] pairs)
                   body-frees (apply set/union #{} (map #(free-syms % bound') body))]
               (set/union binding-frees body-frees))))

         ;; fn/fn*/ftm — params are local to each arity
         :lambda
         (let [arities (if (vector? (second e))
                         [(rest e)]
                         (rest e))]
           (apply set/union #{}
                  (for [arity arities]
                    (let [params (first arity)
                          fn-bound (into bound (filter symbol? params))
                          ;; A walked ftm arity is, after the param vector:
                          ;;   [:- <ret>]?  :raster.walker/source-body <raw-vec>  <walked-body...>
                          ;; The `:- <ret>` is type syntax (not a value), and the
                          ;; source-body payload is the deliberately-unqualified raw
                          ;; body kept for AD transparency. Neither is real code to
                          ;; scan for captures — the walked-body alone is fully
                          ;; qualified (.invk + ns-qualified calls). Scanning the
                          ;; raw payload/annotations leaks type heads (`Array`) and
                          ;; unqualified callees as bogus free vars.
                          ;; Strip leading `:- <ret>` and any `:raster.walker/
                          ;; source-body <vec>` pairs (an ftm may be walked more
                          ;; than once, nesting multiple source-body markers).
                          after (loop [a (rest arity)]
                                  (cond
                                    (= :- (first a)) (recur (drop 2 a))
                                    (= :raster.walker/source-body (first a)) (recur (drop 2 a))
                                    :else a))]
                      (apply set/union #{} (map #(free-syms % fn-bound) after))))))

         ;; try/catch/finally — catch binds exception variable
         :special
         (if (= 'try (first e))
           (apply set/union #{}
                  (for [sub (rest e)]
                    (if (and (seq? sub) (= 'catch (first sub)))
                      (let [[_ _ex-type ex-sym & catch-body] sub
                            catch-bound (conj bound ex-sym)]
                        (apply set/union #{} (map #(free-syms % catch-bound) catch-body)))
                      (free-syms sub bound))))
           ;; recur, throw, new — just recurse into args
           (apply set/union #{} (map #(free-syms % bound) (rest e))))

         ;; par forms — use par-scope-info for unified scope decomposition
         :par
         (if-let [{:keys [scoped-syms inner-exprs outer-exprs]} (par/par-scope-info e)]
           (let [scoped-set (set (filter symbol? scoped-syms))
                 body-bound (into bound scoped-set)
                 outer-frees (apply set/union #{}
                                    (map #(free-syms % bound) outer-exprs))
                 inner-frees (apply set/union #{}
                                    (map #(free-syms % body-bound) inner-exprs))]
             (set/union outer-frees inner-frees))
           ;; No scope info (scatter!, rng-fill!, etc.) — recurse into all args
           (apply set/union #{} (map #(free-syms % bound) (rest e))))

         ;; do, branch, invk, var-ref — head is syntax, not a value reference.
         ;; Recurse into subforms, skip head.
         (:do :branch :invk :var-ref)
         (apply set/union #{} (map #(free-syms % bound) (rest e)))

         ;; :call — head IS a value reference when it's a bare unqualified symbol
         ;; (e.g. (f k1 u t) where f is a local holding a function value).
         ;; Qualified heads (clojure.core/+, Math/pow) are var/class references,
         ;; not local variables — they're excluded by the (namespace head) check
         ;; in the symbol handler. So we simply walk ALL subforms including head.
         ;; :call
         (apply set/union #{} (map #(free-syms % bound) e))))

     ;; Vectors — recurse
     (vector? e) (apply set/union #{} (map #(free-syms % bound) e))

     ;; Literals — no free symbols
     :else #{})))

(defn free-syms-excluding
  "Collect unqualified free symbols, excluding a set of operator symbols.
   Scope-aware — uses free-syms internally."
  [operator-syms e]
  (set/difference (free-syms e) operator-syms))

(defn remake
  "Construct a new list form preserving metadata from orig.
   Use this instead of bare (apply list ...) or (list ...) in compiler passes
   to prevent loss of :tag, :raster.op/original, :line, :column metadata through transformations."
  [orig & children]
  (let [m (when (instance? clojure.lang.IMeta orig) (meta orig))]
    (if m
      (with-meta (apply list children) m)
      (apply list children))))

(defn remake-from
  "Construct a new list form from a seq of children, preserving metadata from orig."
  [orig children]
  (let [m (when (instance? clojure.lang.IMeta orig) (meta orig))]
    (if m
      (with-meta (apply list children) m)
      (apply list children))))

(defn call-head
  "Extract the head symbol from a call expression.
   Handles both direct calls (f x y) and .invk calls (.invk obj x y)."
  [expr]
  (when (seq? expr)
    (let [head (first expr)]
      (if (= '.invk head)
        (second expr)
        head))))

(defn call-args
  "Extract the argument symbols from a call expression."
  [expr]
  (when (seq? expr)
    (let [head (first expr)]
      (if (= '.invk head)
        (vec (nnext expr))
        (vec (rest expr))))))

;; ================================================================
;; .invk form construction with :raster.op/original metadata
;; ================================================================

(defn impl->op
  "Extract the original qualified op symbol from a mangled impl symbol.
   e.g., raster.numeric/_star__m_double_double-impl → raster.numeric/*
         raster.math/exp_m_double-impl → raster.math/exp
   Returns nil if the pattern doesn't match."
  [impl-sym]
  (when (symbol? impl-sym)
    (let [n (name impl-sym)
          ns-part (namespace impl-sym)
          ;; Strip -impl suffix
          base (if (.endsWith n "-impl") (subs n 0 (- (count n) 5)) n)
          ;; Find _m_ separator
          idx (.indexOf ^String base "_m_")]
      (when (and ns-part (pos? idx))
        (let [op-name (subs base 0 idx)
              ;; Demangle: _plus_ → +, _minus_ → -, _star_ → *, _div_ → /, etc.
              demangled (-> op-name
                            (.replace "_plus_" "+")
                            (.replace "_minus_" "-")
                            (.replace "_star_" "*")
                            (.replace "_div_" "/")
                            (.replace "_lt_" "<")
                            (.replace "_gt_" ">")
                            (.replace "_lteq_" "<=")
                            (.replace "_gteq_" ">=")
                            (.replace "_eq_" "=")
                            (.replace "_bang" "!")
                            (.replace "_qmark" "?"))]
          (symbol ns-part demangled))))))

(defn make-invk
  "Construct a (.invk impl args...) form with :raster.op/original metadata.
   Always attaches {:raster.op/original original-op} so downstream passes can identify the operation.
   Preserves any existing metadata from `original-meta`."
  ([impl-sym args]
   (make-invk impl-sym args nil))
  ([impl-sym args original-meta]
   (let [op (or (:raster.op/original original-meta) (impl->op impl-sym))
         m (cond-> (or original-meta {})
             op (assoc :raster.op/original op))]
     (with-meta (list* '.invk impl-sym args) m))))

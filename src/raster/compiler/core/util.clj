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
        ;; map literals carry symbol references too — notably a `case*` clause map
        ;; {hash [test result-expr] …}, whose result exprs reference let-bound
        ;; locals. Without this, a flat-DCE liveness scan misses those uses and
        ;; wrongly eliminates the binding they depend on (→ unbound-sym miscompile).
        (map? e)    (apply set/union #{} (map free-syms-flat (apply concat e)))
        :else       #{}))

(def ^:dynamic *shadowing-locals*
  "Set of symbols the caller KNOWS are locals (deftm params / hoisted bindings)
   in the enclosing scope, even when they collide with a clojure.core name
   (e.g. a param literally named `seq`, `first`, `count`). free-leaf? treats
   these as free-variable references despite resolving to a core var, so a
   loop/closure helper that captures them receives them as helper params
   instead of resolving to the global core fn (which then ClassCasts). Defaults
   empty → no effect on any caller that doesn't bind it."
  #{})

(defn- free-leaf?
  "True if a bare symbol is a free local-variable reference: unqualified, not
   bound, not a method call (.foo), and not a clojure.core var (cast intrinsics
   like double/long/int appear bare in walked IR but are never local vars) —
   UNLESS the symbol is a known shadowing local (see *shadowing-locals*)."
  [e bound]
  (and (symbol? e)
       (not (namespace e))
       (not (contains? bound e))
       (not (.startsWith (name e) "."))
       (or (contains? *shadowing-locals* e)
           (not (when-let [v (try (ns-resolve 'clojure.core e) (catch Exception _ nil))]
                  (var? v))))))

(defn free-syms
  "Collect unqualified free symbols from a (possibly nested) S-expression,
   correctly respecting lexical scoping from EVERY closed-core binding form.

   Generic over `form/scope-info` — the single source of binder knowledge.
   Adding a new binding form to form.clj's scope-info is sufficient; no change
   here. Handles: let*, loop*, dotimes, fn*, ftm, letfn* (recursive), reify*,
   try/catch, and all raster.par/* variants. `quote` is opaque (its body is
   code-as-data, never scanned)."
  ([e] (free-syms e #{}))
  ([e bound]
   (cond
     (symbol? e) (if (free-leaf? e bound) #{e} #{})

     ;; quote is opaque — its contents are data, not code
     (and (seq? e) (= 'quote (first e))) #{}

     (seq? e)
     (if-let [{:keys [scopes outer sequential? rec?]} (form/scope-info e)]
       (let [outer-free (apply set/union #{} (map #(free-syms % bound) outer))
             scope-free
             (apply set/union #{}
                    (for [{:keys [binders inits body]} scopes]
                      (let [bset (set (filter symbol? binders))]
                        (if sequential?
                          ;; let*/loop*: init_k sees bound + prior binders
                          (let [[bound' ifree]
                                (reduce (fn [[b fr] [sym init]]
                                          [(conj b sym) (set/union fr (free-syms init b))])
                                        [bound #{}] (map vector binders inits))
                                bfree (apply set/union #{} (map #(free-syms % bound') body))]
                            (set/union ifree bfree))
                          ;; fn*/letfn*/par/catch/dotimes: inits see bound (or
                          ;; bound+binders if :rec?), body sees bound+binders
                          (let [init-bound (if rec? (into bound bset) bound)
                                ifree (apply set/union #{} (map #(free-syms % init-bound) inits))
                                body-bound (into bound bset)
                                bfree (apply set/union #{} (map #(free-syms % body-bound) body))]
                            (set/union ifree bfree))))))]
         (set/union outer-free scope-free))
       ;; non-binder seq: :call head is a value ref (local fn); else head is
       ;; syntax (.invk/if/do/special/par-without-scope) — skip it
       (if (= :call (:kind (form/form-info e)))
         (apply set/union #{} (map #(free-syms % bound) e))
         (apply set/union #{} (map #(free-syms % bound) (rest e)))))

     (vector? e) (apply set/union #{} (map #(free-syms % bound) e))
     (map? e)    (apply set/union #{} (map #(free-syms % bound) (apply concat e)))
     (set? e)    (apply set/union #{} (map #(free-syms % bound) e))
     :else #{})))

(defn free-syms-excluding
  "Collect unqualified free symbols, excluding a set of operator symbols.
   Scope-aware — uses free-syms internally."
  [operator-syms e]
  (set/difference (free-syms e) operator-syms))

;; ================================================================
;; Capture-avoiding substitution & alpha-conversion (generic over scope-info)
;; ================================================================

(defn- smap-value-frees
  "Free symbols appearing across all smap values — the names that would be
   captured if a scope binder shadowed them."
  [smap]
  (apply set/union #{} (map free-syms (vals smap))))

(defn- fresh-like
  "A fresh gensym carrying sym's metadata, suffixed _α_ for traceability."
  [sym]
  (with-meta (gensym (str (name sym) "_α_")) (meta sym)))

(declare subst-syms remake-from)

(defn- subst-scope
  "Apply capture-avoiding substitution to one scope map. KEY capture: a binder
   that is an smap key is rebound → stop substituting it within. VALUE capture:
   a binder free in some smap value → alpha-rename it so substituted values keep
   referring to the OUTER binding. `leaf-fn` is the symbol-substitution policy
   (see subst-syms)."
  [smap {:keys [binders inits body aux]} sequential? rec? leaf-fn]
  (if sequential?
    ;; let*/loop*: thread — init_k sees bound + prior binders
    (let [[sm binders' inits']
          (reduce (fn [[sm bs is] [b i]]
                    (let [i'   (subst-syms sm i leaf-fn)
                          cap? (and (symbol? b) (contains? (smap-value-frees sm) b))
                          [b' sm'] (if cap?
                                     (let [f (fresh-like b)] [f (assoc (dissoc sm b) b f)])
                                     [b (if (symbol? b) (dissoc sm b) sm)])]
                      [sm' (conj bs b') (conj is i')]))
                  [smap [] []] (map vector binders inits))]
      {:binders binders' :inits inits' :body (mapv #(subst-syms sm % leaf-fn) body)})
    ;; fn*/letfn*/par/catch/dotimes: all binders enter scope at once
    (let [vfrees (smap-value-frees smap)
          [binders' rename]
          (reduce (fn [[bs rn] b]
                    (if (and (symbol? b) (contains? vfrees b))
                      (let [f (fresh-like b)] [(conj bs f) (assoc rn b f)])
                      [(conj bs b) rn]))
                  [[] {}] binders)
          inner (merge (apply dissoc smap (filter symbol? binders)) rename)]
      {:binders binders'
       ;; only :rec? forms (letfn*) let inits see the binders; others' inits
       ;; (none, in the closed core) would see the outer smap
       :inits (mapv #(subst-syms (if rec? inner smap) % leaf-fn) inits)
       :body  (mapv #(subst-syms inner % leaf-fn) body)
       ;; :aux (ftm source-body) shares the arity scope — rename it CONSISTENTLY
       ;; with the body so the raw AD payload never desyncs from the walked code.
       :aux  (some->> aux (mapv #(subst-syms inner % leaf-fn)))})))

(defn subst-syms
  "Capture-avoiding substitution of free symbol occurrences per
   smap {old-sym → replacement-expr}, over a closed-core S-expression.

   Generic over form/scope-info — handles EVERY binder uniformly (let*/loop*/
   dotimes/fn*/ftm/letfn*/reify*/catch and all raster.par/* variants), avoiding
   both KEY capture (binder rebinds an smap key) and VALUE capture (binder
   shadows a name free in an smap value). `quote` is opaque.

   `leaf-fn` (optional) is the symbol-substitution policy `(fn [smap sym] -> expr)`;
   default is plain `(get smap sym sym)`. Pass a custom leaf to preserve use-site
   metadata onto the replacement (the inliner does this for type-tag propagation)."
  ([smap expr] (subst-syms smap expr (fn [sm e] (get sm e e))))
  ([smap expr leaf-fn]
   (if (empty? smap)
     expr
     (cond
       (symbol? expr) (leaf-fn smap expr)

       (and (seq? expr) (= 'quote (first expr))) expr

       (seq? expr)
       (if-let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info expr)]
         (rebuild (mapv #(subst-scope smap % sequential? rec? leaf-fn) scopes)
                  (mapv #(subst-syms smap % leaf-fn) outer))
         (remake-from expr (map #(subst-syms smap % leaf-fn) expr)))

       (vector? expr) (mapv #(subst-syms smap % leaf-fn) expr)
       (map? expr) (into (empty expr) (map (fn [[k v]] [(subst-syms smap k leaf-fn) (subst-syms smap v leaf-fn)]) expr))
       (set? expr) (into (empty expr) (map #(subst-syms smap % leaf-fn) expr))
       :else expr))))

(declare alpha-convert)

(defn- alpha-scope
  "Alpha-rename one scope's binders to fresh names, threading env correctly:
   sequential inits see prior binders; :rec? inits see all binders; otherwise
   inits see only the enclosing env."
  [env {:keys [binders inits body aux]} sequential? rec?]
  (if sequential?
    (let [[env' binders' inits']
          (reduce (fn [[env bs is] [b i]]
                    (let [i' (alpha-convert env i)            ;; init sees prior binders
                          [b' env'] (if (symbol? b)
                                      (let [f (fresh-like b)] [f (assoc env b f)])
                                      [b env])]
                      [env' (conj bs b') (conj is i')]))
                  [env [] []] (map vector binders inits))]
      {:binders binders' :inits inits' :body (mapv #(alpha-convert env' %) body)})
    (let [pairs (mapv (fn [b] [b (if (symbol? b) (fresh-like b) b)]) binders)
          env'  (into env (filter (comp symbol? first) pairs))
          binders' (mapv second pairs)]
      {:binders binders'
       :inits (mapv #(alpha-convert (if rec? env' env) %) inits)
       :body  (mapv #(alpha-convert env' %) body)
       ;; ftm source-body (:aux) shares the arity scope — rename consistently
       :aux  (some->> aux (mapv #(alpha-convert env' %)))})))

(defn alpha-convert
  "Alpha-rename EVERY bound variable in a closed-core form to a fresh gensym,
   consistently rewriting references through env. Generic over form/scope-info.
   Use to make hygienic copies of a body (e.g. unrolled fold iterations) so two
   copies don't conflate their loop indices / accumulators. `quote` is opaque."
  ([form] (alpha-convert {} form))
  ([env form]
   (cond
     ;; Preserve the reference's metadata (esp. :raster.type/tag) onto the fresh
     ;; name — else a re-walk after inlining can't re-devirtualize arithmetic whose
     ;; operand type lived only on the renamed symbol (e.g. an inlined SDPA's
     ;; `(- (* dot scale) mx)` loses mx's element type → bare raster.numeric/- that
     ;; pass-late-cleanup rejects on GPU). Mirrors subst-sym-leaf.
     ;; MERGE ORDER: the fresh name's own meta wins — it comes from the (possibly
     ;; re-stamped) binder, which is more current than the reference. A reference
     ;; carrying a stale generic-phase tag must not overwrite a binder tag that a
     ;; monomorphizing rewalk just narrowed (double → float).
     (symbol? form) (let [r (get env form form)]
                      (if (and (not (identical? r form)) (symbol? r) (meta form))
                        (with-meta r (merge (meta form) (meta r)))
                        r))
     (and (seq? form) (= 'quote (first form))) form
     (seq? form)
     (if-let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info form)]
       (rebuild (mapv #(alpha-scope env % sequential? rec?) scopes)
                (mapv #(alpha-convert env %) outer))
       (remake-from form (map #(alpha-convert env %) form)))
     (vector? form) (mapv #(alpha-convert env %) form)
     (map? form) (into (empty form) (map (fn [[k v]] [(alpha-convert env k) (alpha-convert env v)]) form))
     (set? form) (into (empty form) (map #(alpha-convert env %) form))
     :else form)))

(declare uniquify*)

(defn- uniquify-scope
  "Uniquify one scope's binders, threading [env bound]: a binder that SHADOWS a
   name already in `bound` is renamed to a fresh name (and recorded in env);
   non-shadowing binders are kept and added to bound. Sequential inits see prior
   binders; :rec? inits see all binders."
  [env bound {:keys [binders inits body aux]} sequential? rec?]
  (if sequential?
    (let [[env' bound' binders' inits']
          (reduce (fn [[env bnd bs is] [b i]]
                    (let [i' (uniquify* env bnd i)          ;; init sees prior binders
                          [b' env' bnd'] (if (symbol? b)
                                           (if (contains? bnd b)
                                             (let [f (fresh-like b)] [f (assoc env b f) (conj bnd f)])
                                             [b env (conj bnd b)])
                                           [b env bnd])]
                      [env' bnd' (conj bs b') (conj is i')]))
                  [env bound [] []] (map vector binders inits))]
      {:binders binders' :inits inits' :body (mapv #(uniquify* env' bound' %) body)})
    (let [triples (mapv (fn [b] (if (and (symbol? b) (contains? bound b))
                                  [b (fresh-like b) true] [b b false])) binders)
          env'   (into env (keep (fn [[o n r]] (when r [o n])) triples))
          bound' (into bound (map second triples))]
      {:binders (mapv second triples)
       :inits (mapv #(uniquify* (if rec? env' env) (if rec? bound' bound) %) inits)
       :body  (mapv #(uniquify* env' bound' %) body)
       :aux   (some->> aux (mapv #(uniquify* env' bound' %)))})))

(defn- uniquify*
  [env bound form]
  (cond
    (symbol? form) (get env form form)
    (and (seq? form) (= 'quote (first form))) form
    (seq? form)
    (if-let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info form)]
      (rebuild (mapv #(uniquify-scope env bound % sequential? rec?) scopes)
               (mapv #(uniquify* env bound %) outer))
      (remake-from form (map #(uniquify* env bound %) form)))
    (vector? form) (mapv #(uniquify* env bound %) form)
    (map? form) (into (empty form) (map (fn [[k v]] [(uniquify* env bound k) (uniquify* env bound v)]) form))
    (set? form) (into (empty form) (map #(uniquify* env bound %) form))
    :else form))

(defn uniquify-rebindings
  "SSA-normalize let*/loop* REBINDINGS: rename any binder that SHADOWS a name
   already in scope (seeded with `params`) to a fresh name, capture-avoidingly
   rewriting downstream references. First bindings and non-shadowing names are
   untouched; sibling scopes don't cross-contaminate (each sees only enclosing
   bindings). Generic over form/scope-info.

   Why: a rebinding like `up = (gelu up)` where the op ALLOCATES a fresh output
   buffer conflates the input and output buffers in materialize/buffer analysis —
   the output alloc is bound to `up`, so the map body's read of `up` resolves to
   the uninitialized output. Distinct SSA names remove the hazard (matches what a
   hand-written distinct binding does)."
  ([form] (uniquify-rebindings #{} form))
  ([params form] (uniquify* {} (set params) form)))

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

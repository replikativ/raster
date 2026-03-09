(ns raster.compiler.core.hoist
  "Shape propagation and allocation rewriting for terminal buffer hoisting."
  (:require [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]))

(def ^:private free-syms
  "Collect unqualified free symbols from an S-expression."
  util/free-syms-flat)

(def ^:private alloc-constructor-syms
  "Unqualified symbols that are array constructors — delegates to op-descriptor."
  descriptor/alloc-ops)

(defn free-data-syms
  "Free symbols excluding array constructors and qualified names."
  [e]
  (util/free-syms-excluding alloc-constructor-syms e))

(def ^:private cast-syms
  "Primitive cast symbols — not data references in allocation size expressions."
  #{'long 'int 'double 'float 'byte 'short 'char 'boolean})

(defn build-dim-env
  "Walk let* bindings and build a map from vector-valued symbols to
	param-only dimension expressions. Used by shape-based hoisting to
	replace intermediate-dependent allocations with param-only sizes."
  [let-form params-set]
  (when (and (seq? let-form) (#{'let 'let*} (first let-form)))
    (let [[_ bindings-vec & _] let-form
          pairs (partition 2 bindings-vec)]
      (reduce
       (fn [env [sym expr]]
         (let [dim (cond
                     (and (symbol? expr) (contains? env expr))
                     (get env expr)

                     (and (seq? expr) (= 'do (first expr)))
                     (let [last-expr (last expr)]
                       (when (and (symbol? last-expr) (contains? env last-expr))
                         (get env last-expr)))

                     (and (seq? expr) (#{'clojure.core/let 'let} (first expr)))
                     (let [body (last expr)]
                       (when (and (symbol? body) (contains? env body))
                         (get env body)))

                     (and (seq? expr)
                          (= 'clojure.core/alength (first expr))
                          (= 2 (count expr))
                          (symbol? (second expr)))
                     (let [arg (second expr)]
                       (cond
                         (params-set arg) expr
                         (contains? env arg) (get env arg)
                         :else nil))

                     ;; .invk alength: (.invk raster.arrays/alength_m_*-impl arr)
                     (and (seq? expr)
                          (= '.invk (first expr))
                          (= 3 (count expr))
                          (symbol? (second expr))
                          (let [n (name (second expr))]
                            (.startsWith ^String n "alength_m_"))
                          (symbol? (nth expr 2)))
                     (let [arg (nth expr 2)
                           ;; Normalize to clojure.core/alength for downstream
                           canonical (list 'clojure.core/alength arg)]
                       (cond
                         (params-set arg) canonical
                         (contains? env arg) (get env arg)
                         :else nil))

                     (seq? expr)
                     (let [head (first expr)
                           args (vec (rest expr))]
                       (when-let [[op-descriptor _] (and (symbol? head)
                                                         (descriptor/resolve-op-descriptor head))]
                         (when-let [rule (get-in op-descriptor [:shape :dim-rule])]
                           (rule args env params-set)))))]
           (if (and dim (every? params-set (free-syms dim)))
             (assoc env sym dim)
             env)))
       {} pairs))))

(def ^:private alloc-ctor-remap
  "Array constructor remapping per dtype."
  {:float {'double-array 'float-array}})

(defn- remap-alloc-ctor
  "Rewrite array constructor in an expression according to dtype."
  [expr dtype]
  (if-let [remap (get alloc-ctor-remap dtype)]
    (if (and (seq? expr) (contains? remap (first expr)))
      (cons (get remap (first expr)) (rest expr))
      expr)
    expr))

(defn- param-only-syms?
  "True if all free data symbols in expr are params or primitive casts.
   Uses free-data-syms which already excludes array constructors and qualified names."
  [expr params-set]
  (every? #(or (params-set %) (cast-syms %)) (free-data-syms expr)))

(defn- resolve-dim-expr
  "Recursively resolve a size expression through dim-env.
	Returns a param-only expression or nil if any symbol can't be resolved."
  [expr dim-env params-set]
  (cond
    (symbol? expr)
    (cond
      (params-set expr) expr
      (cast-syms expr) expr
      (contains? dim-env expr)
      (let [resolved (get dim-env expr)]
        (when (param-only-syms? resolved params-set)
          resolved))
      :else nil)

    (and (seq? expr) (>= (count expr) 2))
    (let [op (first expr)
          resolved-args (mapv #(resolve-dim-expr % dim-env params-set)
                              (rest expr))]
      (when (every? some? resolved-args)
        (cons op resolved-args)))

    (number? expr) expr

    :else nil))

(defn rewrite-alloc-exprs
  "Rewrite alloc expressions using shape propagation and dtype remapping.
	For each ^:hoistable binding that references non-param symbols,
	replaces the alloc with a param-only allocation when the referenced
	symbol has a known param-only dimension in dim-env."
  ([fused-pairs dim-env params-set]
   (rewrite-alloc-exprs fused-pairs dim-env params-set nil))
  ([fused-pairs dim-env params-set dtype]
   (let [try-shape-rewrite
         (fn [ctor size-expr]
           (when-let [resolved (resolve-dim-expr size-expr dim-env params-set)]
             ;; Preserve original constructor type (float-array, long-array, etc.)
             ;; rather than using dtype, which could mismatch for non-data arrays.
             (let [rewritten (list ctor resolved)]
               (when (param-only-syms? rewritten params-set)
                 rewritten))))
         remap-pair
         (fn [[sym expr :as pair]]
           (if (:raster.buffer/hoistable (meta sym))
             (if (not (param-only-syms? expr params-set))
               ;; Try shape rewrite only for standard (ctor size) allocs,
               ;; not for multi-arg forms like (zeros-like ref size)
               (if (and (seq? expr) (= 2 (count expr)))
                 (let [ctor (first expr)
                       size-expr (second expr)
                       rewritten (try-shape-rewrite ctor size-expr)]
                   (if rewritten
                     [sym rewritten]
                     [sym (remap-alloc-ctor expr dtype)]))
                 ;; Multi-arg alloc (zeros-like): try resolving the size arg (last)
                 (if (and (seq? expr) (> (count expr) 2))
                   (let [size-expr (last expr)
                         resolved-size (resolve-dim-expr size-expr dim-env params-set)]
                     (if (and resolved-size (param-only-syms? resolved-size params-set))
                       [sym (concat (butlast expr) [resolved-size])]
                       [sym (remap-alloc-ctor expr dtype)]))
                   [sym (remap-alloc-ctor expr dtype)]))
               [sym (remap-alloc-ctor expr dtype)])
             pair))]
     (mapv remap-pair fused-pairs))))

(defn hoist-safe-pair?
  "True when a binding is a pure size-only buffer allocation that depends only
   on params. Primitive casts (long, double, etc.) in size expressions are
   transparent. Rejects allocations with computed content (e.g. object-array
   with a vector of expressions)."
  [params-set [sym alloc-expr]]
  (and (:raster.buffer/hoistable (meta sym))
       ;; Reject allocations with vector/collection args (computed content, not just size)
       (not (and (seq? alloc-expr)
                 (some vector? (rest alloc-expr))))
       (let [free (free-data-syms alloc-expr)]
         (every? #(or (params-set %) (cast-syms %)) free))))

(defn needs-zeroing?
  "True when a hoisted buffer sym needs to be zeroed before each use.
   Checks :write-mode metadata first (set by buffer-fuse), then falls back
   to consumer analysis. Only :accumulate mode needs zeroing; :overwrite
   operations (par/map!, dgemv! with beta=0) write every element."
  [buf-sym inner-pairs]
  ;; Check metadata first — buffer-fuse sets :write-mode on hoisted buf syms
  (if-let [wm (:raster.buffer/write-mode (meta buf-sym))]
    (= :accumulate wm)
    ;; Fallback: consumer analysis (for buffers not created by buffer-fuse)
    (let [consumer (first (filter (fn [[_sym expr]]
                                    (and (seq? expr)
                                         (some #{buf-sym} (rest expr))))
                                  inner-pairs))]
      (if consumer
        (let [[_csym cexpr] consumer
              op (first cexpr)]
          (cond
            ;; Check descriptor registry
            (and (symbol? op) (namespace op))
            (let [write-info (or (descriptor/get-buffer-write-mode op)
                                 (when-let [base (when (.contains ^String (name op) "_m_")
                                                   (let [n (name op)
                                                         idx (.indexOf ^String n "_m_")]
                                                     (symbol (namespace op) (subs n 0 idx))))]
                                   (descriptor/get-buffer-write-mode base)))]
              (if write-info
                (= :accumulate (:mode write-info))
                true))
            :else true))
      ;; No consumer found — buffer unused or used indirectly, zero to be safe
        true))))

(def ^:private non-variable-syms
  "Symbols that appear in code but are not variables — exclude from dependency tracking."
  (into alloc-constructor-syms
        #{'long 'int 'double 'float 'byte 'short 'char 'boolean
          'clojure.core/double-array 'clojure.core/float-array
          'clojure.core/long-array 'clojure.core/int-array
          'clojure.core/alength 'clojure.core/inc 'clojure.core/*
          'clojure.core/+ 'clojure.core/- 'clojure.core/quot}))

(defn- free-dep-syms
  "Free variable symbols from an expression, excluding non-variable forms.
   Qualified deftm call heads are excluded (they're functions, not variables)."
  [e]
  (let [head-exclude (when (and (seq? e) (symbol? (first e)) (namespace (first e)))
                       #{(first e)})]
    (util/free-syms-excluding (if head-exclude
                                (into non-variable-syms head-exclude)
                                non-variable-syms)
                              e)))

(defn- alloc-expr?
  "True if expr is a pure array allocation (zero-init, value-init, or copy).
   Recognizes array constructors, zeros-like, and aclone."
  [expr]
  (and (seq? expr) (symbol? (first expr))
       (let [head (first expr)
             base-name (name head)]
         (or (contains? alloc-constructor-syms head)
             (contains? alloc-constructor-syms (symbol base-name))
             (contains? #{'aclone 'clojure.core/aclone 'raster.arrays/aclone} head)
             ;; zeros-like (unmangled or mangled)
             (.startsWith ^String base-name "zeros-like")))))

(defn infer-hoistable
  "Mark allocation bindings as ^:hoistable when their size expressions
   depend only on function parameters (transitively through the binding graph).
   This removes the need for manual :hoistable annotations in AD templates."
  [let-form params-set]
  (if-not (and (seq? let-form) (contains? #{'let 'let*} (first let-form)))
    let-form
    (let [[head bindings-vec & body] let-form
          pairs (partition 2 bindings-vec)
          ;; Build a set of param-derived symbols: starts with params,
          ;; grows as we find bindings whose values depend only on param-derived syms
          param-derived
          (reduce (fn [derived [sym expr]]
                    (if (every? derived (free-dep-syms expr))
                      (conj derived sym)
                      derived))
                  params-set pairs)
          ;; Mark alloc bindings as hoistable if size is param-derived
          new-bindings
          (vec (mapcat (fn [[sym expr]]
                         (let [is-hoistable-alloc (and (alloc-expr? expr)
                                                       (every? param-derived (free-dep-syms expr)))
                               sym (cond-> sym
                                     (and is-hoistable-alloc (not (:raster.buffer/hoistable (meta sym))))
                                     (vary-meta assoc :raster.buffer/hoistable true))]
                           [sym expr]))
                       pairs))]
      (apply list head new-bindings body))))
(ns raster.core
  "Typed multiple dispatch for Clojure.

  Quick start with ns-raster (experimental):

    (raster.core/ns-raster my.app
      \"Optional docstring.\"
      (:require [raster.linalg.core :as la]))

  This is equivalent to:

    (ns my.app
      (:refer-clojure :exclude [+ - * / zero? min max abs sqrt pow
                                mod rem quot
                                bit-and bit-or bit-xor bit-not
                                bit-shift-left bit-shift-right
                                unsigned-bit-shift-right
                                < <= > >= == even? odd? pos? neg?
                                aget aset alength aclone])
      (:require [raster.core :refer [deftm ftm defvalue defabstract defval
                 specialize
                                     methods-of ambiguities print-methods
                                     invoke prefer-method! remove-method!]]
                [raster.numeric :refer [+ - * / zero? min max abs sqrt pow
                                        mod rem quot
                                        bit-and bit-or bit-xor bit-not
                                        bit-shift-left bit-shift-right
                                        unsigned-bit-shift-right
                                        < <= > >= == even? odd? pos? neg?
                                        zero one sign clamp similar
                                        real-value derivative-value
                                        add! sub! scale! axpy!]]
                [raster.arrays :refer [aget aset alength aclone]]
                [raster.linalg.core :as la]))

  ns-raster is experimental — feedback welcome. You can always use plain ns
  with explicit :refer-clojure :exclude and :require forms instead.

  Inside deftm bodies, typed macros are available:
  - (broadcast [syms...] expr) — functional element-wise map over arrays
  - (muladd expr) — rewrite (+ a (* b c)) to (Math/fma b c a)

  Each deftm call adds a typed method to a generic function.
  Dispatch selects the most-specific matching method based on
  argument types, using instanceof checks at runtime."
  (:refer-clojure :exclude [defn])
  (:require [clojure.string :as str]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.core.specialize :as specialize]
            [raster.compiler.backend.jvm.valhalla :as valhalla]
            [raster.compiler.core.method-entry :as me]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.typed-dispatch :as typed-dispatch]
            [raster.runtime.callsites :as callsites]))

(def ^:dynamic ^:no-doc *simplify?*
  "When true, algebraic simplification is applied to walked deftm bodies.
  Eliminates identity operations (+0, *1, etc.) and folds constants."
  false)

(def ^:dynamic *warn-on-boxed-dispatch*
  "When true, warn when the walker cannot devirtualize a deftm call to a
  typed .invk path — analogous to Clojure's *warn-on-reflection*.
  This means the call will go through boxed IFn dispatch at runtime.
  Enable with (set! *warn-on-boxed-dispatch* true) in the ns form."
  false)

;; ================================================================
;; ns-raster — convenience namespace macro (experimental)
;; ================================================================

(def ^:no-doc raster-clojure-excludes
  "Clojure.core symbols that raster.numeric and raster.arrays replace."
  '[+ - * / zero? min max abs sqrt pow
    mod rem quot
    bit-and bit-or bit-xor bit-not
    bit-shift-left bit-shift-right
    unsigned-bit-shift-right
    < <= > >= == even? odd? pos? neg?
    aget aset alength aclone])

(def ^:no-doc raster-numeric-refers
  "Symbols to refer from raster.numeric."
  '[+ - * / zero? min max abs sqrt pow
    mod rem quot
    bit-and bit-or bit-xor bit-not
    bit-shift-left bit-shift-right
    unsigned-bit-shift-right
    < <= > >= == even? odd? pos? neg?
    zero one sign clamp similar
    real-value derivative-value
    add! sub! scale! axpy!])

(def ^:no-doc raster-core-refers
  "Symbols to refer from raster.core."
  '[defn deftm ftm defvalue defabstract defval
    specialize
    methods-of ambiguities print-methods
    invoke prefer-method! remove-method!])

(def ^:no-doc raster-array-refers
  "Symbols to refer from raster.arrays."
  '[aget aset alength aclone])

(defmacro ns-raster
  "Like clojure.core/ns, but auto-imports raster's polymorphic arithmetic,
  core macros, and typed arrays.  All standard ns clauses (:require, :import,
  :use, :refer-clojure, etc.) work as normal — they are merged with the
  raster defaults.

  EXPERIMENTAL — this macro may change based on user feedback.
  You can always use plain (ns ...) with explicit :refer-clojure/:require instead.

  Usage:
    (raster.core/ns-raster my.app
      \"My awesome app.\"
      (:require [raster.linalg.core :as la]
                [clojure.string :as str])
      (:import (java.util Date)))"
  {:arglists '([name docstring? attr-map? references*])}
  [name & references]
  (let [;; Parse optional docstring and attr-map (same as clojure.core/ns)
        docstring  (when (string? (first references)) (first references))
        references (if docstring (next references) references)
        metadata   (when (map? (first references)) (first references))
        references (if metadata (next references) references)
        ;; Group user references by kind
        user-refs (group-by first references)
        ;; Merge :refer-clojure excludes
        user-rc (first (get user-refs :refer-clojure))
        user-rc-filters (when user-rc (rest user-rc))
        ;; Extract user's :exclude list if any
        user-excludes (when user-rc-filters
                        (let [pairs (partition 2 user-rc-filters)
                              excl-pair (first (filter #(= :exclude (first %)) pairs))]
                          (when excl-pair (second excl-pair))))
        merged-excludes (vec (distinct (concat raster-clojure-excludes user-excludes)))
        ;; Rebuild :refer-clojure with merged excludes + any other filters (:only, :rename)
        other-rc-filters (when user-rc-filters
                           (mapcat identity
                                   (remove #(= :exclude (first %))
                                           (partition 2 user-rc-filters))))
        rc-clause `(:refer-clojure :exclude ~merged-excludes ~@other-rc-filters)
        ;; Merge :require — prepend raster requires before user's
        user-requires (mapcat rest (get user-refs :require))
        raster-requires [`[~'raster.core :refer ~raster-core-refers]
                         `[~'raster.numeric :refer ~raster-numeric-refers]
                         `[~'raster.arrays :refer ~raster-array-refers]]
        merged-require (when-let [all (seq (concat raster-requires user-requires))]
                         `(:require ~@all))
        ;; Pass through all other reference types unchanged
        other-refs (mapcat identity
                           (vals (dissoc user-refs :refer-clojure :require)))
        ;; Assemble final references
        all-refs (concat [rc-clause]
                         (when merged-require [merged-require])
                         other-refs)]
    `(ns ~(if (or docstring metadata)
            (cond-> name
              docstring (vary-meta assoc :doc docstring)
              metadata  (vary-meta merge metadata))
            name)
       ~@all-refs)))

;; walk-body is no longer re-exported from core. Use walker/walk-body directly.

(defn- build-walker-type-env
  "Build a unified type-env from deftm/ftm parameter annotations.
  Delegates to inference/build-param-type-env — single source of truth
  for param→type-env construction."
  [params annotations]
  (let [tags (types/extract-tags params annotations)]
    (inf/build-param-type-env params tags annotations)))

(defn- validate-annotations!
  "Warn about type annotations that don't resolve to known types."
  [fn-label params tags]
  (doseq [[param tag] (map vector params tags)]
    (when (and tag (symbol? tag)
               (not= tag 'Object)
               (not (contains? #{'double 'long 'float 'int 'boolean
                                 'byte 'short 'char 'void
                                 'doubles 'longs 'floats 'ints
                                 'bytes 'shorts 'chars 'booleans
                                 'objects 'Double 'Long 'Float
                                 'Integer 'Boolean 'Byte 'Short
                                 'Character 'Void 'Object} tag))
               (not (try (Class/forName (str tag)) (catch Exception _ nil)))
               (not (try (let [resolved (ns-resolve *ns* tag)]
                           (when (class? resolved) resolved))
                         (catch Exception _ nil))))
      (binding [*out* *err*]
        (println (str "WARNING: " fn-label " — type annotation '" tag
                      "' on param '" param "' cannot be resolved. "
                      "Will dispatch as Object."))))))

(defn- prepare-typed-body
  "Shared preparation for deftm/ftm/defn*: parse annotations, validate,
   build type-env, walk body. Returns a map with all intermediate results.

   opts:
     :validate?     — warn on unresolvable annotations (default true)
     :fn-label      — label for validation warnings (e.g. \"deftm foo\")
     :tc-eager?     — run TC before walking (ftm=true, deftm=false)
     :defer-walk?   — skip walking for typed functions (Julia model)
     :simplify?     — apply simplifier to walked body"
  [param-vec ret-args opts]
  (let [{:keys [validate? fn-label tc-eager? defer-walk? simplify?]
         :or {validate? true tc-eager? false defer-walk? false simplify? *simplify?*}} opts
        ;; Parse return type and body from ret-args
        ;; Return type: inline `:- Ret` after the arg vector, else metadata
        ;; `^{:- Ret}` on the arg vector itself (mirrors Clojure's ^long [x]).
        [ret-type body] (if (= :- (first ret-args))
                          [(second ret-args) (nnext ret-args)]
                          [(types/meta-type-annotation param-vec) ret-args])
        ;; Parse typed params
        {:keys [params annotations]} (types/parse-typed-params param-vec)
        tags (types/extract-tags params annotations)
        explicit-ret-tag (when ret-type (types/annotation->tag ret-type nil))
        ;; Determine generate-typed? early (needed for defer-walk? decision)
        generate-typed? (types/has-primitive-annotations? annotations ret-type)
        ;; Julia model: skip walking for typed deftm — walk happens at JIT time
        skip-walk? (and defer-walk? generate-typed?)
        ;; Infer return type via TC when no explicit annotation and TC is eager.
        ;; When tc-eager? is false (default), the walker uses structural inference
        ;; and TC runs later at compile-aot/specialize time.
        tc-ret-tag (when (and tc-eager? (not explicit-ret-tag) (seq annotations))
                     (try
                       (:ret-tag (inf/tc-analyze-deftm-body
                                  (or fn-label '<infer>) params annotations body *ns*))
                       (catch Throwable _ nil)))
        ret-tag (or explicit-ret-tag tc-ret-tag)
        ;; Validate
        _ (when validate?
            (validate-annotations! (or fn-label "deftm") params tags))
        ;; Build hinted params (for IFn defn path)
        hinted-params (types/build-hinted-params params annotations)
        hinted-params (if-let [ret-hint (if (> (count params) 4)
                                          (types/annotation->hint ret-type)
                                          (types/annotation->return-hint ret-type))]
                        (vary-meta hinted-params assoc :tag ret-hint)
                        hinted-params)
        ;; Build type-env
        full-type-env (build-walker-type-env params annotations)
        plain-type-env (reduce-kv (fn [m s rec] (assoc m s (dissoc rec :fn-info))) {} full-type-env)
        ;; TC binding tags (eager for ftm, deferred for deftm)
        tc-binding-tags (when tc-eager?
                          (try
                            (:binding-tags (inf/tc-analyze-deftm-body
                                            (or fn-label '<ftm>) params annotations body *ns*))
                            (catch Throwable _ nil)))
        ;; Walk body with plain type-env (IFn path) — skipped for Julia-model deftm
        walk-opts (cond-> {:type-env plain-type-env :source-ns *ns*}
                    (seq tc-binding-tags) (assoc :tc-binding-tags tc-binding-tags))
        walked-body (when-not skip-walk?
                      (let [wb (map #(walker/walk-body % walk-opts) body)]
                        (if simplify? (map simplify/simplify wb) wb)))
        ;; Conditional second walk with fn-info (typed .invk path)
        has-fn-params? (some :fn-info (vals full-type-env))
        walked-body-typed (when (and (not skip-walk?) has-fn-params?)
                            (let [full-walk-opts (assoc walk-opts :type-env full-type-env)
                                  wb (map #(walker/walk-body % full-walk-opts) body)]
                              (if simplify? (map simplify/simplify wb) wb)))
        ;; Typed interface info
        typed-iface-info (when generate-typed?
                           (let [prim-tags (mapv #(types/annotation->tag %1 %2) annotations params)]
                             (types/ensure-fn-interface! prim-tags)))
        prim-hinted-params (when generate-typed?
                             (types/build-prim-hinted-params params annotations))]
    {:params params
     :annotations annotations
     :tags tags
     :ret-type ret-type
     :ret-tag ret-tag
     :body body
     :hinted-params hinted-params
     :full-type-env full-type-env
     :plain-type-env plain-type-env
     :walked-body walked-body
     :has-fn-params? has-fn-params?
     :walked-body-typed walked-body-typed
     :generate-typed? generate-typed?
     :typed-iface-info typed-iface-info
     :typed-iface-name (when typed-iface-info (:iface-name typed-iface-info))
     :prim-hinted-params prim-hinted-params}))

;; ================================================================
;; Bytecode-compiled typed impls (opt-in, lazy-loaded)
;; ================================================================

(def ^:dynamic *compile-deftm-bytecode?*
  "When true, deftm typed-impls are lazily compiled with Raster's bytecode
  compiler on first invocation (Julia-style JIT). The compiled impl replaces
  the Clojure deftype via var swap."
  true)

(def ^:dynamic *jit-silent-fallback?*
  "When false (default), bytecode compilation errors are thrown instead of
  silently falling back to the deftype impl."
  false)

(defn- jit-walk-with-tc
  "Walk body with TC binding tags for JIT compilation.
  Julia model: type inference + devirtualization happens at first invocation,
  not at definition time. This gives TC-quality code in the JIT path."
  [source-body params tags annotations source-ns]
  (let [type-env (build-walker-type-env params annotations)
        tc-binding-tags (try
                          (:binding-tags (inf/tc-analyze-deftm-body
                                          '<jit> params annotations source-body source-ns))
                          (catch Throwable _ nil))
        walk-opts (cond-> {:type-env type-env :source-ns (or source-ns *ns*)}
                    (seq tc-binding-tags) (assoc :tc-binding-tags tc-binding-tags))]
    (mapv #(walker/walk-body % walk-opts) source-body)))

(def ^:dynamic *jit-simd?*
  "Opt-in: vectorize par/map and par/reduce in the lazy-JIT bytecode upgrade via
  the JDK Vector API (par-simd/simd-pass), the same codegen compile-aot uses.

  Default FALSE. The emitted vector bytecode is correct and identical to
  compile-aot's, but when a small kernel (e.g. a dim~50 cos-dist) is invoked
  through the deftm `.invk` interface inside a hot loop, C2's escape analysis
  fails to scalarize the per-call DoubleVector temporaries — they heap-allocate
  every call, and at millions of calls that is a large net regression (measured
  ~12x on UMAP kNN) despite the loop body being 'vectorized'. compile-aot does
  not hit this because it inlines the kernel into its caller so EA can see the
  whole region. Until lazy-JIT SIMD is made reliably scalarizable across the
  .invk boundary, SIMD out of the box lives in compile-aot; bind this to true to
  opt a lazy-JIT kernel into SIMD when you know it pays (large bounds, or the
  kernel is the whole hot method). simd-pass keeps a scalar tail + falls back to
  a scalar loop, so results stay correct (reductions reorder FP adds like BLAS)."
  false)

(defn- simd-expand-jit-body
  "Replace par forms in a lazy-JIT walked body with SIMD code when *jit-simd?*.
  Per-form: vectorize via simd-pass, else leave the par form for bytecode's
  scalar expansion. Never throws — a failed form keeps its scalar shape."
  [walked-body source-ns]
  (if-not *jit-simd?*
    walked-body
    (binding [*ns* (or source-ns *ns*)]
      (mapv (fn [form]
              (try (:form (par-simd/simd-pass form))
                   (catch Throwable _ form)))
            walked-body))))

(clojure.core/defn ^:no-doc do-bytecode-upgrade!
  "Compile a deftm body to bytecode and swap the -impl var.
  Julia model: re-walks from source body with TC at JIT time for
  TC-quality devirtualization. Falls back to pre-walked body if
  TC re-walk fails. Returns the compiled impl on success."
  [impl-var mangled-sym params tags return-tag walked-body source-ns typed-iface-name
   & {:keys [annotations source-body]}]
  (try
    (let [;; Create compilation DCL if not already in a compilation scope
          needs-cl? (nil? me/*compilation-classloader*)
          cl (when needs-cl?
               (clojure.lang.DynamicClassLoader.
                (.getContextClassLoader (Thread/currentThread))))]
      (binding [me/*compilation-classloader* (or me/*compilation-classloader* cl)]
        (let [;; Re-walk from source body with TC for better devirtualization
              effective-body (if (and source-body annotations)
                               (try (jit-walk-with-tc (vec source-body) params tags
                                                      annotations source-ns)
                                    (catch Throwable _
                                      walked-body))
                               walked-body)
              ;; Vectorize par/map + par/reduce out of the box (scalar fallback inside).
              effective-body (simd-expand-jit-body effective-body source-ns)
              class-name (str "raster.compiled.DT_"
                              (.replace (str mangled-sym) "." "_")
                              "_" (bit-and (hash effective-body) 0x7FFFFFFF)
                              "_" (mod (System/nanoTime) 1000000))
              impl (bytecode/compile-typed-impl! class-name params tags return-tag
                                                 effective-body source-ns typed-iface-name
                                                 (str (ns-name source-ns)) (str mangled-sym))]
          (when impl
            (let [old-impl (.getRawRoot ^clojure.lang.Var impl-var)]
              (alter-var-root impl-var (constantly impl))
              (try
            ;; Update dispatch table: the cached :typed-impl points to the old
            ;; deftype instance. Find the generic function var and replace the
            ;; old impl with the new BC-compiled one.
                (let [impl-ns (.ns ^clojure.lang.Var impl-var)]
              ;; Walk all dispatch tables in this namespace
                  (doseq [[_ v] (ns-interns impl-ns)]
                    (when-let [table-atom (::dispatch-table (meta v))]
                      (swap! table-atom
                             (fn [table]
                               (into {}
                                     (map (fn [[arity entries]]
                                            [arity (mapv (fn [entry]
                                                           (if (identical? (:typed-impl entry) old-impl)
                                                             (assoc entry :typed-impl impl)
                                                             entry))
                                                         entries)])
                                          table)))))))
            ;; Update the dispatch object's impl field for this overload.
            ;; The typed dispatch class holds impl fields for each IFn__
            ;; interface — update the field that pointed to old-impl.
                (let [impl-ns (.ns ^clojure.lang.Var impl-var)]
                  (doseq [[_ v] (ns-interns impl-ns)]
                    (when (::dispatch-table (meta v))
                      (let [dispatch-obj (.getRawRoot ^clojure.lang.Var v)]
                        (typed-dispatch/update-dispatch-impl! dispatch-obj old-impl impl)))))
            ;; Invalidate MutableCallSites so C2 re-inlines with new impl
                (callsites/invalidate! impl-var impl)
                impl
                (catch Exception e
              ;; Rollback var to old impl on dispatch table or callsite failure
                  (alter-var-root impl-var (constantly old-impl))
                  (throw e))))))))
    (catch Throwable e
      (if *jit-silent-fallback?*
        (do (when (System/getProperty "raster.debug")
              (println "WARN: bytecode compile failed for" mangled-sym
                       ":" (.getMessage e)))
            nil)
        ;; Build a readable error message with source context
        (let [base-name (when-let [idx (clojure.string/index-of (str mangled-sym) "_m_")]
                          (subs (str mangled-sym) 0 idx))
              type-sig  (when-let [idx (clojure.string/index-of (str mangled-sym) "_m_")]
                          (subs (str mangled-sym) (+ idx 3)))
              cause-msg (loop [t e]
                          (if-let [c (.getCause t)]
                            (recur c)
                            (.getMessage t)))]
          (throw (ex-info (str "Bytecode compilation failed for `"
                               (or base-name mangled-sym) "`"
                               (when type-sig (str " [" (clojure.string/replace type-sig "_" ", ") "]"))
                               " in " (ns-name source-ns) ".\n"
                               "  Cause: " cause-msg "\n"
                               "  Params: " (pr-str (mapv vector params tags)) "\n"
                               "  Set raster.core/*jit-silent-fallback?* to true to fall back to deftype.")
                          {:var impl-var
                           :function (or base-name mangled-sym)
                           :ns (ns-name source-ns)
                           :params params
                           :tags tags
                           :return-tag return-tag}
                          e)))))))

;; ================================================================
;; Walked body registry — avoids quoting large walked bodies into
;; macro expansions, which would hit Clojure's 64KB method limit when
;; the body has many .invk calls with metadata.
;; ================================================================

(defonce ^:private walked-body-registry
  (atom {}))

(clojure.core/defn walked-body-for
  "Retrieve a stored walked body by mangled symbol key."
  [key]
  (get @walked-body-registry key))

(clojure.core/defn ^:no-doc ensure-walked-body!
  "Lazily walk a deftm var's source body with TC. Returns walked body.
  Julia model: walking is deferred to first use. If the var already has
  a walked body, returns it immediately. Otherwise walks from source body
  with TC binding tags and caches the result on the var metadata.
  Thread-safe via double-checked locking per var."
  [v]
  (or (seq (::deftm-walked-body (meta v)))
      (locking v
        (or (seq (::deftm-walked-body (meta v)))
            (when-let [src (::deftm-source-body (meta v))]
              (let [m (meta v)
                    walked (try (jit-walk-with-tc (vec src)
                                                  (::deftm-params m)
                                                  (::deftm-tags m)
                                                  (::deftm-annotations m)
                                                  (try (the-ns (::deftm-source-ns m))
                                                       (catch Exception _ *ns*)))
                                (catch Throwable _
                                  ;; Fallback: walk without TC
                                  (let [te (build-walker-type-env (::deftm-params m) (::deftm-annotations m))]
                                    (mapv #(walker/walk-body % {:type-env te :source-ns *ns*}) (vec src)))))]
                (alter-meta! v assoc ::deftm-walked-body (vec walked))
                (vec walked)))))))

;; ================================================================
;; Public API: defn — Clojure defn with walked body for compiler inlining
;; ================================================================

(defmacro defn*
  "Like clojure.core/defn but stores walked body for compile-aot inlining.
  The body is walked at definition time and stored as ::deftm-walked-body.
  At runtime this is a regular Clojure function — no typed dispatch, no
  overhead. But the compiler can see through it and inline it into
  compiled call chains.

  Exported as `defn` — use (:refer-clojure :exclude [defn]) in ns form."
  [name & fdecl]
  (let [;; Parse optional docstring
        [docstring fdecl] (if (string? (first fdecl))
                            [(first fdecl) (rest fdecl)]
                            [nil fdecl])
        ;; Parse optional attr-map
        [attr-map fdecl] (if (map? (first fdecl))
                           [(first fdecl) (rest fdecl)]
                           [nil fdecl])
        ;; Multi-arity? If first element is a list, it's multi-arity
        multi-arity? (and (seq fdecl) (list? (first fdecl)))
        ;; For single arity: walk body for inlining
        ;; For multi-arity: skip walking (too complex to pick which arity)
        [params body tags clean-params walked]
        (if multi-arity?
          [nil nil nil nil nil]
          (let [params (first fdecl)
                body (rest fdecl)
                tags (mapv (fn [p]
                             (if (symbol? p)
                               (or (:tag (meta p)) 'Object)
                               'Object))
                           params)
                clean-params (mapv #(if (symbol? %) (with-meta % nil) %) params)
                flat-env (into {} (map (fn [p t] [(with-meta p nil) t]) params tags))
                type-env (reduce-kv (fn [m s t] (assoc m s {:tag t})) {} flat-env)
                ctx (walker/make-ctx {:type-env type-env :source-ns *ns*})
                walked (try (walker/walk-forms (vec body) ctx)
                            (catch Exception _e nil))]
            [params body tags clean-params walked]))]
    ;; Build the defn form + metadata attachment
    (let [defn-form (cond
                      (and docstring attr-map) `(clojure.core/defn ~name ~docstring ~attr-map ~@fdecl)
                      docstring               `(clojure.core/defn ~name ~docstring ~@fdecl)
                      attr-map                `(clojure.core/defn ~name ~attr-map ~@fdecl)
                      :else                   `(clojure.core/defn ~name ~@fdecl))]
      (if walked
        (let [reg-key (str *ns* "/" name)]
          ;; Store walked body in registry — avoids quoting large walked bodies
          ;; as literals, which hits Clojure's 64KB method limit when the body
          ;; has many .invk calls with metadata (same pattern as deftm).
          (swap! walked-body-registry assoc reg-key
                 {:walked-body (vec walked)
                  :source-body (vec body)})
          `(do ~defn-form
               (let [reg# (walked-body-for ~reg-key)]
                 (alter-meta! (var ~name) assoc
                              ::deftm true
                              ::deftm-walked-body (:walked-body reg#)
                              ::deftm-raw-body (:source-body reg#)
                              ::deftm-source-ns '~(symbol (str *ns*))
                              ::deftm-params '~(vec clean-params)
                              ::deftm-tags '~tags))
               (var ~name)))
        ;; Multi-arity or walk failed — store only params/tags
        (if clean-params
          `(do ~defn-form
               (alter-meta! (var ~name) assoc
                            ::deftm-source-ns '~(symbol (str *ns*))
                            ::deftm-params '~(vec clean-params)
                            ::deftm-tags '~tags)
               (var ~name))
          defn-form)))))

;; Export as `defn` — consumers use (:refer-clojure :exclude [defn])
(def ^{:macro true
       :doc "Like clojure.core/defn but stores walked body for compiler inlining.
  Use (:refer-clojure :exclude [defn]) to import this instead of clojure.core/defn."}
  defn #'defn*)

;; ================================================================
;; Public API: deftm macro
;; ================================================================

(defmacro deftm
  "Define a typed method with Julia-style multiple dispatch.

  Uses typedclojure-compatible :- inline annotations:

    (deftm foo [x :- Long] :- Long (* x x))
    (deftm foo [x :- String] (count x))

  Annotations may equivalently be given as metadata (the Rich-Hickey /
  TypedClojure style). A param type goes on the param symbol; the return
  type goes on the arg vector, before it (mirroring Clojure's ^long [x]):

    (deftm foo ^{:- Long} [x :- Long] (* x x))           ; return via metadata
    (deftm foo [^{:- Long} x] :- Long (* x x))           ; param via metadata
    (deftm foo [^{:typed.clojure/type Long} x] ...)       ; TC's fully-qualified key

  Inline and metadata forms may be freely mixed; inline wins if both are
  present on the same position. The metadata annotation flows through the
  identical machinery, so (Array T), (Fn [..] R), (Param T) etc. all work.

  Type inference is automatic:
  - Record field types inferred via typedclojure (use :field keyword access)
  - Let bindings auto-hinted (no manual ^doubles needed)
  - Calls to other deftm functions devirtualized at compile time

  Also supports legacy ^Tag (:tag) metadata on parameters."
  {:arglists '([name [params...] :- RetType & body]
               [name docstring [params...] & body])}
  [fn-name & args]
  (let [;; Auto-qualify: if fn-name is unqualified but resolves to a generic function
        ;; var from another namespace (via :refer), use the qualified name so that
        ;; deftm extends the existing dispatch table instead of shadowing it.
        ;; This is the Julia semantics: `+(a::MyType, b::MyType) = ...` extends Base.+
        fn-name (if (and (not (qualified-symbol? fn-name))
                         (let [resolved (ns-resolve *ns* fn-name)]
                           (and resolved
                                (not= (the-ns (symbol (str (:ns (meta resolved)))))
                                      *ns*)
                                (:raster.core/generic-function (meta resolved)))))
                  (let [v (ns-resolve *ns* fn-name)]
                    (symbol (str (:ns (meta v))) (str (:name (meta v)))))
                  fn-name)
        ;; Parse optional docstring
        [docstring rest-args] (if (string? (first args))
                                [(first args) (rest args)]
                                [nil args])
        ;; Detect parametric: (deftm name (All [T] [params...] :- Ret body))
        ;; The All form wraps the entire signature + body
        parametric? (and (seq? (first rest-args)) (= 'All (ffirst rest-args)))
        ;; Detect tree-typed annotations in the param vector — these signal
        ;; that this deftm uses the tree pipeline (compile-time flattening +
        ;; structured-arg wrapper). Recognized forms:
        ;;   (HMap ...)       — bare HMap annotation
        ;;   (HVec [...])     — bare HVec annotation
        ;;   (Params <inner>) — back-compat alias; <inner> must be HMap/HVec
        ;; Forward to raster.params/defmodel which owns that machinery. The
        ;; user namespace must require raster.params for the resolution to
        ;; succeed (lazy via requiring-resolve at expansion time below).
        tree-head? #{'Params 'HMap 'HVec}
        params-arg? (and (not parametric?)
                         (vector? (first rest-args))
                         (some (fn [item]
                                 (and (sequential? item)
                                      (tree-head? (first item))))
                               (first rest-args)))]
    (cond
      params-arg?
      `(raster.params/defmodel ~fn-name ~@(if docstring (cons docstring rest-args) rest-args))
      parametric?
      ;; === Parametric deftm: register template + generate concrete double ===
      (do
        ;; Guard: body forms outside (All [T] ...) are silently dropped.
        ;; This catches the common mistake: (deftm f (All [T] [x :- T] :- T) body)
        ;; where body is OUTSIDE the All form. It should be:
        ;; (deftm f (All [T] [x :- T] :- T body))
        (when (> (count rest-args) 1)
          (throw (ex-info (str "deftm " fn-name ": body forms found outside (All [T] ...) form. "
                               "The body must be INSIDE the (All [T] [params...] :- Ret body) form, "
                               "not after it.")
                          {:fn-name fn-name :extra-forms (rest rest-args)})))
        (let [[_All type-vars & all-rest] (first rest-args)
              param-vec (first all-rest)
              after-params (rest all-rest)
              [ret-type body] (if (= :- (first after-params))
                                [(second after-params) (nnext after-params)]
                                [(types/meta-type-annotation param-vec) after-params])
              {:keys [params annotations]} (types/parse-typed-params param-vec)
            ;; Build annotation forms preserving type variables
              ann-forms (mapv (fn [ann] (or ann 'Object)) annotations)
              qname (if (qualified-symbol? fn-name) fn-name
                        (symbol (str *ns*) (str fn-name)))
            ;; Substitute T for the default concrete specialization.
            ;; Inside (Array T) → (Array double) (lowercase, for JVM array types).
            ;; Bare T → Double (boxed) in annotations, double (primitive cast) in body.
            ;; The annotation system handles Double→double lowering for ≤4 params.
              substitute-ann (fn subst [form]
                               (cond
                                 (and (symbol? form) (contains? (set type-vars) form))
                                 'Double  ;; bare T → Double (boxed, for annotations)
                                 (and (sequential? form) (= 'Array (first form))
                                      (contains? (set type-vars) (second form)))
                                 (list 'Array 'double)  ;; (Array T) → (Array double)
                                 (vector? form) (mapv subst form)
                                 (sequential? form) (apply list (map subst form))
                                 :else form))
              substitute-body (fn subst [form]
                                (cond
                                  (and (symbol? form) (contains? (set type-vars) form))
                                  'double  ;; bare T → double (primitive cast fn in body)
                                  (and (sequential? form) (= 'Array (first form))
                                       (contains? (set type-vars) (second form)))
                                  (list 'Array 'double)
                                  (vector? form) (mapv subst form)
                                  (sequential? form) (apply list (map subst form))
                                  :else form))
              concrete-params (mapv substitute-ann param-vec)
              concrete-ret (when ret-type (substitute-ann ret-type))
              concrete-body (map substitute-body body)]
          `(do
             (dispatch/ensure-generic! *ns* '~(symbol (name fn-name)))
             (dispatch/register-parametric!
              '~qname
              '~(vec type-vars)
              '~ann-forms
              '~(or ret-type 'Object)
              '~(vec params)
              '~(if (= 1 (count body)) (first body) (cons 'do body))
              *ns*)
           ;; Generate concrete double specialization so the dispatch table
           ;; has a concrete overload for AD inline expansion and compilation
             (deftm ~fn-name ~concrete-params
               ~@(when concrete-ret [:- concrete-ret])
               ~@concrete-body))))
      :else
      ;; === Normal deftm ===
      (let [param-vec (first rest-args)
            _ (assert (vector? param-vec)
                      (str "deftm requires a parameter vector, got: " (pr-str (first rest-args))))
        ;; Shared preparation: parse, validate, walk (deferred for typed deftm)
            prep (prepare-typed-body param-vec (rest rest-args)
                                     {:fn-label (str "deftm " fn-name)
                                      :defer-walk? true})
            {:keys [params annotations tags ret-type ret-tag body
                    hinted-params walked-body walked-body-typed has-fn-params?
                    generate-typed? typed-iface-name prim-hinted-params]} prep
            simple-name (if (qualified-symbol? fn-name) (symbol (name fn-name)) fn-name)
            mangled (types/mangle simple-name tags)
        ;; Add prim return hint to prim-hinted-params metadata
            prim-hinted-params (when (and prim-hinted-params ret-tag)
                                 (vary-meta prim-hinted-params assoc :tag
                                            (types/annotation->prim-hint ret-type)))
            prim-hinted-params (or prim-hinted-params
                                   (when generate-typed?
                                     (types/build-prim-hinted-params params annotations)))
            mangled-impl-var (when generate-typed?
                               (symbol (str mangled "-impl")))
        ;; Collect (Fn ...) param indices and their expected interface names
            fn-param-checks (when generate-typed?
                              (keep-indexed
                               (fn [i ann]
                                 (when-let [info (when ann (types/annotation->fn-info ann))]
                                   {:index i :iface-name (:iface-name info)}))
                               annotations))]
    ;; Store walked bodies in the registry at macro-expansion time.
    ;; This avoids quoting large walked bodies into the compiled output,
    ;; which would hit Clojure's 64KB method limit for functions with
    ;; many expanded par forms (e.g. vern7-step! with 10 broadcast stages).
        (swap! walked-body-registry assoc (str mangled)
               {:walked-body (when walked-body (vec walked-body))
                :walked-body-typed (when walked-body-typed (vec walked-body-typed))
                :source-body (vec body)})
        `(do
       ;; IFn backward-compat path (boxed). When the bytecode JIT is enabled,
       ;; we only need a stub here — the real code is compiled by our bytecode
       ;; backend (with method splitting) on first call. The walked body is
       ;; stored in the registry for the JIT to retrieve.
       ;; When JIT is disabled, we still need the Clojure-compiled defn as
       ;; the actual implementation.
           ~(if generate-typed?
          ;; Typed path: stub defn triggers JIT on first boxed call, then
          ;; delegates to the bytecode-compiled -impl. No walked body in
          ;; Clojure compilation — avoids 64KB method limit entirely.
              (let [args-sym (gensym "args__")
                    impl-var-sym (symbol (str (ns-name *ns*)) (str mangled-impl-var))]
                `(defn ~mangled ~@(when docstring [docstring])
                   [~'& ~args-sym]
                   (let [iv# (resolve '~impl-var-sym)]
                     (when (::jit-spec (meta iv#))
                       (locking iv#
                         (when-let [spec# (::jit-spec (meta iv#))]
                           (do-bytecode-upgrade!
                            iv# (:mangled-sym spec#)
                            (:params spec#) (:tags spec#) (:return-tag spec#)
                            (:walked-body spec#) (the-ns (:source-ns-name spec#))
                            (:typed-iface-name spec#)
                            :annotations (:annotations spec#)
                            :source-body (:source-body spec#))
                           (alter-meta! iv# dissoc ::jit-spec))))
                     (apply (deref iv#) ~args-sym))))
          ;; No typed interface (rare): must use Clojure-compiled defn
              `(defn ~mangled ~@(when docstring [docstring]) ~hinted-params ~@walked-body))
       ;; Populate metadata from walked-body-registry (stored at macro time).
       ;; This avoids quoting large walked bodies as literals, which would
       ;; hit Clojure's 64KB method limit for functions with many par forms.
           (let [reg# (walked-body-for ~(str mangled))]
             (alter-meta! (var ~mangled) assoc
                          ::deftm true
                          ::deftm-params '~params
                          ::deftm-tags '~tags
                          ::deftm-annotations '~(vec annotations)
                          ::deftm-source-ns '~(symbol (str *ns*))
                          ::deftm-source-body (:source-body reg#)
                          ::deftm-walked-body (:walked-body reg#)
                          ::deftm-walked-body-typed (:walked-body-typed reg#)))
           ~(when ret-tag
              `(alter-meta! (var ~mangled) assoc ::return-tag '~ret-tag))
           ~(when (seq fn-param-checks)
              `(alter-meta! (var ~mangled) assoc ::has-fn-params true))
           ~@(when generate-typed?
           ;; Minimal deftype stub: implements the typed interface for .invk
           ;; and IFn for boxed calls. On first call, triggers lazy JIT which
           ;; replaces this with an optimized bytecode-compiled class.
           ;; The .invk method delegates to the boxed defn (walked body) — only
           ;; runs until JIT kicks in, typically on the very first call.
               (let [this-sym (gensym "this__")
                     invk-sym 'invk  ;; returns Object (interface contract)
                     mangled-impl-type (symbol (str mangled "_Impl"))
                     impl-var-sym (symbol (str (ns-name *ns*)) (str mangled-impl-var))
                 ;; JIT trigger: check ::jit-spec, compile, swap
                     jit-trigger `(when-let [iv# (resolve '~impl-var-sym)]
                                    (when (::jit-spec (meta iv#))
                                      (locking iv#
                                        (when-let [spec# (::jit-spec (meta iv#))]
                                          (do-bytecode-upgrade!
                                           iv# (:mangled-sym spec#)
                                           (:params spec#) (:tags spec#) (:return-tag spec#)
                                           (:walked-body spec#) (the-ns (:source-ns-name spec#))
                                           (:typed-iface-name spec#)
                                           :annotations (:annotations spec#)
                                           :source-body (:source-body spec#))
                                          (alter-meta! iv# dissoc ::jit-spec)))))]
                 [`(import ~typed-iface-name)
              ;; Minimal deftype: .invk triggers JIT then delegates to boxed defn.
              ;; The defn stub also triggers JIT, so double-trigger is safe (locking).
                  `(deftype ~mangled-impl-type []
                     ~typed-iface-name
                     (~invk-sym [~this-sym ~@prim-hinted-params]
                       ~jit-trigger
                       (~mangled ~@prim-hinted-params)))
                  `(def ~(vary-meta mangled-impl-var assoc :tag typed-iface-name)
                     (new ~mangled-impl-type))
              ;; Mark for JIT compilation — walked body retrieved from registry
                  `(let [reg# (walked-body-for ~(str mangled))]
                     (when *compile-deftm-bytecode?*
                       (alter-meta! (var ~mangled-impl-var) assoc
                                    ::jit-spec {:mangled-sym '~mangled
                                                :params '~params
                                                :tags '~tags
                                                :annotations '~annotations
                                                :return-tag '~(or ret-tag 'Object)
                                                :source-body (:source-body reg#)
                                                :walked-body (or (:walked-body-typed reg#) (:walked-body reg#))
                                                :source-ns-name '~(ns-name *ns*)
                                                :typed-iface-name '~typed-iface-name})))]))
       ;; Register: typed-target-fn triggers lazy JIT on first call,
       ;; then routes through the -impl (which after JIT implements IFn).
       ;; Julia model: first dispatch call triggers compilation.
           (dispatch/register-method! ~mangled '~fn-name '~tags *ns*
                                      ~(if generate-typed? mangled-impl-var nil)
                                      ~(if generate-typed? `'~typed-iface-name nil)
                                      ~(if generate-typed?
                                         `(fn [~'& ~'args]
                                        ;; Trigger lazy JIT if not yet compiled
                                            (let [iv# (var ~mangled-impl-var)]
                                              (when (::jit-spec (meta iv#))
                                                (locking iv#
                                                  (when-let [spec# (::jit-spec (meta iv#))]
                                                    (do-bytecode-upgrade!
                                                     iv# (:mangled-sym spec#)
                                                     (:params spec#) (:tags spec#) (:return-tag spec#)
                                                     (:walked-body spec#) (the-ns (:source-ns-name spec#))
                                                     (:typed-iface-name spec#)
                                                     :annotations (:annotations spec#)
                                                     :source-body (:source-body spec#))
                                                    (alter-meta! iv# dissoc ::jit-spec)))))
                                        ;; Route through -impl (after JIT: compiled class with IFn)
                                            (apply (deref (var ~mangled-impl-var)) ~'args))
                                         nil)
                                      ~(meta fn-name)))))))

;; ================================================================
;; Public API: ftm macro (typed anonymous function)
;; ================================================================

(defmacro ftm
  "Typed anonymous function via reify on a shared interface.
  Supports 5+ primitive args without boxing. Backward-compatible as IFn.

  Usage:
    (def f (ftm [du :- (Array double), u :- (Array double), t :- Double]
             (aset du 0 (* 10.0 (aget u 0)))))

    ;; Call via typed interface (primitive fast-path):
    (.invk f du u t)

    ;; Call via IFn (backward compat):
    (f du u t)

  Supports optional return type:
    (ftm [x :- Double, y :- Double] :- Double (+ x y))"
  [param-vec & body-args]
  (let [;; Extract walker-injected source body if present.
        ;; The walker inserts :raster.walker/source-body [body] after the :- RetType.
        ;; body-args is: (:- RetType :raster.walker/source-body [body] walked-body...)
        ;; or: (:raster.walker/source-body [body] walked-body...) when no return type.
        ;; or just: (body...) when not from walker.
        [walker-source-body body-args]
        (let [;; Parse past optional :- RetType
              [after-ret rest-args] (if (= :- (first body-args))
                                      [nil (nnext body-args)]  ;; skip :- and RetType
                                      [nil body-args])]
          (if (= :raster.walker/source-body (first rest-args))
            [(second rest-args)
             ;; Reconstruct body-args: put back :- RetType + remaining body
             (if (= :- (first body-args))
               (list* :- (second body-args) (nnext rest-args))
               (nnext rest-args))]
            [nil body-args]))
        ;; Shared preparation: parse, validate, walk (TC eager for ftm)
        prep (prepare-typed-body param-vec body-args
                                 {:fn-label "ftm"})
        {:keys [params annotations tags ret-type ret-tag body
                walked-body prim-hinted-params typed-iface-name]} prep
        ;; ftm-specific: interface types for reify (primitives stay, others → Object)
        iface-param-tags (mapv types/tag->iface-type tags)
        iface-info (types/ensure-fn-interface! iface-param-tags)
        iface-name (:iface-name iface-info)
        prim-hinted-params (or prim-hinted-params
                               (types/build-prim-hinted-params params annotations))
        ;; Build IFn invoke arity — uses un-walked body for dispatch compatibility
        invoke-params (mapv (fn [p] (gensym (str p "__"))) params)
        invoke-let-bindings (vec (interleave params invoke-params))
        this-sym (gensym "this__")
        invk-sym 'invk]  ;; returns Object (interface contract)
    `(do
       (import ~iface-name)
       (vary-meta
        (reify
          ~iface-name  ;; extends raster.compiler.core.types.TypedFn (marker for function algebra dispatch)
          (~invk-sym [~this-sym ~@prim-hinted-params] ~@walked-body)
          clojure.lang.IFn
          (~'invoke [~this-sym ~@invoke-params]
            (let ~invoke-let-bindings
              ~@(or walker-source-body body))))
        merge {::deftm true
               ::deftm-walked-body '~(vec walked-body)
                ;; Source body for AD: walker's pre-walk source (uses raster.numeric
                ;; dispatch which handles Dual) when ftm is inside a walked deftm body.
                ;; Falls back to the raw body when ftm is used standalone.
               ::deftm-source-body '~(vec (or walker-source-body body))
               ::deftm-params '~(vec params)
               ::deftm-tags '~tags}))))

;; ================================================================
;; Public API: defvalue macro
;; ================================================================

(defmacro defabstract
  "Define an abstract type (Java interface) for dispatch hierarchy.

  Like Julia's `abstract type Foo <: Bar end`, creates a type that
  can be used in deftm dispatch but cannot be instantiated directly.

  Usage:
    (defabstract Algorithm)
    (defabstract AdaptiveAlgorithm :extends Algorithm)
    (defabstract ImplicitAlgorithm :extends [Algorithm Stiff])
    (defabstract Group :extends Monoid)"
  [name & body]
  (let [[doc opts] (if (string? (first body))
                     [(first body) (next body)]
                     [nil body])
        {:keys [extends]} (if (seq opts) (apply hash-map opts) {})
        ns-str (str/replace (str (ns-name *ns*)) "-" "_")
        full-name (symbol (str ns-str "." name))
        extend-spec (cond
                      (nil? extends) []
                      (symbol? extends) [extends]
                      (vector? extends) extends
                      :else (throw (ex-info "defabstract :extends must be a symbol or vector of symbols"
                                            {:extends extends})))
        resolve-parent (fn [s]
                         (if-let [cls (types/resolve-type-ref->class s)]
                           cls
                           (throw (ex-info (str "Cannot resolve abstract type: " s)
                                           {:name s}))))]
    `(let []
       (gen-interface :name ~full-name :extends ~(mapv resolve-parent extend-spec))
       (import ~full-name)
       ~@(when doc
           [`(types/register-abstract-doc! ~(str full-name) ~doc)]))))

;; Registry of parametric defvalue templates. {name -> {:type-vars [...] :fields [...] :opts [...] :source-ns ns}}
(defonce parametric-value-registry (atom {}))

;; Cache of parametric value type classes. Keyed by fully-qualified class name string.
;; Ensures a single Class object is used across make-dual and parametric-specialize!,
;; avoiding classloader mismatches from separate eval calls.
(defonce parametric-class-cache (atom {}))

(defn get-or-create-parametric-class!
  "Get or create a parametric value type class. Returns the Class.
   `fqn` is the fully-qualified class name (e.g. \"raster.ad.forward.Dual__Sym\").
   `create-fn` is called with no args to create the class (via eval) if it doesn't exist.
   The class is cached to ensure all code uses the same Class object."
  [fqn create-fn]
  (or (get @parametric-class-cache fqn)
      (do (create-fn)
          (let [cl (.getContextClassLoader (Thread/currentThread))
                cls (Class/forName fqn true cl)]
            (swap! parametric-class-cache assoc fqn cls)
            cls))))

;; Marker interface for parameter container value-classes (the composable unit
;; for nested model params). Only fields whose type :implements Parameters get
;; typed-value-class field treatment in defvalue — so GA/Dual/etc. value classes
;; (GradedAlgebra, …) are unaffected.
(defabstract Parameters)

(defn- implements-parameters?
  [^Class cls]
  (try (.isAssignableFrom (Class/forName "raster.core.Parameters") cls)
       (catch Throwable _ false)))

(defn- value-class-field-tag
  "If a defvalue field annotation names a (loaded) value-class that implements
  Parameters — bare `Name` or `(Array Name)` — return a JVM descriptor string so
  the generated value class has a TYPED field (a nested Parameters value-class or
  an array of them) instead of Object. Other types (boxed primitives, GA/Dual
  value classes, unknown tags) return nil (caller falls back to annotation->tag)."
  [ann]
  (let [boxed '#{Double Float Long Integer Short Byte Character Boolean
                 java.lang.Double java.lang.Float java.lang.Long java.lang.Integer
                 java.lang.Short java.lang.Byte java.lang.Character java.lang.Boolean}
        vc   (fn [sym]
               (when (and (symbol? sym) (not (boxed sym)))
                 (let [r (try (resolve sym) (catch Exception _ nil))]
                   (when (and (class? r) (not (.isPrimitive ^Class r))
                              (implements-parameters? r)) r))))
        ;; JVM field descriptor string (embeddable in the macro's output; the
        ;; generator's tag->desc resolves a descriptor string to a ClassDesc).
        desc (fn [^Class c] (str "L" (.replace (.getName c) "." "/") ";"))]
    (cond
      (and (sequential? ann) (= 'Array (first ann)) (vc (second ann)))
      (str "[" (desc (vc (second ann))))
      (vc ann)
      (desc (vc ann))
      :else nil)))

(defmacro defvalue
  "Define a value type with :- Type annotations. On Valhalla JDK, generates
  real value class bytecode for stack allocation and array flattening. Falls
  back to defrecord otherwise.

  Supports :implements to participate in abstract type hierarchies:
    (defvalue Vec3 [x :- Double, y :- Double, z :- Double] :implements AbstractVector)

  Supports parametric types:
    (defvalue SparseVector (All [T]) [indices :- (Array int), values :- (Array T), length :- Long])
    ;; Registers template; base name is the double specialization."
  [name fields-or-all & opts]
  ;; Detect parametric: (defvalue Name (All [T]) [fields...] opts...)
  (if (and (seq? fields-or-all) (= 'All (first fields-or-all)))
    (let [type-vars (second fields-or-all)
          fields (first opts)
          rest-opts (rest opts)
          ;; Substitute T -> double for the default concrete specialization
          bindings (zipmap type-vars (repeat 'double))
          substitute (fn subst [form]
                       (cond
                         (and (symbol? form) (contains? bindings form))
                         (get bindings form)
                         (vector? form) (mapv subst form)
                         (sequential? form) (apply list (map subst form))
                         :else form))
          concrete-fields (mapv substitute fields)]
      `(do
         ;; Register the parametric template for JIT specialization of non-double types
         (swap! parametric-value-registry assoc '~name
                {:type-vars '~type-vars
                 :fields '~fields
                 :opts '~(vec rest-opts)
                 :source-ns *ns*})
         ;; Register type vars in dispatch for parametric unification
         ;; (enables Dual__Sym matching Dual annotation → T=Sym)
         (dispatch/register-parametric-value-type! '~name '~type-vars)
         ;; Generate the default double specialization under the base name
         (defvalue ~name ~concrete-fields ~@rest-opts)))
    ;; Concrete defvalue with :- Type annotations
    (let [fields fields-or-all
          {:keys [params annotations]} (types/parse-typed-params fields)
          ;; Build field-types map: {name-string -> dispatch-tag}
          field-types (into {}
                            (for [[p ann] (map vector params annotations)
                                  :when ann
                                  :let [tag (types/annotation->tag ann p)]
                                  :when (not= tag 'Object)]
                              [(clojure.core/name p) tag]))
          ;; Build field specs for Valhalla value class generation. Value-class
          ;; fields (nested Parameters) resolve to a ClassDesc tag so the field
          ;; is typed (not Object).
          field-specs (mapv (fn [p ann]
                              {:name (clojure.core/name p)
                               :tag (or (value-class-field-tag ann)
                                        (if ann (types/annotation->tag ann p) 'Object))})
                            params annotations)
          ;; A value class with any value-class-typed field is not SoA-eligible
          ;; (SoA needs primitive-array fields only).
          has-value-class-field? (boolean (some value-class-field-tag annotations))
          ;; Build hinted params for defrecord fallback (Double→^double, etc.)
          hinted (types/build-record-hinted-params params annotations)
          ;; Parse :implements from opts
          [kv-opts rest-specs] (loop [items (seq opts) kvs {} specs []]
                                 (if (and items (keyword? (first items)))
                                   (recur (nnext items)
                                          (assoc kvs (first items) (second items))
                                          specs)
                                   [(apply hash-map (mapcat identity kvs))
                                    (vec items)]))
          implements (get kv-opts :implements)
          impls (cond
                  (nil? implements) []
                  (symbol? implements) [implements]
                  (vector? implements) implements
                  :else (throw (ex-info "defvalue :implements must be a symbol or vector of symbols"
                                        {:implements implements})))
          ;; SoA companion generation
          soa-eligible (and (not has-value-class-field?)
                            (inf/soa-eligible? field-types))
          flat-field-info (when soa-eligible
                            (vec (mapcat (fn [[p ann]]
                                           (when ann
                                             (let [tag (types/annotation->tag ann p)]
                                               (inf/flatten-one-field (clojure.core/name p) tag [(clojure.core/name p)]))))
                                         (map vector params annotations))))
          orig-field-pairs (vec (for [[p ann] (map vector params annotations)
                                      :when ann
                                      :let [tag (types/annotation->tag ann p)]]
                                  [(clojure.core/name p) tag]))
          soa-name (symbol (str name "SoA"))
          kebab-name (let [raw (str/lower-case
                                (str/replace (clojure.core/name name) #"([A-Z])" "-$1"))]
                       (if (str/starts-with? raw "-") (subs raw 1) raw))
          make-fn-sym   (symbol (str "make-" kebab-name "-soa"))
          aget-fn-sym   (symbol (str "aget-" kebab-name))
          aset-fn-sym   (symbol (str "aset-" kebab-name "!"))
          soa-fields (when soa-eligible
                       (vec (mapcat (fn [{:keys [name array-tag]}]
                                      [(symbol name) :- (list 'Array (get {'doubles 'double 'floats 'float
                                                                           'longs 'long 'ints 'int} array-tag array-tag))])
                                    flat-field-info)))
          soa-make-body (when soa-eligible
                          (let [array-ctor {'doubles 'double-array
                                            'floats  'float-array
                                            'longs   'long-array
                                            'ints    'int-array}]
                            `(~(symbol (str "->" soa-name))
                              ~@(map (fn [{:keys [array-tag]}]
                                       `(~(get array-ctor array-tag) ~'n))
                                     flat-field-info))))
          soa-aget-body (when soa-eligible
                          `(~(symbol (str "->" name))
                            ~@(map (fn [[fname ftag]]
                                     (inf/build-aget-expr ftag fname 'soa 'i))
                                   orig-field-pairs)))
          soa-aset-stmts (when soa-eligible
                           (map (fn [{:keys [name get-path]}]
                                  (let [arr-field (symbol (str "." name))
                                        val-accessor (reduce (fn [form fname]
                                                               `(~(symbol (str "." fname)) ~form))
                                                             'v
                                                             get-path)]
                                    `(aset (~arr-field ~'soa) ~'i ~val-accessor)))
                                flat-field-info))]
      (if valhalla/valhalla-available?
        ;; Generate real Valhalla value class bytecode
        (let [ns-str (str/replace (str (ns-name *ns*)) "-" "_")
              class-name (str ns-str "." name)
              iface-descs (mapv (fn [s]
                                  (let [cls (resolve s)]
                                    (when-not cls
                                      (throw (ex-info (str "Cannot resolve interface: " s) {:name s})))
                                    (let [cn (.getName ^Class cls)
                                          di (.lastIndexOf cn (int \.))
                                          pkg (subs cn 0 di)
                                          sn (subs cn (inc di))]
                                      [pkg sn])))
                                impls)]
          (let [factory-sym (symbol (str "->" name))
                factory-params (vec params)
                ;; Build TC annotation for the constructor
                tc-param-types (mapv (fn [ann]
                                       (if ann
                                         (let [tag (types/annotation->tag ann nil)]
                                           (get dispatch/primitive-tc-ann tag
                                                (or (when-let [^Class cls (get @types/tag-class-registry tag)]
                                                      (symbol (.getName cls)))
                                                    'clojure.core.typed/Any)))
                                         'clojure.core.typed/Any))
                                     annotations)
                tc-ann-form (vec (concat tc-param-types [:-> 'clojure.core.typed/Any]))]
            `(do
               (valhalla/load-value-class! ~class-name '~field-specs
                                           :interfaces ~(if (seq iface-descs)
                                                          `(mapv (fn [[p# n#]] (java.lang.constant.ClassDesc/of p# n#)) '~iface-descs)
                                                          []))
               (import '~(symbol class-name))
               (defn ~factory-sym ~factory-params
                 (new ~name ~@factory-params))
               (try (clojure.core.typed/-ann
                     ['~(ns-name *ns*) '~(symbol (str (ns-name *ns*)) (str factory-sym))
                      '~tc-ann-form {} nil nil])
                    (catch Exception _#))
               ~(when (seq field-types)
                  `(inf/register-field-types! '~name '~field-types))
               ~@(when soa-eligible
                   [`(defvalue ~soa-name ~soa-fields)
                    `(inf/register-soa! '~name '~soa-name '~(mapv #(dissoc % :get-path) flat-field-info))
                    `(defn ~make-fn-sym [~'n]
                       ~soa-make-body)
                    `(defn ~aget-fn-sym [~'soa ~'i]
                       ~soa-aget-body)
                    `(defn ~aset-fn-sym [~'soa ~'i ~'v]
                       (do ~@soa-aset-stmts))]))))
        ;; Fallback: defrecord
        (let [rest-specs (vec (remove #{:implements implements} rest-specs))
              factory-sym (symbol (str "->" name))
              tc-param-types (mapv (fn [ann]
                                     (if ann
                                       (let [tag (types/annotation->tag ann nil)]
                                         (get dispatch/primitive-tc-ann tag
                                              (or (when-let [^Class cls (get @types/tag-class-registry tag)]
                                                    (symbol (.getName cls)))
                                                  'clojure.core.typed/Any)))
                                       'clojure.core.typed/Any))
                                   annotations)
              tc-ann-form (vec (concat tc-param-types [:-> 'clojure.core.typed/Any]))]
          `(do
             ;; deftype (not defrecord) — a value type is a struct, not a Map.
             ;; This matches the Valhalla value-class path and avoids field/method
             ;; collisions (a field `size`/`count` would otherwise resolve to the
             ;; defrecord's java.util.Map .size()/.count). We regenerate the bits
             ;; the glue actually uses: ILookup (:field access), value
             ;; equals/hashCode (deftype defaults to identity), and print-method.
             (deftype ~name ~(vec hinted)
               ~@impls ~@rest-specs
               clojure.lang.ILookup
               (~'valAt [~'_ ~'k] (case ~'k ~@(mapcat (fn [p] [(keyword (clojure.core/name p)) p]) params) nil))
               (~'valAt [~'_ ~'k ~'nf] (case ~'k ~@(mapcat (fn [p] [(keyword (clojure.core/name p)) p]) params) ~'nf))
               Object
               (~'equals [~'_ ~'other]
                 (and (instance? ~name ~'other)
                      ~@(map (fn [p] `(clojure.core/= ~p (. ~'other ~(symbol (str "-" (clojure.core/name p)))))) params)))
               (~'hashCode [~'_] (clojure.core/hash [~@params])))
             (defmethod print-method ~name [~'v ~'w]
               (.write ^java.io.Writer ~'w
                       (str "#" '~name
                            (array-map ~@(mapcat (fn [p] [(keyword (clojure.core/name p))
                                                          `(. ~'v ~(symbol (str "-" (clojure.core/name p))))]) params)))))
             (try (clojure.core.typed/-ann
                   ['~(ns-name *ns*) '~(symbol (str (ns-name *ns*)) (str factory-sym))
                    '~tc-ann-form {} nil nil])
                  (catch Exception _#))
             ~(when (seq field-types)
                `(inf/register-field-types! '~name '~field-types))
             ~@(when soa-eligible
                 [`(defvalue ~soa-name ~soa-fields)
                  `(inf/register-soa! '~name '~soa-name '~(mapv #(dissoc % :get-path) flat-field-info))
                  `(defn ~make-fn-sym [~'n]
                     ~soa-make-body)
                  `(defn ~aget-fn-sym [~'soa ~'i]
                     ~soa-aget-body)
                  `(defn ~aset-fn-sym [~'soa ~'i ~'v]
                     (do ~@soa-aset-stmts))])))))))

(defmacro defval
  "Define a singleton value type (no fields).
  Like Julia's Val{x} pattern — zero-size type for dispatch.

  Usage:
    (defval InPlace)
    (defval OutOfPlace)
    (deftm make-cache [alg :- Euler, mode :- InPlace] ...)"
  [name & opts]
  `(defvalue ~name [] ~@opts))

;; ================================================================
;; Public API: Introspection
;; ================================================================

(defn methods-of
  "Returns the dispatch table for a generic function."
  [fn-name-or-var]
  (let [k (cond
            (var? fn-name-or-var)
            (let [m (meta fn-name-or-var)]
              (symbol (str (:ns m)) (str (:name m))))

            (qualified-symbol? fn-name-or-var)
            fn-name-or-var

            :else
            (throw (ex-info "methods-of requires a qualified symbol or var"
                            {:arg fn-name-or-var})))]
    (when-let [table-atom (get @dispatch/dispatch-tables k)]
      @table-atom)))

(defn ambiguities
  "Returns a seq of ambiguous method pairs for a generic function.
  Each pair is [{:tags [...] :specificity n} {:tags [...] :specificity n}].
  Empty seq means no ambiguities."
  [fn-name-or-var]
  (let [table (methods-of fn-name-or-var)]
    (when table
      (for [[_arity methods] table
            [i a] (map-indexed vector methods)
            b (subvec (vec methods) (inc i))
            :when (dispatch/methods-ambiguous? a b)]
        [(select-keys a [:tags :specificity])
         (select-keys b [:tags :specificity])]))))

(defn print-methods
  "Print all methods of a generic function in a readable format."
  [fn-name-or-var]
  (let [table (methods-of fn-name-or-var)
        fn-name (if (var? fn-name-or-var)
                  (let [m (meta fn-name-or-var)] (str (:ns m) "/" (:name m)))
                  (str fn-name-or-var))]
    (if (empty? table)
      (println "No methods defined for" fn-name)
      (doseq [[arity methods] (sort-by key table)]
        (doseq [m methods]
          (println (str "  " fn-name "("
                        (str/join ", " (map clojure.core/name (:tags m)))
                        ")  [specificity=" (:specificity m) "]")))))))

;; ================================================================
;; Public API: Method disambiguation
;; ================================================================

(defn invoke
  "Call a specific method by explicit type tags, bypassing dispatch.
  Useful when two methods are ambiguous and you want to select one explicitly.

  Usage:
    (invoke #'describe [FixedStepAlgorithm] (->RK4))
    (invoke #'+ [Long Long] 1 2)"
  [fn-var tags & args]
  (let [m (meta fn-var)
        table-atom (::dispatch-table m)
        _ (when-not table-atom
            (throw (ex-info "invoke requires a generic function var"
                            {:var fn-var})))
        arity (count tags)
        methods (get @table-atom arity)
        entry (some #(when (= (mapv symbol (map clojure.core/name (:tags %)))
                              (mapv symbol (map clojure.core/name tags)))
                       %)
                    methods)]
    (if entry
      (apply (:target-fn entry) args)
      (throw (ex-info (str "No method with tags " tags " for " (:name m))
                      {:name (:name m) :tags tags
                       :available (mapv :tags methods)})))))

(defn prefer-method!
  "Declare that method with `preferred-tags` should be preferred over
  `other-tags` when both match. Resolves ambiguity without defining
  a new method.

  Usage:
    (prefer-method! #'describe [FixedStepAlgorithm] [AdaptiveAlgorithm])"
  [fn-var preferred-tags other-tags]
  (let [table-atom (::dispatch-table (meta fn-var))
        _ (when-not table-atom
            (throw (ex-info "prefer-method! requires a generic function var"
                            {:var fn-var})))
        arity (count preferred-tags)]
    (swap! table-atom update arity
           (fn [methods]
             (let [pref-idx (some (fn [[i m]] (when (= (:tags m) preferred-tags) i))
                                  (map-indexed vector methods))
                   other-idx (some (fn [[i m]] (when (= (:tags m) other-tags) i))
                                   (map-indexed vector methods))]
               (if (and pref-idx other-idx (> pref-idx other-idx))
                 ;; Move preferred before other by bumping its specificity
                 (let [other-spec (:specificity (nth methods other-idx))
                       methods (update-in (vec methods) [pref-idx :specificity]
                                          (fn [s] (max s (inc other-spec))))]
                   (vec (sort-by :specificity > methods)))
                 methods))))
    ;; Rebuild dispatch fn
    (let [m (meta fn-var)
          reducible? (::reducible m)
          fn-name (symbol (clojure.core/name (:name m)))]
      (alter-var-root fn-var (constantly (dispatch/make-dispatch-fn fn-name table-atom reducible?))))
    fn-var))

(defn remove-method!
  "Remove a method with the given tags from a generic function.

  Usage:
    (remove-method! #'describe [Euler])"
  [fn-var tags]
  (let [table-atom (::dispatch-table (meta fn-var))
        _ (when-not table-atom
            (throw (ex-info "remove-method! requires a generic function var"
                            {:var fn-var})))
        arity (count tags)]
    (swap! table-atom update arity
           (fn [methods]
             (vec (remove #(= (:tags %) tags) methods))))
    ;; Rebuild dispatch fn
    (let [m (meta fn-var)
          reducible? (::reducible m)
          fn-name (symbol (clojure.core/name (:name m)))]
      (alter-var-root fn-var (constantly (dispatch/make-dispatch-fn fn-name table-atom reducible?))))
    fn-var))

(defn ^:no-doc specialize-fn!
  "Specialize a deftm function for a given element type.
  Delegates to raster.compiler.core.specialize."
  [fn-var opts]
  (specialize/specialize-fn! fn-var opts))

(defmacro specialize
  "Specialize a deftm function for a specific element type.
  Generates a devirtualized version where all raster.numeric dispatch
  calls are resolved to concrete mangled implementations.

  Usage:
    (specialize generic-rk4-step {:element-type Complex
                                   :fn-bindings {f lotka-volterra}})"
  [fn-name opts]
  (let [element-type (:element-type opts)
        fn-bindings (:fn-bindings opts)
        param-hints (:param-hints opts)
        ;; Convert fn-bindings values from symbols to quoted symbols
        fn-bindings-form (when fn-bindings
                           (into {} (map (fn [[k v]] [`'~k `'~v]) fn-bindings)))
        param-hints-form (when param-hints
                           (into {} (map (fn [[k v]] [`'~k `'~v]) param-hints)))]
    `(specialize-fn! (var ~fn-name)
                     {:element-type '~element-type
                      :fn-bindings ~fn-bindings-form
                      :param-hints ~param-hints-form
                      :ns-obj *ns*})))

;; ================================================================
;; Typed macro stubs for clj-kondo
;; ================================================================
;; These are never called at runtime — the walker expands them before
;; Clojure's macro-expander ever sees them.  They exist so that
;; clj-kondo can resolve the symbols when they appear in deftm bodies.

(defmacro broadcast
  "Functional element-wise map over arrays — valid only inside deftm bodies.
  Returns a new array. Output allocation is handled by the compiler.

  (broadcast [x y] (+ x y))        ;; element-wise add, returns new array
  (broadcast [x] (* 2.0 x))        ;; scalar-array multiply

  Expanded by the typed walker into par/map! with compiler-managed output."
  [_syms & _args]
  (throw (ex-info "broadcast is only valid inside deftm bodies" {})))

(defmacro reduce!
  "Array fold — valid only inside deftm bodies.
  Expanded by the typed walker; this stub exists for tooling only."
  [_acc-binding _syms & _args]
  (throw (ex-info "reduce! is only valid inside deftm bodies" {})))

(defmacro scan
  "Functional prefix scan over arrays — valid only inside deftm bodies.
  Returns a new array. Output allocation is handled by the compiler.
  Expanded by the typed walker."
  [_acc-binding _syms & _args]
  (throw (ex-info "scan is only valid inside deftm bodies" {})))

(defmacro stencil!
  "Array stencil loop — valid only inside deftm bodies.
  Expanded by the typed walker; this stub exists for tooling only."
  [_syms & _args]
  (throw (ex-info "stencil! is only valid inside deftm bodies" {})))

(defmacro muladd
  "Fused multiply-add — valid only inside deftm bodies.
  Expanded by the typed walker; this stub exists for tooling only."
  [& _args]
  (throw (ex-info "muladd is only valid inside deftm bodies" {})))

;; ================================================================
;; Parametric specialization callback
;; Breaks dispatch↔core cycle: dispatch detects the parametric match,
;; core generates the concrete deftm.
;; ================================================================

(defn- parametric-specialize!
  "Generate a concrete deftm specialization for a parametric template.
  Called by dispatch when a parametric function is invoked with a new type.

  Uses the bytecode compiler directly instead of Clojure eval. This ensures
  the invoke(Object) path is a bytecoded unboxing bridge (like manual deftm),
  not a Clojure-compiled fn with TC type hints that break array interop."
  [fn-name template bindings args]
  (let [{:keys [annotations ret-annotation params body source-ns]} template
        ;; Specialize parametric value types (defvalue with All [T])
        non-prim-tag (first (filter #(and (symbol? %)
                                          (not (#{'double 'float 'long 'int} %)))
                                    (vals bindings)))
        cache-specs
        (when non-prim-tag
          (into {} (keep (fn [[cname tmpl]]
                           (let [concrete-name (symbol (str cname "__" non-prim-tag))]
                             [cname concrete-name]))
                         @parametric-value-registry)))
        ;; Unified substitution: type vars + parametric type names + constructors
        subst (fn [sym]
                (cond
                  (contains? bindings sym) (get bindings sym)
                  (and cache-specs (contains? cache-specs sym))
                  (get cache-specs sym)
                  (and cache-specs
                       (.startsWith ^String (name sym) "->")
                       (contains? cache-specs (symbol (subs (name sym) 2))))
                  (symbol (str "->" (get cache-specs (symbol (subs (name sym) 2)))))
                  :else sym))
        ;; Substitute in annotations
        concrete-anns (mapv (fn [ann]
                              (let [a (dispatch/substitute-type-var ann bindings)]
                                (clojure.walk/postwalk
                                 #(if (symbol? %) (subst %) %) a)))
                            annotations)
        concrete-ret (clojure.walk/postwalk
                      #(if (symbol? %) (subst %) %)
                      (dispatch/substitute-type-var ret-annotation bindings))
        ;; Determine element type for literal casting (Julia-style)
        ;; When T=float, double literals like 0.0 should become (float 0.0)
        ;; so loop accumulators match the element type and dispatch resolves
        ;; to concrete float+float instead of Number+Number.
        elem-prim (get bindings (first (:type-vars template)))
        literal-cast-fn (case elem-prim
                          float  (fn [x]
                                   (if (and (number? x) (instance? Double (if (int? x) nil x)))
                                     (list 'float x)
                                     x))
                          int    (fn [x]
                                   (if (and (number? x) (instance? Double (if (int? x) nil x)))
                                     (list 'int x)
                                     x))
                          ;; double or unrecognized — no cast needed
                          identity)
        ;; Substitute in body: type vars + literal casting
        body-with-types (clojure.walk/postwalk
                         (fn [x]
                           (cond
                             (symbol? x) (subst x)
                             (number? x) (literal-cast-fn x)
                             :else x))
                         body)
        ;; Compute tags and mangled name
        tags (mapv #(types/annotation->tag %1 %2) concrete-anns params)
        ret-tag (or (types/annotation->tag concrete-ret nil) 'Object)
        mangled (types/mangle fn-name tags)
        mangled-sym (symbol (str mangled))
        fq-mangled (symbol (str (ns-name source-ns)) (str mangled))
        ;; Typed interface for zero-boxing invocation
        iface-info (types/ensure-fn-interface! tags)
        typed-iface-name (symbol (.getName ^Class (:iface-class iface-info)))
        ;; Run TC on the specialized body for binding type inference —
        ;; same as prepare-typed-body does for normal deftm
        tc-binding-tags (try
                          (let [body-vec (if (seq? body-with-types) [body-with-types] (vec body-with-types))]
                            (:binding-tags (inf/tc-analyze-deftm-body fn-name params concrete-anns body-vec source-ns)))
                          (catch Throwable e
                            (println (str "WARNING: TC analysis failed during parametric specialization of `"
                                          fn-name "`: " (.getMessage e)))
                            nil))
        ;; Walk the body — same as deftm macro does
        type-env (build-walker-type-env params concrete-anns)
        plain-type-env (reduce-kv (fn [m s rec] (assoc m s (dissoc rec :fn-info))) {} type-env)
        walk-opts (cond-> {:type-env plain-type-env :source-ns source-ns}
                    (seq tc-binding-tags) (assoc :tc-binding-tags tc-binding-tags))
        walked-body (vec (map #(walker/walk-body % walk-opts) (if (seq? body-with-types)
                                                                [body-with-types]
                                                                body-with-types)))
        ;; Collect defvalue forms for parametric value types
        body-str (str body)
        ann-str (str annotations)
        defvalue-forms (when (and cache-specs non-prim-tag)
                         (vec (keep (fn [[cname tmpl]]
                                      (when (or (.contains ^String body-str (str cname))
                                                (.contains ^String ann-str (str cname)))
                                        (let [concrete-name (get cache-specs cname)
                                              {:keys [type-vars fields opts]} tmpl
                                              sub-bindings (zipmap type-vars (repeat non-prim-tag))
                                              substituted-fields (mapv (fn [f]
                                                                         (if (and (sequential? f) (= 'Array (first f)))
                                                                           (clojure.walk/postwalk
                                                                            #(if (contains? sub-bindings %) (get sub-bindings %) %) f)
                                                                           (if (and (symbol? f) (contains? sub-bindings f))
                                                                             (get sub-bindings f) f)))
                                                                       fields)]
                                          (when concrete-name
                                            (list* 'raster.core/defvalue concrete-name substituted-fields opts)))))
                                    @parametric-value-registry)))]
    ;; Import classes into source-ns so the walker can resolve them.
    ;; Also register parametric value type classes in tag-class-registry
    ;; from the actual args — this ensures we use the exact class/classloader
    ;; that the runtime instances have, avoiding classloader mismatches.
    (doseq [arg args]
      (when (and arg (not (number? arg)) (not (.isArray (class arg))))
        (let [cls (class arg)
              simple (symbol (.getSimpleName cls))]
          (when (contains? @types/array-like-registry simple)
            (.importClass ^clojure.lang.Namespace source-ns cls))
          ;; Register parametric value type classes from actual args.
          ;; Also populate the class cache so all paths use the same Class.
          (when (.contains ^String (str simple) "__")
            (.importClass ^clojure.lang.Namespace source-ns cls)
            (swap! types/tag-class-registry assoc simple cls)
            (swap! parametric-class-cache assoc (.getName cls) cls))
          ;; Import field type classes from parametric value instances
          ;; (e.g., Dual__Sym's .v field is Sym — import Sym into source-ns)
          (doseq [^java.lang.reflect.Field field (.getDeclaredFields cls)]
            (let [ftype (.getType field)]
              (when (and (not (.isPrimitive ftype))
                         (not (.isArray ftype))
                         (not= ftype Object))
                (try (.importClass ^clojure.lang.Namespace source-ns ftype)
                     (catch Exception _ nil))))))))
    ;; Bind a shared compilation classloader
    (let [shared-cl (or me/*compilation-classloader*
                        (let [tcl (.getContextClassLoader (Thread/currentThread))]
                          (if (instance? clojure.lang.DynamicClassLoader tcl)
                            tcl
                            (clojure.lang.DynamicClassLoader. tcl))))]
      (binding [me/*compilation-classloader* shared-cl]
        (let [prev-tcl (.getContextClassLoader (Thread/currentThread))]
          (.setContextClassLoader (Thread/currentThread) shared-cl)
          (try
            ;; Eval defvalue forms via shared class cache — ensures a single
            ;; Class object across make-dual and parametric-specialize!
            (binding [*ns* source-ns]
              (doseq [dv-form defvalue-forms]
                (let [concrete-name (second dv-form)
                      fqn (str (.replace (str source-ns) "-" "_") "." concrete-name)]
                  (get-or-create-parametric-class!
                   fqn
                   (fn []
                     (try (eval dv-form)
                          (catch Throwable e
                            (when-not (instance? LinkageError (.getCause e))
                              (binding [*out* *err*]
                                (println (str "WARNING: parametric defvalue failed: "
                                              concrete-name " — " (.getMessage e))))))))))))
            ;; Import specialized cache classes from the shared cache
            (when cache-specs
              (doseq [[_ specialized-name] cache-specs]
                (let [fqn (str (.replace (str source-ns) "-" "_") "." specialized-name)
                      cls (or (get @parametric-class-cache fqn)
                              (try (Class/forName fqn true shared-cl)
                                   (catch Exception _ nil)))]
                  (when cls
                    (.importClass ^clojure.lang.Namespace source-ns cls)
                    (swap! types/tag-class-registry assoc specialized-name cls)))))
            ;; Create the mangled defn var (minimal — just a placeholder for metadata)
            (binding [*ns* source-ns]
              (eval (list 'defn mangled-sym (vec params)
                          (list 'throw (list 'UnsupportedOperationException.
                                             "bytecode-compiled — should not reach here")))))
            ;; Set metadata on the mangled var
            (let [mangled-var (ns-resolve source-ns mangled-sym)]
              (alter-meta! mangled-var assoc
                           ::deftm-params (vec params)
                           ::deftm-tags tags
                           ::return-tag ret-tag
                           ::deftm-walked-body walked-body
                           ::deftm-source-body (vec [body-with-types])
                           ::deftm-annotations concrete-anns
                           ::deftm-source-ns (ns-name source-ns))
              ;; Create the -impl var with a stub, then bytecode-upgrade it
              (let [impl-var-sym (symbol (str mangled-sym "-impl"))
                    _ (binding [*ns* source-ns]
                        (eval (list 'def impl-var-sym nil)))
                    impl-var (ns-resolve source-ns impl-var-sym)]
                ;; Bytecode compile: creates invk (typed) + invoke(Object) bridge
                ;; This is the same path as lazy JIT — no Clojure compiler involved
                (do-bytecode-upgrade!
                 impl-var mangled-sym params tags ret-tag
                 walked-body source-ns typed-iface-name
                 :annotations concrete-anns
                 :source-body (vec [body-with-types]))
                ;; Update mangled var to delegate to bytecode-compiled impl.
                ;; The mangled defn was a throwing placeholder; now that the impl
                ;; exists, route IFn calls through it so mangled call paths work.
                (alter-var-root mangled-var
                                (constantly (fn [& call-args] (apply @impl-var call-args))))
                ;; Register in dispatch table
                (dispatch/register-method! (str mangled) fn-name tags source-ns
                                           impl-var typed-iface-name
                                           ;; typed-target-fn: routes through -impl
                                           (fn [& call-args]
                                             (apply @impl-var call-args))
                                           (meta (ns-resolve source-ns (symbol (name fn-name)))))
                ;; Call the newly-compiled impl directly rather than re-dispatching.
                ;; Re-dispatch through the generic function would fail on first call
                ;; because the original args may predate the class cache.
                (apply @impl-var args)))
            (finally
              (.setContextClassLoader (Thread/currentThread) prev-tcl))))))))

;; Register the callback with dispatch — breaks the cycle
(dispatch/set-parametric-specializer! parametric-specialize!)

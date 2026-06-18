(ns raster.compiler.core.walker
  "Source-to-source walker for deftm bodies.

  Walks Clojure source forms, tracking types and rewriting:
  - Auto-adds type hints to let bindings
  - Devirtualizes calls to deftm functions (polymorphic → monomorphic)
  - Expands registered typed macros with type env
  - Emits .invk calls for (Fn ...) annotated parameters

  Each form type is handled by a defmethod of the `walk-form` multimethod,
  dispatched via `classify-form`. This makes the walker:
  - Readable: each form handler is a separate, named function
  - Testable: handlers can be unit-tested with synthetic contexts
  - Extensible: new form types can be added via defmethod"
  (:require [clojure.string :as str]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.inference :as inf]))

(defn- warn-boxed-dispatch?
  "Check if *warn-on-boxed-dispatch* is bound and true."
  []
  (when-let [v (resolve 'raster.core/*warn-on-boxed-dispatch*)]
    @v))

(defn- boxed-dispatch-warning
  "Emit a boxed-dispatch warning, similar to Clojure's reflection warning."
  [fn-sym form reason]
  (let [m (meta form)]
    (binding [*out* *err*]
      (println (str "Boxed dispatch call to `" fn-sym "`"
                    (when (:file m) (str " (" (:file m) ":" (:line m) ")"))
                    " — " reason)))))

;; ================================================================
;; Type Context
;;
;; The walker tracks type information in a unified type-env:
;;   {sym → {:tag dispatch-tag          ;; e.g. 'double, 'doubles, 'DP5Cache
;;           :fn-info {:iface-name ...} ;; for (Fn [...]) typed functions
;;           :element dispatch-tag      ;; element type for Object[] parametric
;;           :tc-type tc-form}}         ;; TypedClojure type (optional)
;;
;; This replaces the previous 5 separate env maps (env, fn-env,
;; element-env, fn-ret-env, tc-env). Each symbol has one type-record
;; that carries all type information, propagated through let bindings.
;;
;; Key benefit: fn-info automatically propagates through let bindings
;; when a tag is an IFn__ type, fixing the ODE devirtualization bug.
;; ================================================================

(declare ctx-assoc-type)

(defn make-ctx
  "Create a walker type context with unified type-env.
  type-env: {symbol → {:tag dispatch-tag, :fn-info map?, :element dispatch-tag?}}
  tc-binding-tags: {symbol → dispatch-tag} pre-computed by TC (optional)
  NOTE: type-env entries are processed through ctx-assoc-type to ensure
  fn-info auto-derivation for IFn__ tagged parameters."
  [{:keys [type-env source-ns tc-binding-tags]
    :or {type-env {} source-ns *ns*}}]
  ;; Build the context by adding each type-env entry through ctx-assoc-type,
  ;; which auto-derives fn-info from IFn__ tags (fixing the ODE fn-param bug).
  (reduce-kv (fn [ctx sym record]
               (ctx-assoc-type ctx sym (:tag record)
                               (:fn-info record)
                               (:element record)))
             {:type-env  {}
              :source-ns source-ns
              :tc-binding-tags (or tc-binding-tags {})}
             type-env))

(defn ctx-assoc-type
  "Add/update a symbol's type record in the context.
  Automatically derives fn-info from IFn__ tags."
  ([ctx sym tag]
   (ctx-assoc-type ctx sym tag nil nil))
  ([ctx sym tag fn-info]
   (ctx-assoc-type ctx sym tag fn-info nil))
  ([ctx sym tag fn-info element]
   (let [record (cond-> {:tag tag}
                  fn-info (assoc :fn-info fn-info)
                  element (assoc :element element)
                  ;; Auto-derive fn-info from IFn__ tags
                  (and (not fn-info) (types/fn-type-tag? tag))
                  (assoc :fn-info {:iface-name tag}))]
     (update ctx :type-env assoc sym record))))

(defn- stamp-type-meta
  "Stamp type metadata onto a symbol. Always sets :raster.type/tag (compiler
  canonical). Only sets Clojure :tag for non-primitive types (arrays, classes,
  interfaces) — Clojure rejects :tag on locals with primitive initializers.
  Also sets :raster.type/fn-info for typed function interfaces.
  This is the ONLY place type metadata should be attached to symbols."
  [sym tag]
  (if tag
    (let [;; Clojure :tag is safe for arrays and classes, not bare primitives
          clj-hint (inf/compute-binding-hint tag sym)]
      (cond-> (vary-meta sym assoc :raster.type/tag tag)
        clj-hint (vary-meta assoc :tag clj-hint)
        (types/fn-type-tag? tag)
        (vary-meta assoc :raster.type/fn-info {:iface-name tag})))
    sym))

(defn ctx-get-tag
  "Get the dispatch tag for a symbol.

  Reads the ctx type-env first (the per-walk cache), then falls back to a tag
  carried on the symbol itself (`:raster.type/tag` metadata stamped at the
  reference site by a prior walk). Carrying the type on the AST means a re-walk
  over restructured code (PE/CSE/inline) keeps a binding's type even when the
  re-derived ctx type-env doesn't reach into a nested scope (e.g. an ftm closure
  capturing an outer var)."
  [ctx sym]
  (or (get-in (:type-env ctx) [sym :tag])
      (when (symbol? sym) (:raster.type/tag (meta sym)))))

(defn ctx-get-fn-info
  "Get fn-info for a symbol."
  [ctx sym]
  (get-in (:type-env ctx) [sym :fn-info]))

(defn ctx-get-element
  "Get element type for an Object[] symbol."
  [ctx sym]
  (get-in (:type-env ctx) [sym :element]))

;; ================================================================
;; Form Classification
;; ================================================================

(declare resolve-aliased-symbol)

(defn classify-form
  "Classify a source form for walker dispatch.
  Returns a keyword identifying which defmethod should handle it."
  [form ctx]
  (cond
    ;; Binding forms — backtick `let expands to clojure.core/let,
    ;; which appears in macroexpanded input before the walker normalizes to let*
    (and (seq? form) (#{`let 'let 'let* 'loop 'loop*} (first form)))
    :let

    (and (seq? form) (#{`fn 'fn 'fn*} (first form)))
    :fn

    (and (seq? form) (= 'ftm (first form)))
    :ftm

    ;; Parallel ops — resolve namespace aliases (par/reduce → raster.par/reduce)
    ;; so user code using aliases gets the typed handlers with proper type-env.
    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/map! resolved)))
    :par-map

    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/reduce resolved)))
    :par-reduce

    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/scan resolved)))
    :par-scan

    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/stencil! resolved)))
    :par-stencil

    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/gather resolved)))
    :par-gather

    ;; Pure par/map — first-class IR form (NOT a typed macro)
    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (= 'raster.par/map resolved)))
    :par-map-pure

    ;; Also catch the IR form directly (raster.par/pmap)
    (and (seq? form) (contains? #{'raster.par/pmap 'par/pmap} (first form)))
    :par-map-pure

    ;; Typed function parameter call (Fn annotation) — must come before typed-macro
    ;; so that local fn parameters shadow global typed macros like 'broadcast
    (and (seq? form) (symbol? (first form)) (ctx-get-fn-info ctx (first form)))
    :fn-param-call

    ;; Typed macro — resolve aliases for qualified heads (e.g., par/map → raster.par/map)
    (and (seq? form)
         (let [head (first form)
               resolved (if (and (symbol? head) (namespace head))
                          (resolve-aliased-symbol head (:source-ns ctx))
                          head)]
           (inf/typed-macro? resolved)))
    :typed-macro

    ;; deftm generic call (devirtualization target)
    ;; Skip aget/aset/aclone on JVM primitive arrays — let :array-op handle those.
    ;; Array-like types (Multivector etc.) go through deftm dispatch.
    ;; alength always goes through deftm dispatch for Long return type.
    (and (seq? form) (inf/generic-fn?* (:source-ns ctx) (first form))
         (not (and (let [fn-name (name (first form))]
                     (contains? #{"aget" "aset" "aclone"} fn-name))
                   (symbol? (second form))
                   (let [arr-tag (ctx-get-tag ctx (second form))]
                     (and arr-tag (contains? types/primitive-array-tags arr-tag))))))
    :deftm-call

    ;; dotimes
    (and (seq? form) (= 'dotimes (first form)))
    :dotimes

    ;; Array op with known type — auto-hint for JVM primitive arrays only.
    ;; Array-like types (registered via register-array-like!) use deftm dispatch.
    ;; alength excluded: goes through deftm dispatch for Long return type.
    (and (seq? form)
         (contains? #{'aget 'aset 'aclone
                      'clojure.core/aget
                      'clojure.core/aset 'clojure.core/aclone
                      'raster.arrays/aget 'raster.arrays/aset
                      'raster.arrays/aclone} (first form))
         (symbol? (second form))
         (let [arr-tag (ctx-get-tag ctx (second form))]
           (and arr-tag (contains? types/primitive-array-tags arr-tag)
                (not (:tag (meta (second form)))))))
    :array-op

    ;; var reference — (var foo) / #'foo — resolve and pass through
    (and (seq? form) (= 'var (first form)) (= 2 (count form)))
    :var-ref

    ;; If with type guard — narrow types in branches for occurrence typing
    (and (seq? form) (= 'if (first form)) (>= (count form) 3))
    :if-form

    ;; Keyword field access on typed record — (:field record) → (.field record)
    ;; Safe when the env proves the receiver is a non-nil record type with that field.
    (and (seq? form) (= 2 (count form))
         (keyword? (first form))
         (symbol? (second form))
         (let [recv-tag (ctx-get-tag ctx (second form))]
           (and recv-tag
                (symbol? recv-tag)
                ;; Check if the receiver type has this field in the registry
                ;; Registry keys are strings, not symbols
                (get-in @inf/field-type-registry
                        [recv-tag (name (first form))]))))
    :keyword-field-access

    ;; Primitive cast — (double x), (long x), etc. — walk inner, don't re-cast
    (and (seq? form) (= 2 (count form))
         (contains? types/primitive-info (first form)))
    :cast

    ;; try/catch/finally
    (and (seq? form) (= 'try (first form)))
    :try

    ;; quote — opaque, don't walk contents (preserves literal data)
    (and (seq? form) (= 'quote (first form)))
    :leaf

    ;; reify — walk method bodies but not interface names or method signatures
    (and (seq? form) (#{'reify 'reify*} (first form)))
    :reify

    ;; letfn — mutual recursion with all names in scope for all bodies
    (and (seq? form) (#{'letfn 'letfn*} (first form)))
    :letfn

    ;; case — walk test expr and result exprs but not dispatch constants
    (and (seq? form) (#{'case 'case*} (first form)))
    :case

    ;; new — walk constructor args but not class name
    (and (seq? form) (= 'new (first form)))
    :new

    ;; Other function call
    (seq? form)
    :call

    ;; Collections
    (vector? form) :vector
    (map? form)    :map
    (set? form)    :set

    ;; Leaf
    :else :leaf))

;; ================================================================
;; Walker Entry Point
;; ================================================================

(declare walk-form)

(defn- resolve-aliased-symbol
  "Resolve a namespace-aliased symbol to its fully qualified name.
   e.g., n/neg-inf in raster.nn → raster.numeric/neg-inf.
   Returns the qualified symbol, or the original if not resolvable.
   Preserves metadata from the original symbol."
  [sym source-ns]
  (if-not (and (symbol? sym) (namespace sym))
    sym
    (try
      (let [v (ns-resolve source-ns sym)]
        (if (instance? clojure.lang.Var v)
          (let [resolved (symbol (str (.name (.ns ^clojure.lang.Var v)))
                                 (str (.sym ^clojure.lang.Var v)))]
            (if-let [m (meta sym)]
              (with-meta resolved m)
              resolved))
          sym))
      (catch Exception _ sym))))

(defn walk
  "Walk a source form with type context. Returns the walked form.
  Preserves metadata and auto-hints field access on typed receivers.
  Resolves namespace-aliased symbols to fully qualified names."
  [form ctx]
  (let [;; Resolve namespace-aliased symbols to fully qualified names
        form (if (and (symbol? form) (namespace form) (not (ctx-get-tag ctx form)))
               (resolve-aliased-symbol form (:source-ns ctx))
               form)
        ;; Qualify bare var references (e.g. GRAVITY → valley.typed-core/GRAVITY)
        ;; so downstream code can always resolve them. Only qualifies non-fn vars
        ;; (constants, def values). Skips locals, special forms, macros, and fns.
        form (if (and (symbol? form) (not (namespace form))
                      (not (ctx-get-tag ctx form))
                      (not (special-symbol? form)))
               (if-let [v (try (ns-resolve (:source-ns ctx) form) (catch Exception _ nil))]
                 (if (and (var? v)
                          (not (.isMacro ^clojure.lang.Var v))
                          (let [root (try (.getRawRoot ^clojure.lang.Var v) (catch Exception _ nil))]
                            (and (some? root) (not (ifn? root)))))
                   (symbol (str (.name (.ns ^clojure.lang.Var v)))
                           (str (.sym ^clojure.lang.Var v)))
                   form)
                 form)
               form)
        result (walk-form form ctx)
        ;; Preserve metadata from original form
        result (if-let [m (meta form)]
                 (if (instance? clojure.lang.IObj result)
                   (with-meta result (merge m (meta result)))
                   result)
                 result)]
    ;; Auto-hint field access on typed receivers: (.field sym) → ^FieldType (.field ^RecvType sym)
    (let [result
          (if (and (seq? result) (= 2 (count result))
                   (not (:tag (meta result)))
                   (symbol? (first result))
                   (let [n (name (first result))]
                     (and (str/starts-with? n ".") (> (count n) 1)
                          (not (str/starts-with? n ".-"))))
                   (symbol? (second result)))
            (if-let [field-tag (inf/infer-field-type result (:type-env ctx))]
              (vary-meta result assoc :tag field-tag :raster.type/tag field-tag)
              result)
            result)]
      ;; Stamp return type on .invk calls and primitive casts.
      ;; Propagates the return type from impl-sym metadata onto the call form,
      ;; so downstream passes and GPU codegen can query (:raster.type/tag (meta expr)).
      ;; Stripped before reify emission in ftm to avoid 64KB bytecode limit.
      (if (and (seq? result)
               (not (:raster.type/tag (meta result)))
               (instance? clojure.lang.IObj result))
        (let [head (first result)
              type-env (:type-env ctx)
              ;; Tag of an already-walked sub-form (bottom-up: inner forms are
              ;; walked+stamped before their parent).
              walked-tag (fn [w]
                           (cond
                             (seq? w)    (or (:raster.type/tag (meta w))
                                             (inf/infer-rewritten-tag w nil type-env))
                             (symbol? w) (or (:raster.type/tag (meta w))
                                             (inf/type-env-tag type-env w))
                             :else       (inf/literal-tag w)))]
          (cond
            ;; .invk — read ret-tag from impl-sym metadata
            (and (= '.invk head) (symbol? (second result)))
            (if-let [ret-tag (:raster.type/ret-tag (meta (second result)))]
              (vary-meta result assoc :raster.type/tag ret-tag)
              result)
            ;; Primitive cast — (double x), (float x), etc.
            (contains? types/primitive-info head)
            (vary-meta result assoc :raster.type/tag head)
            ;; if — result type is the (agreeing) type of its value branches.
            ;; A recur branch carries no value, so the result is the OTHER branch;
            ;; this also types loop bodies of the form (if test (recur ...) acc).
            ;; Only stamp when the type is unambiguous (both value branches agree)
            ;; so a genuine union stays untyped (and thus dispatched — correct).
            (and (= 'if head) (>= (count result) 3))
            (let [recur? (fn [x] (and (seq? x) (= 'recur (first x))))
                  then (nth result 2)
                  els  (when (> (count result) 3) (nth result 3))
                  tt   (when-not (recur? then) (walked-tag then))
                  et   (when (and els (not (recur? els))) (walked-tag els))
                  rtag (cond
                         (and tt et (= tt et))      tt
                         (and tt els (recur? els))  tt
                         (and et (recur? then))     et
                         :else nil)]              ; one-arm/ambiguous → leave untyped
              (if rtag (vary-meta result assoc :raster.type/tag rtag) result))
            ;; let*/loop*/do — result type is the last body form's type.
            (contains? '#{let* loop* do} head)
            (if-let [t (walked-tag (last result))]
              (vary-meta result assoc :raster.type/tag t)
              result)
            ;; case — result type is the (agreeing) type of all clause results
            ;; (every 2nd clause element) plus the optional trailing default.
            (= 'case head)
            (let [clauses (drop 2 result)
                  pairs?  (even? (count clauses))
                  results (concat (map second (partition 2 (if pairs? clauses (butlast clauses))))
                                  (when-not pairs? [(last clauses)]))
                  tags    (map walked-tag results)]
              (if (and (seq tags) (every? some? tags) (apply = tags))
                (vary-meta result assoc :raster.type/tag (first tags))
                result))
            ;; try — result type is the (agreeing) type of the body's last form and
            ;; each catch's last form (finally never yields the value).
            (= 'try head)
            (let [parts   (rest result)
                  cf?     (fn [f] (and (seq? f) (contains? #{'catch 'finally} (first f))))
                  body    (take-while (complement cf?) parts)
                  catches (filter #(and (seq? %) (= 'catch (first %))) parts)
                  vforms  (concat (when (seq body) [(last body)]) (map last catches))
                  tags    (map walked-tag vforms)]
              (if (and (seq tags) (every? some? tags) (apply = tags))
                (vary-meta result assoc :raster.type/tag (first tags))
                result))
            :else result))
        result))))

(defn walk-forms
  "Walk multiple forms with the same context. Returns a vector of walked forms."
  [forms ctx]
  (mapv #(walk % ctx) forms))

;; ================================================================
;; Multimethod
;; ================================================================

(defmulti walk-form
  "Walk a source form, returning the rewritten form.
  Dispatches on (classify-form form ctx)."
  (fn [form ctx] (classify-form form ctx)))

;; ================================================================
;; Branch: let/let*/loop
;; ================================================================

(defmethod walk-form :let [form ctx]
  (let [[let-sym bindings & body] form
        [new-bindings new-ctx]
        (reduce
         (fn [[binds ctx] [sym init]]
           (let [rewritten-init (walk init ctx)
                 ;; ANF flattening: if a typed-macro expansion returns (let* [...] body),
                 ;; splice the inner bindings into the parent let and use the body as
                 ;; the init. This keeps the IR flat for fusion and bytecode compilation.
                 [extra-binds rewritten-init]
                 (if (and (seq? rewritten-init) (= 'let* (first rewritten-init)))
                   (let [[_ inner-bindings & inner-body] rewritten-init
                         ;; CC10 fix: splice intermediate effectful forms as bindings
                         ;; so they are not silently dropped
                         effect-binds (vec (mapcat (fn [form]
                                                     [(gensym "_effect__") form])
                                                   (butlast inner-body)))]
                     [(into (vec inner-bindings) effect-binds) (last inner-body)])
                   [nil rewritten-init])
                 ;; Process extra bindings from flattened let*
                 [binds ctx]
                 (if extra-binds
                   (reduce (fn [[binds ctx] [esym einit]]
                             (let [etag (inf/infer-binding-tag esym einit einit (:type-env ctx)
                                                               {:source-ns (:source-ns ctx)})
                                   esym (if etag (stamp-type-meta esym etag) esym)]
                               [(conj binds esym einit)
                                (if etag (ctx-assoc-type ctx esym etag) ctx)]))
                           [binds ctx]
                           (partition 2 extra-binds))
                   [binds ctx])
                 type-env (:type-env ctx)
                 ;; TC pre-computed hint takes priority (covers core arith, deftm returns)
                 tc-tag (get (:tc-binding-tags ctx) sym)
                 tag (or tc-tag
                         (inf/infer-binding-tag sym init rewritten-init type-env
                                                {:source-ns (:source-ns ctx)}))
                 elem-tag (inf/infer-element-tag init tag type-env)
                 hint (inf/compute-binding-hint tag sym)
                 sym (stamp-type-meta sym (or hint tag))]
             [(conj binds sym rewritten-init)
              (cond-> ctx
                tag (ctx-assoc-type sym tag)
                elem-tag (update-in [:type-env sym] assoc :element elem-tag))]))
         [[] ctx]
         (partition 2 bindings))]
    (list* let-sym (vec new-bindings) (walk-forms body new-ctx))))

;; ================================================================
;; Branch: fn/fn*
;; ================================================================

(defmethod walk-form :fn [form ctx]
  (let [[fn-sym & rest-args] form
        [fname arities] (if (symbol? (first rest-args))
                          [(first rest-args) (rest rest-args)]
                          [nil rest-args])
        arities (if (vector? (first arities))
                  (list arities)
                  arities)
        rewritten-arities
        (for [arity arities]
          (let [params (first arity)
                body (rest arity)
                param-ctx (reduce
                           (fn [ctx p]
                             (if-let [t (inf/hint-tag p)]
                               (ctx-assoc-type ctx p t)
                               ctx))
                           ctx
                           (remove #{'&} params))]
            (cons params (walk-forms body param-ctx))))]
    (if fname
      (list* fn-sym fname rewritten-arities)
      (list* fn-sym rewritten-arities))))

;; ================================================================
;; Branch: try/catch/finally
;; ================================================================

(defmethod walk-form :try [form ctx]
  (let [[_ & body-and-clauses] form
        catch-or-finally? (fn [f] (and (seq? f) (contains? #{'catch 'finally} (first f))))
        [body-forms clauses] (split-with (complement catch-or-finally?) body-and-clauses)]
    (list* 'try
           (concat
            (walk-forms body-forms ctx)
            (map (fn [clause]
                   (if (= 'catch (first clause))
                     (let [[_ ex-class ex-sym & catch-body] clause
                           ex-tag (symbol (.getName ^Class (ns-resolve (:source-ns ctx) ex-class)))
                           catch-ctx (ctx-assoc-type ctx ex-sym ex-tag)]
                       (list* 'catch ex-class ex-sym (walk-forms catch-body catch-ctx)))
                 ;; finally
                     (let [[_ & fin-body] clause]
                       (list* 'finally (walk-forms fin-body ctx)))))
                 clauses)))))

;; ================================================================
;; Branch: ftm (typed anonymous function)
;; ================================================================

(defmethod walk-form :ftm [form ctx]
  (let [[ftm-sym param-vec & body-args] form
        ;; Parse optional return type: (ftm [params] :- RetType body...)
        [ret-type tail] (if (= :- (first body-args))
                          [(second body-args) (nnext body-args)]
                          [nil body-args])
        ;; Idempotency: the pipeline walks ftms more than once (walk → inline →
        ;; pe-rewalk). An already-walked ftm carries its raw body in a
        ;; `:raster.walker/source-body <vec>` marker; preserve that original raw
        ;; body and re-walk only the walked body. Re-wrapping would NEST a second
        ;; source-body and double the (64KB-method-limit-sensitive) payload.
        already-walked? (= :raster.walker/source-body (first tail))
        src-body (if already-walked? (second tail) (vec tail))
        body     (if already-walked? (nnext tail) tail)
        ;; Parse typed params: [a :- Double, b :- Long] → extract names + types
        {:keys [params annotations]} (types/parse-typed-params param-vec)
        ;; Add params to walker context with their type annotations
        param-ctx (reduce
                   (fn [ctx [p ann]]
                     (let [tag (when ann (types/annotation->tag ann p))
                           fn-info (when ann (types/annotation->fn-info ann))]
                       (cond-> ctx
                         tag (ctx-assoc-type p tag fn-info))))
                   ctx
                   (map vector params annotations))
        walked-body (walk-forms body param-ctx)
        ;; Pass original source body to ftm macro via :raster.walker/source-body
        ;; keyword arg. The ftm macro stores this for AD transparency — the source
        ;; body uses raster.numeric dispatch (handles Dual) unlike walked .invk calls.
        result (if ret-type
                 (list* ftm-sym param-vec :- ret-type
                        :raster.walker/source-body src-body
                        walked-body)
                 (list* ftm-sym param-vec
                        :raster.walker/source-body src-body
                        walked-body))]
    result))

;; ================================================================
;; Branch: dotimes
;; ================================================================

(defmethod walk-form :dotimes [form ctx]
  (let [[_ [i-sym bound-expr] & body] form
        walked-bound (walk bound-expr ctx)
        idx-ctx (ctx-assoc-type ctx i-sym 'long)]
    (list* 'dotimes [i-sym walked-bound] (walk-forms body idx-ctx))))

;; ================================================================
;; Branch: parallel ops (map!, reduce, scan, stencil!)
;; ================================================================

(defmethod walk-form :par-map [form ctx]
  ;; Handle both standard and offset variants:
  ;;   Standard: (par/map! out idx bound cast body)
  ;;   Offset:   (par/map! out idx bound :offset base cast body)
  (let [offset? (and (>= (count form) 8) (= :offset (nth form 4)))
        [_ out-sym i-sym bound-expr] form
        [base-expr cast-fn body-expr]
        (if offset?
          [(nth form 5) (nth form 6) (nth form 7)]
          [nil (nth form 4) (nth form 5)])
        out-tag (ctx-get-tag ctx out-sym)
        hinted-out (if (and out-tag (types/array-tag? out-tag))
                     (stamp-type-meta out-sym out-tag)
                     out-sym)
        walked-bound (walk bound-expr ctx)
        idx-ctx (ctx-assoc-type ctx i-sym 'long)
        walked-body (walk body-expr idx-ctx)
        result (if offset?
                 (list 'raster.par/map! hinted-out i-sym walked-bound
                       :offset (walk base-expr ctx) cast-fn walked-body)
                 (list 'raster.par/map! hinted-out i-sym walked-bound cast-fn walked-body))
        elem-type (cond
                    (= cast-fn 'float) :float
                    (= cast-fn 'double) :double
                    (= cast-fn 'int) :int
                    (= cast-fn 'long) :long
                    (= out-tag 'floats) :float
                    (= out-tag 'doubles) :double
                    (= out-tag 'ints) :int
                    (= out-tag 'longs) :long
                    :else nil)]
    (if out-tag
      (with-meta result (cond-> {:tag out-tag :raster.type/tag out-tag}
                          elem-type (assoc :raster.type/elem-type elem-type)))
      (if elem-type
        (with-meta result {:raster.type/elem-type elem-type})
        result))))

(defmethod walk-form :par-reduce [form ctx]
  (let [[_ acc-sym init-expr i-sym bound-expr body-expr] form
        walked-init (walk init-expr ctx)
        walked-bound (walk bound-expr ctx)
        acc-tag (or (inf/hint-tag acc-sym)
                    (inf/literal-tag init-expr)
                    ;; Recognize cast expressions: (float x), (double x)
                    (when (and (seq? init-expr) (symbol? (first init-expr)))
                      (case (first init-expr)
                        float 'float
                        double 'double
                        nil))
                    ;; Infer from type-env if init is a symbol
                    (when (symbol? init-expr) (ctx-get-tag ctx init-expr)))
        _ (when-not acc-tag
            (throw (ex-info (str "par/reduce: cannot infer accumulator type from init expression `"
                                 (pr-str init-expr) "`. "
                                 "Use a typed init like (double 0) or (float 0), "
                                 "or annotate the accumulator symbol with :- Type.")
                            {:form form :init init-expr})))
        body-ctx (-> ctx
                     (ctx-assoc-type i-sym 'long)
                     (ctx-assoc-type acc-sym acc-tag))]
    (let [elem-type (case acc-tag
                      float :float
                      double :double
                      int :int
                      long :long
                      :double)]
      (with-meta
        (list 'raster.par/reduce acc-sym walked-init i-sym walked-bound (walk body-expr body-ctx))
        {:tag acc-tag :raster.type/tag acc-tag :raster.type/elem-type elem-type}))))

(defmethod walk-form :par-scan [form ctx]
  (let [[_ out-sym acc-sym init-expr i-sym bound-expr cast-fn body-expr] form
        out-tag (ctx-get-tag ctx out-sym)
        hinted-out (if (and out-tag (types/array-tag? out-tag))
                     (stamp-type-meta out-sym out-tag)
                     out-sym)
        walked-init (walk init-expr ctx)
        walked-bound (walk bound-expr ctx)
        acc-tag (or (inf/hint-tag acc-sym)
                    (inf/literal-tag init-expr)
                    (when (and (seq? init-expr) (symbol? (first init-expr)))
                      (case (first init-expr)
                        float 'float
                        double 'double
                        nil))
                    (when (symbol? init-expr) (ctx-get-tag ctx init-expr)))
        _ (when-not acc-tag
            (throw (ex-info (str "par/scan: cannot infer accumulator type from init expression `"
                                 (pr-str init-expr) "`. "
                                 "Use a typed init like (double 0) or (float 0).")
                            {:form form :init init-expr})))
        body-ctx (-> ctx
                     (ctx-assoc-type i-sym 'long)
                     (ctx-assoc-type acc-sym acc-tag))]
    (let [elem-type (cond
                      (= cast-fn 'float) :float
                      (= cast-fn 'double) :double
                      (= acc-tag 'float) :float
                      (= acc-tag 'double) :double
                      (= acc-tag 'long) :long
                      (= acc-tag 'int) :int
                      :else (throw (ex-info (str "par/scan: cannot determine element type from cast `"
                                                 cast-fn "` or acc-tag `" acc-tag "`")
                                            {:cast-fn cast-fn :acc-tag acc-tag :form form})))
          result (list 'raster.par/scan hinted-out acc-sym walked-init i-sym walked-bound cast-fn
                       (walk body-expr body-ctx))]
      (with-meta result (cond-> {:raster.type/elem-type elem-type}
                          out-tag (assoc :tag out-tag :raster.type/tag out-tag))))))

(defmethod walk-form :par-stencil [form ctx]
  (let [[_ out-sym in-arrays radius boundary cast-fn i-sym bound-expr body-expr] form
        out-tag (ctx-get-tag ctx out-sym)
        hinted-out (if (and out-tag (types/array-tag? out-tag))
                     (stamp-type-meta out-sym out-tag)
                     out-sym)
        walked-bound (walk bound-expr ctx)
        idx-ctx (ctx-assoc-type ctx i-sym 'long)]
    (let [elem-type (cond
                      (= cast-fn 'float) :float
                      (= cast-fn 'double) :double
                      (= out-tag 'floats) :float
                      (= out-tag 'doubles) :double
                      (= out-tag 'longs) :long
                      (= out-tag 'ints) :int
                      out-tag (throw (ex-info (str "par/stencil: unrecognized out-tag `" out-tag "`")
                                              {:out-tag out-tag :form form}))
                      :else nil)
          result (list 'raster.par/stencil! hinted-out in-arrays radius boundary cast-fn
                       i-sym walked-bound (walk body-expr idx-ctx))]
      (if (or out-tag elem-type)
        (with-meta result (cond-> {}
                            out-tag (assoc :tag out-tag :raster.type/tag out-tag)
                            elem-type (assoc :raster.type/elem-type elem-type)))
        result))))

;; ================================================================
;; Branch: par/gather (first-class IR — preserved for the SIMD vgather pass,
;; not macroexpanded to a scalar loop)
;; ================================================================

(defmethod walk-form :par-gather [form ctx]
  ;; (raster.par/gather out src index n [stride]) — keep symbolic; the SIMD
  ;; pass emits a hardware vgather (DoubleVector.fromArray index-map form),
  ;; else expand-par-gather! lowers it to a scalar loop.
  (let [args       (vec (rest form))
        out-sym    (first args)
        out-tag    (ctx-get-tag ctx out-sym)
        hinted-out (if (and out-tag (types/array-tag? out-tag))
                     (stamp-type-meta out-sym out-tag)
                     out-sym)
        elem-type  (case out-tag
                     floats :float doubles :double longs :long ints :int
                     nil)
        result     (list* 'raster.par/gather hinted-out
                          (mapv #(walk % ctx) (rest args)))]
    (if (or out-tag elem-type)
      (with-meta result (cond-> {}
                          out-tag (assoc :tag out-tag :raster.type/tag out-tag)
                          elem-type (assoc :raster.type/elem-type elem-type)))
      result)))

;; ================================================================
;; Branch: pure par/map (first-class IR, not typed macro)
;; ================================================================

(defmethod walk-form :par-map-pure [form ctx]
  ;; Source form: (par/map [i n] body) or (par/map [i n] :like ref body)
  ;; IR form (rewalk): (raster.par/pmap idx bound cast body)
  (let [source? (and (sequential? (second form)) (vector? (second form)))
        ;; Parse source vs IR form
        [i-sym bound-expr cast-hint body-expr]
        (if source?
          (let [[_ binding & rest-args] form
                [i-sym bound-expr] binding
                [_ body-expr] (if (= :like (first rest-args))
                                [(second rest-args) (nth rest-args 2)]
                                [nil (first rest-args)])]
            [i-sym bound-expr nil body-expr])
          ;; IR form: (raster.par/pmap idx bound cast body)
          (let [[_ i-sym bound-expr cast-fn body-expr] form]
            [i-sym bound-expr cast-fn body-expr]))
        walked-bound (walk bound-expr ctx)
        idx-ctx (ctx-assoc-type ctx i-sym 'long)
        walked-body (walk body-expr idx-ctx)
        ;; Infer output element type from body context or existing cast
        existing-elem (:raster.type/elem-type (meta form))
        inferred-tag (or (inf/infer-body-element-type walked-body (:type-env ctx))
                         ;; No array-typed symbols — infer from body expression type
                         (let [body-tag (or (:tag (meta walked-body))
                                            (inf/infer-arg-tag walked-body (:type-env idx-ctx)))]
                           (case body-tag
                             double 'doubles
                             float 'floats
                             long 'longs
                             int 'ints
                             nil)))
        out-tag (or (when existing-elem
                      (case existing-elem :float 'floats :double 'doubles
                            :long 'longs :int 'ints nil))
                    inferred-tag)
        _ (when (and (not out-tag) (not cast-hint))
            (throw (ex-info (str "par/map: cannot infer output element type from body. "
                                 "Use a typed cast hint or :like annotation.")
                            {:form form})))
        out-tag (or out-tag (case cast-hint
                              float 'floats
                              double 'doubles
                              long 'longs
                              int 'ints
                              (throw (ex-info (str "par/map: unrecognized cast hint `" cast-hint "`")
                                              {:cast-hint cast-hint :form form}))))
        store-cast (or cast-hint (types/array-tag->cast out-tag))
        elem-type (case out-tag
                    floats :float
                    doubles :double
                    longs :long
                    ints :int
                    (throw (ex-info (str "par/map: unrecognized out-tag `" out-tag "`")
                                    {:out-tag out-tag :form form})))]
    (with-meta
      (list 'raster.par/pmap i-sym walked-bound store-cast walked-body)
      {:tag out-tag :raster.type/tag out-tag :raster.type/elem-type elem-type})))

;; ================================================================
;; Branch: typed macro
;; ================================================================

(defmethod walk-form :typed-macro [form ctx]
  (let [head (first form)
        resolved (if (and (symbol? head) (namespace head))
                   (resolve-aliased-symbol head (:source-ns ctx))
                   head)
        expand-fn (inf/get-typed-macro resolved)
        expanded (expand-fn form (:type-env ctx))]
    (walk expanded ctx)))

;; ================================================================
;; Branch: typed function parameter call (Fn annotation)
;; ================================================================

(defmethod walk-form :fn-param-call [form ctx]
  (let [fn-sym (first form)
        fn-info (ctx-get-fn-info ctx fn-sym)
        iface-name (:iface-name fn-info)
        rewritten-args (walk-forms (vec (rest form)) ctx)]
    (let [stamped (stamp-type-meta fn-sym iface-name)
          ret-tag (:ret-tag fn-info)]
      (with-meta
        (list* '.invk
               (if ret-tag (vary-meta stamped assoc :raster.type/ret-tag ret-tag) stamped)
               rewritten-args)
        {:raster.op/original fn-sym}))))

;; ================================================================
;; Branch: deftm call (devirtualization)
;; ================================================================

(defmethod walk-form :deftm-call [form ctx]
  (let [source-ns (:source-ns ctx)
        type-env (:type-env ctx)
        fn-sym (first form)
        rewritten-args (walk-forms (vec (rest form)) ctx)
        original-args (vec (rest form))
        extra-tags (mapv (fn [orig rewr]
                           (or (inf/infer-aget-type orig type-env)
                               (when (seq? rewr)
                                 (binding [*ns* source-ns]
                                   (inf/infer-rewritten-tag rewr orig type-env)))))
                         original-args rewritten-args)]
    ;; Reducible desugaring: when a reducible deftm (like +, *) is called
    ;; with >2 args and no exact-arity overload exists, desugar to nested
    ;; binary calls: (+ a b c) → (+ (+ a b) c). This matches the dispatch
    ;; function's runtime behavior and enables per-pair devirtualization.
    (if (and (> (count original-args) 2)
             (not (binding [*ns* source-ns]
                    (inf/try-resolve-call fn-sym original-args type-env)))
             (when-let [v (ns-resolve source-ns fn-sym)]
               (:raster.core/reducible (meta v))))
      ;; Desugar and re-walk the reduced form
      (let [reduced-form (reduce (fn [acc arg] (list fn-sym acc arg))
                                 (first original-args) (rest original-args))]
        (walk reduced-form ctx))
      ;; Standard path: try exact arity resolution
      (if-let [resolved (binding [*ns* source-ns]
                          (let [r (or (inf/try-resolve-call fn-sym original-args type-env)
                                      (inf/try-resolve-call fn-sym original-args type-env extra-tags))]
                            ;; Don't devirtualize to Object-tagged methods — they are
                            ;; untyped fallbacks that box everything. Number fallbacks
                            ;; are accepted since resolve-method-entry already prefers
                            ;; concrete matches and tries promotion first.
                            (when (and r (not (some #{'Object} (:tags r))))
                              r)))]
        (let [{:keys [mangled-sym typed-impl typed-iface has-fn-params promotion-casts]} resolved]
          (let [;; Apply promotion casts: wrap args that need type promotion
              ;; e.g., when dispatch resolved float+double → double+double,
              ;; wrap the float arg in (double ...)
                rewritten-args (if promotion-casts
                                 (mapv (fn [arg cast]
                                         (if cast (list cast arg) arg))
                                       rewritten-args promotion-casts)
                                 rewritten-args)
              ;; Emit .invk when we have a typed-impl AND either:
              ;; (a) no (Fn ...) params, OR
              ;; (b) all (Fn ...) args are known typed fns in the walker env.
              ;; When a (Fn ...) arg is untyped (plain defn or opaque), the
              ;; typed-impl's body would ClassCastException on .invk, so we
              ;; fall back to the mangled defn which handles both via dispatch.
                fn-args-safe?
                (if-not has-fn-params
                  true
                  (let [tags (:tags resolved)]
                    (every? identity
                            (map (fn [tag arg]
                                   (if (and (symbol? tag)
                                            (.contains (str tag) "IFn__"))
                             ;; (Fn ...) param — check tag OR fn-info for typed fn tag
                                     (or (when-let [arg-tag (when (symbol? arg) (ctx-get-tag ctx arg))]
                                           (.contains (str arg-tag) "IFn__"))
                                         (when (symbol? arg)
                                           (ctx-get-fn-info ctx arg)))
                                     true))
                                 tags (rest form)))))
                impl-var-sym (when (and typed-iface fn-args-safe?)
                               (let [s (symbol (namespace mangled-sym)
                                               (str (name mangled-sym) "-impl"))]
                                 (when (ns-resolve source-ns s) s)))
              ;; Qualify the original op for downstream passes
                qualified-op (if-let [v (ns-resolve source-ns fn-sym)]
                               (symbol (str (.name (.ns ^clojure.lang.Var v)))
                                       (str (.sym ^clojure.lang.Var v)))
                               fn-sym)]
            (cond
            ;; Best: compile-time safe — all (Fn ...) args are known typed
              impl-var-sym
              (let [stamped (stamp-type-meta impl-var-sym typed-iface)
                    ret-tag (:return-tag resolved)
                    stamped (if ret-tag
                              (vary-meta stamped assoc :raster.type/ret-tag ret-tag)
                              stamped)]
                (with-meta
                  (list* '.invk stamped rewritten-args)
                  {:raster.op/original qualified-op}))
            ;; Middle: has (Fn ...) params but args are opaque (e.g. from Object field).
            ;; Use typed-target-fn which does runtime instanceof check → fast .invk path
            ;; when the arg IS a typed fn (which it almost always is at runtime).
              (and has-fn-params
                   (let [s (symbol (namespace mangled-sym)
                                   (str (name mangled-sym) "-typed-fn"))]
                     (ns-resolve source-ns s)))
              (let [typed-fn-sym (symbol (namespace mangled-sym)
                                         (str (name mangled-sym) "-typed-fn"))]
                (with-meta (cons typed-fn-sym rewritten-args)
                  {:raster.op/original qualified-op}))
            ;; Fallback: plain defn (no typed path available)
              :else
              (do (when (warn-boxed-dispatch?)
                    (boxed-dispatch-warning fn-sym form "resolved but no typed impl available"))
                  (with-meta (cons mangled-sym rewritten-args)
                    {:raster.op/original qualified-op})))))
      ;; Devirtualization failed — qualify symbol
        (do (when (warn-boxed-dispatch?)
              (boxed-dispatch-warning fn-sym form
                                      (str "cannot resolve dispatch for arg types "
                                           (mapv #(or (when (symbol? %) (ctx-get-tag ctx %)) (type %))
                                                 (rest form)))))
            (let [v (ns-resolve source-ns fn-sym)
                  qsym (if v
                         (symbol (str (.name (.ns ^clojure.lang.Var v))) (str (.sym ^clojure.lang.Var v)))
                         fn-sym)]
              (cons qsym rewritten-args)))))))

;; ================================================================
;; Branch: array op auto-hinting
;; ================================================================

(defmethod walk-form :array-op [form ctx]
  (let [[f arr-sym & rest-args] form
        arr-tag (ctx-get-tag ctx arr-sym)
        hinted-arr (stamp-type-meta arr-sym arr-tag)
        core-f (symbol "clojure.core" (name f))
        walked-args (map #(walk % ctx) rest-args)
        ;; For aset on float[]: wrap value in (float ...) to ensure correct narrowing.
        ;; Without this, double-typed values stored into float[] would bypass the cast,
        ;; causing downstream SIMD to use DoubleVector on float arrays.
        aset? (contains? #{"aset"} (name f))
        walked-args (if (and aset? (= arr-tag 'floats) (= 2 (count walked-args)))
                      (let [[idx-arg val-arg] walked-args
                            already-float? (and (seq? val-arg) (= 'float (first val-arg)))]
                        (if already-float?
                          walked-args
                          [idx-arg (list 'float val-arg)]))
                      walked-args)]
    (apply list core-f hinted-arr walked-args)))

;; ================================================================
;; Branch: generic function call
;; ================================================================

;; ================================================================
;; If-form with occurrence typing — narrow types in branches
;; ================================================================

(def ^:private type-predicates
  "Map of type-checking predicates to the type they narrow to."
  {'instance?          :instance  ;; special: (instance? Class x)
   'clojure.core/instance? :instance
   'some?              {:narrow-to :non-nil}
   'clojure.core/some? {:narrow-to :non-nil}
   'nil?               {:narrow-to nil :else-narrow-to :non-nil}
   'clojure.core/nil?  {:narrow-to nil :else-narrow-to :non-nil}
   'number?            {:narrow-to 'Number}
   'clojure.core/number? {:narrow-to 'Number}
   'string?            {:narrow-to 'String}
   'clojure.core/string? {:narrow-to 'String}
   'double?            {:narrow-to 'double}
   'int?               {:narrow-to 'long}})

(clojure.core/defn- extract-type-guard
  "Extract type narrowing info from an if test expression.
  Returns {:var sym :then-tag tag :else-tag tag} or nil."
  [test-form type-env]
  (when (and (seq? test-form) (>= (count test-form) 2))
    (let [pred (first test-form)]
      (cond
        ;; (instance? ClassName x)
        (and (= :instance (get type-predicates pred))
             (= 3 (count test-form))
             (symbol? (nth test-form 2)))
        (let [cls-sym (second test-form)
              var-sym (nth test-form 2)]
          (when (inf/type-env-tag type-env var-sym)
            {:var var-sym :then-tag cls-sym}))

        ;; (predicate? x) — e.g. (number? x), (string? x)
        (and (get type-predicates pred)
             (= 2 (count test-form))
             (symbol? (second test-form)))
        (let [info (get type-predicates pred)
              var-sym (second test-form)]
          (when (and (map? info) (inf/type-env-tag type-env var-sym))
            (cond-> {:var var-sym}
              (:narrow-to info) (assoc :then-tag (:narrow-to info))
              (:else-narrow-to info) (assoc :else-tag (:else-narrow-to info)))))

        :else nil))))

(defmethod walk-form :if-form [form ctx]
  (let [[_ test then else] form
        walked-test (walk test ctx)
        guard (extract-type-guard test (:type-env ctx))
        ;; Narrow type-env for then-branch
        then-ctx (if-let [tag (:then-tag guard)]
                   (ctx-assoc-type ctx (:var guard) tag)
                   ctx)
        ;; Narrow type-env for else-branch
        else-ctx (if-let [tag (:else-tag guard)]
                   (ctx-assoc-type ctx (:var guard) tag)
                   ctx)
        walked-then (walk then then-ctx)
        walked-else (when else (walk else else-ctx))]
    (if walked-else
      (list 'if walked-test walked-then walked-else)
      (list 'if walked-test walked-then))))

;; ================================================================
;; Keyword field access → direct field access on typed records
;; (:health mob) → (double (.health mob)) when mob is a known record type
;; ================================================================

(defmethod walk-form :keyword-field-access [form ctx]
  (let [kw (first form)
        recv-sym (second form)
        recv-tag (ctx-get-tag ctx recv-sym)
        field-str (name kw)
        field-type (get-in @inf/field-type-registry [recv-tag field-str])
        ;; Build (.field receiver) form
        dot-field (symbol (str "." field-str))
        access-form (list dot-field recv-sym)]
    ;; Wrap with primitive cast if field type is known
    (if (and field-type (contains? types/primitive-info field-type))
      (list field-type access-form)
      access-form)))

(defmethod walk-form :call [form ctx]
  (let [type-env (:type-env ctx)
        source-ns (:source-ns ctx)
        [f & args] form
        ;; Qualify function symbols to their fully qualified namespace for downstream inlining.
        ;; Handles both unqualified symbols and namespace-aliased symbols (e.g. rev/value+grad).
        ;; Only qualify non-macro, non-special-form vars (macros like case/cond must stay unqualified
        ;; because the walker walks their raw args and cast-wrapping would corrupt macro semantics)
        f (if (and (symbol? f) (not (inf/type-env-tag type-env f)))
            (if-let [v (try (ns-resolve source-ns f) (catch Exception _ nil))]
              (if (and (var? v) (not (:macro (meta v))))
                (symbol (str (.name (.ns ^clojure.lang.Var v)))
                        (str (.sym ^clojure.lang.Var v)))
                f)
              f)
            f)
        f (or f (first form))  ;; fallback if ns-resolve returned nil
        static-call? (and (symbol? f) (some? (namespace f)))
        walked-args (map #(walk % ctx) args)
        cast-args (if static-call?
                    (map (fn [orig walked]
                           (let [tag (or (when (symbol? orig) (inf/type-env-tag type-env orig))
                                         (inf/literal-tag orig))]
                             (if (and tag (contains? types/primitive-info tag)
                                      ;; Don't re-wrap already-cast args
                                      (not (and (seq? walked)
                                                (contains? types/primitive-info (first walked)))))
                               (list tag walked)
                               walked)))
                         args walked-args)
                    walked-args)
        result (apply list (walk f ctx) cast-args)]
    result))

;; ================================================================
;; Branch: primitive cast (idempotent)
;; ================================================================

(defmethod walk-form :var-ref [form ctx]
  ;; (var foo) / #'foo — resolve to fully qualified, handling namespace aliases
  (let [sym (second form)
        resolved (if-let [v (try (ns-resolve (:source-ns ctx) sym) (catch Exception _ nil))]
                   (symbol (str (.name (.ns ^clojure.lang.Var v)))
                           (str (.sym ^clojure.lang.Var v)))
                   sym)]
    (list 'var resolved)))

(defmethod walk-form :cast [form ctx]
  ;; Primitive casts like (double x) — walk the inner form but don't add
  ;; another cast layer on re-walks (inner cast check is in :call handler).
  (let [[cast-sym inner] form]
    (list cast-sym (walk inner ctx))))

;; ================================================================
;; Branch: Clojure special forms
;; ================================================================

(defmethod walk-form :reify [form ctx]
  ;; (reify Interface1 (method1 [this x] body1) Interface2 (method2 [this y] body2) ...)
  ;; Walk method bodies but preserve interface names and method signatures.
  (let [[head & specs] form]
    (list* head
           (loop [remaining specs result []]
             (if-not (seq remaining)
               result
               (let [item (first remaining)]
                 (if (symbol? item)
                   ;; Interface name — keep as-is
                   (recur (rest remaining) (conj result item))
                   (if (and (seq? item) (symbol? (first item)) (vector? (second item)))
                     ;; Method: (name [params...] body...) — walk body only
                     (let [[mname params & body] item]
                       (recur (rest remaining)
                              (conj result (list* mname params (walk-forms body ctx)))))
                     ;; Unknown — pass through
                     (recur (rest remaining) (conj result item))))))))))

(defmethod walk-form :letfn [form ctx]
  ;; (letfn [(f [x] (g x)) (g [x] (f x))] body...)
  ;; All fn names are mutually in scope for all bodies.
  (let [[head bindings & body] form
        ;; Extract fn names from bindings: [[f [x] body] [g [x] body]] → [f g]
        fn-names (mapv first bindings)
        ;; Add all fn names to context (no type info, just present)
        letfn-ctx (reduce (fn [c n] (ctx-assoc-type c n 'Object)) ctx fn-names)
        ;; Walk each fn body with all names in scope
        walked-bindings (mapv (fn [binding]
                                (let [[fname & fn-body] binding]
                                  (vec (list* fname (walk-forms fn-body letfn-ctx)))))
                              bindings)
        walked-body (walk-forms body letfn-ctx)]
    (list* head walked-bindings walked-body)))

(defmethod walk-form :case [form ctx]
  ;; (case test-expr const1 result1 const2 result2 ... default?)
  ;; Walk test-expr and result exprs, but NOT dispatch constants.
  (let [[head test-expr & clauses] form
        walked-test (walk test-expr ctx)]
    (if (even? (count clauses))
      ;; No default: pairs of [const result]
      (list* head walked-test
             (mapcat (fn [[c r]] [c (walk r ctx)])
                     (partition 2 clauses)))
      ;; Has default: pairs + trailing default
      (let [pairs (butlast clauses)
            default (last clauses)]
        (list* head walked-test
               (concat (mapcat (fn [[c r]] [c (walk r ctx)])
                               (partition 2 pairs))
                       [(walk default ctx)]))))))

(defmethod walk-form :new [form ctx]
  ;; (new ClassName arg1 arg2 ...) — walk args but not class name
  (let [[head class-name & args] form]
    (list* head class-name (map #(walk % ctx) args))))

;; ================================================================
;; Branch: collections
;; ================================================================

(defmethod walk-form :vector [form ctx]
  (mapv #(walk % ctx) form))

(defmethod walk-form :map [form ctx]
  (into {} (map (fn [[k v]] [(walk k ctx) (walk v ctx)])) form))

(defmethod walk-form :set [form ctx]
  (into #{} (map #(walk % ctx)) form))

;; ================================================================
;; Branch: leaf (symbol / literal)
;; ================================================================

(defmethod walk-form :leaf [form ctx]
  (let [tag (when (symbol? form) (ctx-get-tag ctx form))]
    (if (and tag (types/array-tag? tag))
      (stamp-type-meta form tag)
      form)))

;; ================================================================
;; walk-body — convenience entry point for walking deftm bodies
;; ================================================================

(defn walk-body
  "Walk a deftm body, tracking types and rewriting:
   - Auto-adds type hints to let bindings via typedclojure inference
   - Devirtualizes calls to other deftm functions
   - Expands registered typed macros with type env
   - Emits .invk calls for (Fn ...) annotated parameters

  type-env: {symbol → {:tag dispatch-tag, :fn-info map?, :element dispatch-tag?}}
  tc-binding-tags: {symbol → dispatch-tag} pre-computed from TC (optional)"
  [form {:keys [type-env source-ns tc-binding-tags]}]
  (let [ctx (make-ctx {:type-env (or type-env {})
                       :source-ns (or source-ns *ns*)
                       :tc-binding-tags tc-binding-tags})]
    (walk form ctx)))

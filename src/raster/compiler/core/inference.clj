(ns raster.compiler.core.inference
  "Type inference, dispatch resolution, typed macros, and expansion helpers.

  This is the layer between dispatch and the walker. Contains:
  - TypedClojure-powered type inference
  - Literal/hint type inference
  - Field type registry
  - SoA registry helpers
  - Tag↔TC type mapping
  - Dispatch resolution (devirtualization)
  - muladd/broadcast/reduce/scan/stencil expansion
  - Typed macro registry
  - Public API functions used by the walker

  All functions that need type information accept a unified type-env:
    {sym → {:tag dispatch-tag, :fn-info map?, :element dispatch-tag?}}
  This is the walker's canonical type representation."
  (:require [clojure.string :as str]
            [clojure.core.typed :as t]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.tc-extensions]))

;; ================================================================
;; Type-env accessors
;; ================================================================

(defn type-env-tag
  "Look up the dispatch tag for a symbol in the type-env."
  [type-env sym]
  (get-in type-env [sym :tag]))

(defn type-env-fn-info
  "Look up fn-info for a symbol in the type-env."
  [type-env sym]
  (get-in type-env [sym :fn-info]))

(defn type-env-element
  "Look up the element type for an Object[] symbol in the type-env."
  [type-env sym]
  (get-in type-env [sym :element]))

;; ================================================================
;; Type inference (typedclojure-powered)
;; ================================================================

(def ^:private class->dispatch-tag
  "Map Java class to dispatch tag. Unboxes primitives, uses short name for others.
  This is the SINGLE canonical mapping — do not duplicate."
  {Long 'long, Double 'double, Integer 'int, Float 'float,
   Boolean 'boolean, Byte 'byte, Short 'short, Character 'char,
   String 'String, Number 'Number, Void 'void})

(defn- class-sym->dispatch-tag
  "Convert a fully-qualified class symbol to a dispatch tag.
  Unboxes java.lang.Double→double etc. For non-primitive classes,
  uses the fully-qualified name to ensure resolvability across namespaces.
  Only java.lang classes use simple names (auto-imported by Clojure)."
  [sym]
  (or (types/boxed->primitive-tag sym)  ;; Short names: Double→double, Long→long
      (try (let [cls (Class/forName (str sym))
                 simple (symbol (.getSimpleName cls))
                 fqn (symbol (.getName cls))]
             ;; Unbox the resolved simple name too
             ;; (java.lang.Double → Double → double)
             (or (types/boxed->primitive-tag simple)
                 ;; java.lang classes: use simple name (auto-imported by Clojure)
                 (when (.startsWith (.getName cls) "java.lang.") simple)
                 ;; All other classes: use fully-qualified name
                 fqn))
           (catch Exception _ nil))))

(def ^:private tc-alias->dispatch-tag
  "Map TC type aliases to dispatch tags. These are TC-specific names
  that can't be resolved as Java classes."
  {'typed.clojure/Int     'long
   'typed.clojure/AnyInteger 'long
   'typed.clojure/Num     'Number
   'typed.clojure/Str     'String
   'typed.clojure/Bool    'boolean})

(defn- tc-type->tag
  "Convert a TypedClojure internal type representation to a dispatch tag symbol.
  Uses class->dispatch-tag for systematic class resolution — no hardcoded
  per-type maps. Handles Value, RClass, Name, PrimitiveArray, Union."
  [t]
  (when t
    (let [type-name (.getSimpleName ^Class (type t))]
      (case type-name
        "PrimitiveArray"
        (let [jtype (:jtype t)]
          (if (= jtype java.lang.Object)
            'objects
            (get types/array-elem->tag
                 (when jtype (symbol (str jtype))))))

        "Value"
        (get class->dispatch-tag (class (:val t)))

        "RClass"
        (let [tc (:the-class t)
              base-tag (or (class-sym->dispatch-tag tc)
                           (symbol (name tc)))
              poly (:poly? t)]
          (if (and (seq poly)
                   (every? #(and (= "Value" (.getSimpleName ^Class (type %)))
                                 (number? (:val %)))
                           poly))
            (apply list base-tag (map :val poly))
            base-tag))

        "Name"
        (let [id (:id t)]
          (or (get tc-alias->dispatch-tag id)
              (class-sym->dispatch-tag id)))

        "Union"
        (let [tags (keep tc-type->tag (:types t))]
          (when (and (seq tags) (apply = tags)) (first tags)))

        ;; Unknown TC type representation — no tag available
        nil))))

;; ================================================================
;; Tag ↔ TC type form mapping
;; ================================================================

(def ^:private tag->tc-type
  {'long 'Long, 'double 'Double, 'int 'Integer, 'float 'Float,
   'boolean 'Boolean, 'byte 'Byte, 'short 'Short, 'char 'Character,
   'String 'String, 'Number 'Number,
   'doubles '(Array double), 'longs '(Array long), 'ints '(Array int),
   'floats '(Array float), 'bytes '(Array byte), 'shorts '(Array short),
   'chars '(Array char), 'booleans '(Array boolean), 'objects '(Array Object)})

(defn tag->tc [tag]
  "Convert a dispatch tag to a typedclojure type form."
  (if (types/compound-tag? tag)
    (types/compound-tag-base tag)
    (get tag->tc-type tag tag)))

(defn- extract-tc-binding-types
  "Walk a checked AST form to collect per-binding TC types.
  Returns {binding-sym → dispatch-tag} for all let bindings."
  [form]
  (let [btk (keyword "typed.clj.ext.clojure.core__let" "binding-types")]
    (when-let [bt (get (meta form) btk)]
      (reduce-kv (fn [m sym tcr]
                   ;; Skip uniquified names (foo__#0), keep original names
                   (if (re-find #"__#\d+$" (name sym))
                     m
                     ;; TCResult (from let) has :t field; raw types (from loop) don't.
                     ;; Also handle [Type FilterSet] vectors.
                     (let [t (cond
                               (vector? tcr) (first tcr)
                               ;; Try :t for TCResult; if it fails, tcr IS the type
                               :else (try (:t tcr)
                                          (catch Exception _ tcr)))
                           ;; Convert TC type to dispatch tag using systematic conversion.
                           ;; Symbol types (from TC's printed repr) use class-sym resolution.
                           ;; TC type objects use tc-type->tag.
                           tag (cond
                                 (nil? t) nil
                                 (symbol? t) (or (class-sym->dispatch-tag t)
                                                 (get tc-alias->dispatch-tag t))
                                 :else (tc-type->tag t))]
                       (if tag (assoc m sym tag) m))))
                 {} bt))))

(defn- collect-binding-types-from-ast
  "Recursively walk an AST form tree collecting binding types from all
   let and loop forms. TC attaches ::binding-types metadata to both
   let* (via core__let extension) and loop* (via check-let)."
  [form]
  (cond
    ;; let* and loop* — both may have TC binding-types metadata
    (and (seq? form) (or (form/binding-form? form)
                         (and (symbol? (first form))
                              (= "loop*" (name (first form))))))
    (let [local-types (extract-tc-binding-types form)
          child-types (reduce merge {}
                              (map collect-binding-types-from-ast (rest form)))]
      (merge child-types local-types))
    (seq? form)
    (reduce merge {} (map collect-binding-types-from-ast form))
    (vector? form)
    (reduce merge {} (map collect-binding-types-from-ast form))
    :else nil))

;; TC stamps each checked node's type under this key (= typed.cljc.checker.utils/expr-type).
(def ^:private tc-expr-type-key :typed.cljc.checker.utils/expr-type)

(defn- tcr->dispatch-tag
  "TCResult / raw type / [Type FilterSet] → raster dispatch tag, or nil."
  [tcr]
  (let [t (cond
            (vector? tcr) (first tcr)
            :else (try (:t tcr) (catch Exception _ tcr)))]
    (cond
      (nil? t) nil
      (symbol? t) (or (class-sym->dispatch-tag t) (get tc-alias->dispatch-tag t))
      :else (tc-type->tag t))))

(defn- collect-binding-types-from-checked-ast
  "Walk a TC :checked-ast, collecting {binding-sym → dispatch-tag} from the
   u/expr-type stamped on each :let / :loop binding node. This replaces reading
   ::binding-types form metadata: let*/loop* and (via Option X) the core/let
   extension all expose per-binding types directly on the binding AST nodes.

   Keyed by the original (de-uniquified) symbol, since the walker looks tags up by
   source name. A name may be bound in more than one scope with DIFFERENT types
   (shadowing — e.g. a macro like `broadcast` reusing a binding name for its loop
   element, scalar, over an outer array of the same name). Conflating those would
   mis-tag the outer binding, so a symbol with conflicting tags across scopes is
   dropped (left for the walker to infer per-scope); a symbol consistently typed
   everywhere keeps its tag."
  [ast]
  (let [acc (atom {})]                       ; sym → set of observed tags
    (letfn [(walk [x]
              (when (map? x)
                (when (#{:let :loop} (:op x))
                  (doseq [b (:bindings x)]
                    (let [sym (:form b)
                          tag (when-let [tcr (get b tc-expr-type-key)]
                                (tcr->dispatch-tag tcr))]
                      (when (and tag (symbol? sym))
                        (swap! acc update sym (fnil conj #{}) tag)))))
                (doseq [k (:children x)]
                  (let [v (get x k)]
                    (if (vector? v) (doseq [c v] (walk c)) (walk v))))))]
      (walk ast))
    (into {} (keep (fn [[sym tags]]
                     (when (= 1 (count tags)) [sym (first tags)]))
                   @acc))))

(defn- raster-ann->tc-type
  "Convert a Raster type annotation to a TypedClojure type form.
  (Fn [Double] Double) → [Double :-> Double]
  (Fn [(Array double) (Array double)]) → [(Array double) (Array double) :-> t/Any]
  Other annotations (Double, (Array double), etc.) pass through unchanged."
  [ann]
  (if (and (seq? ann) (= 'Fn (first ann)))
    (let [parts (rest ann)
          param-types (first parts)
          ret-type (if (> (count parts) 1)
                     (second parts)
                     'clojure.core.typed/Any)]
      (vec (concat (map raster-ann->tc-type param-types) [:-> (raster-ann->tc-type ret-type)])))
    ann))

(def ^:dynamic *tc-warn-on-error?*
  "When true, surface every reason Typed Clojure failed to produce binding types
  (an analysis exception, a TCError return type, or reported type-errors) instead
  of silently falling back to the walker's structural inference. Off by default:
  TC failure is non-fatal (devirtualization still proceeds via structural
  heuristics, just less precisely), but enabling this makes the failures visible
  for diagnosis — e.g. the source-ns resolution holes that silently discard all
  binding types and regress devirtualization."
  false)

(defn tc-analyze-deftm-body
  "Analyze a deftm body once with Typed Clojure.

  Returns a map with:
    :ret-tag           — inferred return type tag
    :stability-warning — warning if return type is unstable (Union)
    :binding-tags      — {sym → dispatch-tag} for all let bindings

  This gives the walker complete type information from a single TC pass.

  source-ns (optional): the namespace the body was written in. TC's analyzer
  resolves bare/refer'd vars (e.g. `pow` from raster.numeric) against *ns*, so at
  re-walk time (compile-aot / lazy JIT, which run in a different ns than the
  definition) *ns* must be bound to the source ns or every refer'd op fails to
  resolve — yielding a TCError return and discarding ALL inferred binding types."
  [fn-name params annotations body & [source-ns]]
  (try
    ;; Skip TC analysis for fully-untyped methods (all Object/Any params).
    ;; These are polymorphic fallbacks where TC can't add value — every operation
    ;; on Any will fail type checking, producing noise.
    (when (some some? annotations)
      (let [check-fn t/check-form-info
            fn-params (vec (mapcat (fn [p ann]
                                     (let [tc-type (if ann
                                                     (raster-ann->tc-type ann)
                                                     'clojure.core.typed/Any)]
                                       [p :- tc-type]))
                                   params annotations))
            body-expr (if (= 1 (count body)) (first body) (cons 'do body))
            wrapped (list 'clojure.core.typed/fn fn-params body-expr)
            ;; Resolve refer'd ops against the body's own namespace.
            result (binding [*ns* (if source-ns
                                    (if (instance? clojure.lang.Namespace source-ns)
                                      source-ns (the-ns source-ns))
                                    *ns*)]
                     (try (check-fn wrapped {:checked-ast true
                                             :check-config {:check-form-eval :never}})
                          (catch Throwable e1
                          ;; Fallback: retry without checked-ast if TC's internal
                          ;; compilation fails (e.g., primitive :tag on let bindings)
                            (try (check-fn wrapped {})
                                 (catch Throwable e2
                                   (println (str "WARNING: TypedClojure check failed for `" fn-name "`: "
                                                 (.getMessage e1) " → retry: " (.getMessage e2)))
                                   nil)))))]
        (when result
          (when (and (seq (:type-errors result))
                     (or *tc-warn-on-error?*
                         (some-> (resolve 'raster.core/*warn-on-boxed-dispatch*) deref)))
            (binding [*out* *err*]
              (println (str "TC type errors in `" fn-name "` (may cause boxed dispatch): "
                            (str/join "; " (map #(-> % ex-data :message (or (str %)))
                                                (:type-errors result)))))))
          (when (empty? (:type-errors result))
            (let [ret-type (:t (:ret result))]
              (when (and *tc-warn-on-error?*
                         (= "TCError" (some-> ret-type type .getSimpleName)))
                (binding [*out* *err*]
                  (println (str "[raster.tc] `" fn-name "` returned TCError — binding types "
                                "discarded, structural fallback used (check source-ns resolution)."))))
              (when-not (= "TCError" (some-> ret-type type .getSimpleName))
                (let [fn-int (first (:types ret-type))
                      f (first (:types fn-int))
                      rng (:t (:rng f))
                      type-name (some-> rng type .getSimpleName)
                      union-tags (when (= "Union" type-name)
                                   (keep tc-type->tag (:types rng)))
                      stability-warning (when (> (count union-tags) 1)
                                          (let [type-strs (str/join " | " (map name union-tags))]
                                            (str "WARNING: Type-unstable method `" fn-name "` "
                                                 (mapv name (types/extract-tags params annotations))
                                                 " may return: " type-strs
                                                 ". Consider adding :- RetType or fixing branches.")))
                    ;; Extract per-binding types straight off the checked AST nodes
                      binding-tags (when-let [ast (:checked-ast result)]
                                     (collect-binding-types-from-checked-ast ast))]
                  {:ret-tag (when rng (tc-type->tag rng))
                   :stability-warning stability-warning
                   :binding-tags (or binding-tags {})})))))))
    (catch Exception e
      ;; TC analysis failure — fall through to walker heuristics
      (when *tc-warn-on-error?*
        (binding [*out* *err*]
          (println (str "[raster.tc] analysis threw for `" fn-name "` — falling back to "
                        "structural inference: " (.getMessage e)))))
      nil)))

(defn safe-tc-binding-tags
  "Run TC binding-type analysis, returning {sym → tag} or nil on failure. TC
  failure is non-fatal (callers fall back to structural inference); set
  *tc-warn-on-error?* to surface the cause. Single entry point so every call
  site shares the same fallback + diagnostics policy."
  [fn-name params annotations body source-ns]
  (try
    (:binding-tags (tc-analyze-deftm-body fn-name params annotations body source-ns))
    (catch Throwable e
      (when *tc-warn-on-error?*
        (binding [*out* *err*]
          (println (str "[raster.tc] binding-tag analysis failed for `" fn-name "`: "
                        (.getMessage e)))))
      nil)))

(defn- desugar-invk-for-tc
  "Rewrite .invk forms to plain function calls for TC analysis.
  (.invk impl-sym arg1 arg2) → (impl-sym arg1 arg2)
  TC can type-check plain calls but not the .invk IR form."
  [form]
  (cond
    (seq? form)
    (let [head (first form)]
      (if (= '.invk head)
        ;; (.invk impl arg1 arg2 ...) → (impl arg1 arg2 ...)
        (let [impl (second form)
              args (nnext form)
              rewritten (apply list impl (map desugar-invk-for-tc args))]
          (if-let [m (meta form)] (with-meta rewritten m) rewritten))
        ;; Recurse into other forms
        (let [r (apply list (map desugar-invk-for-tc form))]
          (if-let [m (meta form)] (with-meta r m) r))))
    (vector? form) (mapv desugar-invk-for-tc form)
    :else form))

(defn tc-infer-binding-tags
  "Infer per-binding dispatch tags for an IR form using TypedClojure.
  param-env: {sym → dispatch-tag} for the function's parameters.
  form: the IR body (may contain .invk calls, raster.numeric/* etc.)

  Returns {sym → dispatch-tag} for all let bindings, or nil on failure.
  Used by the pipeline rewalk pass to type-check post-AD code."
  [param-env form]
  (try
    (let [check-fn t/check-form-info
          ;; Desugar .invk → plain calls so TC can analyze them
          tc-form (desugar-invk-for-tc form)
          ;; Build TC fn params from param-env
          fn-params (vec (mapcat (fn [[sym tag]]
                                   (let [;; Map dispatch tag to TC type form
                                         tc-type (get tag->tc-type tag tag)]
                                     [sym :- tc-type]))
                                 param-env))
          wrapped (list 'clojure.core.typed/fn fn-params tc-form)
          result (try (check-fn wrapped {:checked-ast true
                                         :check-config {:check-form-eval :never}})
                      (catch Throwable _
                        (try (check-fn wrapped {})
                             (catch Throwable _ nil))))]
      (when (and result (empty? (:type-errors result)))
        (when-let [ast (:checked-ast result)]
          (collect-binding-types-from-checked-ast ast))))
    (catch Throwable _ nil)))

;; ================================================================
;; Literal type inference
;; ================================================================

(def ^:private literal-type-tags
  ;; Use instance? checks instead of float? (which is true for both Float and Double)
  [[integer?                       'long]
   [#(instance? Float %)           'float]
   [#(instance? Double %)          'double]
   [string?                        'String]
   [boolean?                       'boolean]
   [keyword?                       'clojure.lang.Keyword]
   [char?                          'char]])

(defn literal-tag [form]
  (some (fn [[pred tag]] (when (pred form) tag)) literal-type-tags))

(defn hint-tag [form]
  (when-let [m (meta form)]
    (or (:raster.type/tag m)
        (:tag m))))

;; ================================================================
;; Field type registry
;; ================================================================

(defonce field-type-registry (atom {}))

(defn register-field-types!
  "Register field types for a record/value type."
  [type-tag fields]
  (swap! field-type-registry assoc type-tag fields))

(defn register-record-fields!
  "Auto-register field types for a defrecord class via Java reflection.
  Reads field names and type hints from the class's declared fields.
  Call after defrecord to enable keyword→field devirtualization."
  [^Class record-class]
  (let [type-tag (symbol (.getSimpleName record-class))
        fields (->> (.getDeclaredFields record-class)
                    (remove #(java.lang.reflect.Modifier/isStatic (.getModifiers ^java.lang.reflect.Field %)))
                    ;; Skip Clojure's internal defrecord fields
                    (remove #(.startsWith (.getName ^java.lang.reflect.Field %) "__"))
                    (reduce (fn [m ^java.lang.reflect.Field f]
                              (let [fname (.getName f)
                                    ftype (.getType f)
                                    tag (cond
                                          (= ftype Double/TYPE) 'double
                                          (= ftype Long/TYPE) 'long
                                          (= ftype Float/TYPE) 'float
                                          (= ftype Integer/TYPE) 'int
                                          (= ftype Boolean/TYPE) 'boolean
                                          (= ftype Byte/TYPE) 'byte
                                          (= ftype Short/TYPE) 'short
                                          (= ftype Character/TYPE) 'char
                                          (.isArray ftype) (symbol (.getSimpleName ftype))
                                          :else (symbol (.getName ftype)))]
                                (assoc m fname tag)))
                            {}))]
    (swap! field-type-registry assoc type-tag fields)
    fields))

;; ================================================================
;; SoA registry helpers
;; ================================================================

(defn register-soa!
  "Register an auto-generated SoA companion for a scalar defvalue type."
  [scalar-tag soa-tag fields]
  (swap! types/soa-registry assoc scalar-tag {:soa-type-tag soa-tag :fields fields})
  (swap! types/soa-reverse-registry assoc soa-tag scalar-tag))

(defn soa-field-eligible?
  "Returns true if ftag is a primitive or a recursively SoA-eligible defvalue type."
  [ftag]
  (or (contains? types/primitive->array-tag ftag)
      (boolean (when-let [nested (get @field-type-registry ftag)]
                 (every? (fn [[_n t]] (soa-field-eligible? t)) nested)))))

(defn soa-eligible?
  "Returns true if all field tags are primitive or recursively SoA-eligible."
  [field-types]
  (and (seq field-types)
       (every? (fn [[_name tag]] (soa-field-eligible? tag)) field-types)))

(defn flatten-one-field
  "Recursively flatten a field to primitive leaves."
  [flat-prefix ftag get-path]
  (if-let [arr-tag (get types/primitive->array-tag ftag)]
    [{:name flat-prefix :element-tag ftag :array-tag arr-tag :get-path get-path}]
    (when-let [nested-fields (get @field-type-registry ftag)]
      (vec (mapcat (fn [[fname2 ftag2]]
                     (let [sname2 (clojure.core/name fname2)]
                       (flatten-one-field (str flat-prefix "_" sname2)
                                          ftag2
                                          (conj get-path sname2))))
                   nested-fields)))))

(defn build-aget-expr
  "Build code to read a value from SoA at index."
  [ftag flat-prefix soa-sym i-sym]
  (if (get types/primitive->array-tag ftag)
    (if (#{'double 'float 'long 'int} ftag)
      `(~ftag (aget (~(symbol (str "." flat-prefix)) ~soa-sym) ~i-sym))
      `(aget (~(symbol (str "." flat-prefix)) ~soa-sym) ~i-sym))
    (let [nested-fields (get @field-type-registry ftag)]
      `(~(symbol (str "->" (clojure.core/name ftag)))
        ~@(map (fn [[fname2 ftag2]]
                 (let [sname2 (clojure.core/name fname2)]
                   (build-aget-expr ftag2 (str flat-prefix "_" sname2) soa-sym i-sym)))
               nested-fields)))))

;; ================================================================
;; Field type inference
;; ================================================================

(def ^:private defrecord-method-names
  #{"size" "count" "empty" "get" "values" "clear" "put" "remove" "replace"
    "merge" "equals" "hashCode" "toString" "meta" "seq" "iterator"
    "entrySet" "keySet" "containsKey" "containsValue" "isEmpty"})

(defn infer-field-type
  "Infer the type of a record field access.
  type-env: {sym → {:tag, :fn-info, :element}}"
  [form type-env]
  (let [[accessor receiver interop?]
        (cond
          (and (seq? form) (= 2 (count form))
               (symbol? (first form))
               (let [n (clojure.core/name (first form))]
                 (and (str/starts-with? n ".") (> (count n) 1))))
          (let [n (clojure.core/name (first form))
                field (if (str/starts-with? n ".-") (subs n 2) (subs n 1))]
            [field (second form) (not (str/starts-with? n ".-"))])

          (and (seq? form) (= 2 (count form)) (keyword? (first form)))
          [(clojure.core/name (first form)) (second form) false]

          :else nil)]
    (when (and accessor (symbol? receiver))
      (when-let [receiver-tag (type-env-tag type-env receiver)]
        (when-let [fields (get @field-type-registry receiver-tag)]
          (when-not (and interop?
                         (contains? defrecord-method-names accessor)
                         (try
                           (let [cls (types/resolve-type-ref->class receiver-tag)]
                             (and cls (.isAssignableFrom clojure.lang.IRecord cls)))
                           (catch Exception _ false)))
            (get fields accessor)))))))

;; ================================================================
;; Dispatch resolution (for devirtualization)
;; ================================================================

(defn generic-fn?
  "Check if sym resolves to a generic function. Uses bare resolve (caller must bind *ns*)."
  [sym]
  (when (symbol? sym)
    (when-let [v (resolve sym)]
      (:raster.core/generic-function (meta v)))))

(defn generic-fn?*
  "Like generic-fn? but resolves sym in the given namespace."
  [source-ns sym]
  (when (symbol? sym)
    (when-let [v (ns-resolve source-ns sym)]
      (:raster.core/generic-function (meta v)))))

(defn- get-dispatch-table [sym]
  (when-let [v (resolve sym)]
    (when-let [table-atom (:raster.core/dispatch-table (meta v))]
      @table-atom)))

(defn- tag-matches?
  ([inferred-tag ^Class method-cls]
   (tag-matches? inferred-tag method-cls nil))
  ([inferred-tag ^Class method-cls method-tag]
   (cond
     (nil? inferred-tag) true
     (= method-cls Object) true
     :else
     (let [inferred-cls (types/tag->check-class inferred-tag)
           class-ok? (and inferred-cls
                          (or (.isAssignableFrom method-cls inferred-cls)
                              (contains? (get types/numeric-widening inferred-cls) method-cls)))]
       (if (and class-ok? method-tag (types/compound-tag? method-tag))
         (if (types/compound-tag? inferred-tag)
           (= (types/compound-tag-params inferred-tag) (types/compound-tag-params method-tag))
           false)
         class-ok?)))))

(defn- resolve-method-entry-exact
  "Try to find an exact (or JVM-widening) match in the dispatch table."
  [methods arg-tags]
  (if (some nil? arg-tags)
    (let [matches (filterv (fn [entry]
                             (every? true? (map tag-matches? arg-tags
                                                (:check-classes entry)
                                                (:tags entry))))
                           methods)]
      (if (<= (count matches) 1)
        (first matches)
        (let [known-idxs (keep-indexed (fn [i t] (when t i)) arg-tags)]
          (if (empty? known-idxs)
            (when (= 1 (count matches)) (first matches))
            (let [dominated? (fn [entry]
                               (some (fn [other]
                                       (and (not (identical? entry other))
                                            (every? (fn [idx]
                                                      (let [ec (nth (:check-classes entry) idx)
                                                            oc (nth (:check-classes other) idx)]
                                                        (and (not (.equals ec oc))
                                                             (.isAssignableFrom ^Class ec oc))))
                                                    known-idxs)))
                                     matches))
                  refined (filterv (complement dominated?) matches)]
              (when (= 1 (count refined)) (first refined)))))))
    (some (fn [entry]
            (when (every? true? (map tag-matches? arg-tags
                                     (:check-classes entry)
                                     (:tags entry)))
              entry))
          methods)))

(def ^:private class->dispatch-tag*
  "Reverse of primitive-info: Class → dispatch tag symbol."
  (into {} (map (fn [[k v]] [v k]) types/primitive-info)))

(defn- promote-tag
  "Given two dispatch tag symbols, return the promoted dispatch tag using the
  canonical promotion registry (raster.types.promote). Returns nil if no rule exists.
  Uses requiring-resolve to avoid cyclic load dependency (promote.clj → core.clj → inference.clj)."
  [tag-a tag-b]
  (when (and tag-a tag-b (not= tag-a tag-b))
    (let [cls-a (get types/primitive-info tag-a)
          cls-b (get types/primitive-info tag-b)]
      (when (and cls-a cls-b)
        (when-let [promote-type-fn (requiring-resolve 'raster.types.promote/promote-type)]
          (when-let [result-cls (promote-type-fn cls-a cls-b)]
            (get class->dispatch-tag* result-cls)))))))

(defn- try-promote-tags
  "Try to promote scalar arg-tags to match a method entry (Julia-style).
  Uses the canonical promotion registry (raster.types.promote) to determine
  valid promotions. Only promotes when the promoted type matches the method's
  expected type. Returns {:entry method-entry, :casts [cast-tag-or-nil ...]} or nil."
  [methods arg-tags]
  (when (every? some? arg-tags)
    ;; Try each method entry, preferring most specific match
    (some
     (fn [entry]
       (let [method-tags (:tags entry)
             check-classes (:check-classes entry)]
          ;; Skip Object/Number fallback entries
         (when-not (some #{'Object 'Number} method-tags)
           (let [casts (mapv
                        (fn [arg-tag method-tag method-cls]
                          (cond
                              ;; Already matches exactly
                            (tag-matches? arg-tag method-cls method-tag) nil
                              ;; Promotable: check if [arg-tag, method-tag] promotes to method-tag
                            (= method-tag (promote-tag arg-tag method-tag))
                            method-tag  ;; cast arg to the method's expected type
                              ;; Not promotable to this method's type
                            :else ::no-match))
                        arg-tags method-tags check-classes)]
             (when-not (some #{::no-match} casts)
               {:entry entry :casts casts})))))
     methods)))

(defn- abstract-tag?
  "True when a dispatch tag is abstract (Number, Object) — i.e. a fallback
  that any concrete type would match. Promotion should be tried before
  accepting these."
  [tag]
  (contains? #{'Object 'Number} tag))

(defn- concrete-match?
  "True when a method entry's tags are all concrete (no abstract fallbacks).
  A concrete match is preferred over promotion."
  [entry]
  (not (some abstract-tag? (:tags entry))))

(defn- resolve-method-entry
  "Resolve a method entry from dispatch table.
  Preference order: concrete exact > promoted > abstract exact (Number/Object).
  Returns {:entry method-entry} for exact match,
  {:entry method-entry, :casts [cast-tag-or-nil ...]} for promoted match,
  or nil."
  [dispatch-table arg-tags]
  (let [arity (count arg-tags)]
    (when-let [methods (get dispatch-table arity)]
      ;; First try exact match (no promotion needed)
      (let [exact (resolve-method-entry-exact methods arg-tags)]
        (if (and exact (concrete-match? exact))
          ;; Good exact match with concrete types — no promotion needed
          {:entry exact}
          ;; Either no exact match, or only abstract fallback matched.
          ;; Try type promotion (Julia-style) before accepting the fallback.
          (or (try-promote-tags methods arg-tags)
              ;; Promotion failed — accept the abstract fallback if it exists
              (when exact {:entry exact})))))))

(defn try-parametric-specialize!
  "When a concrete dispatch entry is not found but a parametric template exists,
  trigger specialization and return the new entry. This is the Julia model:
  specialize on first use at compile time, not just at runtime."
  [fn-sym arg-tags]
  (when (every? some? arg-tags)
    ;; Build fake args with the right types for dispatch to unify against
    (let [fake-args (mapv (fn [tag]
                            (case tag
                              doubles (double-array 0)
                              floats  (float-array 0)
                              ints    (int-array 0)
                              longs   (long-array 0)
                              double  0.0
                              float   (float 0.0)
                              long    0
                              int     (int 0)
                              ;; For user-defined types, try to find a constructor
                              nil))
                          arg-tags)
          try-pd (requiring-resolve 'raster.compiler.core.dispatch/try-parametric-dispatch)]
      (when (and (every? some? fake-args) try-pd)
        (try
          (some? (try-pd fn-sym fake-args))
          (catch Exception _e nil))))))

(defn try-resolve-call
  "Try to resolve a deftm call to a direct mangled call.
  type-env: {sym → {:tag, :fn-info, :element}}

  When no concrete entry exists but arg types are known, triggers parametric
  specialization (Julia-style JIT specialization at compile time)."
  ([fn-sym arg-forms type-env]
   (try-resolve-call fn-sym arg-forms type-env nil))
  ([fn-sym arg-forms type-env extra-tags]
   (when-let [table (get-dispatch-table fn-sym)]
     (let [arg-tags (mapv (fn [form extra]
                            (or (when (symbol? form) (type-env-tag type-env form))
                                (literal-tag form)
                                (hint-tag form)
                                ;; Cast expressions: (double x), (long x), etc.
                                (when (and (seq? form) (contains? types/primitive-info (first form)))
                                  (first form))
                                extra))
                          arg-forms (or extra-tags (repeat nil)))
           result (or (resolve-method-entry table arg-tags)
                      ;; No concrete entry — try parametric specialization
                      (when (try-parametric-specialize! fn-sym arg-tags)
                        ;; Re-read table after specialization added the entry
                        (resolve-method-entry (get-dispatch-table fn-sym) arg-tags)))]
       (when result
         (let [entry (:entry result)
               casts (:casts result)
               mangled-name (str (types/mangle fn-sym (:tags entry)))
               mangled-sym (symbol mangled-name)
               v (resolve fn-sym)
               ns-str (str (:ns (meta v)))
               fq-mangled (symbol ns-str (str (types/mangle fn-sym (:tags entry))))
               resolved (or (resolve fq-mangled)
                            (ns-resolve *ns* mangled-sym)
                            (when-let [mns (:mangled-ns entry)]
                              (ns-resolve (the-ns mns) mangled-sym)))
               actual-sym (when resolved
                            (symbol (str (:ns (meta resolved)))
                                    (str (:name (meta resolved)))))]
           (when actual-sym
             (cond-> {:mangled-sym actual-sym
                      :tags (:tags entry)
                      :typed-impl (:typed-impl entry)
                      :typed-iface (:typed-iface entry)
                      :has-fn-params (boolean (:raster.core/has-fn-params (meta resolved)))
                      :return-tag (:raster.core/return-tag (meta resolved))}
               ;; Include promotion casts if any args need casting
               (and casts (some some? casts))
               (assoc :promotion-casts casts)))))))))

;; ================================================================
;; aget/rewritten-tag inference
;; ================================================================

(defn infer-aget-element-type
  "Infer the element type of an aget expression using element type from type-env."
  [form type-env]
  (when (and (seq? form) (descriptor/aget-op? (first form)) (>= (count form) 3))
    (let [arr-sym (second form)]
      (when (symbol? arr-sym)
        (type-env-element type-env arr-sym)))))

(defn infer-aget-type
  "Infer the result type of an aget expression.
  type-env: {sym → {:tag, :fn-info, :element}}"
  [form type-env]
  (when (and (seq? form)
             (descriptor/aget-op? (first form))
             (>= (count form) 3))
    (let [arr-sym (second form)]
      (when (symbol? arr-sym)
        (let [env-tag (type-env-tag type-env arr-sym)]
          (or (type-env-element type-env arr-sym)
              ;; Element dispatch tag carried on the symbol (stamped at the binding
              ;; site) — survives a re-walk whose env doesn't reach a captured var.
              (:raster.type/element (meta arr-sym))
              (get types/primitive-array-element-types env-tag)
              (get types/primitive-array-element-types (:tag (meta arr-sym)))
              (get @types/soa-reverse-registry env-tag)))))))

(defn infer-expr-tag
  "Infer the type tag of an arbitrary expression given type-env.
  Handles symbols, literals, casts, aget, and deftm calls recursively.
  Used by infer-binding-tag to determine arg types of compound expressions."
  [expr type-env source-ns]
  (cond
    ;; Symbol: look up in type-env
    (symbol? expr)
    (or (type-env-tag type-env expr)
        ;; Const var
        (when (namespace expr)
          (try (when-let [v (ns-resolve source-ns expr)]
                 (when (and (var? v) (not (fn? @v)))
                   (literal-tag @v)))
               (catch Exception _ nil))))
    ;; Literal
    (not (seq? expr)) (literal-tag expr)
    ;; Seq expression
    :else
    (let [head (first expr)]
      (cond
        ;; Cast: (double x) → double
        (contains? types/primitive-info head) head
        ;; aget on typed array
        (descriptor/aget-op? head)
        (infer-aget-type expr type-env)
        ;; Clojure core fns with known long return type
        (contains? '#{clojure.core/quot clojure.core/rem clojure.core/mod
                      clojure.core/bit-and clojure.core/bit-or clojure.core/bit-xor
                      clojure.core/bit-not clojure.core/bit-shift-left
                      clojure.core/bit-shift-right clojure.core/unsigned-bit-shift-right
                      clojure.core/unchecked-add clojure.core/unchecked-subtract
                      clojure.core/unchecked-multiply clojure.core/unchecked-negate
                      clojure.core/unchecked-inc clojure.core/unchecked-dec
                      clojure.core/count clojure.core/alength
                      quot rem mod bit-and bit-or bit-xor bit-not
                      bit-shift-left bit-shift-right unsigned-bit-shift-right
                      count alength}
                   head)
        'long
        ;; deftm generic call: resolve dispatch and get return tag
        (and (symbol? head) (generic-fn?* source-ns head))
        (let [arg-tags (mapv #(infer-expr-tag % type-env source-ns) (rest expr))]
          (when-let [resolved (binding [*ns* source-ns]
                                (try-resolve-call head (rest expr) type-env arg-tags))]
            (let [v (resolve (:mangled-sym resolved))]
              (when v
                (let [ret (or (:raster.core/return-tag (meta v))
                              (:tag (meta v)))]
                  (when (and ret (not= ret 'Object)) ret))))))
        ;; Unknown
        :else nil))))

(declare infer-rewritten-tag)

(defn- prim-class->tag
  "Map a (boxed or primitive) numeric Class to a raster type tag, else nil."
  [c]
  (condp = c
    Double/TYPE 'double Float/TYPE 'float Long/TYPE 'long Integer/TYPE 'int
    java.lang.Double 'double java.lang.Float 'float java.lang.Long 'long java.lang.Integer 'int
    nil))

(defn static-method-return-tag
  "Return tag of a Java static-method call (Class/method args…) via reflection,
   resolving overloads by the args' inferred tags. nil if the class/method can't be
   resolved or the overload stays ambiguous. This lets the inference type interop
   like (Math/floor x) → double, so arithmetic over it devirtualizes AT WALK TIME
   (and thus carries :raster.type/tag), rather than falling back to a later,
   weaker-typed resolution. TC reflects the same return type into :tag; this is the
   raster-side equivalent that feeds :raster.type/tag."
  [head args type-env]
  (when (and (symbol? head) (namespace head))
    (when-let [cls (try (resolve (symbol (namespace head))) (catch Throwable _ nil))]
      (when (class? cls)
        (let [mn (name head)
              n  (count args)
              cand (filter (fn [^java.lang.reflect.Method m]
                             (and (= mn (.getName m))
                                  (java.lang.reflect.Modifier/isStatic (.getModifiers m))
                                  (= n (.getParameterCount m))))
                           (.getMethods cls))
              rets (distinct (map (fn [^java.lang.reflect.Method m] (prim-class->tag (.getReturnType m))) cand))]
          (cond
            ;; unambiguous (e.g. floor, sqrt) — every overload returns the same prim
            (and (= 1 (count rets)) (first rets)) (first rets)
            ;; overloaded (e.g. abs, min, max) — pick by the args' inferred tags
            (seq cand)
            (let [atags (mapv (fn [a] (or (when (symbol? a) (type-env-tag type-env a))
                                          (literal-tag a)
                                          (when (seq? a) (infer-rewritten-tag a nil type-env))))
                              args)]
              (some (fn [^java.lang.reflect.Method m]
                      (when (= atags (mapv prim-class->tag (.getParameterTypes m)))
                        (prim-class->tag (.getReturnType m))))
                    cand))
            :else nil))))))

(defn infer-rewritten-tag
  "Infer the return type tag of a rewritten expression.
  type-env: {sym → {:tag, :fn-info, :element}} (optional, for dispatched calls)"
  ([rewritten-form]
   (infer-rewritten-tag rewritten-form nil nil))
  ([rewritten-form original-form type-env]
   (when (seq? rewritten-form)
     (let [head (first rewritten-form)]
       (or
        (cond
           ;; Primitive cast — (double x), (long x), etc.
          (contains? types/primitive-info head)
          head

           ;; if/when expression — result type = numeric unification of the
           ;; branch types (widen mismatched numeric branches to the larger).
           ;; MUST precede the symbol-head clause below, which would otherwise
           ;; match 'if (a symbol) and short-circuit to nil. Without this, a
           ;; local bound to (if …) is untyped → downstream mixed-precision
           ;; arithmetic on it (double gc × float diff) falls to the boxing
           ;; variadic dispatch (~96x Valhalla / ~888x GraalVM in float kernels).
          (= 'if head)
          (let [branch-tag (fn [b] (or (infer-rewritten-tag b nil type-env)
                                       (literal-tag b)
                                       (when (symbol? b) (type-env-tag type-env b))))
                then-tag (branch-tag (nth rewritten-form 2 nil))
                else-tag (when (>= (count rewritten-form) 4)
                           (branch-tag (nth rewritten-form 3)))
                rank '{int 1 long 2 float 3 double 4}]
            (cond
              (nil? else-tag) then-tag
              (nil? then-tag) else-tag
              (= then-tag else-tag) then-tag
              (and (rank then-tag) (rank else-tag))
              (if (>= (rank then-tag) (rank else-tag)) then-tag else-tag)
              :else (or (promote-tag then-tag else-tag) then-tag else-tag)))

           ;; Devirtualized call (a var) — or a Java static method (Class/method)
          (and (symbol? head) (not= '.invk head))
          (or (when-let [v (try (resolve head) (catch Throwable _ nil))]
                (or (:raster.core/return-tag (meta v))
                    (:tag (meta v))))
              (static-method-return-tag head (rest rewritten-form) type-env))

           ;; .invk call
          (and (= '.invk head) (symbol? (second rewritten-form)))
          (let [impl-sym (second rewritten-form)
                impl-name (name impl-sym)
                impl-ns (namespace impl-sym)]
            (or
               ;; Check impl-sym metadata first (fn-param .invk has :raster.type/ret-tag)
             (:raster.type/ret-tag (meta impl-sym))
               ;; Resolve from method var (mangled deftm -impl convention)
             (when (and impl-ns (.endsWith ^String impl-name "-impl"))
               (let [method-name (subs impl-name 0 (- (count impl-name) 5))
                     method-var (resolve (symbol impl-ns method-name))]
                 (when method-var
                   (or (:raster.core/return-tag (meta method-var))
                       (:tag (meta method-var))))))))

           ;; Dispatched call (original form with known arg types)
          (and original-form type-env (seq? original-form) (symbol? (first original-form)))
          (when-let [table (get-dispatch-table (first original-form))]
            (let [arg-tags (mapv (fn [a] (or (when (symbol? a) (type-env-tag type-env a))
                                             (literal-tag a)))
                                 (rest original-form))]
              (when-let [result (resolve-method-entry table arg-tags)]
                (let [entry (:entry result)
                      mangled (types/mangle (first original-form) (:tags entry))
                      v (resolve (symbol (str mangled)))]
                  (when v
                    (or (:raster.core/return-tag (meta v))
                        (:tag (meta v))))))))

          :else nil)
         ;; Fallback: form-level :raster.type/tag metadata (set by walker handlers,
         ;; e.g. par/reduce sets {:raster.type/tag acc-tag} on the walked result)
        (:raster.type/tag (meta rewritten-form)))))))

;; ================================================================
;; muladd transform
;; ================================================================

(defn muladd-transform [form]
  (cond
    (not (seq? form)) form

    (and (= '+ (first form)) (= 3 (count form)))
    (let [[_ x y] form
          x (muladd-transform x)
          y (muladd-transform y)]
      (cond
        (and (seq? y) (= '* (first y)) (= 3 (count y)))
        (let [[_ b c] y] (list 'Math/fma b c x))
        (and (seq? x) (= '* (first x)) (= 3 (count x)))
        (let [[_ b c] x] (list 'Math/fma b c y))
        :else (list '+ x y)))

    (and (= '+ (first form)) (> (count form) 3))
    (let [args (map muladd-transform (rest form))]
      (reduce (fn [acc arg]
                (if (and (seq? arg) (= '* (first arg)) (= 3 (count arg)))
                  (let [[_ b c] arg] (list 'Math/fma b c acc))
                  (list '+ acc arg)))
              (first args) (rest args)))

    (and (= '- (first form)) (= 3 (count form)))
    (let [[_ x y] form
          x (muladd-transform x)
          y (muladd-transform y)]
      (if (and (seq? y) (= '* (first y)) (= 3 (count y)))
        (let [[_ b c] y] (list 'Math/fma (list '- b) c x))
        (list '- x y)))

    :else
    (apply list (first form) (map muladd-transform (rest form)))))

;; ================================================================
;; broadcast/reduce/scan/stencil expansion
;; ================================================================

(defn- parse-keyword-opts [forms]
  (loop [opts {} remaining (seq forms)]
    (if (and remaining (keyword? (first remaining)) (next remaining))
      (recur (assoc opts (first remaining) (second remaining))
             (nnext remaining))
      [opts (vec remaining)])))

(def ^:private array-tag->alloc-fn
  "Array type tag → constructor symbol for sized allocation."
  {'doubles 'clojure.core/double-array
   'floats  'clojure.core/float-array
   'longs   'clojure.core/long-array
   'ints    'clojure.core/int-array})

(defn infer-body-element-type
  "Scan body expression for array-typed symbols in type-env.
  Returns the first array tag found, or nil. This lets par/map infer
  the output type from the body context -- like Futhark's type inference."
  [body type-env]
  (cond
    (symbol? body)
    (let [tag (type-env-tag type-env body)]
      (when (and tag (types/array-tag? tag)) tag))

    (sequential? body)
    (some #(infer-body-element-type % type-env) (rest body))

    :else nil))

(defn expand-par-map
  "Expand (raster.par/map [i n] body) into allocation + par/map!.
  Functional: returns a NEW array. Output allocation is handled by the compiler.

  (par/map [i n] body) → allocate out[n], par/map! out i n cast body

  The output element type is inferred from array-typed symbols in the body
  (like Futhark's type inference). Falls back to double[]."
  [binding body type-env {:keys [like]}]
  (let [[i-sym bound-expr] binding
        like-tag (when like (type-env-tag type-env like))
        inferred-tag (infer-body-element-type body type-env)
        out-tag (or like-tag inferred-tag)
        _ (when-not out-tag
            (throw (ex-info (str "par/map: cannot infer output element type. "
                                 "Use :like or ensure array-typed symbols appear in body.")
                            {:body body})))
        alloc-fn (or (get array-tag->alloc-fn out-tag)
                     (throw (ex-info (str "par/map: no alloc-fn for array tag `" out-tag "`")
                                     {:out-tag out-tag})))
        store-cast (types/array-tag->cast out-tag)
        n-sym (gensym "n__")
        out-sym (gensym "out__")]
    (list 'let* [n-sym bound-expr
                 out-sym (list alloc-fn n-sym)]
          (list 'raster.par/map! out-sym i-sym n-sym store-cast body))))

(defn expand-broadcast
  "Expand broadcast into a pure par/map form. Functional: all listed symbols
  are INPUT arrays. The output array is generated by the materialize pass.
  Returns a form that produces a new array.

  (broadcast [x y] (+ x y))  → (raster.par/map i (alength x) cast (+ (aget x i) (aget y i)))

  The type-env is used for determining the output array's cast type
  (double vs float) from the first input array's type."
  [syms expr type-env {:keys [offset length]}]
  (let [array-sym-set (set syms)
        ;; Functional: output is a fresh array, same type as first input
        first-input (first syms)
        out-tag (or (type-env-tag type-env first-input) 'doubles)
        store-cast (types/array-tag->cast out-tag)
        compute-cast (types/array-tag->compute-cast out-tag)
        i-sym (gensym "i__")
        off-sym (when offset (gensym "off__"))
        rewrite (fn rewrite [form]
                  (cond
                    (and (symbol? form) (contains? array-sym-set form))
                    (let [aget-expr (list 'aget form (if off-sym
                                                       (list 'unchecked-add-int off-sym i-sym)
                                                       i-sym))]
                      (if (and compute-cast (not= compute-cast store-cast))
                        (list compute-cast aget-expr)
                        aget-expr))
                    (and (number? form) compute-cast (not= compute-cast 'double))
                    (list compute-cast form)
                    (seq? form) (apply list (map rewrite form))
                    (vector? form) (mapv rewrite form)
                    :else form))
        rewritten-expr (rewrite expr)
        len-expr (or length (list 'alength first-input))]
    (if off-sym
      ;; Offset broadcast: still needs explicit allocation + dotimes (imperative pattern)
      (let [out-sym (gensym "out__")
            n-sym (gensym "n__")
            idx-expr (list 'unchecked-add-int off-sym i-sym)
            aset-expr (if store-cast (list store-cast rewritten-expr) rewritten-expr)]
        (list 'let* [out-sym (list 'raster.numeric/similar first-input)
                     off-sym (list 'int offset) n-sym (list 'int len-expr)]
              (list 'dotimes [i-sym n-sym]
                    (list 'aset out-sym idx-expr aset-expr))
              out-sym))
      ;; Non-offset: emit pure par/map form.
      ;; The materialize pass will allocate the output buffer later.
      (list 'raster.par/map [i-sym len-expr] rewritten-expr))))

(defn expand-reduce
  "Expand reduce! into a par/reduce loop. ALL listed symbols are treated
  as arrays. Scalars should not be listed — use them directly in the body."
  [acc-binding syms body type-env {:keys [offset length]}]
  (let [[acc-sym init-expr] acc-binding
        array-sym-set (set syms)
        first-arr (first syms)
        first-tag (or (type-env-tag type-env first-arr) 'doubles)
        compute-type (types/array-tag->compute-cast first-tag)
        i-sym (gensym "i__")
        n-sym (gensym "n__")
        off-sym (when offset (gensym "off__"))
        rewrite (fn rewrite [form]
                  (cond
                    (and (symbol? form) (contains? array-sym-set form))
                    (let [idx (if off-sym (list 'unchecked-add-int off-sym i-sym) i-sym)
                          aget-expr (list 'aget form idx)]
                      (if (and compute-type (not= compute-type 'double))
                        (list compute-type aget-expr)
                        aget-expr))
                    (and (number? form) compute-type (not= compute-type 'double))
                    (list compute-type form)
                    (seq? form) (apply list (map rewrite form))
                    (vector? form) (mapv rewrite form)
                    :else form))
        rewritten-body (rewrite body)
        rewritten-init (rewrite init-expr)
        len-expr (or length (list 'alength first-arr))]
    (if off-sym
      (list 'let* [off-sym (list 'int offset) n-sym (list 'int len-expr)]
            (list 'loop [i-sym 0 acc-sym rewritten-init]
                  (list 'if (list '< i-sym n-sym)
                        (list 'recur (list 'inc i-sym) rewritten-body)
                        acc-sym)))
      (list 'raster.par/reduce acc-sym rewritten-init i-sym len-expr rewritten-body))))

(defn expand-scan
  "Expand scan into a par/scan loop. Functional: all listed symbols are
  INPUT arrays. Output array is generated internally by the compiler.
  Returns a form that produces a new array."
  [acc-binding syms body type-env {:keys [offset length]}]
  (let [[acc-sym init-expr] acc-binding
        array-sym-set (set syms)
        first-arr (first syms)
        out-sym (gensym "scan_out__")
        out-tag (or (type-env-tag type-env first-arr) 'doubles)
        cast-fn (types/array-tag->cast out-tag)
        i-sym (gensym "i__")
        rewrite (fn rewrite [form]
                  (cond
                    (and (symbol? form) (contains? array-sym-set form))
                    (list 'aget form i-sym)
                    (seq? form) (apply list (map rewrite form))
                    (vector? form) (mapv rewrite form)
                    :else form))
        rewritten-body (rewrite body)
        len-expr (or length (list 'alength first-arr))]
    (list 'let* [out-sym (list 'raster.numeric/similar first-arr)]
          (list 'raster.par/scan out-sym acc-sym init-expr i-sym len-expr cast-fn rewritten-body))))

(defn expand-stencil
  "Expand stencil! into a par/stencil! form. ALL listed symbols are arrays.
  First listed symbol is the output array."
  [syms body type-env {:keys [radius boundary] :or {radius 1 boundary :dirichlet}}]
  (let [out-sym (first syms)
        out-tag (or (type-env-tag type-env out-sym) 'doubles)
        cast-fn (types/array-tag->cast out-tag)
        in-arrays (vec (rest syms))
        i-sym 'i]
    (list 'raster.par/stencil! out-sym in-arrays radius boundary cast-fn
          i-sym (list 'alength out-sym) body)))

;; ================================================================
;; Typed macro registry
;; ================================================================

(defonce typed-macros (atom {}))

(defn register-typed-macro!
  "Register a typed macro that will be expanded inside deftm bodies.
  expand-fn receives (form, type-env) where type-env is
  {sym → {:tag dispatch-tag, :fn-info map?, :element dispatch-tag?}}.
  Use (type-env-tag type-env sym) to look up dispatch tags."
  [sym expand-fn]
  (swap! typed-macros assoc sym expand-fn)
  sym)

(defn typed-macro? [sym]
  (contains? @typed-macros sym))

(defn get-typed-macro [sym]
  (get @typed-macros sym))

;; Register built-in typed macros
;; Expand-fns receive (form, type-env) where type-env is {sym → {:tag, :fn-info, :element}}
;; par/map is now a first-class walker form (:par-map-pure), not a typed macro.
;; expand-par-map is still used by the materialize pass.

(register-typed-macro! 'broadcast
                       (fn [form type-env]
                         (let [[_ syms & rest-args] form
                               [opts [expr]] (parse-keyword-opts rest-args)]
                           (expand-broadcast syms expr type-env opts))))

(register-typed-macro! 'reduce!
                       (fn [form type-env]
                         (let [[_ acc-binding syms & rest-args] form
                               [opts [body]] (parse-keyword-opts rest-args)]
                           (expand-reduce acc-binding syms body type-env opts))))

(register-typed-macro! 'scan
                       (fn [form type-env]
                         (let [[_ out-acc-binding syms & rest-args] form
                               [opts [body]] (parse-keyword-opts rest-args)]
                           (expand-scan out-acc-binding syms body type-env opts))))

(register-typed-macro! 'stencil!
                       (fn [form type-env]
                         (let [[_ syms & rest-args] form
                               [opts [body]] (parse-keyword-opts rest-args)]
                           (expand-stencil syms body type-env opts))))

(register-typed-macro! 'muladd
                       (fn [form _type-env]
                         (let [body (rest form)
                               transformed (map muladd-transform body)]
                           (if (= 1 (count transformed))
                             (first transformed)
                             (list* 'do transformed)))))

;; ================================================================
;; Array constructor inference
;; ================================================================

(def ^:private alloc-sym->array-tag
  "Array constructor symbol -> array type tag (from centralized registry)."
  descriptor/alloc-sym->array-tag)

;; ================================================================
;; Compound inference helpers (used by walker)
;; ================================================================

(def ^:dynamic *trace-inference*
  "When true, infer-binding-tag prints provenance information for each
  inferred type. Useful for debugging inference failures."
  false)

(defn- trace-inferred [sym tag provenance]
  (when *trace-inference*
    (println (str "  INFER " sym " → " tag " [" provenance "]")))
  tag)

(defn infer-binding-tag
  "Infer the dispatch tag for a let binding.
  type-env: {sym → {:tag, :fn-info, :element}}

  TC handles most type inference via whole-body analysis (tc-binding-tags
  checked first by the walker). This function covers:
    - Cases TC can't handle (field-type-registry, fn-info, element types,
      compound shapes)
    - Cheap structural fallbacks for when TC didn't run (parametric
      functions, TC errors)"
  [sym init rewritten-init type-env {:keys [source-ns]}]
  (or
   ;; Explicit ^tag metadata — user override, authoritative
   (when-let [t (hint-tag sym)]
     (trace-inferred sym t :hint))
   ;; Literal type — TC covers this, but free and needed for parametric fns
   (when-let [t (literal-tag init)]
     (trace-inferred sym t :literal))
   ;; Symbol alias — propagate known type
   (when-let [t (when (symbol? init) (type-env-tag type-env init))]
     (trace-inferred sym t :env))
   ;; Object[] element type — TC returns Object for these
   (when-let [t (infer-aget-element-type init type-env)]
     (trace-inferred sym t :element))
   ;; Typed array aget — TC covers, but needed for parametric fns
   (when-let [t (infer-aget-type init type-env)]
     (trace-inferred sym t :aget))
   ;; Record field type — TC doesn't know field-type-registry
   (when-let [t (infer-field-type init type-env)]
     (trace-inferred sym t :field))
   ;; Primitive cast — cheap structural check
   (when-let [t (when (and (seq? init) (contains? types/primitive-info (first init)))
                  (first init))]
     (trace-inferred sym t :cast))
   ;; Array allocation — cheap structural check
   (when-let [t (when (and (seq? init) (contains? alloc-sym->array-tag (first init)))
                  (get alloc-sym->array-tag (first init)))]
     (trace-inferred sym t :array-alloc))
   ;; if/when result — type from the (walked) branches (see infer-rewritten-tag).
   ;; A local bound to (if …) is otherwise untyped, forcing downstream
   ;; mixed-precision arithmetic onto the boxing variadic path.
   (when-let [t (when (and (seq? rewritten-init) (= 'if (first rewritten-init)))
                  (infer-rewritten-tag rewritten-init init type-env))]
     (trace-inferred sym t :if))
   ;; Fn-info return type — TC doesn't know walker fn-info
   (when-let [t (when (and (seq? init) (symbol? (first init)))
                  (when-let [info (type-env-fn-info type-env (first init))]
                    (let [ret (:ret-tag info)]
                      (when (and ret (not= ret 'void) (not= ret 'Object)) ret))))]
     (trace-inferred sym t :fn-ret))
   ;; Deftm call return type — needed during incremental walk for typed
   ;; macros (broadcast!, reduce!) that run mid-walk and need to know
   ;; which bindings are arrays. TC can't help here because it runs
   ;; before the walker, not during the incremental let-binding walk.
   (when-let [t (when (and (seq? init) (symbol? (first init)))
                  (let [fn-sym (first init)
                        src-ns (or source-ns *ns*)]
                    (when (generic-fn?* src-ns fn-sym)
                      (let [arg-tags (mapv (fn [a]
                                             (infer-expr-tag a type-env src-ns))
                                           (rest init))]
                        (when-let [resolved (binding [*ns* src-ns]
                                              (try-resolve-call fn-sym (rest init) type-env arg-tags))]
                          (let [v (resolve (:mangled-sym resolved))]
                            (when v
                              (let [ret (or (:raster.core/return-tag (meta v))
                                            (:tag (meta v)))]
                                (when (and ret (not= ret 'Object)) ret)))))))))]
     (trace-inferred sym t :deftm-return))
   ;; .invk return type — check both rewritten-init (walker just devirtualized)
   ;; and init (fixpoint re-walk of already-devirtualized code).
   (when-let [t (let [invk-form (cond
                                  (and (seq? rewritten-init) (= '.invk (first rewritten-init)))
                                  rewritten-init
                                  (and (seq? init) (= '.invk (first init)))
                                  init
                                  :else nil)]
                  (when (and invk-form (symbol? (second invk-form)))
                    (let [impl-sym (second invk-form)
                          ret-from-meta (:raster.type/ret-tag (meta impl-sym))]
                      (or (when (and ret-from-meta (not= ret-from-meta 'Object))
                            ret-from-meta)
                          (let [v (try (ns-resolve (or source-ns *ns*) impl-sym)
                                       (catch Exception _ nil))]
                            (when v
                              (let [ret (or (:raster.core/return-tag (meta v))
                                            (:tag (meta v)))]
                                (when (and ret (not= ret 'Object)) ret))))))))]
     (trace-inferred sym t :invk-return))
   ;; Mangled deftm call return type — after fixpoint re-walks AD-generated code,
   ;; calls may be already-mangled (e.g. raster.arrays/zeros-like_m_floats_long)
   ;; rather than generic names or .invk forms. Look up the var's return-tag directly.
   (when-let [t (when (and (seq? init) (symbol? (first init)))
                  (let [fn-sym (first init)
                        src-ns (or source-ns *ns*)]
                    (when-not (generic-fn?* src-ns fn-sym)
                      (let [v (try (ns-resolve src-ns fn-sym) (catch Exception _ nil))]
                        (when (and v (var? v))
                          (let [ret (or (:raster.core/return-tag (meta v))
                                        (:tag (meta v)))]
                            (when (and ret (not= ret 'Object)) ret)))))))]
     (trace-inferred sym t :mangled-return))
   ;; Compound shape inference — Raster-specific shape lattice
   (when-let [t (when (seq? init)
                  (let [arg-tags (mapv (fn [a]
                                         (cond (symbol? a) (type-env-tag type-env a)
                                               :else (literal-tag a)))
                                       (rest init))]
                    (when (some types/compound-tag? arg-tags)
                      (types/infer-shape-from-rules (first init) arg-tags))))]
     (trace-inferred sym t :shape-rules))))

(defn infer-element-tag
  "Infer the element type for a let binding (for Object[] parametric dispatch).
  Returns the element tag to store, or nil if not applicable.
  Only propagates element types from direct aliasing or aget — never inherits
  from unrelated type-env entries."
  [init tag type-env]
  (cond
    ;; Direct alias: x = y where y has an element type
    (and (symbol? init) (type-env-element type-env init))
    (type-env-element type-env init)
    ;; aget on typed Object[]: element type from the array's element
    (and (= tag 'objects) (infer-aget-element-type init type-env))
    (infer-aget-element-type init type-env)
    :else nil))

(defn compute-binding-hint
  "Compute the :tag metadata hint for a let binding, or nil."
  [tag sym]
  (when (and tag (not (hint-tag sym)))
    (case tag
      ;; Primitive arrays: always hint (Clojure needs these for aget/aset)
      (doubles longs ints floats bytes shorts chars booleans) tag
      ;; Bare primitives: never hint (conflicts with primitive initializers)
      (long double int float boolean byte short char) nil
      ;; Boxed primitives: never hint (conflicts with primitive initializers
      ;; in loop bindings like [best-val (double expr)])
      (Double Long Float Integer Boolean Byte Short Character
              java.lang.Double java.lang.Long java.lang.Float java.lang.Integer
              java.lang.Boolean java.lang.Byte java.lang.Short java.lang.Character
              Number) nil
      ;; Reference types: resolve to class name
      (let [cls (types/tag->check-class tag)]
        (if (and cls (not= cls Object))
          (symbol (.getName ^Class cls))
          tag)))))

;; ================================================================
;; Type environment construction — single source of truth
;; ================================================================

(defn build-param-type-env
  "Build a walker type-env from deftm parameter annotations.
  Returns {sym → {:tag dispatch-tag, :element element-tag?, :fn-info map?}}.

  This is the SINGLE source of truth for param→type-env construction.
  Used by: prepare-typed-body (core.clj), walk-body-with-tc (pipeline.clj),
  pe-rewalk (rewalk.clj). All three MUST use this function — divergent
  type-env construction causes devirtualization inconsistencies."
  ([params tags]
   (build-param-type-env params tags nil))
  ([params tags annotations]
   (reduce (fn [te [p ann tag]]
             (let [;; Extract element type from (Array Class) annotations
                   element (when (and (sequential? ann) (= 'Array (first ann)))
                             (let [elem (second ann)]
                               (when (and elem (symbol? elem) (not= elem 'Object)
                                          (not (contains? types/primitive-array-element-types elem)))
                                 elem)))
                   ;; Extract fn-info from (Fn [...]) annotations
                   fn-info (when ann (types/annotation->fn-info ann))]
               (cond-> te
                 (or (and tag (not= tag 'Object)) element fn-info)
                 (assoc (with-meta p nil)
                        (cond-> {:tag (or tag 'Object)}
                          element (assoc :element element)
                          fn-info (assoc :fn-info fn-info))))))
           {} (map vector params
                   (or annotations (repeat nil))
                   (or tags (repeat 'Object))))))

;; ================================================================
;; Symbol qualification (moved from inline.clj to break cycle)
;; ================================================================

(defn qualify-body-symbols
  "Qualify unqualified symbols in a walked body to their source namespace.
	Skips: params, special forms, Java classes, already-qualified symbols."
  [body source-ns param-set]
  (let [;; Symbols that should not be namespace-qualified: form heads,
        ;; cast/alloc/array ops, and special literals. Uses centralized
        ;; registries so adding a new form/op automatically excludes it.
        non-qualifiable?
        (fn [sym]
          (or (contains? form/known-form-heads sym)
              (descriptor/cast-op? sym)
              (descriptor/alloc-op? sym)
              (descriptor/aget-op? sym)
              (descriptor/aset-op? sym)
              (= 'alength sym)))
        qualify (fn qualify [expr bound]
                  (cond
                    (symbol? expr)
                    (cond
                      (or (contains? param-set expr)
                          (contains? bound expr)
                          (non-qualifiable? expr))
                      expr
                      (and (not (namespace expr))
                           (let [name-str (name expr)]
                             (and (seq name-str)
                                  (Character/isUpperCase ^char (first name-str)))))
                      (let [resolved (try (ns-resolve source-ns expr) (catch Exception _ nil))]
                        (cond
                          (var? resolved)
                          (symbol (str (.name (.ns ^clojure.lang.Var resolved)))
                                  (str (.sym ^clojure.lang.Var resolved)))
                          ;; Bare class name (e.g. (new Foo ...), (catch Foo e),
                          ;; or a class in value position): qualify to its FQN so
                          ;; the backend resolves it with no ns context.
                          (class? resolved)
                          (symbol (.getName ^Class resolved))
                          :else expr))
                      (and (namespace expr)
                           (try (the-ns (symbol (namespace expr))) true
                                (catch Exception _ false)))
                      expr
                      ;; Java static-call head `Class/method` where the namespace
                      ;; part is a simple (imported) class name, not a loaded ns:
                      ;; qualify Class to its FQN so the backend resolves it with no
                      ;; ns context (e.g. MemorySegment/ofArray ->
                      ;; java.lang.foreign.MemorySegment/ofArray when inlined out of
                      ;; the ns that imported it).
                      (and (namespace expr)
                           (class? (try (ns-resolve source-ns (symbol (namespace expr)))
                                        (catch Exception _ nil))))
                      (symbol (.getName ^Class (ns-resolve source-ns (symbol (namespace expr))))
                              (name expr))
                      :else
                      (if-let [v (try (ns-resolve source-ns expr) (catch Exception _ nil))]
                        (cond
                          ;; var: qualify to ns/sym
                          (var? v) (symbol (str (.name (.ns ^clojure.lang.Var v)))
                                           (str (.sym ^clojure.lang.Var v)))
                          ;; bare class name (e.g. macroexpand canonicalized (X. a)
                          ;; -> (new X a), exposing X): qualify to its FQN
                          (class? v) (symbol (.getName ^Class v))
                          :else expr)
                        expr))
                    (seq? expr)
                    (let [h (first expr)]
                      (case (when (symbol? h) (symbol (name h)))
                        (let* loop*)
                        (let [bindings (second expr)
                              pairs (partition 2 bindings)
                              new-bindings
                              (loop [ps (seq pairs) b bound acc []]
                                (if-not ps
                                  (vec (mapcat identity acc))
                                  (let [[sym init] (first ps)
                                        qinit (qualify init b)
                                        b' (conj b sym)]
                                    (recur (next ps) b' (conj acc [sym qinit])))))
                              new-bound (into bound (map first pairs))
                              body-forms (drop 2 expr)
                              qbody (map #(qualify % new-bound) body-forms)]
                          (with-meta (apply list h new-bindings qbody) (meta expr)))
                        fn*
                        (let [arities (rest expr)
                              qualify-arity
                              (fn [arity]
                                (let [params-vec (first arity)
                                      fn-bound (into bound (filter symbol? params-vec))
                                      qbody (map #(qualify % fn-bound) (rest arity))]
                                  (apply list params-vec qbody)))]
                          (with-meta
                            (if (vector? (first arities))
                              (let [params-vec (first arities)
                                    fn-bound (into bound (filter symbol? params-vec))
                                    qbody (map #(qualify % fn-bound) (rest arities))]
                                (apply list h params-vec qbody))
                              (apply list h (map qualify-arity arities)))
                            (meta expr)))
                        (with-meta (apply list (map #(qualify % bound) expr)) (meta expr))))
                    (vector? expr) (mapv #(qualify % bound) expr)
                    :else expr))]
    (qualify body #{})))

;; ================================================================
;; Late-pipeline type inference (moved from inline.clj to break cycle)
;; ================================================================

(defn infer-arg-tag
  "Infer the type tag of an expression in the late pipeline.
   Priority: env lookup → :tag metadata → structural analysis.
   Metadata-based inference is the primary mechanism."
  [expr env]
  (cond
    (symbol? expr) (or (get env expr)
                       (:raster.type/tag (meta expr))
                       (:tag (meta expr)))
    (number? expr) (condp instance? expr
                     Double 'double
                     Float 'float
                     Long 'long
                     Integer 'int
                     nil)
    (not (seq? expr)) nil
    :else
    (or (:raster.type/tag (meta expr))
        (:tag (meta expr))
        (let [head (first expr)]
          (cond
            (contains? #{'double 'float 'long 'int 'byte 'short 'char 'boolean} head) head
            ;; par/reduce returns the accumulator type, inferred from init
            (contains? #{'raster.par/reduce 'par/reduce} head)
            (let [[_ _acc init] expr]
              (infer-arg-tag init env))
            ;; par/scan returns the output array (first arg)
            (contains? #{'raster.par/scan 'par/scan} head)
            (let [out-sym (second expr)]
              (when (symbol? out-sym) (or (get env out-sym) (:tag (meta out-sym)))))
            ;; Allocations: handle both direct (double-array size)
            ;; and devirtualized (.invk zeros-like_m_...-impl ref size) forms
            (let [op (form/effective-op expr)]
              (and (symbol? op) (descriptor/alloc-op? op)))
            (let [op (form/effective-op expr)
                  args (form/effective-args expr)]
              (or (get descriptor/alloc-sym->array-tag (symbol (name op)))
                  ;; Dynamic allocators (zeros-like, aclone, alloc-like, similar):
                  ;; return the same array type as their first argument
                  (when-let [ref-arg (first args)]
                    (infer-arg-tag ref-arg env))))
            (descriptor/aget-op? head)
            (when (symbol? (second expr))
              (let [arr-tag (or (get env (second expr)) (:tag (meta (second expr))))]
                (get {'doubles 'double 'floats 'float 'longs 'long 'ints 'int
                      'bytes 'byte 'shorts 'short 'chars 'char 'booleans 'boolean}
                     arr-tag)))
            (= '.invk head)
            (when-let [impl-sym (second expr)]
              (try (when-let [v (resolve impl-sym)]
                     (let [m (meta v)]
                       (or (:raster.core/return-tag m)
                           (let [n (name impl-sym)
                                 base (when (.endsWith n "-impl") (subs n 0 (- (count n) 5)))]
                             (when base
                               (when-let [bv (ns-resolve (or (some-> (namespace impl-sym) symbol find-ns)
                                                             *ns*)
                                                         (symbol base))]
                                 (:raster.core/return-tag (meta bv))))))))
                   (catch Exception _ nil)))
            ;; grad-acc returns the type of its args (nil-safe +)
            (= 'raster.ad.reverse/grad-acc head)
            (let [arg-tags (keep #(infer-arg-tag % env) (rest expr))]
              (first arg-tags))
            (and (symbol? head) (namespace head)
                 (.contains ^String (name head) "_m_"))
            (try (when-let [v (resolve head)]
                   (or (:raster.core/return-tag (meta v))
                       (:tag (meta v))))
                 (catch Exception _ nil))
            (= 'if head)
            (let [[_ _test then-branch else-branch] expr
                  recur? (fn [f] (and (seq? f) (= 'recur (first f))))
                  then-tag (when (and then-branch (not (recur? then-branch)))
                             (infer-arg-tag then-branch env))
                  else-tag (when (and else-branch (not (recur? else-branch)))
                             (infer-arg-tag else-branch env))]
              (cond
                (nil? then-tag) else-tag
                (nil? else-tag) then-tag
                (= then-tag else-tag) then-tag
                (and (contains? #{'double 'float 'long 'int} then-tag)
                     (contains? #{'double 'float 'long 'int} else-tag))
                (cond
                  (or (= then-tag 'double) (= else-tag 'double)) 'double
                  (or (= then-tag 'float) (= else-tag 'float))
                  (if (or (= then-tag 'long) (= else-tag 'long)) 'double 'float)
                  :else 'long)
                :else (or then-tag else-tag)))
            (= 'do head)
            (when-let [last-expr (last (rest expr))]
              (infer-arg-tag last-expr env))
            (contains? #{'let 'let*} head)
            (let [bindings (second expr)
                  pairs (partition 2 bindings)
                  let-env (reduce (fn [e [sym init]]
                                    (if-let [t (infer-arg-tag init e)]
                                      (assoc e sym t)
                                      e))
                                  env pairs)
                  body (drop 2 expr)
                  last-body (last body)]
              (when last-body
                (infer-arg-tag last-body let-env)))
            (contains? #{'loop 'loop*} head)
            (let [bindings (second expr)
                  pairs (partition 2 bindings)
                  loop-env (reduce (fn [e [sym init]]
                                     (if-let [t (infer-arg-tag init e)]
                                       (assoc e sym t)
                                       e))
                                   env pairs)
                  body (drop 2 expr)
                  last-body (last body)]
              (when last-body
                (infer-arg-tag last-body loop-env)))
            (some? (:return-type-arg (form/form-info expr)))
            (let [arg-idx (:return-type-arg (form/form-info expr))
                  type-arg (nth (rest expr) arg-idx nil)]
              (when type-arg (infer-arg-tag type-arg env)))
            (descriptor/scalar-op? head)
            (let [arg-tags (keep #(infer-arg-tag % env) (rest expr))]
              (cond
                (contains? #{'clojure.core/alength 'raster.arrays/alength} head) 'long
                (some #{'double} arg-tags) 'double
                (some #{'float} arg-tags) 'float
                (some #{'long} arg-tags) 'long
                (empty? arg-tags) nil
                :else 'long))
            (= 'new head) (when-let [cls-sym (second expr)] (symbol (str cls-sym)))
            (and (symbol? head) (.endsWith (name head) "."))
            (symbol (subs (name head) 0 (dec (count (name head)))))
            (and (symbol? head) (namespace head))
            (try
              (when-let [v (resolve head)]
                (when-let [dt-atom (:raster.core/dispatch-table (meta v))]
                  (let [arg-tags (mapv #(infer-arg-tag % env) (rest expr))
                        entries (get @dt-atom (count (rest expr)))]
                    (when (and (every? some? arg-tags) (seq entries))
                      (when-let [match (first (filter #(= (vec arg-tags) (vec (:tags %))) entries))]
                        (let [mn (str (types/mangle (symbol (name head)) (:tags match)))
                              ms (symbol (str (:mangled-ns match)) mn)]
                          (when-let [mv (resolve ms)]
                            (or (:raster.core/return-tag (meta mv))
                                (:tag (meta mv))))))))))
              (catch Exception _ nil))
            :else nil)))))

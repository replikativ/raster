(ns raster.compiler.passes.scalar.local-products
  "Closed lexical presence refinement and nonescaping local vector replacement.
   No calls are evaluated or arithmetic reassociated. An approved optional
   scalar sum projects to addition only with independent present, homogeneous
   producer carriers and the existing checked scalar lowerer. Owners retaining
   definition-freshness guards must capture dependencies before projection. Fields
   are snapshotted in source order; escaping products retain their source form."
  (:require [raster.compiler.ir.form :as form]
            [raster.ad.accumulation :as accumulation]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.value-product :as value-product]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.scalar.normal-return :as normal-return]
            [raster.compiler.core.util :as util]))

(defn- literal? [x]
  (or (nil? x) (boolean? x) (number? x) (string? x) (keyword? x) (char? x)))

(defn- relist [original children]
  (with-meta (apply list children) (meta original)))

(defn- fact [env expression]
  (cond (literal? expression) (cond-> {:literal expression
                                     :presence (if (nil? expression) :absent :present)}
                               (number? expression)
                               (assoc :scalar-type (some-> (inference/infer-arg-tag expression {})
                                                          dtype/dtype-for-scalar-tag)))
        (symbol? expression) (get env expression)
        ;; Object completion changes representation, not the value of a saved
        ;; literal/lexical reference. Transport only existing facts here; do
        ;; not infer a producer recipe from the operand's logical scalar tag.
        (and (completion/canonical-form? expression)
             (= completion/object-input-contract
                (:contract (completion/canonical-parts expression))))
        (fact env (:operand (completion/canonical-parts expression)))
        :else nil))

(defn- product-shape? [expression]
  (or (vector? expression)
      (and (seq? expression) (= 'let* (first expression))
           (= 3 (count expression)) (product-shape? (last expression)))))

(defn- split-product [expression]
  (if-let [fields (value-product/literal-fields expression)]
    [[] fields]
    (let [[bindings fields] (split-product (last expression))]
      [(into (vec (second expression)) bindings) fields])))

(defn- primitive-cast? [expression]
  (when (and (seq? expression) (= 2 (count expression)))
    (let [operation (descriptor/semantic-op expression)]
      (and (symbol? operation) (= "clojure.core" (namespace operation))
           (descriptor/cast-result-tag operation)))))

(defn- literal-tail [expression]
  (cond (literal? expression) {:bindings [] :literal expression}
        (and (seq? expression) (= 'let* (first expression)) (= 3 (count expression)))
        (when-let [tail (literal-tail (last expression))]
          (update tail :bindings #(into (vec (second expression)) %)))
        :else nil))

(declare lower)

(defn- nil-test-branches [env test]
  ;; A canonical direct nil test refines only the value it actually reads.
  ;; Do not infer presence from dtype tags or carry predicate facts across
  ;; later rebinding of a lexical name.
  (if (and (seq? test) (= 2 (count test)) (symbol? (second test))
           (= 'clojure.core/nil? (descriptor/semantic-op test)))
    (let [value (second test) known (get env value)]
      [(assoc env value (-> known (dissoc :fields :present? :scalar-type :array-tag)
                           (assoc :literal nil :presence :absent)))
       (assoc env value (-> known (dissoc :literal) (assoc :present? true :presence :present)))])
    [env env]))

(def ^:private conditional-expansion-limit 4096)
(def ^:dynamic ^:private *conditional-expansion-budget* nil)
(def ^:dynamic ^:private *consumed-evaluation-obligations* nil)
(def ^:dynamic ^:private *evaluation-dependencies* nil)

(defn- product-arity [expression]
  (cond
    (vector? expression) (count expression)
    (and (seq? expression) (= 'let* (first expression)) (= 3 (count expression)))
    (product-arity (last expression))
    (and (seq? expression) (= 'if (first expression)) (= 4 (count expression)))
    (let [left (product-arity (nth expression 2))
          right (product-arity (nth expression 3))]
      (when (and (some? left) (= left right)) left))
    :else nil))

(defn- conditional-product? [expression]
  (and (seq? expression) (= 'if (first expression))
       (= 4 (count expression)) (some? (product-arity expression))))

(defn- spend-conditional-expansion! [continuation]
  ;; Account for the duplicated continuation before constructing it. The
  ;; bounded walk also avoids traversing an arbitrarily large rejected body.
  (let [remaining @*conditional-expansion-budget*
        cost (count (take (inc remaining) (tree-seq coll? seq continuation)))]
    (when (> cost remaining)
      (throw (ex-info "Conditional product replacement exceeds its expansion budget"
                      {:reason :local-product-budget})))
    (swap! *conditional-expansion-budget* - cost)))

(defn- binding-fact [scope id init]
  (let [known (or (fact scope init)
                  ;; Completion retains the operand's value and evaluation
                  ;; context, not a new computation recipe. Reuse only the
                  ;; existing result-fact authorities for a direct producer.
                  (when-let [{:keys [contract operand]} (completion/canonical-parts init)]
                    (when (= completion/object-input-contract contract)
                      (binding-fact scope id operand)))
                  ;; A real primitive conversion cannot return nil. This is a
                  ;; normal-return fact, not permission to remove its source
                  ;; evaluation or the exception raised for a nullable input.
                  (when (primitive-cast? init)
                    (let [type (some-> (descriptor/semantic-op init)
                                       descriptor/cast-result-tag dtype/dtype-for-scalar-tag)]
                      (cond-> {:present? true} type (assoc :scalar-type type))))
                  ;; Numerical payload tags do not establish the JVM overload
                  ;; or physical result width of canonical source arithmetic.
                  ;; Its checked grammar proves presence only. In particular,
                  ;; Object-carried Float + Float may return a Double.
                  (when-let [proof (scalar-body/checked-scalar-normal-return
                                    init (into {} (keep (fn [[argument known]]
                                                         (when (and (= :present (:presence known))
                                                                    (:producer-scalar-type known))
                                                           [argument (:producer-scalar-type known)])))
                                               scope))]
                    (assoc proof :present? true)))
        known (or known
                  (when-let [proof (scalar-body/checked-selected-primitive-result init)]
                    {:present? true :scalar-type (:type proof)})
                  (let [result (normal-return/result-fact init scope)]
                    (when (= :present (:presence result))
                      (assoc result :present? true))))
        retained (some-> (:raster.type/tag (meta id)) dtype/dtype-for-scalar-tag)
        ;; Retained consumer declarations are not physical producer evidence.
        ;; Keep that distinction through aliases for checked sum projection.
        producer-type (or (:producer-scalar-type known) (:scalar-type known))]
    (cond-> known
      (:present? known) (assoc :presence :present)
      producer-type (assoc :producer-scalar-type producer-type)
      (and (:scalar-type known) retained) (assoc :scalar-type retained))))

(defn- project-scalar-accumulation [env expression]
  (when (accumulation/approved-call? expression)
    (let [arguments (:arguments (descriptor/call-description expression))
          facts (mapv (fn [argument]
                        (let [known (binding-fact env (gensym "sum_operand__") argument)
                              dt (:producer-scalar-type known)]
                          (when (and dt (= dt (:scalar-type known))
                                     (or (:present? known)
                                         (and (contains? known :literal) (some? (:literal known)))))
                            {:present? true :dtype dt}))) arguments)]
      (accumulation/project-checked-addition
       expression facts
       (fn [addition dt]
         ;; Validate the selected operation using independently established
         ;; producer widths. Proxies are proof inputs, not emitted temporaries:
         ;; original operands and their evaluation order remain unchanged.
         (let [ports [(gensym "sum_left__") (gensym "sum_right__")]
               probe (with-meta (apply list (first addition) ports) (meta addition))
               proof (scalar-body/checked-primitive-result probe (zipmap ports (repeat dt)))]
           (= dt (:type proof))))))))

(defn- project-array-accumulation [env expression]
  (when (accumulation/approved-call? expression)
    (let [facts (mapv #(normal-return/result-fact % env)
                      (:arguments (descriptor/call-description expression)))]
      (accumulation/project-checked-array-addition
       expression facts
       (fn [addition tag]
         (= {:presence :present :array-tag tag}
            (normal-return/result-fact addition env)))))))

(defn- owned-binding-fact [scope id init]
  (let [known (binding-fact scope id init)
        lexical-read? #(and (symbol? %) (contains? scope %))
        alias? (or (lexical-read? init)
                   (and (completion/canonical-form? init)
                        (= completion/object-input-contract
                           (:contract (completion/canonical-parts init)))
                        (lexical-read? (:operand (completion/canonical-parts init)))))
        ;; Presence and evaluation are separate facts. Even an unknown producer
        ;; must still execute when representation projection removes its last
        ;; saved reference; it may throw before returning any value at all.
        token (when (and *consumed-evaluation-obligations*
                         (or (coll? init) (symbol? init)) (not alias?))
                (Object.))
        ;; Retaining a producer alone is insufficient: normalization can remove
        ;; discarded lexical reads inside it. Record direct dependency edges;
        ;; copying all ancestors at every binding would grow quadratically.
        dependencies (when token
                       (binding [util/*shadowing-locals*
                                 (into util/*shadowing-locals* (keys scope))]
                         (into #{} (mapcat #(get-in scope [% :evaluation-obligations]))
                               (util/free-syms init))))
        _ (when token
            ;; Publish immutable edges before the fact can be consumed. Tokens
            ;; and this graph belong to exactly one analysis invocation.
            (swap! *evaluation-dependencies* assoc token
                   (into (or (:evaluation-obligations known) #{}) dependencies)))]
    [(cond-> known token (assoc :evaluation-obligations #{token})) token]))

(defn- consume-evaluation! [known]
  (when *consumed-evaluation-obligations*
    (loop [pending (vec (:evaluation-obligations known))]
      (when (seq pending)
        (let [token (peek pending) remaining (pop pending)]
          (if (contains? @*consumed-evaluation-obligations* token)
            (recur remaining)
            (let [graph @*evaluation-dependencies*]
              (when-not (contains? graph token)
                (throw (ex-info "Evaluation obligation has no producer in this analysis"
                                {:reason :local-product-evaluation-obligation})))
              (swap! *consumed-evaluation-obligations* conj token)
              (recur (into remaining (get graph token))))))))))

(defn- emit-owned-bindings [entries original]
  (with-meta
    (vec (mapcat (fn [[id init token]]
                   [id (if (and token (contains? @*consumed-evaluation-obligations* token))
                         (vary-meta init assoc :raster.effect/preserve-evaluation true)
                         init)]) entries))
    (meta original)))

(defn- lower-let [env expression products?]
  (let [[head bindings & body] expression]
    (when-not (and (vector? bindings) (even? (count bindings))
                   (every? symbol? (take-nth 2 bindings)))
      (throw (ex-info "Local product replacement requires lexical symbol bindings"
                      {:reason :local-product-decline})))
    (loop [remaining (seq (partition 2 bindings)) scope env output []]
      (if-let [[id original-init] (first remaining)]
        (let [init (if (and products? (symbol? original-init) (:fields (get scope original-init)))
                     original-init (lower scope original-init products?))
              alias (when (symbol? init) (get scope init))]
          (cond
                 ;; Replace a nonescaping product join by its lexical
                 ;; continuation in the selected arm. Predicate, arm effects
                 ;; and continuation each execute once, in their source order.
                 (and products? (conditional-product? init))
                 (let [[_ test then else] init
                       suffix (vec (mapcat identity (next remaining)))
                       continue (fn [arm]
                                  (util/alpha-convert
                                   (relist expression
                                           (concat [head (into [id arm] suffix)] body))))
                       _ (spend-conditional-expansion! (concat suffix body))
                       choice (lower scope (relist init ['if test (continue then) (continue else)])
                                     products?)]
                   (relist expression [head (emit-owned-bindings output bindings) choice]))

                 (and products? (:fields alias))
                 (recur (next remaining) (assoc scope id alias) output)

                 (and products? (product-shape? init))
                 (let [[prefix fields] (split-product (util/alpha-convert init))
                       snapshots (mapv (fn [value]
                                         [(with-meta (gensym "product_field__")
                                            (assoc (select-keys (meta value) [:raster.type/tag])
                                                   :tag 'Object))
                                          ;; A vector consumes each original field in Object
                                          ;; context, once, before any projection. Merely
                                          ;; binding its primitive value loses both that
                                          ;; producer context and the identity of its box.
                                          (completion/complete-value
                                           completion/object-input-contract value)])
                                       fields)
                       lifted (vec (concat prefix (mapcat identity snapshots)))
                       [scope output] (reduce (fn [[local out] [id init]]
                                                (let [[known token] (owned-binding-fact local id init)]
                                                  [(assoc local id known)
                                                   (conj out [id init token])]))
                                              [scope output] (partition 2 lifted))]
                   (recur (next remaining) (assoc scope id {:fields (mapv first snapshots)})
                          output))

                 :else
                 (let [[known token] (owned-binding-fact scope id init)]
                   (recur (next remaining) (assoc scope id known)
                          (conj output [id init token])))))
        ;; Realize consumers first: they record which occurrence-owned presence
        ;; proofs were used. Tokens never leave this analysis invocation.
        (let [body' (mapv #(lower scope % products?) body)]
          (relist expression (concat [head (emit-owned-bindings output bindings)] body')))))))

(defn- lower-scopes [env expression products?]
  (let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info expression)
        mask #(apply dissoc %1 %2)]
    (rebuild
     (mapv (fn [{:keys [binders inits body] :as scope}]
             (let [base (if rec? (mask env binders) env)
                   [local inits']
                   (reduce (fn [[local out] [id init]]
                             [(mask local [id])
                              (conj out (lower (if (or sequential? rec?) local env) init products?))])
                           [base []] (map vector binders inits))]
               (assoc scope :inits inits'
                            :body (mapv #(lower (mask local binders) % products?) body)))) scopes)
     (mapv #(lower env % products?) outer))))

(defn- lower [env expression products?]
  (cond
    (symbol? expression)
    (let [known (get env expression)]
      (cond (:fields known) (throw (ex-info "Local product escapes projection scope"
                                           {:reason :local-product-escape :value expression}))
            :else expression))

    (and (seq? expression) (= 'quote (first expression))) expression

    (and (seq? expression) (contains? #{'let 'let*} (first expression)))
    (lower-let env expression products?)

    ;; Import an inline literal projection through the SAME saved-product
    ;; owner. Selecting a field directly would discard eager sibling fields
    ;; and their Object input contexts, including failures after that field.
    (and products? (seq? expression)
         (value-product/literal-fields (second expression))
         (value-product/constant-projection expression (count (second expression))))
    (let [product (gensym "inline_product__")]
      (lower-let env
                 (relist expression
                         ['let* [product (second expression)]
                          (relist expression [(first expression) product (nth expression 2)])])
                 products?))

    ;; Constant-index projection consumes a saved product without reconstructing
    ;; its allocation. Out-of-bounds/defaulted/dynamic accesses remain unchanged.
    (and (seq? expression) (symbol? (second expression))
         (get-in env [(second expression) :fields])
         (value-product/constant-projection
          expression (count (get-in env [(second expression) :fields]))))
    (let [fields (get-in env [(second expression) :fields])
          projection (value-product/constant-projection expression (count fields))]
      (nth fields (:index projection)))

    (and (seq? expression) (= 'clojure.core/nil? (descriptor/semantic-op expression)) (= 2 (count expression)))
    (let [argument (second expression) known (fact env argument)]
      (cond (:fields known) false
            (:present? known) (do (consume-evaluation! known)
                                 false)
            (contains? known :literal) (nil? (:literal known))
            ;; A primitive cast cannot return nil, but can throw. Retain its
            ;; evaluation once, at the original test position, before false.
            (primitive-cast? argument)
            (list 'let* [(gensym "presence_evaluation__")
                         (vary-meta (lower env argument products?) assoc
                                    :raster.effect/preserve-evaluation true)] false)
            :else (relist expression [(first expression) (lower env argument products?)])))

    (and (seq? expression) (= 'if (first expression)))
    (let [[head test then else] expression
          test' (lower env test products?)
          known (fact env test')
          tail (or (when (contains? known :literal) {:bindings [] :literal (:literal known)})
                   (literal-tail (util/alpha-convert test')))]
      (if tail
        (let [branch (lower env (if (:literal tail) then else) products?)]
          (if (seq (:bindings tail)) (list 'let* (:bindings tail) branch) branch))
        (let [[then-env else-env] (nil-test-branches env test')]
          (relist expression (cond-> [head test' (lower then-env then products?)]
                              (= 4 (count expression)) (conj (lower else-env else products?)))))))

    (and (seq? expression) (form/scope-info expression))
    (lower-scopes env expression products?)

    (seq? expression)
    (let [lowered (relist expression (map #(lower env % products?) expression))]
      (or (project-scalar-accumulation env lowered)
          (project-array-accumulation env lowered) lowered))
    (vector? expression) (with-meta (mapv #(lower env % products?) expression) (meta expression))
    (map? expression) (into (empty expression) (map (fn [[k v]] [(lower env k products?) (lower env v products?)])) expression)
    (set? expression) (into (empty expression) (map #(lower env % products?)) expression)
    :else expression))

(defn lower-local-products
  "Refine only literal/alias presence and scalar-replace projectively used local
   vectors. Unknown calls/parameters never prove presence. If any product escapes,
   retain all product allocations and perform only literal presence refinement.
   Conditional product joins distribute only a bounded lexical continuation;
   an escape or expansion limit rolls the entire product rewrite back."
  [expression]
  (binding [*conditional-expansion-budget* (atom conditional-expansion-limit)
            *consumed-evaluation-obligations* (atom #{})
            *evaluation-dependencies* (atom {})]
    (try (lower {} expression true)
       (catch clojure.lang.ExceptionInfo e
         (case (:reason (ex-data e))
           (:local-product-escape :local-product-budget) (try (lower {} expression false)
                                      (catch clojure.lang.ExceptionInfo e
                                        (if (= :local-product-decline (:reason (ex-data e))) expression (throw e))))
           :local-product-decline expression
           (throw e))))))

(defn- fp-payload [known]
  (when (and (= :present (:presence known))
             (= (:producer-scalar-type known) (:scalar-type known)))
    (case (:producer-scalar-type known) :float Float/TYPE :double Double/TYPE nil)))

(defn- exact-fp-input? [payload consumer]
  (or (= payload consumer)
      (and (= Float/TYPE payload) (= Double/TYPE consumer))))

(defn- object-alias-operand [expression]
  (let [{:keys [contract operand]} (completion/canonical-parts expression)]
    (when (and (= completion/object-input-contract contract) (symbol? operand)) operand)))

(def ^:private native-snapshot-visit-limit 1000000)
(def ^:dynamic ^:private *native-snapshot-visits* nil)

(defn- visit-native-snapshot! []
  (when (and *native-snapshot-visits* (neg? (vswap! *native-snapshot-visits* dec)))
    ;; This optional representation analysis must never publish a partially
    ;; checked rewrite. Compilation stops before C emission, not mid execution.
    (throw (ex-info "Native Object snapshot analysis exceeded its visit budget"
                    {:reason :native-object-storage-budget
                     :visit-limit native-snapshot-visit-limit}))))

(defn- primitive-use-closed?
  "Check uses of one existing Object snapshot, not the producer's computation.
   Lexical aliases copy its identity; only actual primitive inputs or the native
   scalar return ABI may stop that flow. Unknown consumers and joins abstain."
  [expression aliases payload result-carrier]
  (letfn [(free-aliases [value names]
            (binding [util/*shadowing-locals* (into util/*shadowing-locals* names)]
              (some names (util/free-syms value))))
          (uses? [value names expected]
            (visit-native-snapshot!)
            (cond
              (symbol? value) (or (not (contains? names value))
                                  (= :discard expected) (exact-fp-input? payload expected))
              (and (seq? value) (#{'quote 'clojure.core/quote} (first value))) true
              (and (seq? value) (#{'let 'let*} (first value)))
              (let [[_ bindings & body] value]
                (and (vector? bindings) (even? (count bindings))
                     (loop [pairs (seq (partition 2 bindings)) names names]
                       (if-let [[id init] (first pairs)]
                         (let [alias (or (when (symbol? init) init) (object-alias-operand init))]
                           (if (and (symbol? id) (contains? names alias))
                             (let [declared (dtype/dtype-for-scalar-tag (:raster.type/tag (meta id)))
                                   actual (if (= Float/TYPE payload) :float :double)]
                               (and (or (nil? declared) (= actual declared))
                                    (recur (next pairs) (conj names id))))
                             (and (symbol? id) (uses? init names nil)
                                  (recur (next pairs) (disj names id)))))
                         (every? true? (map-indexed
                                        (fn [i form]
                                          (uses? form names (if (= i (dec (count body))) expected :discard)))
                                        body))))))
              (and (seq? value) (= 'if (first value)))
              ;; An Object's truthiness is not its numerical nonzero test.
              (and (not (free-aliases (second value) names))
                   (every? #(uses? % names expected) (nnext value)))
              (and (seq? value) (= 'do (first value)))
              (every? true? (map-indexed
                             (fn [i form]
                               (uses? form names (if (= i (- (count value) 2)) expected :discard)))
                             (rest value)))
              (and (seq? value) (= :lambda (:kind (form/form-info value))))
              (not (free-aliases value names))
              (and (seq? value) (form/scope-info value))
              (let [{:keys [scopes outer]} (form/scope-info value)]
                (and (every? #(uses? % names nil) outer)
                     (every? (fn [{:keys [binders inits body]}]
                               (and (every? #(uses? % names nil) inits)
                                    (every? #(uses? % (apply disj names binders) expected) body)))
                             scopes)))
              (seq? value)
              (let [{:keys [semantic-op implementation-op arguments dispatch? lexical-callee]}
                    (descriptor/call-description value)
                    ;; Explicit core FP conversion is an actual consumer, not
                    ;; a retained dtype hint. Narrowing remains unsupported.
                    cast (when (and (not dispatch?) (not lexical-callee)
                                    (symbol? semantic-op)
                                    (= semantic-op (first value))
                                    (= "clojure.core" (namespace semantic-op))
                                    (= 1 (count arguments)))
                           (case (descriptor/cast-result-tag semantic-op)
                             double Double/TYPE float Float/TYPE nil))
                    parameters (when (and (= '.invk (first value)) dispatch? (not lexical-callee)
                                          (free-aliases value names))
                                 (:parameter-carriers
                                  (method-entry/selected-parameter-carriers
                                   semantic-op implementation-op (count arguments))))]
                (cond
                  cast (uses? (first arguments) names cast)
                  (and parameters (= (count arguments) (count parameters)))
                  (and (not-any? names (take 2 value))
                       (every? true? (map #(uses? %1 names %2) arguments parameters)))
                  :else (every? #(uses? % names nil) value)))
              (coll? value) (every? #(uses? % names nil) value)
              :else true))]
    (boolean (uses? expression aliases result-carrier))))

(defn project-storage-alias-snapshots
  "Discharge Object snapshot storage using actual producer/parameter carriers.
   Array references keep their identity even in Object context, so a proved
   array alias needs no boxing-use closure and may still be nullable. Primitive
   FP aliases require the existing exact primitive-consumer closure instead.
   Every original producer and its position stay unchanged. No call body is
   substituted, no arithmetic overload changes, and no direct producer recipe
   is inferred from a field tag. Unknown snapshots retain their Object IR and
   are refused by native completion admission.

   parameter-facts come from the physical ABI, not semantic annotations;
   result-carrier is that same ABI's actual scalar result Class (or nil). Local
   payload/presence facts reuse this pass's existing producer authority. Requires
   the enclosing definition owner for any selected method observations."
  [expression parameter-facts result-carrier]
  (letfn [(storage-id [id known]
            (if-let [tag (or (when (contains? types/primitive-array-info (:array-tag known))
                               (:array-tag known))
                             (when-let [payload (fp-payload known)]
                               (if (= Float/TYPE payload) 'float 'double)))]
              (vary-meta id assoc :tag tag :raster.type/tag tag)
              id))
          (project [value scope expected]
            (visit-native-snapshot!)
            (cond
              (and (seq? value) (#{'quote 'clojure.core/quote} (first value))) value
              (and (seq? value) (= :lambda (:kind (form/form-info value)))) value
              (and (seq? value) (#{'let 'let*} (first value)))
              (let [[head bindings & body] value]
                (loop [pairs (seq (partition 2 bindings)) scope scope output []]
                  (if-let [[id init] (first pairs)]
                    (let [operand (object-alias-operand init)
                          [known token] (owned-binding-fact scope id init)
                          known (dissoc known :native-erased-box?)
                          payload (fp-payload known)
                          array? (contains? types/primitive-array-info (:array-tag known))
                          _ (when (and array?
                                       (some #(and (contains? types/primitive-array-info %)
                                                   (not= (:array-tag known) %))
                                             [(:tag (meta id)) (:raster.type/tag (meta id))]))
                              (throw (ex-info "Array alias declaration contradicts its physical producer"
                                              {:reason :storage-alias-carrier-conflict
                                               :binding id :producer (:array-tag known)
                                               :declared (select-keys (meta id) [:tag :raster.type/tag])})))
                          continuation (relist value
                                               (concat [head (vec (mapcat identity (next pairs)))] body))
                          unused? (and operand (contains? scope operand)
                                       (binding [util/*shadowing-locals* (conj util/*shadowing-locals* id)]
                                         (not (contains? (util/free-syms continuation) id))))
                          _ (when unused? (consume-evaluation! known))
                          erase? (and operand
                                      (or array?
                                          (and payload
                                               (primitive-use-closed? continuation #{id} payload expected))))
                          inherited? (and (symbol? init) (:native-erased-box? (get scope init)))
                          known (cond-> known (or erase? inherited?) (assoc :native-erased-box? true))
                          id (if (or erase? inherited?) (storage-id id known) id)
                          init (if erase? operand (project init scope nil))]
                      ;; A bound lexical read in Object context cannot add a
                      ;; source conversion failure. If never observed, its box
                      ;; storage may disappear without deleting its producer.
                      (recur (next pairs) (assoc scope id known)
                             (if unused? output (conj output [id init token]))))
                    ;; Child consumers may discharge a snapshot after the
                    ;; producer was visited. Realize them before publishing its
                    ;; initializer and the occurrence-owned evaluation marker.
                    (let [body' (mapv #(project % scope expected) body)]
                      (relist value (concat [head (emit-owned-bindings output bindings)] body'))))))
              (and (seq? value) (form/scope-info value))
              (let [{:keys [scopes outer rebuild]} (form/scope-info value)]
                (rebuild
                 (mapv (fn [{:keys [binders inits body] :as local}]
                         (assoc local :inits (mapv #(project % scope nil) inits)
                                      :body (mapv #(project % (apply dissoc scope binders) expected) body))) scopes)
                 (mapv #(project % scope nil) outer)))
              (seq? value) (relist value (map #(project % scope nil) value))
              :else value))]
    (binding [*native-snapshot-visits* (volatile! native-snapshot-visit-limit)
              *consumed-evaluation-obligations* (atom #{})
              *evaluation-dependencies* (atom {})]
      (project expression parameter-facts result-carrier))))

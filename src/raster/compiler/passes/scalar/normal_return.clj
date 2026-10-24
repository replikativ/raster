(ns raster.compiler.passes.scalar.normal-return
  "Conservative normal-return presence and physical producer-carrier summaries.

   The lattice is :no-return < {:present :absent} < :unknown. A present result
   says nothing about ownership, aliasing, effects, cost or whether evaluation
   throws. Calls and initializers must remain at their original source positions.
   No source code is executed and no operation/function registry is introduced."
  (:require [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.source-arithmetic :as source-arithmetic]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.support.mangled :as mangled]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.parallel.materialize :as materialize]))

(def ^:private unknown {:presence :unknown})

(defn- join [a b]
  (let [pa (:presence a) pb (:presence b)]
    (cond
      (= :no-return pa) b
      (= :no-return pb) a
      :else
      (cond-> {:presence (if (= pa pb) pa :unknown)}
        (and (:array-tag a) (= (:array-tag a) (:array-tag b)))
        (assoc :array-tag (:array-tag a))
        (and (:scalar-type a) (= (:scalar-type a) (:scalar-type b)))
        (assoc :scalar-type (:scalar-type a))))))

(declare summary)

(defn- summary-source
  "Authentic semantic return source, never a replaceable emitted body.
   Primitive return completion is nonnullable; reference returns need body facts.
   Void discards even a value-producing body, so remains outside this rule."
  [{:keys [source-var receiver-var operation implementation-op parameters tags]
    :as definition}]
  (if receiver-var
    (when-let [frontend (method-entry/selected-frontend-source
                        operation implementation-op (count parameters) source-var
                        (dtype/infer-dtype-from-tags tags))]
      (let [lineage (:source-entry frontend)
            return-tag (:return-tag lineage)]
        (when (and (identical? (:source-body definition) (:source-body lineage))
                   (or (= 'Object return-tag) (= Object return-tag)
                       (contains? types/primitive-info return-tag)
                       (contains? types/primitive-array-info return-tag)))
          (assoc frontend :parameters (:params lineage) :tags (:tags lineage)
                 :source-var source-var :source-ns (:source-ns lineage)
                 :return-tag (:return-tag lineage)))))
    ;; Plain Clojure has its own actual frozen body/entry correspondence.
    (method-entry/selected-clojure-entry-body definition)))

(defn- observe-semantic-intrinsic! [body-context operation]
  (when (:source-frontend body-context)
    (method-entry/assert-frontend-source! body-context (:source-var body-context))
    (intrinsic-source/observe-original! (:source-ns body-context) operation)))

(defn- materialized-result [expression env visited body-context]
  (let [key [:materialized-map expression]]
    (if (contains? visited key)
      unknown
      (let [result (try
                     (materialize/materialize-pass expression nil)
                     (catch clojure.lang.ExceptionInfo e
                       ;; Optional analysis may decline exactly the materializer's
                       ;; unsupported domain, never conceal an invariant failure.
                       (if (= :unsupported-pure-map-materialization (:reason (ex-data e)))
                         nil
                         (throw e))))]
        (if (and (contains? result :form) (not= expression (:form result)))
          (summary (:form result) env (conj visited key) body-context)
          unknown)))))

(defn- selected-result* [expression env visited body-context]
  (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
        (descriptor/call-description expression)
        definition (when (and method-entry/*emission-inputs*
                              (not (or lexical-callee (contains? env implementation-op))))
                     (if dispatch?
                       (method-entry/selected-definition semantic-op implementation-op (count arguments))
                       (or
                        (when-let [operation (mangled/impl->op implementation-op)]
                          (method-entry/observed-selected-definition :direct operation implementation-op
                                                                    (count arguments)))
                        ;; Generic array selection uses actual producer carriers,
                        ;; never desired result/formal annotations or fake values.
                        (when (and (qualified-symbol? implementation-op)
                                   (find-ns (symbol (namespace implementation-op))))
                          (let [tags (mapv #(:array-tag (summary % env visited body-context)) arguments)
                                signature (when (and (seq tags) (every? some? tags))
                                            (inference/registered-call-signature implementation-op tags))]
                            (when (and signature (= tags (:tags signature))
                                       (not (seq (:promotion-casts signature))))
                              (method-entry/observed-selected-definition
                               :direct implementation-op (:mangled-sym signature) (count arguments))))))))]
    (if definition
      (if-let [{:keys [source-var parameters body tags] :as selected-body}
               (summary-source definition)]
        (if (and (:source-frontend selected-body)
                 (contains? types/primitive-info (:return-tag selected-body)))
          ;; This is the producer-enforced RETURN completion, not presence from
          ;; a naked annotation or a claim about the body's result carrier.
          ;; A cold stub first compiles; the primitive compute return converts
          ;; and the public Object entry boxes it. Nil/invalid bodies may throw,
          ;; but cannot normally return nil across this boundary. Keep the call.
          (cond-> {:presence :present}
            (dtype/dtype-for-scalar-tag (:return-tag selected-body))
            (assoc :scalar-type (dtype/dtype-for-scalar-tag (:return-tag selected-body))))
          (if (contains? visited source-var)
          unknown
          (let [;; Only historical producer-owned bodies may prove constructors
                ;; or aliases. Current walked metadata is not code evidence.
                body (seq body)]
            ;; Transport actual producer facts, never presence from formal tags.
            ;; A changed input carrier cannot certify a callee's array result.
            (if (seq body)
              (let [result (summary (apply list 'do body)
                                    (zipmap parameters
                                            (map (fn [argument tag]
                                                   (let [fact (summary argument env visited body-context)]
                                                     (cond-> fact
                                                       (not= tag (:array-tag fact))
                                                       (dissoc :array-tag)
                                                       (not= (dtype/dtype-for-scalar-tag tag)
                                                             (:scalar-type fact))
                                                       (dissoc :scalar-type))))
                                                 arguments tags))
                                    (conj visited source-var) selected-body)]
                ;; A declared array return may checkcast, but cannot turn a
                ;; different independently produced array into the desired one.
                (if (and (:source-frontend selected-body) (:array-tag result)
                         (contains? types/primitive-array-info (:return-tag selected-body))
                         (not= (:return-tag selected-body) (:array-tag result)))
                  unknown
                  result))
              unknown))))
        unknown)
      unknown)))

(defn- selected-result [expression env visited body-context]
  (or (method-entry/call-with-optional-definition-inputs
       #(let [result (selected-result* expression env visited body-context)]
          (when (or (not= :unknown (:presence result)) (:array-tag result)) result)))
      unknown))

(defn- source-call [expression env body-context]
  (let [head (when (seq? expression) (first expression))]
    (if-let [operation (when (and body-context (symbol? head)
                                 (nil? (namespace head))
                                 (not (special-symbol? head))
                                 (not (or (:dispatch? (descriptor/call-description expression))
                                          (:lexical-callee (descriptor/call-description expression)))))
                        (method-entry/observed-body-operation body-context head env))]
      (with-meta (cons operation (rest expression)) (meta expression))
      expression)))

(defn- conditional-result [expression env visited body-context]
  (let [[_ test then else] expression
        test (source-call test env body-context)
        argument (when (and (seq? test) (= 2 (count test))
                             (= 'clojure.core/nil? (descriptor/semantic-op test))
                             (not (contains? env (first test))))
                   (second test))
        known (when argument (summary argument env visited body-context))]
    (when argument (observe-semantic-intrinsic! body-context 'clojure.core/nil?))
    (case (:presence known)
      :present (summary else env visited body-context)
      :absent (summary then env visited body-context)
      (let [local? (symbol? argument)
            then-env (if local? (assoc env argument {:presence :absent}) env)
            else-env (if local? (assoc env argument (assoc known :presence :present)) env)]
        (join (summary then then-env visited body-context)
              (summary else else-env visited body-context))))))

(defn- summary
  ([expression env visited] (summary expression env visited nil))
  ([expression env visited body-context]
  (cond
    (nil? expression) {:presence :absent}
    (symbol? expression) (merge unknown (select-keys (get env expression) [:presence :array-tag]))
    (not (seq? expression)) {:presence :present}
    (contains? #{'quote 'clojure.core/quote} (first expression))
    {:presence (if (nil? (second expression)) :absent :present)}
    (= 'throw (first expression)) {:presence :no-return}
    (= 'do (first expression)) (summary (last (rest expression)) env visited body-context)
    (completion/canonical-parts expression)
    (let [{:keys [contract operand]} (completion/canonical-parts expression)
          fact (summary operand env visited body-context)]
      (case (:source-context contract)
        ;; Retaining a physical evaluation context says nothing about value
        ;; presence or a target-neutral producer. Await a separate projection.
        :jvm-bytecode-result-context unknown
        :jvm-bytecode-boxed-parameter-input
        (if (.isPrimitive ^Class (:target-carrier contract))
          {:presence (if (= :no-return (:presence fact)) :no-return :present)}
          fact)
        ;; These contexts preserve the actual reference, including nil. A
        ;; desired CHECKCAST class does not establish a producer array carrier.
        (:jvm-object-input :jvm-checkcast :jvm-bytecode-reference-input) fact
        ;; Presence is conditional on normal return, not proof that a cast
        ;; succeeds or authority to execute it under another compilation owner.
        (:jvm-primitive-input :jvm-bytecode-floating-input :jvm-bytecode-integral-input)
        {:presence (if (= :no-return (:presence fact)) :no-return :present)}))
    (form/binding-form? expression)
    (let [{:keys [sequential? scopes]} (form/scope-info expression)
          {:keys [binders inits body]} (first scopes)]
      (if (and sequential? (= 1 (count scopes)) (every? symbol? binders))
        (let [local (reduce (fn [local [id init]]
                              (assoc local id (summary init local visited body-context)))
                            env (map vector binders inits))]
          (summary (last body) local visited body-context))
        unknown))
    (= 'if (first expression))
    (conditional-result expression env visited body-context)
    :else
    (let [expression (source-call expression env body-context)
          operation (descriptor/semantic-op expression)]
      (cond
        ;; Exact qualified language constructors, not alloc-op?'s compatibility
        ;; lookup or buffer metadata which may only describe a call argument.
        (and (not= '.invk (first expression))
             (symbol? operation) (= "clojure.core" (namespace operation))
             (not (contains? env operation))
             (contains? descriptor/alloc-ops operation))
        (let [tag (get descriptor/alloc-sym->array-tag operation)]
          (observe-semantic-intrinsic! body-context operation)
          (cond-> {:presence :present}
            (contains? types/primitive-array-info tag) (assoc :array-tag tag)))
        ;; pmap is structural compiler IR, not a source helper invocation. Use
        ;; its existing materializer to expose the actual allocation/return,
        ;; retaining constructor guards in a semantic frontend's context too.
        (and (= 'raster.par/pmap operation) (par/par-map-pure-form? expression)
             (= 5 (count expression)) (symbol? (second expression)))
        (materialized-result expression env visited body-context)
        ;; Only explicit destination identity contracts may forward a fact.
        (and (not (:source-frontend body-context))
             (symbol? operation) (= operation (first expression))
             ;; form-info accepts surface aliases such as par/scan. Such a
             ;; spelling is not a closed semantic operation in raw source.
             (= "raster.par" (namespace operation))
             (some? (:return-alias-arg (form/form-info expression))))
        (let [index (:return-alias-arg (form/form-info expression))]
          (summary (nth (descriptor/call-args expression) index nil) env visited body-context))
        :else (selected-result expression env visited body-context))))))

(defn result-fact
  "Normal-return presence plus independent array/domain information.
   Lexical facts come from actual producers; formal/result annotations cannot
   establish presence or a physical carrier. An authenticated primitive return
   COMPLETION can additionally supply :scalar-type: its converted and canonically
   boxed value domain, not a JVM operand Class or nonthrowing claim. Symbolic
   inputs do not import a caller's consumer-overridden scalar tag. Extra owner-local
   facts are ignored, and no call is executed or moved by this analysis.
   Selected definitions require an enclosing compilation-input owner; otherwise
   definition-dependent facts are unknown rather than silently losing provenance."
  ([expression] (result-fact expression {}))
  ([expression lexical-facts]
   (select-keys (summary expression lexical-facts #{}) [:presence :array-tag :scalar-type])))

(defn presence
  "Return :present, :absent, :unknown, or :no-return for a closed source region.
   Unknown is not absence. This function is an analysis, never an evaluator or
   permission to remove/move calls. Input values start unknown; only checked
   producers and lexical aliases establish presence."
  [expression]
  (:presence (result-fact expression)))

(defn- primitive-completion-fact [^Class carrier]
  (when (and (class? carrier) (.isPrimitive carrier) (not= Void/TYPE carrier))
    {:carrier carrier :presence :present
     :builtin-class (get types/primitive-info (symbol (.getName carrier)))}))

(defn- checked-completion-result [fact]
  (when (some? fact)
    (let [{:keys [carrier builtin-class]} (when (map? fact) fact)
          expected (or (primitive-completion-fact carrier)
                       (when (and (= Object carrier)
                                  (some #(identical? builtin-class %) (vals types/primitive-info)))
                         {:carrier Object :presence :present :builtin-class builtin-class}))]
      (when-not (and expected (= expected (select-keys fact [:carrier :presence :builtin-class])))
        (throw (ex-info "Owned completion resolver returned a noncanonical scalar fact"
                        {:reason :invalid-completion-result-fact})))
      expected)))

(defn- completed-scalar-fact* [expression lexical-facts source-ns completion-result]
  (cond
    (symbol? expression)
    ;; Only independently supplied producer facts, never a symbol's dtype tag.
    (let [{:keys [carrier presence] :as fact} (get lexical-facts expression)]
      (when (and (class? carrier) (= :present presence))
        (select-keys fact [:carrier :presence :builtin-class])))

    (not (seq? expression))
    ;; literal-tag alone also describes nonprimitive integers. Exact wrapper
    ;; correspondence prevents BigInt/BigInteger from becoming primitive Long.
    (let [tag (types/literal-tag expression)]
      (when (= (class expression) (get types/primitive-info tag))
        (primitive-completion-fact (types/tag->java-class tag))))

    (completion/canonical-parts expression)
    (or (when completion-result
          (checked-completion-result (completion-result expression)))
        (let [{:keys [contract operand]} (completion/canonical-parts expression)
              fact (completed-scalar-fact* operand lexical-facts source-ns completion-result)]
          (case (:source-context contract)
            :jvm-object-input (when fact (assoc fact :carrier Object))
            ;; Without the owned resolver, a target is not a result carrier:
            ;; source primitive-input projects through RT.box, whereas the
            ;; bytecode-floating form needs its actual emitter context.
            nil)))

    ;; Only the expanded special form, not an unauthenticated source macro.
    (= 'let* (first expression))
    (let [{:keys [sequential? scopes]} (form/scope-info expression)
          {:keys [binders inits body]} (first scopes)]
      (when (and sequential? (= 1 (count scopes)) (every? symbol? binders))
        (let [scope (reduce (fn [scope [id init]]
                              (assoc scope id (completed-scalar-fact* init scope source-ns completion-result)))
                            lexical-facts (map vector binders inits))]
          (completed-scalar-fact* (last body) scope source-ns completion-result))))

    (= 'do (first expression))
    (completed-scalar-fact* (last (rest expression)) lexical-facts source-ns completion-result)

    :else
    (let [{:keys [semantic-op arguments lexical-callee implementation-op]}
          (descriptor/call-description expression)]
      (when-not (or lexical-callee (contains? lexical-facts implementation-op))
        (if (and (= semantic-op (first expression))
                 (symbol? semantic-op)
                 (or (= "clojure.core" (namespace semantic-op))
                     (and (nil? (namespace semantic-op))
                          (not (contains? util/*shadowing-locals* semantic-op))))
                 (= 1 (count arguments)) (descriptor/cast-op? semantic-op))
          ;; Conditional on successful source evaluation. A nullable operand
          ;; can still throw; this fact cannot discharge that evaluation.
          (when (and method-entry/*emission-inputs*
                     (instance? clojure.lang.Namespace source-ns)
                     (intrinsic-source/observe-original!
                      source-ns semantic-op
                      (if (namespace semantic-op) semantic-op
                          (symbol "clojure.core" (name semantic-op)))))
            (some-> semantic-op descriptor/cast-result-tag types/tag->java-class
                    primitive-completion-fact))
          (when method-entry/*emission-inputs*
            (when-let [proof (let [facts (mapv #(completed-scalar-fact* % lexical-facts source-ns completion-result) arguments)]
                              (or (scalar-body/checked-core-fp-evaluation expression facts source-ns)
                                  (scalar-body/checked-selected-primitive-evaluation expression facts)))]
              {:carrier (:result-carrier proof) :presence (:presence proof)
               :builtin-class (:builtin-class proof)})))))))

(defn completed-scalar-fact
  "Opt-in physical producer facts AFTER normal source evaluation.

   Lexical facts must describe actual producers, not logical dtype annotations.
   Selected calls reuse the existing source-owned primitive completion proof;
   Object results remain Object, with a separate exact builtin payload class.
   Missing/nullable/opaque evidence returns nil. This neither executes source
   nor proves that its argument evaluation succeeds, authorizes motion, removes
   evaluation flags, or supplies a kernel/payload lowering recipe. Core casts
   and qualified core arithmetic require the explicit producer namespace,
   bootstrap definition and source owner. Bare casts retain their original
   namespace lookup and require an unshadowed lexical scope.

   Optional completion-result resolves an EXACT canonical occurrence through
   an independently guarded producer. The caller must retain that producer's
   owner through final validation and account for each used completion at its
   source position through an exact executor or a typed lowering rule. This
   imports actual normal-return facts,
   not logical tags, totality or permission to remove/move the completion.
   Copies and unknown observations must return nil; stale owners must throw."
  ([expression] (completed-scalar-fact expression {}))
  ([expression lexical-facts] (completed-scalar-fact expression lexical-facts nil))
  ([expression lexical-facts source-ns]
   (completed-scalar-fact expression lexical-facts source-ns nil))
  ([expression lexical-facts source-ns completion-result]
   (method-entry/call-with-optional-definition-inputs
    #(completed-scalar-fact* expression lexical-facts source-ns completion-result))))

(defn completed-scalar-bindings
  "Normal-completion facts for sequential lexical producers. Unknown producers
   shadow earlier facts; inputs must already have independent carrier evidence.
   This does not execute, reorder, or certify evaluation of any binding."
  ([pairs] (completed-scalar-bindings pairs {}))
  ([pairs lexical-facts] (completed-scalar-bindings pairs lexical-facts nil))
  ([pairs lexical-facts source-ns]
   (completed-scalar-bindings pairs lexical-facts source-ns nil))
  ([pairs lexical-facts source-ns completion-result]
   (reduce (fn [facts [id expression]]
             (assoc facts id (completed-scalar-fact expression facts source-ns completion-result)))
           lexical-facts pairs)))

(defn- double-storage-leaf? [expression]
  (or (symbol? expression) (instance? Double expression)))

(defn- double-storage-initializer? [expression]
  (or (double-storage-leaf? expression)
      (when-let [{:keys [contract operand]} (completion/canonical-parts expression)]
        (and (= completion/object-input-contract contract)
             (double-storage-leaf? operand)))
      (and (seq? expression) (not (:lexical-callee (descriptor/call-description expression)))
           (or (and (= 'clojure.core/double (first expression))
                    (= 2 (count expression))
                    (double-storage-initializer? (second expression)))
               (and (symbol? (first expression))
                    (= "clojure.core" (namespace (first expression)))
                    (source-arithmetic/binary-operator? (name (first expression)))
                    (= 3 (count expression))
                    (every? double-storage-initializer? (rest expression)))
               (and (= '.invk (first expression)) (symbol? (second expression))
                    (every? double-storage-leaf? (nnext expression)))))))

(defn- present-double-payload? [{:keys [carrier presence builtin-class]}]
  (and (= :present presence)
       (or (identical? Double/TYPE carrier)
           (and (contains? #{Object Double} carrier)
                (identical? Double builtin-class)))))

(defn- double-storage-annotations? [expression]
  ;; An annotation supplies NO carrier fact. Once the independent physical
  ;; construction proves Double, reject a contradictory width the downstream
  ;; scalar emitter could otherwise honor. Object storage annotations are not
  ;; silently reinterpreted as primitive producer evidence.
  (every? (fn [tag]
            (or (nil? tag)
                (contains? #{Object Double Double/TYPE} tag)
                (contains? #{Object Double/TYPE}
                           (get types/tag->java-class (get types/boxed->primitive-tag tag tag)))))
          ((juxt :tag :raster.type/tag) (meta expression))))

(defn- double-storage-call-entry [expression locals]
  (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
        (descriptor/call-description expression)]
    (when (and dispatch? (not lexical-callee)
               (not (contains? locals implementation-op)))
      (when-let [definition (method-entry/observed-selected-definition
                            :typed semantic-op implementation-op (count arguments))]
        (let [entry (method-entry/selected-source-entry-carriers
                     semantic-op implementation-op (count arguments) (:source-var definition))]
          (when (and entry (every? #(identical? Double/TYPE %) (:parameter-carriers entry)))
            entry))))))

(defn- double-storage-expression-fact
  [expression facts locals context source-ns]
  (when (and (double-storage-annotations? expression)
             (not (and (seq? expression)
                       (:lexical-callee (descriptor/call-description expression)))))
    (cond
      (instance? Double expression) {:carrier Double/TYPE :presence :present :builtin-class Double}
      (symbol? expression) (get facts expression)
      (seq? expression)
      (let [head (first expression)]
        (cond
          (= 'clojure.core/double head)
          (when (and (= 2 (count expression))
                     (present-double-payload?
                      (double-storage-expression-fact (second expression) facts locals context source-ns))
                     (intrinsic-source/observe-original! source-ns head))
            {:carrier Double/TYPE :presence :present :builtin-class Double})

          (= '.invk head)
          (let [arguments (vec (nnext expression))
                argument-facts (mapv #(double-storage-expression-fact % facts locals context source-ns) arguments)
                invocation ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/typed-invocation-plan)
                            (second expression) (count arguments) locals
                            ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/binding-initializer-context) context))
                payload (when (every? present-double-payload? argument-facts)
                          (scalar-body/checked-selected-fp-payload expression argument-facts))]
            (when (and payload (:global? invocation)
                       (= (:method invocation) (get-in payload [:bridge :invocation-method]))
                       (= (:parameter-carriers invocation) (:parameter-carriers payload))
                       (every? #(identical? Double/TYPE %) (:parameter-carriers payload))
                       (identical? Double/TYPE (:payload-carrier payload))
                       (identical? Object (:interface-result-class invocation)))
              {:carrier Object :presence :present :builtin-class Double}))

          (and (symbol? head) (= "clojure.core" (namespace head))
               (source-arithmetic/binary-operator? (name head)) (= 3 (count expression)))
          (let [arguments (mapv #(double-storage-expression-fact % facts locals context source-ns) (rest expression))
                actual (when (every? #(and (present-double-payload? %)
                                          (identical? Double/TYPE (:carrier %))) arguments)
                         (scalar-body/checked-core-fp-evaluation expression arguments source-ns))]
            (when (and (:nonthrowing? actual) (identical? Double/TYPE (:result-carrier actual)))
              {:carrier Double/TYPE :presence :present :builtin-class Double}))
          :else nil))
      :else nil)))

(defn- construct-double-storage-source
  "Representation-only construction over the actual physical binding plan.
   Whole-spine closure is also the use check: no Object observation/escape or
   unknown operation can consume a replaced snapshot. Original calls stay at
   their original bindings. Original facts are never retagged as primitives."
  [source parameters entry source-ns]
  (let [parameter-locals ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/resolved-parameter-locals)
                          (:invocation-method entry) parameters)
        context {:source-ns source-ns}
        used (into (into (set parameters) (take-nth 2 (second source))) (util/free-syms source))
        ;; The shared physical planner rejects duplicate/rebound binders,
        ;; including parameter shadowing. No earlier replacement can survive a
        ;; new definition: source shadowing must first pass normal SSA hygiene.
        plan (when parameter-locals
               ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/plan-flat-bindings)
                (second source) parameter-locals context))
        leaf-fact (fn [expression facts]
                    (if (instance? Double expression)
                      {:carrier Double/TYPE :presence :present :builtin-class Double}
                      (get facts expression)))]
    (when plan
      (loop [rows (:bindings plan)
             facts (zipmap parameters (repeat {:carrier Double/TYPE :presence :present}))
             replacements {} bindings [] used used]
        (if-let [{:keys [binder initializer before raw-type completion]} (first rows)]
          (let [parts (raster.compiler.core.value-completion/canonical-parts initializer)
                fact
                (if parts
                  (when-let [operand-fact (leaf-fact (:operand parts) facts)]
                          (when (and (present-double-payload? operand-fact)
                                     (intrinsic-source/observe-original! source-ns 'clojure.core/double))
                            (assoc operand-fact :carrier Object :builtin-class Double)))
                  (double-storage-expression-fact initializer facts before context source-ns))
                expected (when fact (if (identical? Double/TYPE (:carrier fact)) :double :ref))
                checkcast (:checkcast-class completion)]
            (when (and (present-double-payload? fact) (= expected raw-type) (nil? checkcast)
                       (double-storage-annotations? binder)
                       (double-storage-annotations? initializer)
                       (or (nil? parts) (double-storage-annotations? (:operand parts))))
              (let [new-id (if parts
                             (loop [candidate (gensym "numeric_snapshot_")]
                               (if (contains? used candidate)
                                 (recur (gensym "numeric_snapshot_"))
                                 (with-meta candidate {:raster.type/tag 'double})))
                             binder)
                    new-init (util/subst-syms replacements
                                             (if parts
                                               (list 'clojure.core/double (:operand parts))
                                               initializer))]
                (recur (next rows) (assoc facts binder fact)
                       (cond-> replacements parts (assoc binder new-id))
                       (conj bindings new-id new-init) (conj used new-id)))))
          (when (and (present-double-payload? (leaf-fact (nth source 2) facts))
                     (double-storage-annotations? (nth source 2)))
            (util/remake-from source
                              (list 'let* bindings (util/subst-syms replacements (nth source 2))))))))))

(defn with-closed-double-storage-source
  "Consume a bounded primitive/Object/primitive source construction lexically.
   The caller owns the actual frontend-to-source transformation and final target
   emission. Source Method Classes, not native ABI/dtypes, seed the physical plan.
   Only a flat all-Double numerical spine is admitted. Boxed AD entry contracts,
   identity observations, arbitrary casts/calls and escaping Objects abstain.

   Explicit preparation compiles only observed original receivers, then checks
   their ORIGINAL observations; any change discards the whole owning attempt.
   consume! receives the new source and a finalizer. The latter accepts the
   actual target parameter order, scalar ABI and result ABI, and returns the
   existing definition guard to run AFTER all public input conversions. Neither
   construction products nor a proof map escape into IR/options."
  [frontend source-var source parameters prepare! consume!]
  (when (and method-entry/*emission-inputs* frontend
             (seq? source) (= 'let* (first source)) (= 3 (count source))
             (vector? (second source)) (even? (count (second source)))
             (vector? parameters) (every? symbol? parameters)
             (completion/contains-completion? source)
             (every? double-storage-initializer? (take-nth 2 (rest (second source))))
             (double-storage-leaf? (nth source 2)))
    (method-entry/call-with-optional-definition-inputs
     (fn []
       (let [selection (method-entry/assert-frontend-selection! frontend source-var)
             entry (method-entry/frontend-body-entry-carriers frontend source-var)
             source-ns (get-in frontend [:source-entry :source-ns])
             locals (into (set parameters) (take-nth 2 (second source)))
             calls (filter #(and (seq? %) (= '.invk (first %)))
                           (tree-seq coll? seq (vec (take-nth 2 (rest (second source))))))
             call-entries (mapv #(double-storage-call-entry % locals) calls)]
         (when (and entry (= parameters (:parameters entry))
                    (every? #(identical? Double/TYPE %) (:parameter-carriers entry))
                    (every? some? call-entries) (:receiver-var selection))
           (method-entry/prepare-owned-definition-inputs!
            prepare! (conj (mapv #(get-in % [:definition :receiver-var]) call-entries)
                           (:receiver-var selection)))
           (when-let [bridge (method-entry/frontend-boxed-entry-bridge frontend)]
             ;; The actual compute return may unbox an Object body result.
             ;; The whole-source fold below proves that very terminal payload
             ;; is a present builtin Double; no logical return annotation
             ;; discharges Number.doubleValue or changes an Object fact.
             (when (and (= :boxed-primitive-static-return (:kind bridge))
                        (identical? Double/TYPE (:payload-carrier bridge))
                        (identical? Double/TYPE
                                    (.getReturnType ^java.lang.reflect.Method (:compute-method bridge)))
                        (= (:invocation-method entry) (:invocation-method bridge))
                        (= (:parameter-carriers entry) (:parameter-carriers bridge))
                        (= :double (get-in bridge [:body-context :return-stack-type]))
                        (contains? #{:ref :double} (get-in bridge [:body-context :emitted-result-type]))
                        (= (completion/bytecode-operand-context {})
                           (get-in bridge [:body-context :result-context])))
               (when-let [constructed (construct-double-storage-source source parameters entry source-ns)]
                 (consume!
                  constructed
                  (fn [target-parameters target-scalars target-result]
                    (when-not (and (= parameters target-parameters)
                                   (= (vec (repeat (count parameters) :double)) target-scalars)
                                   (= :double target-result))
                      (throw (ex-info "scalar storage construction has a different target ABI"
                                      {:reason :scalar-storage-target-mismatch})))
                    (method-entry/assert-frontend-selection! frontend source-var)
                    (let [historical @method-entry/*emission-inputs*]
                      (method-entry/assert-definition-inputs-current! historical)
                      (fn [] (method-entry/assert-definition-inputs-current! historical))))))))))))))

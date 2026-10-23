(ns raster.compiler.passes.scalar.normal-return
  "Conservative normal-return presence and physical producer-carrier summaries.

   The lattice is :no-return < {:present :absent} < :unknown. A present result
   says nothing about ownership, aliasing, effects, cost or whether evaluation
   throws. Calls and initializers must remain at their original source positions.
   No source code is executed and no operation/function registry is introduced."
  (:require [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
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
        (assoc :array-tag (:array-tag a))))))

(declare summary)

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
                       (method-entry/observed-selected-definition :typed semantic-op implementation-op (count arguments))
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
               (method-entry/selected-definition-body definition)]
        (if (contains? visited source-var)
          unknown
          (let [;; Only historical producer-owned bodies may prove constructors
                ;; or aliases. Current walked metadata is not code evidence.
                body (seq body)]
            ;; Transport actual producer facts, never presence from formal tags.
            ;; A changed input carrier cannot certify a callee's array result.
            (if (seq body)
              (summary (apply list 'do body)
                       (zipmap parameters
                               (map (fn [argument tag]
                                      (let [fact (summary argument env visited body-context)]
                                        (if (= tag (:array-tag fact)) fact
                                            (dissoc fact :array-tag))))
                                    arguments tags))
                       (conj visited source-var) selected-body)
              unknown)))
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
          (cond-> {:presence :present}
            (contains? types/primitive-array-info tag) (assoc :array-tag tag)))
        ;; Use the existing pure-map materializer to expose its actual
        ;; constructor and returned local; do not duplicate SOAC lowering.
        (and (= 'raster.par/pmap operation) (par/par-map-pure-form? expression))
        (materialized-result expression env visited body-context)
        ;; Only explicit destination identity contracts may forward a fact.
        (and (symbol? operation) (= operation (first expression))
             ;; form-info accepts surface aliases such as par/scan. Such a
             ;; spelling is not a closed semantic operation in raw source.
             (= "raster.par" (namespace operation))
             (some? (:return-alias-arg (form/form-info expression))))
        (let [index (:return-alias-arg (form/form-info expression))]
          (summary (nth (descriptor/call-args expression) index nil) env visited body-context))
        :else (selected-result expression env visited body-context))))))

(defn result-fact
  "Normal-return presence plus an independently derived primitive-array carrier.
   Lexical facts come from actual producers; formal/result annotations cannot
   establish either presence or a physical array carrier. Extra owner-local
   facts are ignored, and no call is executed or moved by this analysis.
   Selected definitions require an enclosing compilation-input owner; otherwise
   definition-dependent facts are unknown rather than silently losing provenance."
  ([expression] (result-fact expression {}))
  ([expression lexical-facts]
   (select-keys (summary expression lexical-facts #{}) [:presence :array-tag])))

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

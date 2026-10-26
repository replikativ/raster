(ns raster.compiler.passes.scalar.source-import
  "Owner-local source-import adapter for isolated analyzed numerical components.
   Receipts, original JVM return recipes and unresolved source completions stay
   outside the portable numerical frontend and its TypedSOAC programs."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason :front-end :analyzed-source))))

(deftype ^:private TerminalReductionImportReceipt [artifact guard]
  method-entry/RetainedProducerInput
  (assert-producer-current! [this]
    (guard)
    this))

(defn- validate-owned-import!
  [{:keys [receipt] :as imported} required-key reason]
  (when-not (and (instance? TerminalReductionImportReceipt receipt)
                 (contains? imported required-key)
                 (= (set (keys imported))
                    (conj (set (keys (.-artifact ^TerminalReductionImportReceipt receipt))) :receipt))
                 (every? (fn [[key original]] (identical? original (get imported key)))
                         (.-artifact ^TerminalReductionImportReceipt receipt)))
    (fail! reason "terminal reduction artifact differs from its exact owner-local construction" {}))
  (method-entry/assert-producer-current! receipt)
  imported)

(defn- retain-owned-import!
  ([artifact frontend] (retain-owned-import! artifact frontend nil))
  ([artifact frontend additional-guard]
  (let [source-var (get-in frontend [:source-entry :source-var])
        emission (:emission frontend)
        ;; Capture local history before adopting this receipt. Neither caller
        ;; history nor the receipt itself may become a recursive dependency.
        historical @method-entry/*emission-inputs*
        guard #(do (method-entry/assert-definition-inputs-current! historical)
                   (method-entry/assert-frontend-source! frontend source-var)
                   (method-entry/assert-emission-inputs-current! emission)
                   (when additional-guard (additional-guard)))
        receipt (TerminalReductionImportReceipt. artifact guard)
        imported (assoc artifact :receipt receipt)]
    (method-entry/assert-producer-current! receipt)
    (method-entry/adopt-definition-inputs! {:producer-inputs [receipt]})
    imported)))

(defn ^:no-doc validate-terminal-reduction-import!
  "Check the exact isolated component and its historical owner. The receipt and
   Methods are process-local; they are never facts in the portable numerical IR."
  [imported]
  (validate-owned-import! imported :component :terminal-reduction-import-correspondence))

(defn ^:no-doc validate-terminal-reduction-return!
  "Check the exact return recipe and its prepared frontend owner, without analysis or JIT.
   Consumers must check this complete artifact, not treat a detached recipe as permission."
  [returned]
  (validate-owned-import! returned :recipe :terminal-reduction-return-correspondence))

(defn ^:no-doc terminal-reduction-return-contract
  "Authenticate an already prepared singleton reduction's original Object return.
   Returns formal parameter carriers, a canonical recipe/value port and the existing
   owner-local receipt. Validation is the owner check; no numerical analysis, JIT,
   argument adaptation, placement, transfer or execution permission is supplied."
  [{:keys [frontend input-body root]}]
  (when-not method-entry/*emission-inputs*
    (fail! :terminal-reduction-import-owner "terminal import requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (let [source-var (get-in frontend [:source-entry :source-var])
           _ (method-entry/assert-frontend-source! frontend source-var)
           selected (method-entry/assert-frontend-selection! frontend source-var)
           _ (when-not (and (identical? input-body (:body frontend))
                            (= 1 (count input-body)) (identical? root (first input-body))
                            (par/par-reduce-form? root))
               (fail! :terminal-reduction-import-root
                      "terminal import requires the exact original singleton reduction body" {}))
           bridge (method-entry/frontend-boxed-entry-bridge frontend)
           emission (:emission frontend)
           body-context (when bridge
                          ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-body-context)
                           emission input-body))
           observed (when body-context
                      ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-reduction-emission)
                       emission input-body root))
           ^java.lang.reflect.Method compute (:compute-method bridge)
           carrier (when compute (.getReturnType compute))
           result-tag (get {Double/TYPE 'double Float/TYPE 'float} carrier)
           payload (get types/primitive-info result-tag)
           context (:result-context body-context)
           _ (when-not (and bridge body-context observed result-tag
                            (= :boxed-primitive-static-return (:kind bridge))
                            (identical? carrier (:payload-carrier bridge))
                            (= body-context (:body-context bridge))
                            (= :ref (:emitted-result-type body-context))
                            (= (keyword (name result-tag)) (:return-stack-type body-context))
                            (= {:carrier Object :presence :present :builtin-class payload}
                               (:result-fact observed))
                            (= :ref (get-in observed [:root-result :result-type]))
                            (= context (get-in observed [:root-result :context]))
                            (= 1 (count (get-in observed [:root-result :body])))
                            (identical? (get-in observed [:loop :expression])
                                        (first (get-in observed [:root-result :body]))))
               (fail! :terminal-reduction-import-return
                      "terminal import has no exact observed FP compute/Object return bridge" {}))
           parameters (get-in frontend [:source-entry :params])
           parameter-carriers (:compute-parameter-carriers bridge)
           _ (when-not (and (vector? parameters) (every? symbol? parameters)
                            (vector? parameter-carriers)
                            (= (count parameters) (count parameter-carriers) (.getParameterCount compute))
                            (every? true? (map identical? parameter-carriers (.getParameterTypes compute))))
               (fail! :terminal-reduction-import-return
                      "terminal return formals differ from the actual compute Method" {}))
           port (with-meta (gensym "terminal_object_") {:tag 'Object :raster.type/tag result-tag})
           recipe (completion/complete-value
                   completion/object-input-contract
                   (completion/complete-value
                    (completion/planned-bytecode-input-contract carrier context :ref)
                    (completion/complete-value (completion/bytecode-result-context-contract context :ref) port)))]
       (retain-owned-import!
        {:frontend frontend :input-body input-body :root root :selection selected
         :parameters parameters :parameter-carriers parameter-carriers
         :source-policy (get-in frontend [:source-entry :source-policy])
         :result-carrier carrier :result-tag result-tag :value-port port :recipe recipe
         :compute-method compute :invocation-method (:invocation-method bridge)
         :emission emission}
        frontend)))))

(defn ^:no-doc analyze-terminal-reduction-import
  "Construct an isolated analyzed terminal-reduction component under its source owner.
   No pipeline deferral or execution capability is installed. Lexical array arguments
   are the bounded input grammar. Original input, return bridge and caller binder
   completions remain explicit unresolved obligations, not portable payload proofs."
  [{:keys [frontend caller-expression arguments result-symbol source-position
           values array-types scalar-types] :as request}]
  (when-not method-entry/*emission-inputs*
    (fail! :terminal-reduction-import-owner "terminal import requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (let [returned (terminal-reduction-return-contract request)
           _ (validate-terminal-reduction-return! returned)
           {:keys [parameters parameter-carriers result-tag source-policy]} returned
           call-arguments (vec (util/call-args caller-expression))
           _ (when-not (and (seq? caller-expression)
                            (= '.invk (first caller-expression))
                            (= (get-in returned [:selection :implementation-op]) (second caller-expression))
                            (map? values) (map? array-types) (map? scalar-types)
                            (vector? arguments) (= arguments call-arguments)
                            (= (count arguments) (count parameters) (count parameter-carriers))
                            (every? symbol? arguments) (symbol? result-symbol)
                            (not (contains? values result-symbol))
                            (integer? source-position) (<= 0 source-position)
                            (every? (fn [[argument parameter-carrier]]
                                      (let [expected (dtype/dtype-for-jvm-array-class parameter-carrier)
                                            value (get values argument)]
                                        (and (av/abstract-value? value)
                                             (contains? #{:double :float} expected)
                                             (= :tensor (:kind value)) (seq (:shape value))
                                             (= expected (some-> (:dtype value) dtype/canon))
                                             (= expected (some-> (get array-types argument) dtype/canon)))))
                                    (map vector arguments parameter-carriers)))
               (fail! :terminal-reduction-import-arguments
                      "terminal import requires the selected call over preceding compatible lexical arrays" {}))
           policy source-policy
           numerical-result (with-meta (gensym "terminal_numerical_") {:raster.type/tag result-tag})
           source-env (merge (into {} (map (fn [[id type]] [id (:array-tag (dtype/info type))])) array-types)
                             (into {} (map (fn [[id type]] [id (symbol (name type))])) scalar-types))
           expansion (source-policy/call-with-policy
                      policy
                      #((requiring-resolve 'raster.compiler.passes.scalar.inline/expand-original-helper-source)
                        frontend caller-expression arguments numerical-result source-env))
           source (list 'let* (vec (mapcat identity (:bindings expansion))) numerical-result)
           options {:dtype (keyword (name result-tag)) :values values :array-types array-types
                    :scalar-types scalar-types :lexical-inputs (vec (keys values))
                    :source-ns (get-in frontend [:source-entry :source-ns])
                    :preserve-declared-array-storage? true :segmented-plans? true}
           normalized (source-policy/call-with-policy policy #(frontend/normalize-source source options))
           component (source-policy/call-with-policy policy #(frontend/form->program-components normalized options))
           program (:typed-program component)
           _ (when-not (and program (= [numerical-result] (:outputs component))
                            (= (keyword (name result-tag)) (get-in component [:values numerical-result :dtype])))
               (fail! :terminal-reduction-import-analysis
                      "terminal helper did not produce its checked isolated numerical component" {}))
           _ (dialect/validate! program)
           obligations {:inputs (mapv (fn [argument parameter-carrier]
                                        {:argument argument :parameter-carrier parameter-carrier
                                         :status :unresolved}) arguments parameter-carriers)
                        :return {:value-port (:value-port returned) :recipe (:recipe returned)
                                 :compute-method (:compute-method returned)
                                 :invocation-method (:invocation-method returned) :status :unresolved}
                        :import {:result numerical-result :status :unresolved}
                        :placement {:source-position source-position :status :unresolved}
                        :caller-binding {:binder result-symbol :initializer caller-expression :status :unresolved}}
           artifact {:component component :source source :normalized-source normalized
                     :argument-bindings (:argument-bindings expansion)
                     :caller-expression caller-expression :result-symbol result-symbol
                     :source-position source-position :source-policy policy :obligations obligations}
           imported (retain-owned-import! artifact frontend)]
       (validate-terminal-reduction-import! imported)
       imported))))

(defn- source-completion-binding!
  [source-program plan step]
  (when-not (and (invocation/source-completion? step)
                 (identical? plan (get-in source-program [:attributes :invocation-plan]))
                 (= 1 (count (filter #(identical? step %) (:steps plan)))))
    (fail! :source-completion-step-owner
           "completion requires the actual step of the source program's attached plan" {}))
  (parallel-program/validate! source-program)
  (invocation/validate-against! plan source-program)
  (let [order (invocation/source-order plan source-program)
        pairs (vec (partition 2 (second (:source source-program))))
        sites (filterv #(= (:symbol step) (first (second %))) (map-indexed vector pairs))
        [position [binder expression]] (first sites)
        {:keys [contract operand]} (completion/canonical-parts expression)
        operands (:operands step)
        input (first operands)
        entries (filterv #(= (:id step) (:invocation-step %)) order)]
    (when-not (and (= 1 (count sites)) (= 1 (count entries))
                   (= position (:position (first entries)))
                   (identical? binder (:symbol step))
                   (identical? expression (:expression step))
                   (= :source-floating-cast (:source-context contract))
                   (completion/canonical-contract? contract)
                   (= 'float (completion/source-cast-result-tag contract))
                   (symbol? operand) (= 1 (count operands))
                   (identical? operand (:symbol input))
                   (= (av/make {:kind :opaque :memory-space :host :ownership :borrowed})
                      (get-in plan [:values (:value input)]))
                   (= (av/tensor {:dtype :float :shape []}) (:value step)))
      (fail! :source-completion-step-source
             "completion requires its exact retained Float binding over one opaque Object value" {}))
    {:source-position position :binder binder :expression expression
     :operand operand :operand-id (:value input)}))

(defn ^:no-doc validate-source-completion-step!
  "Authenticate a complete owner-local adapter against the CURRENT execution boundary.
   Call before transfer/invocation; a detached region or callable map is not an adapter.
   This validates construction correspondence, not transfer or execution completion."
  [adapter source-program plan step]
  (when-not (and (identical? source-program (:source-program adapter))
                 (identical? plan (:plan adapter)) (identical? step (:step adapter)))
    (fail! :source-completion-step-correspondence
           "completion adapter belongs to another source program, plan or step" {}))
  (validate-owned-import! adapter :region :source-completion-step-correspondence))

(defn ^:no-doc compile-source-completion-step
  "Construct a NEW Object -> primitive Float region for one exact attached source step.
   `{:frontend prepared :source-program semantic :plan attached :step actual}`
   supplies the original frontend only as loader/history owner, never as an observed
   occurrence of this planned recipe. No JIT, result transfer, cursor advancement or
   runtime admission occurs. The caller must validate the complete returned adapter
   against its current program/plan/step before using `[:region :invoke]` on one Object."
  [{:keys [frontend source-program plan step]}]
  (when-not method-entry/*emission-inputs*
    (fail! :source-completion-step-owner "completion construction requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (let [source-var (get-in frontend [:source-entry :source-var])
           _ (method-entry/assert-frontend-source! frontend source-var)
           physical (method-entry/frontend-body-entry-carriers frontend source-var)
           emission (:emission frontend)
           policy (get-in frontend [:source-entry :source-policy])
           _ (when-not (and emission (identical? emission (:emission physical))
                            (not (:pending-source? physical)) (source-policy/valid? policy))
               (fail! :source-completion-step-frontend
                      "completion requires an already prepared exact frontend emission" {}))
           _ (method-entry/retain-emission-input-history! emission)
           binding (source-completion-binding! source-program plan step)
           historical @method-entry/*emission-inputs*
           owner-check #(do (method-entry/assert-definition-inputs-current! historical)
                            ;; The source/program/plan are immutable objects sealed into
                            ;; this adapter after structural validation above. Reuse must
                            ;; check their exact identity plus live historical inputs, not
                            ;; reconstruct the whole source-order proof on every cast.
                            (method-entry/assert-frontend-source! frontend source-var))
           region ((requiring-resolve
                    'raster.compiler.backend.jvm.bytecode/compile-planned-source-floating-completion-region)
                   emission (:expression binding) policy owner-check)
           artifact (merge binding {:frontend frontend :source-program source-program
                                    :plan plan :step step :source-policy policy :region region})
           adapter (retain-owned-import! artifact frontend (:guard region))]
       (validate-source-completion-step! adapter source-program plan step)))))

(ns raster.compiler.passes.scalar.source-import
  "Owner-local source construction, current-emission observations and import evidence.
   Receipts, original JVM return recipes and unresolved source completions stay
   outside the portable numerical frontend and its TypedSOAC programs."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]))

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

(defn- retain-owned-artifact!
  [artifact owner-guard]
  ;; Capture local history before adopting this receipt. Neither caller
  ;; history nor the receipt itself may become a recursive dependency.
  (let [historical @method-entry/*emission-inputs*
        guard #(do (method-entry/assert-definition-inputs-current! historical)
                   (owner-guard))
        receipt (TerminalReductionImportReceipt. artifact guard)
        imported (assoc artifact :receipt receipt)]
    (method-entry/assert-producer-current! receipt)
    (method-entry/adopt-definition-inputs! {:producer-inputs [receipt]})
    imported))

(defn- retain-owned-import!
  ([artifact frontend] (retain-owned-import! artifact frontend nil))
  ([artifact frontend additional-guard]
   (retain-owned-artifact!
    artifact
    #(do (method-entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
         (method-entry/assert-emission-inputs-current! (:emission frontend))
         (when additional-guard (additional-guard))))))

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

(defn ^:no-doc construct-contextual-terminal-return
  "Discharge a selected helper's return into executable canonical source.
   Used only by the generated-program constructor, before caller hygiene. The
   original helper proves this construction; no historical AST identity travels
   with the resulting code. The current contextual reduction still needs its
   own numerical/result-import consumer and actual normal-return evidence.
   In particular, the fresh local's Object carrier is NOT a Double payload fact."
  [{:keys [return-contract argument-bindings instantiated-body binding-facts]}]
  (validate-terminal-reduction-return! return-contract)
  (let [object (completion/canonical-parts (:recipe return-contract))
        returned (completion/canonical-parts (:operand object))
        contextual (completion/canonical-parts (:operand returned))
        carrier (:result-carrier return-contract)
        context (get-in contextual [:contract :operand-context])]
    (when-not (and (= completion/object-input-contract (:contract object))
                   (= :jvm-bytecode-floating-input (get-in returned [:contract :source-context]))
                   (identical? carrier (get-in returned [:contract :target-carrier]))
                   (= :ref (get-in returned [:contract :expected-input-carrier]))
                   (= :jvm-bytecode-result-context (get-in contextual [:contract :source-context]))
                   (= :ref (get-in contextual [:contract :result-carrier]))
                   (= context (get-in returned [:contract :operand-context]))
                   (identical? (:value-port return-contract) (:operand contextual))
                   (vector? argument-bindings) (even? (count argument-bindings))
                   (map? binding-facts) (seq? instantiated-body))
      (fail! :terminal-reduction-return-construction
             "terminal helper has no complete contextual FP/Object return recipe" {}))
    (let [result (with-meta (gensym "contextual_reduction__")
                   {:tag 'Object :raster.type/tag (:result-tag return-contract)})
          ;; Original helper equivalence has been discharged by this constructor.
          ;; Its old JVM occurrence tokens are not metadata of the NEW numerical
          ;; source; the current-source witness will emit fresh owned occurrences.
          numerical-source ((requiring-resolve
                             'raster.compiler.backend.jvm.bytecode/detach-emission-occurrences)
                            instantiated-body)
          root (completion/complete-value (:contract contextual) numerical-source)
          tail (completion/complete-value
                (:contract object)
                (completion/complete-value (:contract returned) result))]
      {:expression (list 'let* (conj argument-bindings result root) tail)
       ;; These are expectations for the NEW program's eventual emission. Its
       ;; observer must check them, not mistake them for original local facts.
       :binding-facts (assoc binding-facts result
                             {:type :ref :hint 'Object :verified false
                              :reference-class Object})})))


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
        result-description (completion/invocation-result-description contract)
        operands (:operands step)
        input (first operands)
        input-value (get-in plan [:values (:value input)])
        literal-carrier (completion/numeric-object-literal-carrier expression)
        scalar-value? #(and (= :tensor (:kind %)) (contains? #{:float :double :long :int} (:dtype %))
                            (= [] (:shape %)))
        opaque-value? #(and (= :opaque (:kind %)) (= [] (:shape %)))
        input-kind (cond literal-carrier :literal
                         (scalar-value? input-value) :scalar (opaque-value? input-value) :opaque)
        input-dtype (if literal-carrier literal-carrier
                        (when (= :scalar input-kind) (:dtype input-value)))
        result-kind (:kind result-description)
        result-dtype (some-> (:tag result-description) dtype/dtype-for-scalar-tag)
        entries (filterv #(= (:id step) (:invocation-step %)) order)]
    (when-not (and (= 1 (count sites)) (= 1 (count entries))
                   (= position (:position (first entries)))
                   (identical? binder (:symbol step))
                   (identical? expression (:expression step))
                   (completion/canonical-contract? contract)
                   ;; Checked integral SOURCE casts retain their existing separate
                   ;; path. This extension consumes canonical numeric ENTRY plans.
                   (not= :source-integral-cast (:source-context contract))
                   (or (and (= :scalar result-kind) (contains? #{:float :double :long :int} result-dtype))
                       (= {:kind :opaque} result-description))
                   (or (and literal-carrier (empty? operands))
                       (and (symbol? operand) (= 1 (count operands))
                            (identical? operand (:symbol input))))
                   input-kind
                   (if (= :opaque result-kind) (opaque-value? (:value step))
                       (and (scalar-value? (:value step)) (= result-dtype (get-in step [:value :dtype])))))
      (fail! :source-completion-step-source
             "completion requires its exact bounded lexical binding or numeric Object literal" {}))
    {:source-position position :binder binder :expression expression
     :operand operand :operand-id (:value input) :input-kind input-kind :result-kind result-kind
     :input-dtype input-dtype :result-dtype result-dtype}))

(declare validate-contextual-source-witness! current-source-binding!)

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
  "Construct a NEW bounded completion region for one exact attached source step.
   `{:frontend prepared :source-program semantic :plan attached :step actual}`
   supplies the original frontend only as loader/history owner, never as an observed
   occurrence of this planned recipe. No JIT, result transfer, cursor advancement or
   runtime admission occurs. The caller must validate the complete returned adapter
   against its current program/plan/step before using `[:region :invoke]`. An optional
   current-source witness authenticates actual bounded numeric/Object ports and result carriers;
   without it, only the original deliberate Object-to-source-Float ABI is supported."
  [{:keys [frontend source-program plan step witness]}]
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
           current (when witness
                     (validate-contextual-source-witness! witness)
                     (when-not (and (identical? (:source source-program) (:scope witness))
                                    (identical? source-var (:source-var witness))
                                    (identical? (:source-ns emission) (:source-ns witness))
                                    (= policy (:source-policy witness)))
                       (fail! :source-completion-step-witness
                              "completion requires the exact current source under its original owner" {}))
                     (current-source-binding!
                      witness (:source-position binding) (:binder binding) (:expression binding)
                      :source-completion-step-witness :source-completion-step-witness))
           observed (when current
                      ((requiring-resolve
                        'raster.compiler.backend.jvm.bytecode/retained-input-completion-context)
                       (:emission witness) (:input-body witness) (:expression binding)))
           literal? (= :literal (:input-kind binding))
           port (when (and current (not literal?)) (get (:before current) (:operand binding)))
           _ (if witness
               (when-not (and observed
                              (identical? (:expression binding) (:source-input observed))
                              (= (:contract (completion/canonical-parts (:expression binding))) (:contract observed))
                              (if literal?
                                (and (nil? port)
                                     (= (:input-dtype binding)
                                        (completion/numeric-object-literal-carrier (:expression binding)))
                                     (= :ref (:input-carrier observed) (:completion-carrier observed)))
                                (and (= port (get (:locals-before observed) (:operand binding)))
                                     (if (= :scalar (:input-kind binding))
                                       (= (:input-dtype binding) (:type port))
                                       (and (= :ref (:type port))
                                            (contains? #{Object Float Double Long Integer}
                                                       (:reference-class port))))))
                              (if (= :opaque (:result-kind binding))
                                (and (= :ref (get-in current [:after :type]))
                                     (identical? Object (get-in current [:after :reference-class])))
                                (= (:result-dtype binding) (get-in current [:after :type]))))
                 (fail! :source-completion-step-carrier
                        "completion source value or binding differs from its actual physical port/result" {}))
               (when-not (and (= :opaque (:input-kind binding)) (= :scalar (:result-kind binding))
                              (= :source-floating-cast
                                 (get-in (completion/canonical-parts (:expression binding)) [:contract :source-context])))
                 (fail! :source-completion-step-source
                        "completion without a current witness requires the deliberate Object-to-source-Float ABI" {})))
           operand-policy (if (= :source-floating-cast (:source-context (:contract observed)))
                            (:operand-source-policy observed) policy)
           historical @method-entry/*emission-inputs*
           owner-check #(do (method-entry/assert-definition-inputs-current! historical)
                            ;; The source/program/plan are immutable objects sealed into
                            ;; this adapter after structural validation above. Reuse must
                            ;; check their exact identity plus live historical inputs, not
                            ;; reconstruct the whole source-order proof on every cast.
                            (method-entry/assert-frontend-source! frontend source-var)
                            (when witness (validate-contextual-source-witness! witness)))
           region (if witness
                    ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-planned-completion-region)
                     emission (:expression binding) port observed operand-policy owner-check)
                    ((requiring-resolve
                      'raster.compiler.backend.jvm.bytecode/compile-planned-source-floating-completion-region)
                     emission (:expression binding) policy owner-check))
           artifact (cond-> (merge binding {:frontend frontend :source-program source-program
                                           :plan plan :step step :source-policy policy :region region
                                           :input-class (when (= :scalar (:input-kind binding))
                                                          (get types/primitive-info (symbol (name (:type port)))))
                                           :result-class (when (= :scalar (:result-kind binding))
                                                           (get types/primitive-info (symbol (name (:result-dtype binding)))))})
                      witness (assoc :witness witness :input-port port :observed observed))
           adapter (retain-owned-import! artifact frontend (:guard region))]
       (validate-source-completion-step! adapter source-program plan step)))))

(declare observe-contextual-reduction!)

(defn ^:no-doc validate-program-result-import-step!
  "Authenticate an internal result-return adapter, not producer completion or read permission."
  [adapter source-program emitted plan step]
  (when-not (and (identical? source-program (:source-program adapter))
                 (identical? emitted (:emitted-program adapter))
                 (identical? plan (:plan adapter)) (identical? step (:step adapter)))
    (fail! :program-result-import-correspondence
           "result adapter belongs to another compilation, plan or import step" {}))
  (validate-owned-import! adapter :region :program-result-import-correspondence))

(defn ^:no-doc compile-program-result-import-step
  "Construct the owned return recipe for one internal numerical-result demand.
   The payload Class comes from an exact current JVM reduction observation, never
   its AV dtype. Checked emitted numerical ownership is separate. No allocation,
   transfer, execution completion or cursor advancement is authorized here."
  [{:keys [frontend source-program emitted plan step witness retained-validation caller-options]}]
  (when-not method-entry/*emission-inputs*
    (fail! :program-result-import-owner "result adapter requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (parallel-program/validate! source-program)
     (invocation/validate-against! plan source-program)
     (validate-contextual-source-witness! witness)
     (let [source-var (get-in frontend [:source-entry :source-var])
           _ (method-entry/assert-frontend-source! frontend source-var)
           original (method-entry/frontend-body-entry-carriers frontend source-var)
           policy (get-in frontend [:source-entry :source-policy])
           _ (when-not (and (:emission frontend) (not (:pending-source? original))
                            (identical? (:emission frontend) (:emission original))
                            (identical? source-var (:source-var witness))
                            (identical? (get-in frontend [:source-entry :source-ns]) (:source-ns witness))
                            (identical? (:source source-program) (:scope witness))
                            (= policy (:source-policy witness))
                            (identical? plan (get-in source-program [:attributes :invocation-plan]))
                            (invocation/program-result-import? step)
                            (= 1 (count (filter #(identical? step %) (:steps plan)))))
               (fail! :program-result-import-owner "result adapter requires its exact prepared source owner" {}))
           _ (method-entry/retain-emission-input-history! (:emission frontend))
           pairs (vec (partition 2 (second (:source source-program))))
           sites (filterv #(= (:symbol step) (first (second %))) (map-indexed vector pairs))
           [position [binder expression]] (first sites)
           roots (filterv #(= (:result step) (first (second %))) (map-indexed vector pairs))
           [producer-position [root-symbol root-source]] (first roots)
           producers (filterv #(= (:equation step) (:id %)) (:equations source-program))
           producer (first producers)
           _ (when-not (and (= 1 (count sites) (count roots) (count producers))
                            (< producer-position position)
                            (identical? binder (:symbol step))
                            (identical? expression (:source-expression step))
                            (= {:equation (:equation step) :result (:result step)}
                               (get (invocation/result-import-demands source-program [[binder expression]]) binder))
                            (identical? producer
                                        (parallel-program/equation-for-binding source-program root-symbol root-source)))
               (fail! :program-result-import-source "result demand lost its exact preceding producer/snapshot" {}))
           observed (observe-contextual-reduction! witness producer-position root-symbol root-source)
           fact (:result-fact observed)
           payload-class (:builtin-class fact)
           payload-carrier (get {Float Float/TYPE Double Double/TYPE} payload-class)
           result-dtype (some-> (:result-tag observed) name keyword)
           checked (if retained-validation
                     (emitted-program/checked-retained-validation! emitted retained-validation caller-options)
                     (emitted-program/validate-with-physical-results! emitted caller-options))
           emitted-producers (filterv #(= (:id producer) (:id %)) (:equations (:program checked)))
           emitted-producer (first emitted-producers)
           _ (when-not (and payload-carrier (= :present (:presence fact))
                            (identical? payload-class (get types/primitive-info (:result-tag observed)))
                            (identical? Object (:carrier fact))
                            (= result-dtype (get-in source-program [:values root-symbol :dtype]))
                            (= [] (get-in source-program [:values root-symbol :shape]))
                            (= 1 (count emitted-producers))
                            (semantic/equivalent? (select-keys producer [:id :operands :results :algorithm])
                                                  (select-keys emitted-producer [:id :operands :results :algorithm]))
                            (= (get-in source-program [:values root-symbol])
                               (get-in (:program checked) [:values root-symbol])))
               (fail! :program-result-import-payload
                      "actual boxed FP return differs from the retained numerical producer" {}))
           history @method-entry/*emission-inputs*
           guard #(do (method-entry/assert-definition-inputs-current! history)
                      (validate-contextual-source-witness! witness)
                      (method-entry/assert-frontend-source! frontend source-var))
           region ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-planned-result-return-region)
                   (:emission witness) expression root-symbol payload-carrier policy guard)
           adapter (retain-owned-import!
                    {:frontend frontend :source-program source-program :emitted-program emitted
                     :plan plan :step step :witness witness :producer producer
                     :emitted-producer emitted-producer :source-position position
                     :producer-position producer-position :result root-symbol :expression expression
                     :payload-class payload-class :payload-carrier payload-carrier :dtype result-dtype
                     :region region}
                    frontend #(do (guard) ((:guard region))))]
       (validate-program-result-import-step! adapter source-program emitted plan step)))))

(defn- unchanged-entry-tag
  "Encode an actual entry Class using existing builtin interface tags, never an AV dtype."
  [^Class carrier]
  (cond
    (.isPrimitive carrier) (when-not (= Void/TYPE carrier) (symbol (.getName carrier)))
    (= Object carrier) 'Object
    (= (.arrayType Object) carrier) 'objects
    :else (some (fn [[tag actual]] (when (identical? actual carrier) tag)) types/primitive-array-info)))

(defn ^:no-doc validate-contextual-source-witness!
  "Validate this exact NEW whole-source construction and its historical inputs.
   No old source occurrence, numerical equivalence or transfer permission is asserted."
  [witness]
  (validate-owned-import! witness :scope :contextual-source-witness-correspondence))

(defn ^:no-doc validate-original-public-float-return!
  "Validate an original Float return contract and its exact current-source owner.
   This does not discharge the current source's terminal value/alias obligations."
  ([returned]
   (validate-owned-import! returned :witness :original-public-float-return-correspondence))
  ([returned witness]
   (validate-original-public-float-return! returned)
   (when-not (identical? witness (:witness returned))
     (fail! :original-public-float-return-correspondence
            "public return contract belongs to another current source witness" {}))
   returned))

(defn ^:no-doc original-public-float-return-contract
  "Read an already prepared ORIGINAL public Float compute/Object wrapper edge.
   The current witness supplies only unchanged entry correspondence and freshness;
   its artificial Object return is never public-return evidence. No preparation,
   JIT, terminal alias analysis, numerical import or execution admission occurs."
  [original witness]
  (when-not method-entry/*emission-inputs*
    (fail! :original-public-float-return-owner "public Float return requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (validate-contextual-source-witness! witness)
     (let [source-var (:source-var witness)
           _ (method-entry/assert-frontend-source! original source-var)
           _ (method-entry/assert-frontend-selection! original source-var)
           entry (:source-entry original)
           bridge (method-entry/frontend-boxed-entry-bridge original)
           ^java.lang.reflect.Method compute (:compute-method bridge)
           ^java.lang.reflect.Method invocation-method (:invocation-method bridge)
           body-context (:body-context bridge)
           raw (:emitted-result-type body-context)
           context (:result-context body-context)
           parameters (:parameters witness)
           carriers (:parameter-carriers witness)
           policy (:source-policy witness)]
       (when-not (and bridge (identical? source-var (:source-var entry))
                      (identical? (:source-ns witness) (:source-ns entry))
                      (= policy (:source-policy entry)) (source-policy/valid? policy)
                      (= parameters (:params entry))
                      (vector? carriers) (every? class? carriers)
                      (= carriers (:compute-parameter-carriers bridge) (:parameter-carriers bridge))
                      (instance? java.lang.reflect.Method compute)
                      (instance? java.lang.reflect.Method invocation-method)
                      (= :boxed-primitive-static-return (:kind bridge))
                      (identical? Float/TYPE (:payload-carrier bridge))
                      (identical? Float/TYPE (.getReturnType compute))
                      (identical? Object (.getReturnType invocation-method))
                      (= (count parameters) (count carriers)
                         (.getParameterCount compute) (.getParameterCount invocation-method))
                      (every? true? (map identical? carriers (.getParameterTypes compute)))
                      (every? true? (map identical? carriers (.getParameterTypes invocation-method)))
                      (= :float (:return-stack-type body-context))
                      (contains? #{:ref :float} raw)
                      (completion/bytecode-operand-context? context))
         (fail! :original-public-float-return
                "original source has no matching prepared primitive Float public return bridge" {}))
       (let [port (with-meta (gensym "public_float_return__") {:tag (if (= :ref raw) 'Object 'float)})
             recipe (completion/complete-value
                     completion/object-input-contract
                     (completion/complete-value
                      (completion/planned-bytecode-input-contract Float/TYPE context raw)
                      (completion/complete-value
                       (completion/bytecode-result-context-contract context raw) port)))]
         (retain-owned-import!
          {:frontend original :witness witness :source-var source-var :source-policy policy
           :parameters parameters :parameter-carriers carriers
           :compute-method compute :invocation-method invocation-method
           :result-carrier Float/TYPE :result-tag 'float :body-context body-context
           :value-port port :recipe recipe}
          original #(validate-contextual-source-witness! witness)))))))

(defn ^:no-doc prepare-contextual-source-witness
  "Explicitly emit one NEW current let* body without executing its numerical code.
   Only an unchanged original flat entry ABI is supported. The original entry
   supplies actual parameter Classes and frozen policy, not occurrence ownership
   of the new code. Receipts, Methods and observations remain owner-local."
  [source {:keys [source-var source-ns lexical-inputs public-parameters source-policy] :as options}]
  (when-not method-entry/*emission-inputs*
    (fail! :contextual-source-witness-owner "current-source emission requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (let [physical (when (var? source-var) (method-entry/source-var-body-entry-carriers source-var))
           lineage (:source-entry physical)
           parameters (:parameters physical)
           carriers (:parameter-carriers physical)
           policy (:source-policy lineage)
           source-ns (or source-ns (:source-ns lineage))
           inputs (if (contains? options :lexical-inputs) lexical-inputs public-parameters)
           tags (when (and (vector? carriers) (every? class? carriers))
                  (mapv unchanged-entry-tag carriers))
           [head bindings & body] (when (seq? source) source)
           _ (when-not (and physical lineage (vector? parameters) (every? symbol? parameters)
                            (= (count parameters) (count (distinct parameters)))
                            (vector? inputs) (= inputs parameters)
                            (every? #(or (not (contains? options %)) (= parameters (get options %)))
                                    [:lexical-inputs :public-parameters])
                            (= (count parameters) (count carriers) (count tags)) (every? some? tags)
                            (instance? clojure.lang.Namespace source-ns)
                            (identical? source-ns (:source-ns lineage)) (source-policy/valid? policy)
                            (or (not (contains? options :source-policy)) (= source-policy policy))
                            (= 'let* head) (vector? bindings) (even? (count bindings)) (= 1 (count body))
                            (every? symbol? (take-nth 2 bindings))
                            (= (quot (count bindings) 2) (count (distinct (take-nth 2 bindings))))
                            (not-any? (set parameters) (take-nth 2 bindings)))
               (fail! :contextual-source-witness-entry
                      "current source requires the unchanged actual public entry and frozen policy" {}))
           ;; Keep this ORIGINAL observation before any source preparation. A
           ;; helper promotion must reject/retry the enclosing construction,
           ;; never refresh the old header around newly emitted code.
           historical @method-entry/*emission-inputs*
           interface (types/ensure-fn-interface! tags)
           input-body [source]
           emitted (source-policy/call-with-policy
                    policy
                    #((requiring-resolve 'raster.core/emit-owned-bytecode!)
                      (requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-whole-source-impl!)
                      "raster.compiled.ContextualSource" parameters tags 'Object input-body source-ns
                      (:iface-name interface) 'contextual-source historical))
           implementation (:implementation emitted)
           emission ((requiring-resolve
                      'raster.compiler.backend.jvm.bytecode/prepare-emitted-global-invocations!)
                     (:emission emitted))
           _ (method-entry/assert-definition-inputs-current! historical)
           bridge ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-boxed-entry-bridge) emission)
           ^java.lang.reflect.Method compute (:compute-method bridge)
           ^java.lang.reflect.Method invocation-method (:invocation-method bridge)
           context ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-body-context)
                    emission input-body)
           _ (when-not (and implementation bridge context
                            (identical? implementation (:implementation emission))
                            (identical? parameters (:params emission)) (= tags (:tags emission))
                            (identical? (:iface-class interface) (:interface bridge))
                            (instance? java.lang.reflect.Method compute)
                            (instance? java.lang.reflect.Method invocation-method)
                            (java.lang.reflect.Modifier/isStatic (.getModifiers compute))
                            (identical? Object (.getReturnType compute))
                            (identical? Object (.getReturnType invocation-method))
                            (identical? (:iface-class interface) (.getDeclaringClass invocation-method))
                            (= (count carriers) (.getParameterCount compute) (.getParameterCount invocation-method))
                            (every? true? (map identical? carriers (.getParameterTypes compute)))
                            (every? true? (map identical? carriers (.getParameterTypes invocation-method))))
               (fail! :contextual-source-witness-method
                      "new contextual source emitted a different physical entry ABI" {}))
           guard (method-entry/prepare-emission-input-guard! emission)
           artifact {:scope source :input-body input-body :source-var source-var :source-entry lineage
                     :source-ns source-ns :source-policy policy :parameters parameters
                     :parameter-carriers carriers :tags tags :implementation implementation
                     :interface-class (:iface-class interface) :method compute
                     :invocation-method invocation-method :emission emission}
           witness (retain-owned-artifact!
                    artifact #(do (method-entry/assert-source-entry-current! lineage) (guard)))]
       (validate-contextual-source-witness! witness)))))

(defn- current-source-binding!
  [witness ordinal binder expression form-reason observation-reason]
  (when-not method-entry/*emission-inputs*
    (fail! :contextual-source-witness-owner "current source consumption requires its compiler owner" {}))
  (validate-contextual-source-witness! witness)
  (let [{:keys [scope input-body emission]} witness
        bindings (second scope)]
    (when-not (and (integer? ordinal) (<= 0 ordinal) (< ordinal (quot (count bindings) 2))
                   (identical? binder (nth bindings (* 2 ordinal)))
                   (identical? expression (nth bindings (inc (* 2 ordinal)))))
      (fail! form-reason "source observation is not the exact current scope binding" {}))
    (let [binding ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-binding-context)
                   emission input-body scope ordinal)]
      (when-not (and binding (identical? scope (:source-input binding)) (= binder (:binder binding)))
        (fail! observation-reason "current binding has no exact owned emission context" {}))
      binding)))

(defn ^:no-doc validate-terminal-float-publication!
  [publication]
  (validate-owned-import! publication :public-return :terminal-float-publication-correspondence))

(defn ^:no-doc terminal-float-publication-candidate?
  "Cheap routing only: a symbolic terminal reaches a checked Float through
   earlier identity/Object bindings. This supplies no ABI, ownership, effect or
   sole-consumer proof; the current-source observer must discharge those."
  [source]
  (let [[head flat & body] (when (seq? source) source)
        pairs (when (and (= 'let* head) (vector? flat) (even? (count flat)))
                (vec (partition 2 flat)))
        indices (zipmap (map first pairs) (range))]
    (boolean
     (when (and pairs (every? symbol? (map first pairs))
                (= (count pairs) (count indices))
                (= 1 (count body)) (symbol? (first body)))
       (loop [symbol (first body) upper (count pairs)]
         (when-let [ordinal (get indices symbol)]
           (when (< ordinal upper)
             (let [[_ expression] (nth pairs ordinal)
                   parts (completion/canonical-parts expression)
                   contract (:contract parts)]
               (if (and (= :source-floating-cast (:source-context contract))
                        (= :float (:operation contract)))
                 true
                 (let [operand (if (symbol? expression) expression
                                   (:operand (completion/object-input-parts expression)))]
                   (when (symbol? operand) (recur operand ordinal))))))))))))

(defn ^:no-doc observe-terminal-float-publication
  "Observe a checked source Float's sole-consumer identity/Object-boxing tail.
   This retains exact current source and order; it neither rewrites source nor
   grants numerical, transfer or runtime publication permission."
  [returned]
  (when-not method-entry/*emission-inputs*
    (fail! :terminal-float-publication-owner "terminal Float observation requires its compiler owner" {}))
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (validate-original-public-float-return! returned)
     (let [witness (:witness returned)
           {:keys [scope emission input-body source-policy]} witness
           [_ flat & body] scope
           pairs (vec (partition 2 flat))
           terminal (first body)
           indices (zipmap (map first pairs) (range))
           reject! #(fail! :terminal-float-publication
                          "current source has no sole-consumer checked Float terminal chain" {})
           _ (when-not (and (= 1 (count body)) (symbol? terminal)) (reject!))
           readers
           (binding [util/*shadowing-locals* (into util/*shadowing-locals*
                                                  (concat (:parameters witness) (keys indices)))]
             (reduce (fn [uses [consumer expression]]
                       (reduce #(update %1 %2 (fnil conj #{}) consumer)
                               uses (util/free-syms expression)))
                     {} (conj (mapv (fn [ordinal [_ expression]] [ordinal expression])
                                    (range) pairs) [:terminal terminal])))
           found
           (loop [symbol terminal consumer :terminal upper (count pairs) chain []]
             (let [ordinal (get indices symbol)]
               (when-not (and (integer? ordinal) (< ordinal upper)
                              (= #{consumer} (get readers symbol))) (reject!))
               (let [[binder expression] (nth pairs ordinal)
                     observed (current-source-binding!
                               witness ordinal binder expression
                               :terminal-float-publication :terminal-float-publication)
                     parts (completion/canonical-parts expression)
                     contract (:contract parts)
                     after (:after observed)
                     entry {:ordinal ordinal :binder binder :expression expression}]
                 (if (= :source-floating-cast (:source-context contract))
                   (let [cast ((requiring-resolve
                                'raster.compiler.backend.jvm.bytecode/retained-input-completion-context)
                               emission input-body expression)]
                     (when-not (and (= :float (:operation contract))
                                    (= source-policy (:source-policy contract))
                                    (identical? expression (:source-input cast))
                                    (= contract (:contract cast))
                                    (= :ref (:input-carrier cast))
                                    (= :float (:completion-carrier cast) (:type after))) (reject!))
                     {:result binder :ordinal ordinal :chain (conj chain (assoc entry :kind :checked-float))})
                   (let [object (completion/object-input-parts expression)
                         operand (if (symbol? expression) expression (:operand object))
                         before (get (:before observed) operand)
                         object-observation
                         (when object
                           ((requiring-resolve
                             'raster.compiler.backend.jvm.bytecode/retained-input-completion-context)
                            emission input-body expression))]
                     (when-not (and (symbol? operand)
                                    (if object
                                      (and (contains? #{:float :ref} (:type before))
                                           (= :ref (:type after))
                                           (identical? Object (:reference-class after))
                                           (identical? expression (:source-input object-observation))
                                           (= completion/object-input-contract (:contract object-observation))
                                           (= :ref (:input-carrier object-observation)
                                              (:completion-carrier object-observation)))
                                      (and (contains? #{:float :ref} (:type before))
                                           (= (select-keys before [:type :reference-class :verified])
                                              (select-keys after [:type :reference-class :verified])))))
                       (reject!))
                     (recur operand ordinal ordinal
                            (conj chain (assoc entry :kind (if object :object-input :identity)))))))))]
       (retain-owned-artifact!
        (merge found {:source scope :terminal terminal :witness witness :public-return returned})
        #(validate-original-public-float-return! returned witness))))))

(defn ^:no-doc construct-terminal-float-publication
  "Construct a distinct singleton writer/read footer after the checked Float.
   Preserve every original binding and its order. The complete candidate still
   requires ordinary numerical admission and resident/result-import validation;
   this constructor itself grants no storage, execution or readback capability."
  [publication]
  (when-not method-entry/*emission-inputs*
    (fail! :terminal-float-publication-owner "terminal Float construction requires its compiler owner" {}))
  (validate-terminal-float-publication! publication)
  (let [source (:source publication)
        occupied (into (set (:parameters (:witness publication)))
                       (filter symbol?) (tree-seq coll? seq source))
        fresh (fn [prefix]
                (loop [id (gensym prefix)]
                  (if (contains? occupied id) (recur (gensym prefix)) id)))
        buffer (with-meta (fresh "rstr_float_publication__")
                 {:tag 'floats :raster.compiler/generated true})
        writer (fresh "rstr_float_publication_writer__")
        index (fresh "rstr_float_publication_index__")
        bindings (into (vec (second source))
                       [buffer (list 'clojure.core/float-array 1)
                        writer (list 'raster.par/map! buffer index 1 'clojure.core/float
                                     (:result publication))])
        candidate (with-meta
                    (list (first source) (with-meta bindings (meta (second source)))
                          (list 'clojure.core/float (list 'clojure.core/aget buffer 0)))
                    (meta source))]
    ;; Reusing a previously checked artifact in a new optional attempt must
    ;; retain its original history, not merely inspect its current freshness.
    (method-entry/adopt-definition-inputs! {:producer-inputs [(:receipt publication)]})
    {:form candidate :buffer buffer :writer writer :publication publication}))

(defn ^:no-doc observe-reference-provenance!
  "Offer the existing reference projector an actual CURRENT before-local array Class.
   `accept!` receives {:formal operand :carrier Class :current-source? true} and
   owns the shared declaration/binder-policy/effect checks. Neither after-locals
   nor hints supply provenance. Only accepted proposals adopt the same witness."
  [witness ordinal binder expression accept!]
  (when-not (fn? accept!)
    (fail! :contextual-source-reference-accept "reference observation requires its compiler predicate" {}))
  (let [binding (current-source-binding!
                 witness ordinal binder expression
                 :contextual-source-reference-form :contextual-source-reference-observation)
        reference (completion/reference-input-parts expression)
        object (completion/object-input-parts expression)
        operand (if (symbol? expression) expression (:operand (or reference object)))
        before (get (:before binding) operand)
        carrier (:reference-class before)
        target (get-in reference [:contract :target-carrier])]
    (when (and (symbol? operand) (= :ref (:type before)) (class? carrier)
               (some? (dtype/dtype-for-jvm-array-class carrier))
               (or (nil? reference)
                   (and (completion/canonical-contract? (:contract reference))
                        (not (true? (get-in reference [:contract :operand-context :void-context])))
                        (class? target) (not (.isPrimitive ^Class target))
                        (.isAssignableFrom ^Class target ^Class carrier))))
      (method-entry/call-with-optional-definition-inputs
       (fn []
         (let [proposal {:formal operand :carrier carrier :current-source? true}]
           (when (accept! proposal)
             (method-entry/adopt-definition-inputs! {:producer-inputs [(:receipt witness)]})
             proposal)))))))

(defn ^:no-doc observe-contextual-reduction!
  "Observe the exact current contextual reduction at its whole-source binding ordinal.
   Result payload and before-locals come from this NEW emission only. They are
   owner-local observations, not AV, initialization, numerical or execution proofs."
  [witness ordinal binder expression]
  (let [binding (current-source-binding!
                 witness ordinal binder expression
                 :contextual-source-reduction-form :contextual-source-reduction-observation)
        {:keys [input-body emission]} witness
        {:keys [contract operand]} (completion/canonical-parts expression)
        root operand]
    (when-not (and (completion/canonical-contract? contract)
                   (= :jvm-bytecode-result-context (:source-context contract))
                   (= :ref (:result-carrier contract)) (par/par-reduce-form? root))
      (fail! :contextual-source-reduction-form
             "contextual reduction is not the exact current scope binding" {}))
    (let [completion ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-completion-context)
                      emission input-body expression)
          observed ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-reduction-emission)
                    emission input-body root)
          fact (:result-fact observed)
          tag (get {Double 'double Float 'float} (:builtin-class fact))
          context (:operand-context contract)]
      (when-not (and completion observed tag
                     (identical? expression (:source-input completion))
                     (identical? root (:source-input observed))
                     (= :ref (get-in binding [:after :type]))
                     ;; The root's normal return does not discharge a later
                     ;; binder CHECKCAST. The actual binding policy records its
                     ;; post-check reference Class. Require Object storage, not
                     ;; the hint: a later pass may have retagged the logical
                     ;; scalar Double while its actual JVM local remains Object.
                     (identical? Object (get-in binding [:after :reference-class]))
                     (= contract (:contract completion))
                     (= context (:operand-context completion))
                     (= :ref (:input-carrier completion) (:completion-carrier completion))
                     (= {:carrier Object :presence :present :builtin-class (get types/primitive-info tag)} fact)
                     (= :ref (get-in observed [:root-result :result-type]))
                     (= context (get-in observed [:root-result :context]))
                     (= 1 (count (get-in observed [:root-result :body])))
                     (identical? (get-in observed [:loop :expression])
                                 (first (get-in observed [:root-result :body]))))
        (fail! :contextual-source-reduction-observation
               "current contextual reduction lacks its exact normal-return observation" {}))
      ;; A cached witness may have been created in an optional attempt whose
      ;; history was rolled back. Every successful consumer must retain the
      ;; SAME historical receipt in its current collector, not just check it.
      (method-entry/adopt-definition-inputs! {:producer-inputs [(:receipt witness)]})
      {:root root :result-tag tag :result-fact fact :before (:before binding)
       :result-context context :ordinal ordinal :binder binder :expression expression
       :witness witness})))

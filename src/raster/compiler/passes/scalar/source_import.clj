(ns raster.compiler.passes.scalar.source-import
  "Owner-local source construction, current-emission observations and import evidence.
   Receipts, original JVM return recipes and unresolved source completions stay
   outside the portable numerical frontend and its TypedSOAC programs."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.inline-call-plan :as call-plan]
            [raster.compiler.core.call-boundary :as call-boundary]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as invocation-source]
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
        compound (completion/planned-compound-integral-input-parts expression)
        result-description (completion/invocation-result-description contract)
        operands (:operands step)
        input (first operands)
        input-value (get-in plan [:values (:value input)])
        literal-carrier (completion/numeric-object-literal-carrier expression)
        scalar-value? #(and (= :tensor (:kind %)) (contains? #{:float :double :long :int} (:dtype %))
                            (= [] (:shape %)))
        opaque-value? #(and (= :opaque (:kind %)) (= [] (:shape %)))
        input-kind (cond compound :scalar literal-carrier :literal
                         (scalar-value? input-value) :scalar (opaque-value? input-value) :opaque)
        input-dtype (cond compound :long literal-carrier literal-carrier
                          (= :scalar input-kind) (:dtype input-value))
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
                       (and compound (seq operands)
                            (every? #(= :long (get-in plan [:values (:value %) :dtype])) operands))
                       (and (symbol? operand) (= 1 (count operands))
                            (identical? operand (:symbol input))))
                   input-kind
                   (if (= :opaque result-kind) (opaque-value? (:value step))
                       (and (scalar-value? (:value step)) (= result-dtype (get-in step [:value :dtype])))))
      (fail! :source-completion-step-source
             "completion requires its exact bounded lexical binding or numeric Object literal" {}))
    {:source-position position :binder binder :expression expression :compound? (boolean compound)
     :operand operand :operand-id (:value input) :input-kind input-kind :result-kind result-kind
     :input-dtype input-dtype :result-dtype result-dtype}))

(declare validate-contextual-source-witness! current-source-binding! current-compound-integral-frame!)

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
           compound-frame (when (:compound? binding)
                            (when-not witness
                              (fail! :source-completion-step-witness "compound input requires its final current witness" {}))
                            (current-compound-integral-frame!
                             witness (:source-position binding) current observed))
           literal? (= :literal (:input-kind binding))
           port (when (and current (not literal?) (not (:compound? binding)))
                  (get (:before current) (:operand binding)))
           _ (if witness
               (when-not (and observed
                              (identical? (:expression binding) (:source-input observed))
                              (= (:contract (completion/canonical-parts (:expression binding))) (:contract observed))
                              (cond
                                (:compound? binding) (boolean compound-frame)
                                literal?
                                (and (nil? port)
                                     (= (:input-dtype binding)
                                        (completion/numeric-object-literal-carrier (:expression binding)))
                                     (= :ref (:input-carrier observed) (:completion-carrier observed)))
                                :else
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
           region (cond
                    compound-frame
                    ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-owned-bytecode-completion-region)
                     (:emission witness) (:input-body witness) (:expression binding))
                    witness
                    ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/compile-planned-completion-region)
                     emission (:expression binding) port observed operand-policy owner-check)
                    :else
                    ((requiring-resolve
                      'raster.compiler.backend.jvm.bytecode/compile-planned-source-floating-completion-region)
                     emission (:expression binding) policy owner-check))
           input-bindings
           (if compound-frame
             (let [operands (into {} (map (juxt :symbol identity)) (:operands step))]
               (when-not (and (= (count operands) (count (:operands step)))
                              (= (set (keys operands)) (set (:parameters region)))
                              (= (:ports compound-frame) (:physical-ports region)))
                 (fail! :source-completion-step-ports "compound completion changed its exact physical input frame" {}))
               (mapv (fn [id]
                       (let [operand (get operands id)
                             value (get-in plan [:values (:value operand)])]
                         (when-not (and (= :tensor (:kind value)) (= [] (:shape value)) (= :long (:dtype value)))
                           (fail! :source-completion-step-ports "compound completion requires a scalar Long representation" {}))
                         {:symbol id :value (:value operand) :input-kind :scalar :input-dtype :long
                          :input-class Long :physical-port (get (:physical-ports region) id)})) (:parameters region)))
             (if literal? []
               [{:symbol (:operand binding) :value (:operand-id binding) :input-kind (:input-kind binding)
                 :input-dtype (:input-dtype binding)
                 :input-class (when (= :scalar (:input-kind binding))
                                (get types/primitive-info (symbol (name (:type port)))))
                 :physical-port port}]))
           artifact (cond-> (merge binding {:frontend frontend :source-program source-program
                                           :plan plan :step step :source-policy policy :region region
                                           :input-bindings input-bindings
                                           :input-class (when (= :scalar (:input-kind binding))
                                                          (if compound-frame Long
                                                            (get types/primitive-info (symbol (name (:type port))))))
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
        reference (or (completion/reference-input-parts expression)
                      (completion/reference-return-parts expression))
        object (completion/object-input-parts expression)
        operand (if (symbol? expression) expression (:operand (or reference object)))
        before (get (:before binding) operand)
        carrier (:reference-class before)
        target (get-in reference [:contract :target-carrier])
        returned? (:reference-return? reference)
        returned-observations
        (when returned?
          (mapv #((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-completion-context)
                  (:emission witness) (:input-body witness) %)
                (:completions reference)))
        actual-return (when returned? (completion/reference-return-parts (:initializer binding)))]
    (when (and (symbol? operand) (= :ref (:type before)) (class? carrier)
               (some? (dtype/dtype-for-jvm-array-class carrier))
               (or (not returned?)
                   (and (= :ref (get-in binding [:after :type]))
                        (identical? carrier (get-in binding [:after :reference-class]))
                        actual-return
                        (= operand (:operand actual-return))
                        (= [:checkcast :identity] (mapv :adaptation returned-observations))
                        (every? true? (map identical? (:completions actual-return)
                                           (map :expression returned-observations)))
                        (every? (fn [[source observed]]
                                  (and observed (identical? source (:source-input observed))
                                       (= (:contract (completion/canonical-parts source)) (:contract observed))
                                       (= :ref (:input-carrier observed) (:completion-carrier observed))))
                                (map vector (:completions reference) returned-observations))))
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

;; JVM lookup is deferred: importing source helpers must not load an emitter.
(defn- helper-backend [name]
  (or (some-> (find-ns 'raster.compiler.backend.jvm.bytecode) (ns-resolve name))
      (fail! :generated-body-unsupported "helper construction needs its existing JVM producer" {:function name})))

(defn- decline! [message data]
  (throw (ex-info message (assoc data :reason :generated-body-unsupported))))

(defn- reference-check [carrier value]
  (if (or (nil? carrier) (= Object carrier)) value
      (completion/complete-value (completion/reference-check-contract carrier) value)))

(def ^:private local-fact-keys [:type :hint :verified :reference-class])

(defn- helper-source-nodes [source]
  (tree-seq #(and (coll? %)
                  (not (and (seq? %) (contains? '#{quote clojure.core/quote} (first %)))))
            seq source))

(defn- import-owned-pmap-spine
  [emission original destination import-child! binding-contexts]
  (when (and destination (= 1 (count (:body emission)))
             (identical? original (first (:body emission))))
    (when-let [{:keys [scope bindings array-class] :as spine}
               ((helper-backend 'retained-pmap-spine) emission original)]
      (when (and (= array-class (:return-class destination))
                 (every? #(= (:context destination) (:input-context %)) bindings))
        (let [children (java.util.IdentityHashMap.)
              _ (doseq [child [(nth original 2) (nth original 4)]]
                  (.put children child (import-child! child)))
              ;; Rebuild only compiler-owned wrappers. Original descendants
              ;; keep their actual invocation/binding checks and lookup owner.
              rebuilt (letfn [(visit [node]
                                (cond
                                  (.containsKey children node) (.get children node)
                                  (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
                                  (and (seq? node) (= 'let* (first node)) (vector? (second node)))
                                  (let [observations (mapv #((helper-backend 'retained-binding-context)
                                                             emission node %)
                                                           (range (quot (count (second node)) 2)))]
                                    (when-not (every? some? observations)
                                      (decline! "Owned pmap wrapper has no actual binding chain" {}))
                                    (let [pairs (mapcat (fn [[id init] observed]
                                                        [(if-let [hint (get-in observed [:after :hint])]
                                                           (vary-meta id assoc :tag hint) id)
                                                         (visit init)])
                                                      (partition 2 (second node)) observations)
                                          result (util/remake-from node
                                                   (list* 'let* (vec pairs) (mapv visit (nnext node))))]
                                      ;; Generated ordinal and outer storage
                                      ;; bindings use their actual emitted local
                                      ;; declaration, never a map-consumer guess.
                                      (.put ^java.util.IdentityHashMap binding-contexts result observations)
                                      result))
                                  (or (seq? node) (vector? node))
                                  (util/remake-if-changed node (mapv visit node))
                                  :else node))]
                        (visit scope))
              result rebuilt]
          ;; The actual count/allocated-array/map-result locals all have stable
          ;; builtin declarations. Contexts match the actual new initializer
          ;; site; no completion is dropped from an arbitrary source node.
          (.put ^java.util.IdentityHashMap binding-contexts result (:bindings spine))
          (when-let [projected? (:projected? destination)]
            (vreset! projected? true))
          result)))))

(defn- checked-helper-body
  "Materialize actual lexical reference checks before hygiene or simplification."
  ([emission receivers binding-contexts] (checked-helper-body emission receivers binding-contexts nil))
  ([emission receivers binding-contexts destination]
  (letfn [(walk [node]
            (cond
              (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
              (par/par-map-pure-form? node)
              (do
                (when-not ((helper-backend 'retained-pmap-expansion) emission node)
                  (decline! "Generated pmap import lacks its original shared expansion"
                            {:operation (first node)}))
                (or (import-owned-pmap-spine emission node destination walk binding-contexts)
                    (util/remake-if-changed node (mapv walk node))))
              (and (seq? node) (= 'let* (first node)))
              (let [[head bindings & tail] node
                    previous (volatile! nil)
                    observations (volatile! [])
                    pairs
                    (mapcat
                     (fn [index [id value]]
                       (let [observed ((helper-backend 'retained-binding-context) emission node index)]
                         (when-not observed
                           (throw (ex-info "Generated import has no owned lexical binding context"
                                           {:reason :generated-body-binding-context :binder id})))
                         (when (and (zero? index)
                                    (= 1 (count (:body emission)))
                                    (identical? node (first (:body emission)))
                                    (not= (:before observed)
                                          (:locals ((helper-backend 'retained-body-context) emission))))
                           (decline! "Generated helper binding chain differs from its actual entry" {:binder id}))
                         (when (and @previous (not= @previous (:before observed)))
                           (throw (ex-info "Generated import has a discontinuous lexical binding chain"
                                           {:reason :generated-body-binding-context :binder id})))
                         (vreset! previous (assoc (:before observed) (:binder observed) (:after observed)))
                         (vswap! observations conj observed)
                         ;; The helper's initializer context is independent of
                         ;; the enclosing return completion. Anchor it before
                         ;; constructing a new outer program: moving the let
                         ;; alone must not switch Object-return call selection
                         ;; or consume a one-shot result flag differently.
                         (let [raw (get-in observed [:after :type])
                               context (:input-context observed)]
                           (when-not (and (contains? #{:ref :double :float :long :int :bool} raw)
                                          (completion/bytecode-operand-context? context))
                             (decline! "Generated helper binding has no retained initializer context"
                                       {:binder id}))
                           (let [anchored (completion/complete-value
                                           (completion/bytecode-result-context-contract context raw)
                                           (walk value))]
                             [(if-let [hint (get-in observed [:after :hint])]
                                (vary-meta id assoc :tag hint)
                                id)
                              (if (= :ref raw)
                                   (reference-check (get-in observed [:after :reference-class]) anchored)
                                   anchored)]))))
                     (range) (partition 2 bindings))]
                (let [bindings (vec pairs)
                      terminal (when (< 1 (count tail))
                                 ((helper-backend 'retained-terminal-context) emission node))
                      _ (when (and (< 1 (count tail))
                                   (not (and terminal
                                             (= (count tail) (count (:body terminal)))
                                             (not (:unretained-sibling-calls? terminal))
                                             (completion/bytecode-operand-context? (:context terminal)))))
                          (decline! "Generated helper statements have no complete owned terminal context" {}))
                      statements
                      (when terminal
                        (mapcat
                         (fn [statement]
                           (let [id (with-meta (gensym "generated_effect__")
                                      {:raster.effect/effectful true})
                                 value (completion/complete-value
                                        (completion/bytecode-result-context-contract
                                         (:context terminal) :ref)
                                        ;; The shared do/emit-body path discards
                                        ;; this statement in its ORIGINAL void
                                        ;; context, then supplies nil as the new
                                        ;; SSA value. No discarded value boxes.
                                        (list 'do (walk statement) nil))]
                             (vswap! observations conj
                                     {:after {:type :ref :hint nil :verified false
                                              :reference-class Object}})
                             [id value]))
                         (butlast tail)))
                      constructed (with-meta
                                    (list* head (into bindings statements)
                                           (if terminal [(walk (last tail))] (mapv walk tail)))
                                    (meta node))]
                  (.put ^java.util.IdentityHashMap binding-contexts constructed @observations)
                  constructed))
              (and (seq? node) (= '.invk (first node)))
              (let [observed ((helper-backend 'retained-invocation-context) emission node)
                    receiver (second node)]
                (when-not (and (= :global-typed-indy (:invocation-mode observed))
                               (qualified-symbol? receiver) (identical? receiver (:receiver observed)))
                  (decline! "Generated helper contains an unobserved or local typed receiver" {}))
                ;; Global typed indy uses literal namespace/name bootstrap
                ;; arguments, not source-ns alias resolution. Protect this exact
                ;; owned receiver from the source qualifier, then restore it.
                (let [placeholder (gensym "owned_receiver__")]
                  (vswap! receivers assoc placeholder receiver)
                  (with-meta (list* '.invk placeholder (mapv walk (nnext node))) (meta node))))
              (or (seq? node) (vector? node))
              (util/remake-if-changed node (mapv walk node))
              (coll? node) (decline! "Generated helper contains an unsupported executable collection" {})
              :else node))]
    (mapv walk (:body emission)))))

(defn- qualified-helper-body
  ([emission outer-ns] (qualified-helper-body emission outer-ns checked-helper-body))
  ([emission outer-ns construct-body]
  (let [receivers (volatile! {})
        binding-contexts (java.util.IdentityHashMap.)
        binding-facts (volatile! {})
        body (construct-body emission receivers binding-contexts)
        parameters (into (set (:params emission)) (keys @receivers))
        used (volatile! (into parameters (filter symbol?) (helper-source-nodes body)))
        allocated (volatile! #{}) allocator util/*fresh-binder*
        hygienic (binding [util/*fresh-binder*
                           (fn [original]
                             (let [fresh (allocator original)]
                               (when-not (and (symbol? fresh) (nil? (namespace fresh))
                                              (not (contains? @used fresh))
                                              (identical? (meta original) (meta fresh)))
                                 (decline! "Generated helper requires fresh metadata-preserving binders" {}))
                               (vswap! used conj fresh)
                               (vswap! allocated conj fresh)
                               fresh))]
                   (mapv #(util/alpha-convert
                           {} %
                           (fn [original constructed _]
                             (when-let [observations (.get binding-contexts original)]
                               (let [ids (vec (take-nth 2 (second constructed)))]
                                 (when-not (= (count ids) (count observations))
                                   (decline! "Generated helper hygiene changed a binding interval" {}))
                                 (doseq [[id observed] (map vector ids observations)]
                                   (when (contains? @binding-facts id)
                                     (decline! "Generated helper hygiene reused a binding" {:binder id}))
                                   (vswap! binding-facts assoc id
                                           (select-keys (:after observed) local-fact-keys))))))) body))
        context {:source-ns (:source-ns emission) :pending-source? false
                 :dependency-snapshots (:dependency-snapshots emission)}
        authenticate!
        (fn [symbol candidate]
          (let [canonical (method-entry/observed-body-operation context symbol #{})
                original (when canonical (ns-resolve (:source-ns emission) canonical))
                target (when canonical (ns-resolve outer-ns canonical))]
            (when-not (and (var? candidate) canonical
                           (identical? candidate original) (identical? candidate target))
              (decline! "Generated helper lookup has no namespace-independent historical meaning"
                        {:symbol symbol}))
            (method-entry/observe-source-resolution!
             {:kind :symbol-resolution :source-ns outer-ns :symbol canonical :result target})
            canonical))
        qualified
        (mapv #(inference/qualify-body-symbols
                % (:source-ns emission) parameters
                (fn [{:keys [kind symbol result]}]
                  (if (and (= :symbol-resolution kind) (var? result))
                    (authenticate! symbol result)
                    (decline! "Generated helper has an unsupported class, alias or unresolved lookup"
                              {:symbol symbol :kind kind})))) hygienic)
        local-names (into parameters @allocated)]
    ;; The shared qualifier intentionally leaves casts alone, because a symbol
    ;; such as `float` can also be positional IR data (pmap's element dtype).
    ;; Only genuine call heads are completed here; quoted data stays opaque.
    (letfn [(walk [node]
              (cond
                (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
                (seq? node)
                (let [head (first node)
                      head (if (and (symbol? head) (nil? (namespace head))
                                    (descriptor/cast-op? head) (not (contains? local-names head)))
                             (let [canonical (method-entry/observed-body-operation context head local-names)]
                               (when-not canonical
                                 (decline! "Generated helper cast lacks its original lookup" {:symbol head}))
                               (with-meta (authenticate! head (ns-resolve (:source-ns emission) canonical)) (meta head)))
                             head)]
                  (util/remake-if-changed node (into [head] (map walk (rest node)))))
                (vector? node) (util/remake-if-changed node (mapv walk node))
                :else node))]
      {:body (mapv #(util/subst-syms @receivers (walk %)) qualified)
       :binding-facts @binding-facts}))))

(defn- expose-anchored-result-spine
  "Distribute result completions through a complete owned sequential binding
   spine. Each moved initializer independently replays its original context;
   conversion of the helper result stays after all bindings, on the tail.
   This constructor is private to authenticated helper import."
  [expression binding-facts]
  (cond
    (completion/canonical-form? expression)
    (let [[bindings tail] (expose-anchored-result-spine (nth expression 2) binding-facts)]
      [bindings (util/remake-from expression [(first expression) (second expression) tail])])

    (and (seq? expression) (= 'let* (first expression)))
    (let [[_ bindings & body] expression
          tail (first body)
          pairs (partition 2 bindings)]
      (when-not (and (= 1 (count body)) (vector? bindings) (even? (count bindings))
                     (every? #(contains? binding-facts (first %)) pairs))
        (decline! "Generated helper result has no complete anchored binding spine"
                  {:body-count (count (nnext expression))
                   :unowned-binders (vec (remove #(contains? binding-facts %)
                                                 (map first pairs)))}))
      ;; Do not descend into an initializer, branch, loop, call or collection.
      ;; Nested helpers/control retain their own existing execution position.
      (let [[later result] (expose-anchored-result-spine tail binding-facts)]
        [(into (vec bindings) later) result]))

    :else [[] expression]))

(defn- optional-terminal-return [frontend]
  (try
    ((requiring-resolve
      'raster.compiler.passes.scalar.source-import/terminal-reduction-return-contract)
     {:frontend frontend :input-body (:body frontend) :root (first (:body frontend))})
    (catch clojure.lang.ExceptionInfo error
      (if (contains? #{:terminal-reduction-import-root :terminal-reduction-import-return}
                     (:reason (ex-data error)))
        nil
        (throw error)))))

(defn- consume-terminal-return
  [consume! {:keys [return-contract argument-bindings binding-facts] :as request}]
  (let [validate! (requiring-resolve
                   'raster.compiler.passes.scalar.source-import/validate-terminal-reduction-return!)]
    (validate! return-contract)
    (when-let [constructed (consume! request)]
      (validate! return-contract)
      (let [expression (when (map? constructed) (:expression constructed))
            facts (when (map? constructed) (:binding-facts constructed))
            [head bindings & body] (when (seq? expression) expression)]
        (when-not (and (map? constructed)
                       (= #{:expression :binding-facts} (set (keys constructed)))
                       (seq? expression) (= 'let* head) (vector? bindings)
                       (even? (count bindings)) (= 1 (count body))
                       (<= (count argument-bindings) (count bindings))
                       (every? true? (map identical? argument-bindings bindings))
                       (map? facts)
                       (every? (fn [[id original]] (= original (get facts id))) binding-facts))
          (decline! "Terminal consumer changed argument completion or existing local obligations" {}))
        constructed))))


(defn ^:no-doc validate-helper-application!
  "Validate the complete owner-local application, not a detached projection.
   This construction grants no numerical projection or execution permission."
  [application]
  (validate-owned-import! application :application :helper-application-correspondence))

(defn- construct-invocation-input-frame
  "Construct positional completions after their actual caller has been checked.
   This is shared hygiene/assembly, not a proof of an operand or selected target."
  ([inputs parameters values body]
   (construct-invocation-input-frame inputs parameters values body (mapv :operand inputs)))
  ([inputs parameters values body source-operands]
  (letfn [(symbols [node]
            (cond
              (symbol? node) #{node}
              (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) #{}
              (map? node) (reduce into #{} (map symbols (mapcat identity node)))
              (coll? node) (reduce into #{} (map symbols node))
              :else #{}))]
    (let [reserved (into (into (into (set parameters) (symbols body)) (mapcat symbols source-operands))
                         (mapcat #(concat (symbols (:operand %)) (keys (:ports %))) inputs))]
      (when-not (and (= (count inputs) (count source-operands) (count parameters) (count values) (count (set values)))
                     (every? #(and (symbol? %) (nil? (namespace %))
                                   (not (contains? reserved %))) values))
        (fail! :nonfresh-source-import-binder
               "helper input slots must not capture caller operands or imported bindings" {}))))
  {:bindings (mapv (fn [value input source-operand]
                     [value (vary-meta (completion/complete-value (:contract input) source-operand)
                                       assoc :raster.effect/preserve-evaluation true)])
                   values inputs source-operands)
   :slots (mapv (fn [ordinal formal value input source-operand]
                  {:ordinal ordinal :formal formal :value value
                   :operand (:operand input) :source-operand source-operand :contract (:contract input)
                   :input input :completion (:completion input)
                   :ports (:ports input) :context (:context input)})
                (range) parameters values inputs source-operands)}))

(defn- construct-helper-application
  [frontend invocation inputs parameters values body binding-facts return-contracts
   return-tag expose-spine?]
  (let [frame (construct-invocation-input-frame inputs parameters values body)
        argument-bindings (:bindings frame)
        instantiated (util/subst-syms (zipmap parameters values) body)
        returned (reduce (fn [value contract] (completion/complete-value contract value))
                         instantiated return-contracts)
        [body-bindings result] (if expose-spine?
                                 (expose-anchored-result-spine returned binding-facts)
                                 [[] returned])
        result (if return-tag (vary-meta result assoc :raster.type/tag return-tag) result)
        application {:caller-occurrence (:expression invocation)
                     :selected-method (:method invocation) :frontend frontend
                     :input-slots (:slots frame)
                     :payload-ports values :numerical-projection nil
                     :return-recipe return-contracts
                     :dependencies @method-entry/*emission-inputs*}]
    {:expression (list 'let* (into (vec (mapcat identity argument-bindings)) body-bindings) result)
     :binding-facts binding-facts :argument-bindings argument-bindings
     :result-form result :payload-expression instantiated :application application}))

(defn- construct-emitted-helper
  [expression locals context capture! consume-terminal-reduction terminal-only?]
  (let [[_ receiver & arguments] expression
        invocation ((helper-backend 'typed-invocation-plan) receiver (count arguments) locals context)]
    (when (and (:global? invocation)
               (not (:lexical-callee (descriptor/call-description expression))))
      (when-let [inputs ((helper-backend 'plan-resolved-invocation-inputs)
                        (:method invocation) (vec arguments) locals context)]
        (let [{:keys [semantic-op implementation-op]} (descriptor/call-description expression)
              owner (capture! semantic-op implementation-op (count arguments) (:method invocation))
              signature (call-boundary/selected-typed-signature
                         semantic-op implementation-op (count arguments) (:method invocation))
              emission (:emission owner)
              source (get-in owner [:signature :definition :source-var])
              frontend (when source (method-entry/original-source-frontend source))]
          (when (and (= :known (:status signature))
                     (ifn? (:guard owner))
                     (= (:method invocation) (:method owner) (get-in owner [:signature :method]))
                     (every? #(identical? (get-in signature [:definition %])
                                         (get-in owner [:signature :definition %]))
                             [:source-var :receiver-var :entry])
                     emission frontend (identical? emission (:emission frontend))
                     (identical? (:implementation emission)
                                 (get-in signature [:definition :entry :typed-impl])))
            ((:guard owner))
            (method-entry/assert-frontend-selection! frontend source)
            (method-entry/assert-emission-inputs-current! emission)
            (let [bridge (when-not terminal-only?
                           (or (method-entry/frontend-primitive-body-return-bridge frontend)
                               (method-entry/frontend-reference-return-bridge frontend)))
                  ;; Authenticate this third, weaker return contract BEFORE
                  ;; hygiene. It is useful only to an actual consuming owner;
                  ;; ordinary JVM/C expansion keeps its existing abstention.
                  terminal-return (when (and (nil? bridge) consume-terminal-reduction)
                                    (optional-terminal-return frontend))
                  saved (or (:body-context bridge)
                            (when terminal-return
                              ((helper-backend 'retained-input-body-context) emission (:input-body terminal-return))))]
              (when (and (or bridge terminal-return)
                         saved (not (:unretained-sibling-calls? saved)))
                (let [{:keys [body binding-facts]} (qualified-helper-body emission (:source-ns context))
                      body (if (= 1 (count body)) (first body) (list* 'do body))
                      ;; Retain the selected formal's existing logical
                      ;; declaration. This is not a raw-carrier inference: the
                      ;; input recipe and final emitter still check conversion.
                      values (mapv (fn [tag]
                                     (with-meta (gensym "generated_input__") {:raster.type/tag tag}))
                                   (:tags emission))
                      bindings (vec (mapcat (fn [value input]
                                             [value (completion/complete-value
                                                     (:contract input) (:operand input))]) values inputs))
                      instantiated (util/subst-syms (zipmap (:params emission) values) body)]
                  ;; All operands, including unused ones, are completed once in
                  ;; positional order before the exact helper body. There is no
                  ;; synthesized receiver term for the global args-first path.
                  (if bridge
                    (let [contracts [(completion/bytecode-result-context-contract
                                      (:result-context saved) (:emitted-result-type saved))]
                          contracts (cond-> contracts
                                      (= :reference-static-return (:kind bridge))
                                      (into (when-not (or (nil? (:body-return-carrier bridge))
                                                          (= Object (:body-return-carrier bridge)))
                                              [(completion/reference-check-contract (:body-return-carrier bridge))]))
                                      (not= :reference-static-return (:kind bridge))
                                      (conj completion/object-input-contract))]
                      (retain-owned-import!
                       (construct-helper-application frontend (assoc invocation :expression expression)
                                                     inputs (:params emission) values body binding-facts
                                                     contracts (:return-tag emission) true)
                       ;; Capture's guard rechecks its ORIGINAL selection and
                       ;; readiness history directly. Reentering a collecting
                       ;; frontend-selection query here would validate the
                       ;; enclosing owner containing this very receipt.
                       frontend (:guard owner)))
                    (consume-terminal-return
                     consume-terminal-reduction
                     {:return-contract terminal-return :frontend frontend :invocation invocation
                      :inputs inputs :expression expression :locals locals :context context
                      :argument-bindings bindings
                      :input-values values :instantiated-body instantiated
                      :saved-return-context saved :binding-facts binding-facts})))))))))))


(defn- effective-inline-body [body]
  (if (> (count body) 1) [(apply list 'do body)] body))

(defn- hygienic-inline-body [body parameters]
  (let [used (volatile! (into (set parameters) (mapcat util/free-syms) body))
        allocate util/*fresh-binder*]
    (binding [util/*fresh-binder*
              (fn [original]
                (let [fresh (allocate original)]
                  (when-not (and (symbol? fresh) (nil? (namespace fresh))
                                 (not (contains? @used fresh))
                                 (identical? (meta original) (meta fresh)))
                    (throw (ex-info "source transformation requires fresh metadata-preserving binders"
                                    {:reason :nonfresh-source-import-binder})))
                  (vswap! used conj fresh)
                  fresh))]
      (mapv util/alpha-convert body))))

(defn ^:no-doc prepare-original-helper-source [frontend body parameters tags source-ns]
  (let [construct
        (fn []
          (when frontend
            (method-entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
            (when-not (and (identical? body (:body frontend))
                           (= parameters (get-in frontend [:source-entry :params]))
                           (= tags (get-in frontend [:source-entry :tags]))
                           (identical? source-ns (get-in frontend [:source-entry :source-ns])))
              (throw (ex-info "helper transformation differs from its original input"
                              {:reason :source-import-input-mismatch})))
            (method-entry/observe-frontend-source-identity! frontend))
          (let [qualified-tags
                (when tags
                  (mapv (fn [tag]
                          (if (or (nil? source-ns) (nil? tag)
                                  (contains? '#{double long float int Double Long Float Integer
                                                doubles floats longs ints objects} tag)
                                  (.contains (str tag) "."))
                            tag
                            (let [lookup (symbol (str tag))
                                  result (try (ns-resolve source-ns lookup) (catch Exception _ nil))]
                              (method-entry/observe-source-resolution!
                               {:kind :symbol-resolution :source-ns source-ns :symbol lookup :result result})
                              (if (class? result) (symbol (.getName ^Class result)) tag)))) tags))
                hygienic (hygienic-inline-body (effective-inline-body body) parameters)
                parameter-set (set (map #(if (symbol? %) % (symbol (name %))) parameters))
                qualified (if source-ns
                            (mapv #(inference/qualify-body-symbols % source-ns parameter-set
                                                           method-entry/observe-source-reference!) hygienic)
                            hygienic)]
            (when frontend
              (method-entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var])))
            {:walked-body qualified :tags qualified-tags}))]
    (construct)))


(defn- construct-static-helper
  "Derive a whole-call application from the exact current candidate and occurrence.
   Rebuilds physical plans and imports the original callee under historical inputs;
   caller-provided bodies, plans and detached application maps are not authority."
  [candidate occurrence source-options]
  (when method-entry/*emission-inputs*
    (when-let [scope (call-plan/source-context source-options)]
      (when-let [planned (.get ^java.util.IdentityHashMap (call-plan/call-plans scope candidate) occurrence)]
        (let [source (get-in planned [:physical :definition :source-var])
              frontend (when source (method-entry/original-source-frontend source))]
          (when frontend
            (method-entry/assert-frontend-source! frontend source)
            (let [lineage (:source-entry frontend)
                  parameters (:params lineage)
                  {:keys [walked-body]} (prepare-original-helper-source
                                        frontend (:body frontend) parameters (:tags lineage) (:source-ns lineage))
                  body (first (effective-inline-body walked-body))
                  arguments (:arguments planned)
                  info {:source-var source :original-frontend frontend :params parameters}]
              (when-let [whole (call-plan/whole-call scope planned info arguments body)]
                (let [values (mapv #(with-meta (gensym (str "completed_" (name %) "_"))
                                     {:raster.effect/effectful true}) parameters)
                      context (:body-context whole)
                      contracts [(completion/bytecode-result-context-contract context (:body-raw whole))
                                 (if (= Object (:return-class whole))
                                   completion/object-input-contract
                                   (completion/planned-bytecode-input-contract
                                    (:return-class whole) context (:body-raw whole)))
                                 completion/object-input-contract]
                      artifact (construct-helper-application
                                frontend (assoc (:plan planned) :expression occurrence)
                                (:inputs whole) parameters values body {} contracts nil false)
                      ;; Keep every operand evaluation, including unused inputs.
                      artifact (update artifact :argument-bindings
                                       #(mapv (fn [[id value]]
                                                [id (vary-meta value assoc :raster.effect/preserve-evaluation true)]) %))
                      artifact (assoc artifact :expression
                                      (list 'let* (vec (mapcat identity (:argument-bindings artifact)))
                                            (:result-form artifact)))]
                  (retain-owned-artifact!
                   artifact
                   #(method-entry/assert-frontend-source! frontend source)))))))))))

(defn ^:no-doc instantiate-emitted-helper
  "Derive one emitted application under a fresh, fully fenced dependency closure.
   The capture callback is independently crosschecked against the actual call."
  [expression locals context capture! consume-terminal-reduction terminal-only?]
  (when (and method-entry/*emission-inputs* (seq? expression) (= '.invk (first expression)))
    (method-entry/call-with-independent-definition-inputs
     #(construct-emitted-helper expression locals context capture!
                                consume-terminal-reduction terminal-only?))))

(defn ^:no-doc instantiate-static-helper
  "Derive one static application under a fresh, fully fenced dependency closure.
   No plan or imported body from before the entry fence is accepted as evidence."
  [candidate occurrence source-options]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     #(construct-static-helper candidate occurrence source-options))))

(defn- construct-reference-application
  "Shared construction AFTER the caller route authenticates its exact inputs,
   selected frontend and reference bridge. This private function grants neither
   occurrence correspondence nor permission to freeze a later source call."
  [frontend emission bridge invocation inputs caller-context source-ns context-for]
  (let [saved (:body-context bridge)
        region-context (if (:object-call-result? caller-context)
                         (context-for caller-context :object-region) caller-context)
        projected? (volatile! false)
        destination {:context (completion/bytecode-operand-context
                               (context-for region-context :initializer))
                     :return-class (:body-return-carrier bridge)
                     :projected? projected?}
        {:keys [body binding-facts]}
        (qualified-helper-body
         emission source-ns
         (fn [module receivers contexts]
           (checked-helper-body module receivers contexts destination)))
        body (if (= 1 (count body)) (first body) (list* 'do body))
        ;; Detach only the newly constructed numerical source, never original
        ;; module descendants or the caller's positional operand occurrences.
        body (if @projected?
               ((helper-backend 'detach-emission-occurrences) body)
               body)
        values (mapv (fn [tag]
                       (with-meta (gensym "reference_input__") {:raster.type/tag tag}))
                     (:tags emission))
        contracts (cond-> [(completion/bytecode-result-context-contract
                            (:result-context saved) (:emitted-result-type saved))]
                    (not= Object (:body-return-carrier bridge))
                    (conj (completion/reference-check-contract (:body-return-carrier bridge))))
        artifact (construct-helper-application
                  frontend invocation inputs (:params emission) values body binding-facts
                  contracts (:return-tag emission) true)]
    (retain-owned-import! (assoc artifact :reference-return bridge) frontend)))

(defn ^:no-doc instantiate-ready-reference-helper
  "Construct an ordinary application using an ALREADY owned reference-return
   emission. This is the emitted alternative to the cold scalar body planner,
   not a request to prepare a helper. Caller inputs are rederived from the exact
   candidate; the complete callee body, lexical checks and return CHECKCAST come
   from its original module. No native representation permission follows."
  [candidate occurrence source-options]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     (fn []
       (when-let [scope (call-plan/source-context source-options)]
         (when-let [planned (.get ^java.util.IdentityHashMap
                                 (call-plan/call-plans scope candidate) occurrence)]
           (let [physical (:physical planned)
                 source (get-in physical [:definition :source-var])
                 frontend (when source (method-entry/original-source-frontend source))]
             (when frontend
               (method-entry/assert-frontend-selection! frontend source)
               (when-let [emission (method-entry/observed-frontend-emission frontend)]
                 (when-let [bridge (method-entry/frontend-reference-return-bridge frontend)]
                   (let [method (get-in planned [:plan :method])
                         saved (:body-context bridge)]
                     (when (and (= method (:invocation-method bridge))
                                (identical? (:implementation emission)
                                            (get-in physical [:definition :entry :typed-impl]))
                                (= (:policy scope) (get-in frontend [:source-entry :source-policy]))
                                (= (:parameter-carriers physical) (:parameter-carriers bridge))
                                (not (:unretained-sibling-calls? saved)))
                       (construct-reference-application
                        frontend emission bridge (assoc (:plan planned) :expression occurrence)
                        (:inputs planned) (:caller-context planned)
                        (get-in scope [:context :source-ns])
                        (get-in scope [:functions 'source-evaluation-context]))))))))))))))

(defn- assert-helper-occurrence-current!
  ([emission input-body original record]
  ;; This is the module authenticated BEFORE receipt issuance. Its original
  ;; history/physical-selection guards are already retained; never reenter a
  ;; collecting frontend query from the enclosing receipt's reuse guard.
  (method-entry/assert-emission-inputs-current! emission)
  (let [current ((helper-backend 'retained-input-invocation-context)
                 emission input-body original)]
    (when-not (and (identical? original (:source-input current))
                   (every? #(identical? (get record %) (get current %))
                           [:expression :method :receiver :inputs :input-completions]))
      (fail! :helper-application-occurrence
             "helper application lost its exact original emitted occurrence" {}))))
  ([frontend emission input-body original record]
   (method-entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
   (when-not (and (identical? emission (:emission frontend)) (identical? input-body (:body frontend)))
     (fail! :helper-application-occurrence "helper application lost its original caller" {}))
   (assert-helper-occurrence-current! emission input-body original record)))

(defn- owned-occurrence-typed-call [original record]
  ;; The emitter may desugar the actual expression to (. receiver invk ...).
  ;; InputOrigins owns that construction edge; this is a NEW typed-call view,
  ;; not a dot-form recognizer or a claim about an original syntax object.
  (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
        (descriptor/call-description original)
        receiver (:receiver record)
        inputs (:inputs record)]
    (when (and dispatch? (not lexical-callee) (qualified-symbol? semantic-op)
               (qualified-symbol? receiver) (= implementation-op receiver)
               (= (count arguments) (count inputs))
               (= :known (:status (call-boundary/selected-typed-signature
                                   semantic-op receiver (count inputs) (:method record)))))
      (with-meta (list* '.invk receiver (mapv :operand inputs))
        {:raster.op/original semantic-op}))))

(defn- construct-helper-occurrence
  [caller-emission input-body original source-ns policy owner-fields owner-guard]
  (owner-guard)
  (when caller-emission
      (when-let [record ((helper-backend 'retained-input-invocation-context)
                        caller-emission input-body original)]
        (when (and (= :known (:status record))
                   (= :global-typed-indy (:invocation-mode record))
                   (identical? Object (:return-carrier record))
                   (identical? original (:source-input record))
                   ((helper-backend 'completed-invocation-inputs?) record)
                   (not-any? :unretained-sibling-calls? (:inputs record)))
          (let [^java.lang.reflect.Method method (:method record)
                carriers (vec (.getParameterTypes method))
                ;; These facts describe slots AFTER the retained positional
                ;; conversions below, never the original unevaluated operands.
                facts (mapv #(hash-map :carrier % :presence :present) carriers)
                typed-call (owned-occurrence-typed-call original record)
                recipe (when typed-call
                         ((requiring-resolve
                           'raster.compiler.passes.parallel.scalar-expression-body/owned-selected-call-fp-payload)
                          typed-call method facts))
                source (get-in recipe [:definition :source-var])
                frontend (when source (method-entry/original-source-frontend source))
                emission (:emission recipe)
                bridge (:bridge recipe) saved (:body-context bridge)]
            (when (and frontend emission (identical? emission (:emission frontend))
                       (= method (:invocation-method bridge)) saved
                       (not (:unretained-sibling-calls? saved))
                       (= policy
                          (get-in frontend [:source-entry :source-policy])))
              (let [{:keys [body binding-facts]}
                    (qualified-helper-body emission source-ns)
                    body (if (= 1 (count body)) (first body) (list* 'do body))
                    inputs (mapv (fn [input completed carrier]
                                   (assoc input :completion completed
                                          :contract (completion/planned-bytecode-input-contract
                                                     carrier (:context input) (:emitted-type completed))))
                                 (:inputs record) (:input-completions record) carriers)
                    values (mapv #(with-meta (gensym "owned_call_input__")
                                    {:raster.type/tag % :raster.effect/effectful true}) (:tags emission))
                    contracts [(completion/bytecode-result-context-contract
                                (:result-context saved) (:emitted-result-type saved))
                               completion/object-input-contract]
                    artifact (construct-helper-application
                              frontend record inputs (:params emission) values body binding-facts
                              contracts (:return-tag emission) true)
                    ;; Only COMPLETED slot IDs are numerical argument ports.
                    ;; The physical/source lexical relation and use-closed box
                    ;; erasure remain separate obligations of a future consumer.
                    projection (assoc recipe :arguments values
                                      :formal-input-values (zipmap (:parameters recipe) values))
                    artifact (update artifact :application merge
                                     owner-fields {:numerical-projection projection
                                     :caller-input-body input-body :original-call original
                                     :physical-call record :derived-call typed-call
                                     :input-ports (mapv :ports (:inputs record))})]
                (retain-owned-import!
                 artifact frontend
                 #(do (owner-guard)
                      (assert-helper-occurrence-current! caller-emission input-body original record))))))))))

(defn- construct-owned-helper-occurrence [caller input-body original]
  (when-let [emission (method-entry/observed-frontend-emission caller)]
    (when (identical? input-body (:body caller))
      (construct-helper-occurrence
       emission input-body original (get-in caller [:source-entry :source-ns])
       (get-in caller [:source-entry :source-policy]) {:caller-frontend caller}
       #(do (method-entry/assert-frontend-source! caller (get-in caller [:source-entry :source-var]))
            (method-entry/assert-emission-inputs-current! emission))))))



(defn ^:no-doc instantiate-owned-helper-occurrence
  "Construct an emitted fragment from one exact original caller occurrence.
   Keeps actual positional operands, contexts, completion records and physical
   ports. It grants no substitution into the original source lexical scope,
   box erasure, motion or GPU admission. No helper is prepared by this query."
  [caller-frontend exact-input-body original-call]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     #(construct-owned-helper-occurrence caller-frontend exact-input-body original-call))))

(defn ^:no-doc instantiate-contextual-helper-occurrence
  "Derive one current-source application from its existing explicit emission owner.
   This query validates that owner; it never creates a witness or prepares code."
  [witness original-call]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     #(do (validate-contextual-source-witness! witness)
          (construct-helper-occurrence
           (:emission witness) (:input-body witness) original-call
           (:source-ns witness) (:source-policy witness) {:caller-witness witness}
           (fn [] (validate-contextual-source-witness! witness)))))))

(defn- closed-prefix-local
  "A leaf load is closed only at the exact physical local version. Conditional
   normal-return facts are not promoted to presence or an earlier definition."
  [before facts operand]
  (when (and (symbol? operand) (contains? before operand))
    (let [local (get before operand)
          retained (get facts operand)]
      (cond-> {:local local}
        (= local (:local retained)) (merge (:result retained))
        (contains? #{Float Double} (:reference-class local))
        (assoc :builtin-class (:reference-class local))))))

(defn- construct-prefix-reference-application
  "Consume one exact witnessed call AFTER its ordered prefix has been closed.
   Only direct lexical operands are admitted here; actual conversions remain in
   the shared application. This does not close the callee for a later call."
  [witness original binding facts]
  (let [{:keys [emission input-body source-ns source-policy]} witness
        record ((helper-backend 'retained-input-invocation-context) emission input-body original)
        typed (when record (owned-occurrence-typed-call original record))
        description (when typed (descriptor/call-description typed))
        signature (when typed
                    (call-boundary/selected-typed-signature
                     (:semantic-op description) (:implementation-op description)
                     (count (:inputs record)) (:method record)))
        original-arguments (vec (descriptor/call-args original))
        before (:before binding)
        direct-inputs?
        (and (= (count original-arguments) (count (:inputs record))
                (count (:input-completions record)))
             (every? true?
                     (map (fn [argument input completed]
                            (let [local (closed-prefix-local before facts argument)]
                              (and local (= argument (:operand input) (:operand completed))
                                   (= {argument (:local local)} (:ports input))
                                   (false? (:unretained-sibling-calls? input))
                                   (or (not= :ref (:emitted-type completed))
                                       (not (.isPrimitive ^Class (:parameter-carrier completed)))
                                       (contains? #{Float Double} (:builtin-class local))))))
                          original-arguments (:inputs record) (:input-completions record))))
        source (get-in signature [:definition :source-var])
        frontend (when (and direct-inputs? (= :known (:status signature)) source)
                   (method-entry/original-source-frontend source))]
    (when (and frontend (= :known (:status record))
               (identical? (:initializer binding) (:expression record))
               (= :global-typed-indy (:invocation-mode record))
               (identical? Object (:return-carrier record))
               ((helper-backend 'completed-invocation-inputs?) record))
      (method-entry/assert-frontend-selection! frontend source)
      (when-let [callee-emission (method-entry/observed-frontend-emission frontend)]
        (when-let [bridge (method-entry/frontend-reference-return-bridge frontend)]
          (when (and (= (:method record) (:invocation-method bridge))
                     (= (:parameter-classes signature) (:parameter-carriers bridge))
                     (identical? (:implementation callee-emission)
                                 (get-in signature [:definition :entry :typed-impl]))
                     (= source-policy (get-in frontend [:source-entry :source-policy]))
                     (not (get-in bridge [:body-context :unretained-sibling-calls?])))
            (let [inputs (mapv (fn [input completed]
                                 (assoc input :completion completed
                                        :contract (completion/planned-bytecode-input-contract
                                                   (:parameter-carrier completed) (:context input)
                                                   (:emitted-type completed))))
                               (:inputs record) (:input-completions record))
                  application (construct-reference-application
                               frontend callee-emission bridge (assoc record :expression original)
                               inputs (:input-context binding) source-ns
                               (helper-backend 'source-evaluation-context))]
              (retain-owned-artifact!
               (assoc (dissoc application :receipt) :caller-witness witness :physical-call record)
               #(do (validate-helper-application! application)
                    (validate-contextual-source-witness! witness)
                    (assert-helper-occurrence-current! emission input-body original record))))))))))

(defn- owned-effect-record
  "Join an actual operand edge OR an original catalog node to an existing
   operation observation. A copied token alone cannot establish either edge."
  [emission node observer]
  (let [token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta node))
        original (get (:invocation-catalog emission) token)
        record (when original ((helper-backend observer) emission original))]
    (when (and record (or (identical? node original) (identical? node (:expression record))))
      (assoc record :source original))))

(defn- array-origin-at-local [result local]
  (if (and (:array-origin result)
           (not (and (= :ref (:type local)) (class? (:reference-class local))
                     (.isAssignableFrom ^Class (:reference-class local) ^Class (:array-class result)))))
    (dissoc result :array-origin :array-class)
    result))

(defn- closed-source-effect-projector
  "A private, non-executable effect projection of actual owned producers.
   Unknown operations stop coverage. No arbitrary AST operation is classified
   by name, and this grants no motion, bounds, totality or concurrent-code claim."
  ([memo] (closed-source-effect-projector memo {}))
  ([memo {:keys [required?]}]
  (let [primitive? #(contains? #{:int :long :float :double :bool} %)
        builtin? #(boolean (some (fn [known] (identical? known (:builtin-class %)))
                                (vals types/primitive-info)))
        reference? (fn [carrier]
                     (and (class? carrier)
                          (or (identical? Object carrier)
                              (and (.isArray ^Class carrier)
                                   (.isPrimitive (.getComponentType ^Class carrier))))))
        ports? (fn [locals ports] (every? (fn [[id local]] (= local (get locals id))) ports))]
    (letfn [(local-result [locals values id]
              (when-let [local (get locals id)]
                (let [old (get values id)]
                  (cond-> {:stack (:type local) :actual id}
                    (= local (:local old)) (merge (dissoc (:result old) :actual))
                    (and (= :ref (:type local))
                         (some #(identical? % (:reference-class local)) (vals types/primitive-info)))
                    (assoc :builtin-class (:reference-class local))))))
            (close-input [emission locals values input raw]
              (when (and (false? (:unretained-sibling-calls? input)) (ports? locals (:ports input)))
                (when-let [result (close-value emission locals values (:operand input))]
                  (when (= raw (:stack result)) result))))
            (close-call [emission locals values record]
              (when (and (= :global-typed-indy (:invocation-mode record))
                         (identical? Object (:return-carrier record))
                         ((helper-backend 'completed-invocation-inputs?) record))
                (let [results (mapv #(close-input emission locals values %1 (:emitted-type %2))
                                    (:inputs record) (:input-completions record))
                      inputs-closed?
                      (and (= (count results) (count (:inputs record)) (count (:input-completions record)))
                           (every? some? results)
                           (every? true?
                                   (map (fn [result completed]
                                          (let [carrier (:parameter-carrier completed)]
                                            (and (or (.isPrimitive ^Class carrier) (reference? carrier))
                                                 (or (not (.isPrimitive ^Class carrier))
                                                     (primitive? (:emitted-type completed))
                                                     (builtin? result)))))
                                        results (:input-completions record))))
                      typed (when inputs-closed? (owned-occurrence-typed-call (:source record) record))
                      description (when typed (descriptor/call-description typed))
                      selected (when typed
                                 (call-boundary/selected-typed-signature
                                  (:semantic-op description) (:implementation-op description)
                                  (count results) (:method record)))
                      source (get-in selected [:definition :source-var])
                      callee (when (= :known (:status selected)) (method-entry/original-source-frontend source))
                      _ (when (and required? source (nil? (:emission callee)))
                          (when-let [request (method-entry/selected-source-preparation-request
                                              :physical (:semantic-op description) (:implementation-op description)
                                              (count results) source)]
                            ;; Only the explicit construction route requests
                            ;; work. The existing outer owner discards this
                            ;; discovery and restarts after independent prepare.
                            (method-entry/demand-selected-source-preparation! request)))]
                  (when callee
                    (when-let [result (close-owner callee)]
                      (when (and (= (:method record) (:method result))
                                 (identical? (get-in selected [:definition :entry :typed-impl])
                                             (:implementation result)))
                        ;; A callee formal/slot is not the caller's source view.
                        ;; Array origin needs a separate actual argument join.
                        (assoc (dissoc result :method :implementation :array-origin :array-class)
                               :actual (:expression record))))))))
            (close-scope [emission locals values node]
              (let [[head bindings & body] (when (seq? node) node)
                    token (:raster.compiler.backend.jvm.bytecode/binding-scope-occurrence (meta node))
                    scope (get (:binding-catalog emission) token)]
                (when (and (identical? node scope)
                           (= 'let* head) (vector? bindings) (even? (count bindings)) (= 1 (count body)))
                  (loop [ordinal 0 locals locals values values]
                    (if (= ordinal (quot (count bindings) 2))
                      (when-let [result (close-value emission locals values (first body))]
                        (assoc result :actual node))
                      (let [observation (when scope ((helper-backend 'retained-binding-context) emission scope ordinal))
                            binder (nth bindings (* 2 ordinal))
                            original (nth bindings (inc (* 2 ordinal)))]
                        (when (and observation (= locals (:before observation))
                                   (= binder (:binder observation)))
                          (when-let [result (close-value emission locals values original)]
                            (when (and (identical? (:actual result) (:initializer observation))
                                       (= (:stack result) (get-in observation [:after :type])))
                              (recur (inc ordinal)
                                     (assoc locals binder (:after observation))
                                     (assoc values binder {:local (:after observation)
                                                           :result (array-origin-at-local result (:after observation))})))))))))))
            (close-completion [emission locals values node]
              (let [token (:raster.compiler.backend.jvm.bytecode/completion-occurrence (meta node))
                    source (get (:completion-catalog emission) token)
                    record (when source ((helper-backend 'retained-owned-completion-context) emission source))
                    parts (completion/canonical-parts node)]
                (when (and record (= locals (:locals-before record))
                           (or (identical? node source) (identical? node (:expression record)))
                           (= (:contract parts) (:contract record)))
                  ;; Traverse the actual recorded operand, not a token-bearing
                  ;; source copy. Completion observers do not equate the two.
                  (when-let [input (close-value emission locals values (:operand record))]
                    (let [context (get-in record [:contract :source-context])
                          target (get-in record [:contract :target-carrier])
                          raw (:input-carrier record)
                          adaptation (:adaptation record)
                          closed?
                          (case context
                            :jvm-object-input (= :identity adaptation)
                            (:jvm-checkcast :jvm-bytecode-reference-input)
                            (and (= :checkcast adaptation) (reference? target))
                            :jvm-bytecode-boxed-parameter-input
                            (or (and (= :checkcast adaptation) (reference? target))
                                (and (= :boxed-unbox adaptation) (builtin? input)))
                            (:jvm-bytecode-floating-input :jvm-bytecode-integral-input
                             :source-floating-cast :source-integral-cast)
                            (or (primitive? raw) (and (= :ref raw) (builtin? input)))
                            :jvm-bytecode-result-context (= :identity adaptation)
                            false)]
                      (when (and closed?
                                 (or (= raw (:stack input))
                                     ;; Object completion's recorded raw carrier
                                     ;; is AFTER the actual root boxing producer.
                                     (and (= :jvm-object-input context)
                                          (= (:stack input) (:operand-before-box-carrier record)))))
                        (merge (when (= :ref (:returned-carrier record))
                                 (merge (select-keys input [:builtin-class])
                                        (when (and (= :ref raw (:stack input))
                                                   (contains? #{:identity :checkcast} adaptation))
                                          (select-keys input [:boxed-primitive-carrier]))
                                        (when (and (:array-origin input)
                                                   (or (and (contains? #{:jvm-object-input :jvm-bytecode-result-context} context)
                                                            (= :ref raw) (= :identity adaptation))
                                                       (and (= :checkcast adaptation) (class? target)
                                                            (.isAssignableFrom ^Class target ^Class (:array-class input)))))
                                          (select-keys input [:array-origin :array-class]))))
                               (:result-fact record)
                               {:stack (:returned-carrier record) :actual (:expression record)})))))))
            (close-map-effect [emission locals values expansion]
                (let [{:keys [output index cast body actual-expanded bound-observation]} expansion
                      ordinal-source body
                      ordinal ((helper-backend 'retained-binding-context) emission ordinal-source 0)
                      effect actual-expanded
                      bound-pair (when (= 'let* (first effect)) (second effect))
                      bound-id (first bound-pair)
                      loop-node (nth effect 2 nil)
                      branch (nth loop-node 2 nil)
                      store (second (nth branch 2 nil))
                      actual-cast (nth store 3 nil)
                      actual-ordinal (second actual-cast)
                      index-local (get-in ordinal [:before index])
                      bound-local (get-in ordinal [:before bound-id])
                      context {:source-ns (:source-ns emission) :pending-source? false
                               :dependency-snapshots (:dependency-snapshots emission)}
                      canonical-cast (when (symbol? cast)
                                       (method-entry/observed-body-operation context cast (set (keys (:before ordinal)))))
                      cast-tag (descriptor/cast-result-tag canonical-cast)
                      expected-store (list 'clojure.core/aset output index actual-cast)
                      expected-effect
                      (list 'let* [bound-id (:initializer bound-observation)]
                            (list 'loop* [index (list 'int 0)]
                                  (list 'if (list '< index bound-id)
                                        (list 'do expected-store
                                              (list 'recur (list 'clojure.core/unchecked-inc-int index)))
                                        (:returned expansion))))]
                  ;; This is the exact consumed compiler recipe, not admission
                  ;; of arbitrary loops or stores. All user-controlled work is
                  ;; in count/body/cast and is closed separately below. The
                  ;; scaffold contributes only primitive integer control and
                  ;; primitive-array storage (which may still throw).
                  (when (and (= locals (:before bound-observation))
                             (= effect expected-effect)
                             (= 2 (count bound-pair))
                             (= :int (:type index-local) (:type bound-local))
                             (= locals
                                (dissoc (:before ordinal) index bound-id))
                             (= (:after bound-observation) bound-local)
                             (let [carrier (get-in locals [output :reference-class])]
                               (and (class? carrier) (.isArray ^Class carrier)
                                    (.isPrimitive (.getComponentType ^Class carrier))))
                             (= 2 (count actual-cast)) (= cast (first actual-cast))
                             (= 'let* (first actual-ordinal)) (= 3 (count actual-ordinal))
                             (vector? (second actual-ordinal)) (= 2 (count (second actual-ordinal)))
                             (= (:binder ordinal) (first (second actual-ordinal)))
                             (identical? (:initializer ordinal) (second (second actual-ordinal)))
                             (= (meta actual-ordinal) (meta ordinal-source))
                             (contains? '#{float double long int} cast-tag)
                             (intrinsic-source/observe-original! (:source-ns context) canonical-cast)
                             (close-value emission locals values (:initializer bound-observation)))
                    (let [body-locals (assoc (:before ordinal) (:binder ordinal) (:after ordinal))
                          original-body (close-value emission body-locals values (nth ordinal-source 2))
                          ;; Generic scopes are catalog-identity-only. The
                          ;; rebuilt actual wrapper above has a separate exact
                          ;; parent/binding/body edge from this owned recipe.
                          body-result (close-scope emission (:before ordinal) values ordinal-source)]
                      (when (and original-body body-result (primitive? (:stack body-result))
                                 (identical? (:actual original-body) (nth actual-ordinal 2 nil)))
                        (assoc (select-keys (local-result locals values output) [:array-origin :array-class])
                               :stack :ref :actual effect))))))
            (close-pmap [emission locals values source]
              (when-let [spine ((helper-backend 'retained-pmap-spine) emission source)]
                (let [[count-binding allocation-binding effect-binding] (:bindings spine)]
                  (when (and (= locals (:before count-binding))
                             (close-value emission locals values (:initializer count-binding))
                             (close-value emission (:before allocation-binding) values (:initializer allocation-binding))
                             (identical? (:initializer effect-binding) (get-in spine [:map-effect :actual-expanded]))
                             (close-map-effect emission (:before effect-binding) values (:map-effect spine)))
                    (cond-> {:stack :ref :actual source}
                      (= :int (get-in spine [:allocation :count :input-type]))
                      (assoc :array-class (:array-class spine)
                             :array-origin {:kind :int-count-allocation :emission emission
                                            :allocation (:allocation spine)}))))))
            (close-current-map [emission locals values node]
              (some (fn [[_ record]]
                      (when (= :map-effect (:source-kind record))
                        (when-let [expansion ((helper-backend 'retained-map-effect-expansion)
                                             emission (:source record))]
                          (when (identical? node (:actual-expanded expansion))
                            (close-map-effect emission locals values expansion)))))
                    (:pmap-expansions emission)))
            (close-reduction [emission locals values node]
              (let [token (:raster.compiler.backend.jvm.bytecode/reduction-occurrence (meta node))
                    witness (:raster.compiler.backend.jvm.bytecode/reduction-witness (meta node))
                    source (or (when token (get (:reduction-catalog emission) token))
                               (when witness
                                 (some (fn [[_ expansion]]
                                         (when (identical? witness (:witness expansion)) (:source expansion)))
                                       (:reduction-expansions emission))))
                    record (when source ((helper-backend 'retained-reduction-emission) emission source))
                    {:keys [bound loop recur exit root-result result-fact]} record
                    [index carry] (:variables loop)
                    [index-init carry-init] (:initializers loop)
                    [index-next carry-next] (:values recur)
                    loop-locals (:locals loop)
                    count-id (:binder bound)
                    actual-bound (:initializer bound)
                    control (:control exit)
                    expected-exit (list 'if (:predicate control) (:expression recur) carry)]
                (when (and record root-result result-fact
                           (or (identical? node source) (identical? node (:expression bound)))
                           (= locals (:before bound))
                           (= 2 (count (:variables loop)) (count (:initializers loop)) (count (:values recur)))
                           (= 1 (count (:body loop)))
                           (identical? (:expression exit) (first (:body loop)))
                           (= expected-exit (:expression exit))
                           (= :int-less-than (:kind control))
                           (= [index count-id] (:operands control))
                           (= [:int :int] (:input-types control))
                           (= loop-locals (:locals control))
                           (= :if-icmpge (:false-branch control))
                           (= (list '< index count-id) (:predicate control))
                           (= (list 'int 0) (:initializer index-init))
                           (= :int (:input-type index-init) (:completed-type index-init)
                              (get-in index-init [:after :type]))
                           (= (:outer-locals loop) (:before index-init))
                           (= (assoc (:before index-init) index (:after index-init)) (:before carry-init))
                           (= (assoc (:before carry-init) carry (:after carry-init)) loop-locals)
                           (= (list 'clojure.core/unchecked-inc-int index) (:operand index-next))
                           (= :int (:input-type index-next) (:type index-next))
                           (= loop-locals (:locals index-next) (:locals carry-next))
                           (= 'int (first actual-bound)) (= 2 (count actual-bound))
                           ;; Only the owned expansion supplies these Int
                           ;; control operations. The evaluated bound, seed and
                           ;; entire update remain ordinary recursive obligations.
                           (primitive? (:input-type carry-init)) (primitive? (:completed-type carry-init))
                           (primitive? (:input-type carry-next)) (primitive? (:type carry-next)))
                  (let [count-result (close-value emission locals values (second actual-bound))
                        seed-result (close-value emission (:before carry-init) values (:initializer carry-init))
                        update-result (close-value emission loop-locals values (:operand carry-next))]
                    (when (and count-result (or (primitive? (:stack count-result)) (builtin? count-result))
                               (= (:input-type carry-init) (:stack seed-result))
                               (= (:input-type carry-next) (:stack update-result)))
                      (merge result-fact {:stack (:result-type root-result) :actual (:expression bound)}))))))
            (close-value [emission locals values node]
              (cond
                (symbol? node) (local-result locals values node)
                (types/numeric-literal-carrier node)
                {:stack (types/numeric-literal-carrier node) :actual node}
                (completion/canonical-form? node) (close-completion emission locals values node)
                :else
                (let [arithmetic (owned-effect-record emission node 'retained-arithmetic)
                      conversion (owned-effect-record emission node 'retained-scalar-conversion)
                      read (owned-effect-record emission node 'retained-array-read)
                      length (owned-effect-record emission node 'retained-array-length)
                      allocation (owned-effect-record emission node 'retained-zero-array-allocation)
                      call (owned-effect-record emission node 'retained-invocation-context)]
                  (cond
                    arithmetic
                    (when (and (every? #(primitive? (:emitted-type %)) (:input-completions arithmetic))
                               (every? some? (map #(close-input emission locals values %1 (:emitted-type %2))
                                                  (:inputs arithmetic) (:input-completions arithmetic))))
                      {:stack (:result-carrier arithmetic) :actual (:expression arithmetic)})

                    conversion
                    (when (ports? locals (:ports conversion))
                      (when-let [input (close-value emission locals values (:operand conversion))]
                        (when (and (= (:stack input) (:input-type conversion))
                                   (or (primitive? (:input-type conversion)) (builtin? input)))
                          {:stack (:result-carrier conversion) :actual (:expression conversion)})))

                    read
                    ;; The existing index-route proof closes actual primitive
                    ;; operations; it explicitly leaves unknown calls open.
                    (when (and (symbol? (:array read)) (= (:array-local read) (get locals (:array read)))
                               (= :ref (get-in read [:array-local :type]))
                               (true? (:primitive-index? read)) (ports? locals (get-in read [:input :ports]))
                               (primitive? (:result-type read)))
                      {:stack (:result-type read) :actual (:expression read)})

                    length
                    (when (and (= :jvm-arraylength (:operation length))
                               (symbol? (:array length))
                               (= {(:array length) (get locals (:array length))} (get-in length [:input :ports]))
                               (= :ref (get-in locals [(:array length) :type]) (:input-carrier length)))
                      {:stack (:result-carrier length) :actual (:expression length)})

                    allocation
                    (let [count-input (:count allocation)
                          count-result (close-input emission locals values count-input (:input-type count-input))]
                      (when (and (true? (:fresh? allocation)) (= :jvm-zero (:initialization allocation))
                                 (= :int (:completed-type count-input))
                                 (= :raw-jvm-int (:completion count-input))
                                 count-result
                                 (or (primitive? (:stack count-result))
                                     (and (builtin? count-result)
                                          (identical? (:actual count-result) (:operand count-input)))))
                        ;; A recursively closed boxed count has no user Number
                        ;; callback. That is weaker than portable index/capacity
                        ;; evidence: retain the old Int theorem's original gate.
                        (cond-> {:stack :ref :actual (:expression allocation)}
                          (and (true? (:primitive-count? allocation)) (= :int (:input-type count-input)))
                          (assoc :array-class (:array-class allocation)
                                 :array-origin {:kind :int-count-allocation :emission emission
                                                :allocation allocation}))))

                    call (close-call emission locals values call)
                    :else (or (close-pmap emission locals values node)
                              (close-current-map emission locals values node)
                              (close-reduction emission locals values node)
                              (close-scope emission locals values node))))))
            (close-owner [owner]
              (when-let [emission (method-entry/observed-frontend-emission owner)]
                (if (.containsKey ^java.util.IdentityHashMap memo emission)
                  (let [result (.get ^java.util.IdentityHashMap memo emission)] (when (map? result) result))
                  (do
                    ;; Recursion is not a callback-free proof. No fixed-point
                    ;; guess or refreshed global definition is introduced.
                    (.put ^java.util.IdentityHashMap memo emission ::in-progress)
                    (let [bridge (method-entry/frontend-boxed-entry-bridge owner)
                          context (:body-context bridge)
                          body (:body emission)
                          result (when (and context (= 1 (count body))
                                            (not (:unretained-sibling-calls? context)))
                                   (close-value emission (:locals context) {} (first body)))
                          result
                          (when (and result (= (:stack result) (:emitted-result-type context)))
                            (case (:kind bridge)
                              :reference-static-return
                              (when (and (= :ref (:stack result)) (reference? (:body-return-carrier bridge)))
                                (assoc result :stack :ref))
                              :boxed-primitive-static-return
                              (when (or (primitive? (:stack result)) (builtin? result))
                                (when-let [builtin (get types/primitive-info
                                                       (symbol (.getName ^Class (:payload-carrier bridge))))]
                                  {:stack :ref :builtin-class builtin
                                   :boxed-primitive-carrier (:payload-carrier bridge)}))
                              nil))
                          result (when result
                                   (assoc result :method (:invocation-method bridge)
                                          :implementation (:implementation emission)))]
                      (.put ^java.util.IdentityHashMap memo emission (or result ::unsupported))
                      result)))))]
      {:owner close-owner :value close-value}))))

(defn- closed-long-frame-ports
  "Relate actual scalar ports to value-only Long use, without retyping them.
   A ref/Object local needs an exact preceding boxed-primitive producer, not a
   Long hint. Its uses must be direct inputs of actual primitive Long formals."
  [witness binding facts input]
  (let [emission (:emission witness)
        before (:before binding)
        ports (:ports input)
        references (into #{} (keep (fn [[id port]] (when (= :ref (:type port)) id))) ports)
        consumer (when (seq references)
                   (owned-effect-record emission (:operand input) 'retained-invocation-context))]
    (when (and (map? ports) (seq ports)
               (every? (fn [[id port]]
                         (and (= port (get before id))
                              (or (= :long (:type port))
                                  (and (contains? references id)
                                       (contains? #{Object Long} (:reference-class port))
                                       (= port (get-in facts [id :local]))
                                       (identical? Long/TYPE (get-in facts [id :result :boxed-primitive-carrier]))
                                       (identical? Long (get-in facts [id :result :builtin-class])))))) ports)
               (or (empty? references)
                   (and (= :global-typed-indy (:invocation-mode consumer))
                        ((helper-backend 'completed-invocation-inputs?) consumer)
                        (every?
                         (fn [id]
                           (let [uses (filter (fn [[operand _]] (contains? (:ports operand) id))
                                              (map vector (:inputs consumer) (:input-completions consumer)))]
                             (and (seq uses)
                                  (every? (fn [[operand completed]]
                                            (and (= id (:operand operand))
                                                 (= {id (get before id)} (:ports operand))
                                                 (= :ref (:emitted-type completed))
                                                 (identical? Long/TYPE (:parameter-carrier completed))
                                                 (= :coerce (:adaptation completed)))) uses)))) references))))
      {:ports ports :reference-ports references})))

(defn- current-compound-integral-frame!
  "Rederive the compound Int/Long step's representation proof from its FINAL witness.
   Earlier normalization receipts or logical Long tags are not physical facts."
  [witness position current observed]
  (let [bindings (second (:scope witness))
        projector (closed-source-effect-projector (java.util.IdentityHashMap.))
        emission (:emission witness)
        target (get-in observed [:contract :target-carrier])
        target-stack (cond
                       (identical? Integer/TYPE target) :int
                       (identical? Long/TYPE target) :long)]
    (loop [ordinal 0 facts {}]
      (let [id (nth bindings (* 2 ordinal))
            expression (nth bindings (inc (* 2 ordinal)))
            binding (current-source-binding! witness ordinal id expression
                                             :source-completion-compound-binding :source-completion-compound-binding)
            result ((:value projector) emission (:before binding) facts (:initializer binding))]
        (when-not (and result (= (:stack result) (get-in binding [:after :type]))
                       (identical? (:actual result) (:initializer binding)))
          (fail! :source-completion-compound-prefix
                 "compound completion crosses an unclosed source definition"
                 {:position ordinal :binding id}))
        (if (= ordinal position)
          (let [frame (closed-long-frame-ports witness binding facts (:operand-input observed))]
            (when-not (and frame (= (:before binding) (:before current) (:locals-before observed))
                           (identical? (:initializer binding) (:initializer current))
                           (identical? (:initializer binding) (:expression observed))
                           target-stack
                           (= target-stack (:stack result) (get-in current [:after :type]))
                           (= target-stack (:completion-carrier observed)
                              (:returned-carrier observed))
                           (false? (:object-parent? observed)))
              (fail! :source-completion-compound-ports
                     "compound completion lacks its exact scalar integral physical frame"
                     {:position position :target target}))
            frame)
          (recur (inc ordinal) (assoc facts id {:local (:after binding) :result result})))))))

(defn- direct-array-constructor-class
  "Resolve only the shared dtype table's primitive-array constructor facet.
   This is syntax routing plus Class lookup; current bytecode evidence below is
   still required before constructing any source replacement."
  [expression]
  (when (and (seq? expression) (= 2 (count expression)) (symbol? (first expression)))
    (some (fn [[_ facets]]
            (when (= (first expression) (:jvm-array-constructor facets))
              (Class/forName (:jvm-array-class-name facets))))
          dtype/dtype-info)))

(defn- construct-contextual-array-count-frame
  "Name one exact owned newarray count at its original constructor site.
   The physical count is JVM Int, while every public/frame input remains in
   the existing Long domain. This transports wrapping and failure order only;
   it grants no equality with a mathematical Long extent or capacity fact."
  [witness original binding facts projector]
  (let [{:keys [emission input-body]} witness
        array-class (direct-array-constructor-class original)
        record (when (and array-class (seq? (second original)))
                 ((helper-backend 'retained-input-array-count-context)
                  emission input-body original))
        allocation (:allocation record)
        allocation-count (:count allocation)
        raw (:input-type record)
        count-result (when record
                       ((:value projector) emission (:before binding) facts (:operand record)))
        frame-proof (when record
                      (closed-long-frame-ports witness binding facts record))]
    (when (and record allocation
               (identical? original (:source-input record))
               (identical? (second original) (:source-original record))
               (identical? (:initializer binding) (:expression allocation))
               (identical? array-class (:array-class allocation))
               (true? (:fresh? allocation)) (= :jvm-zero (:initialization allocation))
               (= :int (:completed-type record)) (= :raw-jvm-int (:completion record))
               (source-policy/valid? (:source-policy record))
               (= (:source-policy witness) (:source-policy record))
               (identical? (:operand record) (:operand allocation-count))
               (= (select-keys allocation-count
                               [:operand :context :ports :unretained-sibling-calls?
                                :source-policy :input-type :completed-type :completion])
                  (select-keys record
                               [:operand :context :ports :unretained-sibling-calls?
                                :source-policy :input-type :completed-type :completion]))
               (contains? #{:int :long :ref} raw)
               count-result (identical? (:actual count-result) (:operand record))
               (= raw (:stack count-result))
               (or (contains? #{:int :long} raw)
                   (identical? Long (:builtin-class count-result)))
               frame-proof)
      (let [contract (completion/planned-bytecode-input-contract
                      Integer/TYPE (:context record) raw)
            input (assoc (select-keys record
                                      [:operand :context :ports :unretained-sibling-calls?])
                         :contract contract :completion allocation)
            parameter (gensym "array_count_formal__")
            value (with-meta (gensym "completed_array_count__") {:raster.type/tag 'int})
            source-count (:source-original record)
            frame (construct-invocation-input-frame
                   [input] [parameter] [value] (:scope witness) [source-count])
            call (util/remake-from original (list (first original) value))
            artifact {:frame frame :exported-ordinals [0]
                      :exported-bindings (:bindings frame)
                      :call call :original original :witness witness
                      :physical-count record :allocation allocation}]
        (retain-owned-artifact!
         artifact
         #(let [current ((helper-backend 'retained-input-array-count-context)
                         emission input-body original)]
            (validate-contextual-source-witness! witness)
            (when-not (and current
                           (identical? (:source-input record) (:source-input current))
                           (identical? (:source-original record) (:source-original current))
                           (identical? (:owned-source record) (:owned-source current))
                           (identical? allocation (:allocation current))
                           (identical? (:operand record) (:operand current))
                           (= (select-keys record
                                           [:context :ports :unretained-sibling-calls?
                                            :source-policy :input-type :completed-type :completion])
                              (select-keys current
                                           [:context :ports :unretained-sibling-calls?
                                            :source-policy :input-type :completed-type :completion])))
              (fail! :source-input-array-count-correspondence
                     "array count frame differs from its exact current constructor"
                     {}))))))))

(defn- construct-contextual-input-frame
  "Name one exact call's completed arguments without importing its body.
   The prefix and every operand must already close under actual observations.
   Initial compound support is Long over proved scalar Long values, not buffers
   or opaque values. Their physical ports may remain Object references."
  [witness original binding facts projector]
  (let [{:keys [emission input-body]} witness
        record ((helper-backend 'retained-input-invocation-context) emission input-body original)
        arguments (when (and (seq? original) (= '.invk (first original))) (vec (drop 2 original)))
        inputs (:inputs record)
        completed (:input-completions record)
        before (:before binding)
        results (when (and arguments
                           (= :global-typed-indy (:invocation-mode record))
                           ((helper-backend 'completed-invocation-inputs?) record)
                           (identical? (:initializer binding) (:expression record))
                           (= (count arguments) (count inputs) (count completed))
                           (some seq? arguments))
                  (mapv (fn [input]
                          (when (and (false? (:unretained-sibling-calls? input))
                                     (every? (fn [[id port]] (= port (get before id))) (:ports input)))
                            ((:value projector) emission before facts (:operand input)))) inputs))
        closed?
        (and results (every? some? results)
             (every? true?
                     (map (fn [argument input completion result]
                            (let [^Class carrier (:parameter-carrier completion)
                                  raw (:emitted-type completion)
                                  compound? (seq? argument)]
                              (and (= (:stack result) raw)
                                   (if compound?
                                     (and (contains? #{Integer/TYPE Long/TYPE} carrier)
                                          (contains? #{:int :long :ref} raw)
                                          (or (contains? #{:int :long} raw)
                                              (identical? Long (:builtin-class result)))
                                          (boolean (closed-long-frame-ports witness binding facts input)))
                                     (and (symbol? argument)
                                          (= {argument (get before argument)} (:ports input))
                                          (if (.isPrimitive carrier)
                                            (and (= :identity (:adaptation completion))
                                                 (= raw (:type (get before argument))))
                                            (let [port (get before argument)
                                                  source-class (:reference-class port)]
                                              (and (= :ref raw) (true? (:verified port))
                                                   (class? source-class)
                                                   (or (identical? Object carrier)
                                                       (and (.isArray carrier)
                                                            (.isPrimitive (.getComponentType carrier))))
                                                   (.isAssignableFrom carrier ^Class source-class)))))))))
                          arguments inputs completed results)))
        typed (when closed? (owned-occurrence-typed-call original record))
        description (when typed (descriptor/call-description typed))
        signature (when typed
                    (call-boundary/selected-typed-signature
                     (:semantic-op description) (:implementation-op description)
                     (count arguments) (:method record)))
        entry (when (= :known (:status signature))
                (method-entry/selected-source-entry-carriers
                 (:semantic-op description) (:implementation-op description)
                 (count arguments) (get-in signature [:definition :source-var])))
        source-operands
        (when closed?
          (let [parent (owned-effect-record emission (:expression record) 'retained-invocation-context)]
            (mapv (fn [ordinal argument input]
                    (if (seq? argument)
                      (let [edge (when parent
                                   ((helper-backend 'retained-input-source-context)
                                    emission (:source parent) ordinal))]
                        (when (and edge (identical? (:operand input) (:operand edge))
                                   (= (:ports input) (:ports edge)) (= (:context input) (:context edge)))
                          (:source-original edge)))
                      argument)) (range) arguments inputs)))
        parameters (:parameters entry)
        tags (get-in entry [:definition :tags])]
    (when (and entry (= (:method record) (:invocation-method entry))
               (every? some? source-operands)
               (= (:parameter-classes signature) (:parameter-carriers entry))
               (= (count arguments) (count parameters) (count tags)))
      (let [inputs (mapv (fn [input completion]
                           (assoc input :completion completion
                                  :contract (completion/planned-bytecode-input-contract
                                             (:parameter-carrier completion) (:context input)
                                             (:emitted-type completion)))) inputs completed)
            values (mapv #(with-meta (gensym "completed_call_input__") {:raster.type/tag %}) tags)
            frame (construct-invocation-input-frame inputs parameters values (:scope witness) source-operands)
            ;; Only compounds need a new definition. The predicates above have
            ;; proved every remaining argument an inert lexical load with an
            ;; identity/assignable-reference conversion. Keep its EXACT source
            ;; symbol, rather than inventing a new reference-view alias.
            exported-ordinals (vec (keep-indexed #(when (seq? %2) %1) arguments))
            exported-bindings (mapv #(nth (:bindings frame) %) exported-ordinals)
            call-arguments (mapv (fn [argument value] (if (seq? argument) value argument)) arguments values)
            call (util/remake-from original (list* '.invk (second original) call-arguments))]
        (retain-owned-artifact!
         {:frame frame :exported-ordinals exported-ordinals :exported-bindings exported-bindings
          :call call :original original :witness witness :physical-call record}
         #(do (validate-contextual-source-witness! witness)
              (method-entry/assert-source-entry-current! (:source-entry entry))
              (assert-helper-occurrence-current! emission input-body original record)))))))

(defn ^:no-doc project-contextual-invocation-arguments
  "Normalize callback-free compound Long arguments at their original call site,
   including the JVM Int count consumed by an exact primitive-array constructor.
   Reuse the existing ordered input-frame constructor; retain the same global
   call and return behavior. Unknown prefixes/operands stop coverage. No source
   execution, numerical equation, mathematical capacity or successful-check credit follows."
  ([witness] (project-contextual-invocation-arguments witness {}))
  ([witness options]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     (fn []
       (validate-contextual-source-witness! witness)
       (let [source (:scope witness)
             bindings (second source)
             projector (closed-source-effect-projector (java.util.IdentityHashMap.) options)
             finish (fn [constructed frames]
                      (when (seq frames)
                        (retain-owned-artifact!
                         {:source source :form (util/remake-from source
                                               (list* 'let* constructed (drop 2 source)))
                          :frames frames :witness witness}
                         #(do (validate-contextual-source-witness! witness)
                              (doseq [frame frames]
                                (validate-owned-import! frame :frame :source-input-frame-correspondence))))))]
         (loop [ordinal 0 facts {} constructed [] frames []]
           (if (= ordinal (quot (count bindings) 2))
             (finish constructed frames)
             (let [binder (nth bindings (* 2 ordinal))
                   original (nth bindings (inc (* 2 ordinal)))
                   observed (current-source-binding!
                             witness ordinal binder original :source-input-frame-binding :source-input-frame-observation)
                   frame (or (construct-contextual-array-count-frame
                              witness original observed facts projector)
                             (construct-contextual-input-frame
                              witness original observed facts projector))
                   constructed (cond-> constructed
                                 ;; The frame keeps original occurrence evidence privately.
                                 ;; Its NEW source initializers must acquire new occurrences
                                 ;; from the final witness, not leak the old JVM tokens.
                                 frame (into (mapcat (fn [[id initializer]]
                                                       [id ((helper-backend 'detach-emission-occurrences) initializer)])
                                                     (:exported-bindings frame))))
                   constructed (conj constructed binder (or (:call frame) original))
                   frames (cond-> frames frame (conj frame))
                   result ((:value projector) (:emission witness) (:before observed) facts (:initializer observed))]
               (if (and result (= (:stack result) (get-in observed [:after :type]))
                        (identical? (:actual result) (:initializer observed)))
                 (recur (inc ordinal) (assoc facts binder {:local (:after observed) :result result})
                        constructed frames)
                 (finish (into constructed (drop (* 2 (inc ordinal)) bindings)) frames)))))))))))

(defn ^:no-doc contextual-invocation-argument-candidate?
  "Cheap routing only; exact current invocation/constructor and callback closure are separate."
  [source]
  (boolean
   (and (seq? source) (= 'let* (first source)) (vector? (second source))
        (some (fn [[_ expression]]
                (and (seq? expression)
                     (or (and (= '.invk (first expression))
                              (some seq? (drop 2 expression)))
                         (and (direct-array-constructor-class expression)
                              (seq? (second expression))))))
              (partition 2 (second source))))))

(defn ^:no-doc project-contextual-reference-helpers
  "Project an exact witnessed flat prefix using owned operation evidence.
   Unknown callbacks/coordination stop the interval. Crossing an imported call
   additionally closes its whole original producer, including owned pmap or
   reduction scaffolds and every evaluated operand. No preparation, motion,
   scalar evaluation, totality or global concurrent stability claim."
  [witness]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     (fn []
       (validate-contextual-source-witness! witness)
       (let [source (:scope witness)
             bindings (second source)
             projector (closed-source-effect-projector (java.util.IdentityHashMap.))
             finish (fn [constructed applications endpoint]
                      (when (seq applications)
                        (let [form (util/remake-from source
                                     (list* 'let* constructed (drop 2 source)))]
                          (retain-owned-artifact!
                           {:source source :form form :applications applications
                            :witness witness :prefix-endpoint endpoint}
                           #(do (validate-contextual-source-witness! witness)
                                (doseq [application applications]
                                  (validate-helper-application! application)))))))]
         (loop [ordinal 0 facts {} constructed bindings applications [] endpoint nil]
           (if (= ordinal (quot (count bindings) 2))
             (finish constructed applications endpoint)
             (let [binder (nth bindings (* 2 ordinal))
                   original (nth bindings (inc (* 2 ordinal)))
                   observed (current-source-binding!
                             witness ordinal binder original
                             :closed-prefix-binding :closed-prefix-observation)
                   application (construct-prefix-reference-application witness original observed facts)
                   constructed (if application
                                 (assoc constructed (inc (* 2 ordinal)) (:expression application)) constructed)
                   applications (cond-> applications application (conj application))
                   endpoint (if application ordinal endpoint)
                   result ((:value projector) (:emission witness) (:before observed) facts (:initializer observed))]
               (if (and result (= (:stack result) (get-in observed [:after :type]))
                        (identical? (:actual result) (:initializer observed)))
                 (recur (inc ordinal) (assoc facts binder {:local (:after observed) :result result})
                        constructed applications endpoint)
                 (finish constructed applications endpoint))))))))))

(defn ^:no-doc contextual-reference-helper-candidate?
  "Syntax-only routing for generated flat source whose earlier completions
   prevent static call substitution. This grants no callee/effect authority."
  [source]
  (when (and (seq? source) (= 'let* (first source)) (vector? (second source)))
    (loop [pairs (seq (partition 2 (second source))) completed? false]
      (when-let [[_ expression] (first pairs)]
        (or (and completed? (:dispatch? (descriptor/call-description expression)))
            (recur (next pairs) (or completed? (completion/canonical-form? expression))))))))

(defn- contextual-array-entry-facts [witness]
  (let [context ((helper-backend 'retained-input-body-context)
                 (:emission witness) (:input-body witness))]
    (into {}
          (keep (fn [[parameter carrier]]
                  (let [local (get-in context [:locals parameter])]
                    (when (and (dtype/dtype-for-jvm-array-class carrier)
                               (= :ref (:type local)) (identical? carrier (:reference-class local)))
                      [parameter {:local local
                                  :result {:stack :ref :array-class carrier
                                           :array-origin {:kind :entry-array :parameter parameter
                                                          :local local}}}]))))
          (map vector (:parameters witness) (:parameter-carriers witness)))))

(defn- integral-array-length-body? [expression operand]
  (and (seq? expression) (= 2 (count expression))
       (case (first expression)
         clojure.core/alength (= operand (second expression))
         (clojure.core/int clojure.core/long)
         (integral-array-length-body? (second expression) operand)
         false)))

(defn- contextual-int-length-source [witness observation original facts projector]
  ;; Publication is use-closed here: the actual Object(Long) result is consumed
  ;; immediately by this original checked Int initializer, never aliased.
  (when (and (seq? original) (= 2 (count original))
             (= 'clojure.core/int (first original)))
    (let [emission (:emission witness) input-body (:input-body witness)
          call-source (second original)
          conversion ((helper-backend 'retained-input-scalar-conversion) emission input-body original)
          call ((helper-backend 'retained-input-invocation-context) emission input-body call-source)
          input (first (:inputs call))
          completed (first (:input-completions call))
          operand (:operand input)
          local (get-in observation [:before operand])
          result (when (= local (get-in facts [operand :local])) (get-in facts [operand :result]))
          carrier (:array-class result)
          call-result (when call ((:value projector) emission (:before observation) facts (:expression call)))
          typed (when call (owned-occurrence-typed-call call-source call))
          body (when (and typed carrier)
                 (invocation-source/selected-array-length-body typed carrier))]
      (when (and conversion call typed (:array-origin result)
                 (dtype/dtype-for-jvm-array-class carrier)
                 (= :int (get-in observation [:after :type]) (:result-carrier conversion))
                 (= 'clojure.core/int (:source-operation conversion))
                 (identical? Integer/TYPE (:target-carrier conversion))
                 (= :ref (:input-type conversion) (:stack call-result))
                 (identical? Long (:builtin-class call-result))
                 (identical? (:initializer observation) (:expression conversion))
                 (identical? (:operand conversion) (:expression call))
                 ((helper-backend 'completed-invocation-inputs?) call)
                 (= 1 (count (:inputs call)) (count (:input-completions call)))
                 (symbol? operand) (= :ref (:type local) (:emitted-type completed))
                 (= {operand local} (:ports input)) (false? (:unretained-sibling-calls? input))
                 (identical? carrier (:parameter-carrier completed))
                 (integral-array-length-body? body operand)
                 (intrinsic-source/observe-original! (:source-ns witness) 'clojure.core/int))
        ;; The actual view Class, not a logical AV, supplies this source operand
        ;; annotation. Its ordinary emitter check preserves null/CHECKCAST order.
        (let [tag (dtype/array-tag-for-dtype (dtype/dtype-for-jvm-array-class carrier))
              body (util/subst-syms {operand (with-meta operand (assoc (meta operand) :tag tag :raster.type/tag tag))} body)]
          ((helper-backend 'detach-emission-occurrences)
           (with-meta (list 'clojure.core/int body) (meta original))))))))

(defn ^:no-doc project-contextual-array-lengths
  "Construct canonical integral array-length SOURCE at an exact checked Int
   binding after a closed current prefix. Actual entry/allocation view origins,
   selected Method/input conversions and use-closed result publication remain
   private to this constructor. No AV capacity, presence, motion or host/device
   transfer is granted; the downstream source ShapeProjection proves its view."
  [witness]
  (when method-entry/*emission-inputs*
    (method-entry/call-with-independent-definition-inputs
     (fn []
       (validate-contextual-source-witness! witness)
       (let [source (:scope witness) bindings (second source)
             projector (closed-source-effect-projector (java.util.IdentityHashMap.))
             finish (fn [bindings rewrites]
                      (when (seq rewrites)
                        (retain-owned-artifact!
                         {:source source :form (util/remake-from source (list* 'let* bindings (drop 2 source)))
                          :rewrites rewrites :witness witness}
                         #(validate-contextual-source-witness! witness))))]
         (loop [ordinal 0 facts (contextual-array-entry-facts witness) constructed bindings rewrites []]
           (if (= ordinal (quot (count bindings) 2))
             (finish constructed rewrites)
             (let [binder (nth bindings (* 2 ordinal)) original (nth bindings (inc (* 2 ordinal)))
                   observation (current-source-binding! witness ordinal binder original
                                                        :contextual-length-binding :contextual-length-observation)
                   result ((:value projector) (:emission witness) (:before observation) facts (:initializer observation))]
               (if (and result (= (:stack result) (get-in observation [:after :type]))
                        (identical? (:actual result) (:initializer observation)))
                 (let [replacement (contextual-int-length-source witness observation original facts projector)]
                   (recur (inc ordinal) (assoc facts binder {:local (:after observation)
                                                            :result (array-origin-at-local result (:after observation))})
                          (if replacement (assoc constructed (inc (* 2 ordinal)) replacement) constructed)
                          (cond-> rewrites replacement (conj {:ordinal ordinal :source original :expression replacement}))))
                 (finish constructed rewrites))))))))))

(defn ^:no-doc contextual-array-length-candidate?
  "Cheap routing for the whole-final-Int constructor; no selection or proof."
  [source]
  (and (seq? source) (= 'let* (first source)) (vector? (second source))
       (boolean
        (some (fn [[_ expression]]
                (when (and (seq? expression) (= 2 (count expression))
                           (= 'clojure.core/int (first expression)))
                  (let [call (descriptor/call-description (second expression))]
                    (and (:dispatch? call) (descriptor/alength-op? (:semantic-op call))))))
              (partition 2 (second source))))))

(defn- direct-double-map-helper? [node]
  (when (and (par/par-map-form? node) (= 6 (count node)))
    (let [{:keys [cast body]} (par/extract-par-map-info node)
          {:keys [dispatch? semantic-op lexical-callee]} (descriptor/call-description body)]
      (and (contains? '#{double clojure.core/double} cast)
           dispatch? (not lexical-callee)
           (not (some-> semantic-op intrinsics/canonical intrinsics/descriptor))))))

(defn ^:no-doc direct-double-map-helper-candidate?
  "Cheap syntax-only routing. No source query, emission, or admission is done."
  [source]
  (letfn [(candidate? [node]
            (cond
              (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) false
              (direct-double-map-helper? node) true
              (par/par-map-form? node) false
              (and (seq? node) (= 'let* (first node)) (vector? (second node)))
              (boolean (or (some #(candidate? (second %)) (partition 2 (second node)))
                           (some candidate? (nnext node))))
              (and (seq? node) (= 'do (first node))) (boolean (some candidate? (rest node)))
              :else false))]
    (and (seq? source) (= 'let* (first source)) (vector? (second source)) (candidate? source))))

(defn- double-array-carrier? [carrier]
  (and (class? carrier) (.isArray ^Class carrier)
       (identical? Double/TYPE (.getComponentType ^Class carrier))))

(defn- unanchored-numerical-payload? [expression slots]
  ;; Imported lexical context/completion anchors are not a numerical-source
  ;; projection. Leave let/control/Java and unknown grammar to a later owner.
  (cond
    (symbol? expression) (contains? slots expression)
    (number? expression) true
    (seq? expression)
    (let [operation (first expression)]
      (and (symbol? operation) (= "clojure.core" (namespace operation))
           (not (completion/canonical-form? expression))
           (or (descriptor/cast-op? operation) (descriptor/scalar-op? operation))
           (every? #(unanchored-numerical-payload? % slots) (rest expression))))
    :else false))

(defn- direct-map-source-argument?
  [edge index available-parameters]
  (let [{:keys [source input array-read]} edge
        arguments (when (descriptor/aget-call? source) (descriptor/call-args source))
        [sym idx] arguments]
    (or (and (number? source) (identical? source (:operand input))
             (= :double (types/numeric-literal-carrier source)))
        (and (= 2 (count arguments)) (symbol? sym) (= idx index)
             (double-array-carrier? (get available-parameters sym))
             array-read (= :double (:result-type array-read))
             (= sym (:array array-read))
             (= index (get-in array-read [:input :operand]))
             (contains? #{:int :long} (get-in array-read [:input :type]))))))

(defn- project-direct-double-map-result
  [witness node available-parameters lexical]
  (when (direct-double-map-helper? node)
    (let [{:keys [out idx cast body]} (par/extract-par-map-info node)
          source-ns (:source-ns witness)
          context {:source-ns source-ns :pending-source? false
                   :dependency-snapshots (get-in witness [:emission :dependency-snapshots])}]
      (when (and (symbol? idx) (not (contains? available-parameters idx))
                 (double-array-carrier? (get available-parameters out))
                 (= 'clojure.core/double (method-entry/observed-body-operation context cast (conj lexical idx)))
                 (intrinsic-source/observe-original! source-ns 'clojure.core/double))
        (when-let [application (instantiate-contextual-helper-occurrence witness body)]
          (validate-helper-application! application)
          (let [projection (:application application)
                record (:physical-call projection)
                numerical (:numerical-projection projection)
                slots (:input-slots projection)
                values (mapv :value slots)
                edges ((helper-backend 'retained-input-invocation-operands)
                       (:emission witness) (:input-body witness) body)
                expression (:payload-expression application)]
            (when (and (identical? Double/TYPE (:payload-carrier numerical))
                       (= :double (get-in numerical [:bridge :body-context :emitted-result-type]))
                       (= (count edges) (count slots)) (every? some? edges)
                       (every? #(and (identical? Double/TYPE (:parameter-carrier %))
                                     (= :double (:emitted-type %)) (= :identity (:adaptation %)))
                               (:input-completions record))
                       (every? #(direct-map-source-argument? % idx (dissoc available-parameters idx)) edges)
                       (unanchored-numerical-payload? expression (set values)))
              (let [bindings (vec (mapcat (fn [value edge]
                                           [(vary-meta value assoc :raster.type/tag 'double
                                                       :raster.effect/effectful true)
                                            (if (instance? clojure.lang.IObj (:source edge))
                                              (vary-meta (:source edge) assoc :raster.effect/preserve-evaluation true)
                                              (:source edge))]) values edges))
                    result (list 'let* bindings expression)
                    ;; The original payload stays owned by its receipt. Only
                    ;; this newly proved numerical source loses the emitter's
                    ;; private occurrence tokens before portable admission.
                    mapped ((helper-backend 'detach-emission-occurrences)
                            (util/remake-from node (concat (take 5 node) [result])))]
                {:form mapped :application application}))))))))

(defn ^:no-doc project-direct-double-map-results
  "Construct ordinary numerical source only for direct, use-closed Double map sinks.
   Uses an existing exact-candidate emission owner; all argument-origin and
   identity conversions remain checked. Receipts never enter the portable IR."
  [witness]
  (method-entry/call-with-independent-definition-inputs
   (fn []
     (validate-contextual-source-witness! witness)
     (let [applications (volatile! [])
           parameters (zipmap (:parameters witness) (:parameter-carriers witness))]
       (letfn [(walk [node available lexical]
                 (cond
                   (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
                   (and (seq? node) (= 'let* (first node)) (vector? (second node)))
                   (let [[pairs available lexical]
                         (reduce (fn [[pairs available lexical] [id initializer]]
                                   [(conj pairs [id (walk initializer available lexical)])
                                    (dissoc available id) (conj lexical id)])
                                 [[] available lexical] (partition 2 (second node)))]
                     (util/remake-from node (list* (first node) (vec (mapcat identity pairs))
                                                  (mapv #(walk % available lexical) (nnext node)))))
                   (par/par-map-form? node)
                   (if-let [projected (try (project-direct-double-map-result witness node available lexical)
                                          (catch clojure.lang.ExceptionInfo error
                                            (if (= :generated-body-unsupported (:reason (ex-data error)))
                                              nil (throw error))))]
                     (do (vswap! applications conj (:application projected)) (:form projected))
                     node)
                   ;; Unknown lexical/control owners are not traversed: a
                   ;; surrounding branch/fn/loop must supply its own contract.
                   (and (seq? node) (= 'do (first node)))
                   (util/remake-if-changed node (into [(first node)] (map #(walk % available lexical) (rest node))))
                   :else node))]
         (let [source (:scope witness)
               result (walk source parameters (set (keys parameters)))]
           (when (seq @applications)
             (retain-owned-artifact!
              {:scope source :form result :applications @applications}
              #(do (validate-contextual-source-witness! witness)
                   (doseq [application @applications] (validate-helper-application! application)))))))))))

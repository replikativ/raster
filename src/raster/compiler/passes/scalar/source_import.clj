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
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason :front-end :analyzed-source))))

(deftype ^:private TerminalReductionImportReceipt [artifact guard]
  method-entry/RetainedProducerInput
  (assert-producer-current! [this]
    (guard)
    this))

(defn ^:no-doc validate-terminal-reduction-import!
  "Check the exact isolated component and its historical owner. The receipt and
   Methods are process-local; they are never facts in the portable numerical IR."
  [{:keys [receipt] :as imported}]
  (when-not (and (instance? TerminalReductionImportReceipt receipt)
                 (= (set (keys imported))
                    (conj (set (keys (.-artifact ^TerminalReductionImportReceipt receipt))) :receipt))
                 (every? (fn [[key original]] (identical? original (get imported key)))
                         (.-artifact ^TerminalReductionImportReceipt receipt)))
    (fail! :terminal-reduction-import-correspondence
           "terminal reduction import differs from its exact analyzed component" {}))
  (method-entry/assert-producer-current! receipt)
  imported)

(defn ^:no-doc analyze-terminal-reduction-import
  "Construct an isolated analyzed terminal-reduction component under its source owner.
   No pipeline deferral or execution capability is installed. Lexical array arguments
   are the bounded input grammar. Original input, return bridge and caller binder
   completions remain explicit unresolved obligations, not portable payload proofs."
  [{:keys [frontend input-body root caller-expression arguments result-symbol source-position
           values array-types scalar-types]}]
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
           call-arguments (vec (util/call-args caller-expression))
           _ (when-not (and (seq? caller-expression)
                            (= '.invk (first caller-expression))
                            (= (:implementation-op selected) (second caller-expression))
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
           policy (get-in frontend [:source-entry :source-policy])
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
           port (with-meta (gensym "terminal_object_") {:tag 'Object :raster.type/tag result-tag})
           recipe (completion/complete-value
                   completion/object-input-contract
                   (completion/complete-value
                    (completion/planned-bytecode-input-contract carrier context :ref)
                    (completion/complete-value (completion/bytecode-result-context-contract context :ref) port)))
           obligations {:inputs (mapv (fn [argument parameter-carrier]
                                        {:argument argument :parameter-carrier parameter-carrier
                                         :status :unresolved}) arguments parameter-carriers)
                        :return {:value-port port :recipe recipe :compute-method compute
                                 :invocation-method (:invocation-method bridge) :status :unresolved}
                        :import {:result numerical-result :status :unresolved}
                        :placement {:source-position source-position :status :unresolved}
                        :caller-binding {:binder result-symbol :initializer caller-expression :status :unresolved}}
           artifact {:component component :source source :normalized-source normalized
                     :argument-bindings (:argument-bindings expansion)
                     :caller-expression caller-expression :result-symbol result-symbol
                     :source-position source-position :source-policy policy :obligations obligations}
           historical @method-entry/*emission-inputs*
           guard #(do (method-entry/assert-definition-inputs-current! historical)
                      (method-entry/assert-frontend-source! frontend source-var)
                      (method-entry/assert-emission-inputs-current! emission))
           receipt (TerminalReductionImportReceipt. artifact guard)
           imported (assoc artifact :receipt receipt)]
       (validate-terminal-reduction-import! imported)
       (method-entry/adopt-definition-inputs! {:producer-inputs [receipt]})
       imported))))

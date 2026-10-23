(ns raster.compiler.backend.jvm.generated-body
  "Construct a new generated program from exact owned helper bodies.
   This is not an original-source occurrence relation or GPU admission proof.
   The caller owns the resulting program and must retain its collected inputs.
   No source frontend, numerical invocation or phantom caller module is created."
  (:require [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.normal-return :as normal-return]))

(defn- decline! [message data]
  (throw (ex-info message (assoc data :reason :generated-body-unsupported))))

(defn ^:no-doc physical-entry-view
  "Read the same generated body/parameter/wrapper boundary used by runtime.
   No equal reconstructed body, implementation or parameter vector can stand
   in for that boundary. The caller still retains its enclosing producer."
  [{:keys [emission input-body implementation interface-class]} parameters]
  (let [bridge (bytecode/retained-boxed-entry-bridge emission)]
    (when-not (and bridge (vector? parameters)
                   (bytecode/retained-input-body-context emission input-body)
                   (identical? parameters (:params emission))
                   (identical? implementation (:implementation emission))
                   (identical? interface-class (:interface bridge))
                   (= (count parameters) (count (get-in bridge [:boxed-entry :parameter-carriers])))
                   (= (count parameters) (count (set parameters))))
      (throw (ex-info "Generated call has no exact boxed entry input boundary"
                      {:reason :generated-body-boxed-entry-inputs})))
    {:bridge bridge :body input-body
     :parameter-carriers (get-in bridge [:boxed-entry :parameter-carriers])}))

(defn instantiate-boxed-entry-inputs
  "Construct the input boundary of an authenticated generated public entry.
   All original operands complete in Object context before any wrapper conversion,
   just as applyTo/invoke does. The caller must retain the physical owner/guard;
   these bindings alone grant no numerical realization or portable import."
  [{:keys [input-body] :as physical} parameters arguments]
  (let [{:keys [bridge parameter-carriers]} (physical-entry-view physical parameters)
        carriers parameter-carriers]
    (when-not (and (vector? arguments) (= (count parameters) (count arguments) (count carriers)))
      (throw (ex-info "Generated call has no exact boxed entry input boundary"
                      {:reason :generated-body-boxed-entry-inputs})))
    (let [used (volatile! (into (set parameters) (filter symbol?)
                                (tree-seq coll? seq [input-body arguments])))
          fresh! (fn [prefix]
                   (loop []
                     (let [id (gensym prefix)]
                       (if (contains? @used id) (recur)
                           (do (vswap! used conj id) id)))))
          slots (mapv (fn [ordinal formal argument carrier]
                        {:ordinal ordinal :formal formal :original-argument argument
                         :object-binder (fresh! "entry_object__")
                         :input-binder (fresh! "entry_input__")
                         :parameter-class carrier
                         :contract (completion/bytecode-boxed-parameter-contract carrier)})
                      (range) parameters arguments carriers)
          objects (mapv (fn [{:keys [object-binder original-argument]}]
                          [object-binder (completion/complete-value completion/object-input-contract
                                                                    original-argument)]) slots)
          inputs (mapv (fn [{:keys [input-binder object-binder contract]}]
                         [input-binder (completion/complete-value contract object-binder)]) slots)]
      {:slots slots :object-bindings objects :input-bindings inputs
       :bindings (into objects inputs)
       :substitutions (into {} (map (juxt :formal :input-binder)) slots)
       :boxed-entry-bridge bridge})))

(defn- fact-dtype [fact]
  ;; Invert the existing builtin carrier table; no new type/function registry.
  ;; Object itself is not a numeric type. Only an independently established
  ;; exact builtin payload may supply its conditional normal-return dtype.
  (let [matches (into #{}
                      (keep (fn [[tag boxed]]
                              (when (or (identical? boxed (:builtin-class fact))
                                        (identical? (types/tag->java-class tag) (:carrier fact)))
                                (dtype/dtype-for-scalar-tag tag))))
                      types/primitive-info)]
    (when (= 1 (count matches)) (first matches))))

(defn- project-numeric-completion [value from target details]
  (let [to (fact-dtype {:carrier target})
        ;; Core has no unchecked-double. Its ordinary double cast performs the
        ;; same primitive numeric adaptation; narrowing keeps the explicit
        ;; unchecked spelling and must still resolve through the shared policy.
        operation (when to (if (= :double to) 'clojure.core/double
                               (symbol "clojure.core" (str "unchecked-" (name to)))))
        policy (when (and from to (not= from to)) (conversion/source-policy from to operation))]
    (when-not (and from to)
      (decline! "Generated numeric completion lacks an unambiguous builtin source and target"
                (assoc details :source from :target to)))
    {:source-dtype from :target-dtype to
     :classification (if (= from to) :numeric-identity :scalar-conversion)
     :candidate (if (= from to) value
                    (if policy
                      (let [[rounding overflow] policy]
                        (list conversion/scalar-convert-symbol
                              {:source-dtype from :target-dtype to :rounding rounding
                               :overflow overflow :source-op operation} value))
                      (decline! "Generated numeric completion has no exact scalar conversion"
                                (assoc details :source from :target to))))}))

(defn- lexical-id? [id] (and (symbol? id) (nil? (namespace id))))

(defn- generated-flat-source! [input-body parameters]
  (let [[head bindings outputs :as source] (when (= 1 (count input-body)) (first input-body))]
    (when-not (and (vector? parameters)
                   (every? lexical-id? parameters)
                   (= (count parameters) (count (set parameters)))
                   (= 'let* head) (= 3 (count source))
                   (vector? bindings) (even? (count bindings)) (vector? outputs)
                   (every? lexical-id? (take-nth 2 bindings))
                   (= (quot (count bindings) 2) (count (set (take-nth 2 bindings))))
                   (not-any? (set parameters) (take-nth 2 bindings)))
      (decline! "Generated module requires its complete flat binding and result spine" {}))
    source))

(defn construct-value-source-template
  "Construct the pending numerical template BEFORE outer entry emission.

   Only explicit primitive input recipes determine a conversion here. Unknown
   reference payload conversions remain fresh canonical-completion holes, one
   per traversal occurrence even when the original expression object is shared.
   All wrappers/checks remain pending manifest obligations. No emitted result,
   inferred scalar default, physical observation or numerical evaluation is used.
   The owning preparation must retain this exact template and later validate its
   complete occurrence spine; the returned map alone grants no admission."
  [input-body parameters]
  (let [source (generated-flat-source! input-body parameters)
        obligations (volatile! [])]
    (letfn [(walk [node path]
              (cond
                (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
                (and (seq? node) (= completion/completion-symbol (first node)))
                (let [parts (or (completion/canonical-parts node)
                                (throw (ex-info "Malformed generated value completion"
                                                {:reason :invalid-value-completion :path path})))
                      {:keys [contract operand]} parts
                      value (walk operand (conj path :operand))
                      context (:source-context contract)
                      _ (when (= :jvm-primitive-input context)
                          (decline! "Source-policy input completion needs its own ordered projection"
                                    {:path path}))
                      numeric? (and (contains? #{:jvm-bytecode-floating-input :jvm-bytecode-integral-input
                                                :jvm-bytecode-boxed-parameter-input} context)
                                    (.isPrimitive ^Class (:target-carrier contract)))
                      raw (:expected-input-carrier contract)
                      from (when (and numeric? (contains? #{:int :long :float :double} raw)) raw)
                      numeric (when from
                                (project-numeric-completion value from (:target-carrier contract) {:path path}))
                      hole (when (and numeric? (nil? numeric))
                             (completion/complete-value contract value))
                      candidate (or (:candidate numeric) hole value)]
                  (vswap! obligations conj
                          {:token (Object.) :source node :path path :contract contract
                           :candidate candidate :hole hole :status :pending
                           :source-dtype (:source-dtype numeric) :target-dtype (:target-dtype numeric)
                           :classification (case context
                                             :jvm-object-input :object-materialization
                                             :jvm-bytecode-result-context :result-context
                                             :jvm-checkcast :reference-check
                                             (if numeric? (or (:classification numeric) :numeric-boundary-hole)
                                                 (if (identical? Object (:target-carrier contract))
                                                   :reference-identity :reference-check)))})
                  candidate)
                (or (seq? node) (vector? node))
                (util/remake-if-changed node (mapv #(walk %1 (conj path %2)) node (range)))
                (coll? node) (if (completion/contains-completion? node)
                              (decline! "Completion in an unordered value container is not a module region"
                                        {:path path}) node)
                :else node))]
      (let [candidate (walk source [])]
        {:input-body input-body :parameters parameters :original source :source candidate
         :completion-obligations @obligations :admitted? false}))))

(defn- with-value-source-template [program]
  (try
    (assoc program ::value-source-template-candidate
           (construct-value-source-template (:walked-body program) (:params program)))
    (catch clojure.lang.ExceptionInfo error
      (if (= :generated-body-unsupported (:reason (ex-data error)))
        program
        (throw error)))))

(defn- template-completion-spine! [template input-body parameters spine]
  (let [obligations (:completion-obligations template)]
    (when-not (and template (identical? input-body (:input-body template))
                   (identical? parameters (:parameters template))
                   (identical? (first input-body) (:original template))
                   (vector? obligations) (some? spine)
                   (= (count obligations) (count spine))
                   (= (count obligations) (count (set (map :token obligations))))
                   (every? (fn [[obligation record]]
                             (and (:token obligation) (= :pending (:status obligation))
                                  (identical? (:source obligation) (:source-input record))
                                  (= (:contract obligation) (:contract record))
                                  (or (not (contains? (:contract obligation) :expected-input-carrier))
                                      (= :diverge (:input-carrier record))
                                      (= (get-in obligation [:contract :expected-input-carrier])
                                         (:input-carrier record)))))
                           (map vector obligations spine)))
      (throw (ex-info "Generated template lacks its complete original completion occurrence spine"
                      {:reason :unclosed-generated-module-completion})))
    spine))

(defn retain-value-source-template
  "Attach the SAME pre-emission template after actual emission validation.
   The enclosing GeneratedPreparationOwner remains the publication authority.
   Unsupported construction omits the template without disabling CPU AD."
  [program {:keys [emission input-body] :as physical}]
  (if-let [template (::value-source-template-candidate program)]
    (if-let [spine (bytecode/retained-input-completion-spine emission input-body)]
      (do (template-completion-spine! template input-body (:params program) spine)
          (assoc physical ::value-source-template template))
      ;; A missing/unemitted occurrence cannot authorize a numerical import,
      ;; but need not invalidate the original CPU implementation.
      physical)
    physical))

(defn propose-value-source
  "Refine only the pending numeric boundary holes of the retained template.
   Numerical operations come from pre-emission construction, not reconstructed
   outer JVM code. Every check/box/completion remains pending; the surrounding
   importer still owes source, normalization, typed and InvocationPlan closure.
   No posthoc reconstruction fallback is available on this production path."
  [{:keys [emission input-body] :as physical} parameters]
  (let [{:keys [parameter-carriers]} (physical-entry-view physical parameters)
        template (::value-source-template physical)
        _ (when-not template
            (decline! "Generated module has no retained pre-emission numerical template" {}))
        spine (template-completion-spine!
               template input-body parameters
               (bytecode/retained-input-completion-spine emission input-body))
        [_ bindings _ :as source] (:original template)
        ;; Normal-return analysis may use an exact unique operand completion.
        ;; Shared syntax with distinct lexical occurrences supplies no global
        ;; fact; occurrence-specific raw input records remain available below.
        contexts (java.util.IdentityHashMap.)
        _ (doseq [record spine]
            (let [node (:source-input record)]
              (.put contexts node (when-not (.containsKey contexts node) record))))
        resolve-fact (fn [node] (:result-fact (.get contexts node)))
        seed (into {} (keep (fn [[id ^Class carrier]]
                              (when (.isPrimitive carrier) [id {:carrier carrier :presence :present}])))
                   (map vector parameters parameter-carriers))
        facts (normal-return/completed-scalar-bindings
               (partition 2 bindings) seed (:source-ns emission) resolve-fact)
        root-locals
        (reduce (fn [locals [ordinal [id _]]]
                  (let [record (bytecode/retained-input-binding-context emission input-body source ordinal)]
                    (when-not record
                      (decline! "Generated module lost its emitted root binding" {:binder id}))
                    (assoc locals id (:after record))))
                (:locals (bytecode/retained-input-body-context emission input-body))
                (map-indexed vector (partition 2 bindings)))
        replacements (java.util.IdentityHashMap.)
        obligations
        (mapv
         (fn [obligation observed]
           (if-let [hole (:hole obligation)]
             (let [{:keys [contract operand]} (completion/canonical-parts (:source obligation))
                   lexical-facts (into {}
                                       (filter (fn [[id _]]
                                                 (and (contains? (:locals-before observed) id)
                                                      (= (get root-locals id)
                                                         (get (:locals-before observed) id))))) facts)
                   fact (normal-return/completed-scalar-fact operand lexical-facts
                                                           (:source-ns emission) resolve-fact)
                   raw (:input-carrier observed)
                   from (or (when (contains? #{:int :long :float :double} raw) raw)
                            (fact-dtype fact))
                   numeric (project-numeric-completion (nth hole 2) from (:target-carrier contract)
                                                       {:path (:path obligation) :source-fact fact})]
               (.put replacements hole (:candidate numeric))
               (merge obligation numeric {:observed observed :source-fact fact}))
             (assoc obligation :observed observed)))
         (:completion-obligations template) spine)]
    (letfn [(specialize [node]
              (cond
                (.containsKey replacements node) (specialize (.get replacements node))
                (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
                (or (seq? node) (vector? node))
                (util/remake-if-changed node (mapv specialize node))
                :else node))]
      {:source (specialize (:source template)) :original source :physical physical
       :template template :normal-return-facts facts
       :completion-obligations (mapv #(update % :candidate specialize) obligations)
       :admitted? false})))

(defn- validate-call-spine! [source binding-count output-count]
  (let [[head bindings outputs] (when (seq? source) source)
        ids (when (vector? bindings) (take-nth 2 bindings))]
    (when-not (and (= 'let* head) (= 3 (count source))
                   (vector? bindings) (= (* 2 binding-count) (count bindings))
                   (= binding-count (count (set ids)))
                   (every? lexical-id? ids)
                   (vector? outputs) (= output-count (count outputs)))
      (decline! "Generated call construction did not preserve the complete flat spine" {})))
  source)

(defn- completion-leaf
  "Follow only canonical completion wrappers, retaining the original source coordinate."
  [expression]
  (loop [expression expression path []]
    (if (completion/canonical-form? expression)
      (recur (:operand (completion/canonical-parts expression)) (conj path :operand))
      {:expression expression :path path})))

(defn- replace-completion-leaf [expression replacement]
  (if (completion/canonical-form? expression)
    (util/remake-from expression
                      [(first expression) (second expression)
                       (replace-completion-leaf (nth expression 2) replacement)])
    replacement))

(defn- ordered-allocation-inputs
  "Construct pending argument completions from one actual global typed invocation.
   Allocator classification selects a candidate only; it grants no allocation contract.
   The original input-region owner still owes execution, storage and correspondence."
  [{:keys [emission input-body]} root first-index fresh!]
  (let [{original :expression original-path :path} (completion-leaf (:original-expression root))
        {candidate :expression} (completion-leaf (:expression root))]
    (when (and (seq? original) (= '.invk (first original))
               (descriptor/alloc-op? (descriptor/semantic-op original))
               (= 2 (count (descriptor/call-args original)))
               (symbol? (first (descriptor/call-args original)))
               (seq? (second (descriptor/call-args original))))
      (let [observed (bytecode/retained-input-completed-invocation-context emission input-body original)
            ^java.lang.reflect.Method method (:method observed)
            carriers (when (instance? java.lang.reflect.Method method)
                       (vec (.getParameterTypes method)))
            size-source (bytecode/retained-input-invocation-input-source emission input-body original 1)
            candidate-arguments (vec (descriptor/call-args candidate))]
        (when-not (and observed (= :global-typed-indy (:invocation-mode observed))
                       (identical? original (:source-input observed))
                       (not (.isVarArgs method)) (= 2 (count carriers))
                       (dtype/dtype-for-jvm-array-class (first carriers))
                       (identical? Long/TYPE (second carriers))
                       size-source
                       (identical? (second (descriptor/call-args original)) (:source-input size-source))
                       (seq? candidate) (= '.invk (first candidate))
                       (= 2 (count candidate-arguments))
                       (identical? (second original) (second candidate))
                       (identical? (second original) (:receiver observed)))
          (decline! "Allocation argument staging requires its exact ordered invocation inputs"
                    {:root-index (:root-index root)}))
        (let [arguments
              (mapv
               (fn [ordinal ^Class carrier input completed operand]
                 (let [contract (completion/planned-bytecode-input-contract
                                 carrier (:context input) (:emitted-type completed))
                       id (fresh!)
                       expression (completion/complete-value contract operand)]
                   {:input-index ordinal :call-binding-index (+ first-index ordinal)
                    :template-source-path (into [1 (inc (* 2 (:root-index root)))]
                                       (conj original-path (+ 2 ordinal)))
                    :original-operand (nth (vec (descriptor/call-args original)) ordinal)
                    :operand operand :contract contract :input input :completion completed
                    :binding [id expression] :status :pending}))
               (range) carriers (:inputs observed) (:input-completions observed) candidate-arguments)
              rebuilt (util/remake-from candidate
                                        (into (vec (take 2 candidate)) (map (comp first :binding) arguments)))]
          {:source-call original :template-call-path original-path
           :candidate-call candidate :constructed-call rebuilt :observed observed
           :input-expression (:expression root)
           :arguments arguments
           :expression (replace-completion-leaf (:expression root) rebuilt)})))))

(defn instantiate-value-source
  "Construct a pending canonical call from an exact generated entry.

   `input-facts` are conditional producer requirements supplied by the caller,
   NOT authenticated here. Every staged Object context, wrapper conversion and
   body completion remains an explicit pending obligation. This constructor
   grants no source correspondence, successful checks, scheduling or execution.
   A later importer must authenticate these facts against the caller and join
   the retained construction to normalization and one ordered InvocationPlan.

   All original arguments are staged before ANY conversion. Unknown numeric
   payloads refuse rather than borrowing the parameter's result declaration."
  [physical parameters arguments input-facts]
  (when-not (and (vector? input-facts) (= (count arguments) (count input-facts)))
    (decline! "Generated call needs one pending producer requirement per original argument" {}))
  (let [proposal (propose-value-source physical parameters)
        plan (instantiate-boxed-entry-inputs physical parameters arguments)
        source-ns (get-in physical [:emission :source-ns])
        _ (when-not (and (instance? clojure.lang.Namespace source-ns)
                         (identical? source-ns (find-ns (ns-name source-ns))))
            (decline! "Generated call source namespace is no longer installed" {}))
        staged (mapv (fn [[id node]] [id (:operand (completion/canonical-parts node))])
                     (:object-bindings plan))
        input-obligations
        (mapv
         (fn [slot fact [_ object-node] [_ conversion-node]]
           (let [{:keys [parameter-class object-binder input-binder]} slot
                 ^Class target parameter-class
                 from (fact-dtype fact)
                 numeric? (.isPrimitive target)
                 numeric (when numeric?
                           (project-numeric-completion object-binder from target
                                                       {:ordinal (:ordinal slot) :input-fact fact}))]
             {:slot slot :input-fact fact :source-dtype from :target-dtype (:target-dtype numeric)
              :object-context object-node :conversion-context conversion-node
              :classification (cond numeric? (:classification numeric)
                                    (identical? Object target) :reference-identity
                                    :else :reference-check)
              :binding [input-binder (if numeric? (:candidate numeric) object-binder)] :status :pending}))
         (:slots plan) input-facts (:object-bindings plan) (:input-bindings plan))
        lookups (volatile! [])
        qualification-construction (volatile! [])
        qualified (inference/qualify-body-symbols
                   (:source proposal) source-ns (set parameters) #(vswap! lookups conj %)
                   (fn [original constructed bound]
                     (vswap! qualification-construction conj {:original original :constructed constructed
                                                               :bound bound})))
        alpha-construction (volatile! [])
        hygienic (util/alpha-convert
                  {} qualified
                  (fn [original constructed environment]
                    (vswap! alpha-construction conj {:original original :constructed constructed
                                                     :environment environment})))
        substitution-construction (volatile! [])
        projected (util/subst-syms
                   (:substitutions plan) hygienic (fn [substitutions id] (get substitutions id id))
                   (fn [original constructed substitutions]
                     (vswap! substitution-construction conj {:original original :constructed constructed
                                                              :substitutions substitutions})))
        binding-count (quot (count (second (:original proposal))) 2)
        output-count (count (last (:original proposal)))
        _ (doseq [spine [(:source proposal) qualified hygienic projected]]
            (validate-call-spine! spine binding-count output-count))
        [_ bindings outputs] projected
        body-bindings (mapv (fn [[id expression]] [(with-meta id nil) expression])
                            (partition 2 bindings))
        roots (mapv (fn [root-index [original-id original-expression] [candidate-id candidate-expression]
                         [id expression]]
                      {:root-index root-index
                       :call-binding-index (+ (count staged) (count input-obligations) root-index)
                       :original-id original-id :original-expression original-expression
                       :candidate-id candidate-id :candidate-expression candidate-expression
                       :id id :expression expression})
                    (range binding-count)
                    (partition 2 (second (:original proposal)))
                    (partition 2 (second (:source proposal))) body-bindings)
        used (volatile! (into #{} (filter symbol?)
                              (tree-seq coll? seq [projected staged arguments (mapv :binding input-obligations)])))
        fresh! (fn []
                 (loop []
                   (let [id (gensym "invocation_input__")]
                     (if (contains? @used id) (recur)
                         (do (vswap! used conj id) id)))))
        body
        (reduce
         (fn [{:keys [bindings] :as state} root]
           (let [first-index (+ (count staged) (count input-obligations) (count bindings))
                 construction (ordered-allocation-inputs physical root first-index fresh!)
                 arguments (:arguments construction)
                 expression (if construction (:expression construction) (:expression root))
                 root (cond-> (assoc root :call-binding-index (+ first-index (count arguments))
                                         :expression expression)
                        construction (assoc :argument-construction construction))]
             (-> state
                 (update :bindings into (map :binding arguments))
                 (update :bindings conj [(:id root) expression])
                 (update :roots conj root))))
         {:bindings [] :roots []} roots)
        roots (:roots body)
        body-bindings (:bindings body)
        ;; Declarations for the candidate normalizer, NOT completed input/body
        ;; facts. The unchanged pending obligations still own every conversion.
        root-requirements
        (into {} (keep (fn [{:keys [original-id id]}]
                         (when-let [dt (fact-dtype (get (:normal-return-facts proposal) original-id))]
                           [id dt]))) roots)
        root-requirements
        (reduce (fn [requirements {:keys [argument-construction]}]
                  (reduce (fn [requirements {:keys [binding completion]}]
                            (let [id (first binding)
                                  ^Class carrier (:parameter-carrier completion)]
                              (if-let [dt (dtype/dtype-for-jvm-array-class carrier)]
                                (assoc-in requirements [:array-types id] dt)
                                (assoc-in requirements [:scalar-types id]
                                          (dtype/dtype-for-scalar-tag (symbol (.getName carrier)))))))
                          requirements (:arguments argument-construction)))
                {:scalar-types root-requirements :array-types {}} roots)
        normalization-requirements
        (reduce (fn [requirements {:keys [slot source-dtype target-dtype]}]
                  (let [{:keys [object-binder input-binder parameter-class]} slot
                        array-dtype (dtype/dtype-for-jvm-array-class parameter-class)]
                    (cond-> requirements
                      source-dtype (assoc-in [:scalar-types object-binder] source-dtype)
                      target-dtype (assoc-in [:scalar-types input-binder] target-dtype)
                      array-dtype (update :array-types assoc object-binder array-dtype input-binder array-dtype))))
                root-requirements input-obligations)]
    {:proposal proposal :argument-plan plan :input-obligations input-obligations
     :qualified-source qualified :hygienic-source hygienic :projected-source projected
     :qualification-lookups @lookups :qualification-construction @qualification-construction
     :alpha-construction @alpha-construction
     :substitution-construction @substitution-construction
     :root-bindings roots
     :normalization-requirements normalization-requirements
     :bindings (into (into staged (map :binding input-obligations)) body-bindings)
     :outputs (mapv #(if (symbol? %) (with-meta % nil) %) outputs)
     :admitted? false}))

(defn- completion-replacement? [before after old-leaf new-leaf]
  (if (completion/canonical-form? before)
    (and (completion/canonical-form? after)
         (identical? (second before) (second after))
         (completion-replacement? (nth before 2) (nth after 2) old-leaf new-leaf))
    (and (identical? before old-leaf) (identical? after new-leaf))))

(defn ^:no-doc value-source-binding-layout
  "Revalidate the complete pending call topology against original invocation observations.
   Returns occurrence coordinates, never admission or successful completion evidence."
  [{:keys [proposal argument-plan input-obligations root-bindings bindings projected-source]}]
  (let [physical (:physical proposal)
        {:keys [emission input-body]} physical
        slots (:slots argument-plan)
        prefix (into (mapv (fn [slot] [(:object-binder slot) (:original-argument slot)]) slots)
                     (map :binding input-obligations))
        original-roots (vec (partition 2 (second (:original proposal))))
        projected-roots (vec (partition 2 (second projected-source)))
        occupied (into #{} (filter symbol?) (tree-seq coll? seq [projected-source prefix]))
        same-pair? (fn [a b] (and (= 2 (count a) (count b))
                                  (every? true? (map identical? a b))))
        expanded (into prefix
                       (mapcat (fn [root]
                                 (concat (map :binding (get-in root [:argument-construction :arguments]))
                                         [[(:id root) (:expression root)]])))
                       root-bindings)
        ids (mapv first expanded)]
    (when (and (vector? bindings) (vector? root-bindings)
               (identical? (:template proposal) (::value-source-template physical))
               (identical? (:original proposal) (get-in proposal [:template :original]))
               (= (count (:completion-obligations proposal))
                  (count (get-in proposal [:template :completion-obligations])))
               (every? true?
                       (map (fn [obligation original]
                              (and (identical? (:token obligation) (:token original))
                                   (identical? (:source obligation) (:source original))
                                   (identical? (:contract obligation) (:contract original))
                                   (= (:path obligation) (:path original))))
                            (:completion-obligations proposal)
                            (get-in proposal [:template :completion-obligations])))
               (= (count root-bindings) (count original-roots) (count projected-roots))
               (= (count expanded) (count bindings))
               (every? lexical-id? ids) (= (count ids) (count (set ids)))
               (every? true? (map same-pair? expanded bindings))
               (loop [remaining (seq root-bindings) next-index (count prefix) ordinal 0]
                 (if-let [root (first remaining)]
                   (let [construction (:argument-construction root)
                         arguments (:arguments construction)
                         expected-root-index (+ next-index (count arguments))
                         original-expression (second (nth original-roots ordinal))
                         projected-expression (second (nth projected-roots ordinal))
                         {original :expression path :path} (completion-leaf original-expression)
                         observed (when construction
                                    (bytecode/retained-input-completed-invocation-context
                                     emission input-body original))
                         ^java.lang.reflect.Method method (:method observed)
                         carriers (when (instance? java.lang.reflect.Method method)
                                    (vec (.getParameterTypes method)))
                         candidate (:candidate-call construction)
                         rebuilt (:constructed-call construction)
                         argument-valid?
                         (fn [ordinal argument]
                           (let [[id expression] (:binding argument)
                                 parts (completion/canonical-parts expression)
                                 input (nth (:inputs observed) ordinal nil)
                                 completed (nth (:input-completions observed) ordinal nil)]
                             (and parts (not (contains? occupied id))
                                  (= :pending (:status argument))
                                  (= ordinal (:input-index argument))
                                  (= (+ next-index ordinal) (:call-binding-index argument))
                                  (= (into [1 (inc (* 2 (:root-index root)))] (conj path (+ 2 ordinal)))
                                     (:template-source-path argument))
                                  (identical? (nth (vec (descriptor/call-args original)) ordinal nil)
                                              (:original-operand argument))
                                  (identical? input (:input argument))
                                  (identical? completed (:completion argument))
                                  (identical? (nth (vec (descriptor/call-args candidate)) ordinal nil)
                                              (:operand argument))
                                  (identical? (:operand argument) (:operand parts))
                                  (identical? (:contract argument) (:contract parts))
                                  (= (:contract argument)
                                     (completion/planned-bytecode-input-contract
                                      (nth carriers ordinal) (:context input) (:emitted-type completed)))
                                  (identical? id (nth (vec (descriptor/call-args rebuilt)) ordinal nil)))))
                         valid?
                         (and (= ordinal (:root-index root))
                              (= expected-root-index (:call-binding-index root))
                              (identical? original-expression (:original-expression root))
                              (if construction
                                (and observed (= :global-typed-indy (:invocation-mode observed))
                                     (identical? original (:source-input observed))
                                     (identical? original (get-in construction [:observed :source-input]))
                                     (= 2 (count arguments) (count carriers))
                                     (not (.isVarArgs method))
                                     (dtype/dtype-for-jvm-array-class (first carriers))
                                     (identical? Long/TYPE (second carriers))
                                     (identical? original (:source-call construction))
                                     (= path (:template-call-path construction))
                                     (identical? (:expression observed) (get-in construction [:observed :expression]))
                                     (identical? method (get-in construction [:observed :method]))
                                     (identical? (:inputs observed) (get-in construction [:observed :inputs]))
                                     (identical? (:input-completions observed)
                                                 (get-in construction [:observed :input-completions]))
                                     (identical? (:receiver observed) (get-in construction [:observed :receiver]))
                                     (= (:invocation-mode observed) (get-in construction [:observed :invocation-mode]))
                                     (identical? projected-expression (:input-expression construction))
                                     (identical? candidate (:expression (completion-leaf projected-expression)))
                                     (seq? rebuilt) (= 4 (count rebuilt))
                                     (identical? (first candidate) (first rebuilt))
                                     (identical? (second candidate) (second rebuilt))
                                     (identical? (second original) (:receiver observed))
                                     (identical? (second original) (second candidate))
                                     (completion-replacement? projected-expression (:expression root) candidate rebuilt)
                                     (every? true? (map-indexed argument-valid? arguments)))
                                (identical? projected-expression (:expression root))))]
                     (and valid? (recur (next remaining) (inc expected-root-index) (inc ordinal))))
                   (= next-index (count bindings)))))
      (let [indices
            (mapv (fn [{:keys [path]}]
               (let [[container position] path
                     index (when (and (= container 1) (integer? position) (pos? position) (odd? position))
                             (quot position 2))
                     root (when (and index (< index (count root-bindings))) (nth root-bindings index))
                     arguments (get-in root [:argument-construction :arguments])
                     matches (filterv (fn [{prefix :template-source-path}]
                                        (= prefix (vec (take (count prefix) path)))) arguments)]
                 (when (and root (<= (count matches) 1))
                   (if-let [argument (first matches)]
                     (:call-binding-index argument) (:call-binding-index root)))))
                  (:completion-obligations proposal))]
        (when (every? integer? indices)
          {:bindings expanded :body-obligation-indices indices})))))

(defn- reference-check [carrier value]
  (if (or (nil? carrier) (= Object carrier)) value
      (completion/complete-value (completion/reference-check-contract carrier) value)))

(def ^:private local-fact-keys [:type :hint :verified :reference-class])

(defn- checked-helper-body
  "Materialize actual lexical reference checks before hygiene or simplification."
  [emission receivers binding-contexts]
  (letfn [(walk [node]
            (cond
              (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
              (and (seq? node) (= 'let* (first node)))
              (let [[head bindings & tail] node
                    previous (volatile! nil)
                    observations (volatile! [])
                    pairs
                    (mapcat
                     (fn [index [id value]]
                       (let [observed (bytecode/retained-binding-context emission node index)]
                         (when-not observed
                           (throw (ex-info "Generated import has no owned lexical binding context"
                                           {:reason :generated-body-binding-context :binder id})))
                         (when (and (zero? index)
                                    (= 1 (count (:body emission)))
                                    (identical? node (first (:body emission)))
                                    (not= (:before observed)
                                          (:locals (bytecode/retained-body-context emission))))
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
                                 (bytecode/retained-terminal-context emission node))
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
              (let [observed (bytecode/retained-invocation-context emission node)
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
    (mapv walk (:body emission))))

(defn- qualified-helper-body [emission outer-ns]
  (let [receivers (volatile! {})
        binding-contexts (java.util.IdentityHashMap.)
        binding-facts (volatile! {})
        body (checked-helper-body emission receivers binding-contexts)
        parameters (into (set (:params emission)) (keys @receivers))
        used (volatile! (into parameters (filter symbol?) (tree-seq coll? seq body)))
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
          (let [canonical (entry/observed-body-operation context symbol #{})
                original (when canonical (ns-resolve (:source-ns emission) canonical))
                target (when canonical (ns-resolve outer-ns canonical))]
            (when-not (and (var? candidate) canonical
                           (identical? candidate original) (identical? candidate target))
              (decline! "Generated helper lookup has no namespace-independent historical meaning"
                        {:symbol symbol}))
            (entry/observe-source-resolution!
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
                             (let [canonical (entry/observed-body-operation context head local-names)]
                               (when-not canonical
                                 (decline! "Generated helper cast lacks its original lookup" {:symbol head}))
                               (with-meta (authenticate! head (ns-resolve (:source-ns emission) canonical)) (meta head)))
                             head)]
                  (util/remake-if-changed node (into [head] (map walk (rest node)))))
                (vector? node) (util/remake-if-changed node (mapv walk node))
                :else node))]
      {:body (mapv #(util/subst-syms @receivers (walk %)) qualified)
       :binding-facts @binding-facts})))

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

(defn- instantiate-helper
  [expression locals context capture!]
  (let [[_ receiver & arguments] expression
        invocation (bytecode/typed-invocation-plan receiver (count arguments) locals context)]
    (when (and (:global? invocation)
               (not (:lexical-callee (descriptor/call-description expression))))
      (when-let [inputs (bytecode/plan-resolved-invocation-inputs
                        (:method invocation) (vec arguments) locals context)]
        (let [{:keys [semantic-op implementation-op]} (descriptor/call-description expression)
              owner (capture! semantic-op implementation-op (count arguments) (:method invocation))
              emission (:emission owner)
              source (get-in owner [:signature :definition :source-var])
              frontend (when source (entry/original-source-frontend source))]
          (when (and emission frontend (identical? emission (:emission frontend)))
            (when-let [bridge (or (entry/frontend-primitive-body-return-bridge frontend)
                                 (entry/frontend-reference-return-bridge frontend))]
              (when-not (get-in bridge [:body-context :unretained-sibling-calls?])
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
                      instantiated (util/subst-syms (zipmap (:params emission) values) body)
                      saved (:body-context bridge)
                      result (completion/complete-value
                              (completion/bytecode-result-context-contract
                               (:result-context saved) (:emitted-result-type saved)) instantiated)
                      result (if (= :reference-static-return (:kind bridge))
                               (reference-check (:body-return-carrier bridge) result)
                               (completion/complete-value completion/object-input-contract result))
                      [body-bindings result] (expose-anchored-result-spine result binding-facts)
                      result (vary-meta result assoc :raster.type/tag (:return-tag emission))]
                  ;; All operands, including unused ones, are completed once in
                  ;; positional order before the exact helper body. There is no
                  ;; synthesized receiver term for the global args-first path.
                  {:expression (list 'let* (into bindings body-bindings) result)
                   :binding-facts binding-facts})))))))))

(defn- generated-interface-method? [program method]
  (when (and (instance? java.lang.reflect.Method method) (vector? (:tags program))
             (= (count (:params program)) (count (:tags program))))
    (let [interface (:iface-class (types/ensure-fn-interface! (:tags program)))
          candidates (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                             (.getDeclaredMethods ^Class interface))]
      (and (= 1 (count candidates)) (= method (first candidates))
           (.isInterface ^Class (.getDeclaringClass ^java.lang.reflect.Method method))
           (not (java.lang.reflect.Modifier/isStatic (.getModifiers ^java.lang.reflect.Method method)))
           (not (.isVarArgs ^java.lang.reflect.Method method))
           (identical? Object (.getReturnType ^java.lang.reflect.Method method))))))

(defn normalize-program
  "Expand a new program's closed top-level typed-helper binding spine once.
   `method` is its real interface Method, not an inferred signature. `capture!`
   is the existing owned, non-evaluating selected-emission producer. Successful
   frontend observers retain helper module history in the enclosing collector.
   Unsupported binding spines abstain; dependency changes still invalidate the
   entire producer attempt. Nested helper calls remain for ordinary lowering;
   their presence is NOT a claim that source or device admission is closed."
  [program method capture!]
  (try
   (entry/call-with-optional-definition-inputs
   (fn []
     (when entry/*emission-inputs*
       (let [{:keys [params source-ns walked-body]} program
             [head bindings outputs :as body] (when (= 1 (count walked-body)) (first walked-body))
             locals (when (generated-interface-method? program method)
                      (bytecode/resolved-parameter-locals method params))
             context {:source-ns source-ns :object-result? true}
             plan (when (and locals (= 'let* head) (= 3 (count body)))
                    (bytecode/plan-flat-bindings bindings locals context))]
         (when plan
           (let [imported-facts (volatile! {})
                 expanded
                 (mapv
                  (fn [{:keys [binder initializer before completion after]}]
                    (let [eligible? (and (seq? initializer) (= '.invk (first initializer)))
                          imported (when eligible?
                                     (instantiate-helper initializer before
                                                         (bytecode/binding-initializer-context context)
                                                         capture!))
                          [_ helper-bindings tail] (:expression imported)
                          value (if imported tail initializer)
                          binder (if-let [hint (:hint after)] (vary-meta binder assoc :tag hint) binder)]
                      (when (or (not eligible?) imported)
                        (vswap! imported-facts merge (:binding-facts imported)
                                {binder (select-keys after local-fact-keys)})
                        (into (vec helper-bindings)
                              [binder (reference-check (:checkcast-class completion) value)]))))
                  (:bindings plan))]
             ;; An unsupported eligible call invalidates the requested whole
             ;; top-level interval; a partially expanded body is not success.
             (when (every? some? expanded)
               (let [bindings (vec (mapcat identity expanded))
                     ids (vec (take-nth 2 bindings))]
                 (when-not (= (count ids) (count (set ids)))
                   (decline! "Generated helper bindings collide with their caller" {}))
                 (with-value-source-template
                   (assoc program ::entry-method method ::binding-facts @imported-facts :walked-body
                          [(with-meta (list head bindings outputs) (meta body))]))))))))))
   (catch clojure.lang.ExceptionInfo error
     (if (= :generated-body-unsupported (:reason (ex-data error))) nil (throw error)))))

(defn validate-program-emission!
  "Check construction-time local expectations against the sole actual outer
   emission before a generated owner is published. Expectations are not carrier
   authority: only these retained observations can discharge the construction."
  [program emission]
  (when-let [expected (::binding-facts program)]
    (let [input (:walked-body program)
          seen (volatile! #{})]
      (letfn [(walk [node]
                (cond
                  (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) nil
                  (and (seq? node) (= 'let* (first node)))
                  (do
                    (doseq [[index [id initializer]] (map-indexed vector (partition 2 (second node)))]
                      (when-let [facts (get expected id)]
                        (let [observed (bytecode/retained-input-binding-context emission input node index)]
                          (when-not (= facts (select-keys (:after observed) local-fact-keys))
                            (throw (ex-info "Generated binding changed its original physical local facts"
                                            {:reason :generated-body-local-facts :binder id
                                             :expected facts
                                             :actual (select-keys (:after observed) local-fact-keys)})))
                          (when (contains? @seen id)
                            (throw (ex-info "Generated binding expectation is ambiguous"
                                            {:reason :generated-body-local-facts :binder id})))
                          (vswap! seen conj id)))
                      (walk initializer))
                    (doseq [tail (nnext node)] (walk tail)))
                  (completion/canonical-form? node)
                  (let [observed (bytecode/retained-input-completion-context emission input node)
                        contract (second node)]
                    (when-not (and observed
                                   (= contract (:contract observed))
                                   (or (not= :jvm-bytecode-result-context (:source-context contract))
                                       (and (= (:operand-context contract) (:operand-context observed))
                                            (= (:result-carrier contract) (:completion-carrier observed))))
                                   (= (:returned-carrier observed)
                                      (if (:object-parent? observed)
                                        (if (= :diverge (:completion-carrier observed)) :diverge :ref)
                                        (:completion-carrier observed))))
                      (throw (ex-info "Generated completion lacks its exact emitted context"
                                      {:reason :generated-body-completion-context
                                       :source-context (:source-context contract)})))
                    (walk (nth node 2)))
                  (coll? node) (doseq [child node] (walk child))))]
        (doseq [body input] (walk body)))
      (when-not (= @seen (set (keys expected)))
        (throw (ex-info "Generated emission lost a required binding interval"
                        {:reason :generated-body-local-facts
                         :missing (vec (remove @seen (keys expected)))})))))
  program)

(ns raster.compiler.backend.jvm.generated-body
  "Construct a new generated program from exact owned helper bodies.
   This is not an original-source occurrence relation or GPU admission proof.
   The caller owns the resulting program and must retain its collected inputs.
   No source frontend, numerical invocation or phantom caller module is created."
  (:require [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.callee-identity :as callee]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.form :as form]))

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
                         ;; A distinct existing contract term identifies this
                         ;; actual snapshot constructor across lexical hygiene.
                         ;; Its data/semantics remain ordinary Object completion.
                         :object-contract (into {} completion/object-input-contract)
                         :contract (completion/bytecode-boxed-parameter-contract carrier)})
                      (range) parameters arguments carriers)
          objects (mapv (fn [{:keys [object-binder original-argument object-contract]}]
                          [object-binder (completion/complete-value object-contract
                                                                    original-argument)]) slots)
          inputs (mapv (fn [{:keys [input-binder object-binder contract]}]
                         [input-binder (completion/complete-value contract object-binder)]) slots)]
      {:slots slots :object-bindings objects :input-bindings inputs
       :bindings (into objects inputs)
       :substitutions (into {} (map (juxt :formal :input-binder)) slots)
       :boxed-entry-bridge bridge})))

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

(defn- instantiate-helper
  [expression locals context capture! consume-terminal-reduction terminal-only?]
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
            (let [bridge (when-not terminal-only?
                           (or (entry/frontend-primitive-body-return-bridge frontend)
                               (entry/frontend-reference-return-bridge frontend)))
                  ;; Authenticate this third, weaker return contract BEFORE
                  ;; hygiene. It is useful only to an actual consuming owner;
                  ;; ordinary JVM/C expansion keeps its existing abstention.
                  terminal-return (when (and (nil? bridge) consume-terminal-reduction)
                                    (optional-terminal-return frontend))
                  saved (or (:body-context bridge)
                            (when terminal-return
                              (bytecode/retained-input-body-context emission (:input-body terminal-return))))]
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
                      instantiated (util/subst-syms (zipmap (:params emission) values) body)
                      result (when bridge (completion/complete-value
                              (completion/bytecode-result-context-contract
                               (:result-context saved) (:emitted-result-type saved)) instantiated))
                      result (when bridge (if (= :reference-static-return (:kind bridge))
                               (reference-check (:body-return-carrier bridge) result)
                               (completion/complete-value completion/object-input-contract result)))
                      [body-bindings result] (when bridge (expose-anchored-result-spine result binding-facts))
                      result (when bridge (vary-meta result assoc :raster.type/tag (:return-tag emission)))]
                  ;; All operands, including unused ones, are completed once in
                  ;; positional order before the exact helper body. There is no
                  ;; synthesized receiver term for the global args-first path.
                  (if bridge
                    {:expression (list 'let* (into bindings body-bindings) result)
                     :binding-facts binding-facts}
                    (consume-terminal-return
                     consume-terminal-reduction
                     {:return-contract terminal-return :frontend frontend :invocation invocation
                      :inputs inputs :expression expression :locals locals :context context
                      :argument-bindings bindings
                      :input-values values :instantiated-body instantiated
                      :saved-return-context saved :binding-facts binding-facts})))))))))))

(defn- optional-terminal-helper
  [construct!]
  (try
    ;; Refused captures are not dependencies of a source program that keeps
    ;; the original call. A fresh child also prevents inherited history from
    ;; becoming the candidate's own preparation/receipt history.
    (entry/call-with-optional-definition-inputs
     #(entry/call-with-independent-definition-inputs construct!))
    (catch clojure.lang.ExceptionInfo error
      (if (= :generated-body-unsupported (:reason (ex-data error))) nil (throw error)))))

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

(defn- normalize-helper-program
  "Expand a new program's closed top-level typed-helper binding spine once.
   `method` is its real interface Method, not an inferred signature. `capture!`
   is the existing owned, non-evaluating selected-emission producer. Successful
   frontend observers retain helper module history in the enclosing collector.
   A successful result retains its exact input program as a construction edge;
   that edge has no authority outside the enclosing preparation owner.
   Unsupported binding spines abstain; dependency changes still invalidate the
   entire producer attempt. Nested helper calls remain for ordinary lowering;
   their presence is NOT a claim that source or device admission is closed."
  [program method capture! consume-terminal-reduction terminal-only?]
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
                                     (let [construct! #(instantiate-helper
                                                        initializer before
                                                        (bytecode/binding-initializer-context context)
                                                        capture! consume-terminal-reduction terminal-only?)]
                                       (if terminal-only? (optional-terminal-helper construct!) (construct!))))
                          [_ helper-bindings tail] (:expression imported)
                          value (if imported tail initializer)
                          unchanged? (and terminal-only? (nil? imported))
                          binder (if unchanged? binder
                                     (if-let [hint (:hint after)] (vary-meta binder assoc :tag hint) binder))]
                      (when (or terminal-only? (not eligible?) imported)
                        ;; The planner's inferred hint is not an observation
                        ;; of an unchanged binding, whose literal declaration
                        ;; was deliberately not strengthened above.
                        (when-not unchanged?
                          (vswap! imported-facts merge (:binding-facts imported)
                                  {binder (select-keys after local-fact-keys)}))
                        (if unchanged?
                          ;; Its original declaration still owns any implicit
                          ;; binding CHECKCAST. Do not hide an ordinary map/helper
                          ;; call inside a newly synthesized completion wrapper.
                          [binder initializer]
                          (into (vec helper-bindings)
                                [binder (reference-check (:checkcast-class completion) value)])))))
                  (:bindings plan))]
             ;; Ordinary expansion still requires a complete interval. The
             ;; targeted mode intentionally retains other exact source calls;
             ;; it grants no import or fallback authority for those calls.
             (when (every? some? expanded)
               (let [bindings (vec (mapcat identity expanded))
                     ids (vec (take-nth 2 bindings))]
                 (when-not (= (count ids) (count (set ids)))
                   (decline! "Generated helper bindings collide with their caller" {}))
                 (assoc program ::normalization-input program
                        ::entry-method method ::binding-facts @imported-facts :walked-body
                        [(with-meta (list head bindings outputs) (meta body))])))))))))
   (catch clojure.lang.ExceptionInfo error
     (if (= :generated-body-unsupported (:reason (ex-data error))) nil (throw error)))))

(defn- independent-code-owner? [node]
  (and (seq? node)
       (or (= :lambda (:kind (form/form-info node)))
           (contains? '#{reify* deftype* var} (first node)))))

(defn- canonical-source-casts
  "Construct source completions before emission, under the actual producer policy.
   Scope structure comes from the common form algebra. Independent code owners,
   quoted data and already-owned completions are opaque. This grants no payload,
   portability, movement or derivative evidence."
  [{:keys [walked-body params source-ns source-policy] :as program}]
  (if-not (contains? program :source-policy)
    {:body walked-body :changed? false}
    (do
      (source-policy/unchecked-math source-policy)
      (let [changed? (volatile! false)]
       (letfn [(walk [node bound]
                (cond
                  (and (seq? node)
                       (or (contains? '#{quote clojure.core/quote}
                                      (first node))
                           (independent-code-owner? node)
                           (completion/canonical-form? node))) node
                  (seq? node)
                  (if-let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info node)]
                    (let [scope! (fn [{:keys [binders inits body aux] :as scope}]
                                   (let [[inner inits]
                                         (if sequential?
                                           (reduce (fn [[bound result] [binder init]]
                                                     [(cond-> bound (symbol? binder) (conj binder))
                                                      (conj result (walk init bound))])
                                                   [bound []] (map vector binders inits))
                                           (let [inner (into bound (filter symbol? binders))]
                                             [inner (mapv #(walk % (if rec? inner bound)) inits)]))]
                                     (cond-> (assoc scope :inits inits :body (mapv #(walk % inner) body))
                                       aux (assoc :aux (mapv #(walk % inner) aux)))))]
                      (rebuild (mapv scope! scopes) (mapv #(walk % bound) outer)))
                    (let [resolved (callee/fresh-resolved-callee (first node) source-ns bound)
                          operation (case resolved
                                      clojure.core/float :float
                                      clojure.core/unchecked-float :unchecked-float nil)]
                      (if (and operation (= 2 (count node)))
                        (do
                          (intrinsic-source/observe-original! source-ns (first node) resolved)
                          (vreset! changed? true)
                          (completion/complete-value
                           (completion/source-floating-cast-contract operation source-policy)
                           (walk (second node) bound)))
                        (util/remake-if-changed node (mapv #(walk % bound) node)))))
                  (vector? node) (util/remake-if-changed node (mapv #(walk % bound) node))
                  (record? node) node
                  (map? node) (util/map-preserving-identity #(walk % bound) node)
                  :else node))]
         (let [body (util/remake-if-changed walked-body (mapv #(walk % (set params)) walked-body))]
           {:body body :changed? @changed?}))))))

(defn- normalize-program*
  [program method capture! options]
  (let [consume-terminal-reduction (:consume-terminal-reduction options)
        terminal-only? (true? (:terminal-only? options))]
   (when-not (and (map? options) (every? #{:consume-terminal-reduction :terminal-only?} (keys options))
                  (or (not (contains? options :terminal-only?)) (instance? Boolean (:terminal-only? options)))
                  (or (nil? consume-terminal-reduction) (fn? consume-terminal-reduction))
                  (or (not terminal-only?) (fn? consume-terminal-reduction)))
     (decline! "Generated terminal construction requires an actual consumer function" {}))
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (generated-interface-method? program method))
       (let [{:keys [body changed?]} (canonical-source-casts program)
             canonical (if changed? (assoc program :walked-body body) program)
             expanded (normalize-helper-program canonical method capture! consume-terminal-reduction terminal-only?)]
         (cond
           expanded (assoc expanded ::normalization-input program)
           changed? (assoc canonical ::normalization-input program
                           ::entry-method method ::binding-facts {})
           :else nil)))))))

(defn normalize-program
  "Construct the generated JVM adapter from its exact numerical producer.
   Source casts become canonical completions under explicit producer policy even
   when helper expansion abstains. Neither construction admits device execution.
   Optional :consume-terminal-reduction is a constructing function receiving an
   authenticated owner-local return contract and instantiated source. It must
   return complete :expression/:binding-facts, preserving the supplied argument
   binding prefix. :terminal-only? true leaves all other helpers unchanged for
   ordinary downstream inlining. No default helper permission is enabled."
  ([program method capture!] (normalize-program* program method capture! {}))
  ([program method capture! options] (normalize-program* program method capture! options)))

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
                  (independent-code-owner? node) nil
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

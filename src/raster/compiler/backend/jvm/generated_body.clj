(ns raster.compiler.backend.jvm.generated-body
  "Construct a new generated program from exact owned helper bodies.
   This is not an original-source occurrence relation or GPU admission proof.
   The caller owns the resulting program and must retain its collected inputs.
   No source frontend, numerical invocation or phantom caller module is created."
  (:require [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]))

(defn- decline! [message data]
  (throw (ex-info message (assoc data :reason :generated-body-unsupported))))

(defn- reference-check [carrier value]
  (if (or (nil? carrier) (= Object carrier)) value
      (completion/complete-value (completion/reference-check-contract carrier) value)))

(defn- checked-helper-body
  "Materialize actual lexical reference checks before hygiene or simplification."
  [emission receivers]
  (letfn [(walk [node]
            (cond
              (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))) node
              (and (seq? node) (= 'let* (first node)))
              (let [[head bindings & tail] node
                    previous (volatile! nil)
                    pairs
                    (mapcat
                     (fn [index [id value]]
                       (let [observed (bytecode/retained-binding-context emission node index)]
                         (when-not observed
                           (throw (ex-info "Generated import has no owned lexical binding context"
                                           {:reason :generated-body-binding-context :binder id})))
                         (when (and @previous (not= @previous (:before observed)))
                           (throw (ex-info "Generated import has a discontinuous lexical binding chain"
                                           {:reason :generated-body-binding-context :binder id})))
                         (vreset! previous (assoc (:before observed) (:binder observed) (:after observed)))
                         [id (if (= :ref (get-in observed [:after :type]))
                               (reference-check (get-in observed [:after :reference-class]) (walk value))
                               (walk value))]))
                     (range) (partition 2 bindings))]
                (with-meta (list* head (vec pairs) (mapv walk tail)) (meta node)))
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
        body (checked-helper-body emission receivers)
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
                   (mapv util/alpha-convert body))
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
      (mapv #(util/subst-syms @receivers (walk %)) qualified))))

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
                (let [body (qualified-helper-body emission (:source-ns context))
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
                      result (vary-meta result assoc :raster.type/tag (:return-tag emission))]
                  ;; All operands, including unused ones, are completed once in
                  ;; positional order before the exact helper body. There is no
                  ;; synthesized receiver term for the global args-first path.
                  (list 'let* bindings result))))))))))

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
           (let [expanded
                 (mapv
                  (fn [{:keys [binder initializer before completion]}]
                    (let [value (if (and (seq? initializer) (= '.invk (first initializer)))
                                  (instantiate-helper initializer before
                                                      (bytecode/binding-initializer-context context)
                                                      capture!)
                                  initializer)]
                      (when (or (some? value) (nil? initializer))
                        [binder (reference-check (:checkcast-class completion) value)])))
                  (:bindings plan))]
             ;; An unsupported eligible call invalidates the requested whole
             ;; top-level interval; a partially expanded body is not success.
             (when (every? some? expanded)
               (assoc program ::entry-method method :walked-body
                      [(with-meta (list head (vec (mapcat identity expanded)) outputs) (meta body))]))))))))
   (catch clojure.lang.ExceptionInfo error
     (if (= :generated-body-unsupported (:reason (ex-data error))) nil (throw error)))))

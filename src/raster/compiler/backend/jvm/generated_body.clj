(ns raster.compiler.backend.jvm.generated-body
  "Construct a new generated program from exact owned helper bodies.
   This is not an original-source occurrence relation or GPU admission proof.
   The caller owns the resulting program and must retain its collected inputs.
   No source frontend, numerical invocation or phantom caller module is created."
  (:require [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.callee-identity :as callee]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
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

(defn- instantiate-helper
  [expression locals context capture! consume-terminal-reduction terminal-only?]
  ((requiring-resolve 'raster.compiler.passes.scalar.source-import/instantiate-emitted-helper)
   expression locals context capture! consume-terminal-reduction terminal-only?))

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

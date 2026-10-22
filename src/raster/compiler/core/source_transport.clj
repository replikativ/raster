(ns raster.compiler.core.source-transport
  "Owner-local source transformations, separate from source execution admission.
   Preserve authentic helper inputs at the transformation that consumes them;
   never reconstruct their origins from later symbol names or equal expressions."
  (:require [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(defn effective-body
  "Existing multi-form helper convention: a single ordered do expression."
  [body]
  (if (> (count body) 1) [(apply list 'do body)] body))

(defn- with-local-history [construct]
  (let [parent entry/*emission-inputs* historical @parent
        local (atom {:static [] :values []})
        _ (entry/assert-definition-inputs-current! historical)
        result (binding [entry/*emission-inputs* local] (construct))]
    (when-not (identical? historical @parent)
      (throw (ex-info "source transformation's enclosing owner changed during construction"
                      {:reason :unstable-definition-proof-input})))
    (entry/assert-definition-inputs-current! historical)
    (entry/assert-definition-inputs-current! @local)
    (entry/adopt-definition-inputs! @local)
    result))

(defn- with-hygienic-renamings [body parameters transform]
  ;; Shared by helper alpha conversion and source uniquification. A trace alone
  ;; cannot make a capturing or metadata-changing binder allocator trustworthy.
  (let [used (volatile! (into (set parameters) (mapcat util/free-syms) body))
        events (volatile! []) allocate util/*fresh-binder*
        result (binding [util/*fresh-binder*
                         (fn [original]
                           (let [fresh (allocate original)]
                             (when-not (and (symbol? fresh) (nil? (namespace fresh))
                                            (not (contains? @used fresh))
                                            (identical? (meta original) (meta fresh)))
                               (throw (ex-info "source transformation requires fresh metadata-preserving binders"
                                               {:reason :nonfresh-source-import-binder})))
                             (vswap! used conj fresh)
                             (vswap! events conj [original fresh])
                             fresh))]
                 (transform))]
    {:body result :renamings @events}))

(deftype ^:private BodyImport
  [frontend parameters source-ns effective hygienic qualified tags renamings
   alpha-occurrences qualification-occurrences observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when-not (identical? source-ns (find-ns (ns-name source-ns)))
      (throw (ex-info "imported source namespace was replaced" {:reason :stale-source-import-namespace})))
    (entry/assert-definition-inputs-current! observations)
    true))

(defn- qualify-tags [tags source-ns]
  (if-not source-ns tags
  (when tags
    (mapv (fn [tag]
            (if (or (nil? tag)
                     (contains? '#{double long float int Double Long Float Integer
                                   doubles floats longs ints objects} tag)
                     (.contains (str tag) "."))
              tag
              (let [lookup (symbol (str tag))
                    result (try (ns-resolve source-ns lookup) (catch Exception _ nil))]
                (entry/observe-source-resolution!
                 {:kind :symbol-resolution :source-ns source-ns :symbol lookup :result result})
                (if (class? result) (symbol (.getName ^Class result)) tag)))) tags))))

(declare completed-reference-body-input)

(defn- prepare-body [frontend body parameters tags source-ns owned? & [completed-call]]
  (let [_ (when frontend
            (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
            (when-not (and (identical? body (:body frontend))
                           (= parameters (get-in frontend [:source-entry :params]))
                           (= tags (get-in frontend [:source-entry :tags]))
                           (identical? source-ns (get-in frontend [:source-entry :source-ns])))
              (throw (ex-info "helper transformation differs from its original input"
                              {:reason :source-import-input-mismatch}))))
        _ (when owned?
            (entry/observe-source-body-input! (get-in frontend [:source-entry :source-var]) :var)
            (entry/observe-frontend-source-identity! frontend))
        completed (when (and owned? completed-call)
                    (completed-reference-body-input frontend completed-call))
        _ (when (and completed-call (nil? completed))
            (throw (ex-info "helper import requires its exact unique completed call"
                            {:reason :source-import-completed-call-mismatch})))
        qualified-tags (qualify-tags tags source-ns)
        effective (effective-body (if completed (:body completed) body))
        alpha-occurrences (when owned? (volatile! []))
        qualification-occurrences (when owned? (volatile! []))
        observe-call (fn [events origins]
                       (when events
                         (fn [original constructed _]
                           (let [origin (if origins (.get ^java.util.IdentityHashMap origins original)
                                            {:original original})]
                             (when (and origin
                                        (or (form/call-form? original)
                                            (and completed (seq? original)
                                                 (not (contains? '#{quote clojure.core/quote} (first original))))))
                               (vswap! events conj (assoc origin :constructed constructed)))))))
        renamed (when owned?
                  (with-hygienic-renamings effective parameters
                    #(mapv (fn [expression]
                             (util/alpha-convert {} expression
                                                 (observe-call alpha-occurrences (:origins completed)))) effective)))
        hygienic (if owned? (:body renamed) (mapv util/alpha-convert effective))
        parameter-set (set (map #(if (symbol? %) % (symbol (name %))) parameters))
        qualified (if source-ns
                    (mapv #(inference/qualify-body-symbols % source-ns parameter-set
                                                         entry/observe-source-resolution!
                                                         (observe-call qualification-occurrences nil)) hygienic)
                    hygienic)
        input (when owned?
                (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
                (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
                (BodyImport. frontend parameters source-ns effective hygienic qualified
                             qualified-tags (:renamings renamed)
                             @alpha-occurrences @qualification-occurrences @entry/*emission-inputs*))]
    {:walked-body qualified :tags qualified-tags :source-import input}))

(defn prepare-inline-body
  "Perform the inliner's existing alpha-renaming and qualification once.
   An authentic frontend and active owner additionally retain an immutable input
   witness with IMPORT-LOCAL history. Legacy bodies acquire no owner. No formal
   argument completion, call substitution, final-step correspondence, normal
   return or permission to execute the imported body is inferred. An optional
   owned completed call is spliced at its original occurrence before hygiene;
   its authority comes from that constructor, not from import."
  [frontend body parameters tags source-ns & [completed-call]]
  (if (and frontend entry/*emission-inputs*)
    (with-local-history #(prepare-body frontend body parameters tags source-ns true completed-call))
    (prepare-body frontend body parameters tags source-ns false completed-call)))

(defn body-import-view
  "Read an exact constructed body under its original history in a collecting owner.
   Wrong/copy/ownerless inputs abstain; stale original inputs throw. Successful
   queries adopt the original observations, never refresh them. This is only a
   body-import relation: call arguments and later transformations remain unproved."
  [input body]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? BodyImport input)
                (identical? body (.-qualified ^BodyImport input)))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^BodyImport input))
       {:frontend (.-frontend ^BodyImport input) :parameters (.-parameters ^BodyImport input)
        :tags (.-tags ^BodyImport input)
        :effective-body (.-effective ^BodyImport input) :hygienic-body (.-hygienic ^BodyImport input)
        :walked-body body :renamings (.-renamings ^BodyImport input)}))))

(defn ^:no-doc unique-construction-edge
  "Identity-only uniqueness check. Caller-supplied edges are not authority."
  [occurrences constructed]
  (let [matches (filterv #(identical? constructed (:constructed %)) occurrences)]
    (when (and (= 1 (count matches))
               (= 1 (count (filter #(identical? (:original (first matches)) (:original %)) occurrences))))
      (first matches))))

(defn body-import-occurrence
  "Follow the exact import's phase-indexed call constructors to ORIGINAL source.
   Unchanged occurrences are recorded too; no form or name matching/reconstruction
   is performed here. Copies, shared/repeated occurrences and ownerless input
   abstain transactionally. This proves origin only, not transformation equivalence."
  [input body constructed]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [imported (body-import-view input body)]
       (let [^BodyImport input input]
         (when-let [qualified (unique-construction-edge (.-qualification-occurrences input) constructed)]
           (when-let [renamed (unique-construction-edge (.-alpha-occurrences input) (:original qualified))]
             (cond-> {:frontend (:frontend imported) :original (:original renamed)
                      :hygienic (:constructed renamed) :constructed constructed
                      :alpha-construction renamed :qualification-construction qualified}
               (:completed-call renamed) (assoc :completed-call (:completed-call renamed))))))))))

(deftype ^:private SourceNormalization [frontend body occurrences renamings observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (entry/assert-definition-inputs-current! observations)
    true))

(defn normalize-frontend-bindings
  "Uniquify the exact authenticated frontend, retaining actual construction edges.
   The witness is producer-local: do not put it in compiler options, IR metadata
   or portable artifacts. An optional opaque completed call carries its own
   constructor authority; normalization alone proves no completion or execution."
  ([frontend parameters] (normalize-frontend-bindings frontend parameters nil))
  ([frontend parameters completed-call]
  (when (and frontend entry/*emission-inputs*)
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when-not (= parameters (get-in frontend [:source-entry :params]))
      (throw (ex-info "source normalization differs from its original parameters"
                      {:reason :source-normalization-input-mismatch})))
    (with-local-history
      (fn []
        (entry/observe-frontend-source-identity! frontend)
        (let [completed (when completed-call (completed-reference-body-input frontend completed-call))
              _ (when (and completed-call (nil? completed))
                  (throw (ex-info "source normalization requires its exact unique completed call"
                                  {:reason :source-normalization-completed-call-mismatch})))
              input-body (if completed (:body completed) (:body frontend))
              parameters (mapv #(if (symbol? %) (symbol (name %)) %) parameters)
              occurrences (volatile! [])
              renamed (with-hygienic-renamings
                        input-body parameters
                        #(mapv (fn [ordinal expression]
                                 (util/uniquify-rebindings
                                  parameters expression
                                  (fn [original constructed renaming bound]
                                    (let [origin (if completed
                                                   (.get ^java.util.IdentityHashMap (:origins completed) original)
                                                   {:original original})]
                                      (when (and origin (seq? original)
                                                 (not (contains? '#{quote clojure.core/quote}
                                                                 (first original))))
                                        (vswap! occurrences conj
                                                (assoc origin :constructed constructed
                                                       :root ordinal :renaming renaming :bound bound)))))))
                               (range) input-body))
              body (:body renamed)]
          {:walked-body body
           :source-normalization (SourceNormalization. frontend body @occurrences
                                                       (:renamings renamed) @entry/*emission-inputs*)}))))))

(defn normalization-occurrence
  "Query a unique exact constructed occurrence under its original history.
   Copied, repeated-shared or ownerless nodes abstain. No transformation or source
   search is rerun; successful adoption retains the original source identity guard."
  [input constructed]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceNormalization input))
       (let [^SourceNormalization input input]
         (when-let [occurrence (unique-construction-edge (.-occurrences input) constructed)]
           (entry/assert-producer-current! input)
           (entry/adopt-definition-inputs! (.-observations input))
           (assoc occurrence :frontend (.-frontend input))))))))

(def ^:dynamic ^:private *source-construction-observer* nil)

(defn observe-type-elaboration!
  "Observe an actual constructor return, only inside the owner-local source phase.
   This retains origins, not equivalence, types, completion or execution evidence."
  [original constructed environment]
  (when (and *source-construction-observer* (not (identical? original constructed)))
    (*source-construction-observer*
     :type-elaboration {:original original :constructed constructed :environment environment})))

(defn observe-builder-expansion!
  "Retain the shared builder's actual inputs/output. This is NOT the final expanded AST."
  [caller callee arguments environment result-symbol expansion]
  (when *source-construction-observer*
    (*source-construction-observer*
     :builder-expansion {:caller caller :source-import (:source-import callee)
                         :walked-body (:walked-body callee) :arguments arguments
                         :environment environment :result-symbol result-symbol
                         :expansion expansion})))

(deftype ^:private SourceConstruction [normalization result edges builders observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-producer-current! normalization)
    (entry/assert-definition-inputs-current! observations)
    true))

(defn with-source-construction-history
  "Run the actual source phase once, retaining its owner-local construction returns.
   No witness is emitted for ownerless input. Nothing enters compiler options, AST
   metadata or portable IR. Successful callee imports join this same local history."
  [normalization construct]
  (if-not (and entry/*emission-inputs* (instance? SourceNormalization normalization))
    {:form (construct)}
    (with-local-history
      (fn []
        (entry/assert-producer-current! normalization)
        (entry/adopt-definition-inputs! (.-observations ^SourceNormalization normalization))
        (let [edges (volatile! []) builders (volatile! [])
              result (binding [*source-construction-observer*
                               (fn [kind event]
                                 (case kind
                                   :type-elaboration (vswap! edges conj event)
                                   :builder-expansion (vswap! builders conj event)))]
                       (construct))]
          {:form result
           :source-construction
           (SourceConstruction. normalization result @edges @builders @entry/*emission-inputs*)})))))

(defn source-construction-view
  "Read only the exact completed phase under its original history.
   Builder observations describe construction, not final placement or execution."
  [input normalization result]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceConstruction input)
                (identical? normalization (.-normalization ^SourceConstruction input))
                (identical? result (.-result ^SourceConstruction input)))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^SourceConstruction input))
       {:type-elaborations (.-edges ^SourceConstruction input)
        :builder-expansions (.-builders ^SourceConstruction input)}))))

(defn construction-occurrence
  "Follow unique actual construction edges back to normalization. Never search by equality.
   A changed tag proves origin only; copied/shared/branching/cyclic histories abstain."
  [input constructed]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceConstruction input))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^SourceConstruction input))
       (let [^SourceConstruction input input edges (.-edges input)
             visited (java.util.IdentityHashMap.)]
         (loop [value constructed chain [] remaining (inc (count edges))]
           (when (and (pos? remaining) (not (.containsKey visited value)))
             (.put visited value true)
             (let [incoming (filterv #(identical? value (:constructed %)) edges)]
               (cond
                 (empty? incoming)
                 (when-let [origin (normalization-occurrence (.-normalization input) value)]
                   (assoc origin :normalized value :constructed constructed
                                 :construction-edges (vec (reverse chain))))

                 (= 1 (count incoming))
                 (let [edge (first incoming)
                       outgoing (filterv #(identical? (:original edge) (:original %)) edges)]
                   (when (= 1 (count outgoing))
                     (recur (:original edge) (conj chain edge) (dec remaining)))))))))))))

(defn builder-construction-view
  "Read one exact recorded builder return and its authenticated imported body.
   This also covers import-local callers and rewritten arguments, but proves NO
   caller correspondence or substitution law. Consumers must check those separately."
  [input expansion]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceConstruction input))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^SourceConstruction input))
       (let [builders (.-builders ^SourceConstruction input)
             matches (filterv #(identical? expansion (:expansion %)) builders)]
         (when (= 1 (count matches))
           (let [{:keys [caller source-import walked-body] :as event} (first matches)
                 uses (filterv #(identical? caller (:caller %)) builders)]
             (when (= 1 (count uses))
               (when-let [imported (body-import-view source-import walked-body)]
                 (assoc event :callee-import imported))))))))))

(defn builder-expansion-view
  "Query one exact shared-builder output and its original caller/imported callee.
   Every actual argument must still be the caller's positional occurrence, including
   unused ones. Recursive argument rewrites without their own proof abstain. This
   supplies no formal completion, return adaptation or final-prefix interval proof."
  [input expansion]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [{:keys [caller arguments] :as event} (builder-construction-view input expansion)]
       (let [original-arguments (descriptor/call-args caller)]
         (when (and (= (count arguments) (count original-arguments))
                    (every? true? (map identical? arguments original-arguments)))
           (when-let [origin (construction-occurrence input caller)]
             (assoc event :caller-origin origin))))))))

(defn recorded-builder-expansions
  "Exact candidates from one authentic source phase. These are construction
   observations, not selected-call ancestry or execution evidence."
  [input]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceConstruction input))
       (entry/assert-producer-current! input)
       (entry/adopt-definition-inputs! (.-observations ^SourceConstruction input))
       (mapv :expansion (.-builders ^SourceConstruction input))))))

(defn- selected-original-invocation [frontend original imported]
  (let [{:keys [dispatch? lexical-callee semantic-op implementation-op arguments]}
        (descriptor/call-description original)]
    (when (and dispatch? (not lexical-callee))
      (when-let [call (entry/frontend-completed-invocation-context frontend original)]
        (let [selected (entry/selected-parameter-carriers semantic-op implementation-op (count arguments))
              definition (:definition selected)
              imported-selection (entry/assert-frontend-selection!
                                  imported (get-in imported [:source-entry :source-var]))]
          (when (and selected (= :global-typed-indy (:invocation-mode call))
                     (= implementation-op (:receiver call) (get-in call [:receiver-input :operand]))
                     (empty? (get-in call [:receiver-input :ports]))
                     (= (:method call) (:invocation-method selected))
                     (= (count arguments) (count (:inputs call)))
                     (= (count arguments) (count (:parameter-carriers selected))
                        (count (:input-completions call)))
                     (every? true? (map identical? (:parameter-carriers selected)
                                        (map :parameter-carrier (:input-completions call))))
                     (every? #(identical? (get definition %) (get imported-selection %))
                             [:source-var :receiver-var :entry])
                     (identical? (get-in definition [:entry :typed-impl]) (:implementation imported)))
            (entry/observe-frontend-source-identity! frontend)
            (entry/observe-frontend-source-identity! imported)
            {:original-call original
             :source-arguments (vec arguments) :call-context call :selected-callee selected}))))))

(defn- selected-original-builder-invocation [event origin]
  (when-let [selected (selected-original-invocation
                      (:frontend origin) (:original origin) (get-in event [:callee-import :frontend]))]
    (assoc selected :builder event :caller-origin origin)))

(deftype ^:private CompletedReferenceCall [caller callee original expression operand-origins observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-frontend-source! caller (get-in caller [:source-entry :source-var]))
    (entry/assert-frontend-source! callee (get-in callee [:source-entry :source-var]))
    (entry/assert-definition-inputs-current! observations)
    true))

(defn construct-original-reference-call
  "Construct one whole original selected call, before source rewriting.
   Receiver and every positional argument complete once before the imported body.
   The saved whole-body result context and actual Method-return check remain.
   Unsupported/cold calls abstain without compiling or executing anything.
   This construction grants no final prefix placement, resident result or AD rule."
  [frontend original]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and frontend entry/*emission-inputs*)
       (with-local-history
         (fn []
           (let [{:keys [dispatch? lexical-callee semantic-op implementation-op arguments]}
                 (descriptor/call-description original)
                 selected (when (and dispatch? (not lexical-callee))
                            (entry/selected-parameter-carriers semantic-op implementation-op (count arguments)))
                 callee (when selected (entry/original-source-frontend (get-in selected [:definition :source-var])))
                 original-call (when callee (selected-original-invocation frontend original callee))
                 call (:call-context original-call)
                 bridge (when original-call (entry/frontend-reference-return-bridge callee))
                 receiver (get-in selected [:definition :receiver-var])]
             (when (and bridge (var? receiver) (not (.isDynamic ^clojure.lang.Var receiver))
                        (false? (get-in bridge [:body-context :unretained-sibling-calls?]))
                        (every? #(false? (:unretained-sibling-calls? %))
                                (cons (:receiver-input call) (:inputs call)))
                        (nil? (.getThreadBinding ^clojure.lang.Var receiver))
                        (= (:method call) (:invocation-method bridge))
                        (= (count arguments)
                           (count (get-in callee [:source-entry :params]))
                           (count (get-in callee [:source-entry :tags])))
                        (every? symbol? (get-in callee [:source-entry :params]))
                        (every? #(and (class? %)
                                      (or (not (.isPrimitive ^Class %))
                                          (contains? #{Integer/TYPE Long/TYPE Float/TYPE Double/TYPE} %)))
                                (:parameter-carriers selected))
                        (identical? (:implementation callee) (:implementation bridge)))
               (let [params (get-in callee [:source-entry :params])
                     tags (get-in callee [:source-entry :tags])
                     source-ns (get-in callee [:source-entry :source-ns])
                     ;; Raw import is deliberately separate from call construction:
                     ;; later wiring the public import producer must not recurse
                     ;; through itself or lose the original callee boundary.
                     imported (with-local-history
                                #(prepare-body callee (:body callee) params tags source-ns true))
                     operand-origins
                     (keep-indexed
                      (fn [index input]
                        (when-let [origin (entry/frontend-invocation-input-source frontend original index)]
                          (when (identical? (:operand input) (:operand origin))
                            [(:operand origin) (:source-input origin)])))
                      (:inputs call))
                     operand-origins (vec operand-origins)
                     body (first (effective-body (:walked-body imported)))
                     all-symbols (filter symbol? (tree-seq coll? seq [original body params]))
                     built
                     (with-hygienic-renamings [original body] all-symbols
                       (fn []
                         (let [receiver-id (util/*fresh-binder*
                                            (with-meta 'completed_receiver {:raster.effect/effectful true}))
                               slots (mapv (fn [formal tag carrier input]
                                             (let [id (util/*fresh-binder*
                                                       (with-meta (symbol (str "completed_" (name formal)))
                                                         {:raster.effect/effectful true
                                                          :tag (symbol (.getName ^Class carrier))
                                                          :raster.type/tag tag}))]
                                               [id (completion/complete-value
                                                    (completion/bytecode-input-contract carrier (:context input))
                                                    (:operand input))]))
                                           params tags (:parameter-carriers selected) (:inputs call))
                               substituted (util/subst-syms (zipmap params (mapv first slots)) body
                                                           util/subst-symbol-preserving-meta)
                               context (get-in bridge [:body-context :result-context])
                               result (completion/complete-value
                                       (completion/bytecode-result-context-contract context :ref) substituted)
                               checked (completion/complete-value
                                        (completion/bytecode-reference-input-contract (:body-return-carrier bridge) context)
                                        result)]
                           (list 'let*
                                 (into [receiver-id
                                        (completion/complete-value
                                         (completion/bytecode-reference-input-contract
                                          Object (get-in call [:receiver-input :context]))
                                         (get-in call [:receiver-input :operand]))]
                                       (mapcat identity slots))
                                 checked))))]
                 (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
                 (CompletedReferenceCall. frontend callee original (:body built) operand-origins
                                          @entry/*emission-inputs*))))))))))

(defn completed-reference-call-view
  "Read the exact constructed call under its historical owner.
   The expression is not portable source or independent execution permission."
  [constructed]
  (when (and entry/*emission-inputs* (instance? CompletedReferenceCall constructed))
    (entry/assert-producer-current! constructed)
    (let [^CompletedReferenceCall constructed constructed]
      (entry/adopt-definition-inputs! (.-observations constructed))
      (entry/adopt-definition-inputs! {:producer-inputs [constructed]})
      {:original (.-original constructed) :expression (.-expression constructed)
       :operand-origins (.-operand-origins constructed)})))

(defn- completed-reference-body-input [frontend constructed]
  (when (and (instance? CompletedReferenceCall constructed)
             (identical? frontend (.-caller ^CompletedReferenceCall constructed)))
    (when-let [{:keys [original expression operand-origins]} (completed-reference-call-view constructed)]
      (let [origins (java.util.IdentityHashMap.) hits (volatile! 0)]
        (letfn [(quoted? [node]
                  (and (seq? node) (contains? '#{quote clojure.core/quote} (first node))))
                (seed [node]
                  (when (and (not (quoted? node)) (or (seq? node) (vector? node)))
                    (.put origins node {:original node})
                    (doseq [child node] (seed child))))
                (go [node]
                  (cond
                    (quoted? node) node

                    (identical? original node)
                    (do (vswap! hits inc)
                        (.put origins expression {:original original :completed-call constructed})
                        expression)

                    (or (seq? node) (vector? node))
                    (let [result (util/remake-if-changed node (mapv go node))]
                      (.put origins result {:original node})
                      result)

                    :else node))]
          (doseq [node (:body frontend)] (seed node))
          (doseq [[actual source] operand-origins]
            (.put origins actual {:original source}))
          (let [body (mapv go (:body frontend))]
            ;; No search by shape/name, no insertion under quoted data and no
            ;; duplication of a shared occurrence. Imported nodes acquire no
            ;; caller-source origins. Parent edges come from actual rebuilding.
            (when (= 1 @hits) {:body body :origins origins})))))))

(defn original-builder-invocation-view
  "Relate one recorded builder's imported callee to an actual original selected call.
   Rewritten arguments need not be unchanged: this authenticates the ORIGINAL
   target and its observed input conversions, not their constructed replacements.
   Import-local callers lacking original occurrence transport still abstain."
  [input expansion]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [{:keys [caller] :as event} (builder-construction-view input expansion)]
       (when-let [origin (construction-occurrence input caller)]
         (selected-original-builder-invocation event origin))))))

(defn imported-builder-invocation-candidate
  "Authenticate a selected original call in an exact recorded parent's import.
   `qualified` is checked against that import, NOT accepted as an origin map.
   Parent/child linkage and the parent's own selection remain UNPROVED here;
   only the inliner's checked substitution ancestry can discharge those duties."
  [input expansion parent-expansion qualified]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [event (builder-construction-view input expansion)]
       (when-let [parent (builder-construction-view input parent-expansion)]
         (when-let [origin (body-import-occurrence (:source-import parent) (:walked-body parent) qualified)]
           (when-let [selected (selected-original-builder-invocation event origin)]
             (assoc selected :parent-linkage :unclosed))))))))

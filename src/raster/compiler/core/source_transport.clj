(ns raster.compiler.core.source-transport
  "Owner-local source transformations, separate from source execution admission.
   Preserve authentic helper inputs at the transformation that consumes them;
   never reconstruct their origins from later symbol names or equal expressions."
  (:require [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]))

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
  [frontend parameters source-ns effective hygienic qualified tags renamings observations]
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

(defn- prepare-body [frontend body parameters tags source-ns owned?]
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
        qualified-tags (qualify-tags tags source-ns)
        effective (effective-body body)
        renamed (when owned?
                  (with-hygienic-renamings effective parameters #(mapv util/alpha-convert effective)))
        hygienic (if owned? (:body renamed) (mapv util/alpha-convert effective))
        parameter-set (set (map #(if (symbol? %) % (symbol (name %))) parameters))
        qualified (if source-ns
                    (mapv #(inference/qualify-body-symbols % source-ns parameter-set
                                                         entry/observe-source-resolution!) hygienic)
                    hygienic)
        input (when owned?
                (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
                (entry/assert-definition-inputs-current! @entry/*emission-inputs*)
                (BodyImport. frontend parameters source-ns effective hygienic qualified
                             qualified-tags (:renamings renamed) @entry/*emission-inputs*))]
    {:walked-body qualified :tags qualified-tags :source-import input}))

(defn prepare-inline-body
  "Perform the inliner's existing alpha-renaming and qualification once.
   An authentic frontend and active owner additionally retain an immutable input
   witness with IMPORT-LOCAL history. Legacy bodies acquire no owner. No formal
   argument completion, call substitution, final-step correspondence, normal
   return or permission to execute the imported body is supplied."
  [frontend body parameters tags source-ns]
  (if (and frontend entry/*emission-inputs*)
    (with-local-history #(prepare-body frontend body parameters tags source-ns true))
    (prepare-body frontend body parameters tags source-ns false)))

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

(deftype ^:private SourceNormalization [frontend body occurrences renamings observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (entry/assert-definition-inputs-current! observations)
    true))

(defn normalize-frontend-bindings
  "Uniquify the exact authenticated frontend, retaining actual construction edges.
   The witness is producer-local: do not put it in compiler options, IR metadata
   or portable artifacts. It proves no completion, inlining or execution rights."
  [frontend parameters]
  (when (and frontend entry/*emission-inputs*)
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when-not (= parameters (get-in frontend [:source-entry :params]))
      (throw (ex-info "source normalization differs from its original parameters"
                      {:reason :source-normalization-input-mismatch})))
    (with-local-history
      (fn []
        (entry/observe-frontend-source-identity! frontend)
        (let [parameters (mapv #(if (symbol? %) (symbol (name %)) %) parameters)
              occurrences (volatile! [])
              renamed (with-hygienic-renamings
                        (:body frontend) parameters
                        #(mapv (fn [ordinal expression]
                                 (util/uniquify-rebindings
                                  parameters expression
                                  (fn [original constructed renaming bound]
                                    (when (and (seq? original)
                                               (not (contains? '#{quote clojure.core/quote}
                                                               (first original))))
                                      (vswap! occurrences conj
                                              {:original original :constructed constructed
                                               :root ordinal :renaming renaming :bound bound})))))
                               (range) (:body frontend)))
              body (:body renamed)]
          {:walked-body body
           :source-normalization (SourceNormalization. frontend body @occurrences
                                                       (:renamings renamed) @entry/*emission-inputs*)})))))

(defn normalization-occurrence
  "Query a unique exact constructed occurrence under its original history.
   Copied, repeated-shared or ownerless nodes abstain. No transformation or source
   search is rerun; successful adoption retains the original source identity guard."
  [input constructed]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? SourceNormalization input))
       (let [^SourceNormalization input input
             occurrences (.-occurrences input)
             matches (filterv #(identical? constructed (:constructed %)) occurrences)]
         (when (and (= 1 (count matches))
                    (= 1 (count (filter #(identical? (:original (first matches)) (:original %))
                                        occurrences))))
           (entry/assert-producer-current! input)
           (entry/adopt-definition-inputs! (.-observations input))
           (assoc (first matches) :frontend (.-frontend input))))))))

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

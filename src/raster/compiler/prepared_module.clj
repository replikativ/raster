(ns raster.compiler.prepared-module
  "Transactional construction of a prepared AD import through source normalization.
   This private compiler phase authenticates construction, never device admission.
   Conversion, source arithmetic, storage, runtime views and publication remain
   obligations of the existing ordered source owner. No object enters portable IR."
  (:require [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.inline :as inline]))

(defn- refuse! [message data]
  (throw (ex-info message (assoc data :reason :unclosed-prepared-module-phase))))

(defn- pairs [source]
  (when-not (and (seq? source) (contains? '#{let let*} (first source))
                 (vector? (second source)) (even? (count (second source))))
    (refuse! "prepared module normalization requires an ordered binding spine" {}))
  (vec (partition 2 (second source))))

(defn- same-pair? [a b]
  (and (= 2 (count a) (count b)) (every? true? (map identical? a b))))

(defn- unique-stage [events stage]
  (let [matches (filterv #(= stage (:stage %)) events)]
    (when-not (= 1 (count matches))
      (refuse! "prepared module normalization lost a unique stage" {:stage stage}))
    (first matches)))

(defn- normalization-origins [source normalized events]
  (let [[copy alpha witness lift spine final]
        (mapv #(unique-stage events %)
              [:copy-allocations :uniquify-root :array-type-witness-root
               :lift-scalar-reductions :normalize-binding-spine :normalize-source])
        stages [copy alpha witness lift spine]
        original-pairs (pairs source)
        copy-pairs (pairs (:constructed copy))
        lift-inputs (pairs (:original lift))
        lift-outputs (pairs (:constructed lift))
        outputs (pairs normalized)
        splits (filterv #(= :reduction-split (:stage %)) events)]
    (when-not (and (identical? source (:original copy))
                   (= [:copy-allocations :uniquify-root :array-type-witness-root
                       :lift-scalar-reductions :normalize-binding-spine :normalize-source]
                      (mapv :stage (filter #(contains? #{:copy-allocations :uniquify-root :array-type-witness-root
                                                        :lift-scalar-reductions :normalize-binding-spine :normalize-source}
                                                       (:stage %)) events)))
                   (identical? source (:original final))
                   (identical? normalized (:constructed spine))
                   (identical? normalized (:constructed final))
                   (every? true? (map #(identical? (:constructed %1) (:original %2))
                                      stages (rest stages)))
                   (= (count original-pairs) (count copy-pairs) (count lift-inputs))
                   (= (count original-pairs) (count (pairs (:constructed alpha))))
                   ;; Actual copy expansion can add bindings. Until it has its
                   ;; own origins, abstain rather than infer them by equality.
                   (every? true? (map same-pair? original-pairs copy-pairs)))
      (refuse! "prepared module normalization stages do not form one exact origin chain" {}))
    (let [lift-origins
          (loop [index 0 output-index 0 origins [] used #{}]
            (if (= index (count lift-inputs))
              (do
                (when-not (and (= output-index (count lift-outputs)) (= (count used) (count splits)))
                  (refuse! "prepared module reduction splits do not cover the lifted spine" {}))
                origins)
              (let [[id expression :as input] (nth lift-inputs index)
                    matches (filterv #(= index (:binding-index %)) splits)]
                (if (empty? matches)
                  (do
                    (when-not (and (< output-index (count lift-outputs))
                                   (same-pair? input (nth lift-outputs output-index)))
                      (refuse! "untraced prepared module binding rewrite" {:index index}))
                    (recur (inc index) (inc output-index)
                           (conj origins {:source-index index :lift-role :source-binding}) used))
                  (let [split (first matches)]
                    (when-not (and (= 1 (count matches))
                                   (identical? (:scope split) (:original lift))
                                   (identical? (:output-scope split) (:constructed lift))
                                   (= output-index (:output-binding-index split))
                                   (identical? id (:destination split))
                                   (identical? expression (:original split))
                                   (= 2 (count (:bindings split)))
                                   (<= (+ output-index 2) (count lift-outputs))
                                   (same-pair? [(:result split) (:reduction split)] (first (:bindings split)))
                                   (same-pair? [id (:constructed split)] (second (:bindings split)))
                                   (every? true? (map same-pair? (:bindings split)
                                                      (subvec lift-outputs output-index (+ output-index 2)))))
                      (refuse! "prepared module reduction split lost its exact occurrence" {:index index}))
                    (recur (inc index) (+ output-index 2)
                           (into origins [{:source-index index :lift-role :reduction}
                                          {:source-index index :lift-role :continuation}])
                           (conj used index)))))))
          entries (:entries spine)
          flattened (vec (mapcat :outputs entries))]
      (when-not (and (= (count entries) (count lift-outputs))
                     (= (count flattened) (count outputs))
                     (= (count outputs) (count (set (map first outputs))))
                     (every? true?
                             (map-indexed
                              (fn [index {:keys [input-index input-binding outputs]}]
                                (and (= index input-index)
                                     (same-pair? input-binding (nth lift-outputs index))
                                     (seq outputs)
                                     (= :source-binding (:origin (last outputs)))
                                     (identical? (first input-binding) (get-in (last outputs) [:binding 0]))
                                     (every? #(= :derived-normalization-binding (:origin %)) (butlast outputs))))
                              entries))
                     (every? true? (map-indexed (fn [index output]
                                                  (and (= index (:index output))
                                                       (same-pair? (:binding output) (nth outputs index))))
                                                flattened)))
        (refuse! "prepared module normalization origins do not cover the final spine" {}))
      (into [] (mapcat (fn [origin event]
                        (map #(assoc origin :index (:index %) :binding (:binding %)
                                            :normalization-role (:origin %)) (:outputs event)))
                      lift-origins entries)))))

(defn- assert-import-current! [{:keys [prepared physical]}]
  ;; Requerying asserts the actual GeneratedPreparationOwner, program, epoch
  ;; and physical module. A newly equal map cannot refresh the original phase.
  (when-not (identical? physical (reverse/prepared-bytecode-entry prepared))
    (refuse! "prepared module physical entry changed" {}))
  (let [source-ns (get-in physical [:emission :source-ns])]
    (when-not (and (instance? clojure.lang.Namespace source-ns)
                   (identical? source-ns (find-ns (ns-name source-ns))))
      (refuse! "prepared module physical namespace changed" {}))))

(defn- checked-type-union [options additions]
  (let [result (reduce
   (fn [options kind]
     (update options kind
             (fn [declared]
               (reduce (fn [result [id dt]]
                            (when-not (and (dtype/known? dt)
                                           (or (not (contains? result id))
                                               (and (dtype/known? (get result id))
                                                    (= (dtype/canon (get result id)) (dtype/canon dt)))))
                              (refuse! "prepared module normalization declarations disagree"
                                       {:kind kind :binding id}))
                            (assoc result id (dtype/canon dt)))
                          {} (concat declared (get additions kind {}))))))
   options [:array-types :scalar-types])]
    (when-let [id (some (set (keys (:array-types result))) (keys (:scalar-types result)))]
      (refuse! "prepared module value cannot be declared as both scalar and array" {:binding id}))
    result))

(defn- import-origins [source events origins imports]
  (let [source-pairs (pairs source)
        before-lift (pairs (:original (unique-stage events :lift-scalar-reductions)))
        by-source (group-by :source-index origins)]
    (mapv
     (fn [import-index {:keys [call placement import physical]}]
       (let [prefix (get-in placement [:binding-site :prefix])
             start (count prefix)
             bindings (:bindings call)
             end (+ start (count bindings))
             ninputs (count (:input-obligations call))
             slots (get-in call [:argument-plan :slots])
             layout (generated/value-source-binding-layout call)
             expected-layout (:bindings layout)]
         (when-not (and (identical? physical (get-in call [:proposal :physical]))
                        layout (= ninputs (count slots))
                        (= (count expected-layout) (count bindings))
                        (every? true? (map #(identical? %1 (:slot %2)) slots (:input-obligations call)))
                        (every? true? (map same-pair? expected-layout bindings)))
           (refuse! "prepared module obligation layout differs from its staged argument and body plan" {:import-index import-index}))
         (when-not (and (<= end (count source-pairs))
                        (every? true? (map same-pair? prefix (take start source-pairs)))
                        (every? true? (map same-pair? bindings (subvec source-pairs start end)))
                        (every? true? (map #(identical? (first %1) (first %2))
                                           bindings (subvec before-lift start end))))
           (refuse! "prepared module bindings lost their exact cut insertion interval" {:import-index import-index}))
         (let [binding-origins
               (mapv (fn [index binding]
                       (let [source-index (+ start index) final (get by-source source-index)]
                         (when-not (seq final)
                           (refuse! "prepared module binding has no final normalization origin" {:import-index import-index :index index}))
                         {:call-index index :source-index source-index :binding binding :origins final}))
                     (range) bindings)
               input-obligations
               (mapv (fn [index obligation]
                       {:id [import-index :input index] :obligation obligation
                        :binding-origins [(nth binding-origins index)
                                          (nth binding-origins (+ ninputs index))]})
                     (range) (:input-obligations call))
               body-obligations
               (mapv (fn [index obligation call-binding-index]
                       (let [[container position] (:path obligation)
                             root-index (when (and (= 1 container) (integer? position) (pos? position) (odd? position))
                                          (quot position 2))]
                         (when-not (and root-index (< root-index (count (:root-bindings call)))
                                        (integer? call-binding-index)
                                        (<= 0 call-binding-index) (< call-binding-index (count binding-origins)))
                           (refuse! "prepared completion is outside the retained root binding interval" {:path (:path obligation)}))
                         {:id [import-index :body index] :obligation obligation
                          :binding-origins [(nth binding-origins call-binding-index)]}))
                     (range) (get-in call [:proposal :completion-obligations])
                     (:body-obligation-indices layout))
               argument-obligations
               (into []
                     (mapcat
                      (fn [{:keys [root-index argument-construction]}]
                        (map (fn [{:keys [input-index call-binding-index] :as argument}]
                               {:id [import-index :argument root-index input-index]
                                :obligation argument
                                :binding-origins [(nth binding-origins call-binding-index)]})
                             (:arguments argument-construction))))
                     (:root-bindings call))]
           {:import import :call call :binding-origins binding-origins
            :obligation-origins (into (into input-obligations body-obligations) argument-obligations)})))
     (range) imports)))

(deftype ^:private PreparedModulePhase [context observations]
  entry/RetainedProducerInput
  (assert-producer-current! [_]
    (entry/assert-producer-current! (:source-construction context))
    (entry/assert-definition-inputs-current! observations)
    (doseq [import (:imports context)] (assert-import-current! import))
    true))

(defn construct-source
  "Construct, represent and normalize one exact frontend under its source history.
   Requires an active definition collector. Only direct public-parameter inputs
   currently have prepared-call carrier evidence; other imported expressions refuse.
   Compiler preparation precedes the final AD cut. Representation runs once after
   closing that pre-SoA source construction; imported spines currently require an
   identity representation. Ordinary no-import sources use the same construction
   and normalization path without manufacturing a PreparedModulePhase.
   The returned handoff map is NOT authority: its retained objects own provenance.
   A prepared phase seals construction only, never admission or marker closure."
  ([source-frontend options] (construct-source source-frontend options {}))
  ([source-frontend options {:keys [prepare-source represent-source]
                             :or {prepare-source identity
                                  represent-source (fn [source options]
                                                     {:body source :options options})}}]
  (when-not entry/*emission-inputs*
    (refuse! "prepared module construction requires a definition owner" {}))
  (let [source-var (get-in source-frontend [:source-entry :source-var])
        _ (entry/assert-frontend-source! source-frontend source-var)
        caller (delay (entry/frontend-body-entry-carriers source-frontend source-var))
        parameters (get-in source-frontend [:source-entry :params])
        normalization (transport/normalize-frontend-bindings source-frontend parameters)
        normalized-source (first (transport/effective-body (:walked-body normalization)))
        environment (zipmap parameters (get-in source-frontend [:source-entry :tags]))
        minted (volatile! [])
        expanded
        (transport/with-source-construction-history
         (:source-normalization normalization)
         #(let [prepared-source
                ;; Do not expose the insertion consumer to earlier passes. A
                ;; generated spine inserted here would be rewritten again by
                ;; preparation, invalidating its occurrence-owned obligations.
                (binding [transport/*prepared-program-consumer* nil
                          inline/*defer-prepared-ad-expansion?* true]
                  (prepare-source normalized-source))]
            (binding [inline/*defer-prepared-ad-expansion?* false
                      transport/*prepared-program-consumer*
                  (fn [event placement]
                    (let [caller @caller
                          prepared (:prepared event)
                          physical (reverse/prepared-bytecode-entry prepared)
                          _ (when-not (and physical (reverse/adopt-prepared-program-inputs! prepared))
                              (refuse! "prepared module producer could not join its enclosing owner" {}))
                          _ (when-not caller
                              (refuse! "prepared module has no actual caller entry carriers" {}))
                          facts (mapv (:parameter-facts caller) (:arguments event))
                          _ (when-not (every? some? facts)
                              (refuse! "prepared module call requires exact public parameter arguments" {}))
                          call (generated/instantiate-value-source physical (:params prepared)
                                                                   (vec (:arguments event)) facts)
                          nonce (Object.)
                          import {:event event :call call :status :pending :admitted? false :nonce nonce}
                          ports (if (= :grad (:mode event)) (vec (rest (:outputs call))) (:outputs call))]
                      (vswap! minted conj {:event event :placement placement :prepared prepared
                                           :physical physical :call call :import import :nonce nonce})
                      {:ports ports :bindings (:bindings call) :import import}))]
              (inline/inline-ad-applications prepared-source environment))))
        construction (transport/source-construction-view (:source-construction expanded)
                                                         (:source-normalization normalization) (:form expanded))
        cuts (:prepared-program-cuts construction)]
    (when-not (and construction (= (count cuts) (count @minted))
                   (every? true?
                           (map (fn [cut {:keys [event placement call import nonce]}]
                                  (and (identical? event (:event cut))
                                       (identical? placement (:placement cut))
                                       (identical? import (:import cut))
                                       (identical? call (get-in cut [:import :call]))
                                       (identical? nonce (get-in cut [:import :nonce]))
                                       (identical? (:bindings call) (:canonical-bindings cut)))) cuts @minted)))
      (refuse! "prepared module calls and inserted cuts are not a bijection" {}))
    (doseq [import @minted] (assert-import-current! import))
    (let [caller (when (seq @minted) @caller)
          actual-options
          {:array-types (into {} (keep (fn [[id fact]] (when-let [dt (dtype/dtype-for-jvm-array-class (:carrier fact))] [id dt])))
                              (:parameter-facts caller))
           :scalar-types (into {} (keep (fn [[id fact]]
                                         (when (.isPrimitive ^Class (:carrier fact))
                                           (when-let [dt (dtype/dtype-for-scalar-tag (symbol (.getName ^Class (:carrier fact))))]
                                             [id dt]))))
                               (:parameter-facts caller))}
          source-ns (get-in source-frontend [:source-entry :source-ns])
          _ (when-not (and (or (not (contains? options :source-ns)) (identical? source-ns (:source-ns options)))
                           (or (not (contains? options :public-parameters)) (= parameters (:public-parameters options))))
              (refuse! "prepared module options differ from their authenticated frontend" {}))
          options (reduce (fn [options {:keys [call]}]
                            (checked-type-union options (:normalization-requirements call)))
                          (checked-type-union options actual-options) @minted)
          options (assoc options :source-ns source-ns
                                 :public-parameters parameters)
          represented (represent-source (:form expanded) options)
          _ (when-not (and (map? represented) (contains? represented :body)
                           (seq? (:body represented)))
              (refuse! "source representation did not return its actual body" {}))
          physical-source (:body represented)
          physical-options (get represented :options options)
          _ (when-not (map? physical-options)
              (refuse! "source representation did not return compiler options" {}))
          _ (when (and (seq @minted)
                       (not (and (identical? physical-source (:form expanded))
                                 (identical? options physical-options)
                                 (nil? (:representation represented)))))
              (refuse! "prepared module representation changed its unclosed interval"
                       {:representation :unclosed}))
          events (volatile! [])
          normalized (frontend/normalize-source physical-source physical-options #(vswap! events conj %))
          origins (when (seq @minted)
                    (normalization-origins physical-source normalized @events))
          import-origins (when (seq @minted)
                           (import-origins physical-source @events origins @minted))
          context {:source-construction (:source-construction expanded)
                   :normalization (:source-normalization normalization)
                   :source (:form expanded) :physical-source physical-source
                   :normalized normalized :options physical-options
                   :representation (:representation represented)
                   :imports @minted :events @events :origins origins :import-origins import-origins :admitted? false}
          observations @entry/*emission-inputs*
          phase (when (seq @minted) (PreparedModulePhase. context observations))]
      (when phase (entry/assert-producer-current! phase))
      {:normalization (:source-normalization normalization)
       :normalized-source normalized-source
       :construction (:source-construction expanded)
       :represented-source (:form expanded) :physical-source physical-source
       :frontend-source normalized :frontend-options physical-options
       :representation (:representation represented) :prepared-phase phase}))))

(defn capture
  "Test/diagnostic adapter over the shared source constructor, requiring an import.
   No provisional admission or original prepared-import marker is discharged."
  ([source-frontend options] (capture source-frontend options identity))
  ([source-frontend options prepare-source]
   (or (:prepared-phase (construct-source source-frontend options {:prepare-source prepare-source}))
       (refuse! "prepared module construction did not contain an import" {}))))

(defn view
  "Query only this phase's exact normalized object under its original history.
   The returned pending descriptors are not standalone certificates. Copies,
   ownerless queries and objects from another phase abstain without normalizing."
  [phase normalized]
  (when (and entry/*emission-inputs* (instance? PreparedModulePhase phase))
    (let [^PreparedModulePhase phase phase
          context (.-context phase)]
      (when (identical? normalized (:normalized context))
        (entry/assert-producer-current! phase)
        (entry/adopt-definition-inputs! (update (.-observations phase) :producer-inputs (fnil conj []) phase))
        context))))

(defn normalized-source
  "Read the exact constructed source, checking phase freshness. Not admission."
  [phase]
  (when (and entry/*emission-inputs* (instance? PreparedModulePhase phase))
    (entry/assert-producer-current! phase)
    (:normalized (.-context ^PreparedModulePhase phase))))

(defn- final-binding-origin [binding-origin]
  ;; No reconstruction of a split reduction or derived normalization temporary.
  ;; These require their own correspondence, not a guess at its last result.
  (let [origins (:origins binding-origin)]
    (when (and (= 1 (count origins))
               (= :source-binding (:lift-role (first origins)))
               (= :source-binding (:normalization-role (first origins))))
      (first origins))))

(defn- input-port-origin [joined current-binding-origin current-final parameter physical-port]
  (let [call (:call joined)
        physical (get-in call [:proposal :physical])
        {:keys [emission input-body]} physical
        slots (get-in call [:argument-plan :slots])
        formals (filterv #(= parameter (:formal %)) slots)
        roots (filterv #(= parameter (:original-id %)) (:root-bindings call))
        ;; A physical port is either a generated formal or a prior original
        ;; root. The latter is common even for sizes: explicit completed inputs
        ;; such as rows and the literal 1 have their own verifier local slots.
        correspondence
        (cond
          (and (= 1 (count formals)) (empty? roots))
          (let [slot (first formals)
                index (+ (count slots) (:ordinal slot))
                body (bytecode/retained-input-body-context emission input-body)]
            (when (= physical-port (get-in body [:locals parameter]))
              {:kind :formal :slot slot :call-index index :physical-context body}))

          (and (empty? formals) (= 1 (count roots)))
          (let [root (first roots)
                scope (get-in call [:proposal :original])
                index (:root-index root)
                record (bytecode/retained-input-binding-context emission input-body scope index)]
            ;; This is the input-origin join to the emitted binding, which may
            ;; retag/expand its initializer. Never compare the expanded syntax
            ;; to the original initializer, or treat an equal symbol as proof.
            (when (and record (identical? scope (:source-input record))
                       (= parameter (:binder record))
                       (identical? (:original-expression root)
                                   (nth (second scope) (inc (* 2 index)) nil))
                       (= physical-port (:after record)))
              {:kind :root :root root :call-index (:call-binding-index root)
               :physical-context record})))
        binding-origin (when correspondence
                         (nth (:binding-origins joined) (:call-index correspondence) nil))
        final (final-binding-origin binding-origin)]
    (when (and final
               (< (:call-index binding-origin) (:call-index current-binding-origin))
               (< (:source-index binding-origin) (:source-index current-binding-origin))
               (< (:index final) (:index current-final)))
      (assoc correspondence :parameter parameter :physical-port physical-port
                            :binding-origin binding-origin :origin final
                            :binding (:binding final)))))

(defn ^:no-doc input-occurrence
  "Compiler-private correspondence for one exact staged invocation input.
   The pair must be the actual normalized binding (its two objects, not printed
   or structural equality). Ports follow original generated formal/root slots
   through the retained normalization history. No code is compiled or executed.
   This query grants neither completion nor admission; every obligation stays
   pending until the whole source owner has validated its InvocationPlan."
  [phase normalized binding]
  (when-let [context (view phase normalized)]
    (let [matches
          (into []
                (mapcat
                 (fn [joined]
                   (when (generated/value-source-binding-layout (:call joined))
                     (keep
                      (fn [{:keys [id obligation binding-origins] :as pending}]
                        (when (and (= :argument (second id)) (= 1 (count binding-origins)))
                          (let [binding-origin (first binding-origins)
                                final (final-binding-origin binding-origin)
                                root (nth (get-in joined [:call :root-bindings]) (nth id 2) nil)
                                construction (:argument-construction root)
                                argument (nth (:arguments construction) (:input-index obligation) nil)]
                            (when (and final (same-pair? binding (:binding final))
                                       (identical? argument obligation)
                                       (= (:call-binding-index argument) (:call-index binding-origin)))
                              (let [ports (mapv (fn [[parameter port]]
                                                  (input-port-origin joined binding-origin final parameter port))
                                                (sort-by (comp :slot val) (get-in argument [:input :ports])))]
                                (when (every? some? ports)
                                  {:kind :prepared-module-invocation-input
                                   :phase phase :import (:import joined)
                                   :physical (get-in joined [:call :proposal :physical])
                                   :root root :construction construction
                                   :obligation pending :argument argument
                                   :origin final :binding (:binding final) :ports ports}))))))
                      (:obligation-origins joined)))))
                (:import-origins context))]
      (when (= 1 (count matches)) (first matches)))))

(defn ^:no-doc analyze-source-components
  "Analyze this exact phase once, retaining pending input identities separately
   from the conditional logical/carrier facts. This neither continues semantic
   admission nor constructs a source owner or executable InvocationPlan."
  [phase normalized]
  (when-let [context (view phase normalized)]
    (let [inputs
          (into []
                (mapcat
                 (fn [joined]
                   (keep (fn [{:keys [id binding-origins]}]
                           (when (= :argument (second id))
                             (let [final (when (= 1 (count binding-origins))
                                           (final-binding-origin (first binding-origins)))
                                   occurrence (when final (input-occurrence phase normalized (:binding final)))]
                               (when-not occurrence
                                 (refuse! "prepared input lost its exact analysis occurrence" {:id id}))
                               [(:index final) occurrence])))
                         (:obligation-origins joined))))
                (:import-origins context))
          _ (when-not (= (count inputs) (count (set (map first inputs))))
              (refuse! "prepared inputs do not have distinct source producers" {}))
          components (frontend/analyze-source-components normalized (:options context) (into {} inputs))]
      (entry/assert-producer-current! phase)
      components)))

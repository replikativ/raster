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
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
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

(defn- import-insertion-site
  [source {:keys [placement call assembly]}]
  (let [original-prefix (get-in placement [:binding-site :prefix])
        bindings (:bindings call)
        source-pairs (pairs source)
        prefix (if assembly (:prefix assembly) original-prefix)
        start (if assembly (:start assembly) (count prefix))
        end (when (integer? start) (+ start (count bindings)))]
    (when assembly
      (let [original-pairs (pairs (:original-source assembly))
            original-start (count original-prefix)
            original-end (+ original-start (count bindings))]
        (when-not (and (identical? source (:constructed-source assembly))
                       (= original-start (:original-start assembly))
                       (<= original-end (count original-pairs))
                       (every? true? (map same-pair? original-prefix (subvec original-pairs 0 original-start)))
                       (every? true? (map same-pair? bindings (subvec original-pairs original-start original-end))))
          (refuse! "import assembly lost its original exact cut" {}))))
    (when-not (and (vector? original-prefix) (vector? prefix) (vector? bindings) (seq bindings)
                   (integer? start) (= start (count prefix))
                   (<= 0 start end (count source-pairs))
                   (every? true? (map same-pair? prefix (subvec source-pairs 0 start)))
                   (every? true? (map same-pair? bindings (subvec source-pairs start end))))
      (refuse! "import assembly lost its constructed insertion interval" {}))
    {:prefix prefix :start start :end end}))

(defn- import-origins [source events origins imports]
  (let [before-lift (pairs (:original (unique-stage events :lift-scalar-reductions)))
        by-source (group-by :source-index origins)]
    (mapv
     (fn [import-index {:keys [call import physical] :as descriptor}]
       (let [{:keys [start end]} (import-insertion-site source descriptor)
             bindings (:bindings call)
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
         (when-not (and (<= end (count before-lift))
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

(declare authored-source-layout prepare-authored-layout)

(defn construct-source
  "Construct, represent and normalize one exact frontend under its source history.
   Requires an active definition collector. Only direct public-parameter inputs
   currently have prepared-call carrier evidence; other imported expressions refuse.
   The optional authored-segment preparer discovers real cuts first, prepares
   only their caller segments, then records a new assembly construction. Ordinary
   preparation still runs on whole no-import source. Representation runs once after
   closing that pre-SoA source construction; imported spines currently require an
   identity representation. Ordinary no-import sources use the same construction
   and normalization path without manufacturing a PreparedModulePhase. Their
   optional representation canonicalizer runs once before final normalization;
   imported source never enters that ordinary-only rewrite.
   The returned handoff map is NOT authority: its retained objects own provenance.
   A prepared phase seals construction only, never admission or marker closure."
  ([source-frontend options] (construct-source source-frontend options {}))
  ([source-frontend options {:keys [prepare-source represent-source prepare-authored-segment
                                  canonicalize-ordinary-source]
                             :or {prepare-source identity
                                  represent-source (fn [source options]
                                                     {:body source :options options})
                                  canonicalize-ordinary-source (fn [source _] source)}}]
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
        assembly (volatile! nil)
        consume
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
            {:ports ports :bindings (:bindings call) :import import}))
        discover
        (fn [source]
          (binding [inline/*defer-prepared-ad-expansion?* false
                    transport/*prepared-program-consumer* consume]
            (inline/inline-ad-applications source environment)))
        expanded
        (transport/with-source-construction-history
         (:source-normalization normalization)
         #(let [discovered (when prepare-authored-segment (discover normalized-source))
                prepared-source
                ;; Whole-source preparation is only legal before a cut exists.
                ;; Once discovered, generated spines stay outside all rewriting.
                (when-not (seq @minted)
                  (binding [transport/*prepared-program-consumer* nil
                            inline/*defer-prepared-ad-expansion?* true]
                    (prepare-source normalized-source)))
                source (if (seq @minted) discovered (discover prepared-source))]
            (if (and prepare-authored-segment (seq @minted))
              (let [prepared (binding [transport/*prepared-program-consumer* nil
                                       inline/*defer-prepared-ad-expansion?* true]
                               (prepare-authored-layout
                                (authored-source-layout source @minted) environment
                                (assoc options :source-ns (get-in source-frontend [:source-entry :source-ns]))
                                prepare-authored-segment))
                    result (:source prepared)
                    imports (mapv (fn [import]
                                    (update import :assembly assoc :original-source source :constructed-source result))
                                  (:imports prepared))]
                (vreset! assembly (assoc prepared :original source :constructed result :imports imports))
                (vreset! minted imports)
                (transport/observe-construction! source result environment)
                result)
              source)))
        construction (transport/source-construction-view (:source-construction expanded)
                                                         (:source-normalization normalization) (:form expanded))
        cuts (:prepared-program-cuts construction)]
    (when @assembly
      (let [{:keys [original constructed]} @assembly
            edges (filterv #(and (identical? original (:original %))
                                  (identical? constructed (:constructed %)))
                           (:type-elaborations construction))]
        (when-not (= 1 (count edges))
          (refuse! "authored assembly lacks its exact construction-history edge" {}))))
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
          represented-body (:body represented)
          physical-options (get represented :options options)
          _ (when-not (map? physical-options)
              (refuse! "source representation did not return compiler options" {}))
          _ (when (and (seq @minted)
                       (not (and (identical? represented-body (:form expanded))
                                 (identical? options physical-options)
                                 (nil? (:representation represented)))))
              (refuse! "prepared module representation changed its unclosed interval"
                       {:representation :unclosed}))
          ;; Ordinary-only canonicalization precedes final normalization. An
          ;; imported spine cannot undergo such rewriting without new origins.
          physical-source (if (seq @minted) represented-body
                              (canonicalize-ordinary-source represented-body physical-options))
          events (volatile! [])
          normalized (frontend/normalize-source physical-source physical-options #(vswap! events conj %))
          origins (when (seq @minted)
                    (normalization-origins physical-source normalized @events))
          import-origins (when (seq @minted)
                           (import-origins physical-source @events origins @minted))
          context {:source-construction (:source-construction expanded)
                   :authored-assembly @assembly
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

(defn- import-output-contexts
  [{:keys [call physical]}]
  (when-not (= (:outputs call) (nth (:projected-source call) 2 nil))
    (refuse! "prepared outputs differ from their original substituted result" {}))
  (mapv
   (fn [output]
     (when output
       (let [roots (filterv #(= output (:id %)) (:root-bindings call))
             root (when (= 1 (count roots)) (first roots))
             record (when root
                      (bytecode/retained-input-binding-context
                       (:emission physical) (:input-body physical)
                       (get-in call [:proposal :original]) (:root-index root)))
             after (:after record)
             array-dtype (when (= :ref (:type after))
                           (dtype/dtype-for-jvm-array-class (:reference-class after)))
             normal-fact (get-in call [:proposal :normal-return-facts (:original-id root)])
             scalar-tag
             (if (= :ref (:type after))
               (when (and (identical? (:reference-class after) (:carrier normal-fact))
                          (:builtin-class normal-fact))
                 (some (fn [[tag carrier]]
                         (when (identical? carrier (:builtin-class normal-fact)) tag))
                       types/primitive-info))
               (when (contains? #{:int :long :float :double :bool} (:type after))
                 (if (= :bool (:type after)) 'boolean (symbol (name (:type after))))))
             tag (or (when array-dtype (dtype/array-tag-for-dtype array-dtype)) scalar-tag)]
         (when-not (and root record (true? (:verified after))
                        (= (:original-id root) (:binder record)) tag)
           (refuse! "prepared output lacks its original emitted carrier"
                    {:output output :root-index (:root-index root)}))
         ;; This is a declaration conditional on the import's normal return,
         ;; not presence, capacity, successful completion or executable authority.
         {:output output :root root :binding-context record :tag tag
          :normal-return-fact (when scalar-tag normal-fact)})))
   (:outputs call)))

(defn- authored-source-layout
  [source imports]
  (let [source-pairs (pairs source)]
    (loop [remaining imports cursor 0 stages []]
      (if-let [import (first remaining)]
        (let [{:keys [start end]} (import-insertion-site source import)
              bindings (get-in import [:call :bindings])]
          (when-not (and (<= cursor start)
                         (generated/value-source-binding-layout (:call import))
                         (identical? (:physical import) (get-in import [:call :proposal :physical])))
            (refuse! "authored preparation lost an exact nonoverlapping import interval"
                     {:start start :end end :previous-end cursor}))
          (recur (next remaining) end
                 (conj stages
                       {:kind :authored :start cursor :end start
                        :bindings (subvec source-pairs cursor start)
                        :live-outs (mapv first (subvec source-pairs cursor start))}
                       {:kind :import :start start :end end :import import :bindings bindings
                        :outputs (import-output-contexts import)})))
        (conj stages {:kind :authored :start cursor :end (count source-pairs)
                      :bindings (subvec source-pairs cursor)
                      :live-outs (mapv first (subvec source-pairs cursor))
                      :tail (vec (drop 2 source))})))))

(defn ^:no-doc authored-layout
  "Partition this phase's pre-normalization source by its exact retained cuts.
   Every authored binding is a live-out until a later assembly proves otherwise.
   Imported output declarations come from original emitted binding contexts.
   This query neither prepares segments nor grants assembly/import admission."
  [phase normalized]
  (when-let [context (view phase normalized)]
    (let [layout (authored-source-layout (:source context) (:imports context))]
      (entry/assert-producer-current! phase)
      layout)))

(defn- prepare-authored-layout
  "Prepare only authored regions, exporting every intermediate definition.
   The compiler callback transforms syntax; its return value is not admission.
   Imports retain original member objects and acquire explicit assembly offsets."
  [layout entry-tags options prepare-segment]
  (when-not (and (vector? layout) (seq layout)
                 (= :authored (:kind (peek layout))) (contains? (peek layout) :tail))
    (refuse! "authored preparation has no final source continuation" {}))
  (let [original-ids (mapv first (mapcat :bindings layout))
        entry-scope (set (keys entry-tags))
        reserved (into entry-scope original-ids)]
    (when-not (= (count original-ids) (count (set original-ids)))
      (refuse! "authored preparation requires unique source binding identities" {}))
    (loop [remaining layout environment entry-tags scope (set (keys entry-tags))
           occupied (set (keys entry-tags))
           output [] segments [] imports []]
      (if-let [{:keys [kind bindings live-outs tail] :as segment} (first remaining)]
        (case kind
          :authored
          (if (and (empty? bindings) (not (contains? segment :tail)))
            (recur (next remaining) environment scope occupied output
                   (conj segments (assoc segment :prepared-bindings [] :assembly-start (count output))) imports)
            (let [source (list* 'let* (vec (mapcat identity bindings))
                                (if (contains? segment :tail) tail [live-outs]))
                  missing (binding [util/*shadowing-locals* scope]
                            (remove scope (util/free-syms source)))
                  _ (when (seq missing)
                      (refuse! "authored segment refers outside its preceding lexical scope"
                               {:start (:start segment) :unavailable (set missing)}))
                  visible (into scope (map first) bindings)
                  segment-options (reduce (fn [options key]
                                            (if (contains? options key)
                                              (update options key select-keys visible) options))
                                          (assoc options :param-env environment)
                                          [:array-types :scalar-types])
                  segment-options (cond-> segment-options
                                    (contains? segment-options :values) (update :values select-keys entry-scope))
                  prepared (prepare-segment source segment-options)
                  prepared-pairs (pairs prepared)
                  prepared-ids (mapv first prepared-pairs)
                  own-ids (set (map first bindings))
                  prepared-tail (vec (drop 2 prepared))
                  defined (into scope (map first) prepared-pairs)
                  prepared-tags (reduce (fn [tags [id _]]
                                          (if-let [tag (types/sym-type-tag id)]
                                            (assoc tags id tag) (dissoc tags id)))
                                        {} prepared-pairs)
                  missing (binding [util/*shadowing-locals* scope]
                            (remove scope (util/free-syms prepared)))]
              (when (or (not= (count prepared-ids) (count (set prepared-ids)))
                        (some occupied prepared-ids)
                        (some #(and (contains? reserved %) (not (contains? own-ids %))) prepared-ids))
                (refuse! "authored preparation would capture another source binding"
                         {:start (:start segment) :bindings prepared-ids}))
              (when (seq missing)
                (refuse! "authored preparation introduced an unavailable value"
                         {:start (:start segment) :unavailable (set missing)}))
              (when (and (not (contains? segment :tail))
                         (not (and (= [live-outs] prepared-tail) (every? defined live-outs))))
                (refuse! "authored preparation changed its exported lexical boundary"
                         {:start (:start segment) :end (:end segment)}))
              (recur (next remaining)
                     (merge (apply dissoc environment live-outs) (select-keys prepared-tags live-outs))
                     (into scope live-outs)
                     (into occupied prepared-ids)
                     (into output prepared-pairs)
                     (conj segments (assoc segment :source source :prepared prepared
                                                  :prepared-bindings prepared-pairs
                                                  :assembly-start (count output)))
                     imports)))

          :import
          (let [import (:import segment)
                assembled (assoc import :assembly {:original-start (:start segment)
                                                  :start (count output) :prefix output})]
            (when (some occupied (map first bindings))
              (refuse! "import assembly would shadow an earlier binding" {:start (:start segment)}))
            (recur (next remaining)
                   (reduce (fn [environment context]
                             (if context (assoc environment (:output context) (:tag context)) environment))
                           environment (:outputs segment))
                   (into scope (keep :output) (:outputs segment))
                   (into occupied (map first) bindings)
                   (into output bindings)
                   (conj segments (assoc segment :assembly-start (count output)))
                   (conj imports assembled)))

          (refuse! "unknown authored preparation segment" {:kind kind}))
        (let [last-segment (peek segments)
              tail (if-let [prepared (:prepared last-segment)]
                     (drop 2 prepared) (:tail last-segment))]
          {:source (list* 'let* (vec (mapcat identity output)) tail)
           :segments segments :imports imports :environment environment})))))

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

(defn ^:no-doc prepare-scalar-input-region
  "Prepare one exact staged scalar input using its original bytecode emitter.
   Arguments remain the original physical ports in slot order, not reconstructed
   source expressions. This prepares execution only: the enclosing source owner
   still owes plan placement, dependency completion and whole-import closure.
   Reference inputs deliberately abstain; a resident buffer is not a JVM array."
  [phase normalized binding]
  (when-let [occurrence (input-occurrence phase normalized binding)]
    (let [{:keys [physical construction argument ports]} occurrence
          carrier (get-in argument [:contract :target-carrier])]
      (when (and (class? carrier) (.isPrimitive ^Class carrier)
                 (not (identical? Void/TYPE carrier)))
        (let [region (bytecode/compile-input-invocation-region
                      (:emission physical) (:input-body physical)
                      (:source-call construction) (:input-index argument))
              expected-parameters (mapv :parameter ports)]
          (when-not (and region
                         (= expected-parameters (:parameters region))
                         (= (into {} (map (juxt :parameter :physical-port)) ports)
                            (:physical-ports region))
                         ;; The generated recipe additionally pins the raw
                         ;; operand carrier. Outlining reports that carrier in
                         ;; its actual completion, not in the input contract.
                         (= (:contract argument)
                            (assoc (:input-contract region) :expected-input-carrier
                                   (get-in region [:input-completion :emitted-type])))
                         (= (:completion argument) (:input-completion region)))
            (refuse! "prepared scalar input differs from its original physical region"
                     {:id (get-in occurrence [:obligation :id])}))
          (let [invoke (:invoke region)
                region-guard (:guard region)
                guard (fn []
                        (entry/assert-producer-current! phase)
                        (region-guard))]
            (guard)
            (assoc region :occurrence occurrence :guard guard
                   :invoke (fn [& arguments]
                             (guard)
                             (let [result (apply invoke arguments)]
                               (guard)
                               result)))))))))

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

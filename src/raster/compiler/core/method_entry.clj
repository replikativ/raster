(ns raster.compiler.core.method-entry
  "MethodEntry — dispatch table entry as a plain map.

  Fields:
    :tags             — vector of type tags [doubles double ...]
    :arity            — argument count
    :check-classes    — vector of Class objects for runtime type checking
    :target-fn        — the backing function
    :specificity      — dispatch priority score
    :typed-impl       — deftype singleton for .invk, or nil
    :typed-iface      — typed interface symbol, or nil
    :typed-target-fn  — wrapper fn for typed dispatch, or nil
    :warning-meta     — metadata for warning suppression
    :has-compound-tags — true if any tag is compound
    :mangled-ns       — namespace symbol where mangled defn lives

  Previously a defrecord, now a plain map to avoid classloader conflicts.
  Maps use value equality across classloaders, eliminating ClassCastException
  when compiled (eval'd) code calls back into dispatch."
  (:require [raster.compiler.core.types :as types]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.support.mangled :as mangled]
            [raster.compiler.core.source-policy :as source-policy]))

(defn var-snapshot
  "Observe an existing Var's effective value and retained definition metadata.
   This is compiler-owner data, not a portable or authenticated certificate."
  [v]
  (let [is-bound (bound? v)]
    {:var v :bound? is-bound :value (when is-bound (var-get v)) :metadata (meta v)}))

(defn same-var-snapshot?
  "Compare observations without resolving names or executing the definition.
   A staleness guard must supply an actual fresh observation, not asserted data."
  [a b]
  (and (var? (:var a)) (var? (:var b))
       (identical? (:var a) (:var b))
       (= (:bound? a) (:bound? b))
       (identical? (:value a) (:value b))
       (= (:metadata a) (:metadata b))))

(defn emission-definition-snapshot
  "Observe definition inputs, excluding only this emitter's static cache output.
   Actually consumed static artifacts are retained separately by the emitter."
  [v]
  (assoc (update (var-snapshot v) :metadata dissoc :raster.core/deftm-bytecode)
         :snapshot-kind :emission-definition))

(def ^:dynamic *emission-inputs*
  "Request-local collector {:static [...], :values [...]} of consumed inputs,
   including definitions consulted by normalization before emission. The owner
   must validate collected observations before publication (and retain a guard
   for dynamic execution). This is neither a registry nor a cache."
  nil)

(defprotocol RetainedProducerInput
  "Process-local compiler dependency, separate from portable IR and Var snapshots.
   Validate this immutable producer's own freshness. Transitive dependencies are
   flattened into the enclosing collector, not recursively rediscovered."
  (assert-producer-current! [producer]))

(defn- assert-producer-inputs-current! [inputs]
  (doseq [producer (:producer-inputs inputs)]
    (when-not (satisfies? RetainedProducerInput producer)
      (throw (ex-info "compiler input has no retained producer guard"
                      {:reason :unowned-producer-input})))
    (assert-producer-current! producer))
  true)

(defn- retain-producer-inputs [inputs producers]
  (reduce (fn [inputs producer]
            (if (some #(identical? producer %) (:producer-inputs inputs))
              inputs
              (update inputs :producer-inputs (fnil conj []) producer)))
          inputs producers))

(defn- var-reference-current?
  "Match historical Var identity and original lookup sites.
   A detached Var can retain its old root after ns-unmap/reintern or namespace
   recreation; reobserving that object alone would bless a different callee."
  [{:keys [var metadata lookups]}]
  (let [source-ns (:ns metadata)
        source-name (:name metadata)]
    (and (var? var) (instance? clojure.lang.Namespace source-ns)
         (symbol? source-name)
         (identical? source-ns (find-ns (ns-name source-ns)))
         (identical? var (.findInternedVar ^clojure.lang.Namespace source-ns source-name))
         (every? (fn [[lookup-ns lookup-symbol]]
                   (and (instance? clojure.lang.Namespace lookup-ns)
                        (symbol? lookup-symbol)
                        (identical? lookup-ns (find-ns (ns-name lookup-ns)))
                        (identical? var (ns-resolve lookup-ns lookup-symbol)))) lookups))))

(defn var-value-current?
  "Compare a historical observation directly to the effective live Var value.
   Equivalent to same-var-snapshot? against a fresh observation, without building
   that temporary map. Name/lookup identity remains the caller's responsibility.
   Definition observations exclude only this emitter's static cache output."
  [{:keys [var value metadata snapshot-kind] :as snapshot}]
  (and (var? var)
       (let [is-bound (bound? var)]
         (and (= (:bound? snapshot) is-bound)
              (identical? value (when is-bound (var-get var)))
              (= metadata (if (= :emission-definition snapshot-kind)
                            (dissoc (meta var) :raster.core/deftm-bytecode)
                            (meta var)))))))

(defn var-snapshot-current?
  "Match both the original lookup and its effective definition observation."
  [snapshot]
  (and (var-reference-current? snapshot)
       (var-value-current? snapshot)))

(defn- retain-definition-observation [inputs v snapshot]
  (if-let [prior (get-in inputs [:definition-proofs v])]
    (do
      (when-not (same-var-snapshot? prior snapshot)
        (throw (ex-info "definition changed while deriving a compiler proof"
                        {:reason :unstable-definition-proof-input :source v})))
      (let [updated (update prior :lookups into (:lookups snapshot))]
        (-> inputs
            (assoc-in [:definition-proofs v] updated)
            ;; Emission artifacts serialize :values, not the request-local
            ;; proof index. Preserve unrelated entries and every lookup edge.
            (update :values
                    (fn [values]
                      (mapv #(if (identical? prior %) updated %) values))))))
    (-> inputs
        (assoc-in [:definition-proofs v] snapshot)
        (update :values (fnil conj []) snapshot))))

(defn observe-definition-input!
  "Retain a definition actually consulted by a normalization proof.
   Return false without a collecting owner: optional analysis must then decline.
   Keep the first observation and all lookup sites, never refresh a stale proof."
  [v lookup-ns lookup-symbol]
  (when *emission-inputs*
    (let [snapshot (assoc (emission-definition-snapshot v)
                          :lookups #{[lookup-ns lookup-symbol]})]
      (swap! *emission-inputs* retain-definition-observation v snapshot)
      true)))

(defn observe-source-body-input!
  "Record an actual source-body import in the existing producer collector.
   Origin selects metadata on the historical Var or its captured function value;
   it is not a second copy of the definition or an inferred compilation policy.
   Optional proof transactions discard this along with their other observations."
  [v origin]
  (when-not (contains? #{:var :value} origin)
    (throw (ex-info "source body import needs its actual metadata origin"
                    {:reason :source-body-input-origin :origin origin})))
  (when *emission-inputs*
    (observe-definition-input! v (:ns (meta v)) (:name (meta v)))
    (swap! *emission-inputs* update :source-body-inputs (fnil conj #{}) [v origin])
    true))

(defn assert-definition-inputs-current!
  "Validate normalization's historical definition observations, not a fresh
   post-hoc source traversal. Sequential staleness fence, not an ABA guarantee."
  [inputs]
  (doseq [snapshot (vals (:definition-proofs inputs))]
    (when-not (var-snapshot-current? snapshot)
      (throw (ex-info "compiler proof inputs changed before publication or execution"
                      {:reason :stale-definition-proof-input :source (:var snapshot)}))))
  (assert-producer-inputs-current! inputs)
  true)

(defn call-with-optional-definition-inputs
  "Retain observations only when an optional proof returns a truthy result.
   A refused query is not an executable dependency. The child starts with the
   parent's HISTORICAL observations, so it cannot refresh an already stale proof.
   This request-local transaction assumes sequential ownership; a competing
   owner update retries construction rather than losing either observation."
  [prove!]
  (if-not *emission-inputs*
    (prove!)
    (let [owner *emission-inputs*
          historical @owner
          child (atom historical)
          result (binding [*emission-inputs* child] (prove!))]
      (when result
        (assert-definition-inputs-current! @child)
        (when-not (compare-and-set! owner historical @child)
          (throw (ex-info "compiler proof owner changed during optional analysis"
                          {:reason :unstable-definition-proof-input}))))
      result)))

(defn adopt-definition-inputs!
  "Import a producer's retained observations into an enclosing compiler owner.
   Caller authenticates the producer; this map alone is not a certificate.
   Never reobserve definitions or replace an earlier observation. Preserve all
   lookup edges and unclosed imports. Without a collector, abstain."
  [historical]
  (when *emission-inputs*
    (let [owner *emission-inputs*
          prior-inputs @owner
          _ (assert-definition-inputs-current! historical)
          _ (assert-definition-inputs-current! prior-inputs)
          merged
          (reduce-kv retain-definition-observation
                     prior-inputs (or (:definition-proofs historical) {}))
          merged (reduce (fn [inputs key]
                           (if-let [facts (seq (get historical key))]
                             (update inputs key (fnil into #{}) facts)
                             inputs))
                         merged [:source-body-inputs :unclosed-source-imports])
          merged (retain-producer-inputs merged (:producer-inputs historical))]
      (assert-definition-inputs-current! merged)
      (when-not (compare-and-set! owner prior-inputs merged)
        (throw (ex-info "compiler proof owner changed during historical adoption"
                        {:reason :unstable-definition-proof-input})))
      true)))

(defn prepare-pending-definition-inputs!
  "Prepare observed pending JIT receivers before publishing a long-lived artifact.
   `prepare!` is the existing non-evaluating compiler entry, not a numerical call.
   Preparation has separate ownership. Afterwards ORIGINAL observations are checked:
   a promotion must discard/retry construction, never refresh the rejected result.
   Refused optional queries must not contribute observations to this owner."
  [prepare!]
  (when-not *emission-inputs*
    (throw (ex-info "definition preparation requires a compiler owner"
                    {:reason :missing-definition-preparation-owner})))
  (let [historical @*emission-inputs*
        pending (into [] (keep (fn [[v snapshot]]
                                 (when (map? (get-in snapshot [:metadata :raster.core/jit-spec]))
                                   v)))
                      (:definition-proofs historical))]
    (assert-definition-inputs-current! historical)
    ;; One batch may promote several entries. Do not expose the deliberately
    ;; stale enclosing owner to each dependency's own compilation transaction.
    (binding [*emission-inputs* nil]
      (doseq [v pending] (prepare! v)))
    (assert-definition-inputs-current! historical)))

(defn call-with-definition-inputs
  "Own normalization and emission together for a static compiled artifact.
   Dynamic executables additionally retain these observations in their existing
   execution guard. A nested owner contributes to the enclosing collector.
   Lazy compilation may replace a consulted definition: retry the WHOLE owning
   compilation at most twice, never refresh observations around old output.
   Resource-producing owners supply their own discard! for rejected results;
   compile! remains responsible for cleanup when it throws before returning."
  ([compile!] (call-with-definition-inputs compile! nil))
  ([compile! discard!]
  (let [attempt! (fn []
                   (let [result (compile!)]
                     (try
                       (assert-definition-inputs-current! @*emission-inputs*)
                       result
                       (catch Throwable failure
                         (when discard!
                           (try (discard! result)
                                (catch Throwable discard-failure
                                  (.addSuppressed discard-failure failure)
                                  (throw (ex-info "rejected compiler result could not be released"
                                                  {:reason :definition-proof-discard-failed}
                                                  discard-failure)))))
                         (throw failure)))))]
    (if *emission-inputs*
      ;; An enclosing compilation must restart from its own earliest proof,
      ;; not keep its stale observations while an inner owner retries.
      (attempt!)
      (loop [attempt 0]
        (let [outcome (try
                        {:result (binding [*emission-inputs* (atom {:static [] :values []})]
                                   (attempt!))}
                        (catch clojure.lang.ExceptionInfo failure {:failure failure}))]
          (if-let [failure (:failure outcome)]
            (if (and (< attempt 2)
                     (contains? #{:stale-definition-proof-input
                                  :unstable-definition-proof-input}
                                (:reason (ex-data failure))))
              (recur (inc attempt))
              (throw failure))
            (:result outcome))))))))

(declare validate-static-emission!)

(defn- validate-emission-inputs!
  "Fence reuse of an actual emitter's historical input observations.

   A fresh snapshot taken at differentiation time cannot certify an older
   emission. Require its original observations and reject a definition change
   across compilation, even if all currently observed roots happen to match.
   This checks sequential reuse, not concurrent mutation/ABA or call-site
   receiver correspondence; the owner must establish those separately."
  [{:keys [dependency-snapshots static-inputs definition-revision completed-definition-revision]
    :as emission} seen]
  (when-not (and (vector? dependency-snapshots)
                 (every? #(and (map? %) (var? (:var %))
                               (boolean? (:bound? %)) (map? (:metadata %))
                               (set? (:lookups %)) (seq (:lookups %))
                               (every? (fn [lookup]
                                         (and (vector? lookup) (= 2 (count lookup))
                                              (instance? clojure.lang.Namespace (first lookup))
                                              (symbol? (second lookup)))) (:lookups %)))
                         dependency-snapshots)
                 (vector? static-inputs)
                 (every? #(and (map? %) (var? (:source-var %)) (map? (:artifact %))) static-inputs)
                 (integer? definition-revision)
                 (integer? completed-definition-revision))
    (throw (ex-info "bytecode emission has no historical dependency observations"
                    {:reason :missing-emission-input-observations})))
  (when-not (= definition-revision completed-definition-revision)
    (throw (ex-info "compiler definitions changed during bytecode emission"
                    {:reason :unstable-emission-inputs
                     :before definition-revision :after completed-definition-revision})))
  (let [changed (filterv (complement var-snapshot-current?) dependency-snapshots)]
    (when (seq changed)
      (throw (ex-info "bytecode emission inputs no longer match their historical definitions"
                      {:reason :stale-emission-inputs
                       :dependencies (mapv (fn [{:keys [var]}]
                                             (symbol (str (:ns (meta var)))
                                                     (str (:name (meta var))))) changed)}))))
  (doseq [{:keys [source-var artifact]} static-inputs]
    (validate-static-emission! source-var artifact seen))
  (assert-producer-inputs-current! emission)
  emission)

(defn- validate-static-emission!
  [source cached ^java.util.IdentityHashMap seen]
  (let [{:keys [source-snapshot] :as inputs} (:raster.compiler/emission-inputs cached)]
      (when-not (and (map? inputs) (map? source-snapshot)
                     (identical? source (:var source-snapshot)))
        (throw (ex-info "static code has no historical input lineage"
                        {:reason :unowned-static-emission-inputs :source source})))
      ;; Owner correspondence is checked BEFORE memoization: the same artifact
      ;; cannot be attached to another Var and bypass its historical owner.
      (when-not (and (= :emission-definition (:snapshot-kind source-snapshot))
                     (var-snapshot-current? source-snapshot))
        (throw (ex-info "static code no longer matches its emitted source definition"
                        {:reason :stale-static-emission-source :source source})))
      (case (.get seen cached)
        :checked cached
        :checking (throw (ex-info "static emission input lineage is cyclic"
                                  {:reason :cyclic-static-emission-inputs}))
        (do
          (.put seen cached :checking)
          (validate-emission-inputs! inputs seen)
          (.put seen cached :checked)
          cached))))

(defn assert-emission-inputs-current!
  "Validate historical input observations and recursive static-code lineage.
   Sequential reuse fence only; not concurrent mutation/ABA authentication."
  [emission]
  (validate-emission-inputs! emission (java.util.IdentityHashMap.)))

(defn prepare-emission-input-guard!
  "Validate immutable historical lineage once, retaining its live reuse checks.
   Each invocation still checks the ORIGINAL lookup sites, effective Var values
   and metadata, including every consumed static source. No epoch shortcut or
   refreshed observation certifies old code. Sequential fence only, like
   assert-emission-inputs-current!; not concurrent mutation/ABA protection."
  [emission]
  (assert-emission-inputs-current! emission)
  (let [seen (java.util.IdentityHashMap.)
        checks (atom [])]
    (letfn [(retain! [inputs]
              (let [snapshots (:dependency-snapshots inputs)]
                (swap! checks conj
                       (fn [] (every? var-snapshot-current? snapshots))))
              (let [producers (:producer-inputs inputs)]
                (when (seq producers)
                  (swap! checks conj
                         (fn [] (assert-producer-inputs-current! {:producer-inputs producers})))))
              (doseq [{:keys [source-var artifact]} (:static-inputs inputs)]
                (let [nested (:raster.compiler/emission-inputs artifact)
                      snapshot (:source-snapshot nested)]
                  ;; Owner correspondence and acyclicity were validated above.
                  ;; Retain the source check even on a shared artifact edge.
                  (swap! checks conj
                         (fn [] (var-snapshot-current? snapshot)))
                  (when-not (.containsKey seen artifact)
                    (.put seen artifact true)
                    (retain! nested)))))]
      (retain! emission)
      (let [checks @checks]
        (fn []
          ;; Keep the full validator as the single diagnostic authority. The
          ;; immutable shape is specialized, not its live success/failure.
          (when-not (every? (fn [check] (check)) checks)
            (assert-emission-inputs-current! emission))
          emission)))))

(defn retryable-emission-change?
  "Lazy compilation may materialize an input definition during emission. Only
   these historical-input failures authorize a bounded fresh emission, never
   blessing the already emitted code with a new snapshot."
  [failure]
  (contains? #{:stale-emission-inputs :stale-static-emission-source
               :unstable-emission-inputs :unowned-static-emission-inputs}
             (:reason (ex-data failure))))

(defn assert-static-emission-current!
  "Validate one actual static artifact against its historical source and inputs.
   A fresh observation cannot certify an older unowned cache entry."
  [source cached]
  (validate-static-emission! source cached (java.util.IdentityHashMap.)))

(defn- retained-global-var [callee]
  (when (and (symbol? callee) (namespace callee))
    (when-let [source-ns (find-ns (symbol (namespace callee)))]
      (.findInternedVar ^clojure.lang.Namespace source-ns (symbol (name callee))))))

(defn selected-definition
  "Project the live registered definition of an already selected typed call.
   This is correspondence evidence, not a purity, type-inference or result
   presence rule. No method is selected from guessed operand types or invoked.
   Both the receiver and its delegated source root must still match registration.
   Unresolved/generic calls and legacy mutable-Var entries have no projection."
  [operation implementation-op argument-count]
  (let [generic (retained-global-var operation)
        receiver (retained-global-var implementation-op)
        table (:raster.core/dispatch-table (meta generic))]
    (when (and generic receiver table)
      (some
       (fn [{:keys [tags mangled-ns typed-impl target-fn] :as entry}]
         (when (and (some? typed-impl) (not (var? typed-impl))
                    (identical? typed-impl @receiver))
           (let [source-name (types/mangle (symbol (name operation)) tags)
                 source (when mangled-ns (retained-global-var (symbol (str mangled-ns) (str source-name))))
                 source-meta (meta source)
                 params (:raster.core/deftm-params source-meta)]
             (when (and source
                        (= implementation-op (symbol (str mangled-ns) (str source-name "-impl")))
                        (identical? target-fn @source)
                        (= tags (:raster.core/deftm-tags source-meta))
                        (= argument-count (count params) (count tags)))
               {:entry entry :operation operation :implementation-op implementation-op
                :source-var source :generic-var generic :receiver-var receiver :parameters params :tags tags
                :return-tag (:raster.core/return-tag source-meta)
                :source-body (:raster.core/deftm-source-body source-meta)
                :walked-body (:raster.core/deftm-walked-body source-meta)}))))
       (get @table argument-count)))))

(declare direct-selected-definition)

(defn capture-source-entry
  "Capture ORIGINAL inputs at the actual compiler producer, before publication.
   This owner-private lineage is not a portable certificate or a type registry.
   Mutable walked metadata/cache output are deliberately not source identity.
   A plain Clojure producer may additionally freeze its exact compiled body.
   Nil means the supplied plan does not correspond to this retained definition."
  [source-var {:keys [params tags annotations source-body source-ns typed-interface return-tag
                     clojure-body] :as source}]
  (when (and (var? source-var) (bound? source-var)
             (instance? clojure.lang.Namespace source-ns))
    (let [{:keys [value metadata]} (var-snapshot source-var)]
      (when (and (:raster.core/deftm metadata)
                 (source-policy/valid? (:source-policy source))
                 (= (:source-policy source) (:raster.core/source-policy metadata))
                 (identical? source-ns (:ns metadata))
                 (= (ns-name source-ns) (:raster.core/deftm-source-ns metadata))
                 (= params (:raster.core/deftm-params metadata))
                 (= tags (:raster.core/deftm-tags metadata))
                 (= annotations (:raster.core/deftm-annotations metadata))
                 (= return-tag (or (:raster.core/return-tag metadata) 'Object))
                 (some? source-body)
                 (identical? source-body (:raster.core/deftm-source-body metadata))
                 (or (some? typed-interface) (vector? clojure-body)))
        (cond-> {:source-var source-var :source-root value :source-name (:name metadata)
                 :params params :tags tags :annotations annotations :source-body source-body
                 :source-policy (:source-policy source)
                 ;; Interface spelling is nominal, not a physical Class proof.
                 ;; Pending specs use Symbols while emission callers may use
                 ;; Strings. Keep one representation in frozen source identity;
                 ;; selected receiver/Class correspondence is checked separately.
                 :source-ns source-ns :typed-interface (some-> typed-interface str) :return-tag return-tag}
          (vector? clojure-body) (assoc :clojure-body clojure-body))))))

(defn source-entry-current?
  "Validate historical source-entry lineage, not a fresh proof around old code.
   Exact raw-body identity preserves its metadata. Body-entry carriers do not
   grant successful invocation, totality, result presence or conversion motion."
  [lineage]
  (boolean
   (when (map? lineage)
     (let [{:keys [source-var source-root source-ns source-name]} lineage]
       (and (var? source-var) (bound? source-var)
            (instance? clojure.lang.Namespace source-ns)
            (symbol? source-name) (nil? (namespace source-name))
            (identical? source-ns (find-ns (ns-name source-ns)))
            (identical? source-var (.findInternedVar ^clojure.lang.Namespace source-ns source-name))
            (identical? source-root (var-get source-var))
            (some? (capture-source-entry source-var lineage)))))))

(defn assert-source-entry-current!
  "Fence publication of an actual captured source entry; sequential, not ABA-safe."
  [lineage]
  (when-not (source-entry-current? lineage)
    (throw (ex-info "typed source entry changed during compilation"
                    {:reason :stale-definition-proof-input :source (:source-var lineage)})))
  true)

(defn observed-selected-definition
  "Optional owner-local definition projection, observed BEFORE deriving facts.
   The first lookup discovers Vars only; never analyze its captured source.
   Reselect after historical snapshots, validate them, and return only the
   reselected definition. Consumers retain the enclosing owner's final guard
   after their analysis. This is a sequential staleness fence, not an ABA proof.
   Selection is :typed for an invk receiver or :direct for a mangled call."
  [selection operation implementation-op argument-count]
  (let [select! (case selection :typed selected-definition :direct direct-selected-definition)]
    (when *emission-inputs*
      (when-let [initial (select! operation implementation-op argument-count)]
        (doseq [[v lookup] [[(:generic-var initial) operation]
                            [(:receiver-var initial) implementation-op]
                            [(:source-var initial)
                             (symbol (str (:ns (meta (:source-var initial))))
                                     (str (:name (meta (:source-var initial)))))]]
                :when v]
          (observe-definition-input! v (:ns (meta v)) lookup))
        (let [definition (select! operation implementation-op argument-count)]
          (assert-definition-inputs-current! @*emission-inputs*)
          (when (and definition
                     (every? #(identical? (get initial %) (get definition %))
                             [:generic-var :receiver-var :source-var]))
            definition))))))

(defn selected-parameter-carriers
  "Observe the actual JVM interface of an already selected, registered invk.

   Owner-private correspondence evidence, not overload selection or a type-tag
   presence rule. Primitive parameters are primitive only INSIDE this method,
   after original argument completion. Reference parameters remain nullable and
   the actual Object return is never upgraded from a logical return annotation.
   The caller must establish that this is the source entry/call being analyzed,
   retain the owning definition observations, and preserve operand completion.
   No interface is generated, method invoked or registry added."
  [operation implementation-op argument-count]
  (when *emission-inputs*
    (when-let [{:keys [entry tags parameters] :as definition}
               (observed-selected-definition :typed operation implementation-op argument-count)]
      (let [{:keys [iface-name iface-class param-tags]}
            (get @types/fn-interface-by-name (:typed-iface entry))
            expected-tags (mapv types/tag->iface-type tags)
            expected-carriers (mapv types/tag->java-class expected-tags)]
        (when (and (= iface-name (:typed-iface entry))
                   (= expected-tags param-tags)
                   (class? iface-class) (.isInterface ^Class iface-class)
                   (.isInstance ^Class iface-class (:typed-impl entry))
                   (every? class? expected-carriers))
          (let [methods (filter #(= "invk" (.getName ^java.lang.reflect.Method %))
                                (.getDeclaredMethods ^Class iface-class))]
            (when (= 1 (count methods))
              (let [^java.lang.reflect.Method method (first methods)
                    carriers (vec (.getParameterTypes method))]
                (when (and (= expected-carriers carriers)
                           (= argument-count (count parameters) (count carriers))
                           (not (.isVarArgs method))
                           (not (java.lang.reflect.Modifier/isStatic (.getModifiers method))))
                  (assert-definition-inputs-current! @*emission-inputs*)
                  (when (identical? entry
                                    (:entry (selected-definition operation implementation-op argument-count)))
                    {:definition definition :invocation-method method
                     :parameters parameters :parameter-carriers carriers
                     :result-carrier (.getReturnType method)}))))))))))

(defn- usable-emission?
  "Optional analyses may abstain on unavailable historical evidence. Execution
   and publication still use the throwing guards; invariant failures propagate."
  [emission]
  (try
    (assert-emission-inputs-current! emission)
    true
    (catch clojure.lang.ExceptionInfo failure
      (if (or (retryable-emission-change? failure)
              (= :missing-emission-input-observations (:reason (ex-data failure))))
        false
        (throw failure)))))

(defn- source-entry-matches-selection?
  [lineage {:keys [source-var parameters tags entry]} expected-source-var]
  (and lineage (identical? expected-source-var source-var)
       (identical? source-var (:source-var lineage))
       (= parameters (:params lineage)) (= tags (:tags lineage))
       (= (str (:typed-iface entry)) (str (:typed-interface lineage)))))

(defn selected-source-entry-carriers
  "Project primitive formal facts INSIDE an exact retained typed source entry.
   Cold jit-spec and warm owned emission must correspond to the same source Var
   supplied by the frontend. Never compile/invoke merely to acquire evidence.
   Original argument completion stays outside these facts; reference formals
   remain nullable and the interface's Object result remains Object. Requires
   the existing enclosing definition owner and its final publication guard."
  [operation implementation-op argument-count expected-source-var]
  (when-let [{:keys [definition parameters parameter-carriers] :as projection}
             (selected-parameter-carriers operation implementation-op argument-count)]
    (let [{:keys [source-var receiver-var entry tags]} definition
          metadata (meta receiver-var)
          spec (:raster.core/jit-spec metadata)
          emission (:raster.core/bytecode-emission metadata)
          source-ns (:ns (meta source-var))
          lineage
          (when (identical? expected-source-var source-var)
            (if spec
              (when (and (symbol? (:mangled-sym spec))
                         (identical? source-var (ns-resolve source-ns (:mangled-sym spec)))
                         (= (ns-name source-ns) (:source-ns-name spec)))
                (capture-source-entry
                 source-var (assoc spec :source-ns source-ns
                                        :typed-interface (:typed-iface-name spec))))
              (when (and (= :jvm-bytecode (:backend emission))
                         (identical? (:typed-impl entry) (:implementation emission))
                         (= parameters (:params emission)) (= tags (:tags emission))
                         (= (str (:typed-iface entry)) (str (:typed-interface emission))))
                (when (and (source-entry-current? (:source-entry emission))
                           (usable-emission? emission))
                  (:source-entry emission)))))]
      (when (source-entry-matches-selection? lineage definition expected-source-var)
        (assert-definition-inputs-current! @*emission-inputs*)
        (assoc projection :scope :typed-source-body-entry :source-entry lineage
               :pending-source? (boolean spec) :emission (when-not spec emission)
               :parameter-facts
               (zipmap parameters
                       (mapv (fn [^Class carrier]
                               {:carrier carrier :presence (if (.isPrimitive carrier) :present :unknown)})
                             parameter-carriers)))))))

(deftype ^:private SourceFrontend [lineage body context preparation-inputs])
(deftype ^:private FrontendSourceOwner [frontend implementation])

(defn freeze-source-frontend
  "Original producer: freeze semantic source before choosing a target module.
   Preparation inputs fence publication, not later macro redefinition: the
   already-expanded body retains its meaning just like compiled Clojure code."
  [lineage body context preparation-inputs]
  (assert-source-entry-current! lineage)
  (assert-definition-inputs-current! preparation-inputs)
  (when-not (and (vector? body) (map? preparation-inputs)
                 (= #{:element-dtype :parameter-dtype} (set (keys context))))
    (throw (ex-info "frontend producer needs its original body and context"
                    {:reason :unclosed-original-frontend})))
  (SourceFrontend. lineage body context preparation-inputs))

(defn bind-source-frontend
  "Producer-owned correspondence, separate from the frozen semantic frontend.
   A pending binding identifies its stub/spec, not an executed numerical body."
  [frontend implementation]
  (when-not (and (instance? SourceFrontend frontend) implementation)
    (throw (ex-info "frontend binding requires an original producer"
                    {:reason :unclosed-original-frontend})))
  (assert-source-entry-current! (.-lineage ^SourceFrontend frontend))
  (FrontendSourceOwner. frontend implementation))

(defn own-frontend-source
  "Original producer boundary: bind pre-backend source to its actual implementation.
   The opaque owner is not reconstructed from a later emission-metadata map."
  [lineage implementation body context]
  (bind-source-frontend (freeze-source-frontend lineage body context {}) implementation))

(defn frontend-source-input!
  "Read immutable producer inputs for this exact pending/published implementation."
  [owner implementation]
  (when-not (and (instance? FrontendSourceOwner owner)
                 (identical? implementation (.-implementation ^FrontendSourceOwner owner)))
    (throw (ex-info "frontend source is not owned by this original implementation"
                    {:reason :retained-frontend-owner-mismatch})))
  (let [frontend (.-frontend ^FrontendSourceOwner owner)
        lineage (.-lineage ^SourceFrontend frontend)]
    (assert-source-entry-current! lineage)
    {:source-frontend frontend :source-entry lineage :source-owner owner
     :implementation implementation :body (.-body ^SourceFrontend frontend)
     :frontend-context (.-context ^SourceFrontend frontend)}))

(defn- original-frontend [owner lineage implementation expected-dtype]
  (when owner
    (let [original (frontend-source-input! owner implementation)]
      (when-not (identical? lineage (:source-entry original))
        (throw (ex-info "frontend source is not owned by this original implementation"
                        {:reason :retained-frontend-owner-mismatch})))
      (let [context (:frontend-context original)]
        (when-not (or (nil? (:parameter-dtype context))
                      (= expected-dtype (:parameter-dtype context)))
          (throw (ex-info "retained source specialization cannot be retyped for this request"
                          {:reason :retained-frontend-dtype-mismatch
                           :source (:source-var lineage) :requested-dtype expected-dtype
                           :source-dtype (:parameter-dtype context)})))
        original))))

(defn assert-frontend-source!
  "Check original producer identity, not asserted body/context fields in a map."
  [frontend source-var]
  (let [{:keys [source-owner source-entry implementation frontend-context body]} frontend
        expected (original-frontend source-owner source-entry implementation
                                    (:parameter-dtype frontend-context))]
    (when-not (and expected (identical? source-var (:source-var source-entry))
                   (identical? (:source-frontend frontend) (:source-frontend expected))
                   (identical? body (:body expected))
                   (= frontend-context (:frontend-context expected)))
      (throw (ex-info "checked source requires its original frontend producer"
                      {:reason :unowned-checked-result-frontend})))
    frontend))

(defn selected-frontend-source
  "Original typed frontend of an exactly selected specialization.
   Its opaque original owner authenticates source independently of any transformed
   module's freshness. This permits new source compilation, not invocation of an
   old module or observed-bytecode analysis. Optional :emission is an unvalidated
   observation; every emission consumer must check its own historical guard.
   Missing cold/legacy ownership abstains without walking or emitting code."
  [operation implementation-op argument-count source-var expected-dtype]
  (call-with-optional-definition-inputs
   (fn []
     (let [projection (selected-parameter-carriers operation implementation-op argument-count)
           direct-op (symbol (str (:ns (meta source-var))) (str (:name (meta source-var))))
           direct (when-not projection
                    (observed-selected-definition :direct operation direct-op argument-count))
           spec (some-> projection :definition :receiver-var meta :raster.core/jit-spec)
           pending? (boolean spec)
           receiver (get-in projection [:definition :receiver-var])
           emission (when (and projection (not pending?))
                      (:raster.core/bytecode-emission (meta receiver)))
           owner (if projection (if pending? (:frontend-source-owner spec)
                                   (:frontend-source-owner emission))
                     (:raster.core/frontend-source-owner (meta source-var)))
           ;; Authenticate against the actual selected receiver, never an
           ;; implementation identity asserted by an editable emission map.
           implementation (if projection (get-in projection [:definition :entry :typed-impl])
                              (when direct (var-get source-var)))
           original (when (and owner implementation)
                      (frontend-source-input! owner implementation))
           lineage (if projection (:source-entry original)
                       (when (identical? source-var (:source-var direct))
                         (:raster.core/clojure-source-entry (meta source-var))))]
       (when (and lineage
                  (or (not projection)
                      (source-entry-matches-selection? lineage (:definition projection) source-var)))
         (when-let [frontend (original-frontend owner lineage implementation expected-dtype)]
           (observe-source-body-input! source-var :var)
           (assert-definition-inputs-current! @*emission-inputs*)
           (merge {:source-entry lineage :definition (or (:definition projection) direct)
                   :source-owner owner :implementation implementation
                   :emission emission} frontend)))))))

(defn frontend-reduction-emission
  "Observe an unchanged frozen frontend reduction through its actual emitter.
   Source ownership, input occurrence identity and emitted completion are separate
   proofs. Cold/legacy entries and transformed emitter inputs abstain; this query
   never materializes bytecode or infers correspondence from equal expressions."
  [frontend original]
  (let [{:keys [emission implementation source-owner source-entry body definition]} frontend
        ;; The opaque origin relation seals bodies, not editable observation maps.
        ;; Reuse the ALREADY observed receiver metadata, never refresh it around
        ;; a caller-supplied emission or an older materialized implementation.
        historical (when *emission-inputs*
                     (get-in @*emission-inputs*
                             [:definition-proofs (:receiver-var definition)
                              :metadata :raster.core/bytecode-emission]))]
    (assert-frontend-source! frontend (:source-var source-entry))
    (when (and emission (identical? emission historical)
               (identical? implementation (:implementation emission))
               (identical? source-owner (:frontend-source-owner emission))
               (identical? source-entry (:source-entry emission)))
      ;; Source availability did not attest this transformed module. Actual
      ;; instruction/occurrence consumers still owe its complete input history.
      (assert-emission-inputs-current! emission)
      (assert-definition-inputs-current! @*emission-inputs*)
      (let [observed
            ((requiring-resolve 'raster.compiler.backend.jvm.bytecode/retained-input-reduction-emission)
             emission body original)]
        (assert-definition-inputs-current! @*emission-inputs*)
        observed))))

(defn original-source-frontend
  "Optional original frontend for an exact already-resolved deftm source Var.
   Shared mangling only discovers the operation; selected correspondence proves it."
  [source-var]
  (when (and *emission-inputs* (var? source-var))
    (let [metadata (meta source-var)
          source-op (symbol (str (:ns metadata)) (str (:name metadata)))
          operation (mangled/impl->op source-op)]
      (when operation
        (selected-frontend-source operation
                                  (symbol (namespace source-op) (str (name source-op) "-impl"))
                                  (count (:raster.core/deftm-params metadata)) source-var
                                  (dtype/infer-dtype-from-tags (:raster.core/deftm-tags metadata)))))))

(defn selected-definition-body
  "Historical body for an already observed selected definition, never current
   walked metadata. Typed methods retain actual receiver/emission correspondence.
   Plain deftm declarations freeze the exact Clojure body at their producer.
   Missing historical evidence abstains without compiling or invoking anything."
  [{:keys [source-var receiver-var operation implementation-op parameters] :as definition}]
  (when (and *emission-inputs* definition)
    (let [projection (when receiver-var
                       (selected-source-entry-carriers operation implementation-op
                                                       (count parameters) source-var))
          lineage (if receiver-var
                    (:source-entry projection)
                    (:raster.core/clojure-source-entry (meta source-var)))
          body (when (and (identical? source-var (:source-var lineage))
                          (identical? (:source-body definition) (:source-body lineage))
                          (source-entry-current? lineage))
                 (if receiver-var
                   (if (:pending-source? projection)
                     (:source-body lineage)
                     (get-in projection [:emission :body]))
                   (:clojure-body lineage)))]
      (when (seq body)
        (observe-source-body-input! source-var :var)
        (assert-definition-inputs-current! @*emission-inputs*)
        {:body body :parameters (:params lineage) :tags (:tags lineage)
         :source-var source-var :source-ns (:source-ns lineage)
         :pending-source? (boolean (:pending-source? projection))
         :dependency-snapshots (get-in projection [:emission :dependency-snapshots])}))))

(defn original-body-entry-carriers
  "Actual parameters of the original typed or plain Clojure body entry.
   Plain invokeStatic signatures come from the historically owned target class,
   not inferred dtypes. An array source hint on an Object port is an explicit
   :required-array-class obligation, never an unconditional payload fact: the
   execution owner must check it before using shape projections or doing work."
  [operation implementation-op argument-count expected-source-var]
  (or (selected-source-entry-carriers operation implementation-op argument-count expected-source-var)
      (call-with-optional-definition-inputs
       (fn []
         (when-let [definition (observed-selected-definition :direct operation implementation-op argument-count)]
           (when (and (identical? expected-source-var (:source-var definition))
                      (selected-definition-body definition))
             (let [lineage (:raster.core/clojure-source-entry (meta expected-source-var))
                   target (:source-root lineage)
                   methods (when (and target (source-entry-current? lineage))
                             (filter (fn [^java.lang.reflect.Method method]
                                       (and (= "invokeStatic" (.getName method))
                                            (java.lang.reflect.Modifier/isStatic (.getModifiers method))
                                            (java.lang.reflect.Modifier/isPublic (.getModifiers method))
                                            (not (.isVarArgs method))
                                            (= argument-count (.getParameterCount method))))
                                     (.getDeclaredMethods (class target))))]
               (when (= 1 (count methods))
                 (let [method (first methods)
                       carriers (vec (.getParameterTypes ^java.lang.reflect.Method method))]
                   (assert-definition-inputs-current! @*emission-inputs*)
                   {:scope :clojure-source-body-entry :definition definition
                    :parameters (:parameters definition) :parameter-carriers carriers
                    :invocation-method method :result-carrier (.getReturnType ^java.lang.reflect.Method method)
                    :parameter-facts
                    (into {} (map (fn [symbol ^Class carrier tag]
                                    [symbol (cond-> {:carrier carrier
                                                     :presence (if (.isPrimitive carrier) :present :unknown)}
                                              (and (= Object carrier) (get types/primitive-array-info tag))
                                              (assoc :required-array-class (get types/primitive-array-info tag)))])
                                  (:parameters definition) carriers (:tags definition)))})))))))))

(defn frontend-body-entry-carriers
  "Actual body-entry carriers for this exact opaque original frontend.
   Re-observe the registered selection before consulting physical entry facts;
   the editable frontend :definition is never carrier authority. This neither
   emits nor invokes code and requires the enclosing definition owner. Missing
   physical evidence abstains; contradictory or changed ownership throws."
  [frontend source-var]
  (assert-frontend-source! frontend source-var)
  (let [current (original-source-frontend source-var)
        selected (:definition current)
        claimed (:definition frontend)
        selection-matches?
        (and current
             (every? #(identical? (get frontend %) (get current %))
                     [:source-frontend :source-owner :implementation :source-entry])
             (= (select-keys claimed [:operation :implementation-op :parameters :tags])
                (select-keys selected [:operation :implementation-op :parameters :tags]))
             (every? #(identical? (get claimed %) (get selected %))
                     [:source-var :generic-var :receiver-var :entry]))]
    (when-not selection-matches?
      (throw (ex-info "frontend physical entry differs from its original producer"
                      {:reason :retained-frontend-entry-mismatch})))
    (let [physical (original-body-entry-carriers
                    (:operation selected) (:implementation-op selected)
                    (count (:parameters selected)) source-var)
          definition (:definition physical)
          actual-implementation (if (:receiver-var definition)
                                  (get-in definition [:entry :typed-impl])
                                  (get-in definition [:entry :target-fn]))]
      (when (and physical
                 (not (and (source-entry-matches-selection? (:source-entry frontend) definition source-var)
                           (identical? (:entry selected) (:entry definition))
                           (identical? (:implementation frontend) actual-implementation))))
        (throw (ex-info "physical carrier facts belong to another frontend entry"
                        {:reason :retained-frontend-entry-mismatch})))
      (assert-definition-inputs-current! @*emission-inputs*)
      physical)))

(defn observed-body-operation
  "Resolve a nonlexical call only with the selected body's lookup authority.
   Pending source may observe current resolution before its eventual compilation.
   An already compiled body needs its ORIGINAL lookup edge; a current namespace
   mapping cannot certify old code. Retain accepted historical inputs in the new
   artifact's existing owner. Missing legacy or plain-Clojure edges abstain."
  [{:keys [source-ns pending-source? dependency-snapshots]} operation locals]
  (when (and *emission-inputs* (instance? clojure.lang.Namespace source-ns)
             (symbol? operation) (not (special-symbol? operation))
             (not= '.invk operation) (not (contains? locals operation)))
    (let [lookup [source-ns operation]
          historical (when-not pending-source?
                       (some #(when (contains? (:lookups %) lookup) %) dependency-snapshots))
          v (if pending-source?
              (ns-resolve source-ns operation)
              (:var historical))]
      (when (and (var? v)
                 (or pending-source? (var-snapshot-current? historical)))
        (observe-definition-input! v source-ns operation)
        ;; Never refresh an old snapshot merely because a new observation was
        ;; possible. Both owners must describe the same historical lookup.
        (when (or pending-source? (var-snapshot-current? historical))
          (assert-definition-inputs-current! @*emission-inputs*)
          (let [metadata (if pending-source?
                           (get-in @*emission-inputs* [:definition-proofs v :metadata])
                           (:metadata historical))]
            (symbol (str (ns-name (:ns metadata))) (str (:name metadata)))))))))

(defn direct-selected-definition
  "Correspondence for an already selected plain mangled function call.
   The dispatcher and table must have their producer-owned identity; the exact
   registered invocation target must still occupy the mangled Var. This selects
   no overload, executes nothing and does not infer presence or argument types."
  [operation implementation-op argument-count]
  (let [generic (retained-global-var operation)
        source (retained-global-var implementation-op)
        table (:raster.core/dispatch-table (meta generic))
        owner (:raster.core/dispatch-owner (meta generic))]
    (when (and generic source table
               (identical? generic (:var owner))
               (identical? table (:table owner))
               (bound? generic) (identical? (var-get generic) (:root owner)))
      (some
       (fn [{:keys [tags mangled-ns target-fn typed-target-fn] :as entry}]
         (let [name (types/mangle (symbol (name operation)) tags)
               metadata (meta source)
               parameters (:raster.core/deftm-params metadata)]
           (when (and (= implementation-op (symbol (str mangled-ns) (str name)))
                      (or (nil? typed-target-fn) (identical? target-fn typed-target-fn))
                      (bound? source) (identical? target-fn (var-get source))
                      (= tags (:raster.core/deftm-tags metadata))
                      (= argument-count (count parameters) (count tags)))
             {:entry entry :operation operation :implementation-op implementation-op
              :source-var source :generic-var generic :parameters parameters :tags tags
              :return-tag (:raster.core/return-tag metadata)
              :source-body (:raster.core/deftm-source-body metadata)
              :walked-body (:raster.core/deftm-walked-body metadata)})))
       (get @table argument-count)))))

(def ^:dynamic *compilation-classloader*
  "When bound, ALL defineClass calls during BC compilation use this DCL.
  Prevents class identity mismatches from nested eval/require."
  nil)

(def ^:dynamic *bytecode-compiler*
  "When bound, compile-typed-impl! function for lazy JIT bytecode upgrade.
  Set by bytecode.clj at load time. Used by core.clj's do-bytecode-upgrade!."
  nil)

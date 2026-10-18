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
  (:require [raster.compiler.core.types :as types]))

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

(defn var-snapshot-current?
  "Match both the original lookup and its effective definition observation."
  [snapshot]
  (and (var-reference-current? snapshot)
       (same-var-snapshot? snapshot
                           (if (= :emission-definition (:snapshot-kind snapshot))
                             (emission-definition-snapshot (:var snapshot))
                             (var-snapshot (:var snapshot))))))

(defn observe-definition-input!
  "Retain a definition actually consulted by a normalization proof.
   Return false without a collecting owner: optional analysis must then decline.
   Keep the first observation and all lookup sites, never refresh a stale proof."
  [v lookup-ns lookup-symbol]
  (when *emission-inputs*
    (let [snapshot (assoc (emission-definition-snapshot v)
                          :lookups #{[lookup-ns lookup-symbol]})]
      (swap! *emission-inputs*
             (fn [inputs]
               (if-let [prior (get-in inputs [:definition-proofs v])]
                 (do
                   (when-not (same-var-snapshot? prior snapshot)
                     (throw (ex-info "definition changed while deriving a compiler proof"
                                     {:reason :unstable-definition-proof-input :source v})))
                   (let [updated (update prior :lookups into (:lookups snapshot))]
                     (-> inputs
                         (assoc-in [:definition-proofs v] updated)
                         ;; Emission artifacts serialize :values, not the
                         ;; request-local proof index. Retain every lookup edge.
                         (update :values
                                 (fn [values]
                                   (mapv #(if (identical? prior %) updated %) values))))))
                 (-> inputs
                     (assoc-in [:definition-proofs v] snapshot)
                     (update :values (fnil conj []) snapshot)))))
      true)))

(defn assert-definition-inputs-current!
  "Validate normalization's historical definition observations, not a fresh
   post-hoc source traversal. Sequential staleness fence, not an ABA guarantee."
  [inputs]
  (doseq [snapshot (vals (:definition-proofs inputs))]
    (when-not (var-snapshot-current? snapshot)
      (throw (ex-info "compiler proof inputs changed before publication or execution"
                      {:reason :stale-definition-proof-input :source (:var snapshot)}))))
  true)

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

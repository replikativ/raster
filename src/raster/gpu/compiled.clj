(ns raster.gpu.compiled
  "The `Compiled` artifact — a functional, inspectable GPU program value (S4 §2).

   `(r/compile #'train-step args opts)` returns a `Compiled` that implements `IFn`: calling it
   replays the resident program and returns device values, not host arrays. For composition,
   `(r/lower ...)` returns an allocation-free `Prepared`; independently lowered values compose
   through semantic boundaries and only the composite is instantiated. Resident descriptors and
   compositions are certified into the same LinkPlan/LinkedExecutable path.

   The three artifact primitives, honestly scoped to the whole-program MVP:
     • device value      — outputs are DeviceArrays over resident buffers; no host round-trip.
     • functional invoke  — `(step inputs)` → `{out-key → DeviceArray}`; mutation of resident
                            :state is invisible (the old input value is consumed/invalidated).
     • donation           — a donated in→out pair reuses the resident buffer; the input value
                            is marked consumed (reads throw), the output value is fresh.

   Roles are DERIVED (§4.2): `:donate` syms → :state (donated), `:constants` syms → :constant
   (captured once at bind, never per-call), the remainder default to the descriptor's derived
   read-only→:input / written→:output. Backend-neutral: `:target` selects the runtime; nothing
   here hardcodes `:ze`."
  (:refer-clojure :exclude [compile])
  (:require [clojure.set :as set]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.equation-artifact-store :as equation-artifact-store]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.build-manifest :as build-manifest]
            [raster.compiler.backend.gpu.storage-representation :as storage-probe]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.link-composition :as link-composition]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.source-dependencies :as source-dependencies]
            [raster.core :as rcore]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.schedule :as gpu-schedule]
            [raster.gpu.value :as v]
            [raster.runtime.numerical-content :as numerical-content])
  (:import [java.lang.foreign MemorySegment]))

(declare invoke-compiled)

;; ================================================================
;; The record
;; ================================================================

(defrecord Compiled
           [lowering     ;; certified resident or composition lowering into LinkPlan
            executable  ;; LinkedExecutable: the sole resident runtime representation
            in-tree      ;; ordered [{:key :sym :role :donate? :shape :dtype} …] — the arg spec
            out-tree     ;; ordered [{:key :sym :shape :dtype :from} …] — the result spec (multi-output)
            donated      ;; {in-key → out-key} — the alias plan (JAX input_output_aliases)
            schedule     ;; the S6 Schedule map (reserved; nil until S6 fills it)
            target       ;; device-id + (future) HardwareDescriptor
            descriptor   ;; raw descriptor or inspectable composite descriptor
            args         ;; captured specialization arguments (empty for a composite)
            preparation-report ;; compact host-side lowering/cache/certification timings
            prepared     ;; original compiler-owned Prepared, retained for explicit evidence
            provenance-seal
            live-outputs] ;; atom holding the DeviceArrays projected by the LAST invocation. They
                          ;; alias resident buffers the next replay overwrites, so they are
                          ;; invalidated (marked dead) at the start of the next invoke and at close!
                          ;; — otherwise a retained old output would silently observe a mutation.
  clojure.lang.IFn
  (invoke [this inputs] (invoke-compiled this inputs))
  (invoke [this] (invoke-compiled this {}))
  (applyTo [this argseq] (apply invoke-compiled this argseq)))

(defrecord Prepared
           [lowering in-tree out-tree donated schedule target descriptor args preparation-report
            provenance-seal])

(defn compiled? [x] (instance? Compiled x))
(defn prepared? [x] (instance? Prepared x))

(def ^:private artifact-seal-token (Object.))

(defn- seal-artifact
  "Bind in-process provenance to one exact immutable artifact object. A copied or associated
   record may retain the closure but cannot satisfy its identity check. This is an optimization
   witness, never a replacement for the public, re-derivable lowering certificate."
  [artifact]
  (let [owner (volatile! nil)
        sealed (assoc artifact :provenance-seal
                      (fn [candidate]
                        (when (identical? candidate @owner) artifact-seal-token)))]
    (vreset! owner sealed)
    sealed))

(defn- sealed-artifact? [artifact]
  (let [seal (:provenance-seal artifact)]
    (and (fn? seal) (identical? artifact-seal-token (seal artifact)))))

(defrecord CompletedEvidence [data lease executable epoch values released? provenance-seal]
  java.io.Closeable
  (close [this]
    (when-not (sealed-artifact? this)
      (throw (ex-info "completed evidence requires its original owner"
                      {:reason :compiled-completed-evidence-owner})))
    (locking (:lifetime-lock executable)
      (when (compare-and-set! released? false true)
        (try (doseq [value (vals values)] (v/free! value))
             (finally (.close ^java.io.Closeable lease))))))
  clojure.lang.IDeref
  (deref [this]
    (when-not (sealed-artifact? this)
      (throw (ex-info "completed evidence requires its original owner"
                      {:reason :compiled-completed-evidence-owner})))
    data))

(defn completed-evidence? [value]
  (and (instance? CompletedEvidence value) (sealed-artifact? value)))

(defn completed-output-values
  "Return pinned DeviceArray outputs from original completed evidence, until it is closed.
   Dereferencing evidence itself returns historical portable metadata, including after close."
  [evidence]
  (when-not (completed-evidence? evidence)
    (throw (ex-info "completed output access requires original evidence"
                    {:reason :compiled-completed-evidence-owner})))
  (locking (:lifetime-lock (:executable evidence))
    @(:lease evidence)
    (:values evidence)))

;; Compilation templates are immutable and argument-independent.  LinkPlan lowering below still
;; runs for every invocation, so shapes, weights, roles, views, and ownership never enter this
;; process-local cache or leak between instances.
(defonce ^:private compilation-template-cache (atom {}))
(defonce ^:private resident-plan-template-cache (atom {}))
(defonce ^:private compilation-template-stats
  (atom {:hits 0 :misses 0 :misses-by-reason {}
         :compilations 0 :failures 0 :compile-nanos 0}))

(def ^:private template-identity-schema :raster.compiled/template-identity-v1)

(deftype WeakIdentity [^java.lang.ref.WeakReference reference ^int identity-hash]
  Object
  (hashCode [_] identity-hash)
  (equals [this other]
    (or (identical? this other)
        (and (instance? WeakIdentity other)
             (let [root (.get reference)
                   other-root (.get (.-reference ^WeakIdentity other))]
               (and (some? root) (identical? root other-root)))))))

(defn- weak-identity [root]
  ;; A hash only selects a map bucket. Actual process-cache reuse must compare the live Var
  ;; roots by identity, even when two roots happen to have the same 32-bit identity hash.
  (WeakIdentity. (java.lang.ref.WeakReference. root)
                 (System/identityHashCode root)))

(def ^:dynamic *compilation-template-observer*
  "Internal per-request observer. It receives only cache/timing facts, never source or artifacts."
  nil)

(def ^:dynamic ^:private *compilation-template-owner*
  "Per-request slot for the exact stable process-cache owner; never part of public reports."
  nil)

(def ^:dynamic *equation-artifact-store*
  "Persistent equation-first store. Bind to a bounded temporary store in tests."
  (equation-artifact-store/make-store))

(defn clear-compilation-cache!
  "Clear reusable compiler templates and their counters. Runtime modules, buffers, and tuning
   results belong to their respective caches and are deliberately unaffected."
  []
  (reset! compilation-template-cache {})
  (reset! resident-plan-template-cache {})
  (reset! compilation-template-stats
          {:hits 0 :misses 0 :misses-by-reason {}
           :compilations 0 :failures 0 :compile-nanos 0})
  nil)

(defn compilation-cache-stats
  "Small, source-free instrumentation for structural compilation reuse."
  []
  (let [entries @compilation-template-cache]
    (assoc @compilation-template-stats
           :entries (count entries)
           :entries-by-compiler (frequencies (map (comp :compiler val) entries)))))

(defn- resident-plan-miss-reason [entries key]
  (if (and (:compiler-template-fingerprint key)
           (some #(= (:compiler-template-fingerprint %)
                     (:compiler-template-fingerprint key))
                 (keys entries)))
    :specialization
    :compulsory))

(defn- cached-resident-plan-template
  [key thunk]
  (let [candidate (delay (resident-plan/source-free-template (thunk)))
        [before after]
        (swap-vals! resident-plan-template-cache
                    #(if (contains? % key) % (assoc % key candidate)))
        hit? (contains? before key)
        miss-reason (when-not hit? (resident-plan-miss-reason before key))
        started (System/nanoTime)
        entry (get after key)]
    (try
      {:template @entry :cache-hit? hit?
       :miss-reason miss-reason
       :resolution-ns (- (System/nanoTime) started)}
      (catch Throwable error
        (swap! resident-plan-template-cache
               #(if (identical? entry (get % key)) (dissoc % key) %))
        (throw error)))))

(defn- qualified-var-symbol [v]
  (let [{:keys [ns name]} (meta v)]
    (when (and ns name) (symbol (str (ns-name ns)) (str name)))))

(defn- source-specialization-identity
  [fn-var dtype]
  (let [resolved (or (try
                       (equation-first/physical-function fn-var dtype)
                       (catch clojure.lang.ExceptionInfo _ nil))
                     fn-var)
        metadata (meta resolved)
        source-body (:raster.core/deftm-source-body metadata)
        build (build-manifest/current-identity)
        dependency-evidence
        (when (and source-body (:complete? build))
          (source-dependencies/manifest
           resolved dtype
           {:build-owned-namespaces (get-in build [:manifest :source-namespaces])}))
        stable {:requested (qualified-var-symbol fn-var)
                :resolved (qualified-var-symbol resolved)
                :tags (:raster.core/deftm-tags metadata)
                :source-body-fingerprint
                (when source-body (semantic-fingerprint/fingerprint source-body))
                :parameter-projection
                (when (:raster.core/deftm-params metadata)
                  (:projection (equation-first/parameter-representation fn-var dtype)))
                :source-dependency-fingerprint (:fingerprint dependency-evidence)}
        persistence-blockers
        (cond-> #{}
          (nil? source-body) (conj :retained-deftm-source)
          (nil? dependency-evidence) (conj :resolved-source-dependencies)
          (and dependency-evidence (not (:complete? dependency-evidence)))
          (conj :resolved-source-dependencies))]
    {:semantic stable
     :persistent-cache-eligible? (empty? persistence-blockers)
     :persistence-blockers persistence-blockers
     :source-dependency-blockers (:blockers dependency-evidence)
     ;; Redefinition with textually equal source must not retain compiler state tied to an old Var
     ;; root (for example a changed closed-over helper or dispatch table).
     :root-identity (weak-identity @resolved)}))

(defn- target-specialization-identity
  ([target]
   (try
     (target-specialization-identity target (hardware/descriptor-for target))
     (catch Exception _
       {:semantic {:target target :descriptor-fingerprint nil}
        :persistent-cache-eligible? false
        :persistence-blockers #{:target-descriptor}})))
  ([target descriptor]
   ;; Captured target facts must be immutable fingerprintable compiler data. Collapsing an
   ;; unsupported snapshot to a nil fingerprint would alias different targets in process memory.
   {:semantic {:target target
               :descriptor-fingerprint (semantic-fingerprint/fingerprint descriptor)}
    :persistent-cache-eligible? true
    :persistence-blockers #{}}))

(defn- fingerprint-or-nil [value]
  (try (semantic-fingerprint/fingerprint value)
       (catch clojure.lang.ExceptionInfo error
         (if (= :semantic-fingerprint-unsupported (:reason (ex-data error)))
           nil
           (throw error)))))

(defn- template-cache-key
  [kind source revision target options pipeline-identity]
  (let [build (build-manifest/current-identity)
        build-fingerprint (:fingerprint build)
        build-blockers
        (cond
          (nil? build) #{:compiler-build-fingerprint}
          (not (:complete? build)) (:blockers build)
          :else #{})
        semantic-request {:schema template-identity-schema
                          :kind kind
                          :compiler-build-fingerprint build-fingerprint
                          :source (:semantic source)
                          :target (:semantic target)
                          :options options}
        family-request {:schema template-identity-schema
                        :kind kind :source (:semantic source)}
        semantic-id (fingerprint-or-nil semantic-request)
        family-id (fingerprint-or-nil family-request)
        persistence-blockers
        (cond-> (into build-blockers
                      (into (:persistence-blockers source) (:persistence-blockers target)))
          (nil? semantic-id) (conj :unsupported-semantic-option)
          (nil? family-id) (conj :unsupported-source-identity))]
    {:kind kind
     ;; Retain the structural value as the in-process equality fallback when callers supply an
     ;; unsupported option. Such entries are never eligible for persistence.
     :semantic-request semantic-request
     :semantic-fingerprint semantic-id
     :family-fingerprint family-id
     :persistent-cache-eligible?
     (and (empty? persistence-blockers) semantic-id family-id
          (:persistent-cache-eligible? source)
          (:persistent-cache-eligible? target))
     :persistence-blockers persistence-blockers
     :source-dependency-blockers (:source-dependency-blockers source)
     :guards {:compiler-revision revision
              :source-root-identity (:root-identity source)
              :pipeline-identity pipeline-identity}}))

(defn- compilation-miss-reason [entries key]
  (let [keys (keys entries)
        semantic-id (:semantic-fingerprint key)
        family-id (:family-fingerprint key)]
    (cond
      (and semantic-id
           (some #(= semantic-id (:semantic-fingerprint %)) keys)) :invalidation
      (and family-id
           (some #(= family-id (:family-fingerprint %)) keys)) :specialization
      :else :compulsory)))

(defn- persistent-artifact-identity [key]
  {:semantic-request-fingerprint (:semantic-fingerprint key)
   :compiler-build-fingerprint
   (get-in key [:semantic-request :compiler-build-fingerprint])
   :source-dependency-fingerprint
   (get-in key [:semantic-request :source :source-dependency-fingerprint])
   :target-descriptor-fingerprint
   (get-in key [:semantic-request :target :descriptor-fingerprint])})

(defn- resolve-compilation-template
  [key compiler thunk persistent-report]
  (let [persistent? (and (= :equation-first compiler)
                         (:persistent-cache-eligible? key))
        identity (when persistent? (persistent-artifact-identity key))
        loaded (when persistent?
                 (equation-artifact-store/load-artifact
                  *equation-artifact-store* (:semantic-fingerprint key) identity))]
    (if (= :hit (:status loaded))
      (do (reset! persistent-report (dissoc loaded :value))
          (:value loaded))
      (let [started (System/nanoTime)
            value (try
                    (let [value (thunk)]
                      (swap! compilation-template-stats update :compilations inc)
                      value)
                    (catch Throwable error
                      (swap! compilation-template-stats update :failures inc)
                      (throw error))
                    (finally
                      (swap! compilation-template-stats update :compile-nanos +
                             (- (System/nanoTime) started))))
            stored
            (when persistent?
              (try
                (equation-artifact-store/store-artifact!
                 *equation-artifact-store* (:semantic-fingerprint key) identity value)
                (catch Exception error
                  {:status :write-failed :error-class (.getName (class error))})))]
        (reset! persistent-report
                (cond
                  (not persistent?) {:status :ineligible}
                  stored (assoc stored :load-miss-reason (:reason loaded)
                                :load-artifact-reason (:artifact-reason loaded))
                  :else {:status :not-stored}))
        value))))

(defn- cached-compilation-template
  [key compiler thunk]
  (let [persistent-report (atom nil)
        value (delay (resolve-compilation-template key compiler thunk persistent-report))
        candidate
        {:compiler compiler
         :value value
         ;; Process-local evidence belongs to this existing template owner, not to the serialized
         ;; compilation. Resolve/load/store the ordinary artifact before deriving this evidence.
         :emitted-validation
         (when (= :equation-first compiler)
           (delay
             (let [compilation @value
                   validator @#'emitted-program/validate-with-physical-results!]
               (when (equation-first/equation-first-compilation? compilation)
                 {:validation (validator (:emitted compilation))
                  :validator-identity (weak-identity validator)}))))
         :persistent-report persistent-report}
        [before after]
        (swap-vals! compilation-template-cache
                    #(if (contains? % key) % (assoc % key candidate)))
        hit? (contains? before key)
        miss-reason (when-not hit? (compilation-miss-reason before key))
        entry (get after key)
        resolution-started (System/nanoTime)]
    (swap! compilation-template-stats update (if hit? :hits :misses) inc)
    (when miss-reason
      (swap! compilation-template-stats update-in [:misses-by-reason miss-reason] (fnil inc 0)))
    (try
      (let [value @(:value entry)]
        (when *compilation-template-owner*
          (reset! *compilation-template-owner* {:key key :entry entry}))
        (when *compilation-template-observer*
          (*compilation-template-observer*
           {:compiler compiler :cache-hit? hit? :success? true
            :miss-reason miss-reason
            :semantic-fingerprint (:semantic-fingerprint key)
            ;; Reuse the artifact owner's build/source/target identity even on process hits.
            ;; This is template evidence, not a bound invocation or checkpoint producer proof.
            :persistent-artifact-identity
            (when (and (= :equation-first compiler) (:persistent-cache-eligible? key))
              (persistent-artifact-identity key))
            ;; The template owner retains the original store/load result across process hits.
            ;; These hashes identify that exact artifact, including its generated value names.
            :retained-artifact
            (select-keys @(:persistent-report entry)
                         [:compilation-fingerprint :payload-fingerprint])
            :persistent-cache-eligible? (:persistent-cache-eligible? key)
            :persistence-blockers (:persistence-blockers key)
            :source-dependency-blockers (:source-dependency-blockers key)
            :persistent-artifact
            (if hit?
              {:status :not-consulted :reason :process-cache-hit}
              @(:persistent-report entry))
            :resolution-ns (- (System/nanoTime) resolution-started)}))
        value)
      (catch Throwable error
        (when *compilation-template-observer*
          (*compilation-template-observer*
           {:compiler compiler :cache-hit? hit? :success? false
            :miss-reason miss-reason
            :semantic-fingerprint (:semantic-fingerprint key)
            :persistent-cache-eligible? (:persistent-cache-eligible? key)
            :persistence-blockers (:persistence-blockers key)
            :source-dependency-blockers (:source-dependency-blockers key)
            :persistent-artifact
            (if hit?
              {:status :not-consulted :reason :process-cache-hit}
              @(:persistent-report entry))
            :resolution-ns (- (System/nanoTime) resolution-started)}))
        ;; A failed compilation is not a durable negative result: a hot reload or newly registered
        ;; specialization may make the same request valid on its next attempt.
        (swap! compilation-template-cache
               #(if (identical? entry (get % key)) (dissoc % key) %))
        (throw error)))))

(def ^:private max-template-stabilization-attempts 8)

(defn- stable-compilation-template
  "Resolve one cached template in a compiler-definition epoch that remains unchanged for the
   complete compilation. Derived deftm specializations do not change this semantic epoch, but a
   concurrent source/type/dispatch redefinition may. In that case the result belongs only to the
   old key: retry under the new epoch instead of unsafely aliasing it.

   This makes a single preparation request converge the cache while preserving the invariant that
   every returned template was produced during a stable compiler-definition interval."
  [key-for-revision compiler thunk]
  (let [request-observer *compilation-template-observer*
        request-owner *compilation-template-owner*]
    (loop [attempt 1]
      (let [revision-before (dispatch/compiler-definition-revision)
            report (atom nil)
            owner (atom nil)
            key (key-for-revision revision-before)
            outcome (try
                      {:value
                       (binding [*compilation-template-observer* #(reset! report %)
                                 *compilation-template-owner* owner]
                         (cached-compilation-template key compiler thunk))}
                      (catch Throwable error {:error error}))
            _ (when-let [error (:error outcome)]
                (when request-observer (request-observer @report))
                (throw error))
            value (:value outcome)
            revision-after (dispatch/compiler-definition-revision)
            _ (when (not= revision-before revision-after)
                ;; No future request can legally use an entry built across an epoch change. A
                ;; concurrent waiter already holds its delay and will independently retry.
                (swap! compilation-template-cache dissoc key))]
        (if (= revision-before revision-after)
          (do
            (when request-owner (reset! request-owner @owner))
            (when request-observer
              (request-observer
               (assoc @report
                      :stabilization-attempts attempt
                      :compiler-revision revision-after)))
            value)
          (if (< attempt max-template-stabilization-attempts)
            (recur (inc attempt))
            (let [error
                  (ex-info "compiler definitions did not stabilize while caching a template"
                           {:reason :compilation-template-unstable
                            :compiler compiler
                            :attempts attempt
                            :revision-before revision-before
                            :revision-after revision-after})]
              (when request-observer
                (request-observer
                 (assoc @report :success? false
                        :stabilization-attempts attempt
                        :compiler-revision-before revision-before
                        :compiler-revision-after revision-after)))
              (throw error))))))))

(defn- owned-emitted-validation
  "Reuse only facts belonging to the exact stable compilation owner and current pipeline.
   Stale guards fall back to independent validation; a failed proof is never cached negatively."
  [{:keys [key entry]} compilation]
  (let [current-owner? (fn []
                         (and (identical? entry (get @compilation-template-cache key))
                              (= (get-in key [:guards :compiler-revision])
                                 (dispatch/compiler-definition-revision))
                              (= (get-in key [:guards :pipeline-identity])
                                 (weak-identity @#'equation-first/compile))
                              (identical? compilation @(:value entry))))]
    (when (and (:emitted-validation entry) (current-owner?))
      (try
        (let [{:keys [validation validator-identity]} @(:emitted-validation entry)]
          (when (and (current-owner?)
                     (= validator-identity
                        (weak-identity @#'emitted-program/validate-with-physical-results!))
                     (emitted-program/retained-validation? (:emitted compilation) validation))
            validation))
        (catch Throwable error
          (swap! compilation-template-cache
                 #(if (identical? entry (get % key)) (dissoc % key) %))
          (throw error))))))

;; ================================================================
;; Role derivation (§4.2) and tree construction
;; ================================================================

(defn- derive-roles
  "Effective {sym → role} for certified LinkPlan lowering: donated → :state, constants →
   :constant, and the rest fall through to the descriptor's derived defaults."
  [descriptor donate constants explicit-roles]
  (let [donate-set   (set donate)
        constant-set (set constants)
        both         (set/intersection donate-set constant-set)]
    (when (seq both)
      (throw (ex-info (str "compile: syms are both :donate and :constants — a param is either "
                           "donated (mutable :state) or frozen (:constant), not both: " both)
                      {:conflict both})))
    (merge (into {} (map (fn [s] [s :state]) donate-set))
           (into {} (map (fn [s] [s :constant]) constant-set))
           explicit-roles)))

(defn- build-in-tree
  [lowering descriptor arguments donate]
  (let [donate-set (set donate)
        values (:values (:certificate lowering))
        argument-map (zipmap (:all-params descriptor) arguments)]
    (vec (for [p (:array-params descriptor)
               :let [{:keys [node role shape dtype]} (get values p)]]
           {:key     (keyword (name p))
            :sym     p
            :node    node
            :role    role
            :donate? (contains? donate-set p)
            :shape   shape
            :dtype   dtype
            :default (get argument-map p)}))))

(defn- build-out-tree
  "Out-tree = donated in→out nodes + any explicit :outputs + the functional :result-sym + taps.
   Each projects to a DeviceArray over a resident buffer (§3.4 multi-output)."
  [lowering donate outputs result-sym taps]
  (let [values (:values (:certificate lowering))
        output-node (fn [key sym from]
                      (let [{:keys [node shape dtype]} (get values sym)]
                        {:key key :sym sym :node node :shape shape :dtype dtype :from from}))
        donate-nodes  (for [s donate]
                        (output-node (keyword (str (name s) "'")) s :donated))
        output-nodes  (for [s outputs]
                        (output-node (keyword (name s)) s :output))
        result-node   (when result-sym
                        [(output-node (keyword (name result-sym)) result-sym :result)])
        tap-nodes     (for [s taps]
                        (output-node (keyword (name s)) s :tap))]
    (vec (concat donate-nodes output-nodes result-node tap-nodes))))

(defn- compilation-id
  [fn-var target dtype descriptor args preserve-declared-array-storage?]
  (let [m (meta fn-var)
        qualified (symbol (str (ns-name (:ns m))) (str (:name m)))
        arrays (set (:array-params descriptor))
        argmap (zipmap (:all-params descriptor) args)
        signature (mapv (fn [parameter]
                          (let [value (get argmap parameter)]
                            (if (contains? arrays parameter)
                              [:array (.getName (.getComponentType (class value)))
                               (java.lang.reflect.Array/getLength value)]
                              [:scalar (semantic-fingerprint/fingerprint value)])))
                        (:all-params descriptor))]
    ;; Ordinary equality identifies signed zeros and cannot reliably identify equal NaN bits.
    ;; Canonical scalar fingerprints select the cache entry; bind-template independently checks
    ;; those bits rather than trusting the digest. Primitive-array contents remain absent.
    (cond-> [::compiled qualified target dtype signature (:schedule descriptor)]
      preserve-declared-array-storage?
      (conj {:preserve-declared-array-storage? true}))))

;; ================================================================
;; Pure lowering, composition, and runtime compilation
;; ================================================================

(defn- lower-resident-descriptor
  "Lower a deftm var into a certified, allocation-free `Prepared` artifact.

   args  — example args in the descriptor's :all-params order. Supplies BOTH the shapes
           compile-gpu-program derives AND the eventual resident initializers.
   opts  — {:target :ze:0            device-id (default :ze:0)
            :dtype  :float           element dtype (default :float)
            :preserve-declared-array-storage? true ; retain resolved pointer storage tags
            :donate  [sym …]         resident :state threaded as values (donation)
            :constants [sym …]       frozen, captured once at bind, never per-call
            :outputs [sym …]         additional written params to project as outputs
            :taps    [sym …]         internal nodes to additionally expose (§5.1)
            :roles   {sym → role}    explicit role override (last word)
            :gemm-precision :mixed-f16-f32|:f32-scalar
            :on-non-resident :nil|:throw
            :schedule <map>}         reserved S6 schedule (threaded into the cache key)"
  [fn-var args {:keys [target dtype donate constants outputs taps roles
                       gemm-precision on-non-resident schedule preserve-declared-array-storage?]
                :or {target :ze:0 dtype :float on-non-resident :nil}}]
  (let [preparation-started (System/nanoTime)
        template-report (atom nil)
        schedule (if schedule (gpu-schedule/normalize-override schedule) schedule)
        compile-arguments
        (cond-> [:dtype dtype :on-non-resident on-non-resident]
          gemm-precision (conj :gemm-precision gemm-precision)
          ;; forward the S6 schedule so it is resolved + gated by compile-gpu-program;
          ;; the RESOLVED schedule is read back off the descriptor below (never the raw
          ;; input). Harmless where compile-gpu-program predates :schedule (ignored kwarg).
          schedule (conj :schedule schedule)
          preserve-declared-array-storage?
          (conj :preserve-declared-array-storage? true))
        template-key
        (fn [revision]
          (template-cache-key
           ::resident-template
           (source-specialization-identity fn-var dtype)
           revision
           (target-specialization-identity target)
           (cond-> {:dtype dtype :gemm-precision (or gemm-precision :mixed-f16-f32)
                    :on-non-resident on-non-resident :schedule schedule}
             preserve-declared-array-storage?
             (assoc :preserve-declared-array-storage? true))
           ;; Keeps with-redefs and hot compiler reloads honest.
           (weak-identity @#'pl/compile-gpu-program)))
        prog (binding [*compilation-template-observer* #(reset! template-report %)]
               (stable-compilation-template
                template-key :resident-descriptor
                #(apply pl/compile-gpu-program fn-var target compile-arguments)))
        _ (when-not prog
            (throw (ex-info "compile: compile-gpu-program returned nil — a step fell back to host (non-resident). Pass :on-non-resident :throw to see which."
                            {:fn fn-var :target target})))
        eff-roles (derive-roles prog donate constants roles)
        ;; Effect-only deftm descriptors may retain a synthetic scalar `body_result_*` even
        ;; though the resident ABI returns Void. Only pointer results are projectable device
        ;; values; explicit written buffers remain available through :outputs.
        pointer-symbols (link-plan/descriptor-pointer-symbols prog)
        result-sym (when (contains? pointer-symbols (:result-sym prog)) (:result-sym prog))
        public-symbols (vec (distinct (concat donate outputs
                                              (when result-sym [result-sym]) taps)))
        lowering-started (System/nanoTime)
        plan-id (compilation-id fn-var target dtype prog args preserve-declared-array-storage?)
        compiler-template-key (template-key (get @template-report :compiler-revision))
        plan-template-key
        {:kind ::resident-plan-template
         :compiler-template-key compiler-template-key
         :compiler-template-fingerprint (:semantic-fingerprint compiler-template-key)
         :plan-id plan-id :roles eff-roles :public-symbols public-symbols}
        plan-template-report
        (cached-resident-plan-template
         plan-template-key
         #(resident-plan/lower
           {:id plan-id :target target :descriptor prog :arguments args
            :roles eff-roles :outputs public-symbols}))
        lowering (resident-plan/bind-template (:template plan-template-report) args)
        lowering-ns (- (System/nanoTime) lowering-started)
        in-tree  (build-in-tree lowering prog args donate)
        out-tree (build-out-tree lowering donate outputs result-sym taps)
        donated  (into {} (map (fn [s] [(keyword (name s)) (keyword (str (name s) "'"))]) donate))
        report {:kind :resident-descriptor
                :timing-source :host-monotonic
                :total-ns (- (System/nanoTime) preparation-started)
                :template @template-report
                :resident-plan-template (dissoc plan-template-report :template)
                :link-plan-lowering-ns lowering-ns
                :nodes (count (get-in lowering [:plan :nodes]))
                :instances (count (get-in lowering [:plan :instances]))}]
    (seal-artifact
     (->Prepared lowering in-tree out-tree donated (:schedule prog) target prog args report nil))))

(defn- equation-first-value
  [plan node role]
  (let [view (get-in plan [:nodes node :view])]
    {:node node :role role :shape (:shape view) :dtype (:dtype view)}))

(defn- equation-first-output-key
  [value index total]
  (cond
    (= 1 total) :result
    (instance? clojure.lang.Named value) (keyword (name value))
    :else (keyword (str "result-" index))))

(defn- project-equation-first-boundary
  "Choose public roles and escaped outputs on the normalized candidate, before its sole proof."
  [raw-plan compilation args {:keys [donate constants outputs taps roles]}]
  (let [invocation-plan (get-in compilation [:semantic :attributes :invocation-plan])
        projection (get-in invocation-plan [:attributes :parameter-projection])
        aggregate-leaves (group-by :binding (filter materialization/projection-path
                                                   (:physical-parameters projection)))
        _ (when (some aggregate-leaves (concat donate outputs taps))
            (throw (ex-info "aggregate donation and aggregate outputs require per-field ownership support"
                            {:reason :compiled-aggregate-output :aggregates (set (keys aggregate-leaves))})))
        expand (fn [symbol] (if-let [leaves (get aggregate-leaves symbol)]
                             (filterv #(contains? (get-in raw-plan [:attributes :public-buffer-bindings]) %)
                                      (mapv :symbol leaves)) [symbol]))
        constants (vec (mapcat expand constants))
        roles (into {} (mapcat (fn [[symbol role]] (map #(vector % role) (expand symbol))) roles))
        attributes (:attributes raw-plan)
        public-bindings (:public-buffer-bindings attributes)
        public-defaults (:public-buffer-roles attributes)
        _ (doseq [[binding leaves] aggregate-leaves
                  leaf leaves
                  :let [role (or (get roles (:symbol leaf))
                                 (when (some #{(:symbol leaf)} constants) :constant)
                                 (get public-defaults (:symbol leaf)))]]
            (when (or (contains? #{:state :output} (get public-defaults (:symbol leaf)))
                      (and role (not (contains? #{:input :constant} role))))
              (throw (ex-info "the initial aggregate vertical admits read-only fields only"
                              {:reason :compiled-aggregate-write :parameter binding :field (:field leaf)
                               :role role}))))
        public-symbols (set (keys public-bindings))
        requested-symbols (set (concat donate constants (keys roles)))
        unknown (set/difference requested-symbols public-symbols)
        _ (when (seq unknown)
            (throw (ex-info "equation-first roles name non-buffer public parameters"
                            {:reason :compiled-equation-first-role-symbols
                             :symbols unknown :available public-symbols})))
        effective-roles (merge public-defaults
                               (derive-roles nil donate constants roles))
        token-roles (into {} (map (fn [[symbol role]]
                                    [(get public-bindings symbol) role]))
                          effective-roles)
        compiler-bindings (:compiler-buffer-bindings attributes)
        compiler-roles (into {} (keep (fn [[compiler-value token]]
                                        (when-let [role (get token-roles token)]
                                          [compiler-value role])))
                             compiler-bindings)
        plan (-> (reduce-kv (fn [plan token role]
                              (assoc-in plan [:nodes token :role] role))
                            raw-plan token-roles)
                 (assoc-in [:instances 0 :roles] compiler-roles)
                 (assoc-in [:attributes :public-buffer-roles] effective-roles))
        parameters (get-in compilation [:semantic :attributes :invocation-plan :parameters])
        argument-map (when-not projection (zipmap (map :symbol parameters) args))
        projected-arguments (when projection
                              (zipmap (map :symbol (:physical-parameters projection))
                                      (materialization/parameter-arguments invocation-plan args)))
        aggregate-scalars (into {}
                                (map (fn [[binding leaves]]
                                       [binding (into {}
                                                      (keep (fn [{:keys [symbol tag] :as leaf}]
                                                              (when-not (types/array-tag? tag)
                                                                [leaf (get projected-arguments symbol)])))
                                                      leaves)]))
                                aggregate-leaves)
        leaves-by-symbol (into {} (map (juxt :symbol identity)) (:physical-parameters projection))
        donate-set (set donate)
        in-tree (vec
                 (keep (fn [{:keys [symbol]}]
                         (when-let [node (get public-bindings symbol)]
                           (let [{:keys [binding] :as leaf} (get leaves-by-symbol symbol)
                                 path (materialization/projection-path leaf)]
                             (merge (cond-> {:key (if path (into [(keyword (name binding))] path)
                                                      (keyword (name symbol)))
                                             :sym symbol
                                             :donate? (contains? donate-set symbol)
                                             :default (if projection (get-in raw-plan [:nodes node :source])
                                                          (get argument-map symbol))}
                                      path (assoc :aggregate-binding binding :aggregate-leaf leaf
                                                   :aggregate-tree (get-in projection [:trees binding])
                                                   :aggregate-scalars (get aggregate-scalars binding)))
                                    (equation-first-value plan node
                                                          (get effective-roles symbol))))))
                       parameters))
        resolve-node (fn [value]
                       (or (get public-bindings value)
                           (get compiler-bindings value)
                           (throw (ex-info "equation-first output names no resident value"
                                           {:reason :compiled-equation-first-output
                                            :value value}))))
        output-entry (fn [key value from]
                       (merge {:key key :sym value :from from}
                              (equation-first-value plan (resolve-node value) nil)))
        donated-nodes (map-indexed
                       (fn [_ symbol]
                         (output-entry (keyword (str (name symbol) "'")) symbol :donated))
                       donate)
        explicit-nodes (map-indexed
                        (fn [_ value]
                          (output-entry (keyword (name value)) value :output))
                        outputs)
        semantic-outputs (vec (:semantic-outputs attributes))
        semantic-nodes (map-indexed
                        (fn [index [value _]]
                          (output-entry
                           (equation-first-output-key value index (count semantic-outputs))
                           value :result))
                        semantic-outputs)
        tap-nodes (map-indexed
                   (fn [_ value]
                     (output-entry (keyword (name value)) value :tap))
                   taps)
        out-tree (reduce (fn [entries entry]
                           (if (some #(= (:node %) (:node entry)) entries)
                             entries
                             (conj entries entry)))
                         [] (concat donated-nodes explicit-nodes semantic-nodes tap-nodes))
        duplicate-keys (->> out-tree (map :key) frequencies
                            (keep (fn [[key count]] (when (< 1 count) key))) set)
        _ (when (seq duplicate-keys)
            (throw (ex-info "equation-first outputs require unique semantic keys"
                            {:reason :compiled-equation-first-output-keys
                             :keys duplicate-keys})))
        escaped (mapv :node out-tree)
        missing (set/difference (set (:outputs plan)) (set escaped))
        _ (when (seq missing)
            (throw (ex-info "escaped invocation outputs must retain the semantic output boundary"
                            {:reason :invocation-link-output-boundary
                             :missing missing :outputs escaped})))]
    {:plan (assoc plan :outputs escaped)
     :projection {:in-tree in-tree :out-tree out-tree}}))

(defn- lower-equation-first
  [fn-var args {:keys [target dtype donate constants outputs taps roles]
                :or {target :ze:0 dtype :float}
                :as opts}]
  (let [preparation-started (System/nanoTime)
        template-report (atom nil)
        template-owner (atom nil)
        target-descriptor (equation-first/validate-target-description!
                           target (hardware/descriptor-for target))
        target-identity (target-specialization-identity target target-descriptor)
        compilation-options (apply dissoc opts
                                   [:compiler :donate :constants :outputs :taps :roles
                                    :profile? :on-non-resident])
        compilation-options (assoc compilation-options :target target :dtype dtype)
        compilation-options (cond-> compilation-options
                              (contains? compilation-options :schedule)
                              (update :schedule gpu-schedule/normalize-override))
        template-key
        (fn [revision]
          (template-cache-key
           ::equation-first-template
           (source-specialization-identity fn-var dtype)
           revision
           target-identity
           compilation-options
           (weak-identity @#'equation-first/compile)))
        compilation (binding [*compilation-template-observer* #(reset! template-report %)
                              *compilation-template-owner* template-owner]
                      (stable-compilation-template
                       template-key :equation-first
                       #(equation-first/compile fn-var compilation-options target-descriptor)))
        lowering-started (System/nanoTime)
        retained-validation (owned-emitted-validation @template-owner compilation)
        equation-lower-phases (atom nil)
        projection-ns (atom 0)
        result (binding [equation-first/*lower-observer*
                         #(reset! equation-lower-phases %)]
                 (equation-first/lower
                  compilation args
                  (fn [plan]
                    (let [started (System/nanoTime)
                          projected (project-equation-first-boundary plan compilation args opts)]
                      (reset! projection-ns (- (System/nanoTime) started))
                      projected))
                  retained-validation))
        equation-lower-ns (- (System/nanoTime) lowering-started @projection-ns)
        plan (:plan result)
        {:keys [in-tree out-tree]} (:projection result)
        attributes (:attributes plan)
        public-bindings (:public-buffer-bindings attributes)
        parameters (get-in compilation [:semantic :attributes :invocation-plan :parameters])
        parameter-projection (get-in compilation [:semantic :attributes :invocation-plan :attributes :parameter-projection])
        logical-binding (into {} (map (juxt :symbol :binding)) (:physical-parameters parameter-projection))
        public-parameters (or (:public-parameters parameter-projection) (mapv :symbol parameters))
        public-arrays (vec (distinct (for [{:keys [symbol]} parameters
                                          :when (contains? public-bindings symbol)]
                                      (get logical-binding symbol symbol))))
        public-array-set (set public-arrays)
        certification-started (System/nanoTime)
        lowering (invocation-link/certify-final-projection result)
        invocation-certification-ns (- (System/nanoTime) certification-started)
        lowering-ns (- (System/nanoTime) lowering-started)
        steps (mapv (fn [index kernel]
                      {:convention :kernel-body
                       :phase (keyword (str "kernel-" index))
                       :kernel-name (:kernel-name kernel)})
                    (range) (:kernels compilation))
        descriptor {:all-params public-parameters
                    :array-params public-arrays
                    :scalar-params (filterv #(not (contains? public-array-set %)) public-parameters)
                    :parameter-projection parameter-projection
                    :steps steps :result-sym nil :equation-first? true}
        schedule (assoc (get-in compilation [:options :schedule])
                        :compiler :equation-first :stats (:stats compilation))
        donated (into {} (map (fn [symbol]
                                [(keyword (name symbol))
                                 (keyword (str (name symbol) "'"))]))
                      donate)
        report {:kind :equation-first
                :timing-source :host-monotonic
                :total-ns (- (System/nanoTime) preparation-started)
                :template @template-report
                :link-plan-lowering-ns lowering-ns
                :phases-ns {:equation-lower equation-lower-ns
                            :role-projection @projection-ns
                            :invocation-certification invocation-certification-ns}
                :equation-lower-phases-ns @equation-lower-phases
                :nodes (count (get-in lowering [:plan :nodes]))
                :instances (count (get-in lowering [:plan :instances]))}]
    (seal-artifact
     (->Prepared lowering in-tree out-tree donated schedule target descriptor args report nil))))

(defn lower
  "Lower a deftm Var into an allocation-free `Prepared` artifact.

   `:compiler :equation-first` selects the typed SOAC→scheduled KernelBody→LinkPlan vertical.
   `:resident-descriptor` remains the default during migration. Selection is explicit: the
   equation-first compiler never falls back to a resident descriptor after a coverage failure."
  [fn-var args {:keys [compiler] :or {compiler :resident-descriptor} :as opts}]
  (case compiler
    :resident-descriptor (lower-resident-descriptor fn-var args opts)
    :equation-first (lower-equation-first fn-var args opts)
    (throw (ex-info "unknown compiled lowering vertical"
                    {:reason :compiled-compiler :compiler compiler
                     :allowed #{:resident-descriptor :equation-first}}))))

(defn instantiate!
  "Instantiate one pure Prepared artifact as a callable Compiled value. All component plans have
   already been composed, so this performs one allocation/binding/graph-recording operation."
  ([prepared] (instantiate! prepared {}))
  ([prepared opts]
   (when-not (prepared? prepared)
     (throw (ex-info "instantiate! requires an allocation-free Prepared artifact"
                     {:reason :compiled-prepared-type :actual (type prepared)})))
   (let [{:keys [lowering in-tree out-tree donated schedule target descriptor args
                 preparation-report]} prepared
         evidence (get-in lowering [:certificate :effect-evidence])
         executable (if (and (sealed-artifact? prepared)
                             (link-plan/retained-effect-evidence? (:plan lowering) evidence))
                      (gpu-link/instantiate-certified! lowering opts)
                      (gpu-link/instantiate! (:plan lowering) opts))]
     (seal-artifact
      (->Compiled lowering executable in-tree out-tree donated schedule target descriptor args
                 preparation-report
                 prepared nil (atom nil))))))

(defn preparation-report
  "Return compact host-side template-cache and LinkPlan preparation facts for a Prepared or
   instantiated Compiled artifact. Composite reports retain their component reports."
  [artifact]
  (when-not (or (prepared? artifact) (compiled? artifact))
    (throw (ex-info "preparation-report requires a Prepared or Compiled artifact"
                    {:reason :compiled-preparation-report-type :actual (type artifact)})))
  (:preparation-report artifact))

(defn- exact-artifact-evidence
  [lowering report]
  (case (:kind report)
    :equation-first
    (let [{:keys [persistent-cache-eligible? persistence-blockers source-dependency-blockers
                 persistent-artifact-identity retained-artifact]} (:template report)]
      (when-not (and persistent-cache-eligible?
                     (empty? persistence-blockers) (empty? source-dependency-blockers)
                     (= 4 (count persistent-artifact-identity))
                     (every? #(and (string? %) (not-empty %)) (vals persistent-artifact-identity))
                     (= #{:compilation-fingerprint :payload-fingerprint}
                        (set (keys retained-artifact)))
                     (every? #(and (string? %) (not-empty %)) (vals retained-artifact)))
        (throw (ex-info "exact program evidence requires a complete retained compiler artifact"
                        {:reason :compiled-execution-identity-incomplete
                         :persistence-blockers persistence-blockers
                         :source-dependency-blockers source-dependency-blockers
                         :retained-artifact? (boolean (seq retained-artifact))})))
      (invocation-link/verify! lowering)
      {:identity persistent-artifact-identity :artifact retained-artifact})

    :composition
    (let [components (:components lowering)
          reports (:components report)]
      (link-composition/verify! lowering)
      (when-not (= (mapv :id components) (mapv :id reports))
        (throw (ex-info "composition reports must follow their exact certified components"
                        {:reason :compiled-execution-identity-components})))
      {:components (mapv (fn [component component-report]
                           {:id (:id component)
                            :artifact (exact-artifact-evidence
                                       (:lowering component) (:report component-report))})
                         components reports)
       :specification (:specification lowering)})

    (throw (ex-info "exact program evidence requires the equation-first vertical"
                    {:reason :compiled-execution-identity-vertical :kind (:kind report)}))))

(defn execution-identity
  "Identify one exact retained, specialized Prepared or Compiled without input-byte lineage.

   Requires complete packaged build/source/target evidence and retained artifact hashes. Includes
   certified calls, storage contracts, composition, roles, donation, outputs and schedule, but never
   host initializer arrays. Generated IDs belong to the exact retained artifact; this is not an
   alpha-equivalence or target-neutral mathematical identity. Caller defaults can be replaced at
   invocation, so the returned identity deliberately does not attest the bytes used in execution."
  [artifact]
  (when-not (and (or (prepared? artifact) (compiled? artifact)) (sealed-artifact? artifact))
    (throw (ex-info "execution-identity requires an original compiler-owned artifact"
                    {:reason :compiled-execution-identity-owner})))
  (let [prepared (if (compiled? artifact) (:prepared artifact) artifact)
        _ (when-not (and (prepared? prepared) (sealed-artifact? prepared))
            (throw (ex-info "compiled evidence requires its original compiler-owned Prepared"
                            {:reason :compiled-execution-identity-owner})))
        artifact (exact-artifact-evidence (:lowering prepared) (:preparation-report prepared))
        plan (:plan (:lowering prepared))
        boundary #(mapv (fn [entry] (dissoc entry :default)) %)
        projection {:kind :raster.compiled/exact-bound-program-v1
                    :artifact artifact
                    :plan (update plan :nodes
                                  (fn [nodes]
                                    (into {} (map (fn [[id node]] [id (dissoc node :source)]) nodes))))
                    :inputs (boundary (:in-tree prepared))
                    :outputs (boundary (:out-tree prepared))
                    :donated (:donated prepared) :schedule (:schedule prepared)
                    :target (:target prepared)}]
    {:scope :exact-bound-program
     :fingerprint (semantic-fingerprint/fingerprint projection)
     :data-slots (mapv #(select-keys % [:key :node :dtype :shape :role]) (:in-tree prepared))
     :attests-input-bytes? false}))

(defn instantiation-report
  "Return the compact host-side construction report for an instantiated Compiled artifact."
  [compiled]
  (when-not (compiled? compiled)
    (throw (ex-info "instantiation-report requires a Compiled artifact"
                    {:reason :compiled-instantiation-report-type :actual (type compiled)})))
  (gpu-link/instantiation-report (:executable compiled)))

(defn- semantic-entry! [components side [component-id key :as reference]]
  (when-not (and (vector? reference) (= 2 (count reference)))
    (throw (ex-info "a semantic artifact reference is [component-id key]"
                    {:reason :compiled-composition-reference :reference reference})))
  (let [prepared (get components component-id)
        entries (case side :input (:in-tree prepared) :output (:out-tree prepared))
        matches (filterv #(= key (:key %)) entries)]
    (when-not (= 1 (count matches))
      (throw (ex-info "a semantic artifact reference must resolve exactly once"
                      {:reason :compiled-composition-reference :side side :reference reference
                       :available (mapv :key entries)})))
    (first matches)))

(defn compose
  "Compose independently lowered Prepared artifacts through semantic boundary keys.

   Request shape:
   {:id stable-id
    :components [{:id component-id :program prepared} ...]
    :connections [{:from [producer-id output-key] :to [consumer-id input-key]} ...]
    :shares [[[component-id input-or-constant-key] ...] ...]
    :mutable-shares [{:owner [component-id donated-input-key]
                      :borrowers [[component-id read-only-input-key] ...]
                      :output [component-id donated-output-key]} ...]
    :outputs [{:key composite-output-key :from [component-id output-key]} ...]}

   Remaining component inputs are exposed under `[component-id input-key]`. Composition happens
   before allocation, so connected intermediates and shared constants have one physical
   allocation. Explicit mutable sharing retains one donated state owner, removes borrower input
   refresh, and exports its final state. Constants cannot be mutable borrowers; ranged views and
   escaped earlier aliases remain rejected. One linked executable owns replay and output leases."
  [{:keys [id components connections shares mutable-shares outputs attributes]
    :or {connections [] shares [] mutable-shares [] attributes {}}}]
  (let [preparation-started (System/nanoTime)
        components (mapv (fn [component]
                           (when-not (and (map? component) (contains? component :id)
                                          (prepared? (:program component)))
                             (throw (ex-info
                                     "each compiled component requires :id and a Prepared :program"
                                     {:reason :compiled-composition-component
                                      :component component})))
                           component)
                         components)
        component-map (into {} (map (juxt :id :program)) components)
        _ (when-not (= (count components) (count component-map))
            (throw (ex-info "compiled component identities must be unique"
                            {:reason :compiled-composition-component-ids
                             :ids (mapv :id components)})))
        donated-components (filterv (comp seq :donated :program) components)
        resolved-connections
        (mapv (fn [{:keys [from to]}]
                {:from from :to to
                 :from-entry (semantic-entry! component-map :output from)
                 :to-entry (semantic-entry! component-map :input to)})
              connections)
        resolved-shares
        (mapv (fn [group]
                (mapv (fn [reference]
                        {:reference reference
                         :entry (semantic-entry! component-map :input reference)})
                      group))
              shares)
        output-specs
        (mapv (fn [{:keys [key from] :as output}]
                (when-not (= #{:key :from} (set (keys output)))
                  (throw (ex-info "a composite output requires exactly :key and :from"
                                  {:reason :compiled-composition-output :output output})))
                {:key key :from from
                 :entry (semantic-entry! component-map :output from)})
              outputs)
        _ (when-not (= (count output-specs) (count (distinct (map :key output-specs))))
            (throw (ex-info "composite output keys must be unique and ordered"
                            {:reason :compiled-composition-output-keys
                             :keys (mapv :key output-specs)})))
        resolved-mutable-shares
        (mapv
         (fn [{:keys [owner borrowers output] :as binding}]
           (when-not (and (= #{:owner :borrowers :output} (set (keys binding)))
                          (vector? borrowers) (seq borrowers))
             (throw (ex-info "mutable sharing requires owner, borrowers and final output"
                             {:reason :compiled-composition-mutable-specification :binding binding})))
           (let [owner-entry (semantic-entry! component-map :input owner)
                 output-entry (semantic-entry! component-map :output output)
                 borrower-entries (mapv #(semantic-entry! component-map :input %) borrowers)
                 selected (filterv #(= output (:from %)) output-specs)]
             (when-not (and (= :state (:role owner-entry)) (:donate? owner-entry)
                            (= (first owner) (first output))
                            (= (get-in component-map [(first owner) :donated (second owner)])
                               (second output))
                            (= (:node owner-entry) (:node output-entry))
                            (every? #(= :input (:role %)) borrower-entries)
                            (= 1 (count selected)))
               (throw (ex-info "mutable sharing must retain a donated owner and its final output"
                               {:reason :compiled-composition-mutable-owner :binding binding})))
             {:owner owner :owner-entry owner-entry
              :borrowers (mapv (fn [reference entry] {:reference reference :entry entry})
                               borrowers borrower-entries)
              :output output :output-entry output-entry :output-key (:key (first selected))}))
         mutable-shares)
        donation-owners (mapv :owner resolved-mutable-shares)
        component-donations
        (set (for [{component-id :id program :program} donated-components
                   key (keys (:donated program))] [component-id key]))
        _ (when-not (and (= (count donation-owners) (count (distinct donation-owners)))
                         (= component-donations (set donation-owners)))
            (throw (ex-info "every component donation requires exactly one mutable owner binding"
                            {:reason :compiled-composition-donation
                             :expected component-donations :owners donation-owners})))
        prevalidated? (every? (comp sealed-artifact? :program) components)
        low-level
        ((if prevalidated?
           link-composition/compose-prevalidated
           link-composition/compose)
         {:id id
          :components (mapv (fn [{:keys [id program]}]
                              {:id id :lowering (:lowering program)})
                            components)
          :connections (mapv (fn [{:keys [from to from-entry to-entry]}]
                               {:from [(first from) (:node from-entry)]
                                :to [(first to) (:node to-entry)]})
                             resolved-connections)
          :shares (mapv (fn [group]
                          (mapv (fn [{:keys [reference entry]}]
                                  [(first reference) (:node entry)])
                                group))
                        resolved-shares)
          :mutable-shares
          (mapv (fn [{:keys [owner owner-entry borrowers output output-entry]}]
                  {:owner [(first owner) (:node owner-entry)]
                   :borrowers (mapv (fn [{:keys [reference entry]}]
                                      [(first reference) (:node entry)]) borrowers)
                   :output [(first output) (:node output-entry)]})
                resolved-mutable-shares)
          :outputs (mapv (fn [{:keys [from entry]}]
                           [(first from) (:node entry)])
                         output-specs)
          :attributes attributes})
        node-mapping (get-in low-level [:certificate :node-mapping])
        mapped-node (fn [component-id node-id]
                      (or (get node-mapping [component-id node-id])
                          (throw (ex-info
                                  "certified semantic node disappeared during composition"
                                  {:reason :compiled-composition-node
                                   :component component-id :node node-id}))))
        consumed-inputs (set (map :to resolved-connections))
        discarded-shares (set (mapcat (fn [group] (map :reference (rest group)))
                                      resolved-shares))
        borrowed-inputs (set (mapcat #(map :reference (:borrowers %)) resolved-mutable-shares))
        in-tree
        (vec
         (for [{component-id :id program :program} components
               entry (:in-tree program)
               :let [reference [component-id (:key entry)]]
               :when (not (contains? consumed-inputs reference))
               :when (not (contains? discarded-shares reference))
           ;; Mutable borrowers are read aliases, not dynamic input refresh slots.
           ;; Their captured host default must never overwrite the owner's updated state.
               :when (not (contains? borrowed-inputs reference))]
           ;; Composition exposes explicit component/leaf references, not ambiguous
           ;; source-record shorthand shared by several independently prepared components.
           (assoc (dissoc entry :aggregate-binding :aggregate-leaf :aggregate-tree :aggregate-scalars)
                  :key reference :sym [component-id (:sym entry)]
                  :node (mapped-node component-id (:node entry)))))
        out-tree
        (mapv (fn [{:keys [key from entry]}]
                (assoc entry :key key :sym [(first from) (:sym entry)] :from :composed
                       :node (mapped-node (first from) (:node entry))))
              output-specs)
        descriptor {:all-params [] :array-params [] :scalar-params []
                    :steps (vec (mapcat (comp :steps :descriptor :program) components))
                    :result-sym nil :composite? true}
        schedules (mapv (comp :schedule :program) components)
        report {:kind :composition
                :timing-source :host-monotonic
                :total-ns (- (System/nanoTime) preparation-started)
                :components (mapv (fn [{:keys [id program]}]
                                    {:id id :report (:preparation-report program)})
                                  components)
                :nodes (count (get-in low-level [:plan :nodes]))
                :instances (count (get-in low-level [:plan :instances]))}]
    (seal-artifact
     (->Prepared low-level in-tree out-tree
                 (into {} (map (juxt :owner :output-key)) resolved-mutable-shares)
                 schedules (:target (:plan low-level))
                 descriptor [] report nil))))

(defn compile
  "Lower and instantiate a deftm as one callable Compiled artifact. Use `lower`, `compose`, then
   `instantiate!` when independently compiled programs must share resident values."
  [fn-var args opts]
  (instantiate! (lower fn-var args opts) (select-keys opts [:profile?])))

;; ================================================================
;; Functional invocation (§2.3)
;; ================================================================

(defn- project-node
  "Wrap one linked output view as an external DeviceArray. LinkedExecutable retains allocation
   ownership; the wrapper owns only its value lifetime."
  [executable {:keys [node]} target]
  (let [resident (gpu-link/node-view executable node)
        buffer (gpu/buffer (:session executable) (:key resident))]
    (v/wrap-external-view buffer target (:view resident))))

(defn- invalidate-live-outputs!
  [^Compiled c]
  (when-let [live-outputs (:live-outputs c)]
    (doseq [device-array @live-outputs] (v/free! device-array))
    (reset! live-outputs nil))
  c)

(defn- checked-donations
  "Validate all provided resident donation handles before any invocation mutation. This is a
   deterministic preflight, not a transaction against another owner concurrently freeing a value."
  [executable in-nodes donated inputs]
  (reduce
   (fn [checked [k _out]]
     (if-let [val (get inputs k)]
       (do
         (when-not (v/device-array? val)
           (throw (ex-info (str "invoke: donated input " k " must be a DeviceArray naming "
                                "this artifact's exact resident view")
                           {:key k :actual (type val)})))
         (when-not (v/live? val)
           (throw (ex-info "donated input is no longer live"
                           {:reason :compiled-donation-dead :key k})))
         (when (= ::v/aliased (:owner val))
           (throw (ex-info "an aliased DeviceArray cannot be donated"
                           {:reason :compiled-donation-alias :key k})))
         (when (some #(identical? val %) checked)
           (throw (ex-info "one DeviceArray cannot fill multiple donated slots"
                           {:reason :compiled-donation-duplicate :key k})))
         (let [{node-id :node :as input-node} (get in-nodes k)
               resident (gpu-link/node-view executable node-id)
               resident-buffer (gpu/buffer (:session executable) (:key resident))]
           (when-not (and (identical? (:buffer val) resident-buffer)
                          (bview/same-range? (:view val) (:view resident)))
             (throw (ex-info (str "invoke: donated input " k " is not this artifact's resident "
                                  "view — thread the exact donated output back")
                             {:key k :node node-id})))
           (when-not (= [(:dtype input-node) (:shape input-node)]
                        [(:dtype val) (:shape val)])
             (throw (ex-info (str "invoke: donated input " k " dtype/shape differs from its slot")
                             {:key k :expected (select-keys input-node [:dtype :shape])
                              :actual (select-keys val [:dtype :shape])})))
           (conj checked val)))
       checked))
   [] donated))

(defn- project-aggregate-inputs
  [in-tree inputs]
  (reduce-kv
   (fn [inputs binding entries]
     (let [key (keyword (name binding))]
       (if-not (contains? inputs key) inputs
               (do
                 (when-not (every? #(= :input (:role %)) entries)
                   (throw (ex-info "captured aggregate fields cannot be replaced at invocation"
                                   {:reason :compiled-aggregate-input-role :parameter binding})))
                 (when (some #(contains? inputs (:key %)) entries)
                   (throw (ex-info "aggregate and individual field inputs cannot both be supplied"
                                   {:reason :compiled-aggregate-input-conflict :parameter binding})))
                 (materialization/validate-aggregate! (get inputs key) binding
                                                     (:aggregate-tree (first entries)))
                 (doseq [[leaf expected] (:aggregate-scalars (first entries))]
                   (let [actual (materialization/aggregate-leaf (get inputs key) leaf)]
                     (when-not (= expected actual)
                       (throw (ex-info "aggregate scalar fields are captured by the prepared specialization; prepare again to change them"
                                       {:reason :compiled-aggregate-scalar-change :parameter binding
                                        :field (:field leaf) :path (materialization/projection-path leaf)
                                        :expected expected :actual actual})))))
                 (into (dissoc inputs key)
                       (map (fn [{:keys [aggregate-leaf] field-key :key}]
                              [field-key (materialization/aggregate-leaf (get inputs key) aggregate-leaf)]))
                       entries))))) inputs
   (group-by :aggregate-binding (filter :aggregate-binding in-tree))))

(defn- preflight-inputs!
  [executable input-nodes inputs]
  (doseq [{:keys [key node default]} input-nodes]
    (gpu-link/validate-write! executable node (get inputs key default))))

(defn- retiring-value-root?
  [previous value]
  (and (v/device-array? value)
       (or (some #(identical? % value) previous)
           (when-let [base (:base value)] (retiring-value-root? previous base)))))

(defn- write-invocation-inputs!
  "Retire public output aliases before mutation, but retain private borrowed reads when a caller
   feeds a previous output back as an input. The executable lifetime lock and completed preflight
   justify this local handoff; no independent owner or asynchronous lifetime is extended."
  [c input-nodes inputs previous donations]
  (let [borrowed (atom [])
        retiring (into (vec previous) donations)]
    (try
      (let [sources (mapv (fn [{:keys [key node default]}]
                            (let [source (get inputs key default)]
                              [node (if (retiring-value-root? retiring source)
                                      (let [read (v/wrap-external-view
                                                  (:buffer source) (:device source) (:view source))]
                                        (swap! borrowed conj read)
                                        read)
                                      source)])) input-nodes)]
        ;; Transfers are not transactional: consume donations at the mutation boundary, so
        ;; even a partially failed write cannot leave a live donated alias to corrupted bytes.
        (doseq [value donations] (v/consume! value))
        (invalidate-live-outputs! c)
        (doseq [[node source] sources]
          (gpu-link/write! (:executable c) node source)))
      (finally (doseq [read @borrowed] (v/free! read))))))

(defn- invoke-compiled-unleased*
  "Replay the artifact and return device values. `inputs` : {in-key → DeviceArray|host-array}.
     1. preflight all donations before any mutation;
     2. consume donations and retire previous output wrappers after complete preflight;
     3. write dynamic inputs (host upload, exact-view no-op, or device-to-device copy);
     4. replay the linked graph with no output download;
     5. project out-tree nodes as external DeviceArrays over stable LinkPlan views.
   Mutation of resident :state is invisible: the caller sees fresh output values and the old
   donated inputs invalidated — never a mutation. Backend failures after preflight consume
   donations too: partially completed device writes cannot be rolled back."
  [^Compiled c inputs before-mutation! before-replay!]
  (let [{:keys [executable in-tree out-tree donated target]} c
        inputs (project-aggregate-inputs in-tree inputs)
        in-nodes     (into {} (map (juxt :key identity)) in-tree)
        input-nodes  (filterv #(= :input (:role %)) in-tree)
        input-keys   (set (map :key input-nodes))
        donated-keys (set (keys donated))
        ;; 0. VALIDATE inputs: every passed key must be an :input-role param or a donated slot —
        ;;    never a :constant/:state key silently ignored (fail-loud, §7.7).
        _ (doseq [[k _v] inputs]
            (when-not (or (contains? input-keys k) (contains? donated-keys k))
              (throw (ex-info (str "invoke: unsupported input key " k " — only :input-role params "
                                   (vec input-keys) " or donated slots " (vec donated-keys)
                                   " may be passed; a :constant/:state slot is captured at bind")
                              {:key k :inputs (keys inputs)}))))
        ;; 1. An invalid later adapter must not consume an earlier handle or write inputs.
        checked-donations (checked-donations executable in-nodes donated inputs)
        ;; A malformed later input must not upload an earlier one. Reuse the LinkNode and
        ;; DeviceArray contracts, including liveness, exact ranges and portable overlap rules.
        _ (preflight-inputs! executable input-nodes inputs)
        previous (when-let [outputs (:live-outputs c)] @outputs)
        ;; 2. Every dynamic input is refreshed on every invocation, preserving the resident-program
        ;;    contract. gpu-link/write! accepts host values and performs D2D for foreign device
        ;;    values; it never materializes a DeviceArray through v/->host.
        _ (when before-mutation! (before-mutation!))
        _ (write-invocation-inputs! c input-nodes inputs previous checked-donations)]
    (when before-replay! (before-replay!))
    ;; 4. replay, no download on the ordinary invocation path.
    (gpu-link/run! executable)
    ;; 5. project outputs as resident device values; record them for next-call invalidation.
    (let [out (into {} (map (fn [{:keys [key] :as node}]
                              [key (project-node executable node target)]))
                    out-tree)]
      (when-let [live-outputs (:live-outputs c)]
        (reset! live-outputs (vec (vals out))))
      out)))

(defn- invoke-compiled-unleased [c inputs]
  (invoke-compiled-unleased* c inputs nil nil))

(defn invoke-compiled
  "Invoke a resident artifact under its linked lifetime guard. A live output lease rejects the
   entire input/donation/replay/output-invalidation sequence before any of those effects run."
  [^Compiled c inputs]
  (gpu-link/with-unleased-execution!
   (:executable c) :invoke-compiled
   #(invoke-compiled-unleased c inputs)))

(defn invoke-leased
  "Invoke a Compiled artifact and pin its owned resident DeviceArray outputs until close.

   Returns a Closeable/IDeref lease. Deref yields the ordinary output map while live. Closing the
   lease invalidates those external DeviceArray wrappers and permits the next invocation or
   artifact close; it never frees the underlying session-owned buffers. Ownership is checked
   before input writes or donation, and projection plus lease acquisition are one serialized
   operation. This is a synchronous output-lifetime contract, not a value-version snapshot or
   authorization for private temporary reuse."
  [^Compiled c inputs]
  (let [output-nodes (set (get-in c [:executable :plan :outputs]))
        projected-nodes (mapv :node (:out-tree c))
        _ (when-not (and (seq projected-nodes)
                         (every? output-nodes projected-nodes))
            (throw (ex-info "leased compiled results must project declared LinkPlan outputs"
                            {:reason :compiled-output-lease-boundary
                             :projected projected-nodes :outputs output-nodes})))
        {:keys [result lease]}
        (gpu-link/execute-and-lease!
         (:executable c) :invoke-leased
         #(invoke-compiled-unleased c inputs))
        released? (atom false)]
    (reify
      java.io.Closeable
      (close [_]
        (when (compare-and-set! released? false true)
          (try
            (doseq [device-array (vals result)] (v/free! device-array))
            (finally (.close ^java.io.Closeable lease)))))
      clojure.lang.IDeref
      (deref [_]
        (when @released?
          (throw (ex-info "compiled output lease has been released"
                          {:reason :compiled-output-lease-released})))
        @lease
        result))))

;; ================================================================
;; Inspection (§2.1) + lifecycle
;; ================================================================

(defn- require-no-evidence-events! [executable]
  (when (seq (:events @(:session executable)))
    (throw (ex-info "resident byte evidence requires no async events"
                    {:reason :compiled-evidence-unready}))))

(defrecord ResidentRepresentationEvidence [data compiled executable session session-id provenance-seal]
  clojure.lang.IDeref
  (deref [this]
    (when-not (sealed-artifact? this)
      (throw (ex-info "representation evidence requires its original owner"
                      {:reason :compiled-representation-owner})))
    data))

(defn representation-evidence? [candidate]
  (and (instance? ResidentRepresentationEvidence candidate) (sealed-artifact? candidate)))

(defn- run-storage-probe! [session dt artifact]
  (let [key (keyword (str "storage-probe-" (random-uuid)))
        allocated? (volatile! false)
        handle (volatile! nil)
        primary (volatile! nil)
        bytes (byte-array (* 2 (dtype/bytes-of dt)))]
    (try
      (gpu/alloc! session {key [dt 2 nil]})
      (vreset! allocated? true)
      (vreset! handle (gpu/bind-kernel-call! session key artifact [key]))
      (gpu/run-kernel-graph! session @handle)
      (gpu/download-range! session key (MemorySegment/ofArray bytes) {:elements 2})
      (let [observation (mapv #(bit-and 255 %) bytes)]
        {:byte-order (storage-probe/classify-bytes dt observation)
         :observation observation
         :content (numerical-content/content-address-of (MemorySegment/ofArray bytes))})
      (catch Throwable error (vreset! primary error) (throw error))
      (finally
        ;; Attempt every release, preserving the computation's first failure. A failed native
        ;; destruction may require session close; never pretend it was successfully reclaimed.
        (let [failure (volatile! @primary)]
          (doseq [release! (cond-> []
                            @handle (conj #(gpu/release-kernel-graph! session @handle))
                            @allocated? (conj #(gpu/free-buffer! session key)))]
            (try (release!)
                 (catch Throwable cleanup
                   (if-let [error @failure]
                     (when-not (identical? error cleanup) (.addSuppressed error cleanup))
                     (vreset! failure cleanup)))))
          (when (and @failure (nil? @primary)) (throw @failure))
          (when (and (nil? @failure)
                     (or (contains? (:buffers @session) key)
                         (contains? (:kernel-graphs @session) key)))
            (throw (ex-info "storage probe cleanup left owned resources live"
                            {:reason :compiled-representation-cleanup}))))))))

(defn measure-storage-representation!
  "Measure one dtype on an original wholly owned Compiled under its exclusive lifetime guard.

   Offline only: invalidates output readiness/continuity, cannot run while outputs are leased,
   and never credits a full-plan replay. Optional type capability declines occur before mutation.
   The returned sealed fact belongs to this exact executable/session; deref is historical data,
   not authority for another owner. No cache or host-order fallback is introduced."
  [c element-dtype]
  (let [program (execution-identity c)
        _ (when-not (compiled? c)
            (throw (ex-info "representation measurement requires a Compiled"
                            {:reason :compiled-evidence-unbound})))
        dt (dtype/canon element-dtype)
        executable (:executable c)
        session (:session executable)
        _ (when-not (:owns-session? executable)
            (throw (ex-info "representation evidence requires an owned session"
                            {:reason :compiled-evidence-ownership})))
        artifact (storage-probe/emit-artifact dt (gpu/kernel-body-c-dialect session))]
    (gpu-link/with-unleased-execution!
     executable :measure-storage-representation
     (fn []
       (require-no-evidence-events! executable)
       (let [device (gpu/execution-device-info session)]
         (when-not (contains? (:storage-types device) dt)
           (throw (ex-info "selected device does not advertise this storage dtype"
                           {:reason :compiled-representation-unsupported :dtype dt})))
         (gpu-link/with-exclusive-mutation!
          executable :measure-storage-representation
          (fn []
            (invalidate-live-outputs! c)
            (let [observation (run-storage-probe! session dt artifact)
                  after (gpu/execution-device-info session)]
              (require-no-evidence-events! executable)
              (when-not (= device after)
                (throw (ex-info "device identity changed during storage measurement"
                                {:reason :compiled-representation-device-changed})))
              (seal-artifact
               (->ResidentRepresentationEvidence
                {:kind :raster.compiled/resident-representation-v1
                 :dtype dt :byte-order (:byte-order observation)
                 :abi {:encoding (if (:fp? (dtype/info dt)) :ieee-binary :twos-complement)
                       :bits (* 8 (dtype/bytes-of dt))}
                 :program-fingerprint (:fingerprint program)
                 :device device
                 :device-fingerprint (semantic-fingerprint/fingerprint device)
                 :probe-fingerprint (semantic-fingerprint/fingerprint artifact)
                 :observed-bytes (:observation observation)
                 :observed-content (:content observation)}
                c executable session (:session-id @session) nil))))))))))

(defn completed-storage-description
  "Project actual completed bytes onto measured raw-array storage for this exact live owner.

   `facts` is a dtype-keyed map of original ResidentRepresentationEvidence. All receipt dtypes
   require their own fact. The receipt's output lease must remain live. Returns portable data,
   not a new authority token or publication; ordinary manifest/byte verification still applies."
  [receipt facts]
  (when-not (completed-evidence? receipt)
    (throw (ex-info "storage projection requires original completed evidence"
                    {:reason :compiled-completed-evidence-owner})))
  (let [executable (:executable receipt)
        session (:session executable)]
    (locking (:lifetime-lock executable)
      @(:lease receipt)
      (let [device-fingerprint (semantic-fingerprint/fingerprint (gpu/execution-device-info session))
            data @receipt
            dtypes (into #{} (map (comp dtype/canon :dtype val))
                         (concat (:inputs data) (:outputs data) (:post-state data)))]
        (doseq [dt dtypes]
          (let [fact (get facts dt)]
            (when-not (and (representation-evidence? fact)
                           (compiled? (:compiled fact)) (sealed-artifact? (:compiled fact))
                           (identical? executable (:executable (:compiled fact)))
                           (identical? executable (:executable fact))
                           (identical? session (:session fact))
                           (= (:session-id @session) (:session-id fact))
                           (= dt (get-in fact [:data :dtype]))
                           (= (:program-fingerprint data) (get-in fact [:data :program-fingerprint]))
                           (= device-fingerprint (get-in fact [:data :device-fingerprint])))
              (throw (ex-info "completed storage requires a matching live owner/dtype measurement"
                              {:reason :compiled-representation-mismatch :dtype dt})))))
        (let [project (fn [nodes]
                        (update-vals nodes
                          (fn [node]
                            (let [dt (dtype/canon (:dtype node))
                                  order (get-in facts [dt :data :byte-order])]
                              (assoc node :storage
                                     {:format :raw-array
                                      :byte-order (if (= :order-invariant order) :little-endian order)})))))]
          {:kind :raster.compiled/completed-storage-description-v1
           :completed-fingerprint (:fingerprint data)
           :program-fingerprint (:program-fingerprint data)
           :representations (into {} (map (fn [dt] [dt @(get facts dt)]) dtypes))
           :inputs (project (:inputs data)) :outputs (project (:outputs data))
           :post-state (project (:post-state data))})))))

(defn- completed-frontier! [c]
  (let [executable (:executable c)
        plan (:plan executable)
        evidence (get-in c [:lowering :certificate :effect-evidence])
        initialization (:initialization evidence)
        fail (fn [reason] (throw (ex-info "compiled execution cannot attest resident byte history"
                                         {:reason reason})))
        _ (when-not (and (:owns-session? executable)
                         (every? #(= :owned (get-in % [:view :allocation :ownership]))
                                 (vals (:nodes plan))))
            (fail :compiled-evidence-ownership))
        _ (when-not (link-plan/retained-effect-evidence? plan evidence)
            (fail :compiled-evidence-initialization))
        _ (require-no-evidence-events! executable)
        _ (when (seq (:record-time-prologue (gpu-link/execution-order executable)))
            (fail :compiled-evidence-record-time-prologue))
        roots (set/union (:requires initialization) (:initializers initialization))
        outputs (set (:outputs plan))
        written (mapv #(get-in plan [:nodes % :view]) (:writes initialization))
        state (into #{} (filter (fn [node]
                                 (some #(bview/overlaps? (get-in plan [:nodes node :view]) %)
                                       written))) roots)]
    (doseq [node (set/union roots outputs)]
      (let [view (get-in plan [:nodes node :view])]
        (when-not (and view (= (:strides view) (bview/dense-strides (:shape view))))
          (fail :compiled-evidence-noncontiguous))))
    {:roots roots :outputs outputs :state state}))

(defn- require-evidence-ready! [executable]
  (require-no-evidence-events! executable)
  (when (or (seq @(:pending-inputs executable)) (seq @(:tainted-inputs executable)))
    (throw (ex-info "resident byte evidence requires initialized storage and no async events"
                    {:reason :compiled-evidence-unready}))))

(defn- snapshot-resident-bytes [executable nodes]
  (require-evidence-ready! executable)
  (into {}
        (map (fn [node]
               (let [view (get-in executable [:plan :nodes node :view])
                     resident (gpu-link/node-view executable node)
                     element-bytes (dtype/bytes-of (:dtype view))
                     content (numerical-content/content-address-from-reader
                              (:byte-length view)
                              (fn [offset ^MemorySegment destination]
                                (let [n (.byteSize destination)]
                                  (gpu/download-range! (:session executable) resident destination
                                                       {:src-element (quot offset element-bytes)
                                                        :elements (quot n element-bytes)})
                                  n)))]
                 [node (assoc (select-keys view [:dtype :shape :strides :byte-length])
                              :representation :device-native
                              :content content)])))
        nodes))

(defn- bound-schedule-evidence! [executable]
  (mapv (fn [index {:keys [instance executable]}]
          (when-not (and (map? executable) (seq (:entry-points executable))
                         (every? string? (:entry-points executable)))
            (throw (ex-info "bound schedule lacks retained executable evidence"
                            {:reason :compiled-evidence-bound-schedule :index index})))
          {:index index :instance instance
           :executable (select-keys executable [:kind :strategy :precision :entry-points :selection])})
        (range) (gpu-link/execution-info executable)))

(defn invoke-with-evidence
  "Execute an original compiler-owned Compiled and attest actual resident bytes offline.

   SHA-256 input snapshots follow all invocation writes, never Prepared defaults or caller hashes.
   Outputs and mutated initialization roots are read after synchronous completion. The returned
   Closeable pins outputs for publication; completed-output-values exposes them while live.
   Deref returns historical portable evidence. Close invalidates its projected DeviceArrays.

   Requires complete retained program identity, wholly owned session/storage, dense views, no
   record-time prologue or async events. Adjacent witnessed mutable state can name the previous
   same-owner receipt; ordinary replay/measurement/tuning/failed evidence breaks continuity.
   This does not publish durable blobs, prove mathematical equivalence, or track raw-session
   mutation outside the LinkedExecutable ownership contract. Bytes retain opaque device-native
   representation; no host/device endianness agreement or canonical numerical codec is inferred.
   Ordinary invocation is unchanged."
  [c inputs]
  (let [program (execution-identity c)
        _ (when-not (compiled? c)
            (throw (ex-info "resident evidence requires an instantiated artifact"
                            {:reason :compiled-evidence-unbound})))
        executable (:executable c)]
    (gpu-link/with-unleased-execution!
     executable :invoke-with-evidence
     (fn []
       (let [{:keys [roots outputs state]} (completed-frontier! c)
             schedules (bound-schedule-evidence! executable)
             owner-state @(:execution-state executable)
             previous (:completed-evidence owner-state)
             mutated? (volatile! false)
             before (volatile! nil)
             held (volatile! nil)]
         (try
           (let [{:keys [result lease]}
                 (gpu-link/execute-and-lease!
                  executable :invoke-with-evidence
                  #(invoke-compiled-unleased*
                    c inputs (fn [] (vreset! mutated? true))
                    (fn [] (vreset! before (snapshot-resident-bytes executable roots)))))
                 _ (vreset! held lease)
                 after (snapshot-resident-bytes executable outputs)
                 post-state (snapshot-resident-bytes executable state)
                 parent (when (and (seq state) (completed-evidence? previous)
                                   (identical? executable (:executable previous))
                                   (= (:epoch previous) (:value-epoch owner-state))
                                   (= (:fingerprint program) (get-in previous [:data :program-fingerprint]))
                                   (= (select-keys @before state) (get-in previous [:data :post-state])))
                          (get-in previous [:data :fingerprint]))
                 data {:kind :raster.compiled/completed-resident-bytes-v1
                       :scope :completed-linked-replay :program-fingerprint (:fingerprint program)
                       :bound-schedules schedules :inputs @before :outputs after :post-state post-state
                       :parent parent :attests-resident-bytes? true}
                 data (assoc data :fingerprint (semantic-fingerprint/fingerprint data))
                 receipt (seal-artifact
                          (->CompletedEvidence data lease executable
                                               (:value-epoch @(:execution-state executable))
                                               result (atom false) nil))]
             (swap! (:execution-state executable) assoc :completed-evidence receipt)
             receipt)
           (catch Throwable error
             (when @mutated?
               (swap! (:execution-state executable) dissoc :completed-evidence)
               (invalidate-live-outputs! c))
             (when-let [lease @held]
               (try (.close ^java.io.Closeable lease)
                    (catch Throwable cleanup
                      (when-not (identical? error cleanup) (.addSuppressed error cleanup)))))
             (throw error))))))))

(defn explain
  "Print the artifact's shape: in-tree / out-tree / donation plan / target / schedule and the
   resident step-kind histogram. Works before or after instantiation and returns its argument."
  [c]
  (let [{:keys [in-tree out-tree donated target schedule descriptor]} c
        kinds (frequencies (map :convention (:steps descriptor)))]
    (println (if (prepared? c) "Prepared artifact" "Compiled artifact"))
    (println "  target   :" target)
    (println "  in-tree  :" (mapv (fn [n] [(:key n) (:role n) (when (:donate? n) :donate)]) in-tree))
    (println "  out-tree :" (mapv (fn [n] [(:key n) (:from n)]) out-tree))
    (println "  donated  :" donated)
    (println "  schedule :" (or schedule :none))
    (println "  steps    :" (count (:steps descriptor)) "resident, by kind:" kinds))
  c)

(defn execution-info
  "Report the instantiated artifact's admitted resident schedules, without running it.
   Unlike `ir`, this observes binding-time selection. It is not a replay or timing report."
  [c]
  (when-not (compiled? c)
    (throw (ex-info "execution info requires an instantiated compiled artifact"
                    {:reason :compiled-execution-info-unbound})))
  (gpu-link/execution-info (:executable c)))

(defn ir
  "Return the descriptor's resident steps (the artifact's lowered IR) for inspection."
  [c]
  (mapv #(select-keys % [:convention :phase :variant :kernel-name]) (:steps (:descriptor c))))

(defn plan
  "Return the validated LinkPlan that is the artifact's public executable representation."
  [c]
  (:plan (:lowering c)))

(defn certificate
  "Return the checkable resident or composition lowering certificate."
  [c]
  (:certificate (:lowering c)))

(defn- refresh-captured-inputs!
  [^Compiled c]
  (let [input-nodes (filterv #(= :input (:role %)) (:in-tree c))
        previous (when-let [outputs (:live-outputs c)] @outputs)]
    (preflight-inputs! (:executable c) input-nodes {})
    (write-invocation-inputs! c input-nodes {} previous []))
  c)

(defn profile
  "Device-event profile of one linked replay. The artifact must have been compiled with
   `{:profile? true}`. Returns the historical profile map, including downloaded semantic results."
  [^Compiled c]
  (gpu-link/with-unleased-execution!
   (:executable c) :profile
   (fn []
     (refresh-captured-inputs! c)
     (let [profile-result (gpu-link/profile! (:executable c))
           result (into {} (map (fn [{:keys [key node]}]
                                  [key (gpu-link/download (:executable c) node)]))
                        (:out-tree c))]
       (assoc profile-result :result result)))))

(defn measure
  "Explicit offline device-event measurement of a Compiled artifact.

   The artifact must have been compiled with {:profile? true}. Returns a stable Measurement;
   options are those of gpu-link/measure!, including the required :before-sample! restore hook
   for stateful programs. This does not choose or cache a schedule by itself."
  [^Compiled c & {:as opts}]
  (gpu-link/with-unleased-execution!
   (:executable c) :measure
   (fn []
     (refresh-captured-inputs! c)
     (apply gpu-link/measure! (:executable c) (mapcat identity opts)))))

(defn cache-key
  "The serializable identity of the artifact minus closures (§2b C5): in/out trees, donation,
   schedule, target, and the descriptor's step kinds + concrete shapes. Excludes the non-
   serializable closures (:n-fn/:m-fn/…) and SPIR-V modules."
  [c]
  {:in-tree  (mapv #(select-keys % [:key :role :donate? :shape :dtype]) (:in-tree c))
   :out-tree (mapv #(select-keys % [:key :from :shape :dtype]) (:out-tree c))
   :donated  (:donated c)
   :schedule (:schedule c)
   :target   (:target c)
   :steps    (frequencies (map :convention (:steps (:descriptor c))))})

(defn close!
  "Release the artifact's linked executable (buffers + graph + kernels). Invalidates
   any still-live projected output values FIRST, so a `->host` on a value returned before close!
   fails loud (use-after-free) instead of copying from a zeMemFree'd segment (SIGSEGV/garbage)."
  [^Compiled c]
  (let [executable (:executable c)]
    (locking (:lifetime-lock executable)
      (if @(:closed? executable)
        (invalidate-live-outputs! c)
        (gpu-link/with-unleased-execution!
         executable :compiled-close!
         #(do (invalidate-live-outputs! c)
              (gpu-link/close! executable))))))
  nil)

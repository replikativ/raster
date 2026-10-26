(ns raster.compiler.ir.emitted-parallel-program
  "Validation for an equation-first emitted ParallelProgram.

   The target backend may replace scheduled operations, but the retained typed algorithms and
   ordered program dataflow remain authoritative. Host-only scalar equations are the only
   equations with an empty emitted operation sequence."
  (:require [clojure.set :as set]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]))

(def ^:private dialect-targets
  {:opencl-parallel :opencl-c
   :cuda-parallel :cuda-c
   :hip-parallel :hip-cpp})

(defn emitted-operation?
  [operation]
  (or (emitted-loop/emitted-loop? operation)
      (equation-dispatch/emitted-equation-dispatch? operation)
      (emitted-equation/emitted-equation? operation)))

(defn- equation-candidates
  [operation caller-options]
  (cond
    (emitted-equation/emitted-equation? operation)
    [(if (nil? caller-options)
       (emitted-equation/validate! operation)
       (emitted-equation/validate! operation caller-options))]
    (equation-dispatch/emitted-equation-dispatch? operation)
    (if (nil? caller-options)
      (equation-dispatch/candidates operation)
      (equation-dispatch/candidates operation caller-options))
    :else []))

(defn- emitted-boundary-with-candidates?
  [equation algorithm candidates-for caller-options]
  (cond
    (control/loop-program? algorithm)
    (and (= 1 (count (:operations equation)))
         (let [operation (first (:operations equation))]
           (and (emitted-loop/emitted-loop? operation)
                (= algorithm (:algorithm (:schedule
                                          (if (nil? caller-options)
                                            (emitted-loop/validate! operation)
                                            (emitted-loop/validate! operation caller-options))))))))

    (soac/program-form? algorithm)
    (if (true? (get-in equation [:attributes :host-only]))
      (and (empty? (:operations equation))
           (every? #(= 'scalar (soac/operation-kind %)) (soac/equations algorithm)))
      (and (= 1 (count (:operations equation)))
           (let [operation (first (:operations equation))
                 candidates (candidates-for operation)]
             (and (seq candidates)
                  (every?
                   (fn [emitted]
                     (let [refinement (:refinement emitted)
                           compound? (seq (get-in equation
                                                  [:attributes :emitted-source-equations]))
                           source-graph (:source refinement)
                           retained-operands
                           (when source-graph
                             (set (concat (map :id (:inputs source-graph))
                                          (map :id (:scalars source-graph)))))
                           expected-operands
                           (if compound?
                             (filterv retained-operands (:inputs (soac/facts algorithm)))
                             (:inputs (soac/facts algorithm)))]
                       (and (= algorithm (:algorithm emitted))
                            (or (not compound?) (some? source-graph))
                            (= (:operands equation) expected-operands)
                            (= (:results equation) (soac/outputs algorithm)))))
                   candidates)))))

    (swr/plan? algorithm)
    (and (= 1 (count (:operations equation)))
         (let [operation (first (:operations equation))
               candidates (candidates-for operation)]
           (and (seq candidates)
                (every? (fn [emitted]
                          (and (= algorithm (:algorithm emitted))
                               (swr/equation-boundary? (get-in emitted [:body :values])
                                                       equation algorithm (:body emitted))))
                        candidates))))

    :else false))

(defn- boundary-for-request?
  [equation algorithm caller-options]
  (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
  (emitted-boundary-with-candidates? equation algorithm
                                   #(equation-candidates % caller-options) caller-options))

(defn emitted-boundary?
  "Independently validate an emitted equation's semantic boundary."
  ([equation algorithm] (boundary-for-request? equation algorithm nil))
  ([equation algorithm caller-options] (boundary-for-request? equation algorithm caller-options)))

(defn- scheduled-equation-view
  "The outer equation contract before target emission replaces its operation sequence."
  [equation]
  (dissoc equation :operations))

(defn- candidate-numerical-equations
  [candidate]
  (filterv #(not (true? (get-in % [:attributes :host-only])))
           (get-in candidate [:body :equations])))

(defn equation-fragment
  "Construct an exact equation view without execution, retaining revalidatable parent provenance.
   No source, InvocationPlan, completion authority or validation seal is transferred."
  [parallel-program equation-id]
  (program/validate! parallel-program)
  (when (contains? (:attributes parallel-program) ::equation-origin)
    (throw (ex-info "nested emitted equation origins are not supported"
                    {:reason :emitted-parallel-program-fragment-origin})))
  (let [equation (some #(when (= equation-id (:id %)) %) (:equations parallel-program))]
    (when-not (and equation (not (true? (get-in equation [:attributes :host-only]))))
      (throw (ex-info "an emitted fragment requires an existing numerical equation"
                      {:reason :emitted-parallel-program-fragment-equation
                       :equation equation-id})))
    (let [prefix (if (emitted-loop/emitted-loop? (first (:operations equation))) []
                    (filterv #(true? (get-in % [:attributes :host-only]))
                             (take-while #(not= equation-id (:id %))
                                         (:equations parallel-program))))
          equations (conj prefix equation)]
      (program/make
       {:dialect (:dialect parallel-program) :values (:values parallel-program)
        :inputs (program/infer-inputs equations) :outputs (:results equation)
        :equations equations :effects (reduce set/union #{} (map :effects equations))
        :source nil :provenance {:source :ordered-numerical-fragment :equation equation-id}
        :attributes (assoc (select-keys (:attributes parallel-program) [:allocations])
                           ::equation-origin {:program parallel-program :equation-id equation-id})}))))

(defn- canonical-candidate-source-region!
  "Recover one exact scheduled source region from already validated emitted alternatives."
  [equation candidates]
  (let [regions (mapv candidate-numerical-equations candidates)
        canonical (first regions)
        source-equations (get-in equation [:attributes :emitted-source-equations])
        exact-region?
        (if source-equations
          #(= source-equations (mapv :id %))
          #(and (= 1 (count %))
                (= (scheduled-equation-view equation)
                   (scheduled-equation-view (first %)))))]
    (when-not (and (seq regions)
                   (every? exact-region? regions)
                   ;; Each candidate already validates its own target schedule. Distinct
                   ;; reference/subgroup operations must retain the same semantic source,
                   ;; not the same independently certified physical schedule.
                   (every? #(semantic-fingerprint/equivalent?
                             (mapv scheduled-equation-view canonical)
                             (mapv scheduled-equation-view %))
                           (next regions)))
      (throw (ex-info "emitted alternatives differ from their retained scheduled source region"
                      {:reason :emitted-parallel-program-source-region
                       :outer-equation (:id equation)
                       :expected-source-equations source-equations
                       :actual-source-equations (mapv #(mapv :id %) regions)
                       :ir :emitted-parallel-program})))
    canonical))

(defn- expanded-source-program!
  "Reconstruct the exact scheduled equation spine retained inside validated emitted candidates.

   Compound emission replaces constituent IDs on the outer spine. The candidate body deliberately
   retains those scheduled constituents, so validation expands them here without reparsing source
   or trusting `:emitted-source-equations` alone. The original InvocationPlan remains the boundary
   owner and must match the expanded dataflow before it can contribute scalar constraints."
  [parallel-program candidates-for]
  (let [{:keys [equations regions]}
        (reduce
         (fn [{:keys [equations regions]} equation]
           (let [candidates (when-let [operation (first (:operations equation))]
                              (seq (candidates-for operation)))
                 region (if candidates
                          (canonical-candidate-source-region! equation (vec candidates))
                          [equation])]
             {:equations (into equations region)
              :regions (assoc regions (:id equation) region)}))
         {:equations [] :regions {}} (:equations parallel-program))
        plan (get-in parallel-program [:attributes :invocation-plan])
        plan (when plan (invocation/validate! plan))
        source-inputs (if plan (:program-inputs plan) (:inputs parallel-program))
        source-outputs (if plan (:program-outputs plan) (:outputs parallel-program))]
    (when-not (= source-outputs (:outputs parallel-program))
      (throw (ex-info "expanded emitted constituents differ from the retained invocation boundary"
                      {:reason :emitted-parallel-program-source-boundary
                       :expected-outputs source-outputs
                       :actual-outputs (:outputs parallel-program)
                       :ir :emitted-parallel-program})))
    {:program (program/validate!
               (assoc parallel-program :inputs source-inputs :equations (vec equations)
                      :outputs source-outputs))
     :regions regions}))

(defn- validate-host-prefix-slices!
  "Bind every narrowed emitted body to its exact enclosing host-scalar execution prefix.

   An EmittedParallelEquation can validate a self-contained numerical graph, but a graph-storage
   extent may also use a preceding host scalar.  This outer check prevents an isolated body from
   substituting a different (though locally valid) prefix: its host equations, terminal numerical
   equation, inferred inputs, outputs, and effects must be the exact slice of this program."
  [parallel-program candidates-for]
  (let [;; validate-program! has independently checked this exact parent/view relation.
        ;; The origin is ordinary provenance, never a stored guard or validation capability.
        source-context (or (get-in parallel-program [:attributes ::equation-origin :program])
                           parallel-program)
        {source-program :program source-regions :regions}
        (when (some (fn [equation]
                      (some (fn [candidate]
                              (equation-graph/invocation-scalar-proof-demand?
                               source-context (candidate-numerical-equations candidate)))
                            (candidates-for (first (:operations equation)))))
                    (:equations parallel-program))
          (expanded-source-program! source-context candidates-for))]
  (loop [host-prefix [] remaining (:equations parallel-program)]
    (when-let [equation (first remaining)]
      (if (true? (get-in equation [:attributes :host-only]))
        (recur (conj host-prefix equation) (next remaining))
        (let [operation (first (:operations equation))]
          (doseq [candidate (candidates-for operation)]
            (let [body (:body candidate)
                  body-equations (:equations body)
                  actual-prefix (filterv #(true? (get-in % [:attributes :host-only]))
                                         body-equations)
                  numerical-body (candidate-numerical-equations candidate)
                  source-equations (get-in equation [:attributes :emitted-source-equations])
                  source-region (get source-regions (:id equation))
                  expected-invocation-preconditions
                  (if source-program
                    (equation-graph/invocation-scalar-preconditions
                     source-program source-region)
                    [])
                  actual-invocation-preconditions
                  (equation-graph/body-invocation-scalar-preconditions body)
                  exact-source?
                  (if source-equations
                    (= source-equations (mapv :id numerical-body))
                    (and (= 1 (count numerical-body))
                         (= (scheduled-equation-view equation)
                            (scheduled-equation-view (first numerical-body)))))
                  expected-inputs (program/infer-inputs body-equations)
                  expected-effects (reduce set/union #{} (map :effects body-equations))]
              (when (swr/plan? (:algorithm equation))
                (let [source-equation (assoc equation :operations
                                             (:operations (first numerical-body)))
                      source-program (update parallel-program :equations
                                             #(mapv (fn [entry]
                                                      (if (= (:id entry) (:id equation))
                                                        source-equation entry)) %))
                      expected-body (:body (equation-graph/make-for-plan-equation
                                            (assoc source-program :dialect (:dialect body))
                                            source-equation))]
                  (when-not (semantic-fingerprint/equivalent? expected-body body)
                    (throw (ex-info "emitted reduction changed its enclosing value or allocation contract"
                                    {:reason :emitted-reduction-outer-slice
                                     :equation (:id equation)})))))
              (when-not (and (= host-prefix actual-prefix)
                             (= expected-invocation-preconditions
                                actual-invocation-preconditions)
                             exact-source?
                             (= expected-inputs (:inputs body))
                             (= (:results equation) (:outputs body))
                             (= expected-effects (:effects body)))
                (throw (ex-info "emitted numerical body differs from its enclosing host-scalar prefix slice"
                                {:reason :emitted-parallel-program-host-prefix
                                 :outer-equation (:id equation)
                                 :expected-prefix (mapv :id host-prefix)
                                 :actual-prefix (mapv :id actual-prefix)
                                 :expected-source-equations source-equations
                                 :actual-source-equations (mapv :id numerical-body)
                                 :expected-invocation-scalar-preconditions
                                 expected-invocation-preconditions
                                 :actual-invocation-scalar-preconditions
                                 actual-invocation-preconditions
                                 :expected-inputs expected-inputs :actual-inputs (:inputs body)
                                 :ir :emitted-parallel-program})))))
          (recur host-prefix (next remaining))))))))

(defn- validate-program!
  [parallel-program candidates-fn caller-options]
  (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
  (when-not (contains? dialect-targets (:dialect parallel-program))
    (throw (ex-info "emitted parallel program requires a supported C-family dialect"
                    {:reason :emitted-parallel-program-dialect
                     :dialect (:dialect parallel-program)
                     :supported (set (keys dialect-targets))
                     :ir :emitted-parallel-program})))
  (let [origin (get-in parallel-program [:attributes ::equation-origin])
        _ (when (contains? (:attributes parallel-program) ::equation-origin)
            (when-not (and (map? origin) (= #{:program :equation-id} (set (keys origin))))
              (throw (ex-info "emitted equation origin requires an exact parent and equation"
                              {:reason :emitted-parallel-program-fragment-origin})))
            ;; Revalidate under CURRENT caller intent. A copied seal or embedded guard vector
            ;; is never authority, and equation-fragment rejects nested origins before recursion.
            (let [expected (equation-fragment (:program origin) (:equation-id origin))]
              (validate-program! (:program origin) #(equation-candidates % caller-options)
                                 caller-options)
              (when-not (= expected parallel-program)
                (throw (ex-info "emitted fragment differs from its exact parent equation view"
                                {:reason :emitted-parallel-program-fragment-origin})))))
        candidate-cache (java.util.IdentityHashMap.)
        candidates-for (fn [operation]
                         ;; A validation invocation checks an immutable operation repeatedly:
                         ;; semantic boundary, enclosing host prefix, and target membership.
                         ;; Reuse only this exact object's checked candidates within this call;
                         ;; no certificate or cached result escapes to a later invocation.
                         (if (.containsKey candidate-cache operation)
                           (.get candidate-cache operation)
                           (let [candidates (candidates-fn operation)]
                             (.put candidate-cache operation candidates)
                             candidates)))
        parallel-program
        (program/validate! parallel-program emitted-operation?
                           (fn [equation algorithm]
                             (emitted-boundary-with-candidates?
                              equation algorithm candidates-for caller-options)))
        expected-target (get dialect-targets (:dialect parallel-program))
        _ (validate-host-prefix-slices! parallel-program candidates-for)
        artifacts
        (vec
        (for [equation (:equations parallel-program)
               operation (:operations equation)
               kernel-graph (if (emitted-loop/emitted-loop? operation)
                              [(:graph operation)]
                              (mapv :graph (candidates-for operation)))
               node (:nodes kernel-graph)]
           ;; Plain/dispatch candidates have already checked every artifact through their
           ;; executable validator. Project target membership from those exact objects.
           ;; Structured-loop graph validation alone is not an artifact validator: keep the
           ;; independent check for that path, including legacy operation certificates.
           (if (emitted-loop/emitted-loop? operation)
             (artifact/validate! (:operation node))
             (:operation node))))
        mismatches (filterv #(not= expected-target (:target %)) artifacts)]
    (when (seq mismatches)
      (throw (ex-info "emitted program dialect disagrees with a contained kernel target"
                      {:reason :emitted-parallel-program-target
                       :dialect (:dialect parallel-program)
                       :expected-target expected-target
                       :artifact-targets (mapv :target artifacts)
                       :ir :emitted-parallel-program})))
    parallel-program))

(defn validate!
  "Validate a fully emitted equation-first program without depending on a target backend."
  ([parallel-program]
   (validate-program! parallel-program #(equation-candidates % nil) nil))
  ([parallel-program caller-options]
   (validate-program! parallel-program #(equation-candidates % caller-options) caller-options)))

(defn- numerical-equations-for-request
  "Project ordered top-level semantic equations from a validated emitted program, before fusion.

   Host-only shape/scalar equations and target operation sequences are excluded. A compound
   emission expands its retained numerical body, not its kernel count or generated name.
   Nested control remains in its enclosing equation's algorithm; this does not flatten loops.
   Every dispatch alternative must preserve the same semantic projection. This is structural
   provenance, not a proof of an application's mathematical interpretation of these equations."
  ([parallel-program caller-options]
   (numerical-equations-for-request parallel-program caller-options false))
  ([parallel-program caller-options include-host?]
  (if (nil? caller-options)
    (validate! parallel-program)
    (validate! parallel-program caller-options))
  (let [equations
        (vec
         (mapcat
          (fn [equation]
            (cond
              (true? (get-in equation [:attributes :host-only]))
              (if include-host? [(scheduled-equation-view equation)] [])
              (seq (get-in equation [:attributes :emitted-source-equations]))
              (let [expected (get-in equation [:attributes :emitted-source-equations])
                    projections
                    (mapv (fn [candidate]
                            (mapv scheduled-equation-view
                                  (candidate-numerical-equations candidate)))
                          (equation-candidates (first (:operations equation)) caller-options))
                    projected (first projections)]
                (when-not (and (seq projections)
                               (= expected (mapv :id projected))
                               (every? #(semantic-fingerprint/equivalent? projected %) projections))
                  (throw (ex-info "compound alternatives differ in retained semantic equations"
                                  {:reason :emitted-program-semantic-projection
                                   :equation (:id equation)})))
                projected)
              :else [(scheduled-equation-view equation)]))
          (:equations parallel-program)))]
    (when-not (= (count equations) (count (distinct (map :id equations))))
      (throw (ex-info "retained semantic equations require distinct identities"
                      {:reason :emitted-program-semantic-identities})))
    equations)))

(defn retained-numerical-equations
  "Project retained numerical equations only after independent validation under caller intent."
  ([parallel-program] (numerical-equations-for-request parallel-program nil))
  ([parallel-program caller-options]
   (numerical-equations-for-request parallel-program caller-options)))

(defn emission-inputs
  "Physical inputs of an emitted equation spine. Compound regions eliminate internal
   intermediates; unchanged spines retain the scheduled entry interface. This constructor
   policy is shared by emission and checked logical projection, not an arbitrary subset rule."
  [scheduled emitted-equations]
  (if (some #(seq (get-in % [:attributes :emitted-source-equations])) emitted-equations)
    (program/infer-inputs emitted-equations)
    (:inputs scheduled)))

(defn- retained-schedule-matches?
  "Relate a scheduled peer to checked emitted alternatives. Attached invocation-plan
   correspondence is exact identity, not ownership proof or portable numerical data."
  [emitted scheduled equations caller-options]
  (program/validate! scheduled)
  (let [by-id (into {} (map (juxt :id identity)) (:equations scheduled))
        numerical-boundary #(update (select-keys % [:values :outputs :effects :attributes])
                                    :attributes dissoc :invocation-plan)]
    (and (contains? #{:segop :scheduled-parallel} (:dialect scheduled))
         (= (contains? (:attributes scheduled) :invocation-plan)
            (contains? (:attributes emitted) :invocation-plan))
         (identical? (get-in scheduled [:attributes :invocation-plan])
                     (get-in emitted [:attributes :invocation-plan]))
         (= (:inputs emitted) (emission-inputs scheduled (:equations emitted)))
         (semantic-fingerprint/equivalent?
          (numerical-boundary scheduled) (numerical-boundary emitted))
         (= (mapv :id (:equations scheduled)) (mapv :id equations))
         (every?
          (fn [equation]
            (if (true? (get-in equation [:attributes :host-only]))
              (semantic-fingerprint/equivalent? equation (get by-id (:id equation)))
              (some (fn [candidate]
                      (every? #(semantic-fingerprint/equivalent? % (get by-id (:id %)))
                              (candidate-numerical-equations candidate)))
                    (equation-candidates (first (:operations equation)) caller-options))))
          (:equations emitted)))))

(defn retained-typed-program-view
  "Project a flat TypedSOAC analysis view only after checking the actual emitted program.
   Retained algorithms, checked host prefixes and allocation context remain authoritative;
   target operations are not accepted by tag or silently discarded before validation.
   An optional scheduled peer must match actual retained source schedules, not just their tags.
   This is not an executable, original-entry completion or invocation-plan capability.
   Mixed algorithm families abstain rather than broadening semantic analyses."
  ([parallel-program] (retained-typed-program-view parallel-program nil nil))
  ([parallel-program caller-options]
   (retained-typed-program-view parallel-program nil caller-options))
  ([parallel-program scheduled-peer caller-options]
   (let [equations (numerical-equations-for-request parallel-program caller-options true)]
     (when (and (every? #(soac/program-form? (:algorithm %)) equations)
                ;; Without a checked peer, a physical compound interface may have lost
                ;; logical intermediates. Abstain rather than invent external inputs.
                (or scheduled-peer
                    (set/subset? (set (program/infer-inputs equations))
                                 (set (:inputs parallel-program))))
                (or (nil? scheduled-peer)
                    (retained-schedule-matches? parallel-program scheduled-peer equations caller-options)))
       (program/make
        {:dialect :typed-soac :source nil
         :values (:values parallel-program)
         :inputs (:inputs (or scheduled-peer parallel-program))
         :outputs (:outputs parallel-program) :effects (:effects parallel-program)
         :diagnostics (:diagnostics parallel-program)
         :provenance (assoc (:provenance parallel-program) :pass :retained-typed-analysis)
         :attributes (dissoc (:attributes parallel-program) :invocation-plan)
         :equations (mapv (fn [equation]
                            (with-meta
                              (program/map->ProgramEquation
                               (assoc equation :operations
                                      (if (true? (get-in equation [:attributes :host-only]))
                                        [] (soac/equations (:algorithm equation)))))
                              (meta equation))) equations)})))))

(def ^:private validation-evidence-seal-token (Object.))

(defn- validation-seal [program-ref owner policy]
  (fn [candidate-program candidate-evidence requested-policy]
    (when (and (some? candidate-program)
               (identical? (.get ^java.lang.ref.WeakReference program-ref) candidate-program)
               (identical? @owner candidate-evidence)
               (= policy requested-policy))
      validation-evidence-seal-token)))

(def ^:private validation-seal-class (class (validation-seal nil nil nil)))

(defn- seal-validation-evidence [parallel-program evidence policy]
  (let [program-ref (java.lang.ref.WeakReference. parallel-program)
        owner (volatile! nil)
        sealed (with-meta evidence
                 {::validation-seal (validation-seal program-ref owner policy)})]
    (vreset! owner sealed)
    sealed))

(defn- retained-validation-for-request?
  "True only for the exact in-process program/evidence objects independently validated here.
   A copied or modified program/evidence does not inherit validation authority."
  [parallel-program evidence caller-options]
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        seal (::validation-seal (meta evidence))]
    (and (.isInstance ^Class validation-seal-class seal)
         (identical? validation-evidence-seal-token (seal parallel-program evidence policy)))))

(defn ^:no-doc retained-validation?
  "True only for the exact program/evidence pair and independently supplied scalar-math request."
  ([parallel-program evidence] (retained-validation-for-request? parallel-program evidence nil))
  ([parallel-program evidence caller-options]
   (retained-validation-for-request? parallel-program evidence caller-options)))

(defn- checked-retained-validation-for-request!
  "Consume only an exact independently validated program/evidence pair. This does not validate
   current invocation bindings, and does not replace the independent public validators."
  [parallel-program evidence caller-options]
  (when-not (retained-validation-for-request? parallel-program evidence caller-options)
    (throw (ex-info "retained emitted-program validation does not belong to this exact program and math request"
                    {:reason :emitted-parallel-program-retained-validation
                     :ir :emitted-parallel-program})))
  evidence)

(defn ^:no-doc checked-retained-validation!
  "Consume only exact independently validated program/evidence/request ownership."
  ([parallel-program evidence] (checked-retained-validation-for-request! parallel-program evidence nil))
  ([parallel-program evidence caller-options]
   (checked-retained-validation-for-request! parallel-program evidence caller-options)))

(defn- physical-validation-for-request!
  "Independently validate a program and retain read-only exact-operation projections.
   Evidence is sealed to this exact immutable program and evidence object; retained-validation?
   checks ownership before internal reuse. Public validators still independently check programs.
   Dispatch reports retain the certified fallback, not a runtime selection. Structured loops
   remain independently validated and have no projection in this map."
  [parallel-program caller-options]
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        projections (java.util.IdentityHashMap.)
        candidates-for
        (fn [operation]
          (cond
            (emitted-equation/emitted-equation? operation)
            (let [projection (if (nil? caller-options)
                               (emitted-equation/validate-with-physical-results operation)
                               (emitted-equation/validate-with-physical-results operation caller-options))
                  boundary (:boundary projection)]
              (.put projections operation (assoc projection :candidates [boundary]))
              [boundary])
            (equation-dispatch/emitted-equation-dispatch? operation)
            (let [projection (if (nil? caller-options)
                               (equation-dispatch/validate-with-boundary operation)
                               (equation-dispatch/validate-with-boundary operation caller-options))]
              (.put projections operation projection)
              (:candidates projection))
            :else (equation-candidates operation caller-options)))
        checked (validate-program! parallel-program candidates-for caller-options)]
    (seal-validation-evidence
     checked {:program checked :projections (java.util.Collections/unmodifiableMap projections)} policy)))

(defn ^:no-doc validate-with-physical-results!
  "Validate independently and seal exact-operation projections to this program and math request.
   Neither the evidence payload nor retained candidate metadata can grant caller consent."
  ([parallel-program] (physical-validation-for-request! parallel-program nil))
  ([parallel-program caller-options] (physical-validation-for-request! parallel-program caller-options)))

(defn ^:no-doc operation-projection
  "Read one exact operation from an already checked enclosing program's projection map.
   This accessor is not an evidence validator. Its synchronous consumers must first check the
   exact program owner; a missing projection is an invariant failure, never permission to reuse
   another operation's facts or silently reconstruct a different boundary."
  [projections operation]
  (when-not (.containsKey ^java.util.Map projections operation)
    (throw (ex-info "validated program has no projection for this exact operation"
                    {:reason :emitted-parallel-program-operation-projection
                     :ir :emitted-parallel-program})))
  (.get ^java.util.Map projections operation))

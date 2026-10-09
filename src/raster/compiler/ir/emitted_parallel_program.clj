(ns raster.compiler.ir.emitted-parallel-program
  "Validation for an equation-first emitted ParallelProgram.

   The target backend may replace scheduled operations, but the retained typed algorithms and
   ordered program dataflow remain authoritative. Host-only scalar equations are the only
   equations with an empty emitted operation sequence."
  (:require [clojure.set :as set]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
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
                                                       equation algorithm)))
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

(defn- validate-host-prefix-slices!
  "Bind every narrowed emitted body to its exact enclosing host-scalar execution prefix.

   An EmittedParallelEquation can validate a self-contained numerical graph, but a graph-storage
   extent may also use a preceding host scalar.  This outer check prevents an isolated body from
   substituting a different (though locally valid) prefix: its host equations, terminal numerical
   equation, inferred inputs, outputs, and effects must be the exact slice of this program."
  [parallel-program candidates-for]
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
                                 :expected-inputs expected-inputs :actual-inputs (:inputs body)
                                 :ir :emitted-parallel-program})))))
          (recur host-prefix (next remaining)))))))

(defn- validate-program!
  [parallel-program candidates-fn caller-options]
  (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
  (when-not (contains? dialect-targets (:dialect parallel-program))
    (throw (ex-info "emitted parallel program requires a supported C-family dialect"
                    {:reason :emitted-parallel-program-dialect
                     :dialect (:dialect parallel-program)
                     :supported (set (keys dialect-targets))
                     :ir :emitted-parallel-program})))
  (let [candidate-cache (java.util.IdentityHashMap.)
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
  [parallel-program caller-options]
  (if (nil? caller-options)
    (validate! parallel-program)
    (validate! parallel-program caller-options))
  (let [equations
        (vec
         (mapcat
          (fn [equation]
            (cond
              (true? (get-in equation [:attributes :host-only])) []
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
    equations))

(defn retained-numerical-equations
  "Project retained numerical equations only after independent validation under caller intent."
  ([parallel-program] (numerical-equations-for-request parallel-program nil))
  ([parallel-program caller-options]
   (numerical-equations-for-request parallel-program caller-options)))

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

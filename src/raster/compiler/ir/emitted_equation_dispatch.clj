(ns raster.compiler.ir.emitted-equation-dispatch
  "A numerical-policy-checked choice among independently certified equation emissions.

   SegmentedWeightedReduction and single plain FP32 contractions share this boundary. Each candidate
   retains its own scheduled body and artifact projection; a shared ABI is only a binding
   contract, never a proof that different floating-point schedules are interchangeable."
  (:require [raster.compiler.ir.emitted-parallel-equation :as equation]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]))

(defrecord EmittedEquationDispatch [alternatives dispatch numerical-policy])

(defn emitted-equation-dispatch?
  [value]
  (and value
       (= "raster.compiler.ir.emitted_equation_dispatch.EmittedEquationDispatch"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :ir :emitted-equation-dispatch))))

(defn- numerical-mode
  [candidate report]
  (if-let [model (:numerical-model report)]
    (:mode model)
    (if (swr/plan? (:algorithm candidate))
    (get-in (last (get-in candidate [:body :equations])) [:operations 0 :numerics :mode])
    (let [certificate (get-in candidate [:graph :nodes 0 :operation :provenance
                                         :scheduled-operation])
          expected (case (get-in certificate [:legality :kind])
                     :ordered-portable-contraction :exact
                     :register-tiled-contraction :reassociated
                     nil)
          actual (get-in certificate [:numerics :mode])]
      (when-not (and expected (= expected actual))
        (fail! :equation-dispatch-numerics
               "contraction schedule did not retain its generated numerical mode"
               {:expected expected :actual actual}))
      actual))))

(defn- validate-model-pair!
  [mode report]
  (let [model (:numerical-model report)]
    (when-not (if model
                (and (= :approximate-model mode)
                     (= :approximate-model (:mode model))
                     (= :mixed-matrix-operational-model (:kind model))
                     (= 1 (:version model)))
                (not= :approximate-model mode))
      (fail! :equation-dispatch-numerics
             "approximate numerical modes require a paired reconstructed operational model"
             {:mode mode :model model}))))

(defn- validation-report
  "Verify every schedule independently and require explicit permission for its numerical mode.

   The reference/default retains exact evaluation order. Reassociation requires :permitted-modes;
   approximation additionally requires an exact operational model in the ordered :permitted-models
   vector. The runtime selector cannot enlarge permission, and no universal finite bound is implied."
  [value]
  (when-not (emitted-equation-dispatch? value)
    (fail! :equation-dispatch-type "expected an EmittedEquationDispatch"
           {:actual (type value)}))
  (let [{:keys [alternatives numerical-policy] selection :dispatch} value]
    (when-not (and (vector? alternatives) (seq alternatives)
                   (every? equation/emitted-equation? alternatives))
      (fail! :equation-dispatch-alternatives
             "equation dispatch requires certified emitted equations" {}))
    (let [candidate-reports (mapv equation/validate-with-result-contracts alternatives)
          _ (when-not (every? #(or (swr/plan? (:algorithm %))
                                  (soac/program-form? (:algorithm %))) alternatives)
              (fail! :equation-dispatch-algorithm
                     "equation dispatch requires a supported retained typed algorithm" {}))
          first-candidate (first alternatives)
          allowed (:permitted-modes numerical-policy)
          domains (mapv :complete-write-domains candidate-reports)
          _ (when-not (every? seq domains)
              (fail! :equation-dispatch-complete-write
                     "each candidate must independently prove its complete-write domain" {}))
          modes (mapv numerical-mode alternatives candidate-reports)
          selection (dispatch/validate! selection)]
      ;; Permission cannot compensate for a missing or incompatible reconstructed model.
      (doseq [[mode report] (map vector modes candidate-reports)]
        (validate-model-pair! mode report))
      (when-not (and (map? numerical-policy)
                     (set? allowed)
                     (contains? allowed :exact)
                     (every? #{:exact :reassociated :approximate-model} allowed)
                     (or (not (contains? allowed :approximate-model))
                         (and (vector? (:permitted-models numerical-policy))
                              (seq (:permitted-models numerical-policy)))))
        (fail! :equation-dispatch-numerical-policy
               "dispatch requires explicit numerical permissions and models for approximate candidates"
               {:policy numerical-policy}))
      ;; Generated kernels may contain NaN literals. Clojure structural equality cannot compare
      ;; independently decoded copies of those graphs, even when their floating-point bits match.
      (when-not (and (= (count alternatives) (count (:alternatives selection)))
                     (every? true?
                             (map semantic-fingerprint/equivalent?
                                  (map :graph alternatives) (:alternatives selection))))
        (fail! :equation-dispatch-executables
               "dispatch alternatives differ from independently certified emitted graphs" {}))
      (when-not (every? #(= (:algorithm first-candidate) (:algorithm %))
                        (rest alternatives))
        (fail! :equation-dispatch-semantics
               "dispatch alternatives must refine the same semantic algorithm" {}))
      (when-not (apply = (map #(select-keys (:graph %) [:inputs :outputs :scalars])
                              alternatives))
        (fail! :equation-dispatch-graph-boundary
               "dispatch alternatives must retain the same external graph values" {}))
      (when-not (apply = (map :physical-results candidate-reports))
        (fail! :equation-dispatch-storage
               "dispatch alternatives must retain the same physical result mapping" {}))
      (when-not (apply = domains)
        (fail! :equation-dispatch-complete-write
               "every dispatch alternative must prove the same complete-write domain" {}))
      (when-not (every? allowed modes)
        (fail! :equation-dispatch-numerics
               "candidate numerical mode is not authorized by the equation policy"
               {:modes modes :permitted-modes allowed}))
      ;; Ordered models avoid Clojure set equality collapsing floating-point bit distinctions.
      ;; These reports were independently reconstructed and matched to every actual artifact.
      (doseq [report candidate-reports
              :let [model (:numerical-model report)]
              :when model]
        (when-not (some #(semantic-fingerprint/equivalent? model %)
                        (:permitted-models numerical-policy))
          (fail! :equation-dispatch-numerics
                 "candidate operational numerical model is not explicitly permitted"
                 {:model model})))
      (let [default-graph (dispatch/default-alternative selection)
            default-index (first (keep-indexed
                                  (fn [index candidate]
                                    (when (semantic-fingerprint/equivalent?
                                           (:graph candidate) default-graph)
                                      index))
                                  alternatives))]
        (when-not (= :exact (nth modes default-index))
          (fail! :equation-dispatch-default-numerics
                 "the safe fallback must retain exact evaluation order"
                 {:default-strategy (:default-strategy selection)}))
        {:value value :complete-write-domains (first domains)
         :boundary (nth alternatives default-index)
         :physical-results (:physical-results (first candidate-reports))
         :candidates alternatives}))))

(defn ^:no-doc validate-with-boundary
  "Independently check all candidates and return their common storage and exact fallback.
   This report grants no reusable authority; the enclosing program validator seals its own
   exact-owner evidence. Runtime admission still chooses the executable independently."
  [value]
  (select-keys (validation-report value) [:boundary :physical-results :candidates]))

(defn validate!
  "Validate every candidate, its numerical permission, and the common full-write domain."
  [value]
  (:value (validation-report value)))

(defn make
  [alternatives selection numerical-policy]
  (validate! (->EmittedEquationDispatch alternatives selection numerical-policy)))

(defn candidates
  "Return each independently certified equation in its dispatch alternative order."
  [value]
  (:alternatives (validate! value)))

(defn complete-write-domains
  "Return only coverage proved for every independently certified alternative."
  [value]
  (:complete-write-domains (validation-report value)))

(defn default-equation
  "Return the exact fallback equation for boundary inspection, not runtime execution."
  [value]
  (:boundary (validation-report value)))

(defn boundary-equation
  "Return the exact equation for shared storage/ABI inspection.

   For a dispatch, all candidates have already proved the same external boundary. This accessor
   never selects a runtime schedule; binding-time admission still owns that choice. A single
   equation is returned unmodified: conservative coverage probes may inspect a modified graph,
   while whole-program and call validation certify executable equations separately."
  [operation]
  (cond
    (emitted-equation-dispatch? operation) (default-equation operation)
    :else operation))

(defn boundary-graph
  [operation]
  (:graph (boundary-equation operation)))

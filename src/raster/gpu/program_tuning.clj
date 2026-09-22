(ns raster.gpu.program-tuning
  "Pure program-level planning for explicit KernelDispatch autotuning.

   A manifest finds tunable dispatch sites in a compiled resident descriptor and groups sites only
   when their stable dispatch identity, emitted alternatives, numerical/layout contract, and
   schedule target agree. A tuning plan then selects a deterministic, explicitly bounded subset.
   Neither operation owns a device or performs measurement."
  (:require [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.gpu.dispatch-tuning :as tuning]))

(def manifest-version 1)
(def plan-version 1)
(def receipt-version 1)

(defn- schedule-target
  [dispatch]
  (let [path (get-in dispatch [:attributes :tuning :schedule-path])
        key (get-in dispatch [:attributes :tuning :schedule-key])]
    (when-not (and (vector? path) (seq path) (every? keyword? path))
      (throw (ex-info "tunable dispatch does not declare a tuning schedule path"
                      {:reason :invalid-tuning-schedule-path
                       :dispatch-id (:id dispatch) :schedule-path path})))
    (when-not (or (nil? key)
                  (and (string? key) (not-empty key))
                  (keyword? key))
      (throw (ex-info "tunable dispatch declares an invalid tuning schedule key"
                      {:reason :invalid-tuning-schedule-key
                       :dispatch-id (:id dispatch) :schedule-key key})))
    {:path path :key key}))

(defn dispatch-signature
  "Stable program-group identity for one tunable dispatch, excluding its current selector policy."
  [dispatch]
  (let [dispatch (kdispatch/validate! dispatch)
        contract (get-in dispatch [:attributes :tuning])]
    (when-not (map? contract)
      (throw (ex-info "tunable dispatch contract must be a map"
                      {:reason :invalid-dispatch-tuning-contract
                       :dispatch-id (:id dispatch) :contract contract})))
    (when (nil? (:numerical-mode contract))
      (throw (ex-info "tunable dispatch contract requires a numerical mode"
                      {:reason :missing-dispatch-numerical-mode :dispatch-id (:id dispatch)})))
    (when (nil? (:layout contract))
      (throw (ex-info "tunable dispatch contract requires a layout"
                      {:reason :missing-dispatch-layout :dispatch-id (:id dispatch)})))
    {:dispatch-id (:id dispatch)
     :default-strategy (:default-strategy dispatch)
     :selector-argument (get-in dispatch [:selector :argument])
     :schedule-target (schedule-target dispatch)
     :numerical-mode (:numerical-mode contract)
     :layout (:layout contract)
     :alternatives (mapv tuning/executable-signature (:alternatives dispatch))}))

(defn- tunable-step?
  [step]
  (and (some? (:dispatch step))
       (some? (get-in step [:dispatch :attributes :tuning]))))

(defn manifest
  "Describe and deduplicate the tunable KernelDispatch sites in a compiled program descriptor.

   Site and group ordering follows descriptor step order. The first equivalent site is the group's
   benchmark representative. Reusing a dispatch ID for a different emitted or schedule identity is
   rejected instead of allowing an unsafe tuning result to leak across operations."
  [descriptor]
  (when-not (map? descriptor)
    (throw (ex-info "program tuning manifest requires a compiled descriptor"
                    {:reason :invalid-program-descriptor :descriptor descriptor})))
  (when-not (vector? (:steps descriptor))
    (throw (ex-info "program tuning manifest requires a descriptor :steps vector"
                    {:reason :invalid-program-steps :steps (:steps descriptor)})))
  (let [sites
        (into []
              (keep (fn [[step-index step]]
                      (when (tunable-step? step)
                        (let [dispatch (kdispatch/validate! (:dispatch step))
                              signature (dispatch-signature dispatch)]
                          {:step-index step-index
                           :phase (:phase step)
                           :dispatch-id (:id dispatch)
                           :group-id (:id dispatch)
                           :signature signature}))))
              (map-indexed vector (:steps descriptor)))
        {:keys [order groups]}
        (reduce
         (fn [{:keys [order groups] :as state} site]
           (let [group-id (:group-id site)
                 prior (get groups group-id)]
             (if prior
               (do
                 (when-not (= (:signature prior) (:signature site))
                   (throw (ex-info "program reuses a dispatch ID for incompatible tuning sites"
                                   {:reason :program-dispatch-identity-collision
                                    :dispatch-id group-id
                                    :first-step (:representative-step-index prior)
                                    :different-step (:step-index site)
                                    :first-signature (:signature prior)
                                    :different-signature (:signature site)})))
                 (assoc state :groups
                        (assoc groups group-id
                               (-> prior
                                   (update :step-indices conj (:step-index site))
                                   (update :phases conj (:phase site))
                                   (update :site-count inc)))))
               {:order (conj order group-id)
                :groups (assoc groups group-id
                               {:id group-id
                                :dispatch-id group-id
                                :representative-step-index (:step-index site)
                                :step-indices [(:step-index site)]
                                :phases [(:phase site)]
                                :site-count 1
                                :signature (:signature site)})})))
         {:order [] :groups {}}
         sites)
        ordered-groups (mapv groups order)]
    {:version manifest-version
     :site-count (count sites)
     :group-count (count ordered-groups)
     :sites sites
     :groups ordered-groups}))

(defn validate-plan!
  [plan]
  (when-not (and (map? plan)
                 (= plan-version (:version plan))
                 (vector? (:selected-groups plan))
                 (vector? (:deferred-groups plan)))
    (throw (ex-info "invalid program dispatch tuning plan"
                    {:reason :invalid-program-tuning-plan :plan plan})))
  plan)

(defn tuning-plan
  "Select a deterministic subset of manifest groups for a later explicit benchmark action.

   Options:
     :group-ids  collection of stable dispatch/group IDs (default: every group)
     :max-groups non-negative upper bound after filtering (default: every selected group)

   Unknown group IDs and invalid budgets fail loudly. A zero budget is a valid dry plan."
  ([manifest] (tuning-plan manifest {}))
  ([manifest {:keys [group-ids max-groups] :as options}]
   (when-not (and (map? manifest)
                  (= manifest-version (:version manifest))
                  (vector? (:groups manifest)))
     (throw (ex-info "program tuning plan requires a valid manifest"
                     {:reason :invalid-program-tuning-manifest :manifest manifest})))
   (when-not (or (nil? max-groups)
                 (and (integer? max-groups) (not (neg? (long max-groups)))))
     (throw (ex-info "program tuning :max-groups must be a non-negative integer"
                     {:reason :invalid-program-tuning-budget :max-groups max-groups})))
   (let [known (set (map :id (:groups manifest)))
         requested (when (some? group-ids) (set group-ids))
         unknown (when requested (seq (remove known requested)))]
     (when unknown
       (throw (ex-info "program tuning plan names unknown dispatch groups"
                       {:reason :unknown-program-tuning-groups
                        :unknown (set unknown) :known known})))
     (let [eligible (if requested
                      (filterv #(contains? requested (:id %)) (:groups manifest))
                      (:groups manifest))
           limit (long (or max-groups (count eligible)))
           selected (vec (take limit eligible))
           deferred (vec (drop limit eligible))]
       (validate-plan!
        {:version plan-version
         :manifest-version (:version manifest)
         :manifest-group-count (:group-count manifest)
         :budget {:max-groups max-groups
                  :eligible-groups (count eligible)
                  :selected-groups (count selected)}
         :options (select-keys options [:group-ids :max-groups])
         :selected-groups selected
         :deferred-groups deferred})))))

(defn- merge-value
  [left right path]
  (cond
    (and (map? left) (map? right))
    (if (or (contains? left :kind) (contains? right :kind))
      (if (= left right)
        left
        (throw (ex-info "program tuning results conflict at one schedule target"
                        {:reason :program-tuning-schedule-conflict
                         :path path :left left :right right})))
      (reduce-kv (fn [result key value]
                   (if (contains? result key)
                     (assoc result key (merge-value (get result key) value (conj path key)))
                     (assoc result key value)))
                 left right))

    (= left right) left

    :else
    (throw (ex-info "program tuning results conflict at one schedule target"
                    {:reason :program-tuning-schedule-conflict
                     :path path :left left :right right}))))

(defn merge-schedule-overrides
  "Merge per-dispatch schedule fragments, rejecting different values at the same leaf."
  [overrides]
  (reduce (fn [result override]
            (when-not (map? override)
              (throw (ex-info "program tuning schedule override must be a map"
                              {:reason :invalid-program-tuning-override :override override})))
            (merge-value result override []))
          {}
          overrides))

(defn- selector-fragment
  [signature selector]
  (let [{:keys [path key]} (:schedule-target signature)
        target-path (cond-> path (some? key) (conj key))]
    (assoc-in {} target-path selector)))

(defn- receipt-selector
  [group tuning schedule-override]
  (let [measured (:selector tuning)
        target (:schedule-target (:signature group))
        path (cond-> (:path target) (some? (:key target)) (conj (:key target)))
        supplied (get-in schedule-override path)
        pinned (when (= :fixed-strategy (:kind measured))
                 (assoc measured :fallback :none))]
    (cond
      (= supplied measured) {:selector measured :pinned? false}
      (and pinned (= supplied pinned)) {:selector pinned :pinned? true}
      :else
      (throw (ex-info "program tuning receipt selector differs from its measured evidence"
                      {:reason :program-tuning-receipt-selector
                       :group-id (:id group) :path path
                       :measured measured :supplied supplied})))))

(defn make-receipt
  "Seal program tuning results and their schedule override into a plain, versioned receipt.

   A receipt may cover the bounded subset selected by `tuning-plan`. It retains each exact
   dispatch signature and complete DispatchTuning evidence; the schedule fragment alone is never
   accepted as proof. Pinned fixed selectors are recorded explicitly."
  [plan results schedule-override]
  (let [plan (validate-plan! plan)]
    (when-not (vector? results)
      (throw (ex-info "program tuning receipt requires an ordered result vector"
                      {:reason :invalid-program-tuning-receipt-results :results results})))
    (let [groups (:selected-groups plan)
          expected-ids (mapv :id groups)
          actual-ids (mapv :group-id results)]
      (when-not (= expected-ids actual-ids)
        (throw (ex-info "program tuning receipt results differ from the selected plan groups"
                        {:reason :program-tuning-receipt-groups
                         :expected expected-ids :actual actual-ids})))
      (let [entries
            (mapv
             (fn [group result]
               (let [evidence (:tuning result)]
                 (when-not (tuning/dispatch-tuning? evidence)
                   (throw (ex-info "program tuning receipt requires DispatchTuning evidence"
                                   {:reason :invalid-program-tuning-receipt-evidence
                                    :group-id (:id group) :evidence evidence})))
                 (let [identity (:identity evidence)
                       signature (:signature group)]
                   (when-not (and (= (:id group) (:dispatch-id identity))
                                  (= (:alternatives signature) (:alternatives identity))
                                  (= (:selector-argument signature)
                                     (:selector-argument identity))
                                  (= (:numerical-mode signature) (:numerical-mode identity))
                                  (= (:layout signature) (:layout identity)))
                     (throw (ex-info "program tuning evidence differs from its manifest group"
                                     {:reason :program-tuning-receipt-identity
                                      :group-id (:id group)
                                      :signature signature :identity identity})))
                   (let [{:keys [selector pinned?]}
                         (receipt-selector group evidence schedule-override)]
                     {:group-id (:id group)
                      :signature signature
                      :selector selector
                      :pinned? pinned?
                      :tuning (tuning/tuning-data evidence)}))))
             groups results)
            reconstructed (merge-schedule-overrides
                           (mapv #(selector-fragment (:signature %) (:selector %)) entries))]
        (when-not (= reconstructed schedule-override)
          (throw (ex-info "program tuning receipt contains schedule data without tuning evidence"
                          {:reason :program-tuning-receipt-override
                           :expected reconstructed :actual schedule-override})))
        {:version receipt-version
         :manifest-version manifest-version
         :plan-version plan-version
         :groups entries
         :schedule-override schedule-override}))))

(defn replay-receipt
  "Validate a tuning receipt against a freshly compiled descriptor and hardware descriptor.

   Returns restored DispatchTunings and the exact schedule override for immutable recompilation.
   Any program, emitted source, ABI, numerical/layout contract, device, driver, capability or
   calibration drift fails loudly; callers choose whether that deployment error triggers a new
   explicit tuning action."
  [descriptor hardware-descriptor receipt]
  (when-not (and (map? receipt)
                 (= receipt-version (:version receipt))
                 (= manifest-version (:manifest-version receipt))
                 (= plan-version (:plan-version receipt))
                 (vector? (:groups receipt))
                 (map? (:schedule-override receipt)))
    (throw (ex-info "invalid program tuning receipt"
                    {:reason :invalid-program-tuning-receipt :receipt receipt})))
  (let [current (manifest descriptor)
        by-id (into {} (map (juxt :id identity)) (:groups current))
        receipt-group-ids (mapv :group-id (:groups receipt))
        _ (when-not (= (count receipt-group-ids) (count (distinct receipt-group-ids)))
            (throw (ex-info "program tuning receipt repeats a dispatch group"
                            {:reason :program-tuning-receipt-duplicate-group
                             :group-ids receipt-group-ids})))
        restored
        (mapv
         (fn [{:keys [group-id signature selector pinned? tuning] :as entry}]
           (let [group (get by-id group-id)]
             (when-not group
               (throw (ex-info "program tuning receipt names a missing dispatch group"
                               {:reason :program-tuning-receipt-missing-group
                                :group-id group-id :known (set (keys by-id))})))
             (when-not (= signature (:signature group))
               (throw (ex-info "program tuning receipt manifest signature changed"
                               {:reason :program-tuning-receipt-signature
                                :group-id group-id
                                :expected (:signature group) :actual signature})))
             (let [step (:representative-step-index group)
                   dispatch (get-in descriptor [:steps step :dispatch])
                   contract (get-in dispatch [:attributes :tuning])
                   evidence (tuning/restore-tuning dispatch tuning hardware-descriptor
                                                    (:numerical-mode contract)
                                                    (:layout contract))]
               (when-not (instance? Boolean pinned?)
                 (throw (ex-info "program tuning receipt :pinned? must be boolean"
                                 {:reason :invalid-program-tuning-receipt-pin
                                  :group-id group-id :pinned? pinned?})))
               (let [expected (if pinned?
                                (assoc (:selector evidence) :fallback :none)
                                (:selector evidence))]
                 (when pinned?
                   (kdispatch/specialize-fixed dispatch expected))
                 (when-not (= expected selector)
                   (throw (ex-info "program tuning receipt selector differs from restored evidence"
                                   {:reason :program-tuning-receipt-restored-selector
                                    :group-id group-id :expected expected :actual selector})))
                 {:group-id group-id :tuning evidence :selector selector
                  :schedule-override (selector-fragment signature selector)
                  :receipt-entry entry}))))
         (:groups receipt))
        schedule-override (merge-schedule-overrides (mapv :schedule-override restored))]
    (when-not (= schedule-override (:schedule-override receipt))
      (throw (ex-info "program tuning receipt schedule override is not evidence-complete"
                      {:reason :program-tuning-receipt-override
                       :expected schedule-override :actual (:schedule-override receipt)})))
    {:receipt receipt
     :tunings restored
     :schedule-override schedule-override}))

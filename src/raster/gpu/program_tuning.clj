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
     :artifacts (mapv tuning/artifact-signature (:alternatives dispatch))}))

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

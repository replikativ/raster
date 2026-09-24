(ns raster.compiler.analysis.physical-liveness
  "Observation-only local storage liveness over validated LinkPlan memory facts.

   This is a shadow analysis, not an allocator. Ordered ABI accesses are submission facts;
   cross-queue completion, external escapes, and actual alias realization remain separate proofs."
  (:require [raster.compiler.ir.link-plan :as link]))

(defn- allocation-nodes [memory]
  (reduce-kv (fn [groups node-id {:keys [allocation]}]
               (update groups allocation (fnil conj []) node-id))
             {} (:nodes memory)))

(defn- allocation-accesses [memory]
  (reduce (fn [groups {:keys [node] :as access}]
            (update groups (get-in memory [:nodes node :allocation]) (fnil conj []) access))
          {} (:accesses memory)))

(defn- slot
  [memory nodes accesses allocation-id]
  (let [allocation (get-in memory [:allocations allocation-id])
        node-ids (get nodes allocation-id)
        node-facts (mapv #(get-in memory [:nodes %]) node-ids)
        events (get accesses allocation-id)
        first-order (some-> events first :order)
        last-order (some-> events last :order)
        reason (cond
                 (not= :owned (:ownership allocation)) :caller-owned
                 (some :output? node-facts) :public-output
                 (some :host-initializer? node-facts) :host-initialized
                 (some #(not (contains? #{:internal :scratch} (:role %))) node-facts)
                 :persistent-role
                 (not= 1 (count node-facts)) :multi-view
                 (empty? events) :unused
                 (not= :write (:access (first events))) :needs-initial-value
                 :else :shadow-candidate)]
    {:allocation allocation-id
     :nodes node-ids
     :byte-size (:byte-size allocation)
     :device (:device allocation)
     :memory-space (:memory-space allocation)
     :alignment (:alignment allocation)
     :dtype (when (= 1 (count node-facts)) (:dtype (first node-facts)))
     :first-order first-order :last-order last-order
     :reason reason}))

(defn- compatible? [left right]
  (= (select-keys left [:byte-size :device :memory-space :alignment :dtype])
     (select-keys right [:byte-size :device :memory-space :alignment :dtype])))

(defn report
  "Find ordered, compatible local storage pairs without changing the LinkPlan.

   A proposal is *not* a reuse proof: it omits runtime event completion, cross-plan escapes,
   and the alias/rebinding validation needed before allocation. No aggregate byte saving is
   claimed, since proposals can compete for the same allocation."
  [plan]
  (let [memory (link/memory-report plan)
        nodes (allocation-nodes memory)
        accesses (allocation-accesses memory)
        slots (into {}
                    (map (fn [allocation-id]
                           [allocation-id (slot memory nodes accesses allocation-id)]))
                    (keys (:allocations memory)))
        accessed-order (reduce (fn [ids {:keys [node]}]
                                 (let [id (get-in memory [:nodes node :allocation])]
                                   (if (some #{id} ids) ids (conj ids id))))
                               [] (:accesses memory))
        candidates (filterv #(= :shadow-candidate (:reason (get slots %))) accessed-order)
        proposals (vec
                   (for [left-id candidates
                         right-id candidates
                         :let [left (get slots left-id)
                               right (get slots right-id)]
                         :when (and (< (:last-order left) (:first-order right))
                                    (compatible? left right))]
                     {:from left-id :to right-id
                      :after-order (:last-order left)
                      :before-order (:first-order right)
                      :bytes (:byte-size right)
                      :status :shadow
                      :pending #{:runtime-order :alias-realization :completion-and-escape}}))]
    {:plan (:plan memory)
     :slots slots
     :proposals proposals
     :current-owned-bytes
     (reduce + 0 (for [[_ allocation] (:allocations memory)
                       :when (= :owned (:ownership allocation))]
                   (:byte-size allocation)))
     :reuse :unproven}))

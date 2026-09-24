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
     :first-complete-write? (boolean (and (= :write (:access (first events)))
                                          (:complete-write? (first events))))
     :reason reason}))

(defn- compatible? [left right]
  (= (select-keys left [:byte-size :device :memory-space :alignment :dtype])
     (select-keys right [:byte-size :device :memory-space :alignment :dtype])))

(defn- source-step [access]
  {:instance (:instance access) :step (:step access)})

(defn- runtime-order-assessment
  "Assess only selected kernel order. A source step split across prologue/replay is deliberately
   rejected: ABI step accesses cannot be attributed to one child kernel without a finer witness."
  [accesses left-id right-id execution-order]
  (when-not (every? #(and (contains? % :source)
                         (contains? (:source %) :instance)
                         (contains? (:source %) :step))
                    (concat (:record-time-prologue execution-order)
                            (:per-replay execution-order)))
    (throw (ex-info "selected graph order lacks linked source-step identities"
                    {:reason :memory-order-source-missing})))
  (let [relevant (filterv (fn [access]
                            (contains? #{left-id right-id} (:allocation access)))
                          accesses)
        prologue (set (map :source (:record-time-prologue execution-order)))
        replay-positions (reduce-kv (fn [positions index {:keys [source]}]
                                      (update positions source (fnil conj []) index))
                                    {} (vec (:per-replay execution-order)))
        relevant-steps (set (map source-step relevant))]
    (cond
      (some prologue relevant-steps)
      {:status :declined :reason :record-time-prologue}

      (some #(not (contains? replay-positions %)) relevant-steps)
      {:status :declined :reason :unmatched-source-step}

      :else
      (let [left-steps (set (map source-step
                                 (filter #(= left-id (:allocation %)) relevant)))
            right-steps (set (map source-step
                                  (filter #(= right-id (:allocation %)) relevant)))
            left-end (apply max (mapcat replay-positions left-steps))
            right-start (apply min (mapcat replay-positions right-steps))]
        (if (< left-end right-start)
          {:status :witnessed :left-end left-end :right-start right-start}
          {:status :declined :reason :selected-order-overlap
           :left-end left-end :right-start right-start})))))

(defn report
  "Find ordered, compatible local storage pairs without changing the LinkPlan.

   An optional linked execution-order witness can discharge selected-order and certified
   cross-replay overwrite obligations. A proposal is *not* a reuse proof: it still omits runtime
   event completion, cross-plan escapes, and alias/rebinding validation before allocation.
   No aggregate byte saving is claimed, since proposals can compete for the same allocation."
  ([plan] (report plan nil))
  ([plan execution-order]
  (let [memory (link/memory-report plan)
        _ (when (and execution-order
                     (not= [(:plan memory) (:target memory)]
                           [(:plan execution-order) (:target execution-order)]))
            (throw (ex-info "selected graph order belongs to a different LinkPlan"
                            {:reason :memory-order-plan-mismatch
                             :plan (:plan memory) :target (:target memory)
                             :witness-plan (:plan execution-order)
                             :witness-target (:target execution-order)})))
        nodes (allocation-nodes memory)
        accesses (allocation-accesses memory)
        ordered-accesses (mapv (fn [access]
                                 (assoc access :allocation
                                        (get-in memory [:nodes (:node access) :allocation])))
                               (:accesses memory))
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
                     (let [assessment (when execution-order
                                        (runtime-order-assessment ordered-accesses
                                                                  left-id right-id
                                                                  execution-order))]
                       (let [replay-initialized? (and (= :witnessed (:status assessment))
                                                      (:first-complete-write? left)
                                                      (:first-complete-write? right))]
                       (cond-> {:from left-id :to right-id
                                :after-order (:last-order left)
                                :before-order (:first-order right)
                                :bytes (:byte-size right)
                                :status :shadow
                                :pending #{:runtime-order :alias-realization
                                           :completion-and-escape}}
                         execution-order
                         (assoc :status (if (= :declined (:status assessment))
                                          :declined :shadow)
                                :runtime-order assessment
                                :pending (cond-> #{:runtime-order :alias-realization
                                                   :completion-and-escape
                                                   :cross-replay-initialization}
                                           (= :witnessed (:status assessment))
                                           (disj :runtime-order)
                                           replay-initialized?
                                           (disj :cross-replay-initialization)))
                         execution-order
                         (assoc :cross-replay-initialization
                                (if replay-initialized? :witnessed :unproven)))))))]
    {:plan (:plan memory)
     :slots slots
     :proposals proposals
     :current-owned-bytes
     (reduce + 0 (for [[_ allocation] (:allocations memory)
                       :when (= :owned (:ownership allocation))]
                   (:byte-size allocation)))
     :reuse :unproven})))

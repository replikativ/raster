(ns raster.compiler.fixtures.distributed-capture
  "Ordinary generated producers with exact Prepared identity, not AMR or a trainer."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as numeric]
            [raster.par :as par]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.link-plan :as link]
            [raster.gpu.compiled :as compiled]))

(deftm twice!
  [out :- (Array float) x :- (Array float) n :- Long] :- (Array float)
  (par/map-void! i n (arrays/aset out i (numeric/* (arrays/aget x i) (float 2.0))))
  out)

(deftm increment!
  [out :- (Array float) x :- (Array float) n :- Long] :- (Array float)
  (par/map-void! i n (arrays/aset out i (numeric/+ (arrays/aget x i) (float 1.0))))
  out)

(defn fixture [target inputs]
  (let [prepared
        (into {}
              (map (fn [[id function]]
                     (let [input (float-array (get inputs id))
                           source (compiled/lower function [(float-array (alength input)) input (alength input)]
                                                  {:target target :compiler :equation-first :dtype :float
                                                   :outputs '[out]})
                           key (:key (first (:out-tree source)))]
                       [id (compiled/compose {:id id :components [{:id id :program source}]
                                              :outputs [{:key id :from [id key]}]})]))
                   [[:twice #'twice!] [:increment #'increment!]]))
        locals (update-vals prepared compiled/plan)
        boundaries
        (update-vals locals
                     (fn [local]
                       (let [outputs (set (link/output-value-ids local))
                             accesses (link/value-accesses local)]
                         (into outputs
                               (keep (fn [[id value]]
                                       (let [node (get-in local [:nodes (:node (first (:leaves value)))])]
                                         (when (and (contains? accesses id)
                                                    (or (contains? #{:input :state} (:role node))
                                                        (and (= :constant (:role node)) (nil? (:source node))))) id))))
                               (:values local)))))
        globals (into {} (for [[entry ids] boundaries id ids]
                           [[entry id] (assoc (get-in locals [entry :values id :abstract])
                                             :sharding {:kind :replicated :devices [:worker]})]))
        plan (distributed/plan
              {:id :two-entry-capture
               :mesh (distributed/mesh [{:name :workers :size 1}] [:worker])
               :topology (distributed/topology [(distributed/device {:id :worker
                                                                      :memory-capacity-bytes 1048576})] [])
               :values globals
               :shards (into {} (map (fn [[id value]]
                                      [id [(distributed/shard {:id id :value id :device :worker
                                                              :shape (:shape value)
                                                              :offsets (vec (repeat (count (:shape value)) 0))
                                                              :ownership :replica})]])) globals)
               :device-plans
               {:worker {:target target
                         :entries (update-vals locals #(hash-map :link-plan %))
                         :steps (into {} (for [[entry ids] boundaries]
                                           [entry {:entry entry
                                                   :bindings (into {} (for [id ids]
                                                                        [id {:value [entry id] :shard [entry id]}]))}]))}}
               :steps (mapv #(distributed/compute-step {:id % :device :worker :duration-ns 1})
                            [:twice :increment])
               :outputs [:twice :increment]})]
    {:plan plan :prepared-by-entry (into {} (map (fn [[id p]] [[:worker id] p])) prepared)
     :options {:device-capacities {target 1048576}}
     :capture-options
     {:id :state-1 :parents [] :logical-coordinate {:step 1}
      :numerical-contract {:mode :fp32 :determinism :bitwise :compatibility-id "two-entry-map-v1"}
      :fields (mapv (fn [id] {:id id :step id :key id
                              :value (av/tensor {:dtype :float :shape [(count (get inputs id))]})})
                    [:twice :increment])}
     :expected {:twice (mapv #(float (* (float %) (float 2.0))) (:twice inputs))
                :increment (mapv #(float (+ (float %) (float 1.0))) (:increment inputs))}}))

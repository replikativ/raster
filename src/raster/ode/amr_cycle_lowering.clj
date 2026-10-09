(ns raster.ode.amr-cycle-lowering
  "Allocation-free numerical provider for one complete ratio-2 diffusion cycle.
   This produces an ordinary certified local composition, not an AMR temporal certificate,
   timestep/synchronization proof, runtime owner or new numerical implementation."
  (:require [raster.arrays :as arrays]
            [raster.compiler.ir.link-composition :as composition]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.validate :refer [fail!]]
            [raster.gpu.compiled :as compiled]
            [raster.ode.amr-subcycle :as subcycle]))

(def ^:private geometry-parameters
  '[cl cr cc co ci cs cv fl fr fc fo fi fs fv
    donors ghosts fine-indices owners interface covered aggregate])

(def ^:private scratch-parameters
  '[coarse-out fine-out coarse-flux predicted old-ghost predicted-ghost
    old-full predicted-full current-0 current-1 boundary-0 boundary-1
    flux-0 flux-1 fine-1 aggregate-0 aggregate-1 register averaged])

(defn- port
  [prepared direction key]
  (let [matches (filter #(= key (:key %)) (get prepared direction))]
    (when-not (= 1 (count matches))
      (fail! "cycle provider requires one exact semantic port"
             :amr-cycle-port {:direction direction :key key}))
    (let [node (:node (first matches))
          owners (for [[id value] (:values (compiled/plan prepared))
                       :when (some #(= node (:node %)) (:leaves value))] id)]
      (when-not (= 1 (count owners))
        (fail! "cycle semantic port requires one logical owner"
               :amr-cycle-port {:direction direction :key key}))
      (first owners))))

(defn lower
  "Lower the existing whole numerical cycle and commit its outputs to coarse/fine state.
   No device allocation/session is opened. The two final state ports borrow the caller's
   initialized arrays; topology and scratch remain private to the local program. Caller bytes
   must initially be synchronized, dt must be stable, and sources must remain valid/stable
   until initialization. A geometry fingerprint describes preparation, not mathematical proof."
  [projection coarse fine dt {:keys [id target] :or {id :amr/cycle} :as options}]
  (when-not (and (map? options) (some? id) (some? target)
                 (every? #{:id :target} (keys options)))
    (fail! "cycle lowering requires an explicit target and closed options"
           :amr-cycle-options {}))
  (let [inputs (subcycle/cycle-inputs projection coarse fine dt)
        common {:target target :compiler :equation-first :dtype :double :inline? true}
        cycle (compiled/lower #'subcycle/diffusion-cycle! (:arguments inputs)
                              (assoc common :constants geometry-parameters
                                     :roles (zipmap scratch-parameters (repeat :scratch))
                                     :outputs '[coarse-out fine-out]))
        commits (into {}
                      (for [[field state] [[:coarse coarse] [:fine fine]]]
                        [field (compiled/lower #'arrays/acopy!
                                               [(get-in inputs [:outputs field]) 0 state 0 (alength ^doubles state)]
                                               (assoc common :roles {'dst :state} :outputs '[dst]))]))
        reference (fn [component prepared direction key]
                    [component (port prepared direction key)])
        owners (into {} (for [[field prepared] commits]
                          [field (reference [field :commit] prepared :in-tree :dst)]))
        local
        (composition/compose
         {:id id
          :components (into [{:id :cycle :lowering (:lowering cycle)}]
                            (map (fn [field]
                                   {:id [field :commit] :lowering (:lowering (get commits field))})
                                 [:coarse :fine]))
          :connections (mapv (fn [field]
                               {:from (reference :cycle cycle :out-tree
                                                 (keyword (str (name field) "-out")))
                                :to (reference [field :commit] (get commits field) :in-tree :src)})
                             [:coarse :fine])
          :mutable-shares (mapv (fn [field]
                                  {:owner (get owners field)
                                   :borrowers [(reference :cycle cycle :in-tree field)]
                                   :output (get owners field)}) [:coarse :fine])
          :outputs (mapv owners [:coarse :fine])})]
    {:lowering local
     :fields (update-vals owners #(get-in local [:certificate :value-mapping %]))
     :dt (double dt) :geometry-fingerprint (fingerprint/fingerprint projection)}))

(ns raster.ode.amr-cycle-lowering
  "Device-allocation-free preparation for one complete ratio-2 diffusion cycle.
   This produces an ordinary certified local composition, not an AMR temporal certificate,
   timestep/synchronization proof, runtime owner or new numerical implementation."
  (:require [raster.arrays :as arrays]
            [raster.compiler.ir.link-plan :as plan]
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

(defn lower
  "Lower the existing whole numerical cycle and commit its outputs to coarse/fine state.
   No device allocation/session is opened. The two final state owners initialize from the
   caller's distinct arrays; topology and scratch remain private to the local program.
   Caller bytes must initially be synchronized, dt must be stable, and sources must remain valid/stable
   until initialization. A geometry fingerprint describes preparation, not mathematical proof."
  [projection coarse fine dt {:keys [id target] :or {id :amr/cycle} :as options}]
  (when-not (and (map? options) (some? id) (some? target)
                 (every? #{:id :target} (keys options)))
    (fail! "cycle lowering requires an explicit target and closed options"
           :amr-cycle-options {}))
  (when (identical? coarse fine)
    (fail! "coarse and fine levels require distinct state sources"
           :amr-cycle-state-alias {}))
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
                                               (assoc common :donate '[dst]))]))
        local
        (compiled/compose
         {:id id
          :components (into [{:id :cycle :program cycle}]
                            (map (fn [field]
                                   {:id [field :commit] :program (get commits field)})
                                 [:coarse :fine]))
          :connections (mapv (fn [field]
                               {:from [:cycle (keyword (str (name field) "-out"))]
                                :to [[field :commit] :src]})
                             [:coarse :fine])
          :mutable-shares (mapv (fn [field]
                                  {:owner [[field :commit] :dst]
                                   :borrowers [[:cycle field]]
                                   :output [[field :commit] :dst']}) [:coarse :fine])
          :outputs (mapv (fn [field] {:key field :from [[field :commit] :dst']})
                         [:coarse :fine])})]
    {:prepared local
     :lowering (:lowering local)
     :fields (zipmap (map :key (:out-tree local))
                     (plan/output-value-ids (compiled/plan local)))
     :dt (double dt) :geometry-fingerprint (fingerprint/fingerprint projection)}))

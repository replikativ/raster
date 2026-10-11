(ns ^:no-doc raster.gpu.compatibility-map
  "Pure empty-extent admission for direct compatibility map APIs, not generic launches."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.segop :as segop]))

(defn empty-map?
  "Check artifact scalar/precondition contracts before admitting an empty map. Plain entries
   retain the direct API's map-bound convention. An artifact needs its checked nonreducing
   SegMap projection and exact extent binding; a zero scalar alone never skips an opaque kernel."
  [registered arguments bound]
  (when (and (zero? bound) (artifact/kernel-artifact? registered))
    (call/validate-preconditions! registered arguments))
  (and (zero? bound)
       (or (not (artifact/kernel-artifact? registered))
           (when-let [operation (get-in registered [:attributes :scheduled-kernel-body])]
             (scheduled/validate-artifact-projection! operation registered)
             (let [source (:source operation)
                   bound-pairs (filter (fn [[slot _]] (= :bound (:role slot)))
                                       (map vector (:abi registered) (:arguments registered)))]
               (and (segop/seg-map? source)
                    (contains? #{:elementwise-map :side-effect-map} (get-in operation [:effects :kind]))
                    (= 1 (count bound-pairs))
                    (= (:bound (segop/seg-space-reduced-dim (:space source)))
                       (second (first bound-pairs)))))))))

(defn realize-launch
  "Pure direct-map geometry: preserve canonical launch authority and only elide a proven empty
   map. Plain compatibility entries retain exact 1-D ceil division and checked overrides."
  [registered arguments bound opts]
  (let [default-workgroup
        (if-let [spec (:launch registered)]
          (let [workgroup (launch/static-workgroup-size spec)]
            (when-not (= 1 (count workgroup))
              (throw (ex-info "1-D kernel path received a multidimensional launch contract"
                              {:kernel-name (:kernel-name registered) :launch spec})))
            (first workgroup))
          (first (:workgroup-size
                  (launch/geometry {:workgroup-size [(or (:workgroup-size registered) 256)]
                                    :group-count [1]}))))
        workgroup (first (:workgroup-size
                          (launch/geometry
                           {:workgroup-size [(get opts :workgroup-size default-workgroup)]
                            :group-count [1]})))
        _ (when (and (artifact/kernel-artifact? registered) (not= default-workgroup workgroup))
            (throw (ex-info "direct map override differs from the emitted workgroup"
                            {:reason :kernel-workgroup-override :kernel-name (:kernel-name registered)
                             :expected default-workgroup :actual workgroup})))
        geometry (when-not (empty-map? registered arguments bound)
                   (if (artifact/kernel-artifact? registered)
                     (call/realize-launch registered arguments)
                     (launch/geometry
                      {:workgroup-size [workgroup]
                       :group-count [(launch/resolve-expression
                                      identity (launch/ceil-div bound workgroup))]})))]
    (when (and geometry (not= 1 (launch/dimensions geometry)))
      (throw (ex-info "direct map requires a one-dimensional launch"
                      {:reason :kernel-launch-dimensionality :launch geometry})))
    geometry))

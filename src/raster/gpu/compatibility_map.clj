(ns ^:no-doc raster.gpu.compatibility-map
  "Pure empty-extent admission for direct compatibility map APIs, not generic launches."
  (:require [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
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

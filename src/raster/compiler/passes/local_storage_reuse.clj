(ns raster.compiler.passes.local-storage-reuse
  "One conservative storage realization over an existing validated LinkPlan and selected order.
   Completion and nonescape must be discharged by the owning runtime scope, not by this pass."
  (:require [raster.compiler.analysis.physical-liveness :as liveness]
            [raster.compiler.ir.link-plan :as link]))

(defn realize-one
  "Realize at most one full-allocation, single-view temporary pair. Never mutate the source plan.
   Returns a revalidated plan and storage delta; the caller still owes completion and nonescape."
  [plan order]
  (let [report (liveness/report plan order)
        full-view? (fn [allocation]
                     (let [slot (get-in report [:slots allocation])
                           view (get-in plan [:nodes (first (:nodes slot)) :view])]
                       (and (pos? (:byte-size slot))
                            (zero? (:byte-offset view))
                            (= (:byte-size slot) (:byte-length view)))))
        contract (fn [allocation]
                   (dissoc (get-in plan [:nodes (first (get-in report [:slots allocation :nodes]))
                                         :view :allocation]) :id))
        candidate (first (filter #(and (= #{:alias-realization :completion-and-escape} (:pending %))
                                        (full-view? (:from %)) (full-view? (:to %))
                                        (= (contract (:from %)) (contract (:to %))))
                                 (:proposals report)))]
    (if-not candidate
      {:plan plan :bytes-saved 0 :allocations-saved 0}
      (let [left (first (get-in report [:slots (:from candidate) :nodes]))
            right (first (get-in report [:slots (:to candidate) :nodes]))
            rewritten (-> plan
                          (assoc-in [:nodes right :view :allocation]
                                    (get-in plan [:nodes left :view :allocation]))
                          (update :aliases conj #{left right})
                          link/validate!)]
        {:plan rewritten :bytes-saved (:bytes candidate) :allocations-saved 1}))))

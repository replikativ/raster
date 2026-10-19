(ns raster.compiler.ir.stage-result
  "Denial-only detection of scheduled result-observation obligations.
   This is not a result validator or execution permission. The emitted-equation
   refinement owner independently reconstructs the obligation; consumers which
   cannot discharge it must refuse before materialization or driver contact.")

(defn requires-observation?
  [value]
  (let [marked? #(or (contains? % :checked-stage-result)
                    (contains? % :checked-stage-results))
        artifact? #(= "raster.compiler.ir.kernel_artifact.KernelArtifact"
                      (some-> % class .getName))]
    (boolean
     (some (fn [node]
             (and (map? node)
                  (or (marked? (:attributes node))
                      (marked? (get-in node [:attributes :kernel-body :attributes]))
                      (marked? (get-in node [:provenance :scheduled-operation :attributes])))))
           ;; Ordinary launch validation must not walk an entire verified body
           ;; on every call. An artifact's obligation is carried at its exact
           ;; scheduled/body projection, inspected directly above.
           (tree-seq #(and (coll? %) (not (artifact? %))) seq value)))))

(defn refuse-unobserved!
  [value operation]
  (when (requires-observation? value)
    (throw (ex-info "checked numerical stage requires an explicit successful-result observation"
                    {:reason :checked-stage-observation-required :operation operation})))
  value)

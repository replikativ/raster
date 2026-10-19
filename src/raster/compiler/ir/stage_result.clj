(ns raster.compiler.ir.stage-result
  "Detection and confinement of scheduled result-observation obligations.
   This is not a result validator or execution permission. The emitted-equation
   refinement owner independently reconstructs the obligation; consumers which
   cannot discharge it must refuse before materialization or driver contact.
   Exact-artifact KernelCall validation may be scoped by the managed observer;
   that scope never authorizes public binding, submission or initialization.")

(deftype ^:private BindingValidationScope [artifacts thread active])
(def ^:dynamic ^:private *pending-call-scope* nil)

(defn- pending-call-artifact? [artifact]
  (when (instance? BindingValidationScope *pending-call-scope*)
    (let [^BindingValidationScope scope *pending-call-scope*]
      (and (identical? (Thread/currentThread) (.-thread scope))
           @(.-active scope)
           (some #(identical? artifact %) (.-artifacts scope))))))

(defn ^:no-doc call-with-checked-binding-validation
  "Validate an exact emitted equation and scope only its KernelCall construction.
   This permits neither public graph binding/submission nor publication. The
   managed runtime binder owes completion, observation and cleanup before its
   enclosing execution owner can credit any result. No capability is returned."
  [equation caller-options bind-and-observe!]
  ;; Resolve at invocation to keep the emitted validator's denial-only dependency
  ;; acyclic. The validator itself never uses this binding-validation context.
  (let [report ((requiring-resolve
                 'raster.compiler.ir.emitted-parallel-equation/validate-with-result-contracts)
                equation caller-options)
        obligations (:stage-results report)]
    (when-not (= 1 (count obligations))
      (throw (ex-info "checked binding requires one reconstructed stage obligation"
                      {:reason :checked-stage-binding-contract})))
    (let [active (volatile! true)
          scope (BindingValidationScope.
                 (mapv :operation (get-in report [:boundary :graph :nodes]))
                 (Thread/currentThread) active)]
      (binding [*pending-call-scope* scope]
        (try (bind-and-observe! report)
             (finally (vreset! active false)))))))

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
  (when (and (requires-observation? value)
             (not (and (contains? #{:kernel-call :make-kernel-call} operation)
                       (pending-call-artifact? value))))
    (throw (ex-info "checked numerical stage requires an explicit successful-result observation"
                    {:reason :checked-stage-observation-required :operation operation})))
  value)

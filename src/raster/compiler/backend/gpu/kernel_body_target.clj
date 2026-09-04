(ns raster.compiler.backend.gpu.kernel-body-target
  "Single target projection for verified ScheduledKernelBody values.

   The fork between scalar/control and matrix instruction spelling is selected from KernelBody
   vocabulary, never from a source operation name.  ABI order, arguments, launch, effects,
   numerical policy, and semantic identity all come from the checked refinement."
  (:require [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as scalar-target]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]))

(defn- record-kind?
  [simple-name value]
  (and value (= simple-name (.getSimpleName (class value)))))

(defn- nested-operations
  [operation]
  (cond
    (record-kind? "IfRegion" operation)
    (concat (:then-operations operation) (:else-operations operation))

    (or (record-kind? "ForLoop" operation)
        (record-kind? "PipelinedFor" operation)
        (record-kind? "Loop" operation)
        (record-kind? "Guard" operation))
    (:operations operation)

    :else []))

(defn- operations
  [kernel-body]
  (letfn [(walk [values]
            (mapcat (fn [operation]
                      (cons operation (walk (nested-operations operation))))
                    values))]
    (walk (:operations kernel-body))))

(defn matrix-body?
  "A body selects matrix target spelling exactly when it contains a scheduled MatrixMad."
  [kernel-body]
  (boolean (some #(record-kind? "MatrixMad" %) (operations (body/validate! kernel-body)))))

(defn- scalar-parameter-names
  [kernel-body overrides]
  (into {}
        (map (fn [parameter]
               [(:id parameter)
                (or (get overrides (:id parameter)) (c-emit/c-symbol (:id parameter)))]))
        (:parameters kernel-body)))

(defn- validate-target-names!
  [kernel-name kernel-body names]
  (when-not (c-emit/c-identifier? kernel-name)
    (throw (ex-info "scheduled kernel entry point is not a portable C-family identifier"
                    {:reason :kernel-body-target-entry-point :kernel-name kernel-name})))
  (let [ordered (mapv #(get names (:id %)) (:parameters kernel-body))]
    (doseq [[parameter target-name] (map vector (:parameters kernel-body) ordered)]
      (when-not (c-emit/c-identifier? target-name)
        (throw (ex-info "scheduled kernel parameter is not a portable C-family identifier"
                        {:reason :kernel-body-target-parameter-name
                         :parameter parameter :target-name target-name}))))
    (when-not (= (count ordered) (count (set ordered)))
      (throw (ex-info "scheduled kernel parameter names are not unique"
                      {:reason :kernel-body-target-name-collision :names ordered})))
    names))

(defn- project-abi
  [scheduled names alignments]
  (let [kernel-body (:body scheduled)
        scalar-bindings (into {} (map (juxt :parameter identity))
                              (:scalar-bindings scheduled))]
  (body-abi/project-contracts
   (mapv (fn [{:keys [id kind dtype role]}]
           (let [binding (get scalar-bindings id)]
             (abi/slot id kind (or (:dtype binding) dtype)
                       :kernel-dtype (or (:kernel-dtype binding) dtype)
                       :c-name (get names id) :role role
                       :alignment (get alignments id))))
         (:parameters kernel-body))
   kernel-body)))

(defn- scalar-target-facts
  [kernel-body dialect target-features]
  (let [consumed (select-keys target-features [:compute-capability :architecture])
        physical-dialect (merge dialect consumed)
        operation-kinds (set (map #(some-> % class .getSimpleName)
                                  (operations kernel-body)))
        collective? (contains? operation-kinds "Collective")
        async? (some operation-kinds
                     #{"AsyncWorkgroupCopy" "AsyncCommit" "AsyncWait" "PipelinedFor"})]
    (cond-> {:target-dialect (:id dialect)}
      collective?
      (assoc :collective-association (c-dialect/collective-association physical-dialect))
      async?
      (assoc :async-copy-mode (c-dialect/async-copy-mode physical-dialect)
             :target-features consumed))))

(defn emit-artifact
  "Emit a ScheduledKernelBody to one C-family target artifact.

   Optional metadata is non-authoritative: checked body/refinement and selected-target facts win
   on key collisions.  `parameter-names` and `target-features` affect target spelling only."
  ([kernel-name scheduled target-dialect]
   (emit-artifact kernel-name scheduled target-dialect {}))
  ([kernel-name scheduled target-dialect
    {:keys [parameter-names target-features provenance attributes]}]
   (let [scheduled (scheduled-body/validate! scheduled)
         kernel-body (:body scheduled)
         matrix? (matrix-body? kernel-body)
         dialect (c-dialect/resolve! target-dialect)
         emitted
         (if matrix?
           (matrix-target/emit-matrix-kernel
            kernel-name kernel-body target-dialect {:parameter-names parameter-names})
           (let [names (validate-target-names!
                        kernel-name kernel-body
                        (scalar-parameter-names kernel-body parameter-names))]
             {:source (scalar-target/emit-scalar-kernel
                       kernel-name kernel-body
                       {:target-dialect target-dialect
                        :target-features target-features
                        :parameter-names names})
              :target (c-dialect/target dialect)
              :target-dialect (:id dialect)
              :parameter-names names
              :parameter-alignments {}
              :target-facts (scalar-target-facts kernel-body dialect target-features)}))
         names (validate-target-names! kernel-name kernel-body (:parameter-names emitted))
         projected-abi (project-abi scheduled names (:parameter-alignments emitted))]
     (scheduled-body/validate-artifact-projection!
      scheduled
      (artifact/make
       {:kernel-name kernel-name
        :target (:target emitted)
        :source (:source emitted)
        :abi projected-abi
        :arguments (:arguments scheduled)
        :launch (scheduled-body/realized-launch scheduled)
        :temporaries []
        :effects (:effects scheduled)
        :provenance (merge provenance
                          (:provenance kernel-body)
                          (:provenance scheduled)
                          {:lowering :scheduled-kernel-body
                           :target-dialect (:target-dialect emitted)
                           :scheduled-operation scheduled
                           :target-facts (:target-facts emitted)})
        :attributes (merge attributes
                          (:attributes scheduled)
                          {:kernel-body kernel-body
                           :scheduled-kernel-body scheduled
                           :emission-route :kernel-body
                           :body-family (if matrix? :matrix :scalar-control)
                           :legality (:legality scheduled)
                           :numerics (:numerics scheduled)
                           :target-facts (:target-facts emitted)})})))))

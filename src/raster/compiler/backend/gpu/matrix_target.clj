(ns raster.compiler.backend.gpu.matrix-target
  "Target selection for verified matrix KernelBody values.

   Contraction scheduling chooses the matrix instruction and records the complete execution in a
   KernelBody. This namespace performs only the final target spelling. It is deliberately the one
   fork at which DPAS and MMA diverge; callers may not select a target template directly or pass a
   second tile/launch/ABI description beside the body."
  (:require [raster.compiler.backend.gpu.cuda-codegen :as cuda-codegen]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.ir.kernel-abi :as kernel-abi]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-launch :as kernel-launch]))

(defn- target-parameter-names
  [body overrides]
  (let [{:keys [m n k]} (get-in body [:attributes :dimension-parameters])]
    (merge
     (into {}
           (map (fn [{:keys [id role]}]
                  [id (case role
                        :lhs "A"
                        :rhs "B"
                        :result "C"
                        (c-emit/c-symbol id))]))
           (:parameters body))
     {m "M" n "N" k "K"}
     overrides)))

(def ^:private matrix-local-names
  #{"warpId" "warp_row" "warp_col" "sg_id" "sg_lid" "sg_row" "sg_col"
    "m_base" "k" "pk" "pr" "row" "col" "k_begin" "k_end"
    "a_wb" "a_pb" "b_wb" "b_pb"})

(defn- generated-matrix-local?
  [generated-locals c-name]
  (or (contains? generated-locals c-name)
      (boolean (re-matches #"(?:a|b|sa|bp|acc|n_base)[0-9]+(?:_[0-9]+)?" c-name))))

(defn- matrix-generated-locals
  [body]
  (into matrix-local-names
        (comp (filter #(and (= :group (:source %)) (= 2 (:axis %))))
              (map #(c-emit/c-symbol (:id %))))
        (:indices body)))

(defn- validate-target-names!
  [kernel-name body names]
  (when-not (c-emit/c-identifier? kernel-name)
    (throw (ex-info "matrix kernel entry point is not a portable C-family identifier"
                    {:reason :matrix-target-entry-point :kernel-name kernel-name})))
  (let [ordered-names (mapv #(get names (:id %)) (:parameters body))
        generated-locals (matrix-generated-locals body)]
    (doseq [[parameter c-name] (map vector (:parameters body) ordered-names)]
      (when-not (c-emit/c-identifier? c-name)
        (throw (ex-info "matrix ABI parameter is not a portable C-family identifier"
                        {:reason :matrix-target-parameter-name
                         :parameter parameter :c-name c-name})))
      (when (generated-matrix-local? generated-locals c-name)
        (throw (ex-info "matrix ABI parameter collides with a generated kernel local"
                        {:reason :matrix-target-name-collision
                         :parameter parameter :c-name c-name}))))
    (when-not (= (count ordered-names) (count (set ordered-names)))
      (throw (ex-info "matrix ABI parameter names are not unique after target spelling"
                      {:reason :matrix-target-name-collision
                       :parameters (:parameters body) :c-names ordered-names})))
    names))

(defn- projected-abi
  [body names target-dialect]
  (let [target-alignment (when (= :cuda target-dialect) 32)]
    (body-abi/project-contracts
     (mapv (fn [{:keys [id kind dtype role]}]
             (kernel-abi/slot id kind dtype
                              :c-name (get names id)
                              :role role
                              :alignment (when (and target-alignment (not= :scalar kind))
                                           target-alignment)))
           (:parameters body))
     body)))

(defn- body-arguments
  [body]
  (let [dimension-values (get-in body [:attributes :dimension-values])]
    (mapv (fn [{:keys [id role]}]
            (if (= :dimension role) (get dimension-values id id) id))
          (:parameters body))))

(defn- body-effects
  [body]
  {:kind :tensor-contraction-stage
   :reads (mapv :id (filter #(contains? #{:input :inout} (:kind %)) (:parameters body)))
   :writes (mapv :id (filter #(contains? #{:output :inout} (:kind %)) (:parameters body)))})

(defn emit-matrix-kernel
  "Emit `body` for one C-family target dialect.

   Returns the target module together with the validated body and concrete artifact target. The
   optional parameter-name map controls spelling only on the Intel OpenCL row; CUDA derives its
   ordered signature from KernelBody parameter roles. Unsupported target/instruction pairs fail
   loudly in the selected target lowerer."
  ([kernel-name body target-dialect]
   (emit-matrix-kernel kernel-name body target-dialect {}))
  ([kernel-name body target-dialect {:keys [parameter-names]}]
   (let [body (kernel-body/validate! body)
         dialect (c-dialect/resolve! target-dialect)
         parameter-names (validate-target-names!
                          kernel-name body
                          (target-parameter-names body parameter-names))
         default-names (target-parameter-names body nil)
         _ (when (and (= :cuda (:id dialect)) (not= default-names parameter-names))
             (throw (ex-info "CUDA matrix lowering does not support ABI spelling overrides"
                             {:reason :cuda-mma-parameter-spelling-unsupported
                              :target :cuda :requested parameter-names
                              :required default-names})))
         source
         (case (:id dialect)
           :opencl-intel
           (kernel-body-opencl/emit-matrix-kernel
            kernel-name body {:parameter-names parameter-names})

           :cuda
           (cuda-codegen/emit-matrix-kernel kernel-name body)

           (throw (ex-info "matrix KernelBody target lowering is not implemented for this dialect"
                           {:reason :matrix-target-dialect-not-lowered
                            :target-dialect (:id dialect)
                            :target (c-dialect/target dialect)
                            :instruction-family
                            (get-in body [:attributes :instruction-family])})))]
     {:source source
      :target (c-dialect/target dialect)
      :target-dialect (:id dialect)
      :parameter-names parameter-names
      :kernel-body body})))

(defn emit-matrix-artifact
  "Project one verified matrix body into a complete target artifact.

   ABI order, memory contracts, compiler arguments and launch geometry are consequences of the
   body. Static dimension specializations become literal artifact arguments; dynamic dimensions
   retain their compiler identities. Target parameter names are spelling policy only."
  ([kernel-name body target-dialect]
   (emit-matrix-artifact kernel-name body target-dialect {}))
  ([kernel-name body target-dialect {:keys [parameter-names provenance attributes]}]
   (let [body (kernel-body/validate! body)
         dimension-values (get-in body [:attributes :dimension-values])
         emitted (emit-matrix-kernel kernel-name body target-dialect
                                     {:parameter-names parameter-names})
         names (:parameter-names emitted)]
     (kernel-artifact/make
      {:kernel-name kernel-name
       :target (:target emitted)
       :source (:source emitted)
       :abi (projected-abi body names (:target-dialect emitted))
       :arguments (body-arguments body)
       :launch (kernel-launch/rebind-spec (:launch body) dimension-values)
       :effects (body-effects body)
       :provenance (merge provenance
                          (:provenance body)
                          {:lowering :matrix-kernel-body
                           :target-dialect (:target-dialect emitted)})
       :attributes (merge attributes
                          {:kernel-body body
                           :emission-route :kernel-body
                           :instruction-family
                           (get-in body [:attributes :instruction-family])})}))))

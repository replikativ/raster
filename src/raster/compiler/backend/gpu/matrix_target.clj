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
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]
            [raster.compiler.ir.kernel-body :as kernel-body]))

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

(defn physical-requirements
  "Return target-derived ABI requirements and identity-bearing physical lowering facts."
  [body target-dialect plan]
  (let [body (kernel-body/validate! body)
        dialect (c-dialect/resolve! target-dialect)
        pointer-alignment (when (= :cuda (:id dialect)) 32)]
    {:parameter-alignments
     (into {}
           (keep (fn [{:keys [id kind]}]
                   (when (and pointer-alignment (not= :scalar kind))
                     [id pointer-alignment])))
           (:parameters body))
     :target-facts
     {:target-dialect (:id dialect)
      :instruction (:instruction plan)
      :instruction-family (get-in plan [:instruction :family])
      :pointer-alignment pointer-alignment}}))

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
         plan (matrix-plan/analyze body)
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
     (merge {:source source
      :target (c-dialect/target dialect)
      :target-dialect (:id dialect)
      :parameter-names parameter-names
      :kernel-body body}
            (physical-requirements body (:id dialect) plan)))))

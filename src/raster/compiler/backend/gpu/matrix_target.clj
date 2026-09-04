(ns raster.compiler.backend.gpu.matrix-target
  "Target selection for verified matrix KernelBody values.

   Contraction scheduling chooses the matrix instruction and records the complete execution in a
   KernelBody. This namespace performs only the final target spelling. It is deliberately the one
   fork at which DPAS and MMA diverge; callers may not select a target template directly or pass a
   second tile/launch/ABI description beside the body."
  (:require [raster.compiler.backend.gpu.cuda-codegen :as cuda-codegen]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.ir.kernel-body :as kernel-body]))

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
      :kernel-body body})))

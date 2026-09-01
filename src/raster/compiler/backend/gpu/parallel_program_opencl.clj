(ns raster.compiler.backend.gpu.parallel-program-opencl
  "Compatibility entry for equation-first OpenCL emission.

   The target-parametric implementation lives in `parallel-program-c-family`. New public compiler
   paths should use that namespace directly."
  (:require [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]))

(defn validate-program!
  [parallel-program]
  (c-family/validate-program! parallel-program))

(defn emit-program
  "Emit an equation-first program to an OpenCL KernelBody source dialect."
  ([parallel-program]
   (emit-program parallel-program {}))
  ([parallel-program options]
   (let [target-dialect (get options :target-dialect :opencl-intel)]
     (when-not (c-dialect/opencl? (c-dialect/resolve! target-dialect))
       (throw (ex-info "OpenCL compatibility entry requires an OpenCL target dialect"
                       {:reason :parallel-program-opencl-target
                        :target-dialect target-dialect
                        :fallback :none})))
     (c-family/emit-program parallel-program
                            (assoc options :target-dialect target-dialect)))))

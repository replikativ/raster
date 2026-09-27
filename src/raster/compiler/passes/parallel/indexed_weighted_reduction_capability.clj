(ns raster.compiler.passes.parallel.indexed-weighted-reduction-capability
  "Pure target-capability admission for indexed weighted-reduction schedules.

   This namespace chooses no schedule and emits no kernel.  It is shared by compatibility routing
   and equation-first certificate construction so both boundaries retain one ordered set of
   hardware, dtype, and storage legality checks."
  (:require [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- declined
  [reason data]
  {:status :declined :reason reason :data data})

(defn score-reuse
  "Check whether `descriptor` can execute the existing subgroup score-reuse body.

   Returns `{:status :supported :width n}` or a structured decline retaining the compatibility
   route's established reason and data. Invalid semantic plans still throw: malformed semantics
   are not a target capability decline. The current production boundary is intentionally the
   proven Intel OpenCL subgroup dialect; CUDA/HIP admission requires its own emitter evidence."
  [plan descriptor]
  (let [{:keys [operands output accumulator-dtype]} (swr/validate! plan)
        storage-dtypes (mapv :dtype (conj operands output))
        subgroup-size (hardware/preferred-subgroup-size descriptor)
        max-workgroup-size (hardware/maximum-workgroup-size descriptor)
        supported-widths (hardware/supported-subgroup-sizes descriptor)
        matrix-family (get-in descriptor [:matrix :family])
        intel-dialect? (gpu-target/intel-opencl-subgroup-dialect? descriptor)]
    (cond
      (and descriptor (not= :gpu (:device-type descriptor)))
      (declined :score-reuse-requires-gpu
                {:device-type (:device-type descriptor)})

      (not intel-dialect?)
      (declined :score-reuse-requires-intel-subgroup-dialect
                {:vendor (:vendor descriptor) :matrix-family matrix-family})

      (or (not (integer? subgroup-size))
          (not (integer? max-workgroup-size)))
      (declined :score-reuse-missing-execution-capability
                {:subgroup-size subgroup-size
                 :max-workgroup-size max-workgroup-size})

      (and (seq supported-widths)
           (not (contains? (set supported-widths) subgroup-size)))
      (declined :score-reuse-subgroup-width-unsupported
                {:subgroup-size subgroup-size
                 :supported-subgroup-sizes (set supported-widths)})

      (not (contains? #{:float :double} accumulator-dtype))
      (declined :score-reuse-accumulator-unsupported
                {:required #{:float :double} :actual accumulator-dtype})

      (not= [accumulator-dtype accumulator-dtype accumulator-dtype
             :long :long accumulator-dtype]
            storage-dtypes)
      (declined :score-reuse-storage-unsupported
                {:required [accumulator-dtype accumulator-dtype accumulator-dtype
                            :long :long accumulator-dtype]
                 :actual storage-dtypes})

      (or (not (pos? (long subgroup-size)))
          (> (long subgroup-size) (long max-workgroup-size)))
      (declined :score-reuse-invalid-subgroup-geometry
                {:subgroup-size subgroup-size
                 :max-workgroup-size max-workgroup-size})

      :else
      {:status :supported :width (long subgroup-size)})))

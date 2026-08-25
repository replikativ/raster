(ns raster.compiler.passes.parallel.paged-kv-append-route
  "Structured routing for backend-neutral paged K/V append problems."
  (:require [raster.compiler.backend.gpu.paged-kv-append :as emit]
            [raster.compiler.ir.paged-kv-append :as append]))

(defn- decline
  [problem reason data]
  {:operation problem :strategy nil :reference? false
   :declines [{:leaf :fp32-to-fp16-reference :reason reason :data data}]})

(defn route
  "Try the portable FP32-to-FP16 assignment leaf or return a structured decline."
  ([problem] (route problem nil))
  ([problem desc]
   (let [{:keys [key-input-dtype value-input-dtype key-storage-dtype
                 value-storage-dtype rounding-mode]
          :as problem} (append/validate! problem)]
     (cond
       (and desc (not= :gpu (:device-type desc)))
       (decline problem :paged-kv-append-requires-gpu
                {:device-type (:device-type desc)})

       (not= [:float :float :half :half]
             [key-input-dtype value-input-dtype key-storage-dtype value-storage-dtype])
       (decline problem :paged-kv-append-dtype-unsupported
                {:required [:float :float :half :half]
                 :actual [key-input-dtype value-input-dtype
                          key-storage-dtype value-storage-dtype]})

       (not= :round-to-nearest-even rounding-mode)
       (decline problem :paged-kv-append-rounding-unsupported
                {:rounding-mode rounding-mode})

       :else
       (let [artifact (emit/emit-fp32-to-fp16-reference problem desc)]
         {:operation problem
          :strategy :fp32-to-fp16-reference
          :reference? true
          :declines []
          :artifact artifact
          :graph (emit/kernel-graph problem artifact)
          :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                     :group-count (:group-count (:launch artifact))}})))))

(defn route!
  "Route paged K/V append or fail with the machine-readable decline trail."
  ([problem] (route! problem nil))
  ([problem desc]
   (let [result (route problem desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable paged K/V append kernel route"
                       {:reason :paged-kv-append-no-kernel-route
                        :route result}))))))

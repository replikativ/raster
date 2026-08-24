(ns raster.compiler.passes.parallel.paged-attention-route
  "Structured routing for backend-neutral paged attention descriptors."
  (:require [raster.compiler.backend.gpu.paged-attention :as emit]
            [raster.compiler.ir.paged-attention :as paged]))

(defn- decline
  [operation reason data]
  {:operation operation
   :strategy nil
   :reference? false
   :declines [{:leaf :fp16-reference :reason reason :data data}]})

(defn route
  "Try the first executable paged-attention leaf and return a route or structured decline.

   Quantized cache formats stay in semantic IR, but decline until their scale/group ABI is
   explicit. This prevents a storage tag from smuggling a dequantization convention into codegen."
  ([operation] (route operation nil))
  ([operation desc]
   (let [{:keys [q-dtype cache-dtype output-dtype accumulator-dtype cache-format]
          :as operation} (paged/validate! operation)]
     (cond
       (and desc (not= :gpu (:device-type desc)))
       (decline operation :paged-attention-requires-gpu
                {:device-type (:device-type desc)})

       (not= :none (:quantization cache-format))
       (decline operation :paged-attention-quantized-cache-abi-unimplemented
                {:cache-format cache-format})

       (not= [:half :half :half]
             [q-dtype cache-dtype output-dtype])
       (decline operation :paged-attention-reference-storage-unsupported
                {:required [:half :half :half]
                 :actual [q-dtype cache-dtype output-dtype]})

       (not= :float accumulator-dtype)
       (decline operation :paged-attention-reference-accumulator-unsupported
                {:required :float :actual accumulator-dtype})

       :else
       (let [artifact (emit/emit-fp16-reference operation desc)]
         {:operation operation
          :strategy :fp16-reference
          :reference? true
          :declines []
          :artifact artifact
          :graph (emit/kernel-graph operation artifact)
          :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                     :group-count (:group-count (:launch artifact))}})))))

(defn route!
  "Route paged attention or fail with the complete machine-readable decline trail."
  ([operation] (route! operation nil))
  ([operation desc]
   (let [result (route operation desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable paged-attention kernel route"
                       {:reason :paged-attention-no-kernel-route :route result}))))))

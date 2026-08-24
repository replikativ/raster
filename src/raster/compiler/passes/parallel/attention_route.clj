(ns raster.compiler.passes.parallel.attention-route
  "Structured routing for backend-neutral attention problems."
  (:require [raster.compiler.backend.gpu.attention :as emit]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.passes.parallel.attention-lower :as lower]))

(defn- decline
  [problem reason data]
  {:operation problem
   :strategy nil
   :reference? false
   :declines [{:leaf :fp16-reference :reason reason :data data}]})

(defn route
  "Try the executable packed/routed attention reference or return a structured decline.

   Quantized K/V formats stay semantic values but decline until their scale/group operands are
   explicit. Dense and CSR routing are both reference leaves with deliberately distinct ABIs."
  ([problem] (route problem nil))
  ([problem desc]
   (let [{:keys [q-dtype k-dtype v-dtype output-dtype accumulator-dtype
                 k-format v-format]
          :as problem} (attention/validate! problem)]
     (cond
       (and desc (not= :gpu (:device-type desc)))
       (decline problem :attention-requires-gpu
                {:device-type (:device-type desc)})

       (or (not= :none (:quantization k-format))
           (not= :none (:quantization v-format)))
       (decline problem :attention-quantized-kv-abi-unimplemented
                {:k-format k-format :v-format v-format})

       (not= [:half :half :half :half]
             [q-dtype k-dtype v-dtype output-dtype])
       (decline problem :attention-reference-storage-unsupported
                {:required [:half :half :half :half]
                 :actual [q-dtype k-dtype v-dtype output-dtype]})

       (not= :float accumulator-dtype)
       (decline problem :attention-reference-accumulator-unsupported
                {:required :float :actual accumulator-dtype})

       :else
       (let [plan (lower/lower problem)
             artifact (emit/emit-fp16-reference plan desc)]
         {:operation problem
          :plan plan
          :strategy :fp16-reference
          :reference? true
          :declines []
          :artifact artifact
          :graph (emit/kernel-graph plan artifact)
          :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                     :group-count (:group-count (:launch artifact))}})))))

(defn route!
  "Route attention or fail with the complete machine-readable decline trail."
  ([problem] (route! problem nil))
  ([problem desc]
   (let [result (route problem desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable attention kernel route"
                       {:reason :attention-no-kernel-route :route result}))))))

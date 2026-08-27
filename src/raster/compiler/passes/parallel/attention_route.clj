(ns raster.compiler.passes.parallel.attention-route
  "Structured routing for backend-neutral attention problems."
  (:require [raster.compiler.backend.gpu.attention :as emit]
            [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.passes.parallel.attention-lower :as lower]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-schedule :as swr-schedule]))

(defn- decline
  [problem reason data]
  {:operation problem
   :strategy nil
   :reference? false
   :declines [{:leaf :fp16-reference :reason reason :data data}]})

(def ^:private cooperative-policies
  #{:subgroup-score-reuse :subgroup-online-score-reuse})

(defn- requested-policy
  [desc]
  (or (:segmented-weighted-reduction-schedule desc) :auto))

(defn- success
  [problem plan artifact declines]
  (let [strategy (get-in artifact [:attributes :strategy])
        reference? (= :reference (get-in artifact [:attributes :optimization-tier]))]
    {:operation problem
     :plan plan
     :strategy strategy
     :reference? reference?
     :declines declines
     :artifact artifact
     :graph (emit/kernel-graph plan artifact)
     :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                :group-count (:group-count (:launch artifact))}}))

(defn route
  "Route packed/routed attention through a cooperative schedule or its semantic oracle.

   Quantized K/V formats stay semantic values but decline until their scale/group operands are
   explicit. Dense and CSR physical routing retain deliberately distinct ABIs while interval and
   CSR logical membership both admit the same one-subgroup online-softmax schedule."
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

       (not (and (contains? #{:half :float} q-dtype)
                 (= :half k-dtype)
                 (= :half v-dtype)
                 (contains? #{:half :float} output-dtype)))
       (decline problem :attention-reference-storage-unsupported
                {:required {:q-dtype #{:half :float}
                            :k-dtype :half
                            :v-dtype :half
                            :output-dtype #{:half :float}}
                 :actual [q-dtype k-dtype v-dtype output-dtype]})

       (not= :float accumulator-dtype)
       (decline problem :attention-reference-accumulator-unsupported
                {:required :float :actual accumulator-dtype})

       :else
       (let [plan (lower/lower problem)
             policy (requested-policy desc)]
         (cond
           (= :reference policy)
           (success problem plan (emit/emit-fp16-reference plan desc) [])

           (or (= :auto policy) (contains? cooperative-policies policy))
           (let [{:keys [ok schedule reason] :as scheduled}
                 (swr-schedule/plan-subgroup-online plan desc)
                 emitter-supported? (gpu-target/intel-opencl-subgroup-dialect? desc)]
             (if (and ok emitter-supported?)
               (success problem plan (emit/emit-fp16-cooperative plan schedule) [])
               (let [cooperative-decline
                     {:leaf :routed-paged-subgroup-online-score-reuse
                      :reason (if ok
                                :score-reuse-requires-intel-subgroup-dialect
                                reason)
                      :data (if ok
                              {:vendor (:vendor desc)
                               :matrix-family (get-in desc [:matrix :family])}
                              (dissoc scheduled :ok :reason))}]
                 (if (= :auto policy)
                   (success problem plan (emit/emit-fp16-reference plan desc)
                            [cooperative-decline])
                   {:operation problem
                    :plan plan
                    :strategy nil
                    :reference? false
                    :declines [cooperative-decline]}))))

           :else
           {:operation problem
            :plan plan
            :strategy nil
            :reference? false
            :declines [{:leaf :attention-schedule-policy
                        :reason :attention-schedule-policy-unsupported
                        :data {:requested policy
                               :supported (into [:auto :reference] cooperative-policies)}}]}))))))

(defn route!
  "Route attention or fail with the complete machine-readable decline trail."
  ([problem] (route! problem nil))
  ([problem desc]
   (let [result (route problem desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable attention kernel route"
                       {:reason :attention-no-kernel-route :route result}))))))

(ns raster.compiler.passes.parallel.indexed-attention-route
  "Structured routing for recognized indexed graph-attention plans."
  (:require [raster.compiler.backend.gpu.indexed-attention :as emit]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- decline
  [plan reason data]
  {:operation plan
   :strategy nil
   :reference? false
   :declines [{:leaf :indexed-edge-list-reference :reason reason :data data}]})

(defn route
  "Try the direct fused correctness leaf using concrete values for symbolic plan extents."
  ([plan shape-env] (route plan shape-env nil))
  ([plan shape-env desc]
   (try
     (let [{:keys [operands output accumulator-dtype] :as plan} (swr/validate! plan)
           storage-dtypes (mapv :dtype (conj operands output))]
       (cond
         (and desc (not= :gpu (:device-type desc)))
         (decline plan :indexed-attention-requires-gpu
                  {:device-type (:device-type desc)})

         (not (contains? #{:float :double} accumulator-dtype))
         (decline plan :indexed-attention-accumulator-unsupported
                  {:required #{:float :double} :actual accumulator-dtype})

         (not= [accumulator-dtype accumulator-dtype accumulator-dtype
                :long :long accumulator-dtype]
               storage-dtypes)
         (decline plan :indexed-attention-storage-unsupported
                  {:required [accumulator-dtype accumulator-dtype accumulator-dtype
                              :long :long accumulator-dtype]
                   :actual storage-dtypes})

         :else
         (let [artifact (emit/emit-reference plan shape-env desc)]
           {:operation plan
            :plan plan
            :strategy :indexed-segmented-reduction-reference
            :reference? true
            :declines []
            :artifact artifact
            :graph (emit/kernel-graph plan shape-env artifact)
            :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                       :group-count (:group-count (:launch artifact))}})))
     (catch clojure.lang.ExceptionInfo e
       (decline plan (or (:reason (ex-data e)) :indexed-segmented-reduction-unsupported)
                (dissoc (ex-data e) :reason))))))

(defn route!
  "Route indexed attention or fail with its machine-readable decline trail."
  ([plan shape-env] (route! plan shape-env nil))
  ([plan shape-env desc]
   (let [result (route plan shape-env desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable indexed-attention kernel route"
                       {:reason :indexed-attention-no-kernel-route :route result}))))))

(defn route-dynamic
  "Route a recognized plan to the shape-polymorphic resident/staging artifact."
  ([plan] (route-dynamic plan nil))
  ([plan desc]
   (try
     (let [{:keys [operands output accumulator-dtype] :as plan} (swr/validate! plan)
           storage-dtypes (mapv :dtype (conj operands output))]
       (cond
         (and desc (not= :gpu (:device-type desc)))
         (decline plan :indexed-attention-requires-gpu
                  {:device-type (:device-type desc)})

         (not (contains? #{:float :double} accumulator-dtype))
         (decline plan :indexed-attention-accumulator-unsupported
                  {:required #{:float :double} :actual accumulator-dtype})

         (not= [accumulator-dtype accumulator-dtype accumulator-dtype
                :long :long accumulator-dtype]
               storage-dtypes)
         (decline plan :indexed-attention-storage-unsupported
                  {:required [accumulator-dtype accumulator-dtype accumulator-dtype
                              :long :long accumulator-dtype]
                   :actual storage-dtypes})

         :else
         (let [artifact (emit/emit-dynamic-reference plan desc)]
           {:operation plan :plan plan
            :strategy :indexed-segmented-reduction-reference
            :reference? true :dynamic-shape? true :declines []
            :artifact artifact
            :graph (emit/dynamic-kernel-graph plan artifact)
            :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                       :group-count (:group-count (:launch artifact))}})))
     (catch clojure.lang.ExceptionInfo e
       (decline plan (or (:reason (ex-data e)) :indexed-segmented-reduction-unsupported)
                (dissoc (ex-data e) :reason))))))

(defn route-dynamic!
  ([plan] (route-dynamic! plan nil))
  ([plan desc]
   (let [result (route-dynamic plan desc)]
     (if (:strategy result)
       result
       (throw (ex-info "no executable dynamic indexed-attention kernel route"
                       {:reason :indexed-attention-no-kernel-route :route result}))))))

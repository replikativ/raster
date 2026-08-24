(ns raster.compiler.passes.parallel.indexed-attention-route
  "Structured routing for recognized indexed graph-attention plans."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.indexed-attention :as emit]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- decline
  ([plan reason data] (decline plan :indexed-edge-list-reference reason data))
  ([plan leaf reason data]
   {:operation plan
    :strategy nil
    :reference? false
    :declines [{:leaf leaf :reason reason :data data}]}))

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

(defn route-dynamic-score-reuse
  "Route to the destination/head subgroup leaf that shares each score across component lanes."
  ([plan] (route-dynamic-score-reuse plan nil))
  ([plan desc]
   (let [leaf :indexed-edge-list-subgroup-score-reuse]
     (try
       (let [{:keys [operands output accumulator-dtype] :as plan} (swr/validate! plan)
             storage-dtypes (mapv :dtype (conj operands output))
             subgroup-size (long (or (:subgroup-size desc) 16))
             max-workgroup-size (long (or (:max-workgroup-size desc) 256))
             vendor (some-> (:vendor desc) str str/lower-case)
             matrix-family (get-in desc [:matrix :family])
             known-non-intel? (and (or vendor matrix-family)
                                   (not (or (= :dpas matrix-family)
                                            (and vendor (str/includes? vendor "intel")))))]
         (cond
           (and desc (not= :gpu (:device-type desc)))
           (decline plan leaf :score-reuse-requires-gpu
                    {:device-type (:device-type desc)})

           known-non-intel?
           (decline plan leaf :score-reuse-requires-intel-subgroup-dialect
                    {:vendor (:vendor desc) :matrix-family matrix-family})

           (not (contains? #{:float :double} accumulator-dtype))
           (decline plan leaf :score-reuse-accumulator-unsupported
                    {:required #{:float :double} :actual accumulator-dtype})

           (not= [accumulator-dtype accumulator-dtype accumulator-dtype
                  :long :long accumulator-dtype]
                 storage-dtypes)
           (decline plan leaf :score-reuse-storage-unsupported
                    {:required [accumulator-dtype accumulator-dtype accumulator-dtype
                                :long :long accumulator-dtype]
                     :actual storage-dtypes})

           (or (not (pos? subgroup-size))
               (> subgroup-size max-workgroup-size))
           (decline plan leaf :score-reuse-invalid-subgroup-geometry
                    {:subgroup-size subgroup-size
                     :max-workgroup-size max-workgroup-size})

           :else
           (let [artifact (emit/emit-dynamic-score-reuse plan desc)]
             {:operation plan :plan plan
              :strategy :indexed-segmented-reduction-subgroup-score-reuse
              :reference? false :dynamic-shape? true :declines []
              :artifact artifact
              :graph (emit/dynamic-kernel-graph plan artifact)
              :schedule {:workgroup-size (:workgroup-size (:launch artifact))
                         :group-count (:group-count (:launch artifact))}})))
       (catch clojure.lang.ExceptionInfo e
         (decline plan leaf
                  (or (:reason (ex-data e)) :indexed-score-reuse-unsupported)
                  (dissoc (ex-data e) :reason)))))))

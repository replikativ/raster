(ns raster.compiler.ir.matrix-stage
  "Exact target-neutral operations introduced by matrix-contraction graph scheduling.

   A MatrixStage is later than a semantic SegRed and earlier than KernelBody. It records the
   representation, batching, reduction partition and input/result transforms chosen for one graph node;
   distinct direct, split-K and batched stages therefore cannot share a fabricated source
   identity in refinement certificates. Input value regions map physical operand elements to
   operand-dtype fragments; the closed typed SSA region retains the conversion policy without
   introducing a second scalar language or assuming target support."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-body :as kernel-body]))

(defrecord MatrixStage
           [id lhs rhs result dimensions batching reduction result-shape epilogue
            operand-dtype accumulator-dtype result-dtype schedule input-value-regions])

(defn matrix-stage?
  [value]
  (and value
       (= "raster.compiler.ir.matrix_stage.MatrixStage" (.getName (class value)))))

(defn validate!
  [stage]
  (when-not (matrix-stage? stage)
    (throw (ex-info "expected a MatrixStage"
                    {:reason :matrix-stage-type :actual (type stage)})))
  (let [{:keys [id lhs rhs result dimensions batching reduction result-shape epilogue
                operand-dtype accumulator-dtype result-dtype schedule input-value-regions]} stage]
    (doseq [[field value] [[:id id] [:lhs lhs] [:rhs rhs] [:result result]]]
      (when (nil? value)
        (throw (ex-info "matrix stage is missing an identity"
                        {:reason :matrix-stage-identity :field field :stage stage}))))
    (when-not (and (vector? dimensions) (= 3 (count dimensions)) (not-any? nil? dimensions))
      (throw (ex-info "matrix stage requires exact M/N/K dimensions"
                      {:reason :matrix-stage-dimensions :dimensions dimensions})))
    (when-not (and (map? reduction)
                   (contains? #{:full :split-k} (:kind reduction))
                   (vector? (:range reduction)) (= 2 (count (:range reduction))))
      (throw (ex-info "matrix stage requires an exact reduction partition"
                      {:reason :matrix-stage-reduction :reduction reduction})))
    (when (= :split-k (:kind reduction))
      (doseq [field [:slice :chunk :partitions]]
        (when (nil? (get reduction field))
          (throw (ex-info "split-K matrix stage is missing partition geometry"
                          {:reason :matrix-stage-reduction :field field
                           :reduction reduction})))))
    (when-not (and (vector? result-shape) (seq result-shape) (not-any? nil? result-shape))
      (throw (ex-info "matrix stage requires its exact result shape"
                      {:reason :matrix-stage-result-shape :result-shape result-shape})))
    (when-not (or (nil? batching)
                  (and (map? batching) (some? (:extent batching))
                       (every? boolean? ((juxt :lhs :rhs) batching))))
      (throw (ex-info "matrix stage batching must state extent and operand broadcast semantics"
                      {:reason :matrix-stage-batching :batching batching})))
    (when-not (or (nil? epilogue) (map? epilogue))
      (throw (ex-info "matrix stage epilogue must be an exact transform description"
                      {:reason :matrix-stage-epilogue :epilogue epilogue})))
    (when-not (or (nil? schedule) (map? schedule))
      (throw (ex-info "matrix stage schedule must be an explicit map"
                      {:reason :matrix-stage-schedule :schedule schedule})))
    (doseq [[field value] [[:operand-dtype operand-dtype]
                           [:accumulator-dtype accumulator-dtype]
                           [:result-dtype result-dtype]]]
      (when-not (and (dtype/known? value) (= value (dtype/canon value)))
        (throw (ex-info "matrix stage requires canonical numerical dtypes"
                        {:reason :matrix-stage-dtype :field field :dtype value}))))
    (when-not (and (map? input-value-regions)
                   (every? (set [lhs rhs]) (keys input-value-regions)))
      (throw (ex-info "matrix input regions must name current operands"
                      {:reason :matrix-stage-input-regions :regions input-value-regions})))
    (doseq [[input region] input-value-regions
            :let [physical-dtype (:accumulator-dtype region)]]
      (when-not (and (dtype/known? physical-dtype)
                     (= physical-dtype (dtype/canon physical-dtype)))
        (throw (ex-info "matrix input region requires a canonical physical input dtype"
                        {:reason :matrix-stage-input-dtype :input input :dtype physical-dtype})))
      (kernel-body/validate-input-value-region! region physical-dtype operand-dtype))
    stage))

(defn make
  [{:keys [id lhs rhs result dimensions batching reduction result-shape epilogue
           operand-dtype accumulator-dtype result-dtype schedule input-value-regions]
    :or {operand-dtype :half accumulator-dtype :float result-dtype :float input-value-regions {}}}]
  (validate!
   (->MatrixStage id lhs rhs result (vec dimensions) batching reduction (vec result-shape)
                  epilogue (dtype/canon operand-dtype) (dtype/canon accumulator-dtype)
                  (dtype/canon result-dtype) schedule input-value-regions)))

(ns raster.compiler.ir.layout-stage
  "Exact target-neutral representation operations inserted into scheduled kernel graphs."
  (:require [raster.compiler.core.dtype :as dtype]))

(defrecord LayoutStage
           [id operation input output input-shape output-shape input-dtype output-dtype policy])

(defn layout-stage?
  [value]
  (and value
       (= "raster.compiler.ir.layout_stage.LayoutStage" (.getName (class value)))))

(defn validate!
  [stage]
  (when-not (layout-stage? stage)
    (throw (ex-info "expected a LayoutStage"
                    {:reason :layout-stage-type :actual (type stage)})))
  (let [{:keys [id operation input output input-shape output-shape
                input-dtype output-dtype policy]} stage]
    (doseq [[field value] [[:id id] [:input input] [:output output]]]
      (when (nil? value)
        (throw (ex-info "layout stage is missing an identity"
                        {:reason :layout-stage-identity :field field :stage stage}))))
    (when-not (contains? #{:cast :transpose} operation)
      (throw (ex-info "layout stage has an unsupported operation"
                      {:reason :layout-stage-operation :operation operation})))
    (doseq [[field shape] [[:input-shape input-shape] [:output-shape output-shape]]]
      (when-not (and (vector? shape) (seq shape) (not-any? nil? shape))
        (throw (ex-info "layout stage requires an exact logical shape"
                        {:reason :layout-stage-shape :field field :shape shape}))))
    (doseq [[field value] [[:input-dtype input-dtype] [:output-dtype output-dtype]]]
      (when-not (and (dtype/known? value) (= value (dtype/canon value)))
        (throw (ex-info "layout stage requires canonical storage dtypes"
                        {:reason :layout-stage-dtype :field field :dtype value}))))
    (when-not (map? policy)
      (throw (ex-info "layout stage requires an explicit semantic policy"
                      {:reason :layout-stage-policy :policy policy})))
    (case operation
      :cast
      (when-not (every? #(contains? policy %) [:rounding :overflow])
        (throw (ex-info "layout cast requires rounding and overflow semantics"
                        {:reason :layout-stage-policy :policy policy})))

      :transpose
      (when-not (= [1 0] (:permutation policy))
        (throw (ex-info "matrix transpose stage requires the [1 0] permutation"
                        {:reason :layout-stage-policy :policy policy}))))
    stage))

(defn make
  [{:keys [id operation input output input-shape output-shape
           input-dtype output-dtype policy]}]
  (validate!
   (->LayoutStage id operation input output (vec input-shape) (vec output-shape)
                  (dtype/canon input-dtype) (dtype/canon output-dtype) policy)))

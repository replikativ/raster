(ns raster.compiler.passes.parallel.mixed-matrix-candidate
  "Explicit mixed-matrix admission from one retained typed equation and its public graph.

   Planning does not emit source or authorize approximation. The caller must separately certify
   emitted artifacts and numerical consent; unsupported target/layout policies decline as data."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.intel-block-io :as block-io]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.mixed-matrix-schedule :as schedule]
            [raster.compiler.passes.parallel.mixed-matrix-validation :as validation]
            [raster.compiler.passes.parallel.typed-contraction-context :as context]))

(defn target-schedule
  "Select existing Intel matrix hardware facts; other instruction families decline honestly."
  [descriptor requested-tile]
  (let [{:keys [family m n k subgroup]} (:matrix descriptor)]
    (when (and (contains? #{:ze :ocl :opencl} (:backend descriptor))
               (= :dpas family) (= [8 16 16] [m n k]) (= 16 subgroup)
               (contains? (hardware/supported-subgroup-sizes descriptor) (long subgroup)))
      (let [tile (or requested-tile (hardware/gemm-tile-for descriptor))
            maximum (hardware/maximum-workgroup-size descriptor)]
        (when (and (contraction-schedule/matrix-tile-valid? tile)
                   (= (:matrix descriptor) (:matrix tile))
                   (integer? maximum) (pos? maximum))
          (let [workgroup-size (*' (quot (:block-m tile) (:sg-m tile))
                                  (quot (:block-n tile) (:sg-n tile)) subgroup)]
            (when (<= workgroup-size maximum)
              {:tile tile :fill-workgroups (hardware/fill-workgroups descriptor workgroup-size)
               :matrix (:matrix descriptor)})))))))

(defn plan
  "Plan a full-K candidate without replacing the independently derived public boundary.

   The public slice admits uniform int or long dimensions and existing NN/NT tile-input fusion, or
   materialized NN/NT/TN/TT storage. Leading batches use the existing fused-slice schedule."
  [algorithm source descriptor {:keys [precision tile input-fusion]
                                :or {input-fusion :tile-inputs} :as options}]
  (if-not (= :mixed-f16-f32 precision)
    {:ok false :reason :matrix-numerical-policy}
    (if-not (= 1 (count (:nodes source)))
      {:ok false :reason :not-single-plain-fp32-contraction}
      (let [operation (get-in source [:nodes 0 :operation])
            {:keys [facts operation-id dtype]} (context/validate-semantic! algorithm operation)
            view (facts/dense-matrix-view facts)
            target (target-schedule descriptor tile)
            types (into {} (map (juxt :id :dtype)) (:scalars source))
            dimensions (cond-> (:dimensions view) (:batched? view) (conj (:batch view)))]
        (cond
          (not= :float (dtype/canon dtype)) {:ok false :reason :mixed-matrix-source-dtype}
          (not (:ok view)) (assoc (dissoc view :ok) :ok false)
          (nil? target) {:ok false :reason :mixed-matrix-target-capability}
          (not (contains? #{:materialized :tile-inputs} input-fusion))
          {:ok false :reason :mixed-matrix-input-fusion-policy}
          (and (:batched? view) (not= :tile-inputs input-fusion))
          {:ok false :reason :mixed-matrix-batched-fusion-policy}
          (and (= :tile-inputs input-fusion)
               (not (contains? #{:nn :nt} (:variant view))))
          {:ok false :reason :mixed-matrix-input-fusion-layout}
          (not (contains? #{#{:int} #{:long}}
                          (set (map #(launch/typed-expression-dtype % types) dimensions))))
          {:ok false :reason :mixed-matrix-index-width}
          (some #(= :inout (:kind %)) (:abi source))
          {:ok false :reason :mixed-matrix-inout-result}
          :else
          (let [[m n k] (:dimensions view)
                {:keys [row col]} (:bindings view)
                failure (block-io/static-failure m n k)]
            (if failure
              {:ok false :reason :matrix-surface-contract :condition failure}
              (let [spec (cond-> {:id [:typed-contraction operation-id]
                                 :a row :b col :c (:out facts) :m m :n n :k k
                                 :axis-symbols (vec (concat (map first (take-last 2 (:free-axes facts)))
                                                           (map first (:contract-axes facts))))
                                 :variant (:variant view) :epilogue (:epilogue view)
                                 :tile (:tile target) :vector-width 4 :split-k? false
                                 :source-operation operation :source-graph source
                                 :external-interface (select-keys source [:abi :arguments :effects])
                                 :strategy (if (= :tile-inputs input-fusion)
                                             :xmx-direct-tile-inputs :xmx-direct)}
                           (= :tile-inputs input-fusion) (assoc :fuse-tile-inputs? true)
                           (:batched? view) (assoc :batch (:batch view) :batching (:batching view)))
                    planned ((if (:batched? view) schedule/plan-batched schedule/plan) spec)]
                (if-not planned
                  {:ok false :reason :mixed-matrix-input-fusion-declined}
                  (assoc (if (contains? options :scalar-math)
                           (validation/validate-reconstruction! algorithm source (:refinement planned)
                                                                (select-keys options [:scalar-math]))
                           (validation/validate-reconstruction! algorithm source (:refinement planned)))
                         :ok true :target-schedule target))))))))))

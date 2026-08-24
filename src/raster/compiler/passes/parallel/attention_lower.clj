(ns raster.compiler.passes.parallel.attention-lower
  "Lower canonical AttentionProblem semantics into the shared segmented weighted-reduction plan."
  (:require [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- scalar-region
  [parameters body accumulator-dtype]
  (swr/region {:parameters parameters :body body :result-dtype accumulator-dtype}))

(defn- ordered-operands
  [problem]
  (let [specs (attention/buffer-specs problem)]
    (mapv (fn [id]
            (let [{:keys [dtype shape elements]} (get specs id)]
              {:id id :dtype dtype :shape shape :elements elements}))
          (attention/ordered-input-buffer-ids problem))))

(defn lower
  "Produce the canonical attention algebra without selecting a GPU schedule.

   Logical visibility is kept apart from physical paged storage. The scalar regions state stable
   softmax exactly: scaled dot -> exp -> additive numerator/denominator -> divide, with zero for
   an empty visible row."
  [problem]
  (let [{:keys [id query route q-heads kv-heads qk-head-dim value-head-dim
                q-dtype k-dtype v-dtype output-dtype accumulator-dtype scale
                k-format v-format k-layout v-layout visibility output]
         :as problem} (attention/validate! problem)
        specs (attention/buffer-specs problem)
        output-spec (get specs output)
        combine (scalar-region ['left 'right]
                               '(raster.numeric/* left right) accumulator-dtype)
        score-finalize (scalar-region ['dot]
                                      (list 'raster.numeric/* 'dot (double scale))
                                      accumulator-dtype)
        weight (scalar-region ['score] '(raster.math/exp score) accumulator-dtype)
        numerator-map (scalar-region ['weight 'value]
                                     '(raster.numeric/* weight value) accumulator-dtype)
        denominator-map (scalar-region ['weight] 'weight accumulator-dtype)]
    (swr/make
     {:id [:segmented-weighted-reduction id]
      :segment-axes [{:name :query-token :extent (:total-tokens query)}
                     {:name :query-head :extent q-heads}]
      :membership {:kind :logical-attention-visibility
                   :visibility-kind (attention/visibility-kind visibility)
                   :position-filter (into {} (attention/position-filter visibility))
                   :duplicate-policy (when (attention/csr-visibility? visibility)
                                       (:duplicate-policy visibility))
                   :buffers (attention/visibility-buffer-ids visibility)}
      :storage {:kind :routed-paged-kv
                :route-kind (attention/route-kind route)
                :page-size (:page-size problem)
                :physical-pages (:physical-pages problem)
                :route-shape (select-keys route [:pages-per-sequence :page-index-capacity])
                :route route
                :buffers (attention/route-buffer-ids route)
                :k-format k-format :v-format v-format
                :k-layout k-layout :v-layout v-layout}
      :score {:kind :dot
              :axis {:name :qk-component :extent qk-head-dim}
              :head-map {:kind :grouped-query :query-heads q-heads :kv-heads kv-heads}
              :left {:kind :packed-query :buffer (:values query) :dtype q-dtype}
              :right {:kind :routed-key :buffer (:k-pages problem) :dtype k-dtype}
              :combine combine
              :arguments []
              :finalize score-finalize}
      :weight weight
      :value {:kind :routed-value :buffer (:v-pages problem)
              :dtype v-dtype :components value-head-dim}
      :numerator (swr/reduction {:operator :sum :identity 0.0
                                 :map-region numerator-map})
      :denominator (swr/reduction {:operator :sum :identity 0.0
                                   :map-region denominator-map})
      :normalization {:kind :divide :epsilon 0.0 :empty-result 0.0}
      :operands (ordered-operands problem)
      :output {:id output :dtype output-dtype :shape (:shape output-spec)
               :elements (:elements output-spec)}
      :accumulator-dtype accumulator-dtype
      :source-operation problem
      :provenance {:semantic-op :attention :operation-id id
                   :lowering :canonical-segmented-weighted-reduction}})))

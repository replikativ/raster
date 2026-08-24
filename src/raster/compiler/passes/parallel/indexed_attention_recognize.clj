(ns raster.compiler.passes.parallel.indexed-attention-recognize
  "Conservative recognition of compositional indexed graph attention.

   This pass recognizes the exact generic-operator chain currently used by GSDM:

     indexed-dot -> scale-clamp-exp -> scatter-add
                                      -> scatter-mul-add -> segment-div

   It produces the shared SegmentedWeightedReductionPlan without rewriting the source program.
   Every dimension, edge direction, intermediate-use count, clamp bound and normalization epsilon
   must agree. A failed proof returns nil; it never guesses that a similar scatter program is
   attention."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(def ^:private indexed-dot-op 'raster.dl.array-ops/indexed-dot)
(def ^:private scale-clamp-exp-op 'raster.dl.array-ops/scale-clamp-exp)
(def ^:private scatter-add-op 'raster.dl.array-ops/scatter-add)
(def ^:private scatter-mul-add-op 'raster.dl.array-ops/scatter-mul-add)
(def ^:private segment-div-op 'raster.dl.array-ops/segment-div)
(def ^:private casts
  '#{double float long int clojure.core/double clojure.core/float
     clojure.core/long clojure.core/int})
(def ^:private sqrt-ops '#{raster.numeric/sqrt raster.math/sqrt Math/sqrt})

(defn- strip-cast
  [expression]
  (if (and (seq? expression) (= 2 (count expression))
           (contains? casts (first expression)))
    (recur (second expression))
    expression))

(defn- same-value?
  [left right]
  (= (strip-cast left) (strip-cast right)))

(defn- call-args
  [expression operation arity]
  (when (and (seq? expression)
             (= operation (descriptor/semantic-op expression)))
    (let [arguments (vec (descriptor/call-args expression))]
      (when (= arity (count arguments)) arguments))))

(defn- product-factors
  [expression]
  (let [expression (strip-cast expression)]
    (if (and (seq? expression)
             (descriptor/multiplication-op? (descriptor/semantic-op expression)))
      (mapcat product-factors (descriptor/call-args expression))
      [expression])))

(defn- product-of?
  [expression factors]
  (= (frequencies (map strip-cast (product-factors expression)))
     (frequencies (map strip-cast factors))))

(defn- inverse-sqrt-of?
  [expression extent]
  (let [expression (strip-cast expression)]
    (when (and (seq? expression)
               (descriptor/division-op? (descriptor/semantic-op expression)))
      (let [[numerator denominator :as arguments]
            (vec (descriptor/call-args expression))
            denominator (strip-cast denominator)]
        (and (= 2 (count arguments))
             (number? (strip-cast numerator))
             (= 1.0 (double (strip-cast numerator)))
             (seq? denominator)
             (contains? sqrt-ops (descriptor/semantic-op denominator))
             (= 1 (count (descriptor/call-args denominator)))
             (same-value? extent (first (descriptor/call-args denominator))))))))

(defn- product
  [values]
  (let [values (vec (remove #(= 1 %) values))]
    (cond
      (empty? values) 1
      (every? number? values) (reduce * values)
      (= 1 (count values)) (first values)
      :else (cons 'clojure.core/* values))))

(defn- buffer
  [id dtype shape]
  {:id id :dtype dtype :shape (vec shape) :elements (product shape)})

(defn- stable-distinct
  [descriptors]
  (second
   (reduce (fn [[seen result] descriptor]
             (if (contains? seen (:id descriptor))
               [seen result]
               [(conj seen (:id descriptor)) (conj result descriptor)]))
           [#{} []]
           descriptors)))

(defn- symbol-uses
  [expressions symbol]
  (count (filter #(= symbol %)
                 (mapcat #(tree-seq coll? seq %) expressions))))

(defn- match-at
  [form bindings use-expressions output dtype accumulator-dtype]
  (when-let [[weighted-sum denominator-sum n-nodes total-dim n-heads epsilon]
             (call-args (get bindings output) segment-div-op 6)]
    (when-let [[weights values destination-edges source-edges
                weighted-n-dst weighted-n-src n-edges slice-dim
                weighted-total-dim weighted-heads]
               (call-args (get bindings weighted-sum) scatter-mul-add-op 10)]
      (when-let [[denominator-weights denominator-destination denominator-n-dst
                  denominator-edges denominator-heads]
                 (call-args (get bindings denominator-sum) scatter-add-op 5)]
        (when-let [[raw-scores scale clamp-bound score-elements]
                   (call-args (get bindings weights) scale-clamp-exp-op 4)]
          (when-let [[queries keys query-indices key-indices dot-n-a dot-n-b dot-edges
                      dot-slice-dim dot-total-dim dot-heads]
                     (call-args (get bindings raw-scores) indexed-dot-op 10)]
            (let [bound (when (number? (strip-cast clamp-bound))
                          (double (strip-cast clamp-bound)))
                  epsilon (when (number? (strip-cast epsilon))
                            (double (strip-cast epsilon)))
                  internal [raw-scores weights denominator-sum weighted-sum]
                  uses [(symbol-uses use-expressions raw-scores)
                        (symbol-uses use-expressions weights)
                        (symbol-uses use-expressions denominator-sum)
                        (symbol-uses use-expressions weighted-sum)]]
              (when (and (every? symbol? internal)
                         (= (count internal) (count (distinct internal)))
                         (= [1 2 1 1] uses)
                         (same-value? weights denominator-weights)
                         (same-value? destination-edges denominator-destination)
                         (same-value? n-nodes denominator-n-dst)
                         (same-value? n-edges denominator-edges)
                         (same-value? n-heads denominator-heads)
                         (same-value? destination-edges query-indices)
                         (same-value? source-edges key-indices)
                         (every? true?
                                 [(same-value? n-nodes weighted-n-dst)
                                  (same-value? n-nodes weighted-n-src)
                                  (same-value? n-nodes dot-n-a)
                                  (same-value? n-nodes dot-n-b)
                                  (same-value? n-edges dot-edges)
                                  (same-value? slice-dim dot-slice-dim)
                                  (same-value? total-dim weighted-total-dim)
                                  (same-value? total-dim dot-total-dim)
                                  (same-value? n-heads weighted-heads)
                                  (same-value? n-heads dot-heads)])
                         (product-of? score-elements [n-edges n-heads])
                         (inverse-sqrt-of? scale slice-dim)
                         (some? bound) (Double/isFinite bound) (pos? bound)
                         (some? epsilon) (Double/isFinite epsilon) (pos? epsilon))
                (let [combine (swr/region
                               {:parameters ['left 'right]
                                :body '(raster.numeric/* left right)
                                :result-dtype accumulator-dtype})
                      score-finalize
                      (swr/region
                       {:parameters ['dot 'scale 'lower 'upper]
                        :body '(raster.numeric/min
                                upper
                                (raster.numeric/max lower
                                                    (raster.numeric/* dot scale)))
                        :result-dtype accumulator-dtype})
                      weight (swr/region
                              {:parameters ['score]
                               :body '(raster.math/exp score)
                               :result-dtype accumulator-dtype})
                      numerator-map (swr/region
                                     {:parameters ['weight 'value]
                                      :body '(raster.numeric/* weight value)
                                      :result-dtype accumulator-dtype})
                      denominator-map (swr/region
                                       {:parameters ['weight] :body 'weight
                                        :result-dtype accumulator-dtype})
                      value-shape [n-nodes total-dim]
                      edge-shape [n-edges]
                      operands (stable-distinct
                                [(buffer queries dtype value-shape)
                                 (buffer keys dtype value-shape)
                                 (buffer values dtype value-shape)
                                 (buffer destination-edges :long edge-shape)
                                 (buffer source-edges :long edge-shape)])]
                  (swr/make
                   {:id [:segmented-weighted-reduction output]
                    :segment-axes [{:name :destination :extent n-nodes}
                                   {:name :head :extent n-heads}]
                    :membership {:kind :edge-list-by-destination
                                 :destination-indices destination-edges
                                 :source-indices source-edges
                                 :edges n-edges
                                 :duplicate-policy :multiset
                                 :buffers [destination-edges source-edges]}
                    :storage {:kind :indexed-dense-values
                              :entity-count n-nodes :total-dim total-dim
                              :buffers [queries keys values]}
                    :score {:kind :dot
                            :axis {:name :head-component :extent slice-dim}
                            :head-map {:kind :identity :heads n-heads}
                            :left {:kind :indexed-query :buffer queries
                                   :indices destination-edges :dtype dtype
                                   :total-dim total-dim}
                            :right {:kind :indexed-key :buffer keys
                                    :indices source-edges :dtype dtype
                                    :total-dim total-dim}
                            :combine combine
                            :arguments [{:parameter 'scale :kind :inverse-sqrt
                                         :extent slice-dim}
                                        {:parameter 'lower :kind :literal :value (- bound)}
                                        {:parameter 'upper :kind :literal :value bound}]
                            :finalize score-finalize}
                    :weight weight
                    :value {:kind :indexed-value :buffer values
                            :indices source-edges :dtype dtype
                            :entity-count n-nodes :total-dim total-dim
                            :components slice-dim}
                    :numerator (swr/reduction
                                {:operator :sum :identity 0.0
                                 :map-region numerator-map})
                    :denominator (swr/reduction
                                  {:operator :sum :identity 0.0
                                   :map-region denominator-map})
                    :normalization {:kind :divide :epsilon epsilon :empty-result 0.0}
                    :operands operands
                    :output (buffer output dtype value-shape)
                    :accumulator-dtype accumulator-dtype
                    :runtime-parameters [n-nodes n-edges total-dim n-heads slice-dim]
                    :source-operation {:form form :output output
                                       :intermediates internal}
                    :provenance {:semantic-op :indexed-graph-attention
                                 :operation-id output
                                 :lowering :recognized-indexed-attention-chain}}))))))))))

(defn recognize
  "Return every proven indexed-attention chain in a closed-core let* form, in binding order.
   `dtype` describes Q/K/V/output storage; `accumulator-dtype` describes scalar reductions."
  [form & {:keys [dtype accumulator-dtype]
           :or {dtype :double accumulator-dtype :double}}]
  (let [dtype (dtype/canon dtype)
        accumulator-dtype (dtype/canon accumulator-dtype)]
    (if-not (and (seq? form) (= 'let* (first form)) (vector? (second form)))
      []
      (let [[_ binding-vector & body] form
            pairs (mapv vec (partition 2 binding-vector))
            bindings (into {} pairs)
            use-expressions (vec (concat (map second pairs) body))]
        (vec (keep (fn [[output expression]]
                     (when (= segment-div-op (descriptor/semantic-op expression))
                       (match-at form bindings use-expressions output dtype accumulator-dtype)))
                   pairs))))))

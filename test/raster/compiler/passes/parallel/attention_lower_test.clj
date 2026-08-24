(ns raster.compiler.passes.parallel.attention-lower-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.attention-lower :as lower]
            [raster.compiler.passes.parallel.attention-route :as route]))

(defn- query
  [& [prefix]]
  (let [prefix (or prefix "")]
    (attention/packed-query-batch
     {:values (symbol (str prefix "q"))
      :row-offsets (symbol (str prefix "q-rows"))
      :positions (symbol (str prefix "q-positions"))
      :total-tokens 5})))

(defn- dense-route
  [& [prefix]]
  (let [prefix (or prefix "")]
    (attention/dense-paged-route
     {:page-table (symbol (str prefix "page-table"))
      :lengths (symbol (str prefix "kv-lengths"))
      :start-positions (symbol (str prefix "kv-starts"))
      :pages-per-sequence 3})))

(defn- csr-route
  [& [prefix]]
  (let [prefix (or prefix "")]
    (attention/csr-paged-route
     {:page-offsets (symbol (str prefix "page-offsets"))
      :page-indices (symbol (str prefix "page-indices"))
      :last-page-lengths (symbol (str prefix "last-page-lengths"))
      :start-positions (symbol (str prefix "kv-starts"))
      :page-index-capacity 6})))

(defn- problem
  [& {:keys [route visibility prefix]
      :or {prefix ""}}]
  (attention/make
   {:id :attention
    :query (query prefix)
    :k-pages (symbol (str prefix "k-pages"))
    :v-pages (symbol (str prefix "v-pages"))
    :route (or route (dense-route prefix))
    :output (symbol (str prefix "output"))
    :batch-size 2 :q-heads 4 :kv-heads 2
    :qk-head-dim 8 :value-head-dim 6
    :page-size 2 :physical-pages 7 :scale 0.25
    :k-layout :kv-head-major :v-layout :page-major
    :visibility (or visibility
                    (attention/visibility
                     {:causal? true :window-left 4 :window-right 0}))}))

(defn- reason
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo e
      (:reason (ex-data e)))))

(deftest canonical-attention-lowers-to-explicit-weighted-reduction-algebra
  (let [problem (problem)
        plan (lower/lower problem)]
    (is (swr/plan? plan))
    (is (= [{:name :query-token :extent 5}
            {:name :query-head :extent 4}]
           (:segment-axes plan)))
    (is (= :logical-attention-visibility (get-in plan [:membership :kind])))
    (is (= :routed-paged-kv (get-in plan [:storage :kind])))
    (is (= :dense-paged (get-in plan [:storage :route-kind])))
    (is (= {:kind :grouped-query :query-heads 4 :kv-heads 2}
           (get-in plan [:score :head-map])))
    (is (= '(raster.numeric/* dot 0.25)
           (get-in plan [:score :finalize :body])))
    (is (= '(raster.math/exp score) (get-in plan [:weight :body])))
    (is (= '(raster.numeric/* weight value)
           (get-in plan [:numerator :map-region :body])))
    (is (= 'weight (get-in plan [:denominator :map-region :body])))
    (is (= {:kind :divide :epsilon 0.0 :empty-result 0.0}
           (:normalization plan)))
    (is (= (attention/ordered-input-buffer-ids problem)
           (swr/ordered-input-ids plan)))
    (is (= :attention (get-in plan [:provenance :semantic-op])))
    (is (= problem (:source-operation plan)))))

(deftest logical-algebra-is-independent-of-physical-page-routing
  (let [dense (lower/lower (problem :route (dense-route)))
        csr (lower/lower (problem :route (csr-route)))]
    (is (= (swr/algebra-key dense) (swr/algebra-key csr)))
    (is (not= (swr/schedule-key dense) (swr/schedule-key csr)))
    (is (= :dense-paged (get-in dense [:storage :route-kind])))
    (is (= :csr-paged (get-in csr [:storage :route-kind])))))

(deftest logical-visibility-remains-part-of-the-algebra
  (let [causal (lower/lower (problem))
        bidirectional (lower/lower
                       (problem :visibility (attention/visibility
                                             {:causal? false
                                              :window-left nil :window-right nil})))]
    (is (not= (swr/algebra-key causal) (swr/algebra-key bidirectional)))))

(deftest buffer-renaming-does-not-change-static-plan-identity
  (let [a (lower/lower (problem :prefix "a-"))
        b (lower/lower (problem :prefix "b-"))]
    (is (= (swr/algebra-key a) (swr/algebra-key b)))
    (is (= (swr/schedule-key a) (swr/schedule-key b)))
    (is (not= (swr/ordered-input-ids a) (swr/ordered-input-ids b)))))

(deftest scalar-plan-legality-fails-closed
  (let [plan (lower/lower (problem))]
    (testing "unknown scalar calls cannot be reordered or fused"
      (is (= :segmented-reduction-region-free-symbol
             (reason #(swr/validate!
                       (assoc-in plan [:weight :body] '(unknown-score-op score)))))))
    (testing "positional numerator contracts are checked"
      (is (= :segmented-reduction-map-arity
             (reason #(swr/validate!
                       (assoc-in plan [:numerator :map-region :parameters]
                                 ['value 'weight]))))))
    (testing "normalization is exact plan semantics, not an emitter default"
      (is (= :segmented-reduction-invalid-normalization
             (reason #(swr/validate!
                       (assoc-in plan [:normalization :epsilon] -1.0))))))))

(deftest executable-attention-routing-consumes-the-shared-plan
  (let [{:keys [plan artifact graph]} (route/route! (problem))]
    (is (swr/plan? plan))
    (is (= (:id plan) (get-in artifact [:provenance :algebra-plan-id])))
    (is (= :segmented-weighted-reduction
           (get-in artifact [:attributes :algebra])))
    (is (= (swr/algebra-key plan)
           (get-in artifact [:attributes :algebra-key])))
    (is (= (swr/ordered-input-ids plan)
           (vec (butlast (:arguments artifact)))))
    (is (= :attention (get-in graph [:provenance :semantic-op])))))

(deftest reference-emission-cannot-ignore-a-changed-algebra
  (let [plan (lower/lower (problem))
        changed (assoc-in plan [:normalization :epsilon] 1.0e-6)]
    (is (not (swr/online-softmax-algebra? changed)))
    (is (= :attention-reference-plan-unsupported
           (reason #((requiring-resolve
                      'raster.compiler.backend.gpu.attention/emit-fp16-reference)
                     changed nil))))))

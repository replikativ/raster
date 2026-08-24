(ns raster.compiler.passes.parallel.indexed-attention-recognize-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.indexed-attention-recognize :as recognize]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.reference.segmented-weighted-reduction :as reference]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.gsdm :as gsdm]))

(defn- chain
  [& {:keys [denominator-destination scale epsilon extra-raw-use?]
      :or {denominator-destination 'dst
           scale '(raster.numeric// 1.0 (raster.numeric/sqrt dk))
           epsilon 1.0e-6
           extra-raw-use? false}}]
  (let [bindings
        ['raw '(raster.dl.array-ops/indexed-dot
                Q K dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
         'weights (list 'raster.dl.array-ops/scale-clamp-exp
                        'raw scale 5.0 '(clojure.core/* n-edges n-heads))
         'denominator (list 'raster.dl.array-ops/scatter-add
                            'weights denominator-destination
                            'n-nodes 'n-edges 'n-heads)
         'weighted '(raster.dl.array-ops/scatter-mul-add
                     weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
         'normalized (list 'raster.dl.array-ops/segment-div
                           'weighted 'denominator 'n-nodes 'emb-dim 'n-heads epsilon)]]
    (list 'let*
          (cond-> bindings extra-raw-use? (into ['raw-copy 'raw]))
          'normalized)))

(defn- reason
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo e
      (:reason (ex-data e)))))

(deftest exact-indexed-chain-lowers-to-the-shared-plan
  (let [plans (recognize/recognize (chain))
        plan (first plans)]
    (is (= 1 (count plans)))
    (is (swr/plan? plan))
    (is (= [{:name :destination :extent 'n-nodes}
            {:name :head :extent 'n-heads}]
           (:segment-axes plan)))
    (is (= {:kind :edge-list-by-destination
            :destination-indices 'dst :source-indices 'src
            :edges 'n-edges :duplicate-policy :multiset
            :buffers '[dst src]}
           (:membership plan)))
    (is (= :indexed-dense-values (get-in plan [:storage :kind])))
    (is (= '(raster.numeric/min
             upper
             (raster.numeric/max lower (raster.numeric/* dot scale)))
           (get-in plan [:score :finalize :body])))
    (is (= [{:parameter 'scale :kind :inverse-sqrt :extent 'dk}
            {:parameter 'lower :kind :literal :value -5.0}
            {:parameter 'upper :kind :literal :value 5.0}]
           (get-in plan [:score :arguments])))
    (is (= {:kind :divide :epsilon 1.0e-6 :empty-result 0.0}
           (:normalization plan)))
    (is (= '[Q K V dst src] (swr/ordered-input-ids plan)))
    (is (= '(clojure.core/* n-nodes emb-dim)
           (get-in plan [:output :elements])))
    (is (false? (swr/online-softmax-algebra? plan))
        "clamp and epsilon must not enter the canonical attention emitter")))

(deftest actual-walked-gsdm-body-is-recognized-by-semantic-call-identity
  (let [walked (first (pipeline/get-walked-body
                       #'gsdm/graph-attention-multihead :double))
        plans (recognize/recognize walked)
        plan (first plans)]
    (is (= 1 (count plans)))
    (is (= :indexed-graph-attention (get-in plan [:provenance :semantic-op])))
    (is (= :recognized-indexed-attention-chain
           (get-in plan [:provenance :lowering])))
    (is (= 'dst-edges (get-in plan [:membership :destination-indices])))
    (is (= 'src-edges (get-in plan [:membership :source-indices])))
    (is (= 1.0e-6 (get-in plan [:normalization :epsilon])))))

(deftest recognition-declines-any-unproven-chain
  (testing "an extra consumer prevents elimination of the raw-score intermediate"
    (is (empty? (recognize/recognize (chain :extra-raw-use? true)))))
  (testing "normalizer and weighted scatter must segment by the same destination"
    (is (empty? (recognize/recognize
                 (chain :denominator-destination 'other-destination)))))
  (testing "the score scale must be the exact inverse square root of the head slice"
    (is (empty? (recognize/recognize (chain :scale 0.5)))))
  (testing "zero epsilon has different empty-row behavior and is not this GSDM contract"
    (is (empty? (recognize/recognize (chain :epsilon 0.0)))))
  (is (empty? (recognize/recognize '(let* [x 1] x)))))

(deftest symbolic-plan-shapes-remain-closed-and-checkable
  (let [plan (first (recognize/recognize (chain)))]
    (is (= :segmented-reduction-invalid-segment-axis
           (reason #(swr/validate!
                     (assoc-in plan [:segment-axes 0 :extent]
                               '(do (launch-side-effect) n-nodes))))))
    (is (= :segmented-reduction-invalid-buffer-descriptor
           (reason #(swr/validate!
                     (assoc-in plan [:output :elements]
                               '(clojure.core/+ n-nodes emb-dim))))))))

(deftest plan-reference-is-bit-identical-to-the-compositional-operators
  (let [plan (first (recognize/recognize (chain)))
        n-nodes 3
        n-edges 4
        emb-dim 5
        n-heads 2
        dk 2
        Q (double-array [1 2 3 4 99, 2 1 0 -1 88, -100 100 80 -80 77])
        K (double-array [0 1 1 0 66, 1 1 2 -1 55, 100 -100 -90 90 44])
        V (double-array [1 2 3 4 33, 2 4 6 8 22, -1 1 -2 2 11])
        dst (long-array [0 0 2 2])
        src (long-array [1 1 0 2])
        raw (array-ops/indexed-dot Q K dst src
                                   n-nodes n-nodes n-edges dk emb-dim n-heads)
        weights (array-ops/scale-clamp-exp
                 raw (/ 1.0 (Math/sqrt dk)) 5.0 (* n-edges n-heads))
        denominator (array-ops/scatter-add weights dst n-nodes n-edges n-heads)
        weighted (array-ops/scatter-mul-add
                  weights V dst src n-nodes n-nodes n-edges dk emb-dim n-heads)
        expected (array-ops/segment-div
                  weighted denominator n-nodes emb-dim n-heads 1.0e-6)
        inputs {:buffers {'Q Q 'K K 'V V 'dst dst 'src src}
                :scalars {'n-nodes n-nodes 'n-edges n-edges
                          'emb-dim emb-dim 'n-heads n-heads 'dk dk}}
        actual (reference/evaluate plan inputs)]
    (is (= (vec expected) (vec actual)))
    (is (= [0.0 0.0 0.0 0.0 0.0]
           (subvec (vec actual) emb-dim (* 2 emb-dim)))
        "an empty destination row stays zero")
    (is (= 0.0 (aget actual (dec emb-dim)))
        "the non-head tail preserves the compositional zero initialization")
    (is (= :segmented-reduction-missing-scalar
           (reason #(reference/evaluate plan
                                        (update inputs :scalars dissoc 'dk)))))))

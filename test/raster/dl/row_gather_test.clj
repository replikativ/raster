(ns raster.dl.row-gather-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]))

(deftm greedy-embedding-tail!
  [logits :- (Array float), token-indices :- (Array int),
   table :- (Array float), rows :- (Array float),
   nrows :- Long, vocab :- Long, width :- Long] :- Void
  (ops/argmax-rows! logits token-indices nrows vocab)
  (ops/gather-rows! table token-indices rows nrows width))

(deftm block-transfer-roundtrip!
  [src :- (Array float), block-indices :- (Array int), paged :- (Array float),
   restored :- (Array float), nblocks :- Long, block-width :- Long,
   paged-blocks :- Long] :- Void
  (ops/scatter-blocks! src block-indices paged nblocks block-width paged-blocks)
  (ops/gather-blocks! paged block-indices restored nblocks block-width paged-blocks))

(deftm block-transfer-half-roundtrip!
  [src :- (Array short), block-indices :- (Array int), paged :- (Array short),
   restored :- (Array short), nblocks :- Long, block-width :- Long,
   paged-blocks :- Long] :- Void
  (ops/scatter-blocks! src block-indices paged nblocks block-width paged-blocks)
  (ops/gather-blocks! paged block-indices restored nblocks block-width paged-blocks))

(deftest gather-rows-test
  (let [width 4
        table (float-array [0 1 2 3
                            10 11 12 13
                            20 21 22 23
                            30 31 32 33])
        indices (int-array [2 0 2])
        rows (float-array (* 3 width))]
    (ops/gather-rows! table indices rows 3 width)
    (testing "rows use shared source storage and may repeat"
      (is (= [20.0 21.0 22.0 23.0
              0.0 1.0 2.0 3.0
              20.0 21.0 22.0 23.0]
             (vec rows))))))

(deftest generic-block-gather-and-scatter-test
  (let [width 3
        nblocks 3
        indexed-blocks 6
        indices (int-array [4 1 5])
        dense (float-array [10 11 12, 20 21 22, 30 31 32])
        paged (float-array (* indexed-blocks width) (float -1.0))
        restored (float-array (* nblocks width))]
    (ops/validate-block-transfer! :scatter dense indices paged
                                  nblocks width indexed-blocks)
    (ops/scatter-blocks! dense indices paged nblocks width indexed-blocks)
    (ops/validate-block-transfer! :gather paged indices restored
                                  nblocks width indexed-blocks)
    (ops/gather-blocks! paged indices restored nblocks width indexed-blocks)
    (testing "scatter changes only unique selected destination blocks"
      (is (= [-1.0 -1.0 -1.0, 20.0 21.0 22.0,
              -1.0 -1.0 -1.0, -1.0 -1.0 -1.0,
              10.0 11.0 12.0, 30.0 31.0 32.0]
             (vec paged))))
    (testing "gather is the inverse for the same block route"
      (is (= (vec dense) (vec restored))))))

(deftest block-index-contract-test
  (let [repeated (int-array [2 0 2])]
    (is (identical? repeated
                    (ops/validate-block-indices! repeated 3 4 :allow)))
    (is (= :block-index-duplicate-destination
           (try
             (ops/validate-block-indices! repeated 3 4 :unique)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :block-index-out-of-bounds
           (try
             (ops/validate-block-indices! (int-array [0 4]) 2 4 :unique)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :block-index-count
           (try
             (ops/validate-block-indices! (int-array [0]) 2 4 :unique)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :block-transfer-buffer-size
           (try
             (ops/validate-block-transfer! :scatter
                                           (float-array 5) (int-array [0 1]) (float-array 12)
                                           2 3 4)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [same (float-array 12)]
      (is (= :block-transfer-overlap
             (try
               (ops/validate-block-transfer! :gather
                                             same (int-array [0 1]) same 2 3 4)
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest block-transfer-resident-lowering-test
  (let [descriptors (into {}
                          (map (fn [target]
                                 [target (pipeline/compile-gpu-program
                                          #'block-transfer-roundtrip! target :dtype :float)]))
                          [:ocl:0 :ze:0])
        descriptor (get descriptors :ocl:0)
        gather-descriptor (pipeline/compile-gpu-program
                           #'ops/gather-blocks! :ocl:0 :dtype :float)]
    (testing "both directions are ordinary allocation-free SOAC maps"
      (is (= [:map-void :map-void] (mapv :convention (:steps descriptor))))
      (is (= [:segmap :segmap]
             (mapv #(get-in % [:artifact :provenance :dialect]) (:steps descriptor)))
          "scatter and gather share the typed scheduled-map vertical")
      (is (= [true nil]
             (mapv #(get-in % [:artifact :attributes :explicit-stores])
                   (:steps descriptor)))
          "scatter owns explicit indexed stores; gather retains the implicit dense result")
      (is (= [:segmap]
             (mapv #(get-in % [:artifact :provenance :dialect])
                   (:steps gather-descriptor)))
          "bounded block gather independently enters TypedSOAC as an inout map")
      (is (empty? (:allocs descriptor)))
      (is (= '[src block-indices paged restored nblocks block-width paged-blocks]
             (:all-params descriptor)))
      (is (= #{'paged 'restored}
             (set (for [step (:steps descriptor)
                        slot (:abi step)
                        :when (contains? #{:output :inout} (:kind slot))]
                    (:name slot))))))
    (testing "OpenCL and Level Zero preserve the same physical ABI"
      (is (apply = (map (fn [[_ compiled]]
                          (mapv (comp #(mapv (juxt :kind :dtype :kernel-dtype) %) :abi)
                                (:steps compiled)))
                        descriptors))))
    (testing "FP16 storage is selected from short-array types, independently of scalar arithmetic"
      (doseq [target [:ocl:0 :ze:0]]
        (let [compiled (pipeline/compile-gpu-program
                        #'block-transfer-half-roundtrip! target :dtype :float)]
          (is (= [:map-void :map-void] (mapv :convention (:steps compiled))))
          (is (= #{:int :half}
                 (set (mapcat #(map :dtype (:abi %)) (:steps compiled))))))))))

(deftest gather-rows-resident-lowering-test
  (let [descriptors (into {}
                          (map (fn [target]
                                 [target (pipeline/compile-gpu-program
                                          #'ops/gather-rows! target :dtype :float)]))
                          [:ocl:0 :ze:0])
        descriptor (get descriptors :ocl:0)
        step (first (:steps descriptor))]
    (testing "the ABI is generic storage plus shape, without embedding policy"
      (is (= '[src indices out nrows width] (:all-params descriptor)))
      (is (= '[src indices out] (:array-params descriptor)))
      (is (= '[nrows width] (:scalar-params descriptor)))
      (is (empty? (:allocs descriptor))))
    (testing "every output element lowers through one target-neutral map"
      (is (= [:map-void] (mapv :convention (:steps descriptor))))
      (is (= :segmap (get-in step [:artifact :provenance :dialect])))
      (is (= '[indices src out width _n_bound] (mapv :name (:abi step))))
      (is (= [:int :float :float :int :int] (mapv :dtype (:abi step))))
      (is (= 'rstr_extent_0 (last (get-in step [:artifact :arguments])))
          "the schedule names the typed scalar extent instead of embedding host code")
      (is (= 15 ((:value-fn (last (:argument-specs step)))
                 [nil nil nil 3 5]))
          "the resident binder evaluates the scalar SSA definition from external arguments"))
    (testing "OpenCL and Level Zero preserve the same physical ABI"
      (is (apply = (map (comp #(mapv (juxt :kind :dtype :kernel-dtype) %)
                              :abi first :steps val)
                        descriptors))))))

(deftest indexed-reduction-composes-with-row-gather-test
  (let [nrows 2
        vocab 513
        width 3
        logits (float-array (* nrows vocab) (float -100.0))
        token-indices (int-array nrows)
        table (float-array (map float (range (* vocab width))))
        rows (float-array (* nrows width))
        arguments [logits token-indices table rows nrows vocab width]
        descriptor (pipeline/compile-gpu-program
                    #'greedy-embedding-tail! :ocl:0 :dtype :float)
        lowering (resident-plan/lower
                  {:id :greedy-embedding-tail
                   :target :ocl:0
                   :descriptor descriptor
                   :arguments arguments
                   :outputs ['token-indices 'rows]})]
    (aset logits 7 (float 9.0))
    (aset logits (+ vocab 400) (float 8.0))
    (greedy-embedding-tail! logits token-indices table rows nrows vocab width)
    (testing "selection and lookup compose without a fused attention primitive"
      (is (= [7 400] (vec token-indices)))
      (is (= [21.0 22.0 23.0 1200.0 1201.0 1202.0] (vec rows))))
    (testing "the compiled graph retains one scheduled reduction and one independent row gather"
      (is (= [:map-void :map-void] (mapv :convention (:steps descriptor))))
      (is (empty? (:allocs descriptor)))
      (is (= :segmented-workgroup-tree
             (get-in descriptor [:steps 0 :artifact :attributes :schedule :strategy]))))
    (testing "the ordinary multi-output descriptor certifies before allocation"
      (is (resident-plan/certified-plan? (resident-plan/verify! lowering))))))

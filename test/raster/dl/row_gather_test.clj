(ns raster.dl.row-gather-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
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
      (is (= '[indices out src width _n_bound] (mapv :name (:abi step))))
      (is (= [:int :float :float :int :int] (mapv :dtype (:abi step))))
      (is (= '(clojure.core/* (long nrows) (long width))
             (last (get-in step [:artifact :arguments])))))
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
    (testing "the compiled graph retains three ordered kernels and only reduction scratch"
      (is (= [:map-void :map-void :map-void] (mapv :convention (:steps descriptor))))
      (is (= [true true]
             (mapv (fn [{:keys [sym]} prefix] (str/starts-with? (name sym) prefix))
                   (:allocs descriptor) ["partial-values" "partial-indices"])))
      (is (= [:float :int] (mapv :dtype (:allocs descriptor)))))
    (testing "the ordinary multi-output descriptor certifies before allocation"
      (is (resident-plan/certified-plan? (resident-plan/verify! lowering))))))

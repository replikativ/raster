(ns raster.compiler.passes.parallel.product-consumer-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.product-consumer-region :as region]
            [raster.core :refer [deftm]]
            [raster.par]))

(deftm projected-product-consumer!
  [input :- (Array int), weights :- (Array int), output :- (Array int), rows :- Long] :- Void
  (let [partials (int-array (* rows 8))]
    (raster.par/product-reduce!
     [partials]
     [[sum 0 :int]]
     [[row rows] [lane 8]]
     chunk 8
     [value (unchecked-add-int
             (raster.arrays/aget input (+ (* (+ (* row 8) lane) 8) chunk))
             (raster.arrays/aget weights (+ (* chunk 8) lane)))]
     [value]
     [[left right]]
     []
     [(unchecked-add-int left right)]
     {:associative? true :commutative? true
      :overflow :wrap :order :implementation-defined})
    (raster.par/map-void!
     row rows
     (let [base (* row 8)
           total (loop [lane 0 left 0 right 0]
                   (if (< lane 8)
                     (let [value (raster.arrays/aget partials (+ base lane))]
                       (recur (inc lane)
                              (unchecked-add-int left value)
                              (unchecked-add-int right value)))
                     (unchecked-add-int left right)))]
       (raster.arrays/aset output row total)))))

(deftm projected-product-pair-consumer!
  [input :- (Array int), weights :- (Array int), output :- (Array int), rows :- Long] :- Void
  (let [left-partials (int-array (* rows 8))
        right-partials (int-array (* rows 8))]
    (raster.par/product-reduce!
     [left-partials right-partials]
     [[left-sum 0 :int] [right-sum 0 :int]]
     [[row rows] [lane 8]]
     chunk 8
     [left-value (raster.arrays/aget input (+ (* (+ (* row 8) lane) 8) chunk))
      right-value (raster.arrays/aget weights (+ (* chunk 8) lane))]
     [left-value right-value]
     [[left-a left-b] [right-a right-b]]
     []
     [(unchecked-add-int left-a left-b)
      (unchecked-add-int right-a right-b)]
     {:associative? true :commutative? true
      :overflow :wrap :order :implementation-defined})
    (raster.par/map-void!
     row rows
     (let [base (* row 8)
           total (loop [lane 0 left 0 right 0]
                   (if (< lane 8)
                     (recur (inc lane)
                            (unchecked-add-int
                             left (raster.arrays/aget left-partials (+ base lane)))
                            (unchecked-add-int
                             right (raster.arrays/aget right-partials (+ base lane))))
                     (unchecked-add-int left right)))]
       (raster.arrays/aset output row total)))))

(defn- scheduled-product-consumer []
  (:segop-lowered
   (pipeline/show-pipeline
    #'projected-product-consumer!
    :target-device :ocl:0 :dtype :int
    :values {'input (av/tensor {:dtype :int :shape [64]})
             'weights (av/tensor {:dtype :int :shape [64]})
             'output (av/tensor {:dtype :int :shape [1]})})))

(defn- scheduled-product-pair-consumer []
  (:segop-lowered
   (pipeline/show-pipeline
    #'projected-product-pair-consumer!
    :target-device :ocl:0 :dtype :int
    :values {'input (av/tensor {:dtype :int :shape [64]})
             'weights (av/tensor {:dtype :int :shape [64]})
             'output (av/tensor {:dtype :int :shape [1]})})))

(defn- numerical-equations [program]
  (filterv (comp seq :operations) (:equations program)))

(deftest exact-product-consumer-region-retains-its-axis-and-numerical-contracts
  (let [program (scheduled-product-consumer)
        plan (region/analyze program (numerical-equations program))]
    (is (= :product-ordered-consumer (:kind plan)))
    (is (= [2 3] (:equations plan)))
    (is (= 'partials (:intermediate plan)))
    (is (= ['row] (mapv :name (get-in plan [:axes :prefix]))))
    (is (= ['row] (get-in plan [:axes :prefix-binding])))
    (is (= {:name 'lane :bound 8} (get-in plan [:axes :ordered])))
    (is (empty? (get-in plan [:axes :local])))
    (is (= {:name 'chunk :bound 8} (get-in plan [:axes :reduced])))
    (is (= [0] (mapv :local-offset (:intermediate-loads plan))))
    (is (= 8 (:workgroup-size plan)))
    (is (= :implementation-defined (get-in plan [:numerics :inner :association])))
    (is (= :ordered (get-in plan [:numerics :outer :association])))
    (is (= #{'partials}
           (set (map :id (get-in plan [:source :graph :temporaries])))))))

(deftest unrelated-equations-decline-without-authorizing-a-fallback-fusion
  (let [program (scheduled-product-consumer)
        equations (numerical-equations program)]
    (try
      (region/analyze program [(second equations)])
      (is false "a singleton cannot masquerade as a fused region")
      (catch clojure.lang.ExceptionInfo exception
        (is (region/declined? exception))
        (is (= :equation-count (:missing-rule (ex-data exception))))))))

(deftest ordered-private-products-retain-component-to-storage-correspondence
  (let [program (scheduled-product-pair-consumer)
        plan (region/analyze program (numerical-equations program))]
    (is (= '[left-partials right-partials] (:intermediates plan)))
    (is (nil? (:intermediate plan)) "the scalar compatibility projection is intentionally absent")
    (is (= [[0 'left-partials 0] [1 'right-partials 0]]
           (mapv (juxt :component :intermediate :local-offset)
                 (:intermediate-loads plan))))
    (is (= 2 (count (:subgroup-collectives plan))))
    (is (= [:int :int] (mapv :dtype (:subgroup-collectives plan))))))

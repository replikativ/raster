(ns raster.dl.indexed-reduction-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.array-ops :as ops]))

(defn- reduction-rows
  [nrows width]
  (float-array
   (for [row (range nrows)
         col (range width)]
     (float
      (case row
        0 (if (contains? #{3 256 512} col) 11.0 (- -1000.0 col))
        1 (if (contains? #{255 257} col) Float/NaN (if (= col 300) 100.0 col))
        2 (- (Math/abs (- col 411)))
        3 (if (contains? #{10 400} col) Float/POSITIVE_INFINITY (- col)))))))

(deftest deterministic-argmax-rows-test
  (let [nrows 4
        width 600
        values (reduction-rows nrows width)
        indices (int-array nrows)]
    (ops/argmax-rows! values indices nrows width)
    (testing "ties cross tile boundaries and choose the first column"
      (is (= 3 (aget indices 0))))
    (testing "NaN outranks numeric values and chooses the first NaN"
      (is (= 255 (aget indices 1))))
    (testing "ordinary negative and infinite rows retain deterministic ordering"
      (is (= [411 10] (subvec (vec indices) 2))))))

(deftest argmax-rows-resident-lowering-test
  (let [nrows 4
        width 600
        values (reduction-rows nrows width)
        indices (int-array nrows)
        arguments [values indices nrows width]
        descriptors (into {}
                          (map (fn [target]
                                 [target (pipeline/compile-gpu-program
                                          #'ops/argmax-rows! target :dtype :float)]))
                          [:ocl:0 :ze:0])
        report (pipeline/compile-report
                #'ops/argmax-rows! :target-device :ocl:0 :dtype :float)
        descriptor (get descriptors :ocl:0)
        allocs (:allocs descriptor)
        steps (:steps descriptor)
        sources (mapv #(get-in % [:artifact :source]) steps)
        lowering (resident-plan/lower
                  {:id :indexed-row-reduction
                   :target :ocl:0
                   :descriptor descriptor
                   :arguments arguments
                   :outputs ['indices]})]
    (testing "the semantic ABI contains no schedule scratch"
      (is (= '[values indices nrows width] (:all-params descriptor)))
      (is (= '[values indices] (:array-params descriptor)))
      (is (= '[nrows width] (:scalar-params descriptor))))
    (testing "schedule scratch is local to the emitted product kernel"
      (is (empty? allocs))
      (is (= :segmented-workgroup-tree
             (get-in steps [0 :artifact :attributes :schedule :strategy]))))
    (testing "the typed product reduction is one resident scheduled step"
      (is (= {:backend :opencl
              :source-dialect :typed-soac
              :typed-validated true
              :declines []}
             (:route report)))
      (is (= 1 (get-in report [:lowering :typed-reused])))
      (is (zero? (get-in report [:lowering :backend-relowered])))
      (is (zero? (get-in report [:lowering :fallback])))
      (is (= [:map-void] (mapv :convention steps)))
      (is (= 1 (count steps)))
      (is (= ['values 'indices]
             (vec (take 2 (get-in steps [0 :artifact :arguments])))))
      (is (= [:float :int]
             (get-in steps [0 :artifact :attributes :component-dtypes]))))
    (testing "OpenCL and Level Zero preserve the same typed lowering"
      (is (apply = (map (comp #(mapv :dtype %) :allocs val) descriptors)))
      (is (apply = (map (comp #(mapv :convention %) :steps val) descriptors)))
      (is (apply = (map (comp #(mapv (fn [step]
                                       (mapv (juxt :kind :dtype :kernel-dtype) (:abi step))) %)
                              :steps val)
                        descriptors))))
    (testing "OpenCL C preserves mixed local products and portable NaN comparison"
      (is (str/includes? (first sources) "__local float shared_0"))
      (is (str/includes? (first sources) "__local int shared_1"))
      (is (str/includes? (first sources) "for (int col = lid"))
      (is (str/includes? (first sources) "barrier(CLK_LOCAL_MEM_FENCE)"))
      (is (every? #(not (str/includes? % "boolean(")) sources))
      (is (every? #(str/includes? % "(float)(elem_0) == (float)(elem_0)") sources)))
    (testing "the descriptor certifies as a composable LinkPlan before allocation"
      (is (resident-plan/certified-plan? (resident-plan/verify! lowering))))))

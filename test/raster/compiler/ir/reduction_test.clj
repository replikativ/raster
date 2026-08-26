(ns raster.compiler.ir.reduction-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.par]))

(def argmax-product-form
  '(raster.par/product-reduce!
    [nil indices]
    [[best-value (float Float/NEGATIVE_INFINITY) :float]
     [best-index (int Integer/MAX_VALUE) :int]]
    [[row nrows]]
    col width
    [candidate (clojure.core/aget values (+ (* row width) col))]
    [candidate (int col)]
    [[left-value right-value] [left-index right-index]]
    [left-nan (int (if (== left-value left-value) 0 1))
     right-nan (int (if (== right-value right-value) 0 1))
     better (int (if (== right-nan 1)
                   (if (== left-nan 1) (if (< right-index left-index) 1 0) 1)
                   (if (== left-nan 1)
                     0
                     (if (> right-value left-value)
                       1
                       (if (== right-value left-value)
                         (if (< right-index left-index) 1 0) 0)))))]
    [(if (== better 1) right-value left-value)
     (if (== better 1) right-index left-index)]
    {:associative? true :commutative? true
     :order {:nan :highest :tie :lowest-index}}))

(deftest canonical-scalar-is-one-component-product-test
  (let [operator (reduction/scalar
                  {:accumulator 'acc :neutral 0.0 :dtype :double :result 'sum
                   :index 'i :step-result '(+ acc (aget values i))})]
    (is (reduction/product-reduction? operator))
    (is (reduction/scalar? operator))
    (is (= [:double] (reduction/dtypes operator)))
    (is (= {:acc 'acc :init 0.0 :lambda '(+ acc (aget values i))}
           (reduction/scalar-op operator)))))

(deftest typed-product-validation-test
  (let [node (soac/par-form->soac '_ argmax-product-form 7 :dtype :float)
        operator (:reduction node)]
    (testing "ordered logical components carry independent dtypes and optional results"
      (is (reduction/product-reduction? operator))
      (is (= [:float :int] (reduction/dtypes operator)))
      (is (= [nil 'indices] (reduction/results operator)))
      (is (= ['best-value 'best-index] (reduction/accumulators operator))))
    (testing "element and closed binary combine regions are separate"
      (is (= '[candidate (clojure.core/aget values (+ (* row width) col))]
             (:bindings (reduction/element-region operator))))
      (is (= '[[left-value right-value] [left-index right-index]]
             (:parameters (reduction/combine-region operator))))
      (is (= true (get-in operator [:algebra :associative?]))))
    (testing "segmentation survives SOAC to SegRed lowering"
      (let [scheduled (first (soac-lower/lower-soac node :cpu:0 :dtype :float))]
        (is (= :product (:phase scheduled)))
        (is (= '[row col] (mapv :name (segop/seg-space-dims (:space scheduled)))))
        (is (= #{'indices} (:outputs scheduled)))
        (is (= [:float :int] (reduction/dtypes (:reduction scheduled))))
        (is (= (* 8 (get-in scheduled [:grid :block-size]))
               (get-in scheduled [:grid :shared-mem-bytes])))
        (testing "the portable target leaf emits a mixed-dtype workgroup tree"
          (let [artifact (segop-opencl/generate-product-reduction-kernel
                          scheduled
                          :scalar-types {'nrows :int 'width :int}
                          :array-types {'values :float})
                source (:source artifact)]
            (is (= [:float :int]
                   (get-in artifact [:attributes :component-dtypes])))
            (is (= [:input :output :scalar :scalar :scalar]
                   (mapv :kind (:abi artifact))))
            (is (str/includes? source (str "__local float shared_0["
                                           (get-in scheduled [:grid :block-size]) "]")))
            (is (str/includes? source (str "__local int shared_1["
                                           (get-in scheduled [:grid :block-size]) "]")))
            (is (str/includes? source "barrier(CLK_LOCAL_MEM_FENCE)"))))))
    (testing "the schedule accounts for every component against the target SLM budget"
      (let [descriptor (assoc-in (hardware/descriptor-for :cpu:0) [:cache :slm] 512)
            scheduled (with-redefs [hardware/descriptor-for (constantly descriptor)]
                        (first (soac-lower/lower-soac node :cpu:0 :dtype :float)))]
        (is (= 64 (get-in scheduled [:schedule :workgroup-size])))
        (is (= 512 (get-in scheduled [:grid :shared-mem-bytes])))
        (is (= 512 (get-in scheduled [:schedule :attributes :slm-budget])))))
    (testing "a computed reduction bound survives into target source"
      (let [scheduled (-> (first (soac-lower/lower-soac node :cpu:0 :dtype :float))
                          (assoc-in [:space :dims 1 :bound] '(- width 1)))
            source (:source (segop-opencl/generate-product-reduction-kernel
                             scheduled
                             :scalar-types {'nrows :int 'width :int}
                             :array-types {'values :float}))]
        (is (str/includes? source "< (width - 1)"))))
    (testing "arity loss is rejected at construction"
      (is (thrown-with-msg?
           Exception #"arity"
           (reduction/make
            {:components [{:id :v :accumulator 'v :neutral 0.0 :dtype :float :result nil}
                          {:id :i :accumulator 'i :neutral 0 :dtype :int :result 'indices}]
             :index 'col
             :element-bindings [] :element-results ['candidate]
             :combine-parameters [['lv 'rv] ['li 'ri]]
             :combine-bindings [] :combine-results ['rv 'ri]}))))))

(deftest product-reduce-reference-semantics-test
  (let [values (float-array [1.0 Float/NaN Float/NaN -2.0
                             4.0 9.0 9.0 3.0])
        indices (int-array 2)
        nrows 2
        width 4]
    (raster.par/product-reduce!
     [nil indices]
     [[best-value (float Float/NEGATIVE_INFINITY) :float]
      [best-index (int Integer/MAX_VALUE) :int]]
     [[row nrows]]
     col width
     [candidate (aget values (+ (* row width) col))]
     [candidate (int col)]
     [[left-value right-value] [left-index right-index]]
     [left-nan (int (if (== left-value left-value) 0 1))
      right-nan (int (if (== right-value right-value) 0 1))
      better (int (if (== right-nan 1)
                    (if (== left-nan 1) (if (< right-index left-index) 1 0) 1)
                    (if (== left-nan 1)
                      0
                      (if (> right-value left-value)
                        1
                        (if (== right-value left-value)
                          (if (< right-index left-index) 1 0) 0)))))]
     [(if (== better 1) right-value left-value)
      (if (== better 1) right-index left-index)]
     {:associative? true :commutative? true
      :order {:nan :highest :tie :lowest-index}})
    (is (= [1 1] (vec indices)))))

(deftest product-reduce-par-scope-test
  (let [{:keys [scopes outer]} ((requiring-resolve 'raster.compiler.ir.form/scope-info)
                                argmax-product-form)]
    (is (contains? (set (:binders (first scopes))) 'col))
    (is (contains? (set (:binders (first scopes))) 'left-value))
    (is (some #{'nrows} outer))
    (is (some #{'width} outer))))

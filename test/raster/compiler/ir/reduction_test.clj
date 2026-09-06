(ns raster.compiler.ir.reduction-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.reference.product-opencl :as product-oracle]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.par :as par-ir]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
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

(defn- product-source-with-index-neutral [neutral]
  (let [form (apply list (assoc-in (vec argmax-product-form) [2 1 1] neutral))
        node (soac/par-form->soac '_ form 7 :dtype :float)
        scheduled (first (soac-lower/lower-soac node :cpu:0 :dtype :float))]
    (:source (product-oracle/generate-product-reduction-kernel
               scheduled :scalar-types {'nrows :int 'width :int}
               :array-types {'values :float}))))

(deftest product-neutral-emission-preserves-checked-casts
  (let [source (product-source-with-index-neutral '(int (float 16777217)))]
    (is (str/includes? source "= 16777216;"))
    (is (not (str/includes? source "= 16777217;"))))
  ;; Unknown callees may be refused earlier by typed scalar binding.
  (is (thrown? clojure.lang.ExceptionInfo (product-source-with-index-neutral '(unknown 0))))
  (doseq [neutral '[(byte 256) user/MAX_VALUE (int user/MAX_VALUE)]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"neutral has no OpenCL literal"
                         (product-source-with-index-neutral neutral)))))

(deftest canonical-scalar-is-one-component-product-test
  (let [operator (reduction/scalar
                  {:accumulator 'acc :neutral 0.0 :dtype :double :result 'sum
                   :index 'i :step-result '(+ acc (aget values i))})]
    (is (reduction/product-reduction? operator))
    (is (reduction/scalar? operator))
    (is (scan/associative-scan? (:algebra operator)))
    (is (= [:double] (reduction/dtypes operator)))
    (is (= {:acc 'acc :init 0.0 :lambda '(+ acc (aget values i))}
           (reduction/scalar-op operator)))))

(deftest lane-strided-tree-cannot-schedule-associativity-alone
  ;; Affine-function composition is associative, not commutative. The existing two-lane tree
  ;; groups e0,e2 then e1,e3 instead of preserving e0,e1,e2,e3.
  (let [compose (fn [[a b] [c d]] [(* a c) (+ (* a d) b)])
        [e0 e1 e2 e3 :as values] [[1 1] [2 0] [1 3] [1 0]]]
    (is (not= (reduce compose [1 0] values)
              (compose (compose e0 e2) (compose e1 e3)))))
  (let [node (soac/par-form->soac '_ argmax-product-form 7 :dtype :float)
        scheduled (first (soac-lower/lower-soac node :cpu:0 :dtype :float))]
    (doseq [[field expected] [[:associative? :product-reduction-not-associative]
                             [:commutative? :product-reduction-not-commutative]]
            flag [false nil]]
      (let [operator (assoc-in (:reduction node) [:algebra field] flag)]
        ;; Both initial schedule construction and emission of a changed retained operator must
        ;; check the same contract; callers cannot bypass it with an already-built schedule.
        (doseq [attempt [#(soac-lower/lower-soac (assoc node :reduction operator) :cpu:0 :dtype :float)
                         #(product-oracle/generate-product-reduction-kernel
                           (assoc scheduled :reduction operator)
                           :scalar-types {'nrows :int 'width :int}
                           :array-types {'values :float})]]
          (try (attempt) (is false "unsupported algebra must decline")
               (catch clojure.lang.ExceptionInfo e
                 (is (= expected (:reason (ex-data e)))))))))))

(deftest scalar-parallel-reduction-requires-the-registered-typed-identity
  (try
    (reduction/scalar
     {:accumulator 'acc :neutral 5.0 :dtype :double :result 'sum
      :index 'i :step-result '(+ acc (aget values i))})
    (is false "a non-identity init must not be duplicated across lanes and blocks")
    (catch clojure.lang.ExceptionInfo exception
      (is (= :reduction-nonidentity-init (:reason (ex-data exception))))
      (is (= 0.0 (:identity (ex-data exception)))))))

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
          (let [artifact (product-oracle/generate-product-reduction-kernel
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
            source (:source (product-oracle/generate-product-reduction-kernel
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

(deftest scalar-reduction-declares-the-element-conversion-into-its-carrier
  (testing "a double-typed element entering a float accumulator gets an explicit float cast"
    (let [element (with-meta '(clojure.core/* scale (clojure.core/aget a i))
                    {:raster.type/tag 'double})
          step (list 'clojure.core/+ 'acc element)
          reduction (reduction/scalar {:accumulator 'acc :neutral 0.0 :dtype :float
                                       :result 's :index 'i :step-result step})
          lambda (first (:results (reduction/fold-region reduction)))
          converted (nth lambda 2)]
      (is (= 'clojure.core/float (first converted)))
      (is (= element (second converted)))
      (is (= 'float (:raster.type/tag (meta converted))))
      (is (= converted (:element (:algebra reduction)))
          "the certificate is re-derived over the converted element")))
  (testing "a matching precision leaves the step untouched"
    (let [element (with-meta '(clojure.core/aget a i) {:raster.type/tag 'float})
          step (list 'clojure.core/+ 'acc element)
          reduction (reduction/scalar {:accumulator 'acc :neutral 0.0 :dtype :float
                                       :result 's :index 'i :step-result step})]
      (is (= step (first (:results (reduction/fold-region reduction)))))))
  (testing "an element without a retained precision is left to the owner"
    (let [step '(clojure.core/+ acc (clojure.core/aget a i))
          reduction (reduction/scalar {:accumulator 'acc :neutral 0.0 :dtype :float
                                       :result 's :index 'i :step-result step})]
      (is (= step (first (:results (reduction/fold-region reduction))))))))

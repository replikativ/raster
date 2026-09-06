(ns raster.compiler.passes.parallel.product-reduction-body-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as reference]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.reduction-test :as fixtures]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.product-reduction-body :as product]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]))

(defn argmax-segred []
  (let [form (vec fixtures/argmax-product-form)
        ;; This fixture supplies retained address widths explicitly, like analyzed source.
        form (update-in form [6 1]
                        (fn [[op array coordinate]]
                          (list op array
                                (walk/postwalk
                                 #(if (seq? %) (with-meta % {:tag 'long}) %) coordinate))))
        node (soac/par-form->soac '_ (apply list form) 7 :dtype :float)]
    (assoc-in (first (soac-lower/lower-soac node :cpu:0 :dtype :float))
              [:schedule :workgroup-size] 32)))

(def options
  {:array-types {'values :float}
   :array-shapes {'values [(launch/product 'nrows 'width)]}
   :scalar-types {'nrows :int 'width :int}
   :element-binding-types {'candidate :float}
   :combine-binding-types {'left-nan :int 'right-nan :int 'better :int}})

(deftest row-product-emits-through-shared-target-pipeline
  (let [segred (argmax-segred)
        scheduled (product/schedule segred options)
        old (reference/generate-product-reduction-kernel segred
              :scalar-types (:scalar-types options) :array-types (:array-types options))]
    (is (= segred (:source scheduled)))
    (is (= [:float :int] (get-in scheduled [:body :attributes :component-dtypes])))
    (is (= (:arguments old) (:arguments scheduled)))
    (is (= (* 8 (get-in segred [:schedule :workgroup-size]))
           (get-in scheduled [:body :launch :shared-memory-bytes])))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [emitted (target/emit-artifact "product_argmax" scheduled dialect)]
        (is (string? (:source emitted)))
        (is (= (mapv #(select-keys % [:name :kind :dtype :role]) (:abi old))
               (mapv #(select-keys % [:name :kind :dtype :role]) (:abi emitted))))))))

(deftest product-candidate-declines-missing-retained-evidence
  (doseq [[changed expected]
          [[(dissoc options :array-shapes) :input-storage]
           [(dissoc options :scalar-types) :scalar-dtype]
           [(dissoc options :combine-binding-types) :scalar-region-binding-dtype]]]
    (try
      (product/schedule (argmax-segred) changed)
      (is false (str "must decline " expected))
      (catch clojure.lang.ExceptionInfo e
        (is (= expected (:missing-rule (ex-data e))))))))

(deftest tuple-storage-and-control-remain-explicit
  (let [scheduled (product/schedule (argmax-segred) options)
        kernel (:body scheduled)
        kind #(.getSimpleName (class %))
        loop-op (first (:operations kernel))
        tree-branches (filter #(and (= "IfRegion" (kind %))
                                    (not= :product-writer (:condition %))) (:operations kernel))]
    (is (= [:float :int] (mapv (comp :type :binding) (:iter-args loop-op))))
    (is (= [:float :int] (mapv :dtype (:allocations kernel))))
    (is (= ['indices] (mapv :id (filter #(= :output (:kind %)) (:parameters kernel))))
        "the hidden winning-value component has scratch but no public output")
    (is (= 5 (count tree-branches)))
    (doseq [branch tree-branches]
      (is (= ["ScalarStore" "ScalarStore" "Yield"]
             (mapv kind (take-last 3 (:then-operations branch)))))
      (is (not-any? #(= "WorkgroupBarrier" (kind %)) (:then-operations branch))))
    (is (= 6 (count (filter #(= "WorkgroupBarrier" (kind %)) (:operations kernel)))))
    (is (false? (get-in scheduled [:attributes :source-storage-certified?])))))

(deftest invalid-neutral-or-unretained-address-width-is-rejected
  (is (thrown? clojure.lang.ExceptionInfo
               (product/schedule
                (assoc-in (argmax-segred) [:reduction :components 1 :neutral] 2147483648)
                options)))
  (let [node (soac/par-form->soac '_ fixtures/argmax-product-form 7 :dtype :float)
        untyped (first (soac-lower/lower-soac node :cpu:0 :dtype :float))]
    (try
      (product/schedule untyped options)
      (is false "unretained compound address widths must not be guessed")
      (catch clojure.lang.ExceptionInfo e
        (is (= :index-expression (:missing-rule (ex-data e))))))))

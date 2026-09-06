(ns raster.compiler.passes.parallel.product-reduction-body-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as reference]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.reduction-test :as fixtures]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.product-reduction-body :as product]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]))

(defn- argmax-form []
  (let [form (vec fixtures/argmax-product-form)
        ;; This fixture supplies retained address widths explicitly, like analyzed source.
        form (update-in form [6 1]
                        (fn [[op array coordinate]]
                          (list op array
                                (walk/postwalk
                                 #(if (seq? %) (with-meta % {:tag 'long}) %) coordinate))))]
    (apply list form)))

(defn- small-workgroup [segred]
  (-> segred
      (assoc-in [:schedule :workgroup-size] 32)
      (assoc-in [:grid :block-size] 32)
      (assoc-in [:grid :shared-mem-bytes] 256)))

(defn argmax-segred []
  (let [node (soac/par-form->soac '_ (argmax-form) 7 :dtype :float)]
    (small-workgroup (first (soac-lower/lower-soac node :cpu:0 :dtype :float)))))

(def options
  {:array-types {'values :float}
   :array-shapes {'values [(launch/product 'nrows 'width)]}
   :scalar-types {'nrows :int 'width :int}
   :element-binding-types {'candidate :float}
   :combine-binding-types {'left-nan :int 'right-nan :int 'better :int}})

(defn typed-argmax-segred []
  (let [tag-bindings (fn [bindings dtypes]
                       (vec (mapcat (fn [[id init]]
                                      [(with-meta id {:raster.type/tag
                                                      (dtype/scalar-tag-for-dtype (get dtypes id))})
                                       init]) (partition 2 bindings))))
        ;; Supply the facts normally attached by the walker; the frontend deliberately does not
        ;; infer types from these raw test S-expressions.
        form (-> (vec (argmax-form))
                 (update 6 tag-bindings (:element-binding-types options))
                 (update 9 tag-bindings (:combine-binding-types options))
                 (update 2 #(mapv (fn [[id neutral type]]
                                    [id (constant/literal-or-original neutral) type]) %)))
        program (frontend/form->program
                 (list 'let* ['effect (apply list form)] 'effect)
                 {:dtype :float :array-types {'values :float 'indices :int}
                  :scalar-types (:scalar-types options)})]
    (small-workgroup (first (soac-lower/lower-typed-product-reduce program :cpu:0 :dtype :float)))))

(defn typed-local-address-segred []
  (let [segred (typed-argmax-segred)
        ;; Use the semantic reduction index, not a spelling inferred from generated code.
        index (get-in segred [:reduction :index])]
    (update-in segred [:reduction :element :bindings]
               (fn [bindings]
                 (into [(with-meta 'offset {:raster.type/tag 'long}) (list 'long index)]
                       (mapcat (fn [[id init]] [id (util/subst-syms {index 'offset} init)])
                               (partition 2 bindings)))))))

(deftest typed-product-projection-retains-local-types-without-caller-reconstruction
  (let [segred (typed-argmax-segred)
        tags (fn [region]
               (into {} (map #(vector % (:raster.type/tag (meta %))))
                     (take-nth 2 (get-in segred [:reduction region :bindings]))))
        inferred-options (dissoc options :element-binding-types :combine-binding-types)]
    (is (= {'candidate 'float} (tags :element)))
    (is (= {'left-nan 'int 'right-nan 'int 'better 'int} (tags :combine)))
    (is (string? (:source (target/emit-artifact "typed_product_argmax"
                                              (product/schedule segred inferred-options)
                                              :opencl-portable))))
    (try
      (product/schedule segred (assoc-in options [:combine-binding-types 'better] :float))
      (is false "retained source types cannot be overridden by candidate options")
      (catch clojure.lang.ExceptionInfo e
        (is (= :binding-dtype-conflict (:missing-rule (ex-data e))))))))

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

(deftest source-product-grid-is-the-actual-segment-schedule
  (doseq [source [(argmax-segred) (typed-argmax-segred)]]
    (is (= 'nrows (get-in source [:grid :num-blocks])))
    (doseq [[field bad] [[:num-blocks 'width] [:block-size 64] [:shared-mem-bytes 128]]]
      (try
        (product/schedule (assoc-in source [:grid field] bad) options)
        (is false (str "contradictory grid " field " must decline"))
        (catch clojure.lang.ExceptionInfo e
          (is (= :source-grid (:missing-rule (ex-data e)))))))))

(deftest source-refinement-is-replayed-not-asserted-by-a-candidate-label
  (let [source (typed-argmax-segred)
        options (dissoc options :element-binding-types :combine-binding-types)
        candidate (product/schedule source options)]
    (is (= candidate (product/validate-source! candidate source options)))
    (doseq [forged [(assoc-in candidate [:numerics :policy] :different-tree)
                    (assoc-in candidate [:body :operations 0 :upper]
                              (body/index-cast 0 :long :exact))
                    (assoc-in candidate [:body :operations 0 :iter-args 1 :initial]
                              (body/literal 0 :int))]]
      (try
        (product/validate-source! forged source options)
        (is false "a well-typed but different computation must not retain the source witness")
        (catch clojure.lang.ExceptionInfo e
          (is (= :source-refinement (:missing-rule (ex-data e)))))))
    (try
      (product/validate-source! candidate (assoc source :id :other-equation) options)
      (is false "a caller-supplied source, not the candidate's own label, closes identity")
      (catch clojure.lang.ExceptionInfo e
        (is (= :source-identity (:missing-rule (ex-data e))))))
    (try
      (product/validate-source!
       candidate source
       (assoc-in options [:array-shapes 'values] [(launch/sum (launch/product 'nrows 'width) 1)]))
      (is false "changed independent storage facts require a different refinement")
      (catch clojure.lang.ExceptionInfo e
        (is (= :source-refinement (:missing-rule (ex-data e))))))))

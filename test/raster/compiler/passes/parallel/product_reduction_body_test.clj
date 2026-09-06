(ns raster.compiler.passes.parallel.product-reduction-body-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.backend.gpu.segop-opencl :as reference]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.reduction-test :as fixtures]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.product-reduction-body :as product]
            [raster.compiler.passes.parallel.product-reduction-regions :as regions]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
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

(defn- typed-argmax-program [values & [scalar-types materialize-value? scale?]]
  (let [tag-bindings (fn [bindings dtypes]
                       (vec (mapcat (fn [[id init]]
                                      [(with-meta id {:raster.type/tag
                                                      (dtype/scalar-tag-for-dtype (get dtypes id))})
                                       init]) (partition 2 bindings))))
        ;; Supply the facts normally attached by the walker; the frontend deliberately does not
        ;; infer types from these raw test S-expressions.
        form (-> (vec (argmax-form))
                 (cond-> materialize-value? (assoc-in [1 0] 'maxima))
                 (cond-> scale? (update-in [6 1]
                                          #(with-meta (list '* 'alpha %) {:raster.type/tag 'float})))
                 (update 6 tag-bindings (:element-binding-types options))
                 (update 9 tag-bindings (:combine-binding-types options))
                 (update 2 #(mapv (fn [[id neutral type]]
                                    [id (constant/literal-or-original neutral) type]) %)))
        program (frontend/form->program
                 (list 'let* ['effect (apply list form)] 'effect)
                 {:dtype :float :array-types {'values :float 'indices :int 'maxima :float}
                  :values values :scalar-types (or scalar-types (:scalar-types options))})]
    program))

(defn typed-argmax-segred
  ([] (typed-argmax-segred (:scalar-types options)))
  ([scalar-types]
   (small-workgroup
    (first (soac-lower/lower-typed-product-reduce
            (typed-argmax-program {} scalar-types) :cpu:0 :dtype :float)))))

(defn typed-local-address-segred
  ([] (typed-local-address-segred (:scalar-types options)))
  ([scalar-types]
  (let [segred (typed-argmax-segred scalar-types)
        ;; Use the semantic reduction index, not a spelling inferred from generated code.
        index (get-in segred [:reduction :index])]
    (update-in segred [:reduction :element :bindings]
               (fn [bindings]
                 (into [(with-meta 'offset {:raster.type/tag 'long}) (list 'long index)]
                       (mapcat (fn [[id init]] [id (util/subst-syms {index 'offset} init)])
                               (partition 2 bindings))))))))

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
        reduction-operations (get-in kernel [:operations 1 :then-operations])
        loop-op (first reduction-operations)
        tree-branches (filter #(and (= "IfRegion" (kind %))
                                    (not= :product-writer (:condition %))) reduction-operations)]
    (is (= :product-active (get-in kernel [:operations 1 :condition])))
    (is (= [:float :int] (mapv (comp :type :binding) (:iter-args loop-op))))
    (is (= [:float :int] (mapv :dtype (:allocations kernel))))
    (is (= ['indices] (mapv :id (filter #(= :output (:kind %)) (:parameters kernel))))
        "the hidden winning-value component has scratch but no public output")
    (is (= 5 (count tree-branches)))
    (doseq [branch tree-branches]
      (is (= ["ScalarStore" "ScalarStore" "Yield"]
             (mapv kind (take-last 3 (:then-operations branch)))))
      (is (not-any? #(= "WorkgroupBarrier" (kind %)) (:then-operations branch))))
    (is (= 6 (count (filter #(= "WorkgroupBarrier" (kind %)) reduction-operations))))
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
    (is (= '(max 1 nrows) (get-in source [:grid :num-blocks])))
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
                    (assoc-in candidate [:body :operations 1 :then-operations 0 :upper]
                              (body/index-cast 0 :long :exact))
                    (assoc-in candidate [:body :operations 1 :then-operations 0 :iter-args 1 :initial]
                              (body/literal 0 :int))
                    (assoc-in candidate [:body :operations 0 :expression :arguments 1]
                              (body/literal 0 :long))
                    (assoc-in candidate [:body :launch :group-count]
                              [(launch/runtime-value '_n_bound)])]]
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

(defn- graph-context [known-input? & [input-options output-options scalar-types materialize-value? scale?]]
  (let [scalar-types (or scalar-types (:scalar-types options))
        algorithm (typed-argmax-program
                   (if known-input?
                     {'values (av/tensor {:dtype :float :shape ['nrows 'width]})}
                     {}) scalar-types materialize-value? scale?)
        ;; Model a compiler-generated typed program as well as the analyzed-source frontend,
        ;; which already declines non-plain aget storage before this boundary.
        algorithm (if input-options
                    (apply list (update-in (vec algorithm) [1 :values 'values] merge input-options))
                    algorithm)
        algorithm (if output-options
                    (apply list
                           (-> (vec algorithm)
                               (update-in [1 :values 'indices] merge output-options)
                               (update-in [1 :values (first (nth (first (nth algorithm 2)) 2))] merge
                                          (select-keys output-options
                                                       [:logical-layout :representation]))))
                    algorithm)
        scheduled (:form (segop-lower/segop-lower-pass
                          (route/program-envelope algorithm)
                          {:dtype :float :target-device :cpu:0
                           :array-types (:array-types options)
                           :scalar-types scalar-types}))
        equation (first (:equations scheduled))
        {:keys [graph body]} (equation-graph/make-for-equation scheduled equation)]
    {:algorithm (:algorithm equation) :body body :graph graph :node (first (:nodes graph))}))

(deftest production-product-emission-is-graph-bound-and-target-neutral
  (doseq [materialize-value? [false true]
          dialect [:opencl-portable :cuda :hip]]
    (let [{:keys [graph algorithm body]} (graph-context false nil nil nil materialize-value?)
          emitted (with-redefs [reference/generate-product-reduction-kernel
                                (fn [& _] (throw (ex-info "legacy product emitter reached" {})))]
                    (reference/generate-kernel-graph
                     graph :target-dialect dialect
                     :scheduled-equation-algorithm algorithm :scheduled-equation-body body))
          artifacts (executable/artifacts emitted)
          artifact (first artifacts)]
      (is (= emitted (executable/validate! emitted)))
      (is (= 1 (count artifacts)))
      (is (= (if materialize-value? 2 1) (count (:outputs emitted))))
      (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
      (is (false? (get-in artifact [:attributes :candidate-only])))
      (is (true? (get-in artifact [:attributes :source-storage-certified?])))))
  (let [{:keys [graph]} (graph-context false)]
    (is (thrown? clojure.lang.ExceptionInfo (reference/generate-kernel-graph graph)))))

(deftest graph-storage-facts-are-derived-from-the-retained-program
  (let [{:keys [algorithm body graph node]} (graph-context true)
        derived (product/graph-options node graph algorithm body)
        candidate (product/schedule (:operation node) derived)]
    (is (= [(launch/product 'nrows 'width)] (get-in derived [:array-shapes 'values])))
    (is (= candidate (product/validate-against-node! candidate node graph algorithm body)))
    (is (false? (get-in candidate [:attributes :source-storage-certified?]))
        "storage correspondence alone does not prove arbitrary indexed reads safe")
    (doseq [forged [(assoc-in graph [:inputs 0 :elements] 1)
                    (assoc-in graph [:inputs 0 :dtype] :double)]]
      (try
        (product/validate-against-node! candidate node forged algorithm body)
        (is false "a graph annotation must not replace the retained storage contract")
        (catch clojure.lang.ExceptionInfo e
          (is (= :scheduled-equation-projection (:reason (ex-data e)))))))
    (try
      (product/graph-options (assoc node :id :unrelated) graph algorithm body)
      (is false "the exact node must belong to the graph")
      (catch clojure.lang.ExceptionInfo e
        (is (= :graph-node (:missing-rule (ex-data e))))))))

(deftest unknown-product-input-storage-is-refined-from-typed-loads
  (let [{:keys [algorithm body graph node]} (graph-context false)
        options (product/graph-options node graph algorithm body)
        candidate (product/schedule (:operation node) options)]
    (is (= [(launch/product 'nrows 'width)] (get-in options [:array-shapes 'values])))
    (is (= candidate (product/validate-against-node! candidate node graph algorithm body)))
    (is (true? (get-in candidate [:attributes :candidate-only])))))

(deftest runtime-allocation-extents-are-not-independent-minimum-shapes
  (let [{:keys [algorithm body graph node]}
        (graph-context true {:shape ['(extent values)]} {:shape ['(extent indices)]})
        admitted (product/schedule-for-node node graph algorithm body)]
    (is (= (launch/product 'nrows 'width) (get-in graph [:inputs 0 :elements])))
    (is (= 'nrows (get-in graph [:outputs 0 :elements])))
    (is (false? (get-in admitted [:attributes :candidate-only])))))

(deftest floating-captures-and-optional-refinement-cannot-bypass-read-capacity
  (let [context #(graph-context true {:shape [1]} nil
                                (assoc (:scalar-types options) 'alpha :float) false true)
        {:keys [algorithm body graph node]} (context)]
    (is (= (launch/maximum (launch/product 'nrows 'width) 1)
           (get-in graph [:inputs 0 :elements])))
    (is (false? (get-in (product/schedule-for-node node graph algorithm body)
                        [:attributes :candidate-only])))
    (with-redefs-fn
      {(ns-resolve 'raster.compiler.passes.parallel.scheduled-equation-graph
                    'product-read-requirements) (constantly {})}
      (fn []
        (let [{:keys [algorithm body graph node]} (context)]
          (try
            (product/schedule-for-node node graph algorithm body)
            (is false "a known capacity cannot substitute for the omitted access requirement")
            (catch clojure.lang.ExceptionInfo e
              (is (= :graph-read-capacity (:missing-rule (ex-data e)))))))))))

(deftest graph-capacity-is-not-an-index-safety-certificate
  (let [{:keys [algorithm body graph node]} (graph-context true {:shape [1]})
        candidate (product/schedule (:operation node)
                                    (product/graph-options node graph algorithm body))]
    ;; Required storage is now the maximum of the AV contract and the proven dense read.
    ;; This still does not authorize production or bypass the runtime buffer-capacity check.
    (is (= (launch/maximum (launch/product 'nrows 'width) 1)
           (get-in graph [:inputs 0 :elements])))
    (is (= candidate (product/validate-against-node! candidate node graph algorithm body)))
    (is (true? (get-in candidate [:attributes :candidate-only])))
    (is (false? (get-in candidate [:attributes :source-storage-certified?])))))

(deftest derived-read-capacity-is-a-checked-minimum
  (doseq [shape [[1] [4096]]]
    (let [{:keys [graph]} (graph-context true {:shape shape})
          required (get-in graph [:inputs 0 :elements])]
      (is (= (max (first shape) 12)
             (launch/resolve-expression {'nrows 3 'width 4} required)))
      (is (= (first shape)
             (launch/resolve-expression {'nrows 0 'width 4} required)))
      (is (thrown? ArithmeticException
                   (launch/resolve-expression {'nrows Long/MAX_VALUE 'width 2} required))))))

(deftest shared-product-input-keeps-every-derived-read-requirement
  (let [{:keys [body node]} (graph-context false)
        source (:operation node)
        second-source (assoc-in source [:space :dims 0 :bound] 'more-rows)
        values (assoc (:values body) 'more-rows (get-in body [:values 'nrows]))
        derive (ns-resolve 'raster.compiler.passes.parallel.scheduled-equation-graph
                           'product-read-requirements)
        requirements (derive values [source second-source]
                             {'more-rows (launch/sum 'nrows 1)})]
    (is (= [(launch/product 'nrows 'width)
            (launch/product (launch/sum 'nrows 1) 'width)]
           (get requirements 'values)))
    (is (= 16 (launch/resolve-expression {'nrows 3 'width 4}
                                         (apply launch/maximum (get requirements 'values)))))))

(deftest typed-read-requirements-do-not-assume-every-access-is-dense
  (let [source (typed-local-address-segred)
        decline! (fn [rule message data] (throw (ex-info message (assoc data :rule rule))))
        derive #(regions/dense-read-requirements % options decline!)
        change-load (fn [coordinate]
                      (assoc-in source [:reduction :element :bindings 3]
                                (list 'clojure.core/aget 'values coordinate)))]
    (is (= {'values (launch/product 'nrows 'width)} (derive source)))
    (is (= (derive source) (regions/dense-read-requirements source
                                                          (dissoc options :array-shapes) decline!))
        "lowering does not need a fabricated shape")
    (doseq [coordinate ['offset 0
                        '(clojure.core/aget indices col)]]
      (is (thrown? clojure.lang.ExceptionInfo (derive (change-load coordinate)))))
    (let [with-combine-read
          (assoc-in source [:reduction :combine :results 0]
                    '(clojure.core/aget values 0))]
      (try (derive with-combine-read)
           (is false "combine reads execute outside the positive element domain")
           (catch clojure.lang.ExceptionInfo e
             (is (= :combine-read (:rule (ex-data e)))))))))

(deftest logical-representations-must-be-lowered-before-raw-product-storage
  (doseq [storage-options [{:representation {:kind :quantized :scheme :q4-k}}
                          {:logical-layout {:kind :strided :strides [2]}}]
          input? [true false]]
    (let [{:keys [algorithm body graph node]}
          (if input? (graph-context true storage-options)
              (graph-context true nil storage-options))]
      (try
        (product/graph-options node graph algorithm body)
        (is false "logical element counts are not raw packed or strided storage capacities")
        (catch clojure.lang.ExceptionInfo e
          (is (= :graph-storage-representation (:missing-rule (ex-data e)))))))))

(deftest output-capacity-needs-an-explicit-relation-to-the-written-segment-domain
  (let [{:keys [algorithm body graph node]} (graph-context true nil {:shape [1]})
        candidate (product/schedule (:operation node)
                                    (product/graph-options node graph algorithm body))]
    (try
      (product/validate-against-node! candidate node graph algorithm body)
      (is false "a retained but unrelated output capacity does not establish the row write contract")
      (catch clojure.lang.ExceptionInfo e
        (is (= :graph-output-storage (:missing-rule (ex-data e))))))))

(deftest long-logical-dimensions-retain-their-abi-width
  (let [{:keys [algorithm body graph node]}
        (graph-context true nil nil {'nrows :long 'width :long})
        derived (product/graph-options node graph algorithm body)
        candidate (product/schedule (:operation node) derived)]
    (is (= candidate (product/validate-against-node! candidate node graph algorithm body)))
    (is (= #{:long} (set (map :dtype (:scalar-bindings candidate)))))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [artifact (target/emit-artifact "long_product" candidate dialect)]
        (is (= [:long :long :long]
               (mapv :kernel-dtype (filter #(= :scalar (:kind %)) (:abi artifact)))))))
    (is (= :long (get-in candidate [:body :operations 1 :then-operations 0 :index :type])))
    (is (= :int (get-in candidate [:body :parameters 1 :dtype]))
        "dimension width does not change the winning-index component representation")
    (let [artifact (target/emit-artifact "long_product_range" candidate :opencl-portable)
          bindings {'values :input 'indices :output
                    'nrows {:type :long :value (+ 2 (long Integer/MAX_VALUE))}
                    'width {:type :long :value 1}}]
      (try
        (call/make artifact (mapv bindings (:arguments artifact)))
        (is false "a long logical row count still needs representable physical group indices")
        (catch clojure.lang.ExceptionInfo e
          (is (= :kernel-launch-index-range (:reason (ex-data e)))))))))

(deftest dimension-widths-are-independent
  (doseq [[row-type width-type] [[:int :long] [:long :int]]]
    (let [{:keys [algorithm body graph node]}
          (graph-context true nil nil {'nrows row-type 'width width-type})
          candidate (product/schedule (:operation node)
                                      (product/graph-options node graph algorithm body))
          artifact (target/emit-artifact "mixed_dimension_product" candidate :opencl-portable)]
      (is (= candidate (product/validate-against-node! candidate node graph algorithm body)))
      (is (= {'nrows row-type 'width width-type '_n_bound row-type}
             (into {} (map (juxt :name :kernel-dtype))
                   (filter #(= :scalar (:kind %)) (:abi artifact))))))))

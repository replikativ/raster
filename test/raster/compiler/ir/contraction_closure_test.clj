(ns raster.compiler.ir.contraction-closure-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.contraction-closure :as closure]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.passes.parallel.staged-contraction-body :as staged]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]))

(defn- program []
  (let [source (fixtures/packed-facts 3 5 3 32)
        values (into {} (map (fn [[id dtype shape]]
                              [id (av/tensor {:dtype dtype :shape shape})]))
                     [['a :byte [288]] ['b :byte [480]] ['da :float [9]]
                      ['db :float [15]] ['out :float [15]] ['result :float [3 5]]])
        attributes {:contraction source :array-parameters '[a b da db]
                    :capture-parameters []}
        equation-facts (assoc (soac/default-equation-facts)
                             :effects #{:memory/read :memory/write}
                             :aliases {'result 'out}
                             :attributes {:result-storage
                                          [{:destination 'out :access :write :host-return :buffer}]})]
    (soac/make
     (soac/default-program-facts {:values values :inputs '[a b da db]
                                 :effects #{:memory/read :memory/write}
                                 :equations {'contraction equation-facts}})
     [(list '= 'contraction '[result] (list 'contract attributes '[a b da db] []))]
     '[result])))

(deftest typed-contraction-retains-the-canonical-payload-through-ssa
  (let [p (program)
        source (get-in (soac/operation-parts (first (soac/equations p)))
                       [:attributes :contraction])
        value-map (zipmap '[a b da db out result]
                         [[:arg 0] [:arg 1] [:arg 2] [:arg 3] [:output 0] [:result 0]])
        renamed (soac/remap-values p value-map)
        equation (first (soac/equations renamed))
        {:keys [attributes arrays captures]} (soac/operation-parts equation)]
    (is (= 'contract (soac/operation-kind equation)))
    (is (identical? source (:contraction attributes)) "no stage reconstruction or flattening")
    (is (= [[:arg 0] [:arg 1] [:arg 2] [:arg 3]] (soac/operation-inputs equation)))
    (is (= [[:output 0]] (soac/physical-results renamed equation)))
    (is (= {'a [:arg 0] 'b [:arg 1] 'da [:arg 2] 'db [:arg 3]}
           (closure/bindings attributes arrays captures)))
    (is (= [:float :int] (mapv :dtype (:stages source))))
    (is (= renamed (first (fusion/fusion-fixpoint renamed)))
        "flat fusion rules leave the staged closure intact")
    (is (= renamed (soac/validate! renamed)))))

(deftest typed-contraction-checks-independent-storage-and-closure
  (let [p (program)
        with-values (fn [f]
                      (list 'soac-program (update (soac/facts p) :values f)
                            (soac/equations p) (soac/outputs p)))
        rewrite-attributes (fn [f]
                             (let [[equals id results [_ attributes arrays captures]]
                                   (first (soac/equations p))]
                               (list 'soac-program (soac/facts p)
                                     [(list equals id results
                                            (list 'contract (f attributes) arrays captures))]
                                     (soac/outputs p))))]
    (doseq [[label transform]
            [[:short-core #(assoc-in % ['a :shape] [287])]
             [:short-scale #(assoc-in % ['da :shape] [8])]
             [:wrong-core-type #(assoc-in % ['a :dtype] :float)]
             [:wrong-scale-type #(assoc-in % ['da :dtype] :byte)]
             [:encoded-input #(assoc-in % ['a :representation] {:kind :quantized})]
             [:strided-input #(assoc-in % ['a :logical-layout] {:strides [2]})]
             [:sharded-input #(assoc-in % ['a :sharding] {:axis 0})]
             [:encoded-output #(-> %
                                   (assoc-in ['result :representation] {:kind :quantized})
                                   (assoc-in ['out :representation] {:kind :quantized}))]
             [:strided-output #(-> %
                                   (assoc-in ['result :logical-layout] {:strides [2]})
                                   (assoc-in ['out :logical-layout] {:strides [2]}))]
             [:short-output #(assoc-in % ['out :shape] [14])]
             [:wrong-result-shape #(assoc-in % ['result :shape] [15])]]]
      (testing (name label)
        (is (thrown? clojure.lang.ExceptionInfo (soac/validate! (with-values transform))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (soac/validate! (rewrite-attributes #(assoc % :array-parameters '[a b da])))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (soac/validate! (rewrite-attributes #(assoc-in % [:contraction :stages 1 :init] 1)))))
    (try
      (soac/validate! (rewrite-attributes #(assoc-in % [:contraction :stages 0 :lift] '(* inner t))))
      (is false "outer lift cannot read an inner-stage coordinate")
      (catch clojure.lang.ExceptionInfo e
        (is (= :stage-lift-scope (:missing-rule (ex-data e))))))))

(deftest retained-contraction-binds-generated-storage-without-source-reparse
  (let [p (soac/remap-values (program) {'a [:arg 0] 'b [:arg 1] 'da [:arg 2]
                                      'db [:arg 3] 'out [:output 0] 'result [:result 0]})
        {:keys [facts bindings array-types]} (projection/contraction-binding p (first (soac/equations p)))
        lowered (staged/lower facts)
        arguments (mapv #(get bindings % %) (:arguments lowered))
        bound (scheduled/make (-> (into {} lowered)
                                  (assoc :arguments arguments)
                                  (assoc-in [:effects :uses]
                                            (scheduled/derive-uses (:body lowered) arguments))))]
    (is (= '{a :byte b :byte da :float db :float out :float} array-types))
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [graph (target/emit-static-dense-graph "typed_closure" bound dialect)]
        (is (= [[:arg 0] [:arg 1] [:arg 2] [:arg 3] [:output 0]] (:arguments graph)))
        (is (= [:byte :byte :float :float :float] (mapv :dtype (:abi graph))))))))

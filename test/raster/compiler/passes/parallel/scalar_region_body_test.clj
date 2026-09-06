(ns raster.compiler.passes.parallel.scalar-region-body-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- lowerer []
  (let [decline! (fn [rule message data] (throw (ex-info message (assoc data :rule rule))))]
    (scalar/make-lowerer
     {:array-types {'x :float} :arrays #{'x} :scalar-types {'i :int}
      :lower-index (fn [expression scope] (index/lower expression (conj scope 'i) decline!))
      :decline! decline!})))

(defn- mixed-region [lowerer]
  ((:lower-region lowerer)
   '{:bindings [candidate (aget x i) better (> candidate old-value)]
     :results [(if better candidate old-value) (if better i old-index)]}
   [:float :int] {'candidate :float 'better :predicate}
   {'old-value :float 'old-index :int}))

(deftest coupled-results-share-bindings-without-sharing-component-types
  (let [{:keys [operations results types] :as lowered} (mixed-region (lowerer))
        kernel
        (body/make
         {:id :mixed-region
          :parameters [(body/->KernelParameter 'x :input :float [16] :global
                                               (layout/row-major [16] :float) :operand)
                       (body/->KernelParameter 'i :scalar :int [] nil nil :parameter)
                       (body/->KernelParameter 'old-value :scalar :float [] nil nil :parameter)
                       (body/->KernelParameter 'old-index :scalar :int [] nil nil :parameter)
                       (body/->KernelParameter 'values :output :float [1] :global
                                               (layout/row-major [1] :float) :result)
                       (body/->KernelParameter 'indices :output :int [1] :global
                                               (layout/row-major [1] :int) :result)]
          :stable-reads [(body/stable-read 'x)]
          :operations (into operations [(body/->ScalarStore 'values [0] (first results) nil)
                                        (body/->ScalarStore 'indices [0] (second results) nil)])
          :launch (launch/spec {:workgroup-size [1] :group-count [1]})
          :provenance {:dialect :test} :attributes {}})]
    (is (= [:float :int] types))
    (is (= ["ScalarLoad" "ScalarCompute" "IfRegion" "IfRegion"]
           (mapv #(.getSimpleName (class %)) operations)))
    (is (= (:condition (nth operations 2)) (:condition (nth operations 3)))
        "both components consume the same predicate SSA value")
    (is (= results (mapv (comp :id first :results) (drop 2 operations))))
    (is (= [:float :int] (mapv (comp :type first :results) (drop 2 operations))))
    (doseq [target [:opencl-portable :cuda :hip]]
      (is (string? (emit/emit-scalar-kernel "mixed_region" kernel {:target-dialect target}))))))

(deftest ordered-bindings-and-fresh-ssa-across-region-instances
  (let [lower (lowerer)
        a (mixed-region lower)
        b (mixed-region lower)
        constants ((:lower-region lower)
                   '{:bindings [a 3 b a] :results [b a]}
                   [:int :int] {'a :int 'b :int} {})]
    (is (empty? (:operations constants)))
    (is (= [(body/literal 3 :int) (body/literal 3 :int)] (:results constants)))
    (is (not-any? (set (:results a)) (:results b)))
    (is (= {:operations [] :results [] :types []}
           ((:lower-region lower) {:bindings [] :results []} [] {} {})))))

(deftest missing-and-inconsistent-type-evidence-is-not-inferred-from-consumers
  (doseq [[region result-types binding-types env rule]
          [['{:bindings [a (aget x i)] :results [a]} [:float] {} {}
            :scalar-region-binding-dtype]
           ['{:bindings [a (aget x i)] :results [a]} [:int] {'a :int} {}
            :scalar-region-dtype]
           ['{:bindings [] :results [a]} [:int] {} {'a :float}
            :scalar-region-dtype]
           ['{:bindings [a 1 a 2] :results [a]} [:int] {'a :int} {}
            :scalar-region-shape]
           ['{:bindings [] :results [1 2]} [:int] {} {} :scalar-region-shape]
           ['{:bindings [a b b 1] :results [a]} [:int] {'a :int 'b :int} {}
            :unbound-scalar]]]
    (try
      ((:lower-region (lowerer)) region result-types binding-types env)
      (is false (str "expected " rule))
      (catch clojure.lang.ExceptionInfo exception
        (is (= rule (:rule (ex-data exception))))))))

(deftest typed-constant-bindings-retain-comparison-and-index-types
  (let [lower (:lower-region (lowerer))
        comparison (lower '{:bindings [threshold 0.5] :results [(> threshold old-value)]}
                          [:predicate] {'threshold :float} {'old-value :float})
        indexed (lower '{:bindings [j 0] :results [(aget x j)]}
                       [:float] {'j :long} {})]
    (is (= (body/literal 0.5 :float)
           (get-in comparison [:operations 0 :expression :arguments 0])))
    (is (= [(body/index-cast 0 :long :exact)]
           (get-in indexed [:operations 0 :coordinates])))
    (doseq [[value type] [[0.5 :float] [256 :byte] [2147483648 :int]]]
      (try
        (lower {:bindings ['j value] :results '[(aget x j)]} [:float] {'j type} {})
        (is false "nonintegral or out-of-range index must decline")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :index-literal (:rule (ex-data exception)))))))))

(deftest region-substitution-respects-nested-lexical-bindings
  (let [lower (:lower-region (lowerer))
        result (lower '{:bindings [a 3]
                        :results [(let* [a 9] a) a]}
                      [:int :int] {'a :int} {})]
    (is (= [(body/literal 9 :int) (body/literal 3 :int)] (:results result)))))

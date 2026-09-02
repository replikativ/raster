(ns raster.compiler.passes.parallel.typed-ordered-effect-map-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private extent (av/tensor {:dtype :long :shape []}))
(def ^:private vector-value (av/tensor {:dtype :float :shape '[n]}))
(def ^:private scalar-slot (av/tensor {:dtype :float :shape '[1]}))

(defn- effect-program
  []
  (let [out-result [:effect 0 0]
        total-result [:effect 0 1]
        reduction (dialect/reducing-scatter-conflict '+ :float)
        equation
        (list '= 0 [out-result total-result]
              (list 'effect-map
                    {:index 'i :extent 'n :dtypes [:float :float]}
                    '[x] [] '[out total]
                    (dialect/effect-lambda-form
                     '[element out-destination total-destination]
                     [(dialect/local-value 'shifted :float '(+ element 1.0))]
                     [(list 'effect 'out-destination :unique 'i '(> element 0.0) 'shifted)
                      (list 'effect 'total-destination reduction 0 true 'element)])))
        storage [{:destination 'out :access :write :host-return :effect}
                 {:destination 'total :access :read-write :host-return :effect}]
        equation-facts
        (assoc (dialect/default-equation-facts)
               :effects #{:memory/write}
               :aliases {out-result 'out total-result 'total}
               :attributes {:result-storage storage})]
    (dialect/make
     (dialect/default-program-facts
      {:values {'n extent 'x vector-value 'out vector-value 'total scalar-slot
                out-result vector-value total-result scalar-slot}
       :inputs '[n x]
       :equations {0 equation-facts}
       :effects #{:memory/write}})
     [equation] [])))

(defn- nested-operations
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (concat (nested-operations (or (:then-operations operation) []))
                          (nested-operations (or (:else-operations operation) [])))))
          operations))

(defn- normalize-fresh-thread-index
  [operation]
  (assoc-in operation [:space :flat-idx] 'thread-index))

(deftest ordered-effects-lower-to-portable-kernelbody-control-and-atomics
  (let [program (effect-program)
        operation (first (soac-lower/lower-typed-effect-map
                          program :ze:0 :dtype :float))
        scheduled (:form (segop-lower/segop-lower-pass
                          (route/program-envelope program)
                          {:target-device :ze:0 :dtype :float}))
        scheduled-operation (first (get-in scheduled [:equations 0 :operations]))]
    (is (= :ordered-effects (:write-conflict operation)))
    (is (= (normalize-fresh-thread-index operation)
           (normalize-fresh-thread-index scheduled-operation))
        "whole-program scheduling dispatches the same semantic effect-map lowering")
    (is (= #{'out 'total} (:outputs operation)))
    (is (= {:locals [{:id 'shifted :dtype :float :init '(+ (clojure.core/aget x i) 1.0)}]
            :effects
            [{:destination 'out :conflict :unique :destination-index 'i
              :predicate '(> (clojure.core/aget x i) 0.0) :value 'shifted}
             {:destination 'total
              :conflict (dialect/reducing-scatter-conflict '+ :float)
              :destination-index 0 :predicate true
              :value '(clojure.core/aget x i)}]}
           (:scalar-region operation)))
    (doseq [[target atomic]
            [[:opencl-portable "atomic_add_float"]
             [:cuda "atomicAdd"]
             [:hip "atomicAdd"]]]
      (testing (name target)
        (let [artifact (segop-opencl/generate-scheduled-segmap-kernel
                        operation :dtype :float :target-dialect target
                        :array-types {'x :float 'out :float 'total :float}
                        :scalar-types {'n :long})
              kernel-body (get-in artifact [:attributes :kernel-body])
              operations (nested-operations (:operations kernel-body))
              pointer-slots (filterv #(not= :scalar (:kind %)) (:abi artifact))]
          (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
          (is (= ['x 'out 'total] (mapv :name pointer-slots)))
          (is (= [:input :output :inout] (mapv :kind pointer-slots)))
          (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
          (is (some #(= "ScalarStore" (some-> % class .getSimpleName)) operations))
          (is (some #(= "AtomicRMW" (some-> % class .getSimpleName)) operations))
          (is (str/includes? (:source artifact) atomic)))))))

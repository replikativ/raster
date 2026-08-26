(ns raster.gpu.compiled-composition-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as gpu-link]))

(defn component [_x _w _n])

(def ^:private kernel
  (artifact/make
   {:kernel-name "compiled_composition_axpy"
    :source "__kernel void compiled_composition_axpy(float* x, float* w, float* y, long n) {}"
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'w :input :float)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x w y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x w] :writes '[y]}}))

(defn- descriptor []
  {:dtype :float
   :all-params '[x w n]
   :array-params '[x w]
   :scalar-params '[n]
   :array-roles {'x :input 'w :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :map :kernel-name "compiled_composition_axpy" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(deftest semantic-artifacts-compose-before-one-runtime-instantiation
  (let [weight (float-array 16)
        prepare #(compiled/lower #'component [(float-array 16) weight 16]
                                 {:target :ze:0 :constants '[w]})
        [first second]
        (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))
                      gpu-link/instantiate! (fn [& _]
                                              (throw (AssertionError.
                                                      "lower must not contact the runtime")))]
          [(prepare) (prepare)])
        composite
        (compiled/compose
         {:id :semantic-two-layers
          :components [{:id :first :program first}
                       {:id :second :program second}]
          :connections [{:from [:first :y] :to [:second :x]}]
          :shares [[[:first :w] [:second :w]]]
          :outputs [{:key :result :from [:second :y]}]})
        plan (compiled/plan composite)
        first-y (get-in first [:out-tree 0 :node])
        second-x (get-in second [:in-tree 0 :node])
        mapping (get-in composite [:lowering :certificate :node-mapping])]
    (is (compiled/prepared? first))
    (is (compiled/prepared? composite))
    (is (= (get mapping [:first first-y]) (get mapping [:second second-x])))
    (is (= 4 (count (:nodes plan))))
    (is (= [[:first :x] [:first :w]] (mapv :key (:in-tree composite))))
    (is (= [:result] (mapv :key (:out-tree composite))))
    (is (= 2 (count (:instances plan))))
    (is (= 2 (count (compiled/ir composite))))
    (is (= {:map 2} (:steps (compiled/cache-key composite))))))

(deftest already-instantiated-artifacts-are-too-late-to-compose-zero-copy
  (is (= :compiled-composition-component
         (:reason
          (ex-data
           (try
             (compiled/compose
              {:id :late :components [{:id :late :program (compiled/map->Compiled {})}]
               :outputs []})
             (catch clojure.lang.ExceptionInfo error error)))))))

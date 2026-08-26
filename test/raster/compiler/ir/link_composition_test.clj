(ns raster.compiler.ir.link-composition-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-composition :as composition]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.resident-plan :as resident]))

(def ^:private kernel
  (artifact/make
   {:kernel-name "composition_axpy"
    :source "__kernel void composition_axpy(float* x, float* w, float* y, long n) {}"
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
   :steps [{:phase :map :kernel-name "composition_axpy" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(defn- lowered [id x w]
  (resident/lower {:id id :target :ze:0 :descriptor (descriptor)
                   :arguments [x w (alength x)] :roles {'w :constant}
                   :outputs ['y]}))

(defn- value-node [lowering symbol]
  (get-in lowering [:certificate :values symbol :node]))

(defn- composed [n]
  (let [weight (float-array n)
        first (lowered :first (float-array n) weight)
        second (lowered :second (float-array n) weight)
        first-y (value-node first 'y)
        second-x (value-node second 'x)
        first-w (value-node first 'w)
        second-w (value-node second 'w)
        second-y (value-node second 'y)]
    (composition/compose
     {:id :two-layers
      :components [{:id :first :lowering first}
                   {:id :second :lowering second}]
      :connections [{:from [:first first-y] :to [:second second-x]}]
      :shares [[[:first first-w] [:second second-w]]]
      :outputs [[:second second-y]]})))

(deftest certified-plans-compose-before-allocation-through-one-node-identity
  (let [lowering (composed 16)
        plan (:plan lowering)
        certificate (:certificate lowering)
        first-y (get (:node-mapping certificate) [:first [:first 'y]])
        second-x (get (:node-mapping certificate) [:second [:second 'x]])
        first-w (get (:node-mapping certificate) [:first [:first 'w]])
        second-w (get (:node-mapping certificate) [:second [:second 'w]])]
    (is (composition/certified-composition? lowering))
    (is (identical? lowering (composition/verify! lowering)))
    (is (link/link-plan? plan))
    (is (= 2 (count (:instances plan))))
    (is (= 4 (count (:nodes plan))) "connection and shared weight each remove one allocation")
    (is (= first-y second-x) "the intermediate is one node, not two buffers plus a copy")
    (is (= first-w second-w) "shared constants retain one allocation identity")
    (is (= :internal (get-in plan [:nodes first-y :role])))
    (is (= first-y (get-in plan [:instances 1 :bindings 'x])))
    (is (= first-w (get-in plan [:instances 1 :bindings 'w])))
    (is (= [(get (:node-mapping certificate) [:second [:second 'y]])]
           (:outputs plan)))))

(deftest composition-is-fail-loud-at-semantic-boundaries
  (testing "a connected view contract must be exact"
    (let [weight (float-array 16)
          producer (lowered :producer (float-array 16) weight)
          consumer (lowered :consumer (float-array 8) (float-array 8))]
      (is (= :link-composition-view-contract
             (:reason
              (ex-data
               (try
                 (composition/compose
                  {:id :bad-shape
                   :components [{:id :producer :lowering producer}
                                {:id :consumer :lowering consumer}]
                   :connections [{:from [:producer (value-node producer 'y)]
                                  :to [:consumer (value-node consumer 'x)]}]
                   :outputs [[:consumer (value-node consumer 'y)]]})
                 (catch clojure.lang.ExceptionInfo error error))))))))
  (testing "raw plans cannot bypass source-lowering verification"
    (let [lowering (lowered :one (float-array 4) (float-array 4))]
      (is (= :link-composition-component-type
             (:reason
              (ex-data
               (try
                 (composition/compose
                  {:id :uncertified
                   :components [{:id :one :lowering (:plan lowering)}]
                   :outputs [[:one (value-node lowering 'y)]]})
                 (catch clojure.lang.ExceptionInfo error error)))))))))

(deftest composition-certificate-detects-target-plan-drift
  (let [lowering (composed 8)
        changed (assoc-in lowering [:plan :outputs] [])]
    (is (= :link-composition-certificate
           (:reason
            (ex-data
             (try (composition/verify! changed)
                  (catch clojure.lang.ExceptionInfo error error))))))))

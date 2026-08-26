(ns raster.compiler.ir.link-composition-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
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

(defn- composite-abstract []
  (av/tensor {:dtype :float :shape ['n]
              :representation {:kind :test-split}}))

(defn- composite-layout []
  {:kind :ordered-fields :field-order [:data :index]})

(def ^:private split-kernel
  (artifact/make
   {:kernel-name "composition_split"
    :source "__kernel void composition_split(const float* x, float* out_data, int* out_index, long n) {}"
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'out_data :output :float :binding 'out :field :data)
          (kabi/slot 'out_index :output :int :binding 'out :field :index)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x out_data out_index n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x] :writes '[out_data out_index]}}))

(def ^:private join-kernel
  (artifact/make
   {:kernel-name "composition_join"
    :source "__kernel void composition_join(const float* in_data, const int* in_index, float* y, long n) {}"
    :abi [(kabi/slot 'in_data :input :float :binding 'in :field :data)
          (kabi/slot 'in_index :input :int :binding 'in :field :index)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[in_data in_index y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[in_data in_index] :writes '[y]}}))

(defn- split-descriptor []
  {:dtype :float
   :all-params '[x n] :array-params '[x] :scalar-params '[n]
   :array-roles {'x :input}
   :allocs [{:sym 'out :abstract (composite-abstract)
             :physical-layout (composite-layout)
             :leaves [{:field :data :dtype :float
                       :size-fn (fn [args] (long (nth args 1)))}
                      {:field :index :dtype :int
                       :size-fn (fn [args] (long (nth args 1)))}]}]
   :steps [{:phase :split :convention :map :artifact split-kernel
            :logical-bindings? true
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :output :sym 'out}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 1)))}]}]
   :result-sym 'out})

(defn- join-descriptor []
  {:dtype :float
   :all-params '[in n] :array-params '[in] :scalar-params '[n]
   :array-roles {'in :input}
   :value-specs {'in {:abstract (composite-abstract)
                      :physical-layout (composite-layout)
                      :leaves [{:field :data :dtype :float}
                               {:field :index :dtype :int}]}}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 1)))}]
   :steps [{:phase :join :convention :map :artifact join-kernel
            :logical-bindings? true
            :argument-specs [{:kind :input :sym 'in}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 1)))}]}]
   :result-sym 'y})

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
        first-y-value (get (:value-mapping certificate) [:first [:first 'y]])
        first-w-value (get (:value-mapping certificate) [:first [:first 'w]])
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
    (is (= first-y-value (get-in plan [:instances 1 :bindings 'x])))
    (is (= first-w-value (get-in plan [:instances 1 :bindings 'w])))
    (is (= [(get (:node-mapping certificate) [:second [:second 'y]])]
           (:outputs plan)))))

(deftest composite-values-compose-atomically-before-allocation
  (let [n 16
        producer (resident/lower {:id :split :target :ze:0
                                  :descriptor (split-descriptor)
                                  :arguments [(float-array n) n]})
        consumer (resident/lower {:id :join :target :ze:0
                                  :descriptor (join-descriptor)
                                  :arguments [{:data (float-array n)
                                               :index (int-array n)} n]})
        producer-value (get-in producer [:certificate :bindings 'out])
        consumer-value (get-in consumer [:certificate :bindings 'in])
        producer-leaves (get-in producer [:plan :values producer-value :leaves])
        consumer-leaves (get-in consumer [:plan :values consumer-value :leaves])
        lowering (composition/compose
                  {:id :split-join
                   :components [{:id :producer :lowering producer}
                                {:id :consumer :lowering consumer}]
                   :connections [{:from [:producer producer-value]
                                  :to [:consumer consumer-value]}]
                   :outputs [[:consumer (get-in consumer [:certificate :bindings 'y])]]})
        plan (:plan lowering)
        certificate (:certificate lowering)
        canonical-value (get (:value-mapping certificate) [:producer producer-value])]
    (is (= [:data :index] (mapv :name producer-leaves)))
    (is (= 2 (count (get-in producer [:certificate :values 'out :leaves]))))
    (is (= canonical-value
           (get (:value-mapping certificate) [:consumer consumer-value])))
    (is (= canonical-value (get-in plan [:instances 1 :bindings 'in])))
    (is (= (mapv #(get (:node-mapping certificate) [:producer (:node %)]) producer-leaves)
           (mapv #(get (:node-mapping certificate) [:consumer (:node %)]) consumer-leaves)))
    (is (= 4 (count (:nodes plan)))
        "the two consumer leaves are unified with producer storage, never copied")
    (is (= :internal
           (get-in plan [:nodes (-> plan :values (get canonical-value) :leaves first :node)
                         :role])))
    (is (composition/certified-composition? (composition/verify! lowering)))))

(deftest composite-constant-leaves-share-one-atomic-value
  (let [n 8
        data (float-array n)
        index (int-array n)
        argument {:data data :index index}
        left (resident/lower {:id :left :target :ze:0 :descriptor (join-descriptor)
                              :arguments [argument n] :roles {'in :constant}})
        right (resident/lower {:id :right :target :ze:0 :descriptor (join-descriptor)
                               :arguments [argument n] :roles {'in :constant}})
        left-in (get-in left [:certificate :bindings 'in])
        right-in (get-in right [:certificate :bindings 'in])
        lowering (composition/compose
                  {:id :shared-composite
                   :components [{:id :left :lowering left} {:id :right :lowering right}]
                   :shares [[[:left left-in] [:right right-in]]]
                   :outputs [[:left (get-in left [:certificate :bindings 'y])]
                             [:right (get-in right [:certificate :bindings 'y])]]})
        plan (:plan lowering)
        certificate (:certificate lowering)
        canonical (get (:value-mapping certificate) [:left left-in])]
    (is (= canonical (get (:value-mapping certificate) [:right right-in])))
    (is (= canonical (get-in plan [:instances 1 :bindings 'in])))
    (is (= 4 (count (:nodes plan))) "two physical constant leaves are allocated only once")
    (is (= [data index]
           (mapv #(get-in plan [:nodes (:node %) :source])
                 (get-in plan [:values canonical :leaves]))))))

(deftest certified-compositions-remain-logical-components
  (let [pair (composed 8)
        third (lowered :third (float-array 8) (float-array 8))
        pair-output (first (link/output-value-ids (:plan pair)))
        third-input (get-in third [:certificate :bindings 'x])
        third-output (get-in third [:certificate :bindings 'y])
        lowering (composition/compose
                  {:id :three-layers
                   :components [{:id :pair :lowering pair} {:id :third :lowering third}]
                   :connections [{:from [:pair pair-output] :to [:third third-input]}]
                   :outputs [[:third third-output]]})]
    (is (= 3 (count (get-in lowering [:plan :instances]))))
    (is (= 1 (count (link/output-value-ids (:plan lowering)))))
    (is (identical? lowering (composition/verify! lowering)))))

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

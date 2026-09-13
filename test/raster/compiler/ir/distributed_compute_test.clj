(ns raster.compiler.ir.distributed-compute-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]))

(def ^:private copy-kernel
  (artifact/make
   {:kernel-name "distributed_copy"
    :source (str "__kernel void distributed_copy(__global const float* x, "
                 "__global const float* weights, __global float* y, long n) {}")
    :abi [(abi/slot 'x :input :float)
          (abi/slot 'weights :input :float)
          (abi/slot 'y :output :float)
          (abi/slot 'n :scalar :long)]
    :arguments '[x weights y n]
    :launch (launch/spec {:workgroup-size [8]
                          :group-count [(launch/ceil-div 'n 8)]})
    :effects {:kind :map :reads '[x weights] :writes '[y]}}))

(def ^:private copy-descriptor
  {:dtype :float
   :all-params '[x weights n]
   :array-params '[x weights]
   :scalar-params '[n]
   :array-roles {'x :input 'weights :constant}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :copy :kernel-name "distributed_copy" :convention :map
            :artifact copy-kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'weights}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(defn- local-abstract
  ([shape] (local-abstract :float shape))
  ([dtype shape]
   (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}})))

(defn- local-link-plan
  ([] (local-link-plan {}))
  ([opts]
   (let [{:keys [x-shape x-dtype target]
          :or {x-shape [2 3] x-dtype :float target :gpu-0}} opts
         weights-source (if (contains? opts :weights-source)
                          (:weights-source opts)
                          (float-array 6))
         node (fn [id role source]
                (link/node {:id id :dtype :float :shape [6] :device target :role role
                            :source source}))
         x (node :x-node :input nil)
         weights (node :weights-node :constant weights-source)
         y (node :y-node :output nil)]
     (link/make
      {:id :local-copy :target target :nodes [x weights y]
       :values [(link/value {:id :local-x :abstract (local-abstract x-dtype x-shape)
                             :leaves [{:name :value :node :x-node}]})
                (link/value {:id :local-weights :abstract (local-abstract [2 3])
                             :leaves [{:name :value :node :weights-node}]})
                (link/value {:id :local-y :abstract (local-abstract [2 3])
                             :leaves [{:name :value :node :y-node}]})]
       :instances [(link/instance {:id :copy :descriptor copy-descriptor
                                   :bindings {'x :local-x
                                              'weights :local-weights
                                              'y :local-y}
                                   :scalars {'n 6}})]
       :outputs [:y-node]}))))

(defn- topology []
  (distributed/topology
   [(distributed/device {:id :gpu-0 :memory-capacity-bytes 1048576})
    (distributed/device {:id :gpu-1 :memory-capacity-bytes 1048576})]
   []))

(defn- global-value []
  (av/tensor {:dtype :float :shape [2 3]
              :representation {:kind :plain}
              :sharding {:kind :replicated :devices [:gpu-0 :gpu-1]}
              :ownership :owned}))

(defn- shards [value prefix]
  [(distributed/shard {:id (keyword (str (name prefix) "-0"))
                       :value value :device :gpu-0 :offsets [0 0]
                       :shape [2 3] :ownership :replica})
   (distributed/shard {:id (keyword (str (name prefix) "-1"))
                       :value value :device :gpu-1 :offsets [0 0]
                       :shape [2 3] :ownership :replica})])

(defn- device-plans
  ([link-plan] (device-plans link-plan {}))
  ([link-plan bindings]
   {:gpu-0
    {:entries {:copy {:link-plan link-plan}}
     :steps {:copy-0
             {:entry :copy
              :bindings (merge {:local-x {:value :x :shard :x-0}
                                :local-y {:value :y :shard :y-0}}
                               bindings)}}}}))

(defn- plan-map
  ([] (plan-map {}))
  ([opts]
   (let [link-plan (or (:link-plan opts) (local-link-plan))
         values (or (:values opts)
                    {:x (global-value) :weights (global-value) :y (global-value)})
         shard-map (or (:shards opts)
                       {:x (shards :x :x)
                        :weights (shards :weights :weights)
                        :y (shards :y :y)})
         steps (or (:steps opts)
                   [(distributed/compute-step
                     {:id :copy-0 :device :gpu-0 :duration-ns 10})
                    (distributed/compute-step
                     {:id :analytical-only :device :gpu-1 :duration-ns 20
                      :dependencies [:copy-0]})])]
     {:id :distributed-copy
      :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
      :topology (topology)
      :values values :shards shard-map
      :device-plans (or (:device-plans opts) (device-plans link-plan))
      :steps steps :outputs [:analytical-only]})))

(defn- make-plan
  ([] (distributed/plan (plan-map)))
  ([overrides] (distributed/plan (plan-map overrides))))

(defn- failure-reason [thunk]
  (:reason (ex-data (try (thunk) (catch clojure.lang.ExceptionInfo error error)))))

(defn- explicit-domain [shape]
  {:local-shape shape
   :placements [{:kind :owned :value :x :shard :x-0 :local-offsets [0 0]}]})

(deftest flat-local-storage-has-an-explicit-multidimensional-domain
  (let [local (local-link-plan {:x-shape [6]})
        plans (device-plans local {:local-x (explicit-domain [2 3])})
        bound (get-in (distributed/compute-bindings (make-plan {:device-plans plans}))
                      [:bindings :copy-0 :values :local-x])]
    (is (= :read (:access bound)))
    (is (= [6] (get-in bound [:leaves 0 :view :shape]))
        "the compiled ABI leaf is retained, not silently rewritten")
    (is (= [2 3] (get-in bound [:domain :view :shape])))
    (is (= [3 1] (get-in bound [:domain :view :strides])))
    (is (= (get-in bound [:leaves 0 :view :allocation])
           (get-in bound [:domain :view :allocation])))
    (is (= (get-in bound [:domain :view :byte-length])
           (get-in bound [:domain :placements 0 :view :byte-length])))
    (is (= :distributed-compute-value-contract
           (failure-reason #(make-plan {:link-plan local})))
        "without explicit coordinates the old exact-shape contract is unchanged")))

(deftest local-domain-does-not-admit-unproven-layouts-or-padding
  (let [local (local-link-plan {:x-shape [6]})
        check (fn [plan reference]
                (failure-reason #(make-plan {:device-plans
                                            (device-plans plan {:local-x reference})})))]
    (is (= :distributed-compute-value-contract (check local (explicit-domain [3 3])))))
  (let [local (local-link-plan {:x-shape [6]})
        plans (device-plans local {:local-x (explicit-domain [2 3])})]
    (doseq [[changed expected]
            [[(assoc-in local [:values :local-x :physical-layout] {:kind :packed})
              :distributed-compute-local-domain]
             [(-> local
                  (assoc-in [:nodes :x-node :view :strides] [0])
                  (assoc-in [:nodes :x-node :view :byte-length] 4))
              :link-noncontiguous-binding]]]
      (is (= expected
             (failure-reason #(make-plan {:device-plans
                                         (assoc-in plans [:gpu-0 :entries :copy :link-plan] changed)})))))
    (is (= :buffer-view-region
           (failure-reason #(make-plan {:device-plans
                                       (assoc-in plans [:gpu-0 :steps :copy-0 :bindings :local-x
                                                        :placements 0 :local-offsets] [1 0])}))))
    (is (= :distributed-compute-placement-coverage
           (failure-reason #(make-plan {:device-plans
                                       (assoc-in plans [:gpu-0 :steps :copy-0 :bindings :local-x
                                                        :placements] [{:kind :replica :transfer :missing}])}))))))

(deftest compute-bindings-retain-exact-link-values-and-derived-accesses
  (let [plan (make-plan)
        report (distributed/compute-bindings plan)
        binding (get-in report [:bindings :copy-0])]
    (is (= [:analytical-only] (:unbound report)))
    (is (= :copy (:entry binding)))
    (is (= :local-copy (get-in binding [:link-plan :id])))
    (is (= {:local-x :read :local-y :write}
           (update-vals (:values binding) :access)))
    (is (= {:value :x :shard :x-0}
           (select-keys (get-in binding [:values :local-x]) [:value :shard])))
    (is (= :x-node
           (get-in binding [:values :local-x :leaves 0 :view :id])))
    (is (= [6]
           (get-in binding [:values :local-x :leaves 0 :view :shape])))))

(deftest bindings-are-qualified-by-global-value-and-exact-shard-shape
  (testing "same-volume transposition is not a shard-shape proof"
    (is (= :distributed-compute-value-contract
         (failure-reason
          #(make-plan {:link-plan (local-link-plan {:x-shape [3 2]})})))))
  (testing "storage dtype must agree with the global value"
    (is (= :distributed-compute-value-contract
         (failure-reason
          #(make-plan {:values (assoc (:values (plan-map)) :x
                                      (assoc (global-value) :dtype :double))})))))
  (testing "a shard on another device cannot satisfy the local compute binding"
    (let [plans (device-plans (local-link-plan))
          plans (assoc-in plans [:gpu-0 :steps :copy-0 :bindings :local-x :shard] :x-1)]
      (is (= :distributed-compute-shard-device
             (failure-reason #(make-plan {:device-plans plans})))))))

(deftest repeated-shards-require-one-structural-physical-realization
  (let [local (local-link-plan)
        base (plan-map)
        plans (-> (device-plans local)
                  (assoc-in [:gpu-0 :entries :again] {:link-plan local})
                  (assoc-in [:gpu-0 :steps :analytical-only]
                            (assoc (get-in (device-plans local) [:gpu-0 :steps :copy-0])
                                   :entry :again)))
        plan (assoc base :device-plans plans
                    :steps (mapv #(assoc % :device :gpu-0) (:steps base)))]
    (is (empty? (:unbound (distributed/compute-bindings (distributed/plan plan)))))
    (is (= :distributed-compute-shard-storage
           (failure-reason
            #(distributed/plan
              (assoc-in plan [:device-plans :gpu-0 :entries :again :link-plan
                              :nodes :x-node :view :allocation :id] :different-allocation)))))
    (is (= :distributed-compute-entry-device
           (failure-reason
            #(make-plan {:device-plans (device-plans (local-link-plan {:target :gpu-1}))}))))))

(deftest distinct-shards-cannot-alias-across-entries
  (let [local (assoc-in (local-link-plan)
                        [:nodes :x-node :view :allocation :byte-size] 48)
        base (plan-map)
        plans (-> (device-plans local)
                  (assoc-in [:gpu-0 :entries :again] {:link-plan local})
                  (assoc-in [:gpu-0 :steps :analytical-only]
                            {:entry :again
                             :bindings {:local-x {:value :weights :shard :weights-0}
                                        :local-y {:value :y :shard :y-0}}}))
        plan (assoc base :device-plans plans
                    :steps (mapv #(assoc % :device :gpu-0) (:steps base)))]
    (is (= :distributed-compute-shard-alias
           (failure-reason #(distributed/plan plan))))
    (testing "disjoint ranges of the same allocation remain distinct legal shards"
      (is (distributed/distributed-plan?
           (distributed/plan
            (assoc-in plan [:device-plans :gpu-0 :entries :again :link-plan
                           :nodes :x-node :view :byte-offset] 24)))))))

(deftest declared-global-memory-space-constrains-physical-leaves
  (let [local (local-link-plan)
        actual (get-in local [:nodes :x-node :view :allocation :memory-space])
        base (plan-map {:link-plan local})]
    (is (distributed/distributed-plan? (distributed/plan base)))
    (is (distributed/distributed-plan?
         (distributed/plan (assoc-in base [:values :x :memory-space] actual))))
    (is (= :distributed-compute-memory-space
           (failure-reason
            #(distributed/plan
              (assoc-in base [:values :x :memory-space]
                        (if (= actual :host) :device :host))))))))

(deftest shared-entry-access-facts-are-derived-once-per-report
  (let [base (plan-map)
        call (get-in base [:device-plans :gpu-0 :steps :copy-0])
        plan (distributed/plan
              (-> base
                  (assoc-in [:device-plans :gpu-0 :steps :analytical-only] call)
                  (update :steps #(mapv (fn [step] (assoc step :device :gpu-0)) %))))
        calls (atom 0)
        original link/value-accesses]
    (with-redefs [link/value-accesses (fn [local] (swap! calls inc) (original local))]
      (is (empty? (:unbound (distributed/compute-bindings plan))))
      (is (= 1 @calls)))))

(deftest bound-shards-cannot-hide-private-aliases
  (let [local (local-link-plan)
        scratch (-> (get-in local [:nodes :x-node])
                    (assoc :id :private-node :role :scratch)
                    (assoc-in [:view :id] :private-node))
        local (-> local
                  (assoc-in [:nodes :private-node] scratch)
                  (assoc-in [:values :private]
                            (link/value {:id :private :abstract (local-abstract [2 3])
                                         :leaves [{:name :value :node :private-node}]}))
                  (assoc :aliases #{#{:x-node :private-node}})
                  (update :instances conj
                          (link/instance {:id :private-write :descriptor copy-descriptor
                                          :bindings {'x :local-y 'weights :local-weights
                                                     'y :private}
                                          :scalars {'n 6}})))]
    (is (link/link-plan? (link/validate! local)))
    (is (= {:local-x :read :local-weights :read :local-y :read-write :private :write}
           (link/value-accesses local)))
    (is (= :distributed-compute-private-alias
           (failure-reason #(make-plan {:link-plan local}))))
    (testing "private scratch can occupy a disjoint range in the same allocation"
      (is (distributed/distributed-plan?
           (make-plan {:link-plan
                       (-> local
                           (assoc :aliases #{})
                           (assoc-in [:nodes :x-node :view :allocation :byte-size] 48)
                           (assoc-in [:nodes :private-node :view :allocation :byte-size] 48)
                           (assoc-in [:nodes :private-node :view :byte-offset] 24))}))))))

(deftest misspelled-local-contract-keys-do-not-silently-declare-analytical-compute
  (let [base (device-plans (local-link-plan))
        misspelled (-> base
                       (assoc-in [:gpu-0 :step] (get-in base [:gpu-0 :steps]))
                       (update :gpu-0 dissoc :steps))]
    (is (= :distributed-compute-device-keys
           (failure-reason #(make-plan {:device-plans misspelled}))))))

(deftest entry-step-value-and-shard-references-fail-closed
  (let [base (device-plans (local-link-plan))]
    (doseq [[label plans]
            [[:entry (assoc-in base [:gpu-0 :steps :copy-0 :entry] :missing)]
             [:step (-> base
                        (assoc-in [:gpu-0 :steps :missing]
                                  (get-in base [:gpu-0 :steps :copy-0]))
                        (update-in [:gpu-0 :steps] dissoc :copy-0))]
             [:local-value
              (-> base
                  (assoc-in [:gpu-0 :steps :copy-0 :bindings :missing]
                            (get-in base [:gpu-0 :steps :copy-0 :bindings :local-x]))
                  (update-in [:gpu-0 :steps :copy-0 :bindings] dissoc :local-x))]
             [:global-value
              (assoc-in base [:gpu-0 :steps :copy-0 :bindings :local-x :value] :missing)]
             [:shard
              (assoc-in base [:gpu-0 :steps :copy-0 :bindings :local-x :shard] :missing)]]]
      (testing (name label)
        (is (keyword? (failure-reason #(make-plan {:device-plans plans}))))))))

(deftest required-boundary-values-and-duplicate-shards-are-checked
  (let [base (device-plans (local-link-plan))]
    (testing "an input cannot disappear from distributed dataflow"
      (is (keyword?
           (failure-reason
            #(make-plan {:device-plans
                         (update-in base [:gpu-0 :steps :copy-0 :bindings]
                                    dissoc :local-x)})))))
    (testing "two local values cannot claim one qualified shard"
      (is (keyword?
           (failure-reason
            #(make-plan {:device-plans
                         (assoc-in base [:gpu-0 :steps :copy-0 :bindings :local-y]
                                   {:value :x :shard :x-0})})))))
    (testing "a sourced constant is local unless deliberately exposed"
      (is (distributed/distributed-plan? (make-plan)))
      (let [plans (device-plans
                   (local-link-plan)
                   {:local-weights {:value :weights :shard :weights-0}})]
        (is (= :read
               (get-in (distributed/compute-bindings
                        (make-plan {:device-plans plans}))
                       [:bindings :copy-0 :values :local-weights :access])))))
    (testing "an unsourced constant is a required distributed input"
      (let [link-plan (local-link-plan {:weights-source nil})]
        (is (keyword? (failure-reason #(make-plan {:link-plan link-plan}))))
        (is (distributed/distributed-plan?
             (make-plan
              {:link-plan link-plan
               :device-plans
               (device-plans link-plan
                             {:local-weights {:value :weights :shard :weights-0}})})))))))

(deftest certificate-detects-compute-binding-mutation
  (let [plan (make-plan)
        certified (distributed/certify plan)
        ;; The swap remains a valid plan because x and y have identical storage contracts, but it
        ;; changes which global shard is read and written by the certified local executable.
        modified (-> certified
                     (assoc-in [:plan :device-plans :gpu-0 :steps :copy-0
                                :bindings :local-x]
                               {:value :y :shard :y-0})
                     (assoc-in [:plan :device-plans :gpu-0 :steps :copy-0
                                :bindings :local-y]
                               {:value :x :shard :x-0}))]
    (is (distributed/distributed-plan? (distributed/validate! (:plan modified))))
    (is (= :distributed-certificate
           (failure-reason #(distributed/verify! modified))))))

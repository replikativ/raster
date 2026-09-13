(ns raster.compiler.ir.distributed-compute-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.scan :as scan]))

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
   (let [{:keys [x-shape x-dtype target elements tail-shape]
          :or {x-shape [2 3] x-dtype :float target :gpu-0 elements 6 tail-shape [2 3]}} opts
         weights-source (if (contains? opts :weights-source)
                          (:weights-source opts)
                          (float-array elements))
         node (fn [id role source]
                (link/node {:id id :dtype :float :shape [elements] :device target :role role
                            :source source}))
         x (node :x-node :input nil)
         weights (node :weights-node :constant weights-source)
         y (node :y-node :output nil)]
     (link/make
      {:id :local-copy :target target :nodes [x weights y]
       :values [(link/value {:id :local-x :abstract (local-abstract x-dtype x-shape)
                             :leaves [{:name :value :node :x-node}]})
                (link/value {:id :local-weights :abstract (local-abstract tail-shape)
                             :leaves [{:name :value :node :weights-node}]})
                (link/value {:id :local-y :abstract (local-abstract tail-shape)
                             :leaves [{:name :value :node :y-node}]})]
       :instances [(link/instance {:id :copy :descriptor copy-descriptor
                                   :bindings {'x :local-x
                                              'weights :local-weights
                                              'y :local-y}
                                   :scalars {'n elements}})]
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
    (is (= :distributed-compute-local-domain (check local (explicit-domain [3 3])))))
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

(defn- periodic-materialization-plan []
  (let [global (assoc (global-value) :shape [2 2]
                      :sharding {:kind :partitioned :axis 0 :devices [:gpu-0 :gpu-1]})
        shards (mapv (fn [i]
                       (distributed/shard {:id (keyword (str "x-" i)) :value :x
                                           :device (keyword (str "gpu-" i))
                                           :offsets [i 0] :shape [1 2] :ownership :owned}))
                     (range 2))
        halo (distributed/schedule-halo
              (distributed/halo-exchange {:id :periodic :value :x :axis 0 :width 1
                                          :boundary :periodic})
              global shards {[:gpu-0 :gpu-1] [:forward] [:gpu-1 :gpu-0] [:backward]} [])
        replicas (mapv (fn [step] {:kind :replica :transfer (:id step)})
                       (filter #(= :gpu-0 (:target %)) (:steps halo)))
        reference {:local-shape [3 2]
                   :placements (into [{:kind :owned :value :x :shard :x-0 :local-offsets [1 0]}]
                                     replicas)}
        base (plan-map {:device-plans (device-plans (local-link-plan {:x-shape [6]})
                                                    {:local-x reference})})]
    (-> base
        (assoc :halos [halo]
               :topology (distributed/topology
                           (vals (:devices (topology)))
                           [(distributed/link {:id :forward :source :gpu-0 :target :gpu-1
                                               :bandwidth-bytes-s 1.0e9 :latency-ns 100})
                            (distributed/link {:id :backward :source :gpu-1 :target :gpu-0
                                               :bandwidth-bytes-s 1.0e9 :latency-ns 100})]))
        (assoc-in [:values :x] global)
        (assoc-in [:shards :x] shards)
        (update :steps (fn [steps]
                         (into (:steps halo)
                               (assoc-in steps [0 :dependencies] (:completions halo))))))))

(deftest periodic-replicas-project-from-the-scheduled-halo
  (let [plan (periodic-materialization-plan)
        report (distributed/compute-bindings (distributed/plan plan))
        placements (get-in report [:bindings :copy-0 :values :local-x :domain :placements])]
    (is (= #{[0 0] [1 0] [2 0]} (set (map #(get-in % [:region :offsets]) placements))))
    (is (= [[1 2] [1 2] [1 2]] (mapv #(get-in % [:view :shape]) placements)))
    (is (= #{0 8 16} (set (map #(get-in % [:view :byte-offset]) placements))))
    (is (= [:analytical-only] (:unbound report)))
    (testing "a gap cannot be treated as an initialized boundary"
      (is (= :distributed-compute-placement-coverage
             (failure-reason #(distributed/plan
                               (update-in plan [:device-plans :gpu-0 :steps :copy-0
                                                :bindings :local-x :placements] pop))))))
    (testing "repeated replicas do not prove coverage"
      (is (= :distributed-compute-placement-coverage
             (failure-reason #(distributed/plan
                               (update-in plan [:device-plans :gpu-0 :steps :copy-0
                                                :bindings :local-x :placements]
                                          (fn [p] (assoc p 2 (nth p 1)))))))))
    (testing "vector order is not a producer dependency"
      (is (= :distributed-compute-replica-dependency
             (failure-reason #(distributed/plan
                               (update plan :steps
                                       (fn [steps] (mapv (fn [s] (if (= :copy-0 (:id s))
                                                                 (assoc s :dependencies []) s)) steps))))))))
    (testing "a transfer to the other shard is not a local replica"
      (is (= :distributed-compute-replica-transfer
             (failure-reason #(distributed/plan
                               (assoc-in plan [:device-plans :gpu-0 :steps :copy-0 :bindings
                                               :local-x :placements 1 :transfer]
                                         [:periodic :edge 0 :forward]))))))))

(deftest replica-transfers-are-transitive-predecessors-not-combining-writes
  (let [plan (periodic-materialization-plan)
        halo (first (:halos plan))
        compute (filterv #(= :compute (:kind %)) (:steps plan))
        relay (distributed/compute-step {:id :relay :device :gpu-0 :duration-ns 1
                                         :dependencies (:completions halo)})
        transitive (assoc plan :steps (into (conj (:steps halo) relay)
                                            (assoc-in compute [0 :dependencies] [:relay])))]
    (is (distributed/distributed-plan? (distributed/plan transitive)))
    (let [combine (scan/certify {:acc 'acc :init 0.0 :lambda '(+ acc element)} :float)
          combining (distributed/schedule-halo
                     (assoc (:exchange halo) :combine combine)
                     (get-in plan [:values :x]) (get-in plan [:shards :x]) (:routes halo) [])
          changed (assoc plan :halos [combining] :steps (into (:steps combining) compute))]
      (is (= :distributed-compute-replica-transfer
             (failure-reason #(distributed/plan changed)))
          "even a certified reduction cannot be silently implemented as a copy replica"))))

(defn- boundary-fill-plan [physical-view]
  (let [kernel (artifact/make
                {:kernel-name "fill_boundary" :source "__kernel void fill_boundary(__global float* y) { y[get_global_id(0)] = 0; }"
                 :abi [(abi/slot 'y :output :float)] :arguments '[y]
                 :launch (launch/spec {:workgroup-size [1] :group-count [2]})
                 :effects {:kind :map :reads [] :writes '[y]}})
        descriptor {:dtype :float :all-params [] :array-params [] :scalar-params []
                    :allocs [{:sym 'y :dtype :float :size-fn (fn [_] 2)}]
                    :steps [{:phase :fill :kernel-name "fill_boundary" :convention :map
                             :artifact kernel :argument-specs [{:kind :output :sym 'y}]}]
                    :result-sym 'y}]
    (link/make {:id :boundary-fill :target (get-in physical-view [:allocation :device])
                :nodes [(link/node {:id :boundary-node :role :output :view physical-view})]
                :values [(link/value {:id :boundary-output :abstract (local-abstract [2])
                                      :leaves [{:name :value :node :boundary-node}]})]
                :instances [(link/instance {:id :fill :descriptor descriptor
                                            :bindings {'y :boundary-output} :scalars {}})]
                :outputs [:boundary-node]})))

(defn- boundary-materialization-plan []
  (let [base (periodic-materialization-plan)
        old (first (:halos base))
        halo (distributed/schedule-halo (assoc (:exchange old) :boundary :nonperiodic)
                                        (get-in base [:values :x]) (get-in base [:shards :x])
                                        (:routes old) [])
        consumer (get-in base [:device-plans :gpu-0 :entries :copy :link-plan])
        boundary-view (view/subview (get-in consumer [:nodes :x-node :view]) {:shape [2]})
        init (distributed/compute-step {:id :boundary-init :device :gpu-0 :duration-ns 1})
        computes (filterv #(= :compute (:kind %)) (:steps base))]
    (-> base
        (assoc :halos [halo] :steps (into (into [init] (:steps halo))
                                         (assoc-in computes [0 :dependencies]
                                                   (conj (:completions halo) :boundary-init))))
        (assoc-in [:device-plans :gpu-0 :entries :boundary] {:link-plan (boundary-fill-plan boundary-view)})
        (assoc-in [:device-plans :gpu-0 :steps :boundary-init] {:entry :boundary :bindings {}})
        (assoc-in [:device-plans :gpu-0 :steps :copy-0 :bindings :local-x :placements 2]
                  {:kind :boundary :region {:offsets [0 0] :shape [1 2]}
                   :provider {:step :boundary-init :local-value :boundary-output}}))))

(deftest nonperiodic-boundary-has-an-exact-executable-producer
  (let [plan (boundary-materialization-plan)
        report (distributed/compute-bindings (distributed/plan plan))]
    (is (= :write (get-in report [:bindings :boundary-init :boundary-outputs :boundary-output :access])))
    (is (empty? (get-in report [:bindings :boundary-init :values]))
        "a private boundary result is not a synthetic globally partitioned ValueShard")
    (is (= [:analytical-only] (:unbound report)))
    (is (= :distributed-compute-boundary-output
           (failure-reason #(distributed/plan
                             (assoc-in plan [:device-plans :gpu-0 :entries :boundary :link-plan
                                             :outputs] []))))
        "an internally written value is not a retained boundary output")
    (is (= :distributed-compute-boundary-write
           (failure-reason #(distributed/plan
                             (assoc-in plan [:device-plans :gpu-0 :steps :copy-0 :bindings :local-x
                                             :placements 2 :provider :local-value] :not-written)))))
    (is (= :distributed-compute-boundary-provider
           (failure-reason #(distributed/plan
                             (update plan :steps (fn [steps]
                                                   (mapv (fn [s] (if (= :copy-0 (:id s))
                                                                   (update s :dependencies pop) s)) steps)))))))
    (is (= :distributed-compute-boundary-storage
           (failure-reason #(distributed/plan
                             (assoc-in plan [:device-plans :gpu-0 :entries :boundary :link-plan
                                             :nodes :boundary-node :view :byte-offset] 8)))))))

(defn- fully-bound-periodic-plan []
  (let [base (periodic-materialization-plan)
        incoming (filter #(= :gpu-1 (:target %)) (get-in base [:halos 0 :steps]))]
    (assoc-in base [:device-plans :gpu-1]
              {:entries {:copy {:link-plan (local-link-plan {:x-shape [6] :target :gpu-1})}}
               :steps {:analytical-only
                       {:entry :copy
                        :bindings {:local-x {:local-shape [3 2]
                                             :placements (into [{:kind :owned :value :x :shard :x-1
                                                                 :local-offsets [1 0]}]
                                                               (map (fn [s] {:kind :replica :transfer (:id s)}) incoming))}
                                   :local-y {:value :y :shard :y-1}}}}})))

(deftest copy-transfer-endpoints-come-from-owned-and-replica-views
  (let [plan (distributed/plan (fully-bound-periodic-plan))
        endpoints (distributed/transfer-bindings plan)]
    (is (= 4 (count endpoints)))
    (doseq [[id {:keys [source target bytes]}] endpoints]
      (is (= 8 bytes))
      (is (= 8 (get-in source [:view :byte-offset]))
          "the source is the owned row, not the beginning of its padded allocation")
      (is (contains? #{0 16} (get-in target [:view :byte-offset])))
      (is (= id (:replica target)))
      (is (not= (:device source) (:device target)))
      (is (= [1 2] (get-in source [:view :shape]) (get-in target [:view :shape]))))
    (is (= :distributed-transfer-target
           (failure-reason #(distributed/transfer-bindings
                             (distributed/plan (periodic-materialization-plan)))))
        "an analytical plan may lack physical destinations, strict lowering may not")
    (is (= :distributed-transfer-source
           (failure-reason #(distributed/transfer-bindings
                             (assoc-in plan [:device-plans :gpu-0] {})))))
    (is (= {} (distributed/transfer-bindings (make-plan))))
    (let [generic (distributed/transfer-step {:id :generic :source :gpu-0 :target :gpu-1
                                              :route [:forward] :value :x :bytes 8})
          analytical (distributed/plan (update plan :steps conj generic))]
      (is (= :distributed-transfer-kind
             (failure-reason #(distributed/transfer-bindings analytical)))
          "an analytical transfer is not silently lowered as a halo copy"))))

(deftest unequal-shard-upper-faces-use-owned-relative-coordinates
  (let [base (fully-bound-periodic-plan)
        global (assoc (get-in base [:values :x]) :shape [6 2])
        shards [(assoc (get-in base [:shards :x 0]) :shape [2 2])
                (assoc (get-in base [:shards :x 1]) :offsets [2 0] :shape [4 2])]
        old (get-in base [:halos 0])
        halo (distributed/schedule-halo (:exchange old) global shards (:routes old) [])
        base (-> base (assoc-in [:values :x] global) (assoc-in [:shards :x] shards)
                 (assoc :halos [halo] :steps (into (:steps halo) (filter #(= :compute (:kind %)) (:steps base)))))
        plan (reduce
              (fn [p [device step rows result]]
                (let [n (* 2 (+ rows 2))
                      local (local-link-plan {:x-shape [n] :tail-shape [n] :elements n :target device})
                      value (assoc (global-value) :shape [n]
                                   :sharding {:kind :replicated :devices [device]})]
                  (-> p
                      (assoc-in [:values result] value)
                      (assoc-in [:shards result] [(distributed/shard {:id result :value result :device device
                                                                      :offsets [0] :shape [n] :ownership :replica})])
                      (assoc-in [:device-plans device :entries :copy :link-plan] local)
                      (assoc-in [:device-plans device :steps step :bindings :local-y] {:value result :shard result})
                      (assoc-in [:device-plans device :steps step :bindings :local-x :local-shape] [(+ rows 2) 2]))))
              base [[:gpu-0 :copy-0 2 :left-result] [:gpu-1 :analytical-only 4 :right-result]])
        endpoints (distributed/transfer-bindings (distributed/plan plan))]
    (is (= 16 (get-in endpoints [[:periodic :edge 0 :forward] :source :view :byte-offset]))
        "left upper face: one padding row plus one owned-relative row")
    (is (= 8 (get-in endpoints [[:periodic :edge 0 :backward] :source :view :byte-offset]))
        "right lower face: subtract global row two, then apply one padding row")
    (is (= 32 (get-in endpoints [[:periodic :edge 1 :forward] :source :view :byte-offset]))
        "right periodic upper face: padding plus three owned-relative rows")
    (is (= 24 (get-in endpoints [[:periodic :edge 0 :backward] :target :view :byte-offset])))
    (is (= 40 (get-in endpoints [[:periodic :edge 1 :backward] :target :view :byte-offset])))))

(defn- initialized-periodic-plan []
  (reduce (fn [plan device]
            (assoc-in plan [:device-plans device :entries :copy :link-plan :nodes :x-node :source]
                      (float-array 6)))
          (fully-bound-periodic-plan) [:gpu-0 :gpu-1]))

(deftest explicit-worker-placement-preserves-logical-shards-and-physical-alias-checks
  (let [base (initialized-periodic-plan)
        placed (reduce
                (fn [plan device]
                  (-> plan
                      (assoc-in [:device-plans device :target] :ze:0)
                      (update-in [:device-plans device :entries :copy :link-plan]
                                 (fn [local]
                                   (-> local (assoc :target :ze:0)
                                       (update :nodes
                                               #(update-vals %
                                                             (fn [node]
                                                               (-> node
                                                                   (assoc-in [:view :allocation :device] :ze:0)
                                                                   (update-in [:view :allocation :id] (fn [id] [device id])))))))))))
                base [:gpu-0 :gpu-1])
        ready (distributed/check-readiness (distributed/plan placed))]
    (is (= 6 (count (:actions ready))))
    (is (= #{:ze:0} (into #{} (map #(get-in % [:view :allocation :device])) (:initializers ready))))
    (is (= #{:gpu-0 :gpu-1} (set (get-in placed [:mesh :devices]))))
    (is (= :distributed-compute-entry-device
           (failure-reason #(distributed/plan (update-in placed [:device-plans :gpu-0] dissoc :target)))))
    (let [aliased (update-in placed [:device-plans :gpu-1 :entries :copy :link-plan :nodes]
                            #(update-vals % (fn [node]
                                              (assoc-in node [:view :allocation :id]
                                                        [:gpu-0 (second (get-in node [:view :allocation :id]))]))))]
      (is (= :distributed-compute-shard-alias (failure-reason #(distributed/plan aliased)))
          "different logical workers cannot hide overlapping physical shard storage"))))

(deftest readiness-composes-initializers-transfers-and-local-contracts
  (let [plan (distributed/plan (initialized-periodic-plan))
        ready (distributed/check-readiness plan)]
    (is (= 4 (count (:initializers ready))))
    (is (= 6 (count (:actions ready))))
    (is (= (mapv :id (:steps plan)) (mapv :id (:actions ready))))
    (is (= #{[8 8]}
           (into #{} (keep (fn [{:keys [view initializer]}]
                             (when (and initializer (= :x-node (get-in view [:allocation :id])))
                               [(:byte-offset view) (:byte-length view)]))) (:final-regions ready)))
        "ghost writes preserve the independently initialized owned row")
    (is (= :distributed-readiness-uninitialized
           (failure-reason #(distributed/check-readiness
                             (assoc-in plan [:device-plans :gpu-0 :entries :copy :link-plan
                                             :nodes :x-node :source] nil)))))
    (is (= :distributed-readiness-unbound
           (failure-reason #(distributed/check-readiness (make-plan)))))))

(deftest nonperiodic-readiness-requires-both-boundary-and-halo-producers
  (let [base (initialized-periodic-plan)
        old (get-in base [:halos 0])
        halo (distributed/schedule-halo (assoc (:exchange old) :boundary :nonperiodic)
                                        (get-in base [:values :x]) (get-in base [:shards :x]) (:routes old) [])
        providers [(distributed/compute-step {:id :lower-boundary :device :gpu-0 :duration-ns 1})
                   (distributed/compute-step {:id :upper-boundary :device :gpu-1 :duration-ns 1})]
        computes (filterv #(= :compute (:kind %)) (:steps base))
        base (assoc base :halos [halo]
                    :steps (into (into providers (:steps halo))
                                 (assoc-in computes [0 :dependencies]
                                           (into (:completions halo) (map :id providers)))))
        plan (reduce
              (fn [plan [device step provider row]]
                (let [base-view (get-in plan [:device-plans device :entries :copy :link-plan :nodes :x-node :view])
                      incoming (first (filter #(= device (:target %)) (:steps halo)))
                      owner (get-in plan [:device-plans device :steps step :bindings :local-x :placements 0])]
                  (-> plan
                      (assoc-in [:device-plans device :entries provider]
                                {:link-plan (boundary-fill-plan (view/subview base-view {:byte-offset (* row 8) :shape [2]}))})
                      (assoc-in [:device-plans device :steps provider] {:entry provider :bindings {}})
                      (assoc-in [:device-plans device :steps step :bindings :local-x :placements]
                                [owner {:kind :replica :transfer (:id incoming)}
                                 {:kind :boundary :region {:offsets [row 0] :shape [1 2]}
                                  :provider {:step provider :local-value :boundary-output}}]))))
              base [[:gpu-0 :copy-0 :lower-boundary 0] [:gpu-1 :analytical-only :upper-boundary 2]])
        report (distributed/check-readiness (distributed/plan plan))
        producers (into #{} (mapcat #(map :producer (:fresh %))) (:actions report))]
    (is (= 6 (count (:actions report))))
    (is (contains? producers :lower-boundary))
    (is (contains? producers :upper-boundary))))

(deftest readiness-rejects-unordered-physical-writes
  (let [plan (initialized-periodic-plan)
        call (get-in plan [:device-plans :gpu-0 :steps :copy-0])
        race (distributed/compute-step {:id :racing-copy :device :gpu-0 :duration-ns 1
                                         :dependencies (get-in plan [:halos 0 :completions])})
        plan (-> plan (update :steps conj race)
                 (assoc-in [:device-plans :gpu-0 :steps :racing-copy] call))]
    (is (distributed/distributed-plan? (distributed/plan plan)))
    (is (= :distributed-readiness-race
           (failure-reason #(distributed/check-readiness (distributed/plan plan)))))
    (is (map? (distributed/check-readiness
               (distributed/plan (update plan :steps
                                         (fn [steps] (mapv #(if (= :racing-copy (:id %))
                                                              (assoc % :dependencies [:copy-0]) %) steps)))))))))

(deftest readiness-orders-pass-through-preconditions-without-abi-reads
  (let [base (initialized-periodic-plan)
        output-view (get-in base [:device-plans :gpu-0 :entries :copy :link-plan :nodes :y-node :view])
        scratch (link/node {:id :unrelated :dtype :float :shape [2] :device :gpu-0 :role :internal})
        passthrough (-> (boundary-fill-plan (:view scratch))
                        (assoc-in [:nodes :pass] (link/node {:id :pass :role :input :view output-view}))
                        (assoc-in [:values :pass]
                                  (link/value {:id :pass :abstract (local-abstract [2 3])
                                               :leaves [{:name :value :node :pass}]}))
                        (assoc :outputs [:pass]))
        plan (-> base
                 (assoc-in [:device-plans :gpu-0 :entries :pass] {:link-plan passthrough})
                 (assoc-in [:device-plans :gpu-0 :steps :pass]
                           {:entry :pass :bindings {:pass {:value :y :shard :y-0}}})
                 (update :steps conj (distributed/compute-step
                                     {:id :pass :device :gpu-0 :duration-ns 1})))
        contract (link/initialization-contract passthrough)]
    (is (contains? (:requires contract) :pass))
    (is (not (contains? (:reads contract) :pass)))
    (is (= :distributed-readiness-race
           (failure-reason #(distributed/check-readiness (distributed/plan plan))))
        "vector order must not substitute for a producer dependency")
    (is (map? (distributed/check-readiness
               (distributed/plan (update plan :steps
                                         (fn [steps] (mapv #(if (= :pass (:id %))
                                                              (assoc % :dependencies [:copy-0]) %) steps)))))))))

(deftest an-initialized-but-overwritten-replica-is-not-fresh
  (let [plan (initialized-periodic-plan)
        plan (-> plan
                 (assoc-in [:device-plans :gpu-0 :entries :copy :link-plan :nodes :x-node :role] :state)
                 (assoc-in [:device-plans :gpu-0 :entries :copy :link-plan :instances 0 :bindings 'y] :local-x)
                 (assoc-in [:device-plans :gpu-0 :entries :copy :link-plan :outputs] [:x-node])
                 (update-in [:device-plans :gpu-0 :steps :copy-0 :bindings] dissoc :local-y))
        ;; The unused old y node is private storage, not an exported result.
        plan (assoc-in plan [:device-plans :gpu-0 :entries :copy :link-plan :nodes :y-node :role] :internal)
        call (get-in plan [:device-plans :gpu-0 :steps :copy-0])
        again (distributed/compute-step {:id :again :device :gpu-0 :duration-ns 1 :dependencies [:copy-0]})
        plan (-> plan (update :steps conj again)
                 (assoc-in [:device-plans :gpu-0 :steps :again] call))]
    (is (distributed/distributed-plan? (distributed/plan plan)))
    (is (= :distributed-readiness-stale
           (failure-reason #(distributed/check-readiness (distributed/plan plan))))
        "a full state write initializes storage but invalidates the earlier transfer's provenance")
    (let [old (get-in plan [:halos 0])
          refresh (distributed/schedule-halo (assoc (:exchange old) :id :refresh)
                                             (get-in plan [:values :x]) (get-in plan [:shards :x])
                                             (:routes old) [:copy-0 :analytical-only])
          retarget (fn [call]
                     (update-in call [:bindings :local-x :placements]
                                (fn [ps] (mapv #(if (= :replica (:kind %))
                                                  (update % :transfer assoc 0 :refresh) %) ps))))
          refreshed (-> plan
                        (update :halos conj refresh)
                        (assoc :steps (into (into (vec (remove #(= :again (:id %)) (:steps plan)))
                                                  (:steps refresh))
                                            [(assoc again :dependencies (:completions refresh))
                                             (distributed/compute-step {:id :other-again :device :gpu-1
                                                                        :duration-ns 1 :dependencies (:completions refresh)})]))
                        (update-in [:device-plans :gpu-0 :steps :again] retarget)
                        (assoc-in [:device-plans :gpu-1 :steps :other-again]
                                  (retarget (get-in plan [:device-plans :gpu-1 :steps :analytical-only]))))]
      (is (= 12 (count (:actions (distributed/check-readiness (distributed/plan refreshed)))))
          "explicitly refreshed transfers establish the next iteration's replica provenance"))))

(deftest strided-faces-require-pack-unpack-not-a-bounding-span-copy
  (let [base (fully-bound-periodic-plan)
        global (assoc-in (get-in base [:values :x]) [:sharding :axis] 1)
        shards (mapv #(assoc % :shape [2 1] :offsets (vec (reverse (:offsets %))))
                     (get-in base [:shards :x]))
        old (get-in base [:halos 0])
        halo (distributed/schedule-halo (assoc (:exchange old) :axis 1) global shards (:routes old) [])
        plan (-> base (assoc-in [:values :x] global) (assoc-in [:shards :x] shards)
                 (assoc :halos [halo] :steps (into (:steps halo) (filter #(= :compute (:kind %)) (:steps base)))))
        plan (reduce (fn [p [device step]]
                       (-> p
                           (assoc-in [:device-plans device :steps step :bindings :local-x :local-shape] [2 3])
                           (assoc-in [:device-plans device :steps step :bindings :local-x :placements 0
                                      :local-offsets] [0 1])))
                     plan [[:gpu-0 :copy-0] [:gpu-1 :analytical-only]])]
    (is (distributed/distributed-plan? (distributed/plan plan)))
    (is (= :distributed-transfer-layout
           (failure-reason #(distributed/transfer-bindings (distributed/plan plan)))))))

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

(deftest owned-identity-is-independent-of-dense-abi-reshape
  (let [local (local-link-plan)
        flat (local-link-plan {:x-shape [6]})
        rectangular (update-in local [:nodes :x-node :view]
                               #(view/subview % {:shape [2 3]}))
        base (plan-map)
        plans (-> (device-plans flat {:local-x (explicit-domain [2 3])})
                  (assoc-in [:gpu-0 :entries :again] {:link-plan rectangular})
                  (assoc-in [:gpu-0 :steps :analytical-only]
                            (assoc (get-in (device-plans local) [:gpu-0 :steps :copy-0])
                                   :entry :again)))
        plan (assoc base :device-plans plans
                    :steps (mapv #(assoc % :device :gpu-0) (:steps base)))
        report (distributed/compute-bindings (distributed/plan plan))]
    (is (empty? (:unbound report)))
    (is (= [6] (get-in report [:bindings :copy-0 :values :local-x :leaves 0 :view :shape])))
    (is (= [2 3] (get-in report [:bindings :analytical-only :values :local-x
                                :leaves 0 :view :shape])))
    (is (= [2 3] (get-in report [:bindings :copy-0 :values :local-x
                                :domain :placements 0 :view :shape])))
    (is (= 0 (get-in report [:bindings :copy-0 :values :local-x
                             :domain :placements 0 :view :byte-offset])))
    (testing "dense normalization cannot hide a different allocation"
      (is (= :distributed-compute-shard-storage
             (failure-reason
              #(distributed/plan
                (assoc-in plan [:device-plans :gpu-0 :entries :again :link-plan
                                :nodes :x-node :view :allocation :id] :different))))))
    (testing "nor a disjoint same-sized range in the same allocation"
      (let [larger (-> plan
                       (assoc-in [:device-plans :gpu-0 :entries :copy :link-plan
                                  :nodes :x-node :view :allocation :byte-size] 48)
                       (assoc-in [:device-plans :gpu-0 :entries :again :link-plan
                                  :nodes :x-node :view :allocation :byte-size] 48)
                       (assoc-in [:device-plans :gpu-0 :entries :again :link-plan
                                  :nodes :x-node :view :byte-offset] 24))]
        (is (= :distributed-compute-shard-storage
               (failure-reason #(distributed/plan larger))))))))

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

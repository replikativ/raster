(ns raster.compiler.ir.distributed-storage-plan-test
  "Declared resident roots and budgets; no device allocation or execution evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.distributed-plan :as plan]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.distributed-compute-test :as fixture]))

(defn- local-plan [] (#'fixture/local-link-plan))

(defn- one-call [local]
  (-> (#'fixture/plan-map {:link-plan local})
      (update :steps #(subvec % 0 1))
      (assoc :outputs [:copy-0])
      plan/plan))

(defn- two-calls [a b]
  (let [p (one-call a)]
    (-> p
        (assoc-in [:device-plans :gpu-0 :entries :second] {:link-plan b})
        (assoc-in [:device-plans :gpu-0 :steps :copy-again]
                  (assoc (get-in p [:device-plans :gpu-0 :steps :copy-0]) :entry :second))
        (update :steps conj (plan/compute-step {:id :copy-again :device :gpu-0
                                               :duration-ns 20 :dependencies [:copy-0]}))
        (assoc :outputs [:copy-again])
        plan/plan)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest one-physical-root-contract-counts-once-across-calls
  (let [local (local-plan) p (two-calls local local)
        pool (plan/resident-storage-plan p {:device-capacities {:gpu-0 72}})
        result (plan/simulate p {:device-capacities {:gpu-0 72}})]
    (is (= 3 (count (:specs pool))))
    (is (= #{:copy-0 :copy-again} (set (keys (:bindings pool)))))
    (is (= {:gpu-0 {:capacity-bytes 72 :resident-bytes 72}} (:allocation-budgets pool)))
    (is (= (:allocation-budgets pool) (get-in result [:resident-storage :allocation-budgets])))
    (is (= :owned-link-plan-roots-until-close (get-in result [:resident-storage :model])))
    (is (= 3 (get-in result [:resident-storage :allocation-count])))
    (is (not (contains? (plan/simulate p) :resident-storage)))
    (is (= :distributed-runtime-memory
           (reason #(plan/resident-storage-plan p {:device-capacities {:gpu-0 71}}))))
    (is (= 1048576 (get-in (plan/resident-storage-plan p)
                           [:allocation-budgets :gpu-0 :capacity-bytes])))))

(deftest distinct-roots-add-and-conflicting-root-contracts-decline
  (let [a (local-plan)
        ;; Bound global shards retain their exact shared realization. Only this private
        ;; constant gets a second physical root; it is not another realization of x/y.
        renamed (update-in a [:nodes :weights-node :view :allocation :id] #(vector :second %))
        pool (plan/resident-storage-plan (two-calls a renamed))
        conflicting (assoc-in a [:nodes :weights-node :view :allocation :byte-size] 32)]
    (is (= 4 (count (:specs pool))))
    (is (= 96 (get-in pool [:allocation-budgets :gpu-0 :resident-bytes])))
    (is (= :distributed-runtime-storage-contract
           (reason #(plan/resident-storage-plan (two-calls a conflicting)))))))

(deftest resident-byte-sums-do-not-overflow
  (let [local (assoc-in (local-plan) [:nodes :weights-node :view :allocation :byte-size]
                        (- Long/MAX_VALUE 3))
        error (try (plan/resident-storage-plan (one-call local)
                                              {:device-capacities {:gpu-0 Long/MAX_VALUE}})
                   nil (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :distributed-runtime-memory (:reason error)))
    (is (> (:bytes error) Long/MAX_VALUE))))

(deftest serial-actions-use-a-scratch-peak-not-a-scratch-sum
  (let [local (local-plan)
        kernel (get-in local [:instances 0 :descriptor :steps 0 :artifact])
        executable (graph/make
                    {:inputs [(graph/buffer 'x :float 6 :device :input)
                              (graph/buffer 'weights :float 6 :device :input)]
                     :outputs [(graph/buffer 'y :float 6 :device :output)]
                     :temporaries [(graph/buffer 'scratch :float 24 :device :temporary)]
                     :scalars [(graph/scalar 'n :long)]
                     :nodes [(graph/->ScheduledKernel
                              :copy kernel [(graph/->ValueUse 'x :read)
                                            (graph/->ValueUse 'weights :read)
                                            (graph/->ValueUse 'y :write)] #{'n} [])]
                     :abi (:abi kernel) :arguments (:arguments kernel)})
        local (assoc-in local [:instances 0 :descriptor :steps 0 :artifact] executable)
        p (two-calls local local)
        options {:device-capacities {:gpu-0 168} :include-graph-temporaries? true}
        pool (plan/resident-storage-plan p options)
        result (plan/simulate p options)]
    (is (= 2 (count (:graph-temporary-plans pool))))
    (is (= {:gpu-0 {:capacity-bytes 168 :resident-bytes 72
                    :graph-temporary-bytes 96 :planned-peak-bytes 168}}
           (:allocation-budgets pool)))
    (is (= (:allocation-budgets pool) (get-in result [:resident-storage :allocation-budgets])))
    (is (= :owned-roots-plus-serial-graph-temporaries (get-in result [:resident-storage :model])))
    (is (= :distributed-runtime-memory
           (reason #(plan/resident-storage-plan p (assoc options :device-capacities {:gpu-0 167})))))))

(deftest remapped-targets-require-an-aggregate-budget
  (let [local (#'fixture/local-link-plan {:target :physical})
        p (-> (#'fixture/plan-map {:link-plan local})
              (assoc-in [:device-plans :gpu-0 :target] :physical)
              (update :steps #(subvec % 0 1)) (assoc :outputs [:copy-0]) plan/plan)]
    (is (= :distributed-runtime-physical-budget (reason #(plan/resident-storage-plan p))))
    (is (= {:physical {:capacity-bytes 72 :resident-bytes 72}}
           (:allocation-budgets (plan/resident-storage-plan p {:device-capacities {:physical 72}}))))
    (is (= :distributed-runtime-memory
           (reason #(plan/simulate p {:device-capacities {:physical 71}}))))))

(deftest colocated-workers-share-one-physical-root-pool
  (let [local (local-plan)
        other (reduce (fn [p node]
                        (update-in p [:nodes node :view :allocation :id] #(vector :worker-1 %)))
                      local [:x-node :y-node])
        base (two-calls local local)
        call (get-in base [:device-plans :gpu-0 :steps :copy-again])
        p (-> base
              (update-in [:device-plans :gpu-0 :steps] dissoc :copy-again)
              (assoc-in [:device-plans :gpu-1]
                        {:target :gpu-0 :entries {:second {:link-plan other}}
                         :steps {:copy-again
                                 (update call :bindings
                                         #(update-vals % (fn [ref]
                                                           (assoc ref :shard
                                                                  (keyword (str (name (:value ref)) "-1"))))))}})
              (assoc-in [:steps 1 :device] :gpu-1)
              plan/plan)
        pool (plan/resident-storage-plan p {:device-capacities {:gpu-0 120}})]
    (is (= 5 (count (:specs pool))))
    (is (= {:gpu-0 {:capacity-bytes 120 :resident-bytes 120}} (:allocation-budgets pool)))
    (is (= :distributed-runtime-memory
           (reason #(plan/resident-storage-plan p {:device-capacities {:gpu-0 119}}))))
    (is (= :distributed-runtime-physical-budget (reason #(plan/resident-storage-plan p))))))

(deftest public-projection-revalidates-storage-and-caller-options
  (let [p (one-call (local-plan))]
    (is (= :distributed-resident-unbound (reason #(plan/resident-storage-plan (#'fixture/make-plan)))))
    (is (= :distributed-runtime-allocation
           (reason #(plan/resident-storage-plan
                     (one-call (assoc-in (local-plan) [:nodes :x-node :view :allocation :ownership] :borrowed))))))
    (is (= :distributed-runtime-allocation
           (reason #(plan/resident-storage-plan
                     (one-call (assoc-in (local-plan) [:nodes :x-node :view :allocation :byte-size] 25))))))
    (doseq [options [nil {:device-capacities true} {:unknown true} {:include-graph-temporaries? :unknown}
                     {:device-capacities {:gpu-0 -1}} {:device-capacities {:gpu-0 Double/NaN}}]]
      (is (= :distributed-runtime-physical-budget (reason #(plan/resident-storage-plan p options)))))
    (is (= :distributed-cost-options
           (reason #(plan/simulate p {:device-capacities {:gpu-0 72} :route-context {}}))))))

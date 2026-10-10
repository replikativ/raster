(ns raster.compiler.ir.distributed-training-test
  "Hardware-free AD/collective/SGD composition and independent numerical oracles.
   Retained in the OpenCL acceptance job as a compiler companion to native tests."
  {:raster.test/opencl-gate true}
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.compiler.fixtures.distributed-training
             :refer [batches initial-theta local-gradient-sum multi-step-training-plan
                     value-node rounded-gradient-sum max-error expected-updates
                     training-plan analytic-gradient-sum]]))

(deftest repeated-local-ad-matches-rounded-independent-oracle
  (doseq [row-counts [[1 3] [1 2 4]]
          theta (cons (initial-theta) (expected-updates row-counts 2))
          {:keys [rows x y] :as batch} (batches row-counts)]
    (is (< (max-error (rounded-gradient-sum theta batch)
                     (local-gradient-sum theta x y rows 17)) 1.0e-7))))

(deftest finite-training-composition-retains-parameters-and-orders-updates
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (constantly descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "assembly allocated device storage" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [plan (multi-step-training-plan device row-counts 3)
              workers (get-in plan [:mesh :devices])
              parameter-allocations
              (into #{} (for [worker workers
                              :let [step [[:epoch 0] [worker :produce]]
                                    local (get-in plan [:device-plans worker :entries step :link-plan])]
                              [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                              :when (= [worker :parameters] (:value binding))]
                          (get-in (value-node local id) [:view :allocation :id])))
              scratch-by-epoch
              (mapv (fn [epoch]
                      (set/difference
                       (into #{} (for [[_ worker-plan] (:device-plans plan)
                                       [[scope _] entry] (:entries worker-plan)
                                       :when (= [:epoch epoch] scope)
                                       [_ node] (get-in entry [:link-plan :nodes])
                                       :let [id (get-in node [:view :allocation :id])]
                                       :when id] id))
                       parameter-allocations)) (range 3))]
          (is (= 3 (count (:refinements plan))))
          (is (= (mapv #(vector [:epoch 2] [% :update]) workers) (:outputs plan)))
          (is (= (distributed/certify plan) (distributed/verify! (distributed/certify plan))))
          (is (= (count workers) (count parameter-allocations)))
          (is (every? seq scratch-by-epoch))
          (doseq [left (range 3) right (range (inc left) 3)]
            (is (empty? (set/intersection (nth scratch-by-epoch left) (nth scratch-by-epoch right)))
                "private scratch and immutable contribution storage are fresh across epochs"))
          (is (thrown? clojure.lang.ExceptionInfo
                       (distributed/check-readiness
                        (update plan :steps
                                #(mapv (fn [step]
                                         (if (= [[:epoch 1] [(first workers) :produce]] (:id step))
                                           (assoc step :dependencies []) step)) %))))
              "an epoch cannot read parameters unordered with respect to the preceding update")
          (doseq [worker workers]
            (let [parameters (for [epoch (range 3) kind [:produce :update]
                                   :let [step [[:epoch epoch] [worker kind]]
                                         local (get-in plan [:device-plans worker :entries step :link-plan])]
                                   [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                                   :when (= [worker :parameters] (:value binding))]
                               (value-node local id))]
              (is (= 6 (count parameters)))
              (is (apply = (map #(dissoc (:view %) :id) parameters)))
              (is (every? #(identical? (:source (first parameters)) (:source %)) parameters)))
            (doseq [epoch [1 2]]
              (is (= [[[:epoch (dec epoch)] [worker :update]]]
                     (:dependencies (first (filter #(= [[:epoch epoch] [worker :produce]] (:id %))
                                                   (:steps plan)))))))))))))

(deftest unequal-batch-local-ad-producers-match-independent-analytic-gradients
  (doseq [rows [[1 3] [1 2 4]]
          {:keys [x y rows] :as batch} (batches rows)]
    (let [actual (local-gradient-sum (initial-theta) x y rows 17)]
      (is (= 17 (alength actual)))
      (is (< (max-error (analytic-gradient-sum batch) actual) 1.0e-7)))))

(deftest differentiated-training-assembly-certifies-without-device-resources
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (fn [target]
                                         (assert (= device target))
                                         descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "hardware-free assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "hardware-free assembly allocated device storage" {})))
                  compiled/instantiate! (fn [& _] (throw (ex-info "hardware-free assembly realized a program" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [n (count row-counts)
              plan (training-plan device row-counts)
              certified (distributed/certify plan)]
          (is (= certified (distributed/verify! certified)))
          (is (= n (count (filter #(= :produce (second (:id %))) (:steps plan)))))
          (is (= n (count (filter #(= :update (second (:id %))) (:steps plan)))))
          (is (= (* 2 (dec n)) (count (filter #(= :transfer (:kind %)) (:steps plan)))))
          (is (= (dec n) (count (filter #(= :combined (second (:id %))) (:steps plan)))))
          (is (= (mapv #(vector % :update) (get-in plan [:mesh :devices])) (:outputs plan)))
          (doseq [[worker ssa] (get-in plan [:refinements :all-reduce :refinement :outputs])
                  :let [local (get-in plan [:device-plans worker :entries [worker :update] :link-plan])
                        matches (for [[value binding]
                                      (get-in plan [:device-plans worker :steps [worker :update] :bindings])
                                      :when (= ssa (:value binding))] value)]]
            (is (= 1 (count matches)))
            (is (nil? (:source (value-node local (first matches))))
                "host gradient initialization cannot stand in for the actual collective producer")))))))


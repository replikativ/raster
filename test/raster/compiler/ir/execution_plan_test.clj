(ns raster.compiler.ir.execution-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as kernel-call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

(defn- one-call
  []
  (let [artifact
        (artifact/make
         {:kernel-name "execution_plan_probe"
          :source "__kernel void execution_plan_probe(__global float* out, int n) {}"
          :abi [(kabi/slot 'out :output :float :role :result)
                (kabi/slot 'n :scalar :int :role :bound)]
          :arguments '[out n]
          :launch (launch/spec {:workgroup-size [64]
                                :group-count [(launch/ceil-div 'n 64)]})
          :effects {:kind :elementwise-map :writes ['out]}})]
    (kernel-call/make artifact [(Object.) {:type :int :value 128}])))

(defn- emitted-graph []
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              93)
        operations (lower/lower-scan node nil :dtype :float)]
    (emit/generate-scan-kernel-graph
     (lower/scan-kernel-graph
      node operations {:array-types {'values :float 'out :float}}))))

(deftest kernel-graph-dependencies-become-logical-event-edges
  (let [graph (emitted-graph)
        ids (set (map :id (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        call (graph-call/make graph (zipmap ids (repeatedly #(Object.)))
                              {'n {:type :int :value 1025}})
        plan (execution/from-kernel-graph-call call)
        [intra block carry] (:operations plan)]
    (is (execution/execution-plan? plan))
    (is (= [:compute :compute :compute]
           (mapv (comp :class :queue) (:operations plan))))
    (is (empty? (:waits intra)))
    (is (= [(:completion intra)] (:waits block)))
    (is (= [(:completion intra) (:completion block)] (:waits carry)))
    (is (= [(:completion carry)] (:outputs plan)))
    (testing "compiler events contain identities, never native handles"
      (is (every? execution/logical-event?
                  (concat (:outputs plan) (mapcat :waits (:operations plan))))))))

(deftest execution-plan-rejects-forward-event-dependencies
  (let [queue (execution/compute-queue)
        later (execution/->LogicalEvent :later)
        done (execution/->LogicalEvent :done)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"available event"
         (execution/validate!
          (execution/->ExecutionPlan
           [queue] []
           [(execution/->ScheduledOperation :first :operation queue [later] done)]
           [done]))))))

(deftest one-kernel-call-uses-the-same-logical-execution-contract
  (let [call (one-call)
        plan (execution/from-kernel-call :candidate call)
        operation (first (:operations plan))]
    (is (execution/execution-plan? plan))
    (is (= call (:operation operation)))
    (is (= :compute (get-in operation [:queue :class])))
    (is (empty? (:waits operation)))
    (is (= [(:completion operation)] (:outputs plan)))))

(deftest logical-transfer-queue-is-backend-neutral
  (let [queue (execution/transfer-queue)]
    (is (execution/logical-queue? queue))
    (is (= :transfer (:class queue)))
    (is (= :in-order (:ordering queue)))
    (is (not= queue (execution/compute-queue)))))

(deftest durable-storage-has-a-distinct-logical-resource-queue
  (let [queue (execution/storage-queue)]
    (is (execution/logical-queue? queue))
    (is (= :storage (:class queue)))
    (is (= :in-order (:ordering queue)))
    (is (not= queue (execution/transfer-queue)))
    (is (not= queue (execution/compute-queue)))))

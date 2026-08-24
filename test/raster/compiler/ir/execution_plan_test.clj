(ns raster.compiler.ir.execution-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

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

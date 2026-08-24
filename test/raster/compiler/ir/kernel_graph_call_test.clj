(ns raster.compiler.ir.kernel-graph-call-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

(defn- emitted-graph []
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              91)
        operations (lower/lower-scan node nil :dtype :float)]
    (emit/generate-scan-kernel-graph
     (lower/scan-kernel-graph
      node operations {:array-types {'values :float 'out :float}}))))

(deftest emitted-graph-becomes-an-ordered-vector-of-kernel-calls
  (let [graph (emitted-graph)
        ids (set (map :id (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        buffers (zipmap ids (repeatedly #(Object.)))
        scalars {'n {:type :int :value 1025}}
        call (graph-call/make graph buffers scalars)
        [intra block carry] (mapv :call (:nodes call))]
    (is (graph-call/kernel-graph-call? call))
    (is (every? kcall/kernel-call? [intra block carry]))
    (is (= [[5] [1] [5]]
           (mapv #(get-in % [:geometry :group-count]) [intra block carry])))
    (testing "the totals extent and stage-2 bound share one checked CeilDiv value"
      (is (= 5 (second (first (vals (graph-call/temporary-specs graph scalars))))))
      (is (= {:type :int :value 5} (last (:arguments block)))))
    (is (= (mapv :dependencies (:nodes graph))
           (mapv :dependencies (:nodes call))))))

(deftest graph-call-fails-before-a-driver-sees-incomplete-bindings
  (let [graph (emitted-graph)
        ids (set (map :id (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        buffers (zipmap ids (repeatedly #(Object.)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly every declared graph buffer"
                          (graph-call/make graph (dissoc buffers (first ids))
                                           {'n {:type :int :value 1025}})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicitly typed"
                          (graph-call/make graph buffers {'n 1025})))))

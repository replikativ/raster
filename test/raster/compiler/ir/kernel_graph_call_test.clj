(ns raster.compiler.ir.kernel-graph-call-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-precondition :as precondition]
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

(deftest direct-scalar-ranges-are-projected-without-evaluating-derived-expressions
  (let [graph (emitted-graph)
        conditions (graph-call/direct-scalar-range-preconditions graph)]
    (is (seq conditions))
    (is (every? #(= 'n (:expression %)) conditions))
    (is (precondition/check! conditions {'n 1025}))
    (doseq [n [-1 2147483648]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"precondition failed"
                            (precondition/check! conditions {'n n}))))))

(deftest scalar-preconditions-precede-temporary-sizing
  (let [graph (assoc-in (emitted-graph) [:nodes 0 :operation :preconditions]
                        [{:expression '_n_bound :op :>= :value 64}])]
    (is (= graph (graph-call/preflight! graph {'n {:type :int :value 1025}})))
    (with-redefs [graph-call/resolve-integer
                  (fn [& _] (throw (ex-info "temporary sizing happened first" {})))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar precondition failed"
                            (graph-call/temporary-specs graph {'n {:type :int :value 32}}))))))

(deftest later-node-preconditions-resolve-derived-physical-scalars
  (let [graph (emitted-graph)
        derived-slot (:name (last (get-in graph [:nodes 1 :operation :abi])))
        guarded (assoc-in graph [:nodes 1 :operation :preconditions]
                          [{:expression derived-slot :op :>= :value 5}])]
    (is (= guarded (graph-call/preflight! guarded {'n {:type :int :value 1025}})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar precondition failed"
                          (graph-call/temporary-specs guarded {'n {:type :int :value 1000}})))))

(deftest public-long-scalars-are-validated-before-private-conversion
  (let [graph (-> (emitted-graph)
                  (update :scalars #(mapv (fn [scalar] (assoc scalar :dtype :long)) %))
                  (update :abi #(mapv (fn [slot]
                                       (if (= :scalar (:kind slot))
                                         (assoc slot :dtype :long) slot)) %))
                  (update :nodes
                          #(mapv (fn [node]
                                   (update-in node [:operation :abi]
                                              (fn [slots]
                                                (mapv (fn [slot]
                                                        (if (= :scalar (:kind slot))
                                                          (assoc slot :dtype :long) slot)) slots)))) %)))]
    (is (= graph (graph-call/preflight! graph {'n {:type :long :value 1025}})))
    (is (= graph (graph-call/preflight! graph {'n {:type :int :value 1025}}))
        "the common executable binder may supply the public slot's physical representation")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"wrong ABI dtype"
                          (graph-call/preflight! graph {'n {:type :float :value 1025.0}})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physical ABI range"
                          (graph-call/preflight! graph
                                                 {'n {:type :long :value (+' (bigint Long/MAX_VALUE) 2)}})))))

(deftest emitted-graph-becomes-an-ordered-vector-of-kernel-calls
  (let [graph (emitted-graph)
        ids (set (map :id (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        buffers (zipmap ids (repeatedly #(Object.)))
        scalars {'n {:type :int :value 1025}}
        call (graph-call/make graph buffers scalars)
        [intra block carry] (mapv :call (:nodes call))]
    (is (= [['n :int]] (mapv (juxt :id :dtype) (:scalars graph))))
    (is (every? set? (map :scalar-uses (:nodes graph))))
    (is (= #{'n} (reduce into #{} (map :scalar-uses (:nodes graph)))))
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
                          (graph-call/make graph buffers {'n 1025})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly every public scalar"
                          (graph-call/make graph buffers {})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"exactly every public scalar"
                          (graph-call/make graph buffers
                                           {'n {:type :int :value 1025}
                                            'extra {:type :int :value 1}})))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"wrong ABI dtype"
                          (graph-call/make graph buffers
                                           {'n {:type :long :value 1025}})))))

(deftest derived-int-scalars-refuse-narrowing-overflow
  (let [graph (emitted-graph)
        ids (set (map :id (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        buffers (zipmap ids (repeatedly #(Object.)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"outside its physical ABI range"
                          (graph-call/make graph buffers
                                           {'n {:type :int :value Long/MAX_VALUE}})))))

(deftest emitted-graph-projects-one-ordered-public-call-to-runtime-binding-maps
  (let [graph (emitted-graph)
        n {:type :int :value 1025}
        bindings (executable/graph-bindings graph [:values-resident :out-resident n])]
    (is (= '[values out n] (:arguments graph)))
    (is (= {'values :values-resident 'out :out-resident} (:buffers bindings)))
    (is (= {'n n} (:scalar-values bindings)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicitly typed"
                          (executable/graph-bindings graph [:values :out 1025])))))

(deftest executable-graphs-cannot-retain-the-pre-emission-nil-scalar-escape
  (let [graph (emitted-graph)
        incomplete (-> graph
                       (assoc :scalars nil)
                       (update :nodes #(mapv (fn [node] (assoc node :scalar-uses nil)) %)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"explicit scalar dependencies"
                          (executable/validate! incomplete)))))

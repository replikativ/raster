(ns raster.compiler.ir.emitted-parallel-equation-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]))

(def ^:private map-source
  '(raster.par/pmap i n float (clojure.core/+ 1.0 (clojure.core/aget values i))))

(defn- scheduled-fixture []
  (let [source (list 'let* ['result map-source] 'result)
        typed (:program (typed-route/attempt
                         source :float {'values :float}
                         {:scalar-types {'n :int}}))
        lowered (:form (segop-lower/segop-lower-pass
                        typed {:target-device :cpu:0 :dtype :float}))
        equation (first (:equations lowered))
        ;; Production equation emission narrows the enclosing program to this equation and retains
        ;; its semantic operand order. The whole-program lowerer sorts independent outer inputs.
        body (assoc lowered :inputs (:operands equation) :outputs (:results equation))
        algorithm (-> body :equations first :algorithm)
        source (equation-graph/make algorithm body)
        source-node (first (:nodes source))
        refined-nodes
        (mapv (fn [index]
                (graph/->ScheduledKernel
                 [:refined index]
                 [:refined-operation index]
                 (:uses source-node)
                 (:scalar-uses source-node)
                 (mapv (fn [earlier] [:refined earlier]) (range index))))
              (range 3))
        refined (graph/validate! (assoc source :nodes refined-nodes))
        witness (refinement/make
                 {:source source :graph refined
                  :schedule {:kind :test-three-stage
                             :semantic-law :fixture-only}
                  :numerics {:mode :exact :policy :fixture-only}})]
    {:algorithm algorithm :body body :source source :refined refined :witness witness}))

(defn- use-kind [access]
  (case access :read :input :write :output :read-write :inout))

(defn- test-artifact
  ([index operation uses scalar-uses buffers scalar-types]
   (test-artifact index operation uses scalar-uses buffers scalar-types :opencl-c))
  ([index operation uses scalar-uses buffers scalar-types target]
   (let [kernel-name (str "refined_" index)
         pointer-slots (mapv (fn [{:keys [buffer access]}]
                               (abi/slot buffer (use-kind access) (:dtype (get buffers buffer))
                                         :c-name (c-emit/c-symbol buffer)))
                             uses)
         ordered-scalars (vec (sort-by str scalar-uses))
         scalar-slots (mapv #(abi/slot % :scalar (get scalar-types %)
                                      :c-name (c-emit/c-symbol %))
                            ordered-scalars)
         slots (into pointer-slots scalar-slots)
         parameters
         (mapv (fn [slot]
                 (if (= :scalar (:kind slot))
                   (str "long " (:c-name slot))
                   (str (if (= :opencl-c target)
                          (if (= :input (:kind slot)) "__global const " "__global ")
                          (if (= :input (:kind slot)) "const " ""))
                        "float* " (:c-name slot))))
               slots)
         prefix (if (= :opencl-c target) "__kernel void " "extern \"C\" __global__ void ")]
     (artifact/certify-scheduled-operation
      (artifact/make
       {:kernel-name kernel-name
        :target target
        :source (str prefix kernel-name "(" (str/join ", " parameters)
                     ") { " (:c-name (first (filter #(not= :input (:kind %)) slots)))
                     "[0] = 0.0f; }")
        :abi slots :arguments (into (mapv :buffer uses) ordered-scalars)
        :launch (launch/spec {:workgroup-size [1] :group-count [1]})})
      operation))))

(defn- graph-interface
  [scheduled]
  (let [external (vec (distinct (concat (:inputs scheduled) (:outputs scheduled))))
        scalar-interface (:scalars scheduled)]
    (graph/validate!
     (assoc scheduled
            :abi (vec
                  (concat
                   (map (fn [{:keys [id dtype role]}]
                          (abi/slot id (case role
                                         :input :input :output :output :inout :inout)
                                    dtype :c-name (c-emit/c-symbol id)))
                        external)
                   (map (fn [{:keys [id dtype]}]
                          (abi/slot id :scalar dtype :c-name (c-emit/c-symbol id)))
                        scalar-interface)))
            :arguments (vec (concat (map :id external) (map :id scalar-interface)))))))

(defn- emit-graph
  [scheduled]
  (let [buffers (into {} (map (juxt :id identity))
                      (concat (:inputs scheduled) (:outputs scheduled) (:temporaries scheduled)))
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars scheduled))]
    (graph-interface
     (graph/map-operations
      scheduled
      (fn [{:keys [id operation uses scalar-uses]}]
        (test-artifact (last id) operation uses scalar-uses buffers scalar-types))))))

(defn- append-int-scalar
  [kernel argument]
  (let [slot (abi/slot 'derived :scalar :int :c-name "derived")]
    (artifact/validate!
     (-> kernel
         (update :abi conj slot)
         (update :arguments conj argument)
         (update :source #(str/replace-first % #"\) \{" ", int derived) {"))))))

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest legacy-one-to-one-emission-remains-the-degenerate-case
  (let [{:keys [algorithm body source]} (scheduled-fixture)
        emitted (emit-graph source)]
    (is (emitted-equation/emitted-equation?
         (emitted-equation/make algorithm body emitted)))))

(deftest emitted-equation-certifies-both-links-of-a-one-to-many-schedule
  (let [{:keys [algorithm body refined witness]} (scheduled-fixture)
        emitted (emit-graph refined)
        value (emitted-equation/make algorithm body emitted {:refinement witness})]
    (is (emitted-equation/emitted-equation? value))
    (is (identical? witness (:refinement value)))
    (testing "emission must retain the refined topology"
      (let [{:keys [source]} (scheduled-fixture)
            wrong-shape (emit-graph source)]
        (is (= :emitted-parallel-equation-dataflow
               (reason-of #(emitted-equation/make
                            algorithm body wrong-shape {:refinement witness}))))))
    (testing "every artifact certifies its exact refined operation"
      (let [wrong-certificate
            (assoc-in emitted [:nodes 1 :operation :provenance :scheduled-operation]
                      :wrong-operation)]
        (is (= :emitted-parallel-equation-operation
               (reason-of #(emitted-equation/make
                            algorithm body wrong-certificate {:refinement witness}))))))))

(deftest emitted-graph-is-an-executable-artifact-projection
  (let [{:keys [algorithm body refined witness]} (scheduled-fixture)
        emitted (emit-graph refined)
        buffers (into {} (map (juxt :id identity))
                      (concat (:inputs refined) (:outputs refined) (:temporaries refined)))
        node (get-in refined [:nodes 1])]
    (testing "a node artifact cannot invent or omit a graph buffer use"
      (is (= :kernel-graph-artifact-buffer
             (reason-of #(emitted-equation/make
                          algorithm body
                          (assoc-in emitted [:nodes 0 :operation :arguments 0] 'undeclared)
                          {:refinement witness}))))
      (let [output-use (filterv #(not= :read (:access %)) (:uses node))
            scalar-types (into {} (map (juxt :id :dtype)) (:scalars refined))
            output-only (test-artifact 99 (:operation node) output-use
                                       (:scalar-uses node) buffers scalar-types)]
        (is (= :kernel-graph-artifact-uses
               (reason-of #(emitted-equation/make
                            algorithm body
                            (assoc-in emitted [:nodes 1 :operation] output-only)
                            {:refinement witness}))))))
    (testing "an emitted graph has one interface and one target dialect"
      (is (= :kernel-graph-executable-interface
             (reason-of #(emitted-equation/make
                          algorithm body (assoc emitted :abi nil :arguments nil)
                          {:refinement witness}))))
      (let [{:keys [id operation uses scalar-uses]} node
            scalar-types (into {} (map (juxt :id :dtype)) (:scalars refined))
            hip (test-artifact (last id) operation uses scalar-uses
                               buffers scalar-types :hip-cpp)]
        (is (= :kernel-graph-executable-targets
               (reason-of #(emitted-equation/make
                            algorithm body (assoc-in emitted [:nodes 1 :operation] hip)
                            {:refinement witness}))))))))

(deftest refinement-source-is-rederived-from-the-typed-equation
  (let [{:keys [algorithm body source refined]} (scheduled-fixture)
        stale-source (assoc-in source [:nodes 0 :operation] :stale-operation)
        stale (refinement/make
               {:source stale-source :graph refined :schedule {:kind :stale}
                :numerics {:mode :exact :policy :structural-identity}})
        emitted (emit-graph refined)]
    (is (= :scheduled-graph-refinement-source
           (reason-of #(emitted-equation/make
                        algorithm body emitted {:refinement stale}))))))

(deftest emitted-equation-cannot-invent-a-public-scalar
  (let [{:keys [algorithm body refined witness]} (scheduled-fixture)
        emitted (emit-graph refined)
        forged (-> emitted
                   (update :scalars conj (graph/scalar 'forged :int))
                   (update :abi conj (abi/slot 'forged :scalar :int))
                   (update :arguments conj 'forged))]
    (is (= :emitted-parallel-equation-dataflow
           (reason-of #(emitted-equation/make
                        algorithm body forged {:refinement witness}))))))

(deftest artifact-private-scalars-are-closed-integral-launch-expressions
  (let [{:keys [algorithm body refined witness]} (scheduled-fixture)
        emitted (emit-graph refined)]
    (testing "a signed literal is a valid private scalar even though it is not a dimension"
      (let [with-literal (update-in emitted [:nodes 0 :operation]
                                    append-int-scalar -1)]
        (is (emitted-equation/emitted-equation?
             (emitted-equation/make
              algorithm body with-literal {:refinement witness})))))
    (testing "a derived integer may not close over a public floating-point scalar"
      (let [alpha (graph/scalar 'alpha :float)
            with-alpha (-> emitted
                           (update :scalars conj alpha)
                           (update :abi conj (abi/slot 'alpha :scalar :float))
                           (update :arguments conj 'alpha)
                           (update-in [:nodes 0 :operation]
                                      append-int-scalar (launch/runtime-value 'alpha)))]
        (is (= :kernel-graph-artifact-scalar
               (reason-of #(emitted-equation/make
                            algorithm body with-alpha {:refinement witness}))))))
    (testing "malformed target-private index algebra is rejected before runtime binding"
      (let [malformed (update-in emitted [:nodes 0 :operation]
                                 append-int-scalar
                                 (kernel-body/->IndexExpr :unknown ['n]))]
        (is (= :kernel-graph-artifact-scalar
               (reason-of #(emitted-equation/make
                            algorithm body malformed {:refinement witness}))))))))

(deftest refinement-cannot-forge-the-rederived-source-interface
  (let [{:keys [algorithm body source refined]} (scheduled-fixture)
        scalar (abi/slot 'forged :scalar :int)
        forged-source (let [interfaced (graph-interface source)]
                        (graph/validate! (-> interfaced
                                             (update :abi conj scalar)
                                             (update :arguments conj 'forged)
                                             (update :scalars conj (graph/scalar 'forged :int)))))
        forged-refined (let [interfaced (graph-interface refined)]
                         (graph/validate! (-> interfaced
                                              (update :abi conj scalar)
                                              (update :arguments conj 'forged)
                                              (update :scalars conj (graph/scalar 'forged :int)))))
        forged (refinement/make
                {:source forged-source :graph forged-refined
                 :schedule {:kind :forged-interface}
                 :numerics {:mode :exact :policy :structural-identity}})
        emitted (emit-graph refined)]
    (is (= :scheduled-graph-refinement-source
           (reason-of #(emitted-equation/make
                        algorithm body emitted {:refinement forged}))))))

(ns raster.compiler.ir.scheduled-graph-refinement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]))

(defn- fixtures []
  (let [a (graph/buffer 'a :float 64 :device :input)
        b (graph/buffer 'b :float 64 :device :input)
        c (graph/buffer 'c :float 64 :device :output)
        ah (graph/buffer 'a-half :half 64 :device :temporary)
        bh (graph/buffer 'b-half :half 64 :device :temporary)
        interface [(abi/slot 'a :input :float)
                   (abi/slot 'b :input :float)
                   (abi/slot 'c :output :float)
                   (abi/slot 'n :scalar :int)]
        arguments '[a b c n]
        scalars [(graph/scalar 'n :int)]
        effects {:semantic #{:numerical}}
        source
        (graph/make
         {:inputs [a b] :outputs [c]
          :nodes [(graph/->ScheduledKernel
                   :semantic-contract :contract
                   [(graph/->ValueUse 'a :read)
                    (graph/->ValueUse 'b :read)
                    (graph/->ValueUse 'c :write)] [])]
          :scalars scalars :abi interface :arguments arguments :effects effects})
        refined
        (graph/make
         {:inputs [a b] :outputs [c] :temporaries [ah bh]
          :nodes [(graph/->ScheduledKernel
                   :cast-a :cast-a
                   [(graph/->ValueUse 'a :read) (graph/->ValueUse 'a-half :write)] [])
                  (graph/->ScheduledKernel
                   :cast-b :cast-b
                   [(graph/->ValueUse 'b :read) (graph/->ValueUse 'b-half :write)] [])
                  (graph/->ScheduledKernel
                   :matrix :matrix
                   [(graph/->ValueUse 'a-half :read)
                    (graph/->ValueUse 'b-half :read)
                    (graph/->ValueUse 'c :write)]
                   [:cast-a :cast-b])]
          :scalars scalars :abi interface :arguments arguments :effects effects})]
    {:a a :b b :c c :source source :refined refined}))

(defn- witness []
  (let [{:keys [source refined]} (fixtures)]
    (refinement/make
     {:source source :graph refined
      :schedule {:kind :mixed-precision-contraction
                 :accumulator-dtype :float}
      :numerics {:mode :bounded-error
                 :policy :mixed-precision-fixture
                 :rounding :nearest-even
                 :accumulator-dtype :float
                 :error-model {:kind :producer-verified-mixed-precision}}})))

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest one-semantic-operation-can-refine-to-a-private-kernel-dag
  (let [{:keys [source refined]} (fixtures)
        value (witness)]
    (is (refinement/scheduled-graph-refinement? value))
    (is (= :contract (refinement/source-operation value)))
    (is (= refined (refinement/scheduled-graph value)))
    (is (= (graph/boundary-contract source)
           (graph/boundary-contract refined)))
    (is (not= (graph/dataflow-contract source)
              (graph/dataflow-contract refined))
        "private topology changes without weakening strict emission equivalence")))

(deftest refinement-rejects-erasure-and-a-non-singleton-source
  (let [{:keys [a b refined]} (fixtures)
        empty-source (graph/make {:inputs [a b] :nodes []})]
    (is (= :scheduled-graph-refinement-source-cardinality
           (reason-of #(refinement/make
                        {:source empty-source :graph refined :schedule {}}))))
    (is (= :scheduled-graph-refinement-source-cardinality
           (reason-of #(refinement/make
                        {:source refined :graph refined :schedule {}}))))
    (let [{:keys [source]} (fixtures)
          empty-refined (graph/make {:inputs [a b] :nodes []})]
      (is (= :scheduled-graph-refinement-empty
             (reason-of #(refinement/make
                          {:source source :graph empty-refined :schedule {}})))))))

(deftest refinement-preserves-the-exact-ordered-public-boundary
  (let [{:keys [source refined]} (fixtures)]
    (testing "tensor extents and ordered inputs"
      (is (= :scheduled-graph-refinement-boundary
             (reason-of #(refinement/make
                          {:source source
                           :graph (graph/validate!
                                   (assoc-in refined [:inputs 0 :elements] 32))
                           :schedule {}}))))
      (is (= :scheduled-graph-refinement-boundary
             (reason-of #(refinement/make
                          {:source source
                           :graph (graph/validate!
                                   (update refined :inputs (fn [values] (vec (reverse values)))))
                           :schedule {}})))))
    (testing "the ordered public ABI and scalar arguments"
      (let [extra-abi (conj (:abi refined) (abi/slot 'tile :scalar :int))]
        (is (= :scheduled-graph-refinement-boundary
               (reason-of #(refinement/make
                            {:source source
                             :graph (graph/validate!
                                     (assoc refined :abi extra-abi
                                                    :arguments (conj (:arguments refined) 'tile)
                                                    :scalars (conj (:scalars refined)
                                                                   (graph/scalar 'tile :int))))
                             :schedule {}}))))))
    (testing "logical effects"
      (is (= :scheduled-graph-refinement-boundary
             (reason-of #(refinement/make
                          {:source source
                           :graph (assoc refined :effects {:semantic #{:io}})
                           :schedule {}})))))))

(deftest refinement-preserves-logical-scalars-without-a-target-abi
  (let [{:keys [source refined]} (fixtures)
        source (graph/validate! (assoc source :abi nil :arguments nil))
        refined (graph/validate! (assoc refined :abi nil :arguments nil))]
    (is (refinement/scheduled-graph-refinement?
         (refinement/make {:source source :graph refined
                           :schedule {:kind :target-neutral-fixture}
                           :numerics {:mode :exact :policy :structural-identity}})))
    (is (= :scheduled-graph-refinement-boundary
           (reason-of #(refinement/make
                        {:source source
                         :graph (assoc refined :scalars [(graph/scalar 'n :long)])
                         :schedule {:kind :target-neutral-fixture}
                         :numerics {:mode :exact :policy :structural-identity}}))))))

(deftest refinement-descriptions-and-source-identity-are-checked
  (let [{:keys [source]} (fixtures)
        value (witness)]
    (doseq [[field bad] [[:schedule []] [:schedule {}]
                         [:provenance nil] [:attributes :bad]]]
      (is (= :scheduled-graph-refinement-description
             (reason-of #(refinement/validate! (assoc value field bad))))))
    (is (= :scheduled-graph-refinement-numerics
           (reason-of #(refinement/validate! (assoc value :numerics {})))))
    (is (= :scheduled-graph-refinement-numerics
           (reason-of #(refinement/validate!
                        (assoc value :numerics {:mode :bounded-error
                                                :policy :missing-evidence})))))
    (let [different-operation
          (graph/validate! (assoc-in source [:nodes 0 :operation] :different-contract))]
      (is (= :scheduled-graph-refinement-source
             (reason-of #(refinement/validate-against! value different-operation)))))))

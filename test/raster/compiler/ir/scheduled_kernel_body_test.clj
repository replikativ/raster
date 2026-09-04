(ns raster.compiler.ir.scheduled-kernel-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]))

(defn- scalar-body []
  (body/make
   {:id :scheduled-body-test
    :parameters [(body/->KernelParameter
                  'x :input :float [1] :global (layout/row-major [1] :float) :operand)
                 (body/->KernelParameter
                  'y :output :float [1] :global (layout/row-major [1] :float) :result)]
    :stable-reads [(body/stable-read 'x)]
    :operations [(body/->ScalarLoad (body/value 'value :float) 'x [0] nil nil :cached)
                 (body/->ScalarStore 'y [0] 'value nil)]
    :schedule {:kind :one-item}
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {}}))

(defn- fixture []
  (scheduled/make
   {:source :semantic-map
    :body (scalar-body)
    :arguments '[input output]
    :effects {:kind :elementwise-map
              :uses [{:value 'input :access :read}
                     {:value 'output :access :write}]}
    :legality {:kind :injective-store}
    :numerics {:mode :exact :policy :same-scalar-evaluation-order}}))

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception (:reason (ex-data exception)))))

(deftest scheduled-body-binds-one-operation-to-one-complete-kernel-plan
  (let [value (fixture)]
    (is (scheduled/scheduled-kernel-body? value))
    (is (= {:kind :elementwise-map
            :uses [{:value 'input :access :read}
                   {:value 'output :access :write}]}
           (:effects value)))
    (is (= :scheduled-kernel-body-effects
           (reason-of #(scheduled/validate!
                        (assoc-in value [:effects :uses]
                                  [{:value 'input :access :read}])))))
    (is (= :scheduled-kernel-body-source
           (reason-of #(scheduled/validate! (assoc value :source nil)))))
    (is (= :scheduled-kernel-body-numerics
           (reason-of #(scheduled/validate! (assoc value :numerics {})))))
    (is (= :scheduled-kernel-body-numerics
           (reason-of #(scheduled/validate!
                        (assoc-in value [:numerics :result-transform]
                                  {:kind :typed-scalar-region
                                   :policy :unverified
                                   :input-dtype :float
                                   :result-dtype :float})))))
    (is (= :scheduled-kernel-body-effects
           (reason-of #(scheduled/validate!
                        (assoc-in value [:effects :reads] ['output])))))))

(deftest duplicate-storage-bindings-require-an-explicit-alias-schedule
  (let [kernel-body (scalar-body)]
    (is (= :scheduled-kernel-body-alias
           (reason-of #(scheduled/derive-uses kernel-body
                                               '[same-buffer same-buffer]))))))

(deftest scheduled-launch-is-closed-over-typed-scalar-parameters
  (let [pointer-launch (assoc (scalar-body) :launch
                              (launch/spec {:workgroup-size [1]
                                            :group-count [(launch/runtime-value 'x)]}))
        with-bound (update (scalar-body) :parameters conj
                           (body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound))
        with-bound (assoc with-bound :launch
                          (launch/spec {:workgroup-size [1]
                                        :group-count [(launch/ceil-div '_n_bound 16)]}))
        value (scheduled/make
               {:source :semantic-map
                :body with-bound
                :arguments '[input output logical-n]
                :effects {:kind :elementwise-map
                          :uses [{:value 'input :access :read}
                                 {:value 'output :access :write}]}
                :legality {:kind :injective-store}
                :numerics {:mode :exact :policy :same-scalar-evaluation-order}})]
    (is (= #{'logical-n}
           (launch/expression-references
            (first (:group-count (scheduled/realized-launch value))))))
    (is (= :scheduled-kernel-body-launch-closure
           (reason-of #(scheduled/make
                        {:source :semantic-map
                         :body pointer-launch
                         :arguments '[input output]
                         :effects {:kind :elementwise-map
                                   :uses [{:value 'input :access :read}
                                          {:value 'output :access :write}]}
                         :legality {:kind :injective-store}
                         :numerics {:mode :exact
                                    :policy :same-scalar-evaluation-order}}))))))

(deftest scheduled-body-validates-against-the-exact-graph-node
  (let [value (fixture)
        node (graph/->ScheduledKernel
              :node :semantic-map
              [(graph/->ValueUse 'input :read) (graph/->ValueUse 'output :write)] [])
        kernel-graph
        (graph/make
         {:inputs [(graph/buffer 'input :float 1 :global :input)]
          :outputs [(graph/buffer 'output :float 1 :global :output)]
          :nodes [node]})]
    (is (identical? value (scheduled/validate-against-node! value node kernel-graph)))
    (testing "source identity and memory effects are independent obligations"
      (is (= :scheduled-kernel-body-source
             (reason-of #(scheduled/validate-against-node!
                          value (assoc node :operation :other) kernel-graph))))
      (is (= :scheduled-kernel-body-node-effects
             (reason-of #(scheduled/validate-against-node!
                          value (assoc node :uses [(graph/->ValueUse 'input :read)])
                          kernel-graph)))))))

(deftest scalar-and-control-bodies-use-the-same-complete-target-projection
  (let [scheduled (fixture)
        emitted (target/emit-artifact "scheduled_scalar" scheduled :opencl-portable)]
    (is (artifact/kernel-artifact? emitted))
    (is (= :opencl-c (:target emitted)))
    (is (= '[input output] (:arguments emitted)))
    (is (= [:input :output] (mapv :kind (:abi emitted))))
    (is (= :scalar-control (get-in emitted [:attributes :body-family])))
    (is (identical? scheduled (get-in emitted [:provenance :scheduled-operation])))
    (is (= (:effects scheduled) (:effects emitted)))
    (is (= {:target-dialect :opencl-portable}
           (get-in emitted [:attributes :target-facts])))
    (is (identical? scheduled (get-in emitted [:attributes :scheduled-kernel-body])))
    (is (re-find #"__kernel void scheduled_scalar" (:source emitted)))
    (is (= :scheduled-kernel-body-artifact-projection
           (reason-of #(scheduled/validate-artifact-projection!
                        scheduled (update-in emitted [:abi 0] dissoc :aliasing))))
        "the certificate proves stable-read ABI preconditions, not just kinds and dtypes")
    (is (= :kernel-body-target-parameter-name
           (reason-of #(target/emit-artifact
                        "scheduled_scalar" scheduled :opencl-portable
                        {:parameter-names {'x "default"}}))))))

(deftest fragment-vocabulary-cannot-fall-through-the-scalar-target
  (let [matrix {:family :dpas :m 1 :n 1 :k 1 :subgroup 1}
        fragment-body (-> (scalar-body)
                          (assoc :fragments
                                 [(body/->Fragment :acc :float [1 1]
                                                   (layout/mma-frag matrix :float))])
                          (update :operations conj (body/->FragmentInit :acc 0.0)))
        value (scheduled/make
               {:source :semantic-map
                :body fragment-body
                :arguments '[input output]
                :effects {:kind :elementwise-map
                          :uses [{:value 'input :access :read}
                                 {:value 'output :access :write}]}
                :legality {:kind :test-fragment-routing}
                :numerics {:mode :exact :policy :same-scalar-evaluation-order}})]
    (is (= :kernel-body-opencl-unimplemented
           (reason-of #(target/emit-artifact
                        "fragment_without_mad" value :opencl-portable))))))

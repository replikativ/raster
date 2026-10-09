(ns raster.compiler.ir.scheduled-kernel-body-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-target :as target]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.numerical-contract :as numerics]
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

(deftest scalar-realization-must-match-retained-numerical-consent
  (let [policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        expression (body/scalar-expression :tanh :float ['value])
        ordinary (assoc-in (fixture) [:body :operations 1 :value] expression)
        widened (assoc-in ordinary [:body :operations 1 :value :options :math-realization]
                          numerics/widened-target-library-math)
        selected (assoc-in widened [:numerics :scalar-math] policy)]
    (is (scheduled/scheduled-kernel-body? (scheduled/validate! ordinary)))
    (is (scheduled/scheduled-kernel-body? (scheduled/validate! selected)))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/validate! widened))))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/validate! (assoc-in ordinary [:numerics :scalar-math] policy)))))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/validate! (update selected :numerics dissoc :scalar-math)))))
    (is (= :scalar-math-policy
           (reason-of #(scheduled/validate! (assoc-in selected [:numerics :scalar-math :extra] true)))))))

(deftest scalar-math-requires-independent-caller-intent
  (let [policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (assoc-in (fixture) [:body :operations 1 :value]
                           (body/scalar-expression :tanh :float ['value]))
        selected (-> ordinary
                     (assoc-in [:body :operations 1 :value :options :math-realization]
                               numerics/widened-target-library-math)
                     (assoc-in [:numerics :scalar-math] policy))]
    (is (= ordinary (scheduled/validate-against-math-policy! ordinary nil)))
    (is (= selected (scheduled/validate-against-math-policy! selected policy)))
    (is (= :scheduled-kernel-body-caller-math-policy
           (reason-of #(scheduled/validate-against-math-policy! selected nil))))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/validate-against-math-policy! ordinary policy))))
    (is (= (fixture) (scheduled/validate-against-math-policy! (fixture) policy))
        "a pure copy may omit an unused math request")
    (is (= :scheduled-kernel-body-caller-math-policy
           (reason-of #(scheduled/validate-against-math-policy!
                        (assoc-in (fixture) [:numerics :scalar-math] policy) nil))))))

(deftest nested-control-retains-scalar-math-consent
  (let [policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (assoc-in (fixture) [:body :masks]
                           [(body/->Mask :active [(body/predicate :lt 0 1)])])
        expression (body/scalar-expression :tanh :float ['value]
                                           {:math-realization numerics/widened-target-library-math})
        nested (assoc-in ordinary [:body :operations 1]
                         (body/->Guard :active [(body/->ScalarStore 'y [0] expression nil)]))
        selected (assoc-in nested [:numerics :scalar-math] policy)]
    (is (scheduled/scheduled-kernel-body? (scheduled/validate! selected)))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/validate! nested))))))

(deftest physical-math-requirements-come-from-checked-executable-leaves
  (let [policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        ordinary (assoc-in (fixture) [:body :operations 1 :value]
                           (body/scalar-expression :tanh :float ['value]))
        selected (-> ordinary
                     (assoc-in [:body :operations 1 :value :options :math-realization]
                               numerics/widened-target-library-math)
                     (assoc-in [:numerics :scalar-math] policy))
        nested (-> selected
                   (assoc-in [:body :masks]
                             [(body/->Mask :active [(body/predicate :lt 0 1)])])
                   (assoc-in [:body :operations 1]
                             (body/->Guard :active [(get-in selected [:body :operations 1])])))]
    (is (= #{} (scheduled/scalar-math-requirements (fixture) policy))
        "an unused override is not a physical FP64 requirement")
    (is (= #{{:operation :tanh :logical-dtype :float :evaluation-dtype :float
              :realization numerics/target-library-math}}
           (scheduled/scalar-math-requirements ordinary nil)))
    (is (= #{{:operation :tanh :logical-dtype :float :evaluation-dtype :double
              :realization numerics/widened-target-library-math}}
           (scheduled/scalar-math-requirements selected policy)))
    (is (= (scheduled/scalar-math-requirements selected policy)
           (scheduled/scalar-math-requirements nested policy)))
    (is (= :scheduled-kernel-body-caller-math-policy
           (reason-of #(scheduled/scalar-math-requirements selected nil))))
    (is (= :scheduled-kernel-body-math-realization
           (reason-of #(scheduled/scalar-math-requirements ordinary policy))))))

(deftest selected-math-needs-affirmative-frozen-target-capability
  (let [policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        selected (-> (fixture)
                     (assoc-in [:body :operations 1 :value]
                               (body/scalar-expression
                                :tanh :float ['value]
                                {:math-realization numerics/widened-target-library-math}))
                     (assoc-in [:numerics :scalar-math] policy))
        supported {:device-id :test-device
                   :execution {:scalar-dtype-support {:double :supported}}}]
    (is (identical? selected (target/validate-math-target! selected policy supported)))
    (doseq [descriptor [{}
                        {:execution {:scalar-dtype-support {:double :unsupported}}}
                        {:execution {:scalar-dtype-support {:double :unknown}}}
                        {:fp64-throughput 1000000000000 :fp64? true}]]
      (is (= :kernel-body-target-math-capability
             (reason-of #(target/validate-math-target! selected policy descriptor)))))
    (is (= (fixture) (target/validate-math-target! (fixture) policy {})))
    (is (= :scheduled-kernel-body-caller-math-policy
           (reason-of #(target/validate-math-target! selected nil supported))))))

(deftest source-arithmetic-cannot-be-dropped-or-forged-in-a-scheduled-certificate
  (let [contract (numerics/blas-source-arithmetic :float)
        value (-> (fixture)
                  (assoc :source {:id :source :source-arithmetic contract})
                  (assoc-in [:numerics :source-arithmetic] contract))]
    (is (= value (scheduled/validate! value)))
    (is (= :scheduled-kernel-body-source-arithmetic
           (reason-of #(scheduled/validate!
                        (update value :numerics dissoc :source-arithmetic)))))
    (is (= :scheduled-kernel-body-source-arithmetic
           (reason-of #(scheduled/validate!
                        (assoc-in value [:numerics :source-arithmetic]
                                  numerics/retained-source-arithmetic)))))
    (is (= :scheduled-kernel-body-source-arithmetic
           (reason-of #(scheduled/validate! (assoc value :source {:id :legacy-source})))))
    (is (= :source-arithmetic
           (reason-of #(scheduled/validate!
                        (assoc-in value [:numerics :source-arithmetic :operands :conversion]
                                  :narrow)))))))

(defn- conditioned-fixture []
  (scheduled/make
   (-> (fixture)
       (update :body update :parameters into
               [(body/->KernelParameter 'n :scalar :int [] nil nil :bound)
                (body/->KernelParameter 'next-n :scalar :int [] nil nil :bound)])
       (assoc :arguments ['input 'output 'rows (launch/sum 'rows 1)]
              :preconditions [{:expression 'n :op :> :value 0}
                              {:expression 'next-n :op :<= :value 8}
                              {:expression (launch/product 'n 'next-n) :op :<= :value 42}])
       (dissoc :scalar-bindings))))

(deftest each-public-schedule-entry-checks-its-body-once
  (let [value (conditioned-fixture)
        check-body body/validate!
        calls (atom 0)
        entries [(fn [value] (scheduled/validate! value))
                 (fn [value] (scheduled/derive-uses (:body value) (:arguments value)))
                 (fn [value] (scheduled/derive-scalar-bindings (:body value) (:arguments value)))
                 (fn [value] (scheduled/derive-scalar-bindings
                              (:body value) (:arguments value) {'rows :int}))
                 scheduled/realized-launch]]
    (with-redefs [body/validate! (fn [kernel-body]
                                  (is (identical? (:body value) kernel-body))
                                  (swap! calls inc)
                                  (check-body kernel-body))]
      (doseq [entry entries]
        (reset! calls 0)
        (entry value)
        (is (= 1 @calls) "one independently checked body per public call")))
    (let [bad (assoc-in value [:body :parameters 0 :dtype] :unknown)]
      (with-redefs [body/validate! (fn [kernel-body]
                                    (swap! calls inc)
                                    (check-body kernel-body))]
        (doseq [entry entries]
          (reset! calls 0)
          (is (thrown? clojure.lang.ExceptionInfo (entry bad)))
          (is (= 1 @calls) "modified bodies are freshly checked, never trusted"))))))

(deftest scheduled-preconditions-use-physical-integral-scalar-identities
  (let [value (conditioned-fixture)]
    (is (= value (scheduled/validate! value)))
    (doseq [symbol ['rows 'input 'unknown]]
      (is (= :kernel-precondition-scope
             (reason-of #(scheduled/validate!
                          (assoc value :preconditions [{:expression symbol :op :> :value 0}]))))))
    (is (= :kernel-preconditions
           (reason-of #(scheduled/validate! (assoc value :preconditions nil)))))
    (is (= :kernel-precondition-scope
           (reason-of #(scheduled/make
                        (-> value
                            (assoc-in [:body :parameters 2 :dtype] :float)
                            (assoc :arguments '[input output rows next-rows])
                            (dissoc :scalar-bindings)))))
        "floating scalar slots are not checked integer-expression inputs")))

(deftest scheduled-preconditions-survive-target-projection-and-graph-preflight
  (let [value (conditioned-fixture)]
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [artifact (target/emit-artifact "conditioned" value dialect)
            graph (target/emit-static-dense-graph "conditioned" value dialect)]
        (is (= (:preconditions value) (:preconditions artifact)))
        (is (= artifact (scheduled/validate-artifact-projection! value artifact)))
        (is (= graph (graph-call/preflight! graph {'rows {:type :int :value 6}})))
        (doseq [rows [0 7 8]]
          (is (= :kernel-precondition-failed
                 (reason-of #(graph-call/preflight! graph {'rows {:type :int :value rows}})))))
        (doseq [conditions [[] (assoc-in (:preconditions value) [2 :value] 100)]]
          (is (= :scheduled-kernel-body-artifact-projection
                 (reason-of #(scheduled/validate-artifact-projection!
                              value (assoc artifact :preconditions conditions))))))))))

(deftest static-dense-body-graph-retains-source-and-required-capacities
  (let [value (-> (fixture)
                  (assoc-in [:body :parameters 0 :shape] [2 3])
                  (assoc-in [:body :operations 0 :coordinates] [0 0])
                  (assoc-in [:body :parameters 0 :layout] (layout/row-major [2 3] :float)))]
    (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
      (let [emitted (target/emit-static-dense-graph "dense_graph" value dialect)
            leaf (get-in emitted [:nodes 0 :operation])]
        (is (= '[input output] (:arguments emitted)))
        (is (= [6 1] (mapv :elements (concat (:inputs emitted) (:outputs emitted)))))
        (is (= (:effects value) (:effects emitted)))
        (is (identical? value (get-in leaf [:attributes :scheduled-kernel-body])))
        (is (= (:source (target/emit-artifact "dense_graph" value dialect)) (:source leaf)))
        (is (empty? (:temporaries emitted)))))))

(deftest static-dense-body-graph-declines-unproved-storage
  (doseq [shape [[0] ['n]]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (target/emit-static-dense-graph
                  "unknown_graph"
                  (-> (fixture)
                      (assoc-in [:body :parameters 0 :shape] shape)
                      (assoc-in [:body :parameters 0 :layout] (layout/row-major shape :float)))
                  :opencl-portable))))
  (is (= :kernel-body-graph-storage
         (reason-of #(target/emit-static-dense-graph
                      "strided_graph"
                      (assoc-in (fixture) [:body :parameters 0 :layout :strides] [2])
                      :opencl-portable))))
  (is (= :kernel-body-graph-capacity
         (reason-of #(target/emit-static-dense-graph
                      "oversized_graph"
                      (-> (fixture)
                          (assoc-in [:body :parameters 0 :shape] [Long/MAX_VALUE])
                          (assoc-in [:body :parameters 0 :layout]
                                    (layout/row-major [Long/MAX_VALUE] :float)))
                      :opencl-portable)))))

(deftest static-dense-body-graph-preserves-public-and-derived-scalars
  (let [base (fixture)
        value (scheduled/make
               (-> base
                   (update :body update :parameters into
                           [(body/->KernelParameter 'n :scalar :int [] nil nil :bound)
                            (body/->KernelParameter 'next-n :scalar :int [] nil nil :bound)])
                   (assoc :arguments ['input 'output 'rows (launch/sum 'rows 1)])
                   (dissoc :scalar-bindings)))
        emitted (target/emit-static-dense-graph "scalar_graph" value :opencl-portable)]
    (is (= '[input output rows] (:arguments emitted)))
    (is (= [(graph/scalar 'rows :int)] (:scalars emitted)))
    (is (= #{'rows} (set (get-in emitted [:nodes 0 :scalar-uses]))))
    (is (= ['input 'output 'rows (launch/sum 'rows 1)]
           (get-in emitted [:nodes 0 :operation :arguments])))))

(deftest static-dense-body-graph-preserves-inout-storage
  (let [base (fixture)
        value (scheduled/make
               (-> base
                   (assoc-in [:body :parameters 1 :kind] :inout)
                   (assoc-in [:effects :uses 1 :access] :read-write)))
        emitted (target/emit-static-dense-graph "inout_graph" value :opencl-portable)]
    (is (= '[input output] (mapv :id (:inputs emitted))))
    (is (= ['output] (mapv :id (:outputs emitted))))
    (is (= [1] (mapv :elements (:outputs emitted))))
    (is (= [:input :inout] (mapv :kind (:abi emitted))))))

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
              [(graph/->ValueUse 'input :read) (graph/->ValueUse 'output :write)] #{} [])
        kernel-graph
        (graph/make
         {:inputs [(graph/buffer 'input :float 1 :global :input)]
          :outputs [(graph/buffer 'output :float 1 :global :output)]
          :scalars []
          :nodes [node]})]
    (is (identical? value (scheduled/validate-against-node! value node kernel-graph)))
    (is (= value (scheduled/validate-against-node!
                  value (graph/map->ScheduledKernel (into {} node)) kernel-graph))
        "equal reconstructed nodes remain valid; membership is not object identity")
    (doseq [storage-path [[:inputs 0 :dtype] [:outputs 0 :dtype]]]
      (is (= :scheduled-kernel-body-node-storage-dtype
             (reason-of #(scheduled/validate-against-node!
                          value node (assoc-in kernel-graph storage-path :double))))
          "matching effects do not permit silently reinterpreting storage"))
    (is (= :scheduled-kernel-body-node-storage-dtype
           (reason-of #(scheduled/validate-against-node!
                        value node
                        (assoc kernel-graph :outputs []
                               :temporaries [(graph/buffer 'output :double 1 :global :temporary)])))))
    (is (= :scheduled-kernel-body-node-storage-dtype
           (reason-of #(scheduled/validate-against-node!
                        value node (assoc-in kernel-graph [:inputs 0 :dtype] :f32))))
        "physical graph storage uses canonical dtype spelling, as the executable ABI requires")
    (is (= :scheduled-kernel-body-node-membership
           (reason-of #(scheduled/validate-against-node!
                        value (assoc node :id :absent-node) kernel-graph))))
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

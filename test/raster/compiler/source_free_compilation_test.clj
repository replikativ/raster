(ns raster.compiler.source-free-compilation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-map-element-test :as map-fixture]
            [raster.ad.normalized-map-sum-test :as sum-fixture]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-surface-map-test :as surface]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as gpu-link]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- pure-map
  ([] (pure-map :double))
  ([dtype]
   (let [array (av/tensor {:dtype dtype :shape '[n]})
         n (av/tensor {:dtype :long :shape []})
         equation (list '= 'map-0 '[y]
                        (list 'map {:index 'i :extent 'n} '[x] []
                              (soac/lambda-form
                               '[element]
                               [(soac/local-value
                                 'squared dtype
                                 (with-meta '(raster.numeric/* element element)
                                   {:raster.type/tag (case dtype :float 'float :double 'double)}))]
                               '[squared])))]
     (soac/make
      (soac/default-program-facts
       {:values {'x array 'n n 'y array}
        :inputs '[x n]
        :equations {'map-0 (soac/default-equation-facts)}})
      [equation] '[y]))))

(defn- synthetic-descriptor [target]
  {:device-id target})

(defn- arguments-for [compilation]
  (mapv (fn [{:keys [value]}]
          (case [(:dtype value) (empty? (:shape value))]
            [:float true] (float 0.5)
            [:double true] (double 0.5)
            [:int true] (int 2)
            [:long true] (long 2)
            [:float false] (float-array [2.0 -3.0])
            [:double false] (double-array [2.0 -3.0])))
        (get-in compilation [:semantic :attributes :invocation-plan :parameters])))

(defn- resident-map-request
  ([compilation] (resident-map-request compilation nil))
  ([compilation input-abstract]
  (let [values (get-in compilation [:emitted :values])
        target (:target compilation)
        input-node (link-plan/node {:id :input-node :dtype :double :shape [2]
                                    :device target :role :input
                                    :source (double-array [2.0 -3.0])})
        output-node (link-plan/node {:id :output-node :dtype :double :shape [2]
                                     :device target :role :output})]
    {:id :resident-square
     :target target
     :nodes {:input-node input-node :output-node output-node}
     :values {:input-value (link-plan/value
                            {:id :input-value :abstract (or input-abstract (get values 'x))
                             :leaves [{:name :value :node :input-node}]})
              :output-value (link-plan/value
                             {:id :output-value :abstract (get values 'y)
                              :leaves [{:name :value :node :output-node}]})}
     :buffer-bindings {'x :input-value 'y :output-value}
     :scalar-values {'n {:type :long :value 2}}})))

(deftest source-free-compilation-uses-the-shared-tail-without-a-frontend
  (let [typed (pure-map)
        calls (atom [])
        reject-frontend (fn [& arguments]
                          (swap! calls conj arguments)
                          (throw (ex-info "frontend must remain unreachable" {:reason :test-frontend})))
        compilation
        (with-redefs [hardware/descriptor-for synthetic-descriptor
                      pipeline/deftm-frontend-facts reject-frontend
                      pipeline/get-walked-body reject-frontend
                      pipeline/run-passes reject-frontend]
          (equation-first/compile-typed
           typed {:target :cuda:source-free
                  :schedule {:precision :f32-scalar}}))]
    (is (empty? @calls))
    (is (equation-first/equation-first-compilation? compilation))
    (is (= :double (:dtype compilation)))
    (is (nil? (:source-ns compilation)))
    (is (= :typed-parallel (get-in compilation [:semantic :dialect])))
    (is (= :scheduled-parallel (get-in compilation [:scheduled :dialect])))
    (is (seq (:kernels compilation)))
    (is (= [:raster.compiler.equation-first/typed-algorithm
            (semantic/fingerprint typed)]
           (:function compilation)))
    (is (= [:raster.compiler.equation-first/typed-compilation
            (semantic/fingerprint typed) :cuda:source-free :double
            {:schedule {:precision :f32-scalar}}]
           (:id compilation)))
    (is (= :equation-artifact-compilation-boundary
           (reason #(#'artifact/validate-compilation! compilation nil)))
        "source-free compilations do not forge a persistent source Var identity")))

(deftest source-free-lowering-materializes-the-exact-public-interface-without-host-evaluation
  (let [input (double-array [2.0 -3.0])
        compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (equation-first/compile-typed
                       (pure-map) {:target :ocl:source-free
                                   :schedule {:precision :f32-scalar}}))
        host-calls (atom 0)
        plan (with-redefs [typed-scalar/evaluate-invocation-step
                           (fn [& _] (swap! host-calls inc) (throw (Exception. "host step")))
                           typed-scalar/evaluate-host-equation
                           (fn [& _] (swap! host-calls inc) (throw (Exception. "host equation")))]
               (equation-first/lower compilation [input 2]))]
    (is (zero? @host-calls))
    (is (= plan (link-plan/validate! plan)))
    (is (= '[x n] (mapv :symbol
                         (get-in compilation [:semantic :attributes
                                              :invocation-plan :parameters]))))
    (let [input-node (get-in plan [:attributes :public-buffer-bindings 'x])]
      (is (identical? input (get-in plan [:nodes input-node :source]))))
    (is (zero? (get-in plan [:attributes :driver-allocations])))
    (is (= #{'y} (set (keys (get-in plan [:attributes :semantic-outputs])))))
    (is (= 1 (count (:outputs plan))))))

(deftest emitted-program-binds-to-existing-resident-link-values-without-initialization-authority
  (let [compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (equation-first/compile-typed
                       (pure-map) {:target :ocl:resident
                                   :schedule {:precision :f32-scalar}}))
        request (resident-map-request compilation)
        emitted (:emitted compilation)
        instance (invocation-link/bind-resident-program emitted request)
        plan (link-plan/make
              {:id :resident-square-plan :target (:target request)
               :nodes (:nodes request) :values (:values request)
               :instances [instance] :outputs [:output-node]})]
    (is (link-plan/program-link-instance? instance))
    (is (= {'x :input-value 'y :output-value}
           (get-in instance [:call :buffers])))
    (is (= plan (link-plan/validate! plan)))
    (is (= #{:input-node}
           (get-in (link-plan/initialization-contract plan) [:reads])))
    (is (= #{:output-node}
           (get-in (link-plan/initialization-contract plan) [:writes])))
    (testing "storage binding does not manufacture initialization"
      (let [uninitialized (-> request
                              (assoc-in [:nodes :input-node :role] :internal)
                              (assoc-in [:nodes :input-node :source] nil))
            bound (invocation-link/bind-resident-program emitted uninitialized)]
        (is (link-plan/program-link-instance? bound))
        (is (= :link-read-before-write
               (reason #(link-plan/make
                         {:id :uninitialized-resident :target (:target request)
                          :nodes (:nodes uninitialized) :values (:values uninitialized)
                          :instances [bound] :outputs [:output-node]}))))))
    (testing "the binder is closed over exact existing storage and scalar identities"
      (is (= :resident-binding-buffers
             (reason #(invocation-link/bind-resident-program
                       emitted (update request :buffer-bindings dissoc 'y)))))
      (is (= :resident-binding-scalars
             (reason #(invocation-link/bind-resident-program
                       emitted (assoc request :scalar-values {})))))
      (is (= :resident-binding-request
             (reason #(invocation-link/bind-resident-program
                       emitted (assoc request :source :forged)))))
      (is (= :link-node-device
             (reason #(invocation-link/bind-resident-program
                       emitted (assoc request :target :ze:resident)))))
      (let [extra-node (link-plan/node {:id :extra-node :dtype :double :shape [2]
                                        :device (:target request) :role :internal})
            multileaf (link-plan/value
                       {:id :input-value :abstract (get-in request [:values :input-value :abstract])
                        :leaves [{:name :left :node :input-node}
                                 {:name :right :node :extra-node}]})]
        (is (= :resident-binding-leaf-count
               (reason #(invocation-link/bind-resident-program
                         emitted
                         (-> request
                             (assoc-in [:nodes :extra-node] extra-node)
                             (assoc-in [:values :input-value] multileaf))))))))))

(deftest resident-binding-preserves-borrowed-program-inputs-and-fresh-validation
  (let [base (pure-map)
        borrowed (apply list
                        (assoc (vec base) 1
                               (assoc-in (soac/facts base) [:values 'x :ownership] :borrowed)))
        compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (equation-first/compile-typed
                       borrowed {:target :ocl:resident-borrowed
                                 :schedule {:precision :f32-scalar}}))
        owned-input (assoc (get-in compilation [:emitted :values 'x]) :ownership :owned)
        request (resident-map-request compilation owned-input)
        emitted (:emitted compilation)
        instance (invocation-link/bind-resident-program emitted request)]
    (is (= :borrowed (get-in instance [:call :program :values 'x :ownership])))
    (is (= :owned (get-in request [:values :input-value :abstract :ownership])))
    (is (= instance
           (link-plan/validate-program-instance-storage!
            (:target request) (:nodes request) (:values request) instance)))
    (is (= :link-value-abstract-ownership
           (reason #(equation-first/lower compilation [(double-array [2.0 -3.0]) 2])))
        "ordinary host materialization remains ownership-strict")
    (testing "full resident validation catches capacity after the logical binding is compatible"
      (let [short-node (link-plan/node {:id :input-node :dtype :double :shape [1]
                                        :device (:target request) :role :input
                                        :source (double-array [2.0])})]
        (is (= :program-link-value-contract
               (reason #(invocation-link/bind-resident-program
                         emitted (assoc-in request [:nodes :input-node] short-node)))))))
    (testing "an inherited operation projection cannot bless a changed emitted owner"
      (let [stale (doto (java.util.IdentityHashMap.))
            tampered (assoc-in emitted [:values 'x :dtype] :float)
            projection-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call
                                       '*validated-boundary-projections*)
            policy-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call
                                   '*validated-projection-policy*)
            expected (reason #(invocation-link/bind-resident-program tampered request))]
        (is (some? expected))
        (is (= expected
               (with-bindings {projection-var stale policy-var {:overrides {}}}
                 (reason #(invocation-link/bind-resident-program tampered request)))))))))

(deftest resident-binding-retains-a-validated-shape-only-array-without-a-kernel-abi-slot
  (let [extent-value (av/tensor {:dtype :double :shape [3]})
        passthrough-value (av/tensor {:dtype :double :shape '[(extent extent-source)]})
        emitted (parallel-program/make
                 {:dialect :opencl-parallel
                  :values {'extent-source extent-value 'passthrough passthrough-value}
                  :inputs '[extent-source passthrough]
                  :equations [] :outputs '[passthrough]})
        validation (emitted-program/validate-with-physical-results! emitted)
        target :ocl:resident-shape
        nodes {:extent-node (link-plan/node {:id :extent-node :dtype :double :shape [3]
                                             :device target :role :input})
               :passthrough-node
               (link-plan/node {:id :passthrough-node :dtype :double :shape [3]
                                :device target :role :input})}
        values {:extent-value
                (link-plan/value {:id :extent-value :abstract extent-value
                                  :leaves [{:name :value :node :extent-node}]})
                :passthrough-value
                (link-plan/value {:id :passthrough-value :abstract passthrough-value
                                  :leaves [{:name :value :node :passthrough-node}]})}
        request {:id :shape-only-resident :target target :nodes nodes :values values
                 :buffer-bindings {'extent-source :extent-value
                                   'passthrough :passthrough-value}
                 :scalar-values {}}
        instance (invocation-link/bind-resident-program emitted request)
        boundaries-fn (ns-resolve 'raster.compiler.ir.invocation-link
                                  'equation-boundaries)
        interface-fn (ns-resolve 'raster.compiler.ir.invocation-link
                                 'resident-runtime-interface)
        boundaries (boundaries-fn emitted (:projections validation))
        interface (interface-fn emitted boundaries (:projections validation))]
    (is (= #{'extent-source 'passthrough} (:buffers interface)))
    (is (empty? (:scalars interface)))
    (is (= (:buffer-bindings request) (get-in instance [:call :buffers])))
    (is (= :passthrough-value (get-in instance [:call :outputs 'passthrough])))
    (is (= instance
           (link-plan/validate-program-instance-storage! target nodes values instance)))
    (testing "an ABI buffer remains a buffer even when its logical shape is rank zero"
      (is (= {:buffers #{'resident} :scalars #{}}
             (interface-fn
              {:values {'resident (av/tensor {:dtype :double :shape []})}
               :equations [] :outputs []}
              [{:graph {:inputs [{:id 'resident}] :outputs [] :scalars []}}]
              (java.util.IdentityHashMap.)))))))

(deftest source-free-compilation-retains-target-policy-and-rejects-interface-overrides
  (doseq [target [:ocl:source-free :ze:source-free :cuda:source-free :hip:source-free]]
    (let [compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                        (equation-first/compile-typed
                         (pure-map :float)
                         {:target target :schedule {:precision :f32-scalar}
                          :scalar-math {:overrides {}}}))]
      (is (= target (:target compilation)))
      (is (= :float (:dtype compilation)))
      (is (seq (:kernels compilation)))
      (is (= {:overrides {}} (get-in compilation [:options :scalar-math])))))
  (doseq [options [{:dtype :double}
                   {:source-ns 'forged.source}
                   {:public-parameters '[other]}
                   {:array-types {'x :float}}
                   {:target :cuda:source-free :frontend :forged}]]
    (is (= :equation-first-typed-options
           (reason #(equation-first/compile-typed (pure-map) options)))))
  (testing "mixed retained floating facts cannot be hidden by a requested carrier"
    (let [base (pure-map)
          typed (apply list
                       (assoc (vec base) 1
                              (update (soac/facts base) :values assoc
                                      'unrelated (av/tensor {:dtype :float :shape []}))))]
      (is (some? (soac/validate! typed)))
      (is (= :equation-first-typed-dtype
             (with-redefs [hardware/descriptor-for synthetic-descriptor]
               (reason #(equation-first/compile-typed
                         typed {:target :cuda:source-free
                                :schedule {:precision :f32-scalar}}))))))))

(deftest existing-deftm-compilation-retains-its-public-boundary-after-tail-factoring
  (doseq [[operation dtype array] [[#'surface/square-double :double double-array]
                                   [#'surface/square-float :float float-array]]
          target [:cuda:source-free :hip:source-free]
          :let [input (array [2.0 -3.0])
                compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                              (equation-first/compile
                               operation {:target target :dtype dtype
                                          :schedule {:precision :f32-scalar}}))
                plan (equation-first/lower compilation [input 2])
                input-node (get-in plan [:attributes :public-buffer-bindings 'xs])]]
    (is (= dtype (:dtype compilation)))
    (is (= dtype (get-in compilation [:semantic :values 'xs :dtype])))
    (is (= 'raster.ad.normalized-surface-map-test (:source-ns compilation)))
    (is (= '[xs size]
           (mapv :symbol (get-in compilation [:semantic :attributes
                                              :invocation-plan :parameters]))))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (identical? input (get-in plan [:nodes input-node :source])))
    (is (= plan (link-plan/validate! plan)))))

(deftest canonical-normalized-phases-compile-and-materializable-inputs-lower
  (doseq [dtype [:float :double]
          :let [map-bundle (normalized/linearize-map-program
                            (#'map-fixture/map-program dtype {})
                            (the-ns 'raster.compiler.source-free-compilation-test))
                sum-bundle (#'sum-fixture/build (#'sum-fixture/source dtype) 'xs)]
          [family bundle facet] [[:map map-bundle :map-programs]
                                 [:sum sum-bundle :sum-programs]]
          phase [:primal :linear :transpose]
          :let [typed (get-in bundle [facet phase])
                borrowed-input? (some #(= :borrowed (:ownership %))
                                      (vals (select-keys (:values (soac/facts typed))
                                                        (:inputs (soac/facts typed)))))
                compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                              (equation-first/compile-typed
                               typed {:target :cuda:source-free
                                      :schedule {:precision :f32-scalar}}))]]
    (is (= dtype (:dtype compilation)) (str family " " phase " carrier"))
    (is (seq (:kernels compilation)) (str family " " phase " kernels"))
    ;; Host primitive arrays materialize as owned uploads. A retained borrowed input needs the
    ;; later same-session/module owner; this front door must expose that boundary, not retag it.
    (if borrowed-input?
      (is (= :link-value-abstract-ownership
             (reason #(equation-first/lower compilation (arguments-for compilation))))
          (str family " " phase " requires a future retained borrowed-buffer owner"))
      (let [plan (equation-first/lower compilation (arguments-for compilation))]
        (is (= plan (link-plan/validate! plan)) (str family " " phase " link plan"))))))

(deftest source-free-compilation-replays-changed-inputs-on-opencl
  (if @opencl/opencl-available?
    (let [initial (float-array [2.0 -3.0 0.5])
          changed (float-array [-1.0 4.0 -2.0])
          initial-before (vec initial)
          changed-before (vec changed)
          compilation (equation-first/compile-typed
                       (pure-map :float)
                       {:target :ocl:0 :schedule {:precision :f32-scalar}})
          plan (equation-first/lower compilation [initial 3])
          input-node (get-in plan [:attributes :public-buffer-bindings 'x])
          output-node (first (:outputs plan))
          executable (gpu-link/instantiate! plan)]
      (try
        (gpu-link/run! executable)
        (is (= [4.0 9.0 0.25]
               (vec (gpu-link/download executable output-node)))
            "the initial source-free invocation executes through the public LinkPlan")
        (gpu-link/upload! executable input-node changed)
        (gpu-link/run! executable)
        (is (= [1.0 16.0 4.0]
               (vec (gpu-link/download executable output-node))))
        (is (= initial-before (vec initial)))
        (is (= changed-before (vec changed)))
        (is (= :none (get-in compilation [:stats :fallback])))
        (is (= 1 (count (:kernels compilation))))
        (finally
          (gpu-link/close! executable)
          (gpu-link/close! executable))))
    (opencl/opencl-skip! "source-free TypedSOAC compilation and changed-input replay")))

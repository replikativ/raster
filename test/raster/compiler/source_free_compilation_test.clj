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
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.core :as gpu]
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

(defn- weighted-sum [dtype]
  (#'sum-fixture/source dtype))

(defn- captured-weighted-sum [dtype]
  (let [program (weighted-sum dtype)
        facts (soac/facts program)
        consumer
        (list '= 'map-2 '[y]
              (list 'map {:index 'j :extent 'size} '[xs] '[loss]
                    (soac/lambda-form '[x total] '[(raster.numeric/* x total)])))]
    (soac/make
     (-> facts
         (assoc-in [:values 'y] (get-in facts [:values 'xs]))
         (assoc-in [:equations 'map-2] (soac/default-equation-facts)))
     (conj (vec (soac/equations program)) consumer)
     '[y])))

(defn- internal-reduction-proof-context [program]
  (let [facts (soac/facts program)
        infos (mapv #'resident/operation-info (soac/equations program))
        info-by-id (into {} (map (juxt :id identity)) infos)
        positions (into {} (map-indexed (fn [index info] [(:id info) index])) infos)
        producers (group-by first
                            (mapcat (fn [info]
                                      (map (fn [result] [result info]) (:results info)))
                                    infos))
        root 'loss]
    {:facts facts :info-by-id info-by-id :positions positions :producers producers
     :outputs (set (soac/outputs program)) :host-reads (set (get-in facts
                                                                    [:attributes :host-read-values]))
     :producer (second (first (get producers root))) :root root
     :sites (get (group-by first (#'resident/use-sites (soac/equations program))) root [])}))

(defn- pure-internal-reduction-root? [program]
  (let [{:keys [facts info-by-id positions producers outputs host-reads producer root sites]}
        (internal-reduction-proof-context program)]
    (#'resident/pure-internal-captured-reduction-root?
     facts info-by-id positions producers outputs host-reads producer root sites)))

(defn- two-weighted-sums [dtype]
  (let [program (weighted-sum dtype)
        equations (vec (soac/equations program))
        reduction (vec (second equations))
        second-reduction (apply list (assoc reduction 1 'sum-2 2 '[loss-2]))
        facts (-> (soac/facts program)
                  (assoc-in [:values 'loss-2] (get-in (soac/facts program) [:values 'loss]))
                  (assoc-in [:equations 'sum-2] (soac/default-equation-facts)))]
    (soac/make facts (conj equations second-reduction) '[loss loss-2])))

(defn- multi-result-weighted-sum [dtype]
  (let [program (weighted-sum dtype)
        equations (vec (soac/equations program))
        reduction (second equations)
        {:keys [attributes arrays captures]} (soac/operation-parts reduction)
        zero (if (= :float dtype) (float 0.0) 0.0)
        left '(raster.numeric/+ left element)
        right '(raster.numeric/+ right element)
        attributes (assoc attributes
                          :accumulators '[left right] :identities [zero zero]
                          :dtypes [dtype dtype]
                          :algebra [(scan/certify-reassociation
                                     {:acc 'left :init zero :lambda left} dtype)
                                    (scan/certify-reassociation
                                     {:acc 'right :init zero :lambda right} dtype)])
        reduction (list '= 'sum-1 '[loss loss-2]
                        (list 'reduce attributes arrays captures
                              (soac/lambda-form '[left right element] [left right])))
        facts (assoc-in (soac/facts program) [:values 'loss-2]
                        (get-in (soac/facts program) [:values 'loss]))]
    (soac/make facts [(first equations) reduction] '[loss loss-2])))

(defn- change-sum-facts [program f]
  (soac/make (f (soac/facts program)) (soac/equations program) (soac/outputs program)))

(defn- float-bits [value]
  (Float/floatToRawIntBits (float value)))

(defn- reassociation-safe-weighted-square-sum [xs weights]
  ;; The source reduction carries an associative-scan certificate, so a device workgroup may
  ;; regroup additions. Test inputs below are exact dyadic Floats whose products and every
  ;; possible partial sum are exact; the serial expression is therefore an independent oracle
  ;; without imposing a left-fold order that the source contract does not promise.
  (reduce (fn [acc [x weight]]
            (let [square (float (* (float x) (float x)))
                  product (float (* square (float weight)))]
              (float (+ (float acc) product))))
          (float 0.0) (map vector xs weights)))

(defn- captured-weighted-sum-result [xs weights]
  (let [loss (reassociation-safe-weighted-square-sum xs weights)]
    (mapv #(float (* (float %) loss)) xs)))

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

(defn- contains-identical?
  [value target]
  (cond
    (identical? value target) true
    (map? value) (or (some #(contains-identical? % target) (keys value))
                     (some #(contains-identical? % target) (vals value)))
    (coll? value) (some #(contains-identical? % target) value)
    :else false))

(defn- resident-layout-fixture []
  (let [input (double-array [2.0 -3.0])
        compilation (with-redefs [hardware/descriptor-for synthetic-descriptor]
                      (equation-first/compile-typed
                       (pure-map) {:target :ocl:resident-layout
                                   :schedule {:precision :f32-scalar}}))
        invocation-plan (get-in compilation [:semantic :attributes :invocation-plan])
        materialized (materialization/materialize
                      invocation-plan [input 2]
                      (fn [& _] (throw (Exception. "unexpected scalar evaluation"))))
        ordinary (equation-first/lower compilation [input 2])
        layout (invocation-link/project-resident-layout
                materialized (:emitted compilation) (:target compilation))]
    {:input input :compilation compilation :materialized materialized
     :ordinary ordinary :layout layout}))

(defn- with-host-equation [emitted]
  (let [answer-value (av/tensor {:dtype :long :shape []})
        host-algorithm
        (soac/make
         (soac/default-program-facts
          {:values {'n (get-in emitted [:values 'n]) 'host-answer answer-value}
           :inputs '[n]
           :equations {'host-scalar (soac/default-equation-facts)}})
         [(list '= 'host-scalar '[host-answer]
                (list 'scalar {:dtypes [:long]} '[n]
                      (soac/lambda-form '[value] [] '[(unchecked-inc value)])))]
         '[host-answer])
        host-equation
        (parallel-program/->ProgramEquation
         [:host-scalar] [:test :host-scalar] nil '[n] '[host-answer]
         host-algorithm [] #{} {:source :test} {:host-only true})]
    (-> emitted
        (assoc-in [:values 'host-answer] answer-value)
        (update :equations conj host-equation))))

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

(deftest resident-layout-is-a-source-free-non-executable-projection-of-ordinary-lowering
  (let [{:keys [input compilation ordinary layout]} (resident-layout-fixture)
        public-bindings (get-in ordinary [:attributes :public-buffer-bindings])
        public-input (get public-bindings 'x)
        ordinary-call (get-in ordinary [:instances 0 :call])]
    (is (= #{:target :nodes :values :buffer-bindings :scalar-values :public-inputs}
           (set (keys layout))))
    (is (not (link-plan/link-plan? layout)))
    (is (not (invocation-link/certified-link? layout)))
    (is (= :ocl:resident-layout (:target layout)))
    (is (= [{:symbol 'x :value public-input}] (:public-inputs layout)))
    (is (= (get-in ordinary [:attributes :compiler-buffer-bindings])
           (:buffer-bindings layout)))
    (is (= (:scalar-values ordinary-call) (:scalar-values layout)))
    (is (= (:values ordinary) (:values layout)))
    (is (identical? input (get-in ordinary [:nodes public-input :source])))
    (is (not (contains-identical? layout input)))
    (is (every? (comp nil? :source val) (:nodes layout)))
    (doseq [[node-id node] (:nodes ordinary)]
      (is (= (cond-> node
               (= node-id public-input) (assoc :role :input :source nil))
             (get-in layout [:nodes node-id]))))
    (let [request (-> layout
                      (dissoc :public-inputs)
                      (assoc :id :resident-layout-binding))
          emitted (:emitted compilation)
          bound (invocation-link/bind-resident-program emitted request)]
      (is (link-plan/program-link-instance? bound))
      (testing "the snapshot grants no proof when its copied storage data is changed"
        (let [input-node (get-in layout [:values public-input :leaves 0 :node])]
          (is (instance?
               clojure.lang.ExceptionInfo
               (try (invocation-link/bind-resident-program
                     emitted (assoc-in request [:nodes input-node :view :shape] [1]))
                    nil (catch clojure.lang.ExceptionInfo error error))))
          (is (some? (reason #(invocation-link/bind-resident-program
                              emitted (assoc-in request [:values public-input :abstract :dtype]
                                                :float)))))
          (is (= :resident-binding-buffers
                 (reason #(invocation-link/bind-resident-program
                           emitted (update request :buffer-bindings dissoc 'x)))))
          (is (= :resident-binding-scalars
                 (reason #(invocation-link/bind-resident-program
                           emitted (assoc request :scalar-values {}))))))))))

(deftest resident-layout-refuses-host-and-non-input-boundaries-before-export
  (let [{:keys [input compilation materialized ordinary]} (resident-layout-fixture)
        project-var (ns-resolve 'raster.compiler.ir.invocation-link
                                'resident-layout-projection)
        order-var (ns-resolve 'raster.compiler.ir.invocation-link
                              'ordered-resident-public-inputs)
        public-input (get-in ordinary [:attributes :public-buffer-bindings 'x])
        output-node (first (:outputs ordinary))
        invoke-project #(project-var % nil)]
    (doseq [role [:state :output]]
      (is (= :resident-layout-public-roles
             (reason #(invoke-project
                       (assoc-in ordinary [:attributes :public-buffer-roles 'x] role))))))
    (is (= :resident-layout-retained-source
           (reason #(invoke-project
                     (-> ordinary
                         (assoc-in [:nodes output-node :role] :state)
                         (assoc-in [:nodes output-node :source] (double-array 2)))))))
    (is (= :resident-layout-host-boundary
           (reason #(invoke-project
                     (assoc-in ordinary [:attributes :driver-allocations] 1)))))
    (is (= :resident-layout-host-boundary
           (reason #(invoke-project
                     (assoc-in ordinary [:attributes :host-outputs] {'host 1})))))
    (is (some? (reason #(invoke-project
                         (update ordinary :instances conj (first (:instances ordinary)))))))
    (is (= :resident-layout-public-inputs
           (reason #(order-var (:plan materialized)
                               {'x public-input 'rogue public-input}))))
    (testing "a malformed initializer is rejected while it is still ordinary materialization"
      (let [changed-buffer (assoc (get-in materialized [:program-buffers 'x])
                                  :source (double-array [2.0]))
            changed (-> materialized
                        (assoc-in [:program-buffers 'x] changed-buffer)
                        (assoc-in [:values (:id changed-buffer)] changed-buffer))]
        (is (= :invocation-materialization-buffer-source
               (reason #(invocation-link/project-resident-layout
                         changed (:emitted compilation) (:target compilation)))))))
    (testing "native initialization is not converted to a resident input premise"
      (let [native (assoc-in (:emitted compilation)
                             [:attributes :native-initialization-providers]
                             {'y {:provider :fixture}})]
        (is (= :invocation-link-native-initialization
               (reason #(invocation-link/project-resident-layout
                         materialized native (:target compilation)))))))
    (testing "a validated host equation reaches the explicit closed-boundary refusal"
      (is (= :resident-layout-host-boundary
             (reason #(invocation-link/project-resident-layout
                       materialized (with-host-equation (:emitted compilation))
                       (:target compilation))))))
    (is (identical? input (get-in ordinary [:nodes public-input :source])))))

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

(deftest source-free-resident-output-demand-is-closed-and-retains-its-emitted-certificate
  (doseq [dtype [:float :double]
          :let [typed (weighted-sum dtype)
                output (first (soac/outputs typed))
                array (case dtype :float float-array :double double-array)
                scheduled-options (atom [])
                schedule-program structured-route/schedule-program
                [compilation plan]
                (with-redefs [hardware/descriptor-for synthetic-descriptor]
                  (with-redefs [structured-route/schedule-program
                                (fn [semantic options]
                                  (swap! scheduled-options conj options)
                                  (schedule-program semantic options))
                                gpu/make-session
                                (fn [& _] (throw (AssertionError. "compile/lower opened a session")))
                                gpu-link/instantiate!
                                (fn [& _] (throw (AssertionError. "compile/lower instantiated")))]
                    (let [compilation
                          (equation-first/compile-typed
                           typed {:target :cuda:resident-output
                                  :resident-outputs [output]
                                  :schedule {:precision :f32-scalar}})]
                      [compilation
                       (equation-first/lower
                        compilation [(array [2.0 -3.0]) (array [0.5 1.0]) 2])])))
                scheduled-value
                (structured-route/validated-public-resident-output
                 (:scheduled compilation) output)
                output-node (first (:outputs plan))]]
    (is (= {:kind :resident-scalar-buffer :elements 1}
           (:representation scheduled-value)))
    (is (= [] (:shape scheduled-value)) "the logical public result remains rank zero")
    (is (= :device (:memory-space scheduled-value)))
    (is (= scheduled-value (get-in compilation [:emitted :values output])))
    (is (= 1 (reduce *' 1 (get-in plan [:nodes output-node :view :shape])))
        "the rank-zero physical view still owns exactly one device element")
    (is (= [output] (get-in compilation [:options :resident-outputs])))
    (is (seq @scheduled-options))
    (is (every? #(not (contains? % :resident-outputs)) @scheduled-options)
        "the demand is consumed once rather than replayed inside component scheduling")
    (is (= [output] (get-in compilation [:semantic :outputs])))
    (is (= (:inputs (soac/facts typed))
           (mapv :symbol (get-in compilation
                                 [:semantic :attributes :invocation-plan :parameters])))
        "physical realization does not reorder the logical public input interface")
    (is (= :none (get-in compilation [:stats :fallback])))
    (testing "absence of a demand preserves the old output-only scalar representation"
      (let [ordinary (with-redefs [hardware/descriptor-for synthetic-descriptor]
                       (equation-first/compile-typed
                        typed {:target :cuda:resident-output
                               :schedule {:precision :f32-scalar}}))]
        (is (= {:kind :plain} (get-in ordinary [:scheduled :values output :representation])))
        (is (nil? (structured-route/validated-public-resident-output
                   (:scheduled ordinary) output)))
        (is (not= (:id compilation) (:id ordinary))
            "the explicit representation demand participates in compilation identity")
        (is (not (contains? (:options ordinary) :resident-outputs)))))))

(deftest source-free-resident-output-demand-preserves-multiple-independent-output-identities
  (let [typed (two-weighted-sums :float)
        outputs (soac/outputs typed)
        compilation
        (with-redefs [hardware/descriptor-for synthetic-descriptor]
          (equation-first/compile-typed
           typed {:target :hip:resident-output :resident-outputs outputs
                  :schedule {:precision :f32-scalar}}))]
    (is (= outputs (get-in compilation [:options :resident-outputs])))
    (is (= outputs (get-in compilation [:scheduled :outputs])))
    (is (every? #(structured-route/validated-public-resident-output
                  (:scheduled compilation) %)
                outputs))
    (is (every? #(= {:kind :resident-scalar-buffer :elements 1}
                    (get-in compilation [:emitted :values % :representation]))
                outputs))))

(deftest source-free-resident-output-demand-selects-an-exact-output-subset
  (let [typed (two-weighted-sums :float)
        outputs (soac/outputs typed)
        requested [(second outputs)]
        compilation
        (with-redefs [hardware/descriptor-for synthetic-descriptor]
          (equation-first/compile-typed
           typed {:target :hip:resident-output :resident-outputs requested
                  :schedule {:precision :f32-scalar}}))]
    (is (= outputs (get-in compilation [:semantic :outputs])))
    (is (= outputs (get-in compilation [:scheduled :outputs])))
    (is (= {:kind :plain}
           (get-in compilation [:emitted :values (first outputs) :representation])))
    (is (= {:kind :resident-scalar-buffer :elements 1}
           (get-in compilation [:emitted :values (second outputs) :representation])))
    (is (nil? (structured-route/validated-public-resident-output
               (:scheduled compilation) (first outputs))))
    (is (some? (structured-route/validated-public-resident-output
                (:scheduled compilation) (second outputs))))))

(deftest source-free-promotion-realizes-a-reduction-captured-by-a-later-map
  (let [typed (captured-weighted-sum :float)
        original-inputs (:inputs (soac/facts typed))
        [realized stats] (resident/realize-source-free-reductions typed [])
        loss (get-in (soac/facts realized) [:values 'loss])
        consumer (soac/operation-parts (last (soac/equations realized)))
        compilation
        (with-redefs [hardware/descriptor-for synthetic-descriptor]
          (equation-first/compile-typed
           typed {:target :cuda:captured-reduction
                  :schedule {:precision :f32-scalar}}))
        operations (mapcat :operations (get-in compilation [:scheduled :equations]))
        consumer-operation
        (first (filter #(some #{'y} (segop/operation-outputs %)) operations))
        producer-operations
        (filter #(some #{'loss} (segop/operation-outputs %)) operations)]
    (is (identical? typed (soac/validate! typed)))
    (is (= 1 (:resident-reductions stats)))
    (is (= {:kind :resident-scalar-buffer :elements 1} (:representation loss)))
    (is (= :device (:memory-space loss)))
    (is (some #{'loss} (:captures consumer)))
    (is (some #{'(clojure.core/aget %capture0 0)}
              (tree-seq coll? seq (:body-results (soac/lambda-parts (:lambda consumer))))))
    (is (= original-inputs (:inputs (soac/facts realized))))
    (is (= original-inputs
           (mapv :symbol (get-in compilation
                                 [:semantic :attributes :invocation-plan :parameters]))))
    (is (= {:kind :resident-scalar-buffer :elements 1}
           (get-in compilation [:scheduled :values 'loss :representation])))
    (is (seq producer-operations))
    (is (= #{'xs 'loss} (segop/operation-inputs consumer-operation)))
    (is (= #{'size} (segop/operation-scalars consumer-operation)))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= ['y] (get-in compilation [:scheduled :outputs])))))

(deftest pure-internal-reduction-residency-requires-the-complete-pristine-boundary
  (let [base (captured-weighted-sum :float)
        reduction-id 'sum-1
        change-facts (fn [program f]
                       (soac/make (f (soac/facts program))
                                  (soac/equations program) (soac/outputs program)))
        facet-fixtures
        [[:logical-layout #(assoc-in % [:values 'loss :logical-layout] {:order :row-major})]
         [:memory-space #(assoc-in % [:values 'loss :memory-space] :host)]
         [:ownership #(assoc-in % [:values 'loss :ownership] :owned)]
         [:value-effects #(assoc-in % [:values 'loss :effects] #{:memory/read})]
         [:attributes #(assoc-in % [:values 'loss :attributes] {:opaque true})]]
        effect
        (change-facts base #(-> %
                                (assoc :effects #{:memory/read})
                                (assoc-in [:equations reduction-id :effects] #{:memory/read})))
        alias
        (change-facts base #(assoc-in % [:equations reduction-id :aliases]
                                      {'loss 'mapped}))
        loss-value (get-in (soac/facts base) [:values 'loss])
        stored
        (change-facts
         base #(-> %
                   (assoc-in [:values 'scratch] (assoc loss-value :shape [1]))
                   (assoc :effects #{:memory/write})
                   (assoc-in [:equations reduction-id :effects] #{:memory/write})
                   (assoc-in [:equations reduction-id :aliases] {'loss 'scratch})
                   (assoc-in [:equations reduction-id :attributes :result-storage]
                             [{:destination 'scratch :access :write :host-return :buffer}])))
        unused-base (weighted-sum :float)
        unused (soac/make (soac/facts unused-base) (soac/equations unused-base) '[mapped])
        multi-base (multi-result-weighted-sum :float)
        multi-facts (soac/facts multi-base)
        multi-consumer
        (list '= 'map-2 '[y]
              (list 'map {:index 'j :extent 'size} '[xs] '[loss]
                    (soac/lambda-form '[x total] '[(raster.numeric/* x total)])))
        multi
        (soac/make
         (-> multi-facts
             (assoc-in [:values 'y] (get-in multi-facts [:values 'xs]))
             (assoc-in [:equations 'map-2] (soac/default-equation-facts)))
         (conj (vec (soac/equations multi-base)) multi-consumer) '[y])]
    (is (pure-internal-reduction-root? base))
    (doseq [[label update-facts] facet-fixtures
            :let [fixture (change-facts base update-facts)]]
      (is (identical? fixture (soac/validate! fixture)) (name label))
      (is (not (pure-internal-reduction-root? fixture)) (name label)))
    (doseq [[label fixture] [[:effect effect] [:alias alias] [:storage stored]
                             [:empty-uses unused] [:multi-result multi]]]
      (is (identical? fixture (soac/validate! fixture)) (name label))
      (is (not (pure-internal-reduction-root? fixture)) (name label)))
    (let [{:keys [facts producers host-reads producer root sites]}
          (internal-reduction-proof-context stored)]
      (is (#'resident/stored-reduction-root?
           facts producers host-reads {'loss 'scratch} producer root sites)
          "valid result storage retains its separately certified historical path"))))

(deftest source-free-reduction-residency-does-not-launder-host-or-extent-scalars
  (let [typed (captured-weighted-sum :float)
        facts (assoc-in (soac/facts typed) [:attributes :host-read-values] ['loss])
        host-escape (soac/make facts (soac/equations typed) (soac/outputs typed))
        [result stats] (resident/realize-source-free-reductions host-escape [])
        base (weighted-sum :float)
        base-facts (soac/facts base)
        dependent-scalar
        (list '= 'scalar-2 '[scaled]
              (list 'scalar {:dtypes [:float]} '[loss]
                    (soac/lambda-form '[total] '[total])))
        dependent-consumer
        (list '= 'map-2 '[y]
              (list 'map {:index 'j :extent 'size} '[xs] '[scaled]
                    (soac/lambda-form '[x total] '[(raster.numeric/* x total)])))
        dependent-host
        (soac/make
         (-> base-facts
             (assoc-in [:values 'scaled] (get-in base-facts [:values 'loss]))
             (assoc-in [:values 'y] (get-in base-facts [:values 'xs]))
             (assoc-in [:equations 'scalar-2] (soac/default-equation-facts))
             (assoc-in [:equations 'map-2] (soac/default-equation-facts))
             (assoc-in [:attributes :host-read-values] ['scaled]))
         (into (vec (soac/equations base)) [dependent-scalar dependent-consumer]) '[y])
        [dependent-result dependent-stats]
        (resident/realize-source-free-reductions dependent-host [])]
    (is (identical? host-escape (soac/validate! host-escape)))
    (is (zero? (:resident-reductions stats)))
    (is (= {:kind :plain} (get-in (soac/facts result) [:values 'loss :representation])))
    (is (= {:kind :plain} (get-in (soac/facts result) [:values 'size :representation])))
    (is (identical? dependent-host (soac/validate! dependent-host)))
    (is (zero? (:resident-reductions dependent-stats)))
    (is (= {:kind :plain}
           (get-in (soac/facts dependent-result) [:values 'loss :representation])))
    (is (some #{'scaled} (mapcat #(nth % 2) (soac/equations dependent-result)))
        "a host-read dependent scalar remains defined rather than beta-reduced")
    (is (some?
         (with-redefs [hardware/descriptor-for synthetic-descriptor]
           (reason #(equation-first/compile-typed
                     host-escape {:target :cuda:captured-reduction
                                  :schedule {:precision :f32-scalar}}))))
        "a blocked scalar capture remains an honest unsupported source-free boundary")))

(deftest empty-resident-output-demand-preserves-supported-reduction-compilation
  (let [typed (weighted-sum :float)
        reduction-realizations (atom [])
        realize-source-free-reductions resident/realize-source-free-reductions
        compile #(with-redefs [hardware/descriptor-for synthetic-descriptor
                               resident/realize-source-free-reductions
                               (fn [program demand]
                                 (swap! reduction-realizations conj demand)
                                 (realize-source-free-reductions program demand))]
                   (equation-first/compile-typed
                    typed (cond-> {:target :cuda:resident-output
                                   :schedule {:precision :f32-scalar}}
                            % (assoc :resident-outputs []))))
        ordinary (compile false)
        explicit-empty (compile true)
        output (first (soac/outputs typed))]
    (is (= [[] []] @reduction-realizations)
        "absent and explicit-empty policies use the same whole-program reduction audit")
    (is (= {:kind :plain}
           (get-in ordinary [:scheduled :values output :representation])))
    (is (semantic/equivalent? (:semantic ordinary) (:semantic explicit-empty)))
    (doseq [field [:inputs :outputs]]
      (is (= (get-in ordinary [:scheduled field])
             (get-in explicit-empty [:scheduled field])))
      (is (= (get-in ordinary [:emitted field])
             (get-in explicit-empty [:emitted field]))))
    (let [public-ids (set (concat (get-in ordinary [:semantic :inputs])
                                  (get-in ordinary [:semantic :outputs])))]
      (is (semantic/equivalent?
           (select-keys (get-in ordinary [:scheduled :values]) public-ids)
           (select-keys (get-in explicit-empty [:scheduled :values]) public-ids)))
      (is (semantic/equivalent?
           (select-keys (get-in ordinary [:emitted :values]) public-ids)
           (select-keys (get-in explicit-empty [:emitted :values]) public-ids))))
    (let [ordinary-numerics
          (mapv #(get-in % [:provenance :scheduled-operation :numerics])
                (:kernels ordinary))
          empty-numerics
          (mapv #(get-in % [:provenance :scheduled-operation :numerics])
                (:kernels explicit-empty))]
      (is (seq ordinary-numerics))
      (is (every? some? ordinary-numerics))
      (is (= ordinary-numerics empty-numerics)
          "the retained numerical schedules are identical"))
    (is (= (get-in ordinary [:stats :emission :emission-routes])
           (get-in explicit-empty [:stats :emission :emission-routes])))
    (is (= [] (get-in explicit-empty [:options :resident-outputs])))
    (is (not (contains? (:options ordinary) :resident-outputs)))
    (is (not= (:id ordinary) (:id explicit-empty))
        "the explicit public policy remains part of compilation identity")))

(deftest source-free-resident-output-demand-refuses-malformed-and-forged-boundaries
  (let [typed (weighted-sum :float)
        output (first (soac/outputs typed))
        facts (soac/facts typed)
        premarked
        (soac/make (-> facts
                       (assoc-in [:values output :representation]
                                 {:kind :resident-scalar-buffer :elements 1})
                       (assoc-in [:values output :memory-space] :device))
                   (soac/equations typed) (soac/outputs typed))
        compile #(with-redefs [hardware/descriptor-for synthetic-descriptor]
                   (equation-first/compile-typed
                    %1 (merge {:target :cuda:resident-output
                               :schedule {:precision :f32-scalar}}
                              %2)))]
    (is (= :equation-first-resident-output-demand
           (reason #(compile typed {:resident-outputs output}))))
    (is (= :equation-first-resident-output-demand
           (reason #(compile typed {:resident-outputs [output output]}))))
    (is (= :equation-first-resident-output-demand
           (reason #(compile typed {:resident-outputs ['not-an-output]}))))
    (is (= :typed-soac-resident-output-demand
           (reason #(compile premarked {:resident-outputs [output]}))))
    (is (= :equation-first-resident-outputs
           (reason #(equation-first/compile
                     #'surface/square-float {:target :cuda:resident-output
                                             :dtype :float
                                             :resident-outputs ['result]}))))))

(deftest source-free-resident-output-demand-does-not-create-authority-for-other-producers
  (let [base (weighted-sum :float)
        output (first (soac/outputs base))
        reduction-id (second (second (soac/equations base)))
        scratch (get-in (soac/facts base) [:values output])
        fixtures
        [[:non-reduction (pure-map :float) 'y]
         [:effect (change-sum-facts
                   base #(-> %
                             (assoc :effects #{:memory/read})
                             (assoc-in [:equations reduction-id :effects] #{:memory/read}))) output]
         [:alias (change-sum-facts
                  base #(-> %
                            (assoc-in [:values 'scratch] scratch)
                            (assoc-in [:equations reduction-id :aliases] {output 'scratch}))) output]
         [:stored (change-sum-facts
                   base #(-> %
                             (assoc-in [:values 'scratch] scratch)
                             (assoc :effects #{:memory/write})
                             (assoc-in [:equations reduction-id :effects] #{:memory/write})
                             (assoc-in [:equations reduction-id :aliases] {output 'scratch})
                             (assoc-in [:equations reduction-id :attributes :result-storage]
                                       [{:destination 'scratch :access :write
                                         :host-return :buffer}]))) output]
         [:multi-result (multi-result-weighted-sum :float) output]]]
    (doseq [[label typed requested] fixtures]
      (is (identical? typed (soac/validate! typed)) (str label " fixture is valid TypedSOAC"))
      (is (some?
           (with-redefs [hardware/descriptor-for synthetic-descriptor]
             (reason #(equation-first/compile-typed
                       typed {:target :cuda:resident-output
                              :resident-outputs [requested]
                              :schedule {:precision :f32-scalar}}))))
          (str label " cannot acquire public resident-output authority")))))

(deftest demanded-source-free-reduction-executes-as-a-resident-scalar-on-opencl-and-level-zero
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                     [:ze:0 ze/gpu-available? ze/gpu-skip!]]]
    (if-not @available?
      (skip! (str "source-free resident scalar reduction on " target))
      (let [xs (float-array [1.5 -2.0 0.5])
            weights (float-array [2.0 0.5 -4.0])
            before-xs (vec xs)
            before-weights (vec weights)
            typed (weighted-sum :float)
            output (first (soac/outputs typed))
            algebra (-> typed soac/equations second soac/operation-parts
                        :attributes :algebra first)
            compilation (equation-first/compile-typed
                         typed {:target target :resident-outputs [output]
                                :schedule {:precision :f32-scalar}})
            plan (equation-first/lower compilation [xs weights 3])
            output-node (first (:outputs plan))
            executable (gpu-link/instantiate! plan)
            expected (reassociation-safe-weighted-square-sum xs weights)]
        (try
          (is (= algebra (scan/validate! algebra))
              "the numerical oracle respects the source's explicit reassociation authority")
          (gpu-link/run! executable)
          (let [actual (first (gpu-link/download executable output-node))]
            (is (= (float-bits expected) (float-bits actual)) (str target)))
          (is (= 1 (reduce *' 1 (get-in plan [:nodes output-node :view :shape]))))
          (is (= before-xs (vec xs)))
          (is (= before-weights (vec weights)))
          (is (= :none (get-in compilation [:stats :fallback])))
          (testing (str target " executes the genuine zero-extent identity reduction")
            (let [empty-xs (float-array 0)
                  empty-weights (float-array 0)
                  zero-plan (equation-first/lower compilation [empty-xs empty-weights 0])
                  zero-node (first (:outputs zero-plan))
                  zero-executable (gpu-link/instantiate! zero-plan)]
              (try
                (gpu-link/run! zero-executable)
                (is (= (float-bits (float 0.0))
                       (float-bits (first (gpu-link/download zero-executable zero-node)))))
                (is (= 1 (reduce *' 1 (get-in zero-plan [:nodes zero-node :view :shape]))))
                (is (empty? empty-xs))
                (is (empty? empty-weights))
                (finally
                  (gpu-link/close! zero-executable)
                  (gpu-link/close! zero-executable)))))
          (finally
            (gpu-link/close! executable)
            (gpu-link/close! executable)))))))

(deftest captured-source-free-reduction-replays-as-a-buffer-capture-on-opencl-and-level-zero
  (doseq [[target available? skip!] [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
                                     [:ze:0 ze/gpu-available? ze/gpu-skip!]]]
    (if-not @available?
      (skip! (str "source-free captured reduction on " target))
      (let [initial (float-array [1.5 -2.0 0.5])
            changed (float-array [-1.0 4.0 -2.0])
            weights (float-array [2.0 0.5 -4.0])
            initial-before (vec initial)
            changed-before (vec changed)
            weights-before (vec weights)
            compilation
            (equation-first/compile-typed
             (captured-weighted-sum :float)
             {:target target :schedule {:precision :f32-scalar}})
            plan (equation-first/lower compilation [initial weights 3])
            input-node (get-in plan [:attributes :public-buffer-bindings 'xs])
            output-node (first (:outputs plan))
            loss-slots
            (for [kernel (:kernels compilation)
                  [slot argument] (map vector (:abi kernel) (:arguments kernel))
                  :when (= 'loss argument)]
              slot)
            executable (gpu-link/instantiate! plan)]
        (try
          (is (seq loss-slots))
          (is (= [:output :input] (mapv :kind loss-slots))
              "the producer and consumer use buffer ABI roles, never a scalar slot")
          (gpu-link/run! executable)
          (is (= (mapv float-bits (captured-weighted-sum-result initial weights))
                 (mapv float-bits (gpu-link/download executable output-node))))
          (gpu-link/upload! executable input-node changed)
          (gpu-link/run! executable)
          (is (= (mapv float-bits (captured-weighted-sum-result changed weights))
                 (mapv float-bits (gpu-link/download executable output-node))))
          (is (= initial-before (vec initial)))
          (is (= changed-before (vec changed)))
          (is (= weights-before (vec weights)))
          (is (= :none (get-in compilation [:stats :fallback])))
          (finally
            (gpu-link/close! executable)
            (gpu-link/close! executable)))))))

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

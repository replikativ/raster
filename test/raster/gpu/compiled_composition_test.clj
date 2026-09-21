(ns raster.gpu.compiled-composition-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as gpu-link]))

(defn component [_x _w _n])

(deftest execution-info-observes-linked-binding-and-rejects-unavailable-evidence
  (let [info {:strategy :chosen :entry-points ["chosen_kernel"]}
        session (atom {:prepared {:phase {:execution-info info}}})
        live (gpu-link/map->LinkedExecutable
              {:session session :phases [:phase] :closed? (atom false)})
        compiled (compiled/map->Compiled {:executable live})]
    (is (= [{:phase :phase :executable info}] (compiled/execution-info compiled)))
    (is (= :compiled-execution-info-unbound
           (try (compiled/execution-info (compiled/map->Prepared {}))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :link-program-execution-info-unsupported
           (try (gpu-link/execution-info (assoc live :prepared-program :equation-first))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (reset! (:closed? live) true)
    (is (thrown? clojure.lang.ExceptionInfo (compiled/execution-info compiled)))))

(deftest compiled-artifact-projects-the-link-instantiation-report
  (let [report {:timing-source :host-monotonic :total-ns 42}
        live (gpu-link/map->LinkedExecutable {:instantiation-report report})
        artifact (compiled/map->Compiled {:executable live})]
    (is (= report (compiled/instantiation-report artifact)))
    (is (= :compiled-instantiation-report-type
           (:reason (ex-data (try (compiled/instantiation-report {})
                                  (catch clojure.lang.ExceptionInfo error error))))))))

(deftest preparation-report-is-available-before-and-after-instantiation
  (let [report {:kind :resident-descriptor :timing-source :host-monotonic}
        prepared (compiled/map->Prepared {:preparation-report report})
        artifact (compiled/map->Compiled {:preparation-report report})]
    (is (= report (compiled/preparation-report prepared)))
    (is (= report (compiled/preparation-report artifact)))
    (is (= :compiled-preparation-report-type
           (:reason (ex-data (try (compiled/preparation-report {})
                                  (catch clojure.lang.ExceptionInfo error error))))))))

(def ^:private kernel
  (artifact/make
   {:kernel-name "compiled_composition_axpy"
    :source "__kernel void compiled_composition_axpy(float* x, float* w, float* y, long n) {}"
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'w :input :float)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x w y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x w] :writes '[y]}}))

(defn- descriptor []
  {:dtype :float
   :all-params '[x w n]
   :array-params '[x w]
   :scalar-params '[n]
   :array-roles {'x :input 'w :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :map :kernel-name "compiled_composition_axpy" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(deftest semantic-artifacts-compose-before-one-runtime-instantiation
  (let [weight (float-array 16)
        prepare #(compiled/lower #'component [(float-array 16) weight 16]
                                 {:target :ze:0 :constants '[w]})
        [first second]
        (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))
                      gpu-link/instantiate! (fn [& _]
                                              (throw (AssertionError.
                                                      "lower must not contact the runtime")))]
          [(prepare) (prepare)])
        composite
        (compiled/compose
         {:id :semantic-two-layers
          :components [{:id :first :program first}
                       {:id :second :program second}]
          :connections [{:from [:first :y] :to [:second :x]}]
          :shares [[[:first :w] [:second :w]]]
          :outputs [{:key :result :from [:second :y]}]})
        plan (compiled/plan composite)
        first-y (get-in first [:out-tree 0 :node])
        second-x (get-in second [:in-tree 0 :node])
        mapping (get-in composite [:lowering :certificate :node-mapping])]
    (is (compiled/prepared? first))
    (is (compiled/prepared? composite))
    (is (= (get mapping [:first first-y]) (get mapping [:second second-x])))
    (is (= 4 (count (:nodes plan))))
    (is (= [[:first :x] [:first :w]] (mapv :key (:in-tree composite))))
    (is (= [:result] (mapv :key (:out-tree composite))))
    (is (= 2 (count (:instances plan))))
    (is (= :composition (:kind (compiled/preparation-report composite))))
    (is (= [:first :second]
           (mapv :id (:components (compiled/preparation-report composite)))))
    (is (pos? (:total-ns (compiled/preparation-report composite))))
    (is (= 2 (count (compiled/ir composite))))
    (is (= {:map 2} (:steps (compiled/cache-key composite))))))

(deftest exact-prepared-values-compose-without-rederiving-component-certificates
  (let [weight (float-array 16)
        prepare #(compiled/lower #'component [(float-array 16) weight 16]
                                 {:target :ze:0 :constants '[w]})
        [first second] (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))]
                         [(prepare) (prepare)])
        request {:id :sealed-components
                 :components [{:id :first :program first}
                              {:id :second :program second}]
                 :connections [{:from [:first :y] :to [:second :x]}]
                 :shares [[[:first :w] [:second :w]]]
                 :outputs [{:key :result :from [:second :y]}]}
        facts-var (ns-resolve 'raster.compiler.ir.link-plan 'instance-access-facts)]
    (with-redefs-fn
      {#'resident-plan/verify!
       (fn [_] (throw (AssertionError. "fresh Prepared was redundantly verified")))
       facts-var
       (fn [_] (throw (AssertionError. "certified component ABIs were reparsed")))}
      #(is (compiled/prepared? (compiled/compose request))))
    (let [copied (assoc first :preparation-report {:copied true})
          calls (atom 0)
          original resident-plan/verify!]
      (with-redefs [resident-plan/verify! (fn [lowering]
                                           (swap! calls inc)
                                           (original lowering))]
        (is (compiled/prepared?
             (compiled/compose
              (assoc-in request [:components 0 :program] copied))))
        (is (= 2 @calls)
            "copying one sealed component makes the whole public composition verify independently")))))

(deftest repeated-lowerings-share-only-the-immutable-compilation-template
  (compiled/clear-compilation-cache!)
  (try
    (let [compilations (atom 0)
          first-input (float-array 16)
          second-input (float-array 32)
          first-weight (float-array 16)
          second-weight (float-array 32)]
      (with-redefs [pipeline/compile-gpu-program
                    (fn [& _] (swap! compilations inc) (descriptor))]
        (let [first (compiled/lower #'component [first-input first-weight 16]
                                    {:target :ze:0 :constants '[w]})
              second (compiled/lower #'component [second-input second-weight 32]
                                     {:target :ze:0 :constants '[w]
                                      :gemm-precision :mixed-f16-f32})
              stats (compiled/compilation-cache-stats)]
          (is (= 1 @compilations)
              "symbolic compilation is shared across concrete sizes and buffer identities")
          (is (= {:hits 1 :misses 1 :misses-by-reason {:compulsory 1}
                  :compilations 1 :failures 0
                  :entries 1 :entries-by-compiler {:resident-descriptor 1}}
                 (dissoc stats :compile-nanos)))
          (is (= first-input (get-in first [:in-tree 0 :default])))
          (is (= second-input (get-in second [:in-tree 0 :default])))
          (is (false? (get-in (compiled/preparation-report first)
                              [:template :cache-hit?])))
          (is (= :compulsory (get-in (compiled/preparation-report first)
                                     [:template :miss-reason])))
          (is (string? (get-in (compiled/preparation-report first)
                               [:template :semantic-fingerprint])))
          (is (false? (get-in (compiled/preparation-report first)
                              [:template :persistent-cache-eligible?]))
              "an ordinary Var without retained deftm source is process-cache-only")
          (is (contains? (get-in (compiled/preparation-report first)
                                 [:template :persistence-blockers])
                         :retained-deftm-source))
          (is (true? (get-in (compiled/preparation-report second)
                             [:template :cache-hit?])))
          (is (nil? (get-in (compiled/preparation-report second)
                            [:template :miss-reason])))
          (is (= :compulsory
                 (get-in (compiled/preparation-report first)
                         [:resident-plan-template :miss-reason])))
          (is (= :specialization
                 (get-in (compiled/preparation-report second)
                         [:resident-plan-template :miss-reason]))
              "a new bound shape is not a failed compiler-template reuse")
          (is (every? #(and (integer? %) (not (neg? %)))
                      ((juxt :total-ns :link-plan-lowering-ns)
                       (compiled/preparation-report second))))
          (is (not= (:lowering first) (:lowering second))
              "argument-dependent LinkPlan certification remains per invocation"))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest repeated-shapes-rebind-a-source-free-resident-plan-template
  (compiled/clear-compilation-cache!)
  (try
    (let [first-x (float-array 16)
          first-w (float-array 16)
          second-x (float-array 16)
          second-w (float-array 16)]
      (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))]
        (let [first (compiled/lower #'component [first-x first-w 16] {:target :ze:0})
              second (compiled/lower #'component [second-x second-w 16] {:target :ze:0})]
          (is (false? (get-in first [:preparation-report :resident-plan-template :cache-hit?])))
          (is (true? (get-in second [:preparation-report :resident-plan-template :cache-hit?])))
          (is (identical? second-x (get-in second [:lowering :plan :nodes
                                                   (get-in second [:lowering :certificate
                                                                   :bindings 'x]) :source])))
          (is (identical? second-w (get-in second [:lowering :plan :nodes
                                                   (get-in second [:lowering :certificate
                                                                   :bindings 'w]) :source])))
          (is (every? (fn [entry]
                        (let [template @entry]
                          (and (every? nil? (map :source (vals (get-in template [:plan :nodes]))))
                               (= [] (get-in template [:plan :instances 0 :arguments])))))
                      (vals @(var-get #'compiled/resident-plan-template-cache)))
              "the cache retains structure but no caller arrays"))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-is-single-flight
  (compiled/clear-compilation-cache!)
  (try
    (let [started (promise)
          release (promise)
          calls (atom 0)
          compile! (fn []
                     (swap! calls inc)
                     (deliver started true)
                     @release
                     :template)
          first-result (future (#'compiled/cached-compilation-template
                                :same :resident-descriptor compile!))]
      @started
      (let [second-result (future (#'compiled/cached-compilation-template
                                   :same :resident-descriptor compile!))]
        (deliver release true)
        (is (= [:template :template] [@first-result @second-result]))
        (is (= 1 @calls))
        (is (= {:hits 1 :misses 1 :misses-by-reason {:compulsory 1}
                :compilations 1 :failures 0
                :entries 1 :entries-by-compiler {:resident-descriptor 1}}
               (dissoc (compiled/compilation-cache-stats) :compile-nanos)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-converges-after-nested-specialization
  (compiled/clear-compilation-cache!)
  (try
    (let [calls (atom 0)
          first? (atom true)
          first-report (atom nil)
          key-for-revision (fn [revision] [:nested-specialization revision])
          compile! (fn []
                     (swap! calls inc)
                     (when (compare-and-set! first? true false)
                       (dispatch/bump-compiler-definition-revision!))
                     :template)
          first-value
          (binding [compiled/*compilation-template-observer* #(reset! first-report %)]
            (#'compiled/stable-compilation-template
             key-for-revision :resident-descriptor compile!))
          second-report (atom nil)
          second-value
          (binding [compiled/*compilation-template-observer* #(reset! second-report %)]
            (#'compiled/stable-compilation-template
             key-for-revision :resident-descriptor compile!))]
      (is (= [:template :template] [first-value second-value]))
      (is (= 2 @calls)
          "the first request recompiles under the installed specialization epoch; the next hits")
      (is (= 2 (:stabilization-attempts @first-report)))
      (is (false? (:cache-hit? @first-report)))
      (is (= 1 (:stabilization-attempts @second-report)))
      (is (true? (:cache-hit? @second-report)))
      (is (= {:hits 1 :misses 2 :misses-by-reason {:compulsory 2}
              :compilations 2 :failures 0
              :entries 1 :entries-by-compiler {:resident-descriptor 1}}
             (dissoc (compiled/compilation-cache-stats) :compile-nanos))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-fails-loud-on-an-unstable-epoch
  (compiled/clear-compilation-cache!)
  (try
    (let [calls (atom 0)
          error (try
                  (#'compiled/stable-compilation-template
                   (fn [revision] [:unstable revision])
                   :resident-descriptor
                   (fn []
                     (swap! calls inc)
                     (dispatch/bump-compiler-definition-revision!)
                     :never-stable))
                  (catch clojure.lang.ExceptionInfo error error))]
      (is (= :compilation-template-unstable (:reason (ex-data error))))
      (is (= 8 @calls))
      (is (zero? (:entries (compiled/compilation-cache-stats)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest stable-template-preserves-failure-observation-and-does-not-cache-errors
  (compiled/clear-compilation-cache!)
  (try
    (let [report (atom nil)
          error (binding [compiled/*compilation-template-observer* #(reset! report %)]
                  (try
                    (#'compiled/stable-compilation-template
                     (fn [revision] [:failure revision])
                     :resident-descriptor
                     #(throw (ex-info "compile failed" {:reason :expected-failure})))
                    (catch clojure.lang.ExceptionInfo error error)))]
      (is (= :expected-failure (:reason (ex-data error))))
      (is (false? (:success? @report)))
      (is (zero? (:entries (compiled/compilation-cache-stats))))
      (is (= 1 (:failures (compiled/compilation-cache-stats)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest compiler-visible-redefinition-invalidates-structural-templates
  (compiled/clear-compilation-cache!)
  (try
    (let [compilations (atom 0)
          arguments [(float-array 16) (float-array 16) 16]]
      (with-redefs [pipeline/compile-gpu-program
                    (fn [& _] (swap! compilations inc) (descriptor))]
        (compiled/lower #'component arguments {:target :ze:0})
        (dispatch/bump-compiler-definition-revision!)
        (let [second (compiled/lower #'component arguments {:target :ze:0})]
          (is (= 2 @compilations)
              "a changed inlined callee cannot reuse a pre-redefinition template")
          (is (= :invalidation
                 (get-in (compiled/preparation-report second) [:template :miss-reason])))
          (is (= 2 (:entries (compiled/compilation-cache-stats)))))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest already-instantiated-artifacts-are-too-late-to-compose-zero-copy
  (is (= :compiled-composition-component
         (:reason
          (ex-data
           (try
             (compiled/compose
              {:id :late :components [{:id :late :program (compiled/map->Compiled {})}]
               :outputs []})
             (catch clojure.lang.ExceptionInfo error error)))))))

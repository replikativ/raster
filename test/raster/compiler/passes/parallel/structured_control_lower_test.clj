(ns raster.compiler.passes.parallel.structured-control-lower-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-loop-call :as loop-call]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.gpu.structured-loop :as loop-runtime]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]
            [raster.compiler.passes.parallel.structured-control-lower :as lower]))

(defn- loop-program
  ([] (loop-program false))
  ([chained?] (loop-program chained? false))
  ([chained? math?] (loop-program chained? math? true))
  ([chained? math? induction?]
   (let [extent (av/tensor {:dtype :int :shape []})
         trip-index (av/tensor {:dtype :long :shape []})
         scalar (av/tensor {:dtype :float :shape []})
         tensor (av/tensor {:dtype :float :shape '[n]})
         inner-tensor (av/tensor {:dtype :float :shape '[n-in]})
         first-result (if chained? 'u-temporary 'u-next)
         equation
         (list '= 'advance '[u-next]
               (list 'map {:index 'i :extent 'n-in}
                     '[u-in] (if induction? '[alpha-in iteration] '[alpha-in])
                     (soac/lambda-form
                      (if induction? '[u-value alpha-value iteration-value] '[u-value alpha-value])
                      (if induction?
                        (if math?
                          '[(Math/tanh (+ u-value alpha-value (* 0.0 iteration-value)))]
                          '[(+ u-value alpha-value (* 0.0 iteration-value))])
                        (if math? '[(Math/tanh (+ u-value alpha-value))]
                            '[(+ u-value alpha-value)])))))
         first-equation (assoc (vec equation) 1 'advance-first 2 [first-result])
         first-equation (apply list first-equation)
         second-equation
         (list '= 'advance-second '[u-next]
               (list 'map {:index 'i :extent 'n-in}
                     '[u-temporary] '[]
                     (soac/lambda-form '[temporary-value] '[(* 2.0 temporary-value)])))
         equations (if chained? [first-equation second-equation] [equation])
         equation-facts (into {}
                              (map (fn [equation]
                                     [(second equation) (soac/default-equation-facts)]))
                              equations)
         body (soac/make
               (soac/default-program-facts
                {:values {'iteration trip-index 'n-in extent 'alpha-in scalar
                          'u-in inner-tensor 'u-temporary inner-tensor
                          'u-next inner-tensor}
                 :inputs (if induction? '[iteration n-in alpha-in u-in] '[n-in alpha-in u-in])
                 :equations equation-facts})
               equations '[u-next])]
     (control/make
      {:id 'time-loop :effects #{} :provenance {:source :test}
       :attributes {:association :sequential}}
      '[iteration steps]
      [{:outer 'n :parameter 'n-in}
       {:outer 'alpha :parameter 'alpha-in}]
      [{:initial 'u0 :parameter 'u-in :result 'u-next :output 'u-final}]
      body
      {'steps trip-index 'n extent 'alpha scalar 'u0 tensor 'u-final tensor}))))

(defn- enclosing-loop-program [emission]
  (let [algorithm (get-in emission [:schedule :algorithm])
        inputs (control/outer-operands algorithm)
        outputs (control/outer-results algorithm)]
    (parallel-program/make
     {:dialect :opencl-parallel :values (control/outer-values algorithm)
      :inputs inputs :outputs outputs
      :equations [(parallel-program/map->ProgramEquation
                   {:id 'time-loop :operands inputs :results outputs :algorithm algorithm
                    :operations [emission] :effects #{} :provenance {} :attributes {}})]})))

(deftest program-validation-evidence-is-bound-to-independent-math-context
  ;; No target-library leaves: both requests validate this program. Their proofs
  ;; must still not be interchangeable, even though this particular code is identical.
  (let [scheduled (lower/schedule (loop-program true) {:target-device :cpu:0 :dtype :float})
        graph (opencl/generate-kernel-graph (:graph scheduled)
                                           :scalar-types {'alpha-in :float 'iteration :long})
        program (enclosing-loop-program (emitted-loop/make scheduled graph))
        request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        default-proof (emitted-program/validate-with-physical-results! program)
        selected-proof (emitted-program/validate-with-physical-results! program request)]
    (is (= program (emitted-program/validate! program request)))
    (is (emitted-program/retained-validation? program default-proof))
    (is (emitted-program/retained-validation? program default-proof {:scalar-math {:overrides {}}}))
    (is (emitted-program/retained-validation? program selected-proof request))
    (is (identical? selected-proof
                    (emitted-program/checked-retained-validation! program selected-proof request)))
    (is (not (emitted-program/retained-validation? program selected-proof)))
    (is (not (emitted-program/retained-validation? program default-proof request)))
    (doseq [[proof options] [[selected-proof {}] [default-proof request]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (emitted-program/checked-retained-validation! program proof options))))
    (is (not (emitted-program/retained-validation? (assoc program :source :copy) selected-proof request)))
    (is (not (emitted-program/retained-validation? program (assoc selected-proof :extra true) request)))
    (is (not (emitted-program/retained-validation?
              program (assoc default-proof :scalar-math (:scalar-math request)) request)))
    (is (= (emitted-program/retained-numerical-equations program)
           (emitted-program/retained-numerical-equations program request)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (emitted-program/retained-validation? program default-proof
                                                       {:scalar-math {:overrides {[:tanh :double] :f64-target-library-rte-f32}}})))))

(deftest prepared-loop-replay-retains-exact-independent-math-owner
  (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        scheduled (lower/schedule (loop-program true true false) {:target-device :cpu:0 :dtype :float})
        certificates (into {} (map (fn [node]
                                    [(:id node) (segmap-body/schedule
                                                 (:operation node)
                                                 (assoc request :scalar-types {'alpha-in :float 'n-in :int}
                                                        :array-types {'u-in :float 'u-temporary :float 'u-next :float}))]))
                           (get-in scheduled [:graph :nodes]))
        graph (opencl/generate-kernel-graph (:graph scheduled)
                                           :scalar-types {'alpha-in :float 'n-in :int}
                                           :scalar-math (:scalar-math request)
                                           :scheduled-bodies certificates)
        emitted (emitted-loop/make scheduled graph {} request)
        call (program-call/make
              (enclosing-loop-program emitted)
              {'u0 :initial 'u-final :output}
              {'steps {:type :long :value 5} 'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {'u-final :scratch} nil {} nil request)
        events (atom [])
        projection-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call '*validated-boundary-projections*)
        policy-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call '*validated-projection-policy*)
        scopes (atom [])
        observe #(vector (var-get projection-var) (var-get policy-var))
        record-scope #(swap! scopes conj (observe) @(future (observe)))
        foreign (fn [invoke]
                  (with-bindings {projection-var (java.util.IdentityHashMap.)
                                  policy-var (:scalar-math request)} (invoke)))
        executor {:bind! (fn [key _ buffers scalars]
                           (record-scope)
                           (swap! events conj [:bind buffers scalars]) key)
                  :run! (fn [handle] (record-scope) (swap! events conj [:run handle]))
                  :release! (fn [handle] (record-scope) (swap! events conj [:release handle]))}]
    (let [instance (link/program-instance {:id :selected :call call} request)
          plan-request {:id :selected-link :target :ocl:0
                        :nodes (mapv #(link/node {:id % :dtype :float :shape [64]
                                                 :device :ocl:0 :role :state})
                                     [:initial :output :scratch])
                        :values (mapv #(link/value {:id % :abstract (get-in call [:program :values 'u0])
                                                    :leaves [{:name :value :node %}]})
                                      [:initial :output :scratch])
                        :instances [instance] :outputs [:output]}
          {:keys [plan effect-evidence]} (link/make-with-effect-evidence plan-request request)
          program-proof (emitted-program/validate-with-physical-results! (:program call) request)
          seal (:raster.compiler.ir.link-plan/validation-seal (meta effect-evidence))
          genuine-token (seal plan effect-evidence (:scalar-math request))]
      (is (identical? plan (link/validate! plan request)))
      (is (= #{:output} (:outputs (link/initialization-contract plan request))))
      (is (contains? (link/value-accesses plan request) :initial))
      (is (= (:id plan) (:plan (link/memory-report plan request))))
      (is (link/link-plan? (:plan (link/borrow-owned-storage plan request))))
      (is (= #{:state} (set (vals (link/instance-roles plan instance request)))))
      (doseq [query [#(link/initialization-contract plan) #(link/value-accesses plan)
                     #(link/memory-report plan) #(link/borrow-owned-storage plan)
                     #(link/instance-roles plan instance)]]
        (is (thrown? clojure.lang.ExceptionInfo (query))))
      (is (link/retained-effect-evidence? plan effect-evidence request))
      (is (not (link/retained-effect-evidence? plan effect-evidence)))
      (is (not (link/retained-effect-evidence?
                plan (with-meta effect-evidence
                       {:raster.compiler.ir.link-plan/validation-seal (fn [& _] genuine-token)}) request)))
      (is (not (link/retained-effect-evidence? (assoc plan :id :copy) effect-evidence request)))
      (is (thrown? clojure.lang.ExceptionInfo (link/validate! plan)))
      (is (thrown? clojure.lang.ExceptionInfo (link/program-instance {:id :selected :call call})))
      (is (thrown? clojure.lang.ExceptionInfo
                   (link/validate-with-effect-evidence! plan program-proof)))
      (is (identical? plan (:plan (link/validate-with-effect-evidence! plan program-proof request))))
      (let [composed (link/validate-with-certified-effect-facts!
                      plan (:step-facts effect-evidence) request)]
        (is (link/retained-effect-evidence? (:plan composed) (:effect-evidence composed) request)))
      (let [runtime-plan (link/make
                          (update plan-request :nodes
                                  #(mapv (fn [node]
                                           (if (= :initial (:id node))
                                             (assoc node :source (float-array 64)) node)) %)) request)
            session (atom {:device-id :ocl:0 :closed? false})
            driver-events (atom [])
            compiler-vars (mapv #(ns-resolve 'raster.compiler.ir.link-plan %)
                                '[*validated-program-instances* *retained-program-validations* *caller-options*])
            driver-scopes (atom [])
            observe #(mapv var-get (into [projection-var policy-var] compiler-vars))
            observe-driver! #(swap! driver-scopes conj (observe) @(future (observe)))]
        (with-redefs [gpu/alloc! (fn [_ specs] (observe-driver!) (swap! driver-events conj [:allocate (count specs)]))
                      gpu/buffer-view (fn [_ key view] {:key key :view view})
                      gpu/upload-range! (fn [& _] (swap! driver-events conj [:upload]))
                      gpu/bind-kernel-graph! (fn [_ key _ _ _ _]
                                               (swap! driver-events conj [:bind]) {:handle key})
                      gpu/run-kernel-graph! (fn [& _] (swap! driver-events conj [:run]))
                      gpu/release-kernel-graph! (fn [& _] (swap! driver-events conj [:release]))
                      gpu/free-buffer! (fn [& _] (swap! driver-events conj [:free]))]
          (is (thrown? clojure.lang.ExceptionInfo
                       (gpu-link/instantiate! runtime-plan {:session session})))
          (is (empty? @driver-events) "default intent rejects before any driver operation")
          (let [executable (with-bindings (merge {projection-var (java.util.IdentityHashMap.)
                                                  policy-var (:scalar-math request)}
                                                 (zipmap compiler-vars [(java.util.IdentityHashMap.)
                                                                        (java.util.IdentityHashMap.) request]))
                             (gpu-link/instantiate! runtime-plan (assoc request :session session)))]
            (try
              (is (= 3 (count (filter #(= :bind (first %)) @driver-events))))
              (is (= #{:output} (set (keys (gpu-link/run! executable)))))
              (is (= 5 (count (filter #(= :run (first %)) @driver-events))))
              (finally (gpu-link/close! executable))))
          (is (= 3 (count (filter #(= :release (first %)) @driver-events))))
          (is (= 3 (count (filter #(= :free (first %)) @driver-events))))
          (is (= [[nil nil nil nil nil] [nil nil nil nil nil]] @driver-scopes)
              "allocation and its future inherit no compiler projection authority")
          (is (false? (:closed? @session)) "attached session remains caller-owned")))
      (let [cache-var (ns-resolve 'raster.compiler.ir.link-plan '*validated-program-instances*)
            retained-var (ns-resolve 'raster.compiler.ir.link-plan '*retained-program-validations*)
            request-var (ns-resolve 'raster.compiler.ir.link-plan '*caller-options*)
            observed (atom [])
            observe #(mapv var-get [cache-var retained-var request-var])
            projected (with-bindings {cache-var (doto (java.util.IdentityHashMap.) (.put instance true))
                                     retained-var (doto (java.util.IdentityHashMap.)
                                                    (.put (:program call) program-proof))
                                     request-var request}
                        (is (thrown? clojure.lang.ExceptionInfo (link/validate-program-instance! instance))
                            "default public validation cannot borrow selected instance evidence")
                        (link/make-with-final-projection
                         plan-request (fn [candidate]
                                        (swap! observed conj (observe) @(future (observe)))
                                        {:plan candidate :projection :checked})
                         program-proof request))]
        (is (= [[nil nil nil] [nil nil nil]] @observed))
        (is (= :checked (:projection projected)))
        (is (link/retained-effect-evidence? (:plan projected) (:effect-evidence projected) request))))
    (is (thrown? clojure.lang.ExceptionInfo (program-runtime/prepare-with! call executor)))
    (is (empty? @events))
    (is (thrown? clojure.lang.ExceptionInfo (program-runtime/staging-plan call :execution)))
    (is (= 3 (count (program-runtime/staging-plan call :execution request))))
    (let [prepared (foreign #(program-runtime/prepare-with! call executor request))]
      (try
        (is (= 3 (count (filter #(= :bind (first %)) @events))))
        (dotimes [_ 2]
          (is (= (:outputs call) (foreign #(program-runtime/run-prepared! prepared)))))
        (is (= 10 (count (filter #(= :run (first %)) @events))))
        (let [before @events
              genuine ((:raster.gpu.parallel-program/request-seal prepared) prepared)]
          (doseq [forged [(assoc prepared :call (assoc call :attributes {:scalar-math (:scalar-math request)}))
                          (with-meta prepared {:scalar-math (:scalar-math request)})
                          (assoc prepared :raster.gpu.parallel-program/request-seal (fn [_] genuine))]]
            (is (= :parallel-program-request-owner
                   (try (program-runtime/run-prepared! forged) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
            (is (= :parallel-program-request-owner
                   (try (program-runtime/release-prepared! forged) nil
                        (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
          (is (= before @events) "forged owners reject before executor callbacks")
          (is (false? @(:closed? prepared)) "forged release cannot close the genuine owner")
          (is (= (:outputs call) (foreign #(program-runtime/run-prepared! prepared)))))
        (let [report (foreign #(program-runtime/profile-prepared!
                               prepared (fn [handle]
                                          (record-scope)
                                          {:profile [{:phase handle}] :kernel-total-ms 0.1 :device-wall-ms 0.2})))]
          (is (= 5 (:program-graph-count report))))
        (finally (foreign #(program-runtime/release-prepared! prepared)))))
    (is (= 3 (count (filter #(= :release (first %)) @events))))
    (let [sequence (foreign #(program-runtime/prepare-sequence-with!
                             [{:id :selected :call call}] executor request))]
      (try
        (is (= {:selected (:outputs call)}
               (foreign #(program-runtime/run-prepared! sequence))))
        (finally (foreign #(program-runtime/release-prepared! sequence)))))
    (is (seq @scopes))
    (is (every? #(= [nil nil] %) @scopes)
        "bind, replay, profile, release callbacks and their futures cannot inherit proof scopes")))

(deftest structured-loop-scalar-math-consent-is-independent
  (let [scheduled (lower/schedule (loop-program true true) {:target-device :cpu:0 :dtype :float})
        graph (:graph scheduled)
        policy {:overrides {[:tanh :float] :f64-target-library-rte-f32}}
        options {:scalar-types {'alpha-in :float 'iteration :long 'n-in :int}
                 :array-types {'u-in :float 'u-temporary :float 'u-next :float}}
        certificates (fn [caller-options]
                       (into {} (map (fn [node]
                                       [(:id node)
                                        (segmap-body/schedule (:operation node)
                                                             (merge options caller-options))]))
                             (:nodes graph)))
        emit (fn [caller-options]
               (opencl/generate-kernel-graph graph
                                            :scalar-types (:scalar-types options)
                                            :scalar-math (:scalar-math caller-options)
                                            :scheduled-bodies (certificates caller-options)))
        selected (emit {:scalar-math policy})
        default (emitted-loop/make scheduled (emit {}))
        accepted (emitted-loop/make scheduled selected {} {:scalar-math policy})]
    (is (= accepted (emitted-loop/validate! accepted {:scalar-math policy})))
    (is (= scheduled (:schedule accepted)) "math realization cannot change loop semantics")
    (is (= (:arguments (:graph default)) (:arguments selected)))
    (is (= (mapv :dependencies (:nodes (:graph default)))
           (mapv :dependencies (:nodes selected))))
    (is (thrown? clojure.lang.ExceptionInfo (emitted-loop/make scheduled selected)))
    (is (thrown? clojure.lang.ExceptionInfo (emitted-loop/validate! accepted)))
    (is (thrown? clojure.lang.ExceptionInfo
                 (emitted-loop/validate! default {:scalar-math policy})))
    (is (thrown? clojure.lang.ExceptionInfo
                 (emitted-loop/validate! (assoc-in accepted [:attributes :scalar-math] policy))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (emitted-loop/validate! accepted {:scalar-math {:overrides {[:tanh :double] :f64-target-library-rte-f32}}})))
    (is (= default (emitted-loop/validate! default)))
    (let [buffers {'u0 :initial-buffer 'u-final :output-buffer}
          scalars {'steps {:type :long :value 3}
                   'n {:type :int :value 64}
                   'alpha {:type :float :value 0.25}}
          scratch {'u-final :scratch-buffer}
          request {:scalar-math policy}
          call (loop-call/make scheduled selected buffers scalars scratch request)
          program (enclosing-loop-program accepted)
          proof (emitted-program/validate-with-physical-results! program request)
          prepared (program-call/make program buffers scalars scratch nil {} proof request)]
      (is (= call (loop-call/validate! call request)))
      (let [events (atom [])
            executor {:bind! (fn [key graph bindings values]
                               (swap! events conj [:bind bindings values])
                               {:key key :graph graph})
                      :run! (fn [_] (swap! events conj [:run]))
                      :release! (fn [_] (swap! events conj [:release]))}]
        (is (thrown? clojure.lang.ExceptionInfo (loop-runtime/run-with! call executor)))
        (is (empty? @events) "mismatched math intent rejects before binding")
        (is (= (:outputs call) (loop-runtime/run-with! call executor request)))
        (is (= [:bind :run :release :bind :run :release :bind :run :release]
               (mapv first @events)))
        (is (= (mapv #(select-keys (loop-call/iteration-binding call % request)
                                  [:buffers :scalar-values]) (range 3))
               (mapv (fn [[_ bindings values]] {:buffers bindings :scalar-values values})
                     (filter #(= :bind (first %)) @events))))
        (reset! events [])
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"execution probe"
                             (loop-runtime/run-with!
                              call (assoc executor :run! (fn [_]
                                                          (swap! events conj [:run])
                                                          (throw (ex-info "execution probe" {}))))
                              request)))
        (is (= [:bind :run :release] (mapv first @events))
            "selected math keeps failed iteration cleanup"))
      (is (= call (loop-call/validate-in-context! call buffers scalars scratch request)))
      (is (= {'u-in :initial-buffer 'u-next :output-buffer}
             (:buffers (loop-call/iteration-binding call 0 request))))
      (is (= {'u-in :output-buffer 'u-next :scratch-buffer}
             (:buffers (loop-call/iteration-binding call 1 request))))
      (is (= {'u-in :scratch-buffer 'u-next :output-buffer}
             (:buffers (loop-call/iteration-binding call 2 request))))
      (is (thrown? clojure.lang.ExceptionInfo (loop-call/make scheduled selected buffers scalars scratch)))
      (is (thrown? clojure.lang.ExceptionInfo (loop-call/validate! call)))
      (is (thrown? clojure.lang.ExceptionInfo (loop-call/iteration-binding call 0)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (loop-call/validate-in-context! call buffers scalars scratch)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (loop-call/validate! (assoc-in call [:attributes :scalar-math] policy))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (loop-call/validate-in-context! (assoc call :trip-count 4) buffers scalars scratch request)))
      (is (= prepared (program-call/validate! prepared request)))
      (is (= prepared (program-call/validate-with-retained-program! prepared proof request)))
      (is (= call (first (:steps prepared))))
      (is (= prepared (program-call/make program buffers scalars scratch nil {} nil request)))
      (is (= #{:initial-buffer :output-buffer :scratch-buffer}
             (set (program-call/buffer-identities prepared request))))
      (doseq [check [#(program-call/validate! prepared)
                     #(program-call/validate-with-retained-program! prepared proof)
                     #(program-call/make program buffers scalars scratch nil {} proof)
                     #(program-call/buffer-bindings prepared)
                     #(program-call/buffer-identities prepared)
                     #(program-call/map-buffers prepared identity)
                     #(program-call/validate! (assoc-in prepared [:attributes :scalar-math] policy))]]
        (is (thrown? clojure.lang.ExceptionInfo (check))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (program-call/validate-with-retained-program!
                    (assoc-in prepared [:steps 0 :trip-count] 4) proof request)))
      (let [projection-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call
                                       '*validated-boundary-projections*)
            policy-var (ns-resolve 'raster.compiler.ir.emitted-parallel-program-call
                                  '*validated-projection-policy*)
            observed (atom [])
            remapped (program-call/map-buffers
                      prepared (fn [buffer]
                                 (swap! observed conj [@projection-var @policy-var])
                                 [:renamed buffer]) request)]
        (is (= [[nil nil] [nil nil] [nil nil]] @observed))
        (is (= remapped (program-call/validate! remapped request)))
        (is (= (:program prepared) (:program remapped)))
        (is (= #{[:renamed :initial-buffer] [:renamed :output-buffer] [:renamed :scratch-buffer]}
               (set (program-call/buffer-identities remapped request))))))
    (let [program (enclosing-loop-program accepted)
          proof (emitted-program/validate-with-physical-results! program {:scalar-math policy})]
      (is (= program (emitted-program/validate! program {:scalar-math policy})))
      (is (emitted-program/retained-validation? program proof {:scalar-math policy}))
      (is (thrown? clojure.lang.ExceptionInfo (emitted-program/validate! program)))
      (is (thrown? clojure.lang.ExceptionInfo (emitted-program/validate-with-physical-results! program)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (emitted-program/validate! (assoc-in program [:attributes :scalar-math] policy))))
      (is (thrown? clojure.lang.ExceptionInfo (emitted-program/retained-numerical-equations program)))
      (is (= ['time-loop]
             (mapv :id (emitted-program/retained-numerical-equations program {:scalar-math policy})))))
    (let [legacy (update-in default [:graph :nodes]
                            (fn [nodes]
                              (mapv (fn [node source-node]
                                      (assoc-in node [:operation :provenance :scheduled-operation]
                                                (:operation source-node)))
                                    nodes (:nodes graph))))]
      (is (= legacy (emitted-loop/validate! legacy)))
      (is (= :emitted-structured-loop-math-owner
             (try (emitted-loop/validate! legacy {:scalar-math policy}) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest structured-control-takes-the-shared-soac-schedule-vertical
  (let [scheduled (lower/schedule (loop-program) {:target-device :cpu:0 :dtype :float})
        body (:body scheduled)
        graph (:graph scheduled)]
    (is (lower/scheduled-loop? scheduled))
    (is (= :segop (:dialect body)))
    (is (parallel-program/parallel-program? body))
    (is (= 1 (count (:nodes graph))))
    (is (= '[u-in] (mapv :id (:inputs graph))))
    (is (= '[u-next] (mapv :id (:outputs graph))))
    (is (= {:kind :host-repetition :association :sequential}
           (:strategy scheduled)))
    (is (= :typed-soac (get-in scheduled [:attributes :body-dialect])))
    (is (= scheduled (lower/validate! scheduled)))
    (testing "the scheduled graph is re-derived from exact SegOp dataflow"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"storage, uses, or dependencies differ"
           (lower/validate!
            (assoc-in scheduled [:graph :inputs 0 :elements] 'different-extent)))))))

(deftest structured-loop-body-dataflow-becomes-one-verified-iteration-graph
  (let [scheduled (lower/schedule (loop-program true) {:target-device :cpu:0 :dtype :float})
        graph (:graph scheduled)
        [first-node second-node] (:nodes graph)]
    (is (= 2 (count (:nodes graph))))
    (is (= '[u-temporary] (mapv :id (:temporaries graph))))
    (is (= [(:id first-node)] (:dependencies second-node)))
    (is (= '[u-in] (mapv :id (:inputs graph))))
    (is (= '[u-next] (mapv :id (:outputs graph))))
    (is (= '[iteration n-in alpha-in] (mapv :id (:scalars graph)))
        "public scalars retain the typed body input order")
    (let [emitted (opencl/generate-kernel-graph
                   graph :scalar-types {'alpha-in :float 'iteration :long})]
      (is (emitted-loop/emitted-loop? (emitted-loop/make scheduled emitted)))
      (is (every? artifact/kernel-artifact? (map :operation (:nodes emitted))))
      (is (= '[u-in u-next iteration n-in alpha-in] (:arguments emitted)))
      (is (= :opencl-c (get-in emitted [:provenance :target-dialect])))
      (is (= (mapv :operation (:nodes graph))
             (mapv #(get-in % [:operation :provenance :scheduled-operation :source])
                   (:nodes emitted))))
      (is (every? #(re-find #"__kernel void graph_segmap" (:source %))
                  (map :operation (:nodes emitted))))
      (is (some #(re-find #"long iteration" (:source %))
                (map :operation (:nodes emitted))))
      (let [certificate (get-in emitted
                                [:nodes 0 :operation :provenance :scheduled-operation])
            replaced (-> certificate
                         (update :arguments
                                 #(mapv (fn [parameter argument]
                                          (if (= 'alpha-in (:id parameter)) 1.0 argument))
                                        (get-in certificate [:body :parameters]) %))
                         (update :scalar-bindings
                                 #(mapv (fn [binding]
                                          (if (= 'alpha-in (:value binding))
                                            (assoc binding :value 1.0)
                                            binding))
                                        %)))]
        (is (= :scheduled-kernel-body-node-scalars
               (try
                 (scheduled-body/validate-against-node! replaced first-node graph)
                 nil
                 (catch clojure.lang.ExceptionInfo exception
                   (:reason (ex-data exception)))))
            "a schedule cannot silently replace one required public scalar by a constant"))
      (let [call (loop-call/make
                  scheduled emitted
                  {'u0 :initial-buffer 'u-final :output-buffer}
                  {'steps {:type :long :value 3}
                   'n {:type :int :value 64}
                   'alpha {:type :float :value 0.25}}
                  {'u-final :scratch-buffer})]
        (is (= {'u-final :output-buffer} (:outputs call)))
        (is (= {'u-in :initial-buffer 'u-next :output-buffer}
               (:buffers (loop-call/iteration-binding call 0))))
        (is (= {'u-in :output-buffer 'u-next :scratch-buffer}
               (:buffers (loop-call/iteration-binding call 1))))
        (is (= {'u-in :scratch-buffer 'u-next :output-buffer}
               (:buffers (loop-call/iteration-binding call 2))))
        (is (= {:type :long :value 2}
               (get-in (loop-call/iteration-binding call 2)
                       [:scalar-values 'iteration])))
        (testing "a copied SegOp ID cannot hide a changed operation certificate"
          (let [tampered (assoc-in emitted
                                   [:nodes 0 :operation :provenance
                                    :scheduled-operation :source :grid :block-size]
                                   128)]
            (is (thrown-with-msg?
                 clojure.lang.ExceptionInfo #"exact operation|operation certificate"
                 (loop-call/validate! (assoc call :graph tampered))))))
        (testing "target emission cannot change buffers or graph effects"
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"scheduled loop dataflow"
               (loop-call/validate!
                (assoc call :graph
                       (assoc-in emitted [:inputs 0 :elements] 'different-extent)))))
          (is (thrown-with-msg?
               clojure.lang.ExceptionInfo #"scheduled loop dataflow"
               (loop-call/validate!
                (assoc call :graph (assoc emitted :effects {:semantic #{:io}}))))))))))

(deftest zero-trip-loop-returns-the-initial-logical-value-without-scratch
  (let [scheduled (lower/schedule (loop-program) {:target-device :cpu:0 :dtype :float})
        emitted (opencl/generate-kernel-graph
                 (:graph scheduled) :scalar-types {'alpha-in :float 'iteration :long})
        call (loop-call/make
              scheduled emitted
              {'u0 :initial-buffer 'u-final :unused-output-buffer}
              {'steps {:type :long :value 0}
               'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {})]
    (is (= 0 (:trip-count call)))
    (is (= {'u-final :initial-buffer} (:outputs call)))))

(deftest dotimes-derived-loop-clamps-a-negative-runtime-bound-to-zero
  (let [program (loop-program)
        program (control/make
                 (assoc-in (control/facts program)
                           [:attributes :trip-count-semantics]
                           :clamp-nonnegative)
                 (control/loop-index program)
                 (control/invariants program)
                 (control/carried program)
                 (control/body program)
                 (control/outer-values program))
        scheduled (lower/schedule program {:target-device :cpu:0 :dtype :float})
        emitted (opencl/generate-kernel-graph
                 (:graph scheduled) :scalar-types {'alpha-in :float 'iteration :long})
        call (loop-call/make
              scheduled emitted
              {'u0 :initial-buffer 'u-final :unused-output-buffer}
              {'steps {:type :long :value -3}
               'n {:type :int :value 64}
               'alpha {:type :float :value 0.25}}
              {})]
    (is (zero? (:trip-count call)))
    (is (= {'u-final :initial-buffer} (:outputs call)))))

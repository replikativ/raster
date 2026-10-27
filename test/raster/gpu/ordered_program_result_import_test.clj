(ns raster.gpu.ordered-program-result-import-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.binding-completion :as binding-completion]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.passes.scalar.source-import :as source-import]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.terminal-reduction-import-test :as completion-fixture]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :as core]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.dl.gpu-grad-parity :as ze-probe]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(core/deftm compound-frame-owner [left :- Long right :- Long] :- Long left)

(defn- compound-completion-request [operand]
  (let [original (#'completion-fixture/selected-frontend #'compound-frame-owner)
        expression (completion/complete-value
                    (completion/planned-bytecode-input-contract
                     Long/TYPE (completion/bytecode-operand-context
                                {:preserve-source-call-results? true}) :long) operand)
        binder (symbol "completed")
        source (list 'let* [binder expression] binder)
        scalar (av/tensor {:dtype :long :shape []})
        plan (invocation/from-prefix
              {:id :compound-input-frame :parameters '[left right]
               :parameter-values {'left scalar 'right scalar}
               :bindings [[binder expression]] :binding-values {binder scalar}
               :program-values {} :program-inputs [] :program-outputs []})
        semantic (program/make {:dialect :typed-soac :source source
                                :attributes {:invocation-plan plan}})
        witness (entry/call-with-definition-inputs
                 #(source-import/prepare-contextual-source-witness
                   source {:source-var (get-in original [:source-entry :source-var])
                           :source-ns (get-in original [:source-entry :source-ns])
                           :lexical-inputs '[left right]}))]
    {:frontend original :source-program semantic :plan plan :step (first (:steps plan))
     :witness witness}))

(deftest compound-completion-executes-exact-current-ports-in-region-order
  (doseq [[operand cases] [['(clojure.core/- left right) [[11 3 8] [4 19 -15]]]
                         ['(clojure.core/- left left) [[11 3 0] [4 19 0]]]]]
    (let [{:keys [source-program plan step] :as request} (compound-completion-request operand)
          adapter (#'completion-fixture/completion-adapter request)
          state (atom {:value-epoch 0})
          owner (cleanup/owner [])
          executable (assoc (link/map->LinkedExecutable {:closed? (atom false) :execution-state state})
                            ::cleanup/owner owner)]
      (try
        (is (#'invocation-link/ordered-source-completion? step))
        (doseq [[left right expected] cases]
          (let [source-state {:materialization (materialization/begin-materialization plan [(long left) (long right)])}
                values (get-in source-state [:materialization :values])
                operands (into {} (map (fn [{:keys [symbol value]}] [symbol (get values value)])) (:operands step))
                invoke #(#'link/execute-ordered-source-completion executable source-state step %)]
            (swap! state assoc :ordered {:current source-state :compilation
                                         {:semantic source-program :source-completion-adapters {(:id step) adapter}}})
            (is (= {:type :long :value (long expected)} (invoke operands)))
            (is (zero? (get-in source-state [:materialization :next-step])))
            (is (= :link-ordered-source-completion-input
                   (reason #(invoke (update-vals operands (fn [v] (with-meta v {:copied true})))))))
            (is (= :link-ordered-source-completion-input
                   (reason #(invoke (update-vals operands (fn [v] (assoc v :value (int (:value v)))))))))
            (when (= 2 (count operands))
              (is (= :link-ordered-source-completion-input
                     (reason #(invoke {'left (operands 'right) 'right (operands 'left)})))))
            (let [copied (update adapter :input-bindings #(with-meta % {:copied true}))]
              (swap! state assoc-in [:ordered :compilation :source-completion-adapters (:id step)] copied)
              (is (= :source-completion-step-correspondence (reason #(invoke operands)))))
            (swap! state assoc-in [:ordered :compilation :source-completion-adapters (:id step)] adapter)
            (swap! state assoc-in [:ordered :current] (with-meta source-state {:copied true}))
            (is (= :link-ordered-source-completion-epoch (reason #(invoke operands))))))
        (finally (cleanup/release! owner))))))

(deftest prepared-fragments-recheck-actual-readiness-and-always-release
  ;; Dispatch/lifecycle only: stand-in bindings grant no program validation or
  ;; device authority. Real view geometry still drives the initialization guard.
  (doseq [mode [:uninitialized :success :run-failure]]
    (let [allocation (view/allocation {:id :storage :byte-size 8 :device :ocl:0
                                       :memory-space :global :alignment 8
                                       :coherence :explicit-transfer :ownership :owned})
          descriptor (view/view allocation {:id :root :dtype :double :shape [1]})
          source (Object.)
          initialized (if (= mode :uninitialized) [] [descriptor])
          state (atom {:ordered {:execution-id :execution :ledger (volatile! [])
                                 :caller-options {} :initialized initialized :prepared {}
                                 :current source :completed-results
                                 {:result {:view descriptor :resident-ready? true}}}})
          executable {:execution-state state :session (atom {})}
          plan {:instances [(link-plan/map->ProgramLinkInstance {:id :stage :call :stand-in})]
                :nodes {:root {:view descriptor}} :aliases #{}}
          contract {:requires #{:root} :writes #{:root} :full-produces #{:root}}
          packet (Object.)
          primary (ex-info "run failed" {:reason :test-run})
          secondary (ex-info "release failed" {:reason :test-release})
          acquisitions (atom 0)
          events (atom [])]
      (with-redefs-fn
        {#'link/acquire-ordered-storage! (fn [& _] (swap! acquisitions inc) {:node-views {}})
         #'link/resident-program-executor (fn [& _] :executor)
         #'link-plan/retained-program-validations! (fn [& _] :validations)
         #'parallel-program/prepare-sequence-with-retained-programs!
         (fn [& _] (swap! events conj :prepare) packet)
         #'parallel-program/run-prepared!
         (fn [actual]
           (is (identical? packet actual))
           (is (false? (get-in @state [:ordered :completed-results :result :resident-ready?])))
           (swap! events conj :run)
           (when (= mode :run-failure) (throw primary)))
         #'parallel-program/release-prepared!
         (fn [actual]
           (is (identical? packet actual))
           (swap! events conj :release)
           (when (= mode :run-failure) (throw secondary)))}
        (fn []
          ;; Predicted contract flow can permit preparation, never execution.
          (let [fragment (#'link/prepare-ordered-program-fragment!
                          executable plan :evidence contract :position [descriptor])]
            (is (= initialized (get-in @state [:ordered :initialized])))
            (is (identical? packet (get-in @state [:ordered :prepared :position])))
            (is (= :link-ordered-prepared-position
                   (reason #(#'link/prepare-ordered-program-fragment!
                             executable plan :evidence contract :position [descriptor]))))
            (is (= 1 @acquisitions))
            (is (= :link-ordered-prepared-owner
                   (reason #(#'link/run-ordered-program-fragment!
                             executable (assoc fragment :prepared (Object.))))))
            (is (= [:prepare] @events))
            (is (identical? packet (get-in @state [:ordered :prepared :position])))
            (is (true? (get-in @state [:ordered :completed-results :result :resident-ready?])))
            (let [error (try (#'link/run-ordered-program-fragment! executable fragment)
                             nil (catch Throwable error error))]
              (case mode
                :uninitialized
                (do (is (= :link-ordered-uninitialized (:reason (ex-data error))))
                    (is (= [:prepare :release] @events))
                    (is (true? (get-in @state [:ordered :completed-results :result :resident-ready?]))))
                :success
                (do (is (nil? error)) (is (= [:prepare :run :release] @events)))
                :run-failure
                (do (is (identical? primary error))
                    (is (= [secondary] (vec (.getSuppressed primary))))
                    (is (= [:prepare :run :release] @events)))))
            ;; A failed release remains owned for existing cleanup retry.
            (is (= (= mode :run-failure)
                   (contains? (get-in @state [:ordered :prepared]) :position)))
            (when-not (= mode :run-failure)
              (let [before @events]
                (is (= :link-ordered-prepared-owner
                       (reason #(#'link/run-ordered-program-fragment! executable fragment))))
                (is (= before @events))
                (is (empty? (get-in @state [:ordered :prepared])))))
            (is (= initialized (get-in @state [:ordered :initialized])))
            (is (identical? source (get-in @state [:ordered :current])))))))))

(deftest import-table-preflight-requires-exact-keys-and-delegates-owner-validation
  ;; This tests dispatch/preflight only: the stand-in validator grants no actual
  ;; import permission, and no session or numerical program is constructed.
  (let [step (invocation/map->ProgramResultImport {:id :snapshot :symbol 'saved})
        plan {:steps [step]}
        semantic (Object.) emitted (Object.) adapter (Object.)
        compilation {:semantic semantic :emitted emitted :program-result-import-adapters {:snapshot adapter}}
        calls (atom 0)]
    (with-redefs [source-import/validate-program-result-import-step!
                  (fn [actual source physical actual-plan actual-step]
                    (swap! calls inc)
                    (is (identical? adapter actual))
                    (is (identical? semantic source))
                    (is (identical? emitted physical))
                    (is (identical? plan actual-plan))
                    (is (identical? step actual-step))
                    actual)]
      (is (identical? (:program-result-import-adapters compilation)
                       (link/validate-ordered-program-result-imports! compilation plan))))
    (is (= 1 @calls))
    (doseq [table [nil {} {:foreign adapter} {:snapshot adapter :foreign adapter} []]]
      (is (= :link-ordered-result-import-adapters
             (reason #(link/validate-ordered-program-result-imports!
                       (assoc compilation :program-result-import-adapters table) plan)))))
    (let [sessions (atom 0)]
      (with-redefs [invocation-link/begin-ordered-realization
                    (fn [& _] {:materialization {:plan plan}})
                    invocation-link/validate-ordered-source-actions! identity
                    gpu/make-session (fn [& _] (swap! sessions inc) (throw (AssertionError. "must not acquire")))]
        (is (= :link-ordered-result-import-adapters
               (reason #(link/instantiate-ordered-invocation!
                         (dissoc compilation :program-result-import-adapters) [] {})))))
      (is (zero? @sessions)))))

(deftest zero-port-literal-completion-executes-only-under-its-current-owned-boundary
  (let [{:keys [source-program plan step] :as request}
        (#'completion-fixture/witnessed-literal-completion-source (double -0.0))
        adapter (#'completion-fixture/completion-adapter request)
        source-state {:materialization (materialization/begin-materialization plan [(float 1.0)])}
        state (atom {:value-epoch 0
                     :ordered {:current source-state
                               :compilation {:semantic source-program
                                             :source-completion-adapters {(:id step) adapter}}}})
        owner (cleanup/owner [])
        ;; No device/session is needed for a host Object completion. Use a real empty cleanup
        ;; owner so current-owner checks remain active, rather than mocking liveness/admission.
        executable (assoc (link/map->LinkedExecutable {:closed? (atom false) :execution-state state})
                          ::cleanup/owner owner)
        invoke #(#'link/execute-ordered-source-completion executable source-state step %)]
    (try
      (is (#'invocation-link/ordered-source-completion? step)
          "structural executor coverage includes the actual zero-port literal adapter")
      (doseq [operand ['(do 0.0) '(+ 0.0 1.0) '(aget buffer 0) 1M]]
        (is (not (#'invocation-link/ordered-source-completion?
                  (assoc step :expression
                         (completion/complete-value completion/object-input-contract operand))))
            "zero free variables do not grant general host-expression execution"))
      (let [left (invoke {}) right (invoke {})]
        (is (instance? Double left))
        (is (= (Double/doubleToRawLongBits (double -0.0)) (Double/doubleToRawLongBits left)))
        (is (identical? (:operand (completion/canonical-parts (:expression step))) left)
            "the executor follows the compiler-owned original Object literal reference")
        (is (identical? left right) "the source literal pool preserves identity across invocations")
        (is (identical? source-state (get-in @state [:ordered :current])))
        (is (zero? (get-in source-state [:materialization :next-step]))
            "the executor alone does not advance its owning source cursor"))
      (is (= :link-ordered-source-completion-input (reason #(invoke {'fake (Object.)}))))
      (let [calls (atom 0)
            fake (assoc-in adapter [:region :invoke] #(do (swap! calls inc) (double 7.0)))]
        (swap! state assoc-in [:ordered :compilation :source-completion-adapters (:id step)] fake)
        (is (= :source-completion-step-correspondence (reason #(invoke {}))))
        (is (zero? @calls)))
      (swap! state assoc-in [:ordered :compilation :source-completion-adapters (:id step)] adapter)
      (swap! state assoc-in [:ordered :current] (with-meta source-state {:copied true}))
      (is (= :link-ordered-source-completion-epoch (reason #(invoke {}))))
      (finally (cleanup/release! owner)))))

(deftest overlapping-writes-retire-only-uncaptured-resident-result-readiness
  (let [allocation (view/allocation {:id :storage :byte-size 16 :device :ocl:0
                                     :memory-space :global :alignment 8 :coherence :explicit-transfer :ownership :owned})
        first-view (view/view allocation {:id :first :dtype :double :shape [1]})
        tail-view (view/view allocation {:id :tail :dtype :double :shape [1] :byte-offset 8})
        root (Object.)
        captured {:view first-view :resident-ready? true :root-object root}
        state (atom {:ordered {:completed-results
                               {:uncaptured {:view first-view :resident-ready? true}
                                :captured captured
                                :disjoint {:view tail-view :resident-ready? true}}}})]
    (#'link/invalidate-ordered-result-reads! state [first-view])
    (is (false? (get-in @state [:ordered :completed-results :uncaptured :resident-ready?])))
    (is (identical? captured (get-in @state [:ordered :completed-results :captured])))
    (is (identical? root (get-in @state [:ordered :completed-results :captured :root-object])))
    (is (true? (get-in @state [:ordered :completed-results :disjoint :resident-ready?])))))

(deftest singleton-binding-check-normalizes-allocation-id-without-granting-readiness
  ;; Immutable descriptors and stand-in buffer/owner objects only; this helper
  ;; alone cannot establish a successful producer, download or live executable.
  (let [execution :execution
        token :result
        key [:raster.gpu.link/allocation execution :storage]
        allocation (view/allocation {:id :storage :byte-size 8 :device :ocl:0
                                     :memory-space :global :alignment 8 :coherence :explicit-transfer :ownership :owned})
        descriptor (view/view allocation {:id token :dtype :double :shape [1]})
        runtime-allocation (assoc allocation :id key)
        resident (gpu/->ResidentBufferView :session key (assoc descriptor :allocation runtime-allocation))
        owner (cleanup/owner [])
        buffer {:dtype :double :byte-size 8 ::cleanup/owner owner}
        session {:session-id :session :device-id :ocl:0 :closed? false
                 :buffers {key buffer} :buffer-owners {key owner} :allocations {key runtime-allocation}}
        ordered {:execution-id execution :roots {:storage {:key key :allocation allocation}}
                 :node-views {token resident}}
        executable {:session (atom session) :execution-state (atom {:ordered ordered})}
        checked (#'link/ordered-result-binding! executable token descriptor)]
    (is (identical? buffer (:backend-buffer checked)))
    (is (identical? owner (:backend-owner checked)))
    (is (identical? resident (:resident checked)))
    (doseq [changed [(assoc session :session-id :foreign)
                     (assoc session :closed? true)
                     (assoc-in session [:buffer-owners key] nil)
                     (assoc-in session [:buffers key :dtype] :float)
                     (assoc-in session [:allocations key :id] :foreign)]]
      (is (= :link-ordered-result-import-view
             (reason #(#'link/ordered-result-binding!
                       (assoc executable :session (atom changed)) token descriptor)))))))

(defn- repeated-snapshot-compilation [target]
  (entry/call-with-definition-inputs
   (fn []
     (let [original (#'completion-fixture/selected-frontend #'completion-fixture/public-float-return)
           context-contract (:contract (completion/canonical-parts
                                        (:expression (#'completion-fixture/contextual-witness-request 'Object))))
           total (with-meta 'total {:tag 'Object :raster.type/tag 'double})
           reduction (completion/complete-value
                      context-contract '(raster.par/reduce acc 0.0 i 3 (+ acc (clojure.core/aget xs i))))
           snapshot #(completion/complete-value completion/object-input-contract total)
           checked #(completion/complete-value
                     (completion/source-floating-cast-contract :float {:unchecked-math false}) %)
           out (with-meta 'out {:tag 'floats :raster.type/tag 'floats})
           source (list 'let* [total reduction
                              'left-snapshot (snapshot) 'left-rounded (checked 'left-snapshot)
                              'right-snapshot (snapshot) 'right-rounded (checked 'right-snapshot)
                              out '(clojure.core/float-array 1)
                              'written '(raster.par/map-void! i 1
                                          (clojure.core/aset out i
                                            (clojure.core/float (clojure.core/+ left-rounded right-rounded))))] out)
           options {:dtype :double :target-dialect :opencl-portable :resident-reductions? false
                    :target target :values {'xs (av/tensor {:dtype :double :shape [3]})}
                    :public-parameters '[xs] :active-params '[xs] :lexical-inputs '[xs]
                    :array-types {'xs :double} :scalar-types {}
                    :source-var (get-in original [:source-entry :source-var])
                    :source-ns (get-in original [:source-entry :source-ns])
                    :source-policy (get-in original [:source-entry :source-policy])
                    :source-binding-context (binding-completion/capture-resolution-context
                                             #'bytecode/binding-completion-plan)}]
       (frontend/call-with-contextual-reduction-constructor
        (#'pipeline/contextual-reduction-constructor options)
        #(let [numerical (:program (typed-route/attempt source :double {'xs :double} options))
               semantic (structured/promote-program numerical options)
               _ (entry/demand-original-source-frontend! (:source-var options))
               scheduled (structured/schedule-program semantic options)
               emitted (:program (c-family/emit-program scheduled options))]
           (#'equation-first/attach-source-action-adapters
            {:semantic semantic :scheduled scheduled :emitted emitted :target target
             :source-ns (:source-ns options)} options)))))))

(defn- check-repeated-snapshots! [target]
  (let [compilation (repeated-snapshot-compilation target)
        plan (get-in compilation [:semantic :attributes :invocation-plan])
        imports (filterv invocation/program-result-import? (:steps plan))
        xs (double-array 3)
        executable (link/instantiate-ordered-invocation! compilation [xs] {})
        previous-root (atom nil)
        previous-replay (atom nil)]
    (try
      (is (= 2 (count imports)) "both independently demanded snapshots survive numerical/source DCE")
      (is (= 1 (count (set (map #(select-keys % [:equation :result]) imports)))))
      (doseq [input [[1.0 2.0 3.0] [4.0 -2.0 7.0]]]
        (dotimes [i 3] (aset xs i (double (nth input i))))
        (let [downloads (atom 0)
              download! gpu/download-range!]
          (with-redefs [gpu/download-range!
                        (fn [& arguments]
                          (swap! downloads inc)
                          (apply download! arguments))]
            (link/run! executable))
          (let [ordered (:ordered @(:execution-state executable))
                current (:current ordered)
                wrappers (mapv #(get-in current [:materialization :values (:id %)]) imports)
                root (:source (first wrappers))
                records (vals (:completed-results ordered))
                sum (reduce + input)
                output (first (keys (link/outputs executable)))
                actual (link/download executable output)]
            (is (= 1 @downloads) "one successful producer execution causes one internal singleton transfer")
            (is (= 1 (count records)))
            (is (instance? Double root))
            (is (identical? root (:source (second wrappers))))
            (is (identical? root (:root-object (first records))))
            (is (= (mapv :id imports) (mapv :id wrappers)))
            (is (= (Double/doubleToRawLongBits sum) (Double/doubleToRawLongBits root)))
            (is (= [(Float/floatToRawIntBits (float (* 2.0 sum)))]
                   (mapv #(Float/floatToRawIntBits %) actual)))
            (when @previous-root
              (is (not (identical? @previous-root root)) "a new replay captures a fresh root object")
              (is (not (identical? @previous-replay (:import-replay ordered)))))
            (reset! previous-root root)
            (reset! previous-replay (:import-replay ordered)))))
      (finally (link/close! executable)))))

(deftest repeated-direct-result-snapshots-share-one-root-per-local-device-replay
  (if @ocl-probe/opencl-available?
    (check-repeated-snapshots! :ocl:0)
    (ocl-probe/opencl-skip! "repeated ordered numerical-result snapshots"))
  (if @ze-probe/gpu-available?
    (check-repeated-snapshots! :ze:0)
    (ze-probe/gpu-skip! "repeated ordered numerical-result snapshots")))

(deftest retained-numerical-view-keeps-attached-plan-correspondence-separate-from-portable-facts
  (let [{:keys [scheduled emitted plan]}
        (entry/call-with-definition-inputs
         #(let [original (#'completion-fixture/selected-frontend #'completion-fixture/public-float-entry)
                {:keys [source-program plan]}
                (#'completion-fixture/witnessed-completion-source original false)
                options {:dtype :float :target-dialect :opencl-portable}
                scheduled (structured/schedule-program source-program options)
                emitted (:program (c-family/emit-program scheduled options))]
            {:scheduled scheduled :emitted emitted :plan plan}))
        logical (emitted-program/retained-typed-program-view emitted scheduled nil)
        unsupported #(try (fingerprint/fingerprint %) nil
                          (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))
        refuses? (fn [candidate]
                   (try (nil? (emitted-program/retained-typed-program-view emitted candidate nil))
                        (catch clojure.lang.ExceptionInfo _ true)))]
    (is (identical? plan (get-in scheduled [:attributes :invocation-plan])))
    (is (identical? plan (get-in emitted [:attributes :invocation-plan])))
    (is (= :semantic-fingerprint-unsupported (unsupported plan))
        "real retained primitive entry completion carries its actual JVM Class")
    (is (some? logical))
    (is (not (contains? (:attributes logical) :invocation-plan))
        "the portable numerical analysis view conveys no invocation capability")
    (is (= (mapv :algorithm (:equations emitted)) (mapv :algorithm (:equations logical))))
    (doseq [candidate [(assoc-in scheduled [:attributes :invocation-plan] (into {} plan))
                       (update scheduled :attributes dissoc :invocation-plan)
                       (assoc scheduled :outputs [])
                       (assoc-in scheduled [:attributes :allocations] [])
                       (assoc-in scheduled [:values 'unboxed :dtype] :double)
                       (update-in scheduled [:equations 0 :operations]
                                  (fn [_] (dialect/equations (get-in scheduled [:equations 0 :algorithm]))))]]
      (is (refuses? candidate)))
    (let [scheduled (assoc-in scheduled [:attributes :unowned-class] Float)
          emitted (assoc-in emitted [:attributes :unowned-class] Float)]
      (is (= :semantic-fingerprint-unsupported
             (try (emitted-program/retained-typed-program-view emitted scheduled nil) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "no arbitrary Class or global fingerprint fallback is admitted"))))

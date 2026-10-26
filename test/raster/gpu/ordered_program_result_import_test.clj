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
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.passes.scalar.source-import :as source-import]
            [raster.compiler.passes.parallel.structured-control-route :as structured]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.terminal-reduction-import-test :as completion-fixture]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.link :as link]
            [raster.dl.gpu-grad-parity :as ze-probe]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

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

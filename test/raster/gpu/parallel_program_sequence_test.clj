(ns raster.gpu.parallel-program-sequence-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.structured-loop-call :as loop-call]
            [raster.gpu.core :as gpu]
            [raster.gpu.test-lifecycle]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]))

(defn- stub-program [id released]
  ((ns-resolve 'raster.gpu.parallel-program 'own-prepared)
   (program/map->PreparedParallelProgram
   {:call {:id id :outputs {id id}
           :steps [(program-call/map->EmittedEquationCall {})]}
    :plan {:step-keys {0 id}}
    :handles {id id}
    :binding-order [id]
    :run! (fn [handle] (swap! released conj [:run handle]))
    :release! (fn [handle] (swap! released conj [:release handle]))
    :closed? (atom false)})))

(deftest ordered-programs-bind-replay-report-and-release-together
  (let [events (atom [])
        prepared (with-redefs [program/prepare-with!
                               (fn [call _] (stub-program (:id call) events))]
                   (program/prepare-sequence-with!
                    [{:id :first :call {:id :a}}
                     {:id :second :call {:id :b}}]
                    {:bind! identity :run! identity :release! identity}))]
    (try
      (is (= [{:instance :first :step 0 :handle :a}
              {:instance :second :step 0 :handle :b}]
             (program/straight-line-handles prepared)))
      (is (= {:first {:a :a} :second {:b :b}}
             (program/run-prepared! prepared)))
      (is (= [[:run :a] [:run :b]] @events))
      (with-redefs [program-call/execution-order
                    (fn [call _]
                      {:record-time-prologue []
                       :per-replay [{:source {:step 0} :phase (:id call)}]
                       :completion :unproven})]
        (is (= [{:instance :first :step 0} {:instance :second :step 0}]
               (mapv :source (:per-replay (program/execution-order prepared identity))))))
      (is (= [:first :second]
             (mapv :instance (program/execution-info prepared identity))))
      (let [profile (program/profile-prepared!
                     prepared (fn [handle]
                                {:profile [{:phase handle}]
                                 :kernel-total-ms 0.25
                                 :device-wall-ms 0.5}))]
        (is (= 1.0 (:device-wall-ms profile)))
        (is (= 2 (:program-graph-count profile)))
        (is (= :sum-of-graph-events (:timing-scope profile)))
        (is (= [:first :second] (mapv :instance (:profile profile)))))
      (finally
        (program/release-prepared! prepared)))
    (is (= [[:release :b] [:release :a]] (take-last 2 @events)))
    (program/release-prepared! prepared)
    (is (= 2 (count (filter #(= :release (first %)) @events))))))

(deftest later-binding-failure-releases-earlier-program
  (let [events (atom [])]
    (with-redefs [program/prepare-with!
                  (fn [call _]
                    (if (= :b (:id call))
                      (throw (ex-info "second binding failed" {:reason :second-binding}))
                      (stub-program (:id call) events)))]
      (is (= :second-binding
             (try
               (program/prepare-sequence-with!
                [{:id :first :call {:id :a}} {:id :second :call {:id :b}}]
                {:bind! identity :run! identity :release! identity})
               (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= [[:release :a]] @events)))))

(deftest direct-graph-and-program-share-ordered-prepared-sequence
  (let [events (atom [])
        graph {:outputs [{:id :y}]}
        executor {:bind! (fn [_ bound-graph buffers scalars]
                           (is (= [graph {:x :input :y :output} {'n {:type :long :value 4}}]
                                  [bound-graph buffers scalars]))
                           (swap! events conj [:bind :graph])
                           :graph)
                  :run! (fn [handle] (swap! events conj [:run handle]))
                  :release! (fn [handle] (swap! events conj [:release handle]))}
        prepared (with-redefs [program/prepare-with!
                               (fn [call _] (stub-program (:id call) events))]
                   (program/prepare-sequence-with!
                    [{:id :graph :kind :graph :call {:graph graph
                                        :bindings {:x :input :y :output}
                                        :scalar-values {'n {:type :long :value 4}}}}
                     {:id :program :call {:id :after}}]
                    executor))]
    (try
      (is (= [{:instance :graph :step nil :handle :graph}
              {:instance :program :step 0 :handle :after}]
             (program/straight-line-handles prepared)))
      (is (= {:graph {:y :output} :program {:after :after}}
             (program/run-prepared! prepared)))
      (is (= [[:bind :graph] [:run :graph] [:run :after]] @events))
      (is (= [:graph :program]
             (mapv :instance (program/execution-info prepared (constantly :observed)))))
      (with-redefs [program-call/execution-order
                    (fn [_ _] {:record-time-prologue []
                               :per-replay [{:source {:step 0}}]
                               :completion :unproven})]
        (is (= [:graph :program]
               (mapv #(get-in % [:source :instance])
                     (:per-replay
                      (program/execution-order
                       prepared (fn [_] {:record-time-prologue []
                                         :per-replay [{:source {:step 0}}]})))))))
      (let [profile (program/profile-prepared!
                     prepared (fn [handle]
                                {:profile [{:phase handle}]
                                 :kernel-total-ms 0.25
                                 :device-wall-ms 0.5}))]
        (is (= [:graph :program]
               (mapv :instance (:profile profile))))
        (is (= 1.0 (:device-wall-ms profile))))
      (finally (program/release-prepared! prepared)))
    (is (= [[:release :after] [:release :graph]] (take-last 2 @events)))))

(deftest later-replay-failure-does-not-publish-partial-outputs
  (let [events (atom [])
        first-program (stub-program :a events)
        second-program (assoc (stub-program :b events)
                              :run! (fn [_]
                                      (throw (ex-info "second replay failed"
                                                      {:reason :second-replay}))))
        prepared ((ns-resolve 'raster.gpu.parallel-program 'own-prepared)
                  (program/map->PreparedParallelSequence
                  {:instances [{:id :first :program first-program}
                               {:id :second :program second-program}]
                   :closed? (atom false)}))
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :two-programs}
                     :prepared-program prepared
                     :pending-inputs (atom #{})
                     :output-ready? (atom true)
                     :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)
                     :output-leases (atom 0)
                     :closed? (atom false)
                     :lifetime-lock (Object.)})]
    (is (= :second-replay
           (try (link/run! executable)
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= [[:run :a]] @events))
    (is (= :link-execution-poisoned
           (try (link/run! executable)
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= [[:run :a]] @events))
    (is (= 1 (:value-epoch @(:execution-state executable))))
    (is (false? @(:output-ready? executable)))
    (is (zero? @(:completed-replays executable)))))

(deftest dynamic-mixed-order-declines-before-session-creation
  (let [plan {:instances [(link-plan/map->LinkInstance {:id :descriptor})
                          (link-plan/map->ProgramLinkInstance
                           {:id :dynamic
                            :call {:steps [(loop-call/map->StructuredLoopCall {})]}})]}
        sessions (atom 0)]
    (with-redefs [link-plan/validate-with-effect-evidence!
                  (fn [_] {:plan plan :effect-evidence {:initialization {}}})
                  gpu/make-session (fn [& _] (swap! sessions inc))]
      (is (= :link-runtime-dynamic-mixed-order
             (try (link/instantiate! plan)
                  (catch clojure.lang.ExceptionInfo error
                    (:reason (ex-data error))))))
      (is (zero? @sessions)))))

(deftest recorded-mixed-executable-replays-only-the-composite-graph
  (let [runs (atom [])
        prepared (stub-program :program runs)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :mixed} :session :session :graph-key :composite
                     :prepared-program prepared :pending-inputs (atom #{})
                     :output-ready? (atom false) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)
                     :output-leases (atom 0) :closed? (atom false)
                     :lifetime-lock (Object.)})]
    (with-redefs [gpu/replay! (fn [_ key] (swap! runs conj [:replay key]))
                  link/outputs (fn [_] {})]
      (is (= {} (link/run! executable)))
      (is (= [[:replay :composite]] @runs))
      (is (= 1 @(:completed-replays executable)))
      (is (true? @(:output-ready? executable))))))

(ns raster.gpu.parallel-program-sequence-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]))

(defn- stub-program [id released]
  (program/map->PreparedParallelProgram
   {:call {:id id :outputs {id id}
           :steps [(program-call/map->EmittedEquationCall {})]}
    :plan {:step-keys {0 id}}
    :handles {id id}
    :binding-order [id]
    :run! (fn [handle] (swap! released conj [:run handle]))
    :release! (fn [handle] (swap! released conj [:release handle]))
    :closed? (atom false)}))

(deftest ordered-programs-bind-replay-report-and-release-together
  (let [events (atom [])
        prepared (with-redefs [program/prepare-with!
                               (fn [call _] (stub-program (:id call) events))]
                   (program/prepare-sequence-with!
                    [{:id :first :call {:id :a}}
                     {:id :second :call {:id :b}}] {}))]
    (try
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
                [{:id :first :call {:id :a}} {:id :second :call {:id :b}}] {})
               (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= [[:release :a]] @events)))))

(deftest later-replay-failure-does-not-publish-partial-outputs
  (let [events (atom [])
        first-program (stub-program :a events)
        second-program (assoc (stub-program :b events)
                              :run! (fn [_]
                                      (throw (ex-info "second replay failed"
                                                      {:reason :second-replay}))))
        prepared (program/map->PreparedParallelSequence
                  {:instances [{:id :first :program first-program}
                               {:id :second :program second-program}]
                   :closed? (atom false)})
        executable (link/map->LinkedExecutable
                    {:plan {:id :two-programs}
                     :prepared-program prepared
                     :pending-inputs (atom #{})
                     :output-ready? (atom true)
                     :completed-replays (atom 0)
                     :output-leases (atom 0)
                     :closed? (atom false)
                     :lifetime-lock (Object.)})]
    (is (= :second-replay
           (try (link/run! executable)
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= [[:run :a]] @events))
    (is (false? @(:output-ready? executable)))
    (is (zero? @(:completed-replays executable)))))

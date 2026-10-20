(ns raster.ci.timed-runner-test
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [cognitect.test-runner :as runner]
            [raster.ci.timed-runner :as timed]))

(deftest timing-preserves-results-and-records-exceptions
  (let [times (atom {})]
    (is (= :result (timed/measured times [:ok] (constantly :result))))
    (is (thrown-with-msg? Exception #"probe"
                          (timed/measured times [:error]
                                          #(throw (Exception. "probe")))))
    (is (every? #(and (integer? %) (not (neg? %))) (vals @times)))
    (is (= #{:ok :error} (set (keys @times))))))

(deftest runner-delegates-selection-and-preserves-failure-summary
  (let [file (java.io.File/createTempFile "raster-timings-" ".edn")
        options {:namespace #{'raster.ci.timed-runner-test} :exclude #{:perf}}
        summary {:test 3 :pass 2 :fail 1 :error 0}
        seen (atom nil)]
    (try
      (with-redefs [runner/test (fn [actual] (reset! seen actual) summary)]
        (is (= summary (timed/run-timed! options file))))
      (is (= options @seen))
      (let [report (edn/read-string (slurp file))]
        (is (= 1 (:schema-version report)))
        (is (= :completed (:run-state report)))
        (is (= summary (:summary report))
            "a completed failing run is not reported as an aborted run"))
      (finally (io/delete-file file)))))

(deftest load-errors-still-publish-the-timing-report
  (let [file (java.io.File/createTempFile "raster-timings-error-" ".edn")]
    (try
      (with-redefs [runner/test (fn [_] (throw (Exception. "load failed")))]
        (is (thrown-with-msg? Exception #"load failed"
                              (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}}
                                                file))))
      (let [report (edn/read-string (slurp file))]
        (is (= 1 (:schema-version report)))
        (is (= :aborted (:run-state report)))
        (is (= {:class "java.lang.Exception"} (:runner-error report)))
        (is (not (contains? report :summary))))
      (finally (io/delete-file file)))))

(deftest outcome-accounting-separates-device-skips-and-preserves-report-events
  (let [file (java.io.File/createTempFile "raster-ci-outcomes-" ".edn")
        v #'timing-preserves-results-and-records-exceptions
        id 'raster.ci.timed-runner-test/timing-preserves-results-and-records-exceptions
        selected #{'raster.ci.timed-runner-test}
        events [{:type :begin-test-var :var v}
                {:type :pass :message "ordinary numerical check"}
                {:type :pass :message "opencl-skip-marker"}
                {:type :pass :message "gpu-skip-marker"}
                {:type :pass :message "gpu-capability-skip-marker"}
                {:type :fail :message "required runtime missing"}
                {:type :error :actual (Exception. "kernel failure")}]
        forwarded (atom [])
        summary {:test 1 :pass 4 :fail 1 :error 1}]
    (try
      (let [actual (with-redefs [clojure.test/report #(swap! forwarded conj %)
                                runner/test
                                (fn [_]
                                  (binding [clojure.test/*testing-vars* (list v)]
                                    (doseq [event events] (clojure.test/report event)))
                                  summary)]
                     (timed/run-timed! {:namespace selected} file))
            report (edn/read-string (slurp file))]
        (is (= summary actual))
        (is (= events @forwarded) "the original reporter receives every unchanged event once")
        (is (= {id {:started 1 :assertions {:pass 4 :fail 1 :error 1}
                    :diagnostics [{:type :fail}
                                  {:type :error :exception {:class "java.lang.Exception"}}]
                    :skip-markers {"opencl-skip-marker" 1 "gpu-skip-marker" 1
                                   "gpu-capability-skip-marker" 1}}}
               (:outcomes report)))
        (is (= 1 (:schema-version report)) "timing fields keep their existing schema"))
      (finally (io/delete-file file)))))

(deftest failure-classifiers-are-portable-bounded-and-do-not-inspect-runtime-inputs
  (let [file (java.io.File/createTempFile "raster-ci-diagnostics-" ".edn")
        v #'timing-preserves-results-and-records-exceptions
        id 'raster.ci.timed-runner-test/timing-preserves-results-and-records-exceptions
        opaque (Object.)
        failure (ex-info "not retained"
                         {:reason :segmap-kernel-body-declined
                          :missing-rule :checked-floating-conversion :fallback :none
                          :compiler opaque :expression opaque})
        events (repeat 12 {:type :error :actual failure :file "probe.clj" :line 17
                           :expected opaque})
        summary {:test 1 :pass 0 :fail 0 :error 12}
        forwarded (atom [])]
    (try
      (let [actual (with-redefs [clojure.test/report #(swap! forwarded conj %)
                                runner/test
                                (fn [_]
                                  (binding [clojure.test/*testing-vars* (list v)]
                                    (doseq [event events] (clojure.test/report event)))
                                  summary)]
                     (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}} file))]
        (is (= summary actual)))
      (let [report (edn/read-string (slurp file))
            outcome (get-in report [:outcomes id])]
        (is (= 12 (get-in outcome [:assertions :error])))
        (is (= 8 (count (:diagnostics outcome))))
        (is (= {:type :error :file "probe.clj" :line 17
                :exception {:class "clojure.lang.ExceptionInfo"
                            :reason :segmap-kernel-body-declined
                            :missing-rule :checked-floating-conversion :fallback :none}}
               (first (:diagnostics outcome))))
        (is (= events @forwarded))
        (is (= summary (:summary report)))
        (is (= :completed (:run-state report))))
      (finally (io/delete-file file)))))

(deftest outcome-accounting-does-not-claim-unselected-or-simulated-tests
  (let [file (java.io.File/createTempFile "raster-ci-selection-outcomes-" ".edn")
        v #'timing-preserves-results-and-records-exceptions]
    (try
      (with-redefs [clojure.test/report (fn [_])
                    runner/test
                    (fn [_]
                      (binding [clojure.test/*testing-vars* (list v)]
                        (clojure.test/report {:type :pass :message "opencl-skip-marker"}))
                      {:test 0 :pass 0 :fail 0 :error 0})]
        (timed/run-timed! {:namespace #{'some.other-test}} file))
      (is (= {} (:outcomes (edn/read-string (slurp file)))))
      (with-redefs [clojure.test/report (fn [_])
                    runner/test
                    (fn [_]
                      ;; Gate-honesty tests replace reporting to simulate device states. Those
                      ;; inner synthetic events must not become real coverage/skip evidence.
                      (with-redefs [clojure.test/report (fn [_])]
                        (binding [clojure.test/*testing-vars* (list v)]
                          (clojure.test/report {:type :pass :message "opencl-skip-marker"})))
                      {:test 0 :pass 0 :fail 0 :error 0})]
        (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}} file))
      (is (= {} (:outcomes (edn/read-string (slurp file)))))
      (finally (io/delete-file file)))))

(deftest outcome-accounting-preserves-real-test-counters
  (let [file (java.io.File/createTempFile "raster-ci-real-outcomes-" ".edn")
        v #'timing-preserves-results-and-records-exceptions
        run-var (fn []
                  (binding [clojure.test/*report-counters* (ref clojure.test/*initial-report-counters*)
                            clojure.test/*test-out* (java.io.StringWriter.)]
                    (clojure.test/test-var v)
                    @clojure.test/*report-counters*))
        baseline (run-var)]
    (try
      (let [observed (with-redefs [runner/test (fn [_] (run-var))]
                       (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}} file))
            outcomes (vals (:outcomes (edn/read-string (slurp file))))]
        (is (= baseline observed))
        (is (= 1 (count outcomes)))
        (is (= 1 (:started (first outcomes))))
        (is (= (:pass baseline) (get-in (first outcomes) [:assertions :pass])))
        (is (nil? (:skip-markers (first outcomes)))))
      (finally (io/delete-file file)))))

(deftest real-failing-assertions-keep-their-counters-and-completed-status
  (let [file (java.io.File/createTempFile "raster-ci-real-failure-" ".edn")]
    (try
      (with-local-vars [probe nil]
        (alter-meta! probe assoc :ns *ns* :name 'failing-ci-probe
                     :test (fn [] (is (= 1 2))))
        (let [run-var #(binding [clojure.test/*report-counters*
                                (ref clojure.test/*initial-report-counters*)
                                clojure.test/*test-out* (java.io.StringWriter.)]
                        (clojure.test/test-var probe)
                        @clojure.test/*report-counters*)
              baseline (run-var)
              observed (with-redefs [runner/test (fn [_] (run-var))]
                         (timed/run-timed! {:namespace #{(ns-name *ns*)}} file))
              report (edn/read-string (slurp file))
              outcome (first (vals (:outcomes report)))]
          (is (= 1 (:fail baseline)))
          (is (= baseline observed (:summary report)))
          (is (= :completed (:run-state report)))
          (is (= {:fail 1} (:assertions outcome)))
          (is (= :fail (get-in outcome [:diagnostics 0 :type])))))
      (finally (io/delete-file file)))))

(deftest outcome-observation-never-retries-or-masks-a-reporting-error
  (let [file (java.io.File/createTempFile "raster-ci-reporter-error-" ".edn")
        v #'timing-preserves-results-and-records-exceptions
        failure (Exception. "original reporter failed")
        calls (atom 0)
        reporter (fn [_] (swap! calls inc) (throw failure))]
    (try
      (let [observed
            (with-redefs [clojure.test/report reporter
                          runner/test
                          (fn [_]
                            (binding [clojure.test/*testing-vars* (list v)]
                              (clojure.test/report {:type :pass})))]
              (let [error (try
                            (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}} file)
                            nil
                            (catch Exception error error))]
                [error (identical? reporter clojure.test/report)]))]
        (is (identical? failure (first observed)))
        (is (true? (second observed)) "the prior reporter is restored even after failure")
        (is (= 1 @calls)))
      (finally (io/delete-file file)))))

(deftest timing-output-failure-does-not-mask-test-outcomes
  (let [file (java.io.File/createTempFile "raster-timings-parent-" ".edn")
        impossible-report (io/file file "timings.edn")
        options {:namespace #{'raster.ci.timed-runner-test}}]
    (try
      (with-redefs [runner/test (constantly {:test 1 :pass 1 :fail 0 :error 0})]
        (is (= {:test 1 :pass 1 :fail 0 :error 0}
               (timed/run-timed! options impossible-report))))
      (with-redefs [runner/test (fn [_] (throw (Exception. "original failure")))]
        (is (thrown-with-msg? Exception #"original failure"
                              (timed/run-timed! options impossible-report))))
      (finally (io/delete-file file)))))

(deftest cli-completes-executor-shutdown-before-returning-exit-status
  (doseq [[summary status] [[{:fail 0 :error 0} 0]
                           [{:fail 1 :error 0} 1]
                           [{:fail 0 :error 1} 1]]]
    (let [events (atom [])]
      (with-redefs [timed/run-timed! (fn [& _] (swap! events conj :tests) summary)
                    clojure.core/shutdown-agents #(swap! events conj :shutdown)]
        (is (= status (#'timed/cli-status ["-n" "raster.ci.timed-runner-test"])))
        (swap! events conj :returned))
      (is (= [:tests :shutdown :returned] @events))))
  (let [failure (Exception. "load failed")
        events (atom [])
        observed (with-redefs [timed/run-timed! (fn [& _] (throw failure))
                              clojure.core/shutdown-agents #(swap! events conj :shutdown)]
                   (try (#'timed/cli-status ["-n" "raster.ci.timed-runner-test"])
                        nil
                        (catch Exception error error)))]
    (is (identical? failure observed))
    (is (= [:shutdown] @events))))

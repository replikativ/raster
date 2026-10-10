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
      (is (= 1 (:schema-version (edn/read-string (slurp file)))))
      (finally (io/delete-file file)))))

(deftest load-errors-still-publish-the-timing-report
  (let [file (java.io.File/createTempFile "raster-timings-error-" ".edn")]
    (try
      (with-redefs [runner/test (fn [_] (throw (Exception. "load failed")))]
        (is (thrown-with-msg? Exception #"load failed"
                              (timed/run-timed! {:namespace #{'raster.ci.timed-runner-test}}
                                                file))))
      (is (= 1 (:schema-version (edn/read-string (slurp file)))))
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

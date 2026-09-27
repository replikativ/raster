(ns raster.ci.shard-plan-test
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [raster.ci.timing-weights :as weights]))

(deftest timing-import-validates-units-and-schema
  (is (= {'a 3 'b 1}
         (weights/report-costs {:schema-version 1
                               :namespaces {'a {:load-ns 1000001 :test-ns 1000000}
                                            'b {:load-ns 0}}})))
  (is (thrown? Exception (weights/report-costs {:schema-version 2})))
  (is (thrown? Exception (weights/report-costs {:schema-version 1})))
  (is (thrown? Exception (weights/report-costs
                          {:schema-version 1 :namespaces {'a {}}})))
  (is (thrown? Exception (weights/report-costs
                          {:schema-version 1 :namespaces {'a {:test-ns -1}}}))))

(deftest measured-plan-preserves-complete-unique-namespace-coverage
  (let [run (fn [timings]
              (shell/sh "bash" "scripts/ci-test-shard.sh" "--plan"
                        :env (assoc (into {} (System/getenv))
                                    "CIRCLE_NODE_TOTAL" "4" "CIRCLE_NODE_INDEX" "0"
                                    "RASTER_TEST_TIMINGS" timings)))
        measured (run "test/resources/ci_test_timings.tsv")
        fallback (run "")
        parse (fn [r] (mapv #(str/split % #"\t") (str/split-lines (:out r))))
        measured-rows (parse measured)
        fallback-rows (parse fallback)]
    (is (zero? (:exit measured)) (:err measured))
    (is (zero? (:exit fallback)) (:err fallback))
    (is (= (:out measured) (:out (run "test/resources/ci_test_timings.tsv"))))
    (is (= (set (vals (weights/test-paths))) (set (map last measured-rows))))
    (is (= (count measured-rows) (count (set (map #(nth % 2) measured-rows)))))
    (is (= (set (map last fallback-rows)) (set (map last measured-rows))))
    (is (= #{"0" "1" "2" "3"} (set (map first measured-rows))))
    (is (every? #(pos? (Long/parseLong (second %))) measured-rows))
    (is (not (zero? (:exit (run "/not-a-ci-timing-file")))))))

(deftest bounded-ci-shards-preserve-the-complete-plan
  (let [result (shell/sh "bash" "scripts/ci-test-shard.sh" "--plan"
                         :env (assoc (into {} (System/getenv))
                                     "RASTER_TEST_SHARDS" "16"
                                     "RASTER_TEST_SHARD" "0"))
        rows (mapv #(str/split % #"\t") (str/split-lines (:out result)))]
    (is (zero? (:exit result)) (:err result))
    (is (= (set (vals (weights/test-paths))) (set (map last rows))))
    (is (= (count rows) (count (set (map #(nth % 2) rows)))))
    (is (= (set (map str (range 16))) (set (map first rows))))))

(deftest opencl-shards-preserve-every-gated-namespace-once
  (let [run (fn [mode shard]
              (shell/sh "bash" "scripts/ci-test-shard.sh" mode
                        :env (assoc (into {} (System/getenv))
                                    "RASTER_TEST_SELECTION" "opencl"
                                    "CIRCLE_NODE_TOTAL" "4"
                                    "CIRCLE_NODE_INDEX" (str shard))))
        plan (run "--plan" 0)
        rows (mapv #(str/split % #"\t") (str/split-lines (:out plan)))
        expected (into #{}
                       (for [path (vals (weights/test-paths))
                             :when (re-find #"opencl-(fp16-|fp64-|gpu-|subgroups-)?available\?"
                                            (slurp path))]
                         path))
        selected (mapv (fn [shard]
                         (let [result (run "--list" shard)]
                           (is (zero? (:exit result)) (:err result))
                           (str/split-lines (:out result))))
                       (range 4))]
    (is (zero? (:exit plan)) (:err plan))
    (is (= expected (set (map last rows))))
    (is (= (count expected) (count rows)))
    (is (= (count expected) (count (distinct (mapcat identity selected)))))
    (is (= (set (map #(nth % 2) rows)) (set (mapcat identity selected))))
    (is (= #{"0" "1" "2" "3"} (set (map first rows))))
    (is (not (zero? (:exit (shell/sh
                            "bash" "scripts/ci-test-shard.sh" "--plan"
                            :env (assoc (into {} (System/getenv))
                                        "RASTER_TEST_SELECTION" "unknown"))))))))

(deftest malformed-timing-data-fails-before-test-selection
  (let [file (java.io.File/createTempFile "raster-invalid-weights-" ".tsv")]
    (try
      (spit file "test/a_test.clj\t-1\n")
      (let [result (shell/sh "bash" "scripts/ci-test-shard.sh" "--plan"
                             :env (assoc (into {} (System/getenv))
                                         "RASTER_TEST_TIMINGS" (str file)))]
        (is (not (zero? (:exit result))))
        (is (str/includes? (:err result) "invalid or duplicate")))
      (finally (io/delete-file file)))))

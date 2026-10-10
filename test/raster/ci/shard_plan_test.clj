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
                             :when (re-find #"opencl-(fp16-|fp64-|gpu-|subgroups-)?available\?|:raster.test/opencl-gate\s+true"
                                            (slurp path))]
                         path))
        selected (mapv (fn [shard]
                         (let [result (run "--list" shard)]
                           (is (zero? (:exit result)) (:err result))
                           (str/split-lines (:out result))))
                       (range 4))]
    (is (zero? (:exit plan)) (:err plan))
    (is (= expected (set (map last rows))))
    (is (contains? expected "test/raster/compiler/ir/distributed_training_test.clj")
        "hardware-free training companions retain OpenCL job coverage after splitting")
    (is (contains? expected "test/raster/gpu/distributed_training_device_test.clj"))
    (is (= (count expected) (count rows)))
    (is (= (count expected) (count (distinct (mapcat identity selected)))))
    (is (= (set (map #(nth % 2) rows)) (set (mapcat identity selected))))
    (is (= #{"0" "1" "2" "3"} (set (map first rows))))
    (is (not (zero? (:exit (shell/sh
                            "bash" "scripts/ci-test-shard.sh" "--plan"
                            :env (assoc (into {} (System/getenv))
                                        "RASTER_TEST_SELECTION" "unknown"))))))))

(defn- declared-tests [path]
  (binding [*read-eval* false]
    (with-open [reader (java.io.PushbackReader. (io/reader path))]
      (loop [names []]
        (let [form (read {:eof ::end} reader)]
          (cond
            (= ::end form) names
            (and (seq? form) (= 'deftest (first form)))
            (recur (conj names (second form)))
            :else (recur names)))))))

(deftest training-partition-preserves-all-eight-cases-once
  (let [compiler (declared-tests "test/raster/compiler/ir/distributed_training_test.clj")
        device (declared-tests "test/raster/gpu/distributed_training_device_test.clj")
        all (concat compiler device)]
    (is (= 4 (count compiler)))
    (is (= 4 (count device)))
    (is (= 8 (count (distinct all))))
    (is (= '#{repeated-local-ad-matches-rounded-independent-oracle
              finite-training-composition-retains-parameters-and-orders-updates
              unequal-batch-local-ad-producers-match-independent-analytic-gradients
              differentiated-training-assembly-certifies-without-device-resources
              finite-multi-step-ad-all-reduce-sgd-on-local-devices
              scalar-loss-as-an-explicit-rank-zero-distributed-output
              actual-ad-all-reduce-sgd-on-colocated-opencl-workers
              actual-ad-all-reduce-sgd-on-colocated-level-zero-workers}
           (set all)))
    (is (empty? (declared-tests "test/raster/compiler/fixtures/distributed_training.clj"))
        "the common fixture cannot cause duplicate test execution")))

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

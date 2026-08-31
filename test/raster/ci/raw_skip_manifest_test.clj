(ns raster.ci.raw-skip-manifest-test
  "Debt ratchet for test paths that print a skip without using an honest shared gate."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]))

(def ^:private manifest-path "test/raster/ci/raw_skip_manifest.edn")

;; Keep the searched token split so this test does not inventory its own regex literal.
(def ^:private raw-skip-pattern
  (re-pattern (str "(?i)\\(" "println[^\\r\\n]*(?:" "skip|skipping)")))

(def ^:private allowed-classes
  #{:device-test :optional-library :shared-gate :spike :toolchain})

(defn- manifest []
  (edn/read-string (slurp manifest-path)))

(defn- clojure-test-files []
  (->> (file-seq (io/file "test"))
       (filter #(.isFile ^java.io.File %))
       (filter #(re-find #"\.clj[cs]?$" (.getName ^java.io.File %)))))

(defn- raw-skip-sites []
  (into (sorted-map)
        (keep (fn [^java.io.File file]
                (let [n (count (re-seq raw-skip-pattern (slurp file)))]
                  (when (pos? n)
                    [(.getPath file) n]))))
        (clojure-test-files)))

(deftest raw-skip-debt-is-exact-and-explained
  (let [{:keys [schema-version sites]} (manifest)
        expected (into (sorted-map)
                       (map (fn [[path entry]] [path (:count entry)]))
                       sites)
        actual (raw-skip-sites)]
    (testing "the manifest is a reviewable compatibility-debt record"
      (is (= 1 schema-version))
      (is (seq sites))
      (doseq [[path {:keys [count class reason]}] sites]
        (is (.isFile (io/file path)) (str "manifest path must exist: " path))
        (is (pos-int? count) (str "manifest count must be positive: " path))
        (is (contains? allowed-classes class) (str "unknown skip class: " path))
        (is (and (string? reason) (not-empty reason))
            (str "skip debt needs a reason: " path))))
    (testing "raw skip sites cannot grow, move, or disappear without updating the debt record"
      (is (= expected actual)
          (str "raw skip inventory changed; migrate to an honest shared gate when possible, "
               "otherwise update " manifest-path " explicitly\n"
               "expected: " (pr-str expected) "\nactual:   " (pr-str actual))))))

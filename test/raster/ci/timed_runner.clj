(ns raster.ci.timed-runner
  "CI-only timing instrumentation for explicit namespace selections around the existing
   Cognitect runner; fixtures, test reporting and failure semantics remain owned by that runner
   and clojure.test. Use the ordinary runner for help, regex or default discovery."
  (:require [clojure.java.io :as io]
            [clojure.test :as test]
            [clojure.tools.cli :as cli]
            [cognitect.test-runner :as runner]))

(def ^:dynamic *loading?* false)

(defn measured
  [timings path f]
  (let [start (System/nanoTime)]
    (try (f)
         (finally
           (swap! timings update-in path (fnil + 0)
                  (- (System/nanoTime) start))))))

(defn run-timed!
  [options report-file]
  (let [started (System/nanoTime)
        timings (atom {:schema-version 1 :namespaces {} :vars {}
                       :load-order [] :test-order []})
        selected (:namespace options)
        require* clojure.core/require
        test-ns* test/test-ns
        test-var* test/test-var]
    (when-not (seq selected)
      (throw (ex-info "timed CI runner requires explicit namespace selection" {})))
    (try
      (with-redefs [clojure.core/require
                    (fn [& args]
                      (if (and (not *loading?*) (= 1 (count args))
                               (contains? selected (first args)))
                        (binding [*loading?* true]
                          (swap! timings update :load-order conj (first args))
                          (measured timings [:namespaces (first args) :load-ns]
                                    #(apply require* args)))
                        (apply require* args)))
                    test/test-ns
                    (fn [n]
                      (swap! timings update :test-order conj (ns-name (the-ns n)))
                      (measured timings [:namespaces (ns-name (the-ns n)) :test-ns]
                                #(test-ns* n)))
                    test/test-var
                    (fn [v]
                      (let [{:keys [ns name]} (meta v)]
                        (if (and (:test (meta v)) (contains? selected (ns-name ns)))
                          (measured timings [:vars (symbol (str (ns-name ns)) (str name))]
                                    #(test-var* v))
                          (test-var* v))))]
        (runner/test options))
      (finally
        (try
          (io/make-parents report-file)
          (spit report-file
                (pr-str (assoc @timings :elapsed-ns (- (System/nanoTime) started)
                               :java-version (System/getProperty "java.version"))))
          (println "CI namespace/var timings:" (str report-file))
          (catch Exception error
            ;; Diagnostics must not mask the compiler/test exception we are diagnosing.
            (binding [*out* *err*]
              (println "Could not write CI timing diagnostics:" (.getMessage error)))))))))

(defn -main [& args]
  (let [{:keys [options errors]} (cli/parse-opts args runner/cli-options)
        report (or (System/getenv "RASTER_TEST_TIMING_REPORT")
                   "test-results/timings.edn")]
    (when (or (seq errors) (:test-help options))
      (throw (ex-info "invalid timed CI runner arguments" {:errors errors})))
    (try
      (let [{:keys [fail error]} (run-timed! options report)]
        (System/exit (if (zero? (+ fail error)) 0 1)))
      (finally (shutdown-agents)))))

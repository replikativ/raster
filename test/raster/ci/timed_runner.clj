(ns raster.ci.timed-runner
  "CI-only timing instrumentation for explicit namespace selections around the existing
   Cognitect runner; fixtures, test reporting and failure semantics remain owned by that runner
   and clojure.test. Use the ordinary runner for help, regex or default discovery."
  (:require [clojure.java.io :as io]
            [clojure.test :as test]
            [clojure.tools.cli :as cli]
            [cognitect.test-runner :as runner])
  (:import [java.nio.file Files CopyOption StandardCopyOption AtomicMoveNotSupportedException]
           [java.nio.file.attribute FileAttribute]))

(def ^:dynamic *loading?* false)

(def ^:private skip-marker-messages
  ;; Existing device gates emit these passing assertions. Keep them distinct from ordinary
  ;; assertions: a passing marker proves an honest skip, not numerical device execution.
  #{"opencl-skip-marker" "gpu-skip-marker" "gpu-capability-skip-marker"})

(defn- exception-diagnostic
  "Keep bounded portable classifier data, never compiler objects or printed inputs."
  [error]
  (when (instance? Throwable error)
    (let [data (ex-data error)
          source (:source data)
          ;; A definition's immutable name is diagnostic, not its root, metadata
          ;; or a source form. In particular do not print arbitrary :source values.
          source-name (cond
                        (var? source)
                        (let [source-ns (.ns ^clojure.lang.Var source)
                              source-symbol (.sym ^clojure.lang.Var source)]
                          (when (and (instance? clojure.lang.Namespace source-ns)
                                     (symbol? source-symbol))
                            (symbol (str (ns-name source-ns)) (str source-symbol))))
                        (qualified-symbol? source) (with-meta source nil))]
      (into (cond-> {:class (.getName (class error))}
              (and source-name (<= (count (str source-name)) 512))
              (assoc :source source-name))
            (keep (fn [key]
                    (let [value (get data key)]
                      (when (keyword? value) [key value]))))
            [:reason :missing-rule :rule :fallback :compiler :front-end]))))

(defn- failure-diagnostic
  [event]
  (cond-> (select-keys event [:type])
    (string? (:file event)) (assoc :file (:file event))
    (integer? (:line event)) (assoc :line (:line event))
    (instance? Throwable (:actual event))
    (assoc :exception (exception-diagnostic (:actual event)))))

(defn- record-outcome!
  [timings selected event]
  (let [v (or (:var event) (first test/*testing-vars*))
        {:keys [ns name]} (meta v)
        namespace (when ns (ns-name ns))
        id (when name (symbol (str namespace) (str name)))
        type (:type event)]
    (when (and id (contains? selected namespace))
      (cond
        (= :begin-test-var type)
        (swap! timings update-in [:outcomes id :started] (fnil inc 0))

        (contains? #{:pass :fail :error} type)
        (swap! timings
               (fn [report]
                 (cond-> (update-in report [:outcomes id :assertions type] (fnil inc 0))
                   ;; Assertion counts remain complete. Retain only eight small examples per
                   ;; test so a failing randomized loop cannot inflate the diagnostic artifact.
                   (and (contains? #{:fail :error} type)
                        (< (count (get-in report [:outcomes id :diagnostics])) 8))
                   (update-in [:outcomes id :diagnostics] (fnil conj [])
                              (failure-diagnostic event))
                   (and (= :pass type) (contains? skip-marker-messages (:message event)))
                   (update-in [:outcomes id :skip-markers (:message event)] (fnil inc 0)))))))))

(defn measured
  [timings path f]
  (let [start (System/nanoTime)]
    (try (f)
         (finally
           (swap! timings update-in path (fnil + 0)
                  (- (System/nanoTime) start))))))

(defn- measured-var
  [timings id compiled? f]
  (let [progress! (fn [phase]
                    (when compiled?
                      (try
                        ;; One entry/exit marker, never a heartbeat or passing assertion. A
                        ;; stalled compile still hits the existing no-output timeout, now with
                        ;; its test Var identified. Print no compiler forms or runtime inputs.
                        (println "CI compiled test" phase id)
                        (flush)
                        (catch Exception _ nil))))]
    (progress! :begin)
    (try (measured timings [:vars id] f)
         (finally (progress! :end)))))

(defn- publish-report!
  "Replace a diagnostic only after a complete sibling file has been written."
  [file report]
  (let [destination (.toPath (.getAbsoluteFile (io/file file)))]
    (io/make-parents file)
    (let [temporary (Files/createTempFile (.getParent destination) ".raster-ci-timing-" ".tmp"
                                          (make-array FileAttribute 0))]
      (try
        (spit (.toFile temporary) (pr-str report))
        (try
          (Files/move temporary destination
                      (into-array CopyOption [StandardCopyOption/ATOMIC_MOVE
                                              StandardCopyOption/REPLACE_EXISTING]))
          (catch AtomicMoveNotSupportedException _
            (Files/move temporary destination
                        (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))))
        (finally (Files/deleteIfExists temporary))))))

(defn run-timed!
  [options report-file]
  (let [started (System/nanoTime)
        timings (atom {:schema-version 1 :run-state :running
                       :namespaces {} :vars {} :outcomes {}
                       :load-order [] :test-order []})
        selected (:namespace options)
        require* clojure.core/require
        test-ns* test/test-ns
        test-var* test/test-var
        report* test/report
        run-id (str (java.util.UUID/randomUUID))
        publication-lock (Object.)
        warned? (atom false)
        checkpoint!
        (fn [event full?]
          ;; Capture under the same run-local lock as publication: a slow old
          ;; namespace checkpoint must not replace the final completed report.
          (locking publication-lock
            (let [snapshot (assoc @timings :run-id run-id
                                  :elapsed-ns (- (System/nanoTime) started)
                                  :java-version (System/getProperty "java.version"))]
              (try
                (when full? (publish-report! report-file snapshot))
                ;; This records the latest observed event, not a unique active
                ;; Var: fixtures can nest calls or run tests concurrently.
                ;; It is constant-sized and never serializes the outcome table.
                (publish-report! (str report-file ".progress")
                                 (assoc (select-keys snapshot [:run-id :run-state :elapsed-ns])
                                        :schema-version 1 :latest-event event))
                (catch Exception error
                  ;; Repeated diagnostic output would itself keep a stuck CI
                  ;; action alive. Warn once, without masking test exceptions.
                  (when (compare-and-set! warned? false true)
                    (try
                      (binding [*out* *err*]
                        (println "Could not write CI timing diagnostics:" (.getMessage error)))
                      (catch Exception _ nil))))))))
        observe (fn [event full-after? f]
                  (checkpoint! (assoc event :phase :begin) false)
                  (try (f)
                       (finally (checkpoint! (assoc event :phase :end) full-after?))))]
    (when-not (seq selected)
      (throw (ex-info "timed CI runner requires explicit namespace selection" {})))
    (try
      (checkpoint! {:scope :runner :phase :begin} true)
      (with-redefs [test/report
                    (fn [event]
                      (record-outcome! timings selected event)
                      (report* event))
                    clojure.core/require
                    (fn [& args]
                      (if (and (not *loading?*) (= 1 (count args))
                               (contains? selected (first args)))
                        (binding [*loading?* true]
                          (swap! timings update :load-order conj (first args))
                          (observe {:scope :load-ns :id (first args)} true
                                   #(measured timings [:namespaces (first args) :load-ns]
                                              (fn [] (apply require* args)))))
                        (apply require* args)))
                    test/test-ns
                    (fn [n]
                      (swap! timings update :test-order conj (ns-name (the-ns n)))
                      (observe {:scope :test-ns :id (ns-name (the-ns n))} true
                               #(measured timings [:namespaces (ns-name (the-ns n)) :test-ns]
                                          (fn [] (test-ns* n)))))
                    test/test-var
                    (fn [v]
                      (let [{:keys [ns name]} (meta v)]
                        (if (and (:test (meta v)) (contains? selected (ns-name ns)))
                          (let [id (symbol (str (ns-name ns)) (str name))]
                            (observe {:scope :test-var :id id} false
                                     #(measured-var timings id (boolean (:compiled (meta v)))
                                                    (fn [] (test-var* v)))))
                          (test-var* v))))]
        (let [summary (runner/test options)]
          ;; A complete run can fail assertions. Do not equate completion with passing,
          ;; or a partial report written during unwinding with complete coverage.
          (swap! timings assoc :run-state :completed
                 :summary (select-keys summary [:test :pass :fail :error]))
          summary))
      (catch Throwable error
        (swap! timings assoc :run-state :aborted
               :runner-error (exception-diagnostic error))
        (throw error))
      (finally
        (checkpoint! {:scope :runner :phase :end} true)
        (try (println "CI namespace/var timings:" (str report-file))
             (catch Exception _ nil))))))

(defn- cli-status [args]
  (let [{:keys [options errors]} (cli/parse-opts args runner/cli-options)
        report (or (System/getenv "RASTER_TEST_TIMING_REPORT")
                   "test-results/timings.edn")]
    (when (or (seq errors) (:test-help options))
      (throw (ex-info "invalid timed CI runner arguments" {:errors errors})))
    (try
      (let [{:keys [fail error]} (run-timed! options report)]
        (if (zero? (+ fail error)) 0 1))
      (finally (shutdown-agents)))))

(defn -main [& args]
  ;; System/exit does not unwind finally blocks. Complete executor shutdown
  ;; before requesting process termination; never turn a native exit into success.
  (System/exit (cli-status args)))

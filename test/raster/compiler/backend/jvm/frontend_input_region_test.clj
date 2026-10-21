(ns raster.compiler.backend.jvm.frontend-input-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-prefix-test :as fixture]
            [raster.numeric :as numeric]))

(def unrelated-input 7)

(defn- captured []
  (fixture/unused-checked-helper-input 2 3)
  (let [frontend (:frontend (#'fixture/capture #'fixture/unused-checked-helper-input))]
    {:frontend frontend :call (second (second (first (:body frontend))))}))

(defn- prepare [frontend call]
  (entry/call-with-definition-inputs
   #(entry/prepare-frontend-invocation-input! frontend call 0)))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- replace-raw-body! [source metadata]
  (let [body (:raster.core/deftm-source-body metadata)]
    (alter-meta! source assoc :raster.core/deftm-source-body
                 (with-meta (mapv identity body) (meta body)))))

(deftest original-unused-helper-argument-outlines-with-its-completed-long-input
  (let [{:keys [frontend call]} (captured)
        region (prepare frontend call)
        completed (:input-completion region)]
    (is (nil? entry/*emission-inputs*))
    (is (= '[rows chunks] (:parameters region)))
    (is (identical? call (:source-input region)))
    (is (= 0 (:source-input-index region)))
    (is (= Long/TYPE (:parameter-carrier completed)))
    (is (= :ref (:emitted-type completed)))
    (is (= :coerce (:adaptation completed)))
    (is (= Long/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (doseq [[rows chunks] [[2 3] [-3 7] [0 Long/MAX_VALUE] [Long/MAX_VALUE 1]]]
      (is (= (numeric/* rows chunks) ((:invoke region) rows chunks))))
    (is (thrown? ArithmeticException ((:invoke region) Long/MAX_VALUE 2)))
    (is (thrown? ArithmeticException (fixture/unused-checked-helper-input Long/MAX_VALUE 2)))
    (is (= 6 ((:invoke region) 2 3)))
    (is (= 7 (alength ^doubles (fixture/unused-checked-helper-input 2 3))))
    (let [forbidden (fn [& _] (throw (AssertionError. "invocation refreshed compiler evidence")))]
      (with-redefs [entry/observe-definition-input! forbidden
                    entry/observe-frontend-source-identity! forbidden
                    bytecode/compile-input-invocation-region forbidden]
        (is (= 6 ((:invoke region) 2 3)))))))

(deftest input-preparation-refuses-foreign-or-copied-or-ownerless-inputs
  (let [{:keys [frontend call]} (captured)
        owner (atom {:static [] :values []}) before @owner]
    (is (nil? (entry/prepare-frontend-invocation-input! frontend call 0)))
    (binding [entry/*emission-inputs* owner]
      (doseq [[frontend call]
              [[frontend (with-meta (apply list call) (meta call))]
               [(assoc frontend :emission (into {} (:emission frontend))) call]
               [(assoc-in frontend [:emission :implementation] (Object.)) call]]]
        (is (nil? (entry/prepare-frontend-invocation-input! frontend call 0)))
        (is (identical? before @owner)))
      (is (= :unowned-bytecode-input-region
             (reason #(entry/prepare-frontend-invocation-input! frontend call 1))))
      (is (identical? before @owner)))))

(deftest prepared-input-retains-raw-source-identity-after-the-query
  (let [{:keys [frontend call]} (captured)
        source (get-in frontend [:source-entry :source-var]) metadata (meta source)
        compile-input bytecode/compile-input-invocation-region
        emitted-guard (atom nil)
        region (with-redefs [bytecode/compile-input-invocation-region
                             (fn [& arguments]
                               (let [region (apply compile-input arguments)]
                                 (reset! emitted-guard (:guard region))
                                 region))]
                 (prepare frontend call))]
    (try
      (replace-raw-body! source metadata)
      (is (= metadata (meta source)) "ordinary metadata equality cannot detect this source replacement")
      (is (nil? (reason #(@emitted-guard)))
          "the old module guard alone does not own raw frontend identity")
      (is (= :stale-definition-proof-input (reason #((:guard region)))))
      (is (= :stale-definition-proof-input (reason #((:invoke region) 2 3))))
      (finally (reset-meta! source metadata)))
    (is (= 6 ((:invoke region) 2 3)))))

(deftest prepared-input-checks-original-source-again-before-returning
  (let [{:keys [frontend call]} (captured)
        source (get-in frontend [:source-entry :source-var]) metadata (meta source)
        compile-input bytecode/compile-input-invocation-region
        region (with-redefs [bytecode/compile-input-invocation-region
                             (fn [& arguments]
                               (let [region (apply compile-input arguments) invoke (:invoke region)]
                                 (assoc region :invoke
                                        (fn [& arguments]
                                          (let [value (apply invoke arguments)]
                                            (replace-raw-body! source metadata)
                                            value)))))]
                 (prepare frontend call))]
    (try
      (is (= :stale-definition-proof-input (reason #((:invoke region) 2 3))))
      (finally (reset-meta! source metadata)))))

(deftest prepared-input-freezes-transitive-but-not-unrelated-parent-history
  (let [{:keys [frontend call]} (captured)
        owner (atom {:static [] :values []})
        region (binding [entry/*emission-inputs* owner]
                 (entry/observe-definition-input! #'unrelated-input
                                                  (the-ns 'raster.compiler.backend.jvm.frontend-input-region-test)
                                                  'unrelated-input)
                 (entry/prepare-frontend-invocation-input! frontend call 0))
        metadata (meta #'unrelated-input)]
    (try
      (alter-meta! #'unrelated-input assoc ::changed true)
      (is (= :stale-definition-proof-input
             (reason #(entry/assert-definition-inputs-current! @owner))))
      (is (= 6 ((:invoke region) 2 3)) "this region did not consume the unrelated enclosing input")
      (finally (reset-meta! #'unrelated-input metadata)))
    (let [dependency (some #(when (= #'numeric/_star__m_long_long (:var %)) (:var %))
                           (get-in frontend [:emission :dependency-snapshots]))
          metadata (meta dependency)]
      (is (var? dependency))
      (try
        (alter-meta! dependency assoc ::changed true)
        (is (contains? #{:stale-definition-proof-input :stale-emission-inputs}
                       (reason #((:invoke region) 2 3))))
        (finally (reset-meta! dependency metadata))))
    (is (= 6 ((:invoke region) 2 3)))))

(ns raster.compiler.core.definition-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]))

(def numeric-calls (atom 0))
(def input-a (fn [] (swap! numeric-calls inc)))
(def input-b (fn [] (swap! numeric-calls inc)))

(defn- observe! [v]
  (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v))))

(deftest refused-optional-analysis-does-not-retain-speculative-dependencies
  (let [owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (is (nil? (entry/call-with-optional-definition-inputs #(do (observe! #'input-a) nil))))
      (is (empty? (:definition-proofs @owner)))
      (is (= :proof (entry/call-with-optional-definition-inputs
                     #(do (observe! #'input-a) :proof))))
      (is (contains? (:definition-proofs @owner) #'input-a))
      (let [before @owner]
        (with-redefs [input-a (fn [] ::changed)]
          (is (= :unstable-definition-proof-input
                 (try (entry/call-with-optional-definition-inputs
                       #(do (observe! #'input-a) :new-proof))
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
        (is (= before @owner) "a new query cannot refresh historical observations")))))

(deftest optional-proof-commit-does-not-overwrite-a-competing-owner-update
  (let [owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (is (= :unstable-definition-proof-input
             (try (entry/call-with-optional-definition-inputs
                   #(do (observe! #'input-a) (swap! owner assoc :another-input true) :proof))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (:another-input @owner))
      (is (empty? (:definition-proofs @owner))))))

(deftest a-pending-batch-is-prepared-without-invocation-and-the-whole-owner-retries
  (let [vars [#'input-a #'input-b]
        roots (mapv deref vars) metadata (mapv meta vars)
        attempts (atom 0) prepared (atom []) before @numeric-calls]
    (try
      (doseq [v vars] (alter-meta! v assoc :raster.core/jit-spec {}))
      (let [result
            (entry/call-with-definition-inputs
             (fn []
               (swap! attempts inc)
               (doseq [v vars] (observe! v))
               (entry/prepare-pending-definition-inputs!
                (fn [v]
                  (is (nil? entry/*emission-inputs*) "dependency compilation has its own owner")
                  (swap! prepared conj v)
                  (alter-var-root v (constantly (fn [] (swap! numeric-calls inc))))
                  (alter-meta! v dissoc :raster.core/jit-spec)))
               @entry/*emission-inputs*))]
        (is (= 2 @attempts))
        (is (= (set vars) (set @prepared)))
        (is (= 2 (count @prepared)))
        (is (= before @numeric-calls))
        (is (entry/assert-definition-inputs-current! result))
        (with-redefs [input-a (fn [] ::different)]
          (is (= :stale-definition-proof-input
                 (try (entry/assert-definition-inputs-current! result)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
      (finally (doseq [[v root m] (map vector vars roots metadata)]
                 (alter-var-root v (constantly root)) (reset-meta! v m))))))

(deftest preparation-failure-is-not-silenced-or-restamped
  (let [original (meta #'input-a) attempts (atom 0)]
    (try
      (alter-meta! #'input-a assoc :raster.core/jit-spec {})
      (is (= :unsupported-test-jit
             (try (entry/call-with-definition-inputs
                   #(do (swap! attempts inc) (observe! #'input-a)
                        (entry/prepare-pending-definition-inputs!
                         (fn [_] (throw (ex-info "unsupported dependency" {:reason :unsupported-test-jit}))))))
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= 1 @attempts))
      (finally (reset-meta! #'input-a original)))))

(deftest a-refused-unsupported-probe-is-never-prepared
  (let [original (meta #'input-a)]
    (try
      (alter-meta! #'input-a assoc :raster.core/jit-spec {})
      (is (= :ready
             (entry/call-with-definition-inputs
              #(do (entry/call-with-optional-definition-inputs
                    (fn [] (observe! #'input-a) nil))
                   (entry/prepare-pending-definition-inputs!
                    (fn [_] (throw (AssertionError. "refused helper was prepared"))))
                   :ready))))
      (finally (reset-meta! #'input-a original)))))

(deftest a-nested-owner-propagates-promotion-instead-of-retrying-locally
  (let [original (meta #'input-a) root @#'input-a
        owner (atom {:static [] :values []}) attempts (atom 0)]
    (try
      (alter-meta! #'input-a assoc :raster.core/jit-spec {})
      (binding [entry/*emission-inputs* owner]
        (is (= :stale-definition-proof-input
               (try (entry/call-with-definition-inputs
                     #(do (swap! attempts inc) (observe! #'input-a)
                          (entry/prepare-pending-definition-inputs!
                           (fn [v]
                             (alter-var-root v (constantly (fn [] ::ready)))
                             (alter-meta! v dissoc :raster.core/jit-spec)))))
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (= 1 @attempts))
      (finally (alter-var-root #'input-a (constantly root))
               (reset-meta! #'input-a original)))))

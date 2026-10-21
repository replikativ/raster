(ns raster.compiler.core.definition-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]))

(def numeric-calls (atom 0))
(def input-a (fn [] (swap! numeric-calls inc)))
(def input-b (fn [] (swap! numeric-calls inc)))
(def input-c (fn [] (swap! numeric-calls inc)))

(defn- observe! [v]
  (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v))))

(defn- observed-inputs [v]
  (binding [entry/*emission-inputs* (atom {:static [] :values []})]
    (observe! v)
    @entry/*emission-inputs*))

(deftest historical-adoption-retains-first-observations-and-all-lookup-edges
  (let [historical (-> (observed-inputs #'input-a)
                       (assoc :source-body-inputs #{[#'input-a :var]}
                              :unclosed-source-imports #{[:test-unclosed #'input-a]}))
        sentinel (Object.)
        parent (atom (-> (observed-inputs #'input-b)
                         (update :values conj sentinel)
                         (assoc :unclosed-source-imports #{[:parent-unclosed]})))
        own-ns (:ns (meta #'input-a))
        qualified (symbol (str (ns-name own-ns)) "input-a")]
    (binding [entry/*emission-inputs* parent]
      (entry/observe-definition-input! #'input-a (the-ns 'clojure.core) qualified)
      (let [prior (get-in @parent [:definition-proofs #'input-a])]
        (with-redefs [entry/emission-definition-snapshot
                      (fn [_] (throw (AssertionError. "historical adoption reobserved a definition")))]
          (is (true? (entry/adopt-definition-inputs! historical))))
        (let [adopted (get-in @parent [:definition-proofs #'input-a])]
          (is (identical? (:value prior) (:value adopted)))
          (is (identical? (:metadata prior) (:metadata adopted)))
          (is (= #{[(the-ns 'clojure.core) qualified]
                   [(:ns (meta #'input-a)) 'input-a]} (:lookups adopted)))
          (is (= 3 (count (:values @parent))))
          (is (some #(identical? sentinel %) (:values @parent)))
          (is (some #(identical? adopted %) (:values @parent)))
          (is (= (:source-body-inputs historical) (:source-body-inputs @parent)))
          (is (= (conj (:unclosed-source-imports historical) [:parent-unclosed])
                 (:unclosed-source-imports @parent))))))))

(deftest historical-adoption-does-not-overwrite-a-competing-owner-update
  (let [historical (observed-inputs #'input-a)
        parent (atom {:static [] :values []})
        validate entry/assert-definition-inputs-current!
        calls (atom 0)]
    (binding [entry/*emission-inputs* parent]
      (with-redefs [entry/assert-definition-inputs-current!
                    (fn [inputs]
                      (let [result (validate inputs)]
                        (when (= 3 (swap! calls inc))
                          (swap! parent assoc :another-input true))
                        result))]
        (is (= :unstable-definition-proof-input
               (try (entry/adopt-definition-inputs! historical) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
    (is (:another-input @parent))
    (is (empty? (:definition-proofs @parent)))))

(deftest stale-historical-adoption-cannot-refresh-or-partially-update-the-parent
  (let [historical (observed-inputs #'input-a)
        parent (atom (observed-inputs #'input-b))
        before @parent]
    (binding [entry/*emission-inputs* parent]
      (with-redefs [input-a (fn [] ::changed)]
        (is (= :stale-definition-proof-input
               (try (entry/adopt-definition-inputs! historical) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
    (is (identical? before @parent))))

(deftest adoption-does-not-refresh-a-stale-parent-with-newer-history
  (let [parent (atom (observed-inputs #'input-a)) before @parent]
    (with-redefs [input-a (fn [] ::changed)]
      (let [newer (observed-inputs #'input-a)]
        (binding [entry/*emission-inputs* parent]
          (is (= :stale-definition-proof-input
                 (try (entry/adopt-definition-inputs! newer) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))
    (is (identical? before @parent))))

(deftest historical-adoption-abstains-without-an-owner
  (binding [entry/*emission-inputs* nil]
    (is (nil? (entry/adopt-definition-inputs! (observed-inputs #'input-a))))))

(deftest adopted-lookup-edges-remain-required-after-handoff
  (let [scope (create-ns (gensym "raster.adoption-lookup."))
        historical (atom {:static [] :values []})
        parent (atom {:static [] :values []})]
    (try
      (binding [*ns* scope] (alias 'chosen (ns-name (:ns (meta #'input-a)))))
      (binding [entry/*emission-inputs* historical]
        (entry/observe-definition-input! #'input-a scope 'chosen/input-a))
      (binding [entry/*emission-inputs* parent]
        (is (true? (entry/adopt-definition-inputs! @historical))))
      (ns-unalias scope 'chosen)
      (is (= :stale-definition-proof-input
             (try (entry/assert-definition-inputs-current! @parent) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
      (finally (remove-ns (ns-name scope))))))

(deftest optional-transactions-retain-producer-guards-only-on-success
  (let [checks (atom 0)
        producer (reify entry/RetainedProducerInput
                   (assert-producer-current! [this] (swap! checks inc) this))
        parent (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* parent]
      (is (nil? (entry/call-with-optional-definition-inputs
                 #(do (entry/adopt-definition-inputs! {:producer-inputs [producer]}) nil))))
      (is (empty? (:producer-inputs @parent)))
      (is (= :accepted (entry/call-with-optional-definition-inputs
                        #(do (entry/adopt-definition-inputs! {:producer-inputs [producer]}) :accepted))))
      (is (= [producer] (:producer-inputs @parent)))
      (is (empty? (:values @parent)))
      (reset! checks 0)
      (is (entry/assert-definition-inputs-current! @parent))
      (is (= 1 @checks)))))

(deftest arbitrary-objects-cannot-act-as-producer-guards
  (let [parent (atom {:static [] :values []}) before @parent]
    (binding [entry/*emission-inputs* parent]
      (is (= :unowned-producer-input
             (try (entry/adopt-definition-inputs! {:producer-inputs [(Object.)]}) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (is (identical? before @parent))))

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

(deftest selected-preparation-preserves-order-and-does-not-promote-unrequested-inputs
  (let [vars [#'input-a #'input-b #'input-c]
        roots (mapv deref vars) metadata (mapv meta vars)
        prepared (atom []) attempts (atom 0) before @numeric-calls]
    (try
      (doseq [v vars] (alter-meta! v assoc :raster.core/jit-spec {}))
      (let [result
            (entry/call-with-definition-inputs
             (fn []
               (swap! attempts inc)
               (doseq [v vars] (observe! v))
               (entry/prepare-pending-definition-inputs!
                (fn [v]
                  (is (nil? entry/*emission-inputs*))
                  (swap! prepared conj v)
                  (alter-var-root v (constantly (fn [] (swap! numeric-calls inc))))
                  (alter-meta! v dissoc :raster.core/jit-spec))
                [#'input-b #'input-a #'input-b])
               @entry/*emission-inputs*))]
        (is (= [#'input-b #'input-a] @prepared))
        (is (= 2 @attempts) "promotion restarts the whole owning transaction")
        (is (= before @numeric-calls))
        (is (map? (:raster.core/jit-spec (meta #'input-c))))
        (is (entry/assert-definition-inputs-current! result)))
      (finally
        (doseq [[v root metadata] (map vector vars roots metadata)]
          (alter-var-root v (constantly root))
          (reset-meta! v metadata))))))

(deftest selected-preparation-requires-original-observations-before-any-compilation
  (let [metadata (mapv meta [#'input-a #'input-b]) calls (atom [])]
    (try
      (doseq [v [#'input-a #'input-b]] (alter-meta! v assoc :raster.core/jit-spec {}))
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (observe! #'input-a)
        (doseq [[selection reason] [[[#'input-a #'input-b] :unobserved-definition-preparation-input]
                                    [(list #'input-a) :invalid-definition-preparation-selection]
                                    [[42] :invalid-definition-preparation-selection]]]
          (is (= reason
                 (try (entry/prepare-pending-definition-inputs! #(swap! calls conj %) selection)
                      nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
        (is (empty? @calls))
        (is (entry/assert-definition-inputs-current! @entry/*emission-inputs*)))
      (finally
        (doseq [[v metadata] (map vector [#'input-a #'input-b] metadata)]
          (reset-meta! v metadata))))))

(deftest an-explicit-empty-preparation-subset-does-not-mean-all-inputs
  (let [metadata (meta #'input-a)]
    (try
      (alter-meta! #'input-a assoc :raster.core/jit-spec {})
      (is (= :ready
             (entry/call-with-definition-inputs
              #(do (observe! #'input-a)
                   (entry/prepare-pending-definition-inputs!
                    (fn [_] (throw (AssertionError. "empty selection prepared a receiver"))) [])
                   :ready))))
      (is (map? (:raster.core/jit-spec (meta #'input-a))))
      (finally (reset-meta! #'input-a metadata)))))

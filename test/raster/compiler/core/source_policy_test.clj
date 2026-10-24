(ns raster.compiler.core.source-policy-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-policy :as policy]
            [raster.compiler.pipeline :as pipeline]))

(defmacro definition-policy []
  (case *unchecked-math* false 0.0 true 1.0 :warn-on-boxed 2.0))

(defn- define! [mode form]
  (binding [*ns* (the-ns 'raster.compiler.core.source-policy-test)
            *unchecked-math* mode]
    (eval form)))

(defn- receiver [source]
  (ns-resolve (:ns (meta source)) (symbol (str (:name (meta source)) "-impl"))))

(deftest policy-is-owned-not-a-default-for-missing-evidence
  (doseq [mode [false true :warn-on-boxed]]
    (let [retained {:unchecked-math mode}]
      (is (policy/valid? retained))
      (is (= mode (binding [*unchecked-math* (not mode)]
                    (policy/call-with-policy retained #(identity *unchecked-math*)))))))
  (doseq [missing [nil {} {:unchecked-math nil} {:unchecked-math false :extra true}]]
    (is (false? (policy/valid? missing)))
    (is (thrown? clojure.lang.ExceptionInfo (policy/call-with-policy missing (constantly nil))))))

(deftest named-source-policy-survives-late-walking-and-bytecode-publication
  (doseq [mode [false true :warn-on-boxed]]
    (let [generic (define! mode '(core/deftm named-policy [x :- Double] :- Double
                                  (definition-policy)))
          source (core/resolve-deftm-var generic {:dtype :double})
          impl (receiver source)
          expected {:unchecked-math mode}
          expected-value ({false 0.0 true 1.0 :warn-on-boxed 2.0} mode)]
      (is (= expected (:raster.core/source-policy (meta source))))
      (is (= expected (get-in (meta impl) [:raster.core/jit-spec :source-policy])))
      (binding [*unchecked-math* (not mode)]
        (core/ensure-walked-body! source)
        (core/ensure-bytecode-implementation! impl)
        (is (= expected-value (@impl 7.0)))
        (is (= expected (:source-policy (pipeline/deftm-frontend-facts source :double)))))
      (let [lineage (get-in (meta impl) [:raster.core/bytecode-emission :source-entry])
            original (meta source)]
        (is (= expected (:source-policy lineage)))
        (is (entry/source-entry-current? lineage))
        (try
          (alter-meta! source assoc :raster.core/source-policy {:unchecked-math (not mode)})
          (is (false? (entry/source-entry-current? lineage)))
          (finally (reset-meta! source original)))))))

(deftest policy-changes-before-first-jit-cannot-replace-the-pending-definition
  (let [generic (define! false '(core/deftm pending-policy [x :- Double] :- Double
                                 (definition-policy)))
        source (core/resolve-deftm-var generic {:dtype :double})
        impl (receiver source)
        original (meta source)
        implementation @impl
        spec (:raster.core/jit-spec (meta impl))]
    (try
      (doseq [replacement [{:unchecked-math true} nil ::absent]]
        (if (= ::absent replacement)
          (alter-meta! source dissoc :raster.core/source-policy)
          (alter-meta! source assoc :raster.core/source-policy replacement))
        (is (= :source-policy-mismatch
               (try (core/ensure-bytecode-implementation! impl) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (identical? implementation @impl))
        (is (identical? spec (:raster.core/jit-spec (meta impl)))))
      (finally (reset-meta! source original)))
    (core/ensure-bytecode-implementation! impl)
    (is (= 0.0 (@impl 1.0)))))

(deftest policy-change-during-capture-cannot-downgrade-to-unowned-emission
  (let [generic (define! false '(core/deftm capture-policy [x :- Double] :- Double x))
        source (core/resolve-deftm-var generic {:dtype :double})
        impl (receiver source)
        original (meta source)
        implementation @impl
        spec (:raster.core/jit-spec (meta impl))
        capture entry/capture-source-entry]
    (try
      (with-redefs [entry/capture-source-entry
                    (fn [source-var plan]
                      (alter-meta! source-var assoc :raster.core/source-policy {:unchecked-math true})
                      (capture source-var plan))]
        (is (= :unclosed-bytecode-source-entry
               (try (core/ensure-bytecode-implementation! impl) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (identical? implementation @impl))
      (is (identical? spec (:raster.core/jit-spec (meta impl))))
      (finally (reset-meta! source original)))))

(deftest parametric-specialization-inherits-its-template-not-the-requesting-context
  (doseq [mode [false true :warn-on-boxed]]
    (let [generic (define! mode '(core/deftm template-policy
                                  (All [T] [values :- (Array T)] :- Double
                                       (double (definition-policy)))))
          expected {:unchecked-math mode}
          expected-value ({false 0.0 true 1.0 :warn-on-boxed 2.0} mode)
          source (binding [*unchecked-math* (not mode)]
                   (core/resolve-deftm-var generic {:dtype :float}))
          impl (receiver source)]
      (is (= expected (:source-policy
                       (first (get @dispatch/parametric-registry
                                   'raster.compiler.core.source-policy-test/template-policy)))))
      (is (= expected (:raster.core/source-policy (meta source))))
      (is (= expected (get-in (meta impl) [:raster.core/bytecode-emission :source-entry :source-policy])))
      (is (= expected-value (@impl (float-array 0)))))))

(deftest plain-source-metadata-captures-policy-without-user-override
  (doseq [mode [false true :warn-on-boxed]
          declaration ['(core/defn* plain-policy
                          {:raster.core/source-policy :not-authority}
                          [^double x] (definition-policy))
                       '(core/defn* plain-policy
                          ([x] x) ([x y] y))]]
    (let [source (define! mode declaration)]
      (is (= {:unchecked-math mode} (:raster.core/source-policy (meta source)))))))

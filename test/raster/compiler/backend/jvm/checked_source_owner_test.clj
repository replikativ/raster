(ns raster.compiler.backend.jvm.checked-source-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.source-owner :as owner]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.inline :as inline]))

(core/deftm public-source [x :- Double] :- Object (float x))
(core/deftm checked-helper [x :- Double] :- Object (let [y (float x)] y))

(def value-helper
  (with-meta (fn [x] (float x))
    {:raster.core/deftm-walked-body ['(let* [y (clojure.core/float x)] y)]
     :raster.core/deftm-params '[x] :raster.core/deftm-tags '[Double]
     :raster.core/return-tag 'Object :raster.core/source-policy {:unchecked-math false}}))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- source []
  (public-source 1.0)
  (core/resolve-deftm-var #'public-source {:dtype :double}))
(defn- frontend [v] (pipeline/retained-frontend-source #'public-source v :double))
(defn- helper []
  (checked-helper 1.0)
  (core/resolve-deftm-var #'checked-helper {:dtype :double}))
(defn- import! [v]
  (binding [inline/*inline-scalar-bodies?* true]
    ((deref (ns-resolve 'raster.compiler.passes.scalar.inline 'try-resolve-walked-body)) v)))

(defn- compile-owned
  ([v] (compile-owned v true))
  ([v required?]
  ;; These opaque identities model only the compiler-producer ownership seam.
  ;; They grant no semantic validation, device completion or initialized storage.
  (let [program (Object.) emission (Object.)]
    (entry/call-with-definition-inputs
     (fn []
       (let [public (source)]
         (entry/observe-definition-input! public (:ns (meta public)) (:name (meta public)))
         (is (some? (import! v)))
         {:program program :emission emission :collector entry/*emission-inputs*
          :owner (owner/compile-ordered-source program emission public (frontend public) required?)}))))))

(deftest imported-policy-is-owned-at-the-actual-body-extraction
  (let [v (helper)
        {:keys [owner program emission collector]} (compile-owned v)]
    (is (contains? (:source-body-inputs @collector) [v :var]))
    (is (identical? owner (owner/assert-ordered-source! owner program emission)))
    (is (= :ordered-source-owner-mismatch
           (reason #(owner/assert-ordered-source! owner (Object.) emission))))
    (let [original @v]
      (try
        (alter-var-root v (constantly (fn [_] :replaced)))
        (reset! collector {})
        (is (= :stale-definition-proof-input
               (reason #(owner/assert-ordered-source! owner program emission))))
        (finally (alter-var-root v (constantly original)))))))

(deftest unknown-and-unchecked-imports-cannot-borrow-the-public-policy
  (doseq [mode [true :warn-on-boxed]]
    (let [generic (binding [*ns* (the-ns 'raster.compiler.backend.jvm.checked-source-owner-test)
                            *unchecked-math* mode]
                    (eval `(core/deftm ~(gensym "unchecked-helper")
                             [~'x :- Double] :- Object (let [~'y (float ~'x)] ~'y))))
          v (core/resolve-deftm-var generic {:dtype :double})
          oracle (binding [*unchecked-math* mode] (eval '(fn [^double x] (float x))))
          checked-caller (binding [*unchecked-math* false]
                           (eval '(fn [helper ^double x] (helper x))))]
      (is (Float/isInfinite (oracle Double/MAX_VALUE)) "independent unchecked source oracle")
      (is (Float/isInfinite (checked-caller oracle Double/MAX_VALUE))
          "independent primitive-source composition; boxed operands have different cast semantics")
      (is (= :unsupported-checked-result-import-policy (reason #(compile-owned v))))
      (is (nil? (:owner (compile-owned v false)))
          "ordinary eager compilation may lack the strict ordered capability")
      (entry/call-with-definition-inputs
       (fn []
         (entry/observe-definition-input! v (:ns (meta v)) (:name (meta v)))
         (let [original-frontend (pipeline/retained-frontend-source generic v :double)]
           (is (nil? (owner/compile-ordered-source
                      (Object.) (Object.) v original-frontend false)))
           (is (= :unsupported-checked-result-source-policy
                  (reason #(owner/compile-ordered-source
                            (Object.) (Object.) v original-frontend)))))))))
  (let [v (helper) original (meta v)]
    (try
      (alter-meta! v dissoc :raster.core/source-policy)
      (is (= :stale-definition-proof-input
             (reason #(entry/call-with-definition-inputs (fn [] (import! v)))))
          "removing the policy invalidates the original frontend before import")
      (finally (reset-meta! v original)))))

(deftest function-value-body-retains-its-own-policy-origin
  (is (= :unclosed-checked-result-source-imports (reason #(compile-owned #'value-helper)))
      "a retained policy alone cannot authenticate a generated function's body")
  (is (nil? (:owner (compile-owned #'value-helper false))))
  (let [original @#'value-helper]
    (try
      (alter-var-root #'value-helper #(with-meta % (dissoc (meta %) :raster.core/source-policy)))
      (is (= :unsupported-checked-result-import-policy (reason #(compile-owned #'value-helper))))
      (finally (alter-var-root #'value-helper (constantly original))))))

(deftest rejected-optional-import-does-not-become-an-executable-dependency
  (entry/call-with-definition-inputs
   (fn []
     (let [before @entry/*emission-inputs*]
       (is (nil? (entry/call-with-optional-definition-inputs
                  #(do (import! (helper)) nil))))
       (is (= before @entry/*emission-inputs*))))))

(deftest optional-source-capability-abstains-without-masking-stale-evidence
  (entry/call-with-definition-inputs
   (fn []
     (let [public (source)]
       (entry/observe-definition-input! public (:ns (meta public)) (:name (meta public)))
       (is (some? (import! #'value-helper)))
       (let [original-frontend (frontend public)
             before @entry/*emission-inputs*
             metadata (meta public)]
         (is (nil? (owner/compile-ordered-source
                    (Object.) (Object.) public original-frontend false)))
         (is (= before @entry/*emission-inputs*)
             "unavailable owner capability retains no speculative observations")
         (try
           (alter-meta! public assoc ::changed-source true)
           (is (= :stale-definition-proof-input
                  (reason #(owner/compile-ordered-source
                            (Object.) (Object.) public original-frontend false))))
           (finally (reset-meta! public metadata))))))))

(deftest derived-source-without-transitive-history-is-not-certified-by-its-caller
  (entry/call-with-definition-inputs
   (fn []
     (let [public (source)]
       (entry/observe-definition-input! public (:ns (meta public)) (:name (meta public)))
       (swap! entry/*emission-inputs* assoc :unclosed-source-imports #{[:prepared-reverse-ad 'loss]})
       (is (= :unclosed-checked-result-source-imports
              (reason #(owner/compile-ordered-source (Object.) (Object.) public
                                                           (frontend public)))))))))

(deftest missing-original-frontend-cannot-acquire-a-checked-source-owner
  (entry/call-with-definition-inputs
   (fn []
     (let [public (source)]
       (entry/observe-definition-input! public (:ns (meta public)) (:name (meta public)))
       (is (= :unowned-checked-result-frontend
              (reason #(owner/compile-ordered-source (Object.) (Object.) public nil))))))))

(deftest checked-owner-cannot-authenticate-replaced-bootstrap-intrinsics
  ;; Materialize original fixtures BEFORE replacing a bootstrap dependency.
  ;; Otherwise their genuinely stale emissions must continue refusing afterward.
  (let [public (source)]
    (entry/call-with-definition-inputs
     (fn []
       (let [original-frontend (frontend public)]
         (with-redefs [clojure.core/float (fn [_] :replaced)]
           (is (= :intrinsic-source-definition-mismatch
                  (reason #(owner/compile-ordered-source
                            (Object.) (Object.) public original-frontend))))))))))

(ns raster.ad.preparation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.method-entry :as entry]))

(deftm squared-loss [x :- Double] :- Double (numeric/* x x))
(deftm cold-squared-loss [x :- Double] :- Double (numeric/* x x))

(defn- result [prepared x]
  (((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared) x))

(deftest preparation-uses-the-frozen-frontend-not-replaceable-walk-caches
  (is (= 9.0 (squared-loss 3.0)))
  (let [source (core/resolve-deftm-var #'squared-loss)
        metadata (meta source)]
    (try
      ;; This is derived cache data, not a source definition or original body.
      ;; It must not replace what the published implementation actually means.
      (alter-meta! source assoc
                   :raster.core/deftm-walked-body '[(clojure.core/* x 7.0)]
                   :raster.core/deftm-walked-body-typed '[(clojure.core/* x 7.0)])
      (is (= [9.0 6.0] (result (reverse/prepare-value+grad #'squared-loss [0]) 3.0)))
      (finally (reset-meta! source metadata)))))

(deftest cold-preparation-observes-the-producer-without-running-its-numeric-body
  (let [source (core/resolve-deftm-var #'cold-squared-loss)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        implementation (var-get receiver)
        prepared (reverse/prepare-value+grad #'cold-squared-loss [0])]
    (is (identical? implementation (var-get receiver)))
    (is (some? (:raster.core/jit-spec (meta receiver))))
    (is (nil? (:raster.core/bytecode-emission (meta receiver))))
    (is (= [16.0 8.0] (result prepared 4.0)))))

(deftest enclosing-owner-detects-a-source-change-during-ad-preparation
  (let [source (core/resolve-deftm-var #'squared-loss)
        metadata (meta source)
        prepare reverse/ad-prepare]
    (try
      (binding [entry/*emission-inputs* (atom {:static [] :values []})]
        (with-redefs [reverse/ad-prepare
                      (fn [& arguments]
                        (let [body (apply prepare arguments)]
                          (alter-meta! source assoc ::changed-during-preparation (Object.))
                          body))]
          (is (= :stale-definition-proof-input
                 (try (reverse/prepare-value+grad #'squared-loss [0]) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
      (finally (reset-meta! source metadata)))))

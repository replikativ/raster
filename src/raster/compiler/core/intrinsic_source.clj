(ns raster.compiler.core.intrinsic-source
  "Historical definitions assumed by source intrinsic projections.
   This owns definition observations, not types, dispatch or conversion rules.
   Operation vocabulary and semantics remain in op-descriptor/source-arithmetic."
  (:require [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.source-arithmetic :as arithmetic]))

(defonce ^:private observations
  ;; Standard Clojure definitions at compiler bootstrap are the trust premise.
  ;; A fresh consumer snapshot cannot authenticate a later replacement/inline.
  (into {}
        (keep (fn [operation]
                (when-let [v (resolve operation)]
                  (when (var? v) [operation (entry/emission-definition-snapshot v)]))))
        (into #{'clojure.core/nil? 'clojure.core/alength}
              (filter #(= "clojure.core" (namespace %)))
              (concat (keys descriptor/cast-ops) descriptor/comparison-ops
                      (filter #(arithmetic/binary-operator? (name %))
                              (concat descriptor/addition-ops descriptor/subtraction-ops
                                      descriptor/multiplication-ops descriptor/division-ops))))))

(defn assert-original!
  "Check the actual historical Var identity, value and inline/definition metadata."
  [v operation reason]
  (let [original (get observations operation)]
    (when-not (and original (identical? v (:var original)) (entry/var-value-current? original))
      (throw (ex-info "intrinsic no longer matches its compiler bootstrap definition"
                      {:reason reason :operation operation})))))

(defn observe-original!
  "Observe one actual bootstrap intrinsic lookup in the active source owner.
   A qualified spelling or a fresh snapshot cannot authenticate a replacement."
  [source-ns operation]
  (let [v (ns-resolve source-ns operation)]
    (assert-original! v operation :intrinsic-source-definition-mismatch)
    (when-not (entry/observe-definition-input! v source-ns operation)
      (throw (ex-info "intrinsic ownership requires its original definition collector"
                      {:reason :unowned-intrinsic-source}))))
  true)

(defn observe-originals!
  "Retain original primitive source assumptions in the active producer collector.
   Conservatively guards this small existing source-intrinsic vocabulary; it does
   not infer which operations a later consumer wishes had been observed."
  [source-ns]
  (doseq [operation (keys observations)]
    (observe-original! source-ns operation))
  nil)

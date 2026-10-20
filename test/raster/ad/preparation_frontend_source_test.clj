(ns raster.ad.preparation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.core.method-entry :as entry]))

(deftm squared-loss [x :- Double] :- Double (numeric/* x x))
(deftm cold-squared-loss [x :- Double] :- Double (numeric/* x x))
(deftm checked-policy-loss [x :- Double y :- Double] :- Double
  (let [rounded (float y)] (numeric/+ (numeric/* x x) (double rounded))))
(deftm two-squared-loss [x :- Double y :- Double] :- Double
  (numeric/+ (numeric/* x x) (numeric/* y y)))

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

(deftest generated-program-fields-remain-bound-to-their-original-producer
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
    (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))
    (doseq [changed [(update prepared :walked-body #(vec (map identity %)))
                     (update prepared :params #(vec (map identity %)))
                     (update prepared :tags #(vec (map identity %)))
                     (update prepared :param-adj-syms #(vec (map identity %)))
                     (update prepared :result-sym #(with-meta % {:raster.type/tag 'float}))
                     (assoc prepared :source-ns (the-ns 'clojure.core))
                     (assoc prepared ::unexpected true)]]
      (is (= :ad-prepared-program-mismatch
             (try (reverse/assert-prepared-program-source! changed) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (is (= :ad-prepared-program-owner
           (try (reverse/assert-prepared-program-source!
                 (assoc prepared :raster.ad.reverse/generated-program-owner (Object.))) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest generated-program-retains-the-original-template-epoch
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        revision (templates/registry-revision)]
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/assert-prepared-program-source! prepared) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
  (let [revision (atom (templates/registry-revision))
        prepare reverse/ad-prepare]
    (binding [entry/*emission-inputs* (atom {:static [] :values []})]
      (with-redefs [templates/registry-revision (fn [] @revision)
                    reverse/ad-prepare (fn [& arguments]
                                         (let [body (apply prepare arguments)]
                                           (swap! revision inc)
                                           body))]
        (is (= :stale-definition-proof-input
               (try (reverse/prepare-value+grad #'squared-loss [0]) nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))))

(deftest generation-and-compilation-use-the-definition-policy-not-the-caller-policy
  (let [seen (atom [])
        prepare reverse/ad-prepare
        builder (ns-resolve 'raster.ad.reverse 'runtime-function-form)
        function-form @builder]
    (binding [*unchecked-math* true]
      (with-redefs [reverse/ad-prepare (fn [& arguments]
                                       (swap! seen conj [:prepare *unchecked-math*])
                                       (apply prepare arguments))]
        (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
          (with-redefs-fn {builder (fn [& arguments]
                                    (swap! seen conj [:compile *unchecked-math*])
                                    (apply function-form arguments))}
            #(is (= [9.0 6.0] (result prepared 3.0)))))))
    (is (= [[:prepare false] [:compile false]] @seen))))

(deftest legacy-input-does-not-acquire-generated-program-authority
  (binding [entry/*emission-inputs* (atom {:static [] :values []})]
    (with-redefs [entry/original-source-frontend (constantly nil)]
      (let [prepared (reverse/prepare-value+grad #'squared-loss [0])]
        (is (nil? (reverse/assert-prepared-program-source! prepared)))
        (is (seq (:unclosed-source-imports @entry/*emission-inputs*)))))))

(deftest inactive-checked-conversion-keeps-its-definition-policy
  (let [compiled (binding [*unchecked-math* true]
                   ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad))
                    (reverse/prepare-value+grad #'checked-policy-loss [0])))]
    (is (= [11.0 6.0 nil] (compiled 3.0 2.0)))
    (is (thrown? IllegalArgumentException (compiled 3.0 Double/MAX_VALUE)))))

(deftest normalized-parameter-selection-retains-padded-gradient-slots
  (doseq [[wrt expected] [[[0] [25.0 6.0 nil]] [[1] [25.0 nil 8.0]]]]
    (let [prepared (reverse/prepare-value+grad #'two-squared-loss wrt)
          compiled ((deref (ns-resolve 'raster.ad.reverse 'compile-prepared-value+grad)) prepared)]
      (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))
      (is (= expected (compiled 3.0 4.0))))))

(deftest prepared-compilation-retains-history-in-an-enclosing-owner
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* parent]
      (is (= [9.0 6.0] (result prepared 3.0))))
    (is (seq (:definition-proofs @parent)))
    (is (entry/assert-definition-inputs-current! @parent))
    (let [revision (templates/registry-revision) before @parent]
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (binding [entry/*emission-inputs* parent]
          (is (= :ad-prepared-program-stale-rules
                 (try (reverse/adopt-prepared-program-inputs! prepared) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
      (is (identical? before @parent)))))

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

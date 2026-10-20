(ns raster.ad.preparation-frontend-source-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.method-entry :as entry]))

(deftm squared-loss [x :- Double] :- Double (numeric/* x x))
(deftm cold-squared-loss [x :- Double] :- Double (numeric/* x x))
(deftm checked-policy-loss [x :- Double y :- Double] :- Double
  (let [rounded (float y)] (numeric/+ (numeric/* x x) (double rounded))))
(deftm two-squared-loss [x :- Double y :- Double] :- Double
  (numeric/+ (numeric/* x x) (numeric/* y y)))

(deftm gradient-calling-root [x :- Double] :- Double
  (let [vg ((reverse/value+grad (var squared-loss) :wrt [0]) x)]
    (clojure.core/nth vg 1)))

(deftm same-signature-owner-donor [x :- Double] :- Double (numeric/+ x 1.0))

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

(deftest frozen-source-remains-available-when-a-transformed-module-has-stale-ad-rules
  (is (= 6.0 (gradient-calling-root 3.0)))
  (let [source (core/resolve-deftm-var #'gradient-calling-root)
        receiver (ns-resolve (:ns (meta source))
                             (symbol (str (:name (meta source)) "-impl")))
        implementation (var-get receiver)
        emission (:raster.core/bytecode-emission (meta receiver))
        old (reverse/prepare-value+grad #'squared-loss [0])
        revision (templates/registry-revision)]
    (is (= :jvm-bytecode (:backend emission)))
    (is (seq (:producer-inputs emission)) "the real emitted root retains its generated AD owner")
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (entry/assert-emission-inputs-current! emission) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/assert-prepared-program-source! old) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (entry/call-with-definition-inputs
       (fn []
         (let [frontend (entry/original-source-frontend source)]
           (is (some? frontend))
           (is (identical? frontend (entry/assert-frontend-source! frontend source)))
           (is (identical? implementation (var-get receiver)))
           (is (= :ad-prepared-program-stale-rules
                  (try (entry/frontend-reduction-emission frontend (first (:body frontend))) nil
                       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
      (let [fresh (reverse/prepare-value+grad #'squared-loss [0])]
        (is (identical? fresh (reverse/assert-prepared-program-source! fresh)))
        (is (= [9.0 6.0] (result fresh 3.0)))))))

(deftest frozen-source-rejects-a-same-signature-owner-from-another-implementation
  (is (= 9.0 (squared-loss 3.0)))
  (is (= 4.0 (same-signature-owner-donor 3.0)))
  (let [source (core/resolve-deftm-var #'squared-loss)
        receiver-for (fn [source]
                       (ns-resolve (:ns (meta source))
                                   (symbol (str (:name (meta source)) "-impl"))))
        receiver (receiver-for source)
        donor (receiver-for (core/resolve-deftm-var #'same-signature-owner-donor))
        foreign (get-in (meta donor) [:raster.core/bytecode-emission :frontend-source-owner])
        before (meta receiver)]
    (is (some? foreign))
    (try
      (alter-meta! receiver assoc-in [:raster.core/bytecode-emission :frontend-source-owner] foreign)
      (is (= :retained-frontend-owner-mismatch
             (try (entry/call-with-definition-inputs #(entry/original-source-frontend source)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (finally (reset-meta! receiver before)))
    (is (some? (entry/call-with-definition-inputs #(entry/original-source-frontend source))))))

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
    (is (= 1 (count (:producer-inputs @parent))))
    (is (entry/assert-definition-inputs-current! @parent))
    (let [revision (templates/registry-revision) before @parent]
      (with-redefs [templates/registry-revision (constantly (inc revision))]
        (binding [entry/*emission-inputs* parent]
          (is (= :ad-prepared-program-stale-rules
                 (try (reverse/adopt-prepared-program-inputs! prepared) nil
                      (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))
      (is (identical? before @parent)))))

(deftest enclosing-owner-retains-the-preparations-template-guard
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []})
        revision (templates/registry-revision)]
    (binding [entry/*emission-inputs* parent]
      (reverse/adopt-prepared-program-inputs! prepared)
      (reverse/adopt-prepared-program-inputs! prepared))
    (is (= 1 (count (:producer-inputs @parent))))
    (is (every? #(not (satisfies? entry/RetainedProducerInput %)) (:values @parent)))
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (entry/assert-definition-inputs-current! @parent) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))

(deftest whole-call-projection-uses-shared-nested-scope-hygiene
  ;; Exercise the structural legacy path without manufacturing an opaque owner.
  ;; Existing public AD tests separately exercise authenticated preparations.
  (let [loss (with-meta 'loss {:raster.type/tag 'double :projection-marker :kept})
        source (list 'let*
                     [loss '(let* [reused x
                                   f (fn* [x] (clojure.core/* x 2.0))
                                   walked (loop* [x 0 total reused]
                                            (if (clojure.core/< x 2)
                                              (recur (clojure.core/inc x)
                                                     (clojure.core/+ total (clojure.core/inc x)))
                                              total))]
                              (clojure.core/+ (f reused) walked))
                      'gradient 'x]
                     [loss 'gradient])
        prepared {:walked-body [source] :params '[x] :source-ns (the-ns 'clojure.core)}
        project #(reverse/project-prepared-value+grad-call prepared '[reused] :value+grad {})
        first-call (project)
        second-call (project)
        source-of #(list 'let* (vec (mapcat identity (:bindings %))) (:elements %))
        binders (fn [expression]
                  (set (mapcat #(mapcat :binders (:scopes (form/scope-info %)))
                               (tree-seq coll? seq expression))))
        first-binders (binders (source-of first-call))]
    (is (= [12.0 3.0] ((eval (list 'fn* '[reused] (source-of first-call))) 3.0)))
    (is (empty? (set/intersection first-binders (binders (source-of second-call)))))
    (is (empty? (set/intersection first-binders '#{loss reused f walked x total gradient})))
    (is (= :kept (:projection-marker (meta (:loss-sym first-call)))))
    (is (nil? (reverse/assert-prepared-program-source! prepared)))
    (is (= 2 (count (:elements first-call))))))

(deftest whole-call-projection-retains-owned-padding-and-rejects-stale-inputs
  (doseq [[wrt expected] [[[] [25.0 nil nil]] [[0] [25.0 6.0 nil]] [[1] [25.0 nil 8.0]]]]
    (let [prepared (reverse/prepare-value+grad #'two-squared-loss wrt)
          projection (reverse/project-prepared-value+grad-call prepared '[a b] :value+grad {})
          execute (eval (list 'fn* '[a b]
                              (list 'let* (vec (mapcat identity (:bindings projection)))
                                    (:elements projection))))]
      (is (= expected (execute 3.0 4.0)))
      (is (= 3 (count (:elements projection))))))
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        revision (templates/registry-revision)]
    (with-redefs [templates/registry-revision (constantly (inc revision))]
      (is (= :ad-prepared-program-stale-rules
             (try (reverse/project-prepared-value+grad-call prepared '[x] :value+grad {}) nil
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
    (is (= :ad-prepared-program-mismatch
           (try (reverse/project-prepared-value+grad-call
                 (assoc prepared :walked-body '[(let* [] [0.0 0.0])]) '[x] :value+grad {}) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(deftest whole-call-projection-rejects-private-or-malformed-source
  (doseq [source '[42 (let* [] [0.0]) (let* [a] [0.0 0.0]) (let* [] [0.0 0.0] 9.0)]]
    (is (= :ad-prepared-call-projection
           (try (reverse/project-prepared-value+grad-call
                 {:walked-body [source] :params '[x] :source-ns (the-ns 'clojure.core)}
                 '[x] :value+grad {}) nil
                (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))
  (is (= :unprojected-jvm-source-owner
         (try (reverse/project-prepared-value+grad-call
               {:source-owner nil :walked-body '[(let* [] [0.0 0.0])]
                :params '[x] :source-ns (the-ns 'clojure.core)}
               '[x] :value+grad {}) nil
              (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))

(deftest whole-call-boundary-retains-its-own-unclosed-marker-only-on-success
  (let [prepared (reverse/prepare-value+grad #'squared-loss [0])
        parent (atom {:static [] :values []}) before @parent]
    (binding [entry/*emission-inputs* parent]
      (is (thrown? clojure.lang.ExceptionInfo
                   (reverse/project-prepared-value+grad-call prepared '[] :value+grad {})))
      (is (identical? before @parent))
      (reverse/project-prepared-value+grad-call prepared '[x] :value+grad {})
      (is (seq (:producer-inputs @parent)))
      (is (some #(= :prepared-reverse-ad (first %)) (:unclosed-source-imports @parent))))))

(deftest lazy-jit-gradient-root-retains-its-actual-ad-producer
  (let [scope (create-ns (gensym "raster.ad-preparation-jit."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm square [x :- Double] :- Double (raster.numeric/* x x)))
        (eval '(raster.core/deftm gradient [x :- Double] :- Double
                 (let [vg ((raster.ad.reverse/value+grad (var square)) x)]
                   (clojure.core/nth vg 1)))))
      (let [root (ns-resolve scope 'gradient)
            _ (is (= 6.0 (binding [core/*jit-silent-fallback?* false] (root 3.0))))
            source (core/resolve-deftm-var root)
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            emission (:raster.core/bytecode-emission (meta receiver))
            revision (templates/registry-revision)]
        (is (seq (:producer-inputs emission))
            "a real JIT root retains preparation consumed before bytecode emission")
        (let [guard (entry/prepare-emission-input-guard! emission)]
          (with-redefs [templates/registry-revision (constantly (inc revision))]
            (is (= :ad-prepared-program-stale-rules
                   (try (guard) nil
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))))))
      (finally (remove-ns (ns-name scope))))))

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

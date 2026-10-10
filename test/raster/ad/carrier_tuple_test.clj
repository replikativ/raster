(ns raster.ad.carrier-tuple-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inf]
            [raster.ad.fixtures.carrier-result-source :as result-source]
            [raster.ad.forward])
  (:import [raster.ad.forward Dual]
           [java.util Date]))

(deftm asymmetric [x :- Double y :- Double] :- Double (n/+ x y))
(deftm asymmetric [x :- Dual y :- Double] :- Dual (n/+ x y))
(deftm both-active [x :- Double y :- Double] :- Double (asymmetric x y))
(deftm first-active [x :- Double] :- Double (asymmetric x 3.0))
(deftm second-active [y :- Double] :- Double (asymmetric 2.0 y))
(deftm ordinary [x :- Double] :- Double (n/+ x 1.0))
(deftm extracted [x :- Double] :- Double (ordinary (n/real-value x)))
(deftm extracted-alias [x :- Double] :- Double
  (let [primal (n/real-value x)] (ordinary primal)))
(deftm joined [x :- Double control :- Long] :- Double
  (ordinary (if (== control 0) (n/real-value x) x)))
(def unknown-executions (atom 0))
(defn unknown-result [_] (swap! unknown-executions inc) "not numeric")
(deftm unknown-primal-argument [x :- Double control :- Long] :- Double
  (n/+ x (unknown-result control)))

(deftm mixed [x :- Object y :- Double] :- Double 7.0)
(deftm mixed (All [T] [x :- T y :- Double] :- T (n/+ x y)))
(deftm mixed-active [x :- Double] :- Double (mixed x 3.0))
(deftm templated [x :- Object y :- Object] :- Long 7)
(deftm templated (All [T] [x :- T y :- T] :- T (n/+ x y)))
(deftm template-before-fallback [x :- Double] :- Double
  (asymmetric (templated x x) 3.0))
(deftm invalid-template (All [T] [x :- T] :- T (double x)))
(deftm invalid-carrier-body [x :- Double] :- Double (invalid-template x))
(def hidden-executions (atom 0))
(defn ^{:tag 'double} hidden-narrow [x]
  (swap! hidden-executions inc)
  (double x))
(deftm erased-template (All [T] [x :- T] :- Double (hidden-narrow x)))
(deftm erased-carrier-body [x :- Double] :- Double (erased-template x))
(deftm nested-erased-template (All [T] [x :- T] :- Double (erased-template x)))
(deftm nested-erased-carrier-body [x :- Double] :- Double (nested-erased-template x))
(deftm promoted-nested (All [T] [x :- T y :- Double] :- Double (hidden-narrow x)))
(deftm promoted-nested (All [T] [x :- T y :- Float] :- Double (n/real-value x)))
(deftm promoted-outer (All [T] [x :- T y :- Double] :- Double (promoted-nested x y)))
(deftm promoted-carrier-body [x :- Double] :- Double (promoted-outer x (float 3.0)))
(deftm promoted-safe (All [T] [x :- T y :- Double] :- T (n/+ x y)))
(deftm promoted-safe-body [x :- Double] :- Double (promoted-safe x (float 3.0)))
(deftm identity-template (All [T] [x :- T] :- T x))
(deftm derived-identity [x :- Double] :- Double (identity-template x))
(deftm identity-template-two (All [T] [x :- T] :- T x))
(deftm derived-identity-two [x :- Double] :- Double (identity-template-two x))
(deftm unpublished-template (All [T] [x :- T] :- T x))
(deftm unpublished-carrier-body [x :- Double] :- Double (unpublished-template x))
(deftm date-consumer [x :- Double d :- java.sql.Date] :- Double (n/+ x 3.0))
(deftm date-consumer [x :- Dual d :- Date] :- Dual (n/+ x 100.0))
(deftm date-consumer [x :- Dual d :- java.sql.Date] :- Dual (n/+ x 3.0))
(deftm qualified-result [x :- Double] :- Double
  (date-consumer x (result-source/date-result x)))
(deftm uncovered-producer [x :- Double] :- Double (n/* x (Math/expm1 x)))
(deftm loop-as-operand [x :- Double count :- Long] :- Double
  (n/+ x (loop [i 0 a 0.0]
           (if (< i count) (recur (inc i) (n/+ a x)) a))))
(deftm loop-primal-result [x :- Double count :- Long] :- Double
  (ordinary (loop [i 0 a x]
              (if (< i count) (recur (inc i) (n/+ a x)) (n/real-value a)))))

(deftest unsupported-tuples-decline-at-construction
  (doseq [v [#'both-active #'second-active #'joined]]
    (let [coverage (reverse/forward-coverage v)
          error (try (reverse/value+grad v :mode :forward)
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (false? (:admissible? coverage)))
      (is (instance? clojure.lang.ExceptionInfo error))
      (is (= (:call-declines coverage) (:call-declines (ex-data error)))))))

(deftest actual-results-propagate-through-call-and-binding
  (doseq [[v args expected] [[#'first-active [2.0] [5.0 1.0]]
                            [#'extracted [2.0] [3.0 0.0]]
                            [#'extracted-alias [2.0] [3.0 0.0]]]]
    (is (:admissible? (reverse/forward-coverage v)))
    (is (= expected (apply (reverse/value+grad v :mode :forward) args)))))

(deftest unknown-primal-result-is-not-a-tuple-witness
  (let [before @unknown-executions
        coverage (reverse/forward-coverage #'unknown-primal-argument)]
    (is (false? (:admissible? coverage)))
    (is (nil? (get-in coverage [:call-declines 0 :argument-tuples])))
    (is (thrown? clojure.lang.ExceptionInfo
                 (reverse/value+grad #'unknown-primal-argument :mode :forward)))
    (is (= before @unknown-executions) "admission does not execute an unknown helper")))

(deftest selection-priority-matches-runtime
  (is (false? (:admissible? (reverse/forward-coverage #'mixed-active)))
      "a mixed Object/Double method blocks parametric evidence")
  (let [op 'raster.ad.carrier-tuple-test/templated
        tuple '[raster.ad.forward.Dual raster.ad.forward.Dual]
        selected (binding [*ns* (the-ns 'raster.ad.carrier-tuple-test)]
                   (dispatch/selected-call-signature
                    op tuple (inf/registered-call-selection op tuple)))]
    (is (#{:parametric :registered} (:selection selected)))
    (is (= :parametric (:selection (dispatch/selected-call-signature
                                    op tuple {:tags '[Object Object]
                                              :signature {:return-tag 'long}}))))
    (is (nil? (dispatch/selected-call-signature op tuple {:tags '[Object Double]}))
        "an unprojectable mixed method still blocks unrelated template evidence")
    (is (some? (:return-tag selected))))
  (is (:admissible? (reverse/forward-coverage #'template-before-fallback)))
  (is (= [7.0 2.0] ((reverse/value+grad #'template-before-fallback :mode :forward) 2.0)))
  (is (:admissible? (reverse/forward-coverage #'template-before-fallback))
      "warm specialization preserves admission"))

(deftest invalid-carrier-specialization-declines-before-invocation
  (dotimes [_ 2]
    (let [coverage (reverse/forward-coverage #'invalid-carrier-body)]
      (is (false? (:admissible? coverage)))
      (is (seq (get-in coverage [:call-declines 0 :signature-failures])))
      (is (thrown? clojure.lang.ExceptionInfo
                   (reverse/value+grad #'invalid-carrier-body :mode :forward))))))

(deftest erased-host-call-is-not-certified-by-compilation
  (let [before @hidden-executions]
    (doseq [v [#'erased-carrier-body #'nested-erased-carrier-body]
            _ (range 2)]
      (let [coverage (reverse/forward-coverage v)]
        (is (false? (:admissible? coverage)))
        (is (seq (get-in coverage [:call-declines 0 :signature-failures])))
        (is (thrown? clojure.lang.ExceptionInfo
                     (reverse/value+grad v :mode :forward)))))
    (is (= before @hidden-executions) "validation never invokes the host helper")))

(deftest derived-identity-retains-its-carrier
  (dotimes [_ 2]
    (is (:admissible? (reverse/forward-coverage #'derived-identity)))
    (is (= [2.0 1.0] ((reverse/value+grad #'derived-identity :mode :forward) 2.0)))))

(deftest result-class-resolves-in-defining-namespace
  (is (:admissible? (reverse/forward-coverage #'qualified-result)))
  (is (= [5.0 1.0] ((reverse/value+grad #'qualified-result :mode :forward) 2.0))))

(deftest uncovered-producer-does-not-mislabel-a-lifted-consumer
  (let [coverage (reverse/forward-coverage #'uncovered-producer)]
    (is (false? (:admissible? coverage)))
    (is (= '[Math/expm1] (:uncovered-ops coverage)))
    (is (some #(= 'raster.numeric/* (:operation %)) (:call-declines coverage))
        "unknown-result propagation remains a refusal with full diagnostic evidence")))

(deftest derived-body-sees-post-promotion-argument-facts
  (binding [*ns* (the-ns 'raster.ad.carrier-tuple-test)]
    ;; Publish the Double implementations before presenting a Float caller.
    ;; The safe Float nested overload must not certify the actual Double path.
    (doseq [op '[promoted-outer promoted-safe]]
      (dispatch/register-parametric-call-tags! op '[raster.ad.forward.Dual double]))
    (let [selection (inf/registered-call-signature 'promoted-outer '[raster.ad.forward.Dual float])
          before @hidden-executions]
      (is (= '[nil double] (:promotion-casts selection)))
      (is (false? (:admissible? (reverse/forward-coverage #'promoted-carrier-body))))
      (is (= before @hidden-executions)))
    (is (:admissible? (reverse/forward-coverage #'promoted-safe-body)))
    (is (= [5.0 1.0] ((reverse/value+grad #'promoted-safe-body :mode :forward) 2.0)))))

(deftest declaration-without-publication-is-not-an-implementation
  (with-redefs [dispatch/register-parametric-call-tags! (fn [& _] nil)]
    (let [coverage (reverse/forward-coverage #'unpublished-carrier-body)]
      (is (false? (:admissible? coverage)))
      (is (= :unpublished-carrier-specialization
             (get-in coverage [:call-declines 0 :signature-failures 0 :reason]))))))

(deftest retained-body-proof-fails-closed-and-is-bounded
  ;; Materialize real implementations first. Replace only their retained-body
  ;; seam to exercise malformed and cyclic proof graphs without executing them.
  (is (:admissible? (reverse/forward-coverage #'derived-identity)))
  (is (:admissible? (reverse/forward-coverage #'derived-identity-two)))
  (let [ns' (the-ns 'raster.ad.carrier-tuple-test)
        lookup (fn [op]
                 (binding [*ns* ns']
                   (resolve (:mangled-sym
                             (inf/registered-call-signature op '[raster.ad.forward.Dual])))))
        a (lookup 'identity-template)
        b (lookup 'identity-template-two)
        original core/ensure-walked-body!
        refusal (fn [] (reverse/forward-coverage #'derived-identity))]
    (with-redefs [core/ensure-walked-body! (fn [v] (if (= v a) nil (original v)))]
      (is (= :missing-carrier-specialization-body
             (get-in (refusal) [:call-declines 0 :signature-failures 0 :reason]))))
    (doseq [mutual? [false true]]
      (with-redefs [core/ensure-walked-body!
                   (fn [v]
                     (cond (= v a) [(list (if mutual?
                                           'raster.ad.carrier-tuple-test/identity-template-two
                                           'raster.ad.carrier-tuple-test/identity-template)
                                         (first (:raster.core/deftm-params (meta a))))]
                           (= v b) [(list 'raster.ad.carrier-tuple-test/identity-template
                                         (first (:raster.core/deftm-params (meta b))))]
                           :else (original v)))]
        (let [coverage (refusal)]
          (is (false? (:admissible? coverage)))
          (is (some #(= :recursive-carrier-specialization (:reason %))
                    (tree-seq coll? seq (:call-declines coverage)))))))
    (let [plan (#'reverse/forward-conversion-plan
                '(raster.ad.carrier-tuple-test/identity-template x) '[x] '[double] ns'
                {:validation-state {:visiting (volatile! #{}) :remaining (volatile! 0)}})]
      (is (= :carrier-validation-budget
             (get-in plan [:call-declines 0 :signature-failures 0 :reason]))))))

(deftest canonical-loop-results-do-not-include-recur
  (doseq [[v expected] [[#'loop-as-operand [8.0 4.0 nil]]
                       [#'loop-primal-result [9.0 0.0 nil]]]]
    (is (:admissible? (reverse/forward-coverage v)))
    (is (= expected ((reverse/value+grad v :mode :forward) 2.0 3)))))

(deftest carrier-identity-does-not-follow-basename
  (is (#'reverse/dual-tag? 'raster.ad.forward.Dual))
  (is (not (#'reverse/dual-tag? 'foreign.namespace.Dual)))
  (is (not (#'reverse/dual-tag? 'Object))))

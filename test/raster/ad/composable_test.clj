(ns raster.ad.composable-test
  "Tests for composable AD operators: value+grad, grad.

  These operators return IFn objects that work in two modes:
  1. Runtime: directly callable as functions
  2. Compiled: carry deftm metadata for pipeline inlining"
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as r]
            [raster.numeric]
            [raster.ad.reverse :as rev]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Test functions
;; ================================================================

(r/deftm quad-loss [x :- Double, y :- Double] :- Double
  (raster.numeric/+ (raster.numeric/* x x) (raster.numeric/* y y)))

(r/deftm cubic-fn [x :- Double] :- Double
  (raster.numeric/* x (raster.numeric/* x x)))

;; Function with nested let in call args (tests let-hoisting)
(r/deftm nested-let-fn [x :- Double, y :- Double] :- Double
  (raster.numeric/+ (let [a (raster.numeric/* x x)] a)
                    (let [b (raster.numeric/* y y)] b)))

;; The wrapper makes its VJP depend on a nested, registry-provided rule rather
;; than only on rules named directly by the VJP target's own body.
(r/deftm cache-rule-inner [x :- Double] :- Double
  (raster.numeric/* x x))

(r/deftm cache-rule-outer [x :- Double] :- Double
  (cache-rule-inner x))

(defn- cache-rule-template [factor]
  {:params '[x]
   :result nil
   :adjoint 'dy
   :grads-fn
   (fn [ctx [x] _result adjoint gensym-fn]
     (let [g (gensym-fn "cache_rule_grad")]
       [(update ctx :bindings into
                [g (list 'raster.numeric/* adjoint
                         (list 'raster.numeric/* factor x))])
        [g]]))})

;; ================================================================
;; value+grad runtime
;; ================================================================

(deftest value+grad-basic-test
  (testing "value+grad computes [value grad1 grad2]"
    (let [vg (rev/value+grad #'quad-loss)
          [val dx dy] (vg 3.0 4.0)]
      (is (= 25.0 val) "f(3,4) = 9+16 = 25")
      (is (= 6.0 dx)   "df/dx = 2x = 6")
      (is (= 8.0 dy)   "df/dy = 2y = 8"))))

(deftest value+grad-single-arg-test
  (testing "value+grad on single-arg function"
    (let [vg (rev/value+grad #'cubic-fn)
          [val dx] (vg 2.0)]
      (is (= 8.0 val) "f(2) = 8")
      (is (= 12.0 dx) "df/dx = 3x^2 = 12"))))

(deftest vjp-cache-observes-nested-rule-replacement-test
  (testing "fresh VJP acquisition recompiles after merge; acquired pullbacks remain snapshots"
    (let [op 'raster.ad.composable-test/cache-rule-inner]
      (tmpl/register-template! op (cache-rule-template 2.0))
      (try
        (let [[value-before pullback-before] (rev/vjp #'cache-rule-outer 2.0)]
          (is (= 4.0 value-before))
          (is (= [4.0] (pullback-before 1.0)))

          ;; Exercise merge-into-template! specifically: replacing the generator
          ;; must invalidate transforms whose dependency is this nested rule.
          (tmpl/merge-into-template! op
                                     {:grads-fn (:grads-fn (cache-rule-template 3.0))})
          (let [[value-after-merge pullback-after-merge]
                (rev/vjp #'cache-rule-outer 2.0)]
            (is (= 4.0 value-after-merge))
            (is (= [6.0] (pullback-after-merge 1.0))
                "new acquisition uses the merged nested rule")
            (is (= [4.0] (pullback-before 1.0))
                "a previously acquired pullback retains snapshot semantics"))

          ;; Full replacement is the other public registry mutation path.
          (tmpl/register-template! op (cache-rule-template 4.0))
          (let [[_ pullback-after-register] (rev/vjp #'cache-rule-outer 2.0)]
            (is (= [8.0] (pullback-after-register 1.0))
                "register-template! also invalidates the nested transform")))
        (finally
          (tmpl/register-template! op (cache-rule-template 2.0)))))))

(deftest vjp-cache-retries-a-registry-mutation-during-transform-test
  (testing "a transform spanning two registry revisions is discarded and retried"
    (let [op 'raster.ad.composable-test/cache-rule-inner
          original-transform rev/transform-body
          mutated? (atom false)]
      (tmpl/register-template! op (cache-rule-template 4.0))
      (try
        (with-redefs [rev/transform-body
                      (fn [& args]
                        (let [transformed (apply original-transform args)]
                          (when (compare-and-set! mutated? false true)
                            (tmpl/merge-into-template!
                             op {:grads-fn (:grads-fn (cache-rule-template 5.0))}))
                          transformed))]
          (let [[_ pullback] (rev/vjp #'cache-rule-outer 2.0)]
            (is @mutated?)
            (is (= [10.0] (pullback 1.0))
                "the pre-mutation transform was not returned or cached")))
        (finally
          (tmpl/register-template! op (cache-rule-template 2.0)))))))

(deftest value+grad-metadata-test
  (testing "value+grad result carries deftm metadata"
    (let [vg (rev/value+grad #'quad-loss)
          m (meta vg)]
      (is (:raster.ad.reverse/value+grad m))
      (is (vector? (:raster.core/deftm-walked-body m)))
      (is (vector? (:raster.core/deftm-params m)))
      (is (vector? (:raster.core/deftm-tags m))))))

;; ================================================================
;; grad runtime
;; ================================================================

(deftest grad-basic-test
  (testing "grad computes [grad1 grad2] without value"
    (let [g (rev/grad #'quad-loss)
          [dx dy] (g 3.0 4.0)]
      (is (= 6.0 dx) "df/dx = 2x = 6")
      (is (= 8.0 dy) "df/dy = 2y = 8"))))

(deftest grad-metadata-test
  (testing "grad result carries deftm metadata"
    (let [g (rev/grad #'quad-loss)
          m (meta g)]
      (is (:raster.ad.reverse/grad m))
      (is (vector? (:raster.core/deftm-walked-body m)))
      (is (vector? (:raster.core/deftm-params m))))))

;; ================================================================
;; Forward mode
;; ================================================================

(deftest value+grad-forward-mode-test
  (testing "forward mode via Dual numbers on walked body"
    (let [vg (rev/value+grad #'quad-loss :mode :forward)
          result (vg 3.0 4.0)]
      (is (= 25.0 (first result)) "value = 25")
      (is (= 6.0 (nth result 1))  "df/dx = 6")
      (is (= 8.0 (nth result 2))  "df/dy = 8"))))

;; ================================================================
;; Pipeline integration: inlining metadata
;; ================================================================

(deftest value+grad-walked-body-is-let-star-test
  (testing "walked body is a let* form (inlinable)"
    (let [vg (rev/value+grad #'quad-loss)
          wb (:raster.core/deftm-walked-body (meta vg))]
      (is (= 1 (count wb)) "single walked body form")
      (is (seq? (first wb)) "body is a seq")
      (is (= 'let* (first (first wb))) "body starts with let*"))))

(deftest value+grad-params-match-original-test
  (testing "params match original deftm params"
    (let [vg (rev/value+grad #'quad-loss)
          params (:raster.core/deftm-params (meta vg))]
      (is (= '[x y] params)))))

;; ================================================================
;; Auto mode selection (Griewank heuristic)
;; ================================================================

(deftest value+grad-auto-single-param-test
  (testing "auto mode selects forward for single-param functions"
    (let [vg (rev/value+grad #'cubic-fn :mode :auto)
          [val dx] (vg 2.0)]
      (is (= 8.0 val))
      (is (= 12.0 dx)))))

(deftest value+grad-auto-multi-param-test
  (testing "auto mode selects reverse for multi-param functions"
    (let [vg (rev/value+grad #'quad-loss :mode :auto)
          [val dx dy] (vg 3.0 4.0)]
      (is (= 25.0 val))
      (is (= 6.0 dx))
      (is (= 8.0 dy)))))

;; ================================================================
;; Nested let hoisting
;; ================================================================

(deftest value+grad-nested-let-test
  (testing "AD handles let expressions inside call arguments"
    (let [vg (rev/value+grad #'nested-let-fn)
          [val dx dy] (vg 3.0 4.0)]
      (is (= 25.0 val) "f(3,4) = 9+16 = 25")
      (is (= 6.0 dx)   "df/dx = 2x = 6")
      (is (= 8.0 dy)   "df/dy = 2y = 8"))))

;; ================================================================
;; Forward-mode single arg
;; ================================================================

(deftest value+grad-forward-single-arg-test
  (testing "forward mode on single-arg function"
    (let [vg (rev/value+grad #'cubic-fn :mode :forward)
          [val dx] (vg 2.0)]
      (is (= 8.0 val))
      (is (= 12.0 dx)))))

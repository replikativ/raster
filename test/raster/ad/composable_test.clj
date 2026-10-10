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

(r/deftm forward-rounded-primal [x :- Double] :- Double (double (float x)))
(r/deftm forward-integral-primal [x :- Double] :- Double (double (long x)))
(r/deftm forward-qualified-rounded-primal [x :- Double] :- Double
  (clojure.core/double (clojure.core/float x)))
(r/deftm forward-double-identity [x :- Double] :- Double (double x))
(r/deftm forward-aliased-rounded-primal [x :- Double] :- Double
  (let [alias x] (double (float alias))))
(r/deftm forward-inactive-rounded-primal [x :- Double n :- Long] :- Double
  (let [rounded (float n)]
    (raster.numeric/+ (raster.numeric/* x x) (double rounded))))
(r/deftm forward-inactive-checked-primal [x :- Double n :- Long] :- Double
  (let [narrowed (int n)]
    (raster.numeric/+ (raster.numeric/* x x) (double narrowed))))
(r/deftm forward-shadowed-primal [x :- Double n :- Long] :- Double
  (let [x n] (double x)))
(r/deftm forward-conditional-rounded-primal [x :- Double] :- Double
  (if (> x 0.0) (double (float x)) x))
(r/deftm forward-carried-rounded-primal [x :- Double] :- Double
  (loop [i 0 acc 0.0]
    (if (< i 2) (recur (inc i) (double (float x))) acc)))
(r/deftm forward-transferred-carry-primal [x :- Double] :- Double
  (loop [a x b 0.0 i 0]
    (if (< i 2) (recur a a (inc i)) (double (float b)))))

(deftest forward-conversions-retain-primal-semantics
  (testing "active rounding and discrete conversions decline before execution"
    (doseq [v [#'forward-rounded-primal #'forward-integral-primal
               #'forward-qualified-rounded-primal #'forward-aliased-rounded-primal
               #'forward-conditional-rounded-primal #'forward-carried-rounded-primal
               #'forward-transferred-carry-primal]]
      (let [coverage (rev/forward-coverage v)]
        (is (false? (:admissible? coverage)))
        (is (seq (:conversion-declines coverage)))
        (is (thrown? clojure.lang.ExceptionInfo (rev/value+grad v :mode :forward)))))
    (is (= 16777216.0 (forward-rounded-primal 16777217.0)))
    (is (= 3.0 (forward-integral-primal 3.7))))
  (testing "auto excludes the unsupported Dual interpretation and retains source primals"
    (is (= (forward-rounded-primal 16777217.0)
           (first ((rev/value+grad #'forward-rounded-primal :mode :auto) 16777217.0))))
    (is (= (forward-integral-primal 3.7)
           (first ((rev/value+grad #'forward-integral-primal :mode :auto) 3.7)))))
  (testing "certified Double identity retains primal and partial"
    (is (= [3.7 1.0] ((rev/value+grad #'forward-double-identity :mode :forward) 3.7))))
  (testing "inactive Float rounding is real, not erased alongside active arithmetic"
    (let [vg (rev/value+grad #'forward-inactive-rounded-primal :mode :forward)]
      (is (= [(forward-inactive-rounded-primal 3.0 16777217) 6.0 nil]
             (vg 3.0 16777217)))))
  (testing "inactive checked narrowing retains its failure"
    (let [vg (rev/value+grad #'forward-inactive-checked-primal :mode :forward)]
      (is (= [12.0 6.0 nil] (vg 3.0 3)))
      (is (thrown? ArithmeticException (forward-inactive-checked-primal 3.0 2147483648)))
      (is (thrown? ArithmeticException (vg 3.0 2147483648)))))
  (testing "lexical shadowing does not inherit the outer seeded carrier"
    (let [vg (rev/value+grad #'forward-shadowed-primal :mode :forward)]
      (is (= [(forward-shadowed-primal 3.0 9007199254740993) 0.0 nil]
             (vg 3.0 9007199254740993))))))

(deftest forward-conversion-semantic-call-representation
  ;; Semantic operation metadata is authoritative even before undevirtualization.
  (let [plan @(ns-resolve 'raster.ad.reverse 'forward-conversion-plan)
        call (with-meta '(.invk fake-cast-implementation x) {:raster.op/original 'float})
        result (plan call '[x] '[double] *ns*)]
    (is (= :float (get-in result [:declines 0 :target-dtype])))
    (is (= call (:body result)))
    (is (= (meta call) (meta (:body result)))))
  (let [plan @(ns-resolve 'raster.ad.reverse 'forward-conversion-plan)]
    (doseq [body ['{:value (float x)} '#{(float x)}
                 '(letfn* [f (fn* [] (float x))] (f))]]
      (is (seq (:declines (plan body '[x] '[double] *ns*))) (pr-str body)))))

(r/deftm cubic-fn [x :- Double] :- Double
  (raster.numeric/* x (raster.numeric/* x x)))

(r/deftm flagged-quadratic [x :- Double flag :- Boolean] :- Double
  (if flag (raster.numeric/* x x) (raster.numeric/+ x x)))

(r/deftm quadratic-with-unused-control [x :- Double flag :- Boolean count :- Long] :- Double
  (raster.numeric/* x x))

(r/deftm integer-scaled-quadratic [x :- Double count :- Long] :- Double
  (raster.numeric/* count (raster.numeric/* x x)))

(r/deftm discrete-only-value [flag :- Boolean count :- Long] :- Long
  (if flag count (unchecked-inc count)))

(r/deftm quadratic-with-array-control [x :- Double values :- (Array double)] :- Double
  (raster.numeric/* x x))

(deftest forward-mode-preserves-discrete-primals-and-gradient-slots
  (is (= [0] (:active-indices (rev/forward-coverage #'quadratic-with-unused-control))))
  (is (= '[flag count] (:constant-params (rev/forward-coverage #'quadratic-with-unused-control))))
  (let [coverage (rev/forward-coverage #'quadratic-with-array-control)]
    (is (= [0] (:active-indices coverage)))
    (is (empty? (:constant-params coverage)) "An array tangent is unsupported, not discrete")
    (is (= '[values] (:array-params coverage)))
    (is (false? (:admissible? coverage)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"array-typed params"
                          (rev/value+grad #'quadratic-with-array-control :mode :forward))))
  (doseq [mode [:forward :auto]]
    (let [unused (rev/value+grad #'quadratic-with-unused-control :mode mode)
          flagged (rev/value+grad #'flagged-quadratic :mode mode)
          scaled (rev/value+grad #'integer-scaled-quadratic :mode mode)
          discrete (rev/value+grad #'discrete-only-value :mode mode)]
      (doseq [x [3.0 -2.0] flag [true false]]
        (is (= [(* x x) (* 2.0 x) nil nil] (unused x flag Long/MAX_VALUE)))
        (is (= (if flag [(* x x) (* 2.0 x) nil] [(+ x x) 2.0 nil])
               (flagged x flag)))
        (is (= [(* 3.0 x x) (* 6.0 x) nil] (scaled x 3))))
      (is (= [Long/MAX_VALUE nil nil] (discrete true Long/MAX_VALUE)))
      (is (= [9007199254740994 nil nil] (discrete false 9007199254740993))))))

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

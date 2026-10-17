(ns raster.ad.absent-pullback-seeds-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.math :as math]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.core.op-descriptor :as opdesc]
            [raster.compiler.pipeline :as pipeline]))

(deftm scaled [x :- Double] :- Double (n/* x 2.0))
(deftm scaled-gradient [x :- Double] :- Double
  (let [result ((reverse/value+grad #'scaled :wrt [0]) x)] (nth result 1)))

(deftm density [x :- Double s :- Double] :- Double
  ;; Exponential log-density, x >= 0 and s > 0.
  (n/- (math/log s) (n/* s x)))
(deftm density-gradient [x :- Double s :- Double] :- Double
  (let [result ((reverse/value+grad #'density :wrt [0]) x s)] (nth result 1)))

(deftm cubic [x :- Double] :- Double (n/* x (n/* x x)))

(deftest absent-lifting-must-preserve-higher-order-composition
  ;; Publication gate: guarding one pullback must not make its reified scalar
  ;; gradient opaque to the next AD pass. Do not waive product/nth failures.
  (let [second-gradient (reverse/value+grad (reverse/reified-grad #'cubic 'x))]
    (doseq [x [-2.0 0.0 0.7 1.5]]
      (let [[first second] (second-gradient x)]
        (is (< (Math/abs (- first (* 3.0 x x))) 1.0e-12))
        (is (< (Math/abs (- second (* 6.0 x))) 1.0e-12))))))

(deftest ordinary-gradient-native-admission-is-preserved
  (doseq [target [:jvm :c]
          [source arguments expected] [[#'scaled-gradient [2.0] 2.0]
                                       [#'density-gradient [0.5 2.0] -2.0]]]
    (let [compiled (pipeline/compile-aot source :target target :dtype :double)]
      (try
        (is (< (Math/abs (- (double expected) (double (apply compiled arguments)))) 1.0e-12))
        (finally
          (when (instance? java.lang.AutoCloseable compiled)
            (.close ^java.lang.AutoCloseable compiled)))))))

(deftest template-projection-is-lazy-only-for-an-absent-edge
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [gradient (with-meta 'gradient {:raster.type/tag 'double})
          before {:bindings [] :marker :retained}
          after (assoc before :bindings
                       [gradient '(do (swap! calls inc) (/ dy x))])
          [context slots] (#'reverse/lift-pullback-over-absence before after [gradient nil] 'dy)
          project (eval (list 'fn* '[dy x calls]
                              (list 'let* (:bindings context) slots)))
          calls (atom 0)]
      (is (= :retained (:marker context)))
      (is (nil? (second slots)) "a statically absent gradient remains absent")
      (is (= [nil nil] (project (identity nil) 0.0 calls)))
      (is (= 0 @calls))
      (is (thrown? ArithmeticException (project 0.0 0.0 calls))
          "an explicit zero seed still evaluates the singular pullback")
      (is (= 1 @calls))
      (is (= [2.0 nil] (project 4.0 2.0 calls)))
      (is (= 2 @calls)))))

(deftest lifting-preserves-present-float-values-without-a-new-cast
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [gradient (with-meta 'gradient {:raster.type/tag 'float})
          [context slots] (#'reverse/lift-pullback-over-absence
                           {:bindings []} {:bindings [gradient 'dy]} [gradient] 'dy)
          project (eval (list 'fn* '[dy] (list 'let* (:bindings context) slots)))
          value (Float/valueOf (float -0.0))]
      (is (= [nil] (project nil)))
      (is (identical? value (first (project value))))
      (is (= (Float/floatToRawIntBits value)
             (Float/floatToRawIntBits (first (project value))))))))

(deftest absent-fragment-does-not-guard-or-duplicate-existing-prefix
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [gradient (with-meta 'gradient {:raster.type/tag 'double})
          before {:bindings ['shared '(do (swap! prefix-calls inc) 2.0)]}
          after (update before :bindings into
                        [gradient '(do (swap! body-calls inc) (+ shared dy))])
          [context slots] (#'reverse/lift-pullback-over-absence before after [gradient] 'dy)
          project (eval (list 'fn* '[dy prefix-calls body-calls]
                              (list 'let* (:bindings context) slots)))
          prefix-calls (atom 0) body-calls (atom 0)]
      (is (= (:bindings before) (subvec (:bindings context) 0 2)))
      (is (= [nil] (project nil prefix-calls body-calls)))
      (is (= [1 0] [@prefix-calls @body-calls]))
      (is (= [5.0] (project 3.0 prefix-calls body-calls)))
      (is (= [2 1] [@prefix-calls @body-calls])))))

(deftest absent-closure-seeds-skip-both-factory-and-invocation
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [factories (atom 0) calls (atom 0)]
      (with-redefs-fn
        {#'templates/resolve-template (fn [_] [nil 'probe])
         #'templates/get-pullback-factory
         (fn [_]
           (fn [& _]
             (swap! factories inc)
             (fn [dy] (swap! calls inc) [(/ dy 0.0)])))}
        (fn []
          (let [state (#'reverse/emit-backward
                       {:type :call :sym 'value :base-op 'probe :args '[x] :active-args '#{x}}
                       'dy {} {:adj-env {} :rev-ctx {:bindings []}})
                project (eval (list 'fn* '[value x dy]
                                    (list 'let* (get-in state [:rev-ctx :bindings])
                                          (first (get-in state [:adj-env 'x])))))]
            (is (nil? (project 1.0 1.0 nil)))
            (is (= [0 0] [@factories @calls]))
            (is (Double/isNaN (project 1.0 1.0 0.0)))
            (is (= [1 1] [@factories @calls]))))))))

(deftest presence-tests-are-discrete-not-differentiable-calls
  (is (opdesc/comparison-op? 'clojure.core/nil?)))

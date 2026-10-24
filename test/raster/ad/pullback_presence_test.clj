(ns raster.ad.pullback-presence-test
  "Absent cotangents skip pullbacks; present zeros retain their arithmetic."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.math :as math]
            [raster.ad.reverse :as reverse]
            [raster.ad.templates :as templates]
            [raster.compiler.pipeline :as pipeline]))

(deftm scaled [x :- Double] :- Double (n/* x 2.0))
(deftm scaled-gradient [x :- Double] :- Double
  (let [result ((reverse/value+grad #'scaled :wrt [0]) x)] (nth result 1)))
(deftm density [x :- Double s :- Double] :- Double
  (n/- (math/log s) (n/* s x)))
(deftm density-gradient [x :- Double s :- Double] :- Double
  (let [result ((reverse/value+grad #'density :wrt [0]) x s)] (nth result 1)))
(deftm cubic [x :- Double] :- Double (n/* x (n/* x x)))

(defn- context []
  {:bindings []
   :gensym-fn (fn
                ([prefix] (gensym (str prefix "__")))
                ([prefix tag]
                 (cond-> (gensym (str prefix "__"))
                   tag (with-meta {:raster.type/tag tag}))))})

(deftest absence-lifting-preserves-higher-order-composition
  (let [second-gradient (reverse/value+grad (reverse/reified-grad #'cubic 'x))]
    (doseq [x [-2.0 0.0 0.7 1.5]]
      (let [[first second] (second-gradient x)]
        (is (< (Math/abs (- first (* 3.0 x x))) 1.0e-12))
        (is (< (Math/abs (- second (* 6.0 x))) 1.0e-12))))))

(deftest ordinary-gradients-retain-invoked-jvm-and-native-admission
  (doseq [target [:jvm :c]
          [source arguments expected] [[#'scaled-gradient [2.0] 2.0]
                                       [#'density-gradient [0.5 2.0] -2.0]]]
    (let [compiled (pipeline/compile-aot source :target target :dtype :double)]
      (try
        (is (< (Math/abs (- expected (double (apply compiled arguments)))) 1.0e-12))
        (finally
          (when (instance? java.lang.AutoCloseable compiled)
            (.close ^java.lang.AutoCloseable compiled)))))))

(deftest density-native-retains-double-math-and-negation-boundaries
  (let [compiled (pipeline/compile-aot #'density-gradient :target :c :dtype :double)]
    (try
      ;; Log's primitive domain failures are IEEE values, not exceptions. Its
      ;; primal remains in the source; wrt x still gives -s at these inputs.
      (doseq [[x s] [[0.5 2.0] [-3.0 0.5] [1.0 -2.0] [3.0 0.0] [4.0 -0.0]
                     [0.0 Double/POSITIVE_INFINITY] [2.0 Double/NaN]]]
        (let [actual (double (compiled x s))
              expected (clojure.core/- (double s))]
          (is (or (and (Double/isNaN actual) (Double/isNaN expected))
                  (= (Double/doubleToRawLongBits expected) (Double/doubleToRawLongBits actual))))))
      (is (thrown? NullPointerException (compiled 0.5 nil)))
      (is (thrown? ClassCastException (compiled "bad" 2.0)))
      (let [source-ns (the-ns 'raster.math)
            original-class (ns-resolve source-ns 'Math)]
        (try
          (ns-unmap source-ns 'Math)
          (.importClass source-ns 'Math java.lang.StrictMath)
          (is (= :stale-source-resolution
                 (:reason (ex-data (try (compiled 0.5 2.0) nil
                                       (catch clojure.lang.ExceptionInfo error error))))))
          (is (thrown? NullPointerException (compiled 0.5 nil))
              "public conversion failure still precedes the retained source guard")
          (finally
            (ns-unmap source-ns 'Math)
            (.importClass source-ns 'Math original-class))))
      (finally
        (when (instance? java.lang.AutoCloseable compiled)
          (.close ^java.lang.AutoCloseable compiled))))))

(deftest template-fragment-is-lazy-only-for-an-absent-edge
  (let [gradient (with-meta 'gradient {:raster.type/tag 'double})
        before (assoc (context) :marker :retained)
        after (assoc before :bindings [gradient '(do (swap! calls inc) (/ dy x))])
        [lifted slots] (#'reverse/lift-pullback-over-absence before after [gradient nil] 'dy)
        project (eval (list 'fn* '[dy x calls] (list 'let* (:bindings lifted) slots)))
        calls (atom 0)]
    (is (= :retained (:marker lifted)))
    (is (nil? (second slots)) "a statically absent gradient remains absent")
    (is (= [nil nil] (project nil 0.0 calls)))
    (is (zero? @calls))
    (is (thrown? ArithmeticException (project 0.0 0.0 calls))
        "a present zero must execute the original singular arithmetic")
    (is (= 1 @calls))
    (is (= [2.0 nil] (project 4.0 2.0 calls)))
    (is (= 2 @calls))))

(deftest present-float-preserves-reference-and-signed-zero
  (let [gradient (with-meta 'gradient {:raster.type/tag 'float})
        before (context)
        [lifted slots] (#'reverse/lift-pullback-over-absence
                        before (assoc before :bindings [gradient 'dy]) [gradient] 'dy)
        project (eval (list 'fn* '[dy] (list 'let* (:bindings lifted) slots)))
        value (Float/valueOf (float -0.0))]
    (is (= [nil] (project nil)))
    (let [actual (first (project value))]
      (is (identical? value actual))
      (is (= (Float/floatToRawIntBits value) (Float/floatToRawIntBits actual))))))

(deftest existing-prefix-is-neither-guarded-nor-duplicated
  (let [gradient (with-meta 'gradient {:raster.type/tag 'double})
        before (assoc (context) :bindings ['shared '(do (swap! prefix-calls inc) 2.0)])
        after (update before :bindings into
                      [gradient '(do (swap! body-calls inc) (+ shared dy))])
        [lifted slots] (#'reverse/lift-pullback-over-absence before after [gradient] 'dy)
        project (eval (list 'fn* '[dy prefix-calls body-calls]
                            (list 'let* (:bindings lifted) slots)))
        prefix-calls (atom 0) body-calls (atom 0)]
    (is (= (:bindings before) (subvec (:bindings lifted) 0 2)))
    (is (= [nil] (project nil prefix-calls body-calls)))
    (is (= [1 0] [@prefix-calls @body-calls]))
    (is (= [5.0] (project 3.0 prefix-calls body-calls)))
    (is (= [2 1] [@prefix-calls @body-calls]))))

(deftest absent-closure-skips-both-factory-and-invocation
  (with-bindings {#'reverse/*gensym-counter* (atom 0)}
    (let [factories (atom 0) calls (atom 0)]
      ;; Exercise only the factory boundary, without changing any AD registry.
      (with-redefs-fn
        {#'templates/get-pullback-factory
         (fn [_]
           (fn [& _]
             (swap! factories inc)
             (fn [dy] (swap! calls inc) [(/ dy 0.0)])))}
        (fn []
          (let [state (#'reverse/emit-backward
                       {:type :call :sym 'value :base-op 'probe :args '[x]
                        :active-args '#{x} :resolved-template nil}
                       'dy {} {:adj-env {} :rev-ctx (context)})
                project (eval (list 'fn* '[value x dy]
                                    (list 'let* (get-in state [:rev-ctx :bindings])
                                          (first (get-in state [:adj-env 'x])))))]
            (is (nil? (project 1.0 1.0 nil)))
            (is (= [0 0] [@factories @calls]))
            (is (Double/isNaN (project 1.0 1.0 0.0)))
            (is (= [1 1] [@factories @calls]))))))))

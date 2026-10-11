(ns raster.compiler.backend.jvm.lexical-core-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.walker :as walker]))

(defn float-feedback-oracle [h0]
  ;; Ordinary Clojure independently executes the actual lexical callable.
  (let [double (fn [v] (clojure.core/float v))]
    (loop [i 0 acc h0]
      (if (< i 2)
        (recur (inc i)
               (if (= i 0) (double (+ acc 0.0))
                   (let [z (- acc 16777216.0)] (* z z))))
        acc))))

(deftm lexical-float-feedback [h0 :- Double] :- Double
  (let [double (fn [v] (clojure.core/float v))]
    (loop [i 0 acc h0]
      (if (< i 2)
        (recur (inc i)
               (if (n/== i 0) (double (n/+ acc 0.0))
                   (let [z (n/- acc 16777216.0)] (n/* z z))))
        acc))))

(deftm core-double-feedback [h0 :- Double] :- Double
  (loop [i 0 acc h0]
    (if (< i 2)
      (recur (inc i)
             (if (n/== i 0) (clojure.core/double (n/+ acc 0.0))
                 (let [z (n/- acc 16777216.0)] (n/* z z))))
      acc)))

(deftm lexical-inc-call [x :- Double] :- Double
  (let [inc (fn [v] (+ v 100.0))] (inc x)))

(deftm parameter-held-core-call [inc :- clojure.lang.IFn, x :- Double] :- Object
  (inc x))

(deftest public-lexical-call-selection-preserves-source-primal
  (let [x 16777216.5 h 0.0625]
    (doseq [value [(- x h) x (+ x h)]]
      (is (= 0.0 (float-feedback-oracle value) (lexical-float-feedback value))))
    (is (= 0.0 (/ (- (lexical-float-feedback (+ x h))
                     (lexical-float-feedback (- x h))) (* 2.0 h))))
    (is (= 0.25 (core-double-feedback x)))
    (is (= 1.0 (/ (- (core-double-feedback (+ x h))
                     (core-double-feedback (- x h))) (* 2.0 h))))
    (is (= [0.25 1.0] ((rev/value+grad #'core-double-feedback) x))))
  (is (= 103.0 (lexical-inc-call 3.0)))
  (let [result (Object.) calls (atom [])]
    (is (identical? result
                    (parameter-held-core-call (fn [x] (swap! calls conj x) result) 3.0)))
    (is (= [3.0] @calls) "boxed lexical invocation executes exactly once")))

(deftest lexical-callee-spelling-is-not-a-return-type-proof
  (let [stack-type (ns-resolve 'raster.compiler.backend.jvm.bytecode
                               'infer-arg-stack-type-structural)
        value-tag (ns-resolve 'raster.compiler.core.walker 'walked-value-tag)]
    (doseq [head ['double 'float 'long 'int '+]]
      (is (nil? (stack-type (list head 1.0) head {head {:type :ref}})))
      (is (nil? (value-tag (list head 1.0) {head {:tag nil}}))))
    (is (= :double (stack-type '(clojure.core/double 1.0) 'clojure.core/double {})))
    (is (= 'double (value-tag '(clojure.core/double 1.0) {})))))

(deftest accepted-ad-must-preserve-the-actual-lexical-primal
  (doseq [[construct invoke] [[rev/value+grad #(% 16777216.5)]
                              [jvp/jvp #(% 16777216.5 1.0)]]]
    ;; Both modes already admit this concrete fixture. Keep that capability
    ;; while demanding parity; arbitrary construction refusal is not success.
    (is (= [0.0 0.0] (invoke (construct #'lexical-float-feedback))))))

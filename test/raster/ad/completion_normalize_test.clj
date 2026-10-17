(ns raster.ad.completion-normalize-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.value-completion :as completion]))

(def policy {:unchecked-math false})

(defn- source-function [parameters source]
  (binding [*unchecked-math* false]
    (eval (list 'fn parameters (completion/project-to-jvm-source source policy)))))

(deftest completion-preserves-original-primitive-result-context
  (let [contract (completion/primitive-input-contract Float/TYPE policy)
        original (completion/complete-value contract '(Math/abs (double x)))
        [bindings normalized] (anf/anf-normalize-expr original gensym)
        source (source-function '[x] original)
        prepared (source-function '[x] (list 'let* bindings normalized))
        boxed (source-function '[x] (completion/complete-value contract 'x))]
    (is (= [] bindings))
    (is (identical? original normalized))
    (is (= Float/POSITIVE_INFINITY (source Double/MAX_VALUE) (prepared Double/MAX_VALUE)))
    (is (thrown? IllegalArgumentException (boxed Double/MAX_VALUE))
        "boxing the same mathematical value changes checked float input conversion")))

(deftest completed-first-input-still-fails-before-later-operand-effects
  (let [first-input (completion/complete-value
                     (completion/primitive-input-contract Double/TYPE policy) 'x)
        expression (list 'vector first-input '(do (swap! events conj :later) 2.0))
        [bindings result] (anf/normalize-for-ad [] [expression] gensym)
        prepared (source-function '[x events] (list 'let* bindings result))]
    (let [events (atom [])]
      (is (thrown? NullPointerException (prepared nil events)))
      (is (= [] @events)))
    (let [events (atom [])]
      (is (= [3.0 2.0] (prepared 3.0 events)))
      (is (= [:later] @events)))))

(deftest object-completion-does-not-lift-a-compound-operand
  (let [expression (completion/complete-value completion/object-input-contract
                                              '(do (swap! events conj :operand) x))
        [bindings normalized] (anf/anf-normalize-expr expression gensym)
        f (source-function '[x events] (list 'let* bindings normalized))
        input (Object.) events (atom [])]
    (is (= [] bindings))
    (is (identical? expression normalized))
    (is (identical? input (f input events)))
    (is (= [:operand] @events))))

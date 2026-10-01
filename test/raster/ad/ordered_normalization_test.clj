(ns raster.ad.ordered-normalization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as reverse]))

(deftm nested-checked-read
  [x :- Double, observations :- (Array double)] :- Double
  (let [result (let [local x] (aget observations 0) local)]
    (* result result)))

(deftest reverse-retains-unused-checked-primal-reads
  (let [vg (reverse/value+grad #'nested-checked-read :wrt [0])]
    (is (thrown? ArrayIndexOutOfBoundsException
                 (nested-checked-read 2.0 (double-array 0))))
    (is (thrown? ArrayIndexOutOfBoundsException (vg 2.0 (double-array 0))))
    (is (= [4.0 4.0 nil] (vg 2.0 (double-array [7.0]))))))

(defn- run-normalization [expression normalize?]
  (let [events (atom [])
        form (if normalize?
               (reverse/call-with-shared-ad-gensym
                 #(#'reverse/hoist-nested-lets expression))
               expression)
        f (eval `(fn [~'events] ~form))]
    {:value (f events) :events @events}))

(deftest hoisting-preserves-all-body-statements
  (let [form '(let* [result (let* [local 3]
                             (swap! events conj :first)
                             (swap! events conj :second)
                             local)] result)]
    (is (= {:value 3 :events [:first :second]}
           (run-normalization form false)
           (run-normalization form true))))
  (let [form '(let* [result (let* [local 3]
                             (throw (IllegalArgumentException. "must execute"))
                             local)] result)]
    (doseq [normalize? [false true]]
      (is (thrown-with-msg? IllegalArgumentException #"must execute"
                            (run-normalization form normalize?))))))

(deftest hoisting-keeps-sibling-argument-evaluation-order
  (doseq [form ['(vector (do (swap! events conj :first) 1)
                        (let* [local (do (swap! events conj :second) 2)] local))
               '(vector (let* [local 1] (do (swap! events conj :first) local))
                        (let* [local (do (swap! events conj :second) 2)] local))
               '(vector (let* [local 1] (swap! events conj :first) local)
                        (do (swap! events conj :second) 2))]]
    (is (= {:value [1 2] :events [:first :second]}
           (run-normalization form false)
           (run-normalization form true)) (str form))))

(deftest first-argument-terminal-traps-before-later-initializers
  (let [form '(vector
                (let* [local 1] (throw (IllegalArgumentException. "first")))
                (let* [local (throw (IllegalStateException. "second"))] local))]
    (doseq [normalize? [false true]]
      (is (thrown-with-msg? IllegalArgumentException #"first"
                            (run-normalization form normalize?))))))

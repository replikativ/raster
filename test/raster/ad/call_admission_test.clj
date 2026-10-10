(ns raster.ad.call-admission-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as r]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as templates]
            [raster.compiler.core.op-descriptor :as descriptor]))

(def ^:private helper-calls (atom 0))

(defn narrowing-helper [x]
  (swap! helper-calls inc)
  (double (float x)))

(defn identity-helper [x] x)
(defn ^{:tag 'double} inactive-rounding-helper [x] (double (float x)))
(defn ^{:tag 'double} exact-integer-helper [x] (if (= x 9007199254740993) 2.0 9.0))
(defn mutating-implementation [x] (swap! helper-calls inc) x)

(r/deftm active-narrowing [x :- Double] :- Double (narrowing-helper x))
(r/deftm active-identity [x :- Double] :- Double (identity-helper x))
(r/deftm active-alias [x :- Double] :- Double
  (let [alias x] (identity-helper alias)))
(r/deftm inactive-rounding [x :- Double count :- Long] :- Double
  (n/+ (n/* x x) (inactive-rounding-helper count)))
(r/deftm inactive-integer [x :- Double count :- Long] :- Double
  (n/+ (n/* x x) (exact-integer-helper count)))
(r/deftm shadowed-helper [x :- Double count :- Long] :- Double
  (let [x count] (exact-integer-helper x)))

(deftest call-description-keeps-semantic-and-implementation-identities-distinct
  (let [direct '(raster.numeric/* x y)
        dispatch '(.invk fake/_star__m_float_float-impl x y)
        stamped (with-meta dispatch {:raster.op/original 'raster.numeric/* :op 'wrong/legacy})
        legacy (with-meta dispatch {:op 'raster.numeric/*})]
    (is (= {:dispatch? false :semantic-op 'raster.numeric/* :recorded-op 'raster.numeric/*
            :implementation-op 'raster.numeric/* :operation 'raster.numeric/* :arguments '[x y]}
           (descriptor/call-description direct)))
    (doseq [call [stamped legacy]]
      (let [description (descriptor/call-description call)]
        (is (= 'raster.numeric/* (:recorded-op description)))
        (is (= 'raster.numeric/* (:operation description)))
        (is (= 'fake/_star__m_float_float-impl (:implementation-op description)))
        (is (= '[x y] (:arguments description)))))
    (is (nil? (:semantic-op (descriptor/call-description legacy))))
    (is (nil? (:recorded-op (descriptor/call-description dispatch))))
    (is (= 'fake/_star__m_float_float-impl (:operation (descriptor/call-description dispatch))))
    (is (nil? (descriptor/call-description 'x)))))

(deftest forward-and-reverse-rules-use-the-same-recorded-call-identity
  (doseq [call ['(raster.numeric/* x y)
                (with-meta '(.invk fake/_star__m_float_float-impl x y)
                  {:raster.op/original 'raster.numeric/* :op 'wrong/legacy})
                (with-meta '(.invk fake/_star__m_float_float-impl x y)
                  {:op 'raster.numeric/*})]]
    (let [resolved (atom [])
          record (with-redefs [templates/resolve-template (fn [operation]
                                                          (swap! resolved conj operation)
                                                          [{} operation])]
                   (#'reverse/ad-record :call 'result call {'x true 'y true}))
          forward (with-redefs [templates/op-jvp-fn
                               (fn [operation]
                                 (swap! resolved conj operation)
                                 (fn [context _args _tangents _result _gensym]
                                   [context 'directional]))]
                    (#'jvp/fold-call {'x 'dx 'y 'dy} 'result call 'float))]
      (is (= ['raster.numeric/* 'raster.numeric/*] @resolved))
      (is (= 'raster.numeric/* (get-in record [:record :base-op])))
      (is (= 'directional (get-in forward [0 'result])))
      (when (= '.invk (first call))
        (is (= ['float 'float] (mapv #(-> % meta :raster.type/tag)
                                     (get-in record [:record :args])))
            "dispatch tags come from implementation identity, not the semantic name")))))

(deftest active-plain-helpers-decline-at-construction-without-execution
  (reset! helper-calls 0)
  (doseq [v [#'active-narrowing #'active-identity #'active-alias]]
    (let [coverage (reverse/forward-coverage v)
          error (try (reverse/value+grad v :mode :forward)
                     (catch clojure.lang.ExceptionInfo e e))]
      (is (false? (:admissible? coverage)))
      (is (= :unsupported-forward-call (get-in coverage [:call-declines 0 :reason])))
      (is (= [0] (get-in coverage [:call-declines 0 :active-argument-indices])))
      (is (= (:call-declines coverage) (:call-declines (ex-data error))))))
  (is (zero? @helper-calls)))

(deftest inactive-helper-calls-preserve-primal-data-and-nil-gradient-slots
  (let [rounded (reverse/value+grad #'inactive-rounding :mode :forward)
        integer (reverse/value+grad #'inactive-integer :mode :forward)
        shadowed (reverse/value+grad #'shadowed-helper :mode :forward)]
    (is (= [(inactive-rounding 3.0 16777217) 6.0 nil] (rounded 3.0 16777217)))
    (is (= [11.0 6.0 nil] (integer 3.0 9007199254740993)))
    (is (= [2.0 0.0 nil] (shadowed 3.0 9007199254740993)))))

(deftest lexical-call-admission-covers-captures-carries-and-missing-dispatch-identity
  (let [plan @#'reverse/forward-conversion-plan
        helper 'raster.ad.call-admission-test/identity-helper]
    (doseq [expression [(list helper 'x)
                        (list 'let* ['alias 'x] (list helper 'alias))
                        (list 'loop* ['a 'x 'b 0.0 'i 0]
                              (list 'if '(clojure.core/< i 2)
                                    '(recur a a (clojure.core/inc i)) (list helper 'b)))
                        (list 'loop* ['a 'x 'b 0 'i 0]
                              (list 'if '(clojure.core/< i 2)
                                    '(recur a a (clojure.core/inc i)) (list helper 'b)))
                        (list helper (list helper 'x))
                        '(.invk opaque-call x)
                        '(let* [f (fn* [] x)] (f))]]
      (is (seq (:call-declines (plan expression '[x] '[double] *ns*))) (str expression)))
    (is (true? (get-in (plan '(let* [f (fn* [] x)] (f)) '[x] '[double] *ns*)
                      [:call-declines 0 :active-callee?])))
    (let [expression (list 'loop* ['a 'x 'b 0 'i 0]
                           (list 'if '(clojure.core/< i 2)
                                 '(recur a a (clojure.core/inc i)) (list helper 'b)))
          declines (:call-declines (plan expression '[x] '[double] *ns*))]
      (is (some #(= helper (:operation %)) declines))
      (is (not-any? #(= 'clojure.core/inc (:operation %)) declines)
          "an inactive integer counter does not acquire carry activity"))
    (doseq [argument [(with-meta '(raster.numeric/+ x x) {:raster.type/tag 'Object})
                      '(unknown-returning-operation x)]]
      (let [outer-call (list helper argument)
            declines (:call-declines (plan outer-call '[x] '[double] *ns*))]
        (is (some #(= outer-call (:form %)) declines)
            "unknown or Object result types cannot certify a carrier-free active argument")))
    (doseq [expression [(list 'let* ['x 7.0] (list helper 'x))
                        '(quote (identity-helper x))
                        '(Math/expm1 0.5)
                        '(raster.numeric/* x x)]]
      (is (empty? (:call-declines (plan expression '[x] '[double] *ns*))) (str expression)))))

(deftest replay-purity-is-not-derived-from-a-derivative-template
  (with-redefs [templates/resolve-template
                (fn [& _] (throw (AssertionError. "purity must not consult derivative rules")))]
    (is (#'jvp/pure-map-step? '(.invk raster.numeric/_star__m_double_double-impl x y)))
    (is (not (#'jvp/pure-map-step?
              (with-meta '(.invk raster.ad.call-admission-test/mutating-implementation x)
                {:raster.op/original 'raster.numeric/sqrt}))))
    (is (not (#'jvp/pure-map-step? '(.invk missing.namespace/unknown x))))))

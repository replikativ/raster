(ns raster.sym.taylor-ext-test
  "Tests for taylor.clj extensions (sigmoid, probit, log-sum-exp, taylor-log-likelihood)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sym.taylor :as taylor]))

(deftest sigmoid-taylor-test
  (testing "sigmoid Taylor at order 3"
    (let [result (taylor/sigmoid-taylor 'x 3)]
      (is (some? result))
      ;; Sigmoid(0) = 0.5, so constant term should be ~0.5
      )))

(deftest probit-taylor-test
  (testing "probit Taylor at order 3"
    (let [result (taylor/probit-taylor 'x 3)]
      (is (some? result))
      ;; Φ(0) = 0.5
      )))

(deftest log-sum-exp-taylor-test
  (testing "log-sum-exp of two variables at origin"
    (let [result (taylor/log-sum-exp-taylor '[x y] [0.0 0.0] 2)]
      (is (some? result)))))

(deftest taylor-log-likelihood-test
  (testing "Taylor expansion of sin(x) at 0 (pure observation, no params)"
    (let [model {:log-likelihood '(sin x)
                 :params '[]
                 :observations '[x]}
          result (taylor/taylor-log-likelihood model 'x 0.0 :order 5)]
      (is (some? result)))))

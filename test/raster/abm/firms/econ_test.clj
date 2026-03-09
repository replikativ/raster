(ns raster.abm.firms.econ-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.abm.firms.econ :as econ]))

(deftest test-production
  (testing "production with known values"
    ;; Y = a*E + b*E^β
    ;; a=1, b=0.5, β=2, E=2 → 1*2 + 0.5*4 = 4.0
    (is (== 4.0 (econ/production 1.0 0.5 2.0 2.0)))
    ;; a=1, b=0, β=1.5, E=3 → 3.0 (linear only)
    (is (== 3.0 (econ/production 1.0 0.0 1.5 3.0)))
    ;; a=0, b=1, β=1, E=5 → 0 + 1*5 = 5.0
    (is (== 5.0 (econ/production 0.0 1.0 1.0 5.0)))
    ;; a=0.5, b=0.5, β=1.5, E=4 → 0.5*4 + 0.5*8 = 6.0
    (is (== 6.0 (econ/production 0.5 0.5 1.5 4.0)))))

(deftest test-marginal-product
  (testing "marginal product dY/dE = a + b*β*E^(β-1)"
    ;; a=1, b=0.5, β=2, E=3 → 1 + 0.5*2*3 = 4.0
    (is (== 4.0 (econ/marginal-product 1.0 0.5 2.0 3.0)))
    ;; a=1, b=1, β=1, E=any → 1 + 1*1*E^0 = 2.0
    (is (== 2.0 (econ/marginal-product 1.0 1.0 1.0 5.0)))
    ;; β=1 (linear): MP = a + b
    (is (== 1.5 (econ/marginal-product 1.0 0.5 1.0 10.0)))))

(deftest test-log-utility
  (testing "log utility θ*ln(income) + (1-θ)*ln(endow-effort)"
    (let [theta 0.5
          income 1.0
          endowment 1.0
          effort 0.5]
      ;; 0.5*ln(1) + 0.5*ln(0.5) = 0 + 0.5*(-0.693...) ≈ -0.347
      (is (< (Math/abs (- (econ/log-utility theta income endowment effort)
                          (* 0.5 (Math/log 0.5))))
             1e-10))))
  (testing "effort at endowment boundary"
    ;; When effort = endowment, ln(endow-effort) = ln(0) = -Inf
    ;; so log-utility falls back to θ*ln(income) only
    (is (== (* 0.5 (Math/log 2.0))
            (econ/log-utility 0.5 2.0 1.0 1.0)))
    ;; Just below endowment still works normally
    (is (Double/isFinite (econ/log-utility 0.5 2.0 1.0 0.999)))))

(deftest test-wage-equal-shares
  (testing "equal shares wage"
    (is (== 5.0 (econ/wage-equal-shares 10.0 2)))
    (is (== 10.0 (econ/wage-equal-shares 10.0 1)))))

(deftest test-wage-proportional
  (testing "proportional wage"
    (is (== 5.0 (econ/wage-proportional 10.0 2.0 4.0)))
    (is (== 0.0 (econ/wage-proportional 10.0 2.0 0.0)))))

(deftest test-solve-singleton-effort
  (testing "singleton effort converges for typical params"
    (let [e (econ/solve-singleton-effort 1.0 0.5 1.5 0.5 1.0)]
      (is (> e 0.0) "effort should be positive")
      (is (< e 1.0) "effort should be less than endowment")
      ;; Verify FOC residual: dV/de ≈ 0
      ;; f(e) = θ*MP/Y - (1-θ)/(ω-e)
      (let [theta 0.5, a 1.0, b 0.5, beta 1.5, endow 1.0
            mp (econ/marginal-product a b beta e)
            output (econ/production a b beta e)
            residual (- (/ (* theta mp) output)
                        (/ (- 1.0 theta) (- endow e)))]
        (is (< (Math/abs residual) 1e-6)
            (str "FOC residual should be near zero, got " residual)))))

  (testing "singleton effort with β=1 (linear production)"
    (let [e (econ/solve-singleton-effort 1.0 0.5 1.0 0.5 1.0)]
      (is (> e 0.0))
      (is (< e 1.0))))

  (testing "singleton effort with high preference for income"
    (let [e-low  (econ/solve-singleton-effort 1.0 0.5 1.5 0.3 1.0)
          e-high (econ/solve-singleton-effort 1.0 0.5 1.5 0.7 1.0)]
      (is (> e-high e-low)
          "higher θ (income preference) should produce higher effort"))))

(deftest test-solve-effort
  (testing "team effort converges for typical params"
    ;; With small others-effort, agent has interior solution
    (let [e (econ/solve-effort 1.0 0.5 1.5 0.3 0.5 1.0)]
      (is (> e 0.01) (str "effort should be well positive, got " e))
      (is (< e 1.0) "effort should be less than endowment")))

  (testing "free-rider effect: high others-effort reduces own effort"
    ;; With equal shares, high others-effort → free-riding → low own effort
    ;; This is the classic team production free-rider result
    (let [e (econ/solve-effort 1.0 0.5 1.5 2.0 0.5 1.0)]
      (is (< e 0.1) (str "free-riding: effort should be near zero, got " e))))

  (testing "team effort with zero others matches singleton"
    (let [e-solo (econ/solve-singleton-effort 1.0 0.5 1.5 0.5 1.0)
          e-team (econ/solve-effort 1.0 0.5 1.5 0.0 0.5 1.0)]
      (is (< (Math/abs (- e-solo e-team)) 1e-6)
          (str "solo=" e-solo " team=" e-team " should be equal"))))

  (testing "edge case: β = 1.0 (linear returns)"
    (let [e (econ/solve-effort 1.0 0.5 1.0 1.0 0.5 1.0)]
      (is (>= e 0.0))
      (is (<= e 1.0))))

  (testing "effort satisfies FOC at interior solution"
    ;; Use params where interior solution exists (small others-effort)
    (let [a 1.0, b 0.5, beta 1.5, theta 0.5, endow 1.0
          others 0.3
          e (econ/solve-effort a b beta others theta endow)
          E (+ e others)
          mp (econ/marginal-product a b beta E)
          Y (econ/production a b beta E)
          residual (- (/ (* theta mp) Y)
                      (/ (- 1.0 theta) (- endow e)))]
      (is (< (Math/abs residual) 1e-6)
          (str "FOC residual should be near zero, got " residual))))

  (testing "free-rider effect: less effort with more others"
    ;; With equal shares, higher others-effort → more free-riding → lower own effort
    (let [e-low-others  (econ/solve-effort 1.0 0.5 1.5 0.1 0.5 1.0)
          e-high-others (econ/solve-effort 1.0 0.5 1.5 1.0 0.5 1.0)]
      (is (> e-low-others e-high-others)
          (str "free-riding: less others → more own effort: "
               e-low-others " vs " e-high-others)))))

(deftest test-prospective-log-utility
  (testing "prospective utility computation"
    (let [u (econ/prospective-log-utility 1.0 0.5 1.5 0.3 1.0 3 0.5 1.0)]
      (is (not (Double/isNaN u)))
      (is (Double/isFinite u)))))

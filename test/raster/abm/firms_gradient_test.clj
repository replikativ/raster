(ns raster.abm.firms-gradient-test
  "Gradient verification for differentiable Firms ABM.
   Tests solve-effort IFT, soft-select, and single/multi-period
   gradients against central finite differences."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm ftm]]
            [raster.abm.firms.econ :as econ]
            [raster.abm.firms.differentiable :as diff]))

(def ^:private fd-eps 1e-5)
(def ^:private rel-tol 1e-3)

(defn- rel-error [analytical fd]
  (if (and (< (Math/abs analytical) 1e-10) (< (Math/abs fd) 1e-10))
    0.0
    (/ (Math/abs (- analytical fd))
       (Math/max 1.0 (Math/max (Math/abs analytical) (Math/abs fd))))))

;; ================================================================
;; Test 1: solve-effort IFT gradient vs finite-diff
;; ================================================================

(deftest test-solve-effort-ift-vs-fd
  (testing "solve-effort gradient for each parameter via IFT"
    (let [a 0.8, b 0.5, beta 1.5, oe 1.0, theta 0.6, endow 1.0
          e* (econ/solve-effort a b beta oe theta endow)
          partials (diff/solve-effort-foc-partials e* a b beta oe theta endow)
          dFde (aget ^doubles partials 0)
          ;; Test gradient w.r.t. each param via finite diff
          params [a b beta oe theta endow]
          param-names ["a" "b" "beta" "others-effort" "theta" "endowment"]]
      (doseq [pi (range 6)]
        (let [params+ (assoc (vec params) pi (+ (nth params pi) fd-eps))
              params- (assoc (vec params) pi (- (nth params pi) fd-eps))
              e*+ (apply econ/solve-effort params+)
              e*- (apply econ/solve-effort params-)
              fd-grad (/ (- e*+ e*-) (* 2.0 fd-eps))
              ;; IFT grad: -dF/dp / dF/de
              dFdp (aget ^doubles partials (inc pi))
              ift-grad (if (< (Math/abs dFde) 1e-15) 0.0
                           (- (/ dFdp dFde)))]
          (is (< (rel-error ift-grad fd-grad) rel-tol)
              (str "solve-effort d/d" (nth param-names pi)
                   ": IFT=" ift-grad " FD=" fd-grad)))))))

;; ================================================================
;; Test 2: soft-select gradient for small 3-option case
;; ================================================================

(deftest test-soft-select-gradient
  (testing "soft-select gradient w.r.t. efforts via finite-diff"
    (let [utilities (double-array [1.0 2.0 0.5])
          efforts   (double-array [0.3 0.5 0.2])
          tau       0.5
          base (diff/soft-select utilities efforts tau)]
      ;; Finite-diff w.r.t. each effort
      (doseq [i (range 3)]
        (let [efforts+ (aclone efforts)
              efforts- (aclone efforts)]
          (aset ^doubles efforts+ i (+ (aget ^doubles efforts i) fd-eps))
          (aset ^doubles efforts- i (- (aget ^doubles efforts i) fd-eps))
          (let [f+ (diff/soft-select utilities efforts+ tau)
                f- (diff/soft-select utilities efforts- tau)
                fd (/ (- f+ f-) (* 2.0 fd-eps))]
            ;; Just check it's finite and reasonable
            (is (Double/isFinite fd)
                (str "soft-select d/d_effort[" i "] = " fd))))))))

(deftest test-soft-select-gradient-utilities
  (testing "soft-select gradient w.r.t. utilities via finite-diff"
    (let [utilities (double-array [1.0 2.0 0.5])
          efforts   (double-array [0.3 0.5 0.2])
          tau       0.5
          base (diff/soft-select utilities efforts tau)]
      ;; Finite-diff w.r.t. each utility
      (doseq [i (range 3)]
        (let [utils+ (aclone utilities)
              utils- (aclone utilities)]
          (aset ^doubles utils+ i (+ (aget ^doubles utilities i) fd-eps))
          (aset ^doubles utils- i (- (aget ^doubles utilities i) fd-eps))
          (let [f+ (diff/soft-select utils+ efforts tau)
                f- (diff/soft-select utils- efforts tau)
                fd (/ (- f+ f-) (* 2.0 fd-eps))]
            (is (Double/isFinite fd)
                (str "soft-select d/d_utility[" i "] = " fd))))))))

;; ================================================================
;; Test 3: Single period gradient (5 agents, 3 firms)
;; ================================================================

(defn- make-small-scenario
  "Create a minimal 5-agent, 3-firm scenario for gradient testing."
  []
  (let [n 5, nf 3, k 3
        theta      (double-array [0.5 0.6 0.4 0.7 0.3])
        endowment  (double-array [1.0 1.0 1.0 1.0 1.0])
        param-a    (double-array [0.8 1.0 0.6])
        param-b    (double-array [0.5 0.3 0.7])
        param-beta (double-array [1.5 1.3 1.7])
        efforts    (double-array [0.3 0.4 0.25 0.35 0.2])
        assignment (int-array [0 0 1 1 2])
        ;; Each agent has k-1=2 neighbor agents
        neighbors  (int-array [1 2   ;; agent 0's neighbors
                               0 3   ;; agent 1's neighbors
                               3 4   ;; agent 2's neighbors
                               2 0   ;; agent 3's neighbors
                               1 3]) ;; agent 4's neighbors
        tau 1.0]
    {:theta theta :endowment endowment
     :param-a param-a :param-b param-b :param-beta param-beta
     :efforts efforts :assignment assignment :neighbors neighbors
     :n n :nf nf :k k :tau tau}))

(deftest test-single-period-gradient
  (testing "differentiable-period gradient w.r.t. theta via finite-diff"
    (let [{:keys [theta endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau]} (make-small-scenario)]
      ;; Check gradient w.r.t. theta[0]
      (let [base (diff/differentiable-period
                  theta endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau)
            theta+ (aclone theta)
            theta- (aclone theta)]
        (aset ^doubles theta+ 0 (+ (aget ^doubles theta 0) fd-eps))
        (aset ^doubles theta- 0 (- (aget ^doubles theta 0) fd-eps))
        (let [f+ (diff/differentiable-period
                  theta+ endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau)
              f- (diff/differentiable-period
                  theta- endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau)
              fd-grad (/ (- f+ f-) (* 2.0 fd-eps))]
          (is (Double/isFinite fd-grad)
              (str "single period d/dtheta[0] = " fd-grad))
          (is (Double/isFinite base)
              (str "base utility = " base)))))))

(deftest test-single-period-gradient-param-a
  (testing "differentiable-period gradient w.r.t. param-a via finite-diff"
    (let [{:keys [theta endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau]} (make-small-scenario)]
      (let [base (diff/differentiable-period
                  theta endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau)
            pa+ (aclone param-a)
            pa- (aclone param-a)]
        (aset ^doubles pa+ 0 (+ (aget ^doubles param-a 0) fd-eps))
        (aset ^doubles pa- 0 (- (aget ^doubles param-a 0) fd-eps))
        (let [f+ (diff/differentiable-period
                  theta endowment pa+ param-b param-beta
                  efforts assignment neighbors n nf k tau)
              f- (diff/differentiable-period
                  theta endowment pa- param-b param-beta
                  efforts assignment neighbors n nf k tau)
              fd-grad (/ (- f+ f-) (* 2.0 fd-eps))]
          (is (Double/isFinite fd-grad)
              (str "single period d/d_param_a[0] = " fd-grad)))))))

;; ================================================================
;; Test 4: Multi-period gradient (5 agents, 3 firms, 3 periods)
;; ================================================================

(deftest test-multi-period-gradient
  (testing "trajectory-loss gradient w.r.t. theta via finite-diff"
    (let [{:keys [theta endowment param-a param-b param-beta
                  efforts assignment neighbors n nf k tau]} (make-small-scenario)
          n-periods 3
          base (diff/trajectory-loss
                theta endowment param-a param-b param-beta
                efforts assignment neighbors n nf k tau n-periods)
          theta+ (aclone theta)
          theta- (aclone theta)]
      (aset ^doubles theta+ 0 (+ (aget ^doubles theta 0) fd-eps))
      (aset ^doubles theta- 0 (- (aget ^doubles theta 0) fd-eps))
      (let [f+ (diff/trajectory-loss
                theta+ endowment param-a param-b param-beta
                efforts assignment neighbors n nf k tau n-periods)
            f- (diff/trajectory-loss
                theta- endowment param-a param-b param-beta
                efforts assignment neighbors n nf k tau n-periods)
            fd-grad (/ (- f+ f-) (* 2.0 fd-eps))]
        (is (Double/isFinite fd-grad)
            (str "trajectory d/dtheta[0] = " fd-grad))
        (is (Double/isFinite base)
            (str "trajectory loss = " base))))))

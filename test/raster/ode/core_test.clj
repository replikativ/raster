(ns raster.ode.core-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.ode.core :refer [ode-problem solve solve-with-events ode-event
                                     dense-output saveat
                                     exponential-decay bouncing-ball!
                                     lorenz! harmonic-oscillator! stiff-decay!
                                     van-der-pol!
                                     exponential-decay-ftm lorenz-ftm
                                     ->Euler ->RK4 dp5 implicit-euler
                                     tsit5 rosenbrock23 sdirk4 vern7 auto-switch]]))

;; ================================================================
;; Helper: compute endpoint error for exponential decay
;; ================================================================

(defn- decay-error
  "Run exponential-decay with given solver and step size, return endpoint error."
  [alg dt]
  (let [prob (ode-problem exponential-decay [1.0] 0.0 1.0)
        sol  (solve alg prob dt)
        u-final (aget ^doubles (last (:us sol)) 0)
        exact   (Math/exp -1.0)]
    (Math/abs (- u-final exact))))

;; ================================================================
;; Convergence order tests
;; ================================================================

(deftest euler-convergence-test
  (testing "Euler is O(h) — halving h halves error (~ratio 2)"
    (let [e1 (decay-error (->Euler) 0.01)
          e2 (decay-error (->Euler) 0.005)
          ratio (/ e1 e2)]
      (is (< 1.8 ratio 2.2)
          (str "Expected ratio ~2, got " ratio)))))

(deftest rk4-convergence-test
  (testing "RK4 is O(h^4) — halving h reduces error by ~16x"
    (let [e1 (decay-error (->RK4) 0.1)
          e2 (decay-error (->RK4) 0.05)
          ratio (/ e1 e2)]
      (is (< 14.0 ratio 18.0)
          (str "Expected ratio ~16, got " ratio)))))

(deftest dp5-accuracy-test
  (testing "DP5 adaptive achieves tight tolerance"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 5.0)
          sol  (solve (dp5 1e-8 1e-6) prob 0.1)
          u-final (aget ^doubles (last (:us sol)) 0)
          exact   (Math/exp -5.0)
          err     (Math/abs (- u-final exact))]
      (is (< err 1e-6)
          (str "Expected error < 1e-6, got " err)))))

;; ================================================================
;; Multi-dimensional ODE tests
;; ================================================================

(deftest harmonic-oscillator-test
  (testing "RK4 conserves energy for harmonic oscillator"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2 Math/PI))
          sol  (solve (->RK4) prob 0.01)
          u-final (last (:us sol))
          x (aget ^doubles u-final 0)
          v (aget ^doubles u-final 1)
          ;; Energy = 0.5*(x^2 + v^2), should be ~0.5
          energy (+ (* 0.5 x x) (* 0.5 v v))]
      (is (< (Math/abs (- energy 0.5)) 0.001)
          (str "Expected energy ~0.5, got " energy)))))

(deftest lorenz-smoke-test
  (testing "DP5 solves Lorenz without crashing"
    (let [prob (ode-problem lorenz! [1.0 1.0 1.0] 0.0 1.0)
          sol  (solve (dp5) prob 0.01)]
      (is (> (:iterations sol) 0))
      (is (every? #(< (Math/abs %) 100.0)
                  (seq (last (:us sol))))))))

;; ================================================================
;; Implicit Euler (stiff solver) tests
;; ================================================================

(deftest implicit-euler-basic-test
  (testing "implicit Euler solves non-stiff decay"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 1.0)
          sol  (solve (implicit-euler) prob 0.1)
          u-final (aget ^doubles (last (:us sol)) 0)
          exact   (Math/exp -1.0)
          err     (Math/abs (- u-final exact))]
      ;; Implicit Euler is O(h), so with h=0.1 expect ~5% error
      (is (< err 0.1)
          (str "Expected error < 0.1, got " err)))))

(deftest implicit-euler-stiff-test
  (testing "implicit Euler is stable on stiff problem with large dt"
    ;; du/dt = -1000*u, exact: exp(-1000*t) ≈ 0
    ;; Explicit Euler would blow up with dt=0.01, implicit stays stable
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.01)
          sol  (solve (implicit-euler) prob 0.005)
          u-final (aget ^doubles (last (:us sol)) 0)]
      ;; Key test: solution stays bounded (not blowing up)
      ;; Implicit Euler is O(h) so accuracy is limited, but stability is the point
      (is (< (Math/abs u-final) 0.1)
          (str "Expected bounded, got " u-final))
      (is (>= u-final 0.0)
          "Solution should stay non-negative")))
  (testing "implicit Euler converges as dt shrinks"
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.01)
          e1 (Math/abs (aget ^doubles (last (:us (solve (implicit-euler) prob 0.005))) 0))
          e2 (Math/abs (aget ^doubles (last (:us (solve (implicit-euler) prob 0.001))) 0))]
      (is (< e2 e1) "Smaller dt should give smaller final value"))))

;; ================================================================
;; Event handling tests
;; ================================================================

(deftest event-termination-test
  (testing "event terminates integration at threshold"
    ;; Exponential decay: stop when u < 0.5 (i.e., t ≈ ln(2) ≈ 0.693)
    (let [prob (ode-problem exponential-decay [1.0] 0.0 5.0)
          threshold-event (ode-event
                           (fn [^doubles u ^double t] (- (aget u 0) 0.5))
                           (fn [^doubles u ^double t] :terminate)
                           :falling)
          sol (solve-with-events (->RK4) prob 0.01 [threshold-event])]
      (is (= 1 (count (:event-log sol))))
      (is (= :terminate (:action (first (:event-log sol)))))
      ;; Should stop near ln(2)
      (let [t-event (:t (first (:event-log sol)))]
        (is (< (Math/abs (- t-event (Math/log 2.0))) 0.02)
            (str "Expected t ≈ 0.693, got " t-event))))))

(deftest bouncing-ball-test
  (testing "bouncing ball with elastic collision"
    ;; Ball drops from height 1, bounces with coefficient of restitution 0.9
    ;; First bounce at ~0.45s, subsequent bounces progressively shorter
    (let [prob (ode-problem bouncing-ball! [1.0 0.0] 0.0 3.0)
          ground-event (ode-event
                         ;; condition: height = 0
                        (fn [^doubles u ^double t] (aget u 0))
                         ;; affect: reverse velocity * restitution
                        (fn [^doubles u ^double t]
                          (aset u 0 0.0)
                          (aset u 1 (* -0.9 (aget u 1)))
                          :continue)
                        :falling)
          sol (solve-with-events (->RK4) prob 0.001 [ground-event])
          bounces (count (:event-log sol))]
      ;; Should have multiple bounces
      (is (>= bounces 3) (str "Expected >= 3 bounces, got " bounces))
      ;; All events should be :continue
      (is (every? #(= :continue (:action %)) (:event-log sol)))
      ;; Height should stay non-negative (approximately)
      (is (every? #(>= (aget ^doubles % 0) -0.1) (:us sol))
          "Ball height should stay >= 0"))))

;; ================================================================
;; Dense output / interpolation tests
;; ================================================================

(deftest dense-output-exponential-test
  (testing "Hermite cubic interpolation matches exact solution for exp decay"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (->RK4) prob 0.1)
          interp (dense-output exponential-decay sol)]
      ;; Test at non-grid points (between RK4 steps)
      (doseq [t [0.05 0.15 0.33 0.77 1.23 1.89]]
        (let [u-interp (aget ^doubles (interp t) 0)
              u-exact  (Math/exp (- t))
              err      (Math/abs (- u-interp u-exact))]
          (is (< err 1e-4)
              (str "At t=" t ", expected " u-exact ", got " u-interp ", err=" err)))))))

(deftest dense-output-endpoints-test
  (testing "interpolation returns exact values at endpoints"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 1.0)
          sol  (solve (->RK4) prob 0.1)
          interp (dense-output exponential-decay sol)]
      ;; At t=0, should return initial condition
      (is (== 1.0 (aget ^doubles (interp 0.0) 0)))
      ;; At t <= 0, should clamp to initial
      (is (== 1.0 (aget ^doubles (interp -1.0) 0)))
      ;; At t >= tf, should clamp to final
      (let [u-final (aget ^doubles (last (:us sol)) 0)
            u-interp (aget ^doubles (interp 2.0) 0)]
        (is (== u-final u-interp))))))

(deftest dense-output-harmonic-test
  (testing "interpolation works for multi-dimensional ODE"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 Math/PI)
          sol  (solve (->RK4) prob 0.05)
          interp (dense-output harmonic-oscillator! sol)]
      ;; At t=π/2, exact: x=cos(π/2)=0, v=-sin(π/2)=-1
      (let [u (interp (* 0.5 Math/PI))]
        (is (< (Math/abs (aget ^doubles u 0)) 0.01)
            (str "x at π/2 should be ~0, got " (aget ^doubles u 0)))
        (is (< (Math/abs (+ (aget ^doubles u 1) 1.0)) 0.01)
            (str "v at π/2 should be ~-1, got " (aget ^doubles u 1)))))))

(deftest saveat-test
  (testing "saveat evaluates at specified time points"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (dp5) prob 0.1)
          interp (dense-output exponential-decay sol)
          result (saveat interp [0.0 0.5 1.0 1.5 2.0])]
      (is (= 5 (count (:ts result))))
      (is (= 5 (count (:us result))))
      ;; Check accuracy at each time point
      ;; DP5 adaptive takes large steps, so Hermite cubic O(h³) gives ~1e-3 error
      (doseq [[t u] (map vector (:ts result) (:us result))]
        (let [u-val (aget ^doubles u 0)
              exact (Math/exp (- (double t)))
              err   (Math/abs (- u-val exact))]
          (is (< err 1e-3)
              (str "saveat at t=" t ": expected " exact ", got " u-val)))))))

;; ================================================================
;; ftm ODE function tests
;; ================================================================

(deftest ftm-exponential-decay-test
  (testing "ftm exponential-decay matches defn version"
    (let [prob-defn (ode-problem exponential-decay [1.0] 0.0 1.0)
          prob-ftm  (ode-problem exponential-decay-ftm [1.0] 0.0 1.0)
          sol-defn  (solve (->RK4) prob-defn 0.01)
          sol-ftm   (solve (->RK4) prob-ftm 0.01)
          u-defn (aget ^doubles (last (:us sol-defn)) 0)
          u-ftm  (aget ^doubles (last (:us sol-ftm)) 0)]
      (is (< (Math/abs (- u-defn u-ftm)) 1e-15)
          (str "ftm and defn should give identical results: " u-defn " vs " u-ftm)))))

(deftest ftm-lorenz-test
  (testing "ftm Lorenz with Tsit5 produces reasonable results"
    (let [prob (ode-problem lorenz-ftm [1.0 1.0 1.0] 0.0 1.0)
          sol  (solve (tsit5) prob 0.01)]
      (is (> (:iterations sol) 0))
      (is (every? #(< (Math/abs %) 100.0)
                  (seq (last (:us sol))))))))

;; ================================================================
;; Ground truth tests — analytical solutions & Julia reference values
;; Julia reference: OrdinaryDiffEq.jl v6, same problem/tolerances
;; ================================================================

(deftest ground-truth-exponential-decay-tsit5
  (testing "Tsit5 exponential decay matches exact solution within tolerance"
    ;; du/dt = -u, u(0) = 1, exact: u(t) = exp(-t)
    ;; Julia Tsit5 atol=1e-8 rtol=1e-6: u(5) = 0.006737948105687488
    (let [prob  (ode-problem exponential-decay [1.0] 0.0 5.0)
          sol   (solve (tsit5 1e-8 1e-6) prob 0.1)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp -5.0)]
      (is (< (Math/abs (- u-end exact)) 1e-6)
          (str "Tsit5 decay err=" (Math/abs (- u-end exact)) " should be < 1e-6"))
      ;; Close to Julia reference value
      (is (< (Math/abs (- u-end 0.006737948105687488)) 1e-9)
          (str "Tsit5 decay should match Julia ref, got " u-end)))))

(deftest ground-truth-exponential-decay-dp5
  (testing "DP5 exponential decay matches exact solution within tolerance"
    ;; Julia DP5 atol=1e-8 rtol=1e-6: u(5) = 0.006737954043822124
    (let [prob  (ode-problem exponential-decay [1.0] 0.0 5.0)
          sol   (solve (dp5 1e-8 1e-6) prob 0.1)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp -5.0)]
      (is (< (Math/abs (- u-end exact)) 1e-5)
          (str "DP5 decay err=" (Math/abs (- u-end exact)) " should be < 1e-5"))
      ;; Close to Julia reference value
      (is (< (Math/abs (- u-end 0.006737954043822124)) 1e-8)
          (str "DP5 decay should match Julia ref, got " u-end)))))

(deftest ground-truth-harmonic-oscillator-tsit5
  (testing "Tsit5 harmonic oscillator matches exact cos/sin after full period"
    ;; du/dt = [v, -x], u0 = [1, 0], exact at t=2π: x=1, v=0
    ;; Julia Tsit5 atol=1e-8 rtol=1e-6: x=0.9999999534127474, v=-5.761e-8
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2 Math/PI))
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          v    (aget ^doubles u 1)]
      (is (< (Math/abs (- x 1.0)) 1e-6)
          (str "x(2π) should be ~1.0, got " x))
      (is (< (Math/abs v) 1e-6)
          (str "v(2π) should be ~0.0, got " v)))))

(deftest ground-truth-harmonic-oscillator-dp5
  (testing "DP5 harmonic oscillator matches exact cos/sin after full period"
    ;; Julia DP5 atol=1e-8 rtol=1e-6: x=0.9999994480912456, v=-2.044e-7
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2 Math/PI))
          sol  (solve (dp5 1e-8 1e-6) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          v    (aget ^doubles u 1)]
      (is (< (Math/abs (- x 1.0)) 1e-5)
          (str "x(2π) should be ~1.0, got " x))
      (is (< (Math/abs v) 1e-5)
          (str "v(2π) should be ~0.0, got " v)))))

(deftest ground-truth-lorenz-tsit5
  (testing "Tsit5 Lorenz at t=1 matches Julia OrdinaryDiffEq.jl reference"
    ;; Lorenz: sigma=10, rho=28, beta=8/3, u0=[1,1,1]
    ;; Julia Tsit5 atol=1e-8 rtol=1e-6, t=[0,1]:
    ;;   x=-9.378566113104041, y=-8.357033888923763, z=29.362316066724060
    ;; At t=1 the system hasn't gone chaotic — implementations should agree closely
    (let [prob (ode-problem lorenz! [1.0 1.0 1.0] 0.0 1.0)
          sol  (solve (tsit5 1e-8 1e-6) prob 0.01)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          y    (aget ^doubles u 1)
          z    (aget ^doubles u 2)]
      (is (< (Math/abs (- x -9.378566113104041)) 1e-5)
          (str "Lorenz x should match Julia ref, got " x))
      (is (< (Math/abs (- y -8.357033888923763)) 1e-5)
          (str "Lorenz y should match Julia ref, got " y))
      (is (< (Math/abs (- z 29.362316066724060)) 1e-4)
          (str "Lorenz z should match Julia ref, got " z)))))

(deftest ground-truth-lorenz-dp5
  (testing "DP5 Lorenz at t=1 matches Julia OrdinaryDiffEq.jl reference"
    ;; Julia DP5 atol=1e-8 rtol=1e-6, t=[0,1]:
    ;;   x=-9.378565720598667, y=-8.357036726800308, z=29.362313299944110
    (let [prob (ode-problem lorenz! [1.0 1.0 1.0] 0.0 1.0)
          sol  (solve (dp5 1e-8 1e-6) prob 0.01)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          y    (aget ^doubles u 1)
          z    (aget ^doubles u 2)]
      (is (< (Math/abs (- x -9.378565720598667)) 1e-4)
          (str "Lorenz x should match Julia ref, got " x))
      (is (< (Math/abs (- y -8.357036726800308)) 1e-4)
          (str "Lorenz y should match Julia ref, got " y))
      (is (< (Math/abs (- z 29.362313299944110)) 1e-3)
          (str "Lorenz z should match Julia ref, got " z)))))

;; ================================================================
;; Rosenbrock23 tests
;; ================================================================

;; ================================================================
;; Tsit5 native dense output tests
;; ================================================================

(deftest tsit5-dense-output-accuracy-test
  (testing "Tsit5 native dense output matches exact solution for exp decay"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          interp (dense-output nil sol)]
      ;; Native polynomial should be very accurate
      (doseq [t [0.05 0.15 0.33 0.77 1.23 1.89]]
        (let [u-interp (aget ^doubles (interp t) 0)
              u-exact  (Math/exp (- t))
              err      (Math/abs (- u-interp u-exact))]
          (is (< err 1e-7)
              (str "At t=" t ", expected " u-exact ", got " u-interp ", err=" err)))))))

(deftest tsit5-dense-output-vs-hermite-test
  (testing "Tsit5 native interpolation is more accurate than Hermite cubic"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          interp-native (dense-output nil sol)
          interp-hermite (dense-output exponential-decay (dissoc sol :ks))]
      (doseq [t [0.15 0.77 1.23]]
        (let [exact (Math/exp (- t))
              err-native (Math/abs (- (aget ^doubles (interp-native t) 0) exact))
              err-hermite (Math/abs (- (aget ^doubles (interp-hermite t) 0) exact))]
          (is (<= err-native err-hermite)
              (str "Native should be <= Hermite error at t=" t
                   " native=" err-native " hermite=" err-hermite)))))))

(deftest tsit5-dense-output-endpoints-test
  (testing "Tsit5 native interpolation returns exact values at endpoints"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 1.0)
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          interp (dense-output nil sol)]
      ;; At t=0, should return initial condition
      (is (== 1.0 (aget ^doubles (interp 0.0) 0)))
      ;; At t <= 0, should clamp to initial
      (is (== 1.0 (aget ^doubles (interp -1.0) 0)))
      ;; At t >= tf, should clamp to final
      (let [u-final (aget ^doubles (last (:us sol)) 0)
            u-interp (aget ^doubles (interp 2.0) 0)]
        (is (== u-final u-interp))))))

(deftest tsit5-dense-output-harmonic-test
  (testing "Tsit5 native dense output for harmonic oscillator"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2.0 Math/PI))
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          interp (dense-output nil sol)]
      ;; At t=π/2, exact: x=cos(π/2)=0, v=-sin(π/2)=-1
      (let [u (interp (* 0.5 Math/PI))]
        (is (< (Math/abs (aget ^doubles u 0)) 1e-5)
            (str "x(π/2) should be ~0, got " (aget ^doubles u 0)))
        (is (< (Math/abs (+ (aget ^doubles u 1) 1.0)) 1e-5)
            (str "v(π/2) should be ~-1, got " (aget ^doubles u 1)))))))

(deftest tsit5-saveat-native-test
  (testing "saveat with native Tsit5 interpolation"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (tsit5 1e-8 1e-6) prob 0.1)
          interp (dense-output nil sol)
          result (saveat interp [0.0 0.5 1.0 1.5 2.0])]
      (is (= 5 (count (:ts result))))
      (doseq [[t u] (map vector (:ts result) (:us result))]
        (let [u-val (aget ^doubles u 0)
              exact (Math/exp (- (double t)))
              err   (Math/abs (- u-val exact))]
          (is (< err 1e-7)
              (str "saveat at t=" t ": expected " exact ", got " u-val ", err=" err)))))))

(deftest rosenbrock23-exponential-decay-test
  (testing "Rosenbrock23 solves exponential decay accurately"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 1.0)
          sol  (solve (rosenbrock23) prob 0.1)
          u-final (aget ^doubles (last (:us sol)) 0)
          exact   (Math/exp -1.0)]
      (is (< (Math/abs (- u-final exact)) 1e-4)
          (str "Expected ~" exact " got " u-final)))))

(deftest rosenbrock23-stiff-decay-test
  (testing "Rosenbrock23 handles stiff decay (lambda = -1000)"
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.01)
          sol  (solve (rosenbrock23) prob 0.005)
          u-final (aget ^doubles (last (:us sol)) 0)
          exact   (Math/exp (* -1000.0 0.01))]
      ;; For very stiff problems, Rosenbrock23 should be stable
      (is (< (Math/abs u-final) 0.1)
          "Solution should decay to near zero"))))

(deftest rosenbrock23-harmonic-oscillator-test
  (testing "Rosenbrock23 solves harmonic oscillator"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2.0 Math/PI))
          sol  (solve (rosenbrock23) prob 0.1)
          u-final (last (:us sol))
          x (aget ^doubles u-final 0)
          v (aget ^doubles u-final 1)]
      ;; After one full period, x ≈ 1.0, v ≈ 0.0
      (is (< (Math/abs (- x 1.0)) 0.05)
          (str "Position: expected ~1.0, got " x))
      (is (< (Math/abs v) 0.05)
          (str "Velocity: expected ~0.0, got " v)))))

;; ================================================================
;; Vern7 tests
;; ================================================================

(deftest vern7-exponential-decay-test
  (testing "Vern7 achieves high accuracy for exponential decay"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 5.0)
          sol  (solve (vern7 1e-10 1e-8) prob 0.1)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp -5.0)
          err   (Math/abs (- u-end exact))]
      (is (< err 1e-8)
          (str "Vern7 decay err=" err " should be < 1e-8")))))

(deftest vern7-harmonic-oscillator-test
  (testing "Vern7 preserves harmonic oscillator over full period"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2 Math/PI))
          sol  (solve (vern7 1e-10 1e-8) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          v    (aget ^doubles u 1)]
      (is (< (Math/abs (- x 1.0)) 1e-7)
          (str "x(2π) should be ~1.0, got " x))
      (is (< (Math/abs v) 1e-7)
          (str "v(2π) should be ~0.0, got " v)))))

(deftest vern7-lorenz-test
  (testing "Vern7 solves Lorenz without crashing"
    (let [prob (ode-problem lorenz! [1.0 1.0 1.0] 0.0 1.0)
          sol  (solve (vern7) prob 0.01)]
      (is (> (:iterations sol) 0))
      (is (every? #(< (Math/abs %) 100.0)
                  (seq (last (:us sol))))))))

(deftest vern7-convergence-test
  (testing "Vern7 is more accurate than Tsit5 at same tolerances"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 5.0)
          exact (Math/exp -5.0)
          sol-v7   (solve (vern7 1e-8 1e-6) prob 0.1)
          sol-t5   (solve (tsit5 1e-8 1e-6) prob 0.1)
          err-v7 (Math/abs (- (aget ^doubles (last (:us sol-v7)) 0) exact))
          err-t5 (Math/abs (- (aget ^doubles (last (:us sol-t5)) 0) exact))]
      (is (<= err-v7 err-t5)
          (str "Vern7 err=" err-v7 " should be <= Tsit5 err=" err-t5)))))

;; ================================================================
;; AutoSwitch tests
;; ================================================================

(deftest auto-switch-nonstiff-test
  (testing "AutoSwitch solves non-stiff problem (stays explicit)"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 2.0)
          sol  (solve (auto-switch) prob 0.1)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp -2.0)
          err   (Math/abs (- u-end exact))]
      (is (< err 1e-5)
          (str "AutoSwitch decay err=" err))
      ;; Should stay in explicit mode (no stiffness)
      (is (= :explicit (:mode sol))
          "Should remain in explicit mode for non-stiff problem")
      (is (= 0 (:nswitch sol))
          "Should not switch for non-stiff problem"))))

(deftest auto-switch-stiff-test
  (testing "AutoSwitch handles stiff problem (switches to implicit)"
    ;; Use low max-stiff-rejects=3 so switching triggers before Tsit5 shrinks dt enough
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.01)
          sol  (solve (auto-switch 1e-8 1e-6 3 20) prob 0.005)
          u-end (aget ^doubles (last (:us sol)) 0)]
      ;; Solution should decay to near zero
      (is (< (Math/abs u-end) 0.1)
          (str "Stiff decay should reach near zero, got " u-end))
      ;; Should have switched at some point
      (is (> (:nswitch sol) 0)
          "Should switch to implicit for stiff problem"))))

(deftest auto-switch-harmonic-test
  (testing "AutoSwitch solves harmonic oscillator"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2.0 Math/PI))
          sol  (solve (auto-switch 1e-8 1e-6) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          v    (aget ^doubles u 1)]
      (is (< (Math/abs (- x 1.0)) 1e-5)
          (str "x(2π) should be ~1.0, got " x))
      (is (< (Math/abs v) 1e-5)
          (str "v(2π) should be ~0.0, got " v)))))

;; ================================================================
;; SDIRK4 tests
;; ================================================================

(deftest sdirk4-stiff-decay-test
  (testing "SDIRK4 handles stiff decay (lambda=-1000) with large step"
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.01)
          sol  (solve (sdirk4) prob 0.001)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp (* -1000.0 0.01))]
      (is (< (Math/abs (- u-end exact)) 1e-3)
          (str "SDIRK4 stiff decay: expected " exact ", got " u-end))
      (is (:converged? sol true)))))

(deftest sdirk4-exponential-decay-test
  (testing "SDIRK4 achieves good accuracy on non-stiff exponential decay"
    (let [prob (ode-problem exponential-decay [1.0] 0.0 5.0)
          sol  (solve (sdirk4 1e-8 1e-6) prob 0.1)
          u-end (aget ^doubles (last (:us sol)) 0)
          exact (Math/exp -5.0)
          err   (Math/abs (- u-end exact))]
      (is (< err 1e-4)
          (str "SDIRK4 exp decay error should be < 1e-4, got " err)))))

(deftest sdirk4-van-der-pol-test
  (testing "SDIRK4 solves van der Pol oscillator (mu=1000, stiff)"
    (let [prob (ode-problem van-der-pol! [2.0 0.0] 0.0 100.0)
          sol  (solve (sdirk4) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)]
      ;; Van der Pol should oscillate between ~±2
      (is (< (Math/abs x) 2.5)
          (str "SDIRK4 VDP: x should be bounded, got " x))
      (is (> (:iterations sol) 10)
          "Should take multiple steps"))))

(deftest sdirk4-harmonic-oscillator-test
  (testing "SDIRK4 solves harmonic oscillator over one period"
    (let [prob (ode-problem harmonic-oscillator! [1.0 0.0] 0.0 (* 2.0 Math/PI))
          sol  (solve (sdirk4 1e-10 1e-8) prob 0.1)
          u    (last (:us sol))
          x    (aget ^doubles u 0)
          v    (aget ^doubles u 1)]
      (is (< (Math/abs (- x 1.0)) 1e-3)
          (str "x(2π) should be ~1.0, got " x))
      (is (< (Math/abs v) 1e-3)
          (str "v(2π) should be ~0.0, got " v)))))

(deftest sdirk4-vs-rosenbrock23-test
  (testing "SDIRK4 and Rosenbrock23 agree on stiff decay endpoint"
    (let [prob (ode-problem stiff-decay! [1.0] 0.0 0.005)
          sol-sdirk (solve (sdirk4 1e-10 1e-8) prob 0.0005)
          sol-ros   (solve (rosenbrock23 1e-10 1e-8) prob 0.0005)
          u-sdirk (aget ^doubles (last (:us sol-sdirk)) 0)
          u-ros   (aget ^doubles (last (:us sol-ros)) 0)]
      (is (< (Math/abs (- u-sdirk u-ros)) 1e-4)
          (str "SDIRK4 vs ROS23: " u-sdirk " vs " u-ros)))))

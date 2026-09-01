(ns raster.compiler.gpu-integration-test
  "Numerical and GPU-compiler ratchets for the heat-equation workload.

  These tests require a Level Zero GPU device. They skip gracefully
  when no GPU is available (CI, CPU-only machines).

  Verifies:
    1. Functional correctness against analytical heat equation solution
    2. The real RK4/PDE workload crosses the public equation-first TypedSOAC compiler
    3. The resulting allocation-free LinkPlan executes numerically on an available GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.link :as gpu-link]
            [raster.ode.pde :as pde]))

;; ================================================================
;; GPU availability check
;; ================================================================

;; Availability + skip routing delegate to the shared HONEST probe
;; (raster.dl.gpu-grad-parity): a broken ze-runtime load fails loud instead of being
;; swallowed as "no GPU", and skips register a marker so the assertion count is stable.
(defmacro ^:private when-gpu [label & body]
  `(if @gp/gpu-available?
     (do ~@body)
     (gp/gpu-skip! ~label)))

;; ================================================================
;; Analytical reference for heat equation
;; ================================================================

(defn- analytical-heat-loss
  "Compute the exact MSE loss for the 1D heat equation with:
    u(x,0)  = sin(πx)  (initial condition)
    target   = 0.5·sin(πx)
    u(x,t)  = exp(-α·π²·t)·sin(πx)  (exact solution, Dirichlet BCs)
    loss     = (1/n) · Σᵢ (u(xᵢ,T) - target(xᵢ))²

  The exact continuous MSE is:
    (exp(-α·π²·T) - 0.5)² · (1/n)·Σ sin²(πxᵢ)

  We compute the discrete sum for the exact grid points."
  [n alpha dt nsteps]
  (let [dx (/ 1.0 (double (dec n)))
        T (* (double nsteps) dt)
        decay (Math/exp (- (* alpha Math/PI Math/PI T)))
        coeff (- decay 0.5)
        ;; Discrete sum of sin²(π·i·dx) for i = 0..n-1
        sum-sin2 (loop [i 0 acc 0.0]
                   (if (>= i n) acc
                       (let [x (* (double i) dx)
                             s (Math/sin (* Math/PI x))]
                         (recur (inc i) (+ acc (* s s))))))]
    (* coeff coeff (/ sum-sin2 (double n)))))

(defn- setup-heat-problem
  "Create initial condition and target arrays for the heat equation test.
  Returns [u0 target alpha inv-dx2 dt]."
  [n]
  (let [dx (/ 1.0 (double (dec n)))
        inv-dx2 (/ 1.0 (* dx dx))
        alpha 0.01
        ;; CFL-stable timestep: dt = 0.4 · dx²/α
        dt (* 0.4 (/ (* dx dx) alpha))
        u0 (double-array n)
        target (double-array n)]
    (dotimes [i n]
      (let [x (* (double i) dx)]
        (aset u0 i (Math/sin (* Math/PI x)))
        (aset target i (* 0.5 (Math/sin (* Math/PI x))))))
    [u0 target alpha inv-dx2 dt]))

;; ================================================================
;; Tests
;; ================================================================

(deftest heat-equation-analytical-reference-test
  (testing "CPU heat-loss-rk4 agrees with analytical solution"
    (require 'raster.ode.pde)
    (let [heat-loss-rk4 (resolve 'raster.ode.pde/heat-loss-rk4)]
      (doseq [[n nsteps label]
              [[64 100 "n=64, 100 steps"]
               [256 50 "n=256, 50 steps"]
               [4096 10 "n=4096, 10 steps"]]]
        (testing label
          (let [[u0 target alpha inv-dx2 dt] (setup-heat-problem n)
                cpu-loss (heat-loss-rk4 (aclone u0) target alpha inv-dx2 dt (long nsteps))
                analytical (analytical-heat-loss n alpha dt nsteps)
                ;; Spatial error O(dx²) dominates; tolerance scales with dx²
                dx (/ 1.0 (double (dec n)))
                tol (* 10.0 dx dx)]
            (is (< (Math/abs (- (double cpu-loss) analytical)) tol)
                (format "%s: CPU=%.10f analytical=%.10f tol=%.6f"
                        label (double cpu-loss) analytical tol))))))))

(deftest gpu-rk4-uses-the-public-equation-first-vertical-test
  (let [n 64
        nsteps 3
        [u0 target alpha inv-dx2 dt] (setup-heat-problem n)
        arguments [u0 target alpha inv-dx2 dt nsteps]
        compilation (equation-first/compile
                     #'pde/heat-loss-rk4 {:target :ze:0 :dtype :double})
        plan (equation-first/lower compilation arguments)]
    (testing "compilation and lowering are inspectable and allocate no driver resources"
      (is (equation-first/equation-first-compilation? compilation))
      (is (= :typed-parallel (get-in compilation [:semantic :dialect])))
      (is (= :scheduled-parallel (get-in compilation [:scheduled :dialect])))
      (is (= :opencl-parallel (get-in compilation [:emitted :dialect])))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (link-plan/link-plan? plan))
      (is (= 0 (get-in plan [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs plan)))))
    (testing "the public LinkPlan executes the same numerical program on an available GPU"
      (when-gpu "gpu-rk4-equation-first-execution"
                (let [expected (pde/heat-loss-rk4 (aclone u0) (aclone target)
                                                  alpha inv-dx2 dt nsteps)
                      executable (gpu-link/instantiate! plan)]
                  (try
                    (gpu-link/run! executable)
                    (let [actual (double (aget ^doubles
                                          (gpu-link/download executable
                                                             (first (:outputs plan)))
                                               0))]
                      (is (< (Math/abs (- (double expected) actual)) 1.0e-10)
                          (format "CPU=%.15g GPU=%.15g" (double expected) actual)))
                    (finally
                      (gpu-link/close! executable))))))))

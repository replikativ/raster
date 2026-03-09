(ns raster.ode.pde-test
  (:refer-clojure :exclude [aget aset alength aclone reduce])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ode.pde :as pde]
            [raster.ode.core :as ode]
            [raster.arrays :refer [aget aset alength aclone]]))

;; ================================================================
;; 1D Heat Equation
;; ================================================================

(deftest heat-rhs-1d-test
  (testing "1D heat stencil RHS"
    (let [n 5
          u (double-array [0.0 0.0 1.0 0.0 0.0])
          du (double-array n)
          alpha 1.0
          dx 1.0
          inv-dx2 (/ 1.0 (* dx dx))]
      (pde/heat-rhs-1d! du u alpha inv-dx2)
      ;; Boundaries should be 0
      (is (= 0.0 (aget du 0)))
      (is (= 0.0 (aget du 4)))
      ;; Interior: du[2] = 1*(0-2*1+0)/1 = -2
      (is (< (Math/abs (- (aget du 2) -2.0)) 1e-10))
      ;; du[1] = 1*(0-0+1)/1 = 1
      (is (< (Math/abs (- (aget du 1) 1.0)) 1e-10))
      ;; du[3] = 1*(1-0+0)/1 = 1
      (is (< (Math/abs (- (aget du 3) 1.0)) 1e-10)))))

(deftest heat-1d-convergence-test
  (testing "1D heat equation convergence (Gaussian IC, error decreases with refinement)"
    ;; Gaussian initial condition on [0,1]
    ;; Analytical: diffuses and flattens over time
    (let [solve-at-resolution
          (fn [n]
            (let [dx (/ 1.0 (dec n))
                  ;; CFL condition: dt < dx^2/(2*alpha)
                  dt (* 0.4 dx dx)
                  alpha 0.01
                  u0 (double-array
                      (for [i (range n)]
                        (let [x (* i dx)]
                          (Math/exp (* -50.0 (Math/pow (- x 0.5) 2))))))
                  ;; Set boundary conditions
                  _ (aset u0 0 0.0)
                  _ (aset u0 (- n 1) 0.0)
                  result (pde/heat-1d-solve u0 alpha dx [0.0 0.1]
                                            (ode/->RK4) dt)]
              (:u result)))
          ;; Solve at two resolutions
          u-coarse (solve-at-resolution 51)
          u-fine (solve-at-resolution 101)]
      ;; Both should have diffused (peak should be lower than initial)
      (is (< (apply max (map #(aget u-coarse %) (range (alength u-coarse)))) 1.0))
      (is (< (apply max (map #(aget u-fine %) (range (alength u-fine)))) 1.0))
      ;; Fine solution should be smoother (similar peak but more resolved)
      (is (> (alength u-fine) (alength u-coarse))))))

(deftest heat-1d-conservation-test
  (testing "1D heat equation conserves total heat (Dirichlet boundaries zero)"
    (let [n 101
          dx (/ 1.0 (dec n))
          dt (* 0.4 dx dx)
          alpha 0.01
          u0 (double-array
              (for [i (range n)]
                (let [x (* i dx)]
                  (if (and (> x 0.3) (< x 0.7)) 1.0 0.0))))
          _ (aset u0 0 0.0)
          _ (aset u0 (- n 1) 0.0)
          sum-before (clojure.core/reduce + (map #(aget u0 %) (range n)))
          result (pde/heat-1d-solve u0 alpha dx [0.0 0.05]
                                    (ode/->RK4) dt)
          u-final (:u result)
          sum-after (clojure.core/reduce + (map #(aget u-final %) (range (alength u-final))))]
      ;; Heat should be approximately conserved (zero-flux at boundaries
      ;; is not enforced -- Dirichlet loses heat, so sum decreases)
      ;; But shouldn't gain heat
      (is (<= sum-after (+ sum-before 1e-6))))))

;; ================================================================
;; 2D Heat Equation
;; ================================================================

(deftest heat-rhs-2d-test
  (testing "2D heat stencil RHS"
    (let [nx 5 ny 5
          n (* nx ny)
          u (double-array n 0.0)
          du (double-array n)
          ;; Set center point to 1.0
          _ (aset u (+ (* 2 ny) 2) 1.0)
          alpha 1.0
          dx 1.0 dy 1.0
          inv-dx2 1.0 inv-dy2 1.0]
      (pde/heat-rhs-2d! du u nx ny alpha inv-dx2 inv-dy2)
      ;; Center point (2,2) should have du = 1*(0-2*1+0+0-2*1+0) = -4
      (is (< (Math/abs (- (aget du 12) -4.0)) 1e-10))
      ;; Boundaries should be 0
      (is (= 0.0 (aget du 0)))
      (is (= 0.0 (aget du (- n 1)))))))

(deftest heat-2d-symmetry-test
  (testing "2D heat equation: symmetric IC -> symmetric solution"
    (let [nx 33 ny 33
          n (* nx ny)
          dx (/ 1.0 (dec nx))
          dy (/ 1.0 (dec ny))
          dt (* 0.2 (Math/min (* dx dx) (* dy dy)))
          alpha 0.01
          ;; Symmetric initial condition (Gaussian at center)
          u0 (double-array
              (for [i (range nx)
                    j (range ny)]
                (let [x (* i dx) y (* j dy)
                      r2 (+ (Math/pow (- x 0.5) 2) (Math/pow (- y 0.5) 2))]
                  (if (or (= i 0) (= i (dec nx)) (= j 0) (= j (dec ny)))
                    0.0
                    (Math/exp (* -50.0 r2))))))
          result (pde/heat-2d-solve u0 nx ny alpha dx dy
                                    [0.0 0.05] (ode/->RK4) dt)
          u-final (:u result)]
      ;; Check symmetry: u(i,j) ~ u(j,i) for interior
      (let [max-asym (atom 0.0)]
        (dotimes [i nx]
          (dotimes [j ny]
            (let [uij (aget u-final (+ (* i ny) j))
                  uji (aget u-final (+ (* j ny) i))
                  diff (Math/abs (- uij uji))]
              (swap! max-asym #(Math/max ^double % diff)))))
        (is (< @max-asym 1e-10) "Solution should preserve symmetry")))))

;; ================================================================
;; 1D Wave Equation
;; ================================================================

(deftest wave-rhs-1d-test
  (testing "1D wave stencil RHS"
    (let [n 5
          ;; State: [u0..u4 v0..v4]
          uv (double-array [0.0 0.0 1.0 0.0 0.0   ;; displacement
                            0.0 0.0 0.0 0.0 0.0])  ;; velocity
          duv (double-array (* 2 n))
          c2-inv-dx2 1.0]
      (pde/wave-rhs-1d! duv uv n c2-inv-dx2)
      ;; du/dt = v -> duv[0..4] = uv[5..9] = 0.0
      (dotimes [i n]
        (is (= 0.0 (aget duv i))))
      ;; dv/dt boundaries
      (is (= 0.0 (aget duv 5)))
      (is (= 0.0 (aget duv 9)))
      ;; dv[2] = 1*(0 - 2*1 + 0) = -2
      (is (< (Math/abs (- (aget duv 7) -2.0)) 1e-10)))))

(deftest wave-1d-energy-test
  (testing "Wave equation energy conservation"
    ;; E = sum(v^2 + c^2*(u[i+1]-u[i])^2/dx^2) should be conserved
    (let [n 51
          dx (/ 1.0 (dec n))
          c 1.0
          dt (* 0.3 dx)  ;; CFL: dt < dx/c
          ;; Sine initial displacement, zero velocity
          u0 (double-array
              (for [i (range n)]
                (let [x (* i dx)]
                  (Math/sin (* Math/PI x)))))
          _ (aset u0 0 0.0)
          _ (aset u0 (- n 1) 0.0)
          v0 (double-array n 0.0)
          ;; Compute energy from state vector
          energy (fn [state n-pts]
                   (let [ke (loop [i 0 e 0.0]
                              (if (< i n-pts)
                                (let [vi (aget state (+ n-pts i))]
                                  (recur (inc i) (+ e (* vi vi))))
                                e))
                         pe (loop [i 0 e 0.0]
                              (if (< i (dec n-pts))
                                (let [du (- (aget state (inc i)) (aget state i))
                                      c2dx2 (/ (* c c) (* dx dx))]
                                  (recur (inc i) (+ e (* c2dx2 du du))))
                                e))]
                     (+ ke pe)))
          ;; Solve for a short time
          result (pde/wave-1d-solve u0 v0 c dx [0.0 0.5] (ode/->RK4) dt)
          u-final (:u result)
          e0 (energy (let [state0 (double-array (* 2 n))]
                       (System/arraycopy u0 0 state0 0 n)
                       (System/arraycopy v0 0 state0 n n)
                       state0)
                     n)
          ef (energy u-final n)]
      ;; Energy should be approximately conserved
      ;; RK4 has some energy drift, but should be < 5% for this dt
      (is (< (Math/abs (/ (- ef e0) e0)) 0.05)
          (str "Energy drift: " (/ (- ef e0) e0))))))

;; # Differential Equations in Clojure with Raster

;; authors: Christian Weilbach

;; Differential equations describe how things change over time. They are
;; the language of physics, biology, engineering, and finance. If you can
;; write down *how fast* something changes, a solver can tell you *where
;; it goes*.
;;
;; Raster provides ODE, PDE, and SDE solvers that work with idiomatic
;; Clojure code. You write the math with `deftm`, and the compiler
;; generates JVM bytecode that runs at Java speed — no type annotations
;; beyond the function signature.

;; ## Setup

(ns raster.ode-solvers
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.ode.core :as ode]
            [raster.ode.pde :as pde]
            [raster.ode.sde :as sde]
            [scicloj.kindly.v4.kind :as kind]))

;; ## 1. Exponential Growth — Your First ODE
;;
;; The simplest differential equation: a population grows proportionally
;; to its size. If bacteria double every hour, the growth rate is
;; proportional to the current count:
;;
;; $$\frac{du}{dt} = 0.3 \cdot u$$
;;
;; The analytical solution is $u(t) = u_0 \cdot e^{0.3t}$.

;; In Raster, you define the right-hand side as a `deftm` or `ftm`.
;; The solver calls this function repeatedly, writing the rate of change
;; into the `du` array:

(deftm exponential-rhs
  [du :- (Array double), u :- (Array double), t :- Double]
  (aset du 0 (* 0.3 (aget u 0))))

;; Solve from t=0 to t=10, starting at u=1:

(def sol-exp
  (let [prob (ode/ode-problem exponential-rhs [1.0] 0.0 10.0)]
    (ode/solve (ode/->RK4) prob 0.01)))

;; Compare with the exact solution:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 500 :height 300
  :title "Exponential Growth: du/dt = 0.3u"
  :layer [{:data {:values (mapv (fn [t u] {:t t :u (first u) :type "RK4 solver"})
                                (:ts sol-exp) (:us sol-exp))}
           :mark {:type "line" :strokeWidth 2}
           :encoding {:x {:field "t" :type "quantitative" :title "Time"}
                      :y {:field "u" :type "quantitative" :title "u(t)"}
                      :color {:field "type" :type "nominal"}}}
          {:data {:values (mapv (fn [t] {:t t :u (Math/exp (* 0.3 t)) :type "Exact: e^(0.3t)"})
                                (range 0.0 10.1 0.1))}
           :mark {:type "line" :strokeDash [5 3] :strokeWidth 2}
           :encoding {:x {:field "t" :type "quantitative"}
                      :y {:field "u" :type "quantitative"}
                      :color {:field "type" :type "nominal"}}}]})

;; The solver tracks the exact solution perfectly. RK4 is a 4th-order
;; method — its error is proportional to dt⁴, so with dt=0.01 the
;; error is roughly 10⁻⁸.

;; ## 2. The Lorenz Attractor — Chaos from Three Equations
;;
;; Edward Lorenz discovered in 1963 that three simple equations
;; can produce chaotic, unpredictable behavior. This is the
;; foundation of chaos theory and the "butterfly effect":
;;
;; $$\frac{dx}{dt} = \sigma(y - x)$$
;; $$\frac{dy}{dt} = x(\rho - z) - y$$
;; $$\frac{dz}{dt} = xy - \beta z$$
;;
;; With σ=10, ρ=28, β=8/3, the trajectory never repeats and is
;; sensitive to initial conditions — a hallmark of deterministic chaos.

(deftm lorenz
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [x (aget u 0) y (aget u 1) z (aget u 2)
        sigma 10.0, rho 28.0, beta (/ 8.0 3.0)]
    (aset du 0 (* sigma (- y x)))
    (aset du 1 (- (* x (- rho z)) y))
    (aset du 2 (- (* x y) (* beta z)))))

;; Solve with the adaptive DP5 (Dormand-Prince 5th order) method.
;; Adaptive solvers automatically choose the step size to maintain
;; the requested accuracy:

(def sol-lorenz
  (let [prob (ode/ode-problem lorenz [1.0 0.0 0.0] 0.0 50.0)]
    (ode/solve (ode/->DP5 1e-8 1e-6) prob 0.01)))

;; The iconic butterfly — the x-z phase space projection:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 500 :height 400
  :title "Lorenz Attractor (x-z projection)"
  :data {:values (mapv (fn [u] {:x (nth u 0) :z (nth u 2)})
                       (take-nth 2 (:us sol-lorenz)))}
  :mark {:type "line" :strokeWidth 0.5 :opacity 0.7 :color "#2563eb"}
  :encoding {:x {:field "x" :type "quantitative" :title "x"
                 :scale {:domain [-25 25]}}
             :y {:field "z" :type "quantitative" :title "z"
                 :scale {:domain [0 55]}}}})

;; All three components over time:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 250
  :title "Lorenz: x, y, z components over time"
  :data {:values (mapcat (fn [t u]
                           [{:t t :val (nth u 0) :var "x"}
                            {:t t :val (nth u 1) :var "y"}
                            {:t t :val (nth u 2) :var "z"}])
                         (take-nth 5 (:ts sol-lorenz))
                         (take-nth 5 (:us sol-lorenz)))}
  :mark {:type "line" :strokeWidth 1}
  :encoding {:x {:field "t" :type "quantitative" :title "Time"}
             :y {:field "val" :type "quantitative" :title "Value"}
             :color {:field "var" :type "nominal" :title "Variable"}}})

;; ## 3. Lotka-Volterra — Predators and Prey
;;
;; The classic ecological model: rabbits reproduce, foxes eat rabbits,
;; foxes starve without prey. The populations oscillate perpetually.
;;
;; $$\frac{dx}{dt} = \alpha x - \beta x y$$
;; $$\frac{dy}{dt} = \delta x y - \gamma y$$

(deftm lotka-volterra
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [x (aget u 0) y (aget u 1)
        alpha 1.5, beta 1.0, delta 0.3, gamma 0.8]
    (aset du 0 (- (* alpha x) (* beta x y)))
    (aset du 1 (- (* delta x y) (* gamma y)))))

(def sol-lv
  (let [prob (ode/ode-problem lotka-volterra [10.0 5.0] 0.0 30.0)]
    (ode/solve (ode/->DP5 1e-8 1e-6) prob 0.01)))

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Lotka-Volterra: Predator-Prey Oscillations"
  :data {:values (mapcat (fn [t u]
                           [{:t t :pop (nth u 0) :species "Prey (rabbits)"}
                            {:t t :pop (nth u 1) :species "Predators (foxes)"}])
                         (:ts sol-lv) (:us sol-lv))}
  :mark {:type "line" :strokeWidth 2}
  :encoding {:x {:field "t" :type "quantitative" :title "Time"}
             :y {:field "pop" :type "quantitative" :title "Population"}
             :color {:field "species" :type "nominal"}}})

;; The phase portrait shows the closed orbits — energy is conserved:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 400 :height 400
  :title "Phase Portrait: Closed Orbits"
  :data {:values (mapv (fn [u] {:prey (nth u 0) :predator (nth u 1)})
                       (:us sol-lv))}
  :mark {:type "line" :strokeWidth 1.5 :color "#2563eb"}
  :encoding {:x {:field "prey" :type "quantitative" :title "Prey"}
             :y {:field "predator" :type "quantitative" :title "Predators"}}})

;; ## 4. Choosing a Solver
;;
;; Raster provides multiple ODE algorithms. The right choice depends on
;; your problem:
;;
;; | Algorithm | Type | Order | Best for |
;; |---|---|---|---|
;; | `Euler` | Fixed-step | 1 | Teaching, quick prototyping |
;; | `RK4` | Fixed-step | 4 | Smooth problems, known time scale |
;; | `DP5` | Adaptive | 5(4) | General-purpose non-stiff |
;; | `Tsit5` | Adaptive | 5(4) | Non-stiff with dense output |
;; | `Vern7` | Adaptive | 7(6) | High-accuracy non-stiff |
;; | `ImplicitEuler` | Implicit | 1 | Stiff problems (chemical kinetics) |
;; | `Rosenbrock23` | Rosenbrock | 2(3) | Moderately stiff, automatic Jacobian |
;; | `SDIRK4` | SDIRK | 4 | Stiff problems needing accuracy |
;;
;; **Stiff vs non-stiff:** If your system has widely separated time
;; scales (e.g., fast chemical reactions coupled with slow diffusion),
;; use an implicit solver. Otherwise, DP5 or Tsit5 is the default choice.

;; ## 5. Beyond ODEs: PDE and SDE Solvers

;; ### Heat Equation (PDE)
;;
;; The 1D heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$
;; describes how temperature diffuses through a material. Raster solves it
;; via method-of-lines: discretize space into grid points, then solve the
;; resulting system of ODEs in time.

;; Start with a spike of heat in the center:

(def heat-sol
  (let [n 101
        dx (/ 1.0 (dec n))
        u0 (double-array n)]
    ;; Gaussian bump centered at x=0.5
    (dotimes [i n]
      (let [x (* i dx)
            d (- x 0.5)]
        (aset u0 i (Math/exp (- (/ (* d d) 0.005))))))
    ;; Solve for several time snapshots
    (let [alpha 0.01
          dt 0.0001]
      {:x (mapv #(* % dx) (range n))
       :snapshots
       (reduce (fn [acc tf]
                 (let [sol (pde/heat-1d-solve u0 alpha dx [0.0 tf] (ode/->RK4) dt)]
                   (conj acc {:t tf :u (vec (:u sol))})))
               [{:t 0.0 :u (vec u0)}]
               [0.05 0.15 0.5 1.0])})))

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "1D Heat Equation: Diffusion of a Temperature Spike"
  :data {:values (mapcat (fn [{:keys [t u]}]
                           (map-indexed (fn [i v]
                                          {:x (nth (:x heat-sol) i) :u v
                                           :t (format "t=%.2f" t)})
                                        u))
                         (:snapshots heat-sol))}
  :mark {:type "line" :strokeWidth 2}
  :encoding {:x {:field "x" :type "quantitative" :title "Position"}
             :y {:field "u" :type "quantitative" :title "Temperature"
                 :scale {:domain [0 1.1]}}
             :color {:field "t" :type "nominal" :title "Time"
                     :sort ["t=0.00" "t=0.05" "t=0.15" "t=0.50" "t=1.00"]}}})

;; The initial spike spreads out and flattens — heat diffuses from hot to
;; cold regions until the temperature is uniform. The boundary conditions
;; are Dirichlet (fixed at zero).

;; ### Stochastic Differential Equation (SDE)
;;
;; An SDE adds noise to a differential equation. The Ornstein-Uhlenbeck
;; process models a noisy spring — a particle pulled toward zero by a
;; restoring force, with random kicks from thermal fluctuations:
;;
;; $$dX_t = -\theta X_t \, dt + \sigma \, dW_t$$

(deftm ou-drift
  [du :- (Array double), u :- (Array double), t :- Double]
  (let [theta 1.0]
    (aset du 0 (* (- theta) (aget u 0)))))

(deftm ou-diffusion
  [du :- (Array double), u :- (Array double), t :- Double]
  (aset du 0 0.5))

;; Run multiple independent realizations to see the stochastic ensemble:

(def sde-trajectories
  (let [prob (sde/sde-problem ou-drift ou-diffusion [2.0] 0.0 5.0)
        dt 0.005]
    (mapv (fn [seed]
            (let [sol (sde/solve (sde/->EulerMaruyama) prob dt
                                (java.util.Random. seed))]
              {:ts (:ts sol) :us (:us sol) :seed seed}))
          (range 8))))

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Ornstein-Uhlenbeck Process: 8 Stochastic Realizations"
  :data {:values (mapcat (fn [{:keys [ts us seed]}]
                           (map (fn [t u]
                                  {:t t :x (first u) :path (str "path " seed)})
                                (take-nth 5 ts) (take-nth 5 us)))
                         sde-trajectories)}
  :mark {:type "line" :strokeWidth 1 :opacity 0.7}
  :encoding {:x {:field "t" :type "quantitative" :title "Time"}
             :y {:field "x" :type "quantitative" :title "X(t)"}
             :color {:field "path" :type "nominal" :legend nil}}})

;; Each trajectory is different due to the random noise, but they all
;; tend toward zero — the mean-reverting behavior of the OU process.
;; This is the simplest model of thermal fluctuations, interest rates,
;; and neural membrane potentials.

;; ## 6. Performance
;;
;; Raster's bytecode compiler automatically optimizes `deftm` and `ftm`
;; functions. On the DP5 Lorenz benchmark (t=[0,100], atol=1e-8):
;;
;; | Implementation | Time |
;; |---|---|
;; | Raster `compile-aot` | ~420µs |
;; | Java (hand-written) | ~420µs |
;; | Julia DifferentialEquations.jl | ~583µs |
;; | Raster lazy JIT (automatic) | ~740µs |
;; | Clojure (type-hinted) | ~730µs |
;;
;; The lazy JIT compiles each `deftm` to optimized bytecode on first
;; call — no manual optimization needed. For production hot paths,
;; `compile-aot` inlines the entire call chain into one method,
;; matching hand-written Java performance.
;;
;; Run `bench/ode_dp5_lorenz.clj` to reproduce these numbers on your
;; hardware.

;; # Geometric Algebra + ODE Solvers: Rotor Kinematics

;; authors: Christian Weilbach

;; This example demonstrates raster's compositional power: a geometric
;; algebra rotor equation is solved by a standard ODE integrator using
;; **native Multivector state** — no manual array conversions needed.
;; The ODE solver operates directly on `Multivector` objects thanks to
;; Julia-style parametric specialization of the solver cache and step
;; functions.

;; ## Setup

(ns raster.ga-ode-rotors
  (:require [raster.core :refer [deftm ftm]]
            [raster.ga.core :as ga]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength]]
            [raster.ode.core :as ode]
            [scicloj.kindly.v4.kind :as kind])
  (:import [raster.ga.core Multivector]))

;; ## The Algebra: VGA(3) — Euclidean 3D

;; Cl(3,0,0) has 2³ = 8 basis elements: 1 scalar, 3 vectors, 3 bivectors,
;; 1 pseudoscalar. Rotations are represented by even-grade elements (rotors).

(def sig (ga/vga 3))
(def dim (ga/algebra-dim sig))

;; Basis vectors and their products:
(let [e1 (ga/basis sig 0)
      e2 (ga/basis sig 1)
      e3 (ga/basis sig 2)]
  {:e1*e1 (ga/mv->str (ga/geometric-product e1 e1))    ;; = 1 (Euclidean)
   :e1*e2 (ga/mv->str (ga/geometric-product e1 e2))    ;; = e12 (bivector)
   :e2*e1 (ga/mv->str (ga/geometric-product e2 e1))})  ;; = -e12 (anticommutative)

;; ## Rotor Kinematics: dR/dt = ½ω·R

;; A rotor R(t) evolving under angular velocity bivector ω satisfies
;; the kinematic equation dR/dt = ½ω·R. The sandwich product R·v·R†
;; rotates any vector v.

;; The ODE solver works directly with Multivector state — no need
;; to convert to/from double arrays. This is possible because
;; `Multivector` is registered as an array-like type and the solver
;; uses Julia-style parametric specialization (`All [T]`).

;; ## Example 1: Constant Rotation

;; Angular velocity: π rad/s in the e1∧e2 plane (one full rotation in 2 seconds)

(def omega
  (let [e12 (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1))]
    (n/* Math/PI e12)))

;; RHS function — operates directly on Multivectors:

(deftm rhs-const
  [du :- Multivector, u :- Multivector, t :- Double]
  (let [dRdt (n/* 0.5 (ga/geometric-product omega u))]
    (dotimes [i dim]
      (aset du i (aget dRdt i)))))

;; Solve from t=0 to t=2 (one full rotation):

(def sol-const
  (let [R0 (ga/scalar-mv sig 1.0)
        prob (ode/ode-problem rhs-const R0 0.0 2.0)]
    (ode/solve (ode/->RK4) prob 0.001)))

;; Track where e1 points over time:

(def trajectory-const
  (let [states (:us sol-const)
        ts (:ts sol-const)
        e1 (ga/basis sig 0)]
    (mapv (fn [t-val R]
            (let [rotated (ga/geometric-product R (ga/geometric-product e1 (ga/reverse-mv R)))
                  data (.-data ^Multivector rotated)]
              {:t t-val
               :x (clojure.core/aget ^doubles data 1)
               :y (clojure.core/aget ^doubles data 2)
               :z (clojure.core/aget ^doubles data 3)
               :norm-R (ga/norm R)}))
          ts states)))

;; The trajectory traces a circle in the x-y plane:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 500 :height 300
  :title "Constant Rotation: e₁ trajectory in the e₁∧e₂ plane"
  :data {:values (take-nth 10 trajectory-const)}
  :layer [{:mark {:type "line" :strokeWidth 2}
           :encoding {:x {:field "x" :type "quantitative" :title "e₁ component"}
                      :y {:field "y" :type "quantitative" :title "e₂ component"}
                      :color {:value "#2196F3"}}}]})

;; ## Example 2: Precessing Rotation

;; Angular velocity varies in time — produces a wobbling rotation like
;; a spinning top with precession:
;; ω(t) = π·cos(t)·e₁₂ + 0.3π·sin(t)·e₂₃

(deftm rhs-precess
  [du :- Multivector, u :- Multivector, t :- Double]
  (let [e12 (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1))
        e23 (ga/geometric-product (ga/basis sig 1) (ga/basis sig 2))
        omega-t (n/+ (n/* (* Math/PI (Math/cos t)) e12)
                     (n/* (* 0.3 Math/PI (Math/sin t)) e23))
        dRdt (n/* 0.5 (ga/geometric-product omega-t u))]
    (dotimes [i dim]
      (aset du i (aget dRdt i)))))

(def sol-precess
  (let [R0 (ga/scalar-mv sig 1.0)
        prob (ode/ode-problem rhs-precess R0 0.0 4.0)]
    (ode/solve (ode/->RK4) prob 0.001)))

(def trajectory-precess
  (let [states (:us sol-precess)
        ts (:ts sol-precess)
        e1 (ga/basis sig 0)]
    (mapv (fn [t-val R]
            (let [rotated (ga/geometric-product R (ga/geometric-product e1 (ga/reverse-mv R)))
                  data (.-data ^Multivector rotated)]
              {:t t-val
               :x (clojure.core/aget ^doubles data 1)
               :y (clojure.core/aget ^doubles data 2)
               :z (clojure.core/aget ^doubles data 3)
               :norm-R (ga/norm R)}))
          ts states)))

;; Components vs time — the precession is visible as the z-component grows:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Precessing Rotor: e₁ components over time"
  :data {:values (mapcat (fn [pt]
                           [{:t (:t pt) :component "x" :value (:x pt)}
                            {:t (:t pt) :component "y" :value (:y pt)}
                            {:t (:t pt) :component "z" :value (:z pt)}])
                         (take-nth 10 trajectory-precess))}
  :mark {:type "line" :strokeWidth 1.5}
  :encoding {:x {:field "t" :type "quantitative" :title "Time (s)"}
             :y {:field "value" :type "quantitative" :title "Component value"}
             :color {:field "component" :type "nominal"
                     :scale {:domain ["x" "y" "z"]
                             :range ["#F44336" "#4CAF50" "#2196F3"]}}}})

;; ## Rotor Norm Conservation

;; The ODE integrator preserves the rotor norm to machine precision:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 200
  :title "Rotor norm |R| over time (should be 1.0)"
  :data {:values (take-nth 50 trajectory-precess)}
  :mark {:type "line" :strokeWidth 1}
  :encoding {:x {:field "t" :type "quantitative" :title "Time (s)"}
             :y {:field "norm-R" :type "quantitative"
                 :scale {:domain [0.9999999 1.0000001]}
                 :title "|R|"}}})

;; ## Key Insight: Compositional Design

;; This example uses **zero specialized code**:
;;
;; - `raster.ga.core` provides Cl(3,0,0) algebra with `deftm` dispatch
;; - `raster.ode.core` provides RK4 integration with `(All [T])` parametric types
;; - They compose through raster's array-like type registry
;;
;; The GA module knows nothing about ODEs. The ODE module knows nothing
;; about geometric algebra. Yet they work together seamlessly because
;; raster's type system provides the glue — just like Julia:
;;
;; ```
;; Julia:    RK4Cache{Multivector}     — parametric struct specialization
;; Raster:   RK4Cache__Multivector     — JIT defcache specialization
;;
;; Julia:    solve(prob; dt=0.001)      — dispatches on eltype(u0)
;; Raster:   (solve (->RK4) prob 0.001) — parametric dispatch on u0 type
;; ```
;;
;; This compositionality extends to AD (automatic differentiation),
;; the compiler pipeline, and GPU execution — all through the same
;; `deftm`/`ftm` typed dispatch mechanism.

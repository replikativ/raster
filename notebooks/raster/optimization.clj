;; # Optimization and Root Finding with Raster

;; authors: Christian Weilbach

;; Raster provides numerical optimization and root-finding algorithms
;; that work with typed functions (`deftm`/`ftm`). The typed dispatch
;; ensures primitive-speed function evaluation inside tight solver loops.

;; ## Setup

(ns raster.optimization
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength]]
            [raster.sci.roots :as roots]
            [raster.sci.optim :as optim]
            [raster.sci.quadrature :as quad]
            [raster.ad.forward :as fwd]
            [scicloj.kindly.v4.kind :as kind]))

;; ## 1. Root Finding
;;
;; Find x where f(x) = 0. Three algorithms with different trade-offs:

;; **Bisection** — guaranteed convergence, needs a bracket:

(roots/bisect (ftm [x :- Double] :- Double (- (* x x) 2.0))
              0.0 2.0)
;; => {:root 1.41421356... :converged? true}

;; **Brent's method** — superlinear convergence, needs a bracket:

(roots/brent (ftm [x :- Double] :- Double (- (Math/exp x) 3.0))
             0.0 2.0)
;; => {:root 1.09861... = ln(3)}

;; **Newton's method** — quadratic convergence, needs the derivative:

(let [f  (ftm [x :- Double] :- Double (- (* x x x) (* 2.0 x) 5.0))
      df (ftm [x :- Double] :- Double (- (* 3.0 x x) 2.0))]
  (roots/newton f df 2.0))

;; Newton's method can also use automatic differentiation for the
;; derivative — no need to derive it by hand. Forward-mode AD uses
;; `raster.numeric` operators for Dual number dispatch:

(let [f-poly (fn [x] (n/- (n/- (n/* x x x) (n/* 2.0 x)) 5.0))
      f  (ftm [x :- Double] :- Double (f-poly x))
      df (ftm [x :- Double] :- Double (fwd/derivative f-poly x))]
  (roots/newton f df 2.0))

;; ## 2. Optimization
;;
;; Minimize f(x) over x ∈ Rⁿ.

;; **Nelder-Mead** — derivative-free, robust for noisy or non-smooth problems:

(let [rosenbrock (ftm [x :- (Array double)] :- Double
                   (let [a (- (aget x 0) 1.0)
                         b (- (aget x 1) (* (aget x 0) (aget x 0)))]
                     (+ (* a a) (* 100.0 b b))))]
  (optim/optimize rosenbrock (double-array [0.0 0.0])
                  {:algorithm :nelder-mead :maxiter 1000}))

;; **L-BFGS** — gradient-based, fast convergence for smooth problems.
;; Provide the gradient function explicitly or let AD compute it:

(let [rosenbrock (ftm [x :- (Array double)] :- Double
                   (let [a (- (aget x 0) 1.0)
                         b (- (aget x 1) (* (aget x 0) (aget x 0)))]
                     (+ (* a a) (* 100.0 b b))))
      grad-fn (ftm [x :- (Array double), out :- (Array double)]
                (let [x0 (aget x 0) x1 (aget x 1)]
                  ;; Analytical gradient of Rosenbrock
                  (aset out 0 (+ (* -400.0 x0 (- x1 (* x0 x0))) (* 2.0 (- x0 1.0))))
                  (aset out 1 (* 200.0 (- x1 (* x0 x0))))))]
  (optim/optimize rosenbrock (double-array [-1.0 -1.0])
                  {:algorithm :lbfgs :gradient grad-fn :maxiter 200}))

;; ## 3. Numerical Integration (Quadrature)
;;
;; Compute definite integrals ∫_a^b f(x) dx.

;; **Gauss-Kronrod** — adaptive, high-accuracy:

(quad/quadgk (ftm [x :- Double] :- Double (Math/exp (- (* x x))))
             -5.0 5.0)
;; => {:result 1.7724538... = √π, :error ~1e-15}

;; **Simpson's rule** — fixed-step, simple:

(quad/simps (ftm [x :- Double] :- Double (* x x))
              0.0 1.0 100)
;; => 0.33333... = 1/3

;; ## Solver Selection Guide
;;
;; | Problem | Method | When to use |
;; |---|---|---|
;; | f(x) = 0, bracket known | `bisect` | Guaranteed, slow |
;; | f(x) = 0, bracket known | `brent` | Fast, robust |
;; | f(x) = 0, derivative available | `newton` | Fastest (quadratic) |
;; | min f(x), no derivatives | `nelder-mead` | Noisy, non-smooth |
;; | min f(x), smooth | `lbfgs` | Large-scale, smooth |
;; | ∫f(x)dx, high accuracy | `quadgk` | Adaptive, general |
;; | ∫f(x)dx, simple | `simpson` | Fixed-step, cheap |

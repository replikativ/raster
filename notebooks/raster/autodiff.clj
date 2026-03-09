;; # Automatic Differentiation with Raster

;; authors: Christian Weilbach

;; Raster includes forward-mode and reverse-mode automatic differentiation.
;; Any function written with `deftm` can be differentiated — the compiler
;; transforms the code to compute gradients alongside the primal value.
;; No special tape objects, no manual backward passes.

;; ## Setup

(ns raster.autodiff
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.ad.forward :as fwd]
            [raster.ad.reverse :as rev]
            [scicloj.kindly.v4.kind :as kind]))

;; ## 1. Forward-Mode: Dual Numbers
;;
;; Forward-mode AD computes the derivative of a function alongside its
;; value by propagating "dual numbers" — pairs of (value, derivative).
;; The chain rule is applied automatically at each arithmetic operation.

;; Forward-mode works with plain functions using `raster.numeric`
;; operators, which automatically dispatch on Dual numbers:

(fwd/derivative (fn [x] (n/+ (n/* 3.0 x x) (n/* -2.0 x) 5.0)) 4.0)
;; => 22.0 (= 6*4 - 2, the derivative of 3x² - 2x + 5)

(fwd/derivative (fn [x] (n/* (fwd/sin x) (fwd/cos x))) 0.0)
;; => 1.0 (= cos²(0) - sin²(0))

;; ## 2. Reverse-Mode: Backpropagation
;;
;; Reverse-mode AD is efficient for functions with many inputs and one
;; output (like loss functions). One backward pass gives all gradients.
;;
;; `value+grad` takes a `deftm` var and returns a function that computes
;; both the value and all parameter gradients:

(deftm rosenbrock-2d [x :- Double, y :- Double] :- Double
  (+ (* 100.0 (n/pow (- y (* x x)) 2))
     (n/pow (- 1.0 x) 2)))

(let [vg (rev/value+grad #'rosenbrock-2d)]
  {:at-minimum (vg 1.0 1.0)
   ;; => [0.0, 0.0, 0.0] — value=0, both gradients zero
   :at-origin  (vg 0.0 0.0)
   ;; => [1.0, -2.0, 0.0] — gradient points toward the minimum
   })

;; ## 2. Reverse-Mode: Backpropagation
;;
;; Reverse-mode AD is efficient for functions with many inputs and one
;; output (like loss functions). One backward pass gives all gradients.
;;
;; `value+grad` returns a function that computes both the value and
;; all parameter gradients:

(deftm loss-fn [w1 :- Double, w2 :- Double, x :- Double, y :- Double] :- Double
  (let [pred (+ (* w1 x) (* w2 (* x x)))]
    (n/pow (- pred y) 2)))

;; `value+grad` wraps a deftm var into a gradient function:

(let [vg (rev/value+grad #'loss-fn)]
  (vg 1.0 0.5 2.0 7.0))
;; => [value, d/dw1, d/dw2, d/dx, d/dy]
;; The gradients tell you how to adjust each parameter to reduce the loss.

;; ## 3. Gradient Descent: Learning Parameters
;;
;; Combine `value+grad` with a simple update loop to fit parameters
;; to data. Here we fit a quadratic y = w1*x + w2*x² to noisy samples:

(def training-history
  (let [vg (rev/value+grad #'loss-fn)
        true-w1 0.3, true-w2 1.5
        data (mapv (fn [x]
                     (let [y (+ (* true-w1 x) (* true-w2 x x)
                                (* 0.1 (- (Math/random) 0.5)))]
                       [x y]))
                   (range -2.0 2.1 0.2))]
    (loop [w1 0.0, w2 0.0, iter 0, history []]
      (if (>= iter 200)
        history
        (let [[total-loss dw1 dw2]
              (reduce (fn [[l d1 d2] [x y]]
                        (let [r (vg w1 w2 x y)]
                          [(+ l (nth r 0))
                           (+ d1 (nth r 1))
                           (+ d2 (nth r 2))]))
                      [0.0 0.0 0.0]
                      data)
              n (count data)
              lr 0.001]
          (recur (- w1 (* lr (/ dw1 n)))
                 (- w2 (* lr (/ dw2 n)))
                 (inc iter)
                 (conj history {:iter iter
                                :loss (/ total-loss n)
                                :w1 w1 :w2 w2})))))))

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 500 :height 300
  :title "Gradient Descent: Fitting y = w1*x + w2*x²"
  :data {:values (take-nth 2 training-history)}
  :mark {:type "line" :strokeWidth 2}
  :encoding {:x {:field "iter" :type "quantitative" :title "Iteration"}
             :y {:field "loss" :type "quantitative" :title "Mean Squared Error"
                 :scale {:type "log"}}}})

;; The learned parameters:

(let [final (last training-history)]
  {:w1 (format "%.3f (true: 0.300)" (:w1 final))
   :w2 (format "%.3f (true: 1.500)" (:w2 final))})

;; ## 4. Sensitivity Analysis: Gradients Through ODE Solvers
;;
;; Raster can differentiate through an entire ODE solve. This is called
;; sensitivity analysis — it answers "how does the solution change if I
;; change a parameter?"
;;
;; The Lotka-Volterra model has 4 parameters (α, β, δ, γ). Forward-mode
;; AD propagates Dual numbers through the RK4 solver to compute
;; ∂solution/∂parameter for all time points simultaneously.
;;
;; This works because the RK4 solver is written with `deftm` and
;; `raster.numeric` operators — it doesn't need to know about AD.
;; Dual numbers flow through the solver automatically.

;; ## 5. When to Use Which Mode
;;
;; | Mode | Best for | Cost |
;; |---|---|---|
;; | Forward (`fwd/derivative`) | Few inputs, scalar → scalar | O(n_params) forward passes |
;; | Forward (`fwd/gradient`) | Few inputs, vector → scalar | O(n_params / chunk_size) passes |
;; | Reverse (`rev/value+grad`) | Many inputs, one output (losses) | O(1) backward pass |
;;
;; **Forward mode** is optimal for ODE sensitivity with few parameters
;; (< 10). **Reverse mode** is optimal for neural network training
;; (millions of parameters, one scalar loss).
;;
;; Both modes compose with the compiler pipeline — `compile-aot`
;; can inline and optimize the AD-transformed code.

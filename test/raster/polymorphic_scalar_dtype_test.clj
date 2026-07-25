(ns raster.polymorphic-scalar-dtype-test
  "Regression guard for the `(All [T])` scalar-dtype rule (CLAUDE.md ~line 316, issue #69).

   A scalar that participates in `T`-typed arithmetic must be declared `:- T`, not `:- Double`.
   With `:- Double` the call still DEVIRTUALIZES — Julia-style promotion picks the
   `[Double Double]` overload and the walker inserts a `(double …)` cast on the float operand
   (walker.clj:1160-1162 documents exactly this) — so results stay correct and nothing throws.
   The damage is f64 contamination of an f32 kernel: the loop runs in f64 over float data with a
   narrowing round-trip per element, float-lane SIMD is blocked, and on OpenCL the casts
   force-enable cl_khr_fp64 (works on Arc, HARD-FAILS on fp64-less backends such as WGSL).

   Because it neither throws nor produces wrong answers, no existing test caught it — these
   kernels had ZERO non-double coverage. This namespace supplies it: every kernel below is
   exercised with float[] AND double[] and must agree, which is only possible when the scalar
   is `:- T` (a `:- Double` scalar cannot dispatch against a float array without promotion)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.dl.optim :as optim]
            [raster.dl.nn :as nn]))

(defn- fa ^floats [xs] (float-array (map float xs)))
(defn- da ^doubles [xs] (double-array (map double xs)))
(defn- close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true? (map #(< (Math/abs (- (double %1) (double %2))) tol) xs ys))))

;; ── raster.numeric: the two functions issue #69 reported ────────────────────────────
(deftest numeric-scale!-axpy!-are-element-typed
  (testing "scale! accepts a T-typed scalar for float AND double arrays, same result"
    (let [xs [1 2 3 4]
          f (vec (n/scale! (fa (repeat 4 0)) (float 2) (fa xs)))
          d (vec (n/scale! (da (repeat 4 0)) (double 2) (da xs)))]
      (is (close? f [2 4 6 8] 1e-6))
      (is (close? f d 1e-6) "float and double paths agree")))
  (testing "axpy! likewise"
    (let [a [1 2 3 4] b [10 20 30 40]
          f (vec (n/axpy! (fa (repeat 4 0)) (fa a) (float 3) (fa b)))
          d (vec (n/axpy! (da (repeat 4 0)) (da a) (double 3) (da b)))]
      (is (close? f [31 62 93 124] 1e-4))
      (is (close? f d 1e-4))))
  (testing "oftype is the documented way to pass a literal at the call site"
    (let [a (fa [1 2 3 4])
          out (n/scale! (fa (repeat 4 0)) (n/oftype a 2.0) a)]
      (is (close? (vec out) [2 4 6 8] 1e-6)))))

;; ── raster.par elementwise broadcasts ───────────────────────────────────────────────
(deftest par-axpy-scale-fill-are-element-typed
  (testing "par/scale + par/axpy agree across float and double"
    (let [x [1 2 3 4] y [10 20 30 40]]
      (is (close? (vec (par/scale (float 2) (fa x))) [2 4 6 8] 1e-6))
      (is (close? (vec (par/scale (float 2) (fa x)))
                  (vec (par/scale (double 2) (da x))) 1e-6))
      (is (close? (vec (par/axpy (float 3) (fa x) (fa y)))
                  (vec (par/axpy (double 3) (da x) (da y))) 1e-4))))
  (testing "par/fill"
    (is (close? (vec (par/fill (fa (repeat 3 0)) (float 7))) [7 7 7] 1e-6))
    (is (close? (vec (par/fill (da (repeat 3 0)) (double 7))) [7 7 7] 1e-6))))

;; ── dl/optim: the GPU-live optimizer kernels ────────────────────────────────────────
(deftest optimizer-kernels-are-element-typed
  (testing "sgd-step! (on the resident GPU training path) — float == double"
    (let [p [1.0 2.0 3.0] g [0.5 0.5 0.5] lr 0.1
          pf (fa p) pd (da p)]
      (optim/sgd-step! pf (fa g) 3 (float lr))
      (optim/sgd-step! pd (da g) 3 (double lr))
      (is (close? (vec pf) [0.95 1.95 2.95] 1e-5))
      (is (close? (vec pf) (vec pd) 1e-5))))
  (testing "ema-update! — float == double"
    (let [sf (fa [1 2 3]) sd (da [1 2 3]) p [5.0 5.0 5.0]]
      (optim/ema-update! sf (fa p) 3 (float 0.9))
      (optim/ema-update! sd (da p) 3 (double 0.9))
      (is (close? (vec sf) (vec sd) 1e-5))))
  (testing "adam-step! — float == double (bias correction + eps all T-typed)"
    (let [p [1.0 2.0] g [0.1 0.2]
          pf (fa p) pd (da p)]
      (optim/adam-step! pf (fa g) (fa [0 0]) (fa [0 0]) 2
                        (float 0.01) (float 0.9) (float 0.999) (float 1e-7) 1)
      (optim/adam-step! pd (da g) (da [0 0]) (da [0 0]) 2
                        (double 0.01) (double 0.9) (double 0.999) (double 1e-7) 1)
      (is (close? (vec pf) (vec pd) 1e-4))))
  (testing "adamw-step! — float == double"
    (let [pf (fa [1 2]) pd (da [1 2]) g [0.1 0.2]]
      (optim/adamw-step! pf (fa g) (fa [0 0]) (fa [0 0]) 2
                         (float 0.01) (float 0.9) (float 0.999) (float 1e-7) (float 0.01) 1)
      (optim/adamw-step! pd (da g) (da [0 0]) (da [0 0]) 2
                         (double 0.01) (double 0.9) (double 0.999) (double 1e-7) (double 0.01) 1)
      (is (close? (vec pf) (vec pd) 1e-4)))))

;; ── dl/nn leaky-relu family ─────────────────────────────────────────────────────────
(deftest leaky-relu-is-element-typed
  (testing "leaky-relu / -backward agree across float and double"
    (let [x [-2.0 -1.0 0.0 1.0 2.0] a 0.1]
      (is (close? (vec (nn/leaky-relu (fa x) 5 (float a)))
                  (vec (nn/leaky-relu (da x) 5 (double a))) 1e-6))
      (is (close? (vec (nn/leaky-relu (fa x) 5 (float a)))
                  [-0.2 -0.1 0.0 1.0 2.0] 1e-6))
      (is (close? (vec (nn/leaky-relu-backward (fa (repeat 5 1)) (fa x) 5 (float a)))
                  (vec (nn/leaky-relu-backward (da (repeat 5 1)) (da x) 5 (double a))) 1e-6)))))

;; ── the invariant itself: a Double scalar must NOT dispatch against a float array ────
(deftest double-scalar-cannot-silently-enter-a-float-kernel
  (testing "passing a raw Double to a float[] kernel now FAILS LOUD instead of promoting to f64"
    (is (thrown? clojure.lang.ExceptionInfo
                 (n/scale! (fa [0 0 0 0]) 2.0 (fa [1 2 3 4]))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (optim/sgd-step! (fa [1 2 3]) (fa [1 1 1]) 3 0.1))))
  (testing "…and double[] kernels still take plain literals unchanged"
    (is (close? (vec (n/scale! (da [0 0 0 0]) 2.0 (da [1 2 3 4]))) [2 4 6 8] 1e-9))
    (is (close? (vec (par/scale 2.0 (da [1 2 3 4]))) [2 4 6 8] 1e-9))))

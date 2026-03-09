(ns raster.random
  "Random number generation for numerical code.

  Julia-inspired API using java.util.random.RandomGenerator — the common
  interface for java.util.Random, SplittableRandom, and all JDK RNGs.
  Mutating ops (fill in-place) use ! suffix convention.

  Usage:
    (def rng (java.util.Random. 42))
    (randn! rng (double-array 100))     ; fill with standard normal
    (rand!  rng (double-array 100))     ; fill with uniform [0,1)

    ;; SplittableRandom works too (deterministic parallel RNG):
    (def srng (java.util.SplittableRandom. 42))
    (rand-int rng 100)                  ; uniform int in [0, 100)
    (rand-long rng)                     ; uniform long
    (rand-double rng 0.1 1.5)           ; uniform double in [0.1, 1.5)"
  (:refer-clojure :exclude [rand-int])
  (:require [raster.core :refer [deftm]])
  (:import [java.util.random RandomGenerator]))

;; ================================================================
;; Scalar ops — for use in deftm bodies (ABM agent decisions, etc.)
;; ================================================================

(deftm rand-int
  "Uniform random int in [0, bound). Works with any RandomGenerator."
  [rng :- RandomGenerator, bound :- Long] :- Long
  (long (.nextInt rng (int bound))))

(deftm rand-long
  "Uniform random long. Works with any RandomGenerator."
  [rng :- RandomGenerator] :- Long
  (.nextLong rng))

(deftm rand-double
  "Uniform random double in [origin, bound).
   Works with any RandomGenerator."
  [rng :- RandomGenerator, origin :- Double, bound :- Double] :- Double
  (.nextDouble rng origin bound))

(deftm rand-float
  "Uniform random float in [origin, bound).
   Works with any RandomGenerator."
  [rng :- RandomGenerator, origin :- Float, bound :- Float] :- Float
  (.nextFloat rng origin bound))

(deftm rand-gaussian
  "Standard normal sample. Works with any RandomGenerator."
  [rng :- RandomGenerator] :- Double
  (.nextGaussian rng))

;; ================================================================
;; fill-seeds! — fill long array with random seeds
;; ================================================================

(deftm fill-seeds!
  "Fill a long array with random seeds from rng."
  [rng :- RandomGenerator, seeds :- (Array long), n :- Long] :- Long
  (dotimes [i n]
    (aset seeds i (.nextLong rng)))
  0)

;; ================================================================
;; select-active! — fill int array with random agent indices
;; ================================================================

(deftm select-active!
  "Fill active-ids with n-active random ints in [0, n-agents)."
  [rng :- RandomGenerator, active-ids :- (Array int),
   n-active :- Long, n-agents :- Long] :- Long
  (dotimes [i n-active]
    (aset active-ids i (int (.nextInt rng 0 (int n-agents)))))
  0)

;; ================================================================
;; randn! — fill array with standard normal samples
;; ================================================================

(deftm randn! [rng :- RandomGenerator, out :- (Array double)] :- (Array double)
  (let [n (alength out)]
    (dotimes [i n]
      (aset out i (.nextGaussian rng)))
    out))

(deftm randn! [rng :- RandomGenerator, out :- (Array float)] :- (Array float)
  (let [n (alength out)]
    (dotimes [i n]
      (aset out i (float (.nextGaussian rng))))
    out))

;; ================================================================
;; rand! — fill array with uniform [0,1) samples
;; ================================================================

(deftm rand! [rng :- RandomGenerator, out :- (Array double)] :- (Array double)
  (let [n (alength out)]
    (dotimes [i n]
      (aset out i (.nextDouble rng)))
    out))

(deftm rand! [rng :- RandomGenerator, out :- (Array float)] :- (Array float)
  (let [n (alength out)]
    (dotimes [i n]
      (aset out i (float (.nextDouble rng))))
    out))

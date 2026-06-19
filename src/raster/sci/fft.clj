(ns raster.sci.fft
  "Fast Fourier Transform using iterative radix-2 Cooley-Tukey algorithm.

  Pure-Java implementation with no external dependencies.

  Usage:
    (require '[raster.sci.fft :refer [fft ifft rfft fftfreq]])
    (let [[re im] (fft (double-array [1 0 -1 0]))]
      ;; re = [0 2 0 2], im = [0 0 0 0]  (DC=0, Nyquist=0, freq1=2)
      )"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm broadcast]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; Utilities
;; ================================================================

(deftm next-power-of-2 [n :- Long] :- Long
  (if (<= n 1)
    1
    (Long/highestOneBit (dec (* 2 n)))))

(deftm bit-reverse [i :- Long, bits :- Long] :- Long
  (loop [j 0 result 0]
    (if (>= j bits)
      result
      (recur (inc j)
             (bit-or (bit-shift-left result 1)
                     (bit-and (bit-shift-right i j) 1))))))

;; ================================================================
;; Cooley-Tukey FFT (iterative, radix-2, in-place)
;; ================================================================

(deftm fft-core!
  [re :- (Array double), im :- (Array double), sign :- Double]
  (let [n (alength re)
        bits (Long/numberOfTrailingZeros n)]
    ;; Bit-reversal permutation
    (dotimes [i n]
      (let [j (bit-reverse i bits)]
        (when (< i j)
          (let [tr (aget re i) ti (aget im i)]
            (aset re i (aget re j))
            (aset im i (aget im j))
            (aset re j tr)
            (aset im j ti)))))
    ;; Butterfly stages
    (loop [size 2]
      (when (<= size n)
        (let [half-size (bit-shift-right size 1)
              angle (n/* sign (n// (n/* 2.0 n/pi) (double size)))
              wr-step (m/cos angle)
              wi-step (m/sin angle)]
          (loop [k 0]
            (when (< k n)
              (loop [j 0 wr 1.0 wi 0.0]
                (when (< j half-size)
                  (let [idx1 (+ k j)
                        idx2 (+ k j half-size)
                        tr (n/- (n/* wr (aget re idx2)) (n/* wi (aget im idx2)))
                        ti (n/+ (n/* wr (aget im idx2)) (n/* wi (aget re idx2)))]
                    (aset re idx2 (n/- (aget re idx1) tr))
                    (aset im idx2 (n/- (aget im idx1) ti))
                    (aset re idx1 (n/+ (aget re idx1) tr))
                    (aset im idx1 (n/+ (aget im idx1) ti))
                    (recur (inc j)
                           (n/- (n/* wr wr-step) (n/* wi wi-step))
                           (n/+ (n/* wr wi-step) (n/* wi wr-step))))))
              (recur (+ k size)))))
        (recur (* size 2))))))

;; ================================================================
;; Public API
;; ================================================================

(deftm fft [x :- (Array double)]
  (let [n0 (alength x)
        n (next-power-of-2 n0)
        re (double-array n)
        im (double-array n)]
    (acopy! x 0 re 0 n0)
    (fft-core! re im -1.0)
    [re im]))

(deftm fft-complex [re-in :- (Array double), im-in :- (Array double)]
  (let [n0 (alength re-in)
        n (next-power-of-2 n0)
        re (double-array n)
        im (double-array n)]
    (acopy! re-in 0 re 0 n0)
    (acopy! im-in 0 im 0 n0)
    (fft-core! re im -1.0)
    [re im]))

(deftm ifft [re-in :- (Array double), im-in :- (Array double)]
  (let [n (alength re-in)
        re (aclone re-in)
        im (aclone im-in)]
    (fft-core! re im 1.0)
    (let [inv-n (n// 1.0 (double n))
          re-scaled (broadcast [re] (n/* re inv-n))
          im-scaled (broadcast [im] (n/* im inv-n))]
      [re-scaled im-scaled])))

(deftm rfft [x :- (Array double)]
  (let [[full-re full-im] (fft x)
        n (alength full-re)
        m (inc (bit-shift-right n 1))
        re (double-array m)
        im (double-array m)]
    (acopy! full-re 0 re 0 m)
    (acopy! full-im 0 im 0 m)
    [re im]))

(deftm fftfreq [n :- Long, dt :- Double]
  (let [freqs (double-array n)
        inv-ndt (n// 1.0 (n/* (double n) dt))
        half (bit-shift-right n 1)]
    (dotimes [i half]
      (aset freqs i (n/* (double i) inv-ndt)))
    (loop [i half]
      (when (< i n)
        (aset freqs i (n/* (double (- i n)) inv-ndt))
        (recur (inc i))))
    freqs))

(deftm magnitude [re :- (Array double), im :- (Array double)]
  (broadcast [re im] (n/sqrt (n/+ (n/* re re) (n/* im im)))))

(deftm phase [re :- (Array double), im :- (Array double)]
  (broadcast [im re] (m/atan2 im re)))

;; ================================================================
;; DCT-II (Discrete Cosine Transform)
;; ================================================================

(deftm dct
  "Type-II Discrete Cosine Transform.
  X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))"
  [x :- (Array double) n :- Long] :- (Array double)
  (let [out (double-array n)
        inv-2n (n// n/pi (n/* 2.0 (double n)))]
    (dotimes [k n]
      (let [s (loop [i 0 acc 0.0]
                (if (>= i n) acc
                    (recur (inc i)
                           (n/+ acc (n/* (aget x i)
                                         (m/cos (n/* inv-2n
                                                     (double k)
                                                     (n/+ (n/* 2.0 (double i)) 1.0))))))))]
        (aset out k (n/* 2.0 s))))
    out))

;; ================================================================
;; IDCT-II (Inverse DCT via DCT-III)
;; ================================================================

(deftm idct
  "Inverse DCT (DCT-III, normalized).
  x[n] = (1/N) * [X[0]/2 + sum_{k=1}^{N-1} X[k] * cos(pi*k*(2n+1)/(2N))]"
  [X :- (Array double) n :- Long] :- (Array double)
  (let [out (double-array n)
        inv-2n (n// n/pi (n/* 2.0 (double n)))
        inv-n (n// 1.0 (double n))]
    (dotimes [i n]
      (let [s (loop [k 1 acc (n/* 0.5 (aget X 0))]
                (if (>= k n) acc
                    (recur (inc k)
                           (n/+ acc (n/* (aget X k)
                                         (m/cos (n/* inv-2n
                                                     (double k)
                                                     (n/+ (n/* 2.0 (double i)) 1.0))))))))]
        (aset out i (n/* inv-n s))))
    out))

;; ================================================================
;; DST-I (Discrete Sine Transform)
;; ================================================================

(deftm dst
  "Type-I Discrete Sine Transform.
  X[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(n+1)*(k+1) / (N+1))"
  [x :- (Array double) n :- Long] :- (Array double)
  (let [out (double-array n)
        inv-n1 (n// n/pi (n/+ (double n) 1.0))]
    (dotimes [k n]
      (let [s (loop [i 0 acc 0.0]
                (if (>= i n) acc
                    (recur (inc i)
                           (n/+ acc (n/* (aget x i)
                                         (m/sin (n/* inv-n1
                                                     (n/+ (double i) 1.0)
                                                     (n/+ (double k) 1.0))))))))]
        (aset out k (n/* 2.0 s))))
    out))

;; ================================================================
;; IDST-I (Inverse DST)
;; ================================================================

(deftm idst
  "Inverse DST (DST-I is its own inverse up to normalization).
  x[n] = (1/(N+1)) * sum_{k=0}^{N-1} X[k] * sin(pi*(n+1)*(k+1)/(N+1))"
  [X :- (Array double) n :- Long] :- (Array double)
  (let [out (double-array n)
        inv-n1 (n// n/pi (n/+ (double n) 1.0))
        scale (n// 1.0 (n/+ (double n) 1.0))]
    (dotimes [i n]
      (let [s (loop [k 0 acc 0.0]
                (if (>= k n) acc
                    (recur (inc k)
                           (n/+ acc (n/* (aget X k)
                                         (m/sin (n/* inv-n1
                                                     (n/+ (double i) 1.0)
                                                     (n/+ (double k) 1.0))))))))]
        (aset out i (n/* scale s))))
    out))

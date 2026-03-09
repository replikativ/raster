(ns raster.sci.interpolation
  "Interpolation algorithms following Julia's Interpolations.jl patterns.

  Provides linear interpolation, cubic spline (natural boundary conditions),
  and Akima spline (locally adaptive, avoids oscillation).

  All interpolants return callable objects that support point evaluation.

  Usage:
    (require '[raster.sci.interpolation :refer [linear-interp cubic-spline akima-spline]])
    (def interp (linear-interp (double-array [0 1 2 3])
                               (double-array [0 1 4 9])))
    (interp 1.5)  ;=> 2.5"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm defvalue defabstract]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; Binary search helper
;; ================================================================

(deftm find-interval (All [T] [xs :- (Array T), x :- Double, n :- Long] :- Long
                          (let [n-1 (dec n)]
                            (if (<= x (aget xs 0))
                              0
                              (if (>= x (aget xs n-1))
                                (- n-1 1)
                                (loop [lo 0 hi n-1]
                                  (if (<= (- hi lo) 1)
                                    lo
                                    (let [mid (bit-shift-right (+ lo hi) 1)]
                                      (if (<= (aget xs mid) x)
                                        (recur mid hi)
                                        (recur lo mid))))))))))

;; ================================================================
;; Linear interpolation
;; ================================================================

(deftm linear-interp (All [T] [xs :- (Array T), ys :- (Array T)]
                          (let [n (long (alength xs))]
                            (when (< n 2)
                              (throw (ex-info "linear-interp requires at least 2 points" {:n n})))
                            (ftm [x :- Double] :- Double
                                 (let [i (find-interval xs x n)
                                       x0 (aget xs i)
                                       x1 (aget xs (inc i))
                                       y0 (aget ys i)
                                       y1 (aget ys (inc i))
                                       t (n// (n/- x x0) (n/- x1 x0))]
                                   (n/+ y0 (n/* t (n/- y1 y0))))))))

;; ================================================================
;; Tridiagonal solver
;; ================================================================

(deftm solve-tridiagonal! (All [T] [a :- (Array T), b :- (Array T), c :- (Array T), d :- (Array T), n :- Long]
  ;; Forward sweep
                               (loop [i 1]
                                 (when (< i n)
                                   (let [m (/ (aget a (dec i)) (aget b (dec i)))]
                                     (aset b i (- (aget b i) (* m (aget c (dec i)))))
                                     (aset d i (- (aget d i) (* m (aget d (dec i))))))
                                   (recur (inc i))))
  ;; Back substitution
                               (aset d (dec n) (/ (aget d (dec n)) (aget b (dec n))))
                               (loop [i (- n 2)]
                                 (when (>= i 0)
                                   (aset d i (/ (- (aget d i) (* (aget c i) (aget d (inc i))))
                                                (aget b i)))
                                   (recur (dec i))))
                               d))

;; ================================================================
;; Cubic spline (natural boundary conditions)
;; ================================================================

(deftm cubic-spline (All [T] [xs :- (Array T), ys :- (Array T)]
                         (let [n (long (alength xs))
                               _ (when (< n 3)
                                   (throw (ex-info "cubic-spline requires at least 3 points" {:n n})))
                               n-1 (dec n)
        ;; Compute intervals h_i = x_{i+1} - x_i
                               h (raster.arrays/alloc-like xs n-1)
                               _ (dotimes [i n-1]
                                   (aset h i (- (aget xs (inc i)) (aget xs i))))
        ;; Set up tridiagonal system for second derivatives (moments) M
                               nm (- n 2)
                               sub (raster.arrays/alloc-like xs (max 1 (dec nm)))
                               diag (raster.arrays/alloc-like xs (max 1 nm))
                               sup (raster.arrays/alloc-like xs (max 1 (dec nm)))
                               rhs (raster.arrays/alloc-like xs (max 1 nm))]
                           (if (< nm 1)
      ;; Only 3 points: nm=1, trivial
                             (let [M (raster.arrays/alloc-like xs n)
                                   _ (when (== nm 1)
                                       (aset diag 0 (* 2.0 (+ (aget h 0) (aget h 1))))
                                       (aset rhs 0 (* 6.0 (- (/ (- (aget ys 2) (aget ys 1)) (aget h 1))
                                                             (/ (- (aget ys 1) (aget ys 0)) (aget h 0)))))
                                       (aset M 1 (/ (aget rhs 0) (aget diag 0))))
                                   cs (raster.arrays/alloc-like xs n)]
                               (dotimes [i n] (aset cs i (aget M i)))
                               (ftm [x :- Double] :- Double
                                    (let [i (find-interval xs x n)
                                          hi (aget h i)
                                          dx (n/- x (aget xs i))
                                          yi (aget ys i)
                                          yi+1 (aget ys (inc i))
                                          mi (aget cs i)
                                          mi+1 (aget cs (inc i))
                                          a (n// (n/- mi+1 mi) (n/* 6.0 hi))
                                          b (n/* 0.5 mi)
                                          c-coeff (n/- (n// (n/- yi+1 yi) hi) (n/* hi (n// (n/+ (n/* 2.0 mi) mi+1) 6.0)))]
                                      (n/+ yi (n/* dx (n/+ c-coeff (n/* dx (n/+ b (n/* dx a)))))))))
      ;; General case: nm >= 2
                             (do
                               (dotimes [i nm]
                                 (let [j (inc i)]
                                   (aset diag i (* 2.0 (+ (aget h (dec j)) (aget h j))))
                                   (aset rhs i (* 6.0 (- (/ (- (aget ys (inc j)) (aget ys j)) (aget h j))
                                                         (/ (- (aget ys j) (aget ys (dec j))) (aget h (dec j))))))))
                               (dotimes [i (dec nm)]
                                 (aset sub i (aget h (inc i)))
                                 (aset sup i (aget h (inc i))))
                               (solve-tridiagonal! sub diag sup rhs nm)
                               (let [M (raster.arrays/alloc-like xs n)]
                                 (dotimes [i nm]
                                   (aset M (inc i) (aget rhs i)))
                                 (let [cs (raster.arrays/alloc-like xs n)]
                                   (dotimes [i n] (aset cs i (aget M i)))
                                   (ftm [x :- Double] :- Double
                                        (let [i (find-interval xs x n)
                                              hi (aget h i)
                                              dx (n/- x (aget xs i))
                                              yi (aget ys i)
                                              yi+1 (aget ys (inc i))
                                              mi (aget cs i)
                                              mi+1 (aget cs (inc i))
                                              a (n// (n/- mi+1 mi) (n/* 6.0 hi))
                                              b (n/* 0.5 mi)
                                              c-coeff (n/- (n// (n/- yi+1 yi) hi) (n/* hi (n// (n/+ (n/* 2.0 mi) mi+1) 6.0)))]
                                          (n/+ yi (n/* dx (n/+ c-coeff (n/* dx (n/+ b (n/* dx a)))))))))))))))

;; ================================================================
;; Akima spline (locally adaptive)
;; ================================================================

(deftm akima-spline (All [T] [xs :- (Array T), ys :- (Array T)]
                         (let [n (long (alength xs))
                               _ (when (< n 5)
                                   (throw (ex-info "akima-spline requires at least 5 points" {:n n})))
                               n-1 (dec n)
        ;; Compute slopes m_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
                               m (raster.arrays/alloc-like xs (+ n-1 4))  ;; padded: m[-2], m[-1], m[0]..m[n-2], m[n-1], m[n]
                               _ (dotimes [i n-1]
                                   (aset m (+ i 2) (/ (- (aget ys (inc i)) (aget ys i))
                                                      (- (aget xs (inc i)) (aget xs i)))))
        ;; Pad ends
                               _ (let [m0 (aget m 2) m1 (aget m 3)]
                                   (aset m 1 (- (* 2.0 m0) m1))
                                   (aset m 0 (- (* 2.0 (aget m 1)) m0)))
                               _ (let [mn-2 (aget m (dec n)) mn-1 (aget m n)]
                                   (aset m (inc n) (- (* 2.0 mn-1) mn-2))
                                   (aset m (+ n 2) (- (* 2.0 (aget m (inc n))) mn-1)))
        ;; Compute Akima weights and slopes at each knot
                               slopes (raster.arrays/alloc-like xs n)
                               _ (dotimes [j n]
                                   (let [i (+ j 2)
                                         t-im1 (n/abs (- (aget m (dec i)) (aget m (- i 2))))
                                         t-ip1 (n/abs (- (aget m (inc i)) (aget m i)))
                                         denom (+ t-ip1 t-im1)]
                                     (aset slopes j
                                           (if (< denom 1e-30)
                                             (* 0.5 (+ (aget m (dec i)) (aget m i)))
                                             (/ (+ (* t-ip1 (aget m (dec i))) (* t-im1 (aget m i)))
                                                denom)))))
        ;; Compute cubic polynomial coefficients per interval
                               b-arr (raster.arrays/alloc-like xs n-1)
                               c-arr (raster.arrays/alloc-like xs n-1)
                               d-arr (raster.arrays/alloc-like xs n-1)
                               _ (dotimes [i n-1]
                                   (let [hi (- (aget xs (inc i)) (aget xs i))
                                         si (aget slopes i)
                                         si+1 (aget slopes (inc i))
                                         mi (/ (- (aget ys (inc i)) (aget ys i)) hi)]
                                     (aset b-arr i si)
                                     (aset c-arr i (/ (- (* 3.0 mi) (* 2.0 si) si+1) hi))
                                     (aset d-arr i (/ (+ si si+1 (* -2.0 mi)) (* hi hi)))))]
                           (ftm [x :- Double] :- Double
                                (let [i (find-interval xs x n)
                                      dx (n/- x (aget xs i))]
                                  (n/+ (aget ys i)
                                       (n/* dx (n/+ (aget b-arr i)
                                                    (n/* dx (n/+ (aget c-arr i)
                                                                 (n/* dx (aget d-arr i))))))))))))

;; ================================================================
;; Derivative evaluation for cubic splines
;; ================================================================

;; ================================================================
;; PCHIP interpolation (monotone-preserving)
;; ================================================================

(deftm pchip
  "Piecewise Cubic Hermite Interpolating Polynomial.
  Preserves monotonicity — no overshoot unlike cubic spline.
  Returns a callable closure."
  (All [T] [xs :- (Array T), ys :- (Array T)]
       (let [n (long (alength xs))
             _ (when (< n 3)
                 (throw (ex-info "pchip requires at least 3 points" {:n n})))
             n-1 (dec n)
        ;; Compute slopes
             h (raster.arrays/alloc-like xs n-1)
             delta (raster.arrays/alloc-like xs n-1)
             _ (dotimes [i n-1]
                 (let [hi (- (aget xs (inc i)) (aget xs i))]
                   (aset h i hi)
                   (aset delta i (/ (- (aget ys (inc i)) (aget ys i)) hi))))
        ;; Compute PCHIP derivatives at each knot
             d (raster.arrays/alloc-like xs n)
             _ (do
            ;; Interior points: weighted harmonic mean of adjacent slopes
                 (loop [i 1]
                   (when (< i n-1)
                     (let [d1 (aget delta (dec i))
                           d2 (aget delta i)]
                       (if (or (<= (* d1 d2) 0.0))
                    ;; Different signs or zero: set derivative to zero
                         (aset d i 0.0)
                    ;; Weighted harmonic mean
                         (let [w1 (+ (* 2.0 (aget h i)) (aget h (dec i)))
                               w2 (+ (aget h i) (* 2.0 (aget h (dec i))))]
                           (aset d i (/ (+ w1 w2) (+ (/ w1 d1) (/ w2 d2)))))))
                     (recur (inc i))))
            ;; Endpoint derivatives: one-sided shape-preserving
                 (let [d0 (aget delta 0)]
                   (aset d 0 d0))
                 (let [dn-1 (aget delta (dec n-1))]
                   (aset d (dec n) dn-1)))
        ;; Cubic coefficients per interval
             c2 (raster.arrays/alloc-like xs n-1)
             c3 (raster.arrays/alloc-like xs n-1)
             _ (dotimes [i n-1]
                 (let [hi (aget h i)
                       di (aget d i)
                       di+1 (aget d (inc i))
                       deli (aget delta i)]
                   (aset c2 i (/ (- (* 3.0 deli) (* 2.0 di) di+1) hi))
                   (aset c3 i (/ (+ di di+1 (* -2.0 deli)) (* hi hi)))))]
         (ftm [x :- Double] :- Double
              (let [i (find-interval xs x n)
                    dx (n/- x (aget xs i))]
                (n/+ (aget ys i)
                     (n/* dx (n/+ (aget d i)
                                  (n/* dx (n/+ (aget c2 i)
                                               (n/* dx (aget c3 i))))))))))))

;; ================================================================
;; 2D bilinear interpolation
;; ================================================================

(deftm interp2d-linear
  "Bilinear interpolation on a regular grid.
  xs[nx], ys[ny], zs[ny*nx] (row-major: z[i,j] = zs[i*nx + j]).
  Returns a callable closure (fn [x y] -> z)."
  (All [T] [xs :- (Array T), ys :- (Array T),
            zs :- (Array T), nx :- Long, ny :- Long]
       (ftm [x :- Double, y :- Double] :- Double
            (let [ix (find-interval xs x nx)
                  iy (find-interval ys y ny)
                  x0 (aget xs ix)
                  x1 (aget xs (inc ix))
                  y0 (aget ys iy)
                  y1 (aget ys (inc iy))
                  tx (n// (n/- x x0) (n/- x1 x0))
                  ty (n// (n/- y y0) (n/- y1 y0))
                  z00 (aget zs (n/+ (n/* iy nx) ix))
                  z01 (aget zs (n/+ (n/* iy nx) (inc ix)))
                  z10 (aget zs (n/+ (n/* (inc iy) nx) ix))
                  z11 (aget zs (n/+ (n/* (inc iy) nx) (inc ix)))]
              (n/+ (n/* (n/- 1.0 tx) (n/- 1.0 ty) z00)
                   (n/* tx (n/- 1.0 ty) z01)
                   (n/* (n/- 1.0 tx) ty z10)
                   (n/* tx ty z11))))))

;; ================================================================
;; 2D bicubic interpolation
;; ================================================================

(deftm interp2d-cubic
  "Bicubic interpolation on a regular grid using Catmull-Rom splines.
  xs[nx], ys[ny], zs[ny*nx] row-major. Requires nx >= 4, ny >= 4.
  Returns a callable closure."
  (All [T] [xs :- (Array T), ys :- (Array T),
            zs :- (Array T), nx :- Long, ny :- Long]
       (when (or (< nx 4) (< ny 4))
         (throw (ex-info "interp2d-cubic requires at least 4 points per axis"
                         {:nx nx :ny ny})))
       (ftm [x :- Double, y :- Double] :- Double
            (let [ix (find-interval xs x nx)
                  iy (find-interval ys y ny)
          ;; Clamp to valid range for 4-point stencil
                  ix (long (n/max 1 (n/min (n/- nx 3) ix)))
                  iy (long (n/max 1 (n/min (n/- ny 3) iy)))
                  tx (n// (n/- x (double (aget xs ix))) (n/- (double (aget xs (inc ix))) (double (aget xs ix))))
                  ty (n// (n/- y (double (aget ys iy))) (n/- (double (aget ys (inc iy))) (double (aget ys iy))))
          ;; Catmull-Rom basis
                  tx2 (n/* tx tx) tx3 (n/* tx2 tx)
                  ty2 (n/* ty ty) ty3 (n/* ty2 ty)
          ;; Weights in x
                  wx0 (n/* 0.5 (n/+ (n/- tx3) (n/* 2.0 tx2) (n/- tx)))
                  wx1 (n/* 0.5 (n/+ (n/* 3.0 tx3) (n/* -5.0 tx2) 2.0))
                  wx2 (n/* 0.5 (n/+ (n/* -3.0 tx3) (n/* 4.0 tx2) tx))
                  wx3 (n/* 0.5 (n/- tx3 tx2))
          ;; Weights in y
                  wy0 (n/* 0.5 (n/+ (n/- ty3) (n/* 2.0 ty2) (n/- ty)))
                  wy1 (n/* 0.5 (n/+ (n/* 3.0 ty3) (n/* -5.0 ty2) 2.0))
                  wy2 (n/* 0.5 (n/+ (n/* -3.0 ty3) (n/* 4.0 ty2) ty))
                  wy3 (n/* 0.5 (n/- ty3 ty2))]
      ;; Evaluate: sum over 4x4 stencil
              (let [i0 (dec iy) i1 iy i2 (inc iy) i3 (n/+ iy 2)
                    j0 (dec ix) j1 ix j2 (inc ix) j3 (n/+ ix 2)]
                (loop [di 0 sum 0.0]
                  (if (>= di 4) sum
                      (let [row (n/+ i0 di)
                            wy (case (int di) 0 wy0 1 wy1 2 wy2 3 wy3)
                            row-val (n/+ (n/* wx0 (aget zs (n/+ (n/* row nx) j0)))
                                         (n/* wx1 (aget zs (n/+ (n/* row nx) j1)))
                                         (n/* wx2 (aget zs (n/+ (n/* row nx) j2)))
                                         (n/* wx3 (aget zs (n/+ (n/* row nx) j3))))]
                        (recur (inc di) (n/+ sum (n/* wy row-val)))))))))))

;; ================================================================
;; Derivative evaluation for cubic splines
;; ================================================================

(deftm cubic-spline-deriv (All [T] [xs :- (Array T), ys :- (Array T)]
                               (let [n (long (alength xs))
                                     _ (when (< n 3)
                                         (throw (ex-info "cubic-spline-deriv requires at least 3 points" {:n n})))
                                     n-1 (dec n)
                                     h (raster.arrays/alloc-like xs n-1)
                                     _ (dotimes [i n-1]
                                         (aset h i (- (aget xs (inc i)) (aget xs i))))
                                     nm (- n 2)
                                     M (raster.arrays/alloc-like xs n)]
                                 (when (>= nm 1)
                                   (let [sub (raster.arrays/alloc-like xs (max 1 (dec nm)))
                                         diag (raster.arrays/alloc-like xs (max 1 nm))
                                         sup (raster.arrays/alloc-like xs (max 1 (dec nm)))
                                         rhs (raster.arrays/alloc-like xs (max 1 nm))]
                                     (if (== nm 1)
                                       (do
                                         (aset diag 0 (* 2.0 (+ (aget h 0) (aget h 1))))
                                         (aset rhs 0 (* 6.0 (- (/ (- (aget ys 2) (aget ys 1)) (aget h 1))
                                                               (/ (- (aget ys 1) (aget ys 0)) (aget h 0)))))
                                         (aset M 1 (/ (aget rhs 0) (aget diag 0))))
                                       (do
                                         (dotimes [i nm]
                                           (let [j (inc i)]
                                             (aset diag i (* 2.0 (+ (aget h (dec j)) (aget h j))))
                                             (aset rhs i (* 6.0 (- (/ (- (aget ys (inc j)) (aget ys j)) (aget h j))
                                                                   (/ (- (aget ys j) (aget ys (dec j))) (aget h (dec j))))))))
                                         (dotimes [i (dec nm)]
                                           (aset sub i (aget h (inc i)))
                                           (aset sup i (aget h (inc i))))
                                         (solve-tridiagonal! sub diag sup rhs nm)
                                         (dotimes [i nm]
                                           (aset M (inc i) (aget rhs i)))))))
                                 (ftm [x :- Double] :- Double
                                      (let [i (find-interval xs x n)
                                            hi (aget h i)
                                            dx (n/- x (aget xs i))
                                            yi (aget ys i)
                                            yi+1 (aget ys (inc i))
                                            mi (aget M i)
                                            mi+1 (aget M (inc i))
                                            a (n// (n/- mi+1 mi) (n/* 6.0 hi))
                                            b (n/* 0.5 mi)
                                            c-coeff (n/- (n// (n/- yi+1 yi) hi) (n/* hi (n// (n/+ (n/* 2.0 mi) mi+1) 6.0)))]
        ;; S'(x) = c + 2*b*(x-x_i) + 3*a*(x-x_i)^2
                                        (n/+ c-coeff (n/* dx (n/+ (n/* 2.0 b) (n/* 3.0 a dx)))))))))

(ns raster.noise
  "Procedural noise functions — Perlin, FBM, ridged noise.

  All core functions use deftm for typed dispatch and devirtualization.
  Permutation tables are seeded int arrays for deterministic results.

  Usage:
    (def perm (make-perm 42))
    (perlin2d perm 1.5 2.3)        ;; → double in [-1, 1]
    (perlin3d perm 1.5 2.3 0.7)    ;; → double in [-1, 1]
    (fbm2d perm 1.5 2.3 4 0.5 2.0) ;; → multi-octave noise
    (ridged2d perm 1.5 2.3 4 0.5 2.0) ;; → ridged noise (canyons/cliffs)"
  (:require [raster.core :refer [deftm]]
            [raster.numeric :as n])
  (:import [java.util Random]))

;; ================================================================
;; Permutation table
;; ================================================================

(defn make-perm
  "Create a seeded permutation table (int[512]) for noise lookups."
  ^ints [^long seed]
  (let [rng (Random. seed)
        p (int-array 512)]
    (dotimes [i 256] (aset p i (int i)))
    (loop [i 255]
      (when (> i 0)
        (let [j (.nextInt rng (inc i))
              tmp (aget p i)]
          (aset p i (aget p j))
          (aset p j (int tmp)))
        (recur (dec i))))
    (dotimes [i 256] (aset p (+ i 256) (aget p i)))
    p))

;; ================================================================
;; Perlin noise primitives (deftm)
;; ================================================================

(deftm fade
  "Quintic interpolation curve: 6t^5 - 15t^4 + 10t^3"
  [t :- Double] :- Double
  (n/* (n/* (n/* t t) t) (n/+ (n/* t (n/- (n/* t 6.0) 15.0)) 10.0)))

(deftm lerp
  "Linear interpolation."
  [t :- Double, a :- Double, b :- Double] :- Double
  (n/+ a (n/* t (n/- b a))))

(deftm grad2d
  "2D gradient dot product."
  [hash :- Long, x :- Double, y :- Double] :- Double
  (let [h (bit-and hash 3)]
    (case (int h)
      0 (n/+ x y)
      1 (n/+ (n/- x) y)
      2 (n/- x y)
      3 (n/- (n/- x) y))))

(deftm grad3d
  "3D gradient dot product."
  [hash :- Long, x :- Double, y :- Double, z :- Double] :- Double
  (let [h (bit-and hash 15)
        u (if (< h 8) x y)
        v (if (< h 4) y (if (or (= h 12) (= h 14)) x z))]
    (n/+ (if (zero? (bit-and h 1)) u (n/- u))
         (if (zero? (bit-and h 2)) v (n/- v)))))

;; ================================================================
;; Perlin noise 2D / 3D
;; ================================================================

(deftm perlin2d
  "2D Perlin noise, returns value in [-1, 1].
  perm: int[512] permutation table from make-perm."
  [perm :- (Array int), x :- Double, y :- Double] :- Double
  (let [xi (bit-and (long (Math/floor x)) 255)
        yi (bit-and (long (Math/floor y)) 255)
        xf (n/- x (Math/floor x))
        yf (n/- y (Math/floor y))
        u (fade xf)
        v (fade yf)
        aa (long (aget perm (+ (aget perm xi) yi)))
        ab (long (aget perm (+ (aget perm xi) yi 1)))
        ba (long (aget perm (+ (aget perm (+ xi 1)) yi)))
        bb (long (aget perm (+ (aget perm (+ xi 1)) yi 1)))]
    (lerp v
          (lerp u (grad2d aa xf yf) (grad2d ba (n/- xf 1.0) yf))
          (lerp u (grad2d ab xf (n/- yf 1.0)) (grad2d bb (n/- xf 1.0) (n/- yf 1.0))))))

(deftm perlin3d
  "3D Perlin noise, returns value in [-1, 1].
  perm: int[512] permutation table from make-perm."
  [perm :- (Array int), x :- Double, y :- Double, z :- Double] :- Double
  (let [xi (bit-and (long (Math/floor x)) 255)
        yi (bit-and (long (Math/floor y)) 255)
        zi (bit-and (long (Math/floor z)) 255)
        xf (n/- x (Math/floor x))
        yf (n/- y (Math/floor y))
        zf (n/- z (Math/floor z))
        u (fade xf)
        v (fade yf)
        w (fade zf)
        a  (+ (aget perm xi) yi)
        aa (+ (aget perm a) zi)
        ab (+ (aget perm (+ a 1)) zi)
        b  (+ (aget perm (+ xi 1)) yi)
        ba (+ (aget perm b) zi)
        bb (+ (aget perm (+ b 1)) zi)]
    (lerp w
          (lerp v
                (lerp u (grad3d (long (aget perm aa)) xf yf zf)
                      (grad3d (long (aget perm ba)) (n/- xf 1.0) yf zf))
                (lerp u (grad3d (long (aget perm ab)) xf (n/- yf 1.0) zf)
                      (grad3d (long (aget perm bb)) (n/- xf 1.0) (n/- yf 1.0) zf)))
          (lerp v
                (lerp u (grad3d (long (aget perm (+ aa 1))) xf yf (n/- zf 1.0))
                      (grad3d (long (aget perm (+ ba 1))) (n/- xf 1.0) yf (n/- zf 1.0)))
                (lerp u (grad3d (long (aget perm (+ ab 1))) xf (n/- yf 1.0) (n/- zf 1.0))
                      (grad3d (long (aget perm (+ bb 1))) (n/- xf 1.0) (n/- yf 1.0) (n/- zf 1.0)))))))

;; ================================================================
;; Multi-octave noise (FBM, ridged)
;; ================================================================

(deftm fbm2d
  "Fractional Brownian Motion — multi-octave 2D Perlin noise.
  Returns value in approximately [-1, 1]."
  [perm :- (Array int), x :- Double, y :- Double,
   octaves :- Long, persistence :- Double, lacunarity :- Double] :- Double
  (loop [i (long 0) amp 1.0 freq 1.0 sum 0.0 max-amp 0.0]
    (if (= i octaves)
      (n// sum max-amp)
      (recur (inc i)
             (n/* amp persistence)
             (n/* freq lacunarity)
             (n/+ sum (n/* amp (perlin2d perm (n/* x freq) (n/* y freq))))
             (n/+ max-amp amp)))))

(deftm fbm3d
  "Fractional Brownian Motion — multi-octave 3D Perlin noise.
  Returns value in approximately [-1, 1]."
  [perm :- (Array int), x :- Double, y :- Double, z :- Double,
   octaves :- Long, persistence :- Double, lacunarity :- Double] :- Double
  (loop [i (long 0) amp 1.0 freq 1.0 sum 0.0 max-amp 0.0]
    (if (= i octaves)
      (n// sum max-amp)
      (recur (inc i)
             (n/* amp persistence)
             (n/* freq lacunarity)
             (n/+ sum (n/* amp (perlin3d perm (n/* x freq) (n/* y freq) (n/* z freq))))
             (n/+ max-amp amp)))))

(deftm ridged2d
  "Ridged multi-fractal 2D noise — abs(noise) inverted to create ridges.
  Good for mountain ridges, canyons, river channels."
  [perm :- (Array int), x :- Double, y :- Double,
   octaves :- Long, persistence :- Double, lacunarity :- Double] :- Double
  (loop [i (long 0) amp 1.0 freq 1.0 sum 0.0 max-amp 0.0]
    (if (= i octaves)
      (n// sum max-amp)
      (let [n (n/- 1.0 (Math/abs (perlin2d perm (n/* x freq) (n/* y freq))))]
        (recur (inc i)
               (n/* amp persistence)
               (n/* freq lacunarity)
               (n/+ sum (n/* amp (n/* n n)))
               (n/+ max-amp amp))))))

(deftm ridged3d
  "Ridged multi-fractal 3D noise."
  [perm :- (Array int), x :- Double, y :- Double, z :- Double,
   octaves :- Long, persistence :- Double, lacunarity :- Double] :- Double
  (loop [i (long 0) amp 1.0 freq 1.0 sum 0.0 max-amp 0.0]
    (if (= i octaves)
      (n// sum max-amp)
      (let [n (n/- 1.0 (Math/abs (perlin3d perm (n/* x freq) (n/* y freq) (n/* z freq))))]
        (recur (inc i)
               (n/* amp persistence)
               (n/* freq lacunarity)
               (n/+ sum (n/* amp (n/* n n)))
               (n/+ max-amp amp))))))

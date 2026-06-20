(ns valley.core
  "Typed numerical core for Valley — compilable deftm chains.

  This module contains the performance-critical numerical kernels that
  compile-aot can inline into single JVM methods. All functions
  take primitive types (Double, Long, arrays) — no Clojure maps.

  The game layer (mobs.clj, hostile.clj, game.clj) extracts typed data
  from Clojure maps, calls these kernels, and reassembles state.

  Domains:
    - Block queries (solid?, transparent? on raw arrays)
    - Physics (gravity, collision, velocity integration)
    - Geometry (distance, direction, rotation, ray intersection)
    - Terrain (biome classification, height sampling, cave detection)"
  (:require [raster.core :refer [deftm]]
            [raster.linalg.core :as lin :refer [->Vec2 ->Vec3 ->Vec4]]
            [raster.noise :as noise])
  (:import [raster.linalg.core Vec2 Vec3 Vec4]))

;; ================================================================
;; Constants — def ^:const inlined by the bytecode compiler
;; ================================================================

(def ^:const GRAVITY 20.0)
(def ^:const SEA-LEVEL 48)
(def ^:const BASE-HEIGHT 52)
(def ^:const TERRAIN-FREQ (/ 1.0 100.0))
(def ^:const BIOME-FREQ (/ 1.0 256.0))
(def ^:const CAVE-FREQ (/ 1.0 40.0))
(def ^:const CAVE-THRESHOLD 0.55)

;; ================================================================
;; Block queries — thin deftm wrappers over property arrays
;; These are the leaves of every physics/meshing chain.
;; ================================================================

;; Block property arrays are byte arrays (0/1) instead of boolean arrays,
;; because the bytecode compiler handles byte arrays natively.
;; Call (valley.typed-core/make-solid-bytes solid-boolean-arr) to convert.

(defn make-solid-bytes
  "Convert boolean property array to byte array for deftm use."
  ^bytes [^booleans bool-arr]
  (let [n (alength bool-arr)
        out (byte-array n)]
    (dotimes [i n] (aset out i (byte (if (aget bool-arr i) 1 0))))
    out))

(deftm block-solid?
  "Is block ID solid? Reads from pre-computed byte array (1=solid, 0=not)."
  [solid-arr :- (Array byte), block-id :- Long] :- Long
  (if (and (>= block-id 0) (<= block-id 61))
    (long (aget solid-arr (int block-id)))
    0))

(deftm block-transparent?
  "Is block ID transparent?"
  [transparent-arr :- (Array byte), block-id :- Long] :- Long
  (if (and (>= block-id 0) (<= block-id 61))
    (long (aget transparent-arr (int block-id)))
    1))

;; ================================================================
;; Chunk block access — typed array indexing
;; ================================================================

(deftm chunk-block
  "Get block ID at local chunk coordinates (16×16×16 cubes). Returns 0 (AIR) if out of bounds."
  [blocks :- (Array int), x :- Long, y :- Long, z :- Long] :- Long
  (if (and (>= x 0) (< x 16) (>= y 0) (< y 16) (>= z 0) (< z 16))
    (long (aget blocks (int (+ x (* z 16) (* y 256)))))
    0))

(deftm world-block-solid?
  "Check if world-coordinate block is solid. Needs chunk blocks + chunk origin."
  [blocks :- (Array int), solid-arr :- (Array byte),
   wx :- Long, wy :- Long, wz :- Long,
   cx :- Long, cy :- Long, cz :- Long] :- Long
  (let [lx (- wx (* cx 16))
        ly (- wy (* cy 16))
        lz (- wz (* cz 16))
        bid (chunk-block blocks lx ly lz)]
    (block-solid? solid-arr bid)))

;; ================================================================
;; Physics — gravity, AABB collision, velocity integration
;; ================================================================

(deftm aabb-collides?
  "Check if AABB at (x,y,z) with half-width hw and height h collides
  with any solid block in the chunk. y = feet position."
  [blocks :- (Array int), solid-arr :- (Array byte),
   x :- Double, y :- Double, z :- Double,
   hw :- Double, h :- Double,
   cx :- Long, cy :- Long, cz :- Long] :- Long
  (let [r (* hw 0.5)
        bx0 (long (Math/floor (- x r)))
        bx1 (long (Math/floor (+ x r)))
        bz0 (long (Math/floor (- z r)))
        bz1 (long (Math/floor (+ z r)))
        by0 (long (Math/floor y))
        by1 (long (Math/floor (+ y h -0.01)))]
    (loop [bx bx0]
      (if (> bx bx1) 0
          (if (pos? (loop [bz bz0]
                      (if (> bz bz1) 0
                          (if (pos? (loop [by by0]
                                      (if (> by by1) 0
                                          (if (pos? (world-block-solid? blocks solid-arr
                                                                        bx by bz cx cy cz))
                                            1
                                            (recur (+ by 1))))))
                            1
                            (recur (+ bz 1))))))
            1
            (recur (+ bx 1)))))))

(deftm ground-check
  "Check if there's a solid block directly below feet.
  Returns the block-top Y if grounded, or -1.0 if not."
  [blocks :- (Array int), solid-arr :- (Array byte),
   x :- Double, y :- Double, z :- Double,
   cx :- Long, cy :- Long, cz :- Long] :- Double
  (let [foot-y (long (Math/floor (- y 0.01)))
        bx (long (Math/floor x))
        bz (long (Math/floor z))]
    (if (pos? (world-block-solid? blocks solid-arr bx foot-y bz cx cy cz))
      (double (+ foot-y 1))
      -1.0)))

(deftm integrate-physics!
  "Full physics integration step. Mutates pos and vel arrays.
  pos = [x y z], vel = [vx vy vz].
  Returns: 1 if on ground, 0 if airborne."
  [pos :- (Array double), vel :- (Array double),
   blocks :- (Array int), solid-arr :- (Array byte),
   cx :- Long, cy :- Long, cz :- Long,
   hw :- Double, h :- Double,
   dx :- Double, dz :- Double,
   dt :- Double] :- Long
  (let [mx (aget pos 0) my (aget pos 1) mz (aget pos 2)
        vy (aget vel 1)
        ;; Gravity
        vy (- vy (* GRAVITY dt))
        ;; Vertical movement
        new-my (+ my (* vy dt))
        ;; Ground check
        ground-y (ground-check blocks solid-arr mx new-my mz cx cy cz)
        on-ground? (> ground-y -0.5)
        new-my (if (and on-ground? (< vy 0.0)) ground-y new-my)
        vy (if (and on-ground? (< vy 0.0)) 0.0 vy)
        ;; Horizontal with 1-block step-up: try X then Z; if a move is blocked by a
        ;; single-block rise that has headroom above it, climb the step (grounded only).
        ;; (aabb-collides? returns 1 when blocked, 0 when clear — `(< _ 1)` means clear.)
        try-mx (+ mx dx)
        coll-x (aabb-collides? blocks solid-arr try-mx new-my mz hw h cx cy cz)
        step-x? (and on-ground? (pos? coll-x)
                     (< (aabb-collides? blocks solid-arr try-mx (+ new-my 1.0) mz hw h cx cy cz) 1))
        new-mx (if (pos? coll-x) (if step-x? try-mx mx) try-mx)
        my-x (if step-x? (+ new-my 1.0) new-my)
        try-mz (+ mz dz)
        coll-z (aabb-collides? blocks solid-arr new-mx my-x try-mz hw h cx cy cz)
        step-z? (and on-ground? (pos? coll-z)
                     (< (aabb-collides? blocks solid-arr new-mx (+ my-x 1.0) try-mz hw h cx cy cz) 1))
        new-mz (if (pos? coll-z) (if step-z? try-mz mz) try-mz)
        my-final (if step-z? (+ my-x 1.0) my-x)]
    ;; Write back
    (aset pos 0 new-mx)
    (aset pos 1 my-final)
    (aset pos 2 new-mz)
    (aset vel 1 vy)
    (if on-ground? 1 0)))

;; ================================================================
;; Geometry — distance, direction, rotation, ray intersection
;; ================================================================

(deftm dist-xz
  "Horizontal distance between two XZ points."
  [x1 :- Double, z1 :- Double, x2 :- Double, z2 :- Double] :- Double
  (let [dx (- x2 x1) dz (- z2 z1)]
    (Math/sqrt (+ (* dx dx) (* dz dz)))))

(deftm dist-3d
  "3D Euclidean distance."
  [x1 :- Double, y1 :- Double, z1 :- Double,
   x2 :- Double, y2 :- Double, z2 :- Double] :- Double
  (let [dx (- x2 x1) dy (- y2 y1) dz (- z2 z1)]
    (Math/sqrt (+ (* dx dx) (* dy dy) (* dz dz)))))

(deftm normalize-xz
  "Normalize XZ direction. Returns a Vec2 (allocation-free value type — the wasm
   backend lowers it to a multi-value return; the JVM scalarizes it)."
  [dx :- Double, dz :- Double] :- Vec2
  (let [len (Math/sqrt (+ (* dx dx) (* dz dz)))]
    ;; scalar `if` in each component keeps the constructor a single value-tail
    (->Vec2 (if (> len 0.001) (/ dx len) 0.0)
            (if (> len 0.001) (/ dz len) 1.0))))

(deftm dir-toward-xz
  "Normalized XZ direction from (mx,mz) toward (tx,tz). Returns [dx dz] vector."
  [mx :- Double, mz :- Double, tx :- Double, tz :- Double]
  (let [dx (- tx mx) dz (- tz mz)
        len (Math/sqrt (+ (* dx dx) (* dz dz)))]
    (if (> len 0.001)
      [(/ dx len) (/ dz len)]
      [0.0 1.0])))

(deftm rotate-x
  "Rotate (x,y,z) around pivot (px,py,pz) by angle on X axis. Returns a Vec3
   (allocation-free value type — see normalize-xz)."
  [x :- Double, y :- Double, z :- Double,
   px :- Double, py :- Double, pz :- Double, angle :- Double] :- Vec3
  (let [dy (- y py) dz (- z pz)
        c (Math/cos angle) s (Math/sin angle)]
    (->Vec3 x
            (+ py (- (* c dy) (* s dz)))
            (+ pz (+ (* s dy) (* c dz))))))

(deftm rotate-y
  "Rotate (x,y,z) around pivot (px,py,pz) by angle on Y axis. Returns a Vec3
   (allocation-free value type — see normalize-xz)."
  [x :- Double, y :- Double, z :- Double,
   px :- Double, py :- Double, pz :- Double, angle :- Double] :- Vec3
  (let [dx (- x px) dz (- z pz)
        c (Math/cos angle) s (Math/sin angle)]
    (->Vec3 (+ px (+ (* c dx) (* s dz)))
            y
            (+ pz (- (* c dz) (* s dx))))))

(deftm ray-intersects-aabb?
  "Ray-AABB intersection test. Returns 1 if ray hits box, 0 if not.
  Ray: origin (ox,oy,oz) + direction (dx,dy,dz), max distance max-d.
  Box: min (bx0,by0,bz0) max (bx1,by1,bz1)."
  [ox :- Double, oy :- Double, oz :- Double,
   dx :- Double, dy :- Double, dz :- Double,
   max-d :- Double,
   bx0 :- Double, by0 :- Double, bz0 :- Double,
   bx1 :- Double, by1 :- Double, bz1 :- Double] :- Long
  (let [;; Inverse direction (avoid division in slab tests)
        idx (if (> (Math/abs dx) 1e-10) (/ 1.0 dx) (if (>= dx 0) 1e10 -1e10))
        idy (if (> (Math/abs dy) 1e-10) (/ 1.0 dy) (if (>= dy 0) 1e10 -1e10))
        idz (if (> (Math/abs dz) 1e-10) (/ 1.0 dz) (if (>= dz 0) 1e10 -1e10))
        tx1 (* (- bx0 ox) idx) tx2 (* (- bx1 ox) idx)
        ty1 (* (- by0 oy) idy) ty2 (* (- by1 oy) idy)
        tz1 (* (- bz0 oz) idz) tz2 (* (- bz1 oz) idz)
        tmin (Math/max (Math/max (Math/min tx1 tx2) (Math/min ty1 ty2)) (Math/min tz1 tz2))
        tmax (Math/min (Math/min (Math/max tx1 tx2) (Math/max ty1 ty2)) (Math/max tz1 tz2))]
    (if (and (<= tmin tmax) (>= tmax 0.0) (<= tmin max-d))
      1 0)))

;; ================================================================
;; Terrain generation — biome, height, caves
;; All noise-based, compiles with raster.noise deftm chain.
;; ================================================================

(deftm biome-climate
  "Compute temperature and humidity at world (wx, wz). Returns a Vec4 of
   [temp humid erosion mushroom] (allocation-free value type — see normalize-xz)."
  [temp-perm :- (Array int), humid-perm :- (Array int),
   detail-perm :- (Array int), mush-perm :- (Array int),
   wx :- Double, wz :- Double] :- Vec4
  (let [temp (+ 0.5 (* 0.5 (noise/fbm2d temp-perm
                                        (* wx BIOME-FREQ) (* wz BIOME-FREQ)
                                        3 0.5 2.0)))
        humid (+ 0.5 (* 0.5 (noise/fbm2d humid-perm
                                         (* wx BIOME-FREQ) (* wz BIOME-FREQ)
                                         3 0.5 2.0)))
        erosion (noise/fbm2d detail-perm
                             (* wx (/ 1.0 180.0)) (* wz (/ 1.0 180.0))
                             2 0.5 2.0)
        mush (noise/perlin2d mush-perm
                             (* wx (/ 1.0 400.0)) (* wz (/ 1.0 400.0)))]
    (->Vec4 temp humid erosion mush)))

(deftm terrain-height
  "Get terrain surface height at world (wx, wz).
  height-scale and base-offset are biome-specific parameters."
  [height-perm :- (Array int), detail-perm :- (Array int),
   wx :- Double, wz :- Double,
   height-scale :- Double, base-offset :- Double] :- Long
  (let [h (noise/fbm2d height-perm
                       (* wx TERRAIN-FREQ) (* wz TERRAIN-FREQ)
                       4 0.5 2.0)
        detail (noise/fbm2d detail-perm
                            (* wx (/ 1.0 20.0)) (* wz (/ 1.0 20.0))
                            2 0.6 2.0)]
    (long (+ (double BASE-HEIGHT) base-offset (* h height-scale) (* detail 4.0)))))

(deftm has-cave?
  "Check if world position (wx,wy,wz) is carved out by cave noise."
  [cave-perm :- (Array int), wx :- Double, wy :- Double, wz :- Double] :- Long
  (let [v (noise/fbm3d cave-perm
                       (* wx CAVE-FREQ) (* wy CAVE-FREQ) (* wz CAVE-FREQ)
                       2 0.5 2.0)]
    (if (> (Math/abs v) CAVE-THRESHOLD) 1 0)))

(deftm has-lava-pool?
  "Check if position should have a lava pool (underground, noise-based)."
  [cave-perm :- (Array int), wx :- Double, wy :- Double, wz :- Double] :- Long
  (if (> wy 20.0) 0
      (let [v (noise/perlin3d cave-perm
                              (* wx (/ 1.0 25.0)) (* wy (/ 1.0 15.0)) (* wz (/ 1.0 25.0)))]
        (if (> v 0.7) 1 0))))

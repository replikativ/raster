(ns valley.swarm
  "Cross-platform mob layer: resident SoA columns + batch physics + wander AI + a
   dynamic cube mesh. Identical code on JVM (batch kernel = bytecode, arrays used
   directly) and browser (batch kernel = wasm, world resident in linear memory via
   k/upload-world!). This is the optimal state model (design/state-model.md) made
   visible: hot mob state lives in columns the kernel sweeps in one call each frame."
  (:require [valley.chunk :as chunk]
            [valley.slice :as slice]
            [valley.kernels :as k]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))

(def ^:const SPEED 2.5)     ; blocks / second
(def ^:const HALF-W 0.6)
(def ^:const HEIGHT 1.4)
(def ^:const MOB-LAYER 3.0) ; 4th atlas layer (see atlas-pixels)

(defn spawn
  "N mobs scattered on the world surface. Uploads the resident world to the kernel
   (JVM no-op; browser writes the block array into wasm memory once)."
  [world n]
  (k/upload-world! (:blocks world) (:solid world))
  (let [pos (chunk/darray (* n 3)) vel (chunk/darray (* n 3))
        dxs (chunk/darray n) dzs (chunk/darray n)
        WX (long (:wx world)) WZ (long (:wz world))]
    (dotimes [i n]
      (let [x (+ 2 (mod (* i 7) (max 1 (- WX 4))))
            z (+ 2 (mod (* i 13) (max 1 (- WZ 4))))
            top (loop [y (dec (long (:wy world)))]
                  (cond (< y 0) 0
                        (pos? (chunk/world-block world x y z)) (inc y)
                        :else (recur (dec y))))]
        (aset pos (* i 3) (+ x 0.5))
        (aset pos (+ (* i 3) 1) (+ (double top) 2.0))   ; spawn above → fall onto terrain
        (aset pos (+ (* i 3) 2) (+ z 0.5))))
    {:pos pos :vel vel :dxs dxs :dzs dzs :n n :t 0.0 :world world}))

(defn update!
  "Wander AI (each mob heads in a slowly-rotating per-mob direction) + one batch
   physics call over all mobs. Mutates the columns in place."
  [mobs dt]
  (let [{:keys [pos vel dxs dzs n world]} mobs
        t (+ (:t mobs) dt)]
    (dotimes [i n]
      (let [h (+ (* t 0.4) (* i 1.7))]
        (aset dxs i (* (cos h) SPEED dt))
        (aset dzs i (* (sin h) SPEED dt))))
    (k/integrate-physics-batch! pos vel dxs dzs (:blocks world) (:solid world)
                                n (:wx world) (:wy world) (:wz world) HALF-W HEIGHT dt)
    (assoc mobs :t t)))

;; --- dynamic cube mesh (triangle list for the vertex-only dynamic mesh) -----------
(def ^:private faces
  [[[0 0 1] [1 0 1] [1 1 1] [0 1 1]]
   [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]
   [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]
   [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]
   [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]
   [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]])
(def ^:private face-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])
(def ^:const VERTS-PER-MOB 36)   ; 6 faces × 2 triangles × 3 verts

(defn verts
  "Flat float seq of all mob cubes (triangle list) for upload to a dynamic mesh.
   Each mob is a HALF-W-wide, HEIGHT-tall box at its feet position, on MOB-LAYER."
  [mobs]
  (let [{:keys [pos n]} mobs
        out (volatile! (transient []))]
    (dotimes [i n]
      (let [px (aget pos (* i 3)) py (aget pos (+ (* i 3) 1)) pz (aget pos (+ (* i 3) 2))
            x0 (- px (* 0.5 HALF-W)) z0 (- pz (* 0.5 HALF-W))
            vert (fn [face ci]
                   (let [c (nth face ci) [u v] (nth face-uv ci)]
                     [(+ x0 (* (nth c 0) HALF-W)) (+ py (* (nth c 1) HEIGHT)) (+ z0 (* (nth c 2) HALF-W))
                      u v MOB-LAYER]))]
        (doseq [f faces]
          (vswap! out (fn [t] (reduce conj! t (concat (vert f 0) (vert f 1) (vert f 2)
                                                      (vert f 0) (vert f 2) (vert f 3))))))))
    (persistent! @out)))

;; --- 4-layer atlas: the 3 terrain layers (grass/dirt/stone) + a red mob layer ----
(defn- mob-pixels [w h]
  (mapcat (fn [i] (let [n (- (mod (* (+ i 1) 2654435761) 40) 20)]
                    [(max 0 (min 255 (+ 200 n))) (max 0 (min 255 (+ 60 n))) (max 0 (min 255 (+ 60 n))) 255]))
          (range (* w h))))

(defn atlas-pixels
  "Terrain atlas (3 layers) + a 4th mob layer — flat RGBA8, layer-major."
  [w h]
  (vec (concat (slice/atlas-pixels w h) (mob-pixels w h))))

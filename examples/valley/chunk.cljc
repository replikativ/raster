(ns valley.chunk
  "Portable single-chunk model for the valley walk slice (Phase 2b): a 16³ block
   array generated from valley.kernels/terrain-height, a face mesh of it, and the
   block-property table the physics kernels read. Cross-platform array helpers keep
   the JVM (Java primitive arrays) and browser (typed arrays in wasm memory) on the
   same code. The player physics (integrate-physics!) runs against this chunk via
   the valley.kernels facade."
  (:require [valley.kernels :as k]))

(def ^:const CS 16)                       ; chunk size (valley CHUNK-SIZE)
(def ^:const STRIDE 24)                   ; pos3 + uv2 + layer (matches valley.slice)

;; --- cross-platform numeric arrays -------------------------------------------
(defn iarray [n] #?(:clj (int-array n)    :cljs (js/Int32Array. n)))
(defn barray [n] #?(:clj (byte-array n)   :cljs (js/Int8Array. n)))
(defn darray [n] #?(:clj (double-array n) :cljs (js/Float64Array. n)))
;; clojure.core aget/aset work on both Java arrays and JS typed arrays.

(defn idx ^long [^long x ^long y ^long z] (+ x (* z CS) (* y CS CS)))

;; block ids: 0 air, 1 grass, 2 dirt, 3 stone. solid table = 62 bytes (1=solid).
(defn solid-table []
  (let [s (barray 62)]
    (aset s 1 #?(:clj (byte 1) :cljs 1))
    (aset s 2 #?(:clj (byte 1) :cljs 1))
    (aset s 3 #?(:clj (byte 1) :cljs 1))
    s))

(defn gen-chunk
  "A self-contained 16³ chunk: per column the surface height is terrain-height
   remapped into 4..13 (so the terrain fits inside one chunk), filled grass/dirt/
   stone below. Returns the int block array."
  []
  (let [b (iarray (* CS CS CS))]
    (dotimes [x CS]
      (dotimes [z CS]
        (let [h (+ 4 (mod (k/terrain-height x z) 10))]   ; 4..13
          (dotimes [y CS]
            (when (< y h)
              (aset b (idx x y z)
                    (cond (= y (dec h)) 1 (>= y (- h 3)) 2 :else 3)))))))
    b))

(defn block-at ^long [blocks ^long x ^long y ^long z]
  (if (and (>= x 0) (< x CS) (>= y 0) (< y CS) (>= z 0) (< z CS))
    (long (aget blocks (idx x y z)))
    0))

;; --- face mesh of the chunk (interior faces culled) --------------------------
(def ^:private faces
  [[[0 0 1]  [[0 0 1] [1 0 1] [1 1 1] [0 1 1]]]
   [[0 0 -1] [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]]
   [[1 0 0]  [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]]
   [[-1 0 0] [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]]
   [[0 1 0]  [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]]
   [[0 -1 0] [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]]])
(def ^:private face-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])

;; --- multi-chunk world (option C): a flat block grid spanning many chunks --------
;; The physics kernel only ever sees ONE 16³ chunk; for a world larger than that we
;; stitch a player-centred 16³ window each frame (the tiny player AABB can never
;; leave a centred window) and call the kernel against it at origin 0. This keeps the
;; wasm kernel + marshaling identical to the single-chunk path — only the host stitch
;; and a pos translation differ. World coords index a flat array: wx + wz*WX + wy*WX*WZ.

(defn world-idx [wx wy wz wx-dim wz-dim]   ; >4 args ⇒ no primitive hints (params or return)
  (+ (long wx) (* (long wz) (long wx-dim)) (* (long wy) (long wx-dim) (long wz-dim))))

(defn gen-world
  "A flat cw×ch×cd-chunk world generated from valley.kernels/terrain-height over
   continuous world coords (so terrain flows across chunk seams — the thing C must
   get right). Returns {:blocks :wx :wy :wz :solid}."
  [^long cw ^long ch ^long cd]
  (let [wx (* cw CS) wy (* ch CS) wz (* cd CS)
        b  (iarray (* wx wy wz))
        base 4
        ;; biome-aware height (cf valley.terrain): smooth, offset by the world min and
        ;; CLAMP into [base, wy) — keeps the profile climbable by the 1-block step-up.
        raw  (vec (for [x (range wx) z (range wz)] (long (k/surface-height-biome x z))))
        minh (reduce min raw)]
    (dotimes [x wx]
      (dotimes [z wz]
        (let [h (max base (min (dec wy) (+ base (- (nth raw (+ (* x wz) z)) minh))))]
          (dotimes [y wy]
            (when (< y h)
              (aset b (world-idx x y z wx wz)
                    (cond (= y (dec h)) 1 (>= y (- h 3)) 2 :else 3)))))))
    {:blocks b :wx wx :wy wy :wz wz :solid (solid-table)}))

(defn gen-flat-world
  "A flat cw×ch×cd-chunk world: solid up to `surface` (grass top, dirt, stone),
   walkable across all chunk seams without step-up. The walker milestone uses this;
   gen-world (real terrain) becomes walkable once the physics gains 1-block step-up."
  [cw ch cd surface]
  (let [wx (* cw CS) wy (* ch CS) wz (* cd CS)
        b  (iarray (* wx wy wz)) s (long surface)]
    (dotimes [x wx]
      (dotimes [z wz]
        (dotimes [y (min s wy)]
          (aset b (world-idx x y z wx wz)
                (cond (= y (dec s)) 1 (>= y (- s 3)) 2 :else 3)))))
    {:blocks b :wx wx :wy wy :wz wz :solid (solid-table)}))

(defn world-block ^long [world ^long wx ^long wy ^long wz]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))]
    (if (and (>= wx 0) (< wx WX) (>= wy 0) (< wy WY) (>= wz 0) (< wz WZ))
      (long (aget (:blocks world) (world-idx wx wy wz WX WZ)))
      0)))

(defn stitch-window!
  "Fill a 16³ scratch int array from `world` centred at world origin (wox,woy,woz)
   — scratch local (i,j,k) ← world (wox+i, woy+j, woz+k). Returns scratch."
  [world scratch wox woy woz]   ; >4 params ⇒ no primitive hints
  (let [wox (long wox) woy (long woy) woz (long woz)]
    (dotimes [j CS]
      (dotimes [k CS]
        (dotimes [i CS]
          (aset scratch (idx i j k)
                #?(:clj (int (world-block world (+ wox i) (+ woy j) (+ woz k)))
                   :cljs (world-block world (+ wox i) (+ woy j) (+ woz k))))))))
  scratch)

(defn mesh-chunk
  "{:verts [floats] :indices [ints]} face mesh of the chunk."
  [blocks]
  (let [vb (volatile! (transient [])) ib (volatile! (transient [])) vc (volatile! 0)]
    (doseq [x (range CS) y (range CS) z (range CS)]
      (let [id (block-at blocks x y z)]
        (when (pos? id)
          (let [layer (double (dec id))]
            (doseq [[[nx ny nz] corners] faces]
              (when (zero? (block-at blocks (+ x nx) (+ y ny) (+ z nz)))
                (let [base @vc]
                  (dotimes [ci 4]
                    (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)]
                      (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox)) (double (+ y oy)) (double (+ z oz)) u v layer])))))
                  (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                  (vswap! vc + 4))))))))
    {:verts (persistent! @vb) :indices (persistent! @ib)}))

(defn mesh-world
  "Face mesh of a whole multi-chunk world (interior faces culled across chunk seams,
   since culling uses world-block, not per-chunk bounds). {:verts :indices}."
  [world]
  (let [WX (long (:wx world)) WY (long (:wy world)) WZ (long (:wz world))
        vb (volatile! (transient [])) ib (volatile! (transient [])) vc (volatile! 0)]
    (doseq [x (range WX) y (range WY) z (range WZ)]
      (let [id (world-block world x y z)]
        (when (pos? id)
          (let [layer (double (dec id))]
            (doseq [[[nx ny nz] corners] faces]
              (when (zero? (world-block world (+ x nx) (+ y ny) (+ z nz)))
                (let [base @vc]
                  (dotimes [ci 4]
                    (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)]
                      (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox)) (double (+ y oy)) (double (+ z oz)) u v layer])))))
                  (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                  (vswap! vc + 4))))))))
    {:verts (persistent! @vb) :indices (persistent! @ib)}))

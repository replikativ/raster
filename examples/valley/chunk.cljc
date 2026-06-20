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

(ns valley.slice
  "Portable valley terrain slice (.cljc): heightmap from valley.kernels/terrain-height
   (deftm→bytecode on JVM, wasm in the browser) + a face mesher + a block atlas,
   rendered through raster.render. Shared by the JVM (Vulkan) and browser (WebGPU)
   shells — the first cross-platform slice of valley, valley's chunk→mesh→draw path
   in miniature (cf. voxel.world).

   This is intentionally separate from valley's full JVM chunk model (world.clj):
   the slice renders terrain from scalar heightmap queries (no chunk storage), so it
   needs no blocks in wasm memory. The in-place port of world.clj (chunk storage for
   the physics kernels) is a later phase; this proves terrain + render end-to-end on
   both platforms first."
  (:require [raster.render :as r]
            [valley.kernels :as k]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- tan [x] (#?(:clj Math/tan :cljs js/Math.tan) x))
(defn- sqrt [x] (#?(:clj Math/sqrt :cljs js/Math.sqrt) x))

(def ^:const W 32)         ; world is W×W columns
(def ^:const STRIDE 24)    ; pos3 (12) + uv2 (8) + layer (4)

;; --- column-major 4x4 (m[col*4+row]) -----------------------------------------
(defn- mat-mul [a b]
  (vec (for [c (range 4) r (range 4)]
         (+ (* (nth a (+ r 0)) (nth b (+ (* c 4) 0)))
            (* (nth a (+ r 4)) (nth b (+ (* c 4) 1)))
            (* (nth a (+ r 8)) (nth b (+ (* c 4) 2)))
            (* (nth a (+ r 12)) (nth b (+ (* c 4) 3)))))))

(defn- perspective [fovy aspect zn zf]
  (let [f (/ 1.0 (tan (* 0.5 fovy)))]            ; Vulkan-style: Y-down, depth [0,1]
    [(/ f aspect) 0.0 0.0 0.0
     0.0 (- f) 0.0 0.0
     0.0 0.0 (/ zf (- zn zf)) -1.0
     0.0 0.0 (/ (* zn zf) (- zn zf)) 0.0]))

(defn- v- [a b] [(- (a 0) (b 0)) (- (a 1) (b 1)) (- (a 2) (b 2))])
(defn- cross [a b] [(- (* (a 1) (b 2)) (* (a 2) (b 1)))
                    (- (* (a 2) (b 0)) (* (a 0) (b 2)))
                    (- (* (a 0) (b 1)) (* (a 1) (b 0)))])
(defn- dot [a b] (+ (* (a 0) (b 0)) (* (a 1) (b 1)) (* (a 2) (b 2))))
(defn- norm [a] (let [l (sqrt (dot a a))] [(/ (a 0) l) (/ (a 1) l) (/ (a 2) l)]))

(defn- look-at [eye center up]
  (let [f (norm (v- center eye)) s (norm (cross f up)) u (cross s f)]
    [(s 0) (u 0) (- (f 0)) 0.0
     (s 1) (u 1) (- (f 1)) 0.0
     (s 2) (u 2) (- (f 2)) 0.0
     (- (dot s eye)) (- (dot u eye)) (dot f eye) 1.0]))

;; --- terrain → blocks --------------------------------------------------------
(defn- surface ^long [x z] (k/terrain-height x z))

(defn heightmap []
  (let [hm (vec (for [x (range W)] (vec (for [z (range W)] (surface x z)))))
        flat (mapcat identity hm)]
    {:hm hm :lo (reduce min flat) :hi (reduce max flat)}))

(defn mvp
  "Orbiting camera around the terrain centre at its mean height."
  [t aspect lo hi]
  (let [cx (* 0.5 W) cz (* 0.5 W) mid (* 0.5 (+ lo hi)) rad (* 1.4 W)
        eye [(+ cx (* rad (cos (* 0.25 t)))) (+ hi 18.0) (+ cz (* rad (sin (* 0.25 t))))]]
    (mat-mul (perspective 0.9 aspect 0.1 400.0)
             (look-at eye [cx (double mid) cz] [0.0 1.0 0.0]))))

(defn- id-at
  "Block id from a heightmap: 0 air, 1 grass (surface), 2 dirt (≤3 below), 3 stone."
  [hm x z y]
  (if (and (>= x 0) (< x W) (>= z 0) (< z W) (>= y 0))
    (let [h (nth (nth hm x) z)]
      (cond (> y h) 0, (= y h) 1, (>= y (- h 3)) 2, :else 3))
    0))

(def ^:private block-faces
  [[[0 0 1]  [[0 0 1] [1 0 1] [1 1 1] [0 1 1]]]
   [[0 0 -1] [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]]
   [[1 0 0]  [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]]
   [[-1 0 0] [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]]
   [[0 1 0]  [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]]
   [[0 -1 0] [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]]])
(def ^:private face-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])

(defn mesh-terrain
  "Face mesh of the terrain with interior faces culled.
   {:verts [floats] :indices [ints] :lo :hi}."
  []
  (let [{:keys [hm lo hi]} (heightmap)
        vb (volatile! (transient []))
        ib (volatile! (transient []))
        vc (volatile! 0)]
    (doseq [x (range W) z (range W) y (range 0 (inc hi))]
      (let [id (id-at hm x z y)]
        (when (pos? id)
          (let [layer (double (dec id))]
            (doseq [[[nx ny nz] corners] block-faces]
              (when (zero? (id-at hm (+ x nx) (+ z nz) (+ y ny)))
                (let [base @vc]
                  (dotimes [ci 4]
                    (let [[ox oy oz] (nth corners ci) [u v] (nth face-uv ci)]
                      (vswap! vb (fn [t] (reduce conj! t [(double (+ x ox)) (double (+ y oy)) (double (+ z oz)) u v layer])))))
                  (vswap! ib (fn [t] (reduce conj! t [base (+ base 1) (+ base 2) base (+ base 2) (+ base 3)])))
                  (vswap! vc + 4))))))))
    {:verts (persistent! @vb) :indices (persistent! @ib) :lo lo :hi hi}))

;; --- 3-layer block atlas (grass / dirt / stone), flat RGBA8, layer-major ------
(defn- noisy [r g b i]
  (let [n (- (mod (* (+ i 1) 2654435761) 48) 24)]
    [(max 0 (min 255 (+ r n))) (max 0 (min 255 (+ g n))) (max 0 (min 255 (+ b n))) 255]))
(defn atlas-pixels [w h]
  (vec (concat
        (mapcat #(noisy 70 150 70 %) (range (* w h)))      ; layer 0 grass
        (mapcat #(noisy 134 96 67 %) (range (* w h)))      ; layer 1 dirt
        (mapcat #(noisy 128 128 130 %) (range (* w h)))))) ; layer 2 stone

;; --- shader (one description → GLSL + WGSL) ----------------------------------
(def shader
  '{:uniform        {:name "U" :fields [[:mvp :mat4]]}
    :attributes     [[inPos vec3 0] [inUv vec2 1] [inLayer float 2]]
    :varyings       [[vUv vec2 0] [vLayer float 1]]
    :texture-arrays [[atlas 0]]
    :vertex         [(set-position (* mvp (vec4 inPos 1.0))) (out vUv inUv) (out vLayer inLayer)]
    :fragment       [(color (sample-layer atlas vUv vLayer))]})

(def pipeline-spec
  {:shader shader :topology :triangles :depth? true :cull :none
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}
                                        {:location 1 :format :float2 :offset 12}
                                        {:location 2 :format :float :offset 20}]}})

(defn draw-terrain!
  "lo/hi are the terrain height bounds from mesh-terrain (for the camera) — the GPU
   `mesh` handle doesn't carry them."
  [_rnd frame pipeline mesh tex t aspect lo hi]
  (r/bind-pipeline! frame pipeline)
  (r/set-uniform! frame pipeline (mvp t aspect lo hi))
  (r/bind-texture! frame pipeline tex)
  (r/bind-mesh! frame mesh)
  (r/draw-indexed! frame (:index-count mesh) 0))

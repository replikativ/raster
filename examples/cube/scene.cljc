(ns cube.scene
  "R4 validation scene: a textured, depth-tested spinning cube — the 3D equivalent
   of the asteroids slice. Shared (.cljc) by the JVM (Vulkan) and browser (WebGPU)
   shells; only the renderer backend differs. Exercises every R4 addition at once:
   an indexed static mesh, a 2D texture + sampler, a depth attachment, and a mat4
   uniform. The mat4 math is plain portable Clojure (column-major), built for the
   Vulkan convention (Y-down, depth [0,1]); the WGSL emitter's Y-flip makes the
   same matrix correct on WebGPU."
  (:require [raster.render :as r]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- tan [x] (#?(:clj Math/tan :cljs js/Math.tan) x))

(def ^:const STRIDE 20)        ; pos3 (12) + uv2 (8)

;; --- column-major 4x4 (vector of 16: m[col*4+row]) ----------------------------
(defn- mat-mul [a b]
  (vec (for [c (range 4) r (range 4)]
         (+ (* (nth a (+ r 0))  (nth b (+ (* c 4) 0)))
            (* (nth a (+ r 4))  (nth b (+ (* c 4) 1)))
            (* (nth a (+ r 8))  (nth b (+ (* c 4) 2)))
            (* (nth a (+ r 12)) (nth b (+ (* c 4) 3)))))))

(defn- perspective [fovy aspect zn zf]
  ;; Vulkan-style: Y-down (−f), depth → [0,1], right-handed looking down −z
  (let [f (/ 1.0 (tan (* 0.5 fovy)))]
    [(/ f aspect) 0.0 0.0 0.0
     0.0 (- f) 0.0 0.0
     0.0 0.0 (/ zf (- zn zf)) -1.0
     0.0 0.0 (/ (* zn zf) (- zn zf)) 0.0]))

(defn- translate [x y z]
  [1.0 0.0 0.0 0.0, 0.0 1.0 0.0 0.0, 0.0 0.0 1.0 0.0, x y z 1.0])

(defn- rot-y [a] (let [c (cos a) s (sin a)]
                   [c 0.0 (- s) 0.0, 0.0 1.0 0.0 0.0, s 0.0 c 0.0, 0.0 0.0 0.0 1.0]))
(defn- rot-x [a] (let [c (cos a) s (sin a)]
                   [1.0 0.0 0.0 0.0, 0.0 c s 0.0, 0.0 (- s) c 0.0, 0.0 0.0 0.0 1.0]))

(defn mvp
  "Model-view-projection (16 floats, column-major) for the cube at time t."
  [t aspect]
  (let [model (mat-mul (rot-y t) (rot-x (* 0.6 t)))
        view  (translate 0.0 0.0 -3.5)]
    (mat-mul (perspective 1.0 aspect 0.1 100.0) (mat-mul view model))))

;; --- cube geometry: 6 faces × 4 verts (pos3+uv2), 36 indices ------------------
(def ^:private faces
  ;; [center right up]  (half-size 0.5)
  [[[0.0 0.0 0.5]  [0.5 0.0 0.0]  [0.0 0.5 0.0]]
   [[0.0 0.0 -0.5] [-0.5 0.0 0.0] [0.0 0.5 0.0]]
   [[0.5 0.0 0.0]  [0.0 0.0 -0.5] [0.0 0.5 0.0]]
   [[-0.5 0.0 0.0] [0.0 0.0 0.5]  [0.0 0.5 0.0]]
   [[0.0 0.5 0.0]  [0.5 0.0 0.0]  [0.0 0.0 -0.5]]
   [[0.0 -0.5 0.0] [0.5 0.0 0.0]  [0.0 0.0 0.5]]])

(defn- v3 [op a b] (mapv op a b))
(def cube-verts
  (vec (mapcat (fn [[c rt up]]
                 (let [bl (v3 - (v3 - c rt) up) br (v3 - (v3 + c rt) up)
                       tr (v3 + (v3 + c rt) up) tl (v3 + (v3 - c rt) up)]
                   (concat bl [0.0 0.0] br [1.0 0.0] tr [1.0 1.0] tl [0.0 1.0])))
               faces)))
(def cube-indices
  (vec (mapcat (fn [f] (let [b (* 4 f)] [b (+ b 1) (+ b 2) b (+ b 2) (+ b 3)])) (range 6))))

;; --- checkerboard texture (flat RGBA8, row-major) -----------------------------
(defn checker [w h]
  (vec (mapcat (fn [i] (let [x (rem i w) y (quot i w)
                             on (zero? (mod (+ (quot x 8) (quot y 8)) 2))
                             v (if on 230 60)]
                         [v v v 255]))
               (range (* w h)))))

;; --- shader (one description → GLSL + WGSL) -----------------------------------
(def shader
  '{:uniform    {:name "U" :fields [[:mvp :mat4]]}
    :attributes [[inPos vec3 0] [inUv vec2 1]]
    :varyings   [[vUv vec2 0]]
    :textures   [[tex 0]]
    :vertex     [(set-position (* mvp (vec4 inPos 1.0))) (out vUv inUv)]
    :fragment   [(color (sample tex vUv))]})

(def pipeline-spec
  {:shader shader
   :topology :triangles
   :depth? true
   :cull :none
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}
                                        {:location 1 :format :float2 :offset 12}]}})

(defn draw-cube! [_rnd frame pipeline mesh tex t aspect]
  (r/bind-pipeline! frame pipeline)
  (r/set-uniform! frame pipeline (mvp t aspect))
  (r/bind-texture! frame pipeline tex)
  (r/bind-mesh! frame mesh)
  (r/draw-indexed! frame (:index-count mesh) 0))

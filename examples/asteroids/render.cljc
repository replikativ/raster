(ns asteroids.render
  "Portable rendering for cross-platform Geometric Asteroids.

   Turns game/render-data into a neutral line-list vertex buffer (pos2 + rgb) and
   records the draw via raster.render. Shared by the JVM (Vulkan) and browser
   (WebGPU) shells — the pipeline spec carries BOTH GLSL and WGSL shader sources,
   and each backend picks its own. The vertex shader maps screen-pixel coords to
   NDC using the `viewport` uniform; the only handedness difference (Vulkan Y-down
   vs WebGPU Y-up) lives in the two shader sources, nowhere else."
  (:require [raster.render :as r]
            [asteroids.game :as game]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(def ^:const TAU (* 2.0 #?(:clj Math/PI :cljs js/Math.PI)))
(def ^:const HALF-PI (* 0.5 #?(:clj Math/PI :cljs js/Math.PI)))

;; matplotlib tab10 palette, by polygon sides, in 0..1
(def palette {3 [0.580 0.404 0.741] 4 [0.839 0.153 0.157] 5 [0.173 0.627 0.173]
              6 [1.000 0.498 0.055] 7 [0.498 0.498 0.498] 8 [0.122 0.467 0.706]
              10 [0.090 0.745 0.812]})
(def ship-color [0.824 0.824 0.824])
(def bullet-color [1.0 1.0 1.0])

(def ^:const MAX-VERTS 16384)               ; line-list verts; buffer is reused each frame
(def ^:const STRIDE 20)                      ; bytes: pos2 (8) + rgb (12)

(defn- corners [x y size sides angle]
  (mapv (fn [k] (let [a (+ angle (/ (* TAU k) (double sides)) (- HALF-PI))]
                  [(+ x (* size (cos a))) (+ y (* size (sin a)))]))
        (range sides)))

(defn- emit-loop! [acc cs [r g b]]
  ;; closed polygon as a line list: each edge contributes its two endpoints
  (let [n (count cs)]
    (reduce (fn [a k]
              (let [[x0 y0] (nth cs k)
                    [x1 y1] (nth cs (mod (inc k) n))]
                (-> a (conj! x0) (conj! y0) (conj! r) (conj! g) (conj! b)
                    (conj! x1) (conj! y1) (conj! r) (conj! g) (conj! b))))
            acc (range n))))

(defn- emit-cross! [acc x y [r g b] s]
  (-> acc (conj! (- x s)) (conj! y) (conj! r) (conj! g) (conj! b)
      (conj! (+ x s)) (conj! y) (conj! r) (conj! g) (conj! b)
      (conj! x) (conj! (- y s)) (conj! r) (conj! g) (conj! b)
      (conj! x) (conj! (+ y s)) (conj! r) (conj! g) (conj! b)))

(defn scene-vertices
  "Flat seq of floats (pos2 + rgb per vertex) for the whole frame, as a line list."
  [rd]
  (persistent!
   (as-> (transient []) acc
     (reduce (fn [a p] (emit-loop! a (corners (:x p) (:y p) (:size p) (:sides p) (:angle p))
                                   (get palette (:sides p) [0.8 0.8 0.8])))
             acc (:polys rd))
     (let [s (:ship rd)]
       (emit-loop! acc (corners (:x s) (:y s) 12.0 3 (+ (:angle s) HALF-PI)) ship-color))
     (reduce (fn [a b] (emit-cross! a (:x b) (:y b) bullet-color 2.5)) acc (:bullets rd)))))

;; --- shader (ONE description → GLSL for Vulkan, WGSL for WebGPU) ---------------
;; The author writes set-position in Vulkan-style (Y-down) clip space; the WGSL
;; emitter flips Y for WebGPU. No hand-written per-platform shader source.
(def shader
  '{:uniform    {:name "U" :fields [[:viewport :vec4]]}
    :attributes [[inPos vec2 0] [inColor vec3 1]]
    :varyings   [[vColor vec3 0]]
    :vertex     [(let ndc vec2 (- (* (/ inPos (swizzle viewport xy)) 2.0) 1.0))
                 (set-position (vec4 (swizzle ndc x) (swizzle ndc y) 0.0 1.0))
                 (out vColor inColor)]
    :fragment   [(color (vec4 vColor 1.0))]})

(def pipeline-spec
  {:shader shader
   :topology :lines
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float2 :offset 0}
                                        {:location 1 :format :float3 :offset 8}]}})

(defn draw-scene!
  "Upload this frame's geometry and record the draw. `rnd` is the renderer, `frame`
   the per-frame recorder, `pipeline`/`mesh` the once-created resources."
  [rnd frame pipeline mesh rd]
  (let [m (r/upload-mesh! rnd mesh (scene-vertices rd))]
    (r/bind-pipeline! frame pipeline)
    (r/set-uniform! frame pipeline [(double game/W) (double game/H) 0.0 0.0])
    (r/bind-mesh! frame m)
    (r/draw! frame (:vertex-count m) 0)))

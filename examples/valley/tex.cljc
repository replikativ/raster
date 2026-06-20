(ns valley.tex
  "Cross-platform procedural block textures (port of the JVM valley.textures terrain
   layers). Pure (x,y)->RGBA pixel functions assembled into a flat layer-major RGBA8
   buffer for raster.render's texture-array. The 32-bit hash uses imul so the JVM and
   browser generate byte-identical textures. Layer indices match the JVM atlas (the
   subset used by the cross-platform terrain + biomes)."
  (:require [clojure.string]))

(def ^:const TEX 16)

;; 32-bit wrapping multiply (Java int* / JS Math.imul) — identical on both platforms.
(defn- imul [a b]
  #?(:clj  (unchecked-int (* (unchecked-int a) (unchecked-int b)))
     :cljs (js/Math.imul a b)))

(defn- hash-byte
  "JVM valley.textures/hash-byte, 32-bit: deterministic 0..255 noise."
  [x y seed]
  (let [h (bit-xor (imul (+ x (* y 37) seed) (unchecked-int 2654435761)) (unchecked-int 0xDEADBEEF))]
    (bit-and (bit-xor h (unsigned-bit-shift-right h 16)) 0xFF)))

(defn- m [v r] (mod v r))   ; like `rem` for the non-negative hash bytes

;; --- per-pixel colour functions (return [r g b a], 0..255) -------------------------
(defn- stone-px [x y] (let [v (hash-byte x y 1) g (+ 100 (m v 30))] [g g g 255]))
(defn- dirt-px  [x y] (let [v (hash-byte x y 2)] [(+ 110 (m v 20)) (+ 75 (m v 15)) (+ 45 (m v 10)) 255]))
(defn- grass-top-px [x y] (let [v (hash-byte x y 3)] [(+ 50 (m v 20)) (+ 140 (m v 30)) (+ 40 (m v 15)) 255]))
(defn- grass-side-px [x y]
  (if (< y 3) (let [v (hash-byte x y 4)] [(+ 50 (m v 20)) (+ 140 (m v 30)) (+ 40 (m v 15)) 255])
      (dirt-px x y)))
(defn- water-px [_ _] [40 80 200 160])
(defn- sand-px  [x y] (let [v (hash-byte x y 5)] [(+ 210 (m v 20)) (+ 190 (m v 15)) (+ 130 (m v 20)) 255]))
(defn- snow-px  [x y] (let [v (hash-byte x y 13)] [(+ 235 (m v 15)) (+ 240 (m v 12)) 255 255]))
(defn- ice-px   [x y] (let [v (hash-byte x y 14)] [(+ 140 (m v 20)) (+ 180 (m v 20)) (+ 230 (m v 15)) 200]))
(defn- clay-px  [x y] (let [v (hash-byte x y 15)] [(+ 150 (m v 15)) (+ 140 (m v 12)) (+ 130 (m v 10)) 255]))
(defn- sandstone-side-px [x y]
  (let [v (hash-byte x y 16) s (if (zero? (m y 4)) -8 0)]
    [(+ 195 s (m v 12)) (+ 175 s (m v 10)) (+ 120 (m v 12)) 255]))
(defn- sandstone-top-px [x y] (let [v (hash-byte x y 17)] [(+ 200 (m v 15)) (+ 185 (m v 12)) (+ 130 (m v 15)) 255]))
(defn- podzol-top-px [x y] (let [v (hash-byte x y 19)] [(+ 90 (m v 15)) (+ 70 (m v 12)) (+ 35 (m v 10)) 255]))
(defn- snowy-grass-top-px [x y] (let [v (hash-byte x y 20)] [(+ 235 (m v 15)) (+ 240 (m v 12)) 255 255]))
(defn- snowy-grass-side-px [x y]
  (if (< y 3) (let [v (hash-byte x y 20)] [(+ 235 (m v 15)) (+ 240 (m v 12)) 255 255]) (dirt-px x y)))
(defn- gravel-px [x y] (let [v (hash-byte x y 11)] [(+ 90 (m v 30)) (+ 85 (m v 25)) (+ 80 (m v 20)) 255]))

;; layer index -> pixel fn (matches the JVM atlas ordering for the terrain subset)
(def ^:private layer-fns
  {0 stone-px 1 dirt-px 2 grass-top-px 3 grass-side-px 4 water-px 5 sand-px
   12 gravel-px 13 snow-px 14 ice-px 15 clay-px 16 sandstone-side-px 17 sandstone-top-px
   19 podzol-top-px 20 snowy-grass-top-px 21 snowy-grass-side-px})

(def ^:const N-LAYERS 22)   ; layers 0..21 (gaps filled with magenta debug)

(defn atlas-pixels
  "Flat layer-major RGBA8 vector for all terrain texture layers (16×16 each)."
  []
  (vec (mapcat (fn [layer]
                 (let [f (get layer-fns layer (fn [_ _] [255 0 255 255]))]
                   (mapcat (fn [i] (let [x (mod i TEX) y (quot i TEX)] (f x y)))
                           (range (* TEX TEX)))))
               (range N-LAYERS))))

;; --- block-id -> per-face texture layer  [top bottom north south east west] ----------
;; (port of valley.world/block-faces for the cross-platform block set)
(def block-faces
  {1 [2 1 3 3 3 3]        ; grass: green top, dirt bottom, grassy side
   2 [1 1 1 1 1 1]        ; dirt
   3 [0 0 0 0 0 0]        ; stone
   4 [5 5 5 5 5 5]        ; sand
   5 [20 1 21 21 21 21]   ; snowy-grass
   6 [13 13 13 13 13 13]  ; snow
   7 [19 1 1 1 1 1]       ; podzol (dirt sides)
   8 [17 17 16 16 16 16]  ; sandstone
   9 [0 0 0 0 0 0]})      ; mountain stone

(defn face-layer
  "Texture layer for block `id` on face `f` (0 top 1 bottom 2 north 3 south 4 east 5 west)."
  ^long [^long id ^long f]
  (long (nth (get block-faces id [0 0 0 0 0 0]) f)))

;; directional face shading (port of valley.mesher/face-shade), indexed like block-faces
(def face-shade [1.0 0.5 0.8 0.8 0.7 0.7])

(def ^:const STRIDE 28)   ; pos3 (12) + uv2 (8) + layer (4) + shade (4)

;; Textured terrain shader: array-sample at the face's layer (fract uv tiles merged
;; quads) modulated by the per-face directional shade. One source → GLSL + WGSL.
(def shader
  '{:uniform        {:name "U" :fields [[:mvp :mat4]]}
    :attributes     [[inPos vec3 0] [inUv vec2 1] [inLayer float 2] [inShade float 3]]
    :varyings       [[vUv vec2 0] [vLayer float 1] [vShade float 2]]
    :texture-arrays [[atlas 0]]
    :vertex         [(set-position (* mvp (vec4 inPos 1.0)))
                     (out vUv inUv) (out vLayer inLayer) (out vShade inShade)]
    :fragment       [(color (* (sample-layer atlas vUv vLayer)
                               (vec4 vShade vShade vShade 1.0)))]})

(def pipeline-spec
  {:shader shader :topology :triangles :depth? true :cull :none
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}
                                        {:location 1 :format :float2 :offset 12}
                                        {:location 2 :format :float  :offset 20}
                                        {:location 3 :format :float  :offset 24}]}})

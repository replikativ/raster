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
(defn- iabs [n] (if (neg? n) (- n) n))

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

;; ore = stone base with clumped coloured speckles (2×2 blob test → speckles, not single px)
(defn- ore-px [x y seed cr cg cb]
  (if (< (m (hash-byte (quot x 2) (quot y 2) seed) 100) 28)
    (let [w (hash-byte x y (inc seed)) j (- (m w 40) 20)]
      [(max 0 (min 255 (+ cr j))) (max 0 (min 255 (+ cg j))) (max 0 (min 255 (+ cb j))) 255])
    (stone-px x y)))
(defn- coal-px    [x y] (ore-px x y 41 38 38 44))
(defn- iron-px    [x y] (ore-px x y 43 205 160 120))
(defn- gold-px    [x y] (ore-px x y 45 230 200 70))
(defn- diamond-px [x y] (ore-px x y 47 120 222 224))

;; birch: pale bark with short dark lenticels; spruce: dark reddish-brown bark
(defn- birch-side-px [x y]
  (let [v (hash-byte x y 51)]
    (if (and (zero? (m y 4)) (< (m (hash-byte x (quot y 4) 52) 10) 4))
      [50 48 46 255]                                   ; dark lenticel dash
      [(+ 205 (m v 25)) (+ 205 (m v 25)) (+ 195 (m v 20)) 255])))
(defn- birch-top-px  [x y] (let [v (hash-byte x y 53)] [(+ 200 (m v 20)) (+ 195 (m v 18)) (+ 165 (m v 15)) 255]))
(defn- birch-leaf-px [x y] (let [v (hash-byte x y 54)] [(+ 95 (m v 30)) (+ 165 (m v 35)) (+ 55 (m v 20)) 255]))
(defn- spruce-side-px [x y] (let [v (hash-byte x y 55) s (if (zero? (m x 4)) -10 0)]
                              [(+ 78 s (m v 18)) (+ 52 s (m v 12)) (+ 34 (m v 10)) 255]))
(defn- spruce-top-px  [x y] (let [v (hash-byte x y 56)] [(+ 92 (m v 16)) (+ 62 (m v 12)) (+ 40 (m v 10)) 255]))
(defn- spruce-leaf-px [x y] (let [v (hash-byte x y 57)] [(+ 28 (m v 18)) (+ 78 (m v 22)) (+ 52 (m v 18)) 255]))
(defn- gravel-px [x y] (let [v (hash-byte x y 11)] [(+ 90 (m v 30)) (+ 85 (m v 25)) (+ 80 (m v 20)) 255]))
;; trees: oak-like log (bark sides + ring top) and mottled leaves
(defn- log-side-px [x y]
  (let [v (hash-byte x y 60) streak (if (< (m (hash-byte (quot x 4) 0 60) 3) 1) -14 0)]
    [(+ 96 streak (m v 16)) (+ 66 streak (m v 12)) (+ 36 (m v 9)) 255]))
(defn- log-top-px [x y]
  (let [ring (if (< (m (+ (iabs (- x 8)) (iabs (- y 8))) 4) 1) -22 0) v (hash-byte x y 61)]
    [(+ 132 ring (m v 12)) (+ 96 ring (m v 10)) (+ 56 (m v 8)) 255]))
(defn- leaves-px [x y]
  (let [v (hash-byte x y 62)]
    (if (< (m v 7) 2) [38 88 34 255] [(+ 44 (m v 34)) (+ 108 (m v 48)) (+ 40 (m v 24)) 255])))

;; --- mob skins (port of valley.textures cow/pig/chicken; layers 22..29) -------------
(defn- cow-body-px [x y]
  (let [v (hash-byte x y 44)]
    (if (< (m (hash-byte (bit-shift-right x 2) (bit-shift-right y 2) 44) 10) 3)
      [230 225 220 255]                                            ; white patch
      [(+ 100 (m v 15)) (+ 60 (m v 12)) (+ 30 (m v 10)) 255])))    ; brown hide
(defn- cow-head-px [x y]
  (let [v (hash-byte x y 45)]
    (if (and (> y 9) (> x 4) (< x 11)) [225 215 205 255]           ; muzzle
        [(+ 100 (m v 15)) (+ 60 (m v 12)) (+ 30 (m v 10)) 255])))
(defn- animal-leg-px [x y] (let [v (hash-byte x y 46)] [(+ 95 (m v 12)) (+ 65 (m v 10)) (+ 35 (m v 8)) 255]))
(defn- pig-body-px [x y] (let [v (hash-byte x y 47)] [(+ 220 (m v 12)) (+ 150 (m v 10)) (+ 140 (m v 8)) 255]))
(defn- pig-head-px [x y]
  (let [v (hash-byte x y 48)]
    (if (and (> y 10) (> x 5) (< x 10)) [200 130 125 255]          ; snout
        [(+ 220 (m v 12)) (+ 150 (m v 10)) (+ 140 (m v 8)) 255])))
(defn- pig-leg-px [x y] (let [v (hash-byte x y 49)] [(+ 205 (m v 10)) (+ 135 (m v 8)) (+ 125 (m v 8)) 255]))
(defn- chicken-body-px [x y]
  (let [v (hash-byte x y 50)] (if (< (m v 2) 1) [240 240 235 255] [220 218 212 255])))
(defn- chicken-leg-px [x y] (let [v (hash-byte x y 51)] [(+ 225 (m v 15)) (+ 150 (m v 12)) 30 255]))

;; --- HUD icons (clip-space overlay; layers 33..37) ----------------------------------
(defn- heartf [x y]
  (let [fx (* (/ (- x 7.5) 8.0) 1.2) fy (* (/ (- 7.5 y) 8.0) 1.25)    ; y up
        a (+ (* fx fx) (* fy fy) -1.0)]
    (<= (- (* a a a) (* fx fx fy fy fy)) 0.0)))
(defn- heart-full-px  [x y] (if (heartf x y) [205 35 45 255] [0 0 0 0]))
(defn- heart-empty-px [x y] (if (heartf x y) [60 60 66 220] [0 0 0 0]))
(defn- slot-px [x y] (if (or (<= x 0) (>= x 15) (<= y 0) (>= y 15)) [205 205 215 205] [38 38 46 150]))
(defn- selector-px [x y] (if (or (<= x 1) (>= x 14) (<= y 1) (>= y 14)) [255 255 255 255] [0 0 0 0]))
(defn- crosshair-px [x y]
  (let [arm (and (>= x 3) (<= x 12) (>= y 3) (<= y 12))]
    (if (and arm (or (and (>= x 7) (<= x 8)) (and (>= y 7) (<= y 8)))) [235 235 235 230] [0 0 0 0])))

;; layer index -> pixel fn (matches the JVM atlas ordering for the terrain subset)
(def ^:private layer-fns
  {0 stone-px 1 dirt-px 2 grass-top-px 3 grass-side-px 4 water-px 5 sand-px
   6 coal-px 7 iron-px 8 gold-px 9 diamond-px
   12 gravel-px 13 snow-px 14 ice-px 15 clay-px 16 sandstone-side-px 17 sandstone-top-px
   19 podzol-top-px 20 snowy-grass-top-px 21 snowy-grass-side-px
   22 cow-body-px 23 cow-head-px 24 animal-leg-px 25 pig-body-px 26 pig-head-px
   27 pig-leg-px 28 chicken-body-px 29 chicken-leg-px
   30 log-side-px 31 log-top-px 32 leaves-px
   33 heart-full-px 34 heart-empty-px 35 slot-px 36 crosshair-px 37 selector-px
   38 birch-side-px 39 birch-top-px 40 birch-leaf-px
   41 spruce-side-px 42 spruce-top-px 43 spruce-leaf-px})

;; HUD layer ids (for valley.hud)
(def ^:const HEART-FULL 33) (def ^:const HEART-EMPTY 34)
(def ^:const SLOT 35) (def ^:const CROSSHAIR 36) (def ^:const SELECTOR 37)

(def ^:const N-LAYERS 44)   ; 0..21 terrain (6..9 ores), 22..29 mobs, 30..32 oak, 33..37 HUD, 38..43 birch/spruce

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
   9 [0 0 0 0 0 0]        ; mountain stone
   10 [4 4 4 4 4 4]       ; water (transparent layer 4)
   11 [31 31 30 30 30 30] ; log: ring top/bottom, bark sides
   12 [32 32 32 32 32 32]  ; leaves
   13 [6 6 6 6 6 6]        ; coal ore
   14 [7 7 7 7 7 7]        ; iron ore
   15 [8 8 8 8 8 8]        ; gold ore
   16 [9 9 9 9 9 9]        ; diamond ore
   17 [39 39 38 38 38 38]  ; birch log
   18 [40 40 40 40 40 40]  ; birch leaves
   19 [42 42 41 41 41 41]  ; spruce log
   20 [43 43 43 43 43 43]})  ; spruce leaves

(defn face-layer
  "Texture layer for block `id` on face `f` (0 top 1 bottom 2 north 3 south 4 east 5 west)."
  ^long [^long id ^long f]
  (long (nth (get block-faces id [0 0 0 0 0 0]) f)))

;; directional face shading (port of valley.mesher/face-shade), indexed like block-faces
(def face-shade [1.0 0.5 0.8 0.8 0.7 0.7])

(def ^:const STRIDE 36)   ; pos3(12) + uv2(8) + layer(4) + shade(4) + light(4) + ao(4)

;; Textured terrain shader: array-sample at the face's layer modulated by the per-face
;; directional shade, baked skylight (vLight) and ambient occlusion (vAo). The day/night
;; level arrives in env.x (day ratio 0.05..1) with env.y ambient floor, so the same baked
;; light dims smoothly into night. One source → GLSL + WGSL.
(def shader
  '{:uniform        {:name "U" :fields [[:mvp :mat4] [:env :vec4]]}   ; env = [dayRatio ambient _ _]
    :attributes     [[inPos vec3 0] [inUv vec2 1] [inLayer float 2] [inShade float 3]
                     [inLight float 4] [inAo float 5]]
    :varyings       [[vUv vec2 0] [vLayer float 1] [vShade float 2] [vLight float 3] [vAo float 4]]
    :texture-arrays [[atlas 0]]
    :vertex         [(set-position (* mvp (vec4 inPos 1.0)))
                     (out vUv inUv) (out vLayer inLayer) (out vShade inShade)
                     (out vLight inLight) (out vAo inAo)]
    :fragment       [(let texel vec4 (sample-layer atlas vUv vLayer))
                     (let lvl float (clamp (+ (swizzle env y) (* (swizzle env x) vLight)) 0.0 1.0))
                     (let lit float (* lvl vShade vAo))
                     (color (vec4 (* (swizzle texel xyz) lit) (swizzle texel w)))]})

(def pipeline-spec
  {:shader shader :topology :triangles :depth? true :cull :none :blend? true
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}
                                        {:location 1 :format :float2 :offset 12}
                                        {:location 2 :format :float  :offset 20}
                                        {:location 3 :format :float  :offset 24}
                                        {:location 4 :format :float  :offset 28}
                                        {:location 5 :format :float  :offset 32}]}})

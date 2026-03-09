(ns valley.textures
  "Procedural block texture generation.
  Creates 16x16 pixel textures for each block face."
)

(set! *warn-on-reflection* true)

(def ^:const TEX-SIZE 16)

(defn- make-rgba
  "Create a byte array with RGBA pixels for a TEX-SIZE x TEX-SIZE texture."
  ^bytes [fill-fn]
  (let [size (* TEX-SIZE TEX-SIZE 4)
        buf (byte-array size)]
    (dotimes [y TEX-SIZE]
      (dotimes [x TEX-SIZE]
        (let [[r g b a] (fill-fn x y)
              off (* (+ (* y TEX-SIZE) x) 4)]
          (aset buf off       (unchecked-byte r))
          (aset buf (+ off 1) (unchecked-byte g))
          (aset buf (+ off 2) (unchecked-byte b))
          (aset buf (+ off 3) (unchecked-byte a)))))
    buf))

(defn- hash-byte ^long [^long x ^long y ^long seed]
  (let [h (bit-xor (* (+ x (* y 37) seed) 2654435761) 0xDEADBEEF)]
    (bit-and (bit-xor h (unsigned-bit-shift-right h 16)) 0xFF)))

;; ================================================================
;; Original block textures (layers 0-12)
;; ================================================================

(defn- stone-pixel [^long x ^long y]
  (let [v (+ 100 (rem (hash-byte x y 1) 30))]
    [v v v 255]))

(defn- dirt-pixel [^long x ^long y]
  (let [v (hash-byte x y 2)]
    [(+ 110 (rem v 20)) (+ 75 (rem v 15)) (+ 45 (rem v 10)) 255]))

(defn- grass-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 3)]
    [(+ 50 (rem v 20)) (+ 140 (rem v 30)) (+ 40 (rem v 15)) 255]))

(defn- grass-side-pixel [^long x ^long y]
  (if (< y 3)
    (let [v (hash-byte x y 4)]
      [(+ 50 (rem v 20)) (+ 140 (rem v 30)) (+ 40 (rem v 15)) 255])
    (dirt-pixel x y)))

(defn- water-pixel [^long _x ^long _y]
  [40 80 200 160])

(defn- sand-pixel [^long x ^long y]
  (let [v (hash-byte x y 5)]
    [(+ 210 (rem v 20)) (+ 190 (rem v 15)) (+ 130 (rem v 20)) 255]))

(defn- wood-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 6)
        stripe (if (zero? (rem y 3)) 10 0)]
    [(+ 120 stripe (rem v 10)) (+ 80 stripe (rem v 8)) (+ 40 (rem v 5)) 255]))

(defn- wood-top-pixel [^long x ^long y]
  (let [cx (- x 8) cy (- y 8)
        dist (Math/sqrt (+ (* cx cx) (* cy cy)))
        ring (rem (int dist) 3)
        v (hash-byte x y 7)]
    (if (< ring 2)
      [(+ 130 (rem v 10)) (+ 90 (rem v 8)) (+ 50 (rem v 5)) 255]
      [(+ 110 (rem v 10)) (+ 75 (rem v 8)) (+ 40 (rem v 5)) 255])))

(defn- leaves-pixel [^long x ^long y]
  (let [v (hash-byte x y 8)
        alpha (if (> (rem v 5) 1) 255 0)]
    [(+ 30 (rem v 25)) (+ 100 (rem v 40)) (+ 20 (rem v 15)) alpha]))

(defn- bedrock-pixel [^long x ^long y]
  (let [v (hash-byte x y 9)]
    [(+ 30 (rem v 20)) (+ 30 (rem v 20)) (+ 30 (rem v 20)) 255]))

(defn- torch-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8))
        cy (Math/abs (- y 8))]
    (if (and (< cx 2) (< cy 5))
      (if (< cy 2) [255 200 50 255] [120 80 40 255])
      [0 0 0 0])))

(defn- glass-pixel [^long x ^long y]
  (let [edge? (or (zero? x) (= x 15) (zero? y) (= y 15))]
    (if edge? [200 220 240 200] [180 200 220 80])))

(defn- gravel-pixel [^long x ^long y]
  (let [v (hash-byte x y 11)]
    [(+ 90 (rem v 30)) (+ 85 (rem v 25)) (+ 80 (rem v 20)) 255]))

;; ================================================================
;; Biome surface textures (layers 13-21)
;; ================================================================

(defn- snow-pixel [^long x ^long y]
  (let [v (hash-byte x y 13)]
    [(+ 235 (rem v 15)) (+ 240 (rem v 12)) 255 255]))

(defn- ice-pixel [^long x ^long y]
  (let [v (hash-byte x y 14)]
    [(+ 140 (rem v 20)) (+ 180 (rem v 20)) (+ 230 (rem v 15)) 200]))

(defn- clay-pixel [^long x ^long y]
  (let [v (hash-byte x y 15)]
    [(+ 150 (rem v 15)) (+ 140 (rem v 12)) (+ 130 (rem v 10)) 255]))

(defn- sandstone-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 16)
        stripe (if (zero? (rem y 4)) -8 0)]
    [(+ 195 stripe (rem v 12)) (+ 175 stripe (rem v 10)) (+ 120 (rem v 12)) 255]))

(defn- sandstone-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 17)]
    [(+ 200 (rem v 15)) (+ 185 (rem v 12)) (+ 130 (rem v 15)) 255]))

(defn- red-sand-pixel [^long x ^long y]
  (let [v (hash-byte x y 18)]
    [(+ 180 (rem v 20)) (+ 100 (rem v 15)) (+ 50 (rem v 15)) 255]))

(defn- podzol-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 19)]
    [(+ 90 (rem v 15)) (+ 70 (rem v 12)) (+ 35 (rem v 10)) 255]))

(defn- snowy-grass-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 20)]
    [(+ 235 (rem v 15)) (+ 240 (rem v 12)) 255 255]))

(defn- snowy-grass-side-pixel [^long x ^long y]
  (if (< y 3)
    (let [v (hash-byte x y 20)]
      [(+ 235 (rem v 15)) (+ 240 (rem v 12)) 255 255])
    (dirt-pixel x y)))

;; ================================================================
;; Ore textures (layers 22-26) — stone base with colored specks
;; ================================================================

(defn- ore-pixel [^long x ^long y ^long seed color]
  (let [base (stone-pixel x y)
        v (hash-byte x y seed)]
    (if (> (rem v 7) 4)
      color
      base)))

(defn- coal-ore-pixel [x y] (ore-pixel x y 22 [40 40 40 255]))
(defn- iron-ore-pixel [x y] (ore-pixel x y 23 [180 130 100 255]))
(defn- copper-ore-pixel [x y] (ore-pixel x y 24 [180 120 70 255]))
(defn- gold-ore-pixel [x y] (ore-pixel x y 25 [240 210 60 255]))
(defn- diamond-ore-pixel [x y] (ore-pixel x y 26 [100 220 240 255]))

;; ================================================================
;; Stone variants (layers 27-31)
;; ================================================================

(defn- cobblestone-pixel [^long x ^long y]
  (let [v (hash-byte x y 27)
        crack? (or (zero? (rem (+ x (* y 3)) 5))
                   (zero? (rem (+ (* x 2) y) 7)))]
    (if crack?
      [(+ 70 (rem v 15)) (+ 70 (rem v 12)) (+ 70 (rem v 10)) 255]
      [(+ 110 (rem v 20)) (+ 110 (rem v 18)) (+ 110 (rem v 16)) 255])))

(defn- granite-pixel [^long x ^long y]
  (let [v (hash-byte x y 28)]
    [(+ 150 (rem v 20)) (+ 100 (rem v 15)) (+ 90 (rem v 12)) 255]))

(defn- andesite-pixel [^long x ^long y]
  (let [v (hash-byte x y 29)]
    [(+ 120 (rem v 20)) (+ 120 (rem v 18)) (+ 115 (rem v 15)) 255]))

(defn- obsidian-pixel [^long x ^long y]
  (let [v (hash-byte x y 30)]
    [(+ 15 (rem v 10)) (+ 10 (rem v 8)) (+ 25 (rem v 12)) 255]))

(defn- basalt-pixel [^long x ^long y]
  (let [v (hash-byte x y 31)
        stripe (if (zero? (rem y 2)) 5 0)]
    [(+ 50 stripe (rem v 12)) (+ 48 stripe (rem v 10)) (+ 55 stripe (rem v 10)) 255]))

;; ================================================================
;; Wood variant textures (layers 32-43)
;; ================================================================

(defn- birch-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 32)
        dark? (and (zero? (rem (+ (* x 3) y) 11)) (> v 100))]
    (if dark?
      [(+ 40 (rem v 10)) (+ 35 (rem v 8)) (+ 30 (rem v 6)) 255]
      [(+ 210 (rem v 12)) (+ 205 (rem v 10)) (+ 195 (rem v 10)) 255])))

(defn- birch-top-pixel [^long x ^long y]
  (let [cx (- x 8) cy (- y 8)
        dist (Math/sqrt (+ (* cx cx) (* cy cy)))
        ring (rem (int dist) 3)
        v (hash-byte x y 33)]
    (if (< ring 2)
      [(+ 200 (rem v 10)) (+ 195 (rem v 8)) (+ 180 (rem v 8)) 255]
      [(+ 180 (rem v 10)) (+ 175 (rem v 8)) (+ 160 (rem v 8)) 255])))

(defn- birch-leaves-pixel [^long x ^long y]
  (let [v (hash-byte x y 34)
        alpha (if (> (rem v 5) 1) 255 0)]
    [(+ 60 (rem v 20)) (+ 130 (rem v 30)) (+ 30 (rem v 15)) alpha]))

(defn- pine-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 35)
        stripe (if (zero? (rem y 3)) 8 0)]
    [(+ 80 stripe (rem v 10)) (+ 55 stripe (rem v 8)) (+ 30 (rem v 6)) 255]))

(defn- pine-top-pixel [^long x ^long y]
  (let [cx (- x 8) cy (- y 8)
        dist (Math/sqrt (+ (* cx cx) (* cy cy)))
        ring (rem (int dist) 3)
        v (hash-byte x y 36)]
    (if (< ring 2)
      [(+ 90 (rem v 10)) (+ 65 (rem v 8)) (+ 35 (rem v 5)) 255]
      [(+ 75 (rem v 10)) (+ 50 (rem v 8)) (+ 25 (rem v 5)) 255])))

(defn- pine-leaves-pixel [^long x ^long y]
  (let [v (hash-byte x y 37)
        alpha (if (> (rem v 4) 0) 255 0)]
    [(+ 20 (rem v 15)) (+ 70 (rem v 25)) (+ 25 (rem v 12)) alpha]))

(defn- jungle-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 38)
        stripe (if (zero? (rem y 2)) 6 0)]
    [(+ 100 stripe (rem v 12)) (+ 75 stripe (rem v 8)) (+ 35 (rem v 6)) 255]))

(defn- jungle-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 39)]
    [(+ 110 (rem v 12)) (+ 85 (rem v 10)) (+ 40 (rem v 6)) 255]))

(defn- jungle-leaves-pixel [^long x ^long y]
  (let [v (hash-byte x y 40)
        alpha (if (> (rem v 6) 1) 255 0)]
    [(+ 20 (rem v 20)) (+ 110 (rem v 35)) (+ 15 (rem v 12)) alpha]))

(defn- dark-oak-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 41)
        stripe (if (zero? (rem y 4)) 8 0)]
    [(+ 60 stripe (rem v 10)) (+ 40 stripe (rem v 8)) (+ 20 (rem v 5)) 255]))

(defn- dark-oak-top-pixel [^long x ^long y]
  (let [cx (- x 8) cy (- y 8)
        dist (Math/sqrt (+ (* cx cx) (* cy cy)))
        ring (rem (int dist) 3)
        v (hash-byte x y 42)]
    (if (< ring 2)
      [(+ 70 (rem v 10)) (+ 50 (rem v 8)) (+ 25 (rem v 5)) 255]
      [(+ 55 (rem v 8)) (+ 35 (rem v 6)) (+ 18 (rem v 4)) 255])))

(defn- dark-oak-leaves-pixel [^long x ^long y]
  (let [v (hash-byte x y 43)
        alpha (if (> (rem v 5) 1) 255 0)]
    [(+ 15 (rem v 15)) (+ 60 (rem v 30)) (+ 10 (rem v 10)) alpha]))

;; ================================================================
;; Plant textures (layers 44-52)
;; ================================================================

(defn- cactus-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 44)
        edge? (or (zero? x) (= x 15))
        spine? (and (zero? (rem y 4)) (or (= x 4) (= x 11)))]
    (cond
      spine? [(+ 180 (rem v 15)) (+ 200 (rem v 10)) (+ 100 (rem v 10)) 255]
      edge? [(+ 40 (rem v 10)) (+ 100 (rem v 15)) (+ 30 (rem v 8)) 255]
      :else [(+ 50 (rem v 15)) (+ 130 (rem v 20)) (+ 40 (rem v 10)) 255])))

(defn- cactus-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 45)
        cx (Math/abs (- x 8)) cy (Math/abs (- y 8))
        center? (and (< cx 3) (< cy 3))]
    (if center?
      [(+ 60 (rem v 15)) (+ 150 (rem v 20)) (+ 50 (rem v 12)) 255]
      [(+ 45 (rem v 12)) (+ 120 (rem v 18)) (+ 35 (rem v 10)) 255])))

(defn- tall-grass-pixel [^long x ^long y]
  (let [v (hash-byte x y 46)
        cx (Math/abs (- x 8))
        blade? (and (< cx (- 7 (quot y 3))) (> y 2))]
    (if blade?
      [(+ 50 (rem v 20)) (+ 130 (rem v 30)) (+ 35 (rem v 12)) 220]
      [0 0 0 0])))

(defn- fern-pixel [^long x ^long y]
  (let [v (hash-byte x y 47)
        cx (- x 8)
        spread (+ 2 (quot (- 15 y) 2))
        frond? (and (< (Math/abs cx) spread) (> y 1)
                    (or (zero? (rem (+ x y) 3)) (< (Math/abs cx) 2)))]
    (if frond?
      [(+ 30 (rem v 18)) (+ 100 (rem v 25)) (+ 25 (rem v 10)) 230]
      [0 0 0 0])))

(defn- dead-bush-pixel [^long x ^long y]
  (let [v (hash-byte x y 48)
        cx (- x 8)
        branch? (and (> y 4)
                     (< (Math/abs (- (Math/abs cx)
                                     (quot (- 15 y) 2))) 2))]
    (if (or branch? (and (< (Math/abs cx) 2) (> y 2)))
      [(+ 120 (rem v 15)) (+ 90 (rem v 10)) (+ 50 (rem v 8)) 255]
      [0 0 0 0])))

(defn- flower-red-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8)) cy (Math/abs (- y 5))]
    (cond
      (and (< cx 3) (< cy 3) (< y 9))
      [220 40 40 255]
      (and (< (Math/abs (- x 8)) 1) (>= y 8))
      [50 120 40 255]
      :else [0 0 0 0])))

(defn- flower-yellow-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8)) cy (Math/abs (- y 5))]
    (cond
      (and (< cx 3) (< cy 3) (< y 9))
      [240 220 40 255]
      (and (< (Math/abs (- x 8)) 1) (>= y 8))
      [50 120 40 255]
      :else [0 0 0 0])))

(defn- mushroom-red-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8)) v (hash-byte x y 51)]
    (cond
      ;; Cap
      (and (< y 8) (< cx (- 6 (quot y 2))))
      (if (zero? (rem (+ x (* y 3)) 5))
        [240 240 230 255]  ;; white spots
        [(+ 180 (rem v 20)) (+ 20 (rem v 10)) (+ 20 (rem v 8)) 255])
      ;; Stem
      (and (>= y 8) (< cx 2))
      [(+ 200 (rem v 12)) (+ 195 (rem v 10)) (+ 185 (rem v 8)) 255]
      :else [0 0 0 0])))

(defn- mushroom-brown-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8)) v (hash-byte x y 52)]
    (cond
      (and (< y 7) (< cx (- 7 y)))
      [(+ 140 (rem v 15)) (+ 95 (rem v 12)) (+ 55 (rem v 10)) 255]
      (and (>= y 7) (< cx 2))
      [(+ 195 (rem v 10)) (+ 190 (rem v 8)) (+ 180 (rem v 8)) 255]
      :else [0 0 0 0])))

;; ================================================================
;; Underground special textures (layers 53-55)
;; ================================================================

(defn- glowstone-pixel [^long x ^long y]
  (let [v (hash-byte x y 53)]
    [(+ 200 (rem v 40)) (+ 180 (rem v 30)) (+ 80 (rem v 30)) 255]))

(defn- crystal-pixel [^long x ^long y]
  (let [v (hash-byte x y 54)
        cx (Math/abs (- x 8))
        facet? (< cx (- 7 (Math/abs (- y 8))))]
    (if facet?
      [(+ 160 (rem v 30)) (+ 100 (rem v 40)) (+ 220 (rem v 25)) 180]
      [0 0 0 0])))

(defn- mossy-cobble-pixel [^long x ^long y]
  (let [[cr cg cb _] (cobblestone-pixel x y)
        v (hash-byte x y 55)
        mossy? (> (rem v 4) 1)]
    (if mossy?
      [(max 30 (- (long cr) 40)) (+ (long cg) 30) (max 20 (- (long cb) 30)) 255]
      [cr cg cb 255])))

;; ================================================================
;; Functional block textures (layers 56-60)
;; ================================================================

(defn- crafting-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 56)
        grid? (or (= x 5) (= x 10) (= y 5) (= y 10))
        edge? (or (zero? x) (= x 15) (zero? y) (= y 15))]
    (cond
      edge? [(+ 80 (rem v 10)) (+ 55 (rem v 8)) (+ 30 (rem v 5)) 255]
      grid? [(+ 70 (rem v 8)) (+ 50 (rem v 6)) (+ 25 (rem v 5)) 255]
      :else [(+ 150 (rem v 12)) (+ 110 (rem v 10)) (+ 60 (rem v 8)) 255])))

(defn- crafting-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 57)
        tool? (and (> x 3) (< x 13) (> y 3) (< y 13)
                   (or (= x 8) (= y 8)))]
    (if tool?
      [(+ 100 (rem v 12)) (+ 100 (rem v 10)) (+ 100 (rem v 8)) 255]
      [(+ 130 (rem v 12)) (+ 90 (rem v 10)) (+ 50 (rem v 8)) 255])))

(defn- furnace-front-pixel [^long x ^long y]
  (let [v (hash-byte x y 58)
        opening? (and (> x 4) (< x 12) (> y 5) (< y 13))]
    (if opening?
      [(+ 30 (rem v 15)) (+ 25 (rem v 10)) (+ 20 (rem v 8)) 255]
      [(+ 115 (rem v 15)) (+ 115 (rem v 12)) (+ 115 (rem v 10)) 255])))

(defn- furnace-side-pixel [^long x ^long y]
  (let [v (hash-byte x y 59)]
    [(+ 110 (rem v 18)) (+ 110 (rem v 15)) (+ 110 (rem v 12)) 255]))

(defn- furnace-top-pixel [^long x ^long y]
  (let [v (hash-byte x y 60)]
    [(+ 105 (rem v 15)) (+ 105 (rem v 12)) (+ 105 (rem v 10)) 255]))

;; ================================================================
;; Lava (layer 61)
;; ================================================================

(defn- lava-pixel [^long x ^long y]
  (let [v (hash-byte x y 61)]
    [(+ 210 (rem v 40)) (+ 80 (rem v 50)) (+ 10 (rem v 20)) 230]))

;; ================================================================
;; HUD icon textures (layers 62-64)
;; ================================================================

(defn- heart-shape?
  "Check if (x,y) is inside a heart shape centered on the 16x16 grid."
  [^long x ^long y]
  ;; Heart shape: two circles on top, triangle on bottom
  ;; Normalized coords (0-15 → centered)
  (let [nx (- (/ (double x) 15.0) 0.5)  ;; -0.5 to 0.5
        ny (- 0.5 (/ (double y) 15.0))  ;; flip Y, -0.5 to 0.5
        ;; Scale up for shape computation
        sx (* nx 2.4)
        sy (* ny 2.4)
        ;; Two-circle + cusp heart equation: (x²+y²-1)³ - x²y³ < 0
        x2 (* sx sx)
        y2 (* sy sy)
        t (- (+ x2 y2) 1.0)]
    (< (- (* t t t) (* x2 y2 sy)) 0.0)))

(defn- heart-full-pixel [^long x ^long y]
  (if (heart-shape? x y)
    [220 30 30 255]   ;; Red heart
    [0 0 0 0]))       ;; Transparent

(defn- heart-half-pixel [^long x ^long y]
  (if (heart-shape? x y)
    (if (< x 8)
      [220 30 30 255]   ;; Left half red
      [60 60 60 180])   ;; Right half dark/empty
    [0 0 0 0]))

(defn- heart-empty-pixel [^long x ^long y]
  (if (heart-shape? x y)
    [60 60 60 180]    ;; Dark gray outline
    [0 0 0 0]))

;; ================================================================
;; Item icon textures (layers 65-87)
;; ================================================================

;; Generic item icon: small centered square with color
(defn- item-icon-pixel [x y r g b]
  (let [x (long x) y (long y)
        cx (Math/abs (- x 8)) cy (Math/abs (- y 8))]
    (if (and (< cx 5) (< cy 5))
      [(int r) (int g) (int b) 255]
      [0 0 0 0])))

(defn- planks-pixel [^long x ^long y]
  (let [v (hash-byte x y 65)
        stripe (if (zero? (rem y 4)) 8 0)]
    (if (and (> x 1) (< x 14) (> y 1) (< y 14))
      [(+ 170 stripe (rem v 12)) (+ 130 stripe (rem v 10)) (+ 70 (rem v 8)) 255]
      [0 0 0 0])))

(defn- coal-pixel [^long x ^long y]
  (item-icon-pixel x y (+ 25 (rem (hash-byte x y 66) 15))
                   (+ 25 (rem (hash-byte x y 66) 12))
                   (+ 25 (rem (hash-byte x y 66) 10))))

(defn- diamond-item-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8)) cy (Math/abs (- y 8))
        inside? (< (+ cx cy) 7)]
    (if inside?
      [(+ 120 (rem (hash-byte x y 67) 30)) (+ 210 (rem (hash-byte x y 67) 20)) (+ 230 (rem (hash-byte x y 67) 15)) 255]
      [0 0 0 0])))

(defn- ingot-pixel [x y r g b seed]
  (let [x (long x) y (long y) seed (long seed)
        v (hash-byte x y seed)
        inside? (and (> x 2) (< x 13) (> y 4) (< y 12))]
    (if inside?
      [(+ (int r) (rem v 15)) (+ (int g) (rem v 12)) (+ (int b) (rem v 10)) 255]
      [0 0 0 0])))

(defn- iron-ingot-pixel [^long x ^long y] (ingot-pixel x y 200 200 200 68))
(defn- gold-ingot-pixel [^long x ^long y] (ingot-pixel x y 230 200 50 69))
(defn- copper-ingot-pixel [^long x ^long y] (ingot-pixel x y 200 130 70 70))

(defn- stick-pixel [^long x ^long y]
  (let [cx (Math/abs (- x 8))]
    (if (and (< cx 2) (> y 2) (< y 14))
      [(+ 140 (rem (hash-byte x y 71) 10)) (+ 100 (rem (hash-byte x y 71) 8)) (+ 50 (rem (hash-byte x y 71) 6)) 255]
      [0 0 0 0])))

;; Tool icon generators: handle + head
(defn- tool-pixel [x y head-color tool-shape]
  (let [x (long x) y (long y)
        ;; Handle: diagonal line from bottom-left to center
        on-handle? (and (< (Math/abs (- (- 15 y) x)) 2) (> y 8) (< x 8))
        ;; Head: top-right area, shape depends on tool
        on-head? (tool-shape x y)]
    (cond
      on-head?   head-color
      on-handle? [(+ 140 (rem (hash-byte x y 99) 10)) (+ 100 (rem (hash-byte x y 99) 8)) 50 255]
      :else      [0 0 0 0])))

(defn- pickaxe-shape [^long x ^long y]
  (and (>= x 4) (<= x 12) (>= y 2) (<= y 4)))

(defn- axe-shape [^long x ^long y]
  (and (>= x 7) (<= x 12) (>= y 2) (<= y 7)))

(defn- shovel-shape [^long x ^long y]
  (and (>= x 6) (<= x 10) (>= y 1) (<= y 5)))

(defn- sword-shape [^long x ^long y]
  (and (< (Math/abs (- (- 12 y) (- x 6))) 2) (< y 9) (> x 5)))

(def ^:private tier-colors
  {:wood    [140 100 50 255]
   :stone   [120 120 120 255]
   :iron    [210 210 210 255]
   :diamond [100 220 240 255]})

(defn- make-tool-pixel-fn [tier shape-fn]
  (let [color (get tier-colors tier)]
    (fn [x y] (tool-pixel x y color shape-fn))))

;; ================================================================
;; Mob textures (88-95)
;; ================================================================

;; Cow body: brown with white patches
(defn- cow-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 88)
        patch? (< (rem (hash-byte (bit-shift-right x 2) (bit-shift-right y 2) 44) 10) 3)]
    (if patch?
      [(+ 230 (rem v 15)) (+ 225 (rem v 12)) (+ 220 (rem v 10)) 255]
      [(+ 100 (rem v 15)) (+ 60 (rem v 12)) (+ 30 (rem v 10)) 255])))

;; Cow head: brown with white nose area
(defn- cow-head-pixel [^long x ^long y]
  (let [v (hash-byte x y 89)
        nose? (and (> x 4) (< x 12) (> y 9) (< y 14))]
    (if nose?
      [(+ 220 (rem v 10)) (+ 210 (rem v 8)) (+ 200 (rem v 8)) 255]
      [(+ 100 (rem v 12)) (+ 60 (rem v 10)) (+ 30 (rem v 8)) 255])))

;; Cow/sheep legs: brown
(defn- animal-leg-pixel [^long x ^long y]
  (let [v (hash-byte x y 90)]
    [(+ 95 (rem v 12)) (+ 65 (rem v 10)) (+ 35 (rem v 8)) 255]))

;; Pig body: pink
(defn- pig-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 91)]
    [(+ 220 (rem v 12)) (+ 150 (rem v 10)) (+ 140 (rem v 8)) 255]))

;; Pig head: pink with darker snout
(defn- pig-head-pixel [^long x ^long y]
  (let [v (hash-byte x y 92)
        snout? (and (> x 5) (< x 11) (> y 8) (< y 13))]
    (if snout?
      [(+ 200 (rem v 10)) (+ 120 (rem v 8)) (+ 110 (rem v 8)) 255]
      [(+ 220 (rem v 10)) (+ 150 (rem v 8)) (+ 140 (rem v 8)) 255])))

;; Pig legs: pink
(defn- pig-leg-pixel [^long x ^long y]
  (let [v (hash-byte x y 93)]
    [(+ 210 (rem v 10)) (+ 140 (rem v 8)) (+ 130 (rem v 8)) 255]))

;; Chicken body: white with feathery texture
(defn- chicken-pixel [^long x ^long y]
  (let [v (hash-byte x y 94)
        feather? (< (rem v 5) 2)]
    (if feather?
      [(+ 240 (rem v 10)) (+ 235 (rem v 8)) (+ 230 (rem v 8)) 255]
      [(+ 220 (rem v 10)) (+ 215 (rem v 8)) (+ 210 (rem v 8)) 255])))

;; Chicken legs: orange
(defn- chicken-leg-pixel [^long x ^long y]
  (let [v (hash-byte x y 95)]
    [(+ 230 (rem v 10)) (+ 160 (rem v 8)) (+ 30 (rem v 8)) 255]))

;; ================================================================
;; Hostile mob textures (96-103)
;; ================================================================

;; Zombie body/limbs: dark green-grey
(defn- zombie-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 96)]
    [(+ 60 (rem v 15)) (+ 90 (rem v 12)) (+ 55 (rem v 10)) 255]))

;; Zombie head: green with darker features
(defn- zombie-head-pixel [^long x ^long y]
  (let [v (hash-byte x y 97)
        eye? (and (> y 5) (< y 9) (or (and (> x 3) (< x 6)) (and (> x 9) (< x 12))))]
    (if eye?
      [10 10 10 255]
      [(+ 70 (rem v 12)) (+ 100 (rem v 10)) (+ 60 (rem v 8)) 255])))

;; Skeleton body/limbs: bone white-grey
(defn- skeleton-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 98)]
    [(+ 200 (rem v 15)) (+ 195 (rem v 12)) (+ 185 (rem v 10)) 255]))

;; Skeleton head: white skull
(defn- skeleton-head-pixel [^long x ^long y]
  (let [v (hash-byte x y 99)
        eye? (and (> y 5) (< y 9) (or (and (> x 3) (< x 6)) (and (> x 9) (< x 12))))]
    (if eye?
      [20 20 20 255]
      [(+ 210 (rem v 10)) (+ 205 (rem v 8)) (+ 195 (rem v 8)) 255])))

;; Lurker body/limbs: mottled green
(defn- lurker-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 100)
        spot? (< (rem v 7) 2)]
    (if spot?
      [(+ 30 (rem v 10)) (+ 100 (rem v 10)) (+ 25 (rem v 8)) 255]
      [(+ 50 (rem v 10)) (+ 140 (rem v 10)) (+ 40 (rem v 8)) 255])))

;; Lurker face: green with dark face pattern
(defn- lurker-face-pixel [^long x ^long y]
  (let [v (hash-byte x y 101)
        ;; Lurker face: two vertical dark rectangles for eyes, frown
        face? (or (and (> y 4) (< y 9) (or (and (> x 3) (< x 6)) (and (> x 9) (< x 12))))
                  (and (> y 8) (< y 12) (> x 5) (< x 11)))]
    (if face?
      [(+ 15 (rem v 8)) (+ 15 (rem v 6)) (+ 10 (rem v 5)) 255]
      [(+ 50 (rem v 10)) (+ 140 (rem v 10)) (+ 40 (rem v 8)) 255])))

;; Spider body: dark brown/black
(defn- spider-body-pixel [^long x ^long y]
  (let [v (hash-byte x y 102)
        eye? (and (> y 3) (< y 7) (or (and (> x 2) (< x 5)) (and (> x 10) (< x 13))))]
    (if eye?
      [(+ 180 (rem v 10)) (+ 20 (rem v 8)) (+ 20 (rem v 8)) 255]  ;; red eyes
      [(+ 30 (rem v 12)) (+ 25 (rem v 10)) (+ 20 (rem v 8)) 255])))

;; Spider head: dark with red eyes
(defn- spider-head-pixel [^long x ^long y]
  (let [v (hash-byte x y 103)
        eye? (and (> y 5) (< y 10)
                  (or (and (> x 2) (< x 6)) (and (> x 9) (< x 13))))]
    (if eye?
      [(+ 200 (rem v 10)) (+ 10 (rem v 8)) (+ 10 (rem v 8)) 255]
      [(+ 35 (rem v 10)) (+ 28 (rem v 8)) (+ 22 (rem v 8)) 255])))

;; ================================================================
;; Food / farming / hunger textures (104-120)
;; ================================================================

;; Raw beef: red meat slab
(defn- raw-beef-pixel [^long x ^long y]
  (let [v (hash-byte x y 104)
        inside? (and (> x 2) (< x 14) (> y 3) (< y 13))]
    (if inside?
      [(+ 180 (rem v 20)) (+ 50 (rem v 15)) (+ 50 (rem v 10)) 255]
      [0 0 0 0])))

;; Cooked beef: brown cooked meat
(defn- cooked-beef-pixel [^long x ^long y]
  (let [v (hash-byte x y 105)
        inside? (and (> x 2) (< x 14) (> y 3) (< y 13))]
    (if inside?
      [(+ 140 (rem v 15)) (+ 80 (rem v 12)) (+ 40 (rem v 10)) 255]
      [0 0 0 0])))

;; Bread: golden loaf shape
(defn- bread-pixel [^long x ^long y]
  (let [v (hash-byte x y 106)
        inside? (and (> x 2) (< x 14) (> y 5) (< y 12))
        crust? (or (= y 5) (= y 11) (= x 3) (= x 13))]
    (cond
      (and inside? crust?) [(+ 180 (rem v 10)) (+ 120 (rem v 8)) (+ 40 (rem v 6)) 255]
      inside? [(+ 210 (rem v 10)) (+ 170 (rem v 8)) (+ 80 (rem v 6)) 255]
      :else [0 0 0 0])))

;; Apple: red circle with green stem
(defn- apple-pixel [^long x ^long y]
  (let [cx (- x 8) cy (- y 9)
        dist-sq (+ (* cx cx) (* cy cy))
        v (hash-byte x y 107)]
    (cond
      (and (< (Math/abs (- x 8)) 2) (< y 5) (> y 2)) [(+ 60 (rem v 10)) (+ 120 (rem v 8)) (+ 30 (rem v 6)) 255]
      (< dist-sq 30) [(+ 200 (rem v 15)) (+ 30 (rem v 10)) (+ 20 (rem v 8)) 255]
      :else [0 0 0 0])))

;; Carrot: orange tapering shape
(defn- carrot-pixel [^long x ^long y]
  (let [v (hash-byte x y 108)
        mid-y 8
        width (max 0 (- 5 (long (/ (Math/abs (- y mid-y)) 1.5))))
        inside? (and (< (Math/abs (- x 8)) width) (> y 3) (< y 14))]
    (if inside?
      [(+ 230 (rem v 10)) (+ 130 (rem v 8)) (+ 20 (rem v 6)) 255]
      [0 0 0 0])))

;; Potato: brown oval
(defn- potato-pixel [^long x ^long y]
  (let [v (hash-byte x y 109)
        cx (- x 8) cy (- y 8)
        inside? (< (+ (* cx cx) (* cy cy 2)) 25)]
    (if inside?
      [(+ 180 (rem v 12)) (+ 150 (rem v 10)) (+ 80 (rem v 8)) 255]
      [0 0 0 0])))

;; Cooked potato: golden-brown baked potato
(defn- cooked-potato-pixel [^long x ^long y]
  (let [v (hash-byte x y 110)
        cx (- x 8) cy (- y 8)
        inside? (< (+ (* cx cx) (* cy cy 2)) 25)]
    (if inside?
      [(+ 200 (rem v 12)) (+ 170 (rem v 10)) (+ 60 (rem v 8)) 255]
      [0 0 0 0])))

;; Wheat seeds: small yellow-green dots
(defn- wheat-seeds-pixel [^long x ^long y]
  (let [v (hash-byte x y 111)
        seed? (and (< (rem v 4) 2) (> x 3) (< x 13) (> y 4) (< y 12))]
    (if seed?
      [(+ 180 (rem v 15)) (+ 190 (rem v 10)) (+ 50 (rem v 8)) 255]
      [0 0 0 0])))

;; Hunger drumstick: full (golden)
(defn- drumstick-full-pixel [^long x ^long y]
  (let [;; Drumstick shape: bone handle left, meat right
        bone? (and (> x 1) (< x 7) (> y 5) (< y 11))
        meat? (and (>= x 7) (< x 14) (> y 3) (< y 13)
                   (< (+ (* (- x 10) (- x 10)) (* (- y 8) (- y 8))) 20))
        v (hash-byte x y 112)]
    (cond
      meat? [(+ 190 (rem v 12)) (+ 130 (rem v 10)) (+ 40 (rem v 8)) 255]
      bone? [(+ 230 (rem v 8)) (+ 225 (rem v 6)) (+ 210 (rem v 5)) 255]
      :else [0 0 0 0])))

;; Hunger drumstick: half (partially eaten)
(defn- drumstick-half-pixel [^long x ^long y]
  (let [bone? (and (> x 1) (< x 7) (> y 5) (< y 11))
        meat? (and (>= x 7) (< x 12) (> y 5) (< y 11))
        v (hash-byte x y 113)]
    (cond
      meat? [(+ 190 (rem v 12)) (+ 130 (rem v 10)) (+ 40 (rem v 8)) 255]
      bone? [(+ 230 (rem v 8)) (+ 225 (rem v 6)) (+ 210 (rem v 5)) 255]
      :else [0 0 0 0])))

;; Hunger drumstick: empty (outline only)
(defn- drumstick-empty-pixel [^long x ^long y]
  (let [bone? (and (> x 1) (< x 7) (> y 5) (< y 11))
        meat? (and (>= x 7) (< x 14) (> y 3) (< y 13)
                   (< (+ (* (- x 10) (- x 10)) (* (- y 8) (- y 8))) 20))
        outline? (or bone? meat?)
        edge? (and outline?
                   (not (and (> x 2) (< x 6) (> y 6) (< y 10)
                             (not meat?))))]
    (if edge?
      [80 70 60 255]
      [0 0 0 0])))

;; Farmland: tilled dirt with lines
(defn- farmland-pixel [^long x ^long y]
  (let [v (hash-byte x y 115)
        row? (zero? (rem y 4))]
    (if row?
      [(+ 100 (rem v 10)) (+ 70 (rem v 8)) (+ 40 (rem v 6)) 255]
      [(+ 120 (rem v 12)) (+ 85 (rem v 10)) (+ 50 (rem v 8)) 255])))

;; Wheat stage 1: short green sprouts
(defn- wheat-stage1-pixel [^long x ^long y]
  (let [v (hash-byte x y 116)
        sprout? (and (> y 10) (zero? (rem x 3)))]
    (if sprout?
      [(+ 50 (rem v 10)) (+ 140 (rem v 10)) (+ 30 (rem v 8)) 255]
      [0 0 0 0])))

;; Wheat stage 2: medium green stalks
(defn- wheat-stage2-pixel [^long x ^long y]
  (let [v (hash-byte x y 117)
        stalk? (and (> y 6) (zero? (rem x 2)))]
    (if stalk?
      [(+ 70 (rem v 10)) (+ 160 (rem v 10)) (+ 40 (rem v 8)) 255]
      [0 0 0 0])))

;; Wheat stage 3: golden mature wheat
(defn- wheat-stage3-pixel [^long x ^long y]
  (let [v (hash-byte x y 118)
        stalk? (and (> y 3) (zero? (rem x 2)))
        head? (and (< y 6) stalk?)]
    (cond
      head? [(+ 210 (rem v 10)) (+ 180 (rem v 8)) (+ 50 (rem v 6)) 255]
      stalk? [(+ 180 (rem v 10)) (+ 160 (rem v 8)) (+ 60 (rem v 6)) 255]
      :else [0 0 0 0])))

;; Carrot stage (green tops, any stage)
(defn- carrot-stage-pixel [^long x ^long y]
  (let [v (hash-byte x y 119)
        leaf? (and (> y 6) (< (Math/abs (- x 8)) 4))]
    (if leaf?
      [(+ 40 (rem v 10)) (+ 130 (rem v 10)) (+ 30 (rem v 8)) 255]
      [0 0 0 0])))

;; Potato stage (green bush, any stage)
(defn- potato-stage-pixel [^long x ^long y]
  (let [v (hash-byte x y 120)
        leaf? (and (> y 5) (< (Math/abs (- x 8)) 5))]
    (if leaf?
      [(+ 50 (rem v 10)) (+ 120 (rem v 10)) (+ 35 (rem v 8)) 255]
      [0 0 0 0])))

;; Generate all 16 tool pixel fns (4 tiers × 4 types)
(def ^:private wood-pickaxe-pixel    (make-tool-pixel-fn :wood pickaxe-shape))
(def ^:private stone-pickaxe-pixel   (make-tool-pixel-fn :stone pickaxe-shape))
(def ^:private iron-pickaxe-pixel    (make-tool-pixel-fn :iron pickaxe-shape))
(def ^:private diamond-pickaxe-pixel (make-tool-pixel-fn :diamond pickaxe-shape))
(def ^:private wood-axe-pixel        (make-tool-pixel-fn :wood axe-shape))
(def ^:private stone-axe-pixel       (make-tool-pixel-fn :stone axe-shape))
(def ^:private iron-axe-pixel        (make-tool-pixel-fn :iron axe-shape))
(def ^:private diamond-axe-pixel     (make-tool-pixel-fn :diamond axe-shape))
(def ^:private wood-shovel-pixel     (make-tool-pixel-fn :wood shovel-shape))
(def ^:private stone-shovel-pixel    (make-tool-pixel-fn :stone shovel-shape))
(def ^:private iron-shovel-pixel     (make-tool-pixel-fn :iron shovel-shape))
(def ^:private diamond-shovel-pixel  (make-tool-pixel-fn :diamond shovel-shape))
(def ^:private wood-sword-pixel      (make-tool-pixel-fn :wood sword-shape))
(def ^:private stone-sword-pixel     (make-tool-pixel-fn :stone sword-shape))
(def ^:private iron-sword-pixel      (make-tool-pixel-fn :iron sword-shape))
(def ^:private diamond-sword-pixel   (make-tool-pixel-fn :diamond sword-shape))

;; ================================================================
;; Texture generator registry — indexed by layer
;; ================================================================

(def texture-generators
  [stone-pixel            ;; 0
   dirt-pixel             ;; 1
   grass-top-pixel        ;; 2
   grass-side-pixel       ;; 3
   water-pixel            ;; 4
   sand-pixel             ;; 5
   wood-side-pixel        ;; 6
   wood-top-pixel         ;; 7
   leaves-pixel           ;; 8
   bedrock-pixel          ;; 9
   torch-pixel            ;; 10
   glass-pixel            ;; 11
   gravel-pixel           ;; 12
   ;; Biome surface (13-21)
   snow-pixel             ;; 13
   ice-pixel              ;; 14
   clay-pixel             ;; 15
   sandstone-side-pixel   ;; 16
   sandstone-top-pixel    ;; 17
   red-sand-pixel         ;; 18
   podzol-top-pixel       ;; 19
   snowy-grass-top-pixel  ;; 20
   snowy-grass-side-pixel ;; 21
   ;; Ores (22-26)
   coal-ore-pixel         ;; 22
   iron-ore-pixel         ;; 23
   copper-ore-pixel       ;; 24
   gold-ore-pixel         ;; 25
   diamond-ore-pixel      ;; 26
   ;; Stone variants (27-31)
   cobblestone-pixel      ;; 27
   granite-pixel          ;; 28
   andesite-pixel         ;; 29
   obsidian-pixel         ;; 30
   basalt-pixel           ;; 31
   ;; Birch (32-34)
   birch-side-pixel       ;; 32
   birch-top-pixel        ;; 33
   birch-leaves-pixel     ;; 34
   ;; Pine (35-37)
   pine-side-pixel        ;; 35
   pine-top-pixel         ;; 36
   pine-leaves-pixel      ;; 37
   ;; Jungle (38-40)
   jungle-side-pixel      ;; 38
   jungle-top-pixel       ;; 39
   jungle-leaves-pixel    ;; 40
   ;; Dark oak (41-43)
   dark-oak-side-pixel    ;; 41
   dark-oak-top-pixel     ;; 42
   dark-oak-leaves-pixel  ;; 43
   ;; Plants (44-52)
   cactus-side-pixel      ;; 44
   cactus-top-pixel       ;; 45
   tall-grass-pixel       ;; 46
   fern-pixel             ;; 47
   dead-bush-pixel        ;; 48
   flower-red-pixel       ;; 49
   flower-yellow-pixel    ;; 50
   mushroom-red-pixel     ;; 51
   mushroom-brown-pixel   ;; 52
   ;; Underground (53-55)
   glowstone-pixel        ;; 53
   crystal-pixel          ;; 54
   mossy-cobble-pixel     ;; 55
   ;; Functional (56-60)
   crafting-top-pixel     ;; 56
   crafting-side-pixel    ;; 57
   furnace-front-pixel    ;; 58
   furnace-side-pixel     ;; 59
   furnace-top-pixel      ;; 60
   ;; Lava (61)
   lava-pixel             ;; 61
   ;; HUD icons (62-64)
   heart-full-pixel       ;; 62
   heart-half-pixel       ;; 63
   heart-empty-pixel      ;; 64
   ;; Item icons (65-71)
   planks-pixel           ;; 65
   coal-pixel             ;; 66
   diamond-item-pixel     ;; 67
   iron-ingot-pixel       ;; 68
   gold-ingot-pixel       ;; 69
   copper-ingot-pixel     ;; 70
   stick-pixel            ;; 71
   ;; Pickaxes (72-75)
   wood-pickaxe-pixel     ;; 72
   stone-pickaxe-pixel    ;; 73
   iron-pickaxe-pixel     ;; 74
   diamond-pickaxe-pixel  ;; 75
   ;; Axes (76-79)
   wood-axe-pixel         ;; 76
   stone-axe-pixel        ;; 77
   iron-axe-pixel         ;; 78
   diamond-axe-pixel      ;; 79
   ;; Shovels (80-83)
   wood-shovel-pixel      ;; 80
   stone-shovel-pixel     ;; 81
   iron-shovel-pixel      ;; 82
   diamond-shovel-pixel   ;; 83
   ;; Swords (84-87)
   wood-sword-pixel       ;; 84
   stone-sword-pixel      ;; 85
   iron-sword-pixel       ;; 86
   diamond-sword-pixel    ;; 87
   ;; Mob textures (88-95)
   cow-body-pixel          ;; 88
   cow-head-pixel          ;; 89
   animal-leg-pixel        ;; 90
   pig-body-pixel          ;; 91
   pig-head-pixel          ;; 92
   pig-leg-pixel           ;; 93
   chicken-pixel           ;; 94
   chicken-leg-pixel       ;; 95
   ;; Hostile mob textures (96-103)
   zombie-body-pixel       ;; 96
   zombie-head-pixel       ;; 97
   skeleton-body-pixel     ;; 98
   skeleton-head-pixel     ;; 99
   lurker-body-pixel      ;; 100
   lurker-face-pixel      ;; 101
   spider-body-pixel       ;; 102
   spider-head-pixel       ;; 103
   ;; Food/farming/hunger (104-115)
   raw-beef-pixel          ;; 104
   cooked-beef-pixel       ;; 105
   bread-pixel             ;; 106
   apple-pixel             ;; 107
   carrot-pixel            ;; 108
   potato-pixel            ;; 109
   cooked-potato-pixel     ;; 110
   wheat-seeds-pixel       ;; 111
   drumstick-full-pixel    ;; 112
   drumstick-half-pixel    ;; 113
   drumstick-empty-pixel   ;; 114
   farmland-pixel          ;; 115
   wheat-stage1-pixel      ;; 116
   wheat-stage2-pixel      ;; 117
   wheat-stage3-pixel      ;; 118
   carrot-stage-pixel      ;; 119
   potato-stage-pixel])    ;; 120

(defn generate-block-textures
  "Generate all block texture images. Returns vec of {:pixels ByteBuffer :width :height :channels 4}."
  []
  (mapv (fn [gen-fn]
          {:pixels (make-rgba gen-fn)
           :width TEX-SIZE
           :height TEX-SIZE
           :channels 4})
        texture-generators))

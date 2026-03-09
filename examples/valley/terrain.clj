(ns valley.terrain
  "Biome-based procedural terrain generation.
  Uses temperature + humidity noise for biome selection, multi-octave noise
  for height, 3D noise for caves, and biome-specific vegetation/ores."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w]
            [raster.noise :as noise]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Terrain parameters
;; ================================================================

(def ^:const SEA-LEVEL 48)
(def ^:const BASE-HEIGHT 52)
(def ^:const HEIGHT-SCALE 24.0)
(def ^:const TERRAIN-FREQ (/ 1.0 80.0))
(def ^:const CAVE-FREQ (/ 1.0 30.0))
(def ^:const CAVE-THRESHOLD 0.55)

;; Biome noise parameters — large scale for smooth transitions
(def ^:const BIOME-FREQ (/ 1.0 256.0))

(defn- hash-byte ^long [^long x ^long y ^long seed]
  (let [h (bit-xor (* (+ x (* y 37) seed) 2654435761) 0xDEADBEEF)]
    (bit-and (bit-xor h (unsigned-bit-shift-right h 16)) 0xFF)))

;; ================================================================
;; Biome definitions
;; ================================================================

;; Biomes are selected by temperature (0-1) and humidity (0-1)
;; Each biome defines surface/subsurface blocks, terrain shape, and features.

(def biomes
  {:desert      {:surface w/SAND         :fill w/SANDSTONE  :height-scale 8.0
                 :base-offset -4  :tree-type nil  :tree-density 0.0
                 :vegetation [:dead-bush :cactus]  :veg-density 0.02}
   :savanna     {:surface w/GRASS        :fill w/DIRT       :height-scale 12.0
                 :base-offset 0   :tree-type :acacia  :tree-density 0.005
                 :vegetation [:tall-grass :dead-bush]  :veg-density 0.06}
   :plains      {:surface w/GRASS        :fill w/DIRT       :height-scale 16.0
                 :base-offset 0   :tree-type :oak  :tree-density 0.015
                 :vegetation [:tall-grass :flower-red :flower-yellow]  :veg-density 0.08}
   :forest      {:surface w/GRASS        :fill w/DIRT       :height-scale 20.0
                 :base-offset 2   :tree-type :oak  :tree-density 0.06
                 :vegetation [:tall-grass :fern :mushroom-brown]  :veg-density 0.05}
   :jungle      {:surface w/GRASS        :fill w/DIRT       :height-scale 28.0
                 :base-offset 4   :tree-type :jungle  :tree-density 0.08
                 :vegetation [:tall-grass :fern :flower-red]  :veg-density 0.12}
   :swamp       {:surface w/GRASS        :fill w/CLAY       :height-scale 6.0
                 :base-offset -6  :tree-type :dark-oak  :tree-density 0.03
                 :vegetation [:mushroom-red :mushroom-brown :fern]  :veg-density 0.08}
   :taiga       {:surface w/PODZOL       :fill w/DIRT       :height-scale 20.0
                 :base-offset 3   :tree-type :pine  :tree-density 0.05
                 :vegetation [:fern :mushroom-brown]  :veg-density 0.04}
   :tundra      {:surface w/SNOWY-GRASS  :fill w/DIRT       :height-scale 10.0
                 :base-offset -2  :tree-type nil  :tree-density 0.0
                 :vegetation [:dead-bush]  :veg-density 0.01}
   :snowy-forest {:surface w/SNOWY-GRASS :fill w/DIRT       :height-scale 18.0
                  :base-offset 2  :tree-type :pine  :tree-density 0.04
                  :vegetation [:fern]  :veg-density 0.02}
   :mountains   {:surface w/STONE        :fill w/STONE      :height-scale 48.0
                 :base-offset 15  :tree-type :pine  :tree-density 0.01
                 :vegetation []  :veg-density 0.0}
   :mushroom    {:surface w/GRASS        :fill w/DIRT       :height-scale 12.0
                 :base-offset 0  :tree-type nil  :tree-density 0.0
                 :vegetation [:mushroom-red :mushroom-brown]  :veg-density 0.15}})

(def vegetation-block
  {:tall-grass w/TALL-GRASS
   :fern w/FERN
   :dead-bush w/DEAD-BUSH
   :flower-red w/FLOWER-RED
   :flower-yellow w/FLOWER-YELLOW
   :mushroom-red w/MUSHROOM-RED
   :mushroom-brown w/MUSHROOM-BROWN
   :cactus w/CACTUS})

;; ================================================================
;; Ore distribution
;; ================================================================

;; Each ore: [block-id min-y max-y vein-size frequency noise-seed]
(def ore-defs
  [[w/COAL-ORE    5  80 8 0.015 100]
   [w/IRON-ORE    5  64 6 0.010 200]
   [w/COPPER-ORE  5  56 5 0.008 300]
   [w/GOLD-ORE    5  32 4 0.005 400]
   [w/DIAMOND-ORE 5  16 3 0.003 500]
   [w/GRANITE     10 80 12 0.020 600]
   [w/ANDESITE    10 80 12 0.020 700]
   [w/BASALT      5  40 8 0.012 800]])

;; ================================================================
;; Generator state
;; ================================================================

(defn make-terrain-gen
  "Create terrain generator with seeded noise tables."
  ([] (make-terrain-gen 42))
  ([^long seed]
   {:height-perm  (noise/make-perm seed)
    :cave-perm    (noise/make-perm (+ seed 1337))
    :tree-perm    (noise/make-perm (+ seed 7777))
    :temp-perm    (noise/make-perm (+ seed 2222))
    :humid-perm   (noise/make-perm (+ seed 3333))
    :ore-perm     (noise/make-perm (+ seed 4444))
    :detail-perm  (noise/make-perm (+ seed 5555))
    :mushroom-perm (noise/make-perm (+ seed 6666))
    :seed seed}))

;; ================================================================
;; Biome selection
;; ================================================================

(defn biome-at
  "Determine biome at world (x, z) using temperature and humidity noise."
  [gen ^long wx ^long wz]
  (let [temp (+ 0.5 (* 0.5 (noise/fbm2d (:temp-perm gen)
                               (* (double wx) BIOME-FREQ)
                               (* (double wz) BIOME-FREQ)
                               3 0.5 2.0)))
        humid (+ 0.5 (* 0.5 (noise/fbm2d (:humid-perm gen)
                                (* (double wx) BIOME-FREQ)
                                (* (double wz) BIOME-FREQ)
                                3 0.5 2.0)))
        ;; Erosion/continentalness for mountains
        erosion (noise/fbm2d (:detail-perm gen)
                  (* (double wx) (/ 1.0 180.0))
                  (* (double wz) (/ 1.0 180.0))
                  2 0.5 2.0)
        ;; Rare mushroom biome check
        mush (noise/perlin2d (:mushroom-perm gen)
               (* (double wx) (/ 1.0 400.0))
               (* (double wz) (/ 1.0 400.0)))]
    (cond
      ;; Mountains at extreme erosion
      (> erosion 0.6) :mountains
      ;; Rare mushroom biome
      (> mush 0.85) :mushroom
      ;; Temperature/humidity grid
      (> temp 0.65)
      (cond
        (< humid 0.35) :desert
        (> humid 0.65) :jungle
        :else :savanna)
      (> temp 0.35)
      (cond
        (< humid 0.3) :savanna
        (> humid 0.7) :swamp
        (> humid 0.5) :forest
        :else :plains)
      ;; Cold
      :else
      (cond
        (> humid 0.45) :snowy-forest
        (> erosion 0.3) :taiga
        :else :tundra))))

;; ================================================================
;; Terrain height
;; ================================================================

(defn surface-height
  "Get terrain surface height at world (x, z) accounting for biome."
  ^long [gen ^long wx ^long wz]
  (let [biome-key (biome-at gen wx wz)
        biome (get biomes biome-key)
        h (noise/fbm2d (:height-perm gen)
            (* (double wx) TERRAIN-FREQ)
            (* (double wz) TERRAIN-FREQ)
            4 0.5 2.0)
        ;; Detail noise for roughness
        detail (noise/fbm2d (:detail-perm gen)
                 (* (double wx) (/ 1.0 20.0))
                 (* (double wz) (/ 1.0 20.0))
                 2 0.6 2.0)
        scale (:height-scale biome)
        offset (:base-offset biome)]
    (long (+ BASE-HEIGHT offset (* h scale) (* detail 4.0)))))

;; ================================================================
;; Caves
;; ================================================================

(defn has-cave?
  "Check if world position is carved out by cave."
  [gen ^long wx ^long wy ^long wz]
  (let [v (Math/abs (double (noise/fbm3d (:cave-perm gen)
            (* (double wx) CAVE-FREQ)
            (* (double wy) CAVE-FREQ)
            (* (double wz) CAVE-FREQ)
            3 0.5 2.0)))]
    (> v CAVE-THRESHOLD)))

(defn has-lava-pool?
  "Check if deep underground position should be lava."
  [gen ^long wx ^long wy ^long wz]
  (and (< wy 12)
       (> wy 1)
       (> (Math/abs (double (noise/fbm3d (:detail-perm gen)
            (* (double wx) (/ 1.0 15.0))
            (* (double wy) (/ 1.0 15.0))
            (* (double wz) (/ 1.0 15.0))
            2 0.5 2.0)))
          0.65)))

;; ================================================================
;; Tree placement
;; ================================================================

(defn should-place-tree?
  "Check if a tree should be placed at (wx, wz) for given biome."
  [gen ^long wx ^long wz ^double density]
  (when (pos? density)
    (let [v (noise/perlin2d (:tree-perm gen) (* (double wx) 0.5) (* (double wz) 0.5))
          threshold (- 1.0 (* density 15.0))]
      (and (> v threshold)
           (zero? (rem (Math/abs (long (+ (* wx 73) (* wz 179)))) 7))))))

(defn- place-oak-tree [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        trunk-h (+ 4 (long (rem (Math/abs (+ (* wx 31) (* wz 17))) 3)))]
    (as-> world w
      (reduce (fn [w3 dy] (w/set-world-block! w3 wx (+ sh 1 dy) wz w/WOOD))
              w (range trunk-h))
      (reduce (fn [w3 [dx dy dz]]
                (let [bx (+ wx (long dx)) by (+ sh trunk-h (long dy)) bz (+ wz (long dz))]
                  (if (= (long (w/get-world-block w3 bx by bz)) w/AIR)
                    (w/set-world-block! w3 bx by bz w/LEAVES)
                    w3)))
              w (for [dx (range -2 3) dy (range -1 3) dz (range -2 3)
                      :when (<= (+ (* dx dx) (* dy dy) (* dz dz)) 6)]
                  [dx dy dz])))))

(defn- place-birch-tree [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        trunk-h (+ 5 (long (rem (Math/abs (+ (* wx 41) (* wz 23))) 3)))]
    (as-> world w
      (reduce (fn [w3 dy] (w/set-world-block! w3 wx (+ sh 1 dy) wz w/BIRCH-WOOD))
              w (range trunk-h))
      (reduce (fn [w3 [dx dy dz]]
                (let [bx (+ wx (long dx)) by (+ sh trunk-h (long dy)) bz (+ wz (long dz))]
                  (if (= (long (w/get-world-block w3 bx by bz)) w/AIR)
                    (w/set-world-block! w3 bx by bz w/BIRCH-LEAVES)
                    w3)))
              w (for [dx (range -2 3) dy (range -1 3) dz (range -2 3)
                      :when (<= (+ (* dx dx) (* dy dy) (* dz dz)) 5)]
                  [dx dy dz])))))

(defn- place-pine-tree [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        trunk-h (+ 6 (long (rem (Math/abs (+ (* wx 37) (* wz 19))) 4)))]
    (as-> world w
      (reduce (fn [w3 dy] (w/set-world-block! w3 wx (+ sh 1 dy) wz w/PINE-WOOD))
              w (range trunk-h))
      ;; Conical canopy
      (reduce (fn [w3 [dx dy dz]]
                (let [bx (+ wx (long dx)) by (+ sh 1 trunk-h (long dy)) bz (+ wz (long dz))
                      r (max (Math/abs (long dx)) (Math/abs (long dz)))]
                  (if (and (= (long (w/get-world-block w3 bx by bz)) w/AIR)
                           (<= r (max 0 (- 3 (long dy)))))
                    (w/set-world-block! w3 bx by bz w/PINE-LEAVES)
                    w3)))
              w (for [dx (range -3 4) dy (range -4 2) dz (range -3 4)]
                  [dx dy dz])))))

(defn- place-jungle-tree [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        trunk-h (+ 8 (long (rem (Math/abs (+ (* wx 47) (* wz 29))) 5)))]
    (as-> world w
      ;; Thick trunk (2x2)
      (reduce (fn [w3 dy]
                (reduce (fn [w4 [dx dz]]
                          (w/set-world-block! w4 (+ wx dx) (+ sh 1 dy) (+ wz dz) w/JUNGLE-WOOD))
                        w3 [[0 0] [1 0] [0 1] [1 1]]))
              w (range trunk-h))
      ;; Large canopy
      (reduce (fn [w3 [dx dy dz]]
                (let [bx (+ wx (long dx)) by (+ sh trunk-h (long dy)) bz (+ wz (long dz))
                      dist (+ (* dx dx) (* dz dz))]
                  (if (and (= (long (w/get-world-block w3 bx by bz)) w/AIR)
                           (<= dist (- 20 (* (long dy) (long dy)))))
                    (w/set-world-block! w3 bx by bz w/JUNGLE-LEAVES)
                    w3)))
              w (for [dx (range -4 5) dy (range -2 4) dz (range -4 5)]
                  [dx dy dz])))))

(defn- place-dark-oak-tree [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        trunk-h (+ 4 (long (rem (Math/abs (+ (* wx 53) (* wz 31))) 3)))]
    (as-> world w
      ;; Thick trunk (2x2)
      (reduce (fn [w3 dy]
                (reduce (fn [w4 [dx dz]]
                          (w/set-world-block! w4 (+ wx dx) (+ sh 1 dy) (+ wz dz) w/DARK-OAK-WOOD))
                        w3 [[0 0] [1 0] [0 1] [1 1]]))
              w (range trunk-h))
      ;; Wide canopy
      (reduce (fn [w3 [dx dy dz]]
                (let [bx (+ wx (long dx)) by (+ sh trunk-h (long dy)) bz (+ wz (long dz))]
                  (if (and (= (long (w/get-world-block w3 bx by bz)) w/AIR)
                           (<= (+ (* dx dx) (* dz dz)) 12))
                    (w/set-world-block! w3 bx by bz w/DARK-OAK-LEAVES)
                    w3)))
              w (for [dx (range -3 4) dy (range -1 3) dz (range -3 4)]
                  [dx dy dz])))))

(defn- place-cactus [world wx sh wz gen]
  (let [wx (long wx) sh (long sh) wz (long wz)
        h (+ 1 (long (rem (Math/abs (+ (* wx 59) (* wz 41))) 3)))]
    (reduce (fn [w dy] (w/set-world-block! w wx (+ sh 1 dy) wz w/CACTUS))
            world (range h))))

(defn- place-tree [world tree-type wx sh wz gen]
  (case tree-type
    :oak (if (> (rem (Math/abs (long (+ (* (long wx) 71) (* (long wz) 113)))) 5) 2)
           (place-birch-tree world wx sh wz gen)
           (place-oak-tree world wx sh wz gen))
    :pine (place-pine-tree world wx sh wz gen)
    :jungle (place-jungle-tree world wx sh wz gen)
    :dark-oak (place-dark-oak-tree world wx sh wz gen)
    :acacia (place-oak-tree world wx sh wz gen) ;; simplified acacia
    world))

;; ================================================================
;; Vegetation placement
;; ================================================================

(defn- place-vegetation
  "Place small vegetation (flowers, grass, mushrooms, cactus) in a chunk column."
  [world gen ^long cx ^long cz]
  (let [wx0 (* cx (long w/CHUNK-SIZE))
        wz0 (* cz (long w/CHUNK-SIZE))]
    (reduce
      (fn [w [lx lz]]
        (let [wx (+ wx0 (long lx))
              wz (+ wz0 (long lz))
              biome-key (biome-at gen wx wz)
              biome (get biomes biome-key)
              veg-types (:vegetation biome)
              density (:veg-density biome)]
          (if (or (empty? veg-types) (zero? density))
            w
            (let [v (noise/perlin2d (:detail-perm gen)
                      (* (double wx) 0.7) (* (double wz) 0.7))]
              (if (> v (- 1.0 density))
                (let [sh (surface-height gen wx wz)]
                  (if (> sh SEA-LEVEL)
                    (let [vi (rem (Math/abs (long (+ (* wx 89) (* wz 61)))) (count veg-types))
                          veg-key (nth veg-types vi)
                          block-id (get vegetation-block veg-key)]
                      (if (and block-id
                               (= (long (w/get-world-block w wx (+ sh 1) wz)) w/AIR))
                        (if (= veg-key :cactus)
                          (place-cactus w wx sh wz gen)
                          (w/set-world-block! w wx (+ sh 1) wz block-id))
                        w))
                    w))
                w)))))
      world
      (for [lx (range w/CHUNK-SIZE) lz (range w/CHUNK-SIZE)] [lx lz]))))

;; ================================================================
;; Ore placement
;; ================================================================

(defn- place-ores-in-chunk
  "Place ore veins within a chunk during generation."
  [blocks gen wx0 wy0 wz0]
  (let [wx0 (long wx0) wy0 (long wy0) wz0 (long wz0)]
    (doseq [[ore-id min-y max-y vein-size freq ore-seed] ore-defs]
    (dotimes [lx w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (dotimes [ly w/CHUNK-SIZE]
          (let [wx (+ wx0 lx) wy (+ wy0 ly) wz (+ wz0 lz)]
            (when (and (>= wy (long min-y)) (<= wy (long max-y)))
              (let [idx (w/block-index lx ly lz)
                    current (aget ^ints blocks idx)]
                (when (= current w/STONE)
                  (let [v (noise/fbm3d (:ore-perm gen)
                            (* (double wx) (/ 1.0 (double vein-size)))
                            (* (double wy) (/ 1.0 (double vein-size)))
                            (* (double wz) (/ 1.0 (double vein-size)))
                            2 0.5 2.0)
                        ;; Offset by seed for each ore type
                        sv (+ v (* (double ore-seed) 0.001))]
                    (when (> (Math/abs sv) (- 1.0 (double freq)))
                      (aset ^ints blocks idx (int ore-id))))))))))))))

;; ================================================================
;; Chunk generation
;; ================================================================

(defn generate-chunk
  "Generate terrain for a single chunk at chunk coords [cx cy cz]."
  [gen ^long cx ^long cy ^long cz]
  (let [chunk (w/create-chunk cx cy cz)
        blocks ^ints (:blocks chunk)
        wx0 (* cx (long w/CHUNK-SIZE))
        wy0 (* cy (long w/CHUNK-SIZE))
        wz0 (* cz (long w/CHUNK-SIZE))]
    ;; Main terrain pass
    (dotimes [lx w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (let [wx (+ wx0 lx)
              wz (+ wz0 lz)
              biome-key (biome-at gen wx wz)
              biome (get biomes biome-key)
              sh (surface-height gen wx wz)
              surface-block (long (:surface biome))
              fill-block (long (:fill biome))]
          (dotimes [ly w/CHUNK-SIZE]
            (let [wy (+ wy0 ly)
                  idx (w/block-index lx ly lz)
                  cave? (and (> wy 1) (< wy (- sh 1)) (has-cave? gen wx wy wz))
                  block (cond
                          ;; Bedrock
                          (zero? wy) w/BEDROCK
                          ;; Deep underground lava
                          (and cave? (has-lava-pool? gen wx wy wz)) w/LAVA
                          ;; Cave
                          cave? w/AIR
                          ;; Deep stone
                          (< wy (- sh 4)) w/STONE
                          ;; Fill layer (dirt/clay/sandstone)
                          (< wy sh) fill-block
                          ;; Surface
                          (= wy sh)
                          (if (< sh SEA-LEVEL)
                            w/SAND  ;; beaches regardless of biome
                            surface-block)
                          ;; Water fill
                          (<= wy SEA-LEVEL) w/WATER
                          ;; Air
                          :else w/AIR)]
              (aset blocks idx (int block)))))))
    ;; Ore placement pass
    (place-ores-in-chunk blocks gen wx0 wy0 wz0)
    ;; Place glowstone and crystal in caves
    (dotimes [lx w/CHUNK-SIZE]
      (dotimes [lz w/CHUNK-SIZE]
        (dotimes [ly w/CHUNK-SIZE]
          (let [wy (+ wy0 ly)
                idx (w/block-index lx ly lz)]
            (when (and (> wy 5) (< wy 35)
                       (= (aget blocks idx) w/AIR))
              ;; Check if ceiling above
              (when (and (< ly (dec w/CHUNK-SIZE))
                         (= (aget blocks (w/block-index lx (inc ly) lz)) w/STONE))
                (let [wx (+ wx0 lx) wz (+ wz0 lz)
                      h (hash-byte wx wz (int wy))]
                  (when (> h 252)
                    (aset blocks idx
                      (int (if (> h 254) w/CRYSTAL w/GLOWSTONE)))))))))))
    chunk))

;; ================================================================
;; Tree placement (post chunk-gen, needs cross-chunk writes)
;; ================================================================

(defn place-trees
  "Place trees on surface within one xz chunk column."
  [world gen ^long cx ^long cz]
  (let [wx0 (* cx (long w/CHUNK-SIZE))
        wz0 (* cz (long w/CHUNK-SIZE))]
    (reduce
      (fn [w [lx lz]]
        (let [wx (+ wx0 (long lx))
              wz (+ wz0 (long lz))
              biome-key (biome-at gen wx wz)
              biome (get biomes biome-key)
              tree-type (:tree-type biome)
              density (:tree-density biome)]
          (if (and tree-type (should-place-tree? gen wx wz density))
            (let [sh (surface-height gen wx wz)]
              (if (> sh SEA-LEVEL)
                (place-tree w tree-type wx sh wz gen)
                w))
            w)))
      world
      (for [lx (range w/CHUNK-SIZE) lz (range w/CHUNK-SIZE)] [lx lz]))))

;; ================================================================
;; World generation
;; ================================================================

(defn generate-world
  "Generate world with chunks in [-radius, radius] xz, [0, height-chunks) y."
  ([gen ^long radius] (generate-world gen radius 6))
  ([gen ^long radius ^long height-chunks]
   (let [world (reduce
                 (fn [w [cx cy cz]]
                   (assoc w [cx cy cz] (generate-chunk gen cx cy cz)))
                 {}
                 (for [cx (range (- radius) (inc radius))
                       cz (range (- radius) (inc radius))
                       cy (range height-chunks)]
                   [cx cy cz]))]
     ;; Post-generation passes: trees, vegetation
     (-> (reduce
           (fn [w [cx cz]] (place-trees w gen cx cz))
           world
           (for [cx (range (- radius) (inc radius))
                 cz (range (- radius) (inc radius))]
             [cx cz]))
         ;; Vegetation pass
         (#(reduce
             (fn [w [cx cz]] (place-vegetation w gen cx cz))
             %
             (for [cx (range (- radius) (inc radius))
                   cz (range (- radius) (inc radius))]
               [cx cz])))))))


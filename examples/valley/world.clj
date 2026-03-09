(ns valley.world
  "Voxel world: block types, chunk storage, coordinate math."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Block types
;; ================================================================

(def ^:const CHUNK-SIZE 16)
(def ^:const CHUNK-VOLUME (* CHUNK-SIZE CHUNK-SIZE CHUNK-SIZE)) ;; 4096

;; Block IDs
(def ^:const AIR 0)
(def ^:const STONE 1)
(def ^:const DIRT 2)
(def ^:const GRASS 3)
(def ^:const WATER 4)
(def ^:const SAND 5)
(def ^:const WOOD 6)        ;; Oak
(def ^:const LEAVES 7)      ;; Oak
(def ^:const BEDROCK 8)
(def ^:const TORCH 9)
(def ^:const GLASS 10)
(def ^:const GRAVEL 11)
;; Biome surface
(def ^:const SNOW-BLOCK 12)
(def ^:const ICE 13)
(def ^:const CLAY 14)
(def ^:const SANDSTONE 15)
(def ^:const RED-SAND 16)
(def ^:const PODZOL 17)
(def ^:const SNOWY-GRASS 18)
;; Ores
(def ^:const COAL-ORE 19)
(def ^:const IRON-ORE 20)
(def ^:const COPPER-ORE 21)
(def ^:const GOLD-ORE 22)
(def ^:const DIAMOND-ORE 23)
;; Stone variants
(def ^:const COBBLESTONE 24)
(def ^:const GRANITE 25)
(def ^:const ANDESITE 26)
(def ^:const OBSIDIAN 27)
(def ^:const BASALT 28)
;; Wood variants
(def ^:const BIRCH-WOOD 29)
(def ^:const BIRCH-LEAVES 30)
(def ^:const PINE-WOOD 31)
(def ^:const PINE-LEAVES 32)
(def ^:const JUNGLE-WOOD 33)
(def ^:const JUNGLE-LEAVES 34)
(def ^:const DARK-OAK-WOOD 35)
(def ^:const DARK-OAK-LEAVES 36)
;; Plants
(def ^:const CACTUS 37)
(def ^:const TALL-GRASS 38)
(def ^:const FERN 39)
(def ^:const DEAD-BUSH 40)
(def ^:const FLOWER-RED 41)
(def ^:const FLOWER-YELLOW 42)
(def ^:const MUSHROOM-RED 43)
(def ^:const MUSHROOM-BROWN 44)
;; Underground special
(def ^:const GLOWSTONE 45)
(def ^:const CRYSTAL 46)
(def ^:const MOSSY-COBBLESTONE 47)
;; Functional
(def ^:const CRAFTING-TABLE 48)
(def ^:const FURNACE 49)
;; Liquids
(def ^:const LAVA 50)
;; Farming
(def ^:const FARMLAND 51)
(def ^:const FARMLAND-WET 52)
(def ^:const WHEAT-1 53)
(def ^:const WHEAT-2 54)
(def ^:const WHEAT-3 55)
(def ^:const CARROT-1 56)
(def ^:const CARROT-2 57)
(def ^:const CARROT-3 58)
(def ^:const POTATO-1 59)
(def ^:const POTATO-2 60)
(def ^:const POTATO-3 61)

(def block-names
  {AIR "air" STONE "stone" DIRT "dirt" GRASS "grass"
   WATER "water" SAND "sand" WOOD "wood" LEAVES "leaves"
   BEDROCK "bedrock" TORCH "torch" GLASS "glass" GRAVEL "gravel"
   SNOW-BLOCK "snow" ICE "ice" CLAY "clay" SANDSTONE "sandstone"
   RED-SAND "red_sand" PODZOL "podzol" SNOWY-GRASS "snowy_grass"
   COAL-ORE "coal_ore" IRON-ORE "iron_ore" COPPER-ORE "copper_ore"
   GOLD-ORE "gold_ore" DIAMOND-ORE "diamond_ore"
   COBBLESTONE "cobblestone" GRANITE "granite" ANDESITE "andesite"
   OBSIDIAN "obsidian" BASALT "basalt"
   BIRCH-WOOD "birch_wood" BIRCH-LEAVES "birch_leaves"
   PINE-WOOD "pine_wood" PINE-LEAVES "pine_leaves"
   JUNGLE-WOOD "jungle_wood" JUNGLE-LEAVES "jungle_leaves"
   DARK-OAK-WOOD "dark_oak_wood" DARK-OAK-LEAVES "dark_oak_leaves"
   CACTUS "cactus" TALL-GRASS "tall_grass" FERN "fern"
   DEAD-BUSH "dead_bush" FLOWER-RED "flower_red"
   FLOWER-YELLOW "flower_yellow" MUSHROOM-RED "mushroom_red"
   MUSHROOM-BROWN "mushroom_brown"
   GLOWSTONE "glowstone" CRYSTAL "crystal"
   MOSSY-COBBLESTONE "mossy_cobblestone"
   CRAFTING-TABLE "crafting_table" FURNACE "furnace" LAVA "lava"
   FARMLAND "farmland" FARMLAND-WET "farmland_wet"
   WHEAT-1 "wheat_1" WHEAT-2 "wheat_2" WHEAT-3 "wheat_3"
   CARROT-1 "carrot_1" CARROT-2 "carrot_2" CARROT-3 "carrot_3"
   POTATO-1 "potato_1" POTATO-2 "potato_2" POTATO-3 "potato_3"})

(def name->id
  (into {} (map (fn [[k v]] [v k]) block-names)))

;; Block properties
(def block-props
  {AIR          {:solid? false :transparent? true  :light 0}
   STONE        {:solid? true  :transparent? false :light 0}
   DIRT         {:solid? true  :transparent? false :light 0}
   GRASS        {:solid? true  :transparent? false :light 0}
   WATER        {:solid? false :transparent? true  :light 0}
   SAND         {:solid? true  :transparent? false :light 0}
   WOOD         {:solid? true  :transparent? false :light 0}
   LEAVES       {:solid? true  :transparent? true  :light 0}
   BEDROCK      {:solid? true  :transparent? false :light 0}
   TORCH        {:solid? false :transparent? true  :light 14}
   GLASS        {:solid? true  :transparent? true  :light 0}
   GRAVEL       {:solid? true  :transparent? false :light 0}
   SNOW-BLOCK   {:solid? true  :transparent? false :light 0}
   ICE          {:solid? true  :transparent? true  :light 0}
   CLAY         {:solid? true  :transparent? false :light 0}
   SANDSTONE    {:solid? true  :transparent? false :light 0}
   RED-SAND     {:solid? true  :transparent? false :light 0}
   PODZOL       {:solid? true  :transparent? false :light 0}
   SNOWY-GRASS  {:solid? true  :transparent? false :light 0}
   COAL-ORE     {:solid? true  :transparent? false :light 0}
   IRON-ORE     {:solid? true  :transparent? false :light 0}
   COPPER-ORE   {:solid? true  :transparent? false :light 0}
   GOLD-ORE     {:solid? true  :transparent? false :light 0}
   DIAMOND-ORE  {:solid? true  :transparent? false :light 0}
   COBBLESTONE  {:solid? true  :transparent? false :light 0}
   GRANITE      {:solid? true  :transparent? false :light 0}
   ANDESITE     {:solid? true  :transparent? false :light 0}
   OBSIDIAN     {:solid? true  :transparent? false :light 0}
   BASALT       {:solid? true  :transparent? false :light 0}
   BIRCH-WOOD   {:solid? true  :transparent? false :light 0}
   BIRCH-LEAVES {:solid? true  :transparent? true  :light 0}
   PINE-WOOD    {:solid? true  :transparent? false :light 0}
   PINE-LEAVES  {:solid? true  :transparent? true  :light 0}
   JUNGLE-WOOD  {:solid? true  :transparent? false :light 0}
   JUNGLE-LEAVES {:solid? true :transparent? true  :light 0}
   DARK-OAK-WOOD   {:solid? true  :transparent? false :light 0}
   DARK-OAK-LEAVES {:solid? true  :transparent? true  :light 0}
   CACTUS       {:solid? true  :transparent? false :light 0}
   TALL-GRASS   {:solid? false :transparent? true  :light 0}
   FERN         {:solid? false :transparent? true  :light 0}
   DEAD-BUSH    {:solid? false :transparent? true  :light 0}
   FLOWER-RED   {:solid? false :transparent? true  :light 0}
   FLOWER-YELLOW {:solid? false :transparent? true :light 0}
   MUSHROOM-RED  {:solid? false :transparent? true :light 0}
   MUSHROOM-BROWN {:solid? false :transparent? true :light 0}
   GLOWSTONE    {:solid? true  :transparent? false :light 14}
   CRYSTAL      {:solid? true  :transparent? true  :light 10}
   MOSSY-COBBLESTONE {:solid? true :transparent? false :light 0}
   CRAFTING-TABLE {:solid? true :transparent? false :light 0}
   FURNACE      {:solid? true  :transparent? false :light 0}
   LAVA         {:solid? false :transparent? true  :light 15}
   FARMLAND     {:solid? true  :transparent? false :light 0}
   FARMLAND-WET {:solid? true  :transparent? false :light 0}
   WHEAT-1      {:solid? false :transparent? true  :light 0}
   WHEAT-2      {:solid? false :transparent? true  :light 0}
   WHEAT-3      {:solid? false :transparent? true  :light 0}
   CARROT-1     {:solid? false :transparent? true  :light 0}
   CARROT-2     {:solid? false :transparent? true  :light 0}
   CARROT-3     {:solid? false :transparent? true  :light 0}
   POTATO-1     {:solid? false :transparent? true  :light 0}
   POTATO-2     {:solid? false :transparent? true  :light 0}
   POTATO-3     {:solid? false :transparent? true  :light 0}})

;; Fast indexed property arrays (avoid hash lookups in hot loops)
(def ^:private max-block-id 61)

(def solid-arr
  (boolean-array (inc max-block-id)
    (mapv #(boolean (get-in block-props [% :solid?] false))
          (range (inc max-block-id)))))

(def transparent-arr
  (boolean-array (inc max-block-id)
    (mapv #(boolean (get-in block-props [% :transparent?] true))
          (range (inc max-block-id)))))

(def light-source-arr
  (byte-array (inc max-block-id)
    (mapv #(byte (get-in block-props [% :light] 0))
          (range (inc max-block-id)))))

;; sunlight_propagates: air, water, glass, leaves, torch, and other transparent non-solid
(def sunlight-propagates-arr
  (boolean-array (inc max-block-id)
    (mapv #(boolean (and (get-in block-props [% :transparent?] true)
                         (not (get-in block-props [% :solid?] false))))
          (range (inc max-block-id)))))

;; light_propagates: same as transparent
(def light-propagates-arr
  (boolean-array (inc max-block-id)
    (mapv #(boolean (get-in block-props [% :transparent?] true))
          (range (inc max-block-id)))))

(defn solid? [^long block-id]
  (if (and (>= block-id 0) (<= block-id max-block-id))
    (aget ^booleans solid-arr (int block-id))
    false))

(defn transparent? [^long block-id]
  (if (and (>= block-id 0) (<= block-id max-block-id))
    (aget ^booleans transparent-arr (int block-id))
    true))

(defn light-source ^long [^long block-id]
  (if (and (>= block-id 0) (<= block-id max-block-id))
    (long (aget ^bytes light-source-arr (int block-id)))
    0))

(defn sunlight-propagates? [^long block-id]
  (if (and (>= block-id 0) (<= block-id max-block-id))
    (aget ^booleans sunlight-propagates-arr (int block-id))
    true))

(defn light-propagates? [^long block-id]
  (if (and (>= block-id 0) (<= block-id max-block-id))
    (aget ^booleans light-propagates-arr (int block-id))
    true))

;; Texture face indices: [top bottom north south east west]
;; Each face maps to a layer in the texture array
;; Texture layers:
;;  0=stone, 1=dirt, 2=grass-top, 3=grass-side, 4=water, 5=sand
;;  6=wood-side, 7=wood-top, 8=leaves, 9=bedrock, 10=torch, 11=glass, 12=gravel
;;  13=snow, 14=ice, 15=clay, 16=sandstone-side, 17=sandstone-top
;;  18=red_sand, 19=podzol-top, 20=snowy_grass-top, 21=snowy_grass-side
;;  22=coal_ore, 23=iron_ore, 24=copper_ore, 25=gold_ore, 26=diamond_ore
;;  27=cobblestone, 28=granite, 29=andesite, 30=obsidian, 31=basalt
;;  32=birch-side, 33=birch-top, 34=birch_leaves
;;  35=pine-side, 36=pine-top, 37=pine_leaves
;;  38=jungle-side, 39=jungle-top, 40=jungle_leaves
;;  41=dark_oak-side, 42=dark_oak-top, 43=dark_oak_leaves
;;  44=cactus-side, 45=cactus-top
;;  46=tall_grass, 47=fern, 48=dead_bush
;;  49=flower_red, 50=flower_yellow
;;  51=mushroom_red, 52=mushroom_brown
;;  53=glowstone, 54=crystal, 55=mossy_cobblestone
;;  56=crafting-top, 57=crafting-side
;;  58=furnace-front, 59=furnace-side, 60=furnace-top
;;  61=lava

(def block-faces
  {STONE        [0 0 0 0 0 0]
   DIRT         [1 1 1 1 1 1]
   GRASS        [2 1 3 3 3 3]
   WATER        [4 4 4 4 4 4]
   SAND         [5 5 5 5 5 5]
   WOOD         [7 7 6 6 6 6]
   LEAVES       [8 8 8 8 8 8]
   BEDROCK      [9 9 9 9 9 9]
   TORCH        [10 10 10 10 10 10]
   GLASS        [11 11 11 11 11 11]
   GRAVEL       [12 12 12 12 12 12]
   SNOW-BLOCK   [13 13 13 13 13 13]
   ICE          [14 14 14 14 14 14]
   CLAY         [15 15 15 15 15 15]
   SANDSTONE    [17 17 16 16 16 16]
   RED-SAND     [18 18 18 18 18 18]
   PODZOL       [19 1 1 1 1 1]
   SNOWY-GRASS  [20 1 21 21 21 21]
   COAL-ORE     [22 22 22 22 22 22]
   IRON-ORE     [23 23 23 23 23 23]
   COPPER-ORE   [24 24 24 24 24 24]
   GOLD-ORE     [25 25 25 25 25 25]
   DIAMOND-ORE  [26 26 26 26 26 26]
   COBBLESTONE  [27 27 27 27 27 27]
   GRANITE      [28 28 28 28 28 28]
   ANDESITE     [29 29 29 29 29 29]
   OBSIDIAN     [30 30 30 30 30 30]
   BASALT       [31 31 31 31 31 31]
   BIRCH-WOOD   [33 33 32 32 32 32]
   BIRCH-LEAVES [34 34 34 34 34 34]
   PINE-WOOD    [36 36 35 35 35 35]
   PINE-LEAVES  [37 37 37 37 37 37]
   JUNGLE-WOOD  [39 39 38 38 38 38]
   JUNGLE-LEAVES [40 40 40 40 40 40]
   DARK-OAK-WOOD [42 42 41 41 41 41]
   DARK-OAK-LEAVES [43 43 43 43 43 43]
   CACTUS       [45 45 44 44 44 44]
   TALL-GRASS   [46 46 46 46 46 46]
   FERN         [47 47 47 47 47 47]
   DEAD-BUSH    [48 48 48 48 48 48]
   FLOWER-RED   [49 49 49 49 49 49]
   FLOWER-YELLOW [50 50 50 50 50 50]
   MUSHROOM-RED  [51 51 51 51 51 51]
   MUSHROOM-BROWN [52 52 52 52 52 52]
   GLOWSTONE    [53 53 53 53 53 53]
   CRYSTAL      [54 54 54 54 54 54]
   MOSSY-COBBLESTONE [55 55 55 55 55 55]
   CRAFTING-TABLE [56 56 57 57 57 57]
   FURNACE      [60 60 58 59 59 59]
   LAVA         [61 61 61 61 61 61]
   ;; Farming blocks (tex layers 115-120)
   FARMLAND     [115 1 1 1 1 1]    ;; tilled top, dirt sides
   FARMLAND-WET [115 1 1 1 1 1]    ;; same appearance (slightly darker in texture)
   WHEAT-1      [116 116 116 116 116 116]
   WHEAT-2      [117 117 117 117 117 117]
   WHEAT-3      [118 118 118 118 118 118]
   CARROT-1     [119 119 119 119 119 119]
   CARROT-2     [119 119 119 119 119 119]
   CARROT-3     [119 119 119 119 119 119]
   POTATO-1     [120 120 120 120 120 120]
   POTATO-2     [120 120 120 120 120 120]
   POTATO-3     [120 120 120 120 120 120]})

;; Face indices for the 6 directions
(def ^:const FACE-TOP 0)
(def ^:const FACE-BOTTOM 1)
(def ^:const FACE-NORTH 2)    ;; +Z
(def ^:const FACE-SOUTH 3)    ;; -Z
(def ^:const FACE-EAST 4)     ;; +X
(def ^:const FACE-WEST 5)     ;; -X

;; ================================================================
;; Coordinate math (needed by light functions below)
;; ================================================================

(defn block-index
  "Convert local block coords (0-15) to flat array index.
  Layout: x + z*16 + y*256 (Y-up, Y is vertical)"
  ^long [^long x ^long y ^long z]
  (+ x (* z CHUNK-SIZE) (* y CHUNK-SIZE CHUNK-SIZE)))

(defn index->local
  "Convert flat index back to [x y z]."
  [^long idx]
  [(rem idx CHUNK-SIZE)
   (quot idx (* CHUNK-SIZE CHUNK-SIZE))
   (rem (quot idx CHUNK-SIZE) CHUNK-SIZE)])

;; ================================================================
;; Light encoding: lower 4 bits = daylight, upper 4 bits = nightlight
;; ================================================================

(defn get-day-light
  "Get daylight value (0-15) at local chunk coords."
  ^long [chunk ^long x ^long y ^long z]
  (if (and (>= x 0) (< x CHUNK-SIZE)
           (>= y 0) (< y CHUNK-SIZE)
           (>= z 0) (< z CHUNK-SIZE))
    (bit-and (aget ^bytes (:light chunk) (block-index x y z)) 0x0F)
    0))

(defn get-night-light
  "Get nightlight value (0-15) at local chunk coords."
  ^long [chunk ^long x ^long y ^long z]
  (if (and (>= x 0) (< x CHUNK-SIZE)
           (>= y 0) (< y CHUNK-SIZE)
           (>= z 0) (< z CHUNK-SIZE))
    (bit-and (unsigned-bit-shift-right (aget ^bytes (:light chunk) (block-index x y z)) 4) 0x0F)
    0))

(defn set-day-light!
  "Set daylight value (0-15) at local chunk coords. Preserves night light."
  [^bytes light-arr ^long idx ^long val]
  (let [old (aget light-arr (int idx))
        new-val (bit-or (bit-and old (unchecked-byte 0xF0))
                        (bit-and val 0x0F))]
    (aset light-arr (int idx) (unchecked-byte new-val))))

(defn set-night-light!
  "Set nightlight value (0-15) at local chunk coords. Preserves day light."
  [^bytes light-arr ^long idx ^long val]
  (let [old (aget light-arr (int idx))
        new-val (bit-or (bit-and old 0x0F)
                        (bit-and (bit-shift-left val 4) (unchecked-byte 0xF0)))]
    (aset light-arr (int idx) (unchecked-byte new-val))))

;; ================================================================
;; Chunk operations
;; ================================================================

(defn create-chunk
  "Create an empty chunk at chunk coordinates [cx cy cz]."
  [cx cy cz]
  {:pos [cx cy cz]
   :blocks (int-array CHUNK-VOLUME)
   :light (byte-array CHUNK-VOLUME)
   :dirty? true
   :mesh nil})

(defn get-block
  "Get block ID at local coords within chunk."
  ^long [chunk ^long x ^long y ^long z]
  (if (and (>= x 0) (< x CHUNK-SIZE)
           (>= y 0) (< y CHUNK-SIZE)
           (>= z 0) (< z CHUNK-SIZE))
    (aget ^ints (:blocks chunk) (block-index x y z))
    AIR))

(defn set-block!
  "Set block ID at local coords. Marks chunk dirty."
  [chunk x y z block-id]
  (when (and (>= x 0) (< x CHUNK-SIZE)
             (>= y 0) (< y CHUNK-SIZE)
             (>= z 0) (< z CHUNK-SIZE))
    (aset ^ints (:blocks chunk) (block-index x y z) (int block-id))
    (assoc chunk :dirty? true)))

;; ================================================================
;; World operations
;; ================================================================

(defn world-to-chunk
  "Convert world block coords to [chunk-pos local-pos]."
  [^long wx ^long wy ^long wz]
  (let [cx (Math/floorDiv wx (long CHUNK-SIZE))
        cy (Math/floorDiv wy (long CHUNK-SIZE))
        cz (Math/floorDiv wz (long CHUNK-SIZE))
        lx (Math/floorMod wx (long CHUNK-SIZE))
        ly (Math/floorMod wy (long CHUNK-SIZE))
        lz (Math/floorMod wz (long CHUNK-SIZE))]
    [[cx cy cz] [lx ly lz]]))

(defn get-world-block
  "Get block at world coordinates from world map."
  [world ^long wx ^long wy ^long wz]
  (let [[cpos [lx ly lz]] (world-to-chunk wx wy wz)]
    (if-let [chunk (get world cpos)]
      (get-block chunk lx ly lz)
      AIR)))

(defn set-world-block!
  "Set block at world coordinates. Returns updated world."
  [world wx wy wz block-id]
  (let [[cpos [lx ly lz]] (world-to-chunk wx wy wz)]
    (if-let [chunk (get world cpos)]
      (let [new-chunk (assoc (set-block! chunk lx ly lz block-id) :blocks-dirty? true)]
        (assoc world cpos new-chunk))
      world)))

(defn block-at-pos
  "Get block at a floating-point world position (floors to block coords)."
  [world ^double x ^double y ^double z]
  (get-world-block world (long (Math/floor x)) (long (Math/floor y)) (long (Math/floor z))))

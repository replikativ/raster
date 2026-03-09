(ns valley.items
  "Item registry: block drops, tool types, mining speeds."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Item definitions
;; ================================================================

;; Items are keywords. Each has properties in the registry.
;; Block items have :block-id to convert back to a placeable block.
;; Tool items have :tool-type and :tier for mining effectiveness.

(def item-props
  {;; Block items (can be placed)
   :cobblestone    {:max-stack 64 :block-id w/COBBLESTONE :tex-layer 27}
   :stone          {:max-stack 64 :block-id w/STONE :tex-layer 0}
   :dirt           {:max-stack 64 :block-id w/DIRT :tex-layer 1}
   :sand           {:max-stack 64 :block-id w/SAND :tex-layer 5}
   :gravel         {:max-stack 64 :block-id w/GRAVEL :tex-layer 12}
   :wood           {:max-stack 64 :block-id w/WOOD :tex-layer 6}
   :birch-wood     {:max-stack 64 :block-id w/BIRCH-WOOD :tex-layer 32}
   :pine-wood      {:max-stack 64 :block-id w/PINE-WOOD :tex-layer 35}
   :jungle-wood    {:max-stack 64 :block-id w/JUNGLE-WOOD :tex-layer 38}
   :dark-oak-wood  {:max-stack 64 :block-id w/DARK-OAK-WOOD :tex-layer 41}
   :planks         {:max-stack 64 :block-id nil :tex-layer 65}  ;; no block yet
   :glass          {:max-stack 64 :block-id w/GLASS :tex-layer 11}
   :torch          {:max-stack 64 :block-id w/TORCH :tex-layer 10}
   :crafting-table {:max-stack 64 :block-id w/CRAFTING-TABLE :tex-layer 57}
   :furnace        {:max-stack 64 :block-id w/FURNACE :tex-layer 59}
   :snow-block     {:max-stack 64 :block-id w/SNOW-BLOCK :tex-layer 13}
   :clay           {:max-stack 64 :block-id w/CLAY :tex-layer 15}
   :sandstone      {:max-stack 64 :block-id w/SANDSTONE :tex-layer 16}
   :obsidian       {:max-stack 64 :block-id w/OBSIDIAN :tex-layer 30}
   :glowstone      {:max-stack 64 :block-id w/GLOWSTONE :tex-layer 53}

   ;; Ore/material items (not directly placeable)
   :coal           {:max-stack 64 :tex-layer 66}
   :iron-ore       {:max-stack 64 :block-id w/IRON-ORE :tex-layer 23}
   :copper-ore     {:max-stack 64 :block-id w/COPPER-ORE :tex-layer 24}
   :gold-ore       {:max-stack 64 :block-id w/GOLD-ORE :tex-layer 25}
   :diamond        {:max-stack 64 :tex-layer 67}
   :iron-ingot     {:max-stack 64 :tex-layer 68}
   :gold-ingot     {:max-stack 64 :tex-layer 69}
   :copper-ingot   {:max-stack 64 :tex-layer 70}
   :stick          {:max-stack 64 :tex-layer 71}

   ;; Tools — pickaxes
   :wood-pickaxe    {:max-stack 1 :tool-type :pickaxe :tier :wood    :durability 60  :tex-layer 72}
   :stone-pickaxe   {:max-stack 1 :tool-type :pickaxe :tier :stone   :durability 132 :tex-layer 73}
   :iron-pickaxe    {:max-stack 1 :tool-type :pickaxe :tier :iron    :durability 250 :tex-layer 74}
   :diamond-pickaxe {:max-stack 1 :tool-type :pickaxe :tier :diamond :durability 1561 :tex-layer 75}
   ;; Tools — axes
   :wood-axe        {:max-stack 1 :tool-type :axe :tier :wood    :durability 60  :tex-layer 76}
   :stone-axe       {:max-stack 1 :tool-type :axe :tier :stone   :durability 132 :tex-layer 77}
   :iron-axe        {:max-stack 1 :tool-type :axe :tier :iron    :durability 250 :tex-layer 78}
   :diamond-axe     {:max-stack 1 :tool-type :axe :tier :diamond :durability 1561 :tex-layer 79}
   ;; Tools — shovels
   :wood-shovel     {:max-stack 1 :tool-type :shovel :tier :wood    :durability 60  :tex-layer 80}
   :stone-shovel    {:max-stack 1 :tool-type :shovel :tier :stone   :durability 132 :tex-layer 81}
   :iron-shovel     {:max-stack 1 :tool-type :shovel :tier :iron    :durability 250 :tex-layer 82}
   :diamond-shovel  {:max-stack 1 :tool-type :shovel :tier :diamond :durability 1561 :tex-layer 83}
   ;; Tools — swords
   :wood-sword      {:max-stack 1 :tool-type :sword :tier :wood    :durability 60  :tex-layer 84}
   :stone-sword     {:max-stack 1 :tool-type :sword :tier :stone   :durability 132 :tex-layer 85}
   :iron-sword      {:max-stack 1 :tool-type :sword :tier :iron    :durability 250 :tex-layer 86}
   :diamond-sword   {:max-stack 1 :tool-type :sword :tier :diamond :durability 1561 :tex-layer 87}
   ;; Hoes
   :wood-hoe        {:max-stack 1 :tool-type :hoe :tier :wood    :durability 60  :tex-layer 84}
   :stone-hoe       {:max-stack 1 :tool-type :hoe :tier :stone   :durability 132 :tex-layer 85}
   :iron-hoe        {:max-stack 1 :tool-type :hoe :tier :iron    :durability 250 :tex-layer 86}
   :diamond-hoe     {:max-stack 1 :tool-type :hoe :tier :diamond :durability 1561 :tex-layer 87}
   ;; Food
   :raw-beef        {:max-stack 64 :tex-layer 104}
   :cooked-beef     {:max-stack 64 :tex-layer 105}
   :bread           {:max-stack 64 :tex-layer 106}
   :apple           {:max-stack 64 :tex-layer 107}
   :carrot          {:max-stack 64 :tex-layer 108}
   :potato          {:max-stack 64 :tex-layer 109}
   :cooked-potato   {:max-stack 64 :tex-layer 110}
   ;; Seeds
   :wheat-seeds     {:max-stack 64 :tex-layer 111}})

;; ================================================================
;; Block → drop mapping
;; ================================================================

(def block->drop
  "Map block ID → item keyword dropped when broken."
  {w/STONE       :cobblestone
   w/DIRT        :dirt
   w/GRASS       :dirt
   w/SAND        :sand
   w/GRAVEL      :gravel
   w/WOOD        :wood
   w/BIRCH-WOOD  :birch-wood
   w/PINE-WOOD   :pine-wood
   w/JUNGLE-WOOD :jungle-wood
   w/DARK-OAK-WOOD :dark-oak-wood
   w/COBBLESTONE :cobblestone
   w/COAL-ORE    :coal
   w/IRON-ORE    :iron-ore
   w/COPPER-ORE  :copper-ore
   w/GOLD-ORE    :gold-ore
   w/DIAMOND-ORE :diamond
   w/GLASS       :glass
   w/TORCH       :torch
   w/CRAFTING-TABLE :crafting-table
   w/FURNACE     :furnace
   w/SANDSTONE   :sandstone
   w/OBSIDIAN    :obsidian
   w/GLOWSTONE   :glowstone
   w/SNOW-BLOCK  :snow-block
   w/CLAY        :clay
   w/GRANITE     :stone
   w/ANDESITE    :stone
   w/BASALT      :stone})

;; ================================================================
;; Block categories for tool effectiveness
;; ================================================================

(def block-category
  "Map block ID → category for tool matching."
  {w/STONE :stone  w/COBBLESTONE :stone  w/GRANITE :stone  w/ANDESITE :stone
   w/BASALT :stone  w/SANDSTONE :stone  w/OBSIDIAN :stone
   w/COAL-ORE :stone  w/IRON-ORE :stone  w/COPPER-ORE :stone
   w/GOLD-ORE :stone  w/DIAMOND-ORE :stone  w/GLOWSTONE :stone
   w/WOOD :wood  w/BIRCH-WOOD :wood  w/PINE-WOOD :wood
   w/JUNGLE-WOOD :wood  w/DARK-OAK-WOOD :wood
   w/CRAFTING-TABLE :wood  w/FURNACE :stone
   w/DIRT :dirt  w/GRASS :dirt  w/SAND :dirt  w/GRAVEL :dirt
   w/CLAY :dirt  w/SNOW-BLOCK :dirt})

;; ================================================================
;; Mining speed
;; ================================================================

;; Tool effectiveness: which tool-type speeds up which block-category
(def tool-effective
  {:pickaxe #{:stone}
   :axe #{:wood}
   :shovel #{:dirt}
   :sword #{}})

;; Tier speed multipliers
(def tier-speed
  {:wood 2.0  :stone 4.0  :iron 6.0  :diamond 8.0})

;; Base mining times (seconds) per block category
(def base-mine-time
  {:stone 1.5  :wood 2.0  :dirt 0.5  nil 1.0})

(defn mining-time
  "Compute mining time in seconds for a block with a given tool item (or nil for hand)."
  ^double [^long block-id tool-item]
  (let [cat (get block-category block-id)
        base (double (get base-mine-time cat 1.0))]
    (if tool-item
      (let [props (get item-props tool-item)
            tool-type (:tool-type props)
            tier (:tier props)
            effective? (and tool-type (contains? (get tool-effective tool-type) cat))]
        (if effective?
          (/ base (double (get tier-speed tier 1.0)))
          base))
      ;; Hand mining: much slower for stone
      (if (= cat :stone) (* base 5.0) base))))

(defn tool?
  "Check if an item is a tool."
  [item-id]
  (boolean (:tool-type (get item-props item-id))))

(def attack-damages
  "Attack damage per tier for swords and other tools."
  {:wood 4  :stone 5  :iron 6  :diamond 7})

(defn attack-damage
  "Get attack damage for a held item (or 1 for bare hand)."
  ^long [item-id]
  (if item-id
    (let [props (get item-props item-id)
          tier (:tier props)
          tool-type (:tool-type props)]
      (if (= tool-type :sword)
        (long (get attack-damages tier 4))
        (if tier
          (long (max 2 (dec (long (get attack-damages tier 2)))))
          1)))
    1))

(defn placeable?
  "Check if an item can be placed as a block."
  [item-id]
  (boolean (:block-id (get item-props item-id))))

(defn item->block
  "Get the block ID for a placeable item, or nil."
  [item-id]
  (:block-id (get item-props item-id)))

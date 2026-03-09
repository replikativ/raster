(ns valley.crafting
  "Crafting recipes: shaped 3x3 grid with offset + mirror matching.
  Smelting recipes for furnace."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.inventory :as inv]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Shaped recipes (3x3 grid)
;; ================================================================

;; Recipe format: {:pattern [[row0] [row1] [row2]]
;;                 :result {:item :keyword :count N}}
;; nil in pattern = empty slot, keyword = item required

(def recipes
  [;; Wood → Planks (4)
   {:pattern [[:wood]]
    :result {:item :planks :count 4}}
   {:pattern [[:birch-wood]]
    :result {:item :planks :count 4}}
   {:pattern [[:pine-wood]]
    :result {:item :planks :count 4}}
   {:pattern [[:jungle-wood]]
    :result {:item :planks :count 4}}
   {:pattern [[:dark-oak-wood]]
    :result {:item :planks :count 4}}

   ;; Planks → Sticks (4)
   {:pattern [[:planks]
              [:planks]]
    :result {:item :stick :count 4}}

   ;; Crafting table
   {:pattern [[:planks :planks]
              [:planks :planks]]
    :result {:item :crafting-table :count 1}}

   ;; Furnace
   {:pattern [[:cobblestone :cobblestone :cobblestone]
              [:cobblestone nil          :cobblestone]
              [:cobblestone :cobblestone :cobblestone]]
    :result {:item :furnace :count 1}}

   ;; Torch (4)
   {:pattern [[:coal]
              [:stick]]
    :result {:item :torch :count 4}}

   ;; --- Pickaxes ---
   {:pattern [[:planks :planks :planks]
              [nil     :stick  nil]
              [nil     :stick  nil]]
    :result {:item :wood-pickaxe :count 1}}
   {:pattern [[:cobblestone :cobblestone :cobblestone]
              [nil          :stick       nil]
              [nil          :stick       nil]]
    :result {:item :stone-pickaxe :count 1}}
   {:pattern [[:iron-ingot :iron-ingot :iron-ingot]
              [nil         :stick      nil]
              [nil         :stick      nil]]
    :result {:item :iron-pickaxe :count 1}}
   {:pattern [[:diamond :diamond :diamond]
              [nil      :stick   nil]
              [nil      :stick   nil]]
    :result {:item :diamond-pickaxe :count 1}}

   ;; --- Axes ---
   {:pattern [[:planks :planks]
              [:planks :stick]
              [nil     :stick]]
    :result {:item :wood-axe :count 1}}
   {:pattern [[:cobblestone :cobblestone]
              [:cobblestone :stick]
              [nil          :stick]]
    :result {:item :stone-axe :count 1}}
   {:pattern [[:iron-ingot :iron-ingot]
              [:iron-ingot :stick]
              [nil         :stick]]
    :result {:item :iron-axe :count 1}}
   {:pattern [[:diamond :diamond]
              [:diamond :stick]
              [nil      :stick]]
    :result {:item :diamond-axe :count 1}}

   ;; --- Shovels ---
   {:pattern [[:planks]
              [:stick]
              [:stick]]
    :result {:item :wood-shovel :count 1}}
   {:pattern [[:cobblestone]
              [:stick]
              [:stick]]
    :result {:item :stone-shovel :count 1}}
   {:pattern [[:iron-ingot]
              [:stick]
              [:stick]]
    :result {:item :iron-shovel :count 1}}
   {:pattern [[:diamond]
              [:stick]
              [:stick]]
    :result {:item :diamond-shovel :count 1}}

   ;; --- Swords ---
   {:pattern [[:planks]
              [:planks]
              [:stick]]
    :result {:item :wood-sword :count 1}}
   {:pattern [[:cobblestone]
              [:cobblestone]
              [:stick]]
    :result {:item :stone-sword :count 1}}
   {:pattern [[:iron-ingot]
              [:iron-ingot]
              [:stick]]
    :result {:item :iron-sword :count 1}}
   {:pattern [[:diamond]
              [:diamond]
              [:stick]]
    :result {:item :diamond-sword :count 1}}

   ;; --- Hoes ---
   {:pattern [[:planks :planks]
              [nil     :stick]
              [nil     :stick]]
    :result {:item :wood-hoe :count 1}}
   {:pattern [[:cobblestone :cobblestone]
              [nil          :stick]
              [nil          :stick]]
    :result {:item :stone-hoe :count 1}}
   {:pattern [[:iron-ingot :iron-ingot]
              [nil         :stick]
              [nil         :stick]]
    :result {:item :iron-hoe :count 1}}
   {:pattern [[:diamond :diamond]
              [nil      :stick]
              [nil      :stick]]
    :result {:item :diamond-hoe :count 1}}

   ;; --- Bread ---
   {:pattern [[:wheat-seeds :wheat-seeds :wheat-seeds]]
    :result {:item :bread :count 1}}])

;; ================================================================
;; Pattern matching
;; ================================================================

(defn- normalize-pattern
  "Remove empty leading/trailing rows and columns from a pattern.
  Returns normalized pattern (list of rows, each a vector of items/nil)."
  [pattern]
  (let [rows (vec (remove (fn [row] (every? nil? row)) pattern))]
    (if (empty? rows)
      [[]]
      (let [min-col (reduce min (map (fn [row]
                                       (or (first (keep-indexed (fn [i v] (when v i)) row))
                                           (count row)))
                                     rows))
            max-col (reduce max (map (fn [row]
                                       (or (last (keep-indexed (fn [i v] (when v i)) row))
                                           -1))
                                     rows))]
        (if (> min-col max-col)
          [[]]
          (mapv (fn [row]
                  (vec (for [c (range min-col (inc max-col))]
                    (get row c))))
                rows))))))

(defn- mirror-pattern
  "Mirror a pattern horizontally."
  [pattern]
  (mapv (fn [row] (vec (reverse row))) pattern))

(defn- grid->pattern
  "Convert a 3x3 grid (vec of 9 slots) to a pattern (3 rows of 3).
  Slots are nil or {:item ... :count ...}."
  [grid]
  (let [rows (mapv (fn [r]
                     (mapv (fn [slot] (when slot (:item slot)))
                           (subvec grid (* r 3) (+ (* r 3) 3))))
                   (range 3))]
    (normalize-pattern rows)))

(defn match-recipe
  "Try to match a 3x3 crafting grid against all recipes.
  Returns {:item :count} of result, or nil if no match.
  Grid is a vec of 9 slots (nil or {:item :count})."
  [grid]
  (let [input (grid->pattern grid)]
    (first
      (for [recipe recipes
            :let [pattern (normalize-pattern (:pattern recipe))
                  mirrored (normalize-pattern (mirror-pattern (:pattern recipe)))]
            :when (or (= input pattern) (= input mirrored))]
        (:result recipe)))))

;; ================================================================
;; Craft execution
;; ================================================================

(defn- count-grid-ingredients
  "Count how many of each item are needed in a recipe pattern."
  [pattern]
  (reduce (fn [m row]
            (reduce (fn [m item]
                      (if item
                        (update m item (fnil inc 0))
                        m))
                    m row))
          {} pattern))

(defn craft!
  "Attempt to craft from inventory using a 3x3 grid.
  Returns [updated-inv result-item result-count] or nil if no match.
  Grid is a vec of 9 slot values (used for matching only).
  Ingredients are consumed from inventory."
  [inv grid]
  (when-let [result (match-recipe grid)]
    (let [input (grid->pattern grid)
          ;; Count ingredients needed
          ingredients (count-grid-ingredients input)
          ;; Check inventory has all ingredients
          has-all? (every? (fn [[item cnt]]
                             (inv/has-item? inv item cnt))
                           ingredients)]
      (when has-all?
        (let [inv (reduce (fn [inv [item cnt]]
                            (inv/remove-item inv item cnt))
                          inv ingredients)
              inv (inv/add-item inv (:item result) (:count result))]
          [inv (:item result) (:count result)])))))

;; ================================================================
;; Smelting recipes
;; ================================================================

(def smelting-recipes
  {:iron-ore    :iron-ingot
   :copper-ore  :copper-ingot
   :gold-ore    :gold-ingot
   :cobblestone :stone
   :sand        :glass
   :raw-beef    :cooked-beef
   :potato      :cooked-potato})

(def fuel-values
  "Fuel items → number of items they can smelt."
  {:coal 8
   :wood 1
   :birch-wood 1
   :pine-wood 1
   :jungle-wood 1
   :dark-oak-wood 1
   :planks 1
   :stick 0})  ;; sticks don't work as fuel alone

(defn smelt-result
  "Get the smelting output item for an input item, or nil."
  [input-item]
  (get smelting-recipes input-item))

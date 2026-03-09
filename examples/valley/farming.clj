(ns valley.farming
  "Farming: farmland blocks, crop growth, planting, harvesting."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w]))

(set! *warn-on-reflection* true)

;; ================================================================
;; New block IDs (51-61)
;; ================================================================

(def ^:const FARMLAND 51)
(def ^:const FARMLAND-WET 52)
(def ^:const WHEAT-1 53)
(def ^:const WHEAT-2 54)
(def ^:const WHEAT-3 55)  ;; mature
(def ^:const CARROT-1 56)
(def ^:const CARROT-2 57)
(def ^:const CARROT-3 58)  ;; mature
(def ^:const POTATO-1 59)
(def ^:const POTATO-2 60)
(def ^:const POTATO-3 61)  ;; mature

;; ================================================================
;; Crop definitions
;; ================================================================

(def crop-stages
  "Crop block IDs by stage [1 2 3(mature)]."
  {:wheat  [WHEAT-1  WHEAT-2  WHEAT-3]
   :carrot [CARROT-1 CARROT-2 CARROT-3]
   :potato [POTATO-1 POTATO-2 POTATO-3]})

(def seed->crop
  "Seed item → crop type."
  {:wheat-seeds :wheat
   :carrot      :carrot
   :potato      :potato})

(def crop-drops
  "Mature crop → drops."
  {:wheat  [{:item :bread :count 1} {:item :wheat-seeds :count 2}]
   :carrot [{:item :carrot :count 3}]
   :potato [{:item :potato :count 3}]})

(defn crop-block?
  "Check if block is any crop stage."
  [^long block-id]
  (and (>= block-id WHEAT-1) (<= block-id POTATO-3)))

(defn crop-type
  "Get crop type for a crop block, or nil."
  [^long block-id]
  (cond
    (and (>= block-id WHEAT-1)  (<= block-id WHEAT-3))  :wheat
    (and (>= block-id CARROT-1) (<= block-id CARROT-3)) :carrot
    (and (>= block-id POTATO-1) (<= block-id POTATO-3)) :potato
    :else nil))

(defn crop-stage
  "Get 0-based stage (0,1,2) of a crop block."
  [^long block-id]
  (let [ct (crop-type block-id)]
    (when ct
      (let [stages (get crop-stages ct)
            base (long (first stages))]
        (- block-id base)))))

(defn mature?
  "Check if a crop block is at final (mature) stage."
  [^long block-id]
  (= (long (crop-stage block-id)) 2))

;; ================================================================
;; Farmland creation
;; ================================================================

(defn use-hoe
  "Use hoe on a block. Returns new block ID or nil."
  [^long block-id]
  (when (or (= block-id w/GRASS) (= block-id w/DIRT))
    FARMLAND))

;; ================================================================
;; Planting
;; ================================================================

(defn plant-seed
  "Plant a seed on farmland. Returns stage-1 block ID or nil."
  [seed-item ^long block-below]
  (when (or (= block-below FARMLAND) (= block-below FARMLAND-WET))
    (let [ct (get seed->crop seed-item)]
      (when ct
        (first (get crop-stages ct))))))

;; ================================================================
;; Crop growth (random tick)
;; ================================================================

(defn- water-nearby?
  "Check if water is within 4 blocks horizontally of position."
  [world bx by bz]
  (let [bx (long bx) by (long by) bz (long bz) r 4]
    (loop [dx (- r)]
      (if (> dx r) false
        (or (loop [dz (- r)]
              (if (> dz r) false
                (or (= (w/get-world-block world (+ bx dx) by (+ bz dz)) w/WATER)
                    (recur (inc dz)))))
            (recur (inc dx)))))))

(defn tick-crop
  "Try to advance a crop block by one stage. Returns new block ID or nil.
  Random tick chance: ~1/4096 per block per second."
  [world bx by bz block-id]
  (let [bx (long bx) by (long by) bz (long bz) block-id (long block-id)]
  (when (crop-block? block-id)
    (let [stage (long (crop-stage block-id))]
      (when (< stage 2)
        ;; Check if farmland below is wet (doubles growth rate)
        (let [below (w/get-world-block world bx (dec by) bz)
              wet? (or (= below FARMLAND-WET)
                       (water-nearby? world bx (dec by) bz))
              chance (if wet? 2048 4096)]
          (when (zero? (rand-int chance))
            ;; Advance to next stage
            (let [ct (crop-type block-id)
                  stages (get crop-stages ct)]
              (nth stages (inc stage))))))))))

(defn harvest-crop
  "Harvest a mature crop. Returns drop items or nil."
  [^long block-id]
  (when (and (crop-block? block-id) (mature? block-id))
    (let [ct (crop-type block-id)]
      (get crop-drops ct))))

;; ================================================================
;; Farmland hydration
;; ================================================================

(defn tick-farmland
  "Check if farmland should be wet or dry. Returns new block ID or nil (no change)."
  [world bx by bz block-id]
  (let [bx (long bx) by (long by) bz (long bz) block-id (long block-id)]
  (when (or (= block-id FARMLAND) (= block-id FARMLAND-WET))
    (let [wet? (water-nearby? world bx by bz)]
      (cond
        (and wet? (= block-id FARMLAND)) FARMLAND-WET
        (and (not wet?) (= block-id FARMLAND-WET)) FARMLAND
        :else nil)))))

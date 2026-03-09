(ns valley.hunger
  "Hunger system: exhaustion → saturation drain → hunger drain.
  Health regen at high hunger, starvation at zero hunger."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Constants
;; ================================================================

(def ^:const MAX-HUNGER 20)
(def ^:const MAX-SATURATION 5.0)
(def ^:const EXHAUSTION-THRESHOLD 4.0)

;; Exhaustion costs per action
(def ^:const SPRINT-EXHAUSTION 0.1)   ;; per meter
(def ^:const JUMP-EXHAUSTION 0.05)
(def ^:const ATTACK-EXHAUSTION 0.1)
(def ^:const MINE-EXHAUSTION 0.005)

;; Regen/starvation
(def ^:const REGEN-THRESHOLD 18)      ;; hunger >= 18 to regen
(def ^:const REGEN-RATE 0.5)          ;; HP per second
(def ^:const STARVATION-RATE 0.25)    ;; HP per second at hunger=0

;; ================================================================
;; Hunger state
;; ================================================================

(defn create-hunger
  "Create initial hunger state."
  []
  {:hunger MAX-HUNGER
   :saturation MAX-SATURATION
   :exhaustion 0.0
   :regen-timer 0.0
   :starve-timer 0.0})

;; ================================================================
;; Exhaustion
;; ================================================================

(defn exhaust
  "Add exhaustion from an action. Returns updated hunger state."
  [hunger-state amount]
  (update hunger-state :exhaustion + (double amount)))

;; ================================================================
;; Tick hunger
;; ================================================================

(defn tick-hunger
  "Process hunger per frame. Returns {:hunger-state :heal :damage}."
  [hunger-state dt]
  (let [dt (double dt)
        exhaustion (double (:exhaustion hunger-state))
        saturation (double (:saturation hunger-state))
        hunger (long (:hunger hunger-state))
        regen-timer (double (:regen-timer hunger-state 0.0))
        starve-timer (double (:starve-timer hunger-state 0.0))
        ;; Process exhaustion → drain saturation/hunger
        [exhaustion saturation hunger]
        (if (>= exhaustion EXHAUSTION-THRESHOLD)
          (let [exhaustion (- exhaustion EXHAUSTION-THRESHOLD)]
            (if (pos? saturation)
              [exhaustion (max 0.0 (- saturation 1.0)) hunger]
              [exhaustion saturation (max 0 (dec hunger))]))
          [exhaustion saturation hunger])

        ;; Health regen when hunger >= 18
        regen-timer (+ regen-timer dt)
        heal (if (and (>= hunger REGEN-THRESHOLD) (>= regen-timer (/ 1.0 REGEN-RATE)))
               1
               0)
        regen-timer (if (pos? heal) 0.0 regen-timer)

        ;; Starvation when hunger = 0
        starve-timer (+ starve-timer dt)
        damage (if (and (zero? hunger) (>= starve-timer (/ 1.0 STARVATION-RATE)))
                 1
                 0)
        starve-timer (if (pos? damage) 0.0 starve-timer)]

    {:hunger-state {:hunger hunger
                    :saturation saturation
                    :exhaustion exhaustion
                    :regen-timer regen-timer
                    :starve-timer starve-timer}
     :heal heal
     :damage damage}))

;; ================================================================
;; Food
;; ================================================================

(def food-values
  "Food item → {:hunger :saturation}."
  {:cooked-beef {:hunger 8 :saturation 3.0}
   :raw-beef    {:hunger 3 :saturation 0.5}
   :bread       {:hunger 5 :saturation 2.0}
   :apple       {:hunger 4 :saturation 1.0}
   :carrot      {:hunger 3 :saturation 1.0}
   :potato      {:hunger 1 :saturation 0.5}
   :cooked-potato {:hunger 5 :saturation 2.0}})

(defn food?
  "Check if item is food."
  [item-id]
  (boolean (get food-values item-id)))

(defn eat-food
  "Eat food item. Returns updated hunger state or nil if can't eat."
  [hunger-state item-id]
  (let [food (get food-values item-id)]
    (when (and food (< (long (:hunger hunger-state)) MAX-HUNGER))
      (let [hunger (min MAX-HUNGER (+ (long (:hunger hunger-state)) (long (:hunger food))))
            saturation (min MAX-SATURATION (+ (double (:saturation hunger-state))
                                              (double (:saturation food))))]
        (assoc hunger-state
          :hunger hunger
          :saturation saturation)))))

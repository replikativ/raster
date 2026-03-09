(ns valley.entity
  "Entity system: player and mob state, damage, healing, death."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Entity creation
;; ================================================================

(defn create-entity
  "Create a new entity with the given type and position."
  [entity-type pos]
  {:id (random-uuid)
   :type entity-type
   :pos pos
   :vel [0.0 0.0 0.0]
   :yaw 0.0
   :health 20
   :max-health 20
   :state :idle
   :on-ground? false
   :data {:fall-distance 0.0
          :hurt-timer 0.0
          :air 10
          :respawn-pos pos}})

(defn create-player
  "Create a player entity at the given spawn position."
  [pos]
  (assoc (create-entity :player pos)
    :max-health 20
    :health 20))

;; ================================================================
;; Damage & healing
;; ================================================================

(defn damage-entity
  "Apply damage to an entity. Returns updated entity."
  [entity ^long amount]
  (let [health (long (:health entity))
        new-health (max 0 (- health amount))]
    (-> entity
        (assoc :health new-health)
        (assoc-in [:data :hurt-timer] 0.4))))

(defn heal-entity
  "Heal an entity by amount, capped at max-health."
  [entity ^long amount]
  (let [health (long (:health entity))
        max-hp (long (:max-health entity))
        new-health (min max-hp (+ health amount))]
    (assoc entity :health new-health)))

(defn dead?
  "Check if entity is dead."
  [entity]
  (<= (long (:health entity)) 0))

;; ================================================================
;; Tick helpers
;; ================================================================

(defn tick-hurt-timer
  "Decrement the hurt timer (invincibility frames after damage)."
  [entity ^double dt]
  (let [timer (double (get-in entity [:data :hurt-timer] 0.0))
        new-timer (max 0.0 (- timer dt))]
    (assoc-in entity [:data :hurt-timer] new-timer)))

(defn can-hurt?
  "Check if entity can receive damage (hurt timer expired)."
  [entity]
  (<= (double (get-in entity [:data :hurt-timer] 0.0)) 0.0))

;; ================================================================
;; Fall damage
;; ================================================================

(def ^:const FALL-DAMAGE-THRESHOLD 3.0)  ;; blocks before damage starts

(defn update-fall-distance
  "Track cumulative fall distance. Call each physics frame."
  [entity ^double vy ^double dy on-ground?]
  (let [fall-dist (double (get-in entity [:data :fall-distance] 0.0))]
    (if on-ground?
      ;; Landing: compute damage, reset
      (let [damage (if (> fall-dist FALL-DAMAGE-THRESHOLD)
                     (long (Math/floor (- fall-dist FALL-DAMAGE-THRESHOLD)))
                     0)
            entity (assoc-in entity [:data :fall-distance] 0.0)]
        (if (pos? damage)
          (damage-entity entity damage)
          entity))
      ;; Falling: accumulate distance (only when going down)
      (if (neg? vy)
        (assoc-in entity [:data :fall-distance] (+ fall-dist (Math/abs dy)))
        entity))))

;; ================================================================
;; Environmental damage
;; ================================================================

(def ^:const LAVA-DAMAGE-PER-SEC 4)
(def ^:const DROWN-DAMAGE-PER-SEC 2)
(def ^:const AIR-MAX 10)
(def ^:const AIR-DRAIN-RATE 1.0)  ;; air units per second

(defn tick-environmental-damage
  "Apply lava and drowning damage based on blocks at player feet/head.
  `feet-block` and `head-block` are block IDs at player's feet and head positions."
  [entity ^long feet-block ^long head-block ^double dt]
  (let [in-lava? (or (= feet-block w/LAVA) (= head-block w/LAVA))
        in-water? (= head-block w/WATER)]
    (cond-> entity
      ;; Lava damage
      (and in-lava? (can-hurt? entity))
      (damage-entity (long (Math/ceil (* LAVA-DAMAGE-PER-SEC dt))))

      ;; Drowning
      in-water?
      (-> (update-in [:data :air]
            (fn [air]
              (let [air (double (or air AIR-MAX))]
                (max 0.0 (- air (* AIR-DRAIN-RATE dt))))))
          (as-> e
            (if (and (<= (double (get-in e [:data :air] AIR-MAX)) 0.0)
                     (can-hurt? e))
              (damage-entity e (long (Math/ceil (* DROWN-DAMAGE-PER-SEC dt))))
              e)))

      ;; Restore air when not in water
      (not in-water?)
      (assoc-in [:data :air] (double AIR-MAX)))))

;; ================================================================
;; Death & respawn
;; ================================================================

(defn respawn
  "Reset player to spawn position with full health."
  [entity]
  (let [spawn-pos (get-in entity [:data :respawn-pos] [0.0 60.0 0.0])]
    (-> entity
        (assoc :health (:max-health entity))
        (assoc :pos spawn-pos)
        (assoc :vel [0.0 0.0 0.0])
        (assoc-in [:data :fall-distance] 0.0)
        (assoc-in [:data :air] (double AIR-MAX))
        (assoc-in [:data :hurt-timer] 1.0))))

;; ================================================================
;; Tick entity (main per-frame update)
;; ================================================================

(defn tick-entity
  "Per-frame update for an entity. Returns updated entity."
  [entity ^double dt]
  (-> entity
      (tick-hurt-timer dt)))

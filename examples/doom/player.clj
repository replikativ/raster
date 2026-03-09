(ns doom.player
  "Player state: health, armor, ammo, weapons, pickups, weapon rendering."
  (:refer-clojure :exclude [defn])
  (:require
   [raster.core :refer [defn]]
   [doom.wad :as wad]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Player state
;; ================================================================

(defn create-player-state []
  {:health 100
   :armor 0
   :max-health 200
   :max-armor 200
   :ammo {:bullets 50 :shells 0 :rockets 0 :cells 0}
   :max-ammo {:bullets 200 :shells 50 :rockets 50 :cells 300}
   :weapons #{:fist :pistol}
   :current-weapon :pistol
   :weapon-state :ready ;; :ready :firing :lowering :raising
   :weapon-frame 0
   :weapon-tics 0
   :keys #{}
   :kills 0
   :items 0
   :secrets 0})

;; ================================================================
;; Weapon definitions
;; ================================================================

(def weapons
  {:fist {:ammo-type nil :ammo-per-shot 0 :damage [2 20] :range 64
          :sprite "PUNG" :fire-tics 8 :hitscan? false :melee? true}
   :pistol {:ammo-type :bullets :ammo-per-shot 1 :damage [5 15] :range 8192
            :sprite "PISG" :fire-tics 6 :hitscan? true :melee? false}
   :shotgun {:ammo-type :shells :ammo-per-shot 1 :damage [5 15] :pellets 7 :range 2048
             :sprite "SHTG" :fire-tics 10 :hitscan? true :melee? false}
   :chaingun {:ammo-type :bullets :ammo-per-shot 1 :damage [5 15] :range 8192
              :sprite "CHGG" :fire-tics 4 :hitscan? true :melee? false}
   :rocket-launcher {:ammo-type :rockets :ammo-per-shot 1 :damage [20 160] :range 8192
                     :sprite "MISG" :fire-tics 12 :hitscan? false :melee? false
                     :projectile? true :projectile-speed 20}
   :chainsaw {:ammo-type nil :ammo-per-shot 0 :damage [2 20] :range 64
              :sprite "SAWG" :fire-tics 4 :hitscan? false :melee? true}})

;; ================================================================
;; Pickup definitions
;; ================================================================

(def pickups
  {:health-bonus {:effect :health :amount 1}
   :stimpack {:effect :health :amount 10}
   :medikit {:effect :health :amount 25}
   :armor-bonus {:effect :armor :amount 1}
   :green-armor {:effect :armor :amount 100}
   :blue-armor {:effect :armor :amount 200}
   :clip {:effect :ammo :ammo-type :bullets :amount 10}
   :ammo-box {:effect :ammo :ammo-type :bullets :amount 50}
   :shells {:effect :ammo :ammo-type :shells :amount 4}
   :shell-box {:effect :ammo :ammo-type :shells :amount 20}
   :rocket {:effect :ammo :ammo-type :rockets :amount 1}
   :rocket-box {:effect :ammo :ammo-type :rockets :amount 5}
   :cell {:effect :ammo :ammo-type :cells :amount 20}
   :cell-pack {:effect :ammo :ammo-type :cells :amount 100}
   :backpack {:effect :backpack}
   :shotgun {:effect :weapon :weapon :shotgun :ammo-type :shells :amount 8}
   :chaingun {:effect :weapon :weapon :chaingun :ammo-type :bullets :amount 20}
   :rocket-launcher {:effect :weapon :weapon :rocket-launcher :ammo-type :rockets :amount 2}
   :bfg {:effect :weapon :weapon :bfg :ammo-type :cells :amount 40}
   :chainsaw {:effect :weapon :weapon :chainsaw}
   :blue-key {:effect :key :key :blue}
   :red-key {:effect :key :key :red}
   :yellow-key {:effect :key :key :yellow}
   :blue-skull {:effect :key :key :blue}
   :red-skull {:effect :key :key :red}
   :yellow-skull {:effect :key :key :yellow}})

;; ================================================================
;; Pickup logic
;; ================================================================

(defn can-pickup?
  "Check if player can pick up this item (not full, etc.)."
  [player pickup-info]
  (case (:effect pickup-info)
    :health (< (:health player) (:max-health player))
    :armor (< (:armor player) (:max-armor player))
    :ammo (let [at (:ammo-type pickup-info)
                cur (get-in player [:ammo at] 0)
                mx (get-in player [:max-ammo at] 999)]
            (< cur mx))
    :weapon true ;; always pick up weapons (get ammo even if owned)
    :key (not (contains? (:keys player) (:key pickup-info)))
    :backpack true
    false))

(defn apply-pickup
  "Apply pickup effect to player state."
  [player pickup-info]
  (case (:effect pickup-info)
    :health (update player :health
                    #(min (:max-health player) (+ % (:amount pickup-info))))
    :armor (update player :armor
                   #(min (:max-armor player) (+ % (:amount pickup-info))))
    :ammo (update-in player [:ammo (:ammo-type pickup-info)]
                     #(min (get-in player [:max-ammo (:ammo-type pickup-info)] 999)
                           (+ (or % 0) (:amount pickup-info))))
    :weapon (-> player
                (update :weapons conj (:weapon pickup-info))
                (cond->
                  (:ammo-type pickup-info)
                  (update-in [:ammo (:ammo-type pickup-info)]
                             #(min (get-in player [:max-ammo (:ammo-type pickup-info)] 999)
                                   (+ (or % 0) (or (:amount pickup-info) 0))))))
    :key (update player :keys conj (:key pickup-info))
    :backpack (-> player
                  (update :max-ammo (fn [m] (into {} (map (fn [[k v]] [k (* v 2)]) m)))))
    player))

;; ================================================================
;; Weapon state machine
;; ================================================================

(defn update-weapon [player input]
  (let [ws (:weapon-state player)
        wt (:weapon-tics player)
        wf (:weapon-frame player)]
    (case ws
      :ready
      (cond
        ;; Fire button
        (and (:fire? input)
             (let [w (get weapons (:current-weapon player))
                   at (:ammo-type w)]
               (or (nil? at) (pos? (get-in player [:ammo at] 0)))))
        (let [w (get weapons (:current-weapon player))
              at (:ammo-type w)]
          (-> player
              (assoc :weapon-state :firing :weapon-frame 0
                     :weapon-tics (:fire-tics w))
              (cond->
                at (update-in [:ammo at] dec))))
        ;; Weapon switch
        (:switch-weapon input)
        (let [target (:switch-weapon input)]
          (if (contains? (:weapons player) target)
            (assoc player :weapon-state :lowering :weapon-tics 6
                   :pending-weapon target)
            player))
        :else player)

      :firing
      (if (<= wt 1)
        (assoc player :weapon-state :ready :weapon-frame 0 :weapon-tics 0)
        (assoc player :weapon-tics (dec wt) :weapon-frame (inc wf)))

      :lowering
      (if (<= wt 1)
        (assoc player :weapon-state :raising :weapon-tics 6
               :current-weapon (:pending-weapon player))
        (assoc player :weapon-tics (dec wt)))

      :raising
      (if (<= wt 1)
        (assoc player :weapon-state :ready :weapon-tics 0)
        (assoc player :weapon-tics (dec wt)))

      player)))

;; ================================================================
;; Damage
;; ================================================================

(defn roll-damage
  "Roll damage in [min max] range."
  [damage-spec]
  (if (vector? damage-spec)
    (let [[lo hi] damage-spec]
      (+ lo (rand-int (inc (- hi lo)))))
    (int damage-spec)))

(defn apply-damage
  "Apply damage to player. Armor absorbs 1/3."
  [player ^long damage]
  (let [armor (:armor player)
        armor-save (min armor (int (/ damage 3)))
        health-dmg (- damage armor-save)]
    (-> player
        (update :armor - armor-save)
        (update :health - health-dmg))))

(defn alive? [player]
  (pos? (:health player)))

;; ================================================================
;; Hitscan
;; ================================================================

(defn hitscan-test
  "Test a hitscan ray against blocking linedefs.
  Returns {:hit? bool :distance double :type :wall/:monster :target ...} or nil."
  [map-data [^double ox ^double oy] [^double dx ^double dy] ^double max-range]
  (let [linedefs (:linedefs map-data)
        vertexes (:vertexes map-data)
        ;; Normalize ray direction
        len (Math/sqrt (+ (* dx dx) (* dy dy)))
        rdx (/ dx len)
        rdy (/ dy len)]
    ;; Test all blocking linedefs
    (loop [remaining linedefs
           best-dist max-range
           best-hit nil]
      (if (empty? remaining)
        best-hit
        (let [ld (first remaining)
              [ax ay] (nth vertexes (:v1 ld))
              [bx by] (nth vertexes (:v2 ld))
              ax (double ax) ay (double ay) bx (double bx) by (double by)
              ;; Ray-segment intersection (2D)
              sdx (- bx ax) sdy (- by ay)
              denom (- (* rdx sdy) (* rdy sdx))]
          (if (< (Math/abs denom) 1e-10)
            (recur (rest remaining) best-dist best-hit)
            (let [t (/ (- (* (- ax ox) sdy) (* (- ay oy) sdx)) denom)
                  u (/ (- (* (- ax ox) rdy) (* (- ay oy) rdx)) denom)]
              (if (and (> t 0) (< t best-dist) (>= u 0) (<= u 1)
                       ;; Only block on one-sided or impassable linedefs
                       (or (nil? (:back-sidedef ld))
                           (not= 0 (bit-and (:flags ld) 1))))
                (recur (rest remaining) t {:hit? true :distance t :type :wall})
                (recur (rest remaining) best-dist best-hit)))))))))

;; ================================================================
;; Per-frame pickup check
;; ================================================================

(def ^:const PICKUP-RADIUS 32.0)

(defn check-pickups
  "Check player proximity to things for pickups. Returns [updated-player updated-things]."
  [player things ^double px ^double py]
  (loop [remaining things
         p player
         kept (transient [])]
    (if (empty? remaining)
      [p (persistent! kept)]
      (let [thing (first remaining)
            type-info (get wad/thing-types (:type thing))]
        (if (and type-info
                 (#{:item :weapon :ammo :key} (:category type-info))
                 (:pickup type-info))
          (let [dx (- (double (:x thing)) px)
                dy (- (double (:y thing)) py)
                dist (Math/sqrt (+ (* dx dx) (* dy dy)))
                pickup-info (get pickups (:pickup type-info))]
            (if (and (< dist PICKUP-RADIUS) pickup-info (can-pickup? p pickup-info))
              (recur (rest remaining) (apply-pickup p pickup-info) kept)
              (recur (rest remaining) p (conj! kept thing))))
          (recur (rest remaining) p (conj! kept thing)))))))

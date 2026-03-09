(ns valley.hostile
  "Hostile mob system: zombies, skeletons, lurkers, spiders.
  Night spawning, chase AI, melee/ranged attacks, sunburn."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.core :as vc]
            [valley.world :as w]
            [valley.mobs :as mobs]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Hostile mob definitions
;; ================================================================

(def hostile-defs
  {:zombie   {:max-health 20 :speed 2.3 :damage 3 :reach 1.5 :attack-rate 1.0
              :aabb [0.4 1.8] :burns-in-sun? true :spawn-light-max 0
              :drops [{:item :dirt :min 0 :max 2}]}
   :skeleton {:max-health 20 :speed 2.5 :damage 2 :reach 16.0 :ranged? true
              :attack-rate 2.0 :aabb [0.4 1.8] :burns-in-sun? true :spawn-light-max 0
              :drops [{:item :stick :min 0 :max 2}]}
   :lurker  {:max-health 20 :speed 1.8 :damage 0 :reach 2.5 :fuse-time 1.5
              :explode-radius 3 :aabb [0.3 1.7] :burns-in-sun? false :spawn-light-max 0
              :drops [{:item :dirt :min 0 :max 1}]}
   :spider   {:max-health 16 :speed 2.8 :damage 2 :reach 1.5 :attack-rate 1.0
              :aabb [0.6 0.6] :neutral-in-day? true :spawn-light-max 0
              :drops [{:item :stick :min 0 :max 2}]}})

(def hostile-types (vec (keys hostile-defs)))

;; ================================================================
;; Hostile mob creation
;; ================================================================

(defn create-hostile
  "Create a new hostile mob entity at position [x y z]."
  [mob-type pos]
  (let [hdef (get hostile-defs mob-type)]
    {:id (random-uuid)
     :mob-type mob-type
     :hostile? true
     :pos pos
     :vel [0.0 0.0 0.0]
     :yaw (* (rand) 2.0 Math/PI)
     :health (long (:max-health hdef))
     :max-health (long (:max-health hdef))
     :state :idle
     :state-timer (+ 2.0 (* (rand) 3.0))
     :walk-phase 0.0
     :wander-dir [(- (rand) 0.5) (- (rand) 0.5)]
     :hurt-timer 0.0
     :attack-timer 0.0
     :fuse-timer 0.0
     :on-ground? false}))

;; ================================================================
;; Line of sight
;; ================================================================

(defn- can-see-player?
  "Step-based raycast to check if mob can see player (no solid blocks in between)."
  [world mob-pos player-pos]
  (let [[mx my mz] mob-pos
        [px py pz] player-pos
        mx (double mx) my (+ (double my) 1.0)  ;; eye height
        mz (double mz)
        px (double px) py (double py) pz (double pz)
        dx (- px mx) dy (- py my) dz (- pz mz)
        dist (Math/sqrt (+ (* dx dx) (* dy dy) (* dz dz)))]
    (if (< dist 0.5)
      true
      (let [steps (long (Math/ceil (* dist 2.0)))
            sx (/ dx steps) sy (/ dy steps) sz (/ dz steps)]
        (loop [i 1]
          (if (>= i steps)
            true
            (let [cx (+ mx (* sx (double i)))
                  cy (+ my (* sy (double i)))
                  cz (+ mz (* sz (double i)))
                  block (w/get-world-block world
                          (long (Math/floor cx))
                          (long (Math/floor cy))
                          (long (Math/floor cz)))]
              (if (w/solid? block)
                false
                (recur (inc i))))))))))

;; ================================================================
;; Hostile AI
;; ================================================================

(def ^:const CHASE-RANGE 16.0)
(def ^:const LOSE-RANGE 24.0)

(defn- pos-x ^double [pos]
  (if (instance? (Class/forName "[D") pos) (aget ^doubles pos 0) (double (nth pos 0))))
(defn- pos-z ^double [pos]
  (if (instance? (Class/forName "[D") pos) (aget ^doubles pos 2) (double (nth pos 2))))
(defn- pos-y ^double [pos]
  (if (instance? (Class/forName "[D") pos) (aget ^doubles pos 1) (double (nth pos 1))))

(defn- dist-xz-pos
  "Horizontal distance between two positions (arrays or vectors)."
  ^double [pos1 pos2]
  (vc/dist-xz (pos-x pos1) (pos-z pos1) (pos-x pos2) (pos-z pos2)))

(defn- dir-toward
  "Normalized XZ direction from mob toward target position."
  [mob-pos target-pos]
  (vc/dir-toward-xz (pos-x mob-pos) (pos-z mob-pos)
                     (pos-x target-pos) (pos-z target-pos)))

(defn tick-hostile-ai
  "Update hostile mob AI. Returns updated mob."
  [mob world player-pos day-ratio dt]
  (let [dt (double dt) day-ratio (double day-ratio)
        hdef (get hostile-defs (:mob-type mob))
        dist (dist-xz-pos (:pos mob) player-pos)
        timer (- (double (:state-timer mob)) dt)
        state (:state mob)
        ;; Spider is neutral during day
        neutral? (and (:neutral-in-day? hdef) (> day-ratio 0.5))
        can-chase? (and (< dist CHASE-RANGE)
                        (not neutral?)
                        (can-see-player? world (:pos mob) player-pos))]
    (case state
      :idle
      (if can-chase?
        (assoc mob :state :chase :state-timer 0.5
               :wander-dir (dir-toward (:pos mob) player-pos))
        (if (> timer 0.0)
          (assoc mob :state-timer timer)
          (assoc mob :state :wander
                 :state-timer (+ 3.0 (* (rand) 5.0))
                 :wander-dir (mobs/pick-wander-dir))))

      :wander
      (if can-chase?
        (assoc mob :state :chase :state-timer 0.5
               :wander-dir (dir-toward (:pos mob) player-pos))
        (if (> timer 0.0)
          (assoc mob :state-timer timer)
          (assoc mob :state :idle
                 :state-timer (+ 2.0 (* (rand) 3.0)))))

      :chase
      (let [reach (double (:reach hdef 1.5))
            in-range? (< dist reach)]
        (cond
          ;; Lost target
          (> dist LOSE-RANGE)
          (assoc mob :state :idle :state-timer 2.0)

          ;; In attack range
          in-range?
          (assoc mob :state :attack :state-timer 0.0
                 :wander-dir (dir-toward (:pos mob) player-pos))

          ;; Keep chasing
          :else
          (assoc mob :wander-dir (dir-toward (:pos mob) player-pos)
                 :state-timer 0.5)))

      :attack
      (if (> dist (* 1.5 (double (:reach hdef 1.5))))
        ;; Target moved out of range
        (assoc mob :state :chase :state-timer 0.5
               :wander-dir (dir-toward (:pos mob) player-pos))
        ;; Stay in attack state, face target
        (assoc mob :wander-dir (dir-toward (:pos mob) player-pos)))

      ;; Default
      (assoc mob :state :idle :state-timer 2.0))))

;; ================================================================
;; Attack logic
;; ================================================================

(defn tick-hostile-attack
  "Process hostile attack. Returns {:mob updated-mob :player-damage amount}."
  [mob player-pos dt]
  (let [dt (double dt)
        hdef (get hostile-defs (:mob-type mob))
        at (+ (double (:attack-timer mob)) dt)
        rate (double (:attack-rate hdef 1.0))
        dist (dist-xz-pos (:pos mob) player-pos)
        reach (double (:reach hdef 1.5))
        in-range? (< dist reach)]
    (if (and (= (:state mob) :attack) in-range? (>= at rate))
      ;; Attack!
      {:mob (assoc mob :attack-timer 0.0)
       :player-damage (long (:damage hdef 0))}
      ;; Not attacking yet
      {:mob (assoc mob :attack-timer at)
       :player-damage 0})))

;; ================================================================
;; Lurker fuse
;; ================================================================

(defn tick-lurker-fuse
  "Tick lurker fuse. Returns {:mob :explode? :pos}."
  [mob player-pos dt]
  (if (not= (:mob-type mob) :lurker)
    {:mob mob :explode? false}
    (let [hdef (get hostile-defs :lurker)
          dist (dist-xz-pos (:pos mob) player-pos)
          reach (double (:reach hdef 2.5))
          fuse (double (:fuse-timer mob 0.0))]
      (if (and (= (:state mob) :attack) (< dist reach))
        ;; Fuse is burning
        (let [new-fuse (+ fuse (double dt))
              fuse-time (double (:fuse-time hdef 1.5))]
          (if (>= new-fuse fuse-time)
            {:mob mob :explode? true :pos (mobs/mob-pos-vec mob)}
            {:mob (assoc mob :fuse-timer new-fuse) :explode? false}))
        ;; Not in range, reset fuse
        {:mob (assoc mob :fuse-timer 0.0) :explode? false}))))

;; ================================================================
;; Sunburn
;; ================================================================

(defn tick-sunburn
  "Burn zombie/skeleton in direct sunlight. Returns updated mob."
  [mob world ^double day-ratio ^double dt]
  (let [hdef (get hostile-defs (:mob-type mob))]
    (if (and (:burns-in-sun? hdef) (> day-ratio 0.7))
      ;; Check if sky-exposed (scan up for solid blocks)
      (let [mx (pos-x (:pos mob)) my (pos-y (:pos mob)) mz (pos-z (:pos mob))
            bx (long (Math/floor mx))
            bz (long (Math/floor mz))
            head-y (long (Math/floor (+ my 1.5)))
            exposed? (loop [y (inc head-y)]
                       (if (> y 128) true
                         (let [b (w/get-world-block world bx y bz)]
                           (if (w/solid? b) false
                             (recur (inc y))))))]
        (if (and exposed? (zero? (double (:hurt-timer mob))))
          (mobs/damage-mob mob 1 (mobs/mob-pos-vec mob))
          mob))
      mob)))

;; ================================================================
;; Spawning
;; ================================================================

(def ^:const MAX-HOSTILES 20)
(def ^:const HOSTILE-SPAWN-RADIUS 64.0)
(def ^:const HOSTILE-MIN-SPAWN-DIST 24.0)

(defn- find-hostile-spawn-pos
  "Find a dark spawn position for hostile mobs. Returns [x y z] or nil."
  [world player-pos]
  (let [[px py pz] player-pos
        px (double px) pz (double pz)
        angle (* (rand) 2.0 Math/PI)
        dist (+ HOSTILE-MIN-SPAWN-DIST (* (rand) (- HOSTILE-SPAWN-RADIUS HOSTILE-MIN-SPAWN-DIST)))
        sx (+ px (* dist (Math/cos angle)))
        sz (+ pz (* dist (Math/sin angle)))
        start-y (long (+ py 10))]
    (loop [y start-y]
      (when (> y 1)
        (let [bx (long (Math/floor sx))
              bz (long (Math/floor sz))
              block (w/get-world-block world bx y bz)
              block-above (w/get-world-block world bx (inc y) bz)
              block-above2 (w/get-world-block world bx (+ y 2) bz)]
          (if (and (w/solid? block)
                   (= block-above w/AIR)
                   (= block-above2 w/AIR))
            ;; Check darkness: scan up for sky exposure
            (let [dark? (loop [sy (+ y 3)]
                          (if (> sy 128) false  ;; sky visible = light
                            (if (w/solid? (w/get-world-block world bx sy bz))
                              true  ;; blocked = dark
                              (recur (inc sy)))))]
              (if dark?
                [sx (double (inc y)) sz]
                (recur (dec y))))
            (recur (dec y))))))))

(defn spawn-hostiles
  "Try to spawn hostile mobs at night. Returns updated mob list."
  [all-mobs world player-pos day-ratio]
  (let [hostiles (filterv :hostile? all-mobs)]
    (if (or (>= (count hostiles) MAX-HOSTILES) (> (double day-ratio) 0.5))
      all-mobs
      (let [mob-type (nth hostile-types (rand-int (count hostile-types)))
            pos (find-hostile-spawn-pos world player-pos)]
        (if pos
          (conj all-mobs (create-hostile mob-type pos))
          all-mobs)))))

;; ================================================================
;; Lurker explosion (modify world)
;; ================================================================

(defn explode-blocks
  "Destroy blocks in a sphere around pos. Returns list of [bx by bz] destroyed."
  [world pos ^long radius]
  (let [[ex ey ez] pos
        ex (long (Math/floor (double ex)))
        ey (long (Math/floor (double ey)))
        ez (long (Math/floor (double ez)))
        destroyed (atom [])]
    (doseq [dx (range (- radius) (inc radius))
            dy (range (- radius) (inc radius))
            dz (range (- radius) (inc radius))]
      (let [bx (+ ex dx) by (+ ey dy) bz (+ ez dz)
            dist-sq (+ (* dx dx) (* dy dy) (* dz dz))]
        (when (and (<= dist-sq (* radius radius))
                   (> by 0)  ;; don't destroy bedrock layer
                   (let [block (w/get-world-block world bx by bz)]
                     (and (not= block w/AIR) (not= block w/BEDROCK))))
          (swap! destroyed conj [bx by bz]))))
    @destroyed))

;; ================================================================
;; Tick hostile mob (combines AI + physics + sunburn)
;; ================================================================

(defn tick-hostile
  "Per-frame update for a hostile mob. Returns updated mob.
  solid-arr: byte-array of block solidity for compiled physics."
  [mob world solid-arr player-pos day-ratio dt]
  (-> mob
      (tick-hostile-ai world player-pos day-ratio dt)
      (mobs/tick-mob world solid-arr dt)
      (tick-sunburn world (double day-ratio) (double dt))))

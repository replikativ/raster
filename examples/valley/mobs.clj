(ns valley.mobs
  "Passive mob system: spawning, AI, physics, drops.
  Mob types: cow, pig, chicken, sheep.
  Physics uses compiled deftm kernels from valley.core."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn]]
            [valley.world :as w]
            [valley.core :as tc]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Mob definitions
;; ================================================================

(def mob-defs
  {:cow     {:max-health 10 :speed 2.0 :aabb [0.45 1.4]
             :drops [{:item :dirt :min 1 :max 3}]
             :passive? true}
   :pig     {:max-health 10 :speed 2.0 :aabb [0.4 0.9]
             :drops [{:item :dirt :min 1 :max 2}]
             :passive? true}
   :chicken {:max-health 4  :speed 1.5 :aabb [0.2 0.7]
             :drops [{:item :dirt :min 1 :max 1}]
             :passive? true}
   :sheep   {:max-health 8  :speed 2.0 :aabb [0.45 1.3]
             :drops [{:item :dirt :min 1 :max 2}]
             :passive? true}
   ;; Hostile mobs (physics/collision needs these)
   :zombie   {:max-health 20 :speed 2.3 :aabb [0.4 1.8]
              :drops [{:item :dirt :min 0 :max 2}]}
   :skeleton {:max-health 20 :speed 2.5 :aabb [0.4 1.8]
              :drops [{:item :stick :min 0 :max 2}]}
   :lurker  {:max-health 20 :speed 1.8 :aabb [0.3 1.7]
              :drops [{:item :dirt :min 0 :max 1}]}
   :spider   {:max-health 16 :speed 2.8 :aabb [0.6 0.6]
              :drops [{:item :stick :min 0 :max 2}]}})

(def mob-types (vec (keys mob-defs)))

;; ================================================================
;; Mob creation
;; ================================================================

(defn create-mob
  "Create a new mob entity at position [x y z].
  pos/vel are mutable double-arrays for compiled physics."
  [mob-type pos]
  (let [def (get mob-defs mob-type)
        pos-arr (if (instance? (Class/forName "[D") pos)
                  pos
                  (double-array [(double (nth pos 0)) (double (nth pos 1)) (double (nth pos 2))]))]
    {:id (random-uuid)
     :mob-type mob-type
     :pos pos-arr
     :vel (double-array [0.0 0.0 0.0])
     :yaw (* (rand) 2.0 Math/PI)
     :health (long (:max-health def))
     :max-health (long (:max-health def))
     :state :idle
     :state-timer (+ 2.0 (* (rand) 3.0))
     :walk-phase 0.0
     :wander-dir [(- (rand) 0.5) (- (rand) 0.5)]
     :hurt-timer 0.0
     :on-ground? false}))

;; ================================================================
;; AI state machine
;; ================================================================

(defn pick-wander-dir []
  (let [angle (* (rand) 2.0 Math/PI)]
    [(Math/cos angle) (Math/sin angle)]))

(defn- tick-ai
  "Update mob AI state. Returns updated mob."
  [mob ^double dt]
  (let [timer (- (double (:state-timer mob)) dt)
        state (:state mob)]
    (if (> timer 0.0)
      ;; Continue current state
      (assoc mob :state-timer timer)
      ;; Transition
      (case state
        :idle (assoc mob
                :state :wander
                :state-timer (+ 3.0 (* (rand) 5.0))
                :wander-dir (pick-wander-dir))
        :wander (assoc mob
                  :state :idle
                  :state-timer (+ 2.0 (* (rand) 3.0)))
        :flee (assoc mob
                :state :idle
                :state-timer (+ 1.0 (* (rand) 2.0)))
        ;; default
        (assoc mob :state :idle :state-timer 2.0)))))

;; ================================================================
;; Mob physics
;; ================================================================

;; ================================================================
;; Mob position accessors (pos/vel are mutable double-arrays)
;; ================================================================

(defn mob-x ^double [mob] (aget ^doubles (:pos mob) 0))
(defn mob-y ^double [mob] (aget ^doubles (:pos mob) 1))
(defn mob-z ^double [mob] (aget ^doubles (:pos mob) 2))
(defn mob-pos-vec
  "Get mob position as a persistent vector (for Clojure interop)."
  [mob]
  (let [^doubles p (:pos mob)]
    [(aget p 0) (aget p 1) (aget p 2)]))

;; ================================================================
;; Mob physics — compiled deftm core + defn boundary
;; ================================================================

(defn- get-mob-chunk
  "Get the chunk blocks array and origin for the chunk containing the mob.
  Returns [blocks-array cx cy cz] or nil if chunk not loaded."
  [world mob]
  (let [mx (long (Math/floor (mob-x mob)))
        my (long (Math/floor (mob-y mob)))
        mz (long (Math/floor (mob-z mob)))
        cx (Math/floorDiv mx (long 16))
        cy (Math/floorDiv my (long 16))
        cz (Math/floorDiv mz (long 16))]
    (when-let [chunk (get world [cx cy cz])]
      [(:blocks chunk) cx cy cz])))

(defn- tick-physics
  "Apply gravity and movement to mob. Uses compiled deftm physics.
  Falls back to simple physics if chunk not available."
  [mob world solid-arr ^double dt]
  (let [^doubles pos (:pos mob)
        ^doubles vel (:vel mob)
        mx (aget pos 0) my (aget pos 1) mz (aget pos 2)
        mob-type (:mob-type mob)
        mdef (get mob-defs mob-type)
        speed (double (:speed mdef))
        [hw h] (:aabb mdef)

        ;; Movement direction from AI state
        [dx dz] (case (:state mob)
                  :wander (let [[wx wz] (:wander-dir mob)]
                            [(* (double wx) speed dt)
                             (* (double wz) speed dt)])
                  :flee   (let [[wx wz] (:wander-dir mob)]
                            [(* (double wx) speed 1.5 dt)
                             (* (double wz) speed 1.5 dt)])
                  [0.0 0.0])

        ;; Try compiled physics path
        chunk-data (get-mob-chunk world mob)
        on-ground?
        (if chunk-data
          ;; Compiled path: deftm integrate-physics! mutates pos/vel in-place
          (let [[blocks cx cy cz] chunk-data]
            (= 1 (tc/integrate-physics! pos vel blocks solid-arr
                    cx cy cz (double hw) (double h)
                    (double dx) (double dz) dt)))
          ;; Fallback: simple gravity without collision
          (do (aset vel 1 (- (aget vel 1) (* 20.0 dt)))
              (aset pos 0 (+ mx (double dx)))
              (aset pos 1 (+ my (* (aget vel 1) dt)))
              (aset pos 2 (+ mz (double dz)))
              false))

        ;; Derived state from updated pos/vel
        new-mx (aget pos 0) new-mz (aget pos 2)
        blocked? (and (= new-mx mx) (= new-mz mz)
                      (or (not (zero? (double dx))) (not (zero? (double dz)))))
        actual-dx (- new-mx mx)
        actual-dz (- new-mz mz)
        yaw (if (or (not (zero? actual-dx)) (not (zero? actual-dz)))
              (Math/atan2 actual-dx actual-dz)
              (double (:yaw mob)))
        walk-phase (if (and (= (:state mob) :wander) (not blocked?))
                     (mod (+ (double (:walk-phase mob)) (* dt 2.0)) 1.0)
                     0.0)]

    (cond-> (assoc mob
              :yaw yaw
              :walk-phase walk-phase
              :on-ground? on-ground?)
      blocked? (assoc :wander-dir (pick-wander-dir)))))

;; ================================================================
;; Damage and death
;; ================================================================

(defn damage-mob
  "Apply damage to a mob. Sets flee state. Returns updated mob."
  [mob ^long amount player-pos]
  (let [health (max 0 (- (long (:health mob)) amount))
        ;; Flee away from player
        mx (mob-x mob) mz (mob-z mob)
        px (double (nth player-pos 0)) pz (double (nth player-pos 2))
        flee-dx (- mx px)
        flee-dz (- mz pz)
        len (Math/sqrt (+ (* flee-dx flee-dx) (* flee-dz flee-dz)))
        flee-dir (if (> len 0.001)
                   [(/ flee-dx len) (/ flee-dz len)]
                   (pick-wander-dir))]
    (assoc mob
      :health health
      :state :flee
      :state-timer 3.0
      :wander-dir flee-dir
      :hurt-timer 0.3)))

(defn mob-dead? [mob] (<= (long (:health mob)) 0))

(defn mob-drops
  "Get drop items for a dead mob. Returns seq of {:item :count}."
  [mob]
  (let [def (get mob-defs (:mob-type mob))]
    (mapv (fn [drop-def]
            {:item (:item drop-def)
             :count (+ (long (:min drop-def))
                       (long (rand-int (inc (- (long (:max drop-def))
                                               (long (:min drop-def)))))))})
          (:drops def))))

;; ================================================================
;; Tick mob
;; ================================================================

(defn tick-mob
  "Per-frame update for a mob. Returns updated mob.
  solid-arr: byte-array of block solidity (from tc/make-solid-bytes)."
  [mob world solid-arr ^double dt]
  (let [mob (-> mob
                (tick-ai dt)
                (tick-physics world solid-arr dt))
        ;; Tick hurt timer
        ht (double (:hurt-timer mob))
        mob (if (pos? ht)
              (assoc mob :hurt-timer (max 0.0 (- ht dt)))
              mob)]
    mob))

;; ================================================================
;; Spawning
;; ================================================================

(def ^:const MAX-MOBS 30)
(def ^:const SPAWN-RADIUS 48.0)
(def ^:const MIN-SPAWN-DIST 24.0)
(def ^:const DESPAWN-RADIUS 128.0)

(defn- find-spawn-pos
  "Find a valid spawn position near the player. Returns [x y z] or nil."
  [world player-pos]
  (let [[px py pz] player-pos
        px (double px) pz (double pz)
        ;; Random position in spawn ring
        angle (* (rand) 2.0 Math/PI)
        dist (+ MIN-SPAWN-DIST (* (rand) (- SPAWN-RADIUS MIN-SPAWN-DIST)))
        sx (+ px (* dist (Math/cos angle)))
        sz (+ pz (* dist (Math/sin angle)))
        ;; Find ground level (scan down from player height + 20)
        start-y (long (+ py 20))]
    (loop [y start-y]
      (when (> y 1)
        (let [block (w/get-world-block world (long (Math/floor sx)) y (long (Math/floor sz)))
              block-above (w/get-world-block world (long (Math/floor sx)) (inc y) (long (Math/floor sz)))
              block-above2 (w/get-world-block world (long (Math/floor sx)) (+ y 2) (long (Math/floor sz)))]
          (if (and (= block w/GRASS)
                   (= block-above w/AIR)
                   (= block-above2 w/AIR))
            [sx (double (+ y 1)) sz]
            (recur (dec y))))))))

(defn spawn-mobs
  "Try to spawn passive mobs near the player. Returns updated mob list."
  [mobs world player-pos]
  (if (>= (count mobs) MAX-MOBS)
    mobs
    ;; Try to spawn a herd of 2-4 mobs
    (let [mob-type (nth mob-types (rand-int (count mob-types)))
          herd-size (+ 2 (rand-int 3))
          base-pos (find-spawn-pos world player-pos)]
      (if base-pos
        (let [[bx by bz] base-pos]
          (reduce (fn [mobs _]
                    (if (>= (count mobs) MAX-MOBS)
                      mobs
                      (let [ox (* (- (rand) 0.5) 6.0)
                            oz (* (- (rand) 0.5) 6.0)
                            pos [(+ (double bx) ox) by (+ (double bz) oz)]]
                        (conj mobs (create-mob mob-type pos)))))
                  mobs (range herd-size)))
        mobs))))

(defn despawn-mobs
  "Remove mobs too far from the player."
  [mobs player-pos]
  (let [[px _py pz] player-pos
        px (double px) pz (double pz)]
    (filterv (fn [mob]
               (let [dx (- (mob-x mob) px)
                     dz (- (mob-z mob) pz)
                     dist (Math/sqrt (+ (* dx dx) (* dz dz)))]
                 (< dist DESPAWN-RADIUS)))
             mobs)))

;; ================================================================
;; Hit detection (player attack → mob)
;; ================================================================

(defn mob-aabb
  "Get AABB [min-x min-y min-z max-x max-y max-z] for a mob."
  [mob]
  (let [mx (mob-x mob) my (mob-y mob) mz (mob-z mob)
        def (get mob-defs (:mob-type mob))
        [hw h] (:aabb def)
        hw (double hw) h (double h)
        r (* hw 0.5)]
    [(- mx r) my (- mz r) (+ mx r) (+ my h) (+ mz r)]))

(defn ray-intersects-aabb?
  "Check if ray from origin in direction hits AABB. Returns distance or nil."
  [ox oy oz dx dy dz min-x min-y min-z max-x max-y max-z]
  (let [ox (double ox) oy (double oy) oz (double oz)
        dx (double dx) dy (double dy) dz (double dz)
        inv-dx (if (zero? dx) Double/MAX_VALUE (/ 1.0 dx))
        inv-dy (if (zero? dy) Double/MAX_VALUE (/ 1.0 dy))
        inv-dz (if (zero? dz) Double/MAX_VALUE (/ 1.0 dz))
        t1x (* (- (double min-x) ox) inv-dx)
        t2x (* (- (double max-x) ox) inv-dx)
        t1y (* (- (double min-y) oy) inv-dy)
        t2y (* (- (double max-y) oy) inv-dy)
        t1z (* (- (double min-z) oz) inv-dz)
        t2z (* (- (double max-z) oz) inv-dz)
        tmin (max (min t1x t2x) (min t1y t2y) (min t1z t2z))
        tmax (min (max t1x t2x) (max t1y t2y) (max t1z t2z))]
    (when (and (<= tmin tmax) (>= tmax 0.0) (<= tmin 6.0))
      tmin)))

(defn find-hit-mob
  "Find the closest mob hit by a ray. Returns [mob distance] or nil."
  [mobs ox oy oz dx dy dz]
  (let [ox (double ox) oy (double oy) oz (double oz)
        dx (double dx) dy (double dy) dz (double dz)]
    (reduce (fn [best mob]
              (when (zero? (double (:hurt-timer mob)))
                (let [[mnx mny mnz mxx mxy mxz] (mob-aabb mob)
                      dist (ray-intersects-aabb? ox oy oz dx dy dz
                                                 mnx mny mnz mxx mxy mxz)]
                  (if (and dist (or (nil? best) (< (double dist) (double (second best)))))
                    [mob dist]
                    best))))
            nil mobs)))

(ns doom.monster
  "Monster AI: state machines for E1M1 monsters (Zombieman, Shotgun Guy, Imp).
  Movement uses Doom's 8-direction movedir/movecount system from P_Enemy.c."
  (:refer-clojure :exclude [defn])
  (:require
   [raster.core :refer [defn]]
   [doom.wad :as wad]
   [doom.bsp :as bsp]
   [doom.player :as player]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Constants
;; ================================================================

(def ^:const TIC-RATE 35)
(def ^:const MELEE-RANGE 64.0)
(def ^:const MISSILE-RANGE 1024.0)
(def ^:const AWARENESS-RANGE 4096.0)

;; 8 directions + no-dir, matching Doom's dirtype_t
(def ^:const DI-EAST 0)
(def ^:const DI-NORTHEAST 1)
(def ^:const DI-NORTH 2)
(def ^:const DI-NORTHWEST 3)
(def ^:const DI-WEST 4)
(def ^:const DI-SOUTHWEST 5)
(def ^:const DI-SOUTH 6)
(def ^:const DI-SOUTHEAST 7)
(def ^:const DI-NODIR 8)

(def opposite [DI-WEST DI-SOUTHWEST DI-SOUTH DI-SOUTHEAST
               DI-EAST DI-NORTHEAST DI-NORTH DI-NORTHWEST DI-NODIR])

;; Movement deltas per direction (unit vectors)
(def dir-dx [1.0  0.7071  0.0 -0.7071 -1.0 -0.7071  0.0  0.7071 0.0])
(def dir-dy [0.0  0.7071  1.0  0.7071  0.0 -0.7071 -1.0 -0.7071 0.0])

;; ================================================================
;; Monster creation
;; ================================================================

(defn create-monster
  "Create a monster entity from a thing."
  [thing]
  (let [type-info (get wad/thing-types (:type thing))]
    (when (= (:category type-info) :monster)
      {:type (:type thing)
       :info type-info
       :x (double (:x thing))
       :y (double (:y thing))
       :angle (double (:angle thing))
       :state :idle
       :health (or (:health type-info) 20)
       :tics (+ 10 (rand-int 20))
       :frame 0
       :active? false
       :movedir DI-NODIR
       :move-count 0
       :reaction-time 8})))

(defn init-monsters
  "Initialize monsters from map things."
  [map-data]
  (vec (keep create-monster (:things map-data))))

;; ================================================================
;; Line of sight
;; ================================================================

(def ^:const EYE-HEIGHT 41.0) ;; Doom player/monster eye height

(defn has-line-of-sight?
  "Check if monster can see player. Uses height-aware ray at eye level.
  One-sided walls block. Two-sided linedefs block if the eye-height ray
  can't pass through the opening between sectors."
  [map-data mx my px py]
  (let [mx (double mx) my (double my) px (double px) py (double py)
        dx (- px mx) dy (- py my)
        dist (Math/sqrt (+ (* dx dx) (* dy dy)))]
    (when (< dist AWARENESS-RANGE)
      (let [linedefs (:linedefs map-data)
            vertexes (:vertexes map-data)
            sidedefs (:sidedefs map-data)
            sectors (:sectors map-data)
            rdx (/ dx (max dist 1.0))
            rdy (/ dy (max dist 1.0))
            ;; Get floor heights at monster and player positions for Z interpolation
            m-sector-idx (doom.bsp/point-sector map-data mx my)
            p-sector-idx (doom.bsp/point-sector map-data px py)
            m-floor (if m-sector-idx
                      (double (:floor-height (nth sectors m-sector-idx)))
                      0.0)
            p-floor (if p-sector-idx
                      (double (:floor-height (nth sectors p-sector-idx)))
                      0.0)
            m-eye (+ m-floor EYE-HEIGHT)
            p-eye (+ p-floor EYE-HEIGHT)]
        (loop [remaining linedefs]
          (if (empty? remaining)
            true
            (let [ld (first remaining)
                  blocks?
                  (let [[ax ay] (nth vertexes (:v1 ld))
                        [bx by] (nth vertexes (:v2 ld))
                        ax (double ax) ay (double ay)
                        bx (double bx) by (double by)
                        sdx (- bx ax) sdy (- by ay)
                        denom (- (* rdx sdy) (* rdy sdx))]
                    (when (> (Math/abs denom) 1e-10)
                      (let [t (/ (- (* (- ax mx) sdy) (* (- ay my) sdx)) denom)
                            u (/ (- (* (- ax mx) rdy) (* (- ay my) rdx)) denom)]
                        (when (and (> t 0.1) (< t dist) (>= u 0.0) (<= u 1.0))
                          (if (nil? (:back-sidedef ld))
                            true ;; one-sided wall always blocks
                            ;; Two-sided: check if eye-height ray passes through opening
                            (let [fs-idx (:front-sidedef ld)
                                  bs-idx (:back-sidedef ld)
                                  fs-sec (:sector (nth sidedefs fs-idx))
                                  bs-sec (:sector (nth sidedefs bs-idx))
                                  fs (nth sectors fs-sec)
                                  bs (nth sectors bs-sec)
                                  open-bottom (max (double (:floor-height fs))
                                                   (double (:floor-height bs)))
                                  open-top (min (double (:ceil-height fs))
                                                (double (:ceil-height bs)))
                                  ;; Interpolate eye height at intersection point
                                  frac (/ t dist)
                                  eye-z (+ m-eye (* frac (- p-eye m-eye)))]
                              ;; Block if opening too small or eye can't fit through
                              (or (< (- open-top open-bottom) 4.0)
                                  (< eye-z (+ open-bottom 4.0))
                                  (> eye-z (- open-top 4.0)))))))))]
              (if blocks?
                false
                (recur (rest remaining))))))))))

;; ================================================================
;; Sound propagation — alert monsters when player fires weapon
;; ================================================================

(defn- build-sector-adjacency
  "Build adjacency map: sector-idx -> set of adjacent sector indices.
  Only through two-sided linedefs without the sound-blocking flag (bit 6)."
  [map-data]
  (let [linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)]
    (reduce
      (fn [adj ld]
        (if (and (:back-sidedef ld)
                 (zero? (bit-and (long (:flags ld)) 64))) ;; not sound-blocking
          (let [fs (:sector (nth sidedefs (:front-sidedef ld)))
                bs (:sector (nth sidedefs (:back-sidedef ld)))]
            (-> adj
                (update fs (fnil conj #{}) bs)
                (update bs (fnil conj #{}) fs)))
          adj))
      {}
      linedefs)))

(defonce ^:private cached-adjacency (atom nil))

(defn propagate-sound
  "Flood-fill from player's sector through non-sound-blocking two-sided linedefs.
  Returns set of sector indices that can 'hear' the sound.
  Doom limits sound propagation to 2 steps through sound-blocking lines,
  but we simplify: flood through all non-blocking adjacencies."
  [map-data px py]
  (let [p-sector (doom.bsp/point-sector map-data (double px) (double py))]
    (when p-sector
      (let [adj (or @cached-adjacency
                    (let [a (build-sector-adjacency map-data)]
                      (reset! cached-adjacency a)
                      a))]
        (loop [frontier #{p-sector}
               visited #{}]
          (if (empty? frontier)
            visited
            (let [next-visited (into visited frontier)
                  next-frontier (reduce
                                  (fn [nf s]
                                    (into nf (remove next-visited (get adj s))))
                                  #{}
                                  frontier)]
              (recur next-frontier next-visited))))))))

(defn alert-monsters-by-sound
  "Alert all monsters in sectors that can hear the player's gunfire."
  [monsters map-data sound-sectors]
  (if (empty? sound-sectors)
    monsters
    (let [_sectors (:sectors map-data)]
      (mapv (fn [m]
              (if (and (= (:state m) :idle)
                       (not= (:state m) :dead))
                (let [m-sector (doom.bsp/point-sector map-data
                                 (double (:x m)) (double (:y m)))]
                  (if (and m-sector (contains? sound-sectors m-sector))
                    (assoc m :state :chase :tics 0 :active? true
                             :reaction-time (+ 2 (rand-int 8)) :move-count 0)
                    m))
                m))
            monsters))))

;; ================================================================
;; Direction-based movement (Doom's P_Move / P_NewChaseDir)
;; ================================================================

(defn- try-move-dir
  "Try to move monster in the given direction. Returns [nx ny] or nil if blocked."
  [map-data mx my speed dir]
  (when (not= dir DI-NODIR)
    (let [ddx (* (double (nth dir-dx dir)) speed)
          ddy (* (double (nth dir-dy dir)) speed)
          [nx ny] (bsp/move-and-slide map-data [mx my] [ddx ddy] 8.0)
          moved-dx (- (double nx) mx)
          moved-dy (- (double ny) my)
          moved-dist (Math/sqrt (+ (* moved-dx moved-dx) (* moved-dy moved-dy)))]
      (when (> moved-dist (* speed 0.3))
        [nx ny]))))

(defn- new-chase-dir
  "Pick a new movement direction toward target, with fallbacks.
  Mimics Doom's P_NewChaseDir from p_enemy.c."
  [monster map-data px py speed]
  (let [mx (double (:x monster))
        my (double (:y monster))
        dx (- (double px) mx)
        dy (- (double py) my)
        old-dir (long (or (:movedir monster) DI-NODIR))
        turnaround (long (nth opposite old-dir))
        ;; Determine primary directions
        d1 (cond (> dx 10.0) DI-EAST (< dx -10.0) DI-WEST :else DI-NODIR)
        d2 (cond (> dy 10.0) DI-NORTH (< dy -10.0) DI-SOUTH :else DI-NODIR)
        ;; Try diagonal first
        diag (cond
               (and (not= d1 DI-NODIR) (not= d2 DI-NODIR))
               (cond
                 (and (> dx 0) (> dy 0)) DI-NORTHEAST
                 (and (> dx 0) (< dy 0)) DI-SOUTHEAST
                 (and (< dx 0) (> dy 0)) DI-NORTHWEST
                 :else DI-SOUTHWEST)
               :else DI-NODIR)]
    ;; Phase 1: try diagonal
    (if (and (not= diag DI-NODIR) (not= diag turnaround)
             (try-move-dir map-data mx my speed diag))
      diag
      ;; Phase 2: try cardinal directions (larger delta first)
      (let [[prim sec] (if (> (Math/abs dx) (Math/abs dy)) [d1 d2] [d2 d1])]
        (or
          (when (and (not= prim DI-NODIR) (not= prim turnaround)
                     (try-move-dir map-data mx my speed prim))
            prim)
          (when (and (not= sec DI-NODIR) (not= sec turnaround)
                     (try-move-dir map-data mx my speed sec))
            sec)
          ;; Phase 3: try old direction
          (when (and (not= old-dir DI-NODIR) (not= old-dir turnaround)
                     (try-move-dir map-data mx my speed old-dir))
            old-dir)
          ;; Phase 4: try all directions
          (let [start (rand-int 8)]
            (loop [i 0]
              (when (< i 8)
                (let [d (mod (+ start i) 8)]
                  (if (and (not= d turnaround)
                           (try-move-dir map-data mx my speed d))
                    d
                    (recur (inc i)))))))
          ;; Phase 5: try turnaround as last resort
          (when (and (not= turnaround DI-NODIR)
                     (try-move-dir map-data mx my speed turnaround))
            turnaround)
          ;; Totally stuck
          DI-NODIR)))))

(defn- move-in-dir
  "Move monster one step in its current movedir. Returns updated monster."
  [monster map-data speed]
  (let [dir (long (or (:movedir monster) DI-NODIR))]
    (if (= dir DI-NODIR)
      monster
      (let [mx (double (:x monster))
            my (double (:y monster))
            ddx (* (double (nth dir-dx dir)) speed)
            ddy (* (double (nth dir-dy dir)) speed)
            [nx ny] (bsp/move-and-slide map-data [mx my] [ddx ddy] 8.0)
            new-angle (Math/toDegrees (Math/atan2 ddy ddx))]
        (assoc monster :x (double nx) :y (double ny) :angle new-angle)))))

(defn- chase-move
  "Doom-style chase movement: use movedir/movecount system."
  [monster map-data px py dt]
  (let [speed (* (double (or (get-in monster [:info :speed]) 8.0)) (double dt) TIC-RATE)
        mc (long (or (:move-count monster) 0))]
    (if (> mc 0)
      ;; Continue in current direction
      (-> (move-in-dir monster map-data speed)
          (assoc :move-count (dec mc)))
      ;; Need new direction
      (let [new-dir (new-chase-dir monster map-data px py speed)]
        (-> (assoc monster :movedir (long new-dir) :move-count (rand-int 16))
            (move-in-dir map-data speed))))))

;; ================================================================
;; Attack
;; ================================================================

(defn monster-attack
  "Monster attacks player. Returns [monster player projectiles-to-add]."
  [monster player-state px py]
  (let [px (double px) py (double py)
        mx (double (:x monster))
        my (double (:y monster))
        dx (- px mx) dy (- py my)
        dist (Math/sqrt (+ (* dx dx) (* dy dy)))
        info (:info monster)]
    (cond
      ;; Melee attack
      (and (< dist MELEE-RANGE) (:melee-dmg info) (not= (:melee-dmg info) 0))
      (let [dmg (player/roll-damage (:melee-dmg info))]
        [(assoc monster :state :attack :tics 12 :frame 0)
         (player/apply-damage player-state dmg)
         []])

      ;; Ranged attack (hitscan for zombieman/shotgun guy)
      (and (:ranged-dmg info) (not (:projectile info)))
      (let [dmg (player/roll-damage (:ranged-dmg info))
            pellets (if (= (:type monster) 9) 3 1)
            total-dmg (* dmg pellets)]
        [(assoc monster :state :attack :tics 12 :frame 0)
         (player/apply-damage player-state total-dmg)
         []])

      ;; Projectile attack (imp fireball)
      (:projectile info)
      (let [speed 10.0
            ndx (/ dx (max dist 1.0)) ndy (/ dy (max dist 1.0))
            proj {:x mx :y my
                  :dx (* ndx speed) :dy (* ndy speed)
                  :sprite (or (:projectile info) "BAL1")
                  :damage (:ranged-dmg info)
                  :tics 0 :alive? true}]
        [(assoc monster :state :attack :tics 16 :frame 0)
         player-state
         [proj]])

      :else
      [monster player-state []])))

;; ================================================================
;; State machine update
;; ================================================================

(defn update-monster
  "Update a single monster. Returns [monster player-state new-projectiles]."
  [monster map-data player-state px py dt]
  (if (= (:state monster) :dead)
    [monster player-state []]
    (let [player-dead? (not (player/alive? player-state))
          px (double px) py (double py) dt (double dt)
          tic-dec (* dt TIC-RATE)
          tics (- (double (:tics monster)) tic-dec)
          mx (double (:x monster))
          my (double (:y monster))
          dist (let [dx (- px mx) dy (- py my)]
                 (Math/sqrt (+ (* dx dx) (* dy dy))))]
      (case (:state monster)
        :idle
        (if player-dead?
          [monster player-state []]
          (if (<= tics 0)
            (if (has-line-of-sight? map-data mx my px py)
              [(assoc monster :state :chase :tics 0 :active? true
                      :reaction-time 4 :move-count 0)
               player-state []]
              [(assoc monster :tics (+ 10 (rand-int 30)))
               player-state []])
            [(assoc monster :tics tics) player-state []]))

        :chase
        (if player-dead?
          [(assoc monster :state :idle :tics 35) player-state []]
          (let [reaction (long (or (:reaction-time monster) 0))
                monster (if (pos? reaction)
                          (assoc monster :reaction-time (dec reaction))
                          monster)]
            (cond
              ;; Melee attack (highest priority)
              (and (< dist MELEE-RANGE) (:melee-dmg (:info monster))
                   (not= (:melee-dmg (:info monster)) 0)
                   (has-line-of-sight? map-data mx my px py))
              (monster-attack monster player-state px py)

              ;; Missile/ranged attack
              (and (zero? reaction)
                   (< dist MISSILE-RANGE)
                   (has-line-of-sight? map-data mx my px py)
                   (< (rand) (* 0.03 dt TIC-RATE)))
              (monster-attack monster player-state px py)

              ;; Chase movement
              :else
              [(-> (chase-move monster map-data px py dt)
                   (update :frame #(mod (inc (or % 0)) 4)))
               player-state []])))

        :attack
        (if (<= tics 0)
          [(assoc monster :state :chase :tics 0 :move-count 0)
           player-state []]
          [(assoc monster :tics tics :frame (inc (:frame monster)))
           player-state []])

        :pain
        (if (<= tics 0)
          [(assoc monster :state :chase :tics 0 :move-count 0)
           player-state []]
          [(assoc monster :tics tics)
           player-state []])

        :death
        (if (<= tics 0)
          [(assoc monster :state :dead :frame 4)
           player-state []]
          [(assoc monster :tics tics
                  :frame (min 4 (int (/ (- 15.0 (max tics 0)) 3.0))))
           player-state []])

        ;; :dead — do nothing
        [monster player-state []]))))

(defn damage-monster
  "Apply damage to a monster."
  [monster ^long damage]
  (let [new-health (- (:health monster) damage)]
    (if (<= new-health 0)
      (assoc monster :state :death :health 0 :tics 15 :frame 0)
      (let [pain-chance (or (get-in monster [:info :pain-chance]) 128)]
        (if (< (rand-int 256) pain-chance)
          (assoc monster :health new-health :state :pain :tics 4 :frame 0)
          (assoc monster :health new-health))))))

;; ================================================================
;; Batch update all monsters
;; ================================================================

(defn update-monsters
  "Update all monsters. Returns [monsters player-state new-projectiles]."
  [monsters map-data player-state px py dt]
  (loop [remaining monsters
         updated (transient [])
         ps player-state
         projs []]
    (if (empty? remaining)
      [(persistent! updated) ps projs]
      (let [[m ps' new-projs] (update-monster (first remaining) map-data ps px py dt)]
        (recur (rest remaining)
               (conj! updated m)
               ps'
               (into projs new-projs))))))

;; ================================================================
;; Projectile update
;; ================================================================

(defn update-projectiles
  "Update all projectiles (imp fireballs etc). Returns [projectiles player-state]."
  [projectiles map-data player-state px py dt]
  (let [px (double px) py (double py) dt (double dt)]
   (loop [remaining projectiles
          updated (transient [])
          ps player-state]
    (if (empty? remaining)
      [(persistent! updated) ps]
      (let [proj (first remaining)
            nx (+ (double (:x proj)) (* (double (:dx proj)) dt TIC-RATE))
            ny (+ (double (:y proj)) (* (double (:dy proj)) dt TIC-RATE))
            dpx (- px nx) dpy (- py ny)
            player-dist (Math/sqrt (+ (* dpx dpx) (* dpy dpy)))
            wall-hit (player/hitscan-test map-data
                       [(double (:x proj)) (double (:y proj))]
                       [(- nx (double (:x proj))) (- ny (double (:y proj)))]
                       40.0)]
        (cond
          (< player-dist 16.0)
          (let [dmg (player/roll-damage (:damage proj))]
            (recur (rest remaining) updated (player/apply-damage ps dmg)))

          wall-hit
          (recur (rest remaining) updated ps)

          :else
          (let [proj' (assoc proj :x nx :y ny :tics (inc (:tics proj)))]
            (if (> (:tics proj') (* TIC-RATE 5))
              (recur (rest remaining) updated ps)
              (recur (rest remaining) (conj! updated proj') ps)))))))))

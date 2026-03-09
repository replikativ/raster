(ns doom.sector
  "Sector specials: doors, lifts, damage floors, flickering lights, switches."
  (:refer-clojure :exclude [defn])
  (:require
   [raster.core :refer [defn]]
   [doom.bsp :as bsp]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Sector state tracking
;; ================================================================

(def ^:const DOOR-SPEED 2.0) ;; Doom units per tic
(def ^:const FAST-DOOR-SPEED 8.0)
(def ^:const LIFT-SPEED 4.0)
(def ^:const DOOR-WAIT-TICS 150) ;; ~4.3 seconds

;; Door linedef specials that matter for E1M1
(def door-specials
  #{1    ;; DR: Door open wait close
    26   ;; DR: Blue key door
    28   ;; DR: Red key door
    27   ;; DR: Yellow key door
    31   ;; D1: Door open stay
    32   ;; D1: Blue key door open stay
    33   ;; D1: Red key door open stay
    34   ;; D1: Yellow key door open stay
    117  ;; DR: Blazing door open wait close
    118  ;; D1: Blazing door open stay
    })

(def lift-specials
  #{62   ;; SR: Lift lower wait raise
    21   ;; S1: Lift lower wait raise
    88   ;; WR: Lift lower wait raise
    10   ;; W1: Lift lower wait raise
    })

(def floor-specials
  #{20 22 23 19 36 37 38 56 58 59 60 96 70 71 98})

;; ================================================================
;; Initialization
;; ================================================================

(defn find-lowest-adjacent-floor
  "Find the lowest floor height among sectors adjacent to sector-idx."
  [map-data ^long sector-idx]
  (let [linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)]
    (reduce
      (fn [^double lowest linedef]
        (let [fs (:front-sidedef linedef)
              bs (:back-sidedef linedef)]
          (cond
            (and fs (= (:sector (nth sidedefs fs)) sector-idx) bs)
            (min lowest (double (:floor-height (nth sectors (:sector (nth sidedefs bs))))))

            (and bs (= (:sector (nth sidedefs bs)) sector-idx) fs)
            (min lowest (double (:floor-height (nth sectors (:sector (nth sidedefs fs))))))

            :else lowest)))
      Double/MAX_VALUE
      linedefs)))

(defn find-highest-adjacent-ceil
  "Find highest ceiling height among sectors adjacent (for door target)."
  [map-data ^long sector-idx]
  (let [linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)]
    (reduce
      (fn [^double highest linedef]
        (let [fs (:front-sidedef linedef)
              bs (:back-sidedef linedef)]
          (cond
            (and fs (= (:sector (nth sidedefs fs)) sector-idx) bs)
            (max highest (double (:ceil-height (nth sectors (:sector (nth sidedefs bs))))))

            (and bs (= (:sector (nth sidedefs bs)) sector-idx) fs)
            (max highest (double (:ceil-height (nth sectors (:sector (nth sidedefs fs))))))

            :else highest)))
      Double/MIN_VALUE
      linedefs)))

(defn init-sector-states
  "Build initial sector state map. Returns {sector-idx {:door-state ...} ...}."
  [map-data]
  (let [sectors (:sectors map-data)
        linedefs (:linedefs map-data)
        sidedefs (:sidedefs map-data)
        states (atom {})]
    ;; Identify door sectors from linedef specials
    (doseq [linedef linedefs]
      (let [special (:special linedef)]
        ;; Door specials
        (when (contains? door-specials special)
          (let [;; Door sector is the back sector for DR/D1 doors
                bs-idx (:back-sidedef linedef)
                door-sector-idx (when bs-idx
                                  (:sector (nth sidedefs bs-idx)))]
            (when door-sector-idx
              (let [sector (nth sectors door-sector-idx)
                    target-ceil (- (double (find-highest-adjacent-ceil map-data door-sector-idx)) 4.0)]
                (swap! states assoc door-sector-idx
                       {:type :door
                        :state :closed
                        :floor (double (:floor-height sector))
                        :ceil (double (:floor-height sector)) ;; doors start closed (ceil=floor)
                        :target-ceil target-ceil
                        :original-ceil (double (:ceil-height sector))
                        :speed (if (#{117 118} special) FAST-DOOR-SPEED DOOR-SPEED)
                        :wait-tics 0
                        :linedef-special special})))))

        ;; Lift specials
        (when (contains? lift-specials special)
          (let [tag (:tag linedef)]
            (doseq [[si sector] (map-indexed vector sectors)]
              (when (= (:tag sector) tag)
                (let [lowest (find-lowest-adjacent-floor map-data si)]
                  (swap! states assoc si
                         {:type :lift
                          :state :idle
                          :floor (double (:floor-height sector))
                          :target-floor (if (< lowest Double/MAX_VALUE) lowest 0.0)
                          :original-floor (double (:floor-height sector))
                          :speed LIFT-SPEED
                          :wait-tics 0}))))))))

    ;; Damage floors from sector specials
    (doseq [[si sector] (map-indexed vector sectors)]
      (let [special (:special sector)]
        (when (and (#{7 5} special) (not (contains? @states si)))
          (swap! states assoc si
                 {:type :damage
                  :damage-amount (case (int special) 5 10 7 5 5)
                  :damage-interval 32 ;; tics
                  :damage-timer 32}))
        ;; Flickering lights
        (when (and (#{1 2 3 12 13 17} special) (not (contains? @states si)))
          (swap! states assoc si
                 {:type :light
                  :light-state :on
                  :original-light (:light-level sector)
                  :min-light 0
                  :flicker-timer (+ 4 (rand-int 16))
                  :special special}))))
    @states))

;; ================================================================
;; Door state machine
;; ================================================================

(defn update-door
  "Update a door sector. Returns updated state."
  [door-state ^double dt]
  (let [tics (* dt 35.0)
        speed (* (:speed door-state) tics)]
    (case (:state door-state)
      :closed door-state ;; waiting for trigger

      :opening
      (let [new-ceil (+ (:ceil door-state) speed)]
        (if (>= new-ceil (:target-ceil door-state))
          (assoc door-state :ceil (:target-ceil door-state)
                 :state :open :wait-tics DOOR-WAIT-TICS)
          (assoc door-state :ceil new-ceil)))

      :open
      (let [wait (- (:wait-tics door-state) tics)]
        (if (<= wait 0)
          ;; Stay-open doors (D1 type) don't close
          (if (#{31 32 33 34 118} (:linedef-special door-state))
            door-state ;; stay open
            (assoc door-state :state :closing :wait-tics 0))
          (assoc door-state :wait-tics wait)))

      :closing
      (let [new-ceil (- (:ceil door-state) speed)]
        (if (<= new-ceil (:floor door-state))
          (assoc door-state :ceil (:floor door-state) :state :closed)
          (assoc door-state :ceil new-ceil)))

      door-state)))

;; ================================================================
;; Lift state machine
;; ================================================================

(defn update-lift
  "Update a lift sector. Returns updated state."
  [lift-state ^double dt]
  (let [tics (* dt 35.0)
        speed (* (:speed lift-state) tics)]
    (case (:state lift-state)
      :idle lift-state

      :lowering
      (let [new-floor (- (:floor lift-state) speed)]
        (if (<= new-floor (:target-floor lift-state))
          (assoc lift-state :floor (:target-floor lift-state)
                 :state :waiting :wait-tics 105) ;; ~3 seconds
          (assoc lift-state :floor new-floor)))

      :waiting
      (let [wait (- (:wait-tics lift-state) tics)]
        (if (<= wait 0)
          (assoc lift-state :state :raising :wait-tics 0)
          (assoc lift-state :wait-tics wait)))

      :raising
      (let [new-floor (+ (:floor lift-state) speed)]
        (if (>= new-floor (:original-floor lift-state))
          (assoc lift-state :floor (:original-floor lift-state) :state :idle)
          (assoc lift-state :floor new-floor)))

      lift-state)))

;; ================================================================
;; Light flicker
;; ================================================================

(defn update-light
  "Update a flickering light. Returns updated state."
  [light-state ^double dt]
  (let [timer (- (:flicker-timer light-state) (* dt 35.0))]
    (if (<= timer 0)
      (-> light-state
          (update :light-state #(if (= % :on) :off :on))
          (assoc :flicker-timer (+ 2 (rand-int 20))))
      (assoc light-state :flicker-timer timer))))

;; ================================================================
;; Player interaction — use key (E key)
;; ================================================================

(defn try-use-line
  "Player tries to use/activate a linedef. Returns updated sector-states or nil."
  [map-data sector-states px py yaw player-state]
  (let [px (double px) py (double py) yaw (double yaw)
        linedefs (:linedefs map-data)
        vertexes (:vertexes map-data)
        sidedefs (:sidedefs map-data)
        ;; Use ray in player facing direction, 64 units
        _use-dx (* (Math/cos (- (/ Math/PI 2.0) yaw)) 64.0)
        _use-dy (* (Math/sin (- (/ Math/PI 2.0) yaw)) 64.0)]
    (reduce
      (fn [states linedef]
        (let [special (:special linedef)]
          (if (zero? special)
            states
            ;; Check if player is close enough to this linedef
            (let [[ax ay] (nth vertexes (:v1 linedef))
                  [bx by] (nth vertexes (:v2 linedef))
                  ax (double ax) ay (double ay) bx (double bx) by (double by)
                  ;; Simple proximity check
                  mid-x (* 0.5 (+ ax bx))
                  mid-y (* 0.5 (+ ay by))
                  dx (- px mid-x) dy (- py mid-y)
                  dist (Math/sqrt (+ (* dx dx) (* dy dy)))]
              (if (> dist 96.0) ;; too far
                states
                ;; Door activation
                (if (contains? door-specials special)
                  (let [bs-idx (:back-sidedef linedef)
                        door-si (when bs-idx (:sector (nth sidedefs bs-idx)))
                        door (when door-si (get states door-si))]
                    (if (and door (= (:state door) :closed))
                      ;; Check key requirement
                      (let [needs-key (case (int special)
                                        26 :blue, 28 :red, 27 :yellow
                                        32 :blue, 33 :red, 34 :yellow
                                        nil)]
                        (if (or (nil? needs-key) (contains? (:keys player-state) needs-key))
                          (assoc states door-si (assoc door :state :opening))
                          states))
                      states))
                  ;; Lift activation
                  (if (contains? lift-specials special)
                    (let [tag (:tag linedef)]
                      (reduce-kv
                        (fn [s si ss]
                          (if (and (= (:type ss) :lift)
                                   (= (:state ss) :idle)
                                   (= (:tag (nth (:sectors map-data) si)) tag))
                            (assoc s si (assoc ss :state :lowering))
                            s))
                        states states))
                    states)))))))
      sector-states
      linedefs)))

;; ================================================================
;; Master update
;; ================================================================

(defn update-sector-states
  "Update all sector states (doors, lifts, lights). Returns updated map."
  [sector-states ^double dt]
  (reduce-kv
    (fn [m k v]
      (assoc m k
             (case (:type v)
               :door (update-door v dt)
               :lift (update-lift v dt)
               :light (update-light v dt)
               :damage (let [timer (- (double (:damage-timer v)) (* dt 35.0))]
                         (if (<= timer 0)
                           (assoc v :damage-timer (double (:damage-interval v))
                                    :damage-fire? true)
                           (assoc v :damage-timer timer :damage-fire? false)))
               v)))
    {}
    sector-states))

(defn check-damage-floor
  "Check if player is standing on a damage floor. Returns damage amount or 0."
  [sector-states map-data px py]
  (let [px (double px) py (double py)
        sector-idx (bsp/point-sector map-data px py)]
    (if-let [ss (when sector-idx (get sector-states sector-idx))]
      (if (and (= (:type ss) :damage) (:damage-fire? ss))
        (:damage-amount ss)
        0)
      0)))

(defn get-sector-overrides
  "Get current floor/ceil heights for sectors with active doors/lifts.
  Returns {sector-idx {:floor-height :ceil-height}} for modified sectors."
  [sector-states _sectors]
  (reduce-kv
    (fn [m si ss]
      (case (:type ss)
        :door (assoc m si {:ceil-height (:ceil ss)
                           :floor-height (:floor ss)})
        :lift (assoc m si {:floor-height (:floor ss)})
        m))
    {}
    sector-states))

(defn get-light-overrides
  "Get current light levels for sectors with flickering lights.
  Returns {sector-idx light-level}."
  [sector-states]
  (reduce-kv
    (fn [m si ss]
      (if (= (:type ss) :light)
        (assoc m si (if (= (:light-state ss) :on)
                      (:original-light ss)
                      (:min-light ss)))
        m))
    {}
    sector-states))

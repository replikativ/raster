(ns doom.bsp
  "BSP tree queries for Doom maps: point-in-sector lookup, collision detection.
  Physics kernels use deftm for typed dispatch and bytecode compilation."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn deftm]]))

(set! *warn-on-reflection* true)

;; ================================================================
;; BSP point-sector lookup
;; ================================================================

(def ^:const NF_SUBSECTOR (int 0x8000))

(deftm point-side-cross
  "Cross product of BSP partition direction and point-to-start vector.
  Positive = right side, negative = left side."
  [px :- Double, py :- Double,
   nx :- Double, ny :- Double,
   dx :- Double, dy :- Double] :- Double
  (- (* (- px nx) dy) (* (- py ny) dx)))

(defn point-side
  "Determine which side of a BSP partition line a point is on.
  Returns :right or :left."
  [^double px ^double py node]
  (let [cross (point-side-cross px py
                (double (:x node)) (double (:y node))
                (double (:dx node)) (double (:dy node)))]
    (if (>= cross 0.0) :right :left)))

(defn point-subsector
  "Find which subsector a point (in Doom coordinates) belongs to.
  Returns subsector index."
  [nodes ^double px ^double py]
  (loop [node-idx (dec (count nodes))]
    (if (neg? node-idx)
      0 ;; fallback
      (let [nf (bit-and node-idx 0xFFFF)]
        (if (not= (bit-and nf NF_SUBSECTOR) 0)
          ;; It's a subsector
          (bit-and nf (bit-not NF_SUBSECTOR))
          ;; It's a node — traverse
          (let [node (nth nodes nf)
                side (point-side px py node)
                child (if (= side :right)
                        (:child-right node)
                        (:child-left node))]
            (if (not= (bit-and child NF_SUBSECTOR) 0)
              (bit-and child (bit-not NF_SUBSECTOR))
              (recur (int child)))))))))

(defn point-sector
  "Find which sector a point (in Doom coordinates) belongs to.
  Returns sector index."
  [map-data ^double px ^double py]
  (let [ssector-idx (point-subsector (:nodes map-data) px py)
        ssector (nth (:ssectors map-data) ssector-idx)
        first-seg (:first-seg ssector)
        seg (nth (:segs map-data) first-seg)
        linedef (nth (:linedefs map-data) (:linedef seg))
        sidedef-idx (if (zero? (:direction seg))
                      (:front-sidedef linedef)
                      (:back-sidedef linedef))]
    (when sidedef-idx
      (:sector (nth (:sidedefs map-data) sidedef-idx)))))

;; ================================================================
;; Collision detection — wall sliding
;; ================================================================

(deftm line-seg-dist
  "Minimum distance from point to line segment. Pure numerical kernel."
  [px :- Double, py :- Double,
   ax :- Double, ay :- Double,
   bx :- Double, by :- Double] :- Double
  (let [dx (- bx ax) dy (- by ay)
        len-sq (+ (* dx dx) (* dy dy))]
    (if (< len-sq 1.0e-10)
      (Math/sqrt (+ (* (- px ax) (- px ax)) (* (- py ay) (- py ay))))
      (let [t (max 0.0 (min 1.0 (/ (+ (* (- px ax) dx) (* (- py ay) dy)) len-sq)))
            cx (+ ax (* t dx))
            cy (+ ay (* t dy))
            ex (- px cx) ey (- py cy)]
        (Math/sqrt (+ (* ex ex) (* ey ey)))))))

(defn- line-segment-distance
  "Compute minimum distance from point (px,py) to line segment (ax,ay)-(bx,by).
  Returns [distance closest-nx closest-ny] where n is the outward normal."
  [px py ax ay bx by]
  (let [px (double px) py (double py) ax (double ax) ay (double ay) bx (double bx) by (double by)
        dist (line-seg-dist px py ax ay bx by)
        ;; Normal perpendicular to line
        dx (- bx ax) dy (- by ay)
        nx (- dy) ny dx
        nl (Math/sqrt (+ (* nx nx) (* ny ny)))
        nx (if (> nl 0) (/ nx nl) 0.0)
        ny (if (> nl 0) (/ ny nl) 0.0)]
    [dist nx ny]))

(defn move-and-slide
  "Try to move from pos by dir, sliding along blocking walls.
  pos and dir are [x y] in Doom coordinates. radius in Doom units.
  sector-overrides: {sector-idx {:floor-height :ceil-height}} for doors/lifts.
  Returns new [x y] position."
  [map-data pos dir radius
   & {:keys [sector-overrides]}]
  (let [px (double (nth pos 0)) py (double (nth pos 1))
        dx (double (nth dir 0)) dy (double (nth dir 1))
        radius (double radius)
        linedefs (:linedefs map-data)
        vertexes (:vertexes map-data)
        sidedefs (:sidedefs map-data)
        sectors (:sectors map-data)
        new-x (+ px dx)
        new-y (+ py dy)]
    ;; Check each blocking linedef
    (loop [rx new-x ry new-y
           remaining linedefs]
      (if (empty? remaining)
        [rx ry]
        (let [linedef (first remaining)
              v1-idx (:v1 linedef)
              v2-idx (:v2 linedef)
              [ax ay] (nth vertexes v1-idx)
              [bx by] (nth vertexes v2-idx)
              ax (double ax) ay (double ay)
              bx (double bx) by (double by)
              ;; Is this a blocking wall?
              blocking? (or
                          ;; One-sided = always blocking
                          (nil? (:back-sidedef linedef))
                          ;; Two-sided but impassable (flags bit 0 = blocks players,
                          ;; or step height > 24 Doom units, or door closed)
                          (let [fs (:front-sidedef linedef)
                                bs (:back-sidedef linedef)]
                            (when (and fs bs)
                              (let [front-si (:sector (nth sidedefs fs))
                                    back-si (:sector (nth sidedefs bs))
                                    front-sector (nth sectors front-si)
                                    back-sector (nth sectors back-si)
                                    ;; Use overrides for door/lift heights
                                    f-floor (if-let [o (when sector-overrides (get sector-overrides front-si))]
                                              (double (or (:floor-height o) (:floor-height front-sector)))
                                              (double (:floor-height front-sector)))
                                    b-floor (if-let [o (when sector-overrides (get sector-overrides back-si))]
                                              (double (or (:floor-height o) (:floor-height back-sector)))
                                              (double (:floor-height back-sector)))
                                    f-ceil (if-let [o (when sector-overrides (get sector-overrides front-si))]
                                             (double (or (:ceil-height o) (:ceil-height front-sector)))
                                             (double (:ceil-height front-sector)))
                                    b-ceil (if-let [o (when sector-overrides (get sector-overrides back-si))]
                                             (double (or (:ceil-height o) (:ceil-height back-sector)))
                                             (double (:ceil-height back-sector)))
                                    ;; Opening height (gap the player can walk through)
                                    opening-floor (max f-floor b-floor)
                                    opening-ceil (min f-ceil b-ceil)
                                    opening-height (- opening-ceil opening-floor)
                                    step-h (Math/abs (- b-floor f-floor))]
                                (or (not= 0 (bit-and (:flags linedef) 1))
                                    (> step-h 24)
                                    ;; Door closed: opening too small for player (56 units)
                                    (< opening-height 56.0))))))]
          (if blocking?
            (let [[dist _nx _ny] (line-segment-distance rx ry ax ay bx by)]
              (if (< dist radius)
                ;; Push out of wall
                (let [push (* (- radius dist) 1.01)
                      ;; Determine push direction (towards the point)
                      _pdx (- rx (+ ax (* 0.5 (- bx ax))))
                      _pdy (- ry (+ ay (* 0.5 (- by ay))))
                      ;; Use segment normal
                      seg-dx (- bx ax)
                      seg-dy (- by ay)
                      seg-nx (- seg-dy)
                      seg-ny seg-dx
                      seg-len (Math/sqrt (+ (* seg-nx seg-nx) (* seg-ny seg-ny)))
                      snx (if (> seg-len 0) (/ seg-nx seg-len) 0.0)
                      sny (if (> seg-len 0) (/ seg-ny seg-len) 0.0)
                      ;; Pick normal direction that faces the player
                      dot (+ (* snx (- rx ax)) (* sny (- ry ay)))
                      fnx (if (>= dot 0) snx (- snx))
                      fny (if (>= dot 0) sny (- sny))]
                  (recur (+ rx (* fnx push))
                         (+ ry (* fny push))
                         (rest remaining)))
                (recur rx ry (rest remaining))))
            (recur rx ry (rest remaining))))))))

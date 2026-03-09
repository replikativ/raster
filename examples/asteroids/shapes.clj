(ns asteroids.shapes
  "Geometric shape types for Asteroids — typed dispatch showcase.

  Each polygon type is a defvalue. Splitting, vertex generation, and scoring
  dispatch on concrete shape type via deftm. Add new polygon types from the
  REPL and the game picks them up on the next frame.

  Hierarchy: Octagon -> Squares, Hexagon -> Rhombi, Pentagon -> Quad+Triangle,
             Square -> Triangles, Triangle -> destroyed."
  (:require [raster.core :refer [deftm defvalue]])
  (:import [org.lwjgl.system MemoryUtil]))

;; ================================================================
;; Shape types — defvalue for typed dispatch
;; ================================================================

(defvalue Triangle [x :- Double, y :- Double,
                    vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double,
                    size :- Double])

(defvalue Square [x :- Double, y :- Double,
                  vx :- Double, vy :- Double,
                  angle :- Double, spin :- Double,
                  size :- Double])

(defvalue Pentagon [x :- Double, y :- Double,
                    vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double,
                    size :- Double])

(defvalue Hexagon [x :- Double, y :- Double,
                   vx :- Double, vy :- Double,
                   angle :- Double, spin :- Double,
                   size :- Double])

(defvalue Octagon [x :- Double, y :- Double,
                   vx :- Double, vy :- Double,
                   angle :- Double, spin :- Double,
                   size :- Double])

;; ================================================================
;; Side count — core property, dispatched
;; ================================================================

(deftm sides [s :- Triangle] :- Long 3)
(deftm sides [s :- Square]   :- Long 4)
(deftm sides [s :- Pentagon]  :- Long 5)
(deftm sides [s :- Hexagon]   :- Long 6)
(deftm sides [s :- Octagon]   :- Long 8)

;; ================================================================
;; Score value — more sides = more points
;; ================================================================

(deftm score-value [s :- Triangle] :- Long 50)
(deftm score-value [s :- Square]   :- Long 30)
(deftm score-value [s :- Pentagon]  :- Long 25)
(deftm score-value [s :- Hexagon]   :- Long 20)
(deftm score-value [s :- Octagon]   :- Long 15)

;; ================================================================
;; Splitting — the geometric decomposition hierarchy
;; ================================================================

(defn- scatter
  "Add random velocity scatter to a child shape's position/velocity."
  [parent-x parent-y parent-vx parent-vy child-size idx total]
  (let [spread-angle (* 2.0 Math/PI (/ (double idx) (double total)))
        scatter-v (+ 20.0 (rand 40.0))
        spin (- (rand 3.0) 1.5)]
    {:x (+ parent-x (* 8.0 (Math/cos spread-angle)))
     :y (+ parent-y (* 8.0 (Math/sin spread-angle)))
     :vx (+ parent-vx (* scatter-v (Math/cos spread-angle)))
     :vy (+ parent-vy (* scatter-v (Math/sin spread-angle)))
     :angle (rand (* 2.0 Math/PI))
     :spin spin
     :size child-size}))

(deftm split [s :- Octagon]
  (let [x (:x s) y (:y s) vx (:vx s) vy (:vy s)
        child-sz (* 0.7 (:size s))]
    (mapv (fn [i]
            (let [p (scatter x y vx vy child-sz i 2)]
              (->Square (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p))))
          (range 2))))

(deftm split [s :- Hexagon]
  (let [x (:x s) y (:y s) vx (:vx s) vy (:vy s)
        child-sz (* 0.65 (:size s))]
    (mapv (fn [i]
            (let [p (scatter x y vx vy child-sz i 3)]
              (->Pentagon (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p))))
          (range 3))))

(deftm split [s :- Pentagon]
  (let [x (:x s) y (:y s) vx (:vx s) vy (:vy s)
        child-sz (* 0.7 (:size s))]
    [(let [p (scatter x y vx vy child-sz 0 2)]
       (->Square (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p)))
     (let [p (scatter x y vx vy child-sz 1 2)]
       (->Triangle (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p)))]))

(deftm split [s :- Square]
  (let [x (:x s) y (:y s) vx (:vx s) vy (:vy s)
        child-sz (* 0.75 (:size s))]
    (mapv (fn [i]
            (let [p (scatter x y vx vy child-sz i 2)]
              (->Triangle (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p))))
          (range 2))))

;; Triangle is terminal — returns nil (destroyed)
(deftm split [s :- Triangle] nil)

;; ================================================================
;; Rendering properties — dispatched per shape type.
;; Extend with new shape types to customize visual size.
;; ================================================================

(deftm render-scale [s :- Triangle] :- Double 0.85)  ;; larger — terminal, needs visibility
(deftm render-scale [s :- Square]   :- Double 0.70)
(deftm render-scale [s :- Pentagon]  :- Double 0.70)
(deftm render-scale [s :- Hexagon]   :- Double 0.68)
(deftm render-scale [s :- Octagon]   :- Double 0.65)  ;; slightly smaller — many sides fill more

;; ================================================================
;; Texture generation — procedural polygon outline with glow
;; ================================================================

(defn- dist-to-segment
  "Minimum distance from point (px,py) to line segment (ax,ay)-(bx,by)."
  [px py ax ay bx by]
  (let [px (double px) py (double py) ax (double ax) ay (double ay)
        bx (double bx) by (double by)
        dx (- bx ax) dy (- by ay)
        len2 (+ (* dx dx) (* dy dy))
        t (if (zero? len2) 0.0
              (max 0.0 (min 1.0 (/ (+ (* (- px ax) dx) (* (- py ay) dy)) len2))))
        cx (+ ax (* t dx)) cy (+ ay (* t dy))]
    (Math/sqrt (+ (* (- px cx) (- px cx)) (* (- py cy) (- py cy))))))

;; Matplotlib tab10 palette — each polygon type gets a distinct color
(def palette
  {3 [0.580 0.404 0.741]  ;; triangle — purple #9467bd
   4 [0.839 0.153 0.157]  ;; square — red #d62728
   5 [0.173 0.627 0.173]  ;; pentagon — green #2ca02c
   6 [1.000 0.498 0.055]  ;; hexagon — orange #ff7f0e
   8 [0.122 0.467 0.706]  ;; octagon — blue #1f77b4
   ;; Default for user-added types
   7 [0.498 0.498 0.498]  ;; heptagon — grey
   9 [0.737 0.741 0.133]  ;; nonagon — olive
   10 [0.090 0.745 0.812]}) ;; decagon — cyan

(defn generate-polygon-texture
  "Generate a sz x sz RGBA ByteBuffer with a colored N-sided polygon outline.
  Crisp lines with small vertex dots, matplotlib-style plot aesthetic.
  Returns {:pixels ByteBuffer :width sz :height sz}."
  [n-sides sz]
  (let [sz (int sz)
        buf (MemoryUtil/memAlloc (int (* sz sz 4)))
        cx (/ (double sz) 2.0)
        cy (/ (double sz) 2.0)
        r (* cx 0.88)
        [cr cg cb] (get palette n-sides [0.8 0.8 0.8])
        cr (double cr) cg (double cg) cb (double cb)
        lw 1.4  ;; crisp thin line
        dot-r 2.5  ;; vertex dot radius
        n (int n-sides)
        verts (double-array (* 2 n))]
    (dotimes [i n]
      (let [a (- (* 2.0 Math/PI (/ (double i) (double n))) (/ Math/PI 2.0))]
        (aset verts (* 2 i) (+ cx (* r (Math/cos a))))
        (aset verts (+ (* 2 i) 1) (+ cy (* r (Math/sin a))))))
    (dotimes [py sz]
      (dotimes [px sz]
        (let [fx (+ (double px) 0.5)
              fy (+ (double py) 0.5)
              ;; Distance to closest edge
              d (loop [i (int 0) min-d Double/MAX_VALUE]
                  (if (< i n)
                    (let [j (mod (inc i) n)
                          ax (aget verts (* 2 i)) ay (aget verts (+ (* 2 i) 1))
                          bx (aget verts (* 2 j)) by (aget verts (+ (* 2 j) 1))
                          d (dist-to-segment fx fy ax ay bx by)]
                      (recur (inc i) (min min-d d)))
                    min-d))
              ;; Distance to closest vertex
              dv (loop [i (int 0) min-d Double/MAX_VALUE]
                   (if (< i n)
                     (let [vx (aget verts (* 2 i)) vy (aget verts (+ (* 2 i) 1))
                           dx (- fx vx) dy (- fy vy)]
                       (recur (inc i) (min min-d (Math/sqrt (+ (* dx dx) (* dy dy))))))
                     min-d))
              ;; Crisp edge: anti-aliased step function
              edge (max 0.0 (min 1.0 (- 1.0 (/ (max 0.0 (- d lw)) 0.8))))
              ;; Vertex dot: filled circle at each corner
              dot (max 0.0 (min 1.0 (- 1.0 (/ (max 0.0 (- dv dot-r)) 0.8))))
              ;; Combine — edge or dot, whichever is stronger
              alpha (min 1.0 (max edge dot))]
          (.put buf (unchecked-byte (int (* cr alpha 255.0))))
          (.put buf (unchecked-byte (int (* cg alpha 255.0))))
          (.put buf (unchecked-byte (int (* cb alpha 255.0))))
          (.put buf (unchecked-byte (int (* alpha 255.0)))))))
    (.flip buf)
    {:pixels buf :width sz :height sz}))

(defn generate-ship-texture
  "Generate the ship triangle texture — light grey, pointing right (+X)."
  [sz]
  (let [sz (int sz)
        buf (MemoryUtil/memAlloc (int (* sz sz 4)))
        cx (/ (double sz) 2.0)
        cy (/ (double sz) 2.0)
        r (* cx 0.88)
        lw 1.4
        dot-r 2.5
        ;; Ship color: light grey
        cr 0.8 cg 0.8 cb 0.8
        ;; Triangle pointing right
        nose-x (+ cx r) nose-y cy
        wing-a1 (* (/ 150.0 180.0) Math/PI)
        wing-a2 (* (/ 210.0 180.0) Math/PI)
        w1x (+ cx (* r (Math/cos wing-a1))) w1y (+ cy (* r (Math/sin wing-a1)))
        w2x (+ cx (* r (Math/cos wing-a2))) w2y (+ cy (* r (Math/sin wing-a2)))]
    (dotimes [py sz]
      (dotimes [px sz]
        (let [fx (+ (double px) 0.5)
              fy (+ (double py) 0.5)
              d1 (dist-to-segment fx fy nose-x nose-y w1x w1y)
              d2 (dist-to-segment fx fy w1x w1y w2x w2y)
              d3 (dist-to-segment fx fy w2x w2y nose-x nose-y)
              d (min d1 (min d2 d3))
              dn (Math/sqrt (+ (* (- fx nose-x) (- fx nose-x)) (* (- fy nose-y) (- fy nose-y))))
              dw1 (Math/sqrt (+ (* (- fx w1x) (- fx w1x)) (* (- fy w1y) (- fy w1y))))
              dw2 (Math/sqrt (+ (* (- fx w2x) (- fx w2x)) (* (- fy w2y) (- fy w2y))))
              dv (min dn (min dw1 dw2))
              edge (max 0.0 (min 1.0 (- 1.0 (/ (max 0.0 (- d lw)) 0.8))))
              dot (max 0.0 (min 1.0 (- 1.0 (/ (max 0.0 (- dv dot-r)) 0.8))))
              alpha (min 1.0 (max edge dot))]
          (.put buf (unchecked-byte (int (* cr alpha 255.0))))
          (.put buf (unchecked-byte (int (* cg alpha 255.0))))
          (.put buf (unchecked-byte (int (* cb alpha 255.0))))
          (.put buf (unchecked-byte (int (* alpha 255.0)))))))
    (.flip buf)
    {:pixels buf :width sz :height sz}))

;; ================================================================
;; Shape constructors — spawn with random velocity
;; ================================================================

(def shape-ctors
  "Registry of shape constructors by keyword. Extend from the REPL!"
  (atom {:triangle ->Triangle
         :square   ->Square
         :pentagon ->Pentagon
         :hexagon  ->Hexagon
         :octagon  ->Octagon}))

(def spawn-pool
  "Which shapes to spawn as initial asteroids. Extend from the REPL!"
  (atom [:hexagon :octagon :pentagon :hexagon :octagon]))

(defn random-shape
  "Spawn a random asteroid shape at given size.
  Avoids the center 300x300 zone (ship spawn)."
  [^double size]
  (let [pool @spawn-pool
        kind (nth pool (rand-int (count pool)))
        ctor (get @shape-ctors kind)
        angle (rand (* 2.0 Math/PI))
        speed (+ 25.0 (rand 45.0))
        ;; Random position, reject center zone, spread across full area
        [x y] (loop []
                (let [x (rand 1024.0) y (rand 768.0)
                      dx (- x 512.0) dy (- y 384.0)]
                  (if (and (< (Math/abs dx) 250.0) (< (Math/abs dy) 200.0))
                    (recur) ;; too close to center
                    [x y])))]
    (ctor x y
          (* speed (Math/cos angle))
          (* speed (Math/sin angle))
          (rand (* 2.0 Math/PI))
          (- (rand 2.0) 1.0)
          size)))

;; ================================================================
;; Shape accessors — generic across all types
;; ================================================================

(defn shape-x ^double [s] (double (:x s)))
(defn shape-y ^double [s] (double (:y s)))
(defn shape-vx ^double [s] (double (:vx s)))
(defn shape-vy ^double [s] (double (:vy s)))
(defn shape-angle ^double [s] (double (:angle s)))
(defn shape-spin ^double [s] (double (:spin s)))
(defn shape-size ^double [s] (double (:size s)))

;; ================================================================
;; Movement — deftm per shape type
;; Eliminates constructor lookup; each type reconstructs directly.
;; Extend with new shape types by adding a deftm move-shape clause.
;; ================================================================

(defn- wrap ^double [^double v ^double max-v]
  (let [v (rem v max-v)] (if (neg? v) (+ v max-v) v)))

(deftm move-shape [s :- Triangle, dt :- Double, w :- Double, h :- Double]
  (->Triangle (wrap (+ (:x s) (* (:vx s) dt)) w)
              (wrap (+ (:y s) (* (:vy s) dt)) h)
              (:vx s) (:vy s)
              (+ (:angle s) (* (:spin s) dt))
              (:spin s) (:size s)))

(deftm move-shape [s :- Square, dt :- Double, w :- Double, h :- Double]
  (->Square (wrap (+ (:x s) (* (:vx s) dt)) w)
            (wrap (+ (:y s) (* (:vy s) dt)) h)
            (:vx s) (:vy s)
            (+ (:angle s) (* (:spin s) dt))
            (:spin s) (:size s)))

(deftm move-shape [s :- Pentagon, dt :- Double, w :- Double, h :- Double]
  (->Pentagon (wrap (+ (:x s) (* (:vx s) dt)) w)
              (wrap (+ (:y s) (* (:vy s) dt)) h)
              (:vx s) (:vy s)
              (+ (:angle s) (* (:spin s) dt))
              (:spin s) (:size s)))

(deftm move-shape [s :- Hexagon, dt :- Double, w :- Double, h :- Double]
  (->Hexagon (wrap (+ (:x s) (* (:vx s) dt)) w)
             (wrap (+ (:y s) (* (:vy s) dt)) h)
             (:vx s) (:vy s)
             (+ (:angle s) (* (:spin s) dt))
             (:spin s) (:size s)))

(deftm move-shape [s :- Octagon, dt :- Double, w :- Double, h :- Double]
  (->Octagon (wrap (+ (:x s) (* (:vx s) dt)) w)
             (wrap (+ (:y s) (* (:vy s) dt)) h)
             (:vx s) (:vy s)
             (+ (:angle s) (* (:spin s) dt))
             (:spin s) (:size s)))

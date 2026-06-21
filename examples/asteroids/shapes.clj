(ns asteroids.shapes
  "Typed-dispatch shape hierarchy for Geometric Asteroids — the JVM half of the
   cross-platform game (asteroids.game). Each polygon is a `defvalue`; `sides`,
   `score-value`, `split` and `move-shape` dispatch on the concrete type via `deftm`.

   This is the live-dispatch pedagogy: define a new `defvalue` + its `deftm` methods
   at the REPL, register a constructor, add it to the spawn pool, and the running game
   picks it up on the next wave — no recompile. (The browser build can't do this — cljs
   has no `deftm` runtime — so asteroids.game keeps an equivalent data form behind a
   reader conditional; see its `split-rule`/`shape-score`.)

   Hierarchy: Octagon → Squares · Hexagon → Pentagons · Pentagon → Square+Triangle ·
              Square → Triangles · Triangle → destroyed."
  (:require [raster.core :refer [deftm defvalue]]))

;; --- shape types — a defvalue per polygon (the dispatch key + its kinematics) ----
(defvalue Triangle [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])
(defvalue Square   [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])
(defvalue Pentagon [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])
(defvalue Hexagon  [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])
(defvalue Octagon  [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])

;; --- side count (dispatched) ----------------------------------------------------
(deftm sides [s :- Triangle] :- Long 3)
(deftm sides [s :- Square]   :- Long 4)
(deftm sides [s :- Pentagon] :- Long 5)
(deftm sides [s :- Hexagon]  :- Long 6)
(deftm sides [s :- Octagon]  :- Long 8)

;; --- score (more sides bigger/easier → fewer points) ----------------------------
(deftm score-value [s :- Triangle] :- Long 50)
(deftm score-value [s :- Square]   :- Long 30)
(deftm score-value [s :- Pentagon] :- Long 25)
(deftm score-value [s :- Hexagon]  :- Long 20)
(deftm score-value [s :- Octagon]  :- Long 15)

;; --- splitting — the geometric decomposition (dispatched) -----------------------
(defn- scatter
  "Random position/velocity scatter for child `idx` of `total` around the parent."
  [px py pvx pvy child-size idx total]
  (let [a (* 2.0 Math/PI (/ (double idx) (double total)))
        v (+ 20.0 (rand 40.0))]
    {:x (+ px (* 8.0 (Math/cos a))) :y (+ py (* 8.0 (Math/sin a)))
     :vx (+ pvx (* v (Math/cos a))) :vy (+ pvy (* v (Math/sin a)))
     :angle (rand (* 2.0 Math/PI)) :spin (- (rand 3.0) 1.5) :size child-size}))

(defn- child [ctor p] (ctor (:x p) (:y p) (:vx p) (:vy p) (:angle p) (:spin p) (:size p)))

(deftm split [s :- Octagon]
  (let [cz (* 0.70 (:size s))]
    (mapv #(child ->Square (scatter (:x s) (:y s) (:vx s) (:vy s) cz % 2)) (range 2))))
(deftm split [s :- Hexagon]
  (let [cz (* 0.65 (:size s))]
    (mapv #(child ->Pentagon (scatter (:x s) (:y s) (:vx s) (:vy s) cz % 3)) (range 3))))
(deftm split [s :- Pentagon]
  (let [cz (* 0.70 (:size s))]
    [(child ->Square   (scatter (:x s) (:y s) (:vx s) (:vy s) cz 0 2))
     (child ->Triangle (scatter (:x s) (:y s) (:vx s) (:vy s) cz 1 2))]))
(deftm split [s :- Square]
  (let [cz (* 0.75 (:size s))]
    (mapv #(child ->Triangle (scatter (:x s) (:y s) (:vx s) (:vy s) cz % 2)) (range 2))))
(deftm split [s :- Triangle] nil)   ; terminal

;; --- movement (dispatched; integrate position + angle, toroidal wrap) -----------
(defn- wrap ^double [^double v ^double m] (let [v (rem v m)] (if (neg? v) (+ v m) v)))

(deftm move-shape [s :- Triangle, dt :- Double, w :- Double, h :- Double]
  (->Triangle (wrap (+ (:x s) (* (:vx s) dt)) w) (wrap (+ (:y s) (* (:vy s) dt)) h)
              (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))
(deftm move-shape [s :- Square, dt :- Double, w :- Double, h :- Double]
  (->Square (wrap (+ (:x s) (* (:vx s) dt)) w) (wrap (+ (:y s) (* (:vy s) dt)) h)
            (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))
(deftm move-shape [s :- Pentagon, dt :- Double, w :- Double, h :- Double]
  (->Pentagon (wrap (+ (:x s) (* (:vx s) dt)) w) (wrap (+ (:y s) (* (:vy s) dt)) h)
              (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))
(deftm move-shape [s :- Hexagon, dt :- Double, w :- Double, h :- Double]
  (->Hexagon (wrap (+ (:x s) (* (:vx s) dt)) w) (wrap (+ (:y s) (* (:vy s) dt)) h)
             (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))
(deftm move-shape [s :- Octagon, dt :- Double, w :- Double, h :- Double]
  (->Octagon (wrap (+ (:x s) (* (:vx s) dt)) w) (wrap (+ (:y s) (* (:vy s) dt)) h)
             (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))

;; --- spawning — registries are atoms so the REPL can extend them -----------------
(def shape-ctors
  "sides → constructor. Extend at the REPL: (swap! shapes/shape-ctors assoc 7 ->Heptagon)."
  (atom {3 ->Triangle 4 ->Square 5 ->Pentagon 6 ->Hexagon 8 ->Octagon}))

(def spawn-pool
  "Which side counts spawn as fresh asteroids. Extend at the REPL: (swap! shapes/spawn-pool conj 7)."
  (atom [6 8 5 6 8]))

(defn random-shape
  "Spawn a random asteroid of the given `size`, avoiding the centre (ship) zone."
  [^double size]
  (let [pool @spawn-pool
        ctor (@shape-ctors (nth pool (rand-int (count pool))))
        a (rand (* 2.0 Math/PI)) sp (+ 25.0 (rand 45.0))
        [x y] (loop []
                (let [x (rand 1024.0) y (rand 768.0)]
                  (if (and (< (Math/abs (- x 512.0)) 250.0) (< (Math/abs (- y 384.0)) 200.0))
                    (recur) [x y])))]
    (ctor x y (* sp (Math/cos a)) (* sp (Math/sin a)) (rand (* 2.0 Math/PI)) (- (rand 2.0) 1.0) size)))

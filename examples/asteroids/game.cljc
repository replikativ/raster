(ns asteroids.game
  "Cross-platform Geometric Asteroids — game logic written once (.cljc), run on the
   JVM and in the browser. Pure/immutable: (update-game state input dt) → state'; the
   platform shell (Vulkan/Swing/WebGPU + input + loop) lives per-platform and reads
   `render-data`.

   The shape hierarchy is the one place the platforms genuinely differ, behind a tiny
   `sh-*` seam:
     • JVM  — a shape IS an asteroids.shapes `defvalue` (Triangle…Octagon); `sides`,
              `score`, `split` and `move` dispatch on its type via `deftm`. Define a new
              `defvalue` + its `deftm` methods at the REPL and the running game uses it —
              live multiple-dispatch, no recompile.
     • cljs — `deftm`/`defvalue` don't exist in ClojureScript, so a shape is a plain
              {:sides n :kin Shape} map and the same hierarchy is the data form below
              (`split-rule`/`data-score`). `kin` is the kinematic value type that crosses
              to wasm (asteroids.kernels/move-shape, compiled to WebAssembly).
   The loop, collision, scoring, ship, bullets, spawn cadence and render-data are shared."
  (:require [asteroids.kernels :as k]
            #?(:clj [asteroids.shapes :as sd])))

(def ^:const W 1024.0)
(def ^:const H 768.0)
(def ^:const TAU (* 2.0 #?(:clj Math/PI :cljs js/Math.PI)))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- rnd ^double [] (rand))

;; --- cljs data form of the hierarchy (the twin of asteroids.shapes' deftm methods) ---
(def split-rule {8 {:kids [4 4]   :scale 0.70}
                 6 {:kids [5 5 5] :scale 0.65}
                 5 {:kids [4 3]   :scale 0.70}
                 4 {:kids [3 3]   :scale 0.75}
                 3 nil})
(defn- data-score [sides] (case sides 3 50 4 30 5 25 6 20 8 15 10))

(defn- ->shape [sides x y vx vy size]
  {:sides sides :kin (k/->Shape x y vx vy (rnd) (- (rnd) 0.5) size)})

(defn- data-split [{:keys [sides kin]}]
  (when-let [{:keys [kids scale]} (split-rule sides)]
    (let [cz (* scale (:size kin)) n (count kids)]
      (map-indexed
       (fn [i ks]
         (let [a (/ (* TAU i) n) sv (+ 20.0 (* (rnd) 40.0))]
           (->shape ks (+ (:x kin) (* 8.0 (cos a))) (+ (:y kin) (* 8.0 (sin a)))
                    (+ (:vx kin) (* sv (cos a))) (+ (:vy kin) (* sv (sin a))) cz)))
       kids))))

;; --- shape seam: bridges the JVM defvalue and the cljs {:sides :kin} map ------------
(defn- sh-x     ^double [s] #?(:clj (:x s)     :cljs (:x (:kin s))))
(defn- sh-y     ^double [s] #?(:clj (:y s)     :cljs (:y (:kin s))))
(defn- sh-angle ^double [s] #?(:clj (:angle s) :cljs (:angle (:kin s))))
(defn- sh-size  ^double [s] #?(:clj (:size s)  :cljs (:size (:kin s))))
(defn- sh-sides ^long   [s] #?(:clj (sd/sides s) :cljs (:sides s)))
(defn- sh-score ^long   [s] #?(:clj (sd/score-value s) :cljs (data-score (:sides s))))
(defn- sh-move  [s dt w h] #?(:clj (sd/move-shape s dt w h)
                              :cljs (update s :kin #(k/move-shape % dt w h))))
(defn- sh-split [s] #?(:clj (sd/split s) :cljs (data-split s)))   ; → child shapes or nil

(defn- spawn-asteroid []
  #?(:clj (sd/random-shape 48.0)
     :cljs (let [sides (nth [6 8 5 6 8] (int (* (rnd) 5)))
                 a (* (rnd) TAU) sp (+ 25.0 (* (rnd) 45.0))]
             (loop []
               (let [x (* (rnd) W) y (* (rnd) H)]
                 (if (and (< (Math/abs (- x (/ W 2.0))) 250.0) (< (Math/abs (- y (/ H 2.0))) 200.0))
                   (recur)
                   (->shape sides x y (* sp (cos a)) (* sp (sin a)) 48.0)))))))

(defn spawn-wave [wave]
  (vec (repeatedly (+ 4 wave) spawn-asteroid)))

;; --- state --------------------------------------------------------------------
(defn init-state []
  {:ship {:x (/ W 2.0) :y (/ H 2.0) :vx 0.0 :vy 0.0 :angle (- (/ TAU 4.0))}
   :asteroids (spawn-wave 1) :bullets [] :score 0 :lives 3 :wave 1 :over false :cooldown 0.0})

(def ^:const SHIP-ACCEL 300.0)
(def ^:const SHIP-TURN 4.0)
(def ^:const BULLET-SPEED 500.0)

(defn- step-ship [{:keys [x y vx vy angle] :as ship} input dt]
  (let [angle (cond-> angle (:left input) (- (* SHIP-TURN dt)) (:right input) (+ (* SHIP-TURN dt)))
        vx (if (:up input) (+ vx (* SHIP-ACCEL (cos angle) dt)) vx)
        vy (if (:up input) (+ vy (* SHIP-ACCEL (sin angle) dt)) vy)
        vx (* vx 0.99) vy (* vy 0.99)]
    {:x (mod (+ x (* vx dt)) W) :y (mod (+ y (* vy dt)) H) :vx vx :vy vy :angle angle}))

(defn- circle-hit? [ax ay bx by r] (let [dx (- ax bx) dy (- ay by)] (< (+ (* dx dx) (* dy dy)) (* r r))))

(defn update-game
  "Advance one frame. input = #{:left :right :up :fire} (a set or map of flags)."
  [{:keys [ship asteroids bullets score lives wave over cooldown] :as state} input dt]
  (if over
    (if (:fire input) (assoc (init-state) :over false) state)
    (let [w (double W) h (double H)
          ;; physics: move every shape (deftm dispatch on JVM / wasm kernel on cljs)
          asteroids (mapv #(sh-move % dt w h) asteroids)
          ship (step-ship ship input dt)
          cooldown (max 0.0 (- cooldown dt))
          ;; fire
          [bullets cooldown] (if (and (:fire input) (<= cooldown 0.0))
                               [(conj bullets {:x (+ (:x ship) (* 20.0 (cos (:angle ship))))
                                               :y (+ (:y ship) (* 20.0 (sin (:angle ship))))
                                               :vx (+ (:vx ship) (* BULLET-SPEED (cos (:angle ship))))
                                               :vy (+ (:vy ship) (* BULLET-SPEED (sin (:angle ship)))) :ttl 1.5})
                                0.15]
                               [bullets cooldown])
          ;; move + cull bullets
          bullets (->> bullets
                       (map (fn [b] (-> b (update :x + (* (:vx b) dt)) (update :y + (* (:vy b) dt))
                                        (update :ttl - dt))))
                       (filterv (fn [b] (and (pos? (:ttl b)) (<= 0 (:x b) w) (<= 0 (:y b) h)))))
          ;; collisions: bullet vs shape → split (sh-split → children or nil)
          [asteroids bullets score]
          (loop [as asteroids, out [], sc score, dead #{}]
            (if (empty? as)
              [out (vec (remove dead bullets)) sc]
              (let [a (first as)
                    hit (first (filter #(and (not (dead %)) (circle-hit? (:x %) (:y %) (sh-x a) (sh-y a) (sh-size a))) bullets))]
                (if hit
                  (recur (rest as) (into out (sh-split a)) (+ sc (sh-score a)) (conj dead hit))
                  (recur (rest as) (conj out a) sc dead)))))
          ;; ship vs shape → lose a life
          ship-hit (some (fn [a] (circle-hit? (:x ship) (:y ship) (sh-x a) (sh-y a) (+ 12.0 (sh-size a)))) asteroids)
          lives (if ship-hit (dec lives) lives)
          ship (if ship-hit (assoc ship :x (/ W 2.0) :y (/ H 2.0) :vx 0.0 :vy 0.0) ship)
          cleared? (empty? asteroids)
          wave (if cleared? (inc wave) wave)
          asteroids (if cleared? (spawn-wave wave) asteroids)]
      {:ship ship :asteroids asteroids :bullets bullets :score score
       :lives lives :wave wave :cooldown cooldown :over (<= lives 0)})))

(defn render-data
  "Platform-agnostic draw list: {:ship {x y angle} :polys [{:x :y :angle :size :sides}] :bullets [{:x :y}] :hud …}."
  [{:keys [ship asteroids bullets score lives wave over]}]
  {:ship (select-keys ship [:x :y :angle])
   :polys (mapv (fn [a] {:x (sh-x a) :y (sh-y a) :angle (sh-angle a) :size (sh-size a) :sides (sh-sides a)}) asteroids)
   :bullets (mapv #(select-keys % [:x :y]) bullets)
   :hud {:score score :lives lives :wave wave :over over}})

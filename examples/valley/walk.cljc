(ns valley.walk
  "Portable single-chunk walker (Phase 2b-A): a 16³ block chunk (valley.chunk) you
   can walk on, with gravity + AABB collision driven by valley.kernels/integrate-physics!
   (deftm→bytecode on the JVM, wasm in the browser — same kernel, same trajectory).

   Camera, shader, atlas and draw path are reused from valley.slice; this ns adds
   the chunk mesh, the player, and the keyboard control loop. The player position and
   velocity live in double-arrays so the physics kernel mutates them in place — the
   cljs facade marshals those arrays through wasm memory (see valley/gen.clj)."
  (:require [valley.slice :as slice]
            [valley.chunk :as chunk]
            [valley.kernels :as k]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- ffloor [x] (long (#?(:clj Math/floor :cljs js/Math.floor) (double x))))
(defn- fabs [x] (#?(:clj Math/abs :cljs js/Math.abs) (double x)))

(def ^:const EYE 1.6)     ; eye height above feet
(def ^:const HALF-W 0.3)  ; player AABB half-width
(def ^:const HEIGHT 1.7)  ; player AABB height
(def ^:const MOVE 5.0)    ; blocks / second
(def ^:const TURN 1.7)    ; radians / second

(defn build-world
  "The single chunk + its derived render mesh + collision tables. Returns everything
   the shells need to mesh, render and simulate against one 16³ chunk."
  []
  (let [blocks (chunk/gen-chunk)
        solid  (chunk/solid-table)
        m      (chunk/mesh-chunk blocks)]
    {:blocks blocks :solid solid
     :verts (:verts m) :indices (:indices m)}))

(defn player-init
  "Spawn the player above the centre column so the first frames are a short fall
   onto the surface — a visible proof the physics kernel runs identically."
  [blocks]
  (let [cx 8 cz 8
        top (loop [y (dec chunk/CS)]
              (cond (< y 0) 0
                    (pos? (chunk/block-at blocks cx y cz)) (inc y)
                    :else (recur (dec y))))]
    {:pos (chunk/darray [(+ cx 0.5) (+ (double top) 3.0) (+ cz 0.5)])
     :vel (chunk/darray [0.0 0.0 0.0])
     :yaw 0.0 :pitch -0.2 :on-ground false}))

(defn player-update
  "Arrows look, WASD move (horizontal, yaw-relative). Gravity + collision come from
   the physics kernel; pos/vel are mutated in place."
  [world player input dt]
  (let [{:keys [pos vel yaw pitch]} player
        on?   (fn [a] (if (contains? input a) 1.0 0.0))
        yaw   (+ yaw (* TURN dt (- (on? :left) (on? :right))))
        pitch (-> (+ pitch (* TURN dt (- (on? :up) (on? :down)))) (max -1.5) (min 1.5))
        ;; XZ basis from yaw (matches slice/forward-vec's XZ components)
        fx (sin yaw)  fz (- (cos yaw))
        rx (cos yaw)  rz (sin yaw)
        fwd (* MOVE dt (- (on? :w) (on? :s)))
        str (* MOVE dt (- (on? :d) (on? :a)))
        dx (+ (* fx fwd) (* rx str))
        dz (+ (* fz fwd) (* rz str))
        g  (k/integrate-physics! pos vel (:blocks world) (:solid world)
                                 0 0 0 HALF-W HEIGHT dx dz dt)]
    (assoc player :yaw yaw :pitch pitch :on-ground (= 1 g))))

;; --- multi-chunk world walker (option C) -------------------------------------
;; Same kernel, same marshaling; the only difference from single-chunk is that each
;; frame we stitch a player-centred 16³ window from the world (chunk/stitch-window!)
;; and call integrate-physics! against it at origin 0, translating the player position
;; into and out of window-local coords around the call.

(defn build-grid
  "A cw×ch×cd-chunk world + its full render mesh + a reusable 16³ stitch scratch.
   Real biome terrain (valley.kernels/surface-height-biome) — walkable because
   integrate-physics! does 1-block step-up. gen-flat-world remains for a flat test plane."
  [cw ch cd]
  (let [world (chunk/gen-world cw ch cd)
        m     (chunk/mesh-world world)]
    (assoc world :verts (:verts m) :indices (:indices m)
           :scratch (chunk/iarray (* chunk/CS chunk/CS chunk/CS)))))

(defn grid-player-init
  "Spawn above the world centre column for a short visible fall onto the terrain."
  [world]
  (let [cx (quot (long (:wx world)) 2)
        cz (quot (long (:wz world)) 2)
        top (loop [y (dec (long (:wy world)))]
              (cond (< y 0) 0
                    (pos? (chunk/world-block world cx y cz)) (inc y)
                    :else (recur (dec y))))]
    {:pos (chunk/darray [(+ cx 0.5) (+ (double top) 3.0) (+ cz 0.5)])
     :vel (chunk/darray [0.0 0.0 0.0])
     :spawn [(+ cx 0.5) (+ (double top) 3.0) (+ cz 0.5)]   ; respawn point
     :yaw 0.0 :pitch -0.2 :on-ground false
     ;; survival state: health (0..max), hotbar block ids + selected index, Q/E latch
     :health 20 :max-health 20 :hotbar [1 2 3 8 11 12] :sel 0 :qe false}))

(def ^:const VOID-Y -10.0)    ; below this → instant death (fell off the world)
(def ^:const FALL-SAFE 13.0)  ; impact speed (blocks/s) you can survive unharmed
(def ^:const FALL-SCALE 1.6)  ; health lost per (blocks/s) above FALL-SAFE

(def ^:const REACH 6.0)   ; block interaction distance

(defn- forward-vec [yaw pitch]
  (let [cp (cos pitch)] [(* (sin yaw) cp) (sin pitch) (* (- (cos yaw)) cp)]))

(defn raycast
  "Amanatides–Woo voxel DDA from (ex,ey,ez) along (dx,dy,dz), up to `maxd` blocks. Returns
   {:hit [x y z] :place [x y z]} for the first opaque block + the empty cell just before it
   (placement target), or nil. Portable (no host types beyond Math floor/abs)."
  [world ex ey ez dx dy dz maxd]
  (let [big 1.0e9
        sx (if (>= dx 0) 1 -1) sy (if (>= dy 0) 1 -1) sz (if (>= dz 0) 1 -1)
        tdx (if (zero? dx) big (fabs (/ 1.0 dx)))
        tdy (if (zero? dy) big (fabs (/ 1.0 dy)))
        tdz (if (zero? dz) big (fabs (/ 1.0 dz)))
        ix (ffloor ex) iy (ffloor ey) iz (ffloor ez)
        bnd (fn [i e d s] (if (zero? d) big (/ (- (+ i (if (>= d 0) 1 0)) e) d)))]
    (loop [vx ix vy iy vz iz
           tmx (bnd ix ex dx sx) tmy (bnd iy ey dy sy) tmz (bnd iz ez dz sz)
           px ix py iy pz iz]
      (cond
        (chunk/opaque? (chunk/world-block world vx vy vz)) {:hit [vx vy vz] :place [px py pz]}
        (> (min tmx tmy tmz) maxd) nil
        (and (<= tmx tmy) (<= tmx tmz)) (recur (+ vx sx) vy vz (+ tmx tdx) tmy tmz vx vy vz)
        (<= tmy tmz)                    (recur vx (+ vy sy) vz tmx (+ tmy tdy) tmz vx vy vz)
        :else                           (recur vx vy (+ vz sz) tmx tmy (+ tmz tdz) vx vy vz)))))

(defn apply-edits!
  "Break (mouse left) / place selected block (mouse right) under the crosshair, in place.
   Returns the changed block coord [x y z] (→ shell re-meshes those columns), or nil. Reads
   this frame's edge-triggered mouse buttons from input metadata (set by both backends)."
  [world player input]
  (let [b (:buttons (meta input))]
    (when (and b (or (contains? b :left) (contains? b :right)))
      (let [p (:pos player)
            [dx dy dz] (forward-vec (:yaw player) (:pitch player))
            r (raycast world (aget p 0) (+ (aget p 1) EYE) (aget p 2) dx dy dz REACH)]
        (when r
          (if (contains? b :left)
            (let [[hx hy hz] (:hit r)] (chunk/set-block! world hx hy hz 0) [hx hy hz])
            (let [[qx qy qz] (:place r)
                  sel (nth (:hotbar player) (:sel player) chunk/LOG)]
              (chunk/set-block! world qx qy qz sel) [qx qy qz])))))))

(defn grid-player-update
  "Player physics against the multi-chunk world via a player-centred window."
  [world player input dt]
  (let [{:keys [pos vel yaw pitch]} player
        on?   (fn [a] (if (contains? input a) 1.0 0.0))
        [mdx mdy] (or (:mouse (meta input)) [0.0 0.0])   ; pointer-lock delta (px)
        sens  0.0025
        yaw   (+ yaw (* TURN dt (- (on? :left) (on? :right))) (* (- sens) mdx))
        pitch (-> (+ pitch (* TURN dt (- (on? :up) (on? :down))) (* (- sens) mdy)) (max -1.5) (min 1.5))
        fx (sin yaw)  fz (- (cos yaw))
        rx (cos yaw)  rz (sin yaw)
        fwd (* MOVE dt (- (on? :w) (on? :s)))
        strf (* MOVE dt (- (on? :d) (on? :a)))
        dx (+ (* fx fwd) (* rx strf))
        dz (+ (* fz fwd) (* rz strf))
        ;; window origin: centre the (tiny) player AABB in a 16³ window
        wox (- (long #?(:clj (Math/floor (aget pos 0)) :cljs (js/Math.floor (aget pos 0)))) 8)
        woy (- (long #?(:clj (Math/floor (aget pos 1)) :cljs (js/Math.floor (aget pos 1)))) 8)
        woz (- (long #?(:clj (Math/floor (aget pos 2)) :cljs (js/Math.floor (aget pos 2)))) 8)
        scratch (chunk/stitch-window! world (:scratch world) wox woy woz)]
    ;; → window-local, step, → world (vel is a delta: translation-invariant)
    (aset pos 0 (- (aget pos 0) wox)) (aset pos 1 (- (aget pos 1) woy)) (aset pos 2 (- (aget pos 2) woz))
    (let [vy0 (aget vel 1)                                  ; downward speed entering the step
          g (k/integrate-physics! pos vel scratch (:solid world)
                                  0 0 0 HALF-W HEIGHT dx dz dt)]
      (aset pos 0 (+ (aget pos 0) wox)) (aset pos 1 (+ (aget pos 1) woy)) (aset pos 2 (+ (aget pos 2) woz))
      ;; hotbar select: Q prev / E next, latched so one press = one step
      (let [n   (count (:hotbar player))
            qe? (or (contains? input :q) (contains? input :e))
            sel (if (and qe? (not (:qe player)))
                  (mod (+ (long (:sel player)) (if (contains? input :e) 1 -1)) n)
                  (:sel player))
            on?     (= 1 g)
            ;; fall damage on the landing frame; void death below the world
            landed? (and on? (not (:on-ground player)))
            fall    (max 0.0 (- (double vy0)))             ; impact speed (vy0 < 0 falling)
            hp      (long (:health player))
            hp'     (if (and landed? (> fall FALL-SAFE))
                      (max 0 (- hp (long (#?(:clj Math/floor :cljs js/Math.floor)
                                          (* (- fall FALL-SAFE) FALL-SCALE)))))
                      hp)
            void?   (< (aget pos 1) VOID-Y)
            dead?   (or void? (<= hp' 0))]
        (if dead?
          (let [[sx sy sz] (:spawn player)]               ; respawn: reset pos/vel + full health
            (aset pos 0 sx) (aset pos 1 sy) (aset pos 2 sz)
            (aset vel 0 0.0) (aset vel 1 0.0) (aset vel 2 0.0)
            (assoc player :yaw yaw :pitch pitch :on-ground false
                   :health (:max-health player) :sel sel :qe qe?))
          (assoc player :yaw yaw :pitch pitch :on-ground on?
                 :health hp' :sel sel :qe qe?))))))

(defn player-cam
  "Render camera built from the player: eye at feet + EYE, reusing slice/fly-mvp."
  [player]
  (let [p (:pos player)]
    {:pos [(aget p 0) (+ (aget p 1) EYE) (aget p 2)]
     :yaw (:yaw player) :pitch (:pitch player)}))

(defn mvp [player aspect]
  (slice/fly-mvp (player-cam player) aspect))

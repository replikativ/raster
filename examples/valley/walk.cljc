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
   Flat for now (walkable across seams without step-up); swap gen-flat-world →
   gen-world once the physics gains step-up to walk real terrain."
  [cw ch cd]
  (let [world (chunk/gen-flat-world cw ch cd 8)
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
     :yaw 0.0 :pitch -0.2 :on-ground false}))

(defn grid-player-update
  "Player physics against the multi-chunk world via a player-centred window."
  [world player input dt]
  (let [{:keys [pos vel yaw pitch]} player
        on?   (fn [a] (if (contains? input a) 1.0 0.0))
        yaw   (+ yaw (* TURN dt (- (on? :left) (on? :right))))
        pitch (-> (+ pitch (* TURN dt (- (on? :up) (on? :down)))) (max -1.5) (min 1.5))
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
    (let [g (k/integrate-physics! pos vel scratch (:solid world)
                                  0 0 0 HALF-W HEIGHT dx dz dt)]
      (aset pos 0 (+ (aget pos 0) wox)) (aset pos 1 (+ (aget pos 1) woy)) (aset pos 2 (+ (aget pos 2) woz))
      (assoc player :yaw yaw :pitch pitch :on-ground (= 1 g)))))

(defn player-cam
  "Render camera built from the player: eye at feet + EYE, reusing slice/fly-mvp."
  [player]
  (let [p (:pos player)]
    {:pos [(aget p 0) (+ (aget p 1) EYE) (aget p 2)]
     :yaw (:yaw player) :pitch (:pitch player)}))

(defn mvp [player aspect]
  (slice/fly-mvp (player-cam player) aspect))

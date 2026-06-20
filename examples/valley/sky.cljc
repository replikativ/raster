(ns valley.sky
  "Cross-platform sky: a camera-centred cube rendered behind the world (depth off) whose
   fragment shader paints a day/night gradient + sun disc/glow + night stars from the
   view direction. One shader → GLSL + WGSL. Day-night colours and sun direction are
   computed on the CPU each frame (port of valley.game's keyframes) and passed as
   uniforms, so no inverse(vp) is needed in the shader."
  (:require [clojure.string]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- tan [x] (#?(:clj Math/tan :cljs js/Math.tan) x))
(def ^:const TAU 6.283185307179586)
(def ^:const DAY 24000.0)
(def ^:const TIME-SPEED 200.0)   ; ticks/sec (faster than the JVM 72 so a demo cycles)
(def ^:const STRIDE 12)          ; pos3 (direction)

;; --- unit cube (direction vectors); 8 verts, 12 triangles ----------------------------
(def cube-verts
  (vec (for [z [-1.0 1.0] y [-1.0 1.0] x [-1.0 1.0] c [x y z]] c)))   ; 8 verts × 3
(def cube-indices
  [0 1 3 0 3 2  4 6 7 4 7 5  0 4 5 0 5 1  2 3 7 2 7 6  0 2 6 0 6 4  1 5 7 1 7 3])

;; --- day-night (CPU): ratio, sky colours, sun direction ------------------------------
(defn- lerp [a b t] (+ a (* (- b a) t)))
(defn- lerp3 [a b t] (mapv #(lerp %1 %2 t) a b))
(defn advance ^double [game-time dt] (mod (+ (double game-time) (* dt TIME-SPEED)) DAY))

(defn- day-ratio
  "0 at deep night → 1 at full day, ramped around dawn (5000) and dusk (19000)."
  [t]
  (cond (< t 5000) 0.05 (< t 7000) (lerp 0.05 1.0 (/ (- t 5000) 2000.0))
        (< t 17000) 1.0  (< t 19000) (lerp 1.0 0.05 (/ (- t 17000) 2000.0)) :else 0.05))

(def ^:private day-top [0.30 0.55 0.92]) (def ^:private day-bot [0.70 0.82 1.0])
(def ^:private night-top [0.02 0.02 0.08]) (def ^:private night-bot [0.05 0.05 0.16])
(def ^:private dusk-bot [0.85 0.45 0.25])

;; continuous triangular bump peaking 1.0 at `c`, 0.0 at ±`w` — replaces the old hard
;; dusk? boolean that snapped the bottom colour at t=16500 (the visible lighting jump).
(defn- bump [t c w] (max 0.0 (- 1.0 (/ (Math/abs (double (- (double t) c))) (double w)))))

(defn- sky-cols [t]
  (let [r  (day-ratio t)
        dw (max (bump t 6000.0 1500.0) (bump t 18000.0 1500.0))   ; dawn + dusk glow
        top (lerp3 night-top day-top r)
        bot (lerp3 night-bot (lerp3 day-bot dusk-bot dw) r)]       ; orange blended in smoothly
    [top bot r]))

(defn day-light
  "Public day-ratio (0.05 deep night → 1.0 full day) for the terrain shader's env uniform,
   so world + mobs dim into night with the same curve the sky uses."
  [t] (day-ratio t))

(defn sky-colors
  "Public [top bot] sky RGB at game-time t — the water shader reflects these."
  [t] (let [[top bot _] (sky-cols t)] [top bot]))

(defn- sun-dir [t]
  (let [ph (* (/ t DAY) TAU)
        x (sin ph) y (- (cos ph)) z 0.3
        l (Math/sqrt (+ (* x x) (* y y) (* z z)))]
    [(/ x l) (/ y l) (/ z l)]))

;; --- camera: projection × rotation-only view (sky centred on the camera) -------------
(defn- forward [yaw pitch] (let [cp (cos pitch)] [(* (sin yaw) cp) (sin pitch) (* (- (cos yaw)) cp)]))
(defn- cross [a b] [(- (* (a 1) (b 2)) (* (a 2) (b 1)))
                    (- (* (a 2) (b 0)) (* (a 0) (b 2)))
                    (- (* (a 0) (b 1)) (* (a 1) (b 0)))])
(defn- nrm [a] (let [l (Math/sqrt (+ (* (a 0) (a 0)) (* (a 1) (a 1)) (* (a 2) (a 2))))] [(/ (a 0) l) (/ (a 1) l) (/ (a 2) l)]))
(defn- dot3 [a b] (+ (* (a 0) (b 0)) (* (a 1) (b 1)) (* (a 2) (b 2))))

(defn- mat-mul [a b]
  (vec (for [c (range 4) r (range 4)]
         (+ (* (a (+ r 0)) (b (+ (* c 4) 0))) (* (a (+ r 4)) (b (+ (* c 4) 1)))
            (* (a (+ r 8)) (b (+ (* c 4) 2))) (* (a (+ r 12)) (b (+ (* c 4) 3)))))))

(defn- perspective [fovy aspect zn zf]
  (let [f (/ 1.0 (tan (* 0.5 fovy)))]
    [(/ f aspect) 0.0 0.0 0.0, 0.0 (- f) 0.0 0.0,
     0.0 0.0 (/ zf (- zn zf)) -1.0, 0.0 0.0 (/ (* zn zf) (- zn zf)) 0.0]))

(defn- rot-view [yaw pitch]   ; look-at with eye at origin (rotation only)
  (let [f (forward yaw pitch) s (nrm (cross f [0.0 1.0 0.0])) u (cross s f)]
    [(s 0) (u 0) (- (f 0)) 0.0, (s 1) (u 1) (- (f 1)) 0.0,
     (s 2) (u 2) (- (f 2)) 0.0, 0.0 0.0 0.0 1.0]))

(defn uniform
  "Flat float vector for the sky uniform: mvp(16) topCol(4) botCol(4) sun(4) starRot(4)
   — 32 floats. starRot = [cos(ph) sin(ph) 0 0] where ph is the sun's hour-angle, so the
   fragment shader can wheel the star field about the celestial axis with the sun."
  [game-time player aspect]
  (let [mvp (mat-mul (perspective 0.9 aspect 0.1 10.0) (rot-view (:yaw player) (:pitch player)))
        [top bot r] (sky-cols game-time)
        [sx sy sz] (sun-dir game-time)
        ph (* (/ (double game-time) DAY) TAU)]
    (vec (concat mvp top [0.0] bot [0.0] [sx sy sz r] [(cos ph) (sin ph) 0.0 0.0]))))

;; --- shader (one source → GLSL + WGSL) -----------------------------------------------
(def shader
  '{:uniform   {:name "Sky" :fields [[:mvp :mat4] [:topCol :vec4] [:botCol :vec4] [:sun :vec4] [:starRot :vec4]]}
    :attributes [[inDir vec3 0]]
    :varyings   [[vDir vec3 0]]
    :vertex     [(set-position (* mvp (vec4 inDir 1.0))) (out vDir inDir)]
    :fragment   [(let d vec3 (normalize vDir))
                 (let h float (clamp (+ (* (swizzle d y) 0.5) 0.5) 0.0 1.0))
                 (let base vec3 (mix (swizzle botCol xyz) (swizzle topCol xyz) h))
                 (let sd float (dot d (swizzle sun xyz)))
                 (let disc float (smoothstep 0.9985 0.9996 sd))
                 (let glow float (* (pow (max sd 0.0) 48.0) 0.4))
                 ;; rotate the view dir's x/y by the sun hour-angle so stars wheel with the sky
                 (let sc float (swizzle starRot x))
                 (let ss float (swizzle starRot y))
                 (let rd vec3 (vec3 (- (* (swizzle d x) sc) (* (swizzle d y) ss))
                                    (+ (* (swizzle d x) ss) (* (swizzle d y) sc))
                                    (swizzle d z)))
                 (let scell vec3 (floor (* rd 110.0)))
                 (let hsh float (fract (* (sin (dot scell (vec3 12.9898 78.233 37.719))) 43758.547)))
                 (let star float (* (smoothstep 0.993 1.0 hsh)
                                    (clamp (- 1.0 (* (swizzle sun w) 1.3)) 0.0 1.0)
                                    (max 0.0 (swizzle d y))))
                 (color (vec4 (+ (+ base (* (vec3 1.0 0.93 0.78) (+ disc glow)))
                                 (vec3 star star star)) 1.0))]})

(def pipeline-spec
  {:shader shader :topology :triangles :depth? false :cull :none
   :vertex {:stride STRIDE :attributes [{:location 0 :format :float3 :offset 0}]}})

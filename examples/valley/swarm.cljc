(ns valley.swarm
  "Cross-platform mob layer: resident SoA columns + batch physics + wander AI + textured
   box-model mobs (cow/pig/chicken/sheep) with walk animation. Identical code on JVM
   (batch kernel = bytecode) and browser (batch kernel = wasm, world resident in linear
   memory via k/upload-world!). Hot mob state (pos/vel/yaw/walk-phase) lives in SoA
   columns the physics kernel sweeps in one call each frame; the box meshes are rebuilt
   per frame into a dynamic vertex buffer (valley.tex shader/atlas)."
  (:require [valley.chunk :as chunk]
            [valley.tex :as tex]
            [valley.kernels :as k]))

(defn- cos [x] (#?(:clj Math/cos :cljs js/Math.cos) x))
(defn- sin [x] (#?(:clj Math/sin :cljs js/Math.sin) x))
(defn- atan2 [y x] (#?(:clj Math/atan2 :cljs js/Math.atan2) y x))
(def ^:const TAU 6.283185307179586)

(def ^:const SPEED 2.5)     ; blocks / second
(def ^:const HALF-W 0.6)
(def ^:const HEIGHT 1.4)

;; --- mob box-models (port of valley.mob-models; :tex = valley.tex mob-skin layer) ----
;; box: {:sz [w h d] :off [ox oy oz] :tex layer :leg n?}  — offset centres x/z, bottom
;; at oy; legs (with :leg) animate by rotating about their top about the X axis.
(defn- animal [body-tex head-tex leg-tex hy hz]
  {:boxes (into [{:sz [0.5 0.4 0.8] :off [0 0.65 0] :tex body-tex}
                 {:sz [0.3 0.3 0.3] :off [0 hy hz] :tex head-tex}]
                (map-indexed (fn [i [dx dz]] {:sz [0.12 0.5 0.12] :off [dx 0 dz] :tex leg-tex :leg i})
                             [[-0.15 -0.25] [0.15 -0.25] [-0.15 0.25] [0.15 0.25]]))})

(def mob-models
  {:cow     (animal 22 23 24 1.05 0.5)
   :sheep   (animal 22 23 24 1.0 0.45)
   :pig     {:boxes (into [{:sz [0.4 0.35 0.6] :off [0 0.45 0] :tex 25}
                           {:sz [0.3 0.3 0.3] :off [0 0.8 0.35] :tex 26}]
                          (map-indexed (fn [i [dx dz]] {:sz [0.1 0.35 0.1] :off [dx 0 dz] :tex 27 :leg i})
                                       [[-0.12 -0.2] [0.12 -0.2] [-0.12 0.2] [0.12 0.2]]))}
   :chicken {:boxes (into [{:sz [0.2 0.2 0.3] :off [0 0.35 0] :tex 28}
                           {:sz [0.15 0.15 0.15] :off [0 0.55 0.18] :tex 28}]
                          (map-indexed (fn [i dx] {:sz [0.04 0.2 0.04] :off [dx 0 0] :tex 29 :leg i})
                                       [-0.06 0.06]))}})
(def ^:private type-keys [:cow :pig :chicken :sheep])
(def ^:const VERTS-PER-MOB (* 6 36))   ; ≤6 boxes × 6 faces × 6 verts

(defn spawn
  "N mobs scattered on the world surface. Uploads the resident world to the kernel
   (JVM no-op; browser writes the block array into wasm memory once)."
  [world n]
  (k/upload-world! (:blocks world) (:solid world))
  (let [pos (chunk/darray (* n 3)) vel (chunk/darray (* n 3))
        dxs (chunk/darray n) dzs (chunk/darray n)
        yaw (chunk/darray n) wph (chunk/darray n) typ (chunk/iarray n)
        WX (long (:wx world)) WZ (long (:wz world))]
    (dotimes [i n]
      (let [x (+ 2 (mod (* i 7) (max 1 (- WX 4))))
            z (+ 2 (mod (* i 13) (max 1 (- WZ 4))))
            top (loop [y (dec (long (:wy world)))]
                  (cond (< y 0) 0
                        (pos? (chunk/world-block world x y z)) (inc y)
                        :else (recur (dec y))))]
        (aset pos (* i 3) (+ x 0.5))
        (aset pos (+ (* i 3) 1) (+ (double top) 2.0))   ; spawn above → fall onto terrain
        (aset pos (+ (* i 3) 2) (+ z 0.5))
        (aset wph i (* i 0.37))
        (aset typ i (int (mod i 4)))))
    {:pos pos :vel vel :dxs dxs :dzs dzs :yaw yaw :wph wph :typ typ :n n :t 0.0 :world world}))

(defn update!
  "Wander AI (each mob heads in a slowly-rotating per-mob direction) + one batch physics
   call over all mobs; per-mob yaw faces the move and walk-phase advances. In-place."
  [mobs dt]
  (let [{:keys [pos vel dxs dzs yaw wph n world]} mobs
        t (+ (:t mobs) dt)]
    (dotimes [i n]
      (let [h (+ (* t 0.4) (* i 1.7))
            dx (* (cos h) SPEED dt) dz (* (sin h) SPEED dt)]
        (aset dxs i dx) (aset dzs i dz)
        (aset yaw i (atan2 dx dz))
        (aset wph i (+ (aget wph i) (* dt 3.0)))))
    (k/integrate-physics-batch! pos vel dxs dzs (:blocks world) (:solid world)
                                n (:wx world) (:wy world) (:wz world) HALF-W HEIGHT dt)
    (assoc mobs :t t)))

(def ^:const REACH 3.5)   ; cull radius (blocks)

(defn cull-nearest!
  "Despawn the nearest mob within REACH of (px,pz): nearest-neighbour query over the
   position column + O(1) swap-remove (last row moved into the freed slot, n--)."
  [mobs px pz]
  (let [{:keys [pos vel yaw wph typ n]} mobs]
    (loop [i 0 best -1 bd (* REACH REACH)]
      (if (< i n)
        (let [dx (- (aget pos (* i 3)) px) dz (- (aget pos (+ (* i 3) 2)) pz)
              d (+ (* dx dx) (* dz dz)) hit? (< d bd)]
          (recur (inc i) (if hit? i best) (if hit? d bd)))
        (if (>= best 0)
          (let [l (dec n) bb (* best 3) lb (* l 3)]
            (aset pos bb (aget pos lb)) (aset pos (+ bb 1) (aget pos (+ lb 1))) (aset pos (+ bb 2) (aget pos (+ lb 2)))
            (aset vel (+ bb 1) (aget vel (+ lb 1)))
            (aset yaw best (aget yaw l)) (aset wph best (aget wph l)) (aset typ best (aget typ l))
            (assoc mobs :n l))
          mobs)))))

;; --- box-model mesh (triangle list for the vertex-only dynamic mesh) -----------------
(def ^:private box-faces   ; [shade [4 corners [cx cy cz]∈{0,1}]]
  [[0.85 [[0 0 1] [1 0 1] [1 1 1] [0 1 1]]]   ; +z
   [0.85 [[1 0 0] [0 0 0] [0 1 0] [1 1 0]]]   ; -z
   [0.7  [[1 0 1] [1 0 0] [1 1 0] [1 1 1]]]   ; +x
   [0.7  [[0 0 0] [0 0 1] [0 1 1] [0 1 0]]]   ; -x
   [1.0  [[0 1 1] [1 1 1] [1 1 0] [0 1 0]]]   ; +y top
   [0.55 [[0 0 0] [1 0 0] [1 0 1] [0 0 1]]]]) ; -y
(def ^:private box-uv [[0.0 0.0] [1.0 0.0] [1.0 1.0] [0.0 1.0]])

(defn- leg-angle [wp leg]
  (* 0.5 (sin (+ (* wp TAU) (if (or (= leg 0) (= leg 3)) 0.0 Math/PI)))))

(defn- emit-box!
  "Append one box's 36 triangle-list verts (after leg-rot, yaw, translate) to `out`."
  [out box wx wy wz yaw wp]
  (let [{:keys [sz off tex leg]} box
        [w h d] sz [ox oy oz] off
        la (if leg (leg-angle wp leg) 0.0)
        cla (cos la) sla (sin la) pvy (+ oy h)        ; leg pivot top = (ox,oy+h,oz)
        cy (cos yaw) sy (sin yaw)]
    (doseq [[shade corners] box-faces]
      (doseq [ci [0 1 2 0 2 3]]
        (let [[gx gy gz] (nth corners ci) [u v] (nth box-uv ci)
              lx (+ (- ox (* 0.5 w)) (* gx w))
              ly (+ oy (* gy h))
              lz (+ (- oz (* 0.5 d)) (* gz d))
              ;; leg rotate-x about pivot (ox,pvy,oz)
              dy (- ly pvy) dz2 (- lz oz)
              ly (if leg (+ pvy (- (* dy cla) (* dz2 sla))) ly)
              lz (if leg (+ oz  (+ (* dy sla) (* dz2 cla))) lz)
              ;; yaw rotate-y about origin, then translate to world
              rx (+ (* lx cy) (* lz sy))
              rz (+ (* (- lx) sy) (* lz cy))]
          (vswap! out (fn [t] (-> t (conj! (+ wx rx)) (conj! (+ wy ly)) (conj! (+ wz rz))
                                  (conj! u) (conj! v) (conj! (double tex)) (conj! shade)))))))))

(defn verts
  "Flat float seq of all mob box-models (triangle list) for the dynamic mesh."
  [mobs]
  (let [{:keys [pos yaw wph typ n]} mobs
        out (volatile! (transient []))]
    (dotimes [i n]
      (let [model (get mob-models (nth type-keys (aget typ i)))
            wx (aget pos (* i 3)) wy (aget pos (+ (* i 3) 1)) wz (aget pos (+ (* i 3) 2))
            yw (aget yaw i) wp (aget wph i)]
        (doseq [box (:boxes model)] (emit-box! out box wx wy wz yw wp))))
    (persistent! @out)))

;; --- atlas: valley.tex terrain + mob-skin layers (0..29) ----------------------------
(def ^:const N-LAYERS tex/N-LAYERS)
(defn atlas-pixels [_w _h] (tex/atlas-pixels))

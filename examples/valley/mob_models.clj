(ns valley.mob-models
  "Box-mesh generation for Box-based cuboid mob assemblies.
  Each mob is a collection of boxes (head, body, legs) with walk animation.
  Uses the same 9-float vertex format as terrain: [x y z u v dayLight nightLight texLayer ao]."
  (:refer-clojure :exclude [defn])
  (:require [raster.core :refer [defn deftm]]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Mob model definitions
;; ================================================================

;; Each box: {:name :keyword :size [w h d] :offset [x y z] :tex layer-idx
;;            :pivot [px py pz] :axis :x/:y/:z}
;; Pivot + axis used for joint rotation (legs, head)

(def mob-models
  {:cow {:boxes [{:name :body   :size [0.5 0.4 0.8] :offset [0.0 0.65 0.0]  :tex 88}
                 {:name :head   :size [0.3 0.3 0.3] :offset [0.0 1.05 0.5]  :tex 89
                  :pivot [0.0 1.05 0.4] :axis :x}
                 {:name :leg-fl :size [0.12 0.5 0.12] :offset [-0.15 0.0 0.25]  :tex 90
                  :pivot [0.0 0.5 0.0] :axis :x}
                 {:name :leg-fr :size [0.12 0.5 0.12] :offset [0.15 0.0 0.25]  :tex 90
                  :pivot [0.0 0.5 0.0] :axis :x}
                 {:name :leg-bl :size [0.12 0.5 0.12] :offset [-0.15 0.0 -0.25] :tex 90
                  :pivot [0.0 0.5 0.0] :axis :x}
                 {:name :leg-br :size [0.12 0.5 0.12] :offset [0.15 0.0 -0.25] :tex 90
                  :pivot [0.0 0.5 0.0] :axis :x}]}

   :pig {:boxes [{:name :body   :size [0.4 0.35 0.6] :offset [0.0 0.45 0.0] :tex 91}
                 {:name :head   :size [0.3 0.3 0.3]  :offset [0.0 0.8 0.35] :tex 92
                  :pivot [0.0 0.8 0.3] :axis :x}
                 {:name :leg-fl :size [0.1 0.35 0.1]  :offset [-0.12 0.0 0.2] :tex 93
                  :pivot [0.0 0.35 0.0] :axis :x}
                 {:name :leg-fr :size [0.1 0.35 0.1]  :offset [0.12 0.0 0.2]  :tex 93
                  :pivot [0.0 0.35 0.0] :axis :x}
                 {:name :leg-bl :size [0.1 0.35 0.1]  :offset [-0.12 0.0 -0.2] :tex 93
                  :pivot [0.0 0.35 0.0] :axis :x}
                 {:name :leg-br :size [0.1 0.35 0.1]  :offset [0.12 0.0 -0.2]  :tex 93
                  :pivot [0.0 0.35 0.0] :axis :x}]}

   :chicken {:boxes [{:name :body :size [0.2 0.2 0.3] :offset [0.0 0.35 0.0] :tex 94}
                     {:name :head :size [0.15 0.15 0.15] :offset [0.0 0.55 0.18] :tex 94
                      :pivot [0.0 0.55 0.15] :axis :x}
                     {:name :leg-l :size [0.04 0.2 0.04] :offset [-0.06 0.0 0.0] :tex 95
                      :pivot [0.0 0.2 0.0] :axis :x}
                     {:name :leg-r :size [0.04 0.2 0.04] :offset [0.06 0.0 0.0]  :tex 95
                      :pivot [0.0 0.2 0.0] :axis :x}]}

   :sheep {:boxes [{:name :body   :size [0.45 0.4 0.7] :offset [0.0 0.6 0.0]  :tex 88}
                   {:name :head   :size [0.25 0.25 0.25] :offset [0.0 0.95 0.4] :tex 89
                    :pivot [0.0 0.95 0.35] :axis :x}
                   {:name :leg-fl :size [0.1 0.45 0.1] :offset [-0.14 0.0 0.22]  :tex 90
                    :pivot [0.0 0.45 0.0] :axis :x}
                   {:name :leg-fr :size [0.1 0.45 0.1] :offset [0.14 0.0 0.22]   :tex 90
                    :pivot [0.0 0.45 0.0] :axis :x}
                   {:name :leg-bl :size [0.1 0.45 0.1] :offset [-0.14 0.0 -0.22] :tex 90
                    :pivot [0.0 0.45 0.0] :axis :x}
                   {:name :leg-br :size [0.1 0.45 0.1] :offset [0.14 0.0 -0.22]  :tex 90
                    :pivot [0.0 0.45 0.0] :axis :x}]}

   ;; Hostile mobs (humanoid model for zombie/skeleton)
   :zombie {:boxes [{:name :body   :size [0.4 0.6 0.25] :offset [0.0 0.75 0.0] :tex 96}
                    {:name :head   :size [0.35 0.35 0.35] :offset [0.0 1.45 0.0] :tex 97
                     :pivot [0.0 1.35 0.0] :axis :x}
                    {:name :leg-fl :size [0.18 0.75 0.18] :offset [-0.11 0.0 0.0] :tex 96
                     :pivot [0.0 0.75 0.0] :axis :x}
                    {:name :leg-fr :size [0.18 0.75 0.18] :offset [0.11 0.0 0.0]  :tex 96
                     :pivot [0.0 0.75 0.0] :axis :x}
                    {:name :arm-l  :size [0.12 0.6 0.12] :offset [-0.3 0.8 0.25] :tex 96}
                    {:name :arm-r  :size [0.12 0.6 0.12] :offset [0.3 0.8 0.25]  :tex 96}]}

   :skeleton {:boxes [{:name :body   :size [0.35 0.55 0.2] :offset [0.0 0.8 0.0] :tex 98}
                      {:name :head   :size [0.35 0.35 0.35] :offset [0.0 1.45 0.0] :tex 99
                       :pivot [0.0 1.35 0.0] :axis :x}
                      {:name :leg-fl :size [0.1 0.8 0.1] :offset [-0.1 0.0 0.0] :tex 98
                       :pivot [0.0 0.8 0.0] :axis :x}
                      {:name :leg-fr :size [0.1 0.8 0.1] :offset [0.1 0.0 0.0]  :tex 98
                       :pivot [0.0 0.8 0.0] :axis :x}
                      {:name :arm-l  :size [0.08 0.6 0.08] :offset [-0.25 0.85 0.0] :tex 98}
                      {:name :arm-r  :size [0.08 0.6 0.08] :offset [0.25 0.85 0.0]  :tex 98}]}

   :lurker {:boxes [{:name :body   :size [0.3 0.8 0.3] :offset [0.0 0.5 0.0] :tex 100}
                     {:name :head   :size [0.35 0.35 0.35] :offset [0.0 1.35 0.0] :tex 101
                      :pivot [0.0 1.3 0.0] :axis :x}
                     {:name :leg-fl :size [0.12 0.5 0.12] :offset [-0.1 0.0 0.1]  :tex 100
                      :pivot [0.0 0.5 0.0] :axis :x}
                     {:name :leg-fr :size [0.12 0.5 0.12] :offset [0.1 0.0 0.1]   :tex 100
                      :pivot [0.0 0.5 0.0] :axis :x}
                     {:name :leg-bl :size [0.12 0.5 0.12] :offset [-0.1 0.0 -0.1] :tex 100
                      :pivot [0.0 0.5 0.0] :axis :x}
                     {:name :leg-br :size [0.12 0.5 0.12] :offset [0.1 0.0 -0.1]  :tex 100
                      :pivot [0.0 0.5 0.0] :axis :x}]}

   :spider {:boxes [{:name :body   :size [0.6 0.25 0.8] :offset [0.0 0.25 0.0] :tex 102}
                    {:name :head   :size [0.35 0.3 0.3] :offset [0.0 0.35 0.5]  :tex 103
                     :pivot [0.0 0.35 0.4] :axis :x}
                    {:name :leg-fl :size [0.04 0.15 0.5] :offset [-0.35 0.15 0.2]  :tex 102}
                    {:name :leg-fr :size [0.04 0.15 0.5] :offset [0.35 0.15 0.2]   :tex 102}
                    {:name :leg-bl :size [0.04 0.15 0.5] :offset [-0.35 0.15 -0.2] :tex 102}
                    {:name :leg-br :size [0.04 0.15 0.5] :offset [0.35 0.15 -0.2]  :tex 102}]}})

;; ================================================================
;; Box mesh generation
;; ================================================================

;; 3D rotation helpers — deftm for bytecode compilation.
;; >4 args so casts are inside the body.

(deftm rotate-x
  "Rotate point [x y z] around pivot by angle (radians) on X axis."
  [x :- Double, y :- Double, z :- Double,
   px :- Double, py :- Double, pz :- Double, angle :- Double]
  (let [dy (- y py) dz (- z pz)
        c (Math/cos angle) s (Math/sin angle)]
    [x
     (+ py (- (* c dy) (* s dz)))
     (+ pz (+ (* s dy) (* c dz)))]))

(deftm rotate-y
  "Rotate point [x y z] around pivot by angle (radians) on Y axis."
  [x :- Double, y :- Double, z :- Double,
   px :- Double, py :- Double, pz :- Double, angle :- Double]
  (let [dx (- x px) dz (- z pz)
        c (Math/cos angle) s (Math/sin angle)]
    [(+ px (+ (* c dx) (* s dz)))
     y
     (+ pz (- (* c dz) (* s dx)))]))

(defn- emit-box-verts!
  "Emit 24 vertices (6 faces x 4 verts) and 36 indices for a box.
  Returns [new-vi new-ii]."
  [^floats verts ^ints indices vi ii
   x0 y0 z0 x1 y1 z1 tex-layer]
  (let [vi (long vi) ii (long ii)
        x0 (float x0) y0 (float y0) z0 (float z0)
        x1 (float x1) y1 (float y1) z1 (float z1)
        tex (float tex-layer)
        day (float 1.0) night (float 0.0) ao (float 1.0)
        ;; 6 faces, each with 4 corners
        ;; front (z+), back (z-), right (x+), left (x-), top (y+), bottom (y-)
        faces [[x0 y0 z1  x1 y0 z1  x1 y1 z1  x0 y1 z1]   ;; front
               [x1 y0 z0  x0 y0 z0  x0 y1 z0  x1 y1 z0]   ;; back
               [x1 y0 z1  x1 y0 z0  x1 y1 z0  x1 y1 z1]   ;; right
               [x0 y0 z0  x0 y0 z1  x0 y1 z1  x0 y1 z0]   ;; left
               [x0 y1 z1  x1 y1 z1  x1 y1 z0  x0 y1 z0]   ;; top
               [x0 y0 z0  x1 y0 z0  x1 y0 z1  x0 y0 z1]]] ;; bottom
    (loop [fi 0, vi vi, ii ii]
      (if (>= fi 6)
        [vi ii]
        (let [face (nth faces fi)
              b (* vi 9)]
          ;; 4 vertices per face
          (dotimes [v 4]
            (let [vb (+ b (* v 9))
                  voff (* v 3)
                  u (float (if (or (= v 1) (= v 2)) 1.0 0.0))
                  vv (float (if (or (= v 2) (= v 3)) 0.0 1.0))]
              (aset verts (+ vb 0) (float (nth face (+ voff 0))))
              (aset verts (+ vb 1) (float (nth face (+ voff 1))))
              (aset verts (+ vb 2) (float (nth face (+ voff 2))))
              (aset verts (+ vb 3) u)
              (aset verts (+ vb 4) vv)
              (aset verts (+ vb 5) day)
              (aset verts (+ vb 6) night)
              (aset verts (+ vb 7) tex)
              (aset verts (+ vb 8) ao)))
          ;; 6 indices (2 triangles)
          (aset indices (+ ii 0) (int vi))
          (aset indices (+ ii 1) (int (+ vi 1)))
          (aset indices (+ ii 2) (int (+ vi 2)))
          (aset indices (+ ii 3) (int vi))
          (aset indices (+ ii 4) (int (+ vi 2)))
          (aset indices (+ ii 5) (int (+ vi 3)))
          (recur (inc fi) (+ vi 4) (+ ii 6)))))))

(defn- leg-angle
  "Sinusoidal leg swing for walk animation. phase=0..1 maps to sin."
  [walk-phase leg-index]
  (let [walk-phase (double walk-phase)
        ;; Alternate legs: FL/BR swing together, FR/BL swing together
        offset (case (long leg-index)
                 0 0.0    ;; front-left
                 1 Math/PI ;; front-right
                 2 Math/PI ;; back-left
                 3 0.0)   ;; back-right
        angle (* 0.5 (Math/sin (+ (* walk-phase 2.0 Math/PI) offset)))]
    angle))

;; ================================================================
;; Build mob mesh
;; ================================================================

(defn build-mob-mesh
  "Build vertex/index arrays for a single mob at world position [wx wy wz]
  facing yaw (radians), with walk-phase (0..1 for animation cycle).
  Returns {:vertices float[] :indices int[] :vert-count :index-count}."
  [mob-type wx wy wz yaw walk-phase]
  (let [wx (double wx) wy (double wy) wz (double wz)
        yaw (double yaw) walk-phase (double walk-phase)
        model (get mob-models mob-type)
        boxes (:boxes model)
        num-boxes (count boxes)
        ;; 24 verts per box (6 faces x 4), 36 indices per box
        max-verts (* num-boxes 24)
        max-indices (* num-boxes 36)
        verts (float-array (* max-verts 9))
        indices (int-array max-indices)]
    (loop [bi 0, vi 0, ii 0]
      (if (>= bi num-boxes)
        {:vertices (java.util.Arrays/copyOf verts (int (* vi 9)))
         :indices (java.util.Arrays/copyOf indices (int ii))
         :vert-count (int vi)
         :index-count (int ii)}
        (let [box (nth boxes bi)
              [sw sh sd] (:size box)
              [ox oy oz] (:offset box)
              sw (double sw) sh (double sh) sd (double sd)
              ox (double ox) oy (double oy) oz (double oz)
              tex (long (:tex box))
              ;; Compute leg rotation if this is a leg box
              is-leg? (.startsWith (name (:name box)) "leg")
              leg-idx (case (:name box)
                        :leg-fl 0 :leg-fr 1 :leg-bl 2 :leg-br 3
                        :leg-l 0 :leg-r 1
                        -1)
              leg-rot (if (and is-leg? (>= leg-idx 0))
                        (leg-angle walk-phase leg-idx)
                        0.0)
              ;; Box local coords (centered on X/Z, bottom at Y=0 offset)
              lx0 (- ox (* 0.5 sw))
              lx1 (+ ox (* 0.5 sw))
              ly0 oy
              ly1 (+ oy sh)
              lz0 (- oz (* 0.5 sd))
              lz1 (+ oz (* 0.5 sd))
              ;; Emit box in local space first
              [new-vi new-ii] (emit-box-verts! verts indices vi ii
                                               lx0 ly0 lz0 lx1 ly1 lz1 tex)]
          ;; Apply leg rotation and world transform to emitted verts
          (let [start-v (* vi 9)
                end-v (* new-vi 9)
                pivot (:pivot box)]
            (loop [v start-v]
              (when (< v end-v)
                (let [lx (double (aget verts v))
                      ly (double (aget verts (+ v 1)))
                      lz (double (aget verts (+ v 2)))
                      ;; Apply leg rotation around pivot
                      [lx ly lz] (if (and is-leg? pivot (not (zero? leg-rot)))
                                   (let [[px py pz] pivot]
                                     (rotate-x lx ly lz px py pz leg-rot))
                                   [lx ly lz])
                      ;; Apply yaw rotation around origin
                      [rx _ry rz] (rotate-y lx ly lz 0.0 0.0 0.0 yaw)
                      ;; Translate to world position
                      fx (+ (double rx) wx)
                      fy (+ ly wy)
                      fz (+ (double rz) wz)]
                  (aset verts v (float fx))
                  (aset verts (+ v 1) (float fy))
                  (aset verts (+ v 2) (float fz)))
                (recur (+ v 9)))))
          (recur (inc bi) new-vi new-ii))))))

;; ================================================================
;; Batched mob mesh (all mobs in one draw call)
;; ================================================================

(defn build-all-mobs-mesh
  "Build a single combined mesh for all mobs. Returns {:vertices :indices} or nil if no mobs."
  [mobs]
  (when (seq mobs)
    (let [mob-meshes (mapv (fn [mob]
                             (let [[mx my mz] (:pos mob)]
                               (build-mob-mesh (:mob-type mob)
                                               mx my mz
                                               (double (:yaw mob 0.0))
                                               (double (:walk-phase mob 0.0)))))
                           mobs)
          total-verts (reduce + (map :vert-count mob-meshes))
          total-indices (reduce + (map :index-count mob-meshes))
          combined-verts (float-array (* total-verts 9))
          combined-indices (int-array total-indices)]
      (loop [mi 0, voff 0, ioff 0, vi-base 0]
        (if (>= mi (count mob-meshes))
          {:vertices combined-verts :indices combined-indices
           :vert-count total-verts :index-count total-indices}
          (let [m (nth mob-meshes mi)
                mv ^floats (:vertices m)
                mind ^ints (:indices m)
                vc (long (:vert-count m))
                ic (long (:index-count m))]
            ;; Copy vertices
            (System/arraycopy mv 0 combined-verts (* voff 9) (* vc 9))
            ;; Copy indices with offset
            (dotimes [i ic]
              (aset combined-indices (+ ioff i) (int (+ (aget mind i) vi-base))))
            (recur (inc mi) (+ voff vc) (+ ioff ic) (+ vi-base vc))))))))

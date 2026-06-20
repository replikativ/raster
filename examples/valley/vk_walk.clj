(ns valley.vk-walk
  "JVM Vulkan shell for the valley multi-chunk walker + mobs (options C + mobs). Physics,
   terrain and the batch mob kernel run as bytecode; the SAME valley.walk/valley.swarm
   (.cljc) run in the browser via WebGPU + wasm (web_walk). The world spans several 16³
   chunks (player uses a per-frame window stitch); mobs are resident SoA columns swept by
   one batch kernel call each frame and drawn as a dynamic cube mesh.
   Run:  clojure -M:vulkan -m valley.vk-walk   Controls: WASD move, arrows look."
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [valley.slice :as slice]
            [valley.walk :as walk]
            [valley.swarm :as swarm]))

(def ^:const NMOBS 48)

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Valley — Walk + Mobs (raster.render)")
        pipeline (r/make-pipeline rnd slice/pipeline-spec)
        world    (walk/build-grid 3 2 3)            ; 48×32×48 biome world
        mesh     (r/make-static-mesh rnd (:verts world) (:indices world) slice/STRIDE)
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 4
                                            :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
        mob-st   (atom (swarm/spawn world NMOBS))
        mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) slice/STRIDE)
        aspect   (/ 1024.0 768.0)]
    (println "valley world mesh:" (quot (count (:verts world)) 6) "verts," (:wx world) "x" (:wy world) "x" (:wz world)
             "blocks," NMOBS "mobs")
    (r/run! rnd
            {:init-state (walk/grid-player-init world)
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [player input dt]
                       (swap! mob-st swarm/update! dt)
                       (when (contains? input :fire)
                         (let [p (:pos player)] (swap! mob-st swarm/cull-nearest! (aget p 0) (aget p 2))))
                       (walk/grid-player-update world player input dt))
             :render (fn [player frame]
                       (r/bind-pipeline! frame pipeline)
                       (r/set-uniform! frame pipeline (walk/mvp player aspect))
                       (r/bind-texture! frame pipeline tex)
                       (r/bind-mesh! frame mesh)
                       (r/draw-indexed! frame (:index-count mesh) 0)
                       (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @mob-st))]
                         (r/bind-mesh! frame mm)
                         (r/draw! frame (:vertex-count mm) 0)))})))

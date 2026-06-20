(ns valley.vk-walk
  "JVM Vulkan shell for the valley multi-chunk walker (option C). Physics + terrain run
   as bytecode; the SAME valley.walk (.cljc) runs in the browser via WebGPU + wasm
   (web_walk). The world spans several 16³ chunks; physics uses a player-centred 16³
   window stitched each frame, so the kernel is unchanged across chunk seams.
   Run:  clojure -M:vulkan -m valley.vk-walk   Controls: WASD move, arrows look."
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [valley.slice :as slice]
            [valley.walk :as walk]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Valley — Walk (raster.render)")
        pipeline (r/make-pipeline rnd slice/pipeline-spec)
        world    (walk/build-grid 3 1 3)            ; 48×16×48 multi-chunk world
        mesh     (r/make-static-mesh rnd (:verts world) (:indices world) slice/STRIDE)
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                            :pixels (slice/atlas-pixels 16 16) :filter :nearest})
        aspect   (/ 1024.0 768.0)]
    (println "valley world mesh:" (quot (count (:verts world)) 6) "verts," (count (:indices world)) "indices,"
             (:wx world) "x" (:wy world) "x" (:wz world) "blocks")
    (r/run! rnd
            {:init-state (walk/grid-player-init world)
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [player input dt] (walk/grid-player-update world player input dt))
             :render (fn [player frame]
                       (r/bind-pipeline! frame pipeline)
                       (r/set-uniform! frame pipeline (walk/mvp player aspect))
                       (r/bind-texture! frame pipeline tex)
                       (r/bind-mesh! frame mesh)
                       (r/draw-indexed! frame (:index-count mesh) 0))})))

(ns cube.vk-main
  "JVM Vulkan shell for the R4 cube validation scene. Same cube.scene (.cljc) as the
   browser WebGPU shell; only the backend differs.
   Run:  clojure -M:vulkan -m cube.vk-main"
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [cube.scene :as scene]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Cube — Vulkan (raster.render R4)")
        pipeline (r/make-pipeline rnd scene/pipeline-spec)
        mesh     (r/make-static-mesh rnd scene/cube-verts scene/cube-indices scene/STRIDE)
        tex      (r/make-texture rnd {:width 64 :height 64 :pixels (scene/checker 64 64) :filter :nearest})
        aspect   (/ 1024.0 768.0)]
    (r/run! rnd
            {:init-state {:t 0.0}
             :clear-color [0.05 0.05 0.08 1.0]
             :update (fn [s _input dt] (update s :t + dt))
             :render (fn [s frame] (scene/draw-cube! rnd frame pipeline mesh tex (:t s) aspect))})))

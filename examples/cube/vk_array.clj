(ns cube.vk-array
  "JVM Vulkan shell for the R5a texture-array cube. Run: clojure -M:vulkan -m cube.vk-array"
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [cube.scene :as scene]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Cube array — Vulkan (R5a)")
        pipeline (r/make-pipeline rnd scene/array-pipeline-spec)
        mesh     (r/make-static-mesh rnd scene/cube-verts-arr scene/cube-indices scene/STRIDE-ARR)
        tex      (r/make-texture-array rnd {:width 64 :height 64 :layers 2
                                            :pixels (scene/atlas-pixels 64 64) :filter :nearest})
        aspect   (/ 1024.0 768.0)]
    (r/run! rnd
            {:init-state {:t 0.0}
             :clear-color [0.05 0.05 0.08 1.0]
             :update (fn [s _input dt] (update s :t + dt))
             :render (fn [s frame] (scene/draw-cube! rnd frame pipeline mesh tex (:t s) aspect))})))

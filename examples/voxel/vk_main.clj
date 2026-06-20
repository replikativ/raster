(ns voxel.vk-main
  "JVM Vulkan shell for the voxel slice. Terrain kernel runs as bytecode; the SAME
   voxel.world (.cljc) renders in the browser via WebGPU + wasm.
   Run:  clojure -M:vulkan -m voxel.vk-main"
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [voxel.world :as world]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Voxel — Vulkan (raster.render)")
        pipeline (r/make-pipeline rnd world/pipeline-spec)
        m        (world/mesh-terrain)
        mesh     (r/make-static-mesh rnd (:verts m) (:indices m) world/STRIDE)
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                            :pixels (world/atlas-pixels 16 16) :filter :nearest})
        aspect   (/ 1024.0 768.0)]
    (println "voxel mesh:" (quot (count (:verts m)) 6) "verts," (count (:indices m)) "indices")
    (r/run! rnd
            {:init-state {:t 0.0}
             :clear-color [0.5 0.7 0.95 1.0]
             :update (fn [s _input dt] (update s :t + dt))
             :render (fn [s frame] (world/draw-terrain! rnd frame pipeline mesh tex (:t s) aspect))})))

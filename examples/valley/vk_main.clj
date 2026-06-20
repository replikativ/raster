(ns valley.vk-main
  "JVM Vulkan shell for the valley terrain slice. The terrain kernels run as
   bytecode; the SAME valley.slice (.cljc) renders in the browser via WebGPU + wasm.
   Run:  clojure -M:vulkan -m valley.vk-main"
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [valley.slice :as slice]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Valley — Vulkan (raster.render)")
        pipeline (r/make-pipeline rnd slice/pipeline-spec)
        m        (slice/mesh-terrain)
        mesh     (r/make-static-mesh rnd (:verts m) (:indices m) slice/STRIDE)
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                            :pixels (slice/atlas-pixels 16 16) :filter :nearest})
        aspect   (/ 1024.0 768.0)]
    (println "valley mesh:" (quot (count (:verts m)) 6) "verts," (count (:indices m)) "indices,"
             "heights" (:lo m) "-" (:hi m))
    (r/run! rnd
            {:init-state {:t 0.0}
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [s _input dt] (update s :t + dt))
             :render (fn [s frame] (slice/draw-terrain! rnd frame pipeline mesh tex (:t s) aspect (:lo m) (:hi m)))})))

(ns valley.web
  "WebGPU browser shell for the valley terrain slice. terrain-height runs as wasm
   (generated valley.kernels, perms in wasm memory); valley.slice meshes the terrain
   by calling it, then renders via the WebGPU backend — the same .cljc as the JVM
   Vulkan shell (vk_main)."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [valley.slice :as slice]
            [valley.kernels :as k]))

(defn ^:export init []
  (-> (k/init! "/valley_kernels.wasm")
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd slice/pipeline-spec)
                     m        (slice/mesh-terrain)
                     mesh     (r/make-static-mesh rnd (:verts m) (:indices m) slice/STRIDE)
                     tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                                        :pixels (slice/atlas-pixels 16 16) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     st       (atom 0.0)]
                 (set! (.-__valley js/window)
                       #js {:t (fn [] @st) :verts (quot (count (:verts m)) 6)
                            :lo (:lo m) :hi (:hi m)})
                 (r/run! rnd
                         {:init-state 0.0
                          :clear-color [0.55 0.75 0.95 1.0]
                          :update (fn [t _input dt] (reset! st (+ t dt)))
                          :render (fn [t frame] (slice/draw-terrain! rnd frame pipeline mesh tex t aspect (:lo m) (:hi m)))}))))
      (.catch (fn [e] (js/console.error "valley init failed:" e)))))

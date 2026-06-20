(ns voxel.web
  "WebGPU browser shell for the voxel slice. terrain-height runs as wasm (generated
   voxel.kernels); voxel.world meshes the chunk by calling it, then renders via the
   WebGPU backend — same .cljc as the JVM Vulkan shell."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [voxel.world :as world]
            [voxel.kernels :as k]))

(defn ^:export init []
  (-> (k/init! "/voxel_kernels.wasm")
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd world/pipeline-spec)
                     m        (world/mesh-terrain)
                     mesh     (r/make-static-mesh rnd (:verts m) (:indices m) world/STRIDE)
                     tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                                         :pixels (world/atlas-pixels 16 16) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     st       (atom 0.0)]
                 (set! (.-__voxel js/window)
                       #js {:t (fn [] @st) :verts (quot (count (:verts m)) 6)})
                 (r/run! rnd
                         {:init-state 0.0
                          :clear-color [0.5 0.7 0.95 1.0]
                          :update (fn [t _input dt] (reset! st (+ t dt)))
                          :render (fn [t frame] (world/draw-terrain! rnd frame pipeline mesh tex t aspect))}))))
      (.catch (fn [e] (js/console.error "voxel init failed:" e)))))

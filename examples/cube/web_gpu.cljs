(ns cube.web-gpu
  "WebGPU browser shell for the R4 cube validation scene — same cube.scene (.cljc)
   as the JVM Vulkan shell (cube.vk-main); only the backend differs."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [cube.scene :as scene]))

(defn ^:export init []
  (-> (gpu/make-renderer (.getElementById js/document "game"))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd scene/pipeline-spec)
                     mesh     (r/make-static-mesh rnd scene/cube-verts scene/cube-indices scene/STRIDE)
                     tex      (r/make-texture rnd {:width 64 :height 64 :pixels (scene/checker 64 64) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     st       (atom 0.0)]
                 ;; expose for headless validation
                 (set! (.-__cube js/window) #js {:t (fn [] @st)})
                 (r/run! rnd
                         {:init-state 0.0
                          :clear-color [0.05 0.05 0.08 1.0]
                          :update (fn [t _input dt] (reset! st (+ t dt)))
                          :render (fn [t frame] (scene/draw-cube! rnd frame pipeline mesh tex t aspect))}))))
      (.catch (fn [e] (js/console.error "cube init failed:" e)))))

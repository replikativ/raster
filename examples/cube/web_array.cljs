(ns cube.web-array
  "WebGPU browser shell for the R5a texture-array cube — same cube.scene (.cljc)."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [cube.scene :as scene]))

(defn ^:export init []
  (-> (gpu/make-renderer (.getElementById js/document "game"))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd scene/array-pipeline-spec)
                     mesh     (r/make-static-mesh rnd scene/cube-verts-arr scene/cube-indices scene/STRIDE-ARR)
                     tex      (r/make-texture-array rnd {:width 64 :height 64 :layers 2
                                                         :pixels (scene/atlas-pixels 64 64) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     st       (atom 0.0)]
                 (set! (.-__cube js/window) #js {:t (fn [] @st)})
                 (r/run! rnd
                         {:init-state 0.0
                          :clear-color [0.05 0.05 0.08 1.0]
                          :update (fn [t _input dt] (reset! st (+ t dt)))
                          :render (fn [t frame] (scene/draw-cube! rnd frame pipeline mesh tex t aspect))}))))
      (.catch (fn [e] (js/console.error "cube-array init failed:" e)))))

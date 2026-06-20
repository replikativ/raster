(ns valley.web-walk
  "WebGPU browser shell for the valley single-chunk walker (Phase 2b-A). The physics
   + terrain kernels run as wasm (generated valley.kernels, perms + chunk scratch in
   wasm memory); valley.walk drives the player and renders via WebGPU — the same
   .cljc as the JVM Vulkan walk shell (vk_walk). Controls: WASD move, arrows look."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [valley.slice :as slice]
            [valley.walk :as walk]
            [valley.kernels :as k]))

(defn ^:export init []
  (-> (k/init! "/valley_kernels.wasm")
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd slice/pipeline-spec)
                     world    (walk/build-world)
                     mesh     (r/make-static-mesh rnd (:verts world) (:indices world) slice/STRIDE)
                     tex      (r/make-texture-array rnd {:width 16 :height 16 :layers 3
                                                        :pixels (slice/atlas-pixels 16 16) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     state    (atom (walk/player-init (:blocks world)))]
                 (set! (.-__valley js/window)
                       #js {:player (fn [] (clj->js {:pos (vec (:pos @state)) :vel (vec (:vel @state))
                                                     :yaw (:yaw @state) :on-ground (:on-ground @state)}))
                            :verts (quot (count (:verts world)) 6)})
                 (r/run! rnd
                         {:init-state @state
                          :clear-color [0.55 0.75 0.95 1.0]
                          :update (fn [p input dt] (reset! state (walk/player-update world p input dt)) @state)
                          :render (fn [p frame]
                                    (r/bind-pipeline! frame pipeline)
                                    (r/set-uniform! frame pipeline (walk/mvp p aspect))
                                    (r/bind-texture! frame pipeline tex)
                                    (r/bind-mesh! frame mesh)
                                    (r/draw-indexed! frame (:index-count mesh) 0))}))))
      (.catch (fn [e] (js/console.error "valley walk init failed:" e)))))

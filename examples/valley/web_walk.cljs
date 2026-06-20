(ns valley.web-walk
  "WebGPU browser shell for the valley multi-chunk walker + mobs (options C + mobs).
   Physics, terrain and the batch mob kernel run as wasm; the world block array is
   uploaded once into wasm linear memory (swarm/spawn → k/upload-world!) and mob SoA
   columns are swept by one batch kernel call each frame. The SAME valley.walk/valley.swarm
   (.cljc) run on the JVM (vk_walk). Controls: WASD move, arrows look."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [valley.slice :as slice]
            [valley.tex :as vtex]
            [valley.walk :as walk]
            [valley.swarm :as swarm]
            [valley.sky :as sky]
            [valley.kernels :as k]))

(def ^:const NMOBS 48)

(defn ^:export init []
  (-> (k/init! "/valley_kernels.wasm")
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd vtex/pipeline-spec)
                     world    (walk/build-grid 4 2 4)
                     mesh     (r/make-static-mesh rnd (:verts world) (:indices world) vtex/STRIDE)
                     tex      (r/make-texture-array rnd {:width 16 :height 16 :layers swarm/N-LAYERS
                                                        :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
                     aspect   (/ 1024.0 768.0)
                     player   (atom (walk/grid-player-init world))
                     swarm-st  (atom (swarm/spawn world NMOBS))   ; uploads resident world
                     mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) vtex/STRIDE)
                     sky-pipe (r/make-pipeline rnd sky/pipeline-spec)
                     sky-mesh (r/make-static-mesh rnd sky/cube-verts sky/cube-indices sky/STRIDE)
                     gt       (atom 6000.0)]
                 (set! (.-__valley js/window)
                       #js {:player (fn [] (clj->js {:pos (vec (:pos @player)) :vel (vec (:vel @player))
                                                    :on-ground (:on-ground @player)}))
                            :mob0   (fn [] (let [p (:pos @swarm-st)] #js [(aget p 0) (aget p 1) (aget p 2)]))
                            :mobs   (fn [] (:n @swarm-st))})
                 (r/run! rnd
                         {:init-state @player
                          :clear-color [0.55 0.75 0.95 1.0]
                          :update (fn [p input dt]
                                    (swap! gt sky/advance dt)
                                    (swap! swarm-st swarm/update! dt)
                                    (when (contains? input :fire)
                                      (let [pp (:pos p)] (swap! swarm-st swarm/cull-nearest! (aget pp 0) (aget pp 2))))
                                    (reset! player (walk/grid-player-update world p input dt)) @player)
                          :render (fn [p frame]
                                    (r/bind-pipeline! frame sky-pipe)
                                    (r/set-uniform! frame sky-pipe (sky/uniform @gt p aspect))
                                    (r/bind-mesh! frame sky-mesh)
                                    (r/draw-indexed! frame (:index-count sky-mesh) 0)
                                    (r/bind-pipeline! frame pipeline)
                                    (r/set-uniform! frame pipeline (walk/mvp p aspect))
                                    (r/bind-texture! frame pipeline tex)
                                    (r/bind-mesh! frame mesh)
                                    (r/draw-indexed! frame (:index-count mesh) 0)
                                    (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @swarm-st))]
                                      (r/bind-mesh! frame mm)
                                      (r/draw! frame (:vertex-count mm) 0)))}))))
      (.catch (fn [e] (js/console.error "valley walk init failed:" e)))))

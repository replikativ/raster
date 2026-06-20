(ns valley.web-walk
  "WebGPU browser shell for the valley multi-chunk walker + mobs + survival HUD + mining.
   Physics, terrain and the batch mob kernel run as wasm; the world block array is uploaded
   once into wasm linear memory (swarm/spawn → k/upload-world!) and mob SoA columns are swept
   by one batch kernel call each frame. The SAME valley.walk/valley.swarm/valley.ui (.cljc)
   run on the JVM (vk_walk).
   Controls: WASD move, arrows/mouse look (click to grab pointer), left-click break,
   right-click place, Q/E cycle hotbar, Space cull mob."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [valley.slice :as slice]
            [valley.tex :as vtex]
            [valley.chunk :as chunk]
            [valley.walk :as walk]
            [valley.swarm :as swarm]
            [valley.sky :as sky]
            [valley.ui :as ui]
            [valley.kernels :as k]))

(def ^:const NMOBS 48)

(defn ^:export init []
  (-> (k/init! "/valley_kernels.wasm")
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd vtex/pipeline-spec)
                     world    (walk/build-grid 4 2 4)
                     ;; skylight computed once; block edits re-mesh with it (newly dug cells
                     ;; read as dark — cave-like — which avoids a full relight on the main
                     ;; thread). Whole-world re-mesh is a brief hitch per edit (cljs has no
                     ;; worker thread here); per-chunk meshing is the future optimization.
                     sky0     (chunk/compute-skylight world)
                     wverts   (atom (chunk/tris (chunk/mesh-world world sky0) 9))
                     wmesh    (atom (r/upload-mesh! rnd (r/make-mesh rnd (* 3 (quot (count @wverts) 9)) vtex/STRIDE) @wverts))
                     tex      (r/make-texture-array rnd {:width 16 :height 16 :layers swarm/N-LAYERS
                                                        :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
                     canvas   (:canvas rnd)
                     aspect   (fn [] (/ (.-width canvas) (.-height canvas)))  ; live (responsive canvas)
                     player   (atom (walk/grid-player-init world))
                     swarm-st (atom (swarm/spawn world NMOBS))   ; uploads resident world
                     mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) vtex/STRIDE)
                     sky-pipe (r/make-pipeline rnd sky/pipeline-spec)
                     sky-mesh (r/make-static-mesh rnd sky/cube-verts sky/cube-indices sky/STRIDE)
                     hud-pipe (r/make-pipeline rnd ui/pipeline-spec)
                     hud-mesh (r/make-mesh rnd 4096 ui/STRIDE)
                     gt       (atom 6000.0)]
                 (set! (.-__valley js/window)
                       #js {:player (fn [] (clj->js {:pos (vec (:pos @player)) :vel (vec (:vel @player))
                                                    :on-ground (:on-ground @player) :sel (:sel @player)}))
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
                                    (reset! player (walk/grid-player-update world p input dt))
                                    ;; mining: edit + sync re-mesh (reuse sky0)
                                    (when (walk/apply-edits! world @player input)
                                      (reset! wverts (chunk/tris (chunk/mesh-world world sky0) 9))
                                      (reset! wmesh (r/upload-mesh! rnd @wmesh @wverts)))
                                    @player)
                          :render (fn [p frame]
                                    (r/bind-pipeline! frame sky-pipe)
                                    (r/set-uniform! frame sky-pipe (sky/uniform @gt p (aspect)))
                                    (r/bind-mesh! frame sky-mesh)
                                    (r/draw-indexed! frame (:index-count sky-mesh) 0)
                                    (r/bind-pipeline! frame pipeline)
                                    (r/set-uniform! frame pipeline
                                                    (into (walk/mvp p (aspect))
                                                          [(sky/day-light @gt) 0.18 0.0 0.0]))  ; env: dayRatio, ambient
                                    (r/bind-texture! frame pipeline tex)
                                    (r/bind-mesh! frame @wmesh)
                                    (r/draw! frame (:vertex-count @wmesh) 0)
                                    (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @swarm-st))]
                                      (r/bind-mesh! frame mm)
                                      (r/draw! frame (:vertex-count mm) 0))
                                    ;; HUD overlay (depth off, blended)
                                    (r/bind-pipeline! frame hud-pipe)
                                    (r/set-uniform! frame hud-pipe [1.0 1.0 1.0 1.0])
                                    (r/bind-texture! frame hud-pipe tex)
                                    (let [hm (r/upload-mesh! rnd hud-mesh (ui/verts p (aspect)))]
                                      (r/bind-mesh! frame hm)
                                      (r/draw! frame (:vertex-count hm) 0)))}))))
      (.catch (fn [e] (js/console.error "valley walk init failed:" e)))))

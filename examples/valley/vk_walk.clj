(ns valley.vk-walk
  "JVM Vulkan shell for the valley multi-chunk walker + mobs (options C + mobs). Physics,
   terrain and the batch mob kernel run as bytecode; the SAME valley.walk/valley.swarm
   (.cljc) run in the browser via WebGPU + wasm (web_walk). The world spans several 16³
   chunks (player uses a per-frame window stitch); mobs are resident SoA columns swept by
   one batch kernel call each frame and drawn as a dynamic cube mesh.
   Run:  clojure -M:vulkan -m valley.vk-walk   Controls: WASD move, arrows look."
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [valley.slice :as slice]
            [valley.tex :as vtex]
            [valley.chunk :as chunk]
            [valley.walk :as walk]
            [valley.swarm :as swarm]
            [valley.sky :as sky]
            [valley.ui :as ui]))

(def ^:const NMOBS 48)

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Valley — Walk + Mobs (raster.render)")
        pipeline (r/make-pipeline rnd vtex/pipeline-spec)
        world    (walk/build-grid 4 2 4)            ; 64×32×64 biome world
        ;; editable world mesh: a dynamic triangle-list re-uploaded after block edits.
        wverts   (atom (chunk/tris (chunk/mesh-world world) 9))
        wmesh    (atom (r/upload-mesh! rnd (r/make-mesh rnd (* 3 (quot (count @wverts) 9)) vtex/STRIDE) @wverts))
        dirty    (atom false) busy (atom false)     ; async re-mesh: edit instant, mesh swaps when ready
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers swarm/N-LAYERS
                                            :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
        mob-st   (atom (swarm/spawn world NMOBS))
        mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) vtex/STRIDE)
        sky-pipe (r/make-pipeline rnd sky/pipeline-spec)
        sky-mesh (r/make-static-mesh rnd sky/cube-verts sky/cube-indices sky/STRIDE)
        hud-pipe (r/make-pipeline rnd ui/pipeline-spec)
        hud-mesh (r/make-mesh rnd 4096 ui/STRIDE)
        gt       (atom 6000.0)
        aspect   (/ 1024.0 768.0)]
    (println "valley world mesh:" (quot (count @wverts) 9) "verts," (:wx world) "x" (:wy world) "x" (:wz world)
             "blocks," NMOBS "mobs")
    (r/run! rnd
            {:init-state (walk/grid-player-init world)
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [player input dt]
                       (swap! gt sky/advance dt)
                       (swap! mob-st swarm/update! dt)
                       (when (contains? input :fire)
                         (let [p (:pos player)] (swap! mob-st swarm/cull-nearest! (aget p 0) (aget p 2))))
                       ;; mining: edit the block array now (instant — collision/raycast see it),
                       ;; rebuild the mesh+light off-thread so the frame never hitches.
                       (when (and (walk/apply-edits! world player input) (compare-and-set! busy false true))
                         (future (let [v (chunk/tris (chunk/mesh-world world (chunk/compute-skylight world)) 9)]
                                   (reset! wverts v) (reset! dirty true) (reset! busy false))))
                       (walk/grid-player-update world player input dt))
             :render (fn [player frame]
                       (when (compare-and-set! dirty true false)
                         (reset! wmesh (r/upload-mesh! rnd @wmesh @wverts)))
                       ;; sky first (depth off) — terrain draws over it
                       (r/bind-pipeline! frame sky-pipe)
                       (r/set-uniform! frame sky-pipe (sky/uniform @gt player aspect))
                       (r/bind-mesh! frame sky-mesh)
                       (r/draw-indexed! frame (:index-count sky-mesh) 0)
                       (r/bind-pipeline! frame pipeline)
                       (r/set-uniform! frame pipeline
                                       (into (walk/mvp player aspect)
                                             [(sky/day-light @gt) 0.18 0.0 0.0]))  ; env: dayRatio, ambient
                       (r/bind-texture! frame pipeline tex)
                       (r/bind-mesh! frame @wmesh)
                       (r/draw! frame (:vertex-count @wmesh) 0)
                       (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @mob-st))]
                         (r/bind-mesh! frame mm)
                         (r/draw! frame (:vertex-count mm) 0))
                       ;; HUD overlay (depth off, blended)
                       (r/bind-pipeline! frame hud-pipe)
                       (r/set-uniform! frame hud-pipe [1.0 1.0 1.0 1.0])
                       (r/bind-texture! frame hud-pipe tex)
                       (let [hm (r/upload-mesh! rnd hud-mesh (ui/verts player aspect))]
                         (r/bind-mesh! frame hm)
                         (r/draw! frame (:vertex-count hm) 0)))})))

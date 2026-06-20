(ns valley.vk-walk
  "JVM Vulkan shell for the valley INFINITE streaming walker + mobs + survival HUD + mining.
   Terrain is generated on demand around the player into a moving active buffer (chunk/make-
   stream + recenter-stream); per-column meshes (absolute world coords) stream in/out of a
   render ring with a GPU-buffer pool. Physics, batch mob kernel and meshing all read the
   active buffer. The SAME valley.walk/swarm/ui (.cljc) run in the browser (web_walk).
   Run:  clojure -M:vulkan -m valley.vk-walk   Controls: WASD move, arrows/mouse look,
   left=break right=place, Q/E hotbar, Space cull mob."
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
(def ^:const RG 5)        ; data radius (chunks) — active buffer covers ±RG
(def ^:const R 4)         ; mesh/render radius (< RG so meshed columns have neighbours)
(def ^:const BUDGET 2)    ; columns meshed per frame (spread pop-in)

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768 :title "Valley — Streaming (raster.render)")
        pipeline (r/make-pipeline rnd vtex/pipeline-spec)
        stream   (atom (chunk/make-stream 0 0 RG))
        meshes   (atom {})                              ; [cx cz] -> gpu mesh
        pool     (atom [])                              ; recycled gpu meshes (bounds buffers)
        pending  (atom (vec (chunk/ring-cols 0 0 R)))   ; keys awaiting mesh
        mesh-one! (fn [[cx cz :as k]]
                    (let [v    (chunk/mesh-stream-column @stream cx cz)
                          base (or (first @pool) (r/make-mesh rnd chunk/MAX-COLUMN-VERTS vtex/STRIDE))]
                      (swap! pool #(if (seq %) (subvec % 1) %))
                      (swap! meshes assoc k (r/upload-mesh! rnd base v))))
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers swarm/N-LAYERS
                                            :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
        mob-st   (atom (swarm/spawn @stream NMOBS))
        mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) vtex/STRIDE)
        sky-pipe (r/make-pipeline rnd sky/pipeline-spec)
        sky-mesh (r/make-static-mesh rnd sky/cube-verts sky/cube-indices sky/STRIDE)
        hud-pipe (r/make-pipeline rnd ui/pipeline-spec)
        hud-mesh (r/make-mesh rnd 4096 ui/STRIDE)
        gt       (atom 6000.0)
        aspect   (/ 1024.0 768.0)]
    (println "valley streaming: active buffer" (:wx @stream) "x" (:wy @stream) "x" (:wz @stream)
             "blocks, mesh ring" R "(" (count @pending) "cols)," NMOBS "mobs")
    (r/run! rnd
            {:init-state (walk/stream-player-init @stream)
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [player input dt]
                       (swap! gt sky/advance dt)
                       ;; recenter the stream + ring when the player crosses a chunk
                       (let [p (:pos player) [pcx pcz] (chunk/chunk-key (aget p 0) (aget p 2))]
                         (when (not= [pcx pcz] (:center @stream))
                           (swap! stream chunk/recenter-stream pcx pcz)
                           (let [{:keys [add remove]} (walk/ring-update (keys @meshes) pcx pcz R)]
                             (doseq [k remove] (swap! pool conj (@meshes k)) (swap! meshes dissoc k))
                             (swap! pending (fn [pend] (into pend (clojure.core/remove (set pend) add)))))))
                       ;; budgeted meshing of pending columns
                       (dotimes [_ BUDGET]
                         (when-let [k (first @pending)] (swap! pending subvec 1) (mesh-one! k)))
                       (swap! mob-st swarm/update! @stream dt)
                       (when (contains? input :fire)
                         (let [p (:pos player)] (swap! mob-st swarm/cull-nearest! (aget p 0) (aget p 2))))
                       ;; mining: edit + re-mesh affected loaded columns
                       (when-let [[ex _ ez] (walk/apply-edits! @stream player input)]
                         (doseq [k (chunk/edit-columns ex ez)] (when (@meshes k) (mesh-one! k))))
                       (walk/grid-player-update @stream player input dt))
             :render (fn [player frame]
                       (r/bind-pipeline! frame sky-pipe)
                       (r/set-uniform! frame sky-pipe (sky/uniform @gt player aspect))
                       (r/bind-mesh! frame sky-mesh)
                       (r/draw-indexed! frame (:index-count sky-mesh) 0)
                       (r/bind-pipeline! frame pipeline)
                       (r/set-uniform! frame pipeline
                                       (into (walk/mvp player aspect) [(sky/day-light @gt) 0.18 0.0 0.0]))
                       (r/bind-texture! frame pipeline tex)
                       (doseq [[_ m] @meshes]
                         (r/bind-mesh! frame m)
                         (r/draw! frame (:vertex-count m) 0))
                       (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @mob-st))]
                         (r/bind-mesh! frame mm)
                         (r/draw! frame (:vertex-count mm) 0))
                       (r/bind-pipeline! frame hud-pipe)
                       (r/set-uniform! frame hud-pipe [1.0 1.0 1.0 1.0])
                       (r/bind-texture! frame hud-pipe tex)
                       (let [hm (r/upload-mesh! rnd hud-mesh (ui/verts player aspect))]
                         (r/bind-mesh! frame hm)
                         (r/draw! frame (:vertex-count hm) 0)))})))

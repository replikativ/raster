(ns valley.shell
  "Shared streaming game shell for JVM (Vulkan) and browser (WebGPU). Given an already-
   constructed renderer (the raster.render protocol), it runs the ENTIRE infinite-streaming
   valley: terrain stream + render ring (mesh pool + per-frame budget) + mobs + HUD + mining +
   fly. This is the cross-platform core — both vk_walk (JVM) and web_walk (browser) are thin
   entry points that build a renderer and call `run!`.

   The only platform-divergent call is k/upload-world! — a no-op on the JVM (the batch kernel
   reads the active buffer array directly) and a copy into wasm linear memory on the browser
   (so the wasm batch kernel sees the current ring). The shell calls it after every recenter."
  (:require [raster.render :as r]
            [valley.tex :as vtex]
            [valley.chunk :as chunk]
            [valley.walk :as walk]
            [valley.swarm :as swarm]
            [valley.sky :as sky]
            [valley.ui :as ui]
            [valley.water :as water]
            [valley.kernels :as k]))

(def ^:const NMOBS 48)
(def ^:const WATER-CAP 8192)   ; per-column water verts (surface faces only — small)

(defn start!
  "Run the streaming valley on renderer `rnd`. opts:
     :aspect    (fn [] aspect)   — viewport aspect (constant on JVM, live canvas in browser)
     :on-state  (fn [getters])   — optional; receives {:player :mobs :mob0} thunks (browser debug)"
  [rnd {:keys [aspect on-state] :or {aspect (constantly (/ 4.0 3.0))}}]
  (let [pipeline (r/make-pipeline rnd vtex/pipeline-spec)
        stream   (atom (chunk/make-stream 0 0 chunk/STREAM-RG))
        meshes   (atom {}) pool (atom [])           ; opaque terrain meshes + buffer pool
        wmeshes  (atom {}) wpool (atom [])          ; water sub-meshes (only columns with water)
        pending  (atom (vec (chunk/ring-cols 0 0 chunk/STREAM-R)))
        mesh-one! (fn [[cx cz :as kk]]
                    (let [{:keys [opaque water]} (chunk/mesh-stream-column @stream cx cz)
                          base (or (first @pool) (r/make-mesh rnd chunk/MAX-COLUMN-VERTS vtex/STRIDE))]
                      (swap! pool #(if (seq %) (subvec % 1) %))
                      (swap! meshes assoc kk (r/upload-mesh! rnd base opaque))
                      (if (seq water)
                        (let [wbase (or (first @wpool) (r/make-mesh rnd WATER-CAP water/STRIDE))]
                          (swap! wpool #(if (seq %) (subvec % 1) %))
                          (swap! wmeshes assoc kk (r/upload-mesh! rnd wbase water)))
                        (when-let [wm (@wmeshes kk)] (swap! wpool conj wm) (swap! wmeshes dissoc kk)))))
        tex      (r/make-texture-array rnd {:width 16 :height 16 :layers swarm/N-LAYERS
                                            :pixels (swarm/atlas-pixels 16 16) :filter :nearest})
        mob-st   (atom (swarm/spawn @stream NMOBS))      ; uploads active buffer (browser)
        mob-mesh (r/make-mesh rnd (* NMOBS swarm/VERTS-PER-MOB) vtex/STRIDE)
        sky-pipe (r/make-pipeline rnd sky/pipeline-spec)
        sky-mesh (r/make-static-mesh rnd sky/cube-verts sky/cube-indices sky/STRIDE)
        water-pipe (r/make-pipeline rnd water/pipeline-spec)
        hud-pipe (r/make-pipeline rnd ui/pipeline-spec)
        hud-mesh (r/make-mesh rnd 4096 ui/STRIDE)
        gt       (atom 6000.0)
        wt       (atom 0.0)        ; wave clock (seconds) for water animation
        player   (atom (walk/stream-player-init @stream))]
    (when on-state
      (on-state {:player (fn [] @player) :mobs (fn [] (:n @mob-st))
                 :mob0   (fn [] (:pos @mob-st))}))
    (r/run! rnd
            {:init-state @player
             :clear-color [0.55 0.75 0.95 1.0]
             :update (fn [pl input dt]
                       (swap! gt sky/advance dt)
                       (swap! wt + dt)
                       ;; recenter the stream + ring when the player crosses a chunk
                       (let [p (:pos pl) [pcx pcz] (chunk/chunk-key (aget p 0) (aget p 2))]
                         (when (not= [pcx pcz] (:center @stream))
                           (swap! stream chunk/recenter-stream pcx pcz)
                           (k/upload-world! (:blocks @stream) (:solid @stream))   ; unified divergence
                           (let [{:keys [add remove]} (walk/ring-update (keys @meshes) pcx pcz chunk/STREAM-R)]
                             (doseq [kk remove]
                               (swap! pool conj (@meshes kk)) (swap! meshes dissoc kk)
                               (when-let [wm (@wmeshes kk)] (swap! wpool conj wm) (swap! wmeshes dissoc kk)))
                             (swap! pending (fn [pend] (into pend (clojure.core/remove (set pend) add)))))))
                       ;; budgeted meshing of pending columns
                       (dotimes [_ chunk/STREAM-BUDGET]
                         (when-let [kk (first @pending)] (swap! pending subvec 1) (mesh-one! kk)))
                       (swap! mob-st swarm/update! @stream dt)
                       (when (contains? input :fire)
                         (let [p (:pos pl)] (swap! mob-st swarm/cull-nearest! (aget p 0) (aget p 2))))
                       (when-let [[ex _ ez] (walk/apply-edits! @stream pl input)]
                         (doseq [kk (chunk/edit-columns ex ez)] (when (@meshes kk) (mesh-one! kk))))
                       (reset! player (walk/player-update @stream pl input dt))
                       @player)
             :render (fn [pl frame]
                       (let [asp (aspect)]
                         (r/bind-pipeline! frame sky-pipe)
                         (r/set-uniform! frame sky-pipe (sky/uniform @gt pl asp))
                         (r/bind-mesh! frame sky-mesh)
                         (r/draw-indexed! frame (:index-count sky-mesh) 0)
                         (r/bind-pipeline! frame pipeline)
                         (r/set-uniform! frame pipeline
                                         (into (walk/mvp pl asp) [(sky/day-light @gt) 0.18 0.0 0.0]))
                         (r/bind-texture! frame pipeline tex)
                         (doseq [[_ m] @meshes]
                           (r/bind-mesh! frame m)
                           (r/draw! frame (:vertex-count m) 0))
                         (let [mm (r/upload-mesh! rnd mob-mesh (swarm/verts @mob-st))]
                           (r/bind-mesh! frame mm)
                           (r/draw! frame (:vertex-count mm) 0))
                         ;; transparent water pass (after opaque terrain) — animated wave shader
                         (when (seq @wmeshes)
                           (r/bind-pipeline! frame water-pipe)
                           (r/set-uniform! frame water-pipe
                                           (water/uniform (walk/mvp pl asp) (:pos (walk/player-cam pl))
                                                          @wt (sky/sky-colors @gt)))
                           (doseq [[_ wm] @wmeshes]
                             (r/bind-mesh! frame wm)
                             (r/draw! frame (:vertex-count wm) 0)))
                         (r/bind-pipeline! frame hud-pipe)
                         (r/set-uniform! frame hud-pipe [1.0 1.0 1.0 1.0])
                         (r/bind-texture! frame hud-pipe tex)
                         (let [hm (r/upload-mesh! rnd hud-mesh (ui/verts pl asp))]
                           (r/bind-mesh! frame hm)
                           (r/draw! frame (:vertex-count hm) 0))))})))

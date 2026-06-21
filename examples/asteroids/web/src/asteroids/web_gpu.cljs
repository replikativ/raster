(ns asteroids.web-gpu
  "WebGPU browser shell for cross-platform Geometric Asteroids. Renders the SAME
   asteroids.game (.cljc) through the SAME asteroids.render (.cljc) scene code as
   the JVM Vulkan shell (asteroids.vk-main) — only the renderer backend differs
   (raster.render.webgpu here, raster.render.vulkan there). The physics is wasm
   via the generated asteroids.kernels. The genuinely-cljs part is just: get the
   canvas, init the kernels + WebGPU device, then hand the loop to raster.render."
  (:require [raster.render :as r]
            [raster.render.webgpu :as gpu]
            [asteroids.render :as scene]
            [asteroids.game :as game]
            [asteroids.kernels :as k]))

(defn- update-hud! [rd]
  (let [{:keys [score lives wave over]} (:hud rd)]
    (set! (.-textContent (.getElementById js/document "hud"))
          (str "SCORE " score "    LIVES " lives "    WAVE " wave
               (when over "    —  GAME OVER (space)")))))

(defn ^:export init []
  (-> (k/init! "kernels.wasm")     ; relative → works at site root (dev) and /raster/asteroids/ (Pages)
      (.then (fn [_] (gpu/make-renderer (.getElementById js/document "game"))))
      (.then (fn [rnd]
               (let [pipeline (r/make-pipeline rnd scene/pipeline-spec)
                     mesh     (r/make-mesh rnd scene/MAX-VERTS scene/STRIDE)
                     ;; mirror state for headless validation
                     st       (atom (game/init-state))]
                 (set! (.-__ast js/window)
                       #js {:stats (fn [] (let [s @st]
                                            #js {:nShapes (count (:asteroids s)) :score (:score s)
                                                 :wave (:wave s) :lives (:lives s) :over (:over s)
                                                 :checksum (reduce (fn [c p] (+ c (* 13.1 (:x (:kin p))) (* 7.7 (:y (:kin p))))) 0.0 (:asteroids s))}))})
                 (r/run! rnd
                         {:init-state (game/init-state)
                          :clear-color [0.0 0.0 0.0 1.0]
                          :update (fn [s input dt] (reset! st (game/update-game s input dt)))
                          :render (fn [s frame]
                                    (let [rd (game/render-data s)]
                                      (update-hud! rd)
                                      (scene/draw-scene! rnd frame pipeline mesh rd)))}))))
      (.catch (fn [e] (js/console.error "WebGPU init failed:" e)))))

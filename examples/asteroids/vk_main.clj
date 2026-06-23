(ns asteroids.vk-main
  "JVM Vulkan shell for cross-platform Geometric Asteroids — the SAME game
   (asteroids.game .cljc) and the SAME scene code (asteroids.render .cljc) as the
   browser build, but rendered through the portable renderer's Vulkan backend
   instead of WebGPU. Demonstrates that one render path drives both APIs.

   Run:  clojure -Sdeps '{:paths [\"src\" \"examples\"]}' -M -m asteroids.vk-main"
  (:require [raster.render :as r]
            [raster.render.vulkan :as vkr]
            [asteroids.game :as game]
            [asteroids.render :as scene]))

(defn -main [& _]
  (let [rnd      (vkr/make-renderer :width 1024 :height 768
                                    :title "Geometric Asteroids — Vulkan (raster.render)")
        pipeline (r/make-pipeline rnd scene/pipeline-spec)
        mesh     (r/make-mesh rnd scene/MAX-VERTS scene/STRIDE)]
    (r/run! rnd
            {:init-state (game/init-state)
             :clear-color [0.0 0.0 0.0 1.0]
             :update (fn [state input dt] (game/update-game state input dt))
             :render (fn [state frame]
                       (scene/draw-scene! rnd frame pipeline mesh (game/render-data state)))})
    (vkr/destroy-renderer! rnd)))

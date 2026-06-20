(ns valley.vk-walk
  "JVM Vulkan entry point for the valley infinite-streaming walker. All game logic lives in
   the shared cross-platform shell (valley.shell); this just builds a Vulkan renderer and runs
   it. The browser entry point (valley.web-walk) is the same shape over WebGPU.
   Run:  clojure -M:vulkan -m valley.vk-walk
   Controls: WASD move, F fly, arrows/mouse look, left=break right=place, Q/E hotbar, Space cull."
  (:require [raster.render.vulkan :as vkr]
            [valley.shell :as shell]))

(defn -main [& _]
  (let [w 1024 h 768
        rnd (vkr/make-renderer :width w :height h :title "Valley — Streaming (raster.render)")]
    (shell/run! rnd {:aspect (constantly (/ (double w) h))})))

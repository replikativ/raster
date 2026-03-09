(ns doom.run
  "Entry point for launching the Doom renderer."
  (:require [doom.game :as game]))

(defn wireframe []
  (game/wireframe!))

(defn textured []
  (game/start!))

(defn -main [& args]
  (let [mode (first args)
        wad (or (second args) "/usr/share/games/doom/doom1.wad")
        map-name (or (nth args 2 nil) "E1M1")]
    (case mode
      "wireframe" (game/wireframe! :wad wad :map map-name)
      "textured" (game/start! :wad wad :map map-name)
      (game/start! :wad wad :map map-name))))

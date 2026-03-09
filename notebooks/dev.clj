(ns dev
  "Clay notebook build helpers.
   Usage from REPL:
     (require '[dev :reload])
     (make-book!)           ;; render all notebooks to docs/
     (make-single! \"notebooks/raster/ga_ode_rotors.clj\")  ;; render one"
  (:require [scicloj.clay.v2.api :as clay]))

(defn make-book!
  "Render all example notebooks as a Quarto book."
  []
  (clay/make! {:format [:quarto :html]
               :base-source-path "notebooks"
               :source-path ["raster/getting_started.clj"
                             "raster/ode_solvers.clj"
                             "raster/autodiff.clj"
                             "raster/linear_algebra.clj"
                             "raster/optimization.clj"
                             "raster/deep_learning.clj"
                             "raster/ga_ode_rotors.clj"
                             "raster/abm_firms.clj"]
               :base-target-path "docs"
               :book {:title "Raster Examples"}}))

(defn make-single!
  "Render a single notebook file."
  [path]
  (clay/make! {:format [:quarto :html]
               :source-path path
               :base-target-path ".clay-temp"
               :show true}))

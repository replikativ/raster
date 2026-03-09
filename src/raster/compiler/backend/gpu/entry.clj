(ns raster.compiler.backend.gpu.entry
  "Backend-specific compiler entry helpers for GPU-facing compilation APIs."
  (:require [raster.compiler.core.closure :as closure]))

(defn- effective-opencl-dtype
  "Choose the effective dtype for Level Zero targets.
  Defaults to :float for ze targets and :double otherwise."
  [dtype target-device]
  (or dtype
      (when (let [td (name target-device)]
              (or (.startsWith td "ze")
                  (.startsWith td ":ze")))
        :float)
      :double))



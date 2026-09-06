(ns raster.hardware-fixture
  "Isolated hardware registries for serial compiler tests.

   CI parallelizes across processes, not concurrent namespaces in one JVM. Like with-redefs,
   this fixture must not overlap unrelated work in the same process. It owns all three registry
   atoms, so initialization, synthetic targets and calibration cannot escape even on failure."
  (:require [raster.runtime.hardware]))

(defn isolated
  [f]
  (with-redefs-fn {#'raster.runtime.hardware/device-registry (atom {})
                  #'raster.runtime.hardware/initialized? (atom false)
                  #'raster.runtime.hardware/measured-registry (atom {})}
    f))

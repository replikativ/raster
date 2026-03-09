(ns raster.abm
  "Public entry point for ABM functionality.

	 The concrete built-in model lives under raster.abm.firms.
	 This namespace intentionally stays thin so shared runtime helpers can
	 remain here without conflating them with the firms model."
  (:require [raster.abm.firms]
            [raster.support :refer [import-vars]]))

(import-vars raster.abm.firms
             ->AgentSoA map->AgentSoA
             ->FirmSoA map->FirmSoA
             ->FirmsConfig map->FirmsConfig
             default-config init-simulation
             collect-stats run-period! run-period-parallel! run-simulation)
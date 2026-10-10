(ns raster.ad.fixtures.carrier-result-source
  (:require [raster.core :refer [deftm]])
  (:import [java.sql Date]
           [raster.ad.forward Dual]))

;; The caller imports java.util.Date. A result tag must retain this defining
;; namespace's class identity rather than resolve Date again in the caller.
(deftm date-result [x :- Double] :- Date (Date. 0))
(deftm date-result [x :- Dual] :- Date (Date. 0))

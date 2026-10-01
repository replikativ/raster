(ns raster.ode.finite-volume
  "Caller-owned conservative face transport expressed through ordinary typed programs.

   Geometry/topology construction and timestep selection belong to the caller. Each face
   has one flux, and each incident cell uses that same flux with opposite orientation.
   These kernels do not implement adaptation, subcycling or reflux. Storage must be
   disjoint where written; CSR bounds, face indices and positive volumes are caller contracts.
   Conservation additionally requires exactly two incidences per face, with +1 and -1
   orientation. These raw numerical kernels do not certify a caller's mesh topology."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]))

(deftm diffusive-face-fluxes!
  [flux :- (Array double), field :- (Array double), left :- (Array int),
   right :- (Array int), conductance :- (Array double), faces :- Long] :- Void
  (par/map-void! f faces
    (arrays/aset flux f
                 (* (arrays/aget conductance f)
                    (- (arrays/aget field (arrays/aget left f))
                       (arrays/aget field (arrays/aget right f)))))))

(deftm divergence-step!
  [out :- (Array double), field :- (Array double), flux :- (Array double),
   offsets :- (Array int), face-indices :- (Array int), orientation :- (Array double),
   inverse-volume :- (Array double), cells :- Long, dt :- Double] :- Void
  (par/map-void! c cells
    (let [start (arrays/aget offsets c)
          degree (- (arrays/aget offsets (inc c)) start)
          outward (loop [i 0 sum 0.0]
                    (if (< i degree)
                      (let [p (+ start i)]
                        (recur (inc i)
                               (+ sum (* (arrays/aget orientation p)
                                         (arrays/aget flux (arrays/aget face-indices p))))))
                      sum))]
      (arrays/aset out c
                   (- (arrays/aget field c)
                      (* dt (arrays/aget inverse-volume c) outward))))))

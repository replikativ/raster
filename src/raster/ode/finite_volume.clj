(ns raster.ode.finite-volume
  "Caller-owned conservative face transport expressed through ordinary typed programs.

   Geometry/topology construction and timestep selection belong to the caller. Each face
   has one flux, and each incident cell uses that same flux with opposite orientation.
   These kernels do not choose adaptation, a subcycle schedule or reflux ordering. Storage must be
   disjoint where written; CSR bounds, face indices and positive volumes are caller contracts.
   Conservation additionally requires exactly two incidences per face, with +1 and -1
   orientation. These raw numerical kernels do not certify a caller's mesh topology."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]))

(deftm temporal-boundary!
  "For nonzero coarse-owner cells, interpolate old/predicted values at theta; other cells
   retain current fine values. Endpoints copy exactly, including signed zero. The caller owns
   the bracket/time policy and supplies 0 <= theta <= 1, valid mask/extents and disjoint output.
   This is a numerical map, not a time/mesh certificate or an implicit boundary exchange."
  [out :- (Array double), old :- (Array double), predicted :- (Array double),
   current :- (Array double), coarse-owner :- (Array int), cells :- Long, theta :- Double] :- Void
  (par/map-void! c cells
                 (arrays/aset out c
                              (if (= 0 (arrays/aget coarse-owner c))
                                (arrays/aget current c)
                                (if (= theta 0.0)
                                  (arrays/aget old c)
                                  (if (= theta 1.0)
                                    (arrays/aget predicted c)
                                    (+ (* (- 1.0 theta) (arrays/aget old c))
                                       (* theta (arrays/aget predicted c)))))))))

(deftm accumulate-interface-transport!
  "Accumulate signed time-integrated extensive face transport on selected interfaces.
   Flux must already include face measure; dt supplies the signed time weight (negative for
   coarse prediction, positive for fine contributions). Each face has one writer. The caller
   clears the register, supplies valid interface mask/extents and orders all contributions
   before correction. CSR divergence with dt=1 applies the resulting register to coarse state;
   it must not be applied a second time to already advanced fine cells."
  [register :- (Array double), flux :- (Array double), interface :- (Array int),
   faces :- Long, dt :- Double] :- Void
  (par/map-void! f faces
                 (when (not= 0 (arrays/aget interface f))
                   (arrays/aset register f
                                (+ (arrays/aget register f) (* dt (arrays/aget flux f)))))))

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

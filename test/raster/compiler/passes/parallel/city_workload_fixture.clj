(ns raster.compiler.passes.parallel.city-workload-fixture
  "City-rstr's two full agent-day kernels, retained as a source-level compiler regression.
   Snapshot of probe.citykernel from 2026-09-25 (#765 validation)."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.par :as par]))

(deftm cast-coordinate-histogram!
  [values :- (Array double), counts :- (Array int), n :- Long] :- Void
  (par/map-void! i n
    (par/atomic-add! counts (int (* (aget values i) 8.0)) (int 1))))

(deftm episode-class-histogram!
  [locations :- (Array int), values :- (Array double), counts :- (Array int),
   n :- Long] :- Void
  (par/map-void! i n
    (do
      (par/atomic-add! counts 0 (int 1))
      (loop [e (int 0)]
        (when (< e 2)
          (let [location (int (aget locations (+ (* i 2) e)))]
            (do
              (when (== location (int 2))
                (let [u (aget values (+ (* i 2) e))
                      class (int (if (< u 0.5) 0 1))]
                  (par/atomic-add! counts (unchecked-add-int 1 class) (int 1))))
              (recur (int (inc e))))))))))

;; Literals are written out inside the kernels: a var reference, even a
;; ^:const one, reaches the C emitter as a bare symbol (`earth_m`).
(deftm haversine-m
  "Great-circle metres between two lon/lat points in degrees; the atan2 form,
   which every backend has, equal to `world/haversine-m`'s asin form."
  [lon1 :- Double, lat1 :- Double, lon2 :- Double, lat2 :- Double] :- Double
  (let [rlat1 (* lat1 0.017453292519943295) rlat2 (* lat2 0.017453292519943295)
        dlat (- rlat2 rlat1) dlon (* (- lon2 lon1) 0.017453292519943295)
        sdlat (Math/sin (* 0.5 dlat)) sdlon (Math/sin (* 0.5 dlon))
        a (+ (* sdlat sdlat) (* (Math/cos rlat1) (Math/cos rlat2) sdlon sdlon))]
    (* 12742000.0 (Math/atan2 (Math/sqrt a) (Math/sqrt (- 1.0 a))))))

(deftm weight
  "A_j^α (1 + d/d₀)^−β, the power kernel. A candidate with non-positive A is
   not in the choice set at any α — zero is how a class vector says a venue
   sells nothing of the class, or is closed — which is `reduce-dense`'s
   convention; α = 0 removes the size term of every candidate that is."
  [att :- Double, d :- Double, alpha :- Double, beta :- Double, d0 :- Double] :- Double
  (let [a (if (<= att 0.0) 0.0 (if (== alpha 0.0) 1.0 (if (== alpha 1.0) att (Math/pow att alpha))))
        x (+ 1.0 (/ d d0))]
    (* a (Math/pow x (- 0.0 beta)))))

(deftm cell-totals!
  "Z_c for every cell: the normaliser of the choice distribution."
  [cell-lon :- (Array double), cell-lat :- (Array double),
   cand-lon :- (Array double), cand-lat :- (Array double), cand-att :- (Array double),
   params :- (Array double), totals :- (Array double),
   n-cells :- Long, nc :- Long] :- Void
  (let [alpha (aget params 0) beta (aget params 1) d0 (aget params 2)]
    (par/map-void! c n-cells
      (let [lon (aget cell-lon c) lat (aget cell-lat c)]
        (loop [q (int 0) z 0.0]
          (if (< q nc)
            (recur (unchecked-add-int q 1)
                   (+ z (weight (aget cand-att q) (haversine-m lon lat (aget cand-lon q) (aget cand-lat q)) alpha beta d0)))
            (aset totals c z)))))))

(deftm retail-visits!
  "The retail half of `step-day` for every person `i`.

   `visits` is int[nc·24]: candidate index × hour, so it is the table-free
   analogue of `step-day`'s firm-indexed visits restricted to retail
   candidates. `counts` is int[2]: [diaries store-visits]."
  [ptype :- (Array int), home-cell :- (Array int), work :- (Array int), work-cell :- (Array int),
   type-offsets :- (Array int), type-cdf :- (Array double),
   diary-offsets :- (Array int), ep-loc :- (Array int), ep-minute :- (Array int),
   cell-lon :- (Array double), cell-lat :- (Array double),
   cand-lon :- (Array double), cand-lat :- (Array double), cand-att :- (Array double),
   totals :- (Array double), params :- (Array double),
   visits :- (Array int), counts :- (Array int),
   dims :- (Array long)] :- Void
  ;; `dims` = [n nc seed]: a Clojure fn takes at most twenty positional
  ;; arguments and this kernel reads nineteen arrays.
  (let [alpha (aget params 0) beta (aget params 1) d0 (aget params 2)
        n (aget dims 0) nc (aget dims 1) seed (aget dims 2)]
    (par/map-void! i n
      (let [t (aget ptype i) wj (aget work i) hc (aget home-cell i)]
        (when (>= (int (+ (if (>= t (int 0)) 1 0) (if (>= hc (int 0)) 1 (if (>= wj (int 0)) 1 0)))) (int 2))
          ;; Every integer local is cast: under a double element type the
          ;; walker types an unannotated local as double, and `double ^ ulong`
          ;; does not compile. raster's ABM kernels do the same.
          (let [a (int (aget type-offsets t)) b (int (aget type-offsets (unchecked-add-int t 1)))
                ;; splitmix inlined: raster lowers 64-bit arithmetic inside a
                ;; kernel body, not inside a helper with integer params (raster#613)
                ;; stream key splitmix(seed·1000003 + i), then the diary draw splitmix(key + 0)
                xs (long (unchecked-add (unchecked-multiply (long seed) 1000003) (long i)))
                zs (long (unchecked-add xs -7046029254386353131))
                zs (long (unchecked-multiply (bit-xor zs (unsigned-bit-shift-right zs 30)) -4658895280553007687))
                zs (long (unchecked-multiply (bit-xor zs (unsigned-bit-shift-right zs 27)) -7723592293110705685))
                key (long (bit-xor zs (unsigned-bit-shift-right zs 31)))
                z0 (long (unchecked-add key -7046029254386353131))
                z0 (long (unchecked-multiply (bit-xor z0 (unsigned-bit-shift-right z0 30)) -4658895280553007687))
                z0 (long (unchecked-multiply (bit-xor z0 (unsigned-bit-shift-right z0 27)) -7723592293110705685))
                z0 (long (bit-xor z0 (unsigned-bit-shift-right z0 31)))
                u0 (/ (double (unsigned-bit-shift-right z0 11)) 9007199254740992.0)
                ;; binary search the type's diary cdf, as step-day does
                d (int (loop [lo (int a) hi (int (unchecked-add-int b -1))]
                         (if (>= lo hi) lo
                             (let [m (int (quot (unchecked-add-int lo hi) 2))]
                               (if (< u0 (aget type-cdf m)) (recur lo m) (recur (int (unchecked-add-int m 1)) hi))))))
                e0 (int (aget diary-offsets d)) e1 (int (aget diary-offsets (unchecked-add-int d 1)))
                start (int (if (>= hc (int 0)) hc (aget work-cell wj)))]
            (par/atomic-add! counts 0 (int 1))
            (loop [e (int e0) anchor (int start)]
              (when (< e e1)
                (let [loc (int (aget ep-loc e))
                      h (int (rem (quot (aget ep-minute e) 60) 24))
                      k (int (unchecked-add-int e (- 0 e0)))]
                  (do (when (== loc (int 2)) (let [x1 (long (unchecked-add key (long (unchecked-add-int k 1))))
                              z1 (long (unchecked-add x1 -7046029254386353131))
                              z1 (long (unchecked-multiply (bit-xor z1 (unsigned-bit-shift-right z1 30)) -4658895280553007687))
                              z1 (long (unchecked-multiply (bit-xor z1 (unsigned-bit-shift-right z1 27)) -7723592293110705685))
                              z1 (long (bit-xor z1 (unsigned-bit-shift-right z1 31)))
                              u (/ (double (unsigned-bit-shift-right z1 11)) 9007199254740992.0)
                              z (aget totals anchor)
                              lon (aget cell-lon anchor) lat (aget cell-lat anchor)
                              ;; walk the candidates to the first cumulative share above u
                              target (* u z)
                              ;; if rounding leaves the target past the last increment, the
                              ;; last candidate with weight; -1 when the cell reaches none
                              j (int (loop [q (int 0) acc 0.0 last (int -1) hit (int -1)]
                                       (if (or (>= q (int nc)) (>= hit (int 0))) (if (>= hit (int 0)) hit last)
                                           (let [wq (weight (aget cand-att q) (haversine-m lon lat (aget cand-lon q) (aget cand-lat q)) alpha beta d0)
                                                 acc2 (+ acc wq)]
                                             (recur (int (unchecked-add-int q 1)) acc2 (int (if (> wq 0.0) q last))
                                                    (int (if (> wq 0.0) (if (< target acc2) q -1) -1)))))))]
                          (when (>= j (int 0))
                            (par/atomic-add! visits (int (unchecked-add-int (* j 24) h)) (int 1))
                            (par/atomic-add! counts 1 (int 1)))))
                      (recur (int (unchecked-add-int e 1)) (int (if (== loc (int 0)) (if (>= hc (int 0)) hc anchor) (if (== (int (+ (if (== loc (int 1)) 1 0) (if (>= wj (int 0)) 1 0))) (int 2)) (aget work-cell wj) anchor))))))))))))))

;; ---- money on the day ---------------------------------------------------------------------

(deftm spend-day!
  "The retail day with money and demand classes, for every person `i`.

   A store episode first draws its class `s` from `pi` (three trip shares,
   cumulative), then its venue from that class's kernel — `att3`, `totals3`
   and `params3` hold the three classes stacked, class-major — and the visit
   carries `spend3[s·n + i]`, the person's spend per visit in that class in
   cents, into `revenue3[(s·nc + j)·24 + h]`. `visits3` counts the same way.
   The class draw uses draw index `100000 + k` so the venue draw keeps the
   index `step-day` uses (`k + 1`) and the two never collide.

   `spend3` is the caller's: a person's annual class potential divided by
   their expected number of class-`s` store visits in a year, so that summing
   this day's revenue over 365 seeds reproduces the class potential exactly in
   expectation. Cents in a 32-bit int: a venue-hour cell would need 21 M € to
   overflow, and Mitte's largest venue takes about 0.2 M € a day.

   `shares` = [π₀ π₀+π₁ 1 | ℓ_short ℓ_medium ℓ_long]: the cumulative class
   shares of shopping trips, then the leakage share per class — the
   probability that a class-s purchase by a resident leaves the city. A
   leaked episode counts in `counts[5+s]` and places no visit and no money;
   its draw index is 200000 + k, and it applies to every person with a home
   cell. In-commuters carry their own `spend3`, sized by the caller from the
   class's measured net inflow. They too have a home cell, at their entry
   point on the boundary (`city.sim.cityworld/place-externals!`), so they
   take the leakage draw as well; that changes nothing as long as no class
   has both an inflow and a leakage share, which
   `city.econ.day/class-balances` guarantees.

   `counts` = [diaries store-visits class-short class-medium class-long
   leaked-short leaked-medium leaked-long].
   Coordinates are interleaved, `cell-xy[2c]` = lon, `cell-xy[2c+1]` = lat,
   so the kernel stays within twenty arguments."
  [ptype :- (Array int), home-cell :- (Array int), work :- (Array int), work-cell :- (Array int),
   type-offsets :- (Array int), type-cdf :- (Array double),
   diary-offsets :- (Array int), ep-loc :- (Array int), ep-minute :- (Array int),
   cell-xy :- (Array double), cand-xy :- (Array double),
   att3 :- (Array double), totals3 :- (Array double), params3 :- (Array double), shares :- (Array double),
   spend3 :- (Array int), revenue3 :- (Array int), visits3 :- (Array int), counts :- (Array int),
   dims :- (Array long)] :- Void
  (let [n (aget dims 0) nc (aget dims 1) seed (aget dims 2) ncell (aget dims 3)]
    (par/map-void! i n
      (let [t (aget ptype i) wj (aget work i) hc (aget home-cell i)]
        (when (>= (int (+ (if (>= t (int 0)) 1 0) (if (>= hc (int 0)) 1 (if (>= wj (int 0)) 1 0)))) (int 2))
          (let [a (int (aget type-offsets t)) b (int (aget type-offsets (unchecked-add-int t 1)))
                ;; stream key splitmix(seed·1000003 + i), then the diary draw splitmix(key + 0)
                xs (long (unchecked-add (unchecked-multiply (long seed) 1000003) (long i)))
                zs (long (unchecked-add xs -7046029254386353131))
                zs (long (unchecked-multiply (bit-xor zs (unsigned-bit-shift-right zs 30)) -4658895280553007687))
                zs (long (unchecked-multiply (bit-xor zs (unsigned-bit-shift-right zs 27)) -7723592293110705685))
                key (long (bit-xor zs (unsigned-bit-shift-right zs 31)))
                z0 (long (unchecked-add key -7046029254386353131))
                z0 (long (unchecked-multiply (bit-xor z0 (unsigned-bit-shift-right z0 30)) -4658895280553007687))
                z0 (long (unchecked-multiply (bit-xor z0 (unsigned-bit-shift-right z0 27)) -7723592293110705685))
                z0 (long (bit-xor z0 (unsigned-bit-shift-right z0 31)))
                u0 (/ (double (unsigned-bit-shift-right z0 11)) 9007199254740992.0)
                d (int (loop [lo (int a) hi (int (unchecked-add-int b -1))]
                         (if (>= lo hi) lo
                             (let [m (int (quot (unchecked-add-int lo hi) 2))]
                               (if (< u0 (aget type-cdf m)) (recur lo m) (recur (int (unchecked-add-int m 1)) hi))))))
                e0 (int (aget diary-offsets d)) e1 (int (aget diary-offsets (unchecked-add-int d 1)))
                start (int (if (>= hc (int 0)) hc (aget work-cell wj)))]
            (par/atomic-add! counts 0 (int 1))
            (loop [e (int e0) anchor (int start)]
              (when (< e e1)
                (let [loc (int (aget ep-loc e))
                      h (int (rem (quot (aget ep-minute e) 60) 24))
                      k (int (unchecked-add-int e (- 0 e0)))]
                  (do (when (== loc (int 2)) (let [;; class draw, index 100000 + k
                              xc (long (unchecked-add key (long (unchecked-add-int k 100000))))
                              zc (long (unchecked-add xc -7046029254386353131))
                              zc (long (unchecked-multiply (bit-xor zc (unsigned-bit-shift-right zc 30)) -4658895280553007687))
                              zc (long (unchecked-multiply (bit-xor zc (unsigned-bit-shift-right zc 27)) -7723592293110705685))
                              zc (long (bit-xor zc (unsigned-bit-shift-right zc 31)))
                              uc (/ (double (unsigned-bit-shift-right zc 11)) 9007199254740992.0)
                              s (int (if (< uc (aget shares 0)) 0 (if (< uc (aget shares 1)) 1 2)))
                              ;; leakage draw, index 200000 + k; residents only
                              xl (long (unchecked-add key (long (unchecked-add-int k 200000))))
                              zl (long (unchecked-add xl -7046029254386353131))
                              zl (long (unchecked-multiply (bit-xor zl (unsigned-bit-shift-right zl 30)) -4658895280553007687))
                              zl (long (unchecked-multiply (bit-xor zl (unsigned-bit-shift-right zl 27)) -7723592293110705685))
                              zl (long (bit-xor zl (unsigned-bit-shift-right zl 31)))
                              ul (/ (double (unsigned-bit-shift-right zl 11)) 9007199254740992.0)
                              leaked (int (if (>= hc (int 0)) (if (< ul (aget shares (unchecked-add-int 3 s))) 1 0) 0))
                              alpha (aget params3 (unchecked-add-int (* s 3) 0))
                              beta (aget params3 (unchecked-add-int (* s 3) 1))
                              d0 (aget params3 (unchecked-add-int (* s 3) 2))
                              ;; venue draw, index k + 1, as step-day
                              x1 (long (unchecked-add key (long (unchecked-add-int k 1))))
                              z1 (long (unchecked-add x1 -7046029254386353131))
                              z1 (long (unchecked-multiply (bit-xor z1 (unsigned-bit-shift-right z1 30)) -4658895280553007687))
                              z1 (long (unchecked-multiply (bit-xor z1 (unsigned-bit-shift-right z1 27)) -7723592293110705685))
                              z1 (long (bit-xor z1 (unsigned-bit-shift-right z1 31)))
                              u (/ (double (unsigned-bit-shift-right z1 11)) 9007199254740992.0)
                              z (aget totals3 (unchecked-add-int (* s ncell) anchor))
                              lon (aget cell-xy (* anchor 2)) lat (aget cell-xy (unchecked-add-int (* anchor 2) 1))
                              target (* u z)
                              abase (int (* s nc))
                              j (int (loop [q (int 0) acc 0.0 last (int -1) hit (int -1)]
                                       (if (or (>= q (int nc)) (>= hit (int 0))) (if (>= hit (int 0)) hit last)
                                           (let [wq (weight (aget att3 (unchecked-add-int abase q))
                                                            (haversine-m lon lat (aget cand-xy (* q 2)) (aget cand-xy (unchecked-add-int (* q 2) 1)))
                                                            alpha beta d0)
                                                 acc2 (+ acc wq)]
                                             (recur (int (unchecked-add-int q 1)) acc2 (int (if (> wq 0.0) q last))
                                                    (int (if (> wq 0.0) (if (< target acc2) q -1) -1)))))))
                              cell (int (unchecked-add-int (* (unchecked-add-int abase (int (if (>= j (int 0)) j 0))) 24) h))]
                          (if (== leaked (int 1))
                            (par/atomic-add! counts (unchecked-add-int 5 s) (int 1))
                            (when (>= j (int 0))
                              (par/atomic-add! visits3 cell (int 1))
                              (par/atomic-add! revenue3 cell (aget spend3 (unchecked-add-int (* s n) i)))
                              (par/atomic-add! counts 1 (int 1))
                              (par/atomic-add! counts (unchecked-add-int 2 s) (int 1))))))
                      (recur (int (unchecked-add-int e 1)) (int (if (== loc (int 0)) (if (>= hc (int 0)) hc anchor) (if (== (int (+ (if (== loc (int 1)) 1 0) (if (>= wj (int 0)) 1 0))) (int 2)) (aget work-cell wj) anchor))))))))))))))

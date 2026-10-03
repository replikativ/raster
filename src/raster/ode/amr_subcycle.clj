(ns raster.ode.amr-subcycle
  "Two-level ratio-2 diffusion geometry for a complete coarse prediction and fine substeps.
   Projection is numerical preparation, not a compiler, timestep, ownership or restart
   certificate. Only one strictly interior rectangular fine patch is admitted initially."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.validate :refer [fail! positive-number? non-negative-number?]]
            [raster.ode.amr-geometry :as geometry]
            [raster.linalg.sparse :as sparse]
            [raster.dl.array-ops :as ops]
            [raster.ode.finite-volume :as fv]
            [raster.ode.multilevel :as multilevel])
  (:import [raster.linalg.sparse CSRMatrix]))

(defn project
  "Derive complete coarse topology, fine interior/open-boundary topology, boundary donors
   and signed fine-face aggregation into coarse interface faces from one validated hierarchy.
   Rates include physical face measure. Registers subsequently multiply by time only.
   Fine faces are oriented left-to-right internally and fine-to-ghost at boundaries; retained
   signs map boundary rates into the coarse face orientation. Covered coarse rows remain
   present until reflux and average-down. :max-patch-cells bounds enumeration before allocation."
  [hierarchy {:keys [domain-lengths diffusivity max-patch-cells boundary]
              :or {diffusivity 1.0 max-patch-cells 1000000 boundary :periodic}}]
  (amr/validate-hierarchy! hierarchy)
  (when-not (and (= 2 (count (:base-shape hierarchy)))
                 (= 2 (count (:levels hierarchy)))
                 (every? #(= 1 (count (:patches %))) (:levels hierarchy)))
    (fail! "subcycling initially requires a 2D two-level single-fine-patch hierarchy"
           :amr-subcycle-hierarchy {}))
  (let [[base-level fine-level] (:levels hierarchy)
        base (first (:patches base-level)) fine (first (:patches fine-level))
        [nx ny] (:base-shape hierarchy) [ox oy] (:offsets fine) [fx fy] (:shape fine)
        total (+' (*' nx ny) (*' fx fy))]
    (when-not (and (= [2 2] (:ratio-to-parent fine-level))
                   (= [0 0] (:offsets base)) (= [nx ny] (:shape base))
                   (every? even? [ox oy fx fy])
                   (<= 2 ox) (<= 2 oy) (<= (+' ox fx) (- (*' 2 nx) 2))
                   (<= (+' oy fy) (- (*' 2 ny) 2)))
      (fail! "subcycling requires aligned ratio-2 refinement strictly inside the complete coarse domain"
             :amr-subcycle-patch {}))
    (when-not (and (= :periodic boundary)
                   (vector? domain-lengths) (= 2 (count domain-lengths))
                   (every? positive-number? domain-lengths) (non-negative-number? diffusivity)
                   (integer? max-patch-cells) (<= 1 max-patch-cells Integer/MAX_VALUE))
      (fail! "subcycling requires periodic boundaries, finite physical geometry and an Int patch-cell budget"
             :amr-subcycle-options {}))
    (when (or (> total max-patch-cells)
              (> (+' (*' 4 nx ny) (*' 4 fx fy) (*' 2 (+' fx fy))) Integer/MAX_VALUE))
      (fail! "subcycle geometry exceeds the row/incidence budget" :amr-subcycle-budget {}))
    (let [coarse-hierarchy (amr/hierarchy (assoc (into {} hierarchy) :levels [base-level]))
          coarse (geometry/project-hierarchy coarse-hierarchy
                                             {:domain-lengths domain-lengths :diffusivity diffusivity
                                              :boundary boundary
                                              :max-patch-cells max-patch-cells})
          coarse-faces (:faces coarse)
          face-index (into {} (map-indexed (fn [i f] [[(:axis f) (:plane f) (:start f)] i]) coarse-faces))
          spacing (mapv / (mapv double domain-lengths) [(* 2 nx) (* 2 ny)])
          nf (* fx fy) fine-id (fn [x y] (+ (* x fy) y))
          covered (vec (for [x (range (quot ox 2) (quot (+ ox fx) 2))
                             y (range (quot oy 2) (quot (+ oy fy) 2))] (+ (* x ny) y)))
          internal (vec (for [x (range fx) y (range fy) axis [0 1]
                              :when (< (if (zero? axis) (inc x) (inc y)) (if (zero? axis) fx fy))]
                          {:left (fine-id x y)
                           :right (fine-id (if (zero? axis) (inc x) x)
                                           (if (zero? axis) y (inc y)))
                           :axis axis :measure (spacing (- 1 axis))
                           :conductance (* diffusivity (/ (spacing (- 1 axis)) (spacing axis)))}))
          boundaries (vec
                      (for [axis [0 1] side [:low :high]
                            t (range (if (zero? axis) fy fx))
                            :let [low? (= :low side)
                                  plane (quot (if (zero? axis)
                                                (+ ox (if low? 0 fx)) (+ oy (if low? 0 fy))) 2)
                                  tangent (quot (+ (if (zero? axis) oy ox) t) 2)
                                  donor-axis (if low? (dec plane) plane)
                                  donor (if (zero? axis) (+ (* donor-axis ny) tangent)
                                            (+ (* tangent ny) donor-axis))
                                  x (if (zero? axis) (if low? 0 (dec fx)) t)
                                  y (if (zero? axis) t (if low? 0 (dec fy)))
                                  coarse-face (get face-index [axis plane tangent])]]
                        {:left (fine-id x y) :axis axis :side side :donor donor
                         :coarse-face coarse-face :coarse-sign (if low? -1.0 1.0)
                         :measure (spacing (- 1 axis))
                         :conductance (* diffusivity (/ (spacing (- 1 axis)) (* 1.5 (spacing axis))))}))
          boundaries (mapv #(assoc %2 :right (+ nf %1)) (range) boundaries)
          faces (into internal boundaries)
          rows (reduce-kv (fn [rows f {:keys [left right]}]
                            (cond-> (update rows left conj [f 1.0])
                              (< right nf) (update right conj [f -1.0])))
                          (vec (repeat nf [])) faces)
          aggregation (reduce-kv (fn [rows i {:keys [coarse-face coarse-sign]}]
                                   (when (nil? coarse-face)
                                     (fail! "fine boundary lacks its coarse interface face"
                                            :amr-subcycle-interface {}))
                                   (update rows coarse-face conj [(+ (count internal) i) coarse-sign]))
                                 (vec (repeat (count coarse-faces) [])) boundaries)
          inverse-volume (/ 1.0 (* (spacing 0) (spacing 1)))]
      (when-not (and (every? #(or (empty? %) (= 2 (count %))) aggregation)
                     (positive-number? inverse-volume)
                     (every? #(and (positive-number? (:measure %))
                                   (non-negative-number? (:conductance %))) faces))
        (fail! "subcycle interface partition or physical storage is not representable"
               :amr-subcycle-interface {}))
      {:schema-version 1 :hierarchy hierarchy :boundary boundary :domain-lengths (mapv double domain-lengths)
       :diffusivity (double diffusivity) :time-ratio 2 :flux-units :face-integrated-rate
       :coarse coarse :fine-shape [fx fy] :fine-cell-count nf :fine-faces faces
       :fine-incidences rows :fine-inverse-volume inverse-volume :covered-coarse covered
       :boundary-donors (mapv :donor boundaries) :coarse-interface-rows aggregation})))

(defn materialize
  "Return fresh arrays for existing finite-volume, gather/scatter and CSR source operations.
   Input is project output; this grants no independent certification or runtime ownership."
  [{:keys [coarse fine-faces fine-incidences fine-cell-count fine-inverse-volume
           covered-coarse boundary-donors coarse-interface-rows]}]
  (let [nz (vec (mapcat identity coarse-interface-rows))]
    {:coarse (geometry/materialize-connectivity coarse)
     :fine {:left (int-array (map :left fine-faces)) :right (int-array (map :right fine-faces))
            :conductance (double-array (map :conductance fine-faces))
            :offsets (int-array (reductions + 0 (map count fine-incidences)))
            :indices (int-array (map first (mapcat identity fine-incidences)))
            :orientation (double-array (map second (mapcat identity fine-incidences)))
            :inverse-volume (double-array (repeat fine-cell-count fine-inverse-volume))
            :cell-count fine-cell-count :face-count (count fine-faces)}
     :covered-coarse (int-array covered-coarse) :boundary-donors (int-array boundary-donors)
     :ghost-indices (int-array (range fine-cell-count (+ fine-cell-count (count boundary-donors))))
     :fine-indices (int-array (range fine-cell-count))
     :coarse-owner (int-array (concat (repeat fine-cell-count 0) (repeat (count boundary-donors) 1)))
     :interface (int-array (map #(if (seq %) 1 0) coarse-interface-rows))
     :aggregation (sparse/csr-matrix
                   (int-array (reductions + 0 (map count coarse-interface-rows)))
                   (int-array (map first nz)) (double-array (map second nz))
                   (count coarse-interface-rows) (count fine-faces))}))

(deftm diffusion-cycle!
  "One ratio-2 cycle: full coarse prediction, two fine substeps, signed reflux, average-down.
   All topology/scratch is caller-owned and must come from the matching project/materialize
   layout. Inputs must be synchronized (covered coarse cells average fine values); dt must be
   stable for the explicit advances. No adaptation, multi-patch or mid-cycle restart is implied."
  [coarse :- (Array double), fine :- (Array double),
   coarse-out :- (Array double), fine-out :- (Array double),
   cl :- (Array int), cr :- (Array int), cc :- (Array double),
   co :- (Array int), ci :- (Array int), cs :- (Array double), cv :- (Array double),
   fl :- (Array int), fr :- (Array int), fc :- (Array double),
   fo :- (Array int), fi :- (Array int), fs :- (Array double), fv :- (Array double),
   donors :- (Array int), ghosts :- (Array int), fine-indices :- (Array int),
   owners :- (Array int), interface :- (Array int), covered :- (Array int),
   aggregate :- CSRMatrix,
   coarse-flux :- (Array double), predicted :- (Array double),
   old-ghost :- (Array double), predicted-ghost :- (Array double),
   old-full :- (Array double), predicted-full :- (Array double),
   current-0 :- (Array double), current-1 :- (Array double),
   boundary-0 :- (Array double), boundary-1 :- (Array double),
   flux-0 :- (Array double), flux-1 :- (Array double), fine-1 :- (Array double),
   aggregate-0 :- (Array double), aggregate-1 :- (Array double),
   register :- (Array double), averaged :- (Array double),
   nc :- Long, nf :- Long, ng :- Long, ncf :- Long, nff :- Long,
   sx :- Long, sy :- Long, dt :- Double] :- Void
  (fv/diffusive-face-fluxes! coarse-flux coarse cl cr cc ncf)
  (fv/divergence-step! predicted coarse coarse-flux co ci cs cv nc dt)
  (ops/gather-blocks! coarse donors old-ghost ng 1 nc)
  (ops/gather-blocks! predicted donors predicted-ghost ng 1 nc)
  (ops/scatter-blocks! old-ghost ghosts old-full ng 1 (+ nf ng))
  (ops/scatter-blocks! predicted-ghost ghosts predicted-full ng 1 (+ nf ng))
  (ops/scatter-blocks! fine fine-indices current-0 nf 1 (+ nf ng))
  (fv/temporal-boundary! boundary-0 old-full predicted-full current-0 owners (+ nf ng) 0.0)
  (fv/diffusive-face-fluxes! flux-0 boundary-0 fl fr fc nff)
  (fv/divergence-step! fine-1 boundary-0 flux-0 fo fi fs fv nf (* 0.5 dt))
  (ops/scatter-blocks! fine-1 fine-indices current-1 nf 1 (+ nf ng))
  (fv/temporal-boundary! boundary-1 old-full predicted-full current-1 owners (+ nf ng) 0.5)
  (fv/diffusive-face-fluxes! flux-1 boundary-1 fl fr fc nff)
  (fv/divergence-step! fine-out boundary-1 flux-1 fo fi fs fv nf (* 0.5 dt))
  (par/map-void! f ncf (arrays/aset register f 0.0))
  (fv/accumulate-interface-transport! register coarse-flux interface ncf (- dt))
  (sparse/spmv aggregate flux-0 aggregate-0 1.0 0.0)
  (sparse/spmv aggregate flux-1 aggregate-1 1.0 0.0)
  (fv/accumulate-interface-transport! register aggregate-0 interface ncf (* 0.5 dt))
  (fv/accumulate-interface-transport! register aggregate-1 interface ncf (* 0.5 dt))
  (fv/divergence-step! coarse-out predicted register co ci cs cv nc 1.0)
  (multilevel/restrict-average-2d! averaged fine-out sx sy)
  (ops/scatter-blocks! averaged covered coarse-out (* sx sy) 1 nc))

(defn cycle-inputs
  "Prepare fresh heap topology and scratch for diffusion-cycle!, without GPU allocation.
   Retains original input arrays; does not silently average/mutate them. The synchronized
  state and timestep-stability obligations remain explicit caller contracts."
  [projection coarse-state fine-state dt]
  (when-not (and (= :double (dtype/dtype-for-jvm-array coarse-state))
                 (= :double (dtype/dtype-for-jvm-array fine-state))
                 (= (count (get-in projection [:coarse :cells])) (count coarse-state))
                 (= (:fine-cell-count projection) (count fine-state)) (positive-number? dt))
    (fail! "cycle inputs require matching FP64 fields and a finite positive timestep"
           :amr-subcycle-inputs {}))
  (let [a (materialize projection) c (:coarse a) f (:fine a)
        nc (:cell-count c) nf (:cell-count f) ng (count (:boundary-donors projection))
        ncf (:face-count c) nff (:face-count f) [fx fy] (:fine-shape projection)
        coarse-out (double-array nc) fine-out (double-array nf)
        coarse-flux (double-array ncf) predicted (double-array nc)
        old-ghost (double-array ng) predicted-ghost (double-array ng)
        old-full (double-array (+ nf ng)) predicted-full (double-array (+ nf ng))
        current-0 (double-array (+ nf ng)) current-1 (double-array (+ nf ng))
        boundary-0 (double-array (+ nf ng)) boundary-1 (double-array (+ nf ng))
        flux-0 (double-array nff) flux-1 (double-array nff) fine-1 (double-array nf)
        aggregate-0 (double-array ncf) aggregate-1 (double-array ncf)
        register (double-array ncf) averaged (double-array (quot (* fx fy) 4))]
    {:arguments
     [coarse-state fine-state coarse-out fine-out
      (:left c) (:right c) (:conductance c) (:offsets c) (:indices c) (:orientation c) (:inverse-volume c)
      (:left f) (:right f) (:conductance f) (:offsets f) (:indices f) (:orientation f) (:inverse-volume f)
      (:boundary-donors a) (:ghost-indices a) (:fine-indices a) (:coarse-owner a) (:interface a)
      (:covered-coarse a) (:aggregation a)
      coarse-flux predicted old-ghost predicted-ghost old-full predicted-full current-0 current-1
      boundary-0 boundary-1 flux-0 flux-1 fine-1 aggregate-0 aggregate-1 register averaged
      nc nf ng ncf nff (quot fx 2) (quot fy 2) (double dt)]
     :outputs {:coarse coarse-out :fine fine-out}
     :scratch {:register register :predicted predicted :fine-1 fine-1 :coarse-flux coarse-flux
               :flux-0 flux-0 :flux-1 flux-1 :boundary-1 boundary-1}}))

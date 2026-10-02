(ns raster.ode.amr-geometry
  "Host-side 2D cell-average hierarchy projection for the existing finite-volume kernels.

   Geometry and patch-row provenance are immutable. Materialization returns fresh arrays, not
   ownership transfers or numerical certificates. Shared two-point face fluxes conserve mass;
   this does not assert spatial order at displaced coarse/fine centres, timestep stability,
   subcycling, reflux, adaptation or distributed execution."
  (:require [raster.compiler.ir.amr-plan :as amr]
            [raster.compiler.ir.validate :refer [fail! positive-number? non-negative-number?]]))

(defn- product [xs] (reduce *' 1 xs))

(defn- active-cells [hierarchy scales]
  (let [levels (:levels hierarchy) finest (peek scales)]
    (vec
     (mapcat
      (fn [level scale]
        (let [widths (mapv quot finest scale)
              ;; Coarsened covered rows, not finest-grid tiles. Their count is bounded by the
              ;; already checked child patch-row budget. Membership avoids O(cells * patches).
              covered (when-let [next-level (get levels (inc (:index level)))]
                        (into #{}
                              (mapcat (fn [patch]
                                        (let [[ox oy] (mapv quot (:offsets patch) (:ratio-to-parent next-level))
                                              [nx ny] (mapv quot (:shape patch) (:ratio-to-parent next-level))]
                                          (for [x (range ox (+' ox nx)) y (range oy (+' oy ny))] [x y])))
                                      (:patches next-level))))]
          (mapcat
           (fn [patch]
             (let [[nx ny] (:shape patch) [ox oy] (:offsets patch)]
               (for [x (range nx) y (range ny)
                     :let [point [(+' ox x) (+' oy y)]]
                     :when (not (contains? covered point))]
                 {:patch (:id patch) :field (:field patch) :device (:device patch)
                  :level (:index level) :local-index (+' (*' x ny) y)
                  :offsets (mapv *' point widths) :shape widths})))
           (:patches level))))
      levels scales))))

(defn- side-index [cells domain boundary]
  (reduce-kv
   (fn [planes cell {:keys [offsets shape]}]
     (reduce
      (fn [planes axis]
        (let [tangent (- 1 axis) lo (offsets axis) hi (+' lo (shape axis))
              start (offsets tangent) end (+' start (shape tangent))
              hi (if (and (= boundary :periodic) (= hi (domain axis))) 0 hi)
              side {:cell cell :start start :end end}]
          (-> planes
              (update-in [[axis lo] :low] (fnil conj []) side)
              (update-in [[axis hi] :high] (fnil conj []) side))))
      planes [0 1]))
   (sorted-map) cells))

(defn- match-sides
  "Merge two disjoint interval partitions. Each overlap is one geometric face, including
   separate periodic faces and self-neighbours. No all-pairs cell enumeration."
  [high low]
  (loop [high (seq (sort-by :start high)) low (seq (sort-by :start low)) faces []]
    (if (and high low)
      (let [h (first high) l (first low)
            start (max (:start h) (:start l)) end (min (:end h) (:end l))]
        (recur (if (<= (:end h) (:end l)) (next high) high)
               (if (<= (:end l) (:end h)) (next low) low)
               (if (< start end)
                 (conj faces {:left (:cell h) :right (:cell l) :start start :end end})
                 faces)))
      faces)))

(defn- faces [cells domain lengths boundary diffusivity]
  (vec
   (mapcat
    (fn [[[axis plane] {:keys [high low]}]]
      (for [{:keys [left right start end] :as face} (match-sides high low)
            :let [measure (* (double (- end start)) (/ (lengths (- 1 axis)) (domain (- 1 axis))))
                  distance (* 0.5 (+ (double (get-in cells [left :shape axis]))
                                    (double (get-in cells [right :shape axis])))
                              (/ (lengths axis) (domain axis)))
                  conductance (* diffusivity (/ measure distance))]]
        (assoc face :axis axis :plane plane :measure measure :conductance conductance)))
    (side-index cells domain boundary))))

(defn project-hierarchy
  "Project a validated 2D hierarchy into packed active rows and geometric faces.

   Level order, patch order and row-major local order determine packed order. Each row retains
   patch/field/device/local-index provenance. Covered coarse rows are absent, not overwritten.
   :max-patch-cells bounds row enumeration before allocation (including covered patch rows),
   not total validation time: the reused hierarchy validator still compares patch pairs.
   Boundaries are :periodic or homogeneous :no-flux. Domain lengths are positive physical
   lengths; diffusivity is one non-negative scalar. Other ranks/materials fail explicitly."
  [hierarchy {:keys [domain-lengths boundary diffusivity max-patch-cells]
              :or {boundary :periodic diffusivity 1.0 max-patch-cells 1000000}}]
  (amr/validate-hierarchy! hierarchy)
  (when-not (= 2 (count (:base-shape hierarchy)))
    (fail! "finite-volume hierarchy projection currently supports two dimensions"
           :amr-geometry-rank {:shape (:base-shape hierarchy)}))
  (when-not (and (vector? domain-lengths) (= 2 (count domain-lengths))
                 (every? positive-number? domain-lengths)
                 (contains? #{:periodic :no-flux} boundary)
                 (non-negative-number? diffusivity)
                 (integer? max-patch-cells) (<= 1 max-patch-cells Integer/MAX_VALUE))
    (fail! "geometry requires finite physical lengths/diffusivity, supported boundaries and an Int row budget"
           :amr-geometry-options {:domain-lengths domain-lengths :boundary boundary
                                  :diffusivity diffusivity :max-patch-cells max-patch-cells}))
  (let [levels (:levels hierarchy)
        patches (vec (mapcat :patches levels))
        count-upper-bound (reduce +' 0 (map #(product (:shape %)) patches))
        _ (when (> count-upper-bound max-patch-cells)
            (fail! "hierarchy patch enumeration exceeds the declared budget"
                   :amr-geometry-budget {:patch-cells count-upper-bound :budget max-patch-cells}))
        scales (reduce (fn [scales level]
                         (conj scales (mapv *' (peek scales) (:ratio-to-parent level))))
                       [[1 1]] (next levels))
        domain (mapv *' (:base-shape hierarchy) (peek scales))
        cells (active-cells hierarchy scales)
        _ (when-not (= (product domain) (reduce +' 0 (map #(product (:shape %)) cells)))
            (fail! "active cells do not cover the domain exactly"
                   :amr-geometry-active-coverage {}))
        lengths (mapv double domain-lengths)
        cell-volumes (mapv #(product (mapv (fn [n length total] (* (double n) (/ length total)))
                                          (:shape %) lengths domain)) cells)
        geometric-faces (faces cells domain lengths boundary (double diffusivity))
        _ (when (or (> (*' 2 (count geometric-faces)) Integer/MAX_VALUE)
                    (not-every? #(and (positive-number? %) (positive-number? (/ 1.0 %))) cell-volumes)
                    (not-every? #(and (positive-number? (:measure %))
                                     (non-negative-number? (:conductance %))) geometric-faces))
            (fail! "projected topology exceeds Int capacity or finite positive physical geometry"
                   :amr-geometry-capacity {}))
        incidences (reduce-kv
                    (fn [rows f {:keys [left right]}]
                      (-> rows (update left conj [f 1.0]) (update right conj [f -1.0])))
                    (vec (repeat (count cells) [])) geometric-faces)
        rows-by-patch (reduce-kv (fn [rows packed cell]
                                  (update rows (:patch cell) (fnil conj []) [(:local-index cell) packed]))
                                {} cells)
        patch-rows (mapv (fn [patch]
                          {:patch (:id patch) :field (:field patch) :device (:device patch)
                           :shape (:shape patch)
                           :rows (get rows-by-patch (:id patch) [])}) patches)]
    {:schema-version 1 :hierarchy hierarchy :domain domain :domain-lengths lengths
     :boundary boundary :diffusivity (double diffusivity)
     :cells cells :cell-volumes cell-volumes :faces geometric-faces
     :incidences incidences :patch-rows patch-rows}))

(defn materialize-connectivity
  "Return fresh arrays for finite-volume's existing FP64/Int ABI and explicit patch packing
   indices. Does not allocate device memory, transfer patch ownership, or choose a timestep.
   Input is the immutable output of project-hierarchy, not an independently certified raw mesh."
  [{:keys [cells cell-volumes faces incidences patch-rows]}]
  {:left (int-array (map :left faces)) :right (int-array (map :right faces))
   :conductance (double-array (map :conductance faces))
   :offsets (int-array (reductions + 0 (map count incidences)))
   :indices (int-array (map first (mapcat identity incidences)))
   :orientation (double-array (map second (mapcat identity incidences)))
   :inverse-volume (double-array (map #(/ 1.0 %) cell-volumes))
   :cell-count (count cells) :face-count (count faces)
   :patch-rows (mapv (fn [patch]
                      (-> patch
                          (dissoc :rows)
                          (assoc :local-indices (int-array (map first (:rows patch)))
                                 :packed-indices (int-array (map second (:rows patch)))))) patch-rows)})

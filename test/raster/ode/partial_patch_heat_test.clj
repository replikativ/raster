(ns raster.ode.partial-patch-heat-test
  "A synchronous conservative partial-patch oracle, not subcycled/refluxed AMR."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.ode.finite-volume :as fv]
            [raster.linalg.sparse :as sparse]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.link :as link]))

(defn- hierarchy
  ([] (hierarchy [{:offsets [2 2] :shape [4 4]}]))
  ([regions]
   (amr/hierarchy
    {:id :heat/partial :base-shape [4 4] :proper-nesting-width 1
     :levels
     [(amr/level {:id :coarse :index 0
                 :patches [(amr/patch {:id :base :level 0 :device :local
                                      :offsets [0 0] :shape [4 4] :field :base-temperature})]})
      (amr/level {:id :fine :index 1 :ratio-to-parent [2 2]
                  :patches (mapv (fn [i region]
                                   (amr/patch (merge region
                                                    {:id [:fine i] :level 1 :device :local
                                                     :field [:fine-temperature i]})))
                                 (range) regions)})]})))

(defn- active-cells [hierarchy]
  ;; Deliberately bounded test projection, not a public general AMR mesh builder.
  (amr/validate-hierarchy! hierarchy)
  (assert (and (= [4 4] (:base-shape hierarchy))
               (= 2 (count (:levels hierarchy)))
               (= [2 2] (:ratio-to-parent (second (:levels hierarchy))))))
  (let [patches (:patches (second (:levels hierarchy)))
        covered? (fn [x y]
                   (some (fn [{[px py] :offsets [sx sy] :shape}]
                           (and (<= px x) (< x (+ px sx))
                                (<= py y) (< y (+ py sy)))) patches))]
    (vec (concat
          (for [x (range 0 8 2) y (range 0 8 2) :when (not (covered? x y))]
            [x y 2])
          (mapcat (fn [{[px py] :offsets [sx sy] :shape}]
                    (for [x (range px (+ px sx)) y (range py (+ py sy))] [x y 1]))
                  patches)))))

(defn- mesh
  ([] (mesh (hierarchy)))
  ([hierarchy]
  ;; Integer 8x8 finest grid on the unit periodic square. Pair enumeration is
  ;; small-oracle-only; it is not a scalable production connectivity algorithm.
  (let [cells (active-cells hierarchy)
        overlaps (fn [a as b bs] (max 0 (- (min (+ a as) (+ b bs)) (max a b))))
        touches (fn [a as b bs]
                  (or (= (+ a as) b) (= (+ b bs) a)
                      (and (= a 0) (= (+ b bs) 8))
                      (and (= b 0) (= (+ a as) 8))))
        faces (vec (for [i (range (count cells)) j (range (inc i) (count cells))
                         :let [[x y size] (cells i) [xx yy other-size] (cells j)
                               length (+ (if (touches x size xx other-size)
                                           (overlaps y size yy other-size) 0)
                                         (if (touches y size yy other-size)
                                           (overlaps x size xx other-size) 0))]
                         :when (pos? length)]
                     [i j (* 0.1 (/ length (* 0.5 (+ size other-size)))) length]))
        incidences (mapv (fn [c]
                          (vec (keep-indexed (fn [f [l r _]]
                                               (cond (= c l) [f 1.0]
                                                     (= c r) [f -1.0])) faces)))
                        (range (count cells)))
        flat (vec (mapcat identity incidences))]
    {:cells cells :faces faces
     :left (int-array (map first faces)) :right (int-array (map second faces))
     :conductance (double-array (map #(nth % 2) faces))
     :offsets (int-array (reductions + 0 (map count incidences)))
     :indices (int-array (map first flat)) :orientation (double-array (map second flat))
     :inverse-volume (double-array (map (fn [[_ _ size]] (/ 64.0 (* size size))) cells))})))

(defn- initial [cells]
  (double-array (map (fn [[x y size]]
                       (+ 2.0 (* 0.3 (Math/sin (* 2.0 Math/PI (/ (+ x (* 0.5 size)) 8.0))))
                          (* 0.2 (Math/cos (* 2.0 Math/PI (/ (+ y (* 0.5 size)) 8.0)))))) cells)))

(defn- mass [field inverse-volume]
  (reduce + (map / field inverse-volume)))

(defn- reference-step [field {:keys [faces inverse-volume]} dt]
  ;; Independent face scatter of extensive mass, not the production CSR gather.
  (let [change (double-array (count field))]
    (doseq [[l r conductance] faces]
      (let [amount (* dt conductance (- (nth field l) (nth field r)))]
        (aset-double change l (- (aget change l) amount))
        (aset-double change r (+ (aget change r) amount))))
    (mapv + field (map * change inverse-volume))))

(deftm pair-step!
  [field :- (Array double), scratch :- (Array double), flux :- (Array double),
   left :- (Array int), right :- (Array int), conductance :- (Array double),
   offsets :- (Array int), indices :- (Array int), orientation :- (Array double),
   inverse-volume :- (Array double), cells :- Long, faces :- Long, dt :- Double] :- Void
  (fv/diffusive-face-fluxes! flux field left right conductance faces)
  (fv/divergence-step! scratch field flux offsets indices orientation inverse-volume cells dt)
  (fv/diffusive-face-fluxes! flux scratch left right conductance faces)
  (fv/divergence-step! field scratch flux offsets indices orientation inverse-volume cells dt))

(deftm pair-step-into!
  [out :- (Array double), field :- (Array double), scratch :- (Array double), flux :- (Array double),
   left :- (Array int), right :- (Array int), conductance :- (Array double),
   offsets :- (Array int), indices :- (Array int), orientation :- (Array double),
   inverse-volume :- (Array double), cells :- Long, faces :- Long, dt :- Double] :- Void
  ;; Same two steps with a read-only input and a distinct owned output. This is the ordinary
  ;; functional dataflow boundary, not an ownership transfer into a second mutable owner.
  (fv/diffusive-face-fluxes! flux field left right conductance faces)
  (fv/divergence-step! scratch field flux offsets indices orientation inverse-volume cells dt)
  (fv/diffusive-face-fluxes! flux scratch left right conductance faces)
  (fv/divergence-step! out scratch flux offsets indices orientation inverse-volume cells dt))

(defn- arguments [{:keys [cells faces left right conductance offsets indices orientation inverse-volume]}]
  [(initial cells) (double-array (repeat (count cells) Double/NaN))
   (double-array (repeat (count faces) Double/NaN))
   left right conductance offsets indices orientation inverse-volume
   (long (count cells)) (long (count faces)) 0.001])

(defn- near? [expected actual]
  (and (= (count expected) (count actual))
       (every? #(< (Math/abs (double %)) 1.0e-11) (map - expected actual))))

(defn- remap-matrix [source-cells target-cells]
  ;; Bounded acceptance fixture only: enumerate geometric overlaps, then use
  ;; the existing CSR operator. Values are cell averages, not extensive mass.
  (let [overlap (fn [a as b bs] (max 0 (- (min (+ a as) (+ b bs)) (max a b))))
        rows (mapv (fn [[tx ty ts]]
                     (vec (keep-indexed
                           (fn [i [sx sy ss]]
                             (let [area (* (overlap tx ts sx ss) (overlap ty ts sy ss))]
                               (when (pos? area) [i (/ (double area) (* ts ts))])))
                           source-cells))) target-cells)
        entries (vec (mapcat identity rows))]
    (sparse/->CSRMatrix (int-array (reductions + 0 (map count rows)))
                        (int-array (map first entries)) (double-array (map second entries))
                        (long (count target-cells)) (long (count source-cells))
                        (long (count entries)))))

(defn- reference-remap [source-cells target-cells field]
  ;; Independent finest-tile lookup/average; it does not read the CSR arrays
  ;; or use the builder's pairwise overlap calculation.
  (let [tiles (into {} (for [[cell [x y size]] (map-indexed vector source-cells)
                             dx (range size) dy (range size)]
                         [[(+ x dx) (+ y dy)] (nth field cell)]))]
    (mapv (fn [[x y size]]
            (/ (reduce + (for [dx (range size) dy (range size)]
                           (get tiles [(+ x dx) (+ y dy)]))) (* size size)))
          target-cells)))

(defn- remap-geometries []
  (mapv #(mesh (hierarchy %))
        [[{:offsets [2 2] :shape [4 4]}]
         [{:offsets [2 2] :shape [2 4]}]
         [{:offsets [4 2] :shape [2 4]}]
         [{:offsets [2 2] :shape [2 2]} {:offsets [4 4] :shape [2 2]}]]))

(deftest csr-remap-preserves-constants-and-volume-weighted-mass
  (doseq [source (remap-geometries) target (remap-geometries)]
    (let [A (remap-matrix (:cells source) (:cells target))
          field (initial (:cells source))
          actual (sparse/spmv A field (double-array (repeat (count (:cells target)) -317.0)) 1.0 0.0)
          constant (sparse/spmv A (double-array (repeat (count (:cells source)) 3.25))
                                (double-array (count (:cells target))) 1.0 0.0)]
      (is (near? (reference-remap (:cells source) (:cells target) field) actual))
      (is (every? #(= 3.25 %) constant))
      (is (< (Math/abs (- (mass field (:inverse-volume source))
                          (mass actual (:inverse-volume target)))) 1.0e-12))
      (is (every? pos? (.-values A)))
      ;; Each source cell's extensive contribution must survive across all
      ;; target rows, including coarsening, refinement and moved/disjoint patches.
      (is (every? true?
                  (for [column (range (count (:cells source)))]
                    (= (/ 1.0 (aget (:inverse-volume source) column))
                       (reduce + (for [row (range (count (:cells target)))
                                       p (range (aget (.-rowptr A) row) (aget (.-rowptr A) (inc row)))
                                       :when (= column (aget (.-colidx A) p))]
                                   (/ (aget (.-values A) p) (aget (:inverse-volume target) row)))))))))))

(deftest hierarchy-projection-covers-the-domain-without-covered-coarse-cells
  (doseq [regions [[{:offsets [2 2] :shape [4 4]}]
                   [{:offsets [2 2] :shape [2 4]}]
                   [{:offsets [4 2] :shape [2 4]}]
                   [{:offsets [2 2] :shape [2 2]}
                    {:offsets [4 4] :shape [2 2]}]]]
    (let [{:keys [cells faces offsets indices orientation inverse-volume] :as geometry}
          (mesh (hierarchy regions))
          tiles (for [[x y size] cells dx (range size) dy (range size)]
                  [(+ x dx) (+ y dy)])
          args (arguments geometry)
          before (mass (first args) inverse-volume)]
      (is (= (set (for [x (range 8) y (range 8)] [x y])) (set tiles)))
      (is (= 64 (count tiles)) "no covered coarse cell survives beside its fine replacement")
      (is (every? true?
                  (map-indexed (fn [c [_ _ size]]
                                 (= (* 4 size)
                                    (reduce + (for [[l r _ length] faces
                                                    :when (or (= c l) (= c r))] length)))) cells)))
      (is (= (* 2 (count faces)) (count indices)))
      (is (every? (fn [f]
                    (= #{[1.0 (first (faces f))] [-1.0 (second (faces f))]}
                       (set (for [c (range (count cells))
                                  p (range (aget offsets c) (aget offsets (inc c)))
                                  :when (= f (aget indices p))]
                              [(aget orientation p) c])))) (range (count faces))))
      (apply pair-step! args)
      (is (near? (nth (iterate #(reference-step % geometry 0.001)
                               (vec (initial cells))) 2) (first args)))
      (is (< (Math/abs (- before (mass (first args) inverse-volume))) 1.0e-12)))))

(deftest partial-interface-is-conservative-and-constant-preserving
  (let [{:keys [cells faces inverse-volume] :as geometry} (mesh)
        args (arguments geometry)
        field (first args)
        expected (nth (iterate #(reference-step % geometry 0.001) (vec field)) 6)
        before (mass field inverse-volume)]
    (is (= {1 16, 2 12} (frequencies (map #(nth % 2) cells))))
    (is (= 60 (count faces)))
    (is (every? true?
                (map-indexed (fn [c [_ _ size]]
                               (= (* 4 size)
                                  (reduce + (for [[l r _ length] faces
                                                  :when (or (= c l) (= c r))] length)))) cells))
        "split coarse/fine and periodic faces cover every cell perimeter exactly")
    (is (every? true?
                (map-indexed (fn [c inverse-volume]
                               (<= (* 0.001 inverse-volume
                                      (reduce + (for [[l r conductance] faces
                                                      :when (or (= c l) (= c r))] conductance)))
                                   1.0)) inverse-volume))
        "the explicit step is a convex combination, including the small fine cells")
    (is (some (fn [[l r _]] (not= (nth (cells l) 2) (nth (cells r) 2))) faces))
    (is (= 1.0 (reduce + (map #(/ 1.0 %) inverse-volume))))
    (dotimes [_ 3] (apply pair-step! args))
    (is (near? expected field))
    (is (some #(> (Math/abs (double %)) 1.0e-4) (map - expected (initial cells))))
    (is (< (Math/abs (- before (mass field inverse-volume))) 1.0e-12))
    (let [constant (assoc args 0 (double-array (repeat (count cells) 3.25)))]
      (apply pair-step! constant)
      (is (every? #(= 3.25 %) (first constant))))))

(defn- run-device [target geometry]
  (let [{:keys [inverse-volume]} geometry
        args (arguments geometry)
        field (first args)
        expected-states (vec (take 7 (iterate #(reference-step % geometry 0.001) (vec field))))
        before (mass field inverse-volume)
        prepared (compiled/lower #'pair-step! args
                                 {:compiler :equation-first :target target :dtype :double
                                  :inline? true :donate '[field]})
        plan (compiled/plan prepared)]
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    (is (= 1 (count (:outputs plan))))
    (is (= 4 (reduce + (map #(count (get-in % [:call :steps])) (:instances plan))))
        "face producers and divergence consumers use the existing four-stage typed program")
    (with-open [executable (link/instantiate! plan)]
      (dotimes [replay 3]
        (link/run! executable)
        (let [actual (link/download executable (first (:outputs plan)))]
          (is (near? (expected-states (* 2 (inc replay))) actual))
          (is (< (Math/abs (- before (mass actual inverse-volume))) 1.0e-12)))))))

(defn- device-geometries []
  [(mesh) (mesh (hierarchy [{:offsets [2 2] :shape [2 2]}
                            {:offsets [4 4] :shape [2 2]}]))])

(deftest generated-partial-interface-on-opencl
  (if @opencl/opencl-fp64-available?
      (doseq [geometry (device-geometries)] (run-device :ocl:0 geometry))
      (ze/gpu-skip! "partial-patch-heat-opencl")))

(deftest generated-partial-interface-on-level-zero
  (if @ze/gpu-available?
      (doseq [geometry (device-geometries)] (run-device :ze:0 geometry))
      (ze/gpu-skip! "partial-patch-heat-level-zero")))

(defn- run-resident-remap [target source destination]
  (let [A (remap-matrix (:cells source) (:cells destination))
        source-args (arguments source)
        evolve (compiled/lower #'pair-step! source-args
                               {:compiler :equation-first :target target :dtype :double :inline? true
                                :donate '[field]
                                :constants '[left right conductance offsets indices orientation inverse-volume]})
        remap (compiled/lower #'sparse/spmv
                              [A (first source-args)
                               (double-array (repeat (count (:cells destination)) -317.0)) 1.0 0.0]
                              {:compiler :equation-first :target target :dtype :double :constants '[A]})
        prepared (with-redefs [gpu/alloc! (fn [& _] (throw (AssertionError. "composition allocated device storage")))]
                   (compiled/compose
                    {:id :heat/resident-remap
                     :components [{:id :evolve :program evolve} {:id :remap :program remap}]
                     :mutable-shares [{:owner [:evolve :field] :borrowers [[:remap :x]]
                                       :output [:evolve :field']}]
                     :outputs [{:key :source :from [:evolve :field']}
                               {:key :remapped :from [:remap :result]}]}))
        plan (compiled/plan prepared)
        mapping (get-in prepared [:lowering :certificate :node-mapping])
        field-node (:node (first (:out-tree evolve)))
        x-node (:node (first (filter #(= :x (:key %)) (:in-tree remap))))
        expected (vec (take 7 (iterate #(reference-step % source 0.001)
                                       (vec (initial (:cells source))))))
        original-mass (mass (first expected) (:inverse-volume source))]
    (is (every? #(= 0 (get-in (compiled/plan %) [:attributes :driver-allocations])) [evolve remap]))
    (is (= (mapping [:evolve field-node]) (mapping [:remap x-node])))
    (is (not-any? #(= [:remap :x] (:key %)) (:in-tree prepared))
        "the consumer must not refresh its initial host input over producer state")
    (is (= 5 (reduce + (map #(count (get-in % [:call :steps])) (:instances plan)))))
    (with-open [executable (link/instantiate! plan)]
      (dotimes [replay 3]
        (link/run! executable)
        (let [source-field (link/download executable (:node (first (:out-tree prepared))))
              result (link/download executable (:node (second (:out-tree prepared))))
              expected-field (expected (* 2 (inc replay)))]
          (is (near? expected-field source-field))
          (is (near? (reference-remap (:cells source) (:cells destination) expected-field) result))
          (is (< (Math/abs (- original-mass (mass result (:inverse-volume destination)))) 1.0e-12)))))))

(defn- run-remap-layouts [target]
  (let [[central _ moved disjoint] (remap-geometries)]
    (run-resident-remap target central moved)
    (run-resident-remap target moved disjoint)))

(deftest resident-conservative-remap-on-opencl
  (if @opencl/opencl-fp64-available? (run-remap-layouts :ocl:0)
      (opencl/opencl-skip! "resident-conservative-remap-opencl")))

(deftest resident-conservative-remap-on-level-zero
  (if @ze/gpu-available? (run-remap-layouts :ze:0)
      (ze/gpu-skip! "resident-conservative-remap-level-zero")))

(defn- run-post-remap-evolution [target source destination]
  (let [A (remap-matrix (:cells source) (:cells destination))
        source-args (arguments source)
        ;; Poison the target field too: its contents must come from the resident remap,
        ;; not the target program's captured host initializer or its previous replay.
        destination-args (assoc (arguments destination) 0
                                (double-array (repeat (count (:cells destination)) Double/NaN)))
        options {:compiler :equation-first :target target :dtype :double :inline? true
                 :donate '[field]
                 :constants '[left right conductance offsets indices orientation inverse-volume]}
        evolve-source (compiled/lower #'pair-step! source-args options)
        remap (compiled/lower #'sparse/spmv
                              [A (first source-args)
                               (double-array (repeat (count (:cells destination)) -317.0)) 1.0 0.0]
                              {:compiler :equation-first :target target :dtype :double :constants '[A]})
        evolve-target (compiled/lower #'pair-step-into!
                                      (into [(double-array (repeat (count (:cells destination)) Double/NaN))]
                                            destination-args)
                                      (assoc (dissoc options :donate) :outputs '[out]))
        prepared (with-redefs [gpu/alloc! (fn [& _] (throw (AssertionError. "composition allocated device storage")))]
                   (compiled/compose
                    {:id :heat/post-remap-evolution
                     :components [{:id :source :program evolve-source}
                                  {:id :remap :program remap}
                                  {:id :destination :program evolve-target}]
                     :mutable-shares [{:owner [:source :field] :borrowers [[:remap :x]]
                                       :output [:source :field']}]
                     :connections [{:from [:remap :result] :to [:destination :field]}]
                     :outputs [{:key :source :from [:source :field']}
                               {:key :destination :from [:destination :out]}]}))
        plan (compiled/plan prepared)
        original (vec (initial (:cells source)))
        original-mass (mass original (:inverse-volume source))
        source-states (vec (take 7 (iterate #(reference-step % source 0.001) original)))]
    (is (every? #(= 0 (get-in (compiled/plan %) [:attributes :driver-allocations]))
                [evolve-source remap evolve-target]))
    (is (= 9 (reduce + (map #(count (get-in % [:call :steps])) (:instances plan)))))
    (is (not-any? #(contains? #{[:remap :x] [:destination :field]} (:key %)) (:in-tree prepared))
        "both resident consumers lose their host refresh slots")
    (with-open [executable (link/instantiate! plan)]
      (dotimes [replay 3]
        (link/run! executable)
        (let [actual-source (link/download executable (:node (first (:out-tree prepared))))
              actual-target (link/download executable (:node (second (:out-tree prepared))))
              expected-source (source-states (* 2 (inc replay)))
              remapped (reference-remap (:cells source) (:cells destination) expected-source)
              expected-target (nth (iterate #(reference-step % destination 0.001) remapped) 2)]
          (is (near? expected-source actual-source))
          (is (near? expected-target actual-target))
          (is (not (near? remapped actual-target)) "the remapped field actually evolves")
          (is (< (Math/abs (- original-mass (mass actual-target (:inverse-volume destination))))
                 1.0e-12)))))))

(defn- run-post-remap-layouts [target]
  (let [[central _ moved disjoint] (remap-geometries)]
    (run-post-remap-evolution target central moved)
    (run-post-remap-evolution target moved disjoint)))

(deftest resident-post-remap-evolution-on-opencl
  (if @opencl/opencl-fp64-available? (run-post-remap-layouts :ocl:0)
      (opencl/opencl-skip! "resident-post-remap-evolution-opencl")))

(deftest resident-post-remap-evolution-on-level-zero
  (if @ze/gpu-available? (run-post-remap-layouts :ze:0)
      (ze/gpu-skip! "resident-post-remap-evolution-level-zero")))

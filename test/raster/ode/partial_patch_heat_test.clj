(ns raster.ode.partial-patch-heat-test
  "A synchronous conservative partial-patch oracle, not subcycled/refluxed AMR."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ode.finite-volume :as fv]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.link :as link]))

(defn- mesh []
  ;; Coordinates use an integer 8x8 finest grid on the unit periodic square.
  ;; Replace only the central 2x2 coarse patch: 12 coarse cells + 16 fine cells.
  (let [cells (vec (mapcat (fn [[x y]]
                            (if (and (<= 2 x 4) (<= 2 y 4))
                              (for [dx [0 1] dy [0 1]] [(+ x dx) (+ y dy) 1])
                              [[x y 2]]))
                          (for [x [0 2 4 6] y [0 2 4 6]] [x y])))
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
     :inverse-volume (double-array (map (fn [[_ _ size]] (/ 64.0 (* size size))) cells))}))

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

(defn- arguments [{:keys [cells faces left right conductance offsets indices orientation inverse-volume]}]
  [(initial cells) (double-array (repeat (count cells) Double/NaN))
   (double-array (repeat (count faces) Double/NaN))
   left right conductance offsets indices orientation inverse-volume
   (long (count cells)) (long (count faces)) 0.001])

(defn- near? [expected actual]
  (and (= (count expected) (count actual))
       (every? #(< (Math/abs (double %)) 1.0e-11) (map - expected actual))))

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

(defn- run-device [target]
  (let [{:keys [inverse-volume] :as geometry} (mesh)
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

(deftest generated-partial-interface-on-opencl
  (if @opencl/opencl-fp64-available? (run-device :ocl:0)
      (ze/gpu-skip! "partial-patch-heat-opencl")))

(deftest generated-partial-interface-on-level-zero
  (if @ze/gpu-available? (run-device :ze:0)
      (ze/gpu-skip! "partial-patch-heat-level-zero")))

(ns raster.ode.amr-subcycle-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.amr-plan :as amr]
            [raster.ode.amr-geometry-test :as oracle]
            [raster.ode.amr-subcycle :as subcycle]
            [raster.linalg.sparse :as sparse]
            [raster.ode.finite-volume :as fv]))

(defn- index-of [x y] (+ (* x 4) y))

(defn- average-down [coarse fine]
  (reduce (fn [out [x y]]
            (assoc out (index-of (inc x) (inc y))
                   (/ (reduce + (for [dx [0 1] dy [0 1]]
                                  (fine (index-of (+ (* 2 x) dx) (+ (* 2 y) dy))))) 4.0)))
          (vec coarse) (for [x [0 1] y [0 1]] [x y])))

(defn- initial-state []
  (let [fine (vec (for [x (range 4) y (range 4)]
                    (+ 2.0 (* 0.1 x) (* 0.2 y) (* 0.07 (Math/sin (+ (* 2.0 x) y))))))
        coarse (vec (for [x (range 4) y (range 4)] (+ 2.0 (* 0.13 x) (* 0.17 y))))]
    {:coarse (average-down coarse fine) :fine fine}))

(defn- composite-mass [coarse fine]
  (+ (/ (reduce + (for [x (range 4) y (range 4)
                        :when (not (and (<= 1 x 2) (<= 1 y 2)))] (coarse (index-of x y)))) 16.0)
     (/ (reduce + fine) 64.0)))

(defn- reference-cycle
  ([coarse fine dt frozen?]
   (reference-cycle coarse fine dt frozen?
                    {:base-shape [4 4] :offsets [2 2] :fine-shape [4 4] :lengths [1.0 1.0]}))
  ([coarse fine dt frozen? {:keys [base-shape offsets fine-shape lengths]}]
  ;; Coordinate loops do not use production CSR incidence, donor indices, interface masks
  ;; or aggregation rows. Unit domain, alpha=0.2, 4x4 coarse grid and interior 4x4 fine patch.
   (let [alpha 0.2 [nx ny] base-shape [ox oy] offsets [fx fy] fine-shape
         coarse-index (fn [x y] (+ (* x ny) y))
         fine-index (fn [x y] (+ (* x fy) y))
         spacing (mapv / lengths base-shape)
         coefficient (mapv (fn [axis] (* alpha (/ (spacing (- 1 axis)) (spacing axis)))) [0 1])
         inverse-coarse-volume (/ 1.0 (reduce * spacing))
         inverse-fine-volume (* 4.0 inverse-coarse-volume)
         lo-x (quot ox 2) lo-y (quot oy 2) hi-x (+ lo-x (quot fx 2)) hi-y (+ lo-y (quot fy 2))
         average (fn [c f]
                   (reduce (fn [out [x y]]
                             (assoc out (coarse-index (+ lo-x x) (+ lo-y y))
                                    (/ (reduce + (for [dx [0 1] dy [0 1]]
                                                   (f (fine-index (+ (* 2 x) dx) (+ (* 2 y) dy))))) 4.0)))
                           (vec c) (for [x (range (quot fx 2)) y (range (quot fy 2))] [x y])))
         coarse-rate (fn [axis plane start]
                       (let [size (base-shape axis) lo (mod (dec plane) size) hi (mod plane size)]
                         (* (coefficient axis) (if (zero? axis)
                                                 (- (coarse (coarse-index lo start)) (coarse (coarse-index hi start)))
                                                 (- (coarse (coarse-index start lo)) (coarse (coarse-index start hi)))))))
         predicted (vec (for [x (range nx) y (range ny)]
                          (- (coarse (coarse-index x y))
                             (* dt inverse-coarse-volume (+ (coarse-rate 0 (mod (inc x) nx) y)
                                                            (- (coarse-rate 0 x y))
                                                            (coarse-rate 1 (mod (inc y) ny) x)
                                                            (- (coarse-rate 1 y x)))))))
         donor (fn [x y theta]
                 (let [i (coarse-index (quot (+ ox x) 2) (quot (+ oy y) 2))]
                   (+ (* (- 1.0 theta) (coarse i)) (* theta (predicted i)))))
         fine-step (fn [state theta]
                     (vec (for [x (range fx) y (range fy)]
                            (let [value (state (fine-index x y))
                                  outward (reduce + (for [[dx dy] [[-1 0] [1 0] [0 -1] [0 1]]
                                                          :let [xx (+ x dx) yy (+ y dy)
                                                                internal? (and (<= 0 xx (dec fx)) (<= 0 yy (dec fy)))]]
                                                      (* (coefficient (if (zero? dx) 1 0)) (if internal? 1.0 (/ 2.0 3.0))
                                                         (- value (if internal? (state (fine-index xx yy))
                                                                      (donor xx yy theta))))))]
                              (- value (* 0.5 dt inverse-fine-volume outward))))))
         fine-1 (fine-step fine 0.0) theta (if frozen? 0.0 0.5)
         fine-2 (fine-step fine-1 theta)
         fine-interface (fn [state weight axis plane start]
                          (let [low? (= plane (quot (offsets axis) 2)) size (fine-shape axis)
                                edge (if low? 0 (dec size)) outside (if low? -1 size)
                                tangent (- (* 2 start) (offsets (- 1 axis)))]
                            (* (if low? -1.0 1.0)
                               (reduce + (for [t [tangent (inc tangent)]
                                               :let [x (if (zero? axis) edge t) y (if (zero? axis) t edge)
                                                     xx (if (zero? axis) outside t) yy (if (zero? axis) t outside)]]
                                           (* (coefficient axis) (/ 2.0 3.0)
                                              (- (state (fine-index x y)) (donor xx yy weight))))))))
         register (fn [axis plane start]
                    (if (and (contains? (if (zero? axis) #{lo-x hi-x} #{lo-y hi-y}) plane)
                             (if (zero? axis) (<= lo-y start (dec hi-y)) (<= lo-x start (dec hi-x))))
                      (+ (* (- dt) (coarse-rate axis plane start))
                         (* 0.5 dt (fine-interface fine 0.0 axis plane start))
                         (* 0.5 dt (fine-interface fine-1 theta axis plane start))) 0.0))
         correction (vec (for [x (range nx) y (range ny)]
                           (* inverse-coarse-volume (+ (register 0 (mod (inc x) nx) y) (- (register 0 x y))
                                                       (register 1 (mod (inc y) ny) x) (- (register 1 y x))))))]
     {:coarse (average (mapv - predicted correction) fine-2) :fine fine-2
      :predicted predicted :register register
      :omitted (average predicted fine-2)
      :wrong-sign (average (mapv + predicted correction) fine-2)})))

(defn- hierarchy []
  (#'oracle/hierarchy [4 4] [{:ratio [2 2] :patches [{:offsets [2 2] :shape [4 4]}]}]))

(deftest complete-coarse-level-and-signed-interface-partition
  (let [p (subcycle/project (hierarchy) {:domain-lengths [2.0 3.0] :diffusivity 0.2})
        arrays (subcycle/materialize p)
        covered (set (:covered-coarse p))
        boundary (filter :side (:fine-faces p))
        coarse-faces (get-in p [:coarse :faces])]
    (is (= 16 (count (get-in p [:coarse :cells]))) "covered coarse rows remain provisional state")
    (is (= #{5 6 9 10} covered))
    (is (= [4 4] (:fine-shape p)))
    (is (= [32 40 16] [(count coarse-faces) (count (:fine-faces p)) (count boundary)]))
    (is (= 8 (count (filter seq (:coarse-interface-rows p)))))
    (is (= 6.0 (+ (* (- 16 (count covered)) (/ 6.0 16))
                  (* (:fine-cell-count p) (/ 6.0 64)))))
    (is (empty? (filter covered (:boundary-donors p))) "ghost donors are uncovered coarse cells")
    (doseq [[coarse-face rows] (map-indexed vector (:coarse-interface-rows p)) :when (seq rows)]
      (let [f (coarse-faces coarse-face)
            fine-faces (map #(get-in p [:fine-faces (first %)]) rows)]
        (is (= 2 (count rows)))
        (is (= (:measure f) (reduce + (map :measure fine-faces))))
        (is (= 1 (count (set (map second rows)))))
        (is (= (set [(:left f) (:right f)])
               (set (concat (map :donor fine-faces)
                            ;; The covered endpoint is derived from the opposite side of
                            ;; the interface, not from the production aggregation signs.
                            (filter covered [(:left f) (:right f)])))))
        (is (= (if (covered (:left f)) 1.0 -1.0) (second (first rows))))))
    (let [fine-flux (double-array (map #(+ 0.25 %) (range 40)))
          actual (sparse/spmv (:aggregation arrays) fine-flux (double-array 32) 1.0 0.0)]
      (doseq [f (range 32)]
        (is (= (reduce + 0.0 (for [[i face] (map-indexed vector (:fine-faces p))
                                   :when (= f (:coarse-face face))]
                               (* (:coarse-sign face) (aget fine-flux i))))
               (aget actual f)))))
    (let [fresh (subcycle/materialize p)]
      (aset-int (:covered-coarse arrays) 0 99)
      (is (= 5 (aget ^ints (:covered-coarse fresh) 0))))))

(deftest constant-fine-state-has-no-open-boundary-transport
  (let [p (subcycle/project (hierarchy) {:domain-lengths [1.0 1.0]})
        {:keys [fine coarse-owner] :as arrays} (subcycle/materialize p)
        old (double-array (repeat 32 3.25)) next (double-array (repeat 32 3.25))
        current (double-array (repeat 32 3.25)) boundary (double-array 32)
        flux (double-array 40) out (double-array 16)]
    (fv/temporal-boundary! boundary old next current coarse-owner 32 0.5)
    (fv/diffusive-face-fluxes! flux boundary (:left fine) (:right fine) (:conductance fine) 40)
    (fv/divergence-step! out boundary flux (:offsets fine) (:indices fine) (:orientation fine)
                         (:inverse-volume fine) 16 0.001)
    (is (every? zero? flux))
    (is (= (vec current) (vec boundary)))
    (is (= (vec (repeat 16 3.25)) (vec out)))))

(deftest subcycle-scope-and-budget-fail-before-materialization
  (let [h (hierarchy) reason #(try (subcycle/project %1 %2) nil
                                   (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))]
    (is (= :amr-subcycle-budget (reason h {:domain-lengths [1.0 1.0] :max-patch-cells 31})))
    (is (= :amr-subcycle-options (reason h {:domain-lengths [Double/NaN 1.0]})))
    (is (= :amr-subcycle-options (reason h {:domain-lengths [1.0 1.0] :diffusivity -1.0})))
    (is (= :amr-subcycle-options (reason h {:domain-lengths [1.0 1.0] :boundary :no-flux})))
    (is (= :amr-subcycle-hierarchy
           (reason (amr/hierarchy (assoc (into {} h) :levels [(first (:levels h))]))
                   {:domain-lengths [1.0 1.0]})))))

(deftest complete-jvm-cycle-matches-coordinates-and-reflux-is-load-bearing
  (let [p (subcycle/project (hierarchy) {:domain-lengths [1.0 1.0] :diffusivity 0.2})
        {:keys [coarse fine]} (initial-state) dt 0.001
        expected (reference-cycle coarse fine dt false)
        inputs (subcycle/cycle-inputs p (double-array coarse) (double-array fine) dt)
        before (composite-mass coarse fine)]
    (apply subcycle/diffusion-cycle! (:arguments inputs))
    (doseq [field [:coarse :fine]]
      (is (every? #(< (Math/abs (double %)) 1.0e-12)
                  (map - (field expected) (get-in inputs [:outputs field])))))
    (is (< (Math/abs (- before (composite-mass (vec (get-in inputs [:outputs :coarse]))
                                               (vec (get-in inputs [:outputs :fine]))))) 1.0e-12))
    (is (> (Math/abs (- before (composite-mass (:omitted expected) (:fine expected)))) 1.0e-8))
    (is (> (Math/abs (- before (composite-mass (:wrong-sign expected) (:fine expected)))) 1.0e-8))
    (is (some #(> (Math/abs (double %)) 1.0e-8) (get-in inputs [:scratch :register])))
    (doseq [[i {:keys [axis plane start]}] (map-indexed vector (get-in p [:coarse :faces]))]
      (is (< (Math/abs (- ((:register expected) axis plane start)
                          (aget ^doubles (get-in inputs [:scratch :register]) i))) 1.0e-12)))
    (is (some #(> (Math/abs (double %)) 1.0e-8)
              (map - (:fine expected) (:fine (reference-cycle coarse fine dt true))))
        "freezing old coarse boundary data is numerically distinguishable")
    (is (= (vec (get-in inputs [:outputs :coarse]))
           (average-down (vec (get-in inputs [:outputs :coarse]))
                         (vec (get-in inputs [:outputs :fine])))))))

(deftest rectangular-anisotropic-cycle-matches-independent-coordinates
  (let [spec {:base-shape [6 5] :offsets [4 2] :fine-shape [4 6] :lengths [2.0 3.0]}
        h (#'oracle/hierarchy [6 5] [{:ratio [2 2] :patches [{:offsets [4 2] :shape [4 6]}]}])
        p (subcycle/project h {:domain-lengths [2.0 3.0] :diffusivity 0.2 :boundary :periodic})
        fine (vec (for [x (range 4) y (range 6)]
                    (+ 1.0 (* 0.17 x) (* 0.09 y) (* 0.08 (Math/cos (+ x (* 2.0 y)))))))
        covered (set (for [x [2 3] y [1 2 3]] (+ (* x 5) y)))
        coarse (reduce (fn [c [x y]]
                         (assoc c (+ (* (+ 2 x) 5) (+ 1 y))
                                (/ (reduce + (for [dx [0 1] dy [0 1]]
                                               (fine (+ (* (+ (* 2 x) dx) 6) (+ (* 2 y) dy))))) 4.0)))
                       (vec (for [x (range 6) y (range 5)] (+ 2.0 (* 0.11 x) (* 0.23 y))))
                       (for [x [0 1] y [0 1 2]] [x y]))
        mass (fn [c f] (+ (* 0.2 (reduce + (keep-indexed #(when-not (covered %1) %2) c)))
                          (* 0.05 (reduce + f))))
        expected (reference-cycle coarse fine 0.001 false spec)
        inputs (subcycle/cycle-inputs p (double-array coarse) (double-array fine) 0.001)]
    (apply subcycle/diffusion-cycle! (:arguments inputs))
    (is (= :periodic (:boundary p)))
    (doseq [field [:coarse :fine]]
      (is (= (count (field expected)) (count (get-in inputs [:outputs field]))))
      (is (every? #(< (Math/abs (double %)) 1.0e-12)
                  (map - (field expected) (get-in inputs [:outputs field])))))
    (is (< (Math/abs (- (mass coarse fine)
                        (mass (vec (get-in inputs [:outputs :coarse]))
                              (vec (get-in inputs [:outputs :fine]))))) 1.0e-12))
    (doseq [[i {:keys [axis plane start]}] (map-indexed vector (get-in p [:coarse :faces]))]
      (is (< (Math/abs (- ((:register expected) axis plane start)
                          (aget ^doubles (get-in inputs [:scratch :register]) i))) 1.0e-12)))))

(ns raster.ad.normalized-segmented-reduction-test
  "Independent flat-matrix oracles for the bounded opt-in segmented AD rule.
   Typed Float controls describe host checkpoints, not a GPU/BLAS rounding promise."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-device :as device]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.gpu.link-module :as link-module]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- close? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? #(<= (Math/abs (double %)) tolerance) (map - expected actual))))

;; These loops encode matrix coordinates directly. They never inspect generated
;; programs, their AxisMaps, scalar derivative proofs, or compiler phase bodies.
(defn- product [rows inner columns a w]
  (vec (for [i (range rows) k (range columns)]
         (reduce + 0.0 (for [j (range inner)]
                         (* (double (nth a (+ (* i inner) j)))
                            (double (nth w (+ (* j columns) k)))))))))

(defn- pull [active rows inner columns a w dy]
  (if (= active 'A)
    (vec (for [i (range rows) j (range inner)]
           (reduce + 0.0 (for [k (range columns)]
                           (* (double (nth dy (+ (* i columns) k)))
                              (double (nth w (+ (* j columns) k))))))))
    (vec (for [j (range inner) k (range columns)]
           (reduce + 0.0 (for [i (range rows)]
                           (* (double (nth a (+ (* i inner) j)))
                              (double (nth dy (+ (* i columns) k))))))))))

(defn- pairing [x y] (reduce + 0.0 (map * x y)))

(defn- float-bits [xs] (mapv #(Float/floatToRawIntBits (float %)) xs))

;; Explicit Float product, Float addition, and Float output checkpoints. There
;; is deliberately no call into Raster's numerical operators or AD implementation.
(defn- float-fold [products]
  (reduce (fn [acc [x y]] (float (+ acc (float (* (float x) (float y))))))
          (float 0.0) products))

(defn- float-product [rows inner columns a w]
  (vec (for [i (range rows) k (range columns)]
         (float-fold (for [j (range inner)]
                       [(nth a (+ (* i inner) j)) (nth w (+ (* j columns) k))])))))

(defn- float-pull [active rows inner columns a w dy]
  (if (= active 'A)
    (vec (for [i (range rows) j (range inner)]
           (float-fold (for [k (range columns)]
                         [(nth dy (+ (* i columns) k)) (nth w (+ (* j columns) k))]))))
    (vec (for [j (range inner) k (range columns)]
           (float-fold (for [i (range rows)]
                         [(nth dy (+ (* i columns) k)) (nth a (+ (* i inner) j))]))))))

(defn- source
  ([dt] (source dt 2 3 4 '[A W]))
  ([dt rows inner columns inputs]
   (let [init (if (= dt :float) '(clojure.core/float 0.0) 0.0)
         body (list 'raster.numeric/+ 'acc
                    (list 'raster.numeric/*
                          (list 'clojure.core/aget 'lhs
                                (list 'clojure.core/+ (list 'clojure.core/* 'i inner) 'j))
                          (list 'clojure.core/aget 'rhs
                                (list 'clojure.core/+ (list 'clojure.core/* 'j columns) 'k))))
         arithmetic (numerics/blas-source-arithmetic dt)
         attributes {:segment-axes [['i rows] ['k columns]] :index 'j :extent inner
                     :result-map {:groups [[['i rows] ['k columns]]]}
                     :accumulators '[acc] :identities [init] :dtypes [dt]
                     :algebra [(scan/certify-reassociation {:acc 'acc :init init :lambda body} dt)]
                     :source-arithmetic arithmetic
                     :attributes {:stable-array-captures '[A W] :pointwise-storage-inputs []}}
         equation (list '= 'dense-product '[C]
                        (list 'segmented-reduce attributes [] '[A W]
                              (soac/lambda-form '[acc lhs rhs] [] [body])))
         value (fn [extent ownership]
                 (av/tensor {:dtype dt :shape [extent] :ownership ownership}))]
     (soac/make (soac/default-program-facts
                 {:values {'A (value (*' rows inner) nil)
                           'W (value (*' inner columns) nil)
                           'C (value (*' rows columns) :owned)}
                  :inputs inputs :attributes {:source-arithmetic arithmetic}
                  :equations {'dense-product (soac/default-equation-facts)}})
                [equation] '[C]))))

(defn- evidence [program active]
  (normalized/linearize-equation-program
   program (the-ns 'raster.ad.normalized-segmented-reduction-test) {:active-input active}))

(defn- arguments [prepared a w] (mapv {'A a 'W w} (:inputs prepared)))
(defn- active-slot [prepared active gradients]
  (nth gradients (.indexOf ^java.util.List (:inputs prepared) active)))

(def ^:private a0 [-2.0 0.5 3.0 1.25 -4.0 2.0])
(def ^:private w0 [0.25 -1.0 2.0 0.75 3.0 -0.5 1.5 -2.0 -1.25 4.0 0.125 2.5])
(def ^:private dy0 [0.5 -1.5 2.0 0.25 -2.5 0.75 1.25 -0.125])
(def ^:private float-a0 [4097.0 0.33333334 -4096.0 -0.125 3.1415927 0.0009765625])
(def ^:private float-w0 [0.33333334 -2.5 0.125 3.1415927
                         2.7182817 0.000244140625 -0.6666667 1.5
                         0.33333334 -2.5 0.125 3.1415927])

(defn- activated-source [dt inputs]
  (let [product (source dt 2 3 4 inputs)
        activation (list '= 'activation '[Y]
                         (list 'map {:index 'p :extent 8} '[C] []
                               (soac/lambda-form '[element] [] '[(raster.numeric/* element element)])))
        facts (-> (soac/facts product)
                  (assoc :attributes {})
                  (assoc-in [:values 'Y] (av/tensor {:dtype dt :shape [8] :ownership :owned}))
                  (assoc-in [:equations 'activation] (soac/default-equation-facts)))]
    (soac/make facts (conj (vec (soac/equations product)) activation) '[Y])))

(defn- summed-source [producer dt]
  (let [boundary (first (soac/outputs producer))
        extent (first (get-in (soac/facts producer) [:values boundary :shape]))
        zero (if (= dt :float) (float 0.0) 0.0)
        body '(raster.numeric/+ total element)
        reduction (list '= 'ordered-loss '[loss]
                        (list 'reduce {:index 'p :extent extent :accumulators '[total]
                                       :identities [zero] :dtypes [dt]
                                       :algebra [(scan/certify-reassociation
                                                  {:acc 'total :init zero :lambda body} dt)]}
                              [boundary] [] (soac/lambda-form '[total element] [body])))
        facts (-> (soac/facts producer)
                  (assoc-in [:values 'loss] (av/tensor {:dtype dt :shape [] :ownership :owned}))
                  (assoc-in [:equations 'ordered-loss] (soac/default-equation-facts)))]
    (soac/make facts (conj (vec (soac/equations producer)) reduction) '[loss])))

(defn- summed-activation-source [dt inputs]
  (summed-source (activated-source dt inputs) dt))

(deftest map-spine-ordered-loss-uses-the-same-composition
  (doseq [dt [:float :double]]
    (let [array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)
          equation (fn [id output input body]
                     (list '= id [output]
                           (list 'map {:index 'i :extent 3} [input] []
                                 (soac/lambda-form '[x] [] [body]))))
          producer (soac/make
                    (soac/default-program-facts
                     {:inputs '[X]
                      :values (into {} (for [id '[X squared Y]]
                                         [id (av/tensor {:dtype dt :shape [3]
                                                        :ownership (when (not= id 'X) :owned)})]))
                      :equations (into {} (for [id '[square twice]]
                                            [id (soac/default-equation-facts)]))})
                    [(equation 'square 'squared 'X '(raster.numeric/* x x))
                     (equation 'twice 'Y 'squared '(raster.numeric/+ x x))] '[Y])
          checked (evidence (summed-source producer dt) 'X)
          prepared (runtime/prepare checked)]
      (doseq [xs [[-2.0 0.5 3.0] [1.25 -1.0 2.0]]]
        (let [input (array xs)
              [value pb] ((:vjp prepared) [input])
              [jvalue tangent] ((:jvp prepared) [input] (array [1.0 1.0 1.0]))
              expected (reduce (fn [a x] (cast (+ a (cast (+ (cast (* x x)) (cast (* x x)))))))
                               (cast 0.0) xs)]
          (is (= expected value jvalue))
          (is (= (reduce + 0.0 (map #(* 4.0 %) xs)) tangent))
          (java.util.Arrays/fill input (cast 99.0))
          (doseq [seed [1.0 -0.5]]
            (is (= (mapv #(cast (* 4.0 seed %)) xs)
                   (vec (first (pb (cast seed))))))))))))

(deftest repeated-input-map-spine-loss-retains-ordered-gradient-ports
  (doseq [dt [:float :double]]
    (let [equations
          (mapv (fn [[id output arrays parameters body]]
                  (list '= id [output]
                        (list 'map {:index 'i :extent 3} arrays []
                              (soac/lambda-form parameters [] [body]))))
                '[[subtract paired [X Q] [x q] (raster.numeric/- x q)]
                  [reuse reused [paired X] [p x] (raster.numeric/* p x)]
                  [scale Y [reused R] [p r] (raster.numeric/* p r)]])
          producer (soac/make
                    (soac/default-program-facts
                     {:inputs '[R Q X]
                      :values (into {} (for [id '[R Q X paired reused Y]]
                                         [id (av/tensor {:dtype dt :shape [3]
                                                        :ownership (when (contains? '#{paired reused Y} id)
                                                                     :owned)})]))
                      :equations (into {} (map #(vector (second %) (soac/default-equation-facts)) equations))})
                    equations '[Y])
          checked (normalized/linearize-equation-program
                   (summed-source producer dt) *ns* {:active-inputs '[R X]})
          owner (canonical/adapt checked)
          prepared (runtime/prepare checked)
          array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)]
      (is (= 3 (count (get-in checked [:map-programs :proof :nodes]))))
      (is (= '[R X] (get-in owner [:interface :active-inputs])))
      (is (every? (fn [[[equation _] binding]]
                    (= equation (get-in binding [:origin :execution-equation])))
                  (:residual-bindings owner)))
      (doseq [xs [[-2.0 0.5 3.0] [1.25 -1.0 2.0]]]
        (let [qs [0.5 -1.0 2.0] rs [2.0 -0.5 1.0]
              args [(array rs) (array qs) (array xs)]
              [value pb] ((:vjp prepared) args)
              [jvalue tangent] ((:jvp prepared) args [(array [0.25 0.25 0.25]) (array [0.5 0.5 0.5])])
              dr (mapv #(* (- %1 %2) %1) xs qs)
              dx (mapv #(* (- (* 2.0 %1) %2) %3) xs qs rs)]
          (is (= (reduce + 0.0 (map * dr rs)) value jvalue))
          (is (= (reduce + 0.0 (map #(+ (* 0.25 %1) (* 0.5 %2)) dr dx)) tangent))
          (doseq [input args] (java.util.Arrays/fill input (cast 99.0)))
          (doseq [seed [1.0 -0.5]]
            (let [[gr inactive gx] (pb (cast seed))]
              (is (= (mapv #(cast (* seed %)) dr) (vec gr)))
              (is (nil? inactive))
              (is (= (mapv #(cast (* seed %)) dx) (vec gx))))))))))

(deftest contraction-activation-ordered-loss-composes-existing-derivatives
  (doseq [dt [:float :double] active '[A W] inputs ['[A W] '[W A]]]
    (let [program (summed-activation-source dt inputs)
          checked (normalized/linearize-equation-program program *ns* {:active-input active})
          owner (canonical/adapt checked)
          prepared (runtime/prepare checked)
          make-array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)
          mul (fn [x y] (cast (* x y)))
          add (fn [x y] (cast (+ x y)))
          a (make-array a0) w (make-array w0)
          c (if (= dt :float) (float-product 2 3 4 a0 w0) (product 2 3 4 a0 w0))
          expected (reduce add (cast 0.0) (map #(mul % %) c))
          args (arguments prepared a w)
          [value pb] ((:vjp prepared) args)
          tangent (make-array (repeat (if (= active 'A) 6 12) 0.25))
          dc (product 2 3 4 (if (= active 'A) tangent a0) (if (= active 'W) tangent w0))
          [jvalue jvp] ((:jvp prepared) args tangent)
          tolerance (if (= dt :float) 1e-4 1e-10)
          seed-array (get-in checked [:sum-programs :interface :seed-array])
          transpose (:transpose owner)]
      (is (= inputs (:inputs prepared)))
      (is (< (Math/abs (- expected value)) tolerance))
      (is (= value jvalue))
      (is (< (Math/abs (- jvp (reduce + 0.0 (map #(* 2.0 %1 %2) c dc)))) tolerance))
      (is (= [8] (get-in (soac/facts transpose) [:values seed-array :shape])))
      (is (= :owned (get-in (soac/facts transpose) [:values seed-array :ownership])))
      (is (every? (fn [[[equation _] binding]]
                    (and (= equation (get-in binding [:origin :execution-equation]))
                         (seq (get-in binding [:origin :source-equations]))
                         (get-in binding [:producer :equation])))
                  (:residual-bindings owner)))
      ;; Delayed pullback owns passive inputs and nonlinear producer residuals.
      (java.util.Arrays/fill a (cast 77.0))
      (java.util.Arrays/fill w (cast -88.0))
      (doseq [seed [1.0 -0.5]]
        (let [seed (cast seed)
              dy (mapv #(add (mul seed %) (mul seed %)) c)
              oracle ((if (= dt :float) float-pull pull) active 2 3 4 a0 w0 dy)
              gradients (pb seed)
              actual (active-slot prepared active gradients)]
          (is (close? oracle actual tolerance))
          (when (= dt :double)
            (let [point (if (= active 'A) a0 w0)
                  loss (fn [values]
                         (reduce + 0.0
                                 (map #(* % %)
                                      (product 2 3 4 (if (= active 'A) values a0)
                                               (if (= active 'W) values w0)))))
                  h 1e-5
                  finite-difference
                  (mapv (fn [i]
                          (* seed (/ (- (loss (update point i + h))
                                         (loss (update point i - h))) (* 2.0 h))))
                        (range (count point)))]
              (is (close? finite-difference actual 1e-6))))
          (is (nil? (nth gradients (.indexOf ^java.util.List inputs (if (= active 'A) 'W 'A))))))))))

(defn- square-push [cast values tangent]
  ;; The existing elementwise template emits two rounded dx*x contributions,
  ;; then their rounded sum. This is deliberately not the expression 2*x*dx.
  (mapv (fn [x dx]
          (let [contribution (cast (* (cast dx) (cast x)))]
            (cast (+ contribution contribution)))) values tangent))

(deftest segmented-product-square-composes-independent-host-derivatives
  (let [base (activated-source :double '[W A])
        [product-equation] (soac/equations base)
        scale-map (fn [id output input]
                    (list '= id [output]
                          (list 'map {:index 'p :extent 8} [input] '[scale]
                                (soac/lambda-form '[element scalar] [] '[(raster.numeric/* element scalar)]))))
        facts (-> (soac/facts base) (assoc :inputs '[scale W A])
                  (assoc-in [:values 'scale] (av/tensor {:dtype :double :shape []}))
                  (assoc-in [:values 'T] (get-in (soac/facts base) [:values 'Y]))
                  (assoc-in [:equations 'scale-one] (soac/default-equation-facts)))
        checked (evidence (soac/make facts [product-equation (scale-map 'scale-one 'T 'C)
                                            (scale-map 'activation 'Y 'T)] '[Y]) 'A)
        owner (canonical/adapt checked) prepared (runtime/prepare checked)
        args [3.0 (double-array w0) (double-array a0)]
        [output pb] ((:vjp prepared) args)
        dx (vec (repeat 6 0.25))
        [_ tangent] ((:jvp prepared) args (double-array dx))
        slots (pb (double-array dy0))
        scalars (filter #(= :immutable-scalar (:storage (val %))) (:residual-bindings owner))]
    (is (= '[scale W A] (:inputs prepared)))
    (is (close? (mapv #(* 9.0 %) (product 2 3 4 a0 w0)) output 1.0e-12))
    (is (close? (mapv #(* 9.0 %) (product 2 3 4 dx w0)) tangent 1.0e-12))
    (is (= [nil nil] (subvec slots 0 2)))
    (is (close? (pull 'A 2 3 4 a0 w0 (mapv #(* 9.0 %) dy0)) (nth slots 2) 1.0e-12))
    (is (= 2 (count scalars)))
    (is (= #{'scale-one 'activation} (set (map (comp first key) scalars))))
    (is (every? #(= 'scale (:id (val %)) (get-in (val %) [:producer :input])) scalars)))
  (doseq [dt [:double :float] active '[A W] inputs ['[A W] '[W A]]]
    (let [checked (evidence (activated-source dt inputs) active)
          owner (canonical/adapt checked)
          prepared (runtime/prepare checked)
          array (if (= dt :float) float-array double-array)
          cast (if (= dt :float) float double)
          a (mapv cast (if (= dt :float) float-a0 a0))
          w (mapv cast (if (= dt :float) float-w0 w0)) dy (mapv cast dy0)
          dx (mapv #(cast (* 0.125 (- % 2))) (range (if (= active 'A) 6 12)))
          c (if (= dt :float) (float-product 2 3 4 a w) (product 2 3 4 a w))
          dc (if (= dt :float)
               (if (= active 'A) (float-product 2 3 4 dx w) (float-product 2 3 4 a dx))
               (if (= active 'A) (product 2 3 4 dx w) (product 2 3 4 a dx)))
          expected-output (mapv #(cast (* % %)) c)
          expected-push (square-push cast c dc)
          seed-c (square-push cast c dy)
          expected-pull (if (= dt :float) (float-pull active 2 3 4 a w seed-c)
                            (pull active 2 3 4 a w seed-c))
          aa (array a) ww (array w)
          [output pb] ((:vjp prepared) (arguments prepared aa ww))
          [jvp-output tangent] ((:jvp prepared) (arguments prepared aa ww) (array dx))
          slots (pb (array dy)) gradient (active-slot prepared active slots)]
      (if (= dt :float)
        (do (is (= (float-bits expected-output) (float-bits output) (float-bits jvp-output)))
            (is (= (float-bits expected-push) (float-bits tangent)))
            (is (= (float-bits expected-pull) (float-bits gradient))))
        (do
          (is (close? expected-output output 1.0e-10))
          (is (= (vec output) (vec jvp-output)))
          (is (close? expected-push tangent 1.0e-10))
          (is (close? expected-pull gradient 1.0e-10))
          (is (< (Math/abs (- (pairing tangent dy) (pairing dx gradient))) 1.0e-9))
          (let [h 1.0e-5
                objective (fn [step]
                            (let [a1 (if (= active 'A) (mapv + a (map #(* step %) dx)) a)
                                  w1 (if (= active 'W) (mapv + w (map #(* step %) dx)) w)]
                              (pairing (first ((:vjp prepared)
                                               (arguments prepared (array a1) (array w1)))) dy)))
                difference (/ (- (objective h) (objective (- h))) (* 2.0 h))]
            (is (< (Math/abs (- difference (pairing dx gradient))) 2.0e-6)))))
      (is (= inputs (:inputs prepared)))
      (is (every? true? (map (fn [id slot] (= (= active id) (some? slot))) inputs slots)))
      (is (= [8] (get-in owner [:interface :primal-result :value :shape])))
      (is (= [(count dx)] (get-in owner [:interface :cotangent-results 0 :value :shape])))
      (is (= [['dense-product] ['activation]]
             (mapv :source-equations (get-in checked [:map-programs :proof :nodes]))))
      (is (= #{'dense-product 'activation} (set (map first (keys (:residual-bindings owner))))))
      (doseq [[[equation _] binding] (:residual-bindings owner)]
        (is (= equation (get-in binding [:origin :execution-equation])))
        (is (= :owned-array (:storage binding)))
        (is (= (if (= equation 'activation) [8] (if (= active 'A) [12] [6]))
               (get-in binding [:value :shape]))))
      ;; Caller mutations include the returned nonlinear activation, not private
      ;; owned snapshots. A delayed pullback must still use its primal coefficients.
      (dotimes [i (count a)] (if (= dt :float) (aset-float aa i (float 19.0)) (aset-double aa i 19.0)))
      (dotimes [i (count w)] (if (= dt :float) (aset-float ww i (float -7.0)) (aset-double ww i -7.0)))
      (dotimes [i 8] (if (= dt :float) (aset-float output i (float 99.0)) (aset-double output i 99.0)))
      (let [again (active-slot prepared active (pb (array dy)))]
        (is (not (identical? gradient again)))
        (if (= dt :float) (is (= (float-bits expected-pull) (float-bits again)))
            (is (close? expected-pull again 1.0e-10))))
      (let [a1 (vec aa) w1 (vec ww)
            c1 (if (= dt :float) (float-product 2 3 4 a1 w1) (product 2 3 4 a1 w1))
            seed1 (square-push cast c1 dy)
            expected1 (if (= dt :float) (float-pull active 2 3 4 a1 w1 seed1)
                          (pull active 2 3 4 a1 w1 seed1))
            [new-output new-pb] ((:vjp prepared) (arguments prepared aa ww))
            new-gradient (active-slot prepared active (new-pb (array dy)))]
        (is (not (identical? output new-output)))
        (if (= dt :float) (is (= (float-bits expected1) (float-bits new-gradient)))
            (is (close? expected1 new-gradient 1.0e-10)))))))

(deftest retained-segmented-node-ports-keep-independent-operand-and-result-spaces
  (doseq [dt [:float :double] active '[A W]]
    (let [program (source dt 2 3 4 '[W A])
          checked (evidence program active)
          node (get-in checked [:map-programs :proof :nodes 0])
          description (#'normalized/segmented-description program {:active-input active})
          reject-allocation (fn [& _] (throw (ex-info "checked port replay allocated an identity" {})))
          built (with-redefs-fn {#'normalized/allocate-map-stage-interface reject-allocation
                                #'normalized/allocate-segmented-interface reject-allocation
                                #'normalized/node-equation-ids reject-allocation
                                #'normalized/linearize-program reject-allocation}
                  #(#'normalized/segmented-node-lifecycle description (:source-ns checked) nil node))
          ports (:ports built) stage (:stage built)
          selected (first (:active ports))
          passive (if (= active 'A) 'W 'A)
          active-value (get-in (soac/facts program) [:values active])
          passive-value (get-in (soac/facts program) [:values passive])
          scalar (:scalar-derivation node)
          coefficient (first (:residuals scalar))
          snapshot (get-in ports [:residual-arrays coefficient])
          ids (:equation-ids node)
          [primal-copy] (rest (soac/equations (:primal stage)))]
      (is (= node (dissoc built :result :active? :ports :stage)))
      (is (= [8] (get-in ports [:result :value :shape])
                 (get-in ports [:tangent-result :value :shape])
                 (get-in ports [:cotangent-input :value :shape])))
      (is (= (if (= active 'A) [6] [12]) (:shape active-value)))
      (is (= active (:source selected) (get-in selected [:tangent :source])
                    (get-in selected [:gradient :source])))
      (is (= active-value (:source-value selected)))
      (is (fingerprint/equivalent? (second (:active-inputs scalar)) (:scalar-source selected)))
      (is (= (assoc active-value :ownership :borrowed) (get-in selected [:tangent :value])))
      (is (= (assoc active-value :ownership :owned) (get-in selected [:gradient :value])))
      (is (= dt (get-in selected [:tangent :value :dtype]) (get-in selected [:gradient :value :dtype])))
      (is (= (if (= active 'A) [12] [6]) (get-in snapshot [:value :shape])))
      (is (= (assoc passive-value :ownership :owned) (:value snapshot)))
      (is (fingerprint/equivalent? coefficient (:coefficient snapshot)))
      (is (= [passive] (:arrays (soac/operation-parts primal-copy))))
      (is (= [(:id snapshot)] (nth primal-copy 2)))
      (is (= {} (:scalar-captures ports)))
      (doseq [phase [:linear :transpose]]
        (let [program (get stage phase)
              ;; Fault only submitted correspondence metadata: the checker must
              ;; retain the passive operand's space, never borrow result extent 8.
              facts (assoc-in (soac/facts program) [:values (:id snapshot) :shape] [8])
              changed (apply list (assoc (vec program) 1 facts))]
          (is (= :ad-normalized-segmented-interface
                 (reason #(#'normalized/checked-segmented-node-ports
                            description scalar (:stage-interface node) (assoc stage phase changed)))))))
      (is (= [(:primal ids) (symbol (str (name (:primal ids)) "__passive_snapshot"))]
             (mapv second (soac/equations (:primal stage)))))
      (doseq [phase [:linear :transpose]]
        (is (= [(get ids phase)] (mapv second (soac/equations (get stage phase)))))))))

(deftest nonsquare-double-products-have-independent-jvp-vjp-and-finite-differences
  (doseq [active '[A W] inputs ['[A W] '[W A]]]
    (testing (str active " with public order " inputs)
      (let [program (source :double 2 3 4 inputs)
            checked (evidence program active)
            owner (canonical/adapt checked)
            prepared (runtime/prepare checked)
            a (double-array a0) w (double-array w0)
            dx (mapv #(* 0.125 (- % 2)) (range (if (= active 'A) 6 12)))
            [output pb] ((:vjp prepared) (arguments prepared a w))
            [jvp-output tangent] ((:jvp prepared) (arguments prepared a w) (double-array dx))
            slots (pb (double-array dy0))
            gradient (active-slot prepared active slots)
            expected-push (if (= active 'A) (product 2 3 4 dx w0) (product 2 3 4 a0 dx))
            expected-pull (pull active 2 3 4 a0 w0 dy0)
            interface (:interface owner)]
        (is (identical? program (:source owner)))
        (is (= inputs (:inputs prepared)))
        (is (= [8] (get-in interface [:primal-result :value :shape])
                   (get-in interface [:cotangent-input :value :shape])))
        (is (= [(count dx)] (get-in interface [:tangent-inputs 0 :value :shape])
                           (get-in interface [:cotangent-results 0 :value :shape])))
        (doseq [phase [:linear :transpose]
                equation (soac/equations (get owner phase))]
          (is (empty? (:locals (soac/lambda-parts (:lambda (soac/operation-parts equation)))))))
        (is (close? (product 2 3 4 a0 w0) output 1.0e-12))
        (is (= (vec output) (vec jvp-output)))
        (is (close? expected-push tangent 1.0e-12))
        (is (close? expected-pull gradient 1.0e-12))
        (is (every? true? (map (fn [id slot] (= (= active id) (some? slot))) inputs slots)))
        (is (< (Math/abs (- (pairing tangent dy0) (pairing dx gradient))) 1.0e-11))
        ;; Re-evaluate the public prepared primal with perturbed independent
        ;; coordinates; no derivative phase participates in this oracle.
        (doseq [coordinate (range (count dx))]
          (let [h 1.0e-5
                objective (fn [delta]
                            (let [aa (double-array a0) ww (double-array w0)
                                  selected (if (= active 'A) aa ww)]
                              (aset ^doubles selected coordinate
                                    (+ (aget ^doubles selected coordinate) delta))
                              (pairing (first ((:vjp prepared) (arguments prepared aa ww))) dy0)))
                finite-difference (/ (- (objective h) (objective (- h))) (* 2.0 h))]
            (is (< (Math/abs (- finite-difference (aget ^doubles gradient coordinate)))
                   2.0e-8))))))))

(deftest float-phases-obey-explicit-typed-host-checkpoints
  (doseq [active '[A W]]
    (let [prepared (runtime/prepare (evidence (source :float 2 3 4 '[W A]) active))
          ;; Non-exact products and cancellation expose missing Float checkpoints.
          a (mapv float float-a0)
          w (mapv float float-w0)
          dx (mapv #(float (* 0.125 (- % 2))) (range (if (= active 'A) 6 12)))
          dy (mapv float dy0)
          args (arguments prepared (float-array a) (float-array w))
          [output pb] ((:vjp prepared) args)
          [jvp-output tangent] ((:jvp prepared) args (float-array dx))
          gradient (active-slot prepared active (pb (float-array dy)))]
      (is (= (float-bits (float-product 2 3 4 a w)) (float-bits output)))
      (is (= (float-bits output) (float-bits jvp-output)))
      (is (= (float-bits (if (= active 'A) (float-product 2 3 4 dx w)
                            (float-product 2 3 4 a dx)))
             (float-bits tangent)))
      (is (= (float-bits (float-pull active 2 3 4 a w dy)) (float-bits gradient)))
      (is (every? #(instance? float/1 %) [output jvp-output tangent gradient])))))

(deftest delayed-pullback-owns-passive-copy-and-returns-fresh-flat-gradients
  (doseq [active '[A W]]
    (let [prepared (runtime/prepare (evidence (source :double 2 3 4 '[W A]) active))
          a (double-array a0) w (double-array w0)
          [output pb] ((:vjp prepared) (arguments prepared a w))
          expected (pull active 2 3 4 a0 w0 dy0)
          invoke-var #'runtime/invoke invoke @invoke-var saved (atom nil)]
      (dotimes [i (alength a)] (aset a i 19.0))
      (dotimes [i (alength w)] (aset w i -7.0))
      (dotimes [i (alength ^doubles output)] (aset ^doubles output i 99.0))
      (let [first-gradient (with-redefs-fn
                             {invoke-var (fn [executable environment]
                                           (reset! saved environment)
                                           (invoke executable environment))}
                             #(active-slot prepared active (pb (double-array dy0))))
            second-gradient (active-slot prepared active (pb (double-array dy0)))]
        (is (close? expected first-gradient 1.0e-12))
        (is (= (vec first-gradient) (vec second-gradient)))
        (is (not (identical? first-gradient second-gradient)))
        (is (not-any? #(or (identical? a %) (identical? w %) (identical? output %))
                      (vals @saved)))
        (is (not-any? #(contains? @saved %) '[A W]))
        (aset ^doubles first-gradient 0 123.0)
        (is (close? expected (active-slot prepared active (pb (double-array dy0))) 1.0e-12))
        ;; A new primal acquisition must take a new passive snapshot, rather than
        ;; retaining the old owner's array contents across unrelated invocations.
        (let [[_ new-pb] ((:vjp prepared) (arguments prepared a w))]
          (is (close? (pull active 2 3 4 (vec a) (vec w) dy0)
                      (active-slot prepared active (new-pb (double-array dy0))) 1.0e-12)))))))

(defn- source-attributes [program f]
  (let [[equation] (soac/equations program)
        operation (nth equation 3)
        changed (apply list (assoc (vec operation) 1 (f (second operation))))]
    (soac/make (soac/facts program)
               [(apply list (assoc (vec equation) 3 changed))] (soac/outputs program))))

(defn- source-fold-body
  ([program f] (source-fold-body program f false))
  ([program f recertify-algebra?]
   (let [[equation] (soac/equations program)
         operation (nth equation 3)
         attributes (second operation)
         {:keys [parameters body-results]} (soac/lambda-parts (last operation))
         body (f (first body-results))
         attributes (if recertify-algebra?
                      (assoc attributes :algebra
                             [(scan/certify-reassociation
                               {:acc (first (:accumulators attributes))
                                :init (first (:identities attributes)) :lambda body}
                               (first (:dtypes attributes)))])
                      attributes)
         changed (apply list (assoc (vec operation) 1 attributes (dec (count operation))
                                    (soac/lambda-form parameters [] [body])))]
     (soac/make (soac/facts program)
                [(apply list (assoc (vec equation) 3 changed))] (soac/outputs program)))))

(defn- colliding-snapshot-source []
  (let [program (source :double)
        collision 'ad-map-primal__passive_snapshot
        [equation] (soac/equations program)
        facts (update (soac/facts program) :values
                      #(-> % (dissoc 'C) (assoc collision (get % 'C))))]
    (soac/make facts [(apply list (assoc (vec equation) 2 [collision]))] [collision])))

(defn- conflicting-accumulator-tag [program]
  (let [[equation] (soac/equations program)
        operation (nth equation 3)
        {:keys [parameters body-results]} (soac/lambda-parts (last operation))
        parameters (assoc parameters 0 (vary-meta (first parameters) assoc :raster.type/tag 'float))
        changed (apply list (assoc (vec operation) (dec (count operation))
                                   (soac/lambda-form parameters [] body-results)))]
    (soac/make (soac/facts program) [(apply list (assoc (vec equation) 3 changed))]
               (soac/outputs program))))

(deftest unsupported-source-and-activity-refuse-before-host-materialization
  (let [program (source :double)
        source-ns (the-ns 'raster.ad.normalized-segmented-reduction-test)
        materializations (atom 0)]
    ;; Exercise the real map entry, observing that dispatch returns its complete
    ;; proof/phase object unchanged rather than rebuilding a look-alike bundle.
    (let [map-source (#'element/map-program :double)
          options {:active-input 'x}
          map-entry normalized/linearize-map-program
          delegated (atom nil)]
      (with-redefs [normalized/linearize-map-program
                    (fn [s n o]
                      (let [result (map-entry s n o)]
                        (reset! delegated [s n o result]) result))]
        (let [result (normalized/linearize-equation-program map-source source-ns options)]
          (is (identical? map-source (nth @delegated 0)))
          (is (identical? source-ns (nth @delegated 1)))
          (is (= options (nth @delegated 2)))
          (is (identical? result (nth @delegated 3))))))
    (with-redefs [route/materialize-host-program
                  (fn [& _] (swap! materializations inc)
                    (throw (ex-info "unexpected materialization" {:reason :test-materialization})))]
      (doseq [[label options] [[:missing {}] [:both {:active-inputs '[A W]}]
                               [:unknown {:active-input 'missing}]]]
        (testing (name label)
          (is (= :ad-normalized-segmented-activity
                 (reason #(normalized/linearize-equation-program program source-ns options))))))
      (doseq [[label expected build]
              [[:rank-two-input
                :ad-normalized-segmented-space
                #(soac/make (assoc-in (soac/facts program) [:values 'A :shape] [2 3])
                            (soac/equations program) (soac/outputs program))]
               [:rank-two-result
                :typed-soac-segmented-reduce-result-type
                #(source-attributes program (fn [attributes] (dissoc attributes :result-map)))]
               [:zero-axis :ad-normalized-segmented-boundary #(source :double 2 0 4 '[A W])]
               [:overflow-capacity :ad-normalized-segmented-space #(source :double Integer/MAX_VALUE 3 4 '[A W])]
               [:wrong-source-arithmetic
                :ad-normalized-segmented-boundary
                #(source-attributes program
                                    (fn [attributes]
                                      (assoc attributes :source-arithmetic numerics/retained-source-arithmetic)))]
               [:wrong-zero
                :reduction-nonidentity-init
                #(source-attributes program (fn [attributes] (assoc attributes :identities [1.0])))]
               [:extra-stage
                :ad-normalized-segmented-boundary
                #(source-attributes program (fn [attributes] (assoc attributes :stages [])))]
               [:result-axis-order
                :typed-soac-syntax
                #(source-attributes program
                                    (fn [attributes]
                                      (assoc attributes :result-map {:groups '[[[k 4] [i 2]]]})))]
               [:accumulator-add
                :reduction-not-associative
                #(source-fold-body program (fn [body] (apply list (assoc (vec body) 0 'raster.numeric/-))))]
               [:unapproved-fold-checkpoint
                :reduction-not-associative
                #(source-fold-body (source :float) (fn [body] (list 'clojure.core/double body)))]
               [:conflicting-formal-tag :ad-normalized-segmented-fold #(conflicting-accumulator-tag program)]
               [:compound-raw-axis
                :typed-soac-syntax
                #(source-attributes program (fn [attributes] (assoc attributes :extent '(clojure.core/+ 1 2))))]]]
        (testing (name label)
          (is (= expected (reason #(evidence (build) 'A))))))
      ;; Valid TypedSOAC and a freshly certified additive fold are insufficient:
      ;; squaring the selected read is not a bilinear product with the passive input.
      (let [square-source (source-fold-body
                           program
                           (fn [body]
                             (let [product (nth body 2) lhs (second product)]
                               (list (first body) (second body) (list (first product) lhs lhs))))
                           true)]
        (is (identical? square-source (soac/validate! square-source)))
        (is (= :ad-normalized-segmented-boundary (reason #(evidence square-source 'A)))))
      (is (= :ad-normalized-segmented-interface
             (reason #(evidence (colliding-snapshot-source) 'A))))
      (is (= :ad-normalized-map-boundary
             (reason #(normalized/linearize-map-program program source-ns {:active-input 'A}))))
      (is (zero? @materializations)))))

(defn- recertify [checked]
  (assoc checked :certificate (#'normalized/certificate checked)))

(deftest producer-loss-replays-retained-identities-and-refuses-before-backend-contact
  (let [program (summed-activation-source :double '[W A])
        checked (evidence program 'A)
        owner (canonical/adapt checked)
        reject-allocation (fn [& _] (throw (ex-info "sum replay allocated an identity" {})))
        [rebuilt validated]
        (with-redefs-fn {#'normalized/allocate-map-stage-interface reject-allocation
                        #'normalized/allocate-segmented-interface reject-allocation
                        #'normalized/node-equation-ids reject-allocation
                        #'normalized/linearize-program reject-allocation
                        #'normalized/linearize-equation-program reject-allocation
                        #'normalized/build-producer-sum-programs reject-allocation}
          #(vector (#'normalized/lift-sum-programs checked program (:interface (:sum-programs checked)))
                   (normalized/validate! checked)))]
    (is (identical? checked validated))
    (doseq [phase [:primal :linear :transpose]]
      (is (fingerprint/equivalent? (get-in checked [:sum-programs phase]) (get rebuilt phase))))
    (doseq [[field value] [[:seed-array 'A] [:seed-equation 'activation]
                          [:seed-index 'W] [:seed nil]]]
      (is (= :ad-normalized-sum-interface
             (reason #(normalized/validate!
                       (recertify (assoc-in checked [:sum-programs :interface field] value)))))))
    (let [coefficient (first (keys (:residual-bindings owner)))
          changed (assoc-in owner [:residual-bindings coefficient :origin :execution-equation] 'wrong)
          changed (assoc changed :certificate (#'canonical/certificate changed))]
      (is (= :ad-program-bundle-correspondence (reason #(canonical/validate! changed)))))
    (let [contacts (atom 0)
          reject-contact (fn [& _] (swap! contacts inc) (throw (Exception. "unexpected backend contact")))]
      (with-redefs-fn {#'equation-first/compile-typed reject-contact
                      #'link-module/instantiate! reject-contact
                      #'route/materialize-host-program reject-contact}
        #(is (= :normalized-device-interface
                (reason (fn [] (device/prepare-sum-owner owner [(double-array w0) (double-array a0)]
                                                        1.0 {:target :ocl:normalized-sum}))))))
      (is (zero? @contacts)))))

(deftest segmented-activation-keeps-node-replay-and-topology-boundaries
  (let [program (activated-source :double '[W A])
        facts (soac/facts program)
        [product activation] (soac/equations program)
        checked (evidence program 'A)
        owner (canonical/adapt checked)
        nodes (get-in checked [:map-programs :proof :nodes])
        reject-allocation (fn [& _] (throw (ex-info "mixed replay allocated an identity" {})))
        [rebuilt validated]
        (with-redefs-fn {#'normalized/allocate-map-stage-interface reject-allocation
                        #'normalized/allocate-segmented-interface reject-allocation
                        #'normalized/node-equation-ids reject-allocation
                        #'normalized/linearize-program reject-allocation}
          #(vector (#'normalized/reconstructed-map-proof checked) (normalized/validate! checked)))
        unary-map (fn [id result input]
                    (list '= id [result]
                          (list 'map {:index 'p :extent 8} [input] []
                                (soac/lambda-form '[element] [] '[(raster.numeric/* element element)]))))
        materializations (atom 0)]
    (is (identical? checked validated))
    (doseq [phase [:primal :linear :transpose]]
      (is (fingerprint/equivalent? (get-in checked [:map-programs phase]) (get rebuilt phase))))
    (with-redefs [route/materialize-host-program
                  (fn [& _] (swap! materializations inc)
                    (throw (ex-info "unexpected materialization" {:reason :test-materialization})))]
      (doseq [[label expected build]
              [[:map-before-product :ad-normalized-equation-boundary
                #(let [pre (list '= 'pre '[A]
                                 (list 'map {:index 'q :extent 6} '[X] []
                                       (soac/lambda-form '[element] [] '[element])))
                       f (-> facts (assoc :inputs '[W X])
                             (assoc-in [:values 'X] (get-in facts [:values 'A]))
                             (assoc-in [:values 'A :ownership] :owned)
                             (assoc-in [:equations 'pre] (soac/default-equation-facts)))]
                   (soac/make f [pre product activation] '[Y]))]
               [:extra-map-array :ad-normalized-segmented-chain-boundary
                #(let [map2 (list '= 'activation '[Y]
                                  (list 'map {:index 'p :extent 8} '[C Q] []
                                        (soac/lambda-form '[element extra] [] '[(raster.numeric/* element extra)])))
                       f (-> facts (update :inputs conj 'Q)
                             (assoc-in [:values 'Q] (av/tensor {:dtype :double :shape [8]})))]
                   (soac/make f [product map2] '[Y]))]
               [:fanout :ad-normalized-map-chain-boundary
                #(let [branch (unary-map 'branch 'Z 'C)
                       join (list '= 'join '[F]
                                  (list 'map {:index 'r :extent 8} '[Y Z] []
                                        (soac/lambda-form '[left right] [] '[(raster.numeric/+ left right)])))
                       f (-> facts
                             (assoc-in [:values 'Z] (get-in facts [:values 'Y]))
                             (assoc-in [:values 'F] (get-in facts [:values 'Y]))
                             (assoc-in [:equations 'branch] (soac/default-equation-facts))
                             (assoc-in [:equations 'join] (soac/default-equation-facts)))]
                   (soac/make f [product activation branch join] '[F]))]
               [:global-arithmetic-scope :ad-normalized-segmented-chain-boundary
                #(soac/make (assoc facts :attributes {:source-arithmetic (numerics/blas-source-arithmetic :double)})
                            [product activation] '[Y])]
               [:member-arithmetic-scope :ad-normalized-segmented-boundary
                #(let [operation (nth product 3)
                       changed (apply list (assoc (vec operation) 1
                                                  (assoc (second operation) :source-arithmetic numerics/retained-source-arithmetic)))
                       product (apply list (assoc (vec product) 3 changed))]
                   (soac/make facts [product activation] '[Y]))]]]
        (is (= expected (reason #(evidence (build) 'A))) (name label)))
      (doseq [[label expected changed]
              [[:node-order :ad-normalized-segmented-proof
                (assoc-in checked [:map-programs :proof :nodes] (vec (reverse nodes)))]
               [:actual-child-id-collision :ad-normalized-segmented-interface
                (assoc-in checked [:map-programs :proof :nodes 1 :equation-ids :primal]
                          (symbol (str (name (get-in nodes [0 :equation-ids :primal])) "__passive_snapshot")))]]]
        (is (= expected (reason #(normalized/validate! (recertify changed)))) (name label)))
      (let [coefficient (first (keys (:residual-bindings owner)))
            changed (assoc-in owner [:residual-bindings coefficient :origin :execution-equation] 'counterfeit)
            changed (assoc changed :certificate (#'canonical/certificate changed))]
        (is (= :ad-program-bundle-correspondence (reason #(canonical/validate! changed)))))
      (let [compiles (atom 0) instantiations (atom 0)]
        (with-redefs [equation-first/compile-typed
                      (fn [& _] (swap! compiles inc) (throw (Exception. "unexpected compile")))
                      link-module/instantiate!
                      (fn [& _] (swap! instantiations inc) (throw (Exception. "unexpected instantiate")))]
          (is (= :normalized-device-interface
                 (reason #(device/prepare-owner owner [(double-array w0) (double-array a0)]
                                                {:target :ocl:normalized-segmented})))))
        (is (zero? @compiles))
        (is (zero? @instantiations)))
      (is (zero? @materializations)))))

(defn- nonalias-accumulator-transpose [scalar]
  (let [program (:transpose scalar)
        gradient (first (soac/outputs program))
        equations (mapv (fn [equation]
                          (if (some #{gradient} (nth equation 2))
                            (let [operation (nth equation 3)
                                  {:keys [parameters body-results]} (soac/lambda-parts (last operation))
                                  alias (first body-results)
                                  _ (assert (symbol? alias) "expected the retained accumulator alias")
                                  lambda (soac/lambda-form parameters [] [(list 'raster.numeric/+ alias alias)])]
                              (apply list (assoc (vec equation) 3
                                                 (apply list (assoc (vec operation) (dec (count operation)) lambda)))))
                            equation))
                        (soac/equations program))]
    (assoc scalar :transpose (soac/make (soac/facts program) equations (soac/outputs program)))))

(defn- altered-snapshot-body [program]
  (let [equations (vec (soac/equations program))
        equation (peek equations)
        {:keys [attributes arrays captures lambda]} (soac/operation-parts equation)
        {:keys [parameters]} (soac/lambda-parts lambda)
        element (first parameters)
        changed (list '= (second equation) (nth equation 2)
                      (list 'map attributes arrays captures
                            (soac/lambda-form parameters []
                                              [(list 'raster.numeric/* element element)])))]
    (soac/make (soac/facts program) (conj (pop equations) changed) (soac/outputs program))))

(defn- altered-snapshot-equation-id [program]
  (let [equations (vec (soac/equations program))
        equation (peek equations)
        old-id (second equation)
        new-id 'counterfeit-snapshot
        facts (update (soac/facts program) :equations
                      #(-> % (dissoc old-id) (assoc new-id (get % old-id))))]
    (soac/make facts (conj (pop equations) (apply list (assoc (vec equation) 1 new-id)))
               (soac/outputs program))))

(deftest recertification-cannot-change-public-activity-flat-ports-or-snapshot-proof
  (let [checked (evidence (source :double) 'A)
        owner (canonical/adapt checked)
        coefficient (first (keys (:residual-bindings owner)))
        materializations (atom 0)]
    (let [reject-allocation (fn [& _] (throw (ex-info "segmented replay allocated an identity" {})))
          [rebuilt validated]
          (with-redefs-fn {#'normalized/allocate-map-stage-interface reject-allocation
                          #'normalized/allocate-segmented-interface reject-allocation
                          #'normalized/node-equation-ids reject-allocation
                          #'normalized/linearize-program reject-allocation}
            #(vector (#'normalized/reconstructed-map-proof checked) (normalized/validate! checked)))]
      (is (identical? checked validated))
      (is (= #{:source-equations :execution-equation :scalar-derivation :stage-interface :equation-ids}
             (set (keys (get-in checked [:map-programs :proof :nodes 0])))))
      (doseq [phase [:primal :linear :transpose]]
        (is (fingerprint/equivalent? (get-in checked [:map-programs phase]) (get rebuilt phase)))))
    (with-redefs [route/materialize-host-program
                  (fn [& _] (swap! materializations inc)
                    (throw (ex-info "unexpected materialization" {:reason :test-materialization})))]
      (doseq [[label expected change]
              [[:activity :ad-normalized-segmented-proof #(assoc-in % [:map-programs :interface :active-inputs] '[W])]
               [:extra-node-key :ad-normalized-segmented-proof #(assoc-in % [:map-programs :proof :nodes 0 :extra-lifecycle-field] true)]
               [:missing-derivation :ad-normalized-segmented-proof #(assoc-in % [:map-programs :proof :nodes 0 :scalar-derivation] nil)]
               [:scalar-namespace :ad-normalized-segmented-proof #(assoc-in % [:map-programs :proof :nodes 0 :scalar-derivation :source-ns]
                                                                          (the-ns 'clojure.core))]
               [:snapshot-body :ad-normalized-map-programs #(update-in % [:map-programs :primal] altered-snapshot-body)]
               [:snapshot-child-id :ad-normalized-map-programs #(update-in % [:map-programs :primal] altered-snapshot-equation-id)]
               [:phase-id-collision :ad-normalized-segmented-proof #(assoc-in % [:map-programs :proof :nodes 0 :equation-ids :linear]
                                              'ad-map-primal__passive_snapshot)]
               [:accumulator-alias :ad-normalized-derivation-correspondence #(update-in % [:map-programs :proof :nodes 0 :scalar-derivation]
                                              (comp recertify nonalias-accumulator-transpose))]
               [:scalar-activity :ad-normalized-bundle-interface #(assoc-in % [:map-programs :proof :nodes 0 :scalar-derivation :active-inputs] [])]
               [:node-link :ad-normalized-segmented-proof #(assoc-in % [:map-programs :proof :nodes 0 :execution-equation] 'counterfeit)]]]
        (testing (name label)
          (is (= expected (reason #(runtime/prepare (recertify (change checked))))))))
      ;; Generic scalar replay may reject this counterfeit before the specialized
      ;; fold guard. Exercise that bounded alias-only obligation directly as well.
      (let [scalar (get-in checked [:map-programs :proof :nodes 0 :scalar-derivation])]
        (is (= :ad-normalized-segmented-fold
               (reason #(#'normalized/exact-accumulator-cotangent!
                          (nonalias-accumulator-transpose scalar) (first (:active-inputs scalar)))))))
      (let [compiles (atom 0) instantiations (atom 0)]
        (with-redefs [equation-first/compile-typed
                      (fn [& _] (swap! compiles inc)
                        (throw (ex-info "unexpected device compilation" {:reason :test-device-compile})))
                      link-module/instantiate!
                      (fn [& _] (swap! instantiations inc)
                        (throw (ex-info "unexpected device instantiation" {:reason :test-device-instantiate})))]
          (is (= :normalized-device-interface
                 (reason #(device/prepare-owner owner [(double-array a0) (double-array w0)]
                                                {:target :ocl:normalized-segmented}))))
          (is (zero? @compiles))
          (is (zero? @instantiations))))
      (doseq [[label changed]
              [[:rank-two-gradient (assoc-in owner [:interface :cotangent-results 0 :value :shape] [2 3])]
               [:scalar-storage-for-array-snapshot (assoc-in owner [:residual-bindings coefficient :storage] :immutable-scalar)]
               [:external-snapshot (assoc-in owner [:residual-bindings coefficient :value :ownership] :external)]
               [:wrong-snapshot-origin (assoc-in owner [:residual-bindings coefficient :origin :equation] 'counterfeit)]]]
        (testing (name label)
          (let [changed (assoc changed :certificate (#'canonical/certificate changed))
                refusal (reason #(runtime/prepare-owner changed))]
            (is (= :ad-program-bundle-correspondence refusal)))))
      (is (zero? @materializations)))))

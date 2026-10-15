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
          a (mapv float [4097.0 0.33333334 -4096.0 -0.125 3.1415927 0.0009765625])
          w (mapv float [0.33333334 -2.5 0.125 3.1415927
                         2.7182817 0.000244140625 -0.6666667 1.5
                         0.33333334 -2.5 0.125 3.1415927])
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
    (with-redefs [route/materialize-host-program
                  (fn [& _] (swap! materializations inc)
                    (throw (ex-info "unexpected materialization" {:reason :test-materialization})))]
      (doseq [[label expected change]
              [[:activity :ad-normalized-segmented-proof #(assoc-in % [:map-programs :interface :active-inputs] '[W])]
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

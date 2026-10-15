(ns raster.ad.normalized-map-sum-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-sum-test :as construction]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- build [source active]
  (normalized/linearize-map-sum-program source (the-ns 'raster.ad.normalized-map-sum-runtime-test)
                                        {:active-array active}))
(defn- build-multi [source active]
  (normalized/linearize-map-sum-program source (the-ns 'raster.ad.normalized-map-sum-runtime-test)
                                        {:active-inputs active}))
(defn- args [prepared environment] (mapv environment (:inputs prepared)))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- surface [v dt] (#'construction/surface-source v dt))
(defn- identity-source [dt]
  (let [source (#'construction/source dt)
        equation (list '= 'map-0 '[mapped]
                       (list 'map {:index 'i :extent 'size} '[xs weights] []
                             (soac/lambda-form '[x w] '[x])))]
    (soac/make (soac/facts source) [equation (second (soac/equations source))] '[loss])))
(defn- product-source [dt]
  (let [source (#'construction/source dt)
        equation (list '= 'map-0 '[mapped]
                       (list 'map {:index 'i :extent 'size} '[xs weights] []
                             (soac/lambda-form '[x w] '[(raster.numeric/* x w)])))]
    (soac/make (soac/facts source) [equation (second (soac/equations source))] '[loss])))
(defn- fmul [left right] (float (* (float left) (float right))))
(defn- fadd [left right] (float (+ (float left) (float right))))
(defn- ordered-float-sum [values] (reduce fadd (float 0.0) values))

(deftest host-hints-do-not-replace-retained-scalar-arithmetic
  (let [project #'route/project-host-scalar-hints
        accumulator (with-meta 'total {:tag 'float :raster.type/tag 'float :line 12})
        array (with-meta 'mapped {:tag 'floats :raster.type/tag 'floats})
        receiver (with-meta 'selected-implementation {:tag 'IFn__float_float
                                                      :raster.op/original 'raster.numeric/+})
        source (list '.invk receiver accumulator (list 'clojure.core/aget array 'i))
        projected (project source)]
    (is (= source projected))
    (is (= (dissoc (meta accumulator) :tag) (meta (nth projected 2))))
    (is (= (meta receiver) (meta (second projected))))
    (is (= (meta array) (meta (second (nth projected 3)))))
    ;; Conflicting or unproved hints are not converted into matching semantic evidence.
    (doseq [symbol [(with-meta 'x {:tag 'double :raster.type/tag 'float})
                    (with-meta 'x {:tag 'float})]]
      (is (= (meta symbol) (meta (project symbol)))))
    (is (= (meta accumulator) (meta (second (project (list 'quote accumulator)))))))
  (let [bundle (build (surface #'construction/weighted-float-sum :float) 'xs)
        projected (route/materialize-host-map-reduce (get-in bundle [:sum-programs :primal]))
        nodes (tree-seq coll? seq (:body projected))
        scalar-uses (filter #(and (symbol? %) (= 'float (:raster.type/tag (meta %)))) nodes)]
    (is (seq scalar-uses))
    (is (every? #(nil? (:tag (meta %))) scalar-uses))
    (is (some #(and (symbol? %) (= 'floats (:tag (meta %)))) nodes))))

(deftest actual-weighted-sum-owns-excluded-and-active-coefficient-snapshots
  (doseq [[v dt array cast] [[#'construction/weighted-double-sum :double double-array double]
                            [#'construction/weighted-float-sum :float float-array float]]
          size [0 3] active '[xs weights]]
    (let [prepared (runtime/prepare (build (surface v dt) active))
          x0 (vec (take size [-2.0 0.0 3.0])) w0 (vec (take size [0.5 2.0 -1.0]))
          xs (array x0) weights (array w0)
          arguments (args prepared {'xs xs 'weights weights 'size size})
          dx (array (repeat size 0.25)) seed (cast -0.5)
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments dx)
          coefficients (mapv (fn [x w] (if (= active 'xs) (* 2.0 x w) (* x x))) x0 w0)
          expected-tangent (cast (reduce + 0.0 (map #(* % 0.25) coefficients)))]
      (is (= (v xs weights size) output jvp-output))
      (is (= (class (cast 0.0)) (class output) (class tangent)))
      (is (= expected-tangent tangent))
      (is (= x0 (vec xs))) (is (= w0 (vec weights)))
      ;; Scalar output/seed are immutable. Neither caller-owned array is retained by pullback.
      (dotimes [i size] (aset xs i (cast 99.0)) (aset weights i (cast -99.0)))
      (let [slots (pullback seed)
            gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) active))]
        (is (= (mapv #(cast (* % -0.5)) coefficients) (vec gradient)))
        (doseq [[id slot] (map vector (:inputs prepared) slots)]
          (is (= (= id active) (some? slot))))
        (is (= (* seed tangent) (reduce + 0.0 (map * (seq dx) (seq gradient)))))
        (dotimes [i size] (aset gradient i (cast 777.0)))
        (is (= (mapv #(cast (* % -0.5)) coefficients)
               (vec (nth (pullback seed) (.indexOf ^java.util.List (:inputs prepared) active)))))))))

(deftest ordered-multiple-active-product-sum-has-independent-host-math-and-snapshots
  (doseq [[dt array cast] [[:float float-array float] [:double double-array double]]
          n [0 3]]
    (let [active '[weights xs]
          bundle (build-multi (product-source dt) active)
          owner (canonical/adapt-map-sum bundle)
          prepared (runtime/prepare-owner owner)
          xs0 (vec (take n [-2.0 0.5 3.0]))
          weights0 (vec (take n [0.5 2.0 -1.0]))
          dxs0 (vec (take n [-1.0 0.75 0.5]))
          dweights0 (vec (take n [0.25 -0.5 2.0]))
          xs (array xs0) weights (array weights0)
          dxs (array dxs0) dweights (array dweights0)
          arguments (args prepared {'xs xs 'weights weights 'size (long n)})
          seed (cast -0.5)
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments [dweights dxs])
          products (if (= dt :float) (mapv fmul xs0 weights0) (mapv * xs0 weights0))
          tangent-lanes (if (= dt :float)
                          (mapv #(fadd (fmul %1 %2) (fmul %3 %4))
                                dxs0 weights0 dweights0 xs0)
                          (mapv #(+ (* %1 %2) (* %3 %4)) dxs0 weights0 dweights0 xs0))
          expected-output (cast (if (= dt :float) (ordered-float-sum products)
                                    (reduce + 0.0 products)))
          expected-tangent (cast (if (= dt :float) (ordered-float-sum tangent-lanes)
                                     (reduce + 0.0 tangent-lanes)))]
      (is (= active (get-in owner [:interface :active-inputs])))
      (is (= active (mapv :source (get-in owner [:interface :tangent-inputs]))
             (mapv :source (get-in owner [:interface :cotangent-results]))))
      (is (= expected-output output jvp-output))
      (is (= expected-tangent tangent))
      (when (= dt :float)
        (is (= (Float/floatToRawIntBits expected-output) (Float/floatToRawIntBits output)))
        (is (= (Float/floatToRawIntBits expected-tangent) (Float/floatToRawIntBits tangent))))
      ;; Both source arrays may change after acquisition: the pullback owns both coefficients.
      (dotimes [i n] (aset xs i (cast 91.0)) (aset weights i (cast -73.0)))
      (let [slots (pullback seed)
            gx (nth slots (.indexOf ^java.util.List (:inputs prepared) 'xs))
            gw (nth slots (.indexOf ^java.util.List (:inputs prepared) 'weights))
            expected-gx (mapv #(cast (* seed %)) weights0)
            expected-gw (mapv #(cast (* seed %)) xs0)]
        (is (= expected-gx (vec gx)))
        (is (= expected-gw (vec gw)))
        (when (= dt :float)
          (is (= (mapv #(Float/floatToRawIntBits %) expected-gx)
                 (mapv #(Float/floatToRawIntBits %) (vec gx))))
          (is (= (mapv #(Float/floatToRawIntBits %) expected-gw)
                 (mapv #(Float/floatToRawIntBits %) (vec gw)))))
        (is (nil? (nth slots (.indexOf ^java.util.List (:inputs prepared) 'size))))
        (is (< (Math/abs (- (* (double seed) (double tangent))
                            (+ (reduce + 0.0 (map * dweights0 (vec gw)))
                               (reduce + 0.0 (map * dxs0 (vec gx))))))
               (if (= dt :float) 1.0e-6 1.0e-12)))
        (is (= expected-gx
               (vec (nth (pullback seed) (.indexOf ^java.util.List (:inputs prepared) 'xs)))))
        (is (= expected-gw
               (vec (nth (pullback seed) (.indexOf ^java.util.List (:inputs prepared) 'weights)))))))))

(deftest multiple-active-product-sum-double-gradient-has-coordinate-fd-oracle
  (let [prepared (runtime/prepare (build-multi (product-source :double) '[weights xs]))
        xs (double-array [-2.0 3.0]) weights (double-array [0.5 -1.0])
        environment {'xs xs 'weights weights 'size (long 2)}
        [_ pullback] ((:vjp prepared) (args prepared environment))
        slots (pullback 0.25)]
    (doseq [active '[xs weights]
            i (range 2)]
      (let [gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) active))
            h 1.0e-5
            objective (fn [delta]
                        (let [changed (aclone ^doubles (get environment active))
                              _ (aset changed i (+ (aget changed i) delta))
                              values (assoc environment active changed)]
                          (* 0.25 (reduce + 0.0 (map * (seq ('xs values))
                                                     (seq ('weights values)))))))
            fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-10))))))

(deftest ordinary-primal-coordinate-fd-checks-either-selected-array
  (doseq [[v dt array cast h tolerance]
          [[#'construction/weighted-double-sum :double double-array double 1.0e-5 1.0e-8]
           [#'construction/weighted-float-sum :float float-array float 0.125 1.0e-6]]
          active '[xs weights]]
    (let [prepared (runtime/prepare (build (surface v dt) active))
          environment {'xs (array [-2.0 0.0 3.0]) 'weights (array [0.5 2.0 -1.0]) 'size 3}
          [_ pullback] ((:vjp prepared) (args prepared environment))
          gradient (nth (pullback (cast 0.5)) (.indexOf ^java.util.List (:inputs prepared) active))]
      (doseq [i (range 3)]
        (let [objective (fn [delta]
                          (let [changed (array (seq (get environment active)))
                                _ (aset changed i (cast (+ (aget changed i) delta)))
                                e (assoc environment active changed)]
                            ;; Existing ordinary deftm objective, not generated derivative code.
                            (* 0.5 (v (get e 'xs) (get e 'weights) 3))))
              fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
          (is (< (Math/abs (- fd (aget gradient i))) tolerance)))))))

(deftest identity-sum-uses-scalar-seed-with-no-array-coefficients
  (doseq [[dt array cast bits large] [[:float float-array float #(Float/floatToRawIntBits %) 33554432.0]
                                     [:double double-array double #(Double/doubleToRawLongBits %) 18014398509481984.0]]
          size [0 3]]
    (let [source (identity-source dt)
          bundle (build source 'xs)
          prepared (runtime/prepare bundle)
          xs (array (take size [large 1.0 (- large)])) weights (array size)
          arguments (args prepared {'xs xs 'weights weights 'size size})
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments xs)]
      (is (empty? (:residuals bundle)))
      ;; Original left-to-right accumulation loses the middle unit. A tree regrouping gives 1.
      (is (= (bits (cast 0.0)) (bits output) (bits jvp-output) (bits tangent)))
      (is (= (class (cast 0.0)) (class output) (class tangent)))
      (when (pos? size) (is (= 1.0 (+ (+ large (- large)) 1.0))))
      (let [gradient (nth (pullback (cast -0.5)) (.indexOf ^java.util.List (:inputs prepared) 'xs))]
        (is (= (vec (repeat size (cast -0.5))) (vec gradient))))
      (let [gradient (nth (pullback (cast -0.0)) (.indexOf ^java.util.List (:inputs prepared) 'xs))]
        (is (= (vec (repeat size (bits (cast -0.0)))) (mapv bits gradient)))))))

(deftest exact-seed-and-array-contracts-refuse-before-derivative-execution
  (doseq [[dt array cast wrong] [[:double double-array double (float 1.0)]
                                [:float float-array float (double 1.0)]]]
    (let [prepared (runtime/prepare (build (identity-source dt) 'xs))
          arguments (args prepared {'xs (array [1.0 2.0]) 'weights (array 2) 'size 2})
          [_ pullback] ((:vjp prepared) arguments)]
      (is (= :ad-map-runtime-scalar (reason #(pullback wrong))))
      (is (= :ad-map-runtime-scalar (reason #(pullback 1))))
      (is (= :ad-map-runtime-array (reason #((:jvp prepared) arguments (array 3)))))
      (is (= :ad-map-runtime-array
             (reason #((:vjp prepared)
                       (args prepared {'xs (array 1) 'weights (array 2) 'size 2}))))))))

(deftest stale-and-forged-producers-refuse-without-executing
  (let [bundle (build (identity-source :double) 'xs)
        prepared (runtime/prepare bundle)
        arguments (args prepared {'xs (double-array [1.0]) 'weights (double-array 1) 'size 1})
        [_ pullback] ((:vjp prepared) arguments)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke)
        compile-projection (ns-resolve 'raster.ad.normalized-map-runtime 'compile-projection)
        executions (atom 0)]
    (with-redefs-fn {revisions (constantly (update current :compiler-revision inc))
                    invoke (fn [& _] (swap! executions inc))}
      (fn []
        (is (= :ad-normalized-stale-rules (reason #((:vjp prepared) arguments))))
        (is (= :ad-normalized-stale-rules (reason #((:jvp prepared) arguments (double-array 1)))))
        (is (= :ad-normalized-stale-rules (reason #(pullback 1.0))))
        (is (zero? @executions))))
    (with-redefs-fn {compile-projection (fn [& _] (swap! executions inc))}
      (fn []
        (let [forged (assoc-in bundle [:sum-programs :interface :seed] 'xs)
              forged (assoc forged :certificate (#'normalized/certificate forged))]
          (is (= :ad-normalized-sum-interface (reason #(runtime/prepare forged))))
          (is (zero? @executions)))))))

(deftest materializer-preserves-exact-extent-ids-and-refuses-ignored-facets
  (let [source (soac/remap-values (identity-source :double) {'size [:shape :rows]})
        bundle (build source 'xs)
        prepared (runtime/prepare bundle)
        [output pullback] ((:vjp prepared)
                          (args prepared {'xs (double-array [2.0 3.0])
                                          'weights (double-array 2) [:shape :rows] 2}))]
    (is (= 5.0 output))
    (is (= [2.0 2.0]
           (vec (nth (pullback 2.0) (.indexOf ^java.util.List (:inputs prepared) 'xs)))))
    (let [program (get-in bundle [:sum-programs :primal])
          reduction (vec (second (soac/equations program)))
          operation (vec (nth reduction 3))
          altered (apply list (assoc (vec program) 2
                                     [(first (soac/equations program))
                                      (apply list (assoc reduction 3
                                                         (apply list (update operation 1 assoc :unknown-policy true))))]))]
      (is (= :typed-soac-host-map-boundary (reason #(route/materialize-host-map-reduce altered)))))
    (let [device-source (#'construction/change-facts source
                        #(reduce (fn [facts id] (assoc-in facts [:values id :memory-space] :device))
                                 % '[xs weights mapped loss]))]
      (is (= :typed-soac-host-map-boundary
             (reason #(runtime/prepare (build device-source 'xs))))))))

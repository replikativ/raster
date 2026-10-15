(ns raster.ad.normalized-excluded-map-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-map-element-test :as element]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as ra]
            [raster.par :as par]))

(deftm weighted-double [xs :- (Array double) weights :- (Array double) size :- Long]
  :- (Array double)
  (par/map [i size] (n/* (n/* (ra/aget xs i) (ra/aget xs i)) (ra/aget weights i))))
(deftm weighted-float [xs :- (Array float) weights :- (Array float) size :- Long]
  :- (Array float)
  (par/map [i size] (n/* (n/* (ra/aget xs i) (ra/aget xs i)) (ra/aget weights i))))
(deftm product-double [xs :- (Array double) weights :- (Array double) size :- Long]
  :- (Array double)
  (par/map [i size] (n/* (ra/aget xs i) (ra/aget weights i))))
(deftm product-float [xs :- (Array float) weights :- (Array float) size :- Long]
  :- (Array float)
  (par/map [i size] (n/* (ra/aget xs i) (ra/aget weights i))))

(defn- source [v dt]
  (let [result (gensym "weighted_result_")
        body (first (core/ensure-walked-body! v))
        options {:dtype dt :array-types {'xs dt 'weights dt} :scalar-types {'size :long}}]
    (frontend/form->program
     (frontend/normalize-source (list 'let* [result body] result) options) options)))

(defn- build [program active]
  (normalized/linearize-map-program program (the-ns 'raster.ad.normalized-excluded-map-test)
                                    {:active-array active}))
(defn- args [prepared environment] (mapv environment (:inputs prepared)))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
(defn- alter-facts [program f]
  (apply list (assoc (vec program) 1 (f (soac/facts program)))))

(deftest actual-weighted-maps-snapshot-both-inputs-with-one-active-array
  (doseq [[v dt array cast] [[#'weighted-double :double double-array double]
                            [#'weighted-float :float float-array float]]
          size [0 3]
          active ['xs 'weights]]
    (let [program (source v dt)
          bundle (build program active)
          prepared (runtime/prepare bundle)
          xs0 (vec (take size [-2.0 0.0 3.0]))
          ws0 (vec (take size [0.5 2.0 -1.0]))
          xs (array xs0) weights (array ws0)
          arguments (args prepared {'xs xs 'weights weights 'size size})
          dx (array (repeat size 0.25)) dy (array (repeat size -0.5))
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments dx)
          coefficients (mapv (fn [x w] (if (= active 'xs) (* 2.0 x w) (* x x))) xs0 ws0)]
      (is (identical? program (get-in bundle [:map-programs :source])))
      (is (= [active] (get-in bundle [:map-programs :interface :active-inputs])))
      (is (= (vec (v xs weights size)) (vec output) (vec jvp-output)))
      (is (= xs0 (vec xs)))
      (is (= ws0 (vec weights)))
      (is (= (mapv #(cast (* % 0.25)) coefficients) (vec tangent)))
      (dotimes [i size]
        (aset xs i (cast 99.0)) (aset weights i (cast -99.0)) (aset output i (cast 123.0)))
      (let [slots (pullback dy)
            gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) active))]
        (is (= (mapv #(cast (* % -0.5)) coefficients) (vec gradient)))
        (doseq [[id slot] (map vector (:inputs prepared) slots)]
          (is (= (= id active) (some? slot))))
        (is (= (reduce + (map * (seq tangent) (seq dy)))
               (reduce + (map * (seq dx) (seq gradient)))))
        (dotimes [i size] (aset gradient i (cast 777.0)))
        (is (= (mapv #(cast (* % -0.5)) coefficients)
               (vec (nth (pullback dy) (.indexOf ^java.util.List (:inputs prepared) active)))))))))

(deftest binary-product-supports-ordered-multiple-active-arrays
  (doseq [[v dt array cast] [[#'product-double :double double-array double]
                            [#'product-float :float float-array float]]]
    (let [program (source v dt)
          active '[weights xs]
          bundle (normalized/linearize-map-program
                  program (the-ns 'raster.ad.normalized-excluded-map-test)
                  {:active-inputs active})
          owner (canonical/adapt-map bundle)
          prepared (runtime/prepare-owner owner)
          xs0 [-2.0 0.0 3.0] ws0 [0.5 2.0 -1.0]
          xs (array xs0) weights (array ws0)
          arguments (args prepared {'xs xs 'weights weights 'size 3})
          dweights (array [0.25 -0.5 2.0]) dxs (array [-1.0 0.75 0.5])
          dy (array [-0.5 2.0 0.25])
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments [dweights dxs])
          expected-y (mapv #(cast (* %1 %2)) xs0 ws0)
          ;; The multiplication rule preserves source operand contribution order, independently
          ;; of the requested tangent-port order: dx*w is accumulated before dw*x.
          expected-tangent (mapv #(cast (+ (cast (* %1 %2)) (cast (* %3 %4))))
                                 (vec dxs) ws0 (vec dweights) xs0)]
      (is (= active (get-in bundle [:map-programs :interface :active-inputs])
             (get-in owner [:interface :active-inputs])))
      (is (= 2 (count (get-in owner [:interface :tangent-inputs]))
             (count (get-in owner [:interface :cotangent-results]))))
      (is (= 2 (count (get-in (canonical/execution-residual-plan owner) [:owned-arrays]))))
      (is (not (contains? (:interface owner) :tangent-input)))
      (is (not (contains? (:interface owner) :cotangent-result)))
      (is (= expected-y (vec output) (vec jvp-output)))
      (is (= expected-tangent (vec tangent)))
      (is (= :ad-map-runtime-tangents
             (reason #((:jvp prepared) arguments dweights))))
      (is (= :ad-map-runtime-tangents
             (reason #((:jvp prepared) arguments [dweights]))))
      (when (= :float dt)
        (is (= (mapv #(Float/floatToRawIntBits %) expected-y)
               (mapv #(Float/floatToRawIntBits %) (vec output))))
        (is (= (mapv #(Float/floatToRawIntBits %) expected-tangent)
               (mapv #(Float/floatToRawIntBits %) (vec tangent)))))
      ;; Delayed reverse owns both coefficient arrays and reads neither caller array nor output.
      (dotimes [i 3]
        (aset xs i (cast 91.0))
        (aset weights i (cast -73.0))
        (aset output i (cast 44.0)))
      (let [slots (pullback dy)
            gx (nth slots (.indexOf ^java.util.List (:inputs prepared) 'xs))
            gw (nth slots (.indexOf ^java.util.List (:inputs prepared) 'weights))]
        (is (= (mapv #(cast (* %1 %2)) ws0 (vec dy)) (vec gx)))
        (is (= (mapv #(cast (* %1 %2)) xs0 (vec dy)) (vec gw)))
        (is (nil? (nth slots (.indexOf ^java.util.List (:inputs prepared) 'size))))
        (is (= (reduce + (map * (vec tangent) (vec dy)))
               (+ (reduce + (map * (vec dweights) (vec gw)))
                  (reduce + (map * (vec dxs) (vec gx))))))))))

(deftest multiple-active-double-gradient-has-independent-finite-difference-corroboration
  (let [program (source #'product-double :double)
        prepared (runtime/prepare
                  (normalized/linearize-map-program
                   program (the-ns 'raster.ad.normalized-excluded-map-test)
                   {:active-inputs '[weights xs]}))
        xs (double-array [-2.0 3.0]) weights (double-array [0.5 -1.0])
        environment {'xs xs 'weights weights 'size 2}
        seed (double-array [0.25 -0.5])
        [_ pullback] ((:vjp prepared) (args prepared environment))
        slots (pullback seed)]
    (doseq [active ['xs 'weights]
            i (range 2)]
      (let [gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) active))
            h 1.0e-5
            objective (fn [delta]
                        (let [changed (aclone ^doubles (get environment active))
                              _ (aset changed i (+ (aget changed i) delta))
                              values (assoc environment active changed)]
                          (reduce + (map * (seq seed)
                                         (seq (product-double ('xs values) ('weights values) 2))))))
            fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-8))))))

(deftest plural-singleton-and-empty-multiple-active-map-retain-host-contracts
  (let [program (source #'product-double :double)
        source-ns (the-ns 'raster.ad.normalized-excluded-map-test)
        singleton (runtime/prepare
                   (normalized/linearize-map-program program source-ns
                                                    {:active-inputs '[weights]}))
        multi (runtime/prepare
               (normalized/linearize-map-program program source-ns
                                                {:active-inputs '[weights xs]}))
        environment {'xs (double-array 0) 'weights (double-array 0) 'size 0}
        singleton-arguments (args singleton environment)
        multi-arguments (args multi environment)
        [singleton-y singleton-pb] ((:vjp singleton) singleton-arguments)
        [multi-y multi-pb] ((:vjp multi) multi-arguments)
        [_ singleton-tangent] ((:jvp singleton) singleton-arguments (double-array 0))
        [_ multi-tangent] ((:jvp multi) multi-arguments [(double-array 0) (double-array 0)])
        singleton-slots (singleton-pb (double-array 0))
        multi-slots (multi-pb (double-array 0))]
    (is (= '[weights] (get-in (canonical/adapt-map
                               (normalized/linearize-map-program
                                program source-ns {:active-inputs '[weights]}))
                              [:interface :active-inputs])))
    (doseq [value [singleton-y multi-y singleton-tangent multi-tangent]]
      (is (zero? (alength value))))
    (is (nil? (nth singleton-slots (.indexOf ^java.util.List (:inputs singleton) 'xs))))
    (is (some? (nth singleton-slots (.indexOf ^java.util.List (:inputs singleton) 'weights))))
    (is (some? (nth multi-slots (.indexOf ^java.util.List (:inputs multi) 'xs))))
    (is (some? (nth multi-slots (.indexOf ^java.util.List (:inputs multi) 'weights))))))

(deftest canonical-multiple-active-ports-cannot-be-dropped-or-reordered
  (let [owner (canonical/adapt-map
               (normalized/linearize-map-program
                (source #'product-double :double)
                (the-ns 'raster.ad.normalized-excluded-map-test)
                {:active-inputs '[weights xs]}))]
    (doseq [changed [(update-in owner [:interface :active-inputs] #(vec (reverse %)))
                     (update-in owner [:interface :tangent-inputs] pop)
                     (update-in owner [:interface :tangent-inputs] #(vec (reverse %)))
                     (update-in owner [:interface :cotangent-results] pop)
                     (update-in owner [:interface :cotangent-results] #(vec (reverse %)))]]
      (is (= :ad-program-bundle-correspondence
             (reason #(canonical/validate! changed)))))))

(deftest independent-coordinate-fd-selects-either-array
  (doseq [active ['xs 'weights]]
    (let [prepared (runtime/prepare (build (source #'weighted-double :double) active))
          xs (double-array [-2.0 3.0]) weights (double-array [0.5 -1.0])
          environment {'xs xs 'weights weights 'size 2}
          seed (double-array [0.25 -0.5])
          [_ pullback] ((:vjp prepared) (args prepared environment))
          gradient (nth (pullback seed) (.indexOf ^java.util.List (:inputs prepared) active))]
      (doseq [i (range 2)]
        (let [h 1.0e-5
              objective (fn [delta]
                          (let [changed (aclone ^doubles (get environment active))
                                _ (aset changed i (+ (aget changed i) delta))
                                values (assoc environment active changed)]
                            ;; Independent ordinary primal, not the generated runtime.
                            (reduce + (map * (seq seed)
                                           (seq (weighted-double ('xs values) ('weights values) 2))))))
              fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
          (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-8)))))))

(deftest array-elements-and-scalar-captures-use-existing-parameter-layout
  (let [base (#'element/map-program :double
                                   {:captures '[scale bias]
                                    :body '(raster.numeric/+ (raster.numeric/* element scale) bias)})
        [equation] (soac/equations base)
        tensor (get-in (soac/facts base) [:values 'x])
        operation (list 'map {:index 'i :extent 'n} '[x weights] '[scale bias]
                        (soac/lambda-form '[element weight scale bias]
                                          '[(raster.numeric/+
                                             (raster.numeric/* (raster.numeric/* element weight) scale)
                                             bias)]))
        program (soac/make (-> (soac/facts base)
                              (assoc-in [:values 'weights] tensor)
                              (update :inputs conj 'weights))
                           [(list '= (second equation) '[y] operation)] '[y])
        bundle (build program 'x)
        prepared (runtime/prepare bundle)
        [_ pullback] ((:vjp prepared) (args prepared {'n 2 'x (double-array [2.0 -1.0])
                                                    'weights (double-array [4.0 5.0])
                                                    'scale 3.0 'bias 1.0}))]
    (is (= {'scale 'scale} (get-in bundle [:map-programs :interface :scalar-captures])))
    (is (contains? (get-in bundle [:map-programs :interface :residual-arrays]) 'weight))
    (is (= [12.0 15.0]
           (vec (nth (pullback (double-array [1.0 1.0]))
                     (.indexOf ^java.util.List (:inputs prepared) 'x)))))))

(deftest ambiguous-or-forged-active-selection-never-grants-a-derivative
  (let [program (source #'weighted-double :double)
        source-ns (the-ns 'raster.ad.normalized-excluded-map-test)]
    (is (= :ad-normalized-map-active-array
           (reason #(normalized/linearize-map-element program source-ns))))
    (is (= :ad-normalized-map-active-array
           (reason #(normalized/linearize-map-program program source-ns))))
    (doseq [options [{:active-array 'missing} {:active-array 'size} {:active-array nil}
                     {:active-array 'xs :active-arrays ['xs 'weights]}
                     {:active-inputs []} {:active-inputs ['xs 'xs]}
                     {:active-inputs ['xs 'size]} {:active-inputs ['xs 'missing]}
                     {:active-inputs ['xs 'weights] :active-input 'xs}
                     {:active-inputs ['xs 'weights] :active-array 'xs}]]
      (is (= :ad-normalized-map-active-array
             (reason #(normalized/linearize-map-program program source-ns options)))))
    (let [bundle (build program 'xs)
          changed (assoc-in bundle [:map-programs :interface :active-inputs] ['weights])
          changed (assoc changed :certificate (#'normalized/certificate changed))]
      ;; The retained scalar witness is now checked against selected activity
      ;; before reconstructing the outer map interface.
      (is (= :ad-normalized-map-node-derivation (reason #(normalized/validate! changed)))))
    (let [one (#'element/map-program :double)]
      (is (= '[element] (:active-inputs (normalized/linearize-map-element one source-ns)))))))

(deftest excluded-arrays-retain-the-same-type-shape-and-storage-fence
  (let [program (source #'weighted-double :double)
        equation-id (second (first (soac/equations program)))]
    (doseq [[path value] [[[:values 'weights :dtype] :float]
                          [[:values 'weights :shape] '[other]]
                          [[:values 'weights :placement] {:device :other}]
                          [[:equations equation-id :aliases] {'weights 'xs}]]]
      (is (some? (reason #(build (alter-facts program (fn [facts] (assoc-in facts path value))) 'xs)))))
    (is (= :ad-normalized-map-boundary
           (reason #(build (alter-facts program
                                        (fn [facts] (-> facts (assoc :effects #{:memory/write})
                                                        (assoc-in [:equations equation-id :effects]
                                                                  #{:memory/write})))) 'xs))))
    (doseq [conversion ['clojure.core/float 'clojure.core/long]]
      (let [[equation] (soac/equations program)
          operation (nth equation 3)
          {:keys [parameters]} (soac/lambda-parts (nth operation 4))
          changed (apply list (assoc (vec operation) 4
                                    (soac/lambda-form parameters
                                                      [(list 'clojure.core/double
                                                             (list conversion (first parameters)))])))
          narrowed (soac/make (soac/facts program)
                              [(list '= (second equation) (nth equation 2) changed)]
                              (soac/outputs program))]
        (is (some? (reason #(build narrowed 'xs))))))
    (let [prepared (runtime/prepare (build program 'xs))
          valid {'xs (double-array 2) 'weights (double-array 2) 'size 2}]
      (doseq [bad [(assoc valid 'weights (float-array 2))
                   (assoc valid 'weights (double-array 1))]]
        (is (= :ad-map-runtime-array
               (reason #((:vjp prepared) (args prepared bad)))))))))

(deftest stale-or-forged-selection-refuses-before-execution
  (let [bundle (build (source #'weighted-double :double) 'xs)
        prepared (runtime/prepare bundle)
        arguments (args prepared {'xs (double-array 2) 'weights (double-array 2) 'size 2})
        [_ pullback] ((:vjp prepared) arguments)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke)
        executions (atom 0)
        forged (assoc-in bundle [:map-programs :interface :active-inputs] ['weights])
        forged (assoc forged :certificate (#'normalized/certificate forged))]
    (with-redefs-fn {invoke (fn [& _] (swap! executions inc))}
      (fn []
        ;; Reject the inconsistent node witness before any runtime invocation.
        (is (= :ad-normalized-map-node-derivation (reason #(runtime/prepare forged))))
        (with-redefs-fn {revisions (constantly (update current :template-revision inc))}
          (fn []
            (is (= :ad-normalized-stale-rules (reason #((:vjp prepared) arguments))))
            (is (= :ad-normalized-stale-rules
                   (reason #((:jvp prepared) arguments (double-array 2)))))
            (is (= :ad-normalized-stale-rules (reason #(pullback (double-array 2)))))))
        (is (zero? @executions))))))

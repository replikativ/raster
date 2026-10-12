(ns raster.ad.normalized-map-runtime-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-element-test :as element]
            [raster.ad.normalized-surface-map-test :as surface]))

(defn- bundle [program]
  (normalized/linearize-map-program program (the-ns 'raster.ad.normalized-map-runtime-test)))
(defn- args [prepared environment] (mapv environment (:inputs prepared)))
(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest real-surface-map-snapshots-and-explicit-seed-pairing
  (doseq [[v dt array cast] [[#'surface/square-double :double double-array double]
                             [#'surface/square-float :float float-array float]]
          size [0 3]]
    (let [prepared (runtime/prepare (bundle (#'surface/source-program v dt)))
          original (vec (take size [-2.0 0.0 3.0]))
          x (array original)
          arguments (args prepared {'xs x 'size size})
          dx (array (repeat size 0.25)) dy (array (repeat size -0.5))
          [output pullback] ((:vjp prepared) arguments)
          [jvp-output tangent] ((:jvp prepared) arguments dx)]
      (is (= (vec (v x size)) (vec output) (vec jvp-output)))
      (is (= (class x) (class output) (class tangent)))
      (dotimes [i size] (aset x i (cast 99.0)) (aset output i (cast -99.0)))
      (let [slots (pullback dy)
            gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) 'xs))]
        (is (= (mapv #(cast (* 2.0 % -0.5)) original) (vec gradient)))
        (is (= (mapv #(when (some? %) (vec %)) slots)
               (mapv #(when (some? %) (vec %)) (pullback dy))))
        (is (= (reduce + (map * (seq tangent) (seq dy)))
               (reduce + (map * (seq dx) (seq gradient)))))
        (dotimes [i size] (aset gradient i (cast 123.0)))
        (is (= (mapv #(cast (* 2.0 % -0.5)) original)
               (vec (nth (pullback dy) (.indexOf ^java.util.List (:inputs prepared) 'xs))))))
      (is (= (vec original) (mapv #(cast (/ % 0.5)) (seq tangent)))))))

(deftest inactive-captures-and-coordinate-finite-differences
  (let [program (#'element/map-program :double
                                      {:captures '[scale bias]
                                       :body '(raster.numeric/+ (raster.numeric/* element scale) bias)})
        prepared (runtime/prepare (bundle program))
        x (double-array [2.0 -1.0])
        environment {'n 2 'x x 'scale 3.0 'bias 5.0}
        arguments (args prepared environment)
        [output pullback] ((:vjp prepared) arguments)
        seed (double-array [0.5 -2.0])
        slots (pullback seed)
        gradient (nth slots (.indexOf ^java.util.List (:inputs prepared) 'x))]
    (is (= [11.0 2.0] (vec output)))
    (is (= [1.5 -6.0] (vec gradient)))
    (doseq [[id slot] (map vector (:inputs prepared) slots)]
      (is (= (= id 'x) (some? slot))))
    (doseq [i (range 2)]
      (let [h 1.0e-5
            objective (fn [delta]
                        (let [xs (aclone x)]
                          (aset xs i (+ (aget xs i) delta))
                          (reduce + (map * (seq seed)
                                         (seq (first ((:vjp prepared)
                                                      (args prepared (assoc environment 'x xs)))))))))
            fd (/ (- (objective h) (objective (- h))) (* 2.0 h))]
        (is (< (Math/abs (- fd (aget ^doubles gradient i))) 1.0e-8))))))

(deftest arguments-must-match-checked-tensor-and-scalar-spaces
  (let [prepared (runtime/prepare (bundle (#'element/map-program :double {:captures '[scale]})))
        valid {'n 2 'x (double-array [1.0 2.0]) 'scale 3.0}]
    (is (= :ad-map-runtime-arity (reason #((:vjp prepared) []))))
    (doseq [bad [(assoc valid 'n -1) (assoc valid 'n (inc (long Integer/MAX_VALUE)))]]
      (is (= :ad-map-runtime-extent (reason #((:vjp prepared) (args prepared bad))))))
    (is (= :ad-map-runtime-extent-type
           (reason #((:vjp prepared) (args prepared (assoc valid 'n (int 2)))))))
    (doseq [bad [(assoc valid 'x (float-array 2)) (assoc valid 'x (double-array 3))]]
      (is (= :ad-map-runtime-array (reason #((:vjp prepared) (args prepared bad))))))
    (is (= :ad-map-runtime-capture
           (reason #((:vjp prepared) (args prepared (assoc valid 'scale (float 3.0)))))))
    (let [[_ pullback] ((:vjp prepared) (args prepared valid))]
      (is (= :ad-map-runtime-array (reason #(pullback (double-array 1)))))
      (is (= :ad-map-runtime-array (reason #(pullback (float-array 2)))))
      (is (= :ad-map-runtime-array
             (reason #((:jvp prepared) (args prepared valid) (double-array 1))))))))

(deftest runtime-does-not-sidestep-bundle-validation
  (let [checked (bundle (#'element/map-program :double))]
    (is (some? (reason #(runtime/prepare (assoc checked :certificate {})))))
    (is (= :ad-program-bundle-scope
           (reason #(runtime/prepare (normalized/linearize-map-element
                                      (#'element/map-program :double)
                                      (the-ns 'raster.ad.normalized-map-runtime-test))))))))

(deftest stale-rules-refuse-before-primal-or-pullback-execution
  (let [checked (bundle (#'element/map-program :double))
        prepared (runtime/prepare checked)
        arguments (args prepared {'n 2 'x (double-array [1.0 2.0])})
        [_ pullback] ((:vjp prepared) arguments)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)
        invoke (ns-resolve 'raster.ad.normalized-map-runtime 'invoke)
        execution-count (atom 0)]
    (with-redefs-fn {revisions (constantly (update current :template-revision inc))
                    invoke (fn [& _] (swap! execution-count inc))}
      (fn []
        (is (= :ad-normalized-stale-rules (reason #((:vjp prepared) arguments))))
        (is (= :ad-normalized-stale-rules
               (reason #((:jvp prepared) arguments (double-array 2)))))
        (is (= :ad-normalized-stale-rules (reason #(pullback (double-array 2)))))
        (is (zero? @execution-count))))))

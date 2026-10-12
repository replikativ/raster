(ns raster.ad.normalized-density-witness-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.math :as math]
            [raster.par :as par]
            [raster.sci.distributions :as dist]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.inline :as inline]))

;; Pointwise prediction spaces, not a claim that vector-latent regression, records,
;; general equation chains or the default AD engine have migrated.
(deftm gaussian-score [predictions :- (Array double) observations :- (Array double)
                       sigma :- Double size :- Long] :- Double
  (let [terms (par/map [i size]
                (let [z (n// (n/- (ra/aget observations i) (ra/aget predictions i)) sigma)]
                  (n/- (n/- (n/- 0.0 (Math/log sigma)) 0.9189385332046727)
                       (n/* 0.5 (n/* z z)))))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget terms i)))))

(deftm logistic-score [predictions :- (Array double) observations :- (Array double)
                       size :- Long] :- Double
  (let [terms (par/map [i size]
                (let [eta (ra/aget predictions i) y (ra/aget observations i)]
                  (n/- (n/* y eta) (Math/log (n/+ 1.0 (Math/exp eta))))))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget terms i)))))

(deftm logistic-float-score [predictions :- (Array float) observations :- (Array float)
                             size :- Long] :- Float
  (let [terms (par/map [i size]
                (let [eta (ra/aget predictions i) y (ra/aget observations i)]
                  (n/- (n/* y eta) (math/log (n/+ (float 1.0) (math/exp eta))))))]
    (par/reduce total (float 0.0) i size (n/+ total (ra/aget terms i)))))

(defn- source-program [v dt]
  (let [body (first (core/ensure-walked-body! v))
        result (gensym "density_result_")
        options {:dtype dt
                 :array-types {'predictions dt 'observations dt}
                 :scalar-types (cond-> {'size :long}
                                 (= v #'gaussian-score) (assoc 'sigma :double))}]
    (frontend/form->program
     (frontend/normalize-source
      (inline/flatten-nested-lets (list 'let* [result body] result)) options) options)))

(deftest actual-density-maps-compose-with-ordered-sum-and-excluded-data
  (doseq [[v dt array cast h tolerance oracle derivative]
          [[#'gaussian-score :double double-array double 1.0e-5 1.0e-8
            (fn [p y] (dist/logpdf (dist/->Normal p 1.7) y))
            (fn [p y] (/ (- y p) (* 1.7 1.7)))]
           [#'logistic-score :double double-array double 1.0e-5 1.0e-8
            (fn [p y]
              ;; Independent probability-domain oracle on moderate logits.
              (let [probability (/ 1.0 (+ 1.0 (Math/exp (- p))))]
                (Math/log (if (= y 1.0) probability (- 1.0 probability)))))
            (fn [p y] (- y (/ 1.0 (+ 1.0 (Math/exp (- p))))))]
           [#'logistic-float-score :float float-array float 0.005 8.0e-4
            (fn [p y]
              (let [probability (/ 1.0 (+ 1.0 (Math/exp (- p))))]
                (Math/log (if (= y 1.0) probability (- 1.0 probability)))))
            (fn [p y] (- y (/ 1.0 (+ 1.0 (Math/exp (- p))))))]]
          size [0 3]]
    (let [p0 (vec (take size [-0.5 0.25 0.75]))
          y0 (vec (take size [1.0 0.0 1.0]))
          predictions (array p0) observations (array y0)
          source (source-program v dt)
          bundle (normalized/linearize-map-sum-program
                  source *ns* {:active-array 'predictions})
          owner (canonical/adapt-map-sum bundle)
          prepared (runtime/prepare bundle)
          args (mapv {'predictions predictions 'observations observations
                      'sigma 1.7 'size (long size)} (:inputs prepared))
          slot (.indexOf ^java.util.List (:inputs prepared) 'predictions)
          [value pb] ((:vjp prepared) args)
          gradients (pb (cast 1.0)) gradient (nth gradients slot)
          direction (array (repeat size 0.125))
          [jvalue tangent] ((:jvp prepared) args direction)
          ordinary (fn [xs]
                     (if (= v #'gaussian-score)
                       (gaussian-score xs observations 1.7 size)
                       (v xs observations size)))
          expected (reduce + 0.0 (map oracle p0 y0))
          arithmetic-tolerance (if (= dt :float) 1.0e-6 1.0e-12)]
      (is (= '[map reduce] (mapv (comp :kind soac/operation-parts) (soac/equations source))))
      (is (identical? source (:source owner)))
      (is (identical? owner (canonical/validate! owner)))
      (is (= (get-in bundle [:sum-programs :primal]) (:primal owner)))
      (is (= (get-in bundle [:sum-programs :linear]) (:linear owner)))
      (is (= (get-in bundle [:sum-programs :transpose]) (:transpose owner)))
      (is (< (Math/abs (- expected value)) arithmetic-tolerance))
      (is (= (ordinary predictions) value jvalue))
      (is (= (class predictions) (class gradient)))
      (doseq [[id g] (map vector (:inputs prepared) gradients)]
        (is (= (= id 'predictions) (some? g))))
      (doseq [i (range size)]
        (let [plus (array p0) minus (array p0)]
          (aset plus i (cast (+ (aget plus i) h)))
          (aset minus i (cast (- (aget minus i) h)))
          (is (< (Math/abs (- (derivative (nth p0 i) (nth y0 i)) (aget gradient i)))
                 arithmetic-tolerance))
          (is (< (Math/abs (- (/ (- (ordinary plus) (ordinary minus)) (* 2.0 h))
                             (aget gradient i))) tolerance))))
      (is (< (Math/abs (- tangent (reduce + 0.0 (map * (seq direction) (seq gradient)))))
             arithmetic-tolerance))
      (is (= p0 (vec predictions))) (is (= y0 (vec observations)))
      (let [[_ fresh-pb] ((:vjp prepared) args)]
        ;; Mutation precedes this closure's first pullback: snapshots belong to
        ;; forward acquisition, not lazy first-backward initialization.
        (dotimes [i size] (aset predictions i (cast 99.0)) (aset observations i (cast -99.0)))
        (is (= (vec gradient) (vec (nth (fresh-pb (cast 1.0)) slot))))
        (is (= (vec gradient) (vec (nth (pb (cast 1.0)) slot))))))))

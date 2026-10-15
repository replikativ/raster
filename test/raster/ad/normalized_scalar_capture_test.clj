(ns raster.ad.normalized-scalar-capture-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.math :as math]
            [raster.par :as par]
            [raster.sci.distributions :as dist]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-map-runtime :as runtime]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.scalar.inline :as inline]))

(deftm gaussian-double [observations :- (Array double) mu :- Double sigma :- Double size :- Long]
  :- Double
  (let [terms (par/map [i size]
                (let [z (n// (n/- (ra/aget observations i) mu) sigma)]
                  (n/- (n/- (n/- 0.0 (Math/log sigma)) 0.9189385332046727)
                       (n/* 0.5 (n/* z z)))))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget terms i)))))

(deftm gaussian-float [observations :- (Array float) mu :- Float sigma :- Float size :- Long]
  :- Float
  (let [terms (par/map [i size]
                (let [z (n// (n/- (ra/aget observations i) mu) sigma)]
                  (n/- (n/- (n/- (float 0.0) (math/log sigma)) (float 0.9189385332046727))
                       (n/* (float 0.5) (n/* z z)))))]
    (par/reduce total (float 0.0) i size (n/+ total (ra/aget terms i)))))

(defn- source [v dt]
  (let [body (first (core/ensure-walked-body! v)) result (gensym "capture_result_")
        options {:dtype dt :array-types {'observations dt}
                 :scalar-types {'mu dt 'sigma dt 'size :long}}]
    (frontend/form->program
     (frontend/normalize-source
      (inline/flatten-nested-lets (list 'let* [result body] result)) options) options)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest scalar-capture-cotangents-use-the-original-ordered-reduction
  (doseq [[v dt array cast h fd-tolerance arithmetic-tolerance]
          [[#'gaussian-double :double double-array double 1.0e-5 1.0e-8 1.0e-12]
           [#'gaussian-float :float float-array float 0.01 8.0e-4 1.0e-6]]
          size [0 3]]
    (let [program (source v dt)
          bundle (normalized/linearize-map-sum-program program *ns* {:active-input 'mu})
          owner (canonical/adapt-map-sum bundle)
          prepared (runtime/prepare bundle)
          ys (array (take size [0.5 -0.25 1.0])) mu (cast 0.25) sigma (cast 1.7)
          args (mapv {'observations ys 'mu mu 'sigma sigma 'size (long size)} (:inputs prepared))
          slot (.indexOf ^java.util.List (:inputs prepared) 'mu)
          [value pb] ((:vjp prepared) args)
          [_ fresh-pb] ((:vjp prepared) args)
          gradients (pb (cast 1.0)) gradient (nth gradients slot)
          [jvalue tangent] ((:jvp prepared) args (cast 0.125))
          analytic (reduce + 0.0 (map #(/ (- (double %) (double mu))
                                         (* (double sigma) (double sigma))) (seq ys)))
          oracle (reduce + 0.0 (map #(dist/logpdf (dist/->Normal (double mu) (double sigma))
                                                (double %)) (seq ys)))
          fd (/ (- (v ys (cast (+ mu h)) sigma size) (v ys (cast (- mu h)) sigma size)) (* 2.0 h))
          original-reduction (soac/operation-parts (second (soac/equations program)))
          gradient-reduction (soac/operation-parts (second (soac/equations (:transpose owner))))]
      (is (= (v ys mu sigma size) value jvalue))
      (is (< (Math/abs (- oracle value)) arithmetic-tolerance))
      (is (< (Math/abs (- analytic gradient)) arithmetic-tolerance))
      (is (< (Math/abs (- fd gradient)) fd-tolerance))
      (is (< (Math/abs (- tangent (* 0.125 gradient))) arithmetic-tolerance))
      (is (= (class mu) (class value) (class gradient) (class tangent)))
      (is (= [] (get-in owner [:interface :tangent-inputs 0 :value :shape])
             (get-in owner [:interface :cotangent-results 0 :value :shape])))
      (is (= '[map reduce] (mapv (comp :kind soac/operation-parts) (soac/equations (:transpose owner)))))
      (is (fingerprint/equivalent? (dissoc original-reduction :arrays)
                                   (dissoc gradient-reduction :arrays)))
      (doseq [[id g] (map vector (:inputs prepared) gradients)]
        (is (= (= id 'mu) (some? g))))
      (doseq [seed [0.25 -2.0]]
        (is (< (Math/abs (- (* seed analytic) (nth (pb (cast seed)) slot)))
               arithmetic-tolerance)))
      (is (= (vec (take size [0.5 -0.25 1.0])) (vec ys)))
      (dotimes [i size] (aset ys i (cast 99.0)))
      (is (= gradient (nth (fresh-pb (cast 1.0)) slot)))
      (is (= gradient (nth (pb (cast 1.0)) slot)))
      (is (= :ad-map-runtime-scalar (reason #((:jvp prepared) args (array size)))))
      (let [revision (normalized/revisions)]
        (with-redefs [normalized/revisions (constantly (update revision :compiler-revision inc))]
          (is (= :ad-normalized-stale-rules (reason #(pb (cast 1.0))))))))))

(deftest unsupported-scalar-spaces-and-standalone-map-transposes-refuse
  (let [program (source #'gaussian-double :double)
        map-source (#'normalized/sum-source-map program *ns*)]
    (is (= :ad-normalized-map-scalar-requires-sum
           (reason #(normalized/linearize-map-program map-source *ns* {:active-input 'mu}))))
    (doseq [options [{:active-input 'size} {:active-input ['mu 'sigma]}
                     {:active-input 'mu :active-array 'observations}
                     {:active-array 'mu}]]
      (is (= :ad-normalized-map-active-array
             (reason #(normalized/linearize-map-sum-program program *ns* options)))))))

(deftest scalar-gradient-contracts-cannot-be-recertified-away
  (let [bundle (normalized/linearize-map-sum-program
                (source #'gaussian-double :double) *ns* {:active-input 'mu})
        transpose (get-in bundle [:sum-programs :transpose])
        gradient (get-in bundle [:sum-programs :interface :gradient-results 0])
        changed (apply list (assoc (vec transpose) 1
                                   (assoc-in (soac/facts transpose) [:values gradient :shape] '[size])))
        forged (assoc-in bundle [:sum-programs :transpose] changed)
        forged (assoc forged :certificate (#'normalized/certificate forged))]
    (is (some? (reason #(normalized/validate! forged))))
    (is (some? (reason #(runtime/prepare forged))))))

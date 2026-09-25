(ns raster.compiler.passes.parallel.typed-soac-resident-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]))

(defn- fused [source]
  (first (fusion/fusion-fixpoint
          (frontend/form->program source
                                  {:dtype :double :array-types {'x :double 'out :double}
                                   :scalar-types {'scale :double}}))))

(deftest resident-uniform-load-is-a-stable-device-capture
  (let [source '(let* [^double alpha (clojure.core/aget params 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset out i (+ alpha (clojure.core/aget out i))))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'params :double 'out :double}
                                         :scalar-types {'n :long}})
        [result stats] (resident/inline-uniform-input-loads program)
        infos (mapv fusion/equation-info (dialect/equations result))
        effect (first (filter #(contains? #{:map :effect-map} (:kind %)) infos))]
    (is (= 1 (:resident-uniform-input-loads stats)))
    (is (not-any? #(some #{'alpha} (:results %)) infos))
    (is (some #{'params} (:captures effect)))
    (is (some #{'params} (get-in effect [:attributes :attributes :stable-array-captures])))
    (is (not (some #{'alpha} (:captures effect))))
    (is (= result (dialect/validate! result)))))

(deftest resident-uniform-load-never-moves-a-launch-extent
  (let [source '(let* [^long n (clojure.core/aget dims 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset out i (double i)))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'dims :long 'out :double}})
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-loads stats)))
    (is (some (fn [equation] (some #{'n} (nth equation 2))) (dialect/equations result)))
    (is (= result (dialect/validate! result)))))

(deftest resident-uniform-load-never-crosses-a-write-to-its-source
  (let [source '(let* [^double alpha (clojure.core/aget params 0)
                       result (raster.par/map-void! i n
                                (clojure.core/aset params i (+ alpha (double i))))]
                  result)
        program (frontend/form->program source
                                        {:dtype :double
                                         :array-types {'params :double}
                                         :scalar-types {'n :long}})
        [result stats] (resident/inline-uniform-input-loads program)]
    (is (zero? (:resident-uniform-input-loads stats)))
    (is (some (fn [equation] (some #{'alpha} (nth equation 2))) (dialect/equations result)))
    (is (= result (dialect/validate! result)))))

(deftest result-transform-integral-casts-state-their-rounding
  (doseq [source [:int :long] target [:float :double]]
    (let [region (body/->ScalarRegion ['acc 'width] (list (symbol (name target)) 'width)
                                      [] target)
          result (scalar-region/lower
                  region {:accumulator 'acc :accumulator-dtype target :store-dtype target
                          :parameters {'width {:id 'width :kind :scalar :role :parameter
                                               :dtype source}}})
          cast (:expression (first (:operations result)))]
      (is (= :cast (:op cast)))
      (is (= {:rounding :nearest-even :overflow :exact} (:options cast))))))

(deftest resident-reduction-preserves-host-scalar-captures
  (let [program (fused
                 '(let* [^long width (alength x)
                         total (raster.par/reduce acc 0.0 i width (+ acc (aget x i)))
                         result (raster.par/map-void! j width
                                  (aset out j (* (aget x j) total)))]
                    result))
        scalar-info (fn [p]
                      (first (filter #(= :scalar (:kind %))
                                     (map fusion/equation-info (dialect/equations p)))))
        before (scalar-info program)
        [result stats] (resident/realize program)
        after (scalar-info result)]
    (is (some? before))
    (is (= 1 (:resident-reductions stats)))
    (is (= (:captures before) (:captures after)))
    (is (= (:parameters before) (:parameters after)))
    (is (= (:body-results before) (:body-results after)))
    (is (= result (dialect/validate! result)))))

(deftest resident-reduction-rewrites-an-effect-map-without-touching-host-scalar-equations
  (let [program (fused
                 '(let* [^long width (alength x)
                         total (raster.par/reduce acc 0.0 i width (+ acc (aget x i)))
                         result (raster.par/map-void! j width
                                  (if (> total 0.0)
                                    (let* [^double scaled (* total 2.0)]
                                      (aset x j (* (aget x j) scaled)))
                                    nil))]
                    result))
        scalar-before (first (filter #(= :scalar (:kind %))
                                     (map fusion/equation-info
                                          (dialect/equations program))))
        [result stats] (resident/realize program)
        infos (mapv fusion/equation-info (dialect/equations result))
        scalar-after (first (filter #(= :scalar (:kind %)) infos))
        effect (first (filter #(= :effect-map (:kind %)) infos))]
    (is (= 1 (:resident-reductions stats)))
    (is (= scalar-before scalar-after))
    (is (= '[x] (:destinations effect)))
    (is (= (inc (count (:captures effect))) (count (:parameters effect))))
    (is (resident/resident-scalar-value? (get-in (dialect/facts result)
                                                  [:values 'total])))
    (is (= result (dialect/validate! result)))))

(deftest resident-reduction-preserves-the-independent-transform-boundary
  (let [program (fused
                 '(let* [total (raster.par/reduce acc 0.0 i n (+ acc (aget x i)))
                         ^double scaled (* ^double total ^double scale)
                         result (raster.par/map-void! j n
                                  (aset out j (* (aget x j) scaled)))]
                    result))
        transform-before (get-in (fusion/equation-info (first (dialect/equations program)))
                                 [:attributes :result-transform])
        [result stats] (resident/realize program)
        reduction (fusion/equation-info (first (dialect/equations result)))]
    (is (some? transform-before))
    (is (= '[scale] (mapv :value (:scalars transform-before))))
    (is (= 1 (:resident-reductions stats)))
    (is (some #{'scale} (:captures reduction)))
    (is (= transform-before (get-in reduction [:attributes :result-transform])))
    (is (resident/resident-scalar-value? (get-in (dialect/facts result) [:values 'scaled])))
    (is (= result (dialect/validate! result)))))

(deftest transform-scalars-do-not-silently-become-resident-pointers
  (let [program (fused
                 '(let* [left (raster.par/reduce a 0.0 i n (+ a (aget x i)))
                         right (raster.par/reduce b 0.0 j n (+ b (* (aget x j) 2.0)))
                         ^double combined (* ^double right ^double left)
                         result (raster.par/map-void! k n
                                  (aset out k (* (aget x k) combined)))]
                    result))
        transform-inputs (into #{} (mapcat #(map :value (get-in (fusion/equation-info %)
                                                                [:attributes :result-transform :scalars])))
                               (dialect/equations program))
        [result _] (resident/realize program)]
    (is (seq transform-inputs))
    (doseq [value transform-inputs]
      (is (not (resident/resident-scalar-value? (get-in (dialect/facts result) [:values value])))))
    (is (= result (dialect/validate! result)))))

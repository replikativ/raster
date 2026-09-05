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

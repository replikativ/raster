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

(deftest resident-realization-preserves-unrelated-fold-map-regions
  (let [program (frontend/form->program
                 '(let* [total (raster.par/reduce acc 0.0 i 4 (+ acc (aget x i)))
                         folded (raster.par/segmented-fold-map!
                                 [out] [[row 1]] col 4
                                 [[sum 0.0 :double 4 (+ sum (aget x col))]]
                                 [(+ sum (aget x col))])
                         result (raster.par/map-void! j 4 (aset out j (* (aget x j) total)))]
                    result)
                 {:dtype :double :array-types {'x :double 'out :double}})
        fold-equation (fn [p]
                        (first (filter #(= :segmented-fold-map
                                           (:kind (fusion/equation-info %)))
                                       (dialect/equations p))))
        [realized stats] (resident/realize program)]
    (is (= 1 (:resident-reductions stats)))
    (is (some? (fold-equation program)))
    (is (= (fold-equation program) (fold-equation realized))
        "unrelated fold regions and their map lambda survive without reconstruction")
    (is (= realized (dialect/validate! realized)))))

(deftest resident-realization-never-replaces-a-multi-region-extent-with-a-buffer
  (doseq [role [:segment :fold]]
    (let [segment-extent (if (= :segment role) 'total 1)
          fold-extent (if (= :fold role) 'total 4)
          source `(let* [~'total (raster.par/reduce ~'acc (long 0) ~'i 4
                                  (clojure.core/unchecked-add ~'acc (clojure.core/aget ~'x ~'i)))
                         ~'result (raster.par/segmented-fold-map!
                                   [~'out] [[~'row ~segment-extent]] ~'col 4
                                   [[~'sum 0 :long ~fold-extent
                                     (clojure.core/unchecked-add ~'sum
                                                                (clojure.core/aget ~'x ~'col))]]
                                   [~'sum])]
                    ~'result)
          program (frontend/form->program source {:dtype :long
                                                  :array-types {'x :long 'out :long}})
          [realized stats] (resident/realize program)]
      (is (= 0 (:resident-reductions stats)))
      (is (= program realized) "every segment/fold extent keeps its scalar representation"))))

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
      (is (= {:rounding (if (and (= :int source) (= :double target))
                         :exact :nearest-even)
              :overflow :exact}
             (:options cast))
          "all Int32 values are exactly representable in Double; other pairs require rounding"))))

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

(defn- two-reduction-transform-program []
  (fused
   '(let* [left (raster.par/reduce a 0.0 i n (+ a (aget x i)))
           right (raster.par/reduce b 0.0 j n (+ b (* (aget x j) 2.0)))
           ^double combined (* ^double right ^double left)
           result (raster.par/map-void! k n
                    (aset out k (* (aget x k) combined)))]
      result)))

(defn- transform-scalar-values [program]
  (into #{} (mapcat #(map :value (get-in (fusion/equation-info %)
                                         [:attributes :result-transform :scalars])))
        (dialect/equations program)))

(deftest audited-transform-scalars-become-explicit-resident-singleton-loads
  (let [program (two-reduction-transform-program)
        transform-inputs (into #{} (mapcat #(map :value (get-in (fusion/equation-info %)
                                                                [:attributes :result-transform :scalars])))
                               (dialect/equations program))
        [result stats] (resident/realize program)
        result-infos (mapv fusion/equation-info (dialect/equations result))
        transform-operands
        (into {} (mapcat #(map (juxt :value identity)
                               (get-in % [:attributes :result-transform :operands])))
              result-infos)
        transform-scalars
        (into #{} (mapcat #(map :value (get-in % [:attributes :result-transform :scalars])))
              result-infos)]
    (is (seq transform-inputs))
    (doseq [value transform-inputs]
      (is (resident/resident-scalar-value? (get-in (dialect/facts result) [:values value])))
      (is (= {:groups [[]]} (:map (get transform-operands value))))
      (is (not (contains? transform-scalars value)))
      (let [transform (some #(when (some #{value}
                                         (map :value (get-in % [:attributes :result-transform
                                                               :operands])))
                               (get-in % [:attributes :result-transform]))
                            result-infos)
            operand (some #(when (= value (:value %)) %)
                          (:operands transform))
            expression (first (:body-results (dialect/lambda-parts (:lambda transform))))]
        (is (some #{(list 'clojure.core/aget (:parameter operand) 0)}
                  (tree-seq coll? seq expression))))
      (is (some (fn [info]
                  (some #{value}
                        (get-in info [:attributes :attributes :stable-array-captures])))
                result-infos)))
    (is (pos? (:resident-reductions stats)))
    (is (= result (dialect/validate! result)))))

(deftest transform-load-realization-is-blocked-by-public-or-host-escape
  (let [program (two-reduction-transform-program)
        value (first (transform-scalar-values program))
        facts (dialect/facts program)
        equations (dialect/equations program)
        outputs (dialect/outputs program)
        public (dialect/make facts equations (vec (distinct (conj outputs value))))
        host (dialect/make (assoc-in facts [:attributes :host-read-values] [value])
                           equations outputs)]
    (doseq [[label candidate] [[:public public] [:host host]]]
      (let [[result _] (resident/realize candidate)
            transform-values (transform-scalar-values result)]
        (is (contains? transform-values value) (name label))
        (is (not (resident/resident-scalar-value?
                  (get-in (dialect/facts result) [:values value])))
            (name label))
        (is (= result (dialect/validate! result)))))))

(deftest transform-load-realization-preserves-incompatible-value-facets
  (let [program (two-reduction-transform-program)
        value (first (transform-scalar-values program))]
    (doseq [[facet facet-value]
            [[:memory-space :host]
             [:ownership :owned]
             [:attributes {:external-contract true}]]]
      (let [facts (assoc-in (dialect/facts program) [:values value facet] facet-value)
            candidate (dialect/make facts (dialect/equations program)
                                    (dialect/outputs program))
            [result _] (resident/realize candidate)]
        (is (contains? (transform-scalar-values result) value) (name facet))
        (is (= facet-value (get-in (dialect/facts result) [:values value facet]))
            (name facet))
        (is (not (resident/resident-scalar-value?
                  (get-in (dialect/facts result) [:values value])))
            (name facet))
        (is (= result (dialect/validate! result)))))))

(deftest transform-load-realization-rejects-a-mixed-extent-use
  (let [program
        (first
         (fusion/fusion-fixpoint
          (frontend/form->program
           '(let* [left (raster.par/reduce a (long 0) i n
                          (clojure.core/unchecked-add a (aget x i)))
                   right (raster.par/reduce b (long 0) j n
                           (clojure.core/unchecked-add b (aget x j)))
                   ^long combined (clojure.core/unchecked-add right left)
                   result (raster.par/map-void! k left
                            (aset out k (clojure.core/unchecked-add (aget x k) combined)))]
              result)
           {:dtype :long :array-types {'x :long 'out :long}
            :scalar-types {'n :long}})))
        value (first (transform-scalar-values program))
        [result _] (resident/realize program)]
    (is (some? value))
    (is (contains? (transform-scalar-values result) value))
    (is (not (resident/resident-scalar-value?
              (get-in (dialect/facts result) [:values value]))))
    (is (= result (dialect/validate! result)))))

(ns raster.compiler.passes.parallel.typed-butterfly-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par]))

(def source
  '(let* [effect (raster.par/butterfly! re im i half wr wi base)] effect))

(def options
  {:scalar-types {'half :long 'base :long}})

(def array-types
  {'re :float 'im :float 'wr :float 'wi :float})

(defn- attempt []
  (route/attempt source :float array-types options))

(deftest butterfly-is-an-ordinary-certified-mixed-radix-effect-map
  (let [result (attempt)
        equation (-> result :program :equations first :algorithm dialect/equations first)
        {:keys [kind attributes lambda]} (dialect/operation-parts equation)
        effects (:body-results (dialect/lambda-parts lambda))]
    (is (= :typed-soac (get-in result [:stats :route])))
    (is (= 'effect-map kind))
    (is (= :independent (:iteration-order attributes)))
    (is (= 1 (get-in result [:stats :effect-row-ownership-proofs])))
    (is (= 4 (count effects)))
    (is (= #{:unique}
           (set (map (comp :conflict dialect/effect-parts)
                     effects))))))

(deftest typed-butterfly-preserves-in-place-semantics-at-a-nonzero-base
  (let [program (:program (attempt))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:target-device :ze:0 :dtype :float
                             :array-types array-types
                             :scalar-types (:scalar-types options)}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[re im wr wi half base] (:form jvm)))
        re (float-array [-91 1 2 3 4 10 20 30 40 -92])
        im (float-array [-81 5 6 7 8 50 60 70 80 -82])
        wr (float-array [1 1 1 1])
        wi (float-array [0 0 0 0])]
    (is (nil? (execute re im wr wi 4 1)))
    (is (= [-91.0 11.0 22.0 33.0 44.0 -9.0 -18.0 -27.0 -36.0 -92.0]
           (vec re)))
    (is (= [-81.0 55.0 66.0 77.0 88.0 -45.0 -54.0 -63.0 -72.0 -82.0]
           (vec im)))))

(deftest butterfly-declines-unknown-or-mismatched-element-types
  (testing "all four arrays share one explicit scalar algebra"
    (let [result (route/attempt source :float
                                (assoc array-types 'wi :double) options)]
      (is (= :typed-soac-source-coverage (get-in result [:declined :reason])))))
  (testing "missing retained array types are not inferred from the operator name"
    (let [result (route/attempt source nil {} options)]
      (is (= :typed-soac-source-coverage (get-in result [:declined :reason]))))))

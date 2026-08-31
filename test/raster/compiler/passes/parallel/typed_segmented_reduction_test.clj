(ns raster.compiler.passes.parallel.typed-segmented-reduction-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-pass]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]))

(defn- tensor [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- program []
  (let [equation
        '(= contract-equation [C]
            (segmented-reduce
             {:segment-axes [[i m] [j n]]
              :index l :extent k
              :accumulators [acc] :identities [0.0]
              :dtypes [:float] :algebra [{}]
              :attributes {:stable-array-captures [A B]}}
             [] [A B n k]
             (lambda [acc lhs rhs width inner]
               (region []
                       [(+ acc
                           (* (aget lhs (+ (* i inner) l))
                              (aget rhs (+ (* l width) j))))]))))
        facts
        (assoc (dialect/default-program-facts
                {:front-end :typed-segmented-reduction-test})
               :values {'A (tensor :float '[m k])
                        'B (tensor :float '[k n])
                        'm (tensor :long [])
                        'n (tensor :long [])
                        'k (tensor :long [])
                        'C (tensor :float '[m n])}
               :inputs '[A B m n k]
               :equations {'contract-equation (dialect/default-equation-facts
                                                {:source :test})})]
    (dialect/make facts [equation] '[C])))

(deftest segmented-reduction-is-a-typed-functional-equation
  (let [program (program)
        equation (first (dialect/equations program))
        attributes (:attributes (dialect/operation-parts equation))]
    (is (= 'segmented-reduce (dialect/operation-kind equation)))
    (is (= '[[i m] [j n]] (:segment-axes attributes)))
    (is (= '[m n k] (dialect/operation-extents equation)))
    (is (= '[A B n k] (dialect/operation-inputs equation)))
    (is (= '[C] (dialect/outputs program)))))

(deftest generic-fusion-keeps-an-unmatched-segmented-reduction-intact
  (let [source (program)
        [result stats] (fusion/fusion-fixpoint source)]
    (is (= source result))
    (is (= :segmented-reduce
           (:kind (fusion/equation-info (first (dialect/equations result))))))
    (is (= 0 (:vertical stats)))
    (is (= 0 (:horizontal stats)))))

(deftest segmented-reduction-lowers-directly-to-canonical-segred
  (with-redefs [legacy/->SoacContract
                (fn [& _]
                  (throw (ex-info "legacy SoacContract was constructed" {})))]
    (let [operation (first (lower/lower-typed-segmented-reduce
                            (program) :ze:0 :dtype :float))
          dims (:dims (:space operation))
          product (:reduction operation)
          step (first (:results (reduction/fold-region product)))]
      (is (instance? raster.compiler.ir.segop.SegRed operation))
      (is (= '[[i m] [j n] [l k]]
             (mapv (juxt :name :bound) dims)))
      (is (= '#{A B} (:inputs operation)))
      (is (= '#{m n k} (:scalars operation)))
      (is (= '#{C} (:outputs operation)))
      (is (= :segmented (:phase operation)))
      (is (nil? (:schedule operation)))
      (is (reduction/product-reduction? product))
      (is (= :float (first (reduction/dtypes product))))
      (is (= 'C (first (reduction/results product))))
      (is (re-find #"A" (pr-str step)))
      (is (re-find #"B" (pr-str step)))
      (is (not (re-find #"lhs|rhs|width|inner" (pr-str step)))))))

(deftest value-remapping-includes-every-segment-extent
  (let [remapped (dialect/remap-values
                  (program) {'m [:binding 'm] 'n [:binding 'n] 'C [:binding 'C]})
        equation (first (dialect/equations remapped))]
    (is (= [['i [:binding 'm]] ['j [:binding 'n]]]
           (get-in (dialect/operation-parts equation) [:attributes :segment-axes])))
    (is (= [[:binding 'm] [:binding 'n] 'k]
           (dialect/operation-extents equation)))
    (is (= [[:binding 'C]] (dialect/outputs remapped)))))

(deftest typed-parallel-program-schedules-the-equation-without-compatibility-relowering
  (let [algorithm (program)
        values (:values (dialect/facts algorithm))
        source '(raster.par/contract C [[i m] [j n]] [[l k]] body)
        equation
        (parallel-program/->ProgramEquation
         'contract-equation [:binding 'C] source '[A B m n k] '[C]
         algorithm [(first (dialect/equations algorithm))] #{}
         {:source-dialect :typed-soac} {})
        envelope
        (parallel-program/make
         {:dialect :typed-soac :source (list 'let* ['C source] 'C)
          :values values :inputs '[A B m n k] :equations [equation] :outputs '[C]
          :effects #{} :operation? (constantly true)
          :algorithm? (fn [_ candidate] (= candidate (dialect/validate! candidate)))})
        lowered (segop-pass/segop-lower-pass
                 envelope {:target-device :ze:0 :dtype :float})
        scheduled-equation (first (:equations (:form lowered)))
        operation (first (:operations scheduled-equation))]
    (is (dialects/valid-typed-soac-program? envelope)
        "the pipeline boundary admits the TypedSOAC segmented-reduce operation")
    (is (= :segop (:dialect (:form lowered))))
    (is (= 1 (get-in lowered [:stats :typed-soac-reused])))
    (is (= 1 (get-in lowered [:stats :segops-lowered])))
    (is (instance? raster.compiler.ir.segop.SegRed operation))
    (is (= :typed-soac (get-in scheduled-equation [:provenance :source-dialect])))
    (is (empty? (:diagnostics (:form lowered))))))

(ns raster.ad.normalized-surface-map-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as ra]
            [raster.par :as par]
            [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-map-execution-test :as execution]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]))

(deftm square-double [xs :- (Array double) size :- Long] :- (Array double)
  (par/map [i size] (n/* (ra/aget xs i) (ra/aget xs i))))
(deftm square-float [xs :- (Array float) size :- Long] :- (Array float)
  (par/map [i size] (n/* (ra/aget xs i) (ra/aget xs i))))

(defn- source-program [v dt]
  (let [body (first (core/ensure-walked-body! v))
        result (gensym "surface_result_")
        options {:dtype dt :array-types {'xs dt} :scalar-types {'size :long}}]
    (frontend/form->program
     (frontend/normalize-source (list 'let* [result body] result) options) options)))

(defn- build [program]
  (normalized/linearize-map-program program (the-ns 'raster.ad.normalized-surface-map-test)))

(deftest real-deftm-map-reaches-normalized-ad-and-shared-host-execution
  (doseq [[v dt array] [[#'square-double :double double-array]
                        [#'square-float :float float-array]]
          size [0 3]]
    (let [source (source-program v dt)
          bundle (build source)
          interface (get-in bundle [:map-programs :interface])
          primal (#'execution/executable (get-in bundle [:map-programs :primal]))
          transpose (#'execution/executable (get-in bundle [:map-programs :transpose]))
          xs (array (take size [-2.0 0.0 3.0]))
          result (#'execution/invoke primal {'size size 'xs xs})
          saved (zipmap (:outputs primal) result)
          expected (vec (v xs size))]
      (is (= expected (vec (first result))))
      (is (identical? source (get-in bundle [:map-programs :source])))
      (is (identical? bundle (normalized/validate! bundle)))
      (dotimes [i size] (aset xs i (if (= dt :float) (float 99.0) 99.0)))
      (is (= (vec (take size [-4.0 0.0 6.0]))
             (vec (first (#'execution/invoke
                          transpose (assoc saved 'size size
                                           (:cotangent-input interface) (array (repeat size 1.0)))))))))))

(defn- reason [program]
  (try (build program) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest frontend-bookkeeping-must-prove-a-single-closed-source-map
  (let [source (source-program #'square-double :double)
        facts (soac/facts source)]
    (doseq [[path value] [[[:attributes :source-bindings] '[other]]
                          [[:attributes :host-binding-ids] [1]]
                          [[:attributes :host-read-values] '[xs]]
                          [[:attributes :host-read-sites] [{:values #{'xs}}]]
                          [[:attributes :allocations] [{:id 'scratch}]]
                          [[:attributes :unknown-policy] true]
                          [[:equations 0 :provenance :source-binding-id] 1]
                          [[:equations 0 :attributes :source-prefix?] false]]]
      (is (= :ad-normalized-map-boundary
             (reason (soac/make (assoc-in facts path value)
                                (soac/equations source) (soac/outputs source))))))))

(deftest identity-conversion-projection-retains-selected-call-metadata
  (let [source (source-program #'square-double :double)
        [equation] (soac/equations source)
        operation (nth equation 3)
        {:keys [parameters body-results]} (soac/lambda-parts (nth operation 4))
        conversion (first body-results)
        operand (:operand (soac/scalar-convert-parts conversion))
        operand (vary-meta operand assoc :raster.type/tag 'double :line 123)
        conversion (apply list (assoc (vec conversion) 2 operand))
        operation (apply list (assoc (vec operation) 4 (soac/lambda-form parameters [conversion])))
        source (soac/make (soac/facts source)
                          [(list '= (second equation) (nth equation 2) operation)]
                          (soac/outputs source))
        bundle (build source)
        calls (for [eq (soac/equations (:primal bundle))
                    term (tree-seq coll? seq (:lambda (soac/operation-parts eq)))
                    :when (and (seq? term) (= '.invk (first term)))] term)]
    (is (= 1 (count calls)))
    (is (= (meta operand) (meta (first calls))))))

(deftest conversion-policies-and-active-precision-changes-still-refuse
  (let [source (source-program #'square-double :double)
        [equation] (soac/equations source)
        operation (nth equation 3)
        {:keys [parameters body-results]} (soac/lambda-parts (nth operation 4))
        conversion (first body-results)]
    (doseq [[field value expected] [[:rounding :nearest-even :typed-soac-syntax]
                                    [:source-dtype :float :ad-normalized-map-space]]]
      (let [changed (apply list (update (vec conversion) 1 assoc field value))
            operation (apply list (assoc (vec operation) 4
                                         (soac/lambda-form parameters [changed])))
            changed-equation (list '= (second equation) (nth equation 2) operation)
            ;; Keep malformed producer data unvalidated until the actual importer boundary.
            changed-source (apply list (assoc (vec source) 2 [changed-equation]))]
        (is (= expected (reason changed-source)))))))

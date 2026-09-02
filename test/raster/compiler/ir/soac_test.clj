(ns raster.compiler.ir.soac-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac :as soac]
            [raster.par :as par]))

;; ================================================================
;; par-form->soac conversion
;; ================================================================

(deftest par-map->soac-test
  (testing "Convert raster.par/map! to SoacMap"
    (let [expr '(raster.par/map! out i n double (* (aget a i) (aget b i)))
          node (soac/par-form->soac 'out expr 0)]
      (is (instance? raster.compiler.ir.soac.SoacMap node))
      (is (= 0 (:id node)))
      (is (= 'out (:sym node)))
      (is (= 'i (:idx node)))
      (is (= 'n (:bound node)))
      (is (= 'double (:cast-fn node)))
      (is (= '(* (aget a i) (aget b i)) (:lambda node)))
      (is (= #{'a 'b} (:inputs node)))
      (is (contains? (:outputs node) 'out))
      ;; n is the bound expression, not in scalars (scalars = free syms of lambda minus inputs/idx)
      (is (not (contains? (:scalars node) 'n))))))

(deftest offset-map-cannot-lose-its-destination-base
  (testing "the compatibility SoacMap has no indexed-store field, so accepting an offset map would
            silently turn out[base+i] into out[i]"
    (try
      (soac/par-form->soac
       'out '(raster.par/map! out i n :offset base float (aget x i)) 0)
      (is false "an offset map must not enter the compatibility SOAC dialect")
      (catch clojure.lang.ExceptionInfo e
        (is (= :offset-map-requires-indexed-store (:reason (ex-data e))))
        (is (= 'base (:offset (ex-data e))))
        (is (= :typed-soac-scatter (:target-dialect (ex-data e))))))))

(deftest par-reduce->soac-test
  (testing "Convert raster.par/reduce to SoacReduce"
    (let [expr '(raster.par/reduce acc 0.0 j n (+ acc (aget a j)))
          node (soac/par-form->soac 'result expr 1)]
      (is (instance? raster.compiler.ir.soac.SoacReduce node))
      (is (= 1 (:id node)))
      (is (= 'result (:sym node)))
      (is (= ['acc] (reduction/accumulators (:reduction node))))
      (is (= [0.0] (reduction/neutrals (:reduction node))))
      (is (= 'j (:index (:reduction node))))
      (is (= 'n (:bound node)))
      (is (= ['(+ acc (aget a j))] (:results (reduction/fold-region (:reduction node)))))
      (is (= #{'a} (:inputs node)))
      (is (= ['result] (:outputs node))))))

(deftest par-scan->soac-test
  (testing "Convert raster.par/scan to SoacScan"
    (let [expr '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i)))
          node (soac/par-form->soac 'out expr 2)]
      (is (instance? raster.compiler.ir.soac.SoacScan node))
      (is (= 'out (:out node)))
      (is (= 'acc (:acc node)))
      (is (= 0.0 (:init node)))
      (is (= 'i (:idx node)))
      (is (= 'n (:bound node)))
      (is (= 'double (:cast-fn node))))))

(deftest par-stencil-has-no-legacy-soac-record
  (testing "Stencil semantics live only in the validated TypedSOAC dialect"
    (let [expr '(raster.par/stencil! out [a] 1 :dirichlet double i n
                                     (+ (aget a (- i 1)) (aget a i) (aget a (+ i 1))))
          node (soac/par-form->soac 'out expr 3)]
      (is (nil? node)))))

(deftest non-par->nil-test
  (testing "Non-par expression returns nil"
    (is (nil? (soac/par-form->soac 'x '(+ 1 2) 0)))))

;; ================================================================
;; Written arrays are OUTPUTS, never scalars. Re-parsing a compatibility
;; map with side-effect stores must classify secondary destinations as
;; array outputs or the GPU backend declares them as scalar
;; kernel params (`float hfuse_out__N`) and extraction emits an
;; unresolvable host reference.
;; ================================================================

(deftest aset-written-array-classified-as-output-test
  (testing "bare aset target in a par/map! lambda → :outputs, not :scalars"
    (let [expr '(raster.par/map! hout1 i n float
                                 (do (raster.arrays/aset hout2 i (float (* (aget d i) (aget a i))))
                                     (* (aget d i) (aget b i))))
          node (soac/par-form->soac 'da expr 0)]
      (is (contains? (:outputs node) 'hout2)
          "aset-written array must be a SOAC output")
      (is (not (contains? (:scalars node) 'hout2))
          "aset-written array must never be a scalar param")
      (is (contains? (:outputs node) 'hout1))
      (is (= #{'d 'a 'b} (:inputs node)))))

  (testing "devirtualized (.invk aset-impl …) target → :outputs, not :scalars"
    (let [aset-invk (with-meta
                      (list '.invk 'raster.arrays/aset_m_floats_long_float-impl
                            'hout2 'i (list 'float '(* (aget d i) (aget a i))))
                      {:raster.op/original 'raster.arrays/aset})
          expr (list 'raster.par/map! 'hout1 'i 'n 'float
                     (list 'do aset-invk '(* (aget d i) (aget b i))))
          node (soac/par-form->soac 'da expr 0)]
      (is (contains? (:outputs node) 'hout2))
      (is (not (contains? (:scalars node) 'hout2))))))

(deftest soac-inputs-test
  (testing "soac-inputs extracts array symbols"
    (let [node (soac/par-form->soac 'out
                                    '(raster.par/map! out i n double (+ (aget a i) (aget b i))) 0)]
      (is (= #{'a 'b} (soac/soac-inputs node))))))

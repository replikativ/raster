(ns raster.compiler.ir.soac-test
  (:require [clojure.test :refer [deftest is testing]]
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

(deftest par-reduce->soac-test
  (testing "Convert raster.par/reduce to SoacReduce"
    (let [expr '(raster.par/reduce acc 0.0 j n (+ acc (aget a j)))
          node (soac/par-form->soac 'result expr 1)]
      (is (instance? raster.compiler.ir.soac.SoacReduce node))
      (is (= 1 (:id node)))
      (is (= 'result (:sym node)))
      (is (= 'acc (:acc node)))
      (is (= 0.0 (:init node)))
      (is (= 'j (:idx node)))
      (is (= 'n (:bound node)))
      (is (= '(+ acc (aget a j)) (:lambda node)))
      (is (= #{'a} (:inputs node)))
      (is (= 'result (:output node))))))

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

(deftest par-stencil->soac-test
  (testing "Convert raster.par/stencil! to SoacStencil"
    (let [expr '(raster.par/stencil! out [a] 1 :dirichlet double i n
                                     (+ (aget a (- i 1)) (aget a i) (aget a (+ i 1))))
          node (soac/par-form->soac 'out expr 3)]
      (is (instance? raster.compiler.ir.soac.SoacStencil node))
      (is (= '[a] (:in-arrays node)))
      (is (= 1 (:radius node)))
      (is (= :dirichlet (:boundary node))))))

(deftest non-par->nil-test
  (testing "Non-par expression returns nil"
    (is (nil? (soac/par-form->soac 'x '(+ 1 2) 0)))))

;; ================================================================
;; soac->par-form round-trip
;; ================================================================

(deftest round-trip-map-test
  (testing "SoacMap round-trip: par→soac→par"
    (let [original '(raster.par/map! out i n double (* (aget a i) 2.0))
          node (soac/par-form->soac 'out original 0)
          reconstructed (soac/soac->par-form node)]
      (is (= (first original) (first reconstructed)))
      (is (= (second original) (second reconstructed)))
      (is (= (nth original 2) (nth reconstructed 2)))
      (is (= (nth original 3) (nth reconstructed 3)))
      (is (= (nth original 4) (nth reconstructed 4)))
      (is (= (nth original 5) (nth reconstructed 5))))))

(deftest round-trip-reduce-test
  (testing "SoacReduce round-trip: par→soac→par"
    (let [original '(raster.par/reduce acc 0.0 j n (+ acc (aget a j)))
          node (soac/par-form->soac 'result original 0)
          reconstructed (soac/soac->par-form node)]
      (is (= 'raster.par/reduce (first reconstructed)))
      (is (= 'acc (second reconstructed)))
      (is (= 0.0 (nth reconstructed 2)))
      (is (= 'j (nth reconstructed 3)))
      (is (= 'n (nth reconstructed 4))))))

(deftest round-trip-scan-test
  (testing "SoacScan round-trip: par→soac→par"
    (let [original '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i)))
          node (soac/par-form->soac 'out original 0)
          reconstructed (soac/soac->par-form node)]
      (is (= 'raster.par/scan (first reconstructed)))
      (is (= 'out (second reconstructed))))))

;; ================================================================
;; let-bindings->nodes / nodes->let-bindings
;; ================================================================

(deftest let-bindings->nodes-test
  (testing "Correctly classifies SOAC vs scalar bindings"
    (let [pairs [['out '(raster.par/map! out i n double (* (aget a i) 2.0))]
                 ['x '42]
                 ['result '(raster.par/reduce acc 0.0 j n (+ acc (aget out j)))]]
          nodes (soac/let-bindings->nodes pairs)]
      (is (= 3 (count nodes)))
      (is (instance? raster.compiler.ir.soac.SoacMap (nth nodes 0)))
      (is (instance? raster.compiler.ir.soac.ScalarBinding (nth nodes 1)))
      (is (instance? raster.compiler.ir.soac.SoacReduce (nth nodes 2))))))

(deftest nodes->let-bindings-roundtrip-test
  (testing "Round-trip through nodes preserves binding order"
    (let [pairs [['out '(raster.par/map! out i n double (* (aget a i) 2.0))]
                 ['x '42]
                 ['result '(raster.par/reduce acc 0.0 j n (+ acc (aget out j)))]]
          nodes (soac/let-bindings->nodes pairs)
          reconstructed (soac/nodes->let-bindings nodes)]
      (is (= 3 (count reconstructed)))
      ;; Scalar binding preserved exactly
      (is (= ['x '42] (nth reconstructed 1)))
      ;; SOAC types preserved
      (is (= 'raster.par/map! (first (second (nth reconstructed 0)))))
      (is (= 'raster.par/reduce (first (second (nth reconstructed 2))))))))

;; ================================================================
;; Predicates and accessors
;; ================================================================

(deftest soac?-test
  (testing "soac? returns true for SOACs, false for ScalarBinding"
    (let [map-node (soac/par-form->soac 'out
                                        '(raster.par/map! out i n double (aget a i)) 0)
          scalar (soac/->ScalarBinding 1 'x '42)]
      (is (true? (soac/soac? map-node)))
      (is (false? (soac/soac? scalar))))))

(deftest soac-inputs-test
  (testing "soac-inputs extracts array symbols"
    (let [node (soac/par-form->soac 'out
                                    '(raster.par/map! out i n double (+ (aget a i) (aget b i))) 0)]
      (is (= #{'a 'b} (soac/soac-inputs node))))))

(deftest soac-bound-test
  (testing "soac-bound extracts iteration bound"
    (let [node (soac/par-form->soac 'out
                                    '(raster.par/map! out i len double (aget a i)) 0)]
      (is (= 'len (soac/soac-bound node))))))

(ns raster.dl.scatter-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.par :as par]
            [raster.compiler.ir.par :as ir.par]))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- arr-approx=
  ([a b] (arr-approx= a b 1e-5))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) (double eps)))))))

;; ================================================================
;; Scalar scatter
;; ================================================================

(deftest scatter-add-test
  (testing "scatter-add accumulates values at indices"
    (let [output (double-array 3)
          src (double-array [1.0 2.0 3.0 4.0])
          index (long-array [0 1 0 2])]
      (par/scatter! output src index 4)
      ;; output[0] = 1.0 + 3.0 = 4.0
      ;; output[1] = 2.0
      ;; output[2] = 4.0
      (is (approx= 4.0 (aget output 0)))
      (is (approx= 2.0 (aget output 1)))
      (is (approx= 4.0 (aget output 2)))))

  (testing "scatter-add with all same index"
    (let [output (double-array 1)
          src (double-array [1.0 2.0 3.0])
          index (long-array [0 0 0])]
      (par/scatter! output src index 3)
      (is (approx= 6.0 (aget output 0))))))

;; ================================================================
;; Strided scatter
;; ================================================================

(deftest scatter-strided-test
  (testing "strided scatter accumulates vectors at indices"
    ;; 3 edges mapped to 2 nodes, stride=2 (2D features)
    (let [output (double-array 4)  ;; 2 nodes * 2 features
          ;; edge 0: [1, 2], edge 1: [3, 4], edge 2: [5, 6]
          src (double-array [1 2 3 4 5 6])
          index (long-array [0 1 0])]
      (par/scatter! output src index 3 2)
      ;; node 0: [1+5, 2+6] = [6, 8]
      ;; node 1: [3, 4]
      (is (approx= 6.0 (aget output 0)))
      (is (approx= 8.0 (aget output 1)))
      (is (approx= 3.0 (aget output 2)))
      (is (approx= 4.0 (aget output 3))))))

;; ================================================================
;; Gather via par/map! (verifying indirect indexing)
;; ================================================================

(deftest gather-test
  (testing "gather via map! with indirect indexing"
    (let [src (double-array [10 20 30 40 50])
          index (long-array [4 2 0])
          out (double-array 3)]
      (par/map! out i 3 double
                (aget src (aget index i)))
      (is (approx= 50.0 (aget out 0)))
      (is (approx= 30.0 (aget out 1)))
      (is (approx= 10.0 (aget out 2))))))

;; ================================================================
;; Par form analysis
;; ================================================================

(deftest par-scatter-form-test
  (testing "par-scatter-form? detects scatter"
    (is (ir.par/par-scatter-form? '(raster.par/scatter! out src idx n)))
    (is (not (ir.par/par-scatter-form? '(raster.par/map! out i n double body)))))

  (testing "extract-par-scatter-info"
    (let [info (ir.par/extract-par-scatter-info
                '(raster.par/scatter! out src idx n))]
      (is (= 'out (:out info)))
      (is (= 'src (:src info)))
      (is (= 'idx (:index info)))
      (is (= 'n (:n info)))
      (is (nil? (:stride info))))

    (let [info (ir.par/extract-par-scatter-info
                '(raster.par/scatter! out src idx n stride))]
      (is (= 'stride (:stride info))))))

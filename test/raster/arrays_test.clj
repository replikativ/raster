(ns raster.arrays-test
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.arrays :refer [aget aset alength aclone
                                   acopy! fill! fill-zero!
                                   zeros-like alloc-like
                                   argmax argmin]]))

;; ================================================================
;; aget
;; ================================================================

(deftest aget-double-test
  (testing "aget on double[]"
    (let [arr (double-array [1.0 2.0 3.0])]
      (is (= 1.0 (aget arr 0)))
      (is (= 3.0 (aget arr 2)))
      (is (instance? Double (aget arr 0))))))

(deftest aget-long-test
  (testing "aget on long[]"
    (let [arr (long-array [10 20 30])]
      (is (= 10 (aget arr 0)))
      (is (= 30 (aget arr 2)))
      (is (instance? Long (aget arr 0))))))

(deftest aget-int-test
  (testing "aget on int[]"
    (let [arr (int-array [100 200 300])]
      (is (= 100 (aget arr 0)))
      (is (instance? Integer (aget arr 0))))))

(deftest aget-float-test
  (testing "aget on float[]"
    (let [arr (float-array [1.5 2.5 3.5])]
      (is (== 1.5 (aget arr 0)))
      (is (instance? Float (aget arr 0))))))

(deftest aget-object-test
  (testing "aget on Object[]"
    (let [arr (object-array ["hello" "world"])]
      (is (= "hello" (aget arr 0))))))

;; ================================================================
;; aset
;; ================================================================

(deftest aset-double-test
  (testing "aset on double[]"
    (let [arr (double-array [0.0 0.0])]
      (aset arr 0 42.0)
      (is (= 42.0 (aget arr 0))))))

(deftest aset-long-test
  (testing "aset on long[]"
    (let [arr (long-array [0 0])]
      (aset arr 0 42)
      (is (= 42 (aget arr 0))))))

(deftest aset-int-test
  (testing "aset on int[]"
    (let [arr (int-array [0 0])]
      (aset arr 0 (int 42))
      (is (= 42 (aget arr 0))))))

(deftest aset-float-test
  (testing "aset on float[]"
    (let [arr (float-array [0.0 0.0])]
      (aset arr 0 (float 3.14))
      (is (< (Math/abs (- 3.14 (double (aget arr 0)))) 1e-6)))))

;; ================================================================
;; alength
;; ================================================================

(deftest alength-double-test
  (testing "alength on double[]"
    (is (= 3 (alength (double-array [1.0 2.0 3.0]))))))

(deftest alength-long-test
  (testing "alength on long[]"
    (is (= 4 (alength (long-array [1 2 3 4]))))))

(deftest alength-int-test
  (testing "alength on int[]"
    (is (= 2 (alength (int-array [1 2]))))))

(deftest alength-float-test
  (testing "alength on float[]"
    (is (= 5 (alength (float-array 5))))))

(deftest alength-object-test
  (testing "alength on Object[]"
    (is (= 3 (alength (object-array ["a" "b" "c"]))))))

;; ================================================================
;; aclone
;; ================================================================

(deftest aclone-double-test
  (testing "aclone on double[] produces independent copy"
    (let [orig (double-array [1.0 2.0 3.0])
          copy (aclone orig)]
      (is (= 3 (alength copy)))
      (is (= 1.0 (aget copy 0)))
      ;; mutation of copy doesn't affect original
      (aset copy 0 99.0)
      (is (= 1.0 (aget orig 0)))
      (is (= 99.0 (aget copy 0))))))

(deftest aclone-long-test
  (testing "aclone on long[] produces independent copy"
    (let [orig (long-array [10 20 30])
          copy (aclone orig)]
      (is (= 3 (alength copy)))
      (is (= 10 (aget copy 0)))
      (aset copy 0 99)
      (is (= 10 (aget orig 0))))))

(deftest aclone-int-test
  (testing "aclone on int[]"
    (let [orig (int-array [1 2 3])
          copy (aclone orig)]
      (is (= 3 (alength copy)))
      (is (= 1 (aget copy 0))))))

(deftest aclone-float-test
  (testing "aclone on float[]"
    (let [orig (float-array [1.0 2.0])
          copy (aclone orig)]
      (is (= 2 (alength copy)))
      (is (== 1.0 (aget copy 0))))))

;; ================================================================
;; aget/aset roundtrip
;; ================================================================

(deftest aget-aset-roundtrip-test
  (testing "double[] roundtrip"
    (let [arr (double-array 3)]
      (aset arr 0 1.5)
      (aset arr 1 2.5)
      (aset arr 2 3.5)
      (is (= 1.5 (aget arr 0)))
      (is (= 2.5 (aget arr 1)))
      (is (= 3.5 (aget arr 2)))))
  (testing "float[] roundtrip"
    (let [arr (float-array 2)]
      (aset arr 0 (float 1.25))
      (aset arr 1 (float 2.75))
      (is (== 1.25 (aget arr 0)))
      (is (== 2.75 (aget arr 1)))))
  (testing "long[] roundtrip"
    (let [arr (long-array 2)]
      (aset arr 0 42)
      (aset arr 1 99)
      (is (= 42 (aget arr 0)))
      (is (= 99 (aget arr 1)))))
  (testing "int[] roundtrip"
    (let [arr (int-array 2)]
      (aset arr 0 (int 7))
      (aset arr 1 (int 13))
      (is (= 7 (aget arr 0)))
      (is (= 13 (aget arr 1))))))

(deftest aset-float-double-narrowing-test
  (testing "aset float[] with Double narrows"
    (let [arr (float-array 1)]
      (aset arr 0 3.14)
      (is (< (Math/abs (- 3.14 (double (aget arr 0)))) 1e-6)))))

;; ================================================================
;; acopy! — region copy
;; ================================================================

(deftest acopy-double-test
  (testing "acopy! full array"
    (let [src (double-array [1.0 2.0 3.0])
          dst (double-array 3)
          result (acopy! src 0 dst 0 3)]
      (is (= [1.0 2.0 3.0] (vec result)))
      (is (identical? dst result))))
  (testing "acopy! with offsets"
    (let [src (double-array [10.0 20.0 30.0 40.0 50.0])
          dst (double-array [0.0 0.0 0.0 0.0 0.0])
          result (acopy! src 1 dst 2 2)]
      ;; copy src[1..2] -> dst[2..3]
      (is (= [0.0 0.0 20.0 30.0 0.0] (vec result))))))

(deftest acopy-long-test
  (testing "acopy! long[]"
    (let [src (long-array [10 20 30])
          dst (long-array 3)
          result (acopy! src 0 dst 0 3)]
      (is (= [10 20 30] (vec result))))))

(deftest acopy-int-test
  (testing "acopy! int[]"
    (let [src (int-array [1 2 3 4])
          dst (int-array 4)
          result (acopy! src 0 dst 0 4)]
      (is (= [1 2 3 4] (vec result))))))

(deftest acopy-float-test
  (testing "acopy! float[]"
    (let [src (float-array [1.5 2.5 3.5])
          dst (float-array 3)
          result (acopy! src 0 dst 0 3)]
      (is (== 1.5 (aget result 0)))
      (is (== 2.5 (aget result 1)))
      (is (== 3.5 (aget result 2))))))

(deftest acopy-partial-test
  (testing "acopy! copies only specified region"
    (let [src (double-array [1.0 2.0 3.0 4.0 5.0])
          dst (double-array [0.0 0.0 0.0 0.0 0.0])]
      (acopy! src 2 dst 0 3)
      (is (= [3.0 4.0 5.0 0.0 0.0] (vec dst))))))

;; ================================================================
;; fill! — fill with value
;; ================================================================

(deftest fill-double-test
  (testing "fill! double[]"
    (let [arr (double-array 5)
          result (fill! arr 7.0)]
      (is (= [7.0 7.0 7.0 7.0 7.0] (vec result)))
      (is (identical? arr result)))))

(deftest fill-float-test
  (testing "fill! float[]"
    (let [arr (float-array 3)
          result (fill! arr (float 2.5))]
      (is (== 2.5 (aget result 0)))
      (is (== 2.5 (aget result 1)))
      (is (== 2.5 (aget result 2))))))

(deftest fill-long-test
  (testing "fill! long[]"
    (let [arr (long-array 4)
          result (fill! arr 42)]
      (is (= [42 42 42 42] (vec result))))))

(deftest fill-int-test
  (testing "fill! int[]"
    (let [arr (int-array 3)
          result (fill! arr (int 7))]
      (is (= [7 7 7] (vec result))))))

;; ================================================================
;; fill-zero! — zero a buffer
;; ================================================================

(deftest fill-zero-double-test
  (testing "fill-zero! double[]"
    (let [arr (double-array [1.0 2.0 3.0])
          result (fill-zero! arr)]
      (is (= [0.0 0.0 0.0] (vec result)))
      (is (identical? arr result)))))

(deftest fill-zero-float-test
  (testing "fill-zero! float[]"
    (let [arr (float-array [1.0 2.0])
          result (fill-zero! arr)]
      (is (== 0.0 (aget result 0)))
      (is (== 0.0 (aget result 1))))))

(deftest fill-zero-long-test
  (testing "fill-zero! long[]"
    (let [arr (long-array [10 20 30])
          result (fill-zero! arr)]
      (is (= [0 0 0] (vec result))))))

(deftest fill-zero-int-test
  (testing "fill-zero! int[]"
    (let [arr (int-array [1 2 3])
          result (fill-zero! arr)]
      (is (= [0 0 0] (vec result))))))

;; ================================================================
;; zeros-like — allocate zero array matching element type
;; ================================================================

(deftest zeros-like-double-test
  (testing "zeros-like double[]"
    (let [ref (double-array [1.0 2.0])
          z (zeros-like ref 5)]
      (is (= 5 (alength z)))
      (is (every? #(== 0.0 %) (seq z)))
      (is (instance? (Class/forName "[D") z)))))

(deftest zeros-like-float-test
  (testing "zeros-like float[]"
    (let [ref (float-array [1.0])
          z (zeros-like ref 3)]
      (is (= 3 (alength z)))
      (is (instance? (Class/forName "[F") z)))))

(deftest zeros-like-long-test
  (testing "zeros-like long[]"
    (let [ref (long-array [1])
          z (zeros-like ref 4)]
      (is (= 4 (alength z)))
      (is (every? #(== 0 %) (seq z)))
      (is (instance? (Class/forName "[J") z)))))

(deftest zeros-like-int-test
  (testing "zeros-like int[]"
    (let [ref (int-array [1])
          z (zeros-like ref 2)]
      (is (= 2 (alength z)))
      (is (instance? (Class/forName "[I") z)))))

;; ================================================================
;; alloc-like — allocate array matching element type (via zeros-like)
;; ================================================================

(deftest alloc-like-double-test
  (testing "alloc-like double[]"
    (let [ref (double-array [1.0 2.0 3.0])
          a (alloc-like ref 4)]
      (is (= 4 (alength a)))
      (is (instance? (Class/forName "[D") a)))))

(deftest alloc-like-float-test
  (testing "alloc-like float[]"
    (let [ref (float-array [1.0])
          a (alloc-like ref 3)]
      (is (= 3 (alength a)))
      (is (instance? (Class/forName "[F") a)))))

;; ================================================================
;; argmax — index of maximum element
;; ================================================================

(deftest argmax-double-test
  (testing "argmax double[] basic"
    (is (= 2 (argmax (double-array [1.0 2.0 5.0 3.0])))))
  (testing "argmax double[] first element is max"
    (is (= 0 (argmax (double-array [10.0 2.0 3.0])))))
  (testing "argmax double[] last element is max"
    (is (= 3 (argmax (double-array [1.0 2.0 3.0 4.0])))))
  (testing "argmax double[] single element"
    (is (= 0 (argmax (double-array [42.0])))))
  (testing "argmax double[] all same values returns first"
    (is (= 0 (argmax (double-array [3.0 3.0 3.0])))))
  (testing "argmax double[] negative values"
    (is (= 2 (argmax (double-array [-5.0 -3.0 -1.0 -2.0]))))))

(deftest argmax-float-test
  (testing "argmax float[] basic"
    (is (= 1 (argmax (float-array [1.0 5.0 3.0])))))
  (testing "argmax float[] single element"
    (is (= 0 (argmax (float-array [42.0])))))
  (testing "argmax float[] all same"
    (is (= 0 (argmax (float-array [2.0 2.0 2.0]))))))

;; ================================================================
;; argmin — index of minimum element
;; ================================================================

(deftest argmin-double-test
  (testing "argmin double[] basic"
    (is (= 0 (argmin (double-array [1.0 2.0 5.0 3.0])))))
  (testing "argmin double[] last element is min"
    (is (= 3 (argmin (double-array [10.0 5.0 3.0 1.0])))))
  (testing "argmin double[] first element is min"
    (is (= 0 (argmin (double-array [-5.0 2.0 3.0])))))
  (testing "argmin double[] single element"
    (is (= 0 (argmin (double-array [42.0])))))
  (testing "argmin double[] all same values returns first"
    (is (= 0 (argmin (double-array [3.0 3.0 3.0])))))
  (testing "argmin double[] negative values"
    (is (= 0 (argmin (double-array [-5.0 -3.0 -1.0 -2.0]))))))

(deftest argmin-float-test
  (testing "argmin float[] basic"
    (is (= 0 (argmin (float-array [1.0 5.0 3.0])))))
  (testing "argmin float[] single element"
    (is (= 0 (argmin (float-array [42.0])))))
  (testing "argmin float[] all same"
    (is (= 0 (argmin (float-array [2.0 2.0 2.0]))))))

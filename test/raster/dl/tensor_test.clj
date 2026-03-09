(ns raster.dl.tensor-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.tensor :as t]))

(deftest tensor-creation-test
  (testing "tensor creates array with shape metadata"
    (let [x (t/tensor [1 2 3 4 5 6] [2 3])]
      (is (= [2 3] (t/tshape x)))
      (is (= 6 (t/tsize x)))
      (is (= 2 (t/trank x)))
      (is (= 1.0 (aget x 0)))
      (is (= 6.0 (aget x 5)))))

  (testing "tensor from double array"
    (let [arr (double-array [1 2 3])
          x (t/tensor arr [3])]
      (is (= [3] (t/tshape x)))
      (is (= 3 (t/tsize x)))))

  (testing "tensor rejects shape mismatch"
    (is (thrown? AssertionError (t/tensor [1 2 3] [2 2])))))

(deftest zeros-ones-test
  (testing "zeros creates zero array"
    (let [x (t/zeros [2 3])]
      (is (= [2 3] (t/tshape x)))
      (is (every? zero? (seq x)))))

  (testing "ones creates all-ones array"
    (let [x (t/ones [3 2])]
      (is (= [3 2] (t/tshape x)))
      (is (every? #(== 1.0 %) (seq x)))))

  (testing "full creates constant array"
    (let [x (t/full [2 2] 3.14)]
      (is (every? #(< (Math/abs (- % 3.14)) 1e-10) (seq x))))))

(deftest randn-test
  (testing "randn has correct shape"
    (let [x (t/randn [10 20])]
      (is (= [10 20] (t/tshape x)))
      (is (= 200 (t/tsize x)))))

  (testing "randn is reproducible with same seed"
    (let [rng1 (java.util.Random. 42)
          rng2 (java.util.Random. 42)
          x1 (t/randn [5] rng1)
          x2 (t/randn [5] rng2)]
      (is (t/tensor-eq? x1 x2)))))

(deftest arange-test
  (testing "arange creates [0..n-1]"
    (let [x (t/arange 5)]
      (is (= [5] (t/tshape x)))
      (is (= 0.0 (aget x 0)))
      (is (= 4.0 (aget x 4))))))

(deftest linspace-test
  (testing "linspace creates evenly spaced values"
    (let [x (t/linspace 0.0 1.0 5)]
      (is (= [5] (t/tshape x)))
      (is (< (Math/abs (- (aget x 0) 0.0)) 1e-10))
      (is (< (Math/abs (- (aget x 2) 0.5)) 1e-10))
      (is (< (Math/abs (- (aget x 4) 1.0)) 1e-10)))))

(deftest tidx-test
  (testing "2D index computation"
    (is (= 0 (t/tidx [2 3] 0 0)))
    (is (= 1 (t/tidx [2 3] 0 1)))
    (is (= 3 (t/tidx [2 3] 1 0)))
    (is (= 5 (t/tidx [2 3] 1 2))))

  (testing "3D index computation"
    ;; shape [2,3,4]: stride = [12, 4, 1]
    (is (= 0 (t/tidx [2 3 4] 0 0 0)))
    (is (= 1 (t/tidx [2 3 4] 0 0 1)))
    (is (= 4 (t/tidx [2 3 4] 0 1 0)))
    (is (= 12 (t/tidx [2 3 4] 1 0 0)))
    (is (= 23 (t/tidx [2 3 4] 1 2 3))))

  (testing "out of bounds throws"
    (is (thrown? AssertionError (t/tidx [2 3] 2 0)))
    (is (thrown? AssertionError (t/tidx [2 3] 0 3)))))

(deftest strides-test
  (testing "2D strides"
    (is (= [3 1] (t/strides [2 3]))))
  (testing "3D strides"
    (is (= [12 4 1] (t/strides [2 3 4]))))
  (testing "4D strides"
    (is (= [60 20 4 1] (t/strides [2 3 5 4])))))

(deftest reshape-test
  (testing "reshape changes shape metadata"
    (let [x (t/tensor [1 2 3 4 5 6] [2 3])
          y (t/reshape x [3 2])]
      (is (= [3 2] (t/tshape y)))
      ;; same underlying data
      (is (= (aget x 0) (aget y 0)))
      (is (= (aget x 5) (aget y 5)))))

  (testing "reshape rejects size mismatch"
    (is (thrown? AssertionError (t/reshape (t/zeros [2 3]) [2 4])))))

(deftest transpose-2d-test
  (testing "transpose swaps rows and columns"
    (let [x (t/tensor [1 2 3 4 5 6] [2 3])
          ;; x = [[1 2 3], [4 5 6]]
          y (t/transpose-2d x 2 3)]
      ;; y = [[1 4], [2 5], [3 6]]
      (is (= [3 2] (t/tshape y)))
      (is (= 1.0 (aget y (t/tidx [3 2] 0 0))))
      (is (= 4.0 (aget y (t/tidx [3 2] 0 1))))
      (is (= 2.0 (aget y (t/tidx [3 2] 1 0))))
      (is (= 5.0 (aget y (t/tidx [3 2] 1 1))))
      (is (= 3.0 (aget y (t/tidx [3 2] 2 0))))
      (is (= 6.0 (aget y (t/tidx [3 2] 2 1))))))

  (testing "double transpose is identity"
    (let [x (t/tensor [1 2 3 4 5 6] [2 3])
          y (t/transpose-2d (t/transpose-2d x 2 3) 3 2)]
      (is (t/tensor-eq? x y)))))

(deftest flatten-tensor-test
  (testing "flatten makes 1D"
    (let [x (t/tensor [1 2 3 4] [2 2])
          y (t/flatten-tensor x)]
      (is (= [4] (t/tshape y))))))

(deftest tensor-add-test
  (testing "element-wise addition"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor [4 5 6] [3])
          c (t/tensor-add a b)]
      (is (= 5.0 (aget c 0)))
      (is (= 7.0 (aget c 1)))
      (is (= 9.0 (aget c 2))))))

(deftest tensor-scale-test
  (testing "scalar multiplication"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor-scale a 2.0)]
      (is (= 2.0 (aget b 0)))
      (is (= 4.0 (aget b 1)))
      (is (= 6.0 (aget b 2))))))

(deftest tensor-eq-test
  (testing "equal tensors"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor [1 2 3] [3])]
      (is (t/tensor-eq? a b))))
  (testing "unequal tensors"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor [1 2 4] [3])]
      (is (not (t/tensor-eq? a b))))))

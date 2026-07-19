(ns raster.par-contract-test
  "W1 step 1: par/contract surface combinator (runtime fallback correctness).
   Verifies the explicit free/contracted-axis contraction computes the same as an
   independent reference across free-axis ranks + a transpose variant, and that its
   output composes with a downstream elementwise op (the fusability the IR must keep)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par :as par]))

(defn- ref-matmul-nn
  "Reference C[m,n] = A[m,k]·B[k,n], row-major flat double[]."
  [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (let [s (loop [l 0 acc 0.0]
                  (if (< l k)
                    (recur (inc l) (+ acc (* (aget A (+ (* i k) l))
                                             (aget B (+ (* l n) j)))))
                    acc))]
          (aset C (+ (* i n) j) s))))
    C))

(deftest contract-matmul-nn
  (testing "2 free axes (i,j), 1 contracted axis (l) = C = A·B :nn"
    (let [m 3 k 4 n 2
          A (double-array (map double (range (* m k))))
          B (double-array (map #(* 0.5 (double %)) (range (* k n))))
          C (double-array (* m n))]
      (par/contract C [[i m] [j n]] [[l k]]
                    (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))
      (is (= (vec C) (vec (ref-matmul-nn A B m k n)))))))

(deftest contract-matmul-nt
  (testing "transpose variant :nt (B stored [n,k]) — axis roles unchanged, index math differs"
    (let [m 3 k 4 n 2
          A (double-array (map double (range (* m k))))
          Bnt (double-array (map #(* 0.25 (double %)) (range (* n k)))) ; B[j,l] row-major [n,k]
          C (double-array (* m n))
          ref (double-array (* m n))]
      (par/contract C [[i m] [j n]] [[l k]]
                    (* (aget A (+ (* i k) l)) (aget Bnt (+ (* j k) l))))
      (dotimes [i m]
        (dotimes [j n]
          (aset ref (+ (* i n) j)
                (loop [l 0 acc 0.0]
                  (if (< l k)
                    (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget Bnt (+ (* j k) l)))))
                    acc)))))
      (is (= (vec C) (vec ref))))))

(deftest contract-single-free-axis
  (testing "1 free axis = matrix-vector y[m] = A[m,k]·x[k]"
    (let [m 5 k 3
          A (double-array (map double (range (* m k))))
          x (double-array (map #(+ 1.0 (double %)) (range k)))
          y (double-array m)
          ref (double-array m)]
      (par/contract y [[i m]] [[l k]]
                    (* (aget A (+ (* i k) l)) (aget x l)))
      (dotimes [i m]
        (aset ref i (loop [l 0 acc 0.0]
                      (if (< l k) (recur (inc l) (+ acc (* (aget A (+ (* i k) l)) (aget x l)))) acc))))
      (is (= (vec y) (vec ref))))))

(deftest contract-three-free-axes
  (testing "3 free axes (batched matmul) — row-major decompose over (b,i,j)"
    (let [btch 2 m 2 n 3 k 2
          A (double-array (map double (range (* btch m k))))       ; A[b,i,l] [btch,m,k]
          B (double-array (map #(* 0.5 (double %)) (range (* btch k n)))) ; B[b,l,j] [btch,k,n]
          C (double-array (* btch m n))
          ref (double-array (* btch m n))]
      (par/contract C [[b btch] [i m] [j n]] [[l k]]
                    (* (aget A (+ (* (+ (* b m) i) k) l))
                       (aget B (+ (* (+ (* b k) l) n) j))))
      (dotimes [b btch]
        (dotimes [i m]
          (dotimes [j n]
            (aset ref (+ (* (+ (* b m) i) n) j)
                  (loop [l 0 acc 0.0]
                    (if (< l k)
                      (recur (inc l) (+ acc (* (aget A (+ (* (+ (* b m) i) k) l))
                                               (aget B (+ (* (+ (* b k) l) n) j)))))
                      acc))))))
      (is (= (vec C) (vec ref))))))

(deftest contract-composes-with-downstream
  (testing "contract output feeds a downstream elementwise map (relu) — compositional"
    (let [m 3 n 2 k 2
          A (double-array (map #(- (double %) 3.0) (range (* m k))))
          B (double-array (map #(- (double %) 2.0) (range (* k n))))
          C (double-array (* m n))]
      (par/contract C [[i m] [j n]] [[l k]]
                    (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))
      (let [relu (par/map [idx (* m n)] (Math/max 0.0 (aget C idx)))
            gemm (ref-matmul-nn A B m k n)]
        (is (= (vec relu) (mapv #(Math/max 0.0 (double %)) (vec gemm))))))))

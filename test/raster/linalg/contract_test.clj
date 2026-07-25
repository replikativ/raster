(ns raster.linalg.contract-test
  "A2: differentiable contraction. Forward matches matmul; reverse-mode value+grad matches
   finite differences to machine precision — proving the Dex/Futhark design that the backward
   of a contraction is TWO contractions (dA = C̄·Bᵀ, dB = Aᵀ·C̄), themselves par/contract forms."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.linalg.contract :as lc]))

(defn- ref-matmul [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m] (dotimes [j n]
      (aset C (+ (* i n) j)
            (loop [l 0 a 0.0] (if (< l k)
                                (recur (inc l) (+ a (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))
                                a)))))
    C))

(deftest contract-mm-forward-is-matmul
  (let [A (double-array [1 2 3 4 5 6]) B (double-array [1 2 3 4 5 6])]  ; 3×2 · 2×3
    (is (= (vec (lc/contract-mm A B 3 2 3)) (vec (ref-matmul A B 3 2 3))))))

;; L(A,B) = ½ Σ C²  where C = contract-mm(A,B);  dL/dC = C, so
;; dL/dA = C·Bᵀ = contract-mm-dA(C,B),  dL/dB = Aᵀ·C = contract-mm-dB(A,C).
(deftm cmm-loss [A :- (Array double), B :- (Array double), m :- Long, k :- Long, n :- Long] :- Double
  (let [C (raster.linalg.contract/contract-mm A B m k n)]
    (raster.par/reduce acc 0.0 i (* m n) (* 0.5 (* (ra/aget C i) (ra/aget C i))))))

(deftest contract-mm-value+grad-matches-finite-diff
  (testing "reverse-mode gradient of a contraction loss == central finite differences"
    (let [m 3 k 4 n 2 rng (java.util.Random. 11)
          A (double-array (repeatedly (* m k) #(.nextGaussian rng)))
          B (double-array (repeatedly (* k n) #(.nextGaussian rng)))
          vg (rev/value+grad #'cmm-loss)
          res (vg A B m k n)
          dA (nth res 1) dB (nth res 2)
          Lf (fn [a b] (nth (vg a b m k n) 0))
          eps 1.0e-6
          fd (fn [arr Lf1] (mapv (fn [p] (let [a+ (aclone arr) a- (aclone arr)]
                                           (aset a+ p (+ (aget arr p) eps)) (aset a- p (- (aget arr p) eps))
                                           (/ (- (Lf1 a+) (Lf1 a-)) (* 2 eps))))
                                 (range (alength arr))))
          fdA (fd A (fn [a] (Lf a B)))
          fdB (fd B (fn [b] (Lf A b)))]
      (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-5) dA fdA))
          (str "dA vs FD: " (vec dA) " / " fdA))
      (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-5) dB fdB))
          "dB vs FD"))))

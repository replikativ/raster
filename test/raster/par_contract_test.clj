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

;; ── A0: par/contract registered as a first-class par form ──────────────────────────
(deftest a0-contract-is-a-par-form
  (require '[raster.compiler.ir.par :as irpar] '[raster.compiler.ir.form :as form])
  (let [cform '(raster.par/contract C [[i 4] [j 3]] [[l 2]]
                 (clojure.core/* (clojure.core/aget A (clojure.core/+ (clojure.core/* i 2) l))
                                 (clojure.core/aget B (clojure.core/+ (clojure.core/* l 3) j))))
        par-form? (resolve 'raster.compiler.ir.par/par-form?)
        expand (resolve 'raster.compiler.ir.par/expand-par-forms)
        form-info (resolve 'raster.compiler.ir.form/form-info)]
    (testing "recognized as a par form (qualified + alias)"
      (is (par-form? cform))
      (is (par-form? '(par/contract C [[i 4]] [[l 2]] x))))
    (testing "form-info returns arg 0 (out) — redomap-shaped, not arg 1 (free-axes)"
      (is (= 0 (:return-type-arg (form-info cform))))
      (is (= :par (:kind (form-info cform)))))
    (testing "expand-par-forms (CPU fallback) computes the matmul"
      (let [A (double-array [1 2 3 4 5 6 7 8]) B (double-array [1 2 3 4 5 6]) C (double-array 12)
            f (eval (list 'fn '[A B C] (expand cform)))]
        (f A B C)
        (is (= (vec C) (mapv double [9 12 15 19 26 33 29 40 51 39 54 69])))))))

(deftest contract-result-transform-reads-the-destination
  (testing ":epilogue is the typed kernels' result transform: C := acc + beta·C on the host too"
    (let [m 3 k 4 n 2
          A (double-array (map double (range (* m k))))
          B (double-array (map #(* 0.5 (double %)) (range (* k n))))
          C (double-array (map #(* 10.0 (double %)) (range (* m n))))
          expected (let [product (ref-matmul-nn A B m k n)]
                     (mapv (fn [p c] (+ p (* 0.5 c))) (vec product) (vec C)))]
      ;; the operand map is macro-time data: the axis-map of the destination over the free axes
      (par/contract C [[i m] [j n]] [[l k]]
                    (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))
                    :epilogue {:acc acc
                               :expr (+ acc (* 0.5 (aget C (+ (* i n) j))))
                               :operands [{:sym C :map {:groups [[[i m]] [[j n]]]} :dtype :double}]
                               :scalars []
                               :dtype :double})
      (is (= expected (vec C))))))

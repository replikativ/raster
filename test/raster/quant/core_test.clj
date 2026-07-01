(ns raster.quant.core-test
  "QMatrix — the extensible quant surface. Validates that (matmul (qmatrix fmt W) x)
   plumbs to the same result as the raw kernel path (the kernels themselves are
   validated bit-faithful in raster.quant.qlinear-k-test), and that a user can register
   a NEW format and have matmul dispatch to it."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.quant.core :as qm]
            [raster.compiler.backend.cpu.quant :as q]
            [raster.quant.qlinear-k :as qk]))

(defn- gen [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(deftest qmatrix-q4k-plumbing
  (testing "QMatrix{:q4_K} · x == the raw qmatmul-q4k path"
    (let [out 8 in 256
          W (gen (* out in) 1) x (gen in 2)
          y (qm/matmul (qm/qmatrix :q4_K W out in) x)
          {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
          {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q4-K)
          yref (float-array out)]
      (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq yref in out 0 out)
      (dotimes [o out]
        (is (= (aget y o) (aget yref o)) (str "q4_K row " o))))))

(deftest qmatrix-q6k-plumbing
  (testing "QMatrix{:q6_K} · x == the raw qmatmul-q6k path"
    (let [out 8 in 256
          W (gen (* out in) 3) x (gen in 4)
          y (qm/matmul (qm/qmatrix :q6_K W out in) x)
          {:keys [wq sc ds]} (q/quantize-weight-q6k W q/q6-K)
          {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
          yref (float-array out)]
      (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds yref in out 0 out)
      (dotimes [o out]
        (is (= (aget y o) (aget yref o)) (str "q6_K row " o))))))

(deftest qmatrix-dispatches-on-format
  (testing "matmul dispatches on the QMatrix's format field"
    (let [out 8 in 256 W (gen (* out in) 5) x (gen in 6)
          y4 (qm/matmul (qm/qmatrix :q4_K W out in) x)
          y6 (qm/matmul (qm/qmatrix :q6_K W out in) x)]
      ;; different formats → different (both valid) results from the same W,x
      (is (not= (seq y4) (seq y6)) "q4_K and q6_K take different code paths")
      (is (= 8 (count y4) (count y6))))))

(deftest user-registered-format
  (testing "a user can register a NEW format and matmul dispatches to it"
    ;; a toy 'format': store the raw f32 weight, matmul = plain dense (proves the
    ;; extension seam — register-quant! + qmatrix + matmul — with zero core changes)
    (qm/register-quant! ::dense-f32
      {:descriptor {:name "dense_f32"}
       :quantize (fn [W _rows _cols _d] {:w W})
       :apply (fn [qmx x]
                (let [W (:w (:arrays qmx)) in (long (:cols qmx)) out (long (:rows qmx))
                      y (float-array out)]
                  (dotimes [o out]
                    (aset y o (float (areduce ^floats x k s (float 0.0)
                                              (+ s (* (aget ^floats W (+ (* o in) k)) (aget ^floats x k)))))))
                  y))})
    (is (contains? (qm/registered-formats) ::dense-f32))
    (let [out 4 in 16 W (gen (* out in) 7) x (gen in 8)
          y (qm/matmul (qm/qmatrix ::dense-f32 W out in) x)
          yref (float-array out)]
      (dotimes [o out]
        (aset yref o (float (areduce ^floats x k s (float 0.0)
                                     (+ s (* (aget ^floats W (+ (* o in) k)) (aget ^floats x k)))))))
      (dotimes [o out] (is (= (aget y o) (aget yref o)) (str "dense row " o))))))

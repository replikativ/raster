(ns raster.compiler.ad.mode-select-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ad.mode-select :as ms]
            [raster.sym.analysis :as sa]))

;; ================================================================
;; Mode selection heuristics (using raw expressions, no deftm needed)
;; ================================================================

(deftest mode-selection-scalar-output
  (testing "scalar output → reverse"
    (let [analysis (ms/analyze-from-exprs
                    ['(+ (* x x) (* y y) (* z z))]
                    '[x y z])]
      (is (= [1 3] (:shape analysis)))
      (is (= :reverse (:recommended-mode analysis))))))

(deftest mode-selection-tall-jacobian
  (testing "tall Jacobian (m > 3n) → forward"
    (let [analysis (ms/analyze-from-exprs
                    ['(* 2 x) '(* 3 x) '(* 4 x) '(* 5 x)]
                    '[x])]
      (is (= [4 1] (:shape analysis)))
      (is (= :forward (:recommended-mode analysis))))))

(deftest mode-selection-sparse
  (testing "sparse diagonal Jacobian → mixed-sparse"
    ;; 4 outputs each depending on only 1 of 4 inputs → diagonal, density 0.25
    (let [analysis (ms/analyze-from-exprs
                    ['(* x x) '(* y y) '(* z z) '(* w w)]
                    '[x y z w])]
      (is (< (:density analysis) 0.3))
      (is (= :mixed-sparse (:recommended-mode analysis))))))

;; ================================================================
;; Linearity analysis integration
;; ================================================================

(deftest linearity-in-analysis
  (testing "linearity per variable"
    (let [analysis (ms/analyze-from-exprs
                    ['(+ (* x x) (* 3 y))]
                    '[x y])]
      (is (= :quadratic (get-in analysis [:linearity 'x])))
      (is (= :linear (get-in analysis [:linearity 'y]))))))

;; ================================================================
;; Sparsity patterns
;; ================================================================

(deftest sparsity-pattern
  (testing "correct sparsity detection"
    (let [analysis (ms/analyze-from-exprs
                    ['(* x y) '(+ x (* z z))]
                    '[x y z])]
      ;; Row 0 (x*y): depends on x,y but not z
      (is (= [true true false] (get-in analysis [:sparsity 0])))
      ;; Row 1 (x+z²): depends on x,z but not y
      (is (= [true false true] (get-in analysis [:sparsity 1]))))))

;; ================================================================
;; Sparse seeding (column coloring)
;; ================================================================

(deftest sparse-seeds-test
  (testing "diagonal sparsity → each column independent"
    (let [sparsity [[true false false]
                    [false true false]
                    [false false true]]
          seeds (ms/sparse-seeds sparsity)]
      ;; All 3 columns are independent (no shared rows) → 1 seed
      (is (= 1 (count seeds)))
      ;; The single seed contains all 3 columns
      (is (= 3 (count (first seeds))))))

  (testing "dense sparsity → one seed per column"
    (let [sparsity [[true true true]
                    [true true true]]
          seeds (ms/sparse-seeds sparsity)]
      ;; All columns share rows → 3 seeds needed
      (is (= 3 (count seeds))))))

;; ================================================================
;; Hessian structure (raw expression)
;; ================================================================

(deftest hessian-analysis-raw
  (testing "diagonal Hessian detection via raw expressions"
    ;; x² + y² has diagonal Hessian
    (let [expr '(+ (* x x) (* y y))
          sparsity (sa/hessian-sparsity expr '[x y])]
      ;; H_xx=2, H_yy=2, H_xy=H_yx=0
      (is (= true (get-in sparsity [0 0])))
      (is (= false (get-in sparsity [0 1])))
      (is (= false (get-in sparsity [1 0])))
      (is (= true (get-in sparsity [1 1]))))))

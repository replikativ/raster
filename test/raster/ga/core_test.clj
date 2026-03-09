(ns raster.ga.core-test
  "Tests for geometric algebra (Clifford algebra) implementation."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ga.core :as ga]
            [raster.numeric :as n]
            [raster.ode.core]
            [raster.arrays]
            [raster.core]))

(def ^:private eps 1e-12)
(defn- approx= [a b] (< (Math/abs (- (double a) (double b))) eps))

;; ================================================================
;; Signatures
;; ================================================================

(deftest signature-test
  (testing "VGA(3) — Euclidean 3D"
    (let [sig (ga/vga 3)]
      (is (= 8 (ga/algebra-dim sig)))
      (is (= {:p 3 :q 0 :r 0} {:p (.p sig) :q (.q sig) :r (.r sig)}))))

  (testing "PGA(3) — Projective 3D"
    (let [sig (ga/pga 3)]
      (is (= 16 (ga/algebra-dim sig)))
      (is (= {:p 3 :q 0 :r 1} {:p (.p sig) :q (.q sig) :r (.r sig)}))))

  (testing "STA — Spacetime algebra"
    (let [sig (ga/sta)]
      (is (= 16 (ga/algebra-dim sig)))
      (is (= {:p 1 :q 3 :r 0} {:p (.p sig) :q (.q sig) :r (.r sig)})))))

;; ================================================================
;; Basis vector metric
;; ================================================================

(deftest metric-test
  (testing "Euclidean: all basis vectors square to +1"
    (let [sig (ga/vga 3)]
      (doseq [i (range 3)]
        (let [ei (ga/basis sig i)
              sq (ga/geometric-product ei ei)]
          (is (approx= 1.0 (ga/scalar-product sq (ga/scalar-mv sig 1.0)))
              (str "e" i "*e" i " should be 1"))))))

  (testing "Spacetime: e0^2=+1, e1^2=e2^2=e3^2=-1"
    (let [sig (ga/sta)]
      (let [e0-sq (ga/scalar-product (ga/geometric-product (ga/basis sig 0) (ga/basis sig 0))
                                     (ga/scalar-mv sig 1.0))]
        (is (approx= 1.0 e0-sq) "timelike basis squares to +1"))
      (doseq [i (range 1 4)]
        (let [sq (ga/scalar-product (ga/geometric-product (ga/basis sig i) (ga/basis sig i))
                                    (ga/scalar-mv sig 1.0))]
          (is (approx= -1.0 sq) (str "spacelike e" i " squares to -1"))))))

  (testing "PGA: degenerate basis squares to 0"
    (let [sig (ga/pga 3)
          e3 (ga/basis sig 3) ;; degenerate (r=1)
          sq (ga/geometric-product e3 e3)]
      (is (approx= 0.0 (ga/norm sq)) "degenerate basis squares to 0"))))

;; ================================================================
;; Product properties
;; ================================================================

(deftest anticommutativity-test
  (testing "distinct basis vectors anticommute"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1)]
      (let [e1e2 (ga/geometric-product e1 e2)
            e2e1 (ga/geometric-product e2 e1)
            sum (n/+ e1e2 e2e1)]
        (is (approx= 0.0 (ga/norm sum)) "e1*e2 + e2*e1 = 0")))))

(deftest associativity-test
  (testing "geometric product is associative"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1) e3 (ga/basis sig 2)
          lhs (ga/geometric-product (ga/geometric-product e1 e2) e3)
          rhs (ga/geometric-product e1 (ga/geometric-product e2 e3))]
      (is (approx= 0.0 (ga/norm (n/- lhs rhs))) "(e1*e2)*e3 = e1*(e2*e3)"))))

(deftest wedge-product-test
  (testing "wedge product equals geometric product for orthogonal vectors"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1)
          gp (ga/geometric-product e1 e2)
          wp (ga/wedge e1 e2)]
      (is (approx= 0.0 (ga/norm (n/- gp wp)))
          "e1*e2 = e1^e2 for orthogonal basis")))

  (testing "wedge product is antisymmetric"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1)]
      (is (approx= 0.0 (ga/norm (n/+ (ga/wedge e1 e2) (ga/wedge e2 e1))))
          "e1^e2 + e2^e1 = 0")))

  (testing "wedge of vector with itself is zero"
    (let [sig (ga/vga 3)
          v (n/+ (ga/basis sig 0) (n/* 2.0 (ga/basis sig 1)))]
      (is (approx= 0.0 (ga/norm (ga/wedge v v))) "v^v = 0"))))

;; ================================================================
;; Involutions
;; ================================================================

(deftest reverse-test
  (testing "reverse is an involution (reverse(reverse(x)) = x)"
    (let [sig (ga/vga 3)
          e12 (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1))]
      (is (approx= 0.0 (ga/norm (n/- (ga/reverse-mv (ga/reverse-mv e12)) e12)))
          "reverse is involutory")))

  (testing "reverse of grade-k flips sign for k(k-1)/2 odd"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0)
          e12 (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1))]
      ;; grade 1: reverse = identity (1*0/2 = 0, even)
      (is (approx= 0.0 (ga/norm (n/- (ga/reverse-mv e1) e1))))
      ;; grade 2: reverse = -identity (2*1/2 = 1, odd)
      (is (approx= 0.0 (ga/norm (n/+ (ga/reverse-mv e12) e12)))))))

;; ================================================================
;; Hodge dual
;; ================================================================

(deftest hodge-test
  (testing "Hodge dual of basis vectors in VGA(3)"
    (let [sig (ga/vga 3)
          e1 (ga/basis sig 0)
          star-e1 (ga/hodge-star e1)]
      ;; ⋆e1 = e2∧e3 in VGA(3)
      (is (approx= 1.0 (ga/norm star-e1)) "hodge preserves norm")
      ;; ⋆(⋆e1) should give ±e1
      (let [dbl-star (ga/hodge-star star-e1)]
        (is (or (approx= 0.0 (ga/norm (n/- dbl-star e1)))
                (approx= 0.0 (ga/norm (n/+ dbl-star e1))))
            "double Hodge gives ±identity")))))

;; ================================================================
;; Numeric dispatch integration
;; ================================================================

(deftest numeric-dispatch-test
  (testing "scalar * multivector"
    (let [sig (ga/vga 2)
          e1 (ga/basis sig 0)]
      (is (approx= 3.0 (ga/norm (n/* 3.0 e1))))))

  (testing "multivector + multivector"
    (let [sig (ga/vga 2)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1)]
      (is (approx= (Math/sqrt 2.0) (ga/norm (n/+ e1 e2))))))

  (testing "multivector - multivector"
    (let [sig (ga/vga 2)
          e1 (ga/basis sig 0)]
      (is (approx= 0.0 (ga/norm (n/- e1 e1)))))))

;; ================================================================
;; Grade selection
;; ================================================================

(deftest grade-select-test
  (testing "grade selection extracts correct components"
    (let [sig (ga/vga 3)
          s (ga/scalar-mv sig 5.0)
          e1 (ga/basis sig 0)
          e12 (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1))
          mixed (n/+ s (n/+ e1 e12))]
      (is (approx= 5.0 (ga/norm (ga/grade-select mixed 0))) "grade 0 = scalar")
      (is (approx= 1.0 (ga/norm (ga/grade-select mixed 1))) "grade 1 = vector")
      (is (approx= 1.0 (ga/norm (ga/grade-select mixed 2))) "grade 2 = bivector")
      (is (approx= 0.0 (ga/norm (ga/grade-select mixed 3))) "grade 3 = 0"))))

;; ================================================================
;; Pseudoscalar
;; ================================================================

(deftest pseudoscalar-test
  (testing "pseudoscalar squared"
    ;; In VGA(3): I^2 = e123*e123 = -1
    (let [sig (ga/vga 3)
          I (ga/pseudoscalar sig)
          I2 (ga/geometric-product I I)]
      (is (approx= -1.0 (ga/scalar-product I2 (ga/scalar-mv sig 1.0)))
          "I^2 = -1 in VGA(3)")))

  (testing "pseudoscalar in VGA(2): I^2 = -1"
    (let [sig (ga/vga 2)
          I (ga/pseudoscalar sig)
          I2 (ga/geometric-product I I)]
      (is (approx= -1.0 (ga/scalar-product I2 (ga/scalar-mv sig 1.0)))))))

;; ================================================================
;; Rotor (rotation) application
;; ================================================================

(deftest rotor-test
  (testing "90-degree rotation in 2D via rotor"
    (let [sig (ga/vga 2)
          e1 (ga/basis sig 0) e2 (ga/basis sig 1)
          ;; Rotor for 90° = exp(-π/4 * e12) = cos(π/4) - sin(π/4)*e12
          angle (/ Math/PI 4.0)
          e12 (ga/geometric-product e1 e2)
          R (n/+ (ga/scalar-mv sig (Math/cos angle))
                 (n/* (- (Math/sin angle)) e12))
          R-rev (ga/reverse-mv R)
          ;; Rotate e1: R*e1*R†
          rotated (ga/geometric-product R (ga/geometric-product e1 R-rev))
          ;; Should be e2 (90° rotation)
          diff (n/- rotated e2)]
      (is (< (ga/norm diff) 1e-10) "90° rotation of e1 gives e2"))))

;; ================================================================
;; GA + ODE: rotor kinematics with native Multivector state
;; ================================================================

(deftest ga-ode-rotor-test
  (testing "180° rotation via ODE integration of dR/dt = 0.5*ω*R"
    (let [sig (ga/vga 3)
          dim (ga/algebra-dim sig)
          omega (n/* Math/PI (ga/geometric-product (ga/basis sig 0) (ga/basis sig 1)))
          rhs (raster.core/ftm [du :- raster.ga.core.Multivector,
                                u :- raster.ga.core.Multivector,
                                t :- Double]
                               (let [dRdt (n/* 0.5 (ga/geometric-product omega u))]
                                 (dotimes [i dim]
                                   (raster.arrays/aset du i (raster.arrays/aget dRdt i)))))
          R0 (ga/scalar-mv sig 1.0)
          prob (raster.ode.core/ode-problem rhs R0 0.0 1.0)
          sol (raster.ode.core/solve (raster.ode.core/->RK4) prob 0.001)
          R-final (last (:us sol))]
      (is (instance? raster.ga.core.Multivector R-final)
          "State type is Multivector")
      (is (< (Math/abs (- 1.0 (ga/norm R-final))) 1e-6)
          "Rotor norm preserved ≈ 1.0")
      (let [e1 (ga/basis sig 0)
            rotated (ga/geometric-product R-final
                                          (ga/geometric-product e1 (ga/reverse-mv R-final)))
            expected (n/* -1.0 e1)
            diff (n/- rotated expected)]
        (is (< (ga/norm diff) 1e-6)
            "e1 rotated 180° gives -e1")))))

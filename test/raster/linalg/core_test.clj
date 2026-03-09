(ns raster.linalg.core-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * /]]
            [raster.linalg.core :refer [->Vec2 ->Vec3 ->Vec4
                                        ->Mat2x2 ->Mat3x3 ->Mat4x4
                                        dot cross norm normalize lerp
                                        det inv transpose identity-mat
                                        ->double-array
                                        vec2-from-array vec3-from-array vec4-from-array
                                        mat2x2-from-array mat3x3-from-array mat4x4-from-array
                                        dimension entry nrows ncols]]))

(def ^:const EPS 1e-12)

(defn- approx= [^double a ^double b]
  (< (Math/abs (clojure.core/- a b)) EPS))

;; ================================================================
;; Vec2
;; ================================================================

(deftest vec2-arithmetic-test
  (testing "Vec2 + Vec2"
    (let [r (+ (->Vec2 1.0 2.0) (->Vec2 3.0 4.0))]
      (is (approx= 4.0 (.x r)))
      (is (approx= 6.0 (.y r)))))
  (testing "Vec2 - Vec2"
    (let [r (- (->Vec2 5.0 3.0) (->Vec2 1.0 2.0))]
      (is (approx= 4.0 (.x r)))
      (is (approx= 1.0 (.y r)))))
  (testing "scalar * Vec2"
    (let [r (* 2.0 (->Vec2 3.0 4.0))]
      (is (approx= 6.0 (.x r)))
      (is (approx= 8.0 (.y r)))))
  (testing "Vec2 / scalar"
    (let [r (/ (->Vec2 6.0 8.0) 2.0)]
      (is (approx= 3.0 (.x r)))
      (is (approx= 4.0 (.y r)))))
  (testing "unary - Vec2"
    (let [r (- (->Vec2 1.0 -2.0))]
      (is (approx= -1.0 (.x r)))
      (is (approx= 2.0 (.y r))))))

(deftest vec2-ops-test
  (testing "dot product"
    (is (approx= 11.0 (dot (->Vec2 1.0 2.0) (->Vec2 3.0 4.0)))))
  (testing "2D cross product (scalar)"
    ;; cross: 1*4 - 2*3 = -2
    (is (approx= -2.0 (cross (->Vec2 1.0 2.0) (->Vec2 3.0 4.0)))))
  (testing "norm"
    (is (approx= 5.0 (norm (->Vec2 3.0 4.0)))))
  (testing "normalize"
    (let [n (normalize (->Vec2 3.0 4.0))]
      (is (approx= 0.6 (.x n)))
      (is (approx= 0.8 (.y n)))
      (is (approx= 1.0 (norm n)))))
  (testing "lerp"
    (let [r (lerp (->Vec2 0.0 0.0) (->Vec2 10.0 20.0) 0.25)]
      (is (approx= 2.5 (.x r)))
      (is (approx= 5.0 (.y r))))))

;; ================================================================
;; Vec3
;; ================================================================

(deftest vec3-arithmetic-test
  (testing "Vec3 + Vec3"
    (let [r (+ (->Vec3 1.0 2.0 3.0) (->Vec3 4.0 5.0 6.0))]
      (is (approx= 5.0 (.x r)))
      (is (approx= 7.0 (.y r)))
      (is (approx= 9.0 (.z r)))))
  (testing "Vec3 * scalar"
    (let [r (* (->Vec3 1.0 2.0 3.0) 3.0)]
      (is (approx= 3.0 (.x r)))
      (is (approx= 6.0 (.y r)))
      (is (approx= 9.0 (.z r))))))

(deftest vec3-ops-test
  (testing "dot product orthogonal"
    (is (approx= 0.0 (dot (->Vec3 1.0 0.0 0.0) (->Vec3 0.0 1.0 0.0)))))
  (testing "dot product parallel"
    (is (approx= 14.0 (dot (->Vec3 1.0 2.0 3.0) (->Vec3 1.0 2.0 3.0)))))
  (testing "cross product i×j = k"
    (let [r (cross (->Vec3 1.0 0.0 0.0) (->Vec3 0.0 1.0 0.0))]
      (is (approx= 0.0 (.x r)))
      (is (approx= 0.0 (.y r)))
      (is (approx= 1.0 (.z r)))))
  (testing "cross product j×i = -k"
    (let [r (cross (->Vec3 0.0 1.0 0.0) (->Vec3 1.0 0.0 0.0))]
      (is (approx= 0.0 (.x r)))
      (is (approx= 0.0 (.y r)))
      (is (approx= -1.0 (.z r)))))
  (testing "norm 3-4-5 triangle extended"
    (is (approx= 5.0 (norm (->Vec3 3.0 4.0 0.0)))))
  (testing "normalize"
    (let [n (normalize (->Vec3 0.0 0.0 5.0))]
      (is (approx= 0.0 (.x n)))
      (is (approx= 0.0 (.y n)))
      (is (approx= 1.0 (.z n))))))

;; ================================================================
;; Vec4
;; ================================================================

(deftest vec4-arithmetic-test
  (testing "Vec4 + Vec4"
    (let [r (+ (->Vec4 1.0 2.0 3.0 4.0) (->Vec4 5.0 6.0 7.0 8.0))]
      (is (approx= 6.0 (.x r)))
      (is (approx= 8.0 (.y r)))
      (is (approx= 10.0 (.z r)))
      (is (approx= 12.0 (.w r)))))
  (testing "dot product"
    (is (approx= 30.0 (dot (->Vec4 1.0 2.0 3.0 4.0) (->Vec4 1.0 2.0 3.0 4.0))))))

;; ================================================================
;; Mat2x2
;; ================================================================

(deftest mat2x2-test
  (testing "identity * vec"
    (let [I (identity-mat 2)
          v (->Vec2 3.0 4.0)
          r (* I v)]
      (is (approx= 3.0 (.x r)))
      (is (approx= 4.0 (.y r)))))
  (testing "mat * mat = identity for rotation"
    ;; 90 degree rotation: [[0 -1][1 0]]
    (let [R  (->Mat2x2 0.0 -1.0 1.0 0.0)
          Rt (->Mat2x2 0.0 1.0 -1.0 0.0)
          I  (* R Rt)]
      (is (approx= 1.0 (.m00 I)))
      (is (approx= 0.0 (.m01 I)))
      (is (approx= 0.0 (.m10 I)))
      (is (approx= 1.0 (.m11 I)))))
  (testing "determinant"
    (is (approx= -2.0 (det (->Mat2x2 1.0 2.0 3.0 4.0)))))
  (testing "inverse"
    (let [m (->Mat2x2 1.0 2.0 3.0 4.0)
          mi (inv m)
          I (* m mi)]
      (is (approx= 1.0 (.m00 I)))
      (is (approx= 0.0 (.m01 I)))
      (is (approx= 0.0 (.m10 I)))
      (is (approx= 1.0 (.m11 I)))))
  (testing "transpose"
    (let [m (->Mat2x2 1.0 2.0 3.0 4.0)
          t (transpose m)]
      (is (approx= 1.0 (.m00 t)))
      (is (approx= 3.0 (.m01 t)))
      (is (approx= 2.0 (.m10 t)))
      (is (approx= 4.0 (.m11 t))))))

;; ================================================================
;; Mat3x3
;; ================================================================

(deftest mat3x3-test
  (testing "identity * vec"
    (let [I (identity-mat 3)
          v (->Vec3 1.0 2.0 3.0)
          r (* I v)]
      (is (approx= 1.0 (.x r)))
      (is (approx= 2.0 (.y r)))
      (is (approx= 3.0 (.z r)))))
  (testing "determinant of identity"
    (is (approx= 1.0 (det (identity-mat 3)))))
  (testing "determinant"
    ;; [[1 2 3][0 1 4][5 6 0]] => det = 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1
    (is (approx= 1.0 (det (->Mat3x3 1.0 2.0 3.0 0.0 1.0 4.0 5.0 6.0 0.0)))))
  (testing "inverse roundtrip"
    (let [m (->Mat3x3 1.0 2.0 3.0 0.0 1.0 4.0 5.0 6.0 0.0)
          mi (inv m)
          I (* m mi)]
      (is (approx= 1.0 (.m00 I)))
      (is (approx= 0.0 (.m01 I)))
      (is (approx= 0.0 (.m02 I)))
      (is (approx= 0.0 (.m10 I)))
      (is (approx= 1.0 (.m11 I)))
      (is (approx= 0.0 (.m12 I)))
      (is (approx= 0.0 (.m20 I)))
      (is (approx= 0.0 (.m21 I)))
      (is (approx= 1.0 (.m22 I))))))

;; ================================================================
;; Mat4x4
;; ================================================================

(deftest mat4x4-test
  (testing "identity * vec"
    (let [I (identity-mat 4)
          v (->Vec4 1.0 2.0 3.0 4.0)
          r (* I v)]
      (is (approx= 1.0 (.x r)))
      (is (approx= 2.0 (.y r)))
      (is (approx= 3.0 (.z r)))
      (is (approx= 4.0 (.w r)))))
  (testing "determinant of identity"
    (is (approx= 1.0 (det (identity-mat 4)))))
  (testing "inverse roundtrip"
    ;; Use a known invertible matrix
    (let [m (->Mat4x4 2.0 0.0 0.0 1.0
                      0.0 1.0 0.0 0.0
                      0.0 0.0 3.0 0.0
                      0.0 0.0 0.0 1.0)
          mi (inv m)
          I (* m mi)]
      (is (approx= 1.0 (.m00 I)))
      (is (approx= 0.0 (.m01 I)))
      (is (approx= 0.0 (.m10 I)))
      (is (approx= 1.0 (.m11 I)))
      (is (approx= 1.0 (.m22 I)))
      (is (approx= 1.0 (.m33 I)))))
  (testing "mat4 + mat4"
    (let [a (identity-mat 4)
          b (identity-mat 4)
          r (+ a b)]
      (is (approx= 2.0 (.m00 r)))
      (is (approx= 2.0 (.m11 r))))))

;; ================================================================
;; Array conversions
;; ================================================================

(deftest array-conversion-test
  (testing "Vec3 roundtrip"
    (let [v (->Vec3 1.0 2.0 3.0)
          a (->double-array v)
          v2 (vec3-from-array a)]
      (is (approx= (.x v) (.x v2)))
      (is (approx= (.y v) (.y v2)))
      (is (approx= (.z v) (.z v2)))))
  (testing "Mat2x2 column-major roundtrip"
    (let [m (->Mat2x2 1.0 2.0 3.0 4.0)
          a (->double-array m)
          m2 (mat2x2-from-array a)]
      (is (approx= (.m00 m) (.m00 m2)))
      (is (approx= (.m01 m) (.m01 m2)))
      (is (approx= (.m10 m) (.m10 m2)))
      (is (approx= (.m11 m) (.m11 m2))))))

;; ================================================================
;; Mat*Vec products
;; ================================================================

(deftest mat-vec-product-test
  (testing "Mat3x3 * Vec3 rotation"
    ;; 90 degree rotation around z-axis
    (let [R (->Mat3x3 0.0 -1.0 0.0
                      1.0  0.0 0.0
                      0.0  0.0 1.0)
          v (->Vec3 1.0 0.0 0.0)
          r (* R v)]
      (is (approx= 0.0 (.x r)))
      (is (approx= 1.0 (.y r)))
      (is (approx= 0.0 (.z r))))))

;; ================================================================
;; Generic accessors
;; ================================================================

(deftest vector-accessors-test
  (testing "dimension"
    (is (= 2 (dimension (->Vec2 1 2))))
    (is (= 3 (dimension (->Vec3 1 2 3))))
    (is (= 4 (dimension (->Vec4 1 2 3 4)))))
  (testing "entry"
    (is (approx= 1.0 (entry (->Vec3 1 2 3) 0)))
    (is (approx= 2.0 (entry (->Vec3 1 2 3) 1)))
    (is (approx= 3.0 (entry (->Vec3 1 2 3) 2)))))

(deftest matrix-accessors-test
  (testing "nrows/ncols"
    (is (= 2 (nrows (->Mat2x2 1 2 3 4))))
    (is (= 2 (ncols (->Mat2x2 1 2 3 4))))
    (is (= 3 (nrows (->Mat3x3 1 0 0 0 1 0 0 0 1))))
    (is (= 3 (ncols (->Mat3x3 1 0 0 0 1 0 0 0 1)))))
  (testing "entry (row-major naming: m[i,j])"
    (let [m (->Mat2x2 1 2 3 4)]
      (is (approx= 1.0 (entry m 0 0)))
      (is (approx= 2.0 (entry m 0 1)))
      (is (approx= 3.0 (entry m 1 0)))
      (is (approx= 4.0 (entry m 1 1))))))

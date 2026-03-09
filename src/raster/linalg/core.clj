(ns raster.linalg.core
  "Linear algebra: small fixed-size types + dense array operations.

   Fixed-size types: Vec2/Vec3/Vec4, Mat2x2/Mat3x3/Mat4x4 with fully
   unrolled arithmetic, dot/cross products, determinants, and inverses.

   Dense array operations: solve, cholesky, lu, lstsq, inv, det,
   cond-number, matrix-norm — backed by LAPACK via raster.lapack.

   Extends raster.numeric/+, -, *, / for vector and matrix arithmetic.

   Usage:
     (require '[raster.linalg.core :refer [->Vec3 dot cross norm normalize solve]])
     (dot (->Vec3 1 0 0) (->Vec3 0 1 0))  ;=> 0.0
     (solve (double-array [2 1 1 3]) (double-array [5 7]) 2)  ;=> [1.4 1.2]"
  (:refer-clojure :exclude [+ - * / aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue defabstract]]
            [raster.types.algebraic-types :as alg]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n :refer [+ - * /]]
            [raster.linalg.lapack :as lapack]))

;; ================================================================
;; Abstract type hierarchy (like Julia's AbstractArray/AbstractVector/AbstractMatrix)
;; ================================================================

(defabstract AbstractArray)
(defabstract AbstractVector :extends [AbstractArray alg/Module])
(defabstract AbstractMatrix :extends [AbstractArray alg/Algebra])

;; ================================================================
;; Vector types
;; ================================================================

(defvalue Vec2 [x :- Double, y :- Double] :implements AbstractVector)
(defvalue Vec3 [x :- Double, y :- Double, z :- Double] :implements AbstractVector)
(defvalue Vec4 [x :- Double, y :- Double, z :- Double, w :- Double] :implements AbstractVector)

;; ================================================================
;; Matrix types (row-major field naming: mRC)
;; ================================================================

(defvalue Mat2x2 [m00 :- Double, m01 :- Double,
                  m10 :- Double, m11 :- Double]
  :implements AbstractMatrix)

(defvalue Mat3x3 [m00 :- Double, m01 :- Double, m02 :- Double,
                  m10 :- Double, m11 :- Double, m12 :- Double,
                  m20 :- Double, m21 :- Double, m22 :- Double]
  :implements AbstractMatrix)

(defvalue Mat4x4 [m00 :- Double, m01 :- Double, m02 :- Double, m03 :- Double,
                  m10 :- Double, m11 :- Double, m12 :- Double, m13 :- Double,
                  m20 :- Double, m21 :- Double, m22 :- Double, m23 :- Double,
                  m30 :- Double, m31 :- Double, m32 :- Double, m33 :- Double]
  :implements AbstractMatrix)

;; ================================================================
;; Generic accessors (Julia's AbstractArray interface)
;; ================================================================

;; --- Vector accessors ---

(deftm dimension
  "Return the number of elements in a vector."
  [v :- Vec2] :- Long 2)
(deftm dimension [v :- Vec3] :- Long 3)
(deftm dimension [v :- Vec4] :- Long 4)

(deftm entry
  "Access element i of a vector, or element (i,j) of a matrix."
  [v :- Vec2, i :- Long] :- Double
  (case (int i) 0 (.x v) 1 (.y v)))

(deftm entry [v :- Vec3, i :- Long] :- Double
  (case (int i) 0 (.x v) 1 (.y v) 2 (.z v)))

(deftm entry [v :- Vec4, i :- Long] :- Double
  (case (int i) 0 (.x v) 1 (.y v) 2 (.z v) 3 (.w v)))

;; --- Matrix accessors ---

(deftm nrows
  "Return the number of rows in a matrix."
  [m :- Mat2x2] :- Long 2)
(deftm ncols
  "Return the number of columns in a matrix."
  [m :- Mat2x2] :- Long 2)
(deftm nrows [m :- Mat3x3] :- Long 3)
(deftm ncols [m :- Mat3x3] :- Long 3)
(deftm nrows [m :- Mat4x4] :- Long 4)
(deftm ncols [m :- Mat4x4] :- Long 4)

(deftm entry [m :- Mat2x2, i :- Long, j :- Long] :- Double
  (let [idx (+ (* (int i) 2) (int j))]
    (case idx
      0 (.m00 m) 1 (.m01 m)
      2 (.m10 m) 3 (.m11 m))))

(deftm entry [m :- Mat3x3, i :- Long, j :- Long] :- Double
  (let [idx (+ (* (int i) 3) (int j))]
    (case idx
      0 (.m00 m) 1 (.m01 m) 2 (.m02 m)
      3 (.m10 m) 4 (.m11 m) 5 (.m12 m)
      6 (.m20 m) 7 (.m21 m) 8 (.m22 m))))

(deftm entry [m :- Mat4x4, i :- Long, j :- Long] :- Double
  (let [idx (+ (* (int i) 4) (int j))]
    (case idx
      0  (.m00 m) 1  (.m01 m) 2  (.m02 m) 3  (.m03 m)
      4  (.m10 m) 5  (.m11 m) 6  (.m12 m) 7  (.m13 m)
      8  (.m20 m) 9  (.m21 m) 10 (.m22 m) 11 (.m23 m)
      12 (.m30 m) 13 (.m31 m) 14 (.m32 m) 15 (.m33 m))))

;; ================================================================
;; Generic fallback implementations (on AbstractVector)
;; Specialized methods for concrete types (below) win by specificity.
;; These provide a safety net for any type that implements accessors.
;; ================================================================

(deftm dot
  "Dot product of two vectors."
  [a :- AbstractVector, b :- AbstractVector] :- Double
  (let [n (long (dimension a))]
    (loop [i (long 0) sum 0.0]
      (if (>= i n)
        sum
        (recur (inc i) (+ sum (* (entry a i) (entry b i))))))))

(deftm norm
  "Euclidean (L2) norm of a vector."
  [v :- AbstractVector] :- Double
  (n/sqrt (dot v v)))

(deftm normalize
  "Return unit vector in the direction of v."
  [v :- AbstractVector]
  (let [inv-n (/ 1.0 (norm v))]
    (raster.numeric/* inv-n v)))

;; ================================================================
;; Vec2 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Vec2, b :- Vec2] :- Vec2
  (->Vec2 (+ (.x a) (.x b))
          (+ (.y a) (.y b))))

(deftm raster.numeric/- [a :- Vec2] :- Vec2
  (->Vec2 (- (.x a))
          (- (.y a))))

(deftm raster.numeric/- [a :- Vec2, b :- Vec2] :- Vec2
  (->Vec2 (- (.x a) (.x b))
          (- (.y a) (.y b))))

(deftm raster.numeric/* [c :- Double, v :- Vec2] :- Vec2
  (->Vec2 (* c (.x v))
          (* c (.y v))))

(deftm raster.numeric/* [v :- Vec2, c :- Double] :- Vec2
  (->Vec2 (* (.x v) c)
          (* (.y v) c)))

(deftm raster.numeric// [v :- Vec2, c :- Double] :- Vec2
  (->Vec2 (/ (.x v) c)
          (/ (.y v) c)))

;; ================================================================
;; Vec3 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Vec3, b :- Vec3] :- Vec3
  (->Vec3 (+ (.x a) (.x b))
          (+ (.y a) (.y b))
          (+ (.z a) (.z b))))

(deftm raster.numeric/- [a :- Vec3] :- Vec3
  (->Vec3 (- (.x a))
          (- (.y a))
          (- (.z a))))

(deftm raster.numeric/- [a :- Vec3, b :- Vec3] :- Vec3
  (->Vec3 (- (.x a) (.x b))
          (- (.y a) (.y b))
          (- (.z a) (.z b))))

(deftm raster.numeric/* [c :- Double, v :- Vec3] :- Vec3
  (->Vec3 (* c (.x v))
          (* c (.y v))
          (* c (.z v))))

(deftm raster.numeric/* [v :- Vec3, c :- Double] :- Vec3
  (->Vec3 (* (.x v) c)
          (* (.y v) c)
          (* (.z v) c)))

(deftm raster.numeric// [v :- Vec3, c :- Double] :- Vec3
  (->Vec3 (/ (.x v) c)
          (/ (.y v) c)
          (/ (.z v) c)))

;; ================================================================
;; Vec4 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Vec4, b :- Vec4] :- Vec4
  (->Vec4 (+ (.x a) (.x b))
          (+ (.y a) (.y b))
          (+ (.z a) (.z b))
          (+ (.w a) (.w b))))

(deftm raster.numeric/- [a :- Vec4] :- Vec4
  (->Vec4 (- (.x a))
          (- (.y a))
          (- (.z a))
          (- (.w a))))

(deftm raster.numeric/- [a :- Vec4, b :- Vec4] :- Vec4
  (->Vec4 (- (.x a) (.x b))
          (- (.y a) (.y b))
          (- (.z a) (.z b))
          (- (.w a) (.w b))))

(deftm raster.numeric/* [c :- Double, v :- Vec4] :- Vec4
  (->Vec4 (* c (.x v))
          (* c (.y v))
          (* c (.z v))
          (* c (.w v))))

(deftm raster.numeric/* [v :- Vec4, c :- Double] :- Vec4
  (->Vec4 (* (.x v) c)
          (* (.y v) c)
          (* (.z v) c)
          (* (.w v) c)))

(deftm raster.numeric// [v :- Vec4, c :- Double] :- Vec4
  (->Vec4 (/ (.x v) c)
          (/ (.y v) c)
          (/ (.z v) c)
          (/ (.w v) c)))

;; ================================================================
;; Dot product
;; ================================================================

(deftm dot [a :- Vec2, b :- Vec2] :- Double
  (m/fma (.x a) (.x b) (* (.y a) (.y b))))

(deftm dot [a :- Vec3, b :- Vec3] :- Double
  (m/fma (.x a) (.x b)
         (m/fma (.y a) (.y b)
                (* (.z a) (.z b)))))

(deftm dot [a :- Vec4, b :- Vec4] :- Double
  (m/fma (.x a) (.x b)
         (m/fma (.y a) (.y b)
                (m/fma (.z a) (.z b)
                       (* (.w a) (.w b))))))

;; ================================================================
;; Cross product
;; ================================================================

(deftm cross
  "Cross product. Vec2: returns scalar (2D wedge). Vec3: returns Vec3."
  [a :- Vec2, b :- Vec2] :- Double
  (- (* (.x a) (.y b))
     (* (.y a) (.x b))))

(deftm cross [a :- Vec3, b :- Vec3] :- Vec3
  (->Vec3 (- (* (.y a) (.z b))
             (* (.z a) (.y b)))
          (- (* (.z a) (.x b))
             (* (.x a) (.z b)))
          (- (* (.x a) (.y b))
             (* (.y a) (.x b)))))

;; ================================================================
;; Norm and normalize
;; ================================================================

(deftm norm [v :- Vec2] :- Double
  (n/sqrt (m/fma (.x v) (.x v) (* (.y v) (.y v)))))

(deftm norm [v :- Vec3] :- Double
  (n/sqrt (m/fma (.x v) (.x v)
                 (m/fma (.y v) (.y v)
                        (* (.z v) (.z v))))))

(deftm norm [v :- Vec4] :- Double
  (n/sqrt (m/fma (.x v) (.x v)
                 (m/fma (.y v) (.y v)
                        (m/fma (.z v) (.z v)
                               (* (.w v) (.w v)))))))

(deftm normalize [v :- Vec2] :- Vec2
  (let [n (norm v)]
    (->Vec2 (/ (.x v) n)
            (/ (.y v) n))))

(deftm normalize [v :- Vec3] :- Vec3
  (let [n (norm v)]
    (->Vec3 (/ (.x v) n)
            (/ (.y v) n)
            (/ (.z v) n))))

(deftm normalize [v :- Vec4] :- Vec4
  (let [n (norm v)]
    (->Vec4 (/ (.x v) n)
            (/ (.y v) n)
            (/ (.z v) n)
            (/ (.w v) n))))

;; ================================================================
;; Linear interpolation
;; ================================================================

(deftm lerp
  "Linear interpolation between vectors a and b: a*(1-t) + b*t."
  [a :- Vec2, b :- Vec2, t :- Double] :- Vec2
  (let [s (- 1.0 t)]
    (->Vec2 (m/fma s (.x a) (* t (.x b)))
            (m/fma s (.y a) (* t (.y b))))))

(deftm lerp [a :- Vec3, b :- Vec3, t :- Double] :- Vec3
  (let [s (- 1.0 t)]
    (->Vec3 (m/fma s (.x a) (* t (.x b)))
            (m/fma s (.y a) (* t (.y b)))
            (m/fma s (.z a) (* t (.z b))))))

(deftm lerp [a :- Vec4, b :- Vec4, t :- Double] :- Vec4
  (let [s (- 1.0 t)]
    (->Vec4 (m/fma s (.x a) (* t (.x b)))
            (m/fma s (.y a) (* t (.y b)))
            (m/fma s (.z a) (* t (.z b)))
            (m/fma s (.w a) (* t (.w b))))))

;; ================================================================
;; Mat2x2 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Mat2x2, b :- Mat2x2] :- Mat2x2
  (->Mat2x2 (+ (.m00 a) (.m00 b)) (+ (.m01 a) (.m01 b))
            (+ (.m10 a) (.m10 b)) (+ (.m11 a) (.m11 b))))

(deftm raster.numeric/- [a :- Mat2x2, b :- Mat2x2] :- Mat2x2
  (->Mat2x2 (- (.m00 a) (.m00 b)) (- (.m01 a) (.m01 b))
            (- (.m10 a) (.m10 b)) (- (.m11 a) (.m11 b))))

(deftm raster.numeric/* [c :- Double, m :- Mat2x2] :- Mat2x2
  (->Mat2x2 (* c (.m00 m)) (* c (.m01 m))
            (* c (.m10 m)) (* c (.m11 m))))

;; Mat2x2 * Mat2x2 (unrolled 8 muls)
(deftm raster.numeric/* [a :- Mat2x2, b :- Mat2x2] :- Mat2x2
  (->Mat2x2 (m/fma (.m00 a) (.m00 b) (* (.m01 a) (.m10 b)))
            (m/fma (.m00 a) (.m01 b) (* (.m01 a) (.m11 b)))
            (m/fma (.m10 a) (.m00 b) (* (.m11 a) (.m10 b)))
            (m/fma (.m10 a) (.m01 b) (* (.m11 a) (.m11 b)))))

;; Mat2x2 * Vec2
(deftm raster.numeric/* [m :- Mat2x2, v :- Vec2] :- Vec2
  (->Vec2 (m/fma (.m00 m) (.x v) (* (.m01 m) (.y v)))
          (m/fma (.m10 m) (.x v) (* (.m11 m) (.y v)))))

;; ================================================================
;; Mat3x3 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Mat3x3, b :- Mat3x3] :- Mat3x3
  (->Mat3x3 (+ (.m00 a) (.m00 b)) (+ (.m01 a) (.m01 b)) (+ (.m02 a) (.m02 b))
            (+ (.m10 a) (.m10 b)) (+ (.m11 a) (.m11 b)) (+ (.m12 a) (.m12 b))
            (+ (.m20 a) (.m20 b)) (+ (.m21 a) (.m21 b)) (+ (.m22 a) (.m22 b))))

(deftm raster.numeric/- [a :- Mat3x3, b :- Mat3x3] :- Mat3x3
  (->Mat3x3 (- (.m00 a) (.m00 b)) (- (.m01 a) (.m01 b)) (- (.m02 a) (.m02 b))
            (- (.m10 a) (.m10 b)) (- (.m11 a) (.m11 b)) (- (.m12 a) (.m12 b))
            (- (.m20 a) (.m20 b)) (- (.m21 a) (.m21 b)) (- (.m22 a) (.m22 b))))

(deftm raster.numeric/* [c :- Double, m :- Mat3x3] :- Mat3x3
  (->Mat3x3 (* c (.m00 m)) (* c (.m01 m)) (* c (.m02 m))
            (* c (.m10 m)) (* c (.m11 m)) (* c (.m12 m))
            (* c (.m20 m)) (* c (.m21 m)) (* c (.m22 m))))

;; Mat3x3 * Mat3x3 (unrolled 27 muls)
(deftm raster.numeric/* [a :- Mat3x3, b :- Mat3x3] :- Mat3x3
  (->Mat3x3
   (m/fma (.m00 a) (.m00 b) (m/fma (.m01 a) (.m10 b) (* (.m02 a) (.m20 b))))
   (m/fma (.m00 a) (.m01 b) (m/fma (.m01 a) (.m11 b) (* (.m02 a) (.m21 b))))
   (m/fma (.m00 a) (.m02 b) (m/fma (.m01 a) (.m12 b) (* (.m02 a) (.m22 b))))
   (m/fma (.m10 a) (.m00 b) (m/fma (.m11 a) (.m10 b) (* (.m12 a) (.m20 b))))
   (m/fma (.m10 a) (.m01 b) (m/fma (.m11 a) (.m11 b) (* (.m12 a) (.m21 b))))
   (m/fma (.m10 a) (.m02 b) (m/fma (.m11 a) (.m12 b) (* (.m12 a) (.m22 b))))
   (m/fma (.m20 a) (.m00 b) (m/fma (.m21 a) (.m10 b) (* (.m22 a) (.m20 b))))
   (m/fma (.m20 a) (.m01 b) (m/fma (.m21 a) (.m11 b) (* (.m22 a) (.m21 b))))
   (m/fma (.m20 a) (.m02 b) (m/fma (.m21 a) (.m12 b) (* (.m22 a) (.m22 b))))))

;; Mat3x3 * Vec3
(deftm raster.numeric/* [m :- Mat3x3, v :- Vec3] :- Vec3
  (->Vec3 (m/fma (.m00 m) (.x v) (m/fma (.m01 m) (.y v) (* (.m02 m) (.z v))))
          (m/fma (.m10 m) (.x v) (m/fma (.m11 m) (.y v) (* (.m12 m) (.z v))))
          (m/fma (.m20 m) (.x v) (m/fma (.m21 m) (.y v) (* (.m22 m) (.z v))))))

;; ================================================================
;; Mat4x4 arithmetic
;; ================================================================

(deftm raster.numeric/+ [a :- Mat4x4, b :- Mat4x4] :- Mat4x4
  (->Mat4x4
   (+ (.m00 a) (.m00 b)) (+ (.m01 a) (.m01 b))
   (+ (.m02 a) (.m02 b)) (+ (.m03 a) (.m03 b))
   (+ (.m10 a) (.m10 b)) (+ (.m11 a) (.m11 b))
   (+ (.m12 a) (.m12 b)) (+ (.m13 a) (.m13 b))
   (+ (.m20 a) (.m20 b)) (+ (.m21 a) (.m21 b))
   (+ (.m22 a) (.m22 b)) (+ (.m23 a) (.m23 b))
   (+ (.m30 a) (.m30 b)) (+ (.m31 a) (.m31 b))
   (+ (.m32 a) (.m32 b)) (+ (.m33 a) (.m33 b))))

(deftm raster.numeric/- [a :- Mat4x4, b :- Mat4x4] :- Mat4x4
  (->Mat4x4
   (- (.m00 a) (.m00 b)) (- (.m01 a) (.m01 b))
   (- (.m02 a) (.m02 b)) (- (.m03 a) (.m03 b))
   (- (.m10 a) (.m10 b)) (- (.m11 a) (.m11 b))
   (- (.m12 a) (.m12 b)) (- (.m13 a) (.m13 b))
   (- (.m20 a) (.m20 b)) (- (.m21 a) (.m21 b))
   (- (.m22 a) (.m22 b)) (- (.m23 a) (.m23 b))
   (- (.m30 a) (.m30 b)) (- (.m31 a) (.m31 b))
   (- (.m32 a) (.m32 b)) (- (.m33 a) (.m33 b))))

(deftm raster.numeric/* [c :- Double, m :- Mat4x4] :- Mat4x4
  (->Mat4x4
   (* c (.m00 m)) (* c (.m01 m))
   (* c (.m02 m)) (* c (.m03 m))
   (* c (.m10 m)) (* c (.m11 m))
   (* c (.m12 m)) (* c (.m13 m))
   (* c (.m20 m)) (* c (.m21 m))
   (* c (.m22 m)) (* c (.m23 m))
   (* c (.m30 m)) (* c (.m31 m))
   (* c (.m32 m)) (* c (.m33 m))))

;; Mat4x4 * Mat4x4 (unrolled 64 muls)
(deftm raster.numeric/* [a :- Mat4x4, b :- Mat4x4] :- Mat4x4
  (->Mat4x4
   ;; Row 0
   (m/fma (.m00 a) (.m00 b) (m/fma (.m01 a) (.m10 b) (m/fma (.m02 a) (.m20 b) (* (.m03 a) (.m30 b)))))
   (m/fma (.m00 a) (.m01 b) (m/fma (.m01 a) (.m11 b) (m/fma (.m02 a) (.m21 b) (* (.m03 a) (.m31 b)))))
   (m/fma (.m00 a) (.m02 b) (m/fma (.m01 a) (.m12 b) (m/fma (.m02 a) (.m22 b) (* (.m03 a) (.m32 b)))))
   (m/fma (.m00 a) (.m03 b) (m/fma (.m01 a) (.m13 b) (m/fma (.m02 a) (.m23 b) (* (.m03 a) (.m33 b)))))
   ;; Row 1
   (m/fma (.m10 a) (.m00 b) (m/fma (.m11 a) (.m10 b) (m/fma (.m12 a) (.m20 b) (* (.m13 a) (.m30 b)))))
   (m/fma (.m10 a) (.m01 b) (m/fma (.m11 a) (.m11 b) (m/fma (.m12 a) (.m21 b) (* (.m13 a) (.m31 b)))))
   (m/fma (.m10 a) (.m02 b) (m/fma (.m11 a) (.m12 b) (m/fma (.m12 a) (.m22 b) (* (.m13 a) (.m32 b)))))
   (m/fma (.m10 a) (.m03 b) (m/fma (.m11 a) (.m13 b) (m/fma (.m12 a) (.m23 b) (* (.m13 a) (.m33 b)))))
   ;; Row 2
   (m/fma (.m20 a) (.m00 b) (m/fma (.m21 a) (.m10 b) (m/fma (.m22 a) (.m20 b) (* (.m23 a) (.m30 b)))))
   (m/fma (.m20 a) (.m01 b) (m/fma (.m21 a) (.m11 b) (m/fma (.m22 a) (.m21 b) (* (.m23 a) (.m31 b)))))
   (m/fma (.m20 a) (.m02 b) (m/fma (.m21 a) (.m12 b) (m/fma (.m22 a) (.m22 b) (* (.m23 a) (.m32 b)))))
   (m/fma (.m20 a) (.m03 b) (m/fma (.m21 a) (.m13 b) (m/fma (.m22 a) (.m23 b) (* (.m23 a) (.m33 b)))))
   ;; Row 3
   (m/fma (.m30 a) (.m00 b) (m/fma (.m31 a) (.m10 b) (m/fma (.m32 a) (.m20 b) (* (.m33 a) (.m30 b)))))
   (m/fma (.m30 a) (.m01 b) (m/fma (.m31 a) (.m11 b) (m/fma (.m32 a) (.m21 b) (* (.m33 a) (.m31 b)))))
   (m/fma (.m30 a) (.m02 b) (m/fma (.m31 a) (.m12 b) (m/fma (.m32 a) (.m22 b) (* (.m33 a) (.m32 b)))))
   (m/fma (.m30 a) (.m03 b) (m/fma (.m31 a) (.m13 b) (m/fma (.m32 a) (.m23 b) (* (.m33 a) (.m33 b)))))))

;; Mat4x4 * Vec4
(deftm raster.numeric/* [m :- Mat4x4, v :- Vec4] :- Vec4
  (->Vec4
   (m/fma (.m00 m) (.x v) (m/fma (.m01 m) (.y v) (m/fma (.m02 m) (.z v) (* (.m03 m) (.w v)))))
   (m/fma (.m10 m) (.x v) (m/fma (.m11 m) (.y v) (m/fma (.m12 m) (.z v) (* (.m13 m) (.w v)))))
   (m/fma (.m20 m) (.x v) (m/fma (.m21 m) (.y v) (m/fma (.m22 m) (.z v) (* (.m23 m) (.w v)))))
   (m/fma (.m30 m) (.x v) (m/fma (.m31 m) (.y v) (m/fma (.m32 m) (.z v) (* (.m33 m) (.w v)))))))

;; ================================================================
;; Determinant
;; ================================================================

(deftm det
  "Determinant of a fixed-size matrix."
  [m :- Mat2x2] :- Double
  (- (* (.m00 m) (.m11 m))
     (* (.m01 m) (.m10 m))))

(deftm det [m :- Mat3x3] :- Double
  ;; Sarrus' rule / cofactor expansion along first row
  (+
   (* (.m00 m) (- (* (.m11 m) (.m22 m))
                  (* (.m12 m) (.m21 m))))
   (* (- (.m01 m)) (- (* (.m10 m) (.m22 m))
                      (* (.m12 m) (.m20 m))))
   (* (.m02 m) (- (* (.m10 m) (.m21 m))
                  (* (.m11 m) (.m20 m))))))

(deftm det [m :- Mat4x4] :- Double
  ;; Laplace expansion along first row using 2x2 subdeterminants
  (let [s0 (- (* (.m00 m) (.m11 m)) (* (.m10 m) (.m01 m)))
        s1 (- (* (.m00 m) (.m12 m)) (* (.m10 m) (.m02 m)))
        s2 (- (* (.m00 m) (.m13 m)) (* (.m10 m) (.m03 m)))
        s3 (- (* (.m01 m) (.m12 m)) (* (.m11 m) (.m02 m)))
        s4 (- (* (.m01 m) (.m13 m)) (* (.m11 m) (.m03 m)))
        s5 (- (* (.m02 m) (.m13 m)) (* (.m12 m) (.m03 m)))
        c5 (- (* (.m22 m) (.m33 m)) (* (.m32 m) (.m23 m)))
        c4 (- (* (.m21 m) (.m33 m)) (* (.m31 m) (.m23 m)))
        c3 (- (* (.m21 m) (.m32 m)) (* (.m31 m) (.m22 m)))
        c2 (- (* (.m20 m) (.m33 m)) (* (.m30 m) (.m23 m)))
        c1 (- (* (.m20 m) (.m32 m)) (* (.m30 m) (.m22 m)))
        c0 (- (* (.m20 m) (.m31 m)) (* (.m30 m) (.m21 m)))]
    (- (+ (* s0 c5)
          (* s2 c3)
          (* s4 c1))
       (+ (* s1 c4)
          (* s3 c2)
          (* s5 c0)))))

;; ================================================================
;; Inverse
;; ================================================================

(deftm inv
  "Inverse of a fixed-size matrix via cofactor expansion."
  [m :- Mat2x2] :- Mat2x2
  (let [d (det m)
        inv-d (/ 1.0 d)]
    (->Mat2x2 (* inv-d (.m11 m))
              (* (- inv-d) (.m01 m))
              (* (- inv-d) (.m10 m))
              (* inv-d (.m00 m)))))

(deftm inv [m :- Mat3x3] :- Mat3x3
  (let [d (det m)
        inv-d (/ 1.0 d)
        ;; Cofactor matrix (transposed for adjugate)
        c00 (- (* (.m11 m) (.m22 m)) (* (.m12 m) (.m21 m)))
        c01 (- (* (.m02 m) (.m21 m)) (* (.m01 m) (.m22 m)))
        c02 (- (* (.m01 m) (.m12 m)) (* (.m02 m) (.m11 m)))
        c10 (- (* (.m12 m) (.m20 m)) (* (.m10 m) (.m22 m)))
        c11 (- (* (.m00 m) (.m22 m)) (* (.m02 m) (.m20 m)))
        c12 (- (* (.m02 m) (.m10 m)) (* (.m00 m) (.m12 m)))
        c20 (- (* (.m10 m) (.m21 m)) (* (.m11 m) (.m20 m)))
        c21 (- (* (.m01 m) (.m20 m)) (* (.m00 m) (.m21 m)))
        c22 (- (* (.m00 m) (.m11 m)) (* (.m01 m) (.m10 m)))]
    (->Mat3x3 (* inv-d c00) (* inv-d c01) (* inv-d c02)
              (* inv-d c10) (* inv-d c11) (* inv-d c12)
              (* inv-d c20) (* inv-d c21) (* inv-d c22))))

(deftm inv [m :- Mat4x4] :- Mat4x4
  ;; Uses the same 2x2 subdeterminants as det for Laplace expansion
  (let [s0 (- (* (.m00 m) (.m11 m)) (* (.m10 m) (.m01 m)))
        s1 (- (* (.m00 m) (.m12 m)) (* (.m10 m) (.m02 m)))
        s2 (- (* (.m00 m) (.m13 m)) (* (.m10 m) (.m03 m)))
        s3 (- (* (.m01 m) (.m12 m)) (* (.m11 m) (.m02 m)))
        s4 (- (* (.m01 m) (.m13 m)) (* (.m11 m) (.m03 m)))
        s5 (- (* (.m02 m) (.m13 m)) (* (.m12 m) (.m03 m)))
        c5 (- (* (.m22 m) (.m33 m)) (* (.m32 m) (.m23 m)))
        c4 (- (* (.m21 m) (.m33 m)) (* (.m31 m) (.m23 m)))
        c3 (- (* (.m21 m) (.m32 m)) (* (.m31 m) (.m22 m)))
        c2 (- (* (.m20 m) (.m33 m)) (* (.m30 m) (.m23 m)))
        c1 (- (* (.m20 m) (.m32 m)) (* (.m30 m) (.m22 m)))
        c0 (- (* (.m20 m) (.m31 m)) (* (.m30 m) (.m21 m)))
        d  (- (+ (* s0 c5)
                 (* s2 c3)
                 (* s4 c1))
              (+ (* s1 c4)
                 (* s3 c2)
                 (* s5 c0)))
        inv-d (/ 1.0 d)]
    (->Mat4x4
     ;; Row 0
     (* inv-d (+ (- (* (.m11 m) c5) (* (.m12 m) c4)) (* (.m13 m) c3)))
     (* inv-d (- (+ (* (.m02 m) c4) (* (- (.m01 m)) c5)) (* (.m03 m) c3)))
     (* inv-d (+ (- (* (.m31 m) s5) (* (.m32 m) s4)) (* (.m33 m) s3)))
     (* inv-d (- (+ (* (.m22 m) s4) (* (- (.m21 m)) s5)) (* (.m23 m) s3)))
     ;; Row 1
     (* inv-d (+ (- (* (.m12 m) c2) (* (.m10 m) c5)) (* (- (.m13 m)) c1)))
     (* inv-d (+ (- (* (.m00 m) c5) (* (.m02 m) c2)) (* (.m03 m) c1)))
     (* inv-d (+ (- (* (.m32 m) s2) (* (.m30 m) s5)) (* (- (.m33 m)) s1)))
     (* inv-d (+ (- (* (.m20 m) s5) (* (.m22 m) s2)) (* (.m23 m) s1)))
     ;; Row 2
     (* inv-d (+ (- (* (.m10 m) c4) (* (.m11 m) c2)) (* (.m13 m) c0)))
     (* inv-d (+ (- (* (.m01 m) c2) (* (.m00 m) c4)) (* (- (.m03 m)) c0)))
     (* inv-d (+ (- (* (.m30 m) s4) (* (.m31 m) s2)) (* (.m33 m) s0)))
     (* inv-d (+ (- (* (.m21 m) s2) (* (.m20 m) s4)) (* (- (.m23 m)) s0)))
     ;; Row 3
     (* inv-d (+ (- (* (.m11 m) c1) (* (.m10 m) c3)) (* (- (.m12 m)) c0)))
     (* inv-d (+ (- (* (.m00 m) c3) (* (.m01 m) c1)) (* (.m02 m) c0)))
     (* inv-d (+ (- (* (.m31 m) s1) (* (.m30 m) s3)) (* (- (.m32 m)) s0)))
     (* inv-d (+ (- (* (.m20 m) s3) (* (.m21 m) s1)) (* (.m22 m) s0))))))

;; ================================================================
;; Transpose
;; ================================================================

(deftm transpose
  "Transpose of a fixed-size matrix."
  [m :- Mat2x2] :- Mat2x2
  (->Mat2x2 (.m00 m) (.m10 m)
            (.m01 m) (.m11 m)))

(deftm transpose [m :- Mat3x3] :- Mat3x3
  (->Mat3x3 (.m00 m) (.m10 m) (.m20 m)
            (.m01 m) (.m11 m) (.m21 m)
            (.m02 m) (.m12 m) (.m22 m)))

(deftm transpose [m :- Mat4x4] :- Mat4x4
  (->Mat4x4 (.m00 m) (.m10 m) (.m20 m) (.m30 m)
            (.m01 m) (.m11 m) (.m21 m) (.m31 m)
            (.m02 m) (.m12 m) (.m22 m) (.m32 m)
            (.m03 m) (.m13 m) (.m23 m) (.m33 m)))

;; ================================================================
;; Identity matrix factory
;; ================================================================

(defn identity-mat
  "Create an identity matrix of the given size (2, 3, or 4)."
  ^{:tag Object}
  [^long n]
  (case n
    2 (->Mat2x2 1.0 0.0 0.0 1.0)
    3 (->Mat3x3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0)
    4 (->Mat4x4 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
                0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0)
    (throw (ex-info "identity-mat supports sizes 2, 3, 4" {:n n}))))

;; ================================================================
;; Array conversions (column-major for OpenGL compatibility)
;; ================================================================

(deftm ->double-array
  "Convert a fixed-size vector or matrix to a double array (column-major for matrices)."
  [v :- Vec2] :- (Array double)
  (double-array [(.x v) (.y v)]))

(deftm ->double-array [v :- Vec3] :- (Array double)
  (double-array [(.x v) (.y v) (.z v)]))

(deftm ->double-array [v :- Vec4] :- (Array double)
  (double-array [(.x v) (.y v) (.z v) (.w v)]))

(deftm ->double-array [m :- Mat2x2] :- (Array double)
  (double-array [(.m00 m) (.m10 m)   ;; column 0
                 (.m01 m) (.m11 m)])) ;; column 1

(deftm ->double-array [m :- Mat3x3] :- (Array double)
  (double-array [(.m00 m) (.m10 m) (.m20 m)   ;; column 0
                 (.m01 m) (.m11 m) (.m21 m)   ;; column 1
                 (.m02 m) (.m12 m) (.m22 m)])) ;; column 2

(deftm ->double-array [m :- Mat4x4] :- (Array double)
  (double-array [(.m00 m) (.m10 m) (.m20 m) (.m30 m)   ;; column 0
                 (.m01 m) (.m11 m) (.m21 m) (.m31 m)   ;; column 1
                 (.m02 m) (.m12 m) (.m22 m) (.m32 m)   ;; column 2
                 (.m03 m) (.m13 m) (.m23 m) (.m33 m)])) ;; column 3

;; ================================================================
;; From-array factories (column-major input)
;; ================================================================

(defn vec2-from-array
  "Create Vec2 from double-array."
  [a]
  (->Vec2 (aget a 0) (aget a 1)))

(defn vec3-from-array
  "Create Vec3 from double-array."
  [a]
  (->Vec3 (aget a 0) (aget a 1) (aget a 2)))

(defn vec4-from-array
  "Create Vec4 from double-array."
  [a]
  (->Vec4 (aget a 0) (aget a 1) (aget a 2) (aget a 3)))

(defn mat2x2-from-array
  "Create Mat2x2 from column-major double-array."
  [a]
  (->Mat2x2 (aget a 0) (aget a 2)
            (aget a 1) (aget a 3)))

(defn mat3x3-from-array
  "Create Mat3x3 from column-major double-array."
  [a]
  (->Mat3x3 (aget a 0) (aget a 3) (aget a 6)
            (aget a 1) (aget a 4) (aget a 7)
            (aget a 2) (aget a 5) (aget a 8)))

(defn mat4x4-from-array
  "Create Mat4x4 from column-major double-array."
  [a]
  (->Mat4x4 (aget a 0) (aget a 4) (aget a 8)  (aget a 12)
            (aget a 1) (aget a 5) (aget a 9)  (aget a 13)
            (aget a 2) (aget a 6) (aget a 10) (aget a 14)
            (aget a 3) (aget a 7) (aget a 11) (aget a 15)))

;; ================================================================
;; Dense array linear algebra (LAPACK-backed)
;; All operations take flat double[] in row-major order.
;; ================================================================

(deftm solve
  "Solve linear system Ax=b. A[n,n] row-major, b[n]. Returns x[n].
  Uses LU factorization via LAPACK dgesv."
  [A :- (Array double) b :- (Array double) n :- Long] :- (Array double)
  (let [nn (* n n)
        A-copy (double-array nn)
        _ (acopy! A 0 A-copy 0 nn)
        x (double-array n)
        _ (acopy! b 0 x 0 n)
        ipiv (int-array n)
        info (lapack/dgesv! A-copy x ipiv n 1)]
    (when-not (== info 0)
      (throw (ex-info "solve failed" {:info info})))
    x))

(deftm solve-multiple
  "Solve AX=B for multiple right-hand sides. A[n,n], B[n,nrhs] row-major.
  Returns X[n,nrhs]."
  [A :- (Array double) B :- (Array double) n :- Long nrhs :- Long]
  :- (Array double)
  (let [nn (* n n)
        nb (* n nrhs)
        A-copy (double-array nn)
        _ (acopy! A 0 A-copy 0 nn)
        X (double-array nb)
        _ (acopy! B 0 X 0 nb)
        ipiv (int-array n)
        info (lapack/dgesv! A-copy X ipiv n nrhs)]
    (when-not (== info 0)
      (throw (ex-info "solve-multiple failed" {:info info})))
    X))

(deftm cholesky
  "Cholesky factorization of SPD matrix. A[n,n] row-major symmetric.
  Returns lower triangular L[n,n] such that A = L*L^T."
  [A :- (Array double) n :- Long] :- (Array double)
  (let [nn (* n n)
        L (double-array nn)
        _ (acopy! A 0 L 0 nn)
        info (lapack/dpotrf! L n)]
    (when-not (== info 0)
      (throw (ex-info "cholesky failed (not positive definite?)" {:info info})))
    ;; dpotrf returns L in column-major lower triangle.
    ;; Transpose to row-major and zero upper triangle.
    (dotimes [i n]
      (loop [j (inc i)]
        (when (< j n)
          ;; Copy col-major lower L[j,i] to row-major L[j,i], zero L[i,j]
          (let [ij (+ (* i n) j)
                ji (+ (* j n) i)
                lower-val (aget L ij)]  ;; col-major L[j,i] is at row-major [i,j]
            (aset L ji lower-val)  ;; place in row-major lower position
            (aset L ij 0.0))       ;; zero upper triangle
          (recur (inc j)))))
    L))

(deftm cholesky-solve
  "Solve Ax=b using Cholesky factor L. L[n,n] lower triangular (row-major), b[n].
  Returns x[n]."
  [L :- (Array double) b :- (Array double) n :- Long] :- (Array double)
  (let [nn (* n n)
        ;; Row-major lower triangular = column-major upper triangular.
        ;; Copy L (no transpose needed) and call dpotrs with uplo='U'.
        L-copy (double-array nn)
        _ (acopy! L 0 L-copy 0 nn)
        x (double-array n)
        _ (acopy! b 0 x 0 n)
        info (lapack/dpotrs-upper! L-copy x n 1)]
    (when-not (== info 0)
      (throw (ex-info "cholesky-solve failed" {:info info})))
    x))

(deftm lu
  "LU factorization with partial pivoting. A[n,n] row-major.
  Returns [LU-combined, pivots] as Object array.
  LU-combined[n,n] contains both L (lower, unit diagonal) and U (upper)."
  [A :- (Array double) n :- Long] :- (Array Object)
  (let [nn (* n n)
        LU (double-array nn)
        _ (acopy! A 0 LU 0 nn)
        ipiv (int-array n)
        info (lapack/dgetrf! LU ipiv n n)]
    (when-not (== info 0)
      (throw (ex-info "lu factorization failed" {:info info})))
    (object-array [LU ipiv])))

(deftm lstsq
  "Least squares solve min||Ax-b||. A[m,n] row-major, b[m].
  Returns x[n] (first n elements of solution).
  m >= n (overdetermined) or m < n (minimum norm)."
  [A :- (Array double) b :- (Array double)
   m :- Long n :- Long] :- (Array double)
  (let [mn (* m n)
        A-copy (double-array mn)
        _ (acopy! A 0 A-copy 0 mn)
        ldb (max m n)
        B (double-array ldb)
        _ (acopy! b 0 B 0 m)
        info (lapack/dgels! A-copy B m n 1)]
    (when-not (== info 0)
      (throw (ex-info "lstsq failed" {:info info})))
    (let [x (double-array n)]
      (acopy! B 0 x 0 n)
      x)))

(deftm array-inv
  "Matrix inverse via LU. A[n,n] row-major. Returns A^{-1}[n,n]."
  [A :- (Array double) n :- Long] :- (Array double)
  (let [nn (* n n)
        result (double-array nn)
        _ (acopy! A 0 result 0 nn)
        ipiv (int-array n)
        info1 (lapack/dgetrf! result ipiv n n)]
    (when-not (== info1 0)
      (throw (ex-info "inv: LU factorization failed" {:info info1})))
    (let [info2 (lapack/dgetri! result ipiv n)]
      (when-not (== info2 0)
        (throw (ex-info "inv: matrix inversion failed" {:info info2})))
      result)))

(deftm array-det
  "Determinant via LU factorization. A[n,n] row-major."
  [A :- (Array double) n :- Long] :- Double
  (let [nn (* n n)
        LU (double-array nn)
        _ (acopy! A 0 LU 0 nn)
        ipiv (int-array n)
        info (lapack/dgetrf! LU ipiv n n)]
    (when-not (== info 0)
      (throw (ex-info "det: LU factorization failed" {:info info})))
    ;; det = product of diagonal * sign from pivoting
    (loop [i 0 d 1.0 swaps 0]
      (if (>= i n)
        (if (even? swaps) d (- d))
        (let [diag-val (aget LU (+ (* i n) i))
              ;; LAPACK uses 1-based indexing for ipiv
              piv (aget ipiv i)
              swap? (not= piv (inc (int i)))]
          (recur (inc i)
                 (* d diag-val)
                 (if swap? (inc swaps) swaps)))))))

(deftm cond-number
  "Condition number via SVD (ratio of largest to smallest singular value).
  A[m,n] row-major."
  [A :- (Array double) m :- Long n :- Long] :- Double
  (let [k (min m n)
        mn (* m n)
        A-copy (double-array mn)
        _ (acopy! A 0 A-copy 0 mn)
        S (double-array k)
        info (lapack/dgesdd-values! A-copy S m n)]
    (when-not (== info 0)
      (throw (ex-info "cond-number: SVD failed" {:info info})))
    (let [s-max (aget S 0)
          s-min (aget S (dec k))]
      (if (< (n/abs s-min) 1e-300)
        n/pos-inf
        (/ s-max s-min)))))

(deftm matrix-norm
  "Matrix norm. ord: 1 = max column sum, 2 = spectral (SVD), -1 = Frobenius.
  A[m,n] row-major."
  [A :- (Array double) m :- Long n :- Long ord :- Long] :- Double
  (let [mn (* m n)]
    (case (int ord)
      ;; Frobenius
      -1 (let [sum (loop [i 0 s 0.0]
                     (if (>= i mn) s
                         (let [v (aget A i)]
                           (recur (inc i) (+ s (* v v))))))]
           (n/sqrt sum))
      ;; 1-norm: max column sum of absolute values
      1 (loop [j 0 mx 0.0]
          (if (>= j n) mx
              (let [col-sum (loop [i 0 s 0.0]
                              (if (>= i m) s
                                  (recur (inc i) (+ s (n/abs (aget A (+ (* i n) j)))))))]
                (recur (inc j) (n/max mx col-sum)))))
      ;; 2-norm: largest singular value
      2 (let [k (min m n)
              A-copy (double-array mn)
              _ (acopy! A 0 A-copy 0 mn)
              S (double-array k)
              info (lapack/dgesdd-values! A-copy S m n)]
          (when-not (== info 0)
            (throw (ex-info "matrix-norm: SVD failed" {:info info})))
          (aget S 0))
      (throw (ex-info "matrix-norm: ord must be -1 (Frobenius), 1, or 2"
                      {:ord ord})))))

;; ================================================================
;; Backward helpers for linear algebra (template-based AD codegen)
;; ================================================================

(deftm solve-backward
  "AD backward for solve: returns [dA, db] given upstream dy."
  [dy :- (Array double) A :- (Array double)
   x :- (Array double) n :- Long]
  :- (Array Object)
  (let [At (double-array (* n n))
        _ (dotimes [i n]
            (dotimes [j n]
              (aset At (clojure.core/+ (* i n) j) (aget A (clojure.core/+ (* j n) i)))))
        db (solve At dy n)
        dA (double-array (* n n))]
    (dotimes [i n]
      (dotimes [j n]
        (aset dA (clojure.core/+ (* i n) j) (- (* (aget db i) (aget x j))))))
    (object-array [dA db])))

(deftm array-det-backward
  "AD backward for det: dA = dy * det(A) * (A^{-1})^T."
  [dy :- Double det-val :- Double
   A :- (Array double) n :- Long]
  :- (Array double)
  (let [Ainv (array-inv A n)
        dA (double-array (* n n))]
    (dotimes [i n]
      (dotimes [j n]
        (aset dA (clojure.core/+ (* i n) j)
              (* dy det-val (aget Ainv (clojure.core/+ (* j n) i))))))
    dA))

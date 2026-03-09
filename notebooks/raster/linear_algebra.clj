;; # Linear Algebra with Raster

;; authors: Christian Weilbach

;; Raster provides linear algebra at two scales:
;; - **Fixed-size** types (Vec2-4, Mat2x2-4x4) for geometry and physics
;; - **Dense array** operations for numerical computing (backed by LAPACK/OpenBLAS)
;;
;; Both integrate with `raster.numeric` — the same `+`, `-`, `*` operators
;; work on vectors, matrices, and scalars.

;; ## Setup

(ns raster.linear-algebra
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.linalg.core :as la]
            [scicloj.kindly.v4.kind :as kind]))

;; ## 1. Fixed-Size Vectors
;;
;; `Vec2`, `Vec3`, `Vec4` are Valhalla value types with fully unrolled
;; arithmetic. No heap allocation, no indirection — just registers.

(def a (la/->Vec3 1.0 2.0 3.0))
(def b (la/->Vec3 4.0 5.0 6.0))

;; Arithmetic via `raster.numeric`:

{:sum  (n/+ a b)
 :diff (n/- a b)
 :scaled (n/* 2.0 a)
 :dot  (la/dot a b)
 :cross (la/cross a b)
 :norm (la/norm a)
 :normalized (la/normalize a)}

;; ## 2. Fixed-Size Matrices
;;
;; `Mat2x2`, `Mat3x3`, `Mat4x4` with determinant, inverse, and
;; matrix-vector multiply:

(def m (la/->Mat3x3 1.0 2.0 3.0
                     0.0 1.0 4.0
                     5.0 6.0 0.0))

{:det (la/det m)
 :inverse (la/inv m)
 :m*v (n/* m a)}

;; ## 3. Dense Linear Systems
;;
;; For larger systems, Raster wraps LAPACK via Panama FFI (OpenBLAS).
;; All operations work on flat `double[]` arrays in row-major order.

;; Solve Ax = b:

(let [A (double-array [2.0 1.0
                        1.0 3.0])
      b (double-array [5.0 7.0])]
  {:solution (vec (la/solve A b 2))
   :expected "[1.6, 1.8]"})

;; ## 4. Matrix Decompositions
;;
;; LU, Cholesky, QR, SVD, and eigenvalue decompositions:

(let [A (double-array [4.0 2.0
                        2.0 3.0])]
  {:cholesky (vec (la/cholesky A 2))
   :cond (la/cond-number A 2 2)})

;; ## 5. Iterative Solvers
;;
;; For large sparse systems, Raster provides Krylov methods:
;; - **CG** (Conjugate Gradient) — symmetric positive definite
;; - **GMRES** — general non-symmetric
;; - **BiCGSTAB** — non-symmetric, lower memory
;; - **Lanczos** — eigenvalues of symmetric matrices
;;
;; These work with any operator that implements matrix-vector product,
;; not just dense arrays.

;; ## 6. Sparse Vectors
;;
;; Compressed sorted format for high-dimensional sparse data:

(require '[raster.linalg.sparse :as sparse])

(let [sv (sparse/sparse-vector (int-array [0 3 7])
                                (double-array [1.0 2.0 3.0])
                                10)]
  {:nnz (sparse/nnz sv)
   :dense (vec (sparse/to-dense sv))})

;; ## Available Operations
;;
;; | Operation | Fixed-size | Dense | Sparse |
;; |---|---|---|---|
;; | Add/subtract | Vec2-4, Mat2-4 | `dense-add!` | `sparse-add` |
;; | Multiply | Mat*Vec, Mat*Mat | BLAS `dgemv`/`dgemm` | — |
;; | Solve | `inv` (small) | LAPACK `solve` | CG/GMRES |
;; | Decompose | det, inv | LU, Cholesky, QR, SVD | Lanczos |
;; | Norm | `norm` | `matrix-norm` | `sparse-norm` |

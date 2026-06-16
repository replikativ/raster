(ns raster.linalg.iterative
  "Iterative linear solvers and eigensolvers.

  Provides Krylov subspace methods for solving Ax = b and eigenvalue
  problems without forming or storing the full matrix. All algorithms
  accept a matvec-fn that computes y = A*x in-place, enabling use
  with sparse (CSR), dense, or matrix-free operators.

  Solvers:
    - cg       Conjugate Gradient (SPD systems)
    - pcg      Preconditioned Conjugate Gradient
    - gmres    Restarted GMRES (general nonsymmetric)
    - bicgstab BiCGSTAB (nonsymmetric, no restart)

  Eigensolvers:
    - lanczos  Lanczos with full reorthogonalization (symmetric)

  Preconditioners:
    - jacobi-precond  Diagonal (Jacobi) preconditioner

  Usage:
    (require '[raster.linalg.iterative :refer [cg gmres bicgstab lanczos pcg]])
    (require '[raster.core :refer [ftm]])

    ;; Define matvec: y = A*x (writes into y, returns y)
    (def mv (ftm [x :- (Array double) y :- (Array double)] :- (Array double)
              (aset y 0 (+ (* 2.0 (aget x 0)) (* 1.0 (aget x 1))))
              (aset y 1 (+ (* 1.0 (aget x 0)) (* 3.0 (aget x 1))))
              y))

    (cg mv (double-array [1.0 2.0]) (double-array [0.0 0.0]) 2 1e-10 100)"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm broadcast reduce!]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.linalg.eigen :as eigen]
            [raster.math :as m]
            [raster.numeric :as n]))

;; ================================================================
;; Vector helper operations
;; ================================================================

(deftm vec-dot (All [T] [a :- (Array T) b :- (Array T) n :- Long] :- Double
                    (reduce! [s 0.0] [a b] (+ s (* a b)))))

;; In-place vector operations for iterative solvers (CG, PCG, GMRES)
;; These mutate their target arrays — iterative algorithms need this.
(deftm vec-axpy! (All [T] [alpha :- Double, a :- (Array T), b :- (Array T), n :- Long] :- (Array T)
                      (dotimes [i n]
                        (aset b i (+ (aget b i) (* alpha (aget a i)))))
                      b))

(deftm vec-copy! (All [T] [src :- (Array T), dst :- (Array T), n :- Long] :- (Array T)
                      (acopy! src 0 dst 0 n)
                      dst))

(deftm vec-scale! (All [T] [a :- (Array T), alpha :- Double, n :- Long] :- (Array T)
                       (dotimes [i n]
                         (aset a i (* alpha (aget a i))))
                       a))

(deftm vec-norm2 (All [T] [a :- (Array T) n :- Long] :- Double
                      (n/sqrt (vec-dot a a n))))

;; ================================================================
;; Conjugate Gradient (for symmetric positive definite systems)
;; ================================================================

(deftm cg (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
                    b :- (Array T) x0 :- (Array T)
                    n :- Long tol :- Double maxiter :- Long] :- (Array Object)
               (let [x   (aclone x0)
                     r   (n/similar b)
                     p   (n/similar b)
                     ap  (n/similar b)
                     _   (matvec-fn x ap)               ;; ap = A*x0
                     _   (vec-copy! b r n)              ;; r = b
                     _   (vec-axpy! -1.0 ap r n)        ;; r = b - A*x0
                     _   (vec-copy! r p n)              ;; p = r
                     bnorm (vec-norm2 b n)
                     tol-abs (if (> bnorm 0.0) (* tol bnorm) tol)]
                 (loop [rr   (vec-dot r r n)
                        iter (int 0)]
                   (let [rnorm (n/sqrt rr)]
                     (if (or (>= iter maxiter)
                             (<= rnorm tol-abs))
                       (object-array [x (Double/valueOf rnorm) (Long/valueOf (long iter))])
                       (let [_     (matvec-fn p ap)              ;; ap = A*p
                             pap   (vec-dot p ap n)
                             alpha (/ rr pap)
                             _     (vec-axpy! alpha p x n)       ;; x += alpha*p
                             _     (vec-axpy! (- alpha) ap r n)  ;; r -= alpha*ap
                             rr-new (vec-dot r r n)
                             beta  (/ rr-new rr)
                ;; p = r + beta*p
                             _     (vec-scale! p beta n)
                             _     (vec-axpy! 1.0 r p n)]
                         (recur rr-new (unchecked-add-int iter 1)))))))))

;; ================================================================
;; Jacobi preconditioner
;; ================================================================

(deftm jacobi-precond (All [T] [diag :- (Array T) r :- (Array T)
                                z :- (Array T) n :- Long] :- (Array T)
                           (dotimes [i n]
                             (aset z i (/ (aget r i) (aget diag i))))
                           z))

;; ================================================================
;; Preconditioned Conjugate Gradient
;; ================================================================

(deftm pcg (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
                     precond-fn :- (Fn [(Array T) (Array T)] (Array T))
                     b :- (Array T) x0 :- (Array T)
                     n :- Long tol :- Double maxiter :- Long] :- (Array Object)
                (let [x   (aclone x0)
                      r   (n/similar b)
                      z   (n/similar b)
                      p   (n/similar b)
                      ap  (n/similar b)
                      _   (matvec-fn x ap)
                      _   (vec-copy! b r n)
                      _   (vec-axpy! -1.0 ap r n)       ;; r = b - A*x0
                      _   (precond-fn r z)              ;; z = M^{-1} r
                      _   (vec-copy! z p n)             ;; p = z
                      bnorm (vec-norm2 b n)
                      tol-abs (if (> bnorm 0.0) (* tol bnorm) tol)]
                  (loop [rz   (vec-dot r z n)
                         iter (int 0)]
                    (let [rnorm (vec-norm2 r n)]
                      (if (or (>= iter maxiter)
                              (<= rnorm tol-abs))
                        (object-array [x (Double/valueOf rnorm) (Long/valueOf (long iter))])
                        (let [_      (matvec-fn p ap)
                              pap    (vec-dot p ap n)
                              alpha  (/ rz pap)
                              _      (vec-axpy! alpha p x n)
                              _      (vec-axpy! (- alpha) ap r n)
                              _      (precond-fn r z)
                              rz-new (vec-dot r z n)
                              beta   (/ rz-new rz)
                              _      (vec-scale! p beta n)
                              _      (vec-axpy! 1.0 z p n)]
                          (recur rz-new (unchecked-add-int iter 1)))))))))

;; ================================================================
;; GMRES (Restarted, with Modified Gram-Schmidt + Givens rotations)
;; ================================================================

(deftm gmres (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
                       b :- (Array T) x0 :- (Array T)
                       n :- Long tol :- Double maxiter :- Long restart :- Long] :- (Array Object)
                  (let [x      (aclone x0)
                        r      (n/similar b)
                        w      (n/similar b)
                        bnorm  (vec-norm2 b n)
                        tol-abs (if (> bnorm 0.0) (* tol bnorm) tol)
        ;; Hessenberg matrix H: (restart+1) x restart stored column-major
        ;; H[i,j] stored at index j*(restart+1)+i
                        m      (int restart)
                        h-size (int (* (+ m 1) m))
        ;; Basis vectors V: (restart+1) vectors of length n
        ;; V[j] stored at offset j*n
                        v-size (int (* (+ m 1) n))]
                    (loop [outer-iter (int 0)
                           total-iter (int 0)]
      ;; Compute initial residual r = b - A*x
                      (let [_     (matvec-fn x w)
                            _     (vec-copy! b r n)
                            _     (vec-axpy! -1.0 w r n)
                            beta  (vec-norm2 r n)]
                        (if (or (>= total-iter maxiter)
                                (<= beta tol-abs))
                          (object-array [x (Double/valueOf beta) (Long/valueOf (long total-iter))])
          ;; Arnoldi iteration
                          (let [H  (double-array h-size)
                                V  (double-array v-size)
                ;; g = right-hand side for least-squares (length m+1)
                                g  (double-array (+ m 1))
                ;; Givens rotation parameters
                                cs (double-array m)
                                sn (double-array m)
                ;; V[0] = r / beta
                                _  (vec-copy! r V n)
                                _  (let [inv-beta (/ 1.0 beta)]
                                     (dotimes [i n]
                                       (aset V i (* inv-beta (aget V i)))))
                                _  (aset g 0 beta)]
            ;; Inner GMRES loop
                            (loop [j     (int 0)
                                   t-iter total-iter]
                              (if (or (>= j m) (>= t-iter maxiter))
                ;; Solve upper triangular H*y = g for the first j entries
                                (let [y (double-array j)]
                  ;; Back substitution
                                  (loop [i (unchecked-subtract-int j 1)]
                                    (when (>= i 0)
                                      (let [sum (loop [k (unchecked-add-int i 1)
                                                       s 0.0]
                                                  (if (>= k j)
                                                    s
                                                    (recur (unchecked-add-int k 1)
                                                           (+ s (* (aget H (+ (* k (+ m 1)) i))
                                                                   (aget y k))))))]
                                        (aset y i (/ (- (aget g i) sum)
                                                     (aget H (+ (* i (+ m 1)) i))))
                                        (recur (unchecked-subtract-int i 1)))))
                  ;; x = x + V * y
                                  (dotimes [i j]
                                    (let [vi-off (int (* i n))
                                          yi     (aget y i)]
                                      (dotimes [k n]
                                        (aset x k (+ (aget x k) (* yi (aget V (+ vi-off k))))))))
                  ;; Continue outer restart loop
                                  (let [rnorm (n/abs (aget g j))]
                                    (if (or (>= t-iter maxiter)
                                            (<= rnorm tol-abs))
                                      (object-array [x (Double/valueOf rnorm) (Long/valueOf (long t-iter))])
                                      (recur (unchecked-add-int outer-iter 1) t-iter))))
                ;; Arnoldi step j
                                (let [vj-off (int (* j n))
                      ;; w = A * V[j]
                                      vj-tmp (double-array n)
                                      _      (acopy! V vj-off vj-tmp 0 n)
                                      _      (matvec-fn vj-tmp w)]
                  ;; Modified Gram-Schmidt
                                  (dotimes [i (+ j 1)]
                                    (let [vi-off (int (* i n))
                          ;; h_{i,j} = dot(V[i], w)
                                          hij (loop [k (int 0)
                                                     s 0.0]
                                                (if (>= k n)
                                                  s
                                                  (recur (unchecked-add-int k 1)
                                                         (+ s (* (aget V (+ vi-off k))
                                                                 (aget w k))))))]
                                      (aset H (+ (* j (+ m 1)) i) hij)
                      ;; w = w - h_{i,j} * V[i]
                                      (dotimes [k n]
                                        (aset w k (- (aget w k)
                                                     (* hij (aget V (+ vi-off k))))))))
                  ;; h_{j+1,j} = ||w||
                                  (let [h-jp1-j (vec-norm2 w n)
                                        h-idx   (int (+ (* j (+ m 1)) (+ j 1)))]
                                    (aset H h-idx h-jp1-j)
                    ;; V[j+1] = w / h_{j+1,j}
                                    (let [vj1-off (int (* (+ j 1) n))
                                          inv-h   (if (> h-jp1-j 1e-30) (/ 1.0 h-jp1-j) 0.0)]
                                      (dotimes [k n]
                                        (aset V (+ vj1-off k) (* inv-h (aget w k)))))
                    ;; Apply previous Givens rotations to column j of H
                                    (dotimes [i j]
                                      (let [h-idx-i  (int (+ (* j (+ m 1)) i))
                                            h-idx-i1 (int (+ (* j (+ m 1)) (+ i 1)))
                                            hi  (aget H h-idx-i)
                                            hi1 (aget H h-idx-i1)
                                            c   (aget cs i)
                                            s   (aget sn i)]
                                        (aset H h-idx-i  (+ (* c hi) (* s hi1)))
                                        (aset H h-idx-i1 (+ (* (- s) hi) (* c hi1)))))
                    ;; Compute new Givens rotation for entries (j, j+1)
                                    (let [hjj  (aget H (+ (* j (+ m 1)) j))
                                          hjj1 (aget H (+ (* j (+ m 1)) (+ j 1)))
                                          denom (n/sqrt (+ (* hjj hjj) (* hjj1 hjj1)))
                                          c    (/ hjj denom)
                                          s    (/ hjj1 denom)]
                                      (aset cs j c)
                                      (aset sn j s)
                      ;; Apply to H
                                      (aset H (+ (* j (+ m 1)) j) (+ (* c hjj) (* s hjj1)))
                                      (aset H (+ (* j (+ m 1)) (+ j 1)) 0.0)
                      ;; Apply to g
                                      (let [gj  (aget g j)
                                            gj1 (aget g (+ j 1))]
                                        (aset g j       (+ (* c gj) (* s gj1)))
                                        (aset g (+ j 1) (+ (* (- s) gj) (* c gj1)))))
                    ;; Check convergence
                                    (let [res (n/abs (aget g (+ j 1)))]
                                      (if (<= res tol-abs)
                        ;; Converged: solve and update x (j+1 columns)
                                        (let [jj (+ j 1)
                                              y  (double-array jj)]
                                          (loop [i (unchecked-subtract-int jj 1)]
                                            (when (>= i 0)
                                              (let [sum (loop [k (unchecked-add-int i 1)
                                                               s 0.0]
                                                          (if (>= k jj)
                                                            s
                                                            (recur (unchecked-add-int k 1)
                                                                   (+ s (* (aget H (+ (* k (+ m 1)) i))
                                                                           (aget y k))))))]
                                                (aset y i (/ (- (aget g i) sum)
                                                             (aget H (+ (* i (+ m 1)) i))))
                                                (recur (unchecked-subtract-int i 1)))))
                                          (dotimes [i jj]
                                            (let [vi-off (int (* i n))
                                                  yi     (aget y i)]
                                              (dotimes [k n]
                                                (aset x k (+ (aget x k) (* yi (aget V (+ vi-off k))))))))
                                          (object-array [x (Double/valueOf res) (Long/valueOf (long (unchecked-add-int t-iter 1)))]))
                        ;; Continue Arnoldi
                                        (recur (unchecked-add-int j 1)
                                               (unchecked-add-int t-iter 1)))))))))))))))

;; ================================================================
;; BiCGSTAB (Bi-Conjugate Gradient Stabilized)
;; ================================================================

(deftm bicgstab (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
                          b :- (Array T) x0 :- (Array T)
                          n :- Long tol :- Double maxiter :- Long] :- (Array Object)
                     (let [x    (aclone x0)
                           r    (n/similar b)
                           r0   (n/similar b)     ;; shadow residual (fixed)
                           p    (n/similar b)
                           v    (n/similar b)     ;; A*p
                           s    (n/similar b)
                           t    (n/similar b)     ;; A*s
                           _    (matvec-fn x v)
                           _    (vec-copy! b r n)
                           _    (vec-axpy! -1.0 v r n)     ;; r = b - A*x0
                           _    (vec-copy! r r0 n)          ;; r0 = r (shadow)
                           _    (vec-copy! r p n)           ;; p = r
                           bnorm (vec-norm2 b n)
                           tol-abs (if (> bnorm 0.0) (* tol bnorm) tol)]
                       (loop [rho   (vec-dot r0 r n)
                              iter  (int 0)]
                         (let [rnorm (vec-norm2 r n)]
                           (if (or (>= iter maxiter)
                                   (<= rnorm tol-abs))
                             (object-array [x (Double/valueOf rnorm) (Long/valueOf (long iter))])
          ;; v = A*p
                             (let [_ (matvec-fn p v)
                                   alpha (/ rho (vec-dot r0 v n))
                ;; s = r - alpha*v
                                   _ (vec-copy! r s n)
                                   _ (vec-axpy! (- alpha) v s n)
                                   snorm (vec-norm2 s n)]
                               (if (<= snorm tol-abs)
              ;; s is small enough, accept x + alpha*p
                                 (let [_ (vec-axpy! alpha p x n)]
                                   (object-array [x (Double/valueOf snorm) (Long/valueOf (long (+ iter 1)))]))
              ;; t = A*s
                                 (let [_ (matvec-fn s t)
                    ;; omega = dot(t,s) / dot(t,t)
                                       omega (/ (vec-dot t s n) (vec-dot t t n))
                    ;; x = x + alpha*p + omega*s
                                       _ (vec-axpy! alpha p x n)
                                       _ (vec-axpy! omega s x n)
                    ;; r = s - omega*t
                                       _ (vec-copy! s r n)
                                       _ (vec-axpy! (- omega) t r n)
                                       rho-new (vec-dot r0 r n)
                                       beta    (* (/ rho-new rho) (/ alpha omega))
                    ;; p = r + beta*(p - omega*v)
                                       _ (vec-axpy! (- omega) v p n)
                                       _ (vec-scale! p beta n)
                                       _ (vec-axpy! 1.0 r p n)]
                                   (if (< (n/abs omega) 1e-30)
                  ;; omega breakdown
                                     (let [rnorm-final (vec-norm2 r n)]
                                       (object-array [x (Double/valueOf rnorm-final) (Long/valueOf (long (+ iter 1)))]))
                                     (recur rho-new (unchecked-add-int iter 1))))))))))))

;; ================================================================
;; Lanczos Eigensolver (symmetric, full reorthogonalization)
;; ================================================================

;; --- Lanczos helpers (split for JVM 64KB method limit + JIT friendliness) ---

(deftm lanczos-tridiag
  "Lanczos iteration with full reorthogonalization.
  Builds tridiagonal matrix (alpha, beta-arr) and orthonormal basis V.
  Returns actual number of Lanczos steps taken."
  (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
            alpha :- (Array double) beta-arr :- (Array double)
            V :- (Array double) w :- (Array T)
            n :- Long m :- Long tol :- Double] :- Long
       (let [;; Initialize v1 = random unit vector
             _ (let [v0 (double-array n)]
                 (dotimes [i n]
                   (aset v0 i (- (* 2.0 (Math/random)) 1.0)))
                 (let [nrm (vec-norm2 v0 n)
                       inv (/ 1.0 nrm)]
                   (dotimes [i n]
                     (aset V i (* inv (aget v0 i))))))]
         (loop [j (int 0)]
           (if (>= j m)
             (long m)
             (let [vj-off (int (* j n))
              ;; w = A * V[j]
                   vj-tmp (n/similar w)
                   _      (acopy! V vj-off vj-tmp 0 n)
                   _      (matvec-fn vj-tmp w)
              ;; alpha[j] = dot(V[j], w)
                   aj     (loop [i (int 0) s 0.0]
                            (if (>= i n)
                              s
                              (recur (unchecked-add-int i 1)
                                     (+ s (* (aget V (+ vj-off i)) (aget w i))))))
                   _      (aset alpha j aj)
              ;; w = w - alpha[j]*V[j]
                   _      (dotimes [i n]
                            (aset w i (- (aget w i)
                                         (* aj (aget V (+ vj-off i))))))
              ;; w = w - beta[j]*V[j-1]  (if j > 0)
                   _      (when (> j 0)
                            (let [bj (aget beta-arr j)
                                  vjm1-off (int (* (- j 1) n))]
                              (dotimes [i n]
                                (aset w i (- (aget w i)
                                             (* bj (aget V (+ vjm1-off i))))))))
              ;; Full reorthogonalization against all previous vectors
                   _      (dotimes [p (+ j 1)]
                            (let [vp-off (int (* p n))
                                  coeff  (loop [i (int 0) s 0.0]
                                           (if (>= i n)
                                             s
                                             (recur (unchecked-add-int i 1)
                                                    (+ s (* (aget V (+ vp-off i))
                                                            (aget w i))))))]
                              (dotimes [i n]
                                (aset w i (- (aget w i)
                                             (* coeff (aget V (+ vp-off i))))))))
              ;; beta[j+1] = ||w||
                   bnext  (vec-norm2 w n)]
               (if (or (< bnext (* tol (n/abs aj)))
                       (>= (+ j 1) m))
                 (long (+ j 1))
                 (let [vj1-off (int (* (+ j 1) n))
                       inv-b   (/ 1.0 bnext)]
                   (aset beta-arr (+ j 1) bnext)
                   (dotimes [i n]
                     (aset V (+ vj1-off i) (* inv-b (aget w i))))
                   (recur (unchecked-add-int j 1))))))))))

(deftm tridiag-qr-eigen
  "Implicit symmetric QR algorithm (Wilkinson shift) on a tridiagonal matrix.
  d = diagonal (eigenvalues on output), e = off-diagonal, Z = eigenvector matrix
  (column-major, am x am, initialized to identity). Mutates d, e, Z in place."
  [d :- (Array double) e :- (Array double) Z :- (Array double)
   am :- Long tol :- Double] :- Long
  (loop [qr-iter (int 0)]
    (if (>= qr-iter (* 30 am))
      (long qr-iter)
      (let [converged
            (loop [sweep (int 0)
                   all-converged true]
              (if (>= sweep (- am 1))
                all-converged
                (if (< (n/abs (aget e sweep))
                       (* tol (+ (n/abs (aget d sweep))
                                 (n/abs (aget d (+ sweep 1))))))
                  (recur (unchecked-add-int sweep 1) all-converged)
                  (recur (unchecked-add-int sweep 1) false))))]
        (if converged
          (long qr-iter)
          (let [dm1   (aget d (- am 1))
                dm2   (aget d (- am 2))
                em1   (aget e (- am 2))
                delta (/ (- dm2 dm1) 2.0)
                sign-d (if (>= delta 0.0) 1.0 -1.0)
                shift (- dm1 (/ (* em1 em1)
                                (+ delta (* sign-d (n/sqrt (+ (* delta delta)
                                                              (* em1 em1)))))))
                _ (loop [i (int 0)
                         g-val (- (aget d 0) shift)
                         s-val (aget e 0)]
                    (when (< i (- am 1))
                      (let [r  (n/sqrt (+ (* g-val g-val) (* s-val s-val)))
                            c  (/ g-val r)
                            sn (/ s-val r)
                            di   (aget d i)
                            di1  (aget d (+ i 1))
                            ei   (if (< i (- am 2)) (aget e (+ i 1)) 0.0)
                            d-new-i   (+ (* c c di) (* 2.0 c sn (aget e i)) (* sn sn di1))
                            d-new-i1  (+ (* sn sn di) (* -2.0 c sn (aget e i)) (* c c di1))
                            e-new-i   (* c sn (- di1 di))
                            e-new-i   (+ e-new-i (* (- (* c c) (* sn sn)) (aget e i)))]
                        (aset d i d-new-i)
                        (aset d (+ i 1) d-new-i1)
                        (aset e i e-new-i)
                        (dotimes [row am]
                          (let [zi  (aget Z (+ (* i am) row))
                                zi1 (aget Z (+ (* (+ i 1) am) row))]
                            (aset Z (+ (* i am) row)       (+ (* c zi) (* sn zi1)))
                            (aset Z (+ (* (+ i 1) am) row) (+ (* (- sn) zi) (* c zi1)))))
                        (let [g-next (aget e i)
                              s-next (if (< i (- am 2)) (* sn ei) 0.0)]
                          (when (< i (- am 2))
                            (aset e (+ i 1) (* c ei)))
                          (recur (unchecked-add-int i 1) g-next s-next)))))]
            (recur (unchecked-add-int qr-iter 1))))))))

(deftm lanczos-extract-eigenvectors
  "Sort eigenvalues by magnitude (largest first), select top k, and
  back-transform eigenvectors from tridiagonal basis Z through Lanczos basis V.
  Returns [eigenvalues eigenvectors] as (Array Object)."
  [d :- (Array double) Z :- (Array double) V :- (Array double)
   n :- Long am :- Long k :- Long] :- (Array Object)
  (let [kk       (int (min (long k) (long am)))
        indices  (int-array am)
        _        (dotimes [i am] (aset indices i (int i)))
        ;; Selection sort for top k by magnitude
        _        (dotimes [i kk]
                   (loop [j    (unchecked-add-int i 1)
                          best i]
                     (if (>= j am)
                       (when (not (== best i))
                         (let [tmp-d (aget d i)
                               tmp-idx (aget indices i)]
                           (aset d i (aget d best))
                           (aset indices i (aget indices best))
                           (aset d best tmp-d)
                           (aset indices best (int tmp-idx))
                           (dotimes [row am]
                             (let [tmp-z (aget Z (+ (* i am) row))]
                               (aset Z (+ (* i am) row) (aget Z (+ (* best am) row)))
                               (aset Z (+ (* best am) row) tmp-z)))))
                       (if (> (n/abs (aget d j)) (n/abs (aget d best)))
                         (recur (unchecked-add-int j 1) j)
                         (recur (unchecked-add-int j 1) best)))))
        eigenvalues (double-array kk)
        _           (acopy! d 0 eigenvalues 0 kk)
        eigenvectors (object-array kk)]
    (dotimes [i kk]
      (let [ev (double-array n)
            col-idx (int i)]
        (dotimes [j am]
          (let [z-ji  (aget Z (+ (* col-idx am) j))
                vj-off (int (* j n))]
            (dotimes [p n]
              (aset ev p (+ (aget ev p) (* z-ji (aget V (+ vj-off p))))))))
        (let [nrm (vec-norm2 ev n)]
          (when (> nrm 1e-30)
            (let [inv (/ 1.0 nrm)]
              (dotimes [p n]
                (aset ev p (* inv (aget ev p)))))))
        (aset eigenvectors i ev)))
    (object-array [eigenvalues eigenvectors])))

;; --- Lanczos orchestrator ---

(deftm lanczos (All [T] [matvec-fn :- (Fn [(Array T) (Array T)] (Array T))
                         n :- Long k :- Long tol :- Double maxiter :- Long] :- (Array Object)
                    (let [m        (int (min (long maxiter) n))
                          alpha    (double-array m)
                          beta-arr (double-array m)
                          V        (double-array (* m n))
                          w        (double-array n)
                          actual-m (lanczos-tridiag matvec-fn alpha beta-arr V w n m tol)
                          am       (int (long actual-m))
                          d        (double-array am)
                          e        (double-array am)
                          _        (acopy! alpha 0 d 0 am)
                          _        (dotimes [i am]
                                     (if (> i 0)
                                       (aset e (- i 1) (aget beta-arr i))
                                       nil))
                          ;; Solve the small am×am tridiagonal eigenproblem via
                          ;; LAPACK eigh (dsyevd). am ≤ maxiter ≪ n, so this dense
                          ;; solve is cheap, and it avoids the hand-rolled
                          ;; tridiag-qr-eigen (which produced NaN — 0/0 Givens with
                          ;; no deflation). eigh's row-major eigenvectors match the
                          ;; Z layout lanczos-extract expects.
                          T        (double-array (* am am))
                          _        (dotimes [i am]
                                     (aset T (+ (* i am) i) (aget d i))
                                     (when (< i (- am 1))
                                       (aset T (+ (* i am) (+ i 1)) (aget e i))
                                       (aset T (+ (* (+ i 1) am) i) (aget e i))))
                          tres     (eigen/eigh T am)
                          tevals   (aget tres 0)
                          tevecs   (aget tres 1)]
                      (lanczos-extract-eigenvectors tevals tevecs V n am k))))

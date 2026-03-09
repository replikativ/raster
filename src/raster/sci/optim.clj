(ns raster.sci.optim
  "Optimization algorithms following Julia's Optim.jl patterns.

  Provides derivative-free (Nelder-Mead) and gradient-based (L-BFGS,
  Gradient Descent, Newton) optimization with line search.

  All core algorithms use deftm with typed function parameters
  via (Fn [(Array double)] Double) for unboxed evaluation.

  Usage:
    (require '[raster.sci.optim :refer [optimize]])
    (require '[raster.core :refer [ftm]])
    (optimize (ftm [x :- (Array double)] :- Double
               (let [a (- (aget x 0) 1.0)
                     b (- (aget x 1) (* (aget x 0) (aget x 0)))]
                 (+ (* a a) (* 100.0 b b))))
              (double-array [0.0 0.0])
              :algorithm :lbfgs)"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm defvalue reduce!]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.par]))

;; ================================================================
;; Array helpers — parametric over element type via (All [T])
;; Auto-specializes for double[], float[], long[], etc. on first call.
;; ================================================================

(deftm dot
  "Compute the dot product of two arrays."
  (All [T] [a :- (Array T), b :- (Array T)] :- Double
       (reduce! [acc 0.0] [a b] (+ acc (* a b)))))

(deftm vec-norm
  "Compute the Euclidean norm of an array."
  (All [T] [a :- (Array T)] :- Double
       (n/sqrt (dot a a))))

(deftm vec-copy
  "Return a fresh copy of array a."
  (All [T] [a :- (Array T)] :- (Array T)
       (let [n (alength a)
             out (n/similar a)]
         (acopy! a 0 out 0 n)
         out)))

(deftm vec-negate!
  "Write the negation of a into out, element-wise."
  (All [T] [out :- (Array T), a :- (Array T)] :- (Array T)
       (dotimes [i (alength out)]
         (aset out i (- (aget a i))))
       out))

(deftm vec-step!
  "In-place x = x + alpha * d."
  (All [T] [x :- (Array T), alpha :- Double, d :- (Array T)]
       :- (Array T)
       (dotimes [i (alength x)]
         (aset x i (+ (aget x i) (* alpha (aget d i)))))
       x))

;; ================================================================
;; Finite-difference gradient
;; ================================================================

(deftm finite-gradient!
  "Compute gradient of f at x via central finite differences, writing into out."
  (All [T] [f :- (Fn [(Array T)] Double),
            x :- (Array T),
            out :- (Array T)] :- (Array T)
       (let [n   (alength x)
             eps 1e-8
             tmp (vec-copy x)]
         (dotimes [i n]
           (let [xi (aget x i)]
             (aset tmp i (+ xi eps))
             (let [fp (f tmp)]
               (aset tmp i (- xi eps))
               (let [fm (f tmp)]
                 (aset out i (/ (- fp fm) (* 2.0 eps)))
                 (aset tmp i xi)))))
         out)))

;; ================================================================
;; Backtracking line search
;; ================================================================

(deftm backtracking-linesearch
  "Armijo backtracking line search. Returns step size alpha."
  (All [T] [f :- (Fn [(Array T)] Double),
            x :- (Array T), d :- (Array T),
            grad-dot-d :- Double, alpha0 :- Double,
            c :- Double, rho :- Double, max-iter :- Long] :- Double
       (let [fx (f x)
             n  (alength x)
             x-new (n/similar x)]
         (loop [alpha alpha0
                iter (int 0)]
           (if (>= iter max-iter)
             alpha
             (do
               (dotimes [i n]
                 (aset x-new i (+ (aget x i) (* alpha (aget d i)))))
               (let [fx-new (f x-new)]
                 (if (<= fx-new (+ fx (* c alpha grad-dot-d)))
                   alpha
                   (recur (* rho alpha) (unchecked-add-int iter 1))))))))))

;; ================================================================
;; Nelder-Mead (derivative-free simplex method)
;; ================================================================

(deftm nm-shrink!
  "Nelder-Mead shrink: contract all vertices toward the best vertex."
  (All [T] [simplex :- (Array Object), fvals :- (Array double),
            order :- (Array int), best-idx :- Long,
            n :- Long, n+1 :- Long,
            f :- (Fn [(Array T)] Double)]
       (let [xb (aget simplex best-idx)]
         (dotimes [j n+1]
           (let [idx (aget order j)]
             (when (not= idx best-idx)
               (let [v (aget simplex idx)]
                 (dotimes [k n]
                   (aset v k (+ (aget xb k)
                                (* 0.5 (- (aget v k) (aget xb k))))))
                 (aset fvals idx (f v)))))))))

(deftm nm-sort-order!
  "Sort simplex vertex indices by function value (insertion sort)."
  [order :- (Array int), fvals :- (Array double), n+1 :- Long]
  (dotimes [i n+1] (aset order i (int i)))
  ;; Insertion sort
  (dotimes [ii (dec n+1)]
    (let [i (inc ii)]
      (loop [j i]
        (when (and (> j 0)
                   (> (aget fvals (aget order (dec j)))
                      (aget fvals (aget order j))))
          (let [tmp (aget order j)]
            (aset order j (aget order (dec j)))
            (aset order (dec j) tmp)
            (recur (dec j))))))))

(deftm nm-centroid!
  "Compute the centroid of all simplex vertices except the worst."
  (All [T] [centroid :- (Array T),
            simplex :- (Array Object), order :- (Array int),
            worst-idx :- Long, n :- Long, n+1 :- Long]
       (dotimes [k n] (aset centroid k 0.0))
       (dotimes [j n+1]
         (let [idx (aget order j)]
           (when (not= idx worst-idx)
             (let [v (aget simplex idx)]
               (dotimes [k n]
                 (aset centroid k (+ (aget centroid k) (aget v k))))))))
       (dotimes [k n]
         (aset centroid k (/ (aget centroid k) (double n))))
       centroid))

(deftm nm-reflect!
  "Nelder-Mead reflection: reflect worst vertex through centroid."
  (All [T] [out :- (Array T), centroid :- (Array T),
            xw :- (Array T), coeff :- Double, n :- Long]
       (dotimes [k n]
         (aset out k (+ (aget centroid k)
                        (* coeff (- (aget centroid k) (aget xw k))))))
       out))

(deftm nm-expand!
  "Nelder-Mead expansion: extend reflected point further from centroid."
  (All [T] [out :- (Array T), centroid :- (Array T),
            xr :- (Array T), gamma :- Double, n :- Long]
       (dotimes [k n]
         (aset out k (+ (aget centroid k)
                        (* gamma (- (aget xr k) (aget centroid k))))))
       out))

(deftm nm-contract-outside!
  "Nelder-Mead outside contraction: move reflected point toward centroid."
  (All [T] [out :- (Array T), centroid :- (Array T),
            xr :- (Array T), beta :- Double, n :- Long]
       (dotimes [k n]
         (aset out k (+ (aget centroid k)
                        (* beta (- (aget xr k) (aget centroid k))))))
       out))

(deftm nm-contract-inside!
  "Nelder-Mead inside contraction: move worst vertex toward centroid."
  (All [T] [out :- (Array T), centroid :- (Array T),
            xw :- (Array T), beta :- Double, n :- Long]
       (dotimes [k n]
         (aset out k (+ (aget centroid k)
                        (* beta (- (aget xw k) (aget centroid k))))))
       out))

(deftm nelder-mead
  "Nelder-Mead derivative-free simplex optimization."
  (All [T] [f :- (Fn [(Array T)] Double),
            x0 :- (Array T),
            maxiter :- Long, tol :- Double]
       (let [n     (alength x0)
             n+1   (inc n)
             simplex (object-array n+1)
             fvals   (double-array n+1)
             order   (int-array n+1)
             centroid (n/similar x0)
             xr      (n/similar x0)
             xc      (n/similar x0)
             xe      (n/similar x0)
             _ (aset simplex 0 (vec-copy x0))
             _ (aset fvals 0 (f x0))]
         (dotimes [i n]
           (let [v  (vec-copy x0)
                 xi (aget x0 i)
                 delta (if (== xi 0.0) 0.00025 (* 0.05 xi))]
             (aset v i (+ xi delta))
             (aset simplex (inc i) v)
             (aset fvals (inc i) (f v))))
         (loop [iter (int 0)]
           (nm-sort-order! order fvals n+1)
           (let [best-idx  (long (aget order 0))
                 worst-idx (long (aget order n))
                 f-best    (aget fvals best-idx)
                 f-worst   (aget fvals worst-idx)
                 f-second  (aget fvals (aget order (dec n)))
                 spread    (- f-worst f-best)]
             (if (or (>= iter maxiter) (< spread tol))
               {:minimizer (vec (aget simplex best-idx))
                :minimum   f-best
                :iterations (long iter)
                :converged? (< spread tol)}
               (let [xw (aget simplex worst-idx)]
                 (nm-centroid! centroid simplex order worst-idx n n+1)
                 (nm-reflect! xr centroid xw 1.0 n)
                 (let [fr (f xr)]
                   (cond
                ;; Accept reflection
                     (and (<= f-best fr) (< fr f-second))
                     (do (acopy! xr 0 (aget simplex worst-idx) 0 n)
                         (aset fvals worst-idx fr)
                         (recur (unchecked-add-int iter 1)))

                ;; Expansion
                     (< fr f-best)
                     (do (nm-expand! xe centroid xr 2.0 n)
                         (let [fe (f xe)]
                           (if (< fe fr)
                             (do (acopy! xe 0 (aget simplex worst-idx) 0 n)
                                 (aset fvals worst-idx fe))
                             (do (acopy! xr 0 (aget simplex worst-idx) 0 n)
                                 (aset fvals worst-idx fr))))
                         (recur (unchecked-add-int iter 1)))

                ;; Contraction
                     :else
                     (if (< fr f-worst)
                  ;; Outside contraction
                       (do (nm-contract-outside! xc centroid xr 0.5 n)
                           (let [fc (f xc)]
                             (if (< fc fr)
                               (do (acopy! xc 0 (aget simplex worst-idx) 0 n)
                                   (aset fvals worst-idx fc))
                               (nm-shrink! simplex fvals order best-idx n n+1 f)))
                           (recur (unchecked-add-int iter 1)))
                  ;; Inside contraction
                       (do (nm-contract-inside! xc centroid xw 0.5 n)
                           (let [fc (f xc)]
                             (if (< fc f-worst)
                               (do (acopy! xc 0 (aget simplex worst-idx) 0 n)
                                   (aset fvals worst-idx fc))
                               (nm-shrink! simplex fvals order best-idx n n+1 f)))
                           (recur (unchecked-add-int iter 1)))))))))))))

;; ================================================================
;; Gradient Descent with backtracking line search
;; ================================================================

(deftm gradient-descent
  "Gradient descent with backtracking line search."
  (All [T] [f :- (Fn [(Array T)] Double),
            grad-fn :- (Fn [(Array T) (Array T)]),
            x0 :- (Array T),
            maxiter :- Long, tol :- Double]
       (let [n (alength x0)
             x (vec-copy x0)
             g (n/similar x0)
             d (n/similar x0)]
         (loop [iter (int 0)]
           (grad-fn x g)
           (let [gnorm (vec-norm g)]
             (if (or (>= iter maxiter) (< gnorm tol))
               {:minimizer (vec x)
                :minimum   (f x)
                :iterations (long iter)
                :converged? (< gnorm tol)}
               (do
                 (vec-negate! d g)
                 (let [gd    (dot g d)
                       alpha (backtracking-linesearch f x d gd 1.0 1e-4 0.5 30)]
                   (vec-step! x alpha d)
                   (recur (unchecked-add-int iter 1))))))))))

;; ================================================================
;; L-BFGS (Limited-memory BFGS)
;; ================================================================

(deftm lbfgs
  "L-BFGS quasi-Newton optimization with m history vectors."
  (All [T] [f :- (Fn [(Array T)] Double),
            grad-fn :- (Fn [(Array T) (Array T)]),
            x0 :- (Array T),
            maxiter :- Long, tol :- Double, m :- Long]
       (let [n      (alength x0)
             x      (vec-copy x0)
             g      (n/similar x0)
             g-prev (n/similar x0)
             s-store (object-array m)
             y-store (object-array m)
             rho-store (double-array m)]
         (grad-fn x g)
         (loop [iter (int 0)
                k    (int 0)]
           (let [gnorm (vec-norm g)]
             (if (or (>= iter maxiter) (< gnorm tol))
               {:minimizer (vec x)
                :minimum   (f x)
                :iterations (long iter)
                :converged? (< gnorm tol)}
          ;; Two-loop recursion
               (let [q     (vec-copy g)
                     bound (min (int k) (int m))
                     alpha-buf (double-array bound)]
            ;; First loop (backward)
                 (loop [i (dec bound)]
                   (when (>= i 0)
                     (let [idx  (int (clojure.core/mod (+ (- k 1) (- bound) 1 i) m))
                           si   (aget s-store idx)
                           yi   (aget y-store idx)
                           rhoi (aget rho-store idx)
                           ai   (* rhoi (dot si q))]
                       (aset alpha-buf i ai)
                       (dotimes [j n]
                         (aset q j (- (aget q j) (* ai (aget yi j)))))
                       (recur (dec i)))))
            ;; Scale by gamma
                 (when (> bound 0)
                   (let [last-idx (int (clojure.core/mod (dec k) m))
                         s-last (aget s-store last-idx)
                         y-last (aget y-store last-idx)
                         gamma  (/ (dot s-last y-last) (dot y-last y-last))]
                     (dotimes [j n] (aset q j (* gamma (aget q j))))))
            ;; Second loop (forward)
                 (loop [i (int 0)]
                   (when (< i bound)
                     (let [idx  (int (clojure.core/mod (+ (- k 1) (- bound) 1 i) m))
                           si   (aget s-store idx)
                           yi   (aget y-store idx)
                           rhoi (aget rho-store idx)
                           beta (* rhoi (dot yi q))]
                       (dotimes [j n]
                         (aset q j (+ (aget q j) (* (- (aget alpha-buf i) beta) (aget si j)))))
                       (recur (unchecked-add-int i 1)))))
            ;; d = -H*g, line search, update
                 (let [d (n/similar x0)
                       _ (dotimes [j n] (aset d j (- (aget q j))))
                       gd    (dot g d)
                       alpha (backtracking-linesearch f x d gd 1.0 1e-4 0.9 30)
                       _     (acopy! g 0 g-prev 0 n)
                       s-new (n/similar x0)]
                   (dotimes [j n]
                     (let [step (* alpha (aget d j))]
                       (aset s-new j step)
                       (aset x j (+ (aget x j) step))))
                   (grad-fn x g)
                   (let [y-new (n/similar x0)]
                     (dotimes [j n]
                       (aset y-new j (- (aget g j) (aget g-prev j))))
                     (let [sy (dot s-new y-new)
                           stored? (> sy 1e-20)]
                       (when stored?
                         (let [idx (int (clojure.core/mod k m))]
                           (aset s-store idx s-new)
                           (aset y-store idx y-new)
                           (aset rho-store idx (/ 1.0 sy))))
                       (recur (unchecked-add-int iter 1)
                              (if stored?
                                (unchecked-add-int k 1)
                                k))))))))))))

;; ================================================================
;; Newton's method (full Hessian via finite differences)
;; ================================================================

(deftm finite-hessian!
  "Compute Hessian of f at x via finite differences, writing into H (row-major)."
  (All [T] [f :- (Fn [(Array T)] Double),
            x :- (Array T),
            H :- (Array Object)] :- (Array Object)
       (let [n   (alength x)
             eps 1e-5
             g1  (n/similar x)
             g2  (n/similar x)
             tmp (vec-copy x)]
         (dotimes [i n]
           (let [xi (aget x i)]
             (aset tmp i (+ xi eps))
             (finite-gradient! f tmp g1)
             (aset tmp i (- xi eps))
             (finite-gradient! f tmp g2)
             (aset tmp i xi)
             (let [row (aget H i)]
               (dotimes [j n]
                 (aset row j (/ (- (aget g1 j) (aget g2 j)) (* 2.0 eps)))))))
         H)))

(deftm solve-linear
  "Solve H*d = -g via Gaussian elimination with partial pivoting."
  (All [T] [H :- (Array Object), g :- (Array T)] :- (Array T)
       (let [n  (alength g)
        ;; Augmented matrix [H | -g]
             A  (object-array n)]
    ;; Build augmented matrix
         (dotimes [i n]
           (let [row (double-array (inc n))
                 h-row (aget H i)]
             (dotimes [j n] (aset row j (aget h-row j)))
             (aset row n (- (aget g i)))
             (aset A i row)))
    ;; Forward elimination with partial pivoting
         (dotimes [k n]
           (loop [max-row k, max-val (n/abs (aget (aget A k) k)), i (inc k)]
             (if (>= i n)
               (when (not= max-row k)
                 (let [tmp (aget A k)]
                   (aset A k (aget A max-row))
                   (aset A max-row tmp)))
               (let [v (n/abs (aget (aget A i) k))]
                 (if (> v max-val)
                   (recur i v (inc i))
                   (recur max-row max-val (inc i))))))
           (let [akk (aget (aget A k) k)]
             (when (> (n/abs akk) 1e-15)
               (loop [i (inc k)]
                 (when (< i n)
                   (let [factor (/ (aget (aget A i) k) akk)]
                     (loop [j k]
                       (when (<= j n)
                         (aset (aget A i) j
                               (- (aget (aget A i) j)
                                  (* factor (aget (aget A k) j))))
                         (recur (inc j)))))
                   (recur (inc i)))))))
    ;; Back substitution
         (let [d (n/similar g)]
           (loop [i (dec n)]
             (when (>= i 0)
               (let [sum (loop [j (inc i) acc 0.0]
                           (if (>= j n) acc
                               (recur (inc j) (+ acc (* (aget (aget A i) j) (aget d j))))))]
                 (aset d i (/ (- (aget (aget A i) n) sum)
                              (aget (aget A i) i))))
               (recur (dec i))))
           d))))

(deftm newton-optimize
  "Newton's method with finite-difference Hessian and line search."
  (All [T] [f :- (Fn [(Array T)] Double),
            grad-fn :- (Fn [(Array T) (Array T)]),
            x0 :- (Array T),
            maxiter :- Long, tol :- Double]
       (let [n (alength x0)
             x (vec-copy x0)
             g (n/similar x0)
             H (object-array n)
             _ (dotimes [i n] (aset H i (n/similar x0)))]
         (loop [iter (int 0)]
           (grad-fn x g)
           (let [gnorm (vec-norm g)]
             (if (or (>= iter maxiter) (< gnorm tol))
               {:minimizer (vec x)
                :minimum   (f x)
                :iterations (long iter)
                :converged? (< gnorm tol)}
               (do
                 (finite-hessian! f x H)
                 (let [d  (solve-linear H g)
                       gd (dot g d)
                  ;; Ensure descent direction
                       d  (if (>= gd 0.0)
                            (let [nd (n/similar x0)]
                              (vec-negate! nd g)
                              nd)
                            d)
                       gd (if (>= gd 0.0) (- (dot g g)) gd)
                       alpha (backtracking-linesearch f x d gd 1.0 1e-4 0.5 30)]
                   (vec-step! x alpha d)
                   (recur (unchecked-add-int iter 1))))))))))

;; ================================================================
;; Unified API
;; ================================================================

(deftm optimize
  "Unified optimization entry point. Dispatches on :algorithm in opts map."
  [f :- Object, x0 :- Object, opts :- Object]
  (let [algorithm (get opts :algorithm :nelder-mead)
        gradient  (get opts :gradient nil)
        maxiter   (long (get opts :maxiter 1000))
        tol       (double (get opts :tol 1e-8))
        m         (long (get opts :m 10))
        x0-arr    (if (instance? (Class/forName "[D") x0)
                    x0
                    (double-array (map double x0)))
        grad-fn   (or gradient
                      (ftm [x :- (Array double), out :- (Array double)]
                           (finite-gradient! f x out)))]
    (case algorithm
      :nelder-mead      (nelder-mead f x0-arr maxiter tol)
      :lbfgs            (lbfgs f grad-fn x0-arr maxiter tol m)
      :gradient-descent (gradient-descent f grad-fn x0-arr maxiter tol)
      :newton           (newton-optimize f grad-fn x0-arr maxiter tol)
      (throw (ex-info (str "Unknown algorithm: " algorithm)
                      {:algorithm algorithm})))))

(deftm optimize [f :- Object, x0 :- Object]
  (optimize f x0 {}))

;; ================================================================
;; Adam optimizer (element-wise, in-place mutation via dotimes/aset)
;; ================================================================

(deftm adam-update-m!
  "Update Adam first moment estimate in-place: m = beta1*m + (1-beta1)*grads."
  (All [T] [m :- (Array T), grads :- (Array T),
            beta1 :- Double] :- (Array T)
       (dotimes [i (alength m)]
         (aset m i (+ (* beta1 (aget m i)) (* (- 1.0 beta1) (aget grads i)))))
       m))

(deftm adam-update-v!
  "Update Adam second moment estimate in-place: v = beta2*v + (1-beta2)*grads^2."
  (All [T] [v :- (Array T), grads :- (Array T),
            beta2 :- Double] :- (Array T)
       (dotimes [i (alength v)]
         (aset v i (+ (* beta2 (aget v i)) (* (- 1.0 beta2) (* (aget grads i) (aget grads i))))))
       v))

(deftm adam-step!
  "Apply one bias-corrected Adam parameter update in-place."
  (All [T] [params :- (Array T), m :- (Array T),
            v :- (Array T), lr :- Double,
            beta1-t :- Double, beta2-t :- Double] :- (Array T)
       (let [lr-corrected (/ lr (- 1.0 beta2-t))]
         (dotimes [i (alength params)]
           (aset params i (- (aget params i)
                             (* lr-corrected (/ (/ (aget m i) (- 1.0 beta1-t))
                                                (+ (n/sqrt (aget v i)) 1e-8))))))
         params)))

(defvalue AdamState [m :- (Array double), v :- (Array double), t :- Long, beta1-t :- Double, beta2-t :- Double])

(defn make-adam-state
  "Create initial Adam optimizer state for n parameters."
  [n]
  (->AdamState (double-array n) (double-array n) 1 1.0 1.0))

(deftm adam-update!
  "One full Adam step: update moments, apply bias-corrected update, return new state."
  (All [T] [params :- (Array T), grads :- (Array T),
            state :- AdamState, lr :- Double, beta1 :- Double,
            beta2 :- Double] :- AdamState
       (let [m (.m state)
             v (.v state)
             t (.t state)
             beta1-t (.beta1-t state)
             beta2-t (.beta2-t state)
             new-beta1-t (* beta1-t beta1)
             new-beta2-t (* beta2-t beta2)]
         (adam-update-m! m grads beta1)
         (adam-update-v! v grads beta2)
         (adam-step! params m v lr new-beta1-t new-beta2-t)
         (->AdamState m v (inc (long t)) new-beta1-t new-beta2-t))))

(deftm adam-update!
  (All [T] [params :- (Array T), grads :- (Array T),
            state :- AdamState] :- AdamState
       (adam-update! params grads state 0.001 0.9 0.999)))

;; ================================================================
;; Levenberg-Marquardt (nonlinear least squares)
;; ================================================================

(deftm compute-jacobian!
  "Compute m-by-n Jacobian of residual function f via forward finite differences."
  (All [T]
       [f :- (Fn [(Array T) (Array T)] (Array T)),
        x :- (Array T), J :- (Array T),
        m :- Long, n :- Long] :- (Array T)
       (let [eps 1e-7
             r0 (raster.arrays/alloc-like x m)
             r1 (raster.arrays/alloc-like x m)
             tmp (vec-copy x)]
         (f x r0)
         (dotimes [j n]
           (let [xj (aget x j)]
             (aset tmp j (+ xj eps))
             (f tmp r1)
             (aset tmp j xj)
             (dotimes [i m]
               (aset J (+ (* i n) j) (/ (- (aget r1 i) (aget r0 i)) eps)))))
         J)))

(deftm levenberg-marquardt
  "Levenberg-Marquardt nonlinear least squares solver."
  (All [T]
       [f :- (Fn [(Array T) (Array T)] (Array T)),
        x0 :- (Array T), m :- Long, n :- Long,
        tol :- Double, maxiter :- Long]
       (let [x (vec-copy x0)
             r (raster.arrays/alloc-like x0 m)
             J (raster.arrays/alloc-like x0 (* m n))
             JtJ (double-array (* n n))
             Jtr (raster.arrays/alloc-like x0 n)
             dx (raster.arrays/alloc-like x0 n)
             x-trial (raster.arrays/alloc-like x0 n)
             r-trial (raster.arrays/alloc-like x0 m)]
         (f x r)
         (loop [iter (int 0)
                lambda 1e-3]
           (let [cost (loop [i 0 s 0.0]
                        (if (>= i m) s (recur (inc i) (+ s (* (aget r i) (aget r i))))))
                 _ (compute-jacobian! f x J m n)
            ;; J^T * J and J^T * r
                 _ (dotimes [i n]
                     (dotimes [j n]
                       (let [s (loop [k 0 acc 0.0]
                                 (if (>= k m) acc
                                     (recur (inc k) (+ acc (* (aget J (+ (* k n) i))
                                                              (aget J (+ (* k n) j)))))))]
                         (aset JtJ (+ (* i n) j) s)))
                     (let [s (loop [k 0 acc 0.0]
                               (if (>= k m) acc
                                   (recur (inc k) (+ acc (* (aget J (+ (* k n) i))
                                                            (aget r k))))))]
                       (aset Jtr i s)))
            ;; Add lambda to diagonal (damping)
                 _ (dotimes [i n]
                     (aset JtJ (+ (* i n) i) (+ (aget JtJ (+ (* i n) i)) lambda)))
            ;; Solve (J^T J + lambda I) dx = -J^T r via Gaussian elimination
                 H-obj (object-array n)
                 _ (dotimes [i n]
                     (let [row (double-array n)]
                       (acopy! JtJ (* i n) row 0 n)
                       (aset H-obj i row)))
            ;; solve-linear solves H*d = -g, so pass Jtr (it will negate)
                 dx-result (solve-linear H-obj Jtr)]
             (acopy! dx-result 0 dx 0 n)
             (let [dx-norm (vec-norm dx)]
               (if (or (>= iter maxiter) (< dx-norm tol))
                 {:minimizer (vec x)
                  :minimum   (* 0.5 cost)
                  :iterations (long iter)
                  :converged? (< dx-norm tol)}
            ;; Trial step
                 (do
                   (dotimes [i n] (aset x-trial i (+ (aget x i) (aget dx i))))
                   (f x-trial r-trial)
                   (let [trial-cost (loop [i 0 s 0.0]
                                      (if (>= i m) s
                                          (recur (inc i) (+ s (* (aget r-trial i) (aget r-trial i))))))]
                     (if (< trial-cost cost)
                  ;; Accept step, decrease lambda
                       (do (acopy! x-trial 0 x 0 n)
                           (acopy! r-trial 0 r 0 m)
                           (recur (unchecked-add-int iter 1) (* lambda 0.1)))
                  ;; Reject step, increase lambda
                       (recur (unchecked-add-int iter 1) (* lambda 10.0))))))))))))

;; ================================================================
;; Gauss-Newton
;; ================================================================

(deftm gauss-newton
  "Gauss-Newton nonlinear least squares (delegates to Levenberg-Marquardt)."
  (All [T]
       [f :- (Fn [(Array T) (Array T)] (Array T)),
        x0 :- (Array T), m :- Long, n :- Long,
        tol :- Double, maxiter :- Long]
       (levenberg-marquardt f x0 m n tol maxiter)))

;; ================================================================
;; Curve fitting convenience
;; ================================================================

(deftm curve-fit
  "Fit model(params, x) -> y to data (xdata, ydata).
  model: (params[n-params], x) -> y.
  Returns {:minimizer :minimum :iterations :converged?}."
  [model :- (Fn [(Array double) Double] Double),
   xdata :- (Array double), ydata :- (Array double),
   p0 :- (Array double), n-data :- Long, n-params :- Long]
  (let [residual-fn (ftm [params :- (Array double), residuals :- (Array double)] :- (Array double)
                         (dotimes [i n-data]
                           (aset residuals i
                                 (- (model params (aget xdata i))
                                    (aget ydata i))))
                         residuals)]
    (levenberg-marquardt residual-fn p0 n-data n-params 1e-8 200)))

;; ================================================================
;; Projected gradient descent (bound constraints)
;; ================================================================

(deftm project-bounds!
  "Clamp each element of x to [lower, upper] bounds in-place."
  (All [T]
       [x :- (Array T), lower :- (Array T),
        upper :- (Array T), n :- Long] :- (Array T)
       (dotimes [i n]
         (aset x i (n/max (aget lower i) (n/min (aget upper i) (aget x i)))))
       x))

(deftm projected-gradient
  "Projected gradient descent with box constraints [lower, upper]."
  (All [T]
       [f :- (Fn [(Array T)] Double),
        grad-fn :- (Fn [(Array T) (Array T)]),
        x0 :- (Array T), lower :- (Array T),
        upper :- (Array T), n :- Long,
        tol :- Double, maxiter :- Long]
       (let [x (vec-copy x0)
             g (n/similar x0)]
         (project-bounds! x lower upper n)
         (loop [iter (int 0)]
           (grad-fn x g)
           (let [gnorm (vec-norm g)]
             (if (or (>= iter maxiter) (< gnorm tol))
               {:minimizer (vec x)
                :minimum   (f x)
                :iterations (long iter)
                :converged? (< gnorm tol)}
               (let [alpha (backtracking-linesearch f x
                                                    (let [d (n/similar x0)]
                                                      (vec-negate! d g) d)
                                                    (- (dot g g)) 1.0 1e-4 0.5 30)]
                 (dotimes [i n]
                   (aset x i (- (aget x i) (* alpha (aget g i)))))
                 (project-bounds! x lower upper n)
                 (recur (unchecked-add-int iter 1)))))))))

;; Note: differential-evolution, levenberg-marquardt, gauss-newton, curve-fit,
;; and compute-jacobian! remain double-only. They have complex allocation
;; patterns with multiple array sizes that aren't easily parametric.
;; The generic loop fallback pattern from nn.clj can be applied when needed.

;; ================================================================
;; Differential Evolution (global optimizer)
;; ================================================================

(deftm differential-evolution
  "Differential evolution for global optimization.
  f: x[n] -> Double. Searches within [lower, upper] bounds.
  Returns {:minimizer :minimum :iterations :converged?}."
  [f :- (Fn [(Array double)] Double),
   lower :- (Array double), upper :- (Array double),
   n :- Long, pop-size :- Long,
   maxiter :- Long, tol :- Double] :- Object
  (let [rng (java.util.concurrent.ThreadLocalRandom/current)
        ;; Initialize population
        pop (object-array pop-size)
        fitness (double-array pop-size)
        trial (double-array n)
        CR 0.7
        F 0.8]
    ;; Random initial population
    (dotimes [i pop-size]
      (let [x (double-array n)]
        (dotimes [j n]
          (aset x j (+ (aget lower j)
                       (* (.nextDouble rng) (- (aget upper j) (aget lower j))))))
        (aset pop i x)
        (aset fitness i (f x))))
    (loop [iter (int 0)]
      ;; Find best
      (let [best-idx (loop [i 1 bi 0 bv (aget fitness 0)]
                       (if (>= i pop-size) bi
                           (if (< (aget fitness i) bv)
                             (recur (inc i) i (aget fitness i))
                             (recur (inc i) bi bv))))
            best-val (aget fitness best-idx)
            best-x (aget pop best-idx)]
        (if (or (>= iter maxiter) (< best-val tol))
          {:minimizer (vec best-x)
           :minimum   best-val
           :iterations (long iter)
           :converged? (< best-val tol)}
          (do
            ;; DE/rand/1/bin mutation + crossover
            (dotimes [i pop-size]
              (let [xi (aget pop i)
                    ;; Pick 3 distinct random indices != i
                    a (loop [] (let [v (.nextInt rng (int pop-size))]
                                 (if (== v i) (recur) v)))
                    b (loop [] (let [v (.nextInt rng (int pop-size))]
                                 (if (or (== v i) (== v a)) (recur) v)))
                    c (loop [] (let [v (.nextInt rng (int pop-size))]
                                 (if (or (== v i) (== v a) (== v b)) (recur) v)))
                    xa (aget pop a)
                    xb (aget pop b)
                    xc (aget pop c)
                    j-rand (.nextInt rng (int n))]
                ;; Mutant + crossover
                (dotimes [j n]
                  (if (or (== j j-rand) (< (.nextDouble rng) CR))
                    (let [v (+ (aget xa j) (* F (- (aget xb j) (aget xc j))))
                          v (n/max (aget lower j) (n/min (aget upper j) v))]
                      (aset trial j v))
                    (aset trial j (aget xi j))))
                ;; Selection
                (let [f-trial (f trial)]
                  (when (<= f-trial (aget fitness i))
                    (acopy! trial 0 xi 0 n)
                    (aset fitness i f-trial)))))
            (recur (unchecked-add-int iter 1))))))))

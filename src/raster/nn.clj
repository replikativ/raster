(ns raster.nn
  "Neural network primitives backed by pure double[]/float[] arrays.

  NN ops are deftm with hand-written rrules for reverse-mode AD.
  The AD system treats them as opaque primitives with known derivatives.

  Supported operations:
    dense            - linear layer: y = W*x + b
    relu             - element-wise ReLU
    softmax          - numerically stable softmax
    cross-entropy    - cross-entropy loss
    softmax-cross-entropy - combined (numerically stable gradient)

  Usage with AD:
    (rad/grad-expr '(let* [h (raster.nn/dense W x b)
                           a (raster.nn/relu h)
                           loss (raster.nn/cross-entropy (raster.nn/softmax a) y)]
                      loss)
                   '[W b])"
  (:refer-clojure :exclude [aget aset alength aclone + - * / < > <= >= == zero? pos? neg? max min abs mod rem])
  (:require [raster.core :refer [deftm ftm broadcast scan reduce!]]
            [raster.arrays :refer [aget aset alength aclone acopy! alloc-like]]
            [raster.numeric :refer [+ - * / < > <= >= == zero? pos? neg? max min abs mod rem]]
            [raster.par :as par]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.random]
            [raster.linalg.blas :as blas]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.ad.templates :as tmpl]))

;; ================================================================
;; Generic fallbacks (All [T]) — MUST come BEFORE BLAS specializations.
;; The (All [T]) auto-generates concrete double overloads which would
;; overwrite BLAS versions if defined after them (last-writer-wins).
;; Defining generics first ensures BLAS versions register later and win.
;; This matches Julia's specificity rule: concrete > parametric.
;; ================================================================

(deftm dense
  "Dense linear layer: y = W*x + b. W is row-major [rows, cols]."
  (All [T] [W :- (Array T) x :- (Array T) b :- (Array T)] :- (Array T)
       (let [rows (alength b)
             cols (alength x)]
         (par/map [i rows]
                  (+ (aget b i)
                     (par/reduce acc 0.0 j cols
                                 (+ acc (* (aget W (+ (* i cols) j)) (aget x j)))))))))

(deftm dense-into!
  "Dense linear layer into pre-allocated buffer: out = W*x + b."
  (All [T]
       [W :- (Array T) x :- (Array T) b :- (Array T) out :- (Array T)] :- (Array T)
       (let [rows (alength b)
             cols (alength x)]
         (acopy! b 0 out 0 rows)
         (dotimes [i rows]
           (dotimes [j cols]
             (aset out i (+ (aget out i) (* (aget W (+ (* i cols) j)) (aget x j))))))
         out)))

(deftm dense-backward-dW
  "Dense backward pass for weights: dW = dy (outer) x^T."
  (All [T] [dy :- (Array T) x :- (Array T)] :- (Array T)
       (let [rows (alength dy)
             cols (alength x)
             dW (n/zero (raster.arrays/alloc-like dy (* rows cols)))]
         (dotimes [i rows]
           (par/map! dW j cols :offset (* i cols) T
                     (* (aget dy i) (aget x j))))
         dW)))

(deftm dense-backward-dW-into!
  "Accumulate weight gradient into buffer: dW += dy (outer) x^T."
  (All [T]
       [dy :- (Array T) x :- (Array T) dW :- (Array T)] :- (Array T)
  ;; dW += dy ⊗ x^T: outer loop over rows, inner par/map! per row for SIMD.
  ;; Each row writes dW[i*cols+j] += dy[i] * x[j] for j=0..cols-1.
       (let [rows (alength dy)
             cols (alength x)]
         (dotimes [i rows]
           (par/map! dW j cols :offset (* i cols) T
                     (+ (aget dW (+ (* i cols) j)) (* (aget dy i) (aget x j)))))
         dW)))

(deftm dense-backward-dx
  "Dense backward pass for input: dx = W^T * dy."
  (All [T]
       [dy :- (Array T) W :- (Array T) x :- (Array T)] :- (Array T)
  ;; dx = W^T * dy: column-reduction via par/map+par/reduce.
  ;; Each output dx[j] = Σᵢ W[i*cols+j]*dy[i] is an independent reduction.
  ;; Strided W access but no read-modify-write contention on dx.
       (let [cols (alength x)
             rows (alength dy)]
         (par/map [j cols]
                  (par/reduce acc 0.0 i rows
                              (+ acc (* (aget W (+ (* i cols) j)) (aget dy i))))))))

(deftm dense-backward-dx-into!
  "Dense backward pass for input into buffer: dx = W^T * dy."
  (All [T]
       [dy :- (Array T) W :- (Array T) dx :- (Array T)] :- (Array T)
  ;; dx = W^T * dy: column-reduction into existing buffer.
       (let [cols (alength dx)
             rows (alength dy)]
         (par/map! dx j cols T
                   (par/reduce acc 0.0 i rows
                               (+ acc (* (aget W (+ (* i cols) j)) (aget dy i)))))
         dx)))

;; ================================================================
;; Dense layer strategy: par/map + par/reduce (Futhark-style SOACs).
;;
;; XLA CPU uses inline LLVM IR (not BLAS) for ALL gemv operations.
;; Futhark never calls BLAS at all — matmul is always map+reduce.
;; dense uses par/map over rows with par/reduce over cols, making
;; the structure visible for SOAC fusion (dense+relu, dense+softmax).
;; dense-into! keeps dotimes for the buffer-fuse rewrite path.
;;
;; For large GEMM (batched training, not gemv): use blas/dgemv!
;; or blas/dgemm! explicitly.
;; ================================================================

(deftm relu
  "Element-wise rectified linear unit: max(0, x)."
  (All [T] [x :- (Array T)] :- (Array T)
       (broadcast [x] (n/max 0.0 x))))

(deftm softmax
  "Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))."
  (All [T] [x :- (Array T)] :- (Array T)
       (let [n (alength x)
             out (n/similar x)
             max-x (loop [i 0 m ##-Inf]
                     (if (< i n)
                       (recur (inc i) (n/max m (aget x i)))
                       m))
             sum-exp (loop [i 0 s 0.0]
                       (if (< i n)
                         (let [e (m/exp (- (aget x i) max-x))]
                           (raster.arrays/aset out i e)
                           (recur (inc i) (+ s e)))
                         s))
             inv-sum (/ 1.0 sum-exp)]
         (broadcast [out] (* out inv-sum)))))

(deftm cross-entropy
  "Cross-entropy loss: -sum(t * log(p)). p = predicted probabilities, t = targets."
  (All [T] [p :- (Array T) t :- (Array T)] :- T
       (let [n (alength p)]
         (loop [i 0 loss 0.0]
           (if (< i n)
             (let [ti (aget t i)]
               (if (zero? ti)
                 (recur (inc i) loss)
                 (recur (inc i) (- loss (* ti (m/log (n/max 1e-15 (aget p i))))))))
             loss)))))

(deftm softmax-cross-entropy
  "Combined softmax + cross-entropy. Returns [loss, softmax-output]."
  (All [T] [logits :- (Array T) t :- (Array T)]
       (let [s (softmax logits)
             loss (cross-entropy s t)]
         [loss s])))

;; relu, softmax, cross-entropy — parametric (All [T]) above
;; dense-backward-db, relu-backward, softmax-backward, cross-entropy-backward-dp
;; — parametric (All [T]) below

;; ================================================================
;; Weight initialization
;; ================================================================

(deftm xavier-init! (All [T]
                         [rng :- Object, fan-in :- Long, fan-out :- Long, out :- (Array T)] :- (Array T)
                         "Xavier/Glorot uniform initialization. Fills out with U[-limit, limit]
  where limit = sqrt(6 / (fan-in + fan-out)).
  Returns out for chaining."
                         (let [limit (n/sqrt (/ 6.0 (+ fan-in fan-out)))
                               n     (alength out)]
                           (raster.random/rand! rng out)
                           (dotimes [i n]
                             (aset out i (- (* (aget out i) 2.0 limit) limit)))
                           out)))

(deftm kaiming-init! (All [T]
                          [rng :- Object, fan-in :- Long, out :- (Array T)] :- (Array T)
                          "Kaiming/He normal initialization for ReLU networks. Fills out with
  N(0, sqrt(2/fan-in)). Returns out for chaining."
                          (let [std (n/sqrt (/ 2.0 fan-in))
                                n   (alength out)]
                            (raster.random/randn! rng out)
                            (dotimes [i n]
                              (aset out i (* (aget out i) std)))
                            out)))

;; ================================================================
;; Backward helper deftm functions (for template-based AD)
;; ================================================================

;; Dense backward functions: generic (All [T]) versions at top of file
;; are now primary. They use transparent dotimes loops, allowing the
;; compiler pipeline to convert them to par/map! for SOAC fusion.
;; BLAS versions (dger!, dgemv-t!) available via raster.linalg.blas
;; for explicit use in performance-critical large-matrix code.

;; Dense backward: db = clone(dy) — parametric
(deftm dense-backward-db
  "Dense backward pass for bias: db = clone(dy)."
  (All [T] [dy :- (Array T)] :- (Array T)
       (aclone dy)))

;; Dense backward into pre-allocated buffer: db = copy(dy)
(deftm dense-backward-db-into!
  "Dense backward pass for bias into buffer: out = copy(dy)."
  (All [T]
       [dy :- (Array T) out :- (Array T)] :- (Array T)
       (acopy! dy 0 out 0 (alength out))
       out))

;; ReLU backward: dx[i] = dy[i] * (x[i] > 0 ? 1 : 0)
(deftm relu-backward
  "ReLU backward: dx[i] = dy[i] if x[i] > 0, else 0."
  (All [T] [dy :- (Array T) x :- (Array T)] :- (Array T)
       (broadcast [dy x] (if (> x 0.0) dy 0.0))))

;; Softmax backward: dx[i] = s[i] * (dy[i] - dot(s, dy))
(deftm softmax-backward
  "Softmax backward: dx[i] = s[i] * (dy[i] - dot(s, dy))."
  (All [T] [dy :- (Array T) s :- (Array T)] :- (Array T)
       (let [s-dot-dy (reduce! [acc 0.0] [s dy] (+ acc (* s dy)))]
         (broadcast [s dy] (* s (- dy s-dot-dy))))))

;; Cross-entropy backward w.r.t. p: dp[i] = -t[i]/p[i] * dy
(deftm cross-entropy-backward-dp
  "Cross-entropy backward w.r.t. probabilities: dp[i] = -t[i]/p[i] * dy."
  (All [T] [dy :- T, p :- (Array T),
            t :- (Array T)] :- (Array T)
       (broadcast [p t] (* (- (/ t (n/max 1e-15 p))) dy))))

;; ================================================================
;; Loss function (composable, for compiler pipeline) — parametric
;; ================================================================

(deftm loss-fn
  "Two-layer MLP forward pass returning cross-entropy loss."
  (All [T] [W1 :- (Array T) b1 :- (Array T)
            W2 :- (Array T) b2 :- (Array T)
            x :- (Array T) y :- (Array T)] :- T
       (let [h (dense W1 x b1)
             a (relu h)
             out (dense W2 a b2)
             p (softmax out)]
         (cross-entropy p y))))

(deftm predict-fn
  "Two-layer MLP forward pass returning logits (no softmax)."
  [W1 :- (Array double) b1 :- (Array double)
   W2 :- (Array double) b2 :- (Array double)
   x :- (Array double)] :- (Array double)
  (let [h (dense W1 x b1)
        a (relu h)]
    (dense W2 a b2)))

;; ================================================================
;; Compiler descriptors for buffer fusion and shape propagation
;; ================================================================

(descriptor/register-buffer-semantics! 'raster.nn/dense
                                       {:allocates? true
                                        :in-place-arg nil
                                        :alloc-form (fn [args _opts]
                                                      (list 'raster.arrays/zeros-like (nth args 2) (list 'raster.arrays/alength (nth args 2))))
                                        :rewrite-fn (fn [args buf-sym] (list* 'raster.nn/dense-into! (conj (vec args) buf-sym)))})

;; Dim rules: result length for shape propagation
(descriptor/register-dim-rule! 'raster.nn/dense
                               (fn [[_W _x b] _dim-env _params-set]
                                 (list 'raster.arrays/alength b)))

(descriptor/register-dim-rule! 'raster.nn/dense-into!
                               (fn [[_W _x b _out] _dim-env _params-set]
                                 (list 'raster.arrays/alength b)))

(descriptor/register-dim-rule! 'raster.nn/relu
                               (fn [[x] dim-env _params-set]
                                 (or (get dim-env x)
                                     (list 'raster.arrays/alength x))))

(descriptor/register-dim-rule! 'raster.nn/softmax
                               (fn [[x] dim-env _params-set]
                                 (or (get dim-env x)
                                     (list 'raster.arrays/alength x))))

;; ================================================================
;; Buffer write mode registration (for hoist zero-fill)
;; ================================================================

;; dense-into!: acopy!(b→out) then dgemv!(W,x,out,m,n,1.0,1.0) — full overwrite
(descriptor/register-buffer-write! 'raster.nn/dense-into! :overwrite 3)

;; dense-backward-dW-into!: dger!(dy,x,dW,m,n,1.0) — ACCUMULATES (+=)
(descriptor/register-buffer-write! 'raster.nn/dense-backward-dW-into! :accumulate 2)

;; dense-backward-dx-into!: dgemv-t!(W,dy,dx,m,n,1.0,0.0) — overwrite (beta=0)
(descriptor/register-buffer-write! 'raster.nn/dense-backward-dx-into! :overwrite 2)

;; dense-backward-db-into!: acopy!(dy→out) — overwrite
(descriptor/register-buffer-write! 'raster.nn/dense-backward-db-into! :overwrite 1)

;; Buffer-SEMANTICS for the dense into-variants (distinct facet from write-mode
;; above): :in-place-arg is the authoritative output-buffer index resolve-alength
;; uses to size-alias (alength result) to the correct returned buffer instead of
;; guessing. :allocates? false so buffer_fuse leaves them untouched.
;; dense-backward-dW-into! [dy-cols/... dW ...] -> dW  (arg 2)
;; dense-backward-dx-into! [dy W dx]            -> dx  (arg 2)
;; dense-backward-db-into! [dy out]            -> out (arg 1)
(descriptor/register-buffer-semantics! 'raster.nn/dense-backward-dW-into!
                                       {:allocates? false :in-place-arg 2})
(descriptor/register-buffer-semantics! 'raster.nn/dense-backward-dx-into!
                                       {:allocates? false :in-place-arg 2})
(descriptor/register-buffer-semantics! 'raster.nn/dense-backward-db-into!
                                       {:allocates? false :in-place-arg 1})

;; ================================================================
;; Reverse-mode AD rules (rrules)
;; ================================================================

;; ================================================================
;; Reverse-mode AD rules (rrules)
;; ================================================================
;; Pullback factories delegate to the backward deftms above, which are
;; (All [T]) parametric — they handle float[], double[], and any future
;; array types without hardcoded dispatch.

;; Dense: y = W*x + b → dW = dy⊗x^T, dx = W^T*dy, db = clone(dy)

;; ReLU: dy/dx_i = dy_i * (x_i > 0 ? 1 : 0)

;; Softmax: dx_i = s_i * (dy_i - dot(s, dy))

;; Cross-entropy: dL/dp_i = -t_i/p_i * dy

;; d_logits = (s - t) * dy, where s is the softmax output from the forward result
(deftm softmax-cross-entropy-backward
  "Combined softmax-cross-entropy backward: d_logits = (s - t) * dy."
  (All [T] [dy :- T, s :- (Array T),
            t :- (Array T)]
       :- (Array T)
       (broadcast [s t] (* dy (- s t)))))

;; Combined softmax-cross-entropy: d_logits = (s - t) * dy

;; ================================================================
;; Template registration for softmax-cross-entropy
;; ================================================================

(tmpl/merge-into-template! 'raster.nn/softmax-cross-entropy
                           {:params '[logits t] :result 'r :adjoint 'dy
                            :grads-fn (fn [ctx [logits t] result-sym adjoint-sym gensym-fn]
               ;; result is [loss s], need to extract s — a PRIMAL intermediate
               ;; (the softmax output), so it carries the primal logits tag
               ;; directly (arg-tag, not Π of it; identical for differentiable
               ;; tags but semantically the primal space).
                                        (let [s-sym (gensym-fn "sce_s" (tmpl/arg-tag logits))
                                              d-logits (gensym-fn "d_logits" (tmpl/grad-tag logits))]
                                          [(update ctx :bindings into
                                                   [s-sym (list 'clojure.core/nth result-sym 1)
                                                    d-logits (list 'raster.nn/softmax-cross-entropy-backward
                                                                   adjoint-sym s-sym t)])
                                           [d-logits nil]]))})

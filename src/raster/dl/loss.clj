(ns raster.dl.loss
  "Loss functions for the Raster deep learning framework.

  All losses are deftm functions with rrules for reverse AD.

  Functions:
    mse-loss           - mean squared error
    cross-entropy-loss - cross-entropy with logits (numerically stable)
    huber-loss         - smooth L1 / Huber loss
    l1-loss            - mean absolute error"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm reduce! broadcast]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.par]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.ad.templates :as tmpl]
            [raster.compiler.core.op-descriptor :as descriptor]))

;; ================================================================
;; MSE Loss: mean((pred - target)^2)
;; ================================================================

(deftm mse-loss (All [T] [pred :- (Array T), target :- (Array T),
                          n :- Long] :- Double
                     (/ (reduce! [sum 0.0] [pred target]
                                 (let [d (- pred target)]
                                   (+ sum (* d d))))
                        (double n))))

;; d_pred[i] = scale*(pred[i]-target[i]); the elementwise mse-loss backward,
;; written as a broadcast SOAC so it lowers to a GPU kernel (resident training)
;; AND SIMD-vectorizes on CPU — replacing the CPU-era ^:no-inline BLAS helper
;; (raster.linalg.blas/daxpy-diff!) that the resident GPU path cannot lower.
;; `scale` is a declared kernel scalar param (a uniform), so at :float dtype it
;; is emitted as a float scalar in the OpenCL C kernel (exactly like rms-norm!'s
;; `eps`/`gain-offset` Double params) — float×float, no T×Double garbage. It
;; lowers to its own par/map kernel step (not inlined as a double intermediate).
(deftm mse-grad (All [T] [pred :- (Array T) target :- (Array T)
                          scale :- Double n :- Long] :- (Array T)
                     (broadcast [pred target] (n/* scale (n/- pred target)))))

;; d_pred = 2*(pred - target)/n
(defn- alloc-like
  "Allocate an array of the same type and given length."
  [x ^long n]
  (if (instance? (Class/forName "[F") x)
    (float-array n)
    (double-array n)))

(defn- aset-like!
  "Set array element, dispatching on array type."
  [arr ^long i ^double v]
  (if (instance? (Class/forName "[F") arr)
    (clojure.core/aset ^floats arr (int i) (float v))
    (clojure.core/aset ^doubles arr (int i) v)))

(defn- aget-like
  "Get array element as double, dispatching on array type."
  ^double [arr ^long i]
  (if (instance? (Class/forName "[F") arr)
    (double (clojure.core/aget ^floats arr (int i)))
    (clojure.core/aget ^doubles arr (int i))))


;; ================================================================
;; Cross-entropy with logits (numerically stable)
;; logits:[batch, classes], target:[batch] (long[], class indices)
;; Returns: -mean(log_softmax[target_class])
;; ================================================================

(deftm cross-entropy-loss-into!
  "Reduce stable cross-entropy into caller-owned `out[0]`.

  Maxima and exponential sums are segmented over classes, exposing the cooperative reduction axis
  to the scheduler. The final batch mean is a separate product reduction because its algebra is
  independent of the per-row log-sum-exp. Floating sums explicitly permit a target reduction tree;
  the maximum has deterministic NaN and tie ordering."
  [logits :- (Array double) target :- (Array long) out :- (Array double)
   batch :- Long classes :- Long] :- Void
  (let [maxima (double-array batch)
        sums (double-array batch)
        losses (double-array batch)]
    (raster.par/product-reduce!
     [maxima nil]
     [[best Double/NEGATIVE_INFINITY :double]
      [best-index (long Long/MAX_VALUE) :long]]
     [[b batch]] c classes
     [candidate (aget logits (+ (* b classes) c))]
     [candidate c]
     [[left right] [left-index right-index]]
     [left-nan (long (if (== left left) 0 1))
      right-nan (long (if (== right right) 0 1))
      better (long (if (== right-nan 1)
                     (if (== left-nan 1) (if (< right-index left-index) 1 0) 1)
                     (if (== left-nan 1)
                       0
                       (if (> right left) 1
                         (if (== right left) (if (< right-index left-index) 1 0) 0)))))]
     [(if (== better 1) right left)
      (if (== better 1) right-index left-index)]
     {:associative? true :commutative? true
      :order {:nan :highest :tie :lowest-index}})
    (raster.par/product-reduce!
     [sums]
     [[sum 0.0 :double]]
     [[b batch]] c classes
     [term (Math/exp (- (aget logits (+ (* b classes) c)) (aget maxima b)))]
     [term]
     [[left right]] []
     [(+ left right)]
     {:associative? true :commutative? true :order :implementation-defined})
    (raster.par/map!
     losses b batch double
     (let [target-class (aget target b)]
       (- (+ (aget maxima b) (Math/log (aget sums b)))
          (aget logits (+ (* b classes) target-class)))))
    (raster.par/product-reduce!
     [out]
     [[total 0.0 :double]]
     [[segment 1]] b batch
     [row-loss (/ (aget losses b) (double batch))]
     [row-loss]
     [[left right]] []
     [(+ left right)]
     {:associative? true :commutative? true :order :implementation-defined})))

(deftm cross-entropy-loss [logits :- (Array double) target :- (Array long)
                           batch :- Long classes :- Long] :- Double
  (let [out (double-array 1)
        _ (cross-entropy-loss-into! logits target out batch classes)]
    (aget out 0)))

;; d_logits = (softmax(logits) - one_hot(target)) / batch
(deftm cross-entropy-loss-backward [dy :- Double logits :- (Array double)
                                    target :- (Array long) batch :- Long classes :- Long]
  :- (Array double)
  (let [maxima (double-array batch)
        sums (double-array batch)
        d-logits (double-array (* batch classes))]
    (raster.par/product-reduce!
     [maxima nil]
     [[best Double/NEGATIVE_INFINITY :double]
      [best-index (long Long/MAX_VALUE) :long]]
     [[b batch]] c classes
     [candidate (aget logits (+ (* b classes) c))]
     [candidate c]
     [[left right] [left-index right-index]]
     [left-nan (long (if (== left left) 0 1))
      right-nan (long (if (== right right) 0 1))
      better (long (if (== right-nan 1)
                     (if (== left-nan 1) (if (< right-index left-index) 1 0) 1)
                     (if (== left-nan 1)
                       0
                       (if (> right left) 1
                         (if (== right left) (if (< right-index left-index) 1 0) 0)))))]
     [(if (== better 1) right left)
      (if (== better 1) right-index left-index)]
     {:associative? true :commutative? true
      :order {:nan :highest :tie :lowest-index}})
    (raster.par/product-reduce!
     [sums]
     [[sum 0.0 :double]]
     [[b batch]] c classes
     [term (Math/exp (- (aget logits (+ (* b classes) c)) (aget maxima b)))]
     [term]
     [[left right]] []
     [(+ left right)]
     {:associative? true :commutative? true :order :implementation-defined})
    (raster.par/map!
     d-logits i (* batch classes) double
     (let [b (quot i classes)
           c (rem i classes)
           softmax (/ (Math/exp (- (aget logits i) (aget maxima b))) (aget sums b))
           grad (if (== c (aget target b)) (- softmax 1.0) softmax)]
       (* dy (/ grad (double batch)))))
    d-logits))


(tmpl/merge-into-template! 'raster.dl.loss/cross-entropy-loss
                           {:params '[logits target batch classes] :adjoint 'dy
                            :grads-fn (fn [ctx [logits target batch classes] _result adjoint gensym-fn]
                                        (let [dl (gensym-fn "d_logits" (tmpl/grad-tag logits))]
                                          [(update ctx :bindings into
                                                   [dl (list 'raster.dl.loss/cross-entropy-loss-backward
                                                             adjoint logits target batch classes)])
                                           [dl nil nil nil]]))})

;; ================================================================
;; Huber Loss (Smooth L1)
;; ================================================================

(deftm huber-loss (All [T] [pred :- (Array T), target :- (Array T),
                            n :- Long, delta :- Double] :- Double
                       (/ (reduce! [sum 0.0] [pred target]
                                   (let [d (n/abs (- pred target))
                                         loss (if (<= d delta) (* 0.5 d d) (- (* delta d) (* 0.5 delta delta)))]
                                     (+ sum loss)))
                          (double n))))

;; huber': |d| <= delta ? d/n : delta*sign(d)/n
(deftm huber-loss-backward [dy :- Double pred :- (Array double) target :- (Array double)
                            n :- Long delta :- Double]
  :- (Array double)
  (let [d-pred (double-array n)]
    (dotimes [i n]
      (let [d (- (clojure.core/aget pred i) (clojure.core/aget target i))
            grad (cond (< (Math/abs d) delta) d
                       (pos? d) delta
                       :else (- delta))]
        (clojure.core/aset d-pred i (* dy (/ grad (double n))))))
    d-pred))


(tmpl/merge-into-template! 'raster.dl.loss/huber-loss
                           {:params '[pred target n delta] :adjoint 'dy
                            :grads-fn (fn [ctx [pred target n delta] _result adjoint gensym-fn]
                                        (let [dp (gensym-fn "d_pred" (tmpl/grad-tag pred))]
                                          [(update ctx :bindings into
                                                   [dp (list 'raster.dl.loss/huber-loss-backward adjoint pred target n delta)])
                                           [dp nil nil nil]]))})

;; ================================================================
;; L1 Loss: mean(|pred - target|)
;; ================================================================

(deftm l1-loss (All [T] [pred :- (Array T), target :- (Array T),
                         n :- Long] :- Double
                    (/ (reduce! [sum 0.0] [pred target]
                                (+ sum (n/abs (- pred target))))
                       (double n))))

;; l1': sign(pred - target) / n
(deftm l1-loss-backward [dy :- Double pred :- (Array double) target :- (Array double) n :- Long]
  :- (Array double)
  (let [d-pred (double-array n)]
    (dotimes [i n]
      (let [d (- (clojure.core/aget pred i) (clojure.core/aget target i))
            grad (Math/signum d)]
        (clojure.core/aset d-pred i (* dy (/ grad (double n))))))
    d-pred))


(tmpl/merge-into-template! 'raster.dl.loss/l1-loss
                           {:params '[pred target n] :adjoint 'dy
                            :grads-fn (fn [ctx [pred target n] _result adjoint gensym-fn]
                                        (let [dp (gensym-fn "d_pred" (tmpl/grad-tag pred))]
                                          [(update ctx :bindings into
                                                   [dp (list 'raster.dl.loss/l1-loss-backward adjoint pred target n)])
                                           [dp nil nil]]))})

;; ================================================================
;; Compiler descriptors for buffer fusion
;; ================================================================

;; mse-loss: returns scalar, does not allocate a buffer
(descriptor/register-buffer-semantics! 'raster.dl.loss/mse-loss
                                       {:allocates? false})

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

(deftm cross-entropy-loss [logits :- (Array double) target :- (Array long)
                           batch :- Long classes :- Long] :- Double
  (loop [b 0 total-loss 0.0]
    (if (< b batch)
      (let [offset (* b (int classes))
            ;; log-sum-exp for numerical stability
            max-logit (loop [c 0 m Double/NEGATIVE_INFINITY]
                        (if (< c classes)
                          (recur (inc c) (Math/max m (clojure.core/aget logits (+ offset c))))
                          m))
            lse (loop [c 0 s 0.0]
                  (if (< c classes)
                    (recur (inc c) (+ s (Math/exp (- (clojure.core/aget logits (+ offset c))
                                                     max-logit))))
                    (+ max-logit (Math/log s))))
            target-class (clojure.core/aget target b)
            log-prob (- (clojure.core/aget logits (+ offset target-class)) lse)]
        (recur (inc b) (- total-loss log-prob)))
      (/ total-loss (double batch)))))

;; d_logits = (softmax(logits) - one_hot(target)) / batch
(deftm cross-entropy-loss-backward [dy :- Double logits :- (Array double)
                                    target :- (Array long) batch :- Long classes :- Long]
  :- (Array double)
  (let [d-logits (double-array (* batch classes))]
    (dotimes [b batch]
      (let [offset (* b (int classes))
            max-logit (loop [c 0 m Double/NEGATIVE_INFINITY]
                        (if (< c classes)
                          (recur (inc c) (Math/max m (clojure.core/aget logits (+ offset c))))
                          m))
            sum-exp (loop [c 0 s 0.0]
                      (if (< c classes)
                        (let [e (Math/exp (- (clojure.core/aget logits (+ offset c)) max-logit))]
                          (clojure.core/aset d-logits (+ offset c) e)
                          (recur (inc c) (+ s e)))
                        s))
            inv-sum (/ 1.0 sum-exp)
            target-c (clojure.core/aget target b)]
        (dotimes [c classes]
          (let [idx (+ offset c)
                si (* (clojure.core/aget d-logits idx) inv-sum)
                grad (if (== c target-c) (- si 1.0) si)]
            (clojure.core/aset d-logits idx (* dy (/ grad (double batch))))))))
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


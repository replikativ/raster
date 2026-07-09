(ns raster.compiler.semantic-invariants-test
  "Corpus tests for the deep semantic validator tier (*validate-deep?*),
  especially I-T3 (dtype closure): under :dtype :float monomorphization no
  double/doubles-stamped binder may appear unless its double-ness is DECLARED —
  an explicit (double ...) cast in its init, a reference to an already-exempt
  double island/param, or a devirtualized callee whose deftm declares `:- Double`.
  This is the enforcement seam for the f64-in-f32 silent-miscompile bug class
  (stale f64 stamps → double arithmetic on CPU, garbage kernels on GPU).

  Positive corpus: kernels with INTENTIONAL double islands must compile at
  :float with the throwing check bound — rms-norm's stability island
  (`(/ (double s) (double features))` → inv), gelu, the fused/batched causal
  SDPA + gqa attention path, and a full AD train step whose non-inlined
  mse-loss returns a declared Double from floats args.

  Negative: a deliberately stale double stamp must throw a named :I-T3 error."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.ir.invariants :as inv]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]))

;; Small float reduction kernel in the fused-SDPA shape: (All [T]) with a
;; `loop ... 0.0` accumulator (monomorphizes to a float stamp at :float) and an
;; n/oftype sentinel (the T-typed mask-literal idiom). Must compile clean.
(deftm it3-row-max-sumexp
  (All [T] [x :- (Array T) n :- Long] :- T
       (let [neg-inf (raster.numeric/oftype x -1.0e38)
             m (loop [i 0 m neg-inf]
                 (if (< i n)
                   (recur (inc i) (raster.numeric/max m (raster.arrays/aget x i)))
                   m))]
         (loop [i 0 s 0.0]
           (if (< i n)
             (recur (inc i)
                    (raster.numeric/+ s (raster.numeric/- (raster.arrays/aget x i) m)))
             s)))))

;; ================================================================
;; Positive corpus — intentional double islands must NOT throw
;; ================================================================

(deftest it3-float-corpus-compiles
  (binding [pl/*validate-deep?* true]
    (testing "float reduction kernel (loop 0.0 acc + oftype sentinel)"
      (let [f (pl/compile-aot #'it3-row-max-sumexp :dtype :float)]
        (is (fn? f))
        ;; x = [1 2 3], max = 3, sum (x_i - max) = -3
        (is (== -3.0 (f (float-array [1.0 2.0 3.0]) 3)))))
    (testing "rms-norm: intentional double stability island ((double s)/(double features) → inv)"
      (let [f (pl/compile-aot #'nn/rms-norm :dtype :float)]
        (is (fn? f))
        ;; 1 row, 2 features, weight=1, gain-offset 0: y_i = x_i/sqrt(mean(x^2)+eps)
        (let [y (f (float-array [3.0 4.0]) (float-array [1.0 1.0]) 1 2 1.0e-6 0.0)]
          (is (< (Math/abs (- (aget ^floats y 0) (/ 3.0 (Math/sqrt 12.5)))) 1.0e-5)))))
    (testing "gelu"
      (is (fn? (pl/compile-aot #'nn/gelu :dtype :float))))
    (testing "batched causal SDPA (fused attention kernel)"
      (is (fn? (pl/compile-aot #'attn/batched-causal-sdpa :dtype :float))))
    (testing "gqa-causal-mha (the GPU-lowerable gqa path)"
      (is (fn? (pl/compile-aot #'attn/gqa-causal-mha :dtype :float))))))

(deftest it3-ad-train-step-declared-double-return
  ;; A full value+grad train step over floats binds the NON-INLINED mse-loss
  ;; result — `(.invk mse-loss_m_floats_floats_long-impl ...)` stamped double —
  ;; whose double-ness is the callee's DECLARED `:- Double` return (resolved via
  ;; the .invk form's :raster.op/original + resolve-deftm-var), not a stale
  ;; stamp. Must compile clean under the throwing check.
  (require 'raster.dl.loss 'raster.dl.optim 'raster.ad.reverse 'raster.arrays)
  (let [_ (eval '(raster.core/deftm it3-ad-probe-loss
                   [W :- (Array float) x :- (Array float) tgt :- (Array float)
                    batch :- Long in-f :- Long out-f :- Long] :- Double
                   (let [pred (raster.dl.nn/linear-nb x W batch in-f out-f)]
                     (raster.dl.loss/mse-loss pred tgt (clojure.core/* batch out-f)))))
        train (eval '(raster.core/deftm it3-ad-probe-train
                       [W :- (Array float) x :- (Array float) tgt :- (Array float)
                        batch :- Long in-f :- Long out-f :- Long lr :- Double] :- Double
                       (let [vg ((raster.ad.reverse/value+grad #'it3-ad-probe-loss)
                                 W x tgt batch in-f out-f)
                             loss (clojure.core/nth vg 0)
                             dW (clojure.core/nth vg 1)]
                         (raster.dl.optim/sgd-step! W dW (raster.arrays/alength W) lr)
                         loss)))]
    (binding [pl/*validate-deep?* true]
      (is (fn? (pl/compile-aot train :dtype :float))
          "AD train step with declared-Double mse-loss compiles under I-T3"))))

;; ================================================================
;; Negative — a stale double stamp must throw a named :I-T3 error
;; ================================================================

(defn- stale-double-form
  "A float-kernel shape with a DELIBERATELY stale 'double stamp on the binder:
  a loop 0.0 accumulator with float-typed body and no double cast anywhere."
  []
  (list 'let* [(with-meta 'acc {:raster.type/tag 'double})
               '(loop* [i 0 s 0.0]
                       (if (clojure.core/< i n)
                         (recur (clojure.core/inc i)
                                (.invk raster.numeric/_plus__m_float_float-impl
                                       s (clojure.core/aget x (long i))))
                         s))]
        'acc))

(deftest it3-negative-stale-double-stamp-throws
  (testing "check-deep! throws a named I-T3 error at a stamped boundary"
    (let [e (try (inv/check-deep! :fixpointed (stale-double-form) :fixpoint
                                  {:params [] :dtype :float :param-env {}})
                 nil
                 (catch clojure.lang.ExceptionInfo e e))]
      (is (some? e) "stale double stamp under :float throws")
      (is (= :I-T3 (:invariant (ex-data e))))
      (is (= 'acc (:binder (ex-data e))))
      (is (= 'double (:tag (ex-data e))))
      (is (= :fixpoint (:pass (ex-data e))))))
  (testing "the same form does NOT throw at :double dtype (I-T3 is :float-gated)"
    (is (nil? (inv/check-deep! :fixpointed (stale-double-form) :fixpoint
                               {:params [] :dtype :double :param-env {}}))))
  (testing "skipped at half-stamped boundaries (:lowered pre-rewalk, post-backend)"
    (is (nil? (inv/check-deep! :lowered (stale-double-form) :lower
                               {:params [] :dtype :float :param-env {}})))
    (is (nil? (inv/check-deep! :backend-applied (stale-double-form) :backend
                               {:params [] :dtype :float :param-env {}})))))

(deftest it3-exemption-unit-cases
  (testing "an explicit (double ...) cast in the init exempts (intentional island)"
    (is (nil? (inv/dtype-closure-violation
               (list 'let* [(with-meta 'ms {:raster.type/tag 'double})
                            '(loop* [i 0 s 0.0]
                                    (if (clojure.core/< i n)
                                      (recur (clojure.core/inc i) s)
                                      (.invk raster.numeric/_div__m_double_double-impl
                                             (double s) (double n))))]
                     'ms)
               {} :float))))
  (testing "a double-tagged deftm param seeds the exempt set"
    (is (nil? (inv/dtype-closure-violation
               (list 'let* [(with-meta 'e {:raster.type/tag 'double}) 'eps] 'e)
               {'eps 'double} :float)))
    (is (some? (inv/dtype-closure-violation
                (list 'let* [(with-meta 'e {:raster.type/tag 'double}) 'eps] 'e)
                {'eps 'floats} :float))
        "a floats param does NOT exempt a double-stamped binder"))
  (testing "exemption flows through references: inv referencing exempt island ms"
    (is (nil? (inv/dtype-closure-violation
               (list 'let* [(with-meta 'ms {:raster.type/tag 'double}) '(double s)
                            (with-meta 'inv {:raster.type/tag 'double})
                            '(.invk raster.numeric/_div__m_double_double-impl one ms)]
                     'inv)
               {} :float)))))

;; ================================================================
;; I-T2-lite — binder/init tag consistency (WARN-only; unit of the detector)
;; ================================================================

(deftest it2-binder-tag-inconsistency-detected
  (is (= [{:sym 'a :init-tag 'float}]
         (vec (inv/binder-tag-inconsistencies
               (list 'let* ['a (with-meta '(.invk raster.numeric/_plus__m_float_float-impl x y)
                                 {:raster.type/tag 'float})]
                     'a)))))
  (is (nil? (inv/binder-tag-inconsistencies
             (list 'let* [(with-meta 'a {:raster.type/tag 'float})
                          (with-meta '(.invk raster.numeric/_plus__m_float_float-impl x y)
                            {:raster.type/tag 'float})]
                   'a)))))

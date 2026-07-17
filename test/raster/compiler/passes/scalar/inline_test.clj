(ns raster.compiler.passes.scalar.inline-test
  "Guards the deftm inliner's call/callee arity contract.

  The inliner substitutes callee params for call args POSITIONALLY — `(nth args i)`
  over the param index. Nothing checked that the two had the same length, so a stale
  call site — a deftm that grew a parameter while one of its callers was not
  updated — died as a bare `IndexOutOfBoundsException: null` out of PersistentVector,
  thrown from deep inside the AD-prep inliner (inline.clj:1108), naming neither the
  callee nor the arities. It read like a compiler bug; it was a wrong-arity CALL.

  (The same stale call against a PARAMETRIC (All [T]) callee fails differently and
  even more obscurely: no overload resolves, the call is never inlined, stays
  symbolic, and reverse-AD reports `No AD template for <callee>` — a message that
  points at the AD registry rather than at the call. Both shapes were hit in the same
  finetune commit, where the gemma block gained a `bs` batch param and two reference
  losses were not updated.)

  A deftm has no variadic or multi-arity form, so a wrong-arity call can never
  dispatch: it is always a bug, and the inliner must say so AT THE CALL — naming the
  callee and both arities."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.par]
            [raster.numeric]
            [raster.arrays]
            [raster.ad.reverse :as rev]))

;; Monomorphic callee taking THREE args (mirrors the concrete-float finetune.train/gblock,
;; which takes 38 and was called with 37).
(deftm arity-callee
  [a :- (Array double) n :- Long s :- Double] :- Double
  (raster.par/reduce acc 0.0 i n
                     (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget a i) s))))

;; The STALE caller: arity-callee takes (a n s); this passes only (a n).
;; deftm walking is lazy, so the bad call is only reached when value+grad forces the
;; AD-prep inline — exactly where the finetune chain died.
(deftm arity-caller-stale
  [a :- (Array double) n :- Long] :- Double
  (raster.compiler.passes.scalar.inline-test/arity-callee a n))

;; The correct caller — the guard must not over-fire.
(deftm arity-caller-good
  [a :- (Array double) n :- Long s :- Double] :- Double
  (raster.compiler.passes.scalar.inline-test/arity-callee a n s))

(deftest wrong-arity-deftm-call-fails-loud-in-ad-inline
  (testing "value+grad over a stale (wrong-arity) deftm call names the callee and both arities"
    (let [t (try (let [vg (rev/value+grad #'arity-caller-stale)]
                   ;; if grad-body construction is deferred, force it with a call
                   (vg (double-array [1.0 2.0]) 2)
                   nil)
                 (catch Throwable e e))]
      (is (some? t) "a wrong-arity deftm call must throw, not silently mis-inline")
      (is (not (instance? IndexOutOfBoundsException t))
          "must not surface as a bare IndexOutOfBoundsException from PersistentVector")
      (is (instance? clojure.lang.ExceptionInfo t)
          (str "must be a NAMED ex-info, got: " (some-> t class str)))
      (when (instance? clojure.lang.ExceptionInfo t)
        (let [d (ex-data t)]
          (is (= 'raster.compiler.passes.scalar.inline-test/arity-callee (:callee d))
              "names the callee")
          (is (= 3 (:expected-arity d)) "reports the callee's declared arity")
          (is (= 2 (:actual-arity d)) "reports the call's actual arity")
          (is (re-find #"Arity mismatch" (ex-message t))
              (str "message must say what is wrong, got: " (ex-message t))))))))

(deftest correct-arity-call-still-inlines-and-differentiates
  (testing "the guard must not over-fire: a well-formed call differentiates through the inline"
    (let [a (double-array [1.0 2.0])
          [v da] ((rev/value+grad #'arity-caller-good) a 2 3.0)]
      ;; L = s * sum(a) = 3 * (1+2) = 9 ; dL/da_i = s = 3
      (is (= 9.0 (double v)))
      (is (= [3.0 3.0] (vec da))))))

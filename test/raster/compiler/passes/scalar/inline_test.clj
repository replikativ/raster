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
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.ad.reverse :as rev]))

(deftest value-gradient-projection-indices-use-checked-constant-evidence
  (let [elements {'vg ['primal 'gradient]}]
    (doseq [index [1 '(long 1) '(clojure.core/long 1) '(clojure.core/int (long 1))]]
      (is (= 'gradient (#'inline/known-vg-element elements 'vg index))))
    (doseq [index [-1 2 'n '(long n) '(int 4294967296) '(float 1)]]
      (is (nil? (#'inline/known-vg-element elements 'vg index))))))

(deftest lifted-arguments-retain-source-types-not-formal-consumer-types
  (doseq [[argument environment] [['(float 1.0) {}]
                                 ['(clojure.core/aget a 0) {'a 'floats}]]]
    (let [bindings (atom [])
          substitution (#'inline/argument-substitution
                        ['x] [argument] ['double] environment #(swap! bindings conj %))
          id (get substitution 'x)]
      (is (= 'float (:raster.type/tag (meta id))))
      (is (= [[id argument]] @bindings)))))

(deftest leaf-bodies-are-inlinable
  (doseq [body ['x 'java.lang.Float/NEGATIVE_INFINITY 42 1.5 nil true]]
    (is (#'inline/inlinable-body? body))))

(deftest inlining-preserves-unused-checked-arguments
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['int] :walked-body [42]})}
    (fn []
      (let [inlined (#'inline/inline-invk '(.invk example/constant (int 4294967296)))]
        (is (thrown? ArithmeticException (eval inlined))
            "an unused checked argument still throws before entering the callee"))
      (let [inlined (#'inline/inline-invk
                    '(.invk example/constant (int (do (swap! calls inc) 3))))]
        (is (= [42 1] (eval (list 'let ['calls '(atom 0)]
                                 [inlined '(deref calls)]))))))))

(deftest direct-inline-arity-is-checked-before-substitution
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body [42]})}
    (fn []
      (doseq [call ['(.invk example/constant)
                   '(.invk example/constant 1 (swap! calls inc))]]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Arity mismatch"
                             (#'inline/inline-invk call)))))))

(deftest unused-collection-arguments-still-evaluate-their-elements
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :walked-body [42]})}
    (fn []
      (doseq [argument ['[(swap! calls inc)] '{:key (swap! calls inc)}
                       '#{(swap! calls inc)}]]
        (let [inlined (#'inline/inline-invk (list '.invk 'example/constant argument))]
          (is (= [42 1] (eval (list 'let ['calls '(atom 0)]
                                   [inlined '(deref calls)])))))))))

(deftest local-function-inlining-preserves-call-by-value
  (doseq [body [nil false 42 '(+ x x)]
          argument ['(int (do (swap! calls inc) 3))]]
    (let [source (list 'let* ['f (list 'ftm ['x] body)
                             'result (list 'f argument)] 'result)
          inlined (:form (#'inline/inline-one-pass source))
          ;; The local function declaration remains for later DCE; use a host fn
          ;; solely to execute this intermediate compiler form.
          executable (clojure.walk/postwalk
                      #(if (and (seq? %) (= 'ftm (first %)))
                         (cons 'fn (rest %)) %) inlined)]
      (is (= [(if (seq? body) 6 body) 1]
             (eval (list 'let ['calls '(atom 0)] [executable '(deref calls)])))))))

(deftest binding-inline-accepts-constant-leaves
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [& _] {:params ['x] :tags ['long] :walked-body [42]})}
    (fn []
      (let [inlined (:form (#'inline/inline-one-pass
                           '(let* [result (example/constant (int 4294967296))] result)))]
        (is (thrown? ArithmeticException (eval inlined)))))))

(deftest nested-argument-calls-are-expanded-before-lifting
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body ['x]})}
    (fn []
      (let [inlined (#'inline/inline-invk
                    '(.invk example/identity (.invk example/identity (+ 1 2))))]
        (is (not-any? #(and (seq? %) (= '.invk (first %)))
                      (tree-seq coll? seq inlined)))
        (is (= 3 (eval inlined)))))))

(deftest inlining-visits-loop-initializers-without-changing-binders
  (with-redefs-fn {#'inline/try-resolve-deftm
                  (fn [_] {:params ['x] :tags ['long] :walked-body ['x]})}
    (fn []
      (let [inlined (#'inline/inline-invk
                    '(loop* [i 0 acc (.invk example/identity (long (+ i 2)))]
                       (if (< i 3) (recur (inc i) (+ acc i)) acc)))]
        (is (= ['i 'acc] (vec (take-nth 2 (second inlined)))))
        (is (not-any? #(and (seq? %) (= '.invk (first %)))
                      (tree-seq coll? seq inlined)))
        (is (= 5 (eval inlined)))))))

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

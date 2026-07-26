(ns raster.compiler.core.inline-pure-lets-test
  "ONE beta-reduction of `let*`. Device-free.

   Three copies of this rewrite existed — the GPU expression emitter's, the SOAC lowerer's, and the
   CPU SIMD emitter's — each built on `walk/postwalk-replace` inside `walk/prewalk`, and each
   carrying a DIFFERENT unstated precondition. Consolidating them is only safe if the shared version
   keeps the one gate that existed and repairs what all three shared:

     • `postwalk-replace` is not capture-avoiding: it rewrites binders and array positions.
     • `clojure.walk` rebuilds seqs with `(apply list …)`, dropping metadata — which in this pipeline
       discards TC `:raster.type/tag` stamps and the `:raster.op/original` stamps the purity
       predicate itself reads, so the gate could be defeated on a later pass.
     • `prewalk` never revisits the form a reduction produces, so a `let*` exposed by inlining
       survived.

   The migration's real risk is a loud→silent trade: two of the three callers had NO purity gate, so
   the natural refactor is to make the shared signature agree by dropping the gate. The final
   section pins each caller's refusal channel so that cannot happen quietly."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.util :as util]))

;; ── the purity predicate the default gate is built from ─────────────────────────────
(deftest effectful-sees-surface-and-devirtualized-writes
  (testing "a surface void op"
    (is (util/effectful? '(raster.arrays/aset O 0 1.0)))
    (is (util/effectful? '(clojure.core/+ 1 (raster.arrays/aset O 0 1.0)))
        "…at any depth, since inlining hoists the whole tree"))
  (testing "a DEVIRTUALIZED write, recognized only via the walker's :raster.op/original stamp —
            an intermediate's aset materializes to .invk, which a head-only check misses"
    (is (util/effectful? (with-meta (list '.invk 'impl 'O 0 1.0)
                           {:raster.op/original 'raster.arrays/aset}))))
  (testing "a tagged form has a return type and is therefore not a statement"
    (is (not (util/effectful? (with-meta (list 'raster.arrays/aset 'O 0 1.0)
                                {:raster.type/tag 'Double})))))
  (testing "pure arithmetic and reads"
    (is (not (util/effectful? '(clojure.core/* (raster.arrays/aget A i) 2)))))
  (testing "quoted data is data, not code to scan"
    (is (not (util/effectful? '(quote (raster.arrays/aset O 0 1.0)))))))

;; ── the rewrite ─────────────────────────────────────────────────────────────────────
(deftest sequential-inits-see-the-prior-bindings
  (testing "let* is sequential: b's init must already carry a's substitution"
    (is (= '(clojure.core/+ (clojure.core/* (raster.arrays/aget A i) 2) 1)
           (util/inline-pure-lets
            '(let* [a (raster.arrays/aget A i)
                    b (clojure.core/* a 2)]
               (clojure.core/+ b 1))))))
  (testing "binding-env exposes the same env for the caller that has already destructured"
    (is (= '{a (raster.arrays/aget A i) b (clojure.core/* (raster.arrays/aget A i) 2)}
           (util/binding-env '[a (raster.arrays/aget A i) b (clojure.core/* a 2)])))))

(deftest an-inner-rebinding-of-an-inlined-name-is-not-corrupted
  (testing "postwalk-replace would rewrite the inner BINDER, producing
            (let* [(aget A i) 99] …) — a form that is not even well-formed"
    (is (= '(clojure.core/+ 99 1)
           (util/inline-pure-lets
            '(let* [x (raster.arrays/aget A i)]
               (let* [x 99] (clojure.core/+ x 1)))))))
  (testing "a name in an ARRAY position is likewise a binder-independent occurrence: substituting
            an array symbol must not be confused with substituting an index"
    (is (= '(raster.arrays/aget A (clojure.core/* 2 4))
           (util/inline-pure-lets
            '(let* [i (clojure.core/* 2 4)] (raster.arrays/aget A i))))))
  (testing "a par binder shadowing an inlined name alpha-renames rather than capturing"
    (let [r (util/inline-pure-lets
             '(let* [s (clojure.core/* n 2)]
                (raster.par/map! O n (fn* [n] (clojure.core/+ n s)))))]
      (is (not= 'n (first (nth (nth r 3) 1)))
          "the fn* binder was renamed, so `s`'s reference to the OUTER n still resolves"))))

(deftest the-reduction-reaches-a-fixpoint
  (testing "prewalk never revisited the form a reduction produced, so a let* exposed by inlining
            survived into the emitter"
    (is (= '(clojure.core/+ (clojure.core/* 7 2) 1)
           (util/inline-pure-lets
            '(let* [x 7] (let* [y (clojure.core/* x 2)] (clojure.core/+ y 1)))))))
  (testing "a let* nested in a binding VALUE reduces too"
    (is (= '(clojure.core/+ 3 1)
           (util/inline-pure-lets '(let* [a (let* [b 3] b)] (clojure.core/+ a 1)))))))

(deftest metadata-survives-the-rewrite
  (testing "clojure.walk drops meta on every rebuilt list. Here the TC tag on an untouched
            subform must survive — the bytecode emitter reads it, and losing it was a live
            ClassCastException in the SIMD-fusion path"
    (let [r (util/inline-pure-lets
             (list 'let* ['a 1]
                   (list 'clojure.core/+
                         (with-meta (list 'g 'a) {:raster.type/tag 'Double})
                         2)))]
      (is (= 'Double (:raster.type/tag (meta (second r)))))))
  (testing "…and losing it would defeat the purity gate itself, which reads
            :raster.op/original from meta"
    (let [devirt (with-meta (list '.invk 'impl 'O 0 1.0)
                   {:raster.op/original 'raster.arrays/aset})]
      (is (util/effectful? (util/inline-pure-lets (list 'do (list 'clojure.core/identity devirt))))
          "the stamp is still readable after a rewrite pass over the enclosing form"))))

(deftest a-multi-form-body-becomes-do-and-is-never-truncated
  (testing "keeping only (last body) drops the earlier forms — which are exactly the ones held
            for their effects. That was a vanished store, not a missed fusion"
    (is (= '(do (clojure.core/aset O 0 1) (clojure.core/+ 1 1))
           (util/inline-pure-lets
            '(let* [a 1] (clojure.core/aset O 0 a) (clojure.core/+ a 1)))))))

(deftest non-let-binders-are-left-alone
  (testing "loop* is not beta-reducible — its binders are re-assigned by recur"
    (is (= '(loop* [i 0] (recur (clojure.core/inc i)))
           (util/inline-pure-lets '(loop* [i 0] (recur (clojure.core/inc i)))))))
  (testing "quote is opaque"
    (is (= '(clojure.core/list (quote (let* [a 1] a)))
           (util/inline-pure-lets '(clojure.core/list (quote (let* [a 1] a))))))))

;; ── the gate: default REFUSES, and :on-impure must escape ────────────────────────────
(deftest the-default-gate-is-the-real-gate
  (testing "the default is (complement effectful?), NOT (constantly true) — the whole point of
            consolidating is that the two ungated callers gain a gate"
    (is (= :impure-binding
           (try (util/inline-pure-lets '(let* [a (raster.arrays/aset O 0 1.0)] a))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :impure-binding
           (try (util/inline-pure-lets
                 (list 'let* ['a (with-meta (list '.invk 'impl 'O 0 1.0)
                                   {:raster.op/original 'raster.arrays/aset})] 'a))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
        "including the devirtualized spelling"))
  (testing "a loop-valued init is refused: inlining it duplicates the whole loop per use"
    (is (= :impure-binding
           (try (util/inline-pure-lets '(let* [a (loop* [i 0] i)] a))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "an :on-impure that RETURNS instead of escaping is a bug, not a licence to inline —
            returning normally would fall through and inline the effect anyway"
    (is (= :raster/bug
           (try (util/inline-pure-lets '(let* [a (raster.arrays/aset O 0 1.0)] a)
                                       :on-impure (fn [_ _] :shrug))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "a caller may still widen the gate deliberately"
    (is (= '(raster.arrays/aset O 0 1.0)
           (util/inline-pure-lets '(let* [a (raster.arrays/aset O 0 1.0)] a)
                                  :pure? (constantly true))))))

;; ── anti-regression: each caller's refusal channel ──────────────────────────────────
;; The migration's failure mode is quiet: make the three signatures agree by dropping the gate, and
;; one loud refusal becomes two silent miscompiles. These pin that each caller still REFUSES.
(deftest each-caller-refuses-rather-than-inlining-an-effect
  (testing "the CPU SIMD emitter returns nil (⇒ scalar loop) for an effectful lane lambda"
    (let [inline-lets (var-get (requiring-resolve 'raster.compiler.backend.cpu.csimd/inline-lets))]
      (is (nil? (inline-lets '(let* [a (raster.arrays/aset O 0 1.0)] a)))
          "impure init ⇒ nil, not a per-lane duplicated store")
      (is (nil? (inline-lets '(let* [a 1] (clojure.core/aset O 0 a) (clojure.core/+ a 1))))
          "a multi-statement body ⇒ nil, not (last body) with the store silently dropped")
      (is (= '(clojure.core/+ (raster.arrays/aget A i) 1)
             (inline-lets '(let* [a (raster.arrays/aget A i)] (clojure.core/+ a 1))))
          "…and the pure case still flattens, so the money path is unchanged")))
  (testing "the GPU vectorizer bails to a scalar loop (its own ::bail marker), not an ex-info"
    (let [ipl (var-get (requiring-resolve 'raster.compiler.backend.gpu.c-emit/inline-pure-lets))]
      (is (true? (try (ipl '(let* [a (raster.arrays/aset O 0 1.0)] a))
                      nil
                      (catch clojure.lang.ExceptionInfo e
                        (:raster.compiler.backend.gpu.c-emit/bail (ex-data e)))))
          "vec-bail! is the refusal channel the enclosing try already catches")))
  (testing "the SOAC lowerer returns nil (⇒ ScalarBinding ⇒ legacy void path) for an effectful
            let-wrapped write, instead of duplicating the effect into a SoacMap lambda"
    (let [sav (var-get (requiring-resolve 'raster.compiler.ir.soac/single-aset-void))]
      (is (nil? (sav '(let* [t (raster.arrays/aset SIDE 0 1.0)]
                        (raster.arrays/aset OUT i t))
                     'i)))
      (is (some? (sav '(let* [g (raster.arrays/aget GATE i)]
                         (raster.arrays/aset OUT i g))
                      'i))
          "…and the pure case is still recognized, so fusion is unchanged"))))

;; ── the one remaining scope-blind substitution, now with a CHECKED precondition ──────
;; CSE keeps a blind substitution deliberately: measured ~4x faster than subst-syms on a
;; 300-binding ANF form, because subst-syms recomputes each smap value's free set at every binder.
;; That is a legitimate optimization — but its precondition ("the pipeline guarantees ANF flatness")
;; used to live only in a docstring. These pin that it is now verified, and that a violation takes
;; the capture-avoiding path instead of corrupting the form.
(deftest scope-blind-substitution-is-gated-on-a-verified-precondition
  (testing "a flat ANF let* with gensym-unique binders is safe — the money path keeps the fast route"
    (is (util/scope-blind-substitution-safe? '{t__1 a__1}
                                             '(let* [t__2 (clojure.core/* t__1 2)] t__2))))
  (testing "KEY capture: a binder rebinds an smap key. Blind substitution rewrites the BINDER"
    (is (not (util/scope-blind-substitution-safe?
              '{k v} '(raster.par/map! O k 4 nil (clojure.core/+ k 1))))))
  (testing "VALUE capture: a binder shadows a name free in an smap VALUE"
    (is (not (util/scope-blind-substitution-safe?
              '{t (clojure.core/* k 9)} '(raster.par/map! O k 4 nil (clojure.core/+ k t))))))
  (testing "quote: blind substitution would rewrite data as if it were code"
    (is (not (util/scope-blind-substitution-safe? '{a b} '(clojure.core/list (quote a))))))
  (testing "an empty smap is trivially safe"
    (is (util/scope-blind-substitution-safe? {} '(let* [a 1] a)))))

(deftest cse-substitution-falls-back-instead-of-corrupting
  (let [subst (var-get (requiring-resolve 'raster.compiler.passes.scalar.cse/subst-syms-in))]
    (testing "the flat-ANF fast path is unchanged"
      (is (= '(let* [t__2 (clojure.core/* a__1 2)] t__2)
             (subst '{t__1 a__1} '(let* [t__2 (clojure.core/* t__1 2)] t__2)))))
    (testing "a binder colliding with an smap key is LEFT ALONE, not rewritten into the value"
      (let [wf '(raster.par/map! O k 4 nil (clojure.core/+ k 1))]
        (is (= wf (subst '{k HIJACKED} wf)))))
    (testing "a value-capturing substitution alpha-renames the binder, so the substituted value
              still refers to the OUTER k — blind substitution produced (+ k (* k 9)), silently
              conflating the loop index with an unrelated outer variable"
      (let [r (subst '{t (clojure.core/* k 9)}
                     '(raster.par/map! O k 4 nil (clojure.core/+ k t)))
            idx (nth r 2)]
        (is (not= 'k idx) "the loop index was renamed")
        (is (= (list 'clojure.core/+ idx '(clojure.core/* k 9)) (nth r 5))
            "the body uses the RENAMED index, and the value's k is the outer one")))
    (testing "the fallback preserves par-form shape (rebuild is arity-faithful)"
      (let [wf '(raster.par/map! O k 4 nil (clojure.core/+ k t))]
        (is (= (count wf) (count (subst '{t (clojure.core/* k 9)} wf))))))))

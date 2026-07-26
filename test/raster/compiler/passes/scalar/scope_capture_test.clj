(ns raster.compiler.passes.scalar.scope-capture-test
  "Regression tests for scope-capture bugs in scalar passes, fixed by routing
   binder knowledge through raster.compiler.ir.form/scope-info (Phase 2)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.passes.scalar.resolve-alength :as ra]))

;; ================================================================
;; pe.clj — const-env must not capture a scope-local binder
;; ================================================================

(deftest pe-par-index-not-captured
  (testing "a par loop index is NOT substituted by a same-named outer constant"
    ;; BUG (pre-fix): par/* fell through to the general-call branch and PE'd the
    ;; body with the full const-env → {i 99} rewrote the loop index to 99.
    (let [r (pe/pe-pass '(raster.par/map! out i n cast (clojure.core/+ x i)) '{x 5 i 99})]
      (is (= 'i (nth r 2)) "loop index stays the binder i, not 99")
      ;; outer constant x IS applied; the loop-local i is preserved
      (is (= '(clojure.core/+ 5 i) (last r)) "x→5 applied; body i is the loop index")))
  (testing "par/reduce acc + idx are both protected"
    (let [r (pe/pe-pass '(raster.par/reduce acc init i n (clojure.core/+ acc i)) '{acc 1 i 2 init q})]
      (is (= '(clojure.core/+ acc i) (last r)) "acc and i stay scope-local")
      (is (= 'q (nth r 2)) "init (outer) is PE'd with full env"))))

(deftest pe-letfn-param-not-captured
  (testing "a letfn* fn param is not substituted by an outer constant of the same name"
    (let [r (pe/pe-pass '(letfn* [f (fn* [x] (clojure.core/+ x 1))] (f x)) '{x 7})]
      ;; the inner fn* param x is shadowed; the outer call (f x) sees the constant
      (is (= '(letfn* [f (fn* [x] (clojure.core/+ x 1))] (f 7)) r)
          "fn* body x not substituted; outer (f x)→(f 7)"))))

;; ================================================================
;; resolve_alength — a nested scope shadowing an alloc name must NOT
;; resolve (alength shadowed) to the OUTER allocation size
;; ================================================================

(deftest resolve-alength-respects-shadowing
  (testing "inner (alength buf) is not resolved to the outer buf's size"
    ;; BUG (pre-fix): size-map keyed by bare name with no shadowing → inner
    ;; (alength buf) resolved to the OUTER buf's size `n`.
    (let [out (:form (ra/resolve-alength-pass
                      '(let* [buf (clojure.core/double-array n)
                              r (let* [buf (clojure.core/double-array m)]
                                      (clojure.core/alength buf))]
                             r)))]
      ;; inner alength stays unresolved (inner buf's size isn't tracked) — the
      ;; key property is it must NOT have become the outer size `n`.
      (is (= '(clojure.core/alength buf)
             (last (nth (second out) 3)))
          "inner (alength buf) preserved, not resolved to outer n")))
  (testing "a NON-shadowing outer alength still resolves (behavior preserved)"
    (let [out (:form (ra/resolve-alength-pass
                      '(let* [buf (clojure.core/double-array n)
                              r (dense-into buf)]
                             (clojure.core/alength buf))))]
      (is (= 'n (last out)) "top-level (alength buf) resolves to its alloc size n"))))

;; ================================================================
;; pe.clj — const-env must not have its VALUES captured either
;; ================================================================
;; The tests above cover KEY capture: a binder that rebinds a const-env key. The other half is
;; VALUE capture: a const-env VALUE that is free in the enclosing scope but BOUND inside this one.
;; Substituting it moves the reference from the outer binding to the inner binder — a silent wrong
;; answer, with no key collision to notice. `pe` protected the first half and not the second.

(deftest pe-const-env-value-not-captured-by-par-index
  (testing "an alias to an OUTER i must not become the par loop's index"
    ;; `x` aliases the outer `i`; the par form binds its own `i`. Substituting x→i inside the loop
    ;; body silently reads the loop index instead of the outer value.
    (let [r (pe/pe-pass '(let* [x i
                               r (raster.par/map! out i n nil (clojure.core/+ x 1))]
                           r)
                        {})]
      (is (some #{'x} (flatten r))
          "x must survive inside the loop body (or be renamed) — it must NOT become the loop index")
      (is (not= '(clojure.core/+ i 1) (last (nth r 1)))
          "the body must not read the loop index in place of the outer i"))))

(deftest pe-const-env-value-not-captured-by-dotimes-index
  (testing "the same hazard on dotimes, substituting from a pre-seeded env"
    (let [r (pe/pe-pass '(dotimes [i n] (clojure.core/aset out i x)) '{x i})]
      (is (= '(clojure.core/aset out i x) (last r))
          "x must stay x: substituting it would make the store read the loop index"))))

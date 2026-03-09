(ns raster.compiler.review-regression-test
  "Regression tests for bugs found in the 2026-04-09 compiler review.
  Each test targets a specific issue (CC1-CC8, D2) from the review."
  (:refer-clojure :exclude [aget aset alength])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength]]
            [raster.compiler.core.inference :as inf]
            [raster.ad.reverse :as rev]))

;; ================================================================
;; CC3: qualify-body-symbols must preserve fn* param vector
;; ================================================================

(deftest cc3-qualify-fn-star-params
  (testing "single-arity fn* preserves parameter vector"
    (let [result (inf/qualify-body-symbols
                  '(fn* [x y] (raster.numeric/+ x y))
                  'user #{})]
      (is (= 'fn* (first result)))
      (is (vector? (second result))
          "param vector must be preserved")
      (is (= '[x y] (second result)))))

  (testing "multi-arity fn* preserves parameter vectors"
    (let [result (inf/qualify-body-symbols
                  '(fn* ([x] x) ([x y] (raster.numeric/+ x y)))
                  'user #{})]
      (is (= 'fn* (first result)))
      (is (vector? (first (second result)))
          "first arity param vector must be preserved")))

  (testing "fn* with destructuring preserves structure"
    (let [result (inf/qualify-body-symbols
                  '(fn* [a b c] (raster.numeric/* a (raster.numeric/+ b c)))
                  'user #{})]
      (is (= '[a b c] (second result))))))

;; ================================================================
;; CC2: loop AD must propagate adjoints through non-trivial init exprs
;; ================================================================

(deftm ^:private cc2-loop-mul-init [x :- Double] :- Double
  (loop [a (n/* x 2.0), i (long 0)]
    (if (< i 5) (recur (n/+ a 1.0) (inc i)) a)))

(deftm ^:private cc2-loop-sq-init [x :- Double] :- Double
  (loop [a (n/* x x), i (long 0)]
    (if (< i 3) (recur (n/+ a 1.0) (inc i)) a)))

(deftest cc2-loop-nontrivial-init-ad
  (testing "loop init (* x 2.0): grad must be 2.0, not 0.0"
    ;; f(x) = x*2 + 5, f'(x) = 2
    (let [[v pb] (rev/vjp #'cc2-loop-mul-init 3.0)]
      (is (= 11.0 v))
      (is (= [2.0] (pb 1.0)))))

  (testing "loop init (* x x): grad must be 2x, not 0.0"
    ;; f(x) = x^2 + 3, f'(x) = 2x
    (let [[v pb] (rev/vjp #'cc2-loop-sq-init 4.0)]
      (is (= 19.0 v))
      (is (= [8.0] (pb 1.0))))))

;; ================================================================
;; CC2b: loop AD must propagate adjoints through compound result exprs
;; ================================================================

(deftm ^:private cc2b-result-only [x :- Double] :- Double
  (loop [i (long 0)]
    (if (< i 3) (recur (inc i)) (n/+ (double i) x))))

(deftm ^:private cc2b-result-mul [x :- Double] :- Double
  (loop [a 0.0, i (long 0)]
    (if (< i 3) (recur (n/+ a 1.0) (inc i)) (n/* a x))))

(deftm ^:private cc2b-both [x :- Double] :- Double
  (loop [a (n/* x x), i (long 0)]
    (if (< i 3) (recur (n/+ a 1.0) (inc i)) (n/* a x))))

(deftest cc2b-loop-compound-result-ad
  (testing "compound result (+ i x): grad must be 1.0, not nil"
    ;; f(x) = 3 + x, f'(x) = 1
    (let [[v pb] (rev/vjp #'cc2b-result-only 5.0)]
      (is (= 8.0 v))
      (is (= [1.0] (pb 1.0)))))

  (testing "compound result (* a x): grad must be 3.0"
    ;; f(x) = 3 * x, f'(x) = 3
    (let [[v pb] (rev/vjp #'cc2b-result-mul 5.0)]
      (is (= 15.0 v))
      (is (= [3.0] (pb 1.0)))))

  (testing "non-trivial init AND compound result"
    ;; f(x) = (x^2+3)*x = x^3+3x, f'(x) = 3x^2+3
    (let [[v pb] (rev/vjp #'cc2b-both 2.0)]
      (is (= 14.0 v))
      (is (= [15.0] (pb 1.0))))))

;; ================================================================
;; CC10: ANF flattening must not drop side-effectful body forms
;; ================================================================

(deftest cc10-anf-flatten-preserves-effects
  (testing "walker.clj ANF flatten splices intermediate effects as bindings"
    (let [src (slurp "src/raster/compiler/core/walker.clj")
          ;; Must use butlast/last pattern, not just (last inner-body)
          idx (.indexOf src "ANF flattening")
          section (subs src idx (min (+ idx 800) (count src)))]
      (is (.contains section "butlast")
          "ANF flattening must use butlast to preserve intermediate forms"))))

;; ================================================================
;; CC5/CC4: dotimes AD backward must not double-invoke mapcat
;; and must use grad-acc (not raster.numeric/+) for nil-init accumulators
;; ================================================================

;; Note: The dotimes AD path has a deeper issue (d_out unbound for
;; scalar-only dotimes). These tests verify the structural fixes
;; (no ((mapcat ...)) and grad-acc usage) are in the generated code.

(deftest cc4-cc5-dotimes-ad-structural
  (testing "reverse.clj has no double-paren ((mapcat ...))"
    (let [src (slurp "src/raster/ad/reverse.clj")]
      (is (not (.contains src "((mapcat"))
          "must not have ((mapcat ...)) double-invocation")))

  (testing "dotimes backward uses grad-acc, not raster.numeric/+"
    ;; Check that the backward recur args use grad-acc for nil-safe accumulation
    (let [src (slurp "src/raster/ad/reverse.clj")
          ;; Find the bwd-recur-args sections in gen-reverse-dotimes and gen-reverse-par-reduce
          ;; They should use raster.ad.reverse/grad-acc, not raster.numeric/+
          dotimes-section (subs src
                                (.indexOf src "gen-reverse-dotimes")
                                (.indexOf src "gen-reverse-par-map"))
          par-reduce-section (subs src
                                   (.indexOf src "gen-reverse-par-reduce")
                                   (min (+ (.indexOf src "gen-reverse-par-reduce") 5000)
                                        (count src)))]
      ;; The bwd-recur-args in these sections should NOT use raster.numeric/+
      ;; for the scalar gradient accumulation (which starts from nil)
      (is (not (.contains dotimes-section "'raster.numeric/+"))
          "dotimes backward must use grad-acc, not raster.numeric/+")
      (is (not (.contains par-reduce-section "'raster.numeric/+"))
          "par-reduce backward must use grad-acc, not raster.numeric/+"))))

;; ================================================================
;; CC7: SIMD reduction identity must not default to 0.0 for unknown ops
;; ================================================================

(deftest cc7-reduction-identity-throws
  (testing "unknown reduction op throws instead of defaulting to 0.0"
    (is (thrown? clojure.lang.ExceptionInfo
                 ((requiring-resolve 'raster.compiler.backend.jvm.segop-simd/reduce-identity)
                  'unknown-op :double))
        "unknown op should throw, not default to 0.0")))

;; ================================================================
;; CC8: cast->elem-type must handle int and long
;; ================================================================

(deftest cc8-cast-elem-type-int-long
  (testing "loop-lift cast->elem-type-kw handles int/long"
    (let [cast-fn (requiring-resolve 'raster.compiler.passes.parallel.loop-lift/cast->elem-type-kw)]
      (is (= :float (cast-fn 'float)))
      (is (= :double (cast-fn 'double)))
      (is (= :long (cast-fn 'long)))
      (is (= :int (cast-fn 'int)))
      (is (nil? (cast-fn nil)))))

  (testing "par-simd cast->elem-type handles int/long"
    (let [cast-fn (requiring-resolve 'raster.compiler.backend.jvm.par-simd/cast->elem-type)]
      (is (= :float (cast-fn 'float)))
      (is (= :double (cast-fn 'double)))
      (is (= :long (cast-fn 'long)))
      (is (= :int (cast-fn 'int)))
      (is (nil? (cast-fn nil))))))

;; ================================================================
;; CC6: ref→double coercion must throw for non-Number, not produce 0.0
;; ================================================================

(deftm coerce-to-double [x :- Object] :- Double
  (double x))

(deftest cc6-ref-double-coercion
  (testing "Number objects correctly coerce to double"
    (is (= 42.0 (coerce-to-double 42)))
    (is (= 3.14 (coerce-to-double 3.14)))
    (is (= 1.0 (coerce-to-double (float 1.0)))))

  (testing "non-Number throws instead of producing 0.0"
    (is (thrown? ClassCastException
                 (coerce-to-double "not a number")))))

;; ================================================================
;; D2: buffer-fuse write-mode must default to :accumulate, not :overwrite
;; ================================================================

(deftest d2-buffer-fuse-conservative-default
  (testing "buffer-fuse source code defaults to :accumulate"
    (let [src (slurp "src/raster/compiler/passes/scalar/buffer_fuse.clj")]
      ;; The fallback when no write mode is registered should be :accumulate
      (is (not (.contains src "(or (:mode wm) :overwrite)"))
          "must not default to :overwrite for unregistered ops"))))

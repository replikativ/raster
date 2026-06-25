(ns raster.compiler.core.macroexpand-test
  "Unit tests for the shared metadata-preserving macroexpander used by both the
   JVM bytecode and wasm backends."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.macroexpand :as mex]))

(deftest preserves-metadata
  (testing "macroexpand-all-preserving keeps :raster.type/tag through expansion"
    (let [form (with-meta '(when c (foo)) {:raster.type/tag 'double})
          out  (mex/macroexpand-all-preserving form)]
      (is (= 'if (first out)))                       ; when → if
      (is (= 'double (:raster.type/tag (meta out)))))))

(deftest skip-head-keeps-form
  (testing "skip-head? leaves a matching form unexpanded but recurses into children"
    (let [form '(dotimes [i n] (when c (bar)))
          out  (mex/macroexpand-all-preserving form #(= "dotimes" (name %)))]
      (is (= 'dotimes (first out)))                  ; dotimes kept
      (is (= 'if (first (nth out 2)))))))            ; inner when → if

(deftest literal-key-case-macroexpands
  (testing "a case with literal integer keys expands to a case* (sanity)"
    (let [form '(case (int h) 0 10 1 20)
          out  (mex/macroexpand-all-preserving form)        ; case → (let* [g …] (case* g …))
          has-case*? (atom false)]
      (clojure.walk/postwalk (fn [x] (when (and (seq? x) (= 'case* (first x))) (reset! has-case*? true)) x) out)
      (is @has-case*? "macroexpanded form contains a case*"))))

(deftest quote-is-opaque
  (testing "macroexpand-all-preserving does NOT descend into quoted code-as-data"
    ;; Regression: the walker classifies (quote ...) as a :leaf and never walks it;
    ;; macroexpand-core runs after the walker and must honor the same contract, or it
    ;; rewrites macros/classes sitting in head position inside quoted data.
    (is (= '(quote (let [a 1] a)) (mex/macroexpand-all-preserving '(quote (let [a 1] a)))))
    (is (= '(quote (when x y))    (mex/macroexpand-all-preserving '(quote (when x y)))))
    (is (= '(quote (Foo. 1 2))    (mex/macroexpand-all-preserving '(quote (Foo. 1 2)))))
    (is (= '(quote (let [a 1] a)) (mex/macroexpand-core '(quote (let [a 1] a))))))
  (testing "a quote nested inside live code is left intact while LIVE siblings expand"
    (let [out (mex/macroexpand-core '(foo (quote (loop [i 0] i)) (when p q)))]
      ;; the quoted loop survives unexpanded somewhere in the expansion
      (let [found-quote (atom false) found-live-if (atom false)]
        (clojure.walk/postwalk
         (fn [x]
           (when (= x '(quote (loop [i 0] i))) (reset! found-quote true))
           ;; the live sibling (when p q) must expand to (if p (do q))
           (when (and (seq? x) (= 'if (first x)) (= 'p (second x))) (reset! found-live-if true))
           x)
         out)
        (is @found-quote "quoted (loop ...) left intact inside expanded code")
        (is @found-live-if "live sibling (when p q) still expands to if")))))

(deftest macroexpand-core-closes-binders-preserves-soac
  (testing "macroexpand-core closes binding forms but preserves SOAC/dotimes/ftm/interop"
    (is (= 'let*  (first (mex/macroexpand-core '(let [a 1] a)))))
    (is (= 'loop* (first (mex/macroexpand-core '(loop [i 0] i)))))
    (is (= 'if    (first (mex/macroexpand-core '(when c d)))))
    (is (= 'dotimes (first (mex/macroexpand-core '(dotimes [i n] (when c (bar)))))))
    (is (= 'if (first (nth (mex/macroexpand-core '(dotimes [i n] (when c (bar)))) 2)))) ; child expands
    ;; raster.par/* head preserved, but kernel body still expanded
    (let [out (mex/macroexpand-core '(raster.par/map! out i n nil (when c (bar))))]
      (is (= 'raster.par/map! (first out)))
      (is (= 'if (first (last out)))))
    ;; raster.params/* is NOT preserved (startsWith pitfall) — only an exact ns match
    (is (mex/core-skip-head? 'raster.par/reduce))
    (is (not (mex/core-skip-head? 'raster.params/defmodel)))))

(deftest resugar-interop-restores-method-sugar
  (testing "canonical (. obj method args) → (.method obj args)"
    (is (= '(.invk impl a b) (mex/resugar-interop '(. impl invk a b))))
    (is (= '(.invk impl a) (mex/resugar-interop '(. impl (invk a))))))   ; nested form
  (testing "recurses into nested forms and preserves meta"
    (let [form (with-meta '(let* [x (. impl invk a)] x) {:k 1})
          out  (mex/resugar-interop form)]
      (is (= '(let* [x (.invk impl a)] x) out))
      (is (= 1 (:k (meta out)))))))

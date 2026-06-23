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

(deftest resugar-interop-restores-method-sugar
  (testing "canonical (. obj method args) → (.method obj args)"
    (is (= '(.invk impl a b) (mex/resugar-interop '(. impl invk a b))))
    (is (= '(.invk impl a) (mex/resugar-interop '(. impl (invk a))))))   ; nested form
  (testing "recurses into nested forms and preserves meta"
    (let [form (with-meta '(let* [x (. impl invk a)] x) {:k 1})
          out  (mex/resugar-interop form)]
      (is (= '(let* [x (.invk impl a)] x) out))
      (is (= 1 (:k (meta out)))))))

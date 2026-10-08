(ns raster.compiler.passes.scalar.typed-arithmetic-simplification-test
  (:require [clojure.test :refer [deftest is]]
            [raster.numeric]
            [raster.compiler.passes.scalar.simplify :as simplify]
            [raster.compiler.passes.scalar.pe :as pe]))

(defn- float-add [a b]
  (with-meta
    (list '.invk
          (with-meta 'raster.numeric/_plus__m_float_float-impl
            {:raster.type/tag 'raster.fn.IFn__float_float
             :tag 'raster.fn.IFn__float_float
             :raster.type/ret-tag 'float})
          a b)
    {:raster.op/original 'raster.numeric/+ :raster.type/tag 'float}))

(deftest scalar-rewriting-preserves-selected-operand-conversions
  (let [form (float-add (float 1.0e8) 1.0)
        expected (eval form)]
    (doseq [rewrite [simplify/simplify-1 simplify/simplify #(pe/pe-pass % {}) #(pe/pe % {})]]
      (is (== (double expected) (double (eval (rewrite form))))))))

(deftest partial-evaluation-preserves-rounding-between-typed-operations
  (let [form (float-add (float-add (float 1.0e8) 1.0) (float -1.0e8))]
    (is (== 0.0 (double (eval form))))
    (doseq [rewrite [simplify/simplify #(pe/pe-pass % {}) #(pe/pe % {})]]
      (is (== 0.0 (double (eval (rewrite form))))))))

(deftest typed-ieee-identities-are-not-generic-algebra
  (let [multiply (with-meta
                   (list '.invk
                         (with-meta 'raster.numeric/_star__m_float_float-impl
                           {:raster.type/tag 'raster.fn.IFn__float_float
                            :raster.type/ret-tag 'float})
                         Float/POSITIVE_INFINITY (float 0.0))
                   {:raster.op/original 'raster.numeric/* :raster.type/tag 'float})]
    (doseq [form [(float-add (float -0.0) (float 0.0)) multiply]
            rewrite [simplify/simplify-1 simplify/simplify #(pe/pe-pass % {}) #(pe/pe % {})]]
      (let [actual (eval (rewrite form)) expected (eval form)]
        (is (instance? Float actual))
        (is (= (Float/floatToIntBits expected) (Float/floatToIntBits actual)))))))

(deftest typed-call-children-still-partially-evaluate
  (let [form (float-add '(clojure.core/+ 2 3) 'x)
        results [(pe/pe-pass form {}) (pe/pe form {})]]
    (doseq [result results]
      (is (= 5 (nth result 2)))
      (is (= '.invk (first result)))
      (is (= (meta form) (meta result)))
      (is (= (meta (second form)) (meta (second result)))))))

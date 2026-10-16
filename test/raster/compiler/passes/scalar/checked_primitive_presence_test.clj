(ns raster.compiler.passes.scalar.checked-primitive-presence-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]))

(deftest canonical-primitive-presence-uses-checked-source-lowering
  (is (= {:type :double} (scalar/checked-primitive-result '(raster.numeric/- x) '{x :double})))
  (is (= {:type :float} (scalar/checked-primitive-result '(raster.numeric/* x y) '{x :float y :float})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/- x) {})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/- nil) {})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/sqrt x x) '{x :double}))))

(deftest unknown-and-lexical-calls-cannot-borrow-a-numeric-name-or-tag
  (doseq [expression [(with-meta '(unknown x) {:raster.type/tag 'double})
                      '(identity nil)
                      '(user/_minus__m_double x)
                      '((fn* [] nil))
                      (with-meta '(raster.numeric/- x) {:raster.op/lexical-callee true})
                      '(- x)]]
    (is (nil? (scalar/checked-primitive-result expression '{x :double})))))

(deftest known-null-typed-bindings-do-not-prove-present-operands
  (let [x (with-meta 'x {:raster.type/tag 'double})
        source (list 'let* [x nil 'result (list 'raster.numeric/- x)]
                     '(clojure.core/nil? result))]
    (is (= source (products/lower-local-products source)))))

(deftest primitive-presence-does-not-replace-values-or-erase-traps
  (let [source '(let* [x 9223372036854775807 y 1 result (clojure.core/+ x y)]
                 (if (clojure.core/nil? result) :absent :present))
        lowered (products/lower-local-products source)
        invoke #(try ((eval (list 'fn* [] %))) nil (catch Throwable e (class e)))]
    (is (= :present (last lowered)))
    (is (= (second source) (second lowered)))
    (is (= ArithmeticException (invoke source) (invoke lowered)))))

(deftm checked-overflow-presence [] :- Long
  (let [a 9223372036854775807 b 1 value (n/+ a b)]
    (if (clojure.core/nil? value) 0 1)))

(deftest compiled-presence-retains-the-original-overflow
  (let [compiled (pipeline/compile-aot #'checked-overflow-presence :dtype :double)
        invoke #(try (%1) nil (catch Throwable e (class e)))]
    (is (= ArithmeticException (invoke checked-overflow-presence) (invoke compiled)))))

(deftest presence-flows-through-snapshots-without-depth-dependent-fixpoints
  (let [pairs (reduce (fn [bindings i]
                        (let [previous (symbol (str "v" (dec i)))
                              tuple (symbol (str "p" i)) value (symbol (str "v" i))]
                          (into bindings [tuple (list 'if (list 'clojure.core/nil? previous)
                                                      [nil] [(list 'raster.numeric/- previous)])
                                          value (list 'clojure.core/nth tuple 0)])))
                      '[v0 1.0] (range 1 10))
        source (list 'let* pairs 'v9)
        lowered (products/lower-local-products source)]
    (is (= -1.0 ((eval (list 'fn* [] source))) ((eval (list 'fn* [] lowered)))))
    (is (not-any? #{'if 'clojure.core/nil? 'clojure.core/nth} (tree-seq coll? seq lowered)))))

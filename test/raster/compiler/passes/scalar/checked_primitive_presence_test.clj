(ns raster.compiler.passes.scalar.checked-primitive-presence-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]))

(deftest canonical-primitive-presence-uses-checked-source-lowering
  (is (= {:type :double} (scalar/checked-primitive-result '(raster.numeric/- x) '{x :double})))
  (is (= {:type :float} (scalar/checked-primitive-result '(raster.numeric/* x y) '{x :float y :float})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/- x) {})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/- nil) {})))
  (is (nil? (scalar/checked-primitive-result '(raster.numeric/sqrt x x) '{x :double}))))

(deftest composed-canonical-presence-is-not-a-result-carrier-proof
  (doseq [expression ['(clojure.core/+ x y)
                      '(clojure.core/+ (clojure.core/double x) (clojure.core/double y))]]
    (is (= {:presence :present}
           (scalar/checked-scalar-normal-return expression '{x :float y :float}))))
  (doseq [expression ['(clojure.core/+ x nil)
                      '(clojure.core/+ x (identity y))
                      '(raster.numeric/+ x y)
                      (with-meta '(clojure.core/+ x y) {:raster.op/lexical-callee true})]]
    (is (nil? (scalar/checked-scalar-normal-return expression '{x :float y :float}))))
  (let [deep (nth (iterate #(list 'clojure.core/double %) 'x) 130)]
    (is (nil? (scalar/checked-scalar-normal-return deep '{x :double})))))

(deftest object-float-arithmetic-does-not-supply-payload-width
  (let [operand {:presence :present :present? true
                 :scalar-type :float :producer-scalar-type :float}
        known (#'products/binding-fact {'x operand 'y operand}
                                      'sum '(clojure.core/+ x y))
        evaluate (eval '(fn [^Object x ^Object y] (clojure.core/+ x y)))
        result (evaluate (float 16777216.0) (float 1.0))]
    (is (= Double (class result)))
    (is (= 16777217.0 result))
    (is (= :present (:presence known)))
    (is (not (contains? known :scalar-type)))
    (is (not (contains? known :producer-scalar-type)))))

(deftest unknown-and-lexical-calls-cannot-borrow-a-numeric-name-or-tag
  (doseq [expression [(with-meta '(unknown x) {:raster.type/tag 'double})
                      '(identity nil)
                      '(user/_minus__m_double x)
                      '((fn* [] nil))
                      (with-meta '(raster.numeric/- x) {:raster.op/lexical-callee true})
                      '(- x)]]
    (is (nil? (scalar/checked-primitive-result expression '{x :double})))))

(deftest unowned-generic-numeric-results-cannot-supply-a-payload-carrier
  ;; A generic numerical name is not proof of the selected implementation's
  ;; result width. Definition-owned selected calls have their separate receipt.
  (let [operand {:presence :present :present? true
                 :scalar-type :float :producer-scalar-type :float}
        known (#'products/binding-fact {'x operand} 'result '(raster.numeric/- x))]
    (is (nil? (:producer-scalar-type known)))))

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

(deftest consumed-presence-keeps-object-arithmetic-evaluation-through-dce
  (let [source (list 'let* ['value '(clojure.core/double x)
                            'saved (completion/complete-value completion/object-input-contract 'value)
                            'sum '(clojure.core// saved saved)
                            'alias 'sum]
                     '(let* [copy alias] (if (clojure.core/nil? copy) 0 1)))
        lowered (products/lower-local-products source)
        again (products/lower-local-products lowered)
        dced (:form (dce/eliminate-dead-bindings again))
        invoke (fn [form x]
                 (try ((eval (list 'fn* '[x] (completion/project-to-jvm-source form))) x)
                      (catch Throwable failure (class failure))))]
    (is (not-any? #{'clojure.core/nil?} (tree-seq coll? seq lowered)))
    (doseq [form [source lowered again dced]]
      (is (= ArithmeticException (invoke form 0.0)))
      (is (= 1 (invoke form 2.0))))))

(deftest presence-obligations-belong-to-binding-occurrences-not-names
  (let [source (list 'let* ['value '(clojure.core/double x)
                            'saved (completion/complete-value completion/object-input-contract 'value)
                            'result '(clojure.core/+ saved saved)]
                     '(let* [result (clojure.core// saved saved)]
                        (if (clojure.core/nil? result) 0 1)))
        lowered (products/lower-local-products source)
        outer-init (nth (second lowered) 5)
        inner-init (second (second (last lowered)))]
    (is (not (:raster.effect/preserve-evaluation (meta outer-init))))
    (is (true? (:raster.effect/preserve-evaluation (meta inner-init))))))

(deftest direct-presence-tests-retain-the-original-conversion
  (let [x (with-meta 'x {:raster.type/tag 'double})
        source (list 'let* [x nil] (list 'clojure.core/nil? (list 'clojure.core/double x)))
        lowered (products/lower-local-products source)
        initializer (second (second (last lowered)))
        invoke (fn [form]
                 (try ((eval (list 'fn* [] (completion/project-to-jvm-source form))))
                      (catch Throwable failure (class failure))))]
    (is (true? (:raster.effect/preserve-evaluation (meta initializer))))
    (is (#'frontend/requires-ordered-evaluation? initializer))
    (doseq [form [source lowered (:form (dce/eliminate-dead-bindings lowered))]]
      (is (= NullPointerException (invoke form))))))

(deftest scalar-accumulation-projection-retains-evaluation-obligations
  (let [addition (with-meta '(raster.ad.reverse/grad-acc left right)
                   {:raster.effect/preserve-evaluation true})
        source (list 'let* '[left (clojure.core/double x) right (clojure.core/double 1.0)]
                     addition)
        lowered (products/lower-local-products source)]
    (is (= 'clojure.core/+ (first (last lowered))))
    (is (true? (:raster.effect/preserve-evaluation (meta (last lowered)))))))

(deftm checked-overflow-presence [] :- Long
  (let [a 9223372036854775807 b 1 value (n/+ a b)]
    (if (clojure.core/nil? value) 0 1)))

(deftest compiled-presence-retains-the-original-overflow
  (let [compiled (pipeline/compile-aot #'checked-overflow-presence :dtype :double)
        invoke #(try (%1) nil (catch Throwable e (class e)))]
    (is (= ArithmeticException (invoke checked-overflow-presence) (invoke compiled)))))

(deftest explicit-carrier-presence-flows-without-depth-dependent-fixpoints
  (let [pairs (reduce (fn [bindings i]
                        (let [previous (symbol (str "v" (dec i)))
                              tuple (symbol (str "p" i)) value (symbol (str "v" i))]
                          (into bindings [tuple (list 'if (list 'clojure.core/nil? previous)
                                                      [nil] [(list 'clojure.core/double
                                                                   (list 'clojure.core/- previous))])
                                          value (list 'clojure.core/nth tuple 0)])))
                      '[v0 1.0] (range 1 10))
        source (list 'let* pairs 'v9)
        lowered (products/lower-local-products source)]
    ;; Internal Object completion is projected by the JVM owner, never invoked
    ;; as an ordinary Clojure function by a source-level oracle.
    (is (= -1.0 ((eval (list 'fn* [] source)))
           ((eval (list 'fn* [] (completion/project-to-jvm-source lowered))))))
    (is (not-any? #{'if 'clojure.core/nil? 'clojure.core/nth} (tree-seq coll? seq lowered)))))

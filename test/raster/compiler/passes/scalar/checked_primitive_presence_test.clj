(ns raster.compiler.passes.scalar.checked-primitive-presence-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.compiler.passes.scalar.cse :as cse]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]))

(defn- source-function [parameters expression]
  (binding [*unchecked-math* false]
    (eval (list 'fn* parameters
                (completion/project-to-jvm-source expression (source-policy/capture))))))

(defn- optimized-presence-source [source]
  (let [lowered (products/lower-local-products
                 source {:source-ns (the-ns 'raster.compiler.passes.scalar.checked-primitive-presence-test)
                         :completed-presence? true})]
    ;; Exercise the real scalar consumers after representation/presence use
    ;; changes, not merely direct evaluation of the local rewrite's result.
    (nth (iterate (fn [expression]
                    (-> expression inline/flatten-nested-lets pe/pe
                        cse/cse-let :form dce/eliminate-dead-bindings :form))
                  lowered) 3)))

(defn- outcome [f & arguments]
  (try {:value (apply f arguments)}
       (catch Throwable e {:error (class e)})))

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
        optimized (optimized-presence-source source)
        invoke #(try ((source-function [] %)) nil (catch Throwable e (class e)))]
    (is (= :present (last lowered)))
    (is (= (second source) (second lowered)))
    (is (= ArithmeticException (invoke source) (invoke lowered) (invoke optimized)))))

(deftm checked-overflow-presence [] :- Long
  (let [a 9223372036854775807 b 1 value (n/+ a b)]
    (if (clojure.core/nil? value) 0 1)))

(deftest compiled-presence-retains-the-original-overflow
  (let [compiled (pipeline/compile-aot #'checked-overflow-presence :dtype :double)
        invoke #(try (%1) nil (catch Throwable e (class e)))]
    (is (= ArithmeticException (invoke checked-overflow-presence) (invoke compiled)))))

(deftest presence-flows-through-snapshots-without-depth-dependent-fixpoints
  (let [pairs-for (fn [complete]
                   (reduce (fn [bindings i]
                        (let [previous (symbol (str "v" (dec i)))
                              tuple (symbol (str "p" i)) value (symbol (str "v" i))]
                          (into bindings [tuple (list 'if (list 'clojure.core/nil? previous)
                                                      [nil] [(complete (list 'raster.numeric/- previous))])
                                          value (list 'clojure.core/nth tuple 0)])))
                           '[v0 1.0] (range 1 10)))
        ;; A raw extensible helper has no selected return carrier. An actual
        ;; source conversion establishes presence only after it completes.
        source (list 'let* (pairs-for #(list 'clojure.core/double %)) 'v9)
        lowered (products/lower-local-products source)]
    (is (= -1.0 ((source-function [] source)) ((source-function [] lowered))))
    (is (not-any? #{'if 'clojure.core/nil? 'clojure.core/nth} (tree-seq coll? seq lowered)))
    (let [raw (list 'let* (pairs-for identity) 'v9)
          raw-lowered (products/lower-local-products raw)]
      (is (some #{'clojure.core/nil?} (tree-seq coll? seq raw-lowered))
          "an unselected helper must not inherit a physical return guarantee")
      (with-redefs [raster.numeric/- (fn [_] nil)]
        (is (nil? ((source-function [] raw))))
        (is (nil? ((source-function [] raw-lowered))))
        (doseq [converted [source lowered]]
          (is (thrown? NullPointerException ((source-function [] converted)))
              "conditional presence never discharges the actual conversion"))))))

(deftest consumed-presence-retains-partial-region-evaluation-through-scalar-passes
  (doseq [initializer ['(do (clojure.core// 1 n) 2.0)
                       '(let* [unused 1] (do (clojure.core// 1 n) 2.0))
                       '(if true (do (clojure.core// 1 n) 2.0) 3.0)
                       '(let* [boom (clojure.core// 1 n)] (do boom 2.0))]
          :let [source (list 'let* ['value initializer] '(clojure.core/nil? value))
                normalized (products/lower-local-products source)
                optimized (optimized-presence-source source)
                original (source-function '[n] source)
                actual (source-function '[n] optimized)]]
    (is (= false (last normalized)) "the new presence proof was actually consumed")
    (is (= {:error ArithmeticException} (outcome original 0) (outcome actual 0)))
    (is (= {:value false} (outcome original 1) (outcome actual 1)))))

(deftest consumed-presence-retains-transitive-opaque-and-number-producers
  (let [source '(let* [a (producer x events)
                       b (do a 2.0)]
                 (clojure.core/nil? b))
        original (source-function '[x events producer] source)
        actual (source-function '[x events producer] (optimized-presence-source source))
        producer (fn [x events]
                   (swap! events conj :producer)
                   (double x))
        number (proxy [Number] []
                 (intValue [] 1) (longValue [] 1) (floatValue [] 1.0)
                 (doubleValue [] (throw (IllegalArgumentException. "actual Number conversion"))))]
    (is (= false (last (products/lower-local-products source))))
    (doseq [x [2.0 number nil]]
      (let [before (atom []) after (atom [])]
        (is (= (outcome original x before producer) (outcome actual x after producer)))
        (is (= [:producer] @before @after))))
    (is (= {:error IllegalArgumentException} (outcome actual number (atom []) producer))))
  (let [source '(let* [a (clojure.core// 1 n) b (do a 2.0)] (clojure.core/nil? b))]
    (is (= {:error ArithmeticException}
           (outcome (source-function '[n] (optimized-presence-source source)) 0)))))

(deftest eager-fields-and-direct-float-presence-retain-original-failures
  (doseq [tail ['(clojure.core/float x) '(clojure.core// 1 n)]
          :let [source (list 'let* ['p [(list 'do '(swap! events conj :first) 1.0) tail]]
                             '(clojure.core/nth p 0))
                original (source-function '[x n events] source)
                actual (source-function '[x n events] (optimized-presence-source source))]]
    (doseq [[x n] [[Double/MAX_VALUE 0] [2.0 1]]]
      (let [before (atom []) after (atom [])]
        (is (= (outcome original x n before) (outcome actual x n after)))
        (is (= [:first] @before @after))
        (when (= x Double/MAX_VALUE)
          (is (= {:error (if (= tail '(clojure.core/float x))
                          IllegalArgumentException ArithmeticException)}
                 (outcome actual x n (atom []))))))))
  (doseq [source ['(clojure.core/nil? (clojure.core/float x))
                  '(let* [value (clojure.core/float x)] (clojure.core/nil? value))]
          :let [original (source-function '[x] source)
                actual (source-function '[x] (optimized-presence-source source))]]
    (doseq [x [Double/MAX_VALUE nil 2.0]]
      (is (= (outcome original x) (outcome actual x))))
    (is (= {:error IllegalArgumentException} (outcome actual Double/MAX_VALUE)))))

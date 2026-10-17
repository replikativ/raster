(ns raster.compiler.passes.scalar.local-products-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.local-products :as products]))

(deftm projected-product [x :- Double] :- Double
  (let [seed 1.0
        values (if (clojure.core/nil? seed) [nil nil]
                   (let [scaled (n/* x 2.0)] [x scaled]))]
    (n/+ (clojure.core/nth values 0) (clojure.core/nth values 1))))

(deftest local-products-share-the-jvm-and-native-front-end
  (doseq [target [:jvm :c]]
    (let [compiled (pipeline/compile-aot #'projected-product :target target :dtype :double)]
      (try
        (doseq [x [2.0 -3.0 1.0e8]]
          (is (= (projected-product x) (compiled x))))
        (finally (when (instance? java.lang.AutoCloseable compiled)
                   (.close ^java.lang.AutoCloseable compiled)))))))

(defn- run [expression & arguments]
  (apply (eval (list 'fn* '[x calls] expression)) arguments))

(deftest conditional-products-preserve-selected-effects-and-lexical-scope
  (let [source '(let* [outside (do (swap! calls conj :prefix) 7.0)
                       p (if (do (swap! calls conj :test) (pos? x))
                           (let* [outside (do (swap! calls conj :then) 100.0)]
                             [outside (do (swap! calls conj :field) x)])
                           (let* [outside (do (swap! calls conj :else) -100.0)]
                             [outside (do (swap! calls conj :field) x)]))
                       alias p
                       value (+ outside (clojure.core/nth alias 0)
                                (clojure.core/nth p 1))]
                 (swap! calls conj :continuation)
                 value)
        lowered (products/lower-local-products source)]
    (doseq [[x expected path] [[2.0 109.0 :then] [-2.0 -95.0 :else]]]
      (let [a (atom []) b (atom [])]
        (is (= expected (run source x a) (run lowered x b)))
        (is (= [:prefix :test path :field :continuation] @a @b))))
    (is (not-any? #{'clojure.core/nth} (tree-seq coll? seq lowered)))))

(deftest conditional-fields-retain-selected-traps-and-do-not-replay-effects
  (let [source '(let* [p (if (do (swap! calls conj :test) (pos? x))
                           [(do (swap! calls conj :then) (/ 1 0))]
                           [(do (swap! calls conj :else) (/ 1 0))])]
                 (swap! calls conj :unreachable)
                 (clojure.core/nth p 0))
        lowered (products/lower-local-products source)]
    (doseq [[x path] [[1.0 :then] [-1.0 :else]]]
      (let [a (atom []) b (atom [])
            failure (fn [expression calls]
                      (try (run expression x calls) nil
                           (catch Throwable e (class e))))]
        (is (= ArithmeticException (failure source a) (failure lowered b)))
        (is (= [:test path] @a @b))))))

(deftest conditional-product-joins-decline-unproved-or-escaping-shapes
  (doseq [source ['(let* [p (if x [1.0] nil)] (clojure.core/nth p 0))
                  '(let* [p (if x [1.0] (opaque-result))] (clojure.core/nth p 0))
                  '(let* [p (if x [1.0] [2.0 3.0])] (clojure.core/nth p 0))
                  '(let* [p (if x [1.0])] (clojure.core/nth p 0))
                  '(let* [p (if x [1.0] [2.0])] (identity p))
                  '(let* [p (if x [1.0] [2.0])] (clojure.core/nth p x))]]
    (is (= source (products/lower-local-products source)))))

(deftest conditional-product-growth-budget-rolls-back-atomically
  (let [source '(let* [p (if x [1.0] [2.0])]
                 (+ (clojure.core/nth p 0) (clojure.core/nth p 0)))]
    (with-redefs-fn {#'products/conditional-expansion-limit 1}
      #(is (= source (products/lower-local-products source))))))

(deftest nested-product-joins-preserve-loop-recur-scope
  (let [source '(loop* [i 0 acc 0.0]
                 (if (< i 2)
                   (let* [p (if (pos? x)
                              (if (zero? i) [1.0 9.0] [2.0 18.0])
                              [3.0 27.0])]
                     (swap! calls conj i)
                     (recur (inc i) (+ acc (clojure.core/nth p 0)
                                      (clojure.core/nth p 1))))
                   acc))
        lowered (products/lower-local-products source)]
    (doseq [[x expected] [[1.0 30.0] [-1.0 60.0]]]
      (let [a (atom []) b (atom [])]
        (is (= expected (run source x a) (run lowered x b)))
        (is (= [0 1] @a @b))))))

(deftest conditional-product-fields-can-be-captured-without-capturing-arm-locals
  (let [source '(let* [outside 7.0
                       p (if (pos? x) (let* [outside 100.0] [outside]) [200.0])
                       f (fn* [x] (+ outside x (clojure.core/nth p 0)))]
                 (f 3.0))
        lowered (products/lower-local-products source)]
    (doseq [[x expected] [[1.0 110.0] [-1.0 210.0]]]
      (is (= expected (run source x (atom [])) (run lowered x (atom [])))))))

(deftest lexical-presence-does-not-execute-or-classify-calls
  (is (= '(let* [a 1.0 b a] :present)
         (products/lower-local-products
          '(let* [a 1.0 b a] (if (clojure.core/nil? b) :absent :present)))))
  (doseq [source ['(clojure.core/nil? x)
                  '(clojure.core/nil? (identity nil))
                  '(clojure.core/nil? (do (swap! calls inc) nil))]]
    (is (= source (products/lower-local-products source))))
  (is (= :present (products/lower-local-products '(if 0 :present :absent))))
  (is (= :absent (products/lower-local-products '(if nil :present :absent)))))

(deftest fields-are-snapshotted-once-in-source-order
  (let [source '(let* [p [(do (swap! calls conj :first) x)
                          (do (swap! calls conj :second) (* x 3.0))]
                       alias p]
                 (+ (clojure.core/nth alias 0) (clojure.core/nth p 1)
                    (clojure.core/nth p 0)))
        lowered (products/lower-local-products source)]
    (doseq [x [2.0 -0.0]]
      (let [a (atom []) b (atom [])]
        (is (= (run source x a) (run lowered x b)))
        (is (= [:first :second] @a @b))))
    (is (not-any? #{'p 'alias} (tree-seq coll? seq lowered)))))

(deftest constructor-scope-cannot-capture-later-bindings
  (let [source '(let* [p (let* [x (do (swap! calls inc) 3.0)] [x (* x 2.0)])
                       x 100.0]
                 (+ (clojure.core/nth p 0) x))
        lowered (products/lower-local-products source)
        a (atom 0) b (atom 0)]
    (is (= 103.0 (run source 9.0 a) (run lowered 9.0 b)))
    (is (= 1 @a @b))))

(deftest aliases-retain-the-old-snapshot-after-shadowing
  (let [source '(let* [p [x] alias p p [100.0]]
                 (+ (clojure.core/nth alias 0) (clojure.core/nth p 0)))
        lowered (products/lower-local-products source)]
    (is (= 102.0 (run source 2.0 nil) (run lowered 2.0 nil)))))

(deftest closure-and-loop-binders-mask-lexical-facts
  (let [source '(let* [x nil p [10.0]]
                 ((fn* [x p] (if (clojure.core/nil? x) 0.0 (clojure.core/nth p 0)))
                  1.0 [20.0]))
        lowered (products/lower-local-products source)]
    (is (= 20.0 (run source 2.0 nil) (run lowered 2.0 nil))))
  (let [source '(let* [i nil] (loop* [i 0] (if (< i 3) (recur (inc i)) i)))]
    (is (= 3 (run (products/lower-local-products source) nil nil)))))

(deftest escapes-and-invalid-projections-retain-allocations-and-traps
  (doseq [source ['(let* [p [x]] p)
                  '(let* [p [x]] (identity p))
                  '(let* [p [x]] (clojure.core/nth p x))
                  '(let* [p [x]] (clojure.core/nth p 2))
                  '(let* [p [x]] (clojure.core/nth p 2 :missing))]]
    (is (= source (products/lower-local-products source))))
  (let [source '(let* [p [(do (swap! calls inc) x)]]
                 (if (clojure.core/nil? p) :bad :good))
        calls (atom 0)]
    (is (= :good (run (products/lower-local-products source) nil calls)))
    (is (= 1 @calls))))

(deftest known-seed-removes-an-unknown-product-branch-without-arithmetic-rewrites
  (let [source '(let* [dy 1.0
                       result (if (clojure.core/nil? dy) [nil nil]
                                  (let* [g (* dy x)] [g g]))]
                 (+ (clojure.core/nth result 0) (clojure.core/nth result 1)))
        lowered (products/lower-local-products source)]
    (is (= 6.0 (run lowered 3.0 nil)))
    (is (not-any? #{'clojure.core/nil? 'clojure.core/nth} (tree-seq coll? seq lowered)))))

(deftest retained-literal-bindings-are-not-substituted-into-arithmetic
  (let [x (with-meta 'x {:raster.type/tag 'float})
        source (list 'let* [x 1.00000006] (list 'raster.numeric/* x x))
        lowered (products/lower-local-products source)]
    (is (= source lowered))
    (is (= 'float (:raster.type/tag (meta (first (second lowered))))))
    (is (every? symbol? (rest (last lowered))))))

(deftest lexical-callees-do-not-acquire-core-projection-or-presence-semantics
  (doseq [expression [(with-meta '(clojure.core/nil? x) {:raster.op/lexical-callee true})
                      (with-meta '(clojure.core/nth p 0) {:raster.op/lexical-callee true})]]
    (let [source (list 'let* '[x nil p [3.0]] expression)]
      (is (= expression (last (products/lower-local-products source))))
      (is (= (meta expression) (meta (last (products/lower-local-products source))))))))

(deftest primitive-cast-presence-retains-conversion-effects-and-traps
  (let [source '(let* [p (if (clojure.core/nil? (clojure.core/double
                                               (do (swap! calls inc) x)))
                           [10.0] [20.0])]
                 (clojure.core/nth p (clojure.core/long 0)))
        lowered (products/lower-local-products source)
        exception-class (fn [expression calls]
                          (try (run expression nil calls) nil (catch Throwable e (class e))))]
    (let [a (atom 0) b (atom 0)]
      (is (= 20.0 (run source 1.0 a) (run lowered 1.0 b)))
      (is (= 1 @a @b)))
    (let [a (atom 0) b (atom 0)]
      (is (= NullPointerException (exception-class source a) (exception-class lowered b)))
      (is (= 1 @a @b)))))

(deftest failed-index-conversions-do-not-discharge-constructor-effects
  (let [source '(let* [p [(do (swap! calls inc) x)]]
                 (clojure.core/nth p (clojure.core/int 9223372036854775807)))
        lowered (products/lower-local-products source)
        invoke (fn [expression calls]
                 (try (run expression 1.0 calls) nil (catch Throwable e (class e))))
        a (atom 0) b (atom 0)]
    (is (= source lowered))
    (is (= ArithmeticException (invoke source a) (invoke lowered b)))
    (is (= 1 @a @b))))

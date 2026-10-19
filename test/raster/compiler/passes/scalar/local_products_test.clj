(ns raster.compiler.passes.scalar.local-products-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.local-products :as products]))

(deftm projected-product [x :- Double] :- Double
  (let [seed 1.0
        values (if (clojure.core/nil? seed) [nil nil]
                   (let [scaled (n/* x 2.0)] [x scaled]))]
    (n/+ (clojure.core/nth values 0) (clojure.core/nth values 1))))

(deftm scalar-square [x :- Double] :- Double (n/* x x))

(deftm scalar-square-gradient [x :- Double] :- Double
  (nth ((reverse/value+grad #'scalar-square) x) 1))

(deftest scalar-ad-fanin-compiles-through-the-shared-native-and-jvm-front-end
  (doseq [target [:jvm :c]]
    (let [compiled (pipeline/compile-aot #'scalar-square-gradient :target target :dtype :double)]
      (try
        (doseq [x [0.0 2.0 -3.5 Math/PI]]
          (is (= (* 2.0 x) (double (compiled x)))))
        (finally (when (instance? java.lang.AutoCloseable compiled)
                   (.close ^java.lang.AutoCloseable compiled)))))))

(deftest local-products-share-the-jvm-and-native-front-end
  (doseq [target [:jvm :c]]
    (let [compiled (pipeline/compile-aot #'projected-product :target target :dtype :double)]
      (try
        (doseq [x [2.0 -3.0 1.0e8]]
          (is (= (projected-product x) (compiled x))))
        (finally (when (instance? java.lang.AutoCloseable compiled)
                   (.close ^java.lang.AutoCloseable compiled)))))))

(defn- run [expression & arguments]
  (apply (eval (list 'fn* '[x calls] (completion/project-to-jvm-source expression))) arguments))

(deftm observed-product-field [x :- Double] :- Double
  (let [product [(clojure.core/* x x)]]
    (if (identical? (nth product 0) (nth product 0)) 1.0 0.0)))

(deftm observed-product-aliases [x :- Double] :- Double
  (let [product [(clojure.core/* x x)]
        saved (nth product 0)
        copy saved]
    (if (identical? saved copy) 1.0 0.0)))

(deftest product-fields-retain-the-original-object-snapshot
  (doseq [function [#'observed-product-field #'observed-product-aliases]]
    (let [compiled (pipeline/compile-aot function :dtype :double)]
      (doseq [x [-3.5 -0.0 0.0 Math/PI]]
        (is (= 1.0 (function x) (compiled x))))))
  (let [source '(let* [p [(clojure.core/* (clojure.core/double x) 2.0)]
                       q p
                       first-field (clojure.core/nth q 0)
                       second-field (clojure.core/nth p 0)]
                 (clojure.core/identical? first-field second-field))
        lowered (products/lower-local-products source)]
    (is (completion/contains-completion? lowered))
    (doseq [x [-3.5 -0.0 0.0 Math/PI]]
      (is (true? (run source x nil)))
      (is (true? (run lowered x nil))))))

(deftest object-snapshots-compose-with-reference-consumers-and-captures
  ;; Identity is not peculiar to identical?: storing a key, returning a field
  ;; from a closure, and selecting a product arm all observe the same snapshot.
  (doseq [source
          ['(let* [p [(clojure.core/double x)]
                   saved (fn* [] (clojure.core/nth p 0))]
              (clojure.core/identical? (saved) (saved)))
           '(let* [p (if (pos? x) [(clojure.core/double x)]
                         [(clojure.core/double (- x))])
                   keys (java.util.IdentityHashMap.)]
              (.put keys (clojure.core/nth p 0) :saved)
              (= :saved (.get keys (clojure.core/nth p 0))))
           '(let* [p [(clojure.core/double x)]
                   saved (clojure.core/nth p 0)]
              (loop* [i 0]
                (if (< i 3) (recur (inc i))
                    (clojure.core/identical? saved (clojure.core/nth p 0)))))]]
    (let [lowered (products/lower-local-products source)]
      (doseq [x [-3.5 -0.0 0.0 Math/PI]]
        (is (true? (run source x nil)))
        (is (true? (run lowered x nil))))))
  (let [source '(let* [p [(clojure.core/double x) (clojure.core/double x)]]
                 (clojure.core/identical? (clojure.core/nth p 0)
                                         (clojure.core/nth p 1)))
        lowered (products/lower-local-products source)]
    (doseq [x [-3.5 -0.0 0.0 Math/PI]]
      (is (false? (run source x nil)))
      (is (false? (run lowered x nil))))))

(deftest snapshot-completion-preserves-failure-before-later-field-effects
  (let [source '(let* [p [(clojure.core/double x)
                         (do (swap! calls conj :later) 2.0)]]
                 (clojure.core/nth p 0))
        lowered (products/lower-local-products source)
        invoke (fn [expression calls]
                 (try (run expression nil calls) :returned
                      (catch Throwable failure (class failure))))
        a (atom []) b (atom [])]
    (is (= NullPointerException (invoke source a) (invoke lowered b)))
    (is (= [] @a @b))))

(deftest scalar-accumulation-lowering-requires-independent-present-carriers
  (let [source '(let* [left (clojure.core/double x)
                       right (clojure.core/double 2.0)]
                  (raster.ad.reverse/grad-acc left right))
        lowered (products/lower-local-products source)]
    (is (some #{'clojure.core/+} (tree-seq coll? seq lowered)))
    (is (not-any? #{'raster.ad.reverse/grad-acc} (tree-seq coll? seq lowered)))
    (doseq [x [0.0 -0.0 Math/PI -2.0]]
      (is (= (run source x nil) (run lowered x nil))))
    (is (thrown? NullPointerException (run lowered nil nil))))
  (doseq [source ['(raster.ad.reverse/grad-acc x 1.0)
                  '(raster.ad.reverse/grad-acc nil 1.0)
                  '(raster.ad.reverse/grad-acc (clojure.core/float x)
                                             (clojure.core/double 1.0))
                  '(raster.ad.reverse/grad-acc [1.0] [2.0])]]
    (is (= source (products/lower-local-products source))
        "unknown, absent, mixed-width and product operands are not scalar proofs")))

(deftest scalar-accumulation-does-not-change-effects-or-association
  (let [source '(let* [a (clojure.core/double (do (swap! calls conj :a) x))
                       b (clojure.core/double (do (swap! calls conj :b) 1.0))
                       c (clojure.core/double (do (swap! calls conj :c) -1.0e16))]
                  (raster.ad.reverse/grad-acc
                   (raster.ad.reverse/grad-acc a b) c))
        lowered (products/lower-local-products source)
        before (atom []) after (atom [])]
    (is (= (run source 1.0e16 before) (run lowered 1.0e16 after)))
    (is (= [:a :b :c] @before @after))))

(deftest float-scalar-accumulation-keeps-the-existing-float-addition
  (let [source '(let* [left (clojure.core/float x)
                       right (clojure.core/float 1.0)]
                  (raster.ad.reverse/grad-acc left right))
        lowered (products/lower-local-products source)]
    (is (some #{'raster.numeric/+} (tree-seq coll? seq lowered)))
    (is (not-any? #{'raster.ad.reverse/grad-acc} (tree-seq coll? seq lowered)))
    (doseq [x [0.0 -0.0 Math/PI 16777216.0]]
      (is (= (Float/floatToIntBits (float (run source x nil)))
             (Float/floatToIntBits (float (run lowered x nil))))))))

(deftest scalar-accumulation-refuses-forged-width-or-replaced-helper
  (let [binder (with-meta 'left {:raster.type/tag 'double})
        source (list 'let* [binder '(clojure.core/float x)]
                     '(raster.ad.reverse/grad-acc left (clojure.core/double 2.0)))]
    (is (some #{'raster.ad.reverse/grad-acc}
              (tree-seq coll? seq (products/lower-local-products source)))))
  (let [source '(raster.ad.reverse/grad-acc (clojure.core/double x)
                                          (clojure.core/double 2.0))]
    (with-redefs [reverse/grad-acc (fn [& _] (throw (AssertionError. "must not execute helper")))]
      (is (= source (products/lower-local-products source))))))

(deftest selected-nil-tests-refine-repeated-optional-products-without-growing-the-budget
  (let [ids (mapv #(symbol (str "p" %)) (range 30))
        bindings (vec (mapcat (fn [[previous id]]
                               [id (list 'if (list 'clojure.core/nil? previous) [nil]
                                         (list 'let* ['tick '(swap! calls inc)] [previous]))
                                (symbol (str id "-field")) (list 'clojure.core/nth id 0)])
                             (map vector (cons 'x (map #(symbol (str % "-field")) (butlast ids))) ids)))
        source (list 'let* bindings (symbol (str (last ids) "-field")))
        lowered (products/lower-local-products source)]
    (is (not-any? #{'clojure.core/nth} (tree-seq coll? seq lowered)))
    (doseq [x [nil false 0.0 2.0]]
      (let [a (atom 0) b (atom 0)]
        (is (= x (run source x a) (run lowered x b)))
        (is (= (if (nil? x) 0 30) @a @b))))))

(deftest nil-test-refinement-does-not-leak-through-shadowing-or-lexical-callees
  (let [source '(if (clojure.core/nil? x)
                 (let* [x 2.0] (clojure.core/nil? x))
                 ((fn* [x] (clojure.core/nil? x)) nil))
        lowered (products/lower-local-products source)]
    (doseq [x [nil 1.0]]
      (is (= (run source x (atom 0)) (run lowered x (atom 0))))))
  (let [test (with-meta '(clojure.core/nil? x) {:raster.op/lexical-callee true})
        source (list 'if test '(clojure.core/nil? x) false)]
    (is (= source (products/lower-local-products source)))))

(deftest strict-cast-bindings-prove-only-normal-return-presence
  (let [source '(let* [value (clojure.core/double (do (swap! calls inc) x))]
                 (if (clojure.core/nil? value) :absent :present))
        lowered (products/lower-local-products source)]
    (is (not-any? #{'clojure.core/nil?} (tree-seq coll? seq lowered)))
    (let [a (atom 0) b (atom 0)]
      (is (= :present (run source 2.0 a) (run lowered 2.0 b)))
      (is (= 1 @a @b)))
    (let [a (atom 0) b (atom 0)]
      (is (thrown? NullPointerException (run source nil a)))
      (is (thrown? NullPointerException (run lowered nil b)))
      (is (= 1 @a @b))))
  (let [cast (with-meta '(clojure.core/double x) {:raster.op/lexical-callee true})
        source (list 'let* ['value cast] '(clojure.core/nil? value))]
    (is (= source (products/lower-local-products source)))))

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

(ns raster.compiler.core.walker-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.walker :as walker]
            [raster.compiler.core.inference :as inf]
            [raster.core :as core :refer [deftm ftm]]
            [raster.arrays]
            [raster.sci.distributions]
            [raster.numeric]))

;; Helper: build type-env from flat {sym → tag}
(defn- te [flat-env]
  (reduce-kv (fn [m s t] (assoc m s {:tag t})) {} flat-env))

(defn- wb
  "Walk a form with a flat env."
  ([form] (walker/walk-body form {:type-env {}}))
  ([form flat-env] (walker/walk-body form {:type-env (te flat-env)})))

(deftest contextual-loop-seed-retains-canonical-source-and-binding-type
  (let [body (walker/walk-body '(loop* [seed 0.0 i 0]
                               (if (< i 1) (recur (raster.numeric/+ seed x) (inc i)) seed))
                             {:type-env (te {'x 'float}) :element-dtype :float})]
    (is (= 'float (:raster.type/tag (meta (first (second body))))))
    (is (= 0.0 (second (second body)))
        "backend realization does not rewrite AD/SOAC source identities")))

(deftest comparison-types-survive-and-macroexpansion
  (let [walked (wb '(and (>= i 0) (< i n)) {'i 'long 'n 'long})
        [_ bindings conditional] walked
        [binder initializer] bindings]
    (is (= 'let* (first walked)))
    (is (= 'boolean (:raster.type/tag (meta initializer))))
    (is (= 'boolean (:raster.type/tag (meta binder))))
    (is (nil? (:tag (meta binder))) "Raster typing does not add a JVM primitive hint")
    (is (= 'if (first conditional)))
    (is (= 'boolean (:raster.type/tag (meta (nth conditional 2)))))))

(deftest qualified-array-read-types-its-consuming-scalar-call
  (let [walked (wb '(raster.numeric/+ sum
                       (raster.sci.distributions/logpdf
                        (raster.sci.distributions/->Normal mu sigma)
                        (raster.arrays/aget ys i)))
                   {'sum 'double 'mu 'double 'sigma 'double
                    'ys 'doubles 'i 'long})]
    (is (= 'double (:raster.type/tag (meta walked))))
    (is (= 'raster.numeric/+ (:raster.op/original (meta walked))))
    (is (= 'raster.sci.distributions/logpdf
           (:raster.op/original (meta (last walked)))))))

;; ================================================================
;; classify-form
;; ================================================================

(deftest loop-initializer-lets-do-not-add-recur-parameters
  (let [walked (wb '(loop* [i 0 acc (let* [seed (+ i 2)] seed)]
                     (if (< i 3) (recur (inc i) (+ acc i)) acc)))]
    (is (= ['i 'acc] (vec (take-nth 2 (second walked)))))
    (is (= 'let* (first (nth (second walked) 3))))
    (is (= 5 (eval walked)))))

(deftest operand-return-contract-is-carried-into-following-bindings
  (doseq [init ['(raster.par/contract out [[i n]] [[j n]] 1.0)
                '(raster.par/reduce-by-key out keys values n +)]]
    (let [walked (wb (list 'let* ['result init] 'result)
                     {'out 'floats 'keys 'ints 'values 'floats 'n 'long})
          binding (first (second walked))]
      (is (= 'floats (:raster.type/tag (meta binding))))
      (is (= 'floats (:raster.type/tag (meta (last walked))))))))

(deftest scalar-and-unknown-par-calls-do-not-inherit-array-return-tags
  (doseq [expression ['(raster.par/atomic-add! out 0 1)
                      '(raster.par/unknown! out 0 1)]]
    (is (nil? (:raster.type/tag (meta (wb expression {'out 'ints})))))))

(deftest classify-let-test
  (testing "let forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :let (walker/classify-form '(let [x 1] x) ctx)))
      (is (= :let (walker/classify-form '(let* [x 1] x) ctx)))
      (is (= :let (walker/classify-form '(loop [i 0] i) ctx))))))

(deftest classify-fn-test
  (testing "fn forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :fn (walker/classify-form '(fn [x] x) ctx)))
      (is (= :fn (walker/classify-form '(fn* [x] x) ctx))))))

(deftest classify-leaf-test
  (testing "literals and symbols"
    (let [ctx (walker/make-ctx {})]
      (is (= :leaf (walker/classify-form 42 ctx)))
      (is (= :leaf (walker/classify-form 3.14 ctx)))
      (is (= :leaf (walker/classify-form 'x ctx)))
      (is (= :leaf (walker/classify-form nil ctx))))))

(deftest classify-deftm-call-test
  (testing "deftm function calls"
    (let [ctx (walker/make-ctx {:type-env (te {'x 'double 'y 'double})
                                :source-ns (the-ns 'raster.compiler.core.walker-test)})]
      (is (= :deftm-call (walker/classify-form '(raster.numeric/+ x y) ctx))))))

(deftest classify-array-op-test
  (testing "array operations"
    (let [ctx (walker/make-ctx {:type-env (te {'a 'doubles})
                                :source-ns (the-ns 'raster.compiler.core.walker-test)})]
      (is (= :array-op (walker/classify-form '(aget a 0) ctx))))))

(deftest retained-primitive-array-hints-do-not-skip-the-array-operation-boundary
  (doseq [tag ['floats 'doubles 'ints 'longs]
          operation ['raster.arrays/aget 'raster.arrays/aset 'raster.arrays/aclone]]
    (let [a (with-meta 'a {:tag tag :raster.type/tag tag})
          expression (case (name operation)
                       "aget" (list operation a 0)
                       "aset" (list operation a 0 1.0)
                       "aclone" (list operation a))
          ctx (walker/make-ctx {:type-env (te {'a tag}) :source-ns *ns*})]
      (is (= :array-op (walker/classify-form expression ctx)))
      (is (= (symbol "clojure.core" (name operation))
             (first (walker/walk expression ctx)))))))

(deftest primitive-float-store-realization-is-stable-under-rewalking
  (let [a (with-meta 'a {:tag 'floats :raster.type/tag 'floats})
        ctx (walker/make-ctx {:type-env (te {'a 'floats 'i 'long 'x 'double})
                              :source-ns *ns*})]
    (doseq [value ['x '(float x) '(clojure.core/float x) '(double x)]]
      (let [once (walker/walk (list 'raster.arrays/aset a 'i value) ctx)
            twice (walker/walk once ctx)
            three-times (walker/walk twice ctx)]
        (is (= 'clojure.core/aset (first once)))
        (is (= once twice three-times))
        (is (= 'floats (:raster.type/tag (meta (second twice)))))
        (is (= 1 (count (filter #(and (seq? %)
                                      (contains? #{'float 'clojure.core/float} (first %)))
                               (tree-seq coll? seq twice)))))))))

(deftest primitive-double-store-does-not-insert-a-float-conversion
  (let [a (with-meta 'a {:tag 'doubles :raster.type/tag 'doubles})
        ctx (walker/make-ctx {:type-env (te {'a 'doubles 'i 'long 'x 'double})
                              :source-ns *ns*})]
    (doseq [value ['x '(double x) '(clojure.core/double x)]]
      (let [once (walker/walk (list 'raster.arrays/aset a 'i value) ctx)
            twice (walker/walk once ctx)]
        (is (= 'clojure.core/aset (first once)))
        (is (= once twice))
        (is (not-any? #(and (seq? %)
                           (contains? #{'float 'clojure.core/float} (first %)))
                      (tree-seq coll? seq twice)))))))

(deftest primitive-array-realization-does-not-override-unknown-or-custom-receivers
  (let [a (with-meta 'a {:tag 'floats :raster.type/tag 'floats})
        expression (list 'raster.arrays/aget a 0)]
    (doseq [tag [nil 'Object 'CustomArray]]
      (let [ctx (walker/make-ctx {:type-env (te {'a tag}) :source-ns *ns*})]
        (is (= :deftm-call (walker/classify-form expression ctx)))))))

(deftest primitive-array-realization-does-not-hijack-lexical-array-operation-names
  (doseq [operation ['aget 'aset 'aclone]
          receiver ['a (with-meta 'a {:tag 'floats :raster.type/tag 'floats})]]
    (let [expression (case operation
                       aget (list operation receiver 0)
                       aset (list operation receiver 0 1.0)
                       aclone (list operation receiver))
          ctx (walker/make-ctx {:type-env (te {'a 'floats operation nil})
                                :source-ns *ns*})]
      (is (= :call (walker/classify-form expression ctx)))
      (is (= operation (first (walker/walk expression ctx)))))))

(deftm array-operation-name-collision
  [a :- (Array float), i :- Long] :- Float
  (float (+ 11.0 (clojure.core/aget a i))))

(deftest primitive-array-realization-uses-resolved-source-callee-identity
  (let [source (create-ns (gensym "raster.compiler.core.walker-array-shadow-fixture-"))
        receiver (with-meta 'a {:tag 'floats :raster.type/tag 'floats})
        ctx (walker/make-ctx {:type-env (te {'a 'floats 'i 'long}) :source-ns source})]
    (try
      (let [helper (intern source 'aget (fn [_ _] 99.0))
            expression (list 'aget receiver 'i)]
        (is (= :call (walker/classify-form expression ctx)))
        (is (= (symbol (str (ns-name source)) "aget")
               (first (walker/walk expression ctx))))
        ;; A genuine custom deftm exposed under the same source name retains
        ;; dispatch authority; its basename is not an intrinsic certificate.
        (alter-var-root helper (constantly @#'array-operation-name-collision))
        (alter-meta! helper merge
                     (dissoc (meta #'array-operation-name-collision) :ns :name))
        (is (= :deftm-call (walker/classify-form expression ctx))))
      (let [shadowed (walker/ctx-assoc-type ctx 'aget nil)
            qualified (list 'raster.arrays/aget receiver 'i)]
        (is (= :array-op (walker/classify-form qualified shadowed)))
        (is (= 'clojure.core/aget (first (walker/walk qualified shadowed)))))
      (finally (remove-ns (ns-name source))))))

(deftest float-store-conversion-does-not-call-a-lexical-float-function
  (let [a (with-meta 'a {:tag 'floats :raster.type/tag 'floats})
        ctx (walker/make-ctx {:type-env (te {'a 'floats 'x 'double 'float nil})
                              :source-ns *ns*})
        walked (walker/walk (list 'raster.arrays/aset a 0 '(float x)) ctx)
        value (last walked)]
    (is (= 'clojure.core/float (first value)))
    (is (= 'float (first (second value))))))

(deftest reduction-uses-typed-compound-initializer-result
  (let [walked (wb '(raster.par/reduce total
                     (raster.numeric/+
                      (raster.sci.distributions/logpdf
                       (raster.sci.distributions/->Normal 0.0 s0) mu)
                      (raster.sci.distributions/logpdf
                       (raster.sci.distributions/->Normal mu s) y))
                     i count
                     (raster.numeric/+ total (aget ys i)))
                   {'s0 'double 'mu 'double 's 'double 'y 'double
                    'ys 'doubles 'count 'long})]
    (is (= 'double (:raster.type/tag (meta walked))))
    (is (= 'double (:raster.type/tag (meta (nth walked 2)))))
    (is (= :double (:raster.type/elem-type (meta walked))))))

(deftest classify-vector-test
  (testing "vector forms"
    (let [ctx (walker/make-ctx {})]
      (is (= :vector (walker/classify-form [1 2 3] ctx))))))

;; ================================================================
;; walk — integration via walk-body
;; ================================================================

(deftest walk-literal-test
  (testing "literals pass through"
    (is (= 42 (wb 42)))
    (is (= 3.14 (wb 3.14)))))

(deftest walk-symbol-test
  (testing "symbols pass through"
    (is (= 'x (wb 'x {'x 'double})))))

(deftest walk-let-hinting-test
  (testing "array bindings get type hints"
    (let [walked (wb '(let [buf (double-array 16)] buf))]
      (is (seq? walked))
      ;; walk-body closes the core: let -> let* (hint still applied below)
      (is (= 'let* (first walked)))
      (let [bindings (partition 2 (second walked))
            [sym _init] (first bindings)]
        ;; buf should be hinted as doubles (array type)
        (is (= 'doubles (:tag (meta sym))))))))

(deftest walk-deftm-devirtualization-test
  (testing "deftm call devirtualized to .invk"
    (let [walked (wb '(let [z (raster.numeric/+ x y)] z)
                     {'x 'double 'y 'double})]
      ;; Should produce .invk call
      (let [bindings (partition 2 (second walked))
            [_ init] (first bindings)]
        (is (seq? init))
        (is (= '.invk (first init)))))))

(deftest walk-op-metadata-test
  (testing ".invk forms carry :raster.op/original metadata"
    (let [walked (wb '(let [z (raster.numeric/+ x y)] z)
                     {'x 'double 'y 'double})]
      (let [bindings (partition 2 (second walked))
            [_ init] (first bindings)]
        (is (= 'raster.numeric/+ (:raster.op/original (meta init))))))))

(deftest bare-scalar-arithmetic-retains-the-central-inferred-result-type
  (let [mixed (wb '(clojure.core/* (double scale) (clojure.core/aget x i))
                  {'scale 'double 'x 'floats 'i 'long})
        uniform (wb '(clojure.core/+ left right) {'left 'float 'right 'float})]
    (is (= 'double (:raster.type/tag (meta mixed))))
    (is (= 'float (:raster.type/tag (meta uniform))))
    (is (nil? (:raster.op/original (meta mixed)))
        "retaining an inferred type does not pretend that a bare core call was devirtualized")))

(deftest unchecked-long-arithmetic-retains-the-existing-inferred-result-type
  (doseq [[operation arguments]
          [['clojure.core/unchecked-add '[state (long i)]]
           ['clojure.core/unchecked-subtract '[state (long i)]]
           ['clojure.core/unchecked-multiply '[state (long i)]]
           ['clojure.core/unchecked-negate '[state]]
           ['clojure.core/unchecked-inc '[state]]
           ['clojure.core/unchecked-dec '[state]]]]
    (let [walked (wb (apply list operation arguments) {'state 'long 'i 'long})]
      (is (= 'long (:raster.type/tag (meta walked))) (str operation))))
  (let [walked (wb '(long (unchecked-add state (long i))) {'state 'long 'i 'long})]
    (is (= 'long (:raster.type/tag (meta (second walked))))
        "the public map result cast retains its compound operand type")))

(deftest effect-map-index-has-a-lexical-type-scope
  (let [walked (wb (with-meta
                    '(raster.par/map-void! i (clojure.core/+ i i)
                       (clojure.core/quot i width))
                    {:line 123})
                   {'i 'double 'width 'long})
        [_ _ bound body] walked]
    (is (= 'double (:raster.type/tag (meta bound)))
        "the bound sees the outer binding, not the induction variable")
    (is (= 'long (:raster.type/tag (meta body)))
        "the existing scalar inference now sees both operand types")
    (is (= 123 (:line (meta walked))))
    (is (nil? (:raster.type/tag (meta walked)))
        "an effect map does not inherit the body's scalar result"))
  (let [walked (wb '(raster.par/map-void! i n (clojure.core/quot i unknown))
                   {'n 'long})]
    (is (nil? (:raster.type/tag (meta (last walked))))
        "index scope does not invent types for unknown operands"))
  (let [walked (wb '(do (raster.par/map-void! i n (clojure.core/quot i width))
                        (clojure.core/+ i i))
                   {'i 'double 'n 'long 'width 'long})]
    (is (= 'double (:raster.type/tag (meta (last walked))))
        "the induction variable does not escape the effect map")))

(deftest malformed-effect-map-does-not-drop-source-operands
  (doseq [form ['(raster.par/map-void! i n)
               '(raster.par/map-void! i n nil (throw (Exception.)))
               '(raster.par/map-void! [i] n nil)]]
    (is (= :invalid-effect-map-form
           (try (wb form) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest java-static-overload-results-survive-nested-conversions
  (doseq [[input-type result-type] [['float 'int] ['double 'long]]]
    (let [walked (wb (list 'clojure.core/long
                           (list 'Math/round (list input-type 'x)))
                     {'x input-type})]
      (is (= result-type (:raster.type/tag (meta (second walked)))))
      (is (= 'long (:raster.type/tag (meta walked))))))
  (is (= 'double (:raster.type/tag (meta (wb '(Math/sqrt (double x)) {'x 'double})))))
  (is (nil? (:raster.type/tag (meta (wb '(Math/round unknown))))))
  (is (nil? (:raster.type/tag (meta (wb '(unknown.namespace/round unknown)))))))

(deftest java-static-result-resolution-uses-the-source-imports
  (let [source-name (gensym "walker-source-")
        source-ns (create-ns source-name)]
    (try
      (.importClass source-ns 'SourceMath java.lang.Math)
      (is (= 'long
             (:raster.type/tag
              (meta (walker/walk-body '(SourceMath/round (clojure.core/double x))
                                      {:source-ns source-ns
                                       :type-env (te {'x 'double})})))))
      (finally (remove-ns source-name)))))

(deftest walk-nested-let-test
  (testing "nested let bindings are walked"
    (let [walked (wb '(let [a (raster.numeric/+ x y)
                            b (raster.numeric/* a z)]
                        b)
                     {'x 'double 'y 'double 'z 'double})]
      ;; Both bindings should be devirtualized
      (let [bindings (vec (partition 2 (second walked)))]
        (is (= 2 (count bindings)))
        (is (= '.invk (first (second (first bindings)))))
        (is (= '.invk (first (second (second bindings)))))))))

(deftest walk-vector-test
  (testing "vectors are walked element-wise"
    (let [walked (wb '(let [a (raster.numeric/+ x y)] [a 42])
                     {'x 'double 'y 'double})]
      (let [body (last (nnext walked))]
        (is (vector? body))))))

(deftest walk-if-test
  (testing "if branches are walked"
    (let [walked (wb '(if true x y) {'x 'double 'y 'double})]
      (is (seq? walked))
      (is (= 'if (first walked))))))

;; ================================================================
;; Object devirtualization guard
;; ================================================================

(deftest walk-object-no-devirtualize-test
  (testing "Object-tagged methods are NOT devirtualized (promotion fallbacks)"
    ;; When both args are Object, walker should leave as qualified dispatch call
    (let [walked (wb '(raster.numeric/* x y) {'x 'Object 'y 'Object})]
      (is (seq? walked))
      (is (= 'raster.numeric/* (first walked))
          "Object+Object should stay as dispatch call, not devirtualize to _star__m_Object_Object"))
    ;; When one arg is inferred Object (e.g. from Object-returning fn),
    ;; the call should still not devirtualize to the Object fallback
    (let [walked (wb '(raster.numeric/* x y) {'x 'Number 'y 'Object})]
      (is (not (and (seq? walked)
                    (let [head (first walked)]
                      (and (symbol? head)
                           (.contains (name head) "Object")))))
          "Should not devirtualize when any arg resolves to Object fallback")))

  (testing "Number-tagged methods ARE devirtualized (concrete implementations)"
    (let [walked (wb '(raster.numeric/* x y) {'x 'Number 'y 'Number})]
      (is (seq? walked))
      (is (= '.invk (first walked))
          "Number+Number should devirtualize to .invk")
      (is (let [impl (second walked)]
            (and (symbol? impl)
                 (.contains (name impl) "Number")))
          "Number+Number impl should contain 'Number'")))

  (testing "double-tagged methods ARE devirtualized"
    (let [walked (wb '(raster.numeric/* x y) {'x 'double 'y 'double})]
      (is (seq? walked))
      (is (= '.invk (first walked))
          "double+double should devirtualize to .invk"))))

;; ================================================================
;; case dispatch constants — qualified-head regression
;; ================================================================

(deftest qualified-case-classified-and-keys-preserved
  (testing "clojure.core/case is classified :case, not a generic call"
    (let [ctx (walker/make-ctx {})]
      (is (= :case (walker/classify-form '(case (clojure.core/int h) 0 a 1 b) ctx)))
      (is (= :case (walker/classify-form '(clojure.core/case (clojure.core/int h) 0 a 1 b) ctx)))))
  (testing "walking a case (bare or qualified) leaves dispatch constants untouched"
    ;; Regression: a qualified clojure.core/case fell through to the generic-call
    ;; handler, which type-tagged integer keys 0 -> (long 0), later breaking
    ;; clojure.core/case macroexpansion in the backends.
    ;; NB: walk-body now closes the core (case -> case*), which would bury the keys
    ;; in the case* hash-table; this regression lives in the *walker*, so test the
    ;; walker's direct (pre-macroexpand) output where the `case` shape is intact.
    (doseq [head '[case clojure.core/case]]
      (let [form   (list head '(clojure.core/int h) 0 'a 1 'b)
            ctx    (walker/make-ctx {:type-env (te {'h 'long 'a 'double 'b 'double})})
            walked (walker/walk form ctx)
            keys   (map first (partition 2 (drop 2 walked)))]
        (is (= [0 1] keys) (str head " keys stay literal (not (long 0)/(long 1))"))))))


;; Known nested returns constrain scalar dispatch before contextual literal typing.
(deftm nested-return-double-inverse
  (All [T] [a :- (Array T) x :- Double eps :- Double] :- Double
    (raster.numeric// 1.0 (raster.numeric/sqrt (raster.numeric/+ x eps)))))
(deftm local-return-double-inverse
  (All [T] [a :- (Array T) x :- Double eps :- Double] :- Double
    (let [root (raster.numeric/sqrt (raster.numeric/+ x eps))]
      (raster.numeric// 1.0 root))))
(deftm explicit-return-double-inverse
  (All [T] [a :- (Array T) x :- Double eps :- Double] :- Double
    (raster.numeric// 1.0 (double (raster.numeric/sqrt (raster.numeric/+ x eps))))))
(deftm nested-return-double-left
  (All [T] [a :- (Array T) x :- Double eps :- Double] :- Double
    (raster.numeric// (raster.numeric/sqrt (raster.numeric/+ x eps)) 3.0)))
(deftm nested-return-double-reducible
  (All [T] [a :- (Array T) x :- Double] :- Double
    (raster.numeric/+ 1.0 (raster.numeric/sqrt x) 3.0)))
(deftm nested-return-element-inverse
  (All [T] [a :- (Array T) x :- T] :- T
    (raster.numeric// 1.0000000596046448 (raster.numeric/sqrt x))))
(deftm local-return-element-inverse
  (All [T] [a :- (Array T) x :- T] :- T
    (let [root (raster.numeric/sqrt x)]
      (raster.numeric// 1.0000000596046448 root))))
(deftm observed-double-return [seen :- Object x :- Double] :- Double
  (do (swap! seen conj x) x))
(deftm observed-double-return [seen :- String x :- Double] :- Float (float x))
(deftm nested-return-observed-inverse
  (All [T] [a :- (Array T) observe :- (Fn [Double] Double) x :- Double] :- Double
    (raster.numeric// 1.0 (observe x))))
(deftm nested-return-observed-pair
  (All [T] [a :- (Array T) observe :- (Fn [Double] Double) x :- Double y :- Double] :- Double
    (raster.numeric/+ (observe x) (observe y))))

(deftest known-nested-return-constrains-both-operand-positions
  (doseq [form ['(raster.numeric// 1.0 (raster.numeric/sqrt x))
                '(raster.numeric// (raster.numeric/sqrt x) 3.0)]]
    (let [walked (walker/walk-body form {:source-ns *ns* :type-env (te {'x 'double})
                                       :element-dtype :float})]
      (is (= 'double (:raster.type/tag (meta walked)))
          "a Float surrounding array cannot narrow a declared Double return"))))

(deftest known-nested-double-return-executes-the-declared-arithmetic
  (binding [core/*compile-deftm-bytecode?* true core/*jit-simd?* false
            core/*jit-silent-fallback?* false]
    (doseq [a [(float-array 1) (double-array 1)]
            x [0.0819668173789978 0.08658681511878967 0.0]]
      (let [eps 1.0e-6 root (Math/sqrt (+ x eps)) expected (/ 1.0 root)
            bits (Double/doubleToRawLongBits expected)]
        (is (= bits (Double/doubleToRawLongBits (nested-return-double-inverse a x eps))))
        (is (= bits (Double/doubleToRawLongBits (local-return-double-inverse a x eps))))
        (is (= bits (Double/doubleToRawLongBits (explicit-return-double-inverse a x eps))))
        (is (= (Double/doubleToRawLongBits (/ root 3.0))
               (Double/doubleToRawLongBits (nested-return-double-left a x eps))))))))

(deftest known-float-return-retains-contextual-literal-narrowing
  (binding [core/*compile-deftm-bytecode?* true core/*jit-simd?* false
            core/*jit-silent-fallback?* false]
    (let [x (float 3.0) root (float (Math/sqrt (double x)))
          expected (float (/ (double (float 1.0000000596046448)) (double root)))
          widened (float (/ 1.0000000596046448 (double root)))
          a (float-array 1)]
      (is (not= (Float/floatToRawIntBits expected) (Float/floatToRawIntBits widened))
          "the witness distinguishes Float contextual literal arithmetic from widening")
      (is (= (Float/floatToRawIntBits expected)
             (Float/floatToRawIntBits (nested-return-element-inverse a x))))
      (is (= (Float/floatToRawIntBits expected)
             (Float/floatToRawIntBits (local-return-element-inverse a x)))))
    (let [x 3.0 expected (/ 1.0000000596046448 (Math/sqrt x))]
      (is (= (Double/doubleToRawLongBits expected)
             (Double/doubleToRawLongBits (nested-return-element-inverse (double-array 1) x)))))))

(deftest reducible-call-retains-nested-double-return-before-literal-narrowing
  (binding [core/*compile-deftm-bytecode?* true core/*jit-simd?* false
            core/*jit-silent-fallback?* false]
    (doseq [a [(float-array 1) (double-array 1)] x [0.0819668173789978 3.0000000000000004]]
      (is (= (Double/doubleToRawLongBits (+ (+ 1.0 (Math/sqrt x)) 3.0))
             (Double/doubleToRawLongBits (nested-return-double-reducible a x)))))))

(deftest nested-return-resolution-preserves-effect-order-and-single-evaluation
  (binding [core/*compile-deftm-bytecode?* true core/*jit-simd?* false
            core/*jit-silent-fallback?* false]
    (let [seen (atom []) x 3.0000000000000004 a (float-array 1)
          observe (ftm [v :- Double] :- Double (do (swap! seen conj v) v))]
      (is (= (Double/doubleToRawLongBits (/ 1.0 x))
             (Double/doubleToRawLongBits (nested-return-observed-inverse a observe x))))
      (is (= [x] @seen))
      (reset! seen [])
      (is (= 7.0 (nested-return-observed-pair a observe 3.0 4.0)))
      (is (= [3.0 4.0] @seen)))))

(deftest nested-return-evidence-does-not-manufacture-an-unknown-type
  (let [source '(identity unknown)
        rewritten (walker/walk-body source {:source-ns *ns* :type-env {}})
        walked (walker/walk-body (list 'raster.numeric// 1.0 source)
                                  {:source-ns *ns* :type-env {} :element-dtype :float})]
    (is (nil? (inf/infer-rewritten-tag rewritten source {})))
    (is (= 'float (:raster.type/tag (meta walked)))
        "characterize existing partial-resolution fallback without changing it")))


(deftest object-generic-return-is-dynamic-across-specific-overloads
  (let [opts {:source-ns (the-ns 'raster.compiler.core.walker-test)
              :type-env (te {'seen 'Object 'x 'double}) :element-dtype :float}
        source '(observed-double-return seen x)
        walked (walker/walk-body source opts)
        seen (atom []) x 3.0000000000000004
        object-result (observed-double-return seen x)
        specific-result (observed-double-return "specific subtype" x)]
    (is (= #'observed-double-return (ns-resolve (:source-ns opts) (first walked)))
        "Object upper bounds preserve live generic dispatch")
    (is (nil? (:raster.type/tag (meta walked))))
    (is (nil? (inf/infer-rewritten-tag walked source (:type-env opts))))
    (is (instance? Double object-result))
    (is (= (Double/doubleToRawLongBits x) (Double/doubleToRawLongBits object-result)))
    (is (instance? Float specific-result))
    (is (= [x] @seen))))

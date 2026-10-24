(ns raster.compiler.passes.scalar.local-products-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as rcore :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.source-policy :as source-policy]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.normal-return :as normal-return]
            [raster.compiler.passes.scalar.inline :as inline]
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

(deftm native-product-with-unused-input [x :- Double unused :- Double] :- Double
  (let [scaled (n/* x 2.0)
        values [x scaled]]
    (n/+ (nth values 0) (nth values 1))))

(defn- observed-double [events label value]
  (proxy [Number] []
    (intValue [] (throw (AssertionError. "unexpected integral conversion")))
    (longValue [] (throw (AssertionError. "unexpected integral conversion")))
    (floatValue [] (throw (AssertionError. "unexpected Float conversion")))
    (doubleValue []
      (swap! events conj label)
      (if (instance? Throwable value) (throw value) (double value)))))

(deftm native-ad-unused-double [x :- Double unused :- Double] :- Double
  (n/* x 2.0))

(deftm native-ad-unused-double-gradient [x :- Double unused :- Double] :- Double
  (let [result ((reverse/value+grad #'native-ad-unused-double :wrt [0]) x unused)]
    (nth result 1)))

(deftm native-ad-integral-boundary [x :- Double unused :- Long] :- Double
  (n/* x 2.0))

(deftm native-ad-integral-boundary-gradient [x :- Double unused :- Long] :- Double
  (let [result ((reverse/value+grad #'native-ad-integral-boundary :wrt [0]) x unused)]
    (nth result 1)))

(deftm native-ad-local-argument-gradient [x :- Double] :- Double
  (let [local x result ((reverse/value+grad #'scalar-square) local)]
    (nth result 1)))

(deftest native-ad-entry-roundtrip-keeps-all-public-conversions
  (let [compiled (pipeline/compile-aot #'native-ad-unused-double-gradient :target :c :dtype :double)
        events (atom [])]
    (try
      (is (= 2.0 (compiled (observed-double events :x 3.0)
                           (observed-double events :unused 11.0))))
      (is (= [:x :unused] @events))
      (reset! events [])
      (is (thrown? NullPointerException (compiled (observed-double events :x 3.0) nil)))
      (is (= [:x] @events))
      (reset! events [])
      (is (thrown? ClassCastException (compiled "bad" (observed-double events :unused 11.0))))
      (is (empty? @events))
      (finally (when (instance? java.lang.AutoCloseable compiled)
                 (.close ^java.lang.AutoCloseable compiled))))))

(deftest native-ad-entry-construction-does-not-borrow-logical-parameter-types
  (doseq [source [#'native-ad-integral-boundary-gradient #'native-ad-local-argument-gradient]]
    (is (= :unclosed-c-family-value-completion
           (:reason (ex-data (try (pipeline/compile-aot source :target :c :dtype :double) nil
                                 (catch clojure.lang.ExceptionInfo error error))))))))

(deftest native-ad-entry-projection-rejects-changed-actual-constructor-plans
  (let [factory (requiring-resolve
                 'raster.compiler.backend.jvm.generated-body/instantiate-boxed-entry-inputs)
        original (var-get factory)]
    (doseq [change [:bridge :slot-contract]]
      (method-entry/call-with-definition-inputs
       (fn []
         (let [source-var (:resolved-var
                           (pipeline/deftm-frontend-facts #'native-ad-unused-double-gradient :double))
               _ (rcore/prepare-deftm-frontend! source-var)
               frontend (method-entry/original-source-frontend source-var)
               policy (get-in frontend [:source-entry :source-policy])
               factory-calls (atom 0) consumers (atom 0)
               result
               (with-redefs-fn
                 {factory
                  (fn [physical parameters arguments]
                    (let [plan (original physical parameters arguments)]
                      (swap! factory-calls inc)
                      (case change
                        :bridge (assoc-in plan [:boxed-entry-bridge :kind] :changed-original-bridge)
                        :slot-contract
                        (let [slot (first (:slots plan))
                              ;; A real canonical contract with the SAME actual
                              ;; target Class, but not this boxed-entry operation.
                              contract (completion/primitive-input-contract (:parameter-class slot) policy)
                              input [(first (first (:input-bindings plan)))
                                     (completion/complete-value contract (:object-binder slot))]
                              inputs (assoc (:input-bindings plan) 0 input)]
                          (assoc plan :slots (assoc-in (:slots plan) [0 :contract] contract)
                                      :input-bindings inputs
                                      :bindings (into (:object-bindings plan) inputs))))))}
                 #(reverse/with-double-source-ad-projection
                   frontend source-var (pipeline/build-param-env source-var :double)
                   (fn [_] (swap! consumers inc) :unexpected-consumer)))]
           (is (pos? @factory-calls) "interpose the actual prepared entry constructor")
           (is (nil? result))
           (is (zero? @consumers) "changed observations never reach the numerical consumer")))))))

(deftest direct-ad-projection-normalization-keeps-the-original-evaluation-site
  (let [source-ns (the-ns 'raster.compiler.passes.scalar.local-products-test)
        application '((raster.ad.reverse/value+grad
                       (var raster.compiler.passes.scalar.local-products-test/scalar-square)) x)
        source (list 'let* '[before (record! :before)] (list 'clojure.core/nth application 1))
        normalized (method-entry/call-with-definition-inputs
                    #(inline/normalize-direct-ad-projections source source-ns))
        bindings (partition 2 (second normalized))]
    (is (= '[before (record! :before)] (first bindings)))
    (is (identical? application (second (second bindings))))
    (is (= (list 'clojure.core/nth (first (second bindings)) 1) (last normalized)))
    (doseq [tail [(list 'consume! '(record! :operand) (list 'clojure.core/nth application 1))
                  (list 'if 'pred (list 'clojure.core/nth application 1) 0.0)
                  (list 'clojure.core/nth application 1 :missing)
                  (list 'clojure.core/nth
                        '((raster.ad.reverse/grad
                           (var raster.compiler.passes.scalar.local-products-test/scalar-square)) x) 0)]]
      (let [original (list 'let* [] tail)]
        (is (= original (method-entry/call-with-definition-inputs
                         #(inline/normalize-direct-ad-projections original source-ns))))))
    (let [original-nth clojure.core/nth]
      (with-redefs [clojure.core/nth (fn [& args] (apply original-nth args))]
        (is (= :intrinsic-source-definition-mismatch
               (:reason (ex-data (try
                                   (method-entry/call-with-definition-inputs
                                    #(inline/normalize-direct-ad-projections source source-ns)) nil
                                   (catch clojure.lang.ExceptionInfo error error))))))))))

(deftest native-object-storage-keeps-entry-conversions-before-the-source-guard
  (let [compiled (pipeline/compile-aot #'native-product-with-unused-input :target :c :dtype :double)
        events (atom [])
        x (observed-double events :x 2.0)
        unused (observed-double events :unused 17.0)
        receiver (ns-resolve 'raster.numeric '_star__m_double_double-impl)]
    (try
      (is (= 6.0 (compiled x unused)))
      (is (= [:x :unused] @events))
      (reset! events [])
      (is (= :native-call-arity
             (:reason (ex-data (try (compiled x unused 1.0) nil
                                   (catch clojure.lang.ExceptionInfo error error))))))
      (is (empty? @events) "arity refusal precedes every public conversion")
      ;; This is an actual selected receiver mutation, not fabricated carrier
      ;; evidence. No replacement numerical helper is ever executed.
      (with-redefs-fn
        {receiver (Object.)}
        (fn []
          (is (thrown? NullPointerException (compiled nil unused)))
          (is (thrown? ClassCastException (compiled "not a Number" unused)))
          (is (empty? @events))
          (let [error (try (compiled x unused) nil
                           (catch clojure.lang.ExceptionInfo error error))]
            (is (= :stale-definition-proof-input (:reason (ex-data error))))
            (is (= [:x :unused] @events)
                "both original conversions occur exactly once before freshness denial"))
          (reset! events [])
          (let [failure (ex-info "unused conversion failed" {:conversion :unused})
                actual (try (compiled x (observed-double events :unused failure)) nil
                            (catch clojure.lang.ExceptionInfo error error))]
            (is (identical? failure actual))
            (is (= [:x :unused] @events)))))
      (reset! events [])
      ;; The source is current when invocation starts. A guard before the
      ;; conversions would miss this mutation and incorrectly execute C.
      (with-redefs-fn
        {receiver (var-get receiver)}
        (fn []
          (let [mutating-x
                (proxy [Number] []
                  (intValue [] (throw (AssertionError. "unexpected integral conversion")))
                  (longValue [] (throw (AssertionError. "unexpected integral conversion")))
                  (floatValue [] (throw (AssertionError. "unexpected Float conversion")))
                  (doubleValue []
                    (swap! events conj :mutating-x)
                    (alter-var-root receiver (constantly (Object.)))
                    2.0))
                error (try (compiled mutating-x unused) nil
                           (catch clojure.lang.ExceptionInfo error error))]
            (is (= [:mutating-x :unused] @events)
                "even a mutation in the first conversion precedes the second conversion")
            (is (= :stale-definition-proof-input (:reason (ex-data error)))))))
      (finally (when (instance? java.lang.AutoCloseable compiled)
                 (.close ^java.lang.AutoCloseable compiled))))))

(deftm native-product-observed-identity [x :- Double] :- Double
  (let [values [x x]]
    (if (identical? (nth values 0) (nth values 1)) 1.0 2.0)))

(deftm native-product-checked-integral [x :- Double] :- Double
  (let [values [(double (long x)) x]]
    (n/+ (nth values 0) (nth values 1))))

(deftest native-object-storage-does-not-admit-identity-or-checked-integral-work
  (doseq [source [#'native-product-observed-identity #'native-product-checked-integral]]
    (is (= :unclosed-c-family-value-completion
           (try (pipeline/compile-aot source :target :c :dtype :double) :unexpected-admission
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "the bounded transport constructor cannot grant either operation")))

(deftm native-product-double-literal [x :- Double] :- Double
  (let [precise 1.0000000596046448
        values [x precise]]
    (n/+ (nth values 0) (nth values 1))))

(deftest native-object-storage-retains-double-emission-policy
  (let [compiled (pipeline/compile-aot #'native-product-double-literal :target :c :dtype :double)]
    (is (= 1.0000000596046448 (compiled 0.0))))
  (is (= :unclosed-c-family-value-completion
         (try (pipeline/compile-aot #'native-product-double-literal :target :c :dtype :float)
              :unexpected-admission
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
      "a Double entry/return does not establish the C emitter's literal policy"))

(deftest scalar-storage-construction-uses-the-existing-ssa-shadow-boundary
  (method-entry/call-with-definition-inputs
   (fn []
     (let [source-var (:resolved-var (pipeline/deftm-frontend-facts #'projected-product :double))
           _ (rcore/prepare-deftm-frontend! source-var)
           frontend (method-entry/original-source-frontend source-var)
           entry (method-entry/frontend-body-entry-carriers frontend source-var)
           parameters (vec (:parameters entry))
           parameter (first parameters)
           snapshot (list completion/completion-symbol completion/object-input-contract parameter)
           source (list 'let* ['p snapshot 'p (list 'clojure.core/double parameter)] 'p)]
       (is (some? entry) "use the actual entry Method, never invented local carrier facts")
       (is (nil? (#'normal-return/construct-double-storage-source
                  source parameters entry (get-in frontend [:source-entry :source-ns])))
           "the shared physical SSA planner refuses a rebinding before any replacement escapes")))))

(defn- run [expression & arguments]
  (binding [*unchecked-math* false]
    (let [policy (source-policy/capture)
          source (completion/project-to-jvm-source (list 'fn* '[x calls] expression) policy)]
      (apply (eval source) arguments))))

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

(deftest literal-nil-product-fields-need-no-object-snapshot
  (let [source '(let* [p [(do (swap! calls conj :left) x) nil
                         (do (swap! calls conj :right) x)]]
                 (clojure.core/nth p 1))
        lowered (products/lower-local-products source)
        source-events (atom []) lowered-events (atom [])]
    (is (nil? (run source 3.0 source-events)))
    (is (nil? (run lowered 3.0 lowered-events)))
    (is (= [:left :right] @source-events @lowered-events))
    (is (= 2 (count (partition 2 (second lowered)))))
    (is (not-any? #(and (completion/canonical-form? %)
                        (nil? (:operand (completion/canonical-parts %))))
                  (tree-seq coll? seq lowered))))
  (let [source '(let* [p [nil (clojure.core/double x)]] (clojure.core/nth p 0))
        lowered (products/lower-local-products source)]
    (is (thrown? NullPointerException (run source nil nil)))
    (is (thrown? NullPointerException (run lowered nil nil)))))

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

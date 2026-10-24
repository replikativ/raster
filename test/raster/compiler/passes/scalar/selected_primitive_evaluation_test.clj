(ns raster.compiler.passes.scalar.selected-primitive-evaluation-test
  (:refer-clojure :exclude [*])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.normal-return :as normal]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- call [operation implementation]
  (with-meta (list '.invk implementation 'x 'y) {:raster.op/original operation}))

(def ^:private divide
  (call 'raster.numeric// 'raster.numeric/_div__m_double_double-impl))
(def ^:private primitive-double {:carrier Double/TYPE :presence :present})
(def ^:private primitive-float {:carrier Float/TYPE :presence :present})
(def ^:private boxed-double {:carrier Object :presence :present :builtin-class Double})
(def ^:private boxed-float {:carrier Object :presence :present :builtin-class Float})

(defn- proof [expression facts]
  (entry/call-with-definition-inputs
   #(scalar/checked-selected-primitive-evaluation expression facts)))

(defn- stale-source-data [operation]
  (try (operation) nil
       (catch clojure.lang.ExceptionInfo failure
         (select-keys (ex-data failure) [:reason :source]))))

(deftest completion-is-separate-from-presence-and-the-object-result-carrier
  (is (= {:type :double}
         (entry/call-with-definition-inputs
          #(scalar/checked-selected-primitive-result divide))))
  (doseq [facts [[primitive-double primitive-double]
                 [primitive-float primitive-double]
                 [primitive-double boxed-double]
                 [boxed-float boxed-double]]]
    (is (= {:type :double :result-carrier Object :presence :present :scope :completed-fp-scalar
            :builtin-class Double :nonthrowing? true}
           (proof divide facts))))
  (doseq [fact [nil {:type :double} {:carrier Double/TYPE}
                {:carrier Object :presence :unknown :builtin-class Double}
                {:carrier Object :presence :present}
                {:carrier String :presence :present :builtin-class Double}
                {:carrier Object :presence :present :builtin-class Number}]]
    (is (nil? (proof divide [primitive-double fact]))))
  (is (nil? (proof divide [primitive-double])))
  (is (nil? (proof divide (list primitive-double primitive-double))))
  (binding [entry/*emission-inputs* nil]
    (is (nil? (scalar/checked-selected-primitive-evaluation
               divide [primitive-double primitive-double])))))

(deftest a-refused-scalar-proof-does-not-create-an-artifact-dependency
  (let [owner (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* owner]
      (is (nil? (scalar/checked-selected-primitive-evaluation divide [primitive-double {}])))
      (is (empty? (:definition-proofs @owner)))
      (is (empty? (:producer-inputs @owner)))))
  (let [source (find-var 'raster.numeric/_div__m_double_double)
        receiver (find-var 'raster.numeric/_div__m_double_double-impl)
        owner (atom {:static [] :values []})]
    (core/prepare-deftm-frontend! source)
    (binding [entry/*emission-inputs* owner]
      (is (= {:presence :present :scalar-type :double} (normal/result-fact divide))
          "successful primitive return completion establishes presence, not a physical scalar proof")
      (is (not (contains? (:definition-proofs @owner) receiver)))
      (is (entry/assert-definition-inputs-current! @owner)))))

(deftest ieee-division-and-comparison-have-real-value-oracles
  (let [receiver @(find-var 'raster.numeric/_div__m_double_double-impl)
        invoke #(clojure.lang.Reflector/invokeInstanceMethod
                 receiver "invk" (object-array [%1 %2]))]
    (doseq [[x y] [[1.0 0.0] [-1.0 0.0] [0.0 0.0]
                    [Double/MAX_VALUE 0.5] [Double/NaN 1.0]]]
      (let [actual (invoke x y) expected (clojure.core// (double x) (double y))]
        (is (instance? Double actual))
        (is (or (and (Double/isNaN expected) (Double/isNaN actual))
                (= (Double/doubleToRawLongBits expected)
                   (Double/doubleToRawLongBits actual)))))))
  (let [comparison (call 'raster.numeric/> 'raster.numeric/_gt__m_double_double-impl)]
    (is (= {:type :predicate :result-carrier Object :presence :present :scope :completed-fp-scalar
            :builtin-class Boolean :nonthrowing? true}
           (proof comparison [boxed-double primitive-double])))))

(deftest float-checkpoints-and-narrowing-are-not-borrowed-from-logical-dtypes
  (let [float-divide (call 'raster.numeric// 'raster.numeric/_div__m_float_float-impl)]
    (is (= Float (:builtin-class (proof float-divide [primitive-float boxed-float]))))
    (is (nil? (proof float-divide [primitive-double primitive-float])))
    (is (nil? (proof float-divide [boxed-double primitive-float]))))
  (testing "checked source float narrowing is not the JVM's unchecked D2F instruction"
    (is (thrown? IllegalArgumentException (float Double/MAX_VALUE))))
  (doseq [expression [(call 'raster.numeric// 'raster.numeric/_div__m_Object_Object-impl)
                       (call 'raster.numeric// 'raster.numeric/_div__m_int_int-impl)
                       '(raster.numeric// x y)
                       (with-meta divide {:raster.op/original 'raster.numeric//
                                          :raster.op/lexical-callee true})]]
    (is (nil? (proof expression [primitive-double primitive-double])))))

(deftest float-division-preserves-ieee-values-and-the-float-result-box
  (let [expression (call 'raster.numeric// 'raster.numeric/_div__m_float_float-impl)
        receiver @(find-var 'raster.numeric/_div__m_float_float-impl)]
    (doseq [[x y] [[1.0 0.0] [-1.0 0.0] [0.0 0.0] [-0.0 2.0]
                    [Float/MAX_VALUE 0.5] [Float/NaN 1.0]]]
      (let [x (float x) y (float y)
            actual (clojure.lang.Reflector/invokeInstanceMethod
                    receiver "invk" (object-array [x y]))
            ;; Widening these already-rounded operands gives an independent
            ;; IEEE division oracle, then Number.floatValue performs raw RTE.
            expected (.floatValue ^Number (clojure.core// (double x) (double y)))]
        (is (some? (proof expression [boxed-float primitive-float])))
        (is (instance? Float actual))
        (is (or (and (Float/isNaN expected) (Float/isNaN actual))
                (= (Float/floatToRawIntBits expected)
                   (Float/floatToRawIntBits actual))))))))

(deftest bare-source-cast-proof-retains-the-original-namespace-lookup
  (let [expression (call 'raster.numeric// 'raster.numeric/_div__m_float_float-impl)
        source-ns (the-ns 'raster.numeric)
        original (ns-resolve source-ns 'float)
        collector (atom {:static [] :values []})]
    (binding [entry/*emission-inputs* collector]
      (is (some? (scalar/checked-selected-primitive-evaluation
                  expression [primitive-float boxed-float]))))
    (is (contains? (:definition-proofs @collector) original))
    (try
      ;; Same bare spelling, different Var identity: source qualification may
      ;; neither borrow a core cast nor bless the historical namespace edge.
      (ns-unmap source-ns 'float)
      (intern source-ns 'float (fn [_] nil))
      (is (nil? (proof expression [primitive-float boxed-float])))
      (is (= :stale-definition-proof-input
             (try (entry/assert-definition-inputs-current! @collector)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (finally
        (ns-unmap source-ns 'float)
        (binding [*ns* source-ns] (refer 'clojure.core :only '[float]))))
    (is (identical? original (ns-resolve source-ns 'float)))
    (is (entry/assert-definition-inputs-current! @collector))))

(deftest changed-selected-definitions-cannot-supply-a-completion-proof
  (doseq [v [(find-var 'raster.numeric/_div__m_double_double)
             (find-var 'raster.numeric/_div__m_double_double-impl)]]
    (with-redefs-fn {v (fn [& _] nil)}
      #(is (nil? (proof divide [primitive-double primitive-double])))))
  (let [collector (atom {:static [] :values []})
        _ (binding [entry/*emission-inputs* collector]
            (is (some? (scalar/checked-selected-primitive-evaluation
                        divide [primitive-double primitive-double]))))
        source (find-var 'raster.numeric/_div__m_double_double)
        original (meta source)]
    (try
      (alter-meta! source assoc :raster.core/deftm-source-body '[(identity nil)])
      (is (nil? (proof divide [primitive-double primitive-double])))
      (is (= :stale-definition-proof-input
             (try (entry/assert-definition-inputs-current! @collector)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (finally (reset-meta! source original)))
    (is (entry/assert-definition-inputs-current! @collector))))

(deftest cold-and-warm-queries-use-the-same-original-source-contract
  (binding [*ns* (the-ns 'raster.compiler.passes.scalar.selected-primitive-evaluation-test)]
    (eval '(deftm * [x :- Double y :- Double] :- Double (clojure.core/* x y))))
  (let [generic #'n/* original (meta generic)
        fixture (find-var 'raster.compiler.passes.scalar.selected-primitive-evaluation-test/*)
        receiver (find-var 'raster.compiler.passes.scalar.selected-primitive-evaluation-test/_star__m_double_double-impl)
        expression (call 'raster.numeric/* (symbol (str (:ns (meta receiver)))
                                                 (str (:name (meta receiver)))))]
    (try
      (alter-meta! generic assoc :raster.core/dispatch-table
                   (:raster.core/dispatch-table (meta fixture)))
      (is (some? (:raster.core/jit-spec (meta receiver))))
      (let [cold (with-redefs [core/ensure-bytecode-implementation!
                              (fn [& _] (throw (AssertionError. "proof invoked the JIT")))]
                   (proof expression [primitive-double boxed-double]))]
        (is (some? cold))
        (core/ensure-bytecode-implementation! receiver)
        (is (= cold (proof expression [primitive-double boxed-double]))))
      (finally (reset-meta! generic original)))))

(deftest a-fresh-owner-cannot-bless-changed-metadata-around-old-code
  (binding [*ns* (the-ns 'raster.compiler.passes.scalar.selected-primitive-evaluation-test)]
    (eval '(deftm * [x :- Double y :- Double] :- Object nil)))
  (let [fixture (find-var 'raster.compiler.passes.scalar.selected-primitive-evaluation-test/*)
        receiver (find-var 'raster.compiler.passes.scalar.selected-primitive-evaluation-test/_star__m_double_double-impl)
        source (find-var 'raster.compiler.passes.scalar.selected-primitive-evaluation-test/_star__m_double_double)
        generic #'n/* original-generic (meta generic) original-source (meta source)
        expression (call 'raster.numeric/* (symbol (str (:ns (meta receiver)))
                                                 (str (:name (meta receiver)))))]
    (try
      (core/ensure-bytecode-implementation! receiver)
      (alter-meta! generic assoc :raster.core/dispatch-table
                   (:raster.core/dispatch-table (meta fixture)))
      (let [body (list 'let* ['x 2.0 'y 3.0 'result expression]
                       '(if (clojure.core/nil? result) :absent :present))
            lowered (entry/call-with-definition-inputs #(products/lower-local-products body))
            evaluate #((eval (list 'fn* [] %)))]
        (is (= :absent (evaluate body) (evaluate lowered))))
      ;; Roots/registration remain unchanged; only the description now claims
      ;; unrelated numeric code. A new definition snapshot alone would accept it.
      (alter-meta! source assoc :raster.core/deftm-source-body '[(clojure.core/* x y)]
                   :raster.core/return-tag 'double)
      (is (nil? (clojure.lang.Reflector/invokeInstanceMethod
                 @receiver "invk" (object-array [2.0 3.0]))))
      (is (nil? (entry/call-with-definition-inputs
                 #(scalar/checked-selected-primitive-result expression))))
      (is (nil? (proof expression [primitive-double primitive-double])))
      (is (= {:reason :stale-definition-proof-input :source source}
             (stale-source-data #(entry/call-with-definition-inputs
                                  (fn [] (normal/result-fact expression))))))
      (let [body (list 'let* ['x 2.0 'y 3.0 'result expression]
                       '(if (clojure.core/nil? result) :absent :present))]
        (is (= {:reason :stale-definition-proof-input :source source}
               (stale-source-data #(entry/call-with-definition-inputs
                                    (fn [] (products/lower-local-products body)))))))
      (finally (reset-meta! source original-source)
               (reset-meta! generic original-generic)))))

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
(def ^:private source-ns (the-ns 'raster.compiler.passes.scalar.selected-primitive-evaluation-test))

(defn- proof [expression facts]
  (entry/call-with-definition-inputs
   #(scalar/checked-selected-primitive-evaluation expression facts)))

(deftest original-core-fp-completion-reuses-physical-operand-plans
  (doseq [[operation result] [['clojure.core/+ Double/TYPE]
                              ['clojure.core/- Double/TYPE]
                              ['clojure.core/* Double/TYPE]
                              ['clojure.core// Double/TYPE]
                              ['clojure.core/< Boolean/TYPE]]]
    (let [expression (list operation 'x 'y)
          summarize #(entry/call-with-definition-inputs
                      (fn [] (scalar/checked-core-fp-evaluation expression % source-ns)))]
      (is (= result (:result-carrier (summarize [primitive-double primitive-double]))))
      (doseq [floating [primitive-float boxed-float boxed-double]
              arguments [[primitive-double floating] [floating primitive-double]]]
        (is (= result (:result-carrier (summarize arguments)))))
      (doseq [unknown [nil {:dtype :double} {:carrier Double/TYPE}
                       {:carrier Object :presence :unknown :builtin-class Double}
                       {:carrier String :presence :present :builtin-class Double}
                       {:carrier Object :presence :present :builtin-class Number}
                       {:carrier Long/TYPE :presence :present}]]
        (is (nil? (summarize [primitive-double unknown]))))
      (is (nil? (summarize [primitive-float primitive-float]))
          "Float-only source promotion needs a separate original emission context")
      (is (nil? (summarize [boxed-double boxed-double]))
          "numeric payloads do not turn reference overloads into primitive result carriers")
      (is (nil? (summarize (list primitive-double primitive-double))))
      (is (= {:carrier result :presence :present
              :builtin-class (if (= result Boolean/TYPE) Boolean Double)}
             (entry/call-with-definition-inputs
              #(normal/completed-scalar-fact expression {'x primitive-double 'y primitive-double} source-ns))))))
  (is (nil? (entry/call-with-definition-inputs
             #(scalar/checked-core-fp-evaluation '(+ x y)
                                                [primitive-double primitive-double] source-ns))))
  (is (nil? (entry/call-with-definition-inputs
             #(normal/completed-scalar-fact '(clojure.core/+ x y)
                                            {'x primitive-double 'y primitive-double
                                             'clojure.core/+ primitive-double} source-ns))))
  (binding [entry/*emission-inputs* nil]
    (is (nil? (scalar/checked-core-fp-evaluation '(clojure.core// x y)
                                               [primitive-double primitive-double] source-ns))))
  (with-redefs-fn {#'clojure.core/- (fn [& _] 42.0)}
    #(is (= :intrinsic-source-definition-mismatch
            (try (entry/call-with-definition-inputs
                  (fn [] (scalar/checked-core-fp-evaluation '(clojure.core/- x y)
                                                           [primitive-double primitive-double] source-ns)))
                 nil
                 (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
  (let [expression '(clojure.core// x y)
        facts {'x primitive-double 'y primitive-double}]
    (entry/call-with-definition-inputs
     #(let [completed (normal/completed-scalar-fact expression facts source-ns)]
        (is (= Double/TYPE (:carrier completed)))
        (is (not (contains? completed :nonthrowing?)))
        (is (not (contains? completed :removable?)))))))

(deftest core-completion-requires-the-actual-producer-lookup
  (let [masked (create-ns (gensym "raster.test.masked-core."))
        producer (create-ns (gensym "raster.test.core-producer."))]
    (try
      (intern masked '+ (fn [& _] nil))
      (intern masked 'double (fn [& _] nil))
      (binding [*ns* producer] (alias 'clojure.core (ns-name masked)))
      (doseq [expression ['(clojure.core/+ 1.0 2.0) '(clojure.core/double nullable)]]
        (is (nil? (entry/call-with-definition-inputs
                   #(normal/completed-scalar-fact expression))))
        (is (= :intrinsic-source-definition-mismatch
               (try (entry/call-with-definition-inputs
                     #(normal/completed-scalar-fact expression {} producer))
                    nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (finally (remove-ns (ns-name producer)) (remove-ns (ns-name masked))))))

(deftest core-double-completion-has-independent-clojure-edge-case-oracles
  (doseq [[operation execute expected-class]
          [['clojure.core/+ (fn [^double x ^double y] (clojure.core/+ x y)) Double]
           ['clojure.core/- (fn [^double x ^double y] (clojure.core/- x y)) Double]
           ['clojure.core/* (fn [^double x ^double y] (clojure.core/* x y)) Double]
           ['clojure.core// (fn [^double x ^double y] (clojure.core// x y)) Double]
           ['clojure.core/< (fn [^double x ^double y] (clojure.core/< x y)) Boolean]]
          [x y] [[0.0 -0.0] [1.0 0.0] [-1.0 0.0]
                 [Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]
                 [Double/NaN 1.0] [Double/MAX_VALUE Double/MAX_VALUE]]]
    (let [actual (execute x y)
          completed (entry/call-with-definition-inputs
                     #(normal/completed-scalar-fact (list operation 'x 'y)
                                                   {'x primitive-double 'y primitive-double}
                                                   source-ns))]
      (is (= expected-class (class actual) (:builtin-class completed))))))

(deftest core-double-anchor-has-independent-reference-overload-oracles
  (doseq [[operation left-reference right-reference expected-class]
          [['clojure.core/+ (fn [x ^double y] (clojure.core/+ x y))
            (fn [^double x y] (clojure.core/+ x y)) Double]
           ['clojure.core/- (fn [x ^double y] (clojure.core/- x y))
            (fn [^double x y] (clojure.core/- x y)) Double]
           ['clojure.core/* (fn [x ^double y] (clojure.core/* x y))
            (fn [^double x y] (clojure.core/* x y)) Double]
           ['clojure.core// (fn [x ^double y] (clojure.core// x y))
            (fn [^double x y] (clojure.core// x y)) Double]
           ['clojure.core/< (fn [x ^double y] (clojure.core/< x y))
            (fn [^double x y] (clojure.core/< x y)) Boolean]]
          [payload fact] [[(float 1.25) boxed-float] [(float -0.0) boxed-float]
                          [(float Float/POSITIVE_INFINITY) boxed-float]
                          [Double/NaN boxed-double] [Double/MAX_VALUE boxed-double]]
          [execute x y facts] [[left-reference payload 0.0 [fact primitive-double]]
                               [right-reference -0.0 payload [primitive-double fact]]]]
    (let [actual (execute x y)
          completed (entry/call-with-definition-inputs
                     #(normal/completed-scalar-fact (list operation 'x 'y)
                                                   (zipmap '[x y] facts) source-ns))]
      (is (= expected-class (class actual) (:builtin-class completed)))
      (is (= Object (:carrier fact)) "numeric use never upgrades its reference producer"))))

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
      (is (= {:presence :unknown} (normal/result-fact divide)))
      (is (empty? (:definition-proofs @owner))))))

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
      ;; Roots/registration remain unchanged; only the description now claims
      ;; unrelated numeric code. A new definition snapshot alone would accept it.
      (alter-meta! source assoc :raster.core/deftm-source-body '[(clojure.core/* x y)]
                   :raster.core/return-tag 'double)
      (is (nil? (clojure.lang.Reflector/invokeInstanceMethod
                 @receiver "invk" (object-array [2.0 3.0]))))
      (is (nil? (entry/call-with-definition-inputs
                 #(scalar/checked-selected-primitive-result expression))))
      (is (nil? (proof expression [primitive-double primitive-double])))
      (let [body (list 'let* ['x 2.0 'y 3.0 'result expression]
                       '(if (clojure.core/nil? result) :absent :present))
            lowered (entry/call-with-definition-inputs #(products/lower-local-products body))
            evaluate #((eval (list 'fn* [] %)))]
        (is (= :absent (evaluate body) (evaluate lowered))))
      (finally (reset-meta! source original-source)
               (reset-meta! generic original-generic)))))

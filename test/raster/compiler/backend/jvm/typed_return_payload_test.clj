(ns raster.compiler.backend.jvm.typed-return-payload-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.numeric :as n]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.backend.jvm.typed-scalar :as typed-scalar]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(def ^:private fp {:carrier Double/TYPE :presence :present})
(def ^:private boxed-fp {:carrier Object :presence :present :builtin-class Double})

(defn- fixture! [scope-name body & [type]]
  (let [scope (create-ns (gensym scope-name))]
    (binding [*ns* scope]
      (refer 'clojure.core)
      (ns-unmap scope '*)
      (eval (list 'raster.core/deftm '*
                  ['x ':- (or type 'Double) 'y ':- (or type 'Double)]
                  ':- (or type 'Double) body)))
    (let [source (core/resolve-deftm-var (ns-resolve scope '*))]
      {:scope scope :generic (ns-resolve scope '*) :source source
       :receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))})))

(defn- with-selected-fixture [fixture run!]
  ;; Only the original operation's table selects this actual typed call. No
  ;; numerical invocation goes through the temporarily described generic root.
  (let [generic #'n/* original (meta generic)
        receiver (:receiver fixture)
        call (with-meta
               (list '.invk (symbol (str (:scope fixture))
                                    (str (:name (meta receiver)))) 'a 'b)
               {:raster.op/original 'raster.numeric/*})]
    (try
      (alter-meta! generic assoc :raster.core/dispatch-table
                   (:raster.core/dispatch-table (meta (:generic fixture))))
      (run! call)
      (finally (reset-meta! generic original)))))

(defn- payload [call facts]
  (entry/call-with-definition-inputs #(scalar/checked-selected-fp-payload call facts)))

(defn- execute-fp-program
  "Independent test oracle for this closed FP leaf, not a compiler lowering."
  [{:keys [operations result]} inputs]
  (letfn [(read-value [values argument]
            (if (instance? raster.compiler.ir.kernel_body.Literal argument)
              (:value argument)
              (get values argument)))
          (round-value [type value]
            (case type :float (.floatValue ^Number value) :double (.doubleValue ^Number value)))]
    (let [values
          (reduce
            (fn [values operation]
              (assert (instance? raster.compiler.ir.kernel_body.ScalarCompute operation))
              (let [{:keys [op arguments result-type]} (:expression operation)
                    arguments (mapv #(read-value values %) arguments)
                    a (first arguments) b (second arguments)
                    value (case op
                            :cast a
                            :* (clojure.core/* (double a) (double b))
                            :+ (clojure.core/+ (double a) (double b))
                            :div (clojure.core// (double a) (double b)))]
                (assoc values (get-in operation [:result :id]) (round-value result-type value))))
            inputs operations)]
      (read-value values result))))

(deftest actual-wrapper-retains-physical-object-and-primitive-payload-separately
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.return-payload." '(clojure.core// x y))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (is (some? (:raster.core/jit-spec (meta receiver))))
          (with-redefs [core/ensure-bytecode-implementation!
                        (fn [& _] (throw (AssertionError. "payload query forced the JIT")))]
            (is (nil? (payload call [fp fp]))))
          (core/ensure-bytecode-implementation! receiver)
          (let [emission (:raster.core/bytecode-emission (meta receiver))
                bridge (bytecode/retained-typed-return-bridge emission)
                proof (payload call [fp boxed-fp])]
            (is (= Object (:result-carrier bridge) (:result-carrier proof)))
            (is (= Double/TYPE (:payload-carrier bridge) (:payload-carrier proof)))
            (is (= :double (get-in bridge [:body-context :emitted-result-type])))
            (is (= :double (get-in proof [:computation :type])))
            (is (seq (get-in proof [:computation :operations])))
            (doseq [[x y] [[3.0 2.0] [-0.0 2.0] [1.0 0.0] [0.0 0.0]
                           [Double/MAX_VALUE 0.5]]]
              (let [actual (clojure.lang.Reflector/invokeInstanceMethod
                            @receiver "invk" (object-array [x y]))
                    expected (clojure.core// (double x) (double y))
                    from-static (.invoke ^java.lang.reflect.Method (:compute-method bridge)
                                         nil (object-array [x y]))
                    projected (execute-fp-program (:computation proof)
                                                 (zipmap (:parameters proof) [x y]))]
                (is (instance? Double actual))
                (is (or (and (Double/isNaN expected) (Double/isNaN actual))
                        (= (Double/doubleToRawLongBits expected)
                           (Double/doubleToRawLongBits actual))))
                (is (or (and (Double/isNaN actual) (Double/isNaN from-static))
                        (= (Double/doubleToRawLongBits actual)
                           (Double/doubleToRawLongBits from-static))))
                (is (or (and (Double/isNaN actual) (Double/isNaN projected))
                        (= (Double/doubleToRawLongBits actual)
                           (Double/doubleToRawLongBits projected))))))
            (is (nil? (bytecode/retained-typed-return-bridge
                       (dissoc emission :typed-return-bridges))))
            (is (nil? (bytecode/retained-typed-return-bridge
                       (assoc emission :implementation (Object.)))))
            (is (nil? (payload call [fp {:carrier Object :presence :unknown}])))
            (is (nil? (payload call [fp]))))))
      (finally (remove-ns (ns-name scope))))))

(deftest a-hidden-object-to-primitive-return-conversion-is-not-a-payload-recipe
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.return-coercion." '(clojure.core/identity x))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (core/ensure-bytecode-implementation! receiver)
          (let [emission (:raster.core/bytecode-emission (meta receiver))
                context (bytecode/retained-body-context emission)]
            (is (= :double (:return-stack-type context)))
            (is (= :ref (:emitted-result-type context)))
            (is (nil? (bytecode/retained-typed-return-bridge emission)))
            (is (nil? (payload call [fp fp]))))))
      (finally (remove-ns (ns-name scope))))))

(deftest an-actual-float-bridge-preserves-float-rounding-and-bits
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.float-return-payload."
                  '(clojure.core/float (clojure.core// x y)) 'Float)]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (core/ensure-bytecode-implementation! receiver)
          (let [proof (payload call [{:carrier Float/TYPE :presence :present}
                                    {:carrier Object :presence :present :builtin-class Float}])]
            (is (= Object (:result-carrier proof)))
            (is (= Float/TYPE (:payload-carrier proof)))
            (is (= :float (get-in proof [:computation :type])))
            (doseq [[x y] [[1.0 3.0] [-0.0 2.0] [1.0 0.0] [0.0 0.0]
                           [Float/MAX_VALUE 0.5]]]
              (let [x (float x) y (float y)
                    actual (clojure.lang.Reflector/invokeInstanceMethod
                            @receiver "invk" (object-array [x y]))
                    projected (execute-fp-program (:computation proof)
                                                 (zipmap (:parameters proof) [x y]))
                    expected (.floatValue ^Number (clojure.core// (double x) (double y)))]
                (is (instance? Float actual))
                (is (or (and (Float/isNaN actual) (Float/isNaN expected))
                        (= (Float/floatToRawIntBits actual) (Float/floatToRawIntBits expected))))
                (is (or (and (Float/isNaN actual) (Float/isNaN projected))
                        (= (Float/floatToRawIntBits actual) (Float/floatToRawIntBits projected)))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest parent-object-arithmetic-does-not-borrow-a-child-payload-carrier
  (let [expression '(clojure.core//
                     (raster.numeric// a b) (raster.numeric// a b))]
    (is (nil? (payload expression [fp fp])))
    (is (nil? (scalar/checked-selected-fp-payload expression [fp fp])))))

(deftest nested-width-stamps-are-not-actual-producer-carriers
  (let [inner (with-meta '(clojure.core/* x x) {:raster.type/tag 'float})
        body (list 'clojure.core/+ inner 'y)
        iface (:iface-name (types/ensure-fn-interface! '[double double]))
        {:keys [implementation emission]}
        (#'core/emit-owned-bytecode!
         bytecode/compile-typed-impl! "raster.compiled.RetainedWidthCounterexample"
         '[x y] '[double double] 'double [body] *ns* iface 'width-counterexample)
        bridge (bytecode/retained-typed-return-bridge emission)
        ;; Deliberately demonstrate why root dtype equality is insufficient.
        unsafe (#'scalar/checked-scalar-program body {'x :double 'y :double})
        x 16777217.0 y 0.25
        actual (clojure.lang.Reflector/invokeInstanceMethod implementation "invk"
                                                           (object-array [x y]))
        expected (clojure.core/+ (clojure.core/* x x) y)]
    (is (= Double/TYPE (:payload-carrier bridge)))
    (is (= :double (:type unsafe)))
    (is (= expected actual))
    (is (not= actual (execute-fp-program unsafe {'x x 'y y})))
    (is (= Double/TYPE (#'scalar/completed-fp-body-carrier body {'x Double/TYPE 'y Double/TYPE})))
    (is (nil? (#'scalar/completed-fp-body-carrier body {'x Double/TYPE 'y Double/TYPE} true))
        "every admitted node must agree before its SSA can become a payload recipe")))

(deftest nonthrowing-intrinsic-evidence-does-not-invent-a-result-carrier-recipe
  (let [descriptor intrinsics/descriptor
        expression '(clojure.core/+ x y)
        carriers {'x Double/TYPE 'y Double/TYPE}]
    (is (= Double/TYPE (#'scalar/completed-fp-body-carrier expression carriers)))
    ;; Model independently extending descriptor totality, without changing
    ;; source identity or teaching this arithmetic-only summary a new recipe.
    (doseq [facts [{:arity 2 :kind :fn :primitive-fp-total true :c {:fn "hypot"}}
                  {:arity 2 :kind :infix :primitive-fp-total true :c "%"}]]
      (with-redefs [intrinsics/descriptor (fn [op]
                                          (if (= op 'clojure.core/+) facts (descriptor op)))]
        (is (nil? (#'scalar/completed-fp-body-carrier expression carriers)))))
    (with-redefs [intrinsics/descriptor
                  (fn [op]
                    (if (= op 'clojure.core/exp)
                      {:arity 1 :kind :fn :primitive-fp-total true :c {:fn "exp"}}
                      (descriptor op)))]
      (is (nil? (#'scalar/completed-fp-body-carrier '(clojure.core/exp x) carriers))))))

(defn- canonical-operation [expression]
  (entry/call-with-definition-inputs #(scalar/canonical-selected-fp-operation expression)))

(deftest canonical-arithmetic-keeps-callee-correspondence-and-object-result-separate
  (doseq [[tag body carrier] [['Double '(clojure.core/* x y) Double/TYPE]
                              ['Float '(clojure.core/float (clojure.core/* x y)) Float/TYPE]]]
    (let [{:keys [scope] :as fixture} (fixture! "raster.canonical-body." body tag)]
      (try
        (with-selected-fixture
          fixture
          (fn [call]
            (with-redefs [core/ensure-bytecode-implementation!
                          (fn [& _] (throw (AssertionError. "correspondence forced the JIT")))]
              (let [proof (canonical-operation call)]
                (is (= :canonical-selected-fp-operation (:scope proof)))
                (is (= :* (:operation proof)))
                (is (= [carrier carrier] (:parameter-carriers proof)))
                (is (= Object (:result-carrier proof)))
                (is (= carrier (:payload-carrier proof)))
                (is (not (contains? proof :nonthrowing?)))))))
        (finally (remove-ns (ns-name scope)))))))

(deftest canonical-operation-does-not-follow-a-nominal-multiply-name
  (doseq [body ['(clojure.core// x y)
                '(clojure.core/* y x)
                '(clojure.core/* x x)
                '(clojure.core/+ (clojure.core/* x y) 1.0)
                '(clojure.core/double (clojure.core/float (clojure.core/* x y)))]]
    (let [{:keys [scope] :as fixture} (fixture! "raster.noncanonical-body." body)]
      (try
        (with-selected-fixture fixture #(is (nil? (canonical-operation %))))
        (finally (remove-ns (ns-name scope)))))))

(deftest selected-float-width-cannot-be-overridden-by-a-double-result-stamp
  (let [{:keys [scope] :as fixture}
        (fixture! "raster.selected-width." '(clojure.core/float (clojure.core/* x y)) 'Float)]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (let [float-call (with-meta call (assoc (meta call) :tag 'float))
                lowered (entry/call-with-definition-inputs
                         #(let [builder (scalar/make-lowerer
                                         {:scalar-types {'a :float 'b :float}
                                          :lower-index (fn [expression _] expression)
                                          :decline! (fn [rule message data]
                                                      (throw (ex-info message (assoc data :rule rule))))})]
                            ((:lower builder) float-call :float {})))]
            (is (= :float (:type lowered)))
            (is (= :float (get-in (last (:operations lowered)) [:expression :result-type]))))
          (let [stamped-call (with-meta call (assoc (meta call) :tag 'double))]
            (is (= Float/TYPE (:payload-carrier (canonical-operation stamped-call))))
            (is (= :selected-scalar-source-width
                   (entry/call-with-definition-inputs
                    #(try
                       (let [builder (scalar/make-lowerer
                                      {:scalar-types {'a :float 'b :float}
                                       :lower-index (fn [expression _] expression)
                                       :decline! (fn [rule message data]
                                                   (throw (ex-info message (assoc data :rule rule))))})]
                         ((:lower builder) stamped-call :double {})
                         nil)
                       (catch clojure.lang.ExceptionInfo error
                         (or (:rule (ex-data error)) (ex-data error))))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest selected-source-host-call-does-not-redispatch-its-nominal-operation
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.selected-host-source." '(clojure.core// x y))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (core/ensure-bytecode-implementation! receiver)
          (is (= 2.0 (clojure.lang.Reflector/invokeInstanceMethod
                       @receiver "invk" (object-array [4.0 2.0]))))
          (is (= :typed-scalar-owned-source-required
                 (try (typed-scalar/evaluate-expression scope {'a 4.0 'b 2.0} call)
                      nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
          (is (= :typed-scalar-owned-source-required
                 (try (typed-scalar/evaluate-expression scope {} call)
                      nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
              "do not begin argument evaluation before refusing a missing source context")
          (entry/call-with-definition-inputs
           (fn []
             (let [projection (entry/selected-parameter-carriers 'raster.numeric/* (second call) 2)
                   method (:invocation-method projection)
                   policy (:raster.core/source-policy (meta (:source fixture)))
                   collector entry/*emission-inputs*
                   execute (source-owner/compile-function
                            '[a b] call (fn [parameters body] (list 'fn* parameters body))
                            {:source-policy policy :source-ns scope :captures []
                             :reference-bindings {} :methods [method]
                             :guard #(entry/assert-definition-inputs-current! @collector)})
                   observations @collector
                   evaluate-source (fn [expression environment]
                                     (assert (identical? call expression))
                                     (entry/assert-definition-inputs-current! observations)
                                     (let [result (execute (get environment 'a) (get environment 'b))]
                                       (entry/assert-definition-inputs-current! observations)
                                       result))
                   region (soac/lambda-form '[a b] [call])]
               (doseq [[a b] [[4.0 2.0] [-9.0 3.0] [0.0 2.0]]]
                 (let [expected (clojure.lang.Reflector/invokeInstanceMethod
                                 @receiver "invk" (object-array [a b]))]
                   (is (= expected (typed-scalar/evaluate-expression
                                    scope {'a a 'b b} call evaluate-source)))
                   (is (= [{:type :double :value expected}]
                          (typed-scalar/evaluate-region
                           scope region [{:type :double :value a} {:type :double :value b}]
                           [:double] evaluate-source)))))
               (is (thrown? NullPointerException
                            (typed-scalar/evaluate-expression
                             scope {'a nil 'b 2.0} call evaluate-source)))
               (let [ordered (with-meta
                               (list '.invk (second call) '(clojure.core/double a)
                                     '(clojure.core// 1 0))
                               (meta call))
                     execute-ordered (source-owner/compile-function
                                      '[a] ordered (fn [parameters body] (list 'fn* parameters body))
                                      {:source-policy policy :source-ns scope :captures []
                                       :reference-bindings {} :methods [method]
                                       :guard #(entry/assert-definition-inputs-current! @collector)})
                     ordered-observations @collector
                     evaluate-ordered (fn [expression environment]
                                        (assert (identical? ordered expression))
                                        (entry/assert-definition-inputs-current! ordered-observations)
                                        (try (execute-ordered (get environment 'a))
                                             (finally
                                               (entry/assert-definition-inputs-current!
                                                ordered-observations))))]
                 (is (thrown? NullPointerException
                              (typed-scalar/evaluate-expression
                               scope {'a nil} ordered evaluate-ordered)))
                 (is (thrown? ArithmeticException
                              (typed-scalar/evaluate-expression
                               scope {'a 1.0} ordered evaluate-ordered)))))))))
      (finally (remove-ns (ns-name scope))))))

(deftest scalar-lowering-refuses-an-actually-selected-noncanonical-implementation
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.selected-source-guard." '(clojure.core// x y))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (is (some? (entry/call-with-definition-inputs
                      #(entry/selected-parameter-carriers 'raster.numeric/* (second call) 2))))
          (is (= :selected-scalar-source-correspondence
                 (entry/call-with-definition-inputs
                  #(try
                     (let [builder (scalar/make-lowerer
                                    {:scalar-types {'a :double 'b :double}
                                     :lower-index (fn [expression _] expression)
                                     :decline! (fn [rule message data]
                                                 (throw (ex-info message (assoc data :rule rule))))})]
                       ((:lower builder) call :double {})
                       nil)
                     (catch clojure.lang.ExceptionInfo error
                       (or (:rule (ex-data error)) (ex-data error)))))))))
      ;; Only after the cold-source refusal: establish the actual implementation
      ;; oracle independently, without changing the retained alias table mid-query.
      (core/ensure-bytecode-implementation! receiver)
      (is (= 2.0 (clojure.lang.Reflector/invokeInstanceMethod
                   @receiver "invk" (object-array [4.0 2.0]))))
      (finally (remove-ns (ns-name scope))))))

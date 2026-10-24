(ns raster.compiler.backend.jvm.typed-return-payload-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.numeric :as n]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(def ^:private fp {:carrier Double/TYPE :presence :present})
(def ^:private boxed-fp {:carrier Object :presence :present :builtin-class Double})

(defn- fixture! [scope-name body & [type]]
  (let [scope (create-ns (gensym scope-name))
        [left right] (if (vector? type) type [(or type 'Double) (or type 'Double)])]
    (binding [*ns* scope]
      (refer 'clojure.core)
      (ns-unmap scope '*)
      (eval (list 'raster.core/deftm '*
                  ['x ':- left 'y ':- right]
                  ':- left body)))
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
                            :neg (clojure.core/- (double a))
                            :log (Math/log (double a))
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

(deftest actual-long-boxing-is-payload-evidence-not-a-primitive-source-carrier
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.long-return-payload." '(clojure.core/* x y) 'Long)]
    (try
      (is (some? (:raster.core/jit-spec (meta receiver))))
      (with-redefs [core/ensure-bytecode-implementation!
                    (fn [& _] (throw (AssertionError. "bridge query forced the JIT")))]
        (is (nil? (bytecode/retained-typed-return-bridge
                   (:raster.core/bytecode-emission (meta receiver))))))
      (core/ensure-bytecode-implementation! receiver)
      (let [emission (:raster.core/bytecode-emission (meta receiver))
            bridge (bytecode/retained-typed-return-bridge emission)
            outcome (fn [f]
                      (try (let [value (f)] [:value (class value) value])
                           (catch Exception error [:exception (class error)])))]
        (is (= :boxed-primitive-static-return (:kind bridge)))
        (is (= Object (:result-carrier bridge)))
        (is (= Long/TYPE (:payload-carrier bridge)))
        (is (= [Long/TYPE Long/TYPE] (:parameter-carriers bridge)))
        (is (= :long (get-in bridge [:body-context :emitted-result-type])))
        (doseq [[x y] [[0 1] [4097 3] [Long/MIN_VALUE 1] [Long/MAX_VALUE -1] [Long/MAX_VALUE 2]]]
          (is (= (outcome #(clojure.core/* (long x) (long y)))
                 (outcome #(clojure.lang.Reflector/invokeInstanceMethod
                            @receiver "invk" (object-array [x y]))))))
        ;; This shared wrapper observation does not broaden the FP SSA recipe.
        (with-selected-fixture fixture
          (fn [call]
            (is (nil? (payload call [{:carrier Long/TYPE :presence :present}
                                    {:carrier Long/TYPE :presence :present}])))))
        (doseq [changed [(dissoc emission :typed-return-bridges)
                         (assoc emission :implementation (Object.))
                         (update emission :typed-return-bridges #(into % %))
                         (assoc-in emission [:typed-return-bridges 0 :kind] :boxed-fp-static-return)
                         (assoc-in emission [:typed-return-bridges 0 :payload-carrier] Double/TYPE)
                         (assoc-in emission [:typed-return-bridges 0 :parameter-carriers] [Object Object])]]
          (is (nil? (bytecode/retained-typed-return-bridge changed)))))
      (finally (remove-ns (ns-name scope))))))

(deftest an-unused-integral-formal-does-not-acquire-fp-argument-completion
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.mixed-return-payload." '(clojure.core/float x) '[Float Long])]
    (try
      (core/ensure-bytecode-implementation! receiver)
      (let [bridge (bytecode/retained-typed-return-bridge
                    (:raster.core/bytecode-emission (meta receiver)))]
        (is (= [Float/TYPE Long/TYPE] (:parameter-carriers bridge)))
        (is (= Float/TYPE (:payload-carrier bridge))))
      (with-selected-fixture fixture
        (fn [call]
          (doseq [argument [{:carrier Float/TYPE :presence :present}
                           {:carrier Object :presence :present :builtin-class Float}
                           {:carrier Long/TYPE :presence :present}]]
            (is (nil? (payload call [{:carrier Float/TYPE :presence :present} argument]))
                "even an unused integral formal retains its original argument conversion"))))
      (finally (remove-ns (ns-name scope))))))

(deftest a-hidden-long-result-unboxing-does-not-become-a-boxing-proof
  (let [{:keys [scope receiver]} (fixture! "raster.long-return-coercion." '(clojure.core/identity x) 'Long)]
    (try
      (core/ensure-bytecode-implementation! receiver)
      (let [emission (:raster.core/bytecode-emission (meta receiver))
            context (bytecode/retained-body-context emission)]
        (is (= :long (:return-stack-type context)))
        (is (= :ref (:emitted-result-type context)))
        (is (nil? (bytecode/retained-typed-return-bridge emission))))
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

(deftest math-payload-requires-the-actual-java-method-occurrence
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.math-return-payload." '(Math/log (clojure.core/double x)))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (with-redefs [core/ensure-bytecode-implementation!
                        (fn [& _] (throw (AssertionError. "Math query forced the JIT")))]
            (is (nil? (payload call [fp fp]))))
          (core/ensure-bytecode-implementation! receiver)
          (let [emitted (:raster.core/bytecode-emission (meta receiver))
                original (first (:body emitted))
                record (bytecode/retained-invocation-context emitted original)
                method (:method record)
                proof (payload call [fp fp])
                owner (atom {:static [] :values []})]
            (is (= :java-static (:invocation-mode record)))
            (is (= (.getMethod java.lang.Math "log" (into-array Class [Double/TYPE])) method))
            (is (= [:double] (mapv :emitted-type (:input-completions record))))
            (is (= [:identity] (mapv :adaptation (:input-completions record))))
            (is (nil? (bytecode/retained-invocation-context
                       emitted (with-meta (apply list original) (meta original)))))
            (is (= Object (:result-carrier proof)))
            (is (= Double/TYPE (:payload-carrier proof)))
            (doseq [x [-1.0 -0.0 0.0 0.5 1.0 Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN]]
              (let [actual (clojure.lang.Reflector/invokeInstanceMethod
                            @receiver "invk" (object-array [x 7.0]))
                    projected (execute-fp-program (:computation proof)
                                                 (zipmap (:parameters proof) [x 7.0]))]
                (is (instance? Double actual))
                (is (or (and (Double/isNaN actual) (Double/isNaN projected))
                        (= (Double/doubleToRawLongBits actual) (Double/doubleToRawLongBits projected))))))
            (binding [entry/*emission-inputs* owner]
              (is (some? (scalar/checked-selected-fp-payload call [fp fp]))))
            (try
              (ns-unmap scope 'Math)
              (.importClass scope 'Math java.lang.StrictMath)
              (is (nil? (payload call [fp fp])) "a current Class lookup cannot replace the emitted Method")
              (is (= :stale-source-resolution
                     (:reason (ex-data (try (entry/assert-definition-inputs-current! @owner) nil
                                           (catch clojure.lang.ExceptionInfo error error))))))
              (finally
                (ns-unmap scope 'Math)
                (.importClass scope 'Math java.lang.Math))))))
      (finally (remove-ns (ns-name scope))))))

(deftest a-different-actual-java-method-is-not-the-math-payload-recipe
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.foreign-math-payload." '(Math/log (clojure.core/double x)))]
    (try
      (ns-unmap scope 'Math)
      (.importClass scope 'Math java.lang.StrictMath)
      (with-selected-fixture
        fixture
        (fn [call]
          (core/ensure-bytecode-implementation! receiver)
          (let [emitted (:raster.core/bytecode-emission (meta receiver))
                original (first (:body emitted))
                record (bytecode/retained-invocation-context emitted original)]
            (is (= :java-static (:invocation-mode record)))
            (is (= java.lang.StrictMath (.getDeclaringClass ^java.lang.reflect.Method (:method record))))
            (is (nil? (payload call [fp fp]))))))
      (finally (remove-ns (ns-name scope))))))

(deftest missing-java-call-history-is-not-reconstructed-from-a-current-class
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.unrecorded-math-payload." '(Math/log (clojure.core/double x)))
        retain-var (ns-resolve 'raster.compiler.backend.jvm.bytecode 'retain-invocation-context!)
        original-retain @retain-var]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          ;; Emit real code without the new optional observation, modeling an
          ;; older actual emission. Neither its owner nor its Classes are forged.
          (with-redefs-fn {retain-var
                          (fn [method receiver arguments mode & remaining]
                            (when-not (= :java-static mode)
                              (apply original-retain method receiver arguments mode remaining)))}
            #(core/ensure-bytecode-implementation! receiver))
          (is (some? (bytecode/retained-typed-return-bridge
                      (:raster.core/bytecode-emission (meta receiver)))))
          (with-redefs [core/ensure-bytecode-implementation!
                        (fn [& _] (throw (AssertionError. "missing history forced preparation")))]
            (is (nil? (payload call [fp fp]))))))
      (finally (remove-ns (ns-name scope))))))

(deftest double-unary-negation-keeps-the-original-primitive-recipe
  (let [{:keys [scope receiver] :as fixture}
        (fixture! "raster.double-negation-payload." '(clojure.core/- x))]
    (try
      (with-selected-fixture
        fixture
        (fn [call]
          (core/ensure-bytecode-implementation! receiver)
          (let [proof (payload call [fp fp])]
            (is (= Double/TYPE (:payload-carrier proof)))
            (doseq [x [-0.0 0.0 -3.0 Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
              (let [actual (clojure.lang.Reflector/invokeInstanceMethod
                            @receiver "invk" (object-array [x 7.0]))
                    projected (execute-fp-program (:computation proof)
                                                 (zipmap (:parameters proof) [x 7.0]))]
                (is (= (Double/doubleToRawLongBits (clojure.core/- (double x)))
                       (Double/doubleToRawLongBits actual) (Double/doubleToRawLongBits projected))))))))
      (finally (remove-ns (ns-name scope))))))

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

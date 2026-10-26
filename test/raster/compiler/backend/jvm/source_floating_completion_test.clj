(ns raster.compiler.backend.jvm.source-floating-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]))

(defn- fixture [tag operation frozen operand-policy & [literal-operand]]
  (let [operand (or literal-operand
                    '(do (clojure.core/aset effects 0
                          (clojure.core/inc (clojure.core/aget effects 0))) value))
        node (completion/complete-value
              (completion/source-floating-cast-contract operation {:unchecked-math frozen}) operand)
        tags [tag 'longs]
        interface (types/ensure-fn-interface! tags)
        body [node]
        emitted (binding [*unchecked-math* operand-policy]
                  (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.SourceFloatingCompletion"
                   '[value effects] tags 'Object body *ns* (:iface-name interface)
                   'source-floating-completion-test))
        parameters [(with-meta 'value {:tag tag}) (with-meta 'effects {:tag 'longs})]
        operand-function (binding [*unchecked-math* operand-policy]
                           (eval (list 'fn parameters operand)))
        source-cast (binding [*unchecked-math* frozen]
                      (eval (list 'fn [(with-meta 'input {:tag tag})]
                                  (list (symbol "clojure.core" (name operation)) 'input))))
        ;; Literal-operand controls contain no policy-sensitive arithmetic. Compile their
        ;; original whole expression directly to retain the exact primitive overload.
        original (if literal-operand
                   (binding [*unchecked-math* frozen]
                     (eval (list 'fn parameters
                                 (list (symbol "clojure.core" (name operation)) operand))))
                   (fn [value effects] (source-cast (operand-function value effects))))
        projected (binding [*unchecked-math* operand-policy]
                    (eval (list 'fn parameters (completion/project-to-jvm-source node))))]
    (assoc emitted :original original :projected projected :node node :input-body body)))

(defn- outcome [invoke make-value]
  (let [effects (long-array 2)
        value (make-value effects)
        result (try (let [result (invoke value effects)]
                      [:value (class result) (Float/floatToRawIntBits (float result))])
                    (catch Throwable error [:error (class error)]))]
    [result (vec effects)]))

(defn- outlined-invocation [{:keys [emission input-body node]}]
  (let [region (bytecode/compile-source-floating-completion-region emission input-body node)]
    (fn [value effects]
      (apply (:invoke region)
             (map {'value value 'effects effects} (:parameters region))))))

(deftest planned-float-completion-chooses-an-object-port-without-fabricating-an-occurrence
  (let [{:keys [emission]} (fixture 'Object :float false true 'value)
        operand (with-meta 'input {:tag 'Object :raster.type/tag 'double})
        expression (completion/complete-value
                    (completion/source-floating-cast-contract :float {:unchecked-math false}) operand)
        stale? (atom false)
        failure (ex-info "planned owner retired" {:reason :test-owner-retired})
        guard #(when @stale? (throw failure))
        region (bytecode/compile-planned-source-floating-completion-region
                emission expression {:unchecked-math true} guard)
        method ^java.lang.reflect.Method (:method region)
        source-cast (binding [*unchecked-math* false]
                      (eval '(fn [^Object input] (clojure.core/float input))))
        result (fn [f value]
                 (try [:value (Float/floatToRawIntBits (f value))]
                      (catch Throwable error [:error (class error)])))]
    (is (= [Object] (vec (.getParameterTypes method))))
    (is (identical? Float/TYPE (.getReturnType method)))
    (doseq [value [(double -0.0) 0.25 Double/MAX_VALUE Double/MIN_VALUE Double/NaN
                   Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY nil true \a]]
      (is (= (result source-cast value) (result (:invoke region) value))))
    (let [calls (atom 0)
          value (proxy [Number] []
                  (intValue [] (throw (AssertionError. "wrong conversion")))
                  (longValue [] (throw (AssertionError. "wrong conversion")))
                  (doubleValue [] (swap! calls inc) (double 0.25))
                  (floatValue [] (swap! calls inc) (float 0.25)))
          expected (result source-cast value)
          expected-calls @calls]
      (reset! calls 0)
      (is (= expected (result (:invoke region) value)))
      (is (= expected-calls @calls))
      (reset! stale? true)
      (reset! calls 0)
      (is (identical? failure (try ((:invoke region) value) nil
                                  (catch Throwable error error))))
      (is (zero? @calls) "owner refusal precedes the Number's conversion effects"))
    (doseq [bad [(completion/complete-value (:contract (completion/canonical-parts expression))
                                          '(do (println :effect) input))
                 '(clojure.core/float input)]]
      (is (= :planned-source-floating-completion-form
             (try (bytecode/compile-planned-source-floating-completion-region
                   emission bad {:unchecked-math true} guard) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest planned-unchecked-float-keeps-the-selected-source-operation
  (let [{:keys [emission]} (fixture 'Object :float false true 'value)
        expression (completion/complete-value
                    (completion/source-floating-cast-contract :unchecked-float
                                                              {:unchecked-math false}) 'input)
        region (bytecode/compile-planned-source-floating-completion-region
                emission expression {:unchecked-math true} (fn []))
        source-cast (binding [*unchecked-math* false]
                      (eval '(fn [^Object input] (clojure.core/unchecked-float input))))]
    (doseq [value [(double -0.0) 1.0e40 Double/NaN]]
      (is (= (outcome (fn [x _] (source-cast x)) (constantly value))
             (outcome (fn [x _] ((:invoke region) x)) (constantly value)))))))

(defn- actual-planned-completion [tag contract]
  (let [expression (completion/complete-value contract 'input)
        body [expression]
        interface (types/ensure-fn-interface! [tag])
        policy {:unchecked-math false}
        emitted (binding [*unchecked-math* false]
                  (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.PlannedCompletionSource"
                   '[input] [tag] (if (= contract completion/object-input-contract) 'Object
                                    (:tag (completion/invocation-result-description contract)))
                   body *ns* (:iface-name interface) 'planned-completion-source))
        observed (bytecode/retained-input-completion-context (:emission emitted) body expression)]
    {:emission (:emission emitted) :expression expression :observed observed
     :port (get (:locals-before observed) 'input) :policy policy}))

(deftest planned-canonical-completions-preserve-actual-port-and-result-abis
  (let [context (completion/bytecode-operand-context {})]
    (doseq [[tag contract] [['float completion/object-input-contract]
                            ['Object completion/object-input-contract]
                            ['Object (completion/bytecode-boxed-parameter-contract Float/TYPE)]
                            ['Object (completion/bytecode-floating-input-contract Float/TYPE context)]
                            ['float (completion/bytecode-boxed-parameter-contract Float/TYPE)]
                            ['float (completion/bytecode-floating-input-contract Float/TYPE context)]
                            ['float (completion/source-floating-cast-contract :float {:unchecked-math false})]]
            :let [{:keys [emission expression observed port policy]}
                  (actual-planned-completion tag contract)
                  region (bytecode/compile-planned-completion-region
                          emission expression port observed policy (fn []))
                  method ^java.lang.reflect.Method (:method region)
                  object? (= contract completion/object-input-contract)]]
      (is (= [(if (= tag 'float) Float/TYPE Object)] (vec (.getParameterTypes method))))
      (is (identical? (if object? Object Float/TYPE) (.getReturnType method)))
      (is (= (assoc port :slot 0) (get-in region [:physical-ports 'input])))
      (doseq [value [(float 0.25) (float -0.0) (Float/intBitsToFloat 0x7fc01234)
                     Float/POSITIVE_INFINITY]]
        (let [actual ((:invoke region) value)]
          (is (instance? Float actual))
          (is (= (Float/floatToRawIntBits value) (Float/floatToRawIntBits actual)))))
      (when (and (= tag 'Object) object?)
        (let [opaque (Object.)]
          (is (identical? opaque ((:invoke region) opaque)))
          (is (nil? ((:invoke region) nil))))))))

(deftest planned-numeric-entries-preserve-each-actual-primitive-result-abi
  (doseq [[tag carrier value] [['float Float/TYPE (float -0.25)]
                               ['double Double/TYPE (double -0.25)]
                               ['long Long/TYPE (long 9007199254740993)]
                               ['int Integer/TYPE (int -17)]]
          input-tag [tag 'Object]
          :let [contract (completion/bytecode-boxed-parameter-contract carrier)
                {:keys [emission expression observed port policy]}
                (actual-planned-completion input-tag contract)
                region (bytecode/compile-planned-completion-region
                        emission expression port observed policy (fn []))
                method ^java.lang.reflect.Method (:method region)]]
    (is (identical? carrier (.getReturnType method)))
    (is (= [(if (= input-tag 'Object) Object carrier)] (vec (.getParameterTypes method))))
    (is (= value ((:invoke region) value)))
    (is (= (class value) (class ((:invoke region) value))))
    (when (= input-tag 'Object)
      (is (thrown? NullPointerException ((:invoke region) nil)))
      (is (thrown? ClassCastException ((:invoke region) "not a number"))))))

(deftest planned-primitive-float-entry-retains-number-effects-and-current-owner
  (let [{:keys [emission expression observed port policy]}
        (actual-planned-completion 'Object (completion/bytecode-boxed-parameter-contract Float/TYPE))
        stale? (atom false)
        failure (Exception. "retired planned completion")
        guard #(when @stale? (throw failure))
        region (bytecode/compile-planned-completion-region emission expression port observed policy guard)
        calls (atom 0)
        value (proxy [Number] []
                (intValue [] (throw (AssertionError. "wrong conversion")))
                (longValue [] (throw (AssertionError. "wrong conversion")))
                (doubleValue [] (throw (AssertionError. "wrong conversion")))
                (floatValue [] (swap! calls inc) (float -0.0)))]
    (is (= (Float/floatToRawIntBits (float -0.0))
           (Float/floatToRawIntBits ((:invoke region) value))))
    (is (= 1 @calls))
    (is (thrown? NullPointerException ((:invoke region) nil)))
    (is (thrown? ClassCastException ((:invoke region) true)))
    (reset! stale? true)
    (is (identical? failure (try ((:invoke region) value) nil (catch Exception error error))))
    (is (= 1 @calls))
    (doseq [[bad-expression bad-port bad-observed]
            [[expression (assoc port :type :double) observed]
             [expression (assoc port :reference-class String) observed]
             [expression port (assoc observed :completion-carrier :ref)]
             [(completion/complete-value (:contract observed) '(do input input)) port observed]
             [(completion/complete-value (completion/bytecode-boxed-parameter-contract Boolean/TYPE) 'input)
              port (assoc observed :contract (completion/bytecode-boxed-parameter-contract Boolean/TYPE))]]]
      (is (= :planned-completion-form
             (try (bytecode/compile-planned-completion-region
                   emission bad-expression bad-port bad-observed policy (fn [])) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :planned-completion-carrier
           (try (bytecode/compile-planned-completion-region
                 emission expression port (assoc observed :adaptation :identity) policy (fn [])) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(defn- actual-literal-completion [value]
  (let [expression (completion/complete-value completion/object-input-contract value)
        body [expression]
        interface (types/ensure-fn-interface! [])
        emitted (binding [*unchecked-math* false]
                  (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.PlannedLiteralCompletion"
                   [] [] 'Object body *ns* (:iface-name interface) 'planned-literal-completion-source))]
    {:emission (:emission emitted) :expression expression :original (:implementation emitted)
     :observed (bytecode/retained-input-completion-context (:emission emitted) body expression)
     :policy {:unchecked-math false}}))

(deftest planned-literal-completion-emits-zero-ports-with-original-numeric-width
  (doseq [value [(float -0.0) (double -0.0) (int -17) (long 9007199254740993)
                (byte -7) (short 257)]]
    (let [{:keys [emission expression observed policy original]} (actual-literal-completion value)
          stale? (atom false)
          guard #(when @stale? (throw (ex-info "literal owner retired" {:reason :literal-owner-retired})))
          region (bytecode/compile-planned-completion-region emission expression nil observed policy guard)
          method ^java.lang.reflect.Method (:method region)]
      (is (empty? (.getParameterTypes method)))
      (is (empty? (:physical-ports region)))
      (is (identical? Object (.getReturnType method)))
      (is (identical? value (original)) "the original owned emission retains the source literal")
      (dotimes [_ 2]
        (let [result ((:invoke region))]
          (is (identical? (original) result)
              "the planned recipe uses the same owned literal reference as the original emission")
          (is (= (class value) (class result)))
          (cond
            (instance? Float value)
            (is (= (Float/floatToRawIntBits value) (Float/floatToRawIntBits result)))
            (instance? Double value)
            (is (= (Double/doubleToRawLongBits value) (Double/doubleToRawLongBits result)))
            :else (is (= value result)))))
      (doseq [[node port observation]
              [[expression {:type :double} observed]
               [expression nil (assoc observed :input-carrier :double)]
               [expression nil (assoc observed :completion-carrier :double)]
               [(completion/complete-value completion/object-input-contract '(do 0.0)) nil observed]]]
        (is (= :planned-completion-form
               (try (bytecode/compile-planned-completion-region emission node port observation policy (fn []))
                    nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (reset! stale? true)
      (is (= :literal-owner-retired
             (try ((:invoke region)) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest planned-result-snapshots-retain-primitive-payload-and-exact-return-recipe
  (let [{:keys [emission]} (fixture 'Object :float false false 'value)
        context (completion/bytecode-operand-context {:object-result? true})
        result 'numerical-result
        snapshot (fn [target]
                   (completion/complete-value
                    completion/object-input-contract
                    (if target
                      (completion/complete-value
                       (completion/planned-bytecode-input-contract target context :ref) result)
                      result)))
        bits (fn [value]
               (cond (instance? Float value) [:float (Float/floatToRawIntBits value)]
                     (instance? Double value) [:double (Double/doubleToRawLongBits value)]))]
    (doseq [payload [Float/TYPE Double/TYPE]
            target [nil Float/TYPE Double/TYPE]
            :let [expression (snapshot target)
                  region (bytecode/compile-planned-result-return-region
                          emission expression result payload {:unchecked-math false} (fn []))
                  box-method ^java.lang.reflect.Method (get-in region [:box-region :method])
                  return-method ^java.lang.reflect.Method (get-in region [:return-region :method])]]
      (is (= [payload] (:parameter-carriers region)))
      (is (= [payload] (vec (.getParameterTypes box-method))))
      (is (identical? Object (.getReturnType box-method)))
      (is (= [Object] (vec (.getParameterTypes return-method))))
      (is (identical? Object (.getReturnType return-method)))
      (is (identical? expression (:snapshot-expression region)))
      (is (not= result (first (:parameters region))) "the new primitive input never claims the old result occurrence")
      (doseq [value (if (identical? Float/TYPE payload)
                     [(float 0.0) (float -0.0) (float 1.25) (Float/intBitsToFloat 0x7fc01234)
                      Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY Float/MAX_VALUE]
                     [0.0 -0.0 1.25 (Double/longBitsToDouble 0x7ff8000000001234)
                      Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/MAX_VALUE])]
        (let [expected (cond
                         (identical? Float/TYPE target) (.floatValue ^Number value)
                         (identical? Double/TYPE target) (.doubleValue ^Number value)
                         :else value)
              actual ((:invoke region) value)]
          (is (= (bits expected) (bits actual))))))))

(deftest planned-result-return-refuses-unowned-grammar-and-stale-construction
  (let [{:keys [emission]} (fixture 'Object :float false false 'value)
        context (completion/bytecode-operand-context {})
        result 'result
        expression (completion/complete-value completion/object-input-contract result)
        stale? (atom false)
        failure (Exception. "retired result recipe")
        region (bytecode/compile-planned-result-return-region
                emission expression result Double/TYPE {:unchecked-math false}
                #(when @stale? (throw failure)))]
    (doseq [[snapshot symbol carrier]
            [[expression result Long/TYPE]
             [expression 'foreign Double/TYPE]
             [(completion/complete-value completion/object-input-contract '(do result result)) result Double/TYPE]
             [(completion/complete-value completion/object-input-contract
                                         (completion/complete-value
                                          (completion/planned-bytecode-input-contract Double/TYPE context :double) result))
              result Double/TYPE]
             [(completion/complete-value completion/object-input-contract
                                         (completion/complete-value
                                          (completion/source-floating-cast-contract :float {:unchecked-math false}) result))
              result Double/TYPE]
             ['(clojure.core/float result) result Double/TYPE]]]
      (is (= :planned-result-return-form
             (try (bytecode/compile-planned-result-return-region
                   emission snapshot symbol carrier {:unchecked-math false} (fn [])) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (reset! stale? true)
    (doseq [invoke [(:invoke region) (get-in region [:box-region :invoke]) (get-in region [:return-region :invoke])]]
      (is (identical? failure (try (invoke 1.0) nil (catch Exception error error)))))))

(deftest result-snapshots-share-only-the-captured-root-not-adapted-boxes
  (let [{:keys [emission]} (fixture 'Object :float false false 'value)
        context (completion/bytecode-operand-context {})
        result 'result
        direct-expression (completion/complete-value completion/object-input-contract result)
        return-expression (completion/complete-value
                           completion/object-input-contract
                           (completion/complete-value
                            (completion/planned-bytecode-input-contract Float/TYPE context :ref) result))]
    (doseq [payload [Float/TYPE Double/TYPE]
            :let [direct (bytecode/compile-planned-result-return-region
                          emission direct-expression result payload {:unchecked-math false} (fn []))
                  returned (bytecode/compile-planned-result-return-region
                            emission return-expression result payload {:unchecked-math false} (fn []))
                  value (if (identical? payload Float/TYPE)
                          (Float/intBitsToFloat 0x7fc01234) (Double/longBitsToDouble 0x7ff8000000001234))
                  root ((get-in direct [:box-region :invoke]) value)
                  first-direct ((get-in direct [:return-region :invoke]) root)
                  second-direct ((get-in direct [:return-region :invoke]) root)
                  first-return ((get-in returned [:return-region :invoke]) root)
                  second-return ((get-in returned [:return-region :invoke]) root)]]
      (is (identical? root first-direct))
      (is (identical? root second-direct))
      (is (identical? (if (identical? payload Float/TYPE) Float Double) (class root)))
      (is (instance? Float first-return))
      (is (= (Float/floatToRawIntBits (.floatValue ^Number root))
             (Float/floatToRawIntBits first-return) (Float/floatToRawIntBits second-return)))
      (is (not (identical? root first-return)) "a primitive-return recipe boxes separately from the captured root")
      (is (not (identical? first-return second-return)) "adapted snapshots are never cached by this adapter")
      (is (identical? root ((get-in direct [:return-region :invoke]) root))))))

(deftest source-float-completion-preserves-overloads-bits-failures-and-evaluation
  (doseq [operation [:float :unchecked-float]
          frozen [false true :warn-on-boxed]
          [tag values] [['long [0 1 -1 Long/MAX_VALUE]]
                        ['double [0.0 -0.0 0.25 1.0e40 Double/MAX_VALUE
                                  Double/MIN_VALUE Double/NaN
                                  Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
                        ['Object [nil true \a (long 3) (double -0.0)
                                  (double 1.0e40) (float 0.25) Double/NaN]]]]
    (let [{:keys [original projected implementation emission node input-body]}
          (fixture tag operation frozen (not frozen))
          observed (bytecode/retained-input-completion-context emission input-body node)
          region (bytecode/compile-source-floating-completion-region emission input-body node)
          outlined (fn [value effects]
                     (apply (:invoke region)
                            (map {'value value 'effects effects} (:parameters region))))]
      (is (= :float (:completion-carrier observed)))
      (is (= :ref (:returned-carrier observed)) "Object parent keeps the original boxing context")
      (is (= {:unchecked-math (not frozen)} (:operand-source-policy observed)))
      (is (identical? Float/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
      (doseq [value values]
        (let [make-value (constantly value)
              expected (outcome original make-value)]
          (is (= expected (outcome projected make-value) (outcome implementation make-value)
                          (outcome outlined make-value))
              (str "source cast " operation " input=" tag " value=" value)))))))

(deftest object-input-keeps-custom-number-conversion-and-its-effects
  (doseq [operation [:float :unchecked-float]]
    (let [{:keys [original projected implementation] :as compiled}
          (fixture 'Object operation false true)
          outlined (outlined-invocation compiled)
          make-number (fn [^longs effects]
                        (proxy [Number] []
                          (intValue [] (throw (AssertionError. "not an int conversion")))
                          (longValue [] (throw (AssertionError. "not a long conversion")))
                          (doubleValue [] (aset effects 1 (inc (aget effects 1))) (double 1.0e40))
                          (floatValue [] (aset effects 1 (inc (aget effects 1))) (float 3.5))))
          expected (outcome original make-number)]
      (is (= 1 (get-in expected [1 0])) "the operand runs exactly once before conversion")
      (is (pos? (get-in expected [1 1])) "the source Number conversion remains observable")
      (is (= expected (outcome projected make-number) (outcome implementation make-number)
                      (outcome outlined make-number))))))

(deftest literal-primitive-void-and-diverging-operands-retain-their-source-context
  (doseq [operation [:float :unchecked-float]
          [operand input-type value]
          [['(clojure.core/int value) :int 7.0]
           ['(clojure.core/unchecked-float value) :float -0.0]
           [true :bool 0.0]
           ;; A source void-returning method is already completed to a nil reference
           ;; by the invocation emitter before the cast consumes its result.
           ['(java.util.Arrays/fill effects (long 7)) :ref 0.0]
           ['(do (clojure.core/aset effects 0 (long 1))
                 (throw (ex-info "operand failure" {}))) :diverge 0.0]]]
    (let [{:keys [original projected implementation emission input-body node]}
          (fixture 'double operation false true operand)
          observed (bytecode/retained-input-completion-context emission input-body node)
          expected (outcome original (constantly value))]
      (is (= input-type (:input-carrier observed)))
      (is (= expected (outcome projected (constantly value))
                      (outcome implementation (constantly value)))))))

(deftest ordinary-core-casts-share-the-frozen-policy-consumer
  (doseq [operation [:float :unchecked-float]
          policy [false true :warn-on-boxed]
          tag ['double 'Object]]
    (let [expression (list (symbol "clojure.core" (name operation)) 'value)
          interface (types/ensure-fn-interface! [tag])
          implementation
          (:implementation
           (binding [*unchecked-math* policy]
             (core/emit-owned-bytecode!
              bytecode/compile-typed-impl! "raster.compiled.OrdinaryFloatingCast"
              '[value] [tag] 'Object [expression] *ns* (:iface-name interface)
              'ordinary-source-floating-cast)))
          original (binding [*unchecked-math* policy]
                     (eval (list 'fn [(with-meta 'value {:tag tag})] expression)))]
      (doseq [value [0.0 -0.0 1.0e40 Double/NaN Double/POSITIVE_INFINITY]]
        (is (= (outcome (fn [value _] (original value)) (constantly value))
               (outcome (fn [value _] (implementation value)) (constantly value))))))))

(deftest throwing-number-is-not-replaced-by-a-guessed-primitive-input
  (let [{:keys [original projected implementation] :as compiled}
        (fixture 'Object :float false true)
        outlined (outlined-invocation compiled)
        make-number (fn [^longs effects]
                      (proxy [Number] []
                        (intValue [] (throw (AssertionError. "wrong conversion")))
                        (longValue [] (throw (AssertionError. "wrong conversion")))
                        (floatValue [] (aset effects 1 (inc (aget effects 1)))
                          (throw (ArithmeticException. "source Number failed")))
                        (doubleValue [] (aset effects 1 (inc (aget effects 1)))
                          (throw (ArithmeticException. "source Number failed")))))
        expected (outcome original make-number)]
    (is (= [[:error ArithmeticException] [1 1]] expected))
    (is (= expected (outcome projected make-number) (outcome implementation make-number)
                    (outcome outlined make-number)))))

(deftest floating-outlining-refuses-copied-or-mismatched-source-owners
  (let [{:keys [emission input-body node]} (fixture 'double :float false true)
        copied-node (with-meta (apply list node) (meta node))
        copied-body (vec (map identity input-body))
        reason (fn [emission input-body node]
                 (try (bytecode/compile-source-floating-completion-region emission input-body node) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (is (= :unowned-source-floating-completion-region (reason emission input-body copied-node)))
    (is (= :unowned-source-floating-completion-region (reason emission copied-body node)))
    (is (= :unowned-source-floating-completion-region
           (reason (dissoc emission :input-origins) input-body node)))
    (is (= :unowned-source-floating-completion-region
           (reason (dissoc emission :completion-contexts) input-body node)))
    (is (= :unowned-source-integral-completion-region
           (try (bytecode/compile-source-integral-completion-region emission input-body node) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

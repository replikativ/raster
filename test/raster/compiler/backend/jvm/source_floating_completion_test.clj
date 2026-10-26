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

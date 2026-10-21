(ns raster.compiler.backend.jvm.invocation-context-owner-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.set :as set]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]))

(deftm scalar-helper [x :- Double] :- Double (n/* x x))
(deftm duplicate-calls [x :- Double] :- Double
  (n/+ (scalar-helper x) (scalar-helper x)))
(deftm shadowed-calls [x :- Double] :- Double
  (let [left (scalar-helper x)
        x (n/+ x 1.0)
        right (scalar-helper x)]
    (n/+ left right)))
(deftm float-helper [x :- Float] :- Float x)
(deftm narrowing-call [x :- Double] :- Double (float-helper (float x)))
(deftm array-input [xs :- (Array double) index :- Long] :- Double
  (scalar-helper (arrays/aget xs index)))
(deftm let-array-input [xs :- (Array double) index :- Long] :- Double
  (let [copy xs] (scalar-helper (arrays/aget copy index))))
(deftm unused-input-helper [x :- Double xs :- (Array double)] :- Long 7)
(deftm unused-input-call [x :- Double xs :- (Array double)] :- Long
  (unused-input-helper x xs))
(deftm nested-input-call [x :- Double] :- Double
  (scalar-helper (scalar-helper x)))
(deftm long-input-helper [value :- Long] :- Long value)
(deftm int-input-helper [value :- Integer] :- Integer value)
(deftm long-input-template [value :- Long] :- Long (long-input-helper value))
(deftm int-input-template [value :- Integer] :- Integer (int-input-helper value))

(use-fixtures :once
  (fn [run]
    ;; Materialize shared numerical helper dependencies before capturing caller
    ;; emissions. A later unrelated lazy JIT is a real historical invalidation,
    ;; not permission to refresh an already captured owner's certificate.
    (scalar-helper 1.0)
    (n/+ 1.0 1.0)
    (run)))

(defn- receiver [v]
  (let [source (core/resolve-deftm-var v)]
    (ns-resolve (.ns ^clojure.lang.Var source)
                (symbol (str (:name (meta source)) "-impl")))))

(defn- emission [v]
  (let [r (receiver v)]
    (core/ensure-bytecode-implementation! r)
    (:raster.core/bytecode-emission (meta r))))

(defn- helper-occurrences [emitted]
  (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/scalar-helper
              (:raster.op/original (meta %)))
          (vals (:invocation-catalog emitted))))

(defn- integral-input-owner [variant]
  (let [[template tags operand]
        (case variant
          :long-int [#'int-input-template '[long] 'value]
          :checked-int [#'int-input-template '[long] '(clojure.core/int value)]
          :object-long [#'long-input-template '[Object] 'value]
          :ordered-object-long [#'long-input-template '[Object Object]
                                '(do (swap! events conj :operand) value)]
          :reference [#'unused-input-call '[double Object] 'value]
          :ordered-reference [#'unused-input-call '[double Object Object]
                              '(do (swap! events conj :operand) value)])
        reference? (contains? #{:reference :ordered-reference} variant)
        params (if reference?
                 (if (= 3 (count tags)) '[scalar value events] '[scalar value])
                 (if (= 2 (count tags)) '[value events] '[value]))
        source (core/resolve-deftm-var template)
        walked (core/ensure-walked-body! source)
        call (first (filter #(and (seq? %) (= '.invk (first %))) (tree-seq coll? seq walked)))
        _ (assert call "fixture needs a walker-selected typed helper call")
        body (with-meta (list* (first call) (second call)
                              (if reference? ['scalar operand] [operand])) (meta call))
        interface (types/ensure-fn-interface! tags)
        emitted (:emission
                 (#'core/emit-owned-bytecode!
                  bytecode/compile-typed-impl! "raster.compiled.IntegralInputOriginal"
                  params tags 'Object [body]
                  (the-ns 'raster.compiler.backend.jvm.invocation-context-owner-test)
                  (:iface-name interface) 'integral-input))
        calls (filter #(bytecode/retained-invocation-context emitted %)
                      (vals (:invocation-catalog emitted)))]
    (is (= 1 (count calls)))
    {:emission emitted :call (first calls)}))

(defn- input-outcome [f & arguments]
  (try [:value (apply f arguments)] (catch Throwable error [:error (class error)])))

(deftest reference-outlining-retains-checkcast-class-null-and-identity
  (let [{:keys [emission call]} (integral-input-owner :reference)
        region (bytecode/compile-invocation-input-region emission call 1)
        ^java.lang.reflect.Method original-method
        (:method (bytecode/retained-invocation-context emission call))
        expected-class (aget (.getParameterTypes original-method) 1)
        original (:implementation emission)]
    (is (identical? expected-class (.getReturnType ^java.lang.reflect.Method (:method region))))
    (is (= :ref (get-in region [:input-completion :emitted-type])))
    (is (= :checkcast (get-in region [:input-completion :adaptation])))
    (is (nil? (:input-contract region)) "reference checking grants no floating AD contract")
    (doseq [value [nil (double-array [1.0 2.0]) (double-array 0) (float-array 1) "wrong" (Object.)]]
      (let [expected (input-outcome original 3.0 value)
            actual (input-outcome (:invoke region) value)]
        (if (= :error (first expected))
          (is (= expected actual))
          (do (is (= [:value 7] expected) "the unused formal is still checked")
              (is (identical? value (second actual))))))))
  (let [{:keys [emission call]} (integral-input-owner :ordered-reference)
        region (bytecode/compile-invocation-input-region emission call 1)]
    (doseq [value [nil (double-array 0) "wrong"]]
      (let [a (atom []) b (atom [])
            expected (input-outcome (:implementation emission) 3.0 value a)
            actual (input-outcome (:invoke region) value b)]
        (is (= [:operand] @a @b))
        (if (= :error (first expected))
          (is (= expected actual))
          (is (identical? value (second actual))))))))

(deftest integral-outlining-keeps-raw-narrowing-distinct-from-checked-source-casts
  (doseq [[source adaptation] [[:long-int :coerce] [:checked-int :identity]]]
    (let [{:keys [emission call]} (integral-input-owner source)
          region (bytecode/compile-invocation-input-region emission call 0)
          original (:implementation emission)]
      (is (= adaptation (get-in region [:input-completion :adaptation])))
      (is (= Integer/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
      (is (nil? (:input-contract region)) "integral outlining grants no floating AD contract")
      (doseq [value [0 1 -1 Integer/MIN_VALUE Integer/MAX_VALUE 2147483648 Long/MIN_VALUE Long/MAX_VALUE]]
        (is (= (input-outcome original (long value))
               (input-outcome (:invoke region) (long value)))))
      (is (= (if (= source :long-int) [:value Integer/MIN_VALUE] [:error ArithmeticException])
             (input-outcome original 2147483648))))))

(deftest object-long-outlining-preserves-number-unboxing-and-operand-order
  (let [{:keys [emission call]} (integral-input-owner :object-long)
        region (bytecode/compile-invocation-input-region emission call 0)
        original (:implementation emission)]
    (is (= :ref (get-in region [:input-completion :emitted-type])))
    (is (= :coerce (get-in region [:input-completion :adaptation])))
    (is (= Long/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (doseq [value [nil "not a number" 7 3.75 3.75M 9223372036854775808N
                   Double/NaN Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
      (is (= (input-outcome original value) (input-outcome (:invoke region) value))))
    (doseq [run [original (:invoke region)]]
      (let [events (atom [])
            value (proxy [Number] []
                    (intValue [] 0) (floatValue [] (float 0.0)) (doubleValue [] 0.0)
                    (longValue [] (swap! events conj :unbox) (long 23)))]
        (is (= 23 (run value)))
        (is (= [:unbox] @events)))))
  (let [{:keys [emission call]} (integral-input-owner :ordered-object-long)
        region (bytecode/compile-invocation-input-region emission call 0)]
    (doseq [run [(:implementation emission) (:invoke region)] value [7 nil]]
      (let [events (atom [])]
        (is (= (if value [:value 7] [:error NullPointerException])
               (input-outcome run value events)))
        (is (= [:operand] @events))))))

(deftest integral-outlining-requires-complete-owned-positional-evidence
  (let [{:keys [emission call]} (integral-input-owner :object-long)
        token (::bytecode/call-occurrence (meta call))
        path [:invocation-contexts token :input-completions]
        region (bytecode/compile-invocation-input-region emission call 0)]
    (doseq [changed [(update-in emission [:invocation-contexts token] dissoc :input-completions)
                     (assoc-in emission path [])
                     (assoc-in emission (conj path 0 :operand) 'foreign)
                     (assoc-in emission (conj path 0 :parameter-carrier) Integer/TYPE)
                     (assoc-in emission (conj path 0 :adaptation) :identity)
                     (assoc-in emission [:invocation-contexts token :inputs 0 :context] nil)
                     (assoc-in emission [:invocation-contexts token :inputs 0 :context :object-result?]
                               :not-a-boolean)]]
      (is (= :unclosed-bytecode-input-completion
             (try (bytecode/compile-invocation-input-region changed call 0) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= :unowned-bytecode-input-region
           (try (bytecode/compile-invocation-input-region
                 emission (with-meta (apply list call) (meta call)) 0) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :unowned-bytecode-input-region
           (try (bytecode/compile-invocation-input-region
                 (assoc-in emission [:invocation-contexts token :status] :unsupported) call 0) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :bytecode-input-result-carrier
           (try (bytecode/compile-invocation-input-region
                 (-> emission (assoc-in (conj path 0 :emitted-type) :long)
                     (assoc-in (conj path 0 :adaptation) :identity)) call 0) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (let [dependency (:var (first (:dependency-snapshots emission)))]
      (is (var? dependency))
      (with-redefs-fn {dependency (fn [& _] :stale)}
        #(is (thrown? clojure.lang.ExceptionInfo ((:invoke region) 1)))))))

(deftest actual-emission-distinguishes-equal-call-occurrences
  (let [emitted (emission #'duplicate-calls)
        calls (vec (helper-occurrences emitted))
        records (mapv #(bytecode/retained-invocation-context emitted %) calls)]
    (is (= 18.0 (duplicate-calls 3.0)))
    (is (= 2 (count calls)))
    (is (= (first calls) (second calls)) "equal syntax is not occurrence identity")
    (is (not (identical? (first calls) (second calls))))
    (is (every? some? records))
    (doseq [record records]
      (is (= :global-typed-indy (:invocation-mode record)))
      (is (= Double/TYPE (:return-carrier record)))
      (is (= [Double/TYPE] (vec (.getParameterTypes ^java.lang.reflect.Method (:method record)))))
      (is (= [:double] (mapv :type (vals (get-in record [:inputs 0 :ports]))))))
    (is (nil? (bytecode/retained-invocation-context
               emitted (with-meta (apply list (first calls)) (meta (first calls))))))
    (is (nil? (bytecode/retained-invocation-context (emission #'shadowed-calls) (first calls))))))

(deftest actual-physical-ports-survive-shadowing-as-distinct-slots
  (let [emitted (emission #'shadowed-calls)
        records (mapv #(bytecode/retained-invocation-context emitted %) (helper-occurrences emitted))
        ports (mapv #(first (vals (get-in % [:inputs 0 :ports]))) records)]
    (is (= 25.0 (shadowed-calls 3.0)))
    (is (= 2 (count ports)))
    ;; Logical Double is not a physical ABI: the shadowing arithmetic result
    ;; is Object-carried, whereas the original parameter is primitive.
    (is (= {:double 1 :ref 1} (frequencies (map :type ports))))
    (is (= 2 (count (set (map :slot ports)))))))

(deftest actual-input-completions-do-not-confuse-ports-with-emitted-carriers
  (let [emitted (emission #'shadowed-calls)
        records (mapv #(bytecode/retained-invocation-context emitted %) (helper-occurrences emitted))]
    (is (= #{:identity :coerce}
           (set (map #(get-in % [:input-completions 0 :adaptation]) records))))
    (doseq [record records]
      (let [input (first (:inputs record)) completion (first (:input-completions record))]
        (is (identical? (:operand input) (:operand completion)))
        (is (= Double/TYPE (:parameter-carrier completion)))
        (is (= (:type (first (vals (:ports input)))) (:emitted-type completion))))))
  (let [emitted (emission #'narrowing-call)
        call (first (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/float-helper
                                (:raster.op/original (meta %)))
                            (vals (:invocation-catalog emitted))))
        completion (first (:input-completions (bytecode/retained-invocation-context emitted call)))]
    (is (= :float (:emitted-type completion)))
    (is (= Float/TYPE (:parameter-carrier completion)))
    (is (= :identity (:adaptation completion))
        "identity AFTER emitting the explicit checked cast does not erase that cast")))

(deftest actual-input-completions-include-unused-reference-and-primitive-formals
  (let [emitted (emission #'unused-input-call)
        call (first (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/unused-input-helper
                                (:raster.op/original (meta %)))
                            (vals (:invocation-catalog emitted))))
        record (bytecode/retained-invocation-context emitted call)
        completions (:input-completions record)]
    (is (= 7 (unused-input-call 3.0 (double-array 0))))
    (is (= 7 (unused-input-call 3.0 nil)) "reference checks preserve null")
    (is (= 2 (count completions)))
    (is (= [:double :ref] (mapv :emitted-type completions)))
    (is (= [Double/TYPE (class (double-array 0))] (mapv :parameter-carrier completions)))
    (is (= [:identity :checkcast] (mapv :adaptation completions)))
    (is (every? true? (map #(identical? (:operand %1) (:operand %2))
                          (:inputs record) completions)))
    (is (= :unclosed-bytecode-input-completion
           (try (bytecode/compile-invocation-input-region
                 (update-in emitted [:invocation-contexts (::bytecode/call-occurrence (meta call))
                                     :input-completions] pop) call 0) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "a valid requested input cannot hide a missing unused formal")
    (is (= :unclosed-bytecode-input-completion
           (try (bytecode/compile-invocation-input-region
                 (assoc-in emitted [:invocation-contexts (::bytecode/call-occurrence (meta call))
                                    :input-completions 1 :parameter-carrier] Object) call 0) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
        "outlining the first formal still validates the second formal's carrier")))

(deftest nested-input-completions-belong-to-their-own-invocations
  (let [emitted (emission #'nested-input-call)
        calls (vec (helper-occurrences emitted))
        records (mapv #(bytecode/retained-invocation-context emitted %) calls)]
    (is (= 81.0 (nested-input-call 3.0)))
    (is (= 2 (count records)))
    (doseq [record records]
      (is (some? record))
      (is (= 1 (count (:input-completions record))))
      (is (identical? (get-in record [:inputs 0 :operand])
                      (get-in record [:input-completions 0 :operand])))
      (is (= Double/TYPE (get-in record [:input-completions 0 :parameter-carrier]))))
    (is (not (identical? (get-in records [0 :input-completions 0 :operand])
                         (get-in records [1 :input-completions 0 :operand]))))
    (let [outer (first (filter #(seq? (get-in % [:inputs 0 :operand])) records))
          inner (first (remove #(identical? outer %) records))]
      (is (some? outer))
      (is (identical? (get-in outer [:input-completions 0 :operand]) (:expression inner))
          "the actual emitted child, not its pre-expansion catalog form, supplies the operand"))))

(deftest outlined-inputs-preserve-the-original-physical-lexical-abi
  (let [emitted (emission #'shadowed-calls)]
    (doseq [call (helper-occurrences emitted)]
      (let [region (bytecode/compile-invocation-input-region emitted call 0)
            original-port (first (vals (:physical-ports region)))
            parameter-class (first (.getParameterTypes ^java.lang.reflect.Method (:method region)))
            run (:invoke region)]
        (is (= (if (= :double (:type original-port)) Double/TYPE Object) parameter-class))
        (is (= (:physical-ports region) (get-in region [:spec ::bytecode/input-port-layout])))
        (doseq [value [1.5 -0.0 Double/POSITIVE_INFINITY Double/NaN]]
          (is (= (Double/doubleToRawLongBits value)
                 (Double/doubleToRawLongBits (run value)))))
        (when (= :ref (:type original-port))
          (is (thrown? NullPointerException (run nil)))
          (is (thrown? ClassCastException (run "not a number"))))))))

(deftest outlined-float-input-keeps-narrowing-and-original-operand
  (let [emitted (emission #'narrowing-call)
        call (first (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/float-helper
                                (:raster.op/original (meta %)))
                            (vals (:invocation-catalog emitted))))
        region (bytecode/compile-invocation-input-region emitted call 0)
        record (bytecode/retained-invocation-context emitted call)]
    (is (identical? (get-in record [:inputs 0 :operand])
                    (get-in region [:input :operand])))
    (is (= Float/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (let [outcome (fn [f value]
                    (try [:value (Double/doubleToRawLongBits (double (f value)))]
                         (catch Exception error [:exception (class error)])))
          oracle (fn [^double value] (double (float value)))]
      (doseq [value [0.1 -0.0 Double/MAX_VALUE Double/POSITIVE_INFINITY
                     Double/NEGATIVE_INFINITY Double/NaN]]
        (is (= (outcome oracle value)
               (outcome narrowing-call value)
               (outcome (:invoke region) value)))))))

(deftest outlining-refuses-copied-occurrences-and-stale-dependencies
  (let [emitted (emission #'duplicate-calls)
        call (first (helper-occurrences emitted))
        region (bytecode/compile-invocation-input-region emitted call 0)]
    (doseq [[node index] [[(with-meta (apply list call) (meta call)) 0]
                         [call -1] [call 1]]]
      (is (= :unowned-bytecode-input-region
             (try (bytecode/compile-invocation-input-region emitted node index)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (with-redefs-fn {(:var (first (:dependency-snapshots emitted))) (fn [& _] :stale)}
      #(is (thrown? clojure.lang.ExceptionInfo ((:invoke region) 2.0))))
    (is (= 2.0 ((:invoke region) 2.0)))))

(deftest outlined-array-input-restores-an-actual-verifier-carrier
  (doseq [v [#'array-input #'let-array-input]]
    (let [emitted (emission v)
          call (first (helper-occurrences emitted))
          region (bytecode/compile-invocation-input-region emitted call 0)
          classes (vec (.getParameterTypes ^java.lang.reflect.Method (:method region)))
          array-port (first (keep (fn [[id port]]
                                    (when (= :ref (:type port)) id)) (:physical-ports region)))
          args (fn [xs i] (mapv #(if (= array-port %) xs i) (:parameters region)))]
      (is (some #{(class (double-array 0))} classes))
      (is (= (class (double-array 0)) (get-in region [:physical-ports array-port :reference-class])))
      (is (= 7.5 (apply (:invoke region) (args (double-array [2.0 7.5]) 1))))
      (is (thrown? NullPointerException (apply (:invoke region) (args nil 0))))
      (is (thrown? ArrayIndexOutOfBoundsException
                   (apply (:invoke region) (args (double-array 0) 0))))
      (let [weakened (update (:spec region) :params
                             #(mapv (fn [p] (if (= p array-port) (vary-meta p assoc :tag 'Object) p)) %))]
        (is (= :bytecode-input-reference-carrier
               (try (bytecode/compile-specialized-class! "raster.compiled.WeakenedInput" [weakened])
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest annotations-are-private-fresh-and-quote-opaque
  (let [quoted '(quote (scalar-helper x))
        qualified-quoted '(clojure.core/quote (scalar-helper x))
        source [(list 'do quoted qualified-quoted '(scalar-helper x) '(scalar-helper x))]
        first-owner (bytecode/retain-invocation-occurrences source)
        second-owner (bytecode/retain-invocation-occurrences (:body first-owner))
        [first-body] (:body first-owner)]
    (is (identical? quoted (second first-body)))
    (is (identical? qualified-quoted (nth first-body 2)))
    (is (= 2 (count (:catalog first-owner))))
    (is (= 2 (count (:catalog second-owner))))
    (is (empty? (set/intersection (set (keys (:catalog first-owner)))
                                        (set (keys (:catalog second-owner))))))
    (is (= source (:body first-owner) (:body second-owner)))))

(deftest runtime-selected-local-inputs-do-not-claim-one-primitive-contract
  (let [call (first (helper-occurrences (emission #'duplicate-calls)))
        local (with-meta 'helper (meta (second call)))
        argument (nth call 2)
        owned (bytecode/retain-invocation-occurrences
               [(with-meta (list '.invk local argument) (meta call))])
        inputs (atom {:static [] :values []})
        compiled
    ;; A typed hint on a parameter does not prove that the receiver implements
    ;; the interface: emission has an IFn fallback with different conversions.
        (binding [entry/*emission-inputs* inputs]
          (bytecode/compile-specialized-class!
           "raster.compiled.GuardedInvocationContext"
           [{:name 'run :params [(with-meta local {:tag 'Object}) argument] :return-tag 'Object
             :walked-body (:body owned)
             :source-ns (the-ns 'raster.compiler.backend.jvm.invocation-context-owner-test)}]))
        method (get-in compiled [:methods 'run])]
    (is (= 4.0 (.invoke ^java.lang.reflect.Method method nil
                        (object-array [@(receiver #'scalar-helper) 2.0]))))
    (is (= 3.0 (.invoke ^java.lang.reflect.Method method nil
                        (object-array [(fn [x] (+ 1.0 x)) 2.0]))))
    (let [record (first (vals (:invocation-contexts @inputs)))]
      (is (= 1 (count (:invocation-contexts @inputs))))
      (is (= :unsupported (:status record)))
      (is (= :runtime-selected-input-context (:reason record)))
      (is (not (contains? record :input-completions))
          "the guarded typed fast path does not describe the IFn fallback")
      (is (= :guarded-local-typed-interface (:invocation-mode record)))
      (is (= :ref (get-in record [:receiver-input :ports local :type])))
      (is (false? (get-in record [:receiver-input :ports local :verified]))))))

(deftest cloned-occurrence-evidence-must-agree-or-refuse
  (let [emitted (emission #'duplicate-calls)
        call (first (helper-occurrences emitted))
        ;; Reuse one occurrence intentionally; real macro expansion can clone
        ;; a source node. Retain its actual receiver hints, not reconstructed ones.
        argument (nth call 2)
        compile! (fn [body]
                   (let [inputs (atom {:static [] :values []})]
                     (binding [entry/*emission-inputs* inputs]
                       (bytecode/compile-specialized-class!
                        "raster.compiled.ClonedInvocationContext"
                        [{:name 'run :params [argument] :return-tag 'Object
                          :walked-body [body]
                          :source-ns (the-ns 'raster.compiler.backend.jvm.invocation-context-owner-test)}]))
                     (vals (:invocation-contexts @inputs))))]
    (let [records (compile! (list 'let* ['a call 'b call] 'b))]
      (is (= 1 (count records)))
      (is (= :ambiguous (:status (first records))))
      (is (not (contains? (first records) :input-completions))))
    (let [records (compile! (list 'let* ['a call] [call]))]
      (is (= 1 (count records)))
      (is (= :ambiguous (:status (first records))))
      (is (= :repeated-emitted-invocation-context (:reason (first records)))))))

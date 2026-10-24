(ns raster.compiler.backend.jvm.object-value-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.tooling.inspect :as inspect]
            [raster.core :refer [deftm]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect InvocationTargetException Proxy InvocationHandler]))

(deftm object-boundary-helper [x :- Double] :- Double x)
(deftm array-boundary-helper [x :- (Array double)] :- (Array double) x)

(defn- compile-value
  ([params expression] (compile-value params expression 'Object))
  ([params expression return-tag]
  (let [{:keys [methods] :as artifact}
        (bytecode/compile-specialized-class!
         "raster.compiled.ObjectValueContext"
         [{:name 'run :params params :return-tag return-tag
           :walked-body [expression] :source-ns *ns*}])
        method (get methods 'run)]
    (assoc artifact :run
           (fn [& arguments]
             (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
                  (catch InvocationTargetException failure (throw (.getCause failure)))))))))

(deftest quoted-cons-bodies-remain-owned-data-not-executable-code
  (let [body (cons 'let* (list ['x 1.0] '(throw (Exception. "quoted only"))))
        expression (list 'quote {:body [body]})
        result ((:run (compile-value [] expression)))]
    (is (= {:body [body]} result))
    (is (identical? body (get-in result [:body 0]))))
  (let [realizations (atom 0)
        tail (lazy-seq (swap! realizations inc) (list :not-source-data))
        body (cons :head tail)
        compiled (:run (compile-value [] (list 'quote body)))]
    (is (identical? body (compiled)))
    (is (zero? @realizations)))
  (let [call '(raster.compiler.backend.jvm.object-value-test/object-boundary-helper_m_double 2.0)
        source (ns-resolve 'raster.compiler.backend.jvm.object-value-test 'object-boundary-helper_m_double)]
    (doseq [quote-head '[quote clojure.core/quote]]
      (is (empty? (#'bytecode/scan-deftm-refs [(list quote-head call)] *ns* []))))
    (is (contains? (#'bytecode/scan-deftm-refs [call] *ns* []) source))))

(deftest wide-typed-apply-uses-the-actual-emitted-owner
  (let [tags (vec (repeat 22 'long))
        params (mapv #(symbol (str "p" %)) (range 22))
        {:keys [iface-name]} (types/ensure-fn-interface! tags)
        loader (clojure.lang.DynamicClassLoader. (clojure.lang.RT/baseLoader))
        compiled (binding [method-entry/*compilation-classloader* loader]
                   (bytecode/compile-typed-impl!
                    (str "raster.compiled.WideObjectOwner" (clojure.lang.RT/nextID))
                    params tags 'long [(last params)] *ns* iface-name))]
    (is (= 21 (apply compiled (mapv long (range 22)))))
    (is (= 22 (apply compiled (mapv (comp long inc) (range 22)))))
    (let [token (Double. 5.125)
          owned (binding [method-entry/*compilation-classloader* loader]
                  (bytecode/compile-typed-impl!
                   (str "raster.compiled.WideCapturedOwner" (clojure.lang.RT/nextID))
                   params tags 'Object [[token (last params)]] *ns* iface-name))]
      (doseq [offset [0 1]]
        (let [[captured last-value] (apply owned (mapv (comp long #(+ offset %)) (range 22)))]
          (is (identical? token captured))
          (is (= (+ 21 offset) last-value)))))))

(deftest destructured-method-parameters-mask-same-named-global-callees
  (let [scope (the-ns 'raster.compiler.backend.jvm.object-value-test)
        local 'object-boundary-helper_m_double
        expression (list local 2.0)
        _ (is (identical? (ns-resolve scope local)
                          #'object-boundary-helper_m_double))
        compiled (binding [*ns* scope]
                   (with-redefs [raster.core/ensure-walked-body!
                              (fn [& _]
                                (throw (AssertionError. "local callee is not a static sibling")))]
                     (compile-value [[local]] expression)))]
    (is (= 7.0 ((:run compiled) [(fn [x] (+ x 5.0))])))))

(deftest bitwise-input-representation-does-not-change-width-or-traps
  (let [outcome (fn [f left right trace]
                  (try (let [value (f left right trace)]
                         {:value value :class (class value)})
                       (catch Throwable failure {:exception (class failure)})))]
    (doseq [[operation arity] [['clojure.core/bit-and 2] ['clojure.core/bit-or 2]
                               ['clojure.core/bit-xor 2] ['clojure.core/bit-not 1]
                               ['clojure.core/bit-shift-left 2] ['clojure.core/bit-shift-right 2]
                               ['clojure.core/unsigned-bit-shift-right 2]]
            cast [nil 'clojure.core/int 'clojure.core/long 'clojure.core/float
                   'clojure.core/double 'clojure.core/boolean]]
      (let [operand (fn [id event]
                      (list 'do (list 'swap! 'trace 'conj event)
                            (if cast (list cast id) id)))
            args (cond-> [(operand 'left :left)]
                   (= arity 2) (conj (operand 'right :right)))
            expression (cons operation args)
            reference (eval (list 'fn '[left right trace] expression))
            execute (:run (compile-value '[left right trace] expression))
            inputs (if cast
                     [[7 40] [-1 33] [123 64] [7 -1] [-1 129] [123 -65]]
                     [[Long/MIN_VALUE 12] [4294967296 40] [-1 33]
                      [(byte 7) (short 3)] [(short -1) (byte 2)]
                      [nil 2] [1.5 2] [(float 1.5) 2] [1N 2] [1/2 2]
                      [\a 2] [true 2] [1 nil] [1 1.5]
                      [-1 -1] [Long/MAX_VALUE 2147483688] [Long/MIN_VALUE Long/MAX_VALUE]])]
        (doseq [[left right] inputs]
          (let [a (atom []) b (atom [])]
            (is (= (outcome reference left right a) (outcome execute left right b))
                (str operation " " cast " " [left right]))
            (is (= @a @b) "input validation must not cross later operand effects")))))))

(deftest bitwise-width-proof-requires-canonical-core-identity
  (doseq [head '[clojure.core/bit-xor bit-xor unrelated.numeric/bit-xor]]
    (is (= (when (= head 'clojure.core/bit-xor) :long)
           (#'bytecode/infer-arg-stack-type-structural (list head 'left 'right) head {}))))
  (is (nil? (#'bytecode/infer-arg-stack-type-structural
             '(clojure.core/bit-xor left right) 'clojure.core/bit-xor
             {'clojure.core/bit-xor {:type :ref}}))))

(deftest variadic-bitwise-calls-retain-all-operands-and-validation-order
  (doseq [operation '[clojure.core/bit-and clojure.core/bit-or clojure.core/bit-xor]]
    (let [expression (list operation
                           '(do (swap! trace conj :left) left)
                           '(do (swap! trace conj :right) right)
                           '(do (swap! trace conj :third) third))
          reference (eval (list 'fn '[left right third trace] expression))
          execute (:run (compile-value '[left right third trace] expression))
          outcome (fn [f x y z trace]
                    (try {:value (f x y z trace)}
                         (catch Throwable failure {:exception (class failure)})))]
      (doseq [[x y z] [[7 3 1] [4294967296 -1 4294967296] [nil 2 3] [1 2 1.5]]]
        (let [a (atom []) b (atom [])]
          (is (= (outcome reference x y z a) (outcome execute x y z b)))
          (is (= @a @b)))))))

(deftest product-fields-and-identity-observers-share-original-literal-references
  (doseq [value [(Long/valueOf 1000) (Double/valueOf 5.125) (float 1.25) \a]]
    (let [expression (list 'let* ['pair [value '(do nil)]]
                           (list 'clojure.core/identical? '(clojure.core/nth pair 0) value))
          source ((eval (list 'fn* [] expression)))
          execute (:run (compile-value [] expression))]
      (is (true? source))
      (is (= source (execute))))))

(deftest literal-pooling-uses-identity-not-numeric-equality
  (let [left (Double. 5.125) right (Double. 5.125)
        execute (:run (compile-value [] [left right left]))
        result (execute)]
    (is (not (identical? left right)))
    (is (identical? left (nth result 0)))
    (is (identical? right (nth result 1)))
    (is (identical? (nth result 0) (nth result 2)))
    (is (not (identical? (nth result 0) (nth result 1))))))

(deftest java-object-inputs-preserve-the-original-reference
  (let [token (Double. 5.125)
        accept (:run (compile-value [] (list 'java.util.Objects/requireNonNull token)))
        construct (:run (compile-value [] (list 'new 'java.util.concurrent.atomic.AtomicReference token)))
        receiver (with-meta 'receiver {:tag 'java.util.concurrent.atomic.AtomicReference})
        store (:run (compile-value [receiver] (list 'do (list '. receiver 'set token) nil)))
        cell (java.util.concurrent.atomic.AtomicReference.)]
    (is (identical? token (accept)))
    (is (identical? token (.get ^java.util.concurrent.atomic.AtomicReference (construct))))
    (store cell)
    (is (identical? token (.get cell)))))

(deftest source-deftm-object-results-use-the-original-callable
  (let [argument (list 'do '(swap! trace conj :argument) 2.0)
        expression (list 'raster.compiler.backend.jvm.object-value-test/object-boundary-helper argument)
        execute (:run (compile-value '[trace] expression))
        token (Double. 5.125)]
    ;; The real deftm stub remains callable before/after specialization.
    (is (= 2.0 (execute (atom []))))
    (is (= 2.0 (execute (atom []))))
    (with-redefs [object-boundary-helper (fn [x] (is (= 2.0 x)) token)]
      (let [trace (atom [])]
        (is (identical? token (execute trace)))
        (is (= [:argument] @trace))))
    (let [local (symbol "raster.compiler.backend.jvm.object-value-test/object-boundary-helper")
          shadow (:run (compile-value [local] (list local 2.0)))]
      (is (identical? token (shadow (fn [_] token)))))))

(deftest object-storage-consumes-original-references
  (let [token (Double. 5.125)
        construct (:run (compile-value [] (list 'clojure.core/object-array [token])))
        array (with-meta 'array {:tag 'objects})
        write-array (:run (compile-value [array] (list 'clojure.core/aset array 0 token)))
        box (with-meta 'box {:tag 'clojure.lang.Box})
        write-field (:run (compile-value [box] (list 'do (list 'set! (list '. box 'val) token) nil)))
        storage (object-array 1) cell (clojure.lang.Box. nil)]
    (is (identical? token (aget ^objects (construct) 0)))
    (is (identical? token (write-array storage)))
    (is (identical? token (aget storage 0)))
    (write-field cell)
    (is (identical? token (.val cell)))))

(deftest cached-array-source-roots-do-not-authorize-old-static-bodies
  (let [source (ns-resolve 'raster.compiler.backend.jvm.object-value-test 'array-boundary-helper_m_doubles)
        head (symbol (str (:ns (meta source))) (str (:name (meta source))))
        expression (list head 'input)
        execute (:run (compile-value '[input] expression))
        input (double-array [1.0]) cached (double-array [9.0])]
    (is (identical? input (execute input)))
    (with-redefs-fn {source (fn [_] cached)}
      (fn []
        (is (identical? cached (execute input)))
        (let [inputs (atom {:static [] :values []})
              artifact (binding [method-entry/*emission-inputs* inputs]
                         (compile-value '[input] expression))]
          (is (identical? cached ((:run artifact) input)))
          (is (empty? (:static @inputs)) "a replacement root is not its retained body"))))
    (is (identical? input (execute input)))))

(deftest owned-source-calls-use-the-real-object-abi-not-declared-scalar-tags
  (let [source (ns-resolve 'raster.compiler.backend.jvm.object-value-test 'object-boundary-helper_m_double)
        head (symbol (str (:ns (meta source))) (str (:name (meta source))))
        token (Double. 5.125) calls (atom 0)
        replacement (fn [_] (swap! calls inc) token)]
    (with-redefs-fn
      {source replacement}
      (fn []
        ;; Old scalar metadata and registration do not authorize interpreting
        ;; the replacement body. Only its actual erased invocation is used.
        (let [inputs (atom {:static [] :values []})
                artifact (binding [method-entry/*emission-inputs* inputs]
                           (compile-value '[input] (list head 'input)))]
            (binding [method-entry/*emission-inputs* inputs]
              (is (some? (#'bytecode/owned-source-call head ['input] {} {:source-ns *ns*}))))
            (is (zero? @calls) "neither planning nor compilation invokes the source")
            (is (identical? token ((:run artifact) (Object.)))
                "the actual source ABI accepts an Object, irrespective of the logical Double declaration")
            (is (= 1 @calls))
            (is (empty? (:static @inputs)))
            (is (empty? (:definition-proofs @inputs))))))))

(deftest guarded-array-source-calls-use-live-roots
  (let [source (ns-resolve 'raster.compiler.backend.jvm.object-value-test 'array-boundary-helper_m_doubles)
        head (symbol (str (:ns (meta source))) (str (:name (meta source))))
        expression (list head 'input) input (double-array [1.0])
        compile-owned (fn []
                        (let [inputs (atom {:static [] :values []})
                              artifact (binding [method-entry/*emission-inputs* inputs]
                                         (compile-value '[input] expression))]
                          {:artifact artifact :inputs inputs}))]
    (doseq [_ (range 2)]
      (let [{:keys [artifact inputs]} (compile-owned)]
        (is (identical? input ((:run artifact) input)))
        (is (empty? (:static @inputs)) "this route calls the source, not recompiled retained code")
        (is (empty? (:definition-proofs @inputs)) "the runtime class guard needs no dispatch/body proof")
        (let [replacement-result (double-array [9.0])]
          (with-redefs-fn {source (fn [_] replacement-result)}
            (fn []
              ;; The actual invocation, not a manual dependency assertion,
              ;; establishes that a different source Fn class is accepted.
              (is (identical? replacement-result ((:run artifact) input))))))))
    (let [metadata (meta source)
          invalid-cache {:not-an-executable :must-not-be-used}]
      (try
        (alter-meta! source assoc :raster.core/deftm-bytecode invalid-cache)
        (let [{:keys [artifact inputs]} (compile-owned)]
          (is (identical? invalid-cache (:raster.core/deftm-bytecode (meta source))))
          (is (empty? (:static @inputs)))
          (is (identical? input ((:run artifact) input))))
        (finally
          (alter-meta! source #(if (contains? metadata :raster.core/deftm-bytecode)
                                (assoc % :raster.core/deftm-bytecode (:raster.core/deftm-bytecode metadata))
                                (dissoc % :raster.core/deftm-bytecode))))))))

(deftest guarded-source-receivers-retain-selection-and-trap-order
  (let [source (ns-resolve 'raster.compiler.backend.jvm.object-value-test 'array-boundary-helper_m_doubles)
        original @source
        head (symbol (str (:ns (meta source))) (str (:name (meta source))))
        expression (list head '(argument))
        source-fn (eval (list 'fn* '[argument] expression))
        artifact (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
                   (compile-value '[argument] expression))
        input (double-array [1.0]) cached (double-array [9.0])
        replacement (fn [_] cached)
        run-case (fn [execute root change-root? throw-argument?]
                   (let [trace (atom [])
                         result (with-redefs-fn {source root}
                                  #(try
                                     [:value (execute (fn []
                                                        (swap! trace conj :argument)
                                                        (when change-root? (alter-var-root source (constantly replacement)))
                                                        (when throw-argument? (throw (IllegalArgumentException. "operand")))
                                                        input))]
                                     (catch Throwable failure [:throw (class failure)])))]
                     {:result result :trace @trace}))]
    (doseq [root [original replacement (Object.) nil]
            change-root? [false true]
            throw-argument? [false true]]
      (let [expected (run-case source-fn root change-root? throw-argument?)
            actual (run-case (:run artifact) root change-root? throw-argument?)]
        (is (= expected actual))
        (when (= :value (first (:result expected)))
          (is (identical? (second (:result expected)) (second (:result actual)))))))
    (let [code (with-out-str (inspect/disassemble (:bytes artifact) {:method "run"}))
          helper (first (filter #(= "__raster_source_callable" (str (.methodName %)))
                                (.methods (.parse (java.lang.classfile.ClassFile/of) ^bytes (:bytes artifact)))))
          instructions (filter #(instance? java.lang.classfile.Instruction %)
                               (seq (.get (.code helper))))]
      (is (re-find #"(?s)INSTANCEOF.*?IFNE.*?INVOKESTATIC.*?__raster_source_callable" code)
          "the receiver validator is behind the failed concrete-class guard")
      (is (= ["ALOAD_0" "CHECKCAST" "ARETURN"]
             (mapv #(str (.opcode ^java.lang.classfile.Instruction %)) instructions))
          "the cold helper only validates and returns the captured receiver"))))

(deftest object-return-literals-have-owned-reference-lowering
  (doseq [value [(int 1000) (short 3) (byte 2) 1/3
                 (bigint "1208925819614629174706176") (bigdec "1.25")]]
    (let [execute (:run (compile-value [] value))]
      (is (identical? value (execute)))
      (is (= (class value) (class (execute)))))))

(deftest quoted-data-and-exception-results-retain-owned-references
  (let [token (Double. 5.125)
        data [token {:value token}]
        quoted (:run (compile-value [] (list 'quote data)))
        selected (:run (compile-value '[fail]
                                      (list 'try
                                            (list 'if 'fail '(throw (new IllegalStateException)) token)
                                            (list 'catch 'IllegalStateException 'failure token))))]
    (is (identical? data (quoted)))
    (is (identical? token (selected false)))
    (is (identical? token (selected true)))))

(deftest preparation-treats-quoted-programs-as-data
  (doseq [head '[quote clojure.core/quote]]
    (let [data '(clojure.core/dotimes [i 3] (raster.par/pmap i 3 i))
          expression (list head data)
          materialize (requiring-resolve 'raster.compiler.passes.parallel.materialize/materialize-pass)]
      (doseq [prepare [bytecode/desugar-invk
                       (ns-resolve 'raster.compiler.backend.jvm.bytecode 'pre-expand-par-forms)
                       (requiring-resolve 'raster.compiler.core.macroexpand/macroexpand-all-preserving)
                       #(:form (materialize % nil))]]
        (is (identical? expression (prepare expression))))
      (is (= 0 (get-in (materialize (list 'let* [] expression) nil) [:stats :materialized]))))))

(deftest literal-owner-method-name-is-not-a-public-function-slot
  (doseq [method-name '[__raster_literal __raster_source_callable]]
    (is (= {:reason :reserved-jvm-method-name :method (str method-name)}
           (try
             (bytecode/compile-specialized-class!
              "raster.compiled.ReservedObjectOwner"
              [{:name method-name :params [] :return-tag 'Object
                :walked-body [nil] :source-ns *ns*}])
             nil
             (catch clojure.lang.ExceptionInfo failure (ex-data failure)))))))

(deftest object-context-stays-in-selected-result-regions
  (let [value (Double. 5.125)
        field (list 'let* ['ignored '(swap! trace conj :enter)]
                    (list 'if 'flag
                          (list 'do '(swap! trace conj :then) value)
                          (list 'loop* ['i '(int 0)]
                                (list 'if '(< i 2) '(recur (unchecked-inc-int i)) value))))
        execute (:run (compile-value '[flag trace] [field value '(do false) '(do nil)]))]
    (doseq [flag [true false]]
      (let [trace (atom []) result (execute flag trace)]
        (is (= [:enter] (take 1 @trace)))
        (is (= (if flag [:enter :then] [:enter]) @trace))
        (is (identical? value (nth result 0)))
        (is (identical? (nth result 0) (nth result 1)))
        (is (= [false nil] (subvec result 2)))))))

(deftest children-and-separate-owners-cannot-substitute-another-pool
  (let [value (Double. 5.125)
        expression [(list 'fn* [] [value value]) value]
        a (compile-value [] expression)
        other (Double. 5.125)
        b (compile-value [] [(list 'fn* [] [other other]) other])
        [child token] ((:run a)) [other-child other-token] ((:run b))]
    (is (not= (:class a) (:class b)))
    (is (not= (.getName ^Class (:class a)) (.getName ^Class (:class b))))
    (is (identical? (.getClassLoader ^Class (:class a)) (.getClassLoader (class child))))
    (is (identical? token (first (child))))
    (is (identical? token (second (child))))
    (is (identical? other-token (first (other-child))))
    (is (not (identical? token other-token)))))

(deftest explicit-object-completion-snapshots-a-primitive-once
  (let [input (with-meta 'input {:tag 'double})
        execute (:run (compile-value
                       [input]
                       (list 'let* ['saved (completion/complete-value completion/object-input-contract input)]
                             ['saved 'saved])))]
    (doseq [value [5.125 1000.0 -0.0]]
      (let [[a b] (execute value)]
        (is (= value a b))
        (is (identical? a b))))))

(deftest object-result-invk-keeps-a-cached-reference
  (let [{:keys [iface-class iface-name]} (types/ensure-fn-interface! '[double])
        token (Double. 5.125) calls (atom 0)
        implementation (Proxy/newProxyInstance
                        (.getClassLoader ^Class iface-class) (into-array Class [iface-class])
                        (reify InvocationHandler
                          (invoke [_ _ method _]
                            (when (= "invk" (.getName ^java.lang.reflect.Method method))
                              (swap! calls inc) token))))
        target (intern (the-ns 'raster.compiler.backend.jvm.object-value-test)
                       'cached-object-result implementation)
        receiver (with-meta 'raster.compiler.backend.jvm.object-value-test/cached-object-result
                   {:tag iface-name :raster.type/tag iface-name :raster.type/ret-tag 'double})
        input (with-meta 'input {:tag 'double})
        call (list '. receiver 'invk input)
        expressions [[call 'expected-value]
                     (list 'let* ['saved call] ['saved 'saved 'expected-value])
                     (list 'if true (list 'let* ['saved call] ['saved 'saved 'expected-value]) nil)
                     (list 'loop* ['i '(int 0)]
                           (list 'if '(< i 1) '(recur (unchecked-inc-int i))
                                 (list 'let* ['saved call] ['saved 'saved 'expected-value])))]]
    (try
      (doseq [expression expressions]
       (let [execute (:run (compile-value [input 'expected-value] expression))
             source (eval (list 'fn* [input 'expected-value] expression))]
        (doseq [input [1.0 -2.0]]
          (let [expected (source input token) actual (execute input token)]
            (is (= expected actual))
            (is (every? #(identical? token %) expected))
            (is (every? #(identical? token %) actual))))))
      (doseq [expression [(list 'let* ['saved call]
                                     '(if (clojure.core/identical? saved expected-value) 1 0))
                          (list 'loop* ['saved call]
                                '(if (clojure.core/identical? saved expected-value) 1 0))
                          (list 'loop* [(with-meta 'saved {:raster.type/tag 'double}) call]
                                '(if (clojure.core/identical? saved expected-value) 1 0))
                          (list 'loop* ['i '(int 0) 'saved call]
                                '(if (< i 2)
                                   (recur (unchecked-inc-int i) saved)
                                   (if (clojure.core/identical? saved expected-value) 1 0)))]]
       (let [source (eval (list 'fn* [input 'expected-value] expression))
            execute (:run (compile-value [input 'expected-value] expression 'long))]
        (doseq [input [1.0 -2.0]]
          (is (= 1 (source input token)))
          (is (= (source input token) (execute input token))))))
      (let [saved 'saved
            expression (list 'let* [saved (list 'clojure.core/double call)]
                             '(clojure.core/identical? saved saved))
            source (eval (list 'fn* [input] expression))
            execute (:run (compile-value [input] expression 'boolean))]
        (is (false? (source 1.0)))
        (is (= (source 1.0) (execute 1.0))))
      (is (= 43 @calls))
      (finally (ns-unmap (:ns (meta target)) (:name (meta target)))))))

(deftest initialization-failure-never-returns-a-callable-artifact
  (let [seen (atom nil) published (atom nil)
        refusal (with-redefs-fn
                  {#'bytecode/initialize-literals!
                   (fn [owner _]
                     (reset! seen owner)
                     (throw (ex-info "injected owner initialization failure" {:reason :injected-literal-init})))}
                  #(try (reset! published (compile-value [] [(Double. 5.125)])) :incorrectly-published
                        (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure)))))]
    (is (= :injected-literal-init refusal))
    (is (nil? @published))
    (is (class? @seen))
    (let [getter (.getMethod ^Class @seen @#'bytecode/literal-getter (into-array Class [Integer/TYPE]))]
      (let [failure (try (.invoke getter nil (object-array [(int 0)])) nil
                         (catch InvocationTargetException failure (.getCause failure)))]
        (is (instance? IllegalStateException failure))
        (is (= "JVM literal owner is not initialized" (ex-message failure)))))))

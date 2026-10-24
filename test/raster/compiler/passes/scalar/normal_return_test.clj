(ns raster.compiler.passes.scalar.normal-return-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm broadcast]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.scalar.normal-return :as normal]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.pipeline :as pipeline]))

(use-fixtures :each
  (fn [test!]
    ;; These tests consume frozen semantic source, not a pending receiver's raw
    ;; syntax or a warm receiver's optimized body. Preparation is an explicit
    ;; producer action before the read-only query's independent owner begins.
    (doseq [name '[allocated effectful-allocation recursive-array borrowed-array
                  nullable-array borrowed-array-with-count throwing-array
                  discarded-array object-array-result mismatched-array-result broadcast-array
                  effectful-primitive-return boxed-reference-return]]
      (when-let [v (ns-resolve 'raster.compiler.passes.scalar.normal-return-test name)]
        (core/prepare-emission-source-walk! (core/resolve-deftm-var v))))
    (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
      (test!))))

(deftest presence-refinement-must-not-delete-a-trapping-read
  (doseq [size [0 1]]
    (let [body (list 'let* ['before '(clojure.core/aset events 0 7)
                            'a (list 'clojure.core/double-array size)
                            'x '(clojure.core/aget a 0)
                            'after '(clojure.core/aset events 1 11)]
                     '(if (clojure.core/nil? x) 1 2))
          lowered (:form (dce/eliminate-dead-bindings (products/lower-local-products body)))
          outcome (fn [expression]
                    (let [events (int-array [0 0])
                          run (eval (list 'fn '[events] expression))]
                      (assoc (try {:value (run events)}
                                  (catch Throwable error {:error (class error)}))
                             :events (vec events))))]
      (is (= (outcome body) (outcome lowered)))
      (is (= (if (zero? size)
               {:error ArrayIndexOutOfBoundsException :events [7 0]}
               {:value 2 :events [7 11]})
             (outcome lowered))))))

(deftest nested-producer-presence-must-not-delete-reference-conversion-checks
  (doseq [operator '[clojure.core/double clojure.core/float]]
    (let [nullable (with-meta 'nullable {:raster.type/tag 'double})
          body (list 'let* ['before '(clojure.core/aset events 0 7)
                            nullable 'input
                            'x (list 'let* ['temporary (list operator nullable)] 'temporary)
                            'after '(clojure.core/aset events 1 11)]
                     '(if (clojure.core/nil? x) 1 2))
          lowered (:form (dce/eliminate-dead-bindings (products/lower-local-products body)))
          outcome (fn [expression input]
                    (let [events (int-array [0 0])
                          run (eval (list 'fn '[input events] expression))]
                      (assoc (try {:value (run input events)}
                                  (catch Throwable error {:error (class error)}))
                             :events (vec events))))]
      (doseq [input [nil "not a Number" 2.0]]
        (is (= (outcome body input) (outcome lowered input))))
      (is (= {:error NullPointerException :events [7 0]} (outcome lowered nil)))
      (is (= {:value 2 :events [7 11]} (outcome lowered 2.0))))))

(deftest definition-dependent-analysis-needs-an-owning-collector
  (binding [method-entry/*emission-inputs* nil]
    (is (= {:presence :present :array-tag 'floats}
           (normal/result-fact '(clojure.core/float-array 0))))
    (is (= {:presence :unknown}
           (normal/result-fact '(raster.numeric/+ a b)
                               {'a {:presence :present :array-tag 'floats}
                                'b {:presence :present :array-tag 'floats}})))))

(deftm allocated [n :- Long] :- (Array double)
  (let [out (clojure.core/double-array n)] out))

(deftm direct-allocated [ignored :- Object] :- Object
  (clojure.core/double-array 0))

(deftm effectful-allocation [counter :- (Array long) n :- Long] :- (Array double)
  (let [out (clojure.core/double-array n)]
    (aset counter 0 (inc (aget counter 0)))
    out))

(deftm recursive-array [n :- Long] :- (Array double) (recursive-array n))

(deftm borrowed-array [x :- (Array double)] :- (Array double) x)

(deftm nullable-array [flag :- Boolean] :- (Array double)
  (if flag (clojure.core/double-array 0) nil))

(deftm borrowed-array-with-count [n :- Long x :- (Array double)] :- (Array double) x)

(deftm discarded-array [n :- Long] :- Void (clojure.core/double-array n))
(deftm object-array-result [n :- Long] :- Object (clojure.core/double-array n))
(deftm mismatched-array-result [n :- Long] :- (Array float) (clojure.core/double-array n))
(deftm broadcast-array [input :- (Array double) n :- Long] :- (Array double)
  (broadcast [input] input))

(deftm effectful-primitive-return [events :- (Array long) value :- Object] :- Double
  (aset events 0 (inc (aget events 0)))
  value)

(deftm boxed-reference-return [n :- Long] :- java.lang.Double nil)

(defn- selected-call [v args]
  (let [m (meta v) operation (symbol (str (:ns m)) (str (:name m)))
        entry (first (get (core/methods-of v) (count args)))
        name (types/mangle (:name m) (:tags entry))]
    (with-meta (list* '.invk (symbol (str (:mangled-ns entry)) (str name "-impl")) args)
      {:raster.op/original operation})))

(deftest semantic-presence-retains-the-original-reference-return-contract
  (is (= {:presence :present :array-tag 'doubles}
         (normal/result-fact (selected-call #'object-array-result '[n]))))
  (is (= {:presence :present :array-tag 'doubles}
         (normal/result-fact (selected-call #'broadcast-array '[input n])))
      "owned structural pmap uses the existing materializer's allocation contract")
  (is (= {:presence :unknown}
         (normal/result-fact (selected-call #'discarded-array '[n]))))
  (is (= {:presence :unknown}
         (normal/result-fact (selected-call #'mismatched-array-result '[n]))))
  (let [call (selected-call #'discarded-array '[0])]
    (core/ensure-bytecode-implementation! (find-var (second call)))
    (is (nil? (discarded-array 0)) "void completion discards the present body value")))

(deftest semantic-presence-survives-only-legitimate-jit-and-intrinsic-history
  (let [v (binding [*ns* (the-ns 'raster.compiler.passes.scalar.normal-return-test)]
            (eval '(core/deftm fresh-semantic-array [n :- Long] :- Object
                     (clojure.core/double-array n))))
        call (selected-call v '[n])
        receiver (find-var (second call))
        cold-root (var-get receiver)
        before @method-entry/*emission-inputs*
        _ (is (= {:presence :unknown} (normal/result-fact call))
              "a read-only query does not prepare a missing semantic frontend")
        _ (is (identical? before @method-entry/*emission-inputs*))
        _ (is (identical? cold-root (var-get receiver)))
        _ (core/prepare-emission-source-walk! (core/resolve-deftm-var v))
        captured (method-entry/call-with-definition-inputs
                  #(let [result (normal/result-fact call)]
                     [result @method-entry/*emission-inputs*]))]
    (is (= {:presence :present :array-tag 'doubles} (first captured)))
    (is (not (contains? (:definition-proofs (second captured)) receiver)))
    (is (some? (:raster.core/jit-spec (meta receiver))))
    (core/ensure-bytecode-implementation! receiver)
    (is (nil? (:raster.core/jit-spec (meta receiver))))
    (is (true? (method-entry/assert-definition-inputs-current! (second captured))))
    (with-redefs [clojure.core/double-array (fn [_] nil)]
      (is (thrown? clojure.lang.ExceptionInfo
                   (method-entry/assert-definition-inputs-current! (second captured))))
      (is (= :intrinsic-source-definition-mismatch
             (try (binding [method-entry/*emission-inputs* nil]
                    (method-entry/call-with-definition-inputs #(normal/result-fact call)))
                  :accepted
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest primitive-return-completion-is-not-totality-or-body-carrier-evidence
  (is (= {:presence :unknown}
         (normal/result-fact (selected-call #'boxed-reference-return '[n]))))
  (is (nil? (boxed-reference-return 0)) "an explicitly boxed reference may return nil")
  (is (= {:presence :unknown}
         (normal/result-fact (with-meta '(unknown-number) {:raster.type/tag 'double}))))
  (let [call (selected-call #'effectful-primitive-return '[events input])
        receiver (find-var (second call))
        raw (list 'let* ['result call] '(if (clojure.core/nil? result) :absent :present))
        captured (method-entry/call-with-definition-inputs
                  #(let [fact (normal/result-fact call)
                         lowered (products/lower-local-products raw)]
                     {:fact fact :lowered lowered :inputs @method-entry/*emission-inputs*}))
        compile! #(eval (list 'fn* '[events input] %))
        original (compile! raw)
        optimized (compile! (:lowered captured))
        observe (fn [function value]
                  (let [events (long-array [0])
                        outcome (try {:value (function events value)}
                                     (catch Throwable error {:error (class error)}))]
                    (assoc outcome :events (vec events))))]
    (is (= {:presence :present :scalar-type :double} (:fact captured)))
    (is (not (contains? (:definition-proofs (:inputs captured)) receiver)))
    (is (= 1 (count (filter #(= call %) (tree-seq coll? seq (:lowered captured))))))
    (doseq [value [2.0 nil (Object.)]]
      (let [expected (observe original value)
            actual (observe optimized value)]
        (is (= expected actual))
        (is (= [1] (:events actual)) "unknown body effects and return conversions run once")
        (if (number? value)
          (is (= :present (:value actual)))
          (is (class? (:error actual)) "presence is conditional on successful return"))))
    (is (true? (method-entry/assert-definition-inputs-current! (:inputs captured)))
        "genuine JIT did not change the semantic return contract")))

(deftest return-domain-does-not-import-hints-or-promote-branch-widths
  (is (= {:presence :present}
         (normal/result-fact 'input {'input {:presence :present :scalar-type :double
                                            :producer-scalar-type :float}})))
  (let [operation 'raster.numeric//
        signature (inference/registered-call-signature operation '[float float])
        implementation (symbol (str (:mangled-sym signature) "-impl"))
        _ (core/prepare-deftm-frontend! (find-var (:mangled-sym signature)))
        float-call (with-meta (list '.invk implementation 'left 'right)
                     {:raster.op/original operation})
        double-call (selected-call #'effectful-primitive-return '[events input])
        fact #(method-entry/call-with-definition-inputs (fn [] (normal/result-fact %)))]
    (is (= {:presence :present :scalar-type :float} (fact float-call)))
    (is (= {:presence :present} (fact (list 'if 'condition float-call double-call))))
    (is (= {:presence :present :scalar-type :float}
           (fact (list 'if 'condition float-call float-call))))))

(deftest presence-never-uses-source-captured-before-its-observation
  (doseq [selection [:typed :direct] phase [:before-observation :after-observation]]
    (let [v (if (= selection :typed) #'allocated #'direct-allocated)
          operation (symbol (str (:ns (meta v))) (str (:name (meta v))))
          entry (first (get (core/methods-of v) 1))
          source-name (symbol (str (:mangled-ns entry))
                              (str (types/mangle (:name (meta v)) (:tags entry))))
          source (find-var source-name)
          saved @source
          implementation (if (= selection :typed) (symbol (str source-name "-impl")) source-name)
          call (if (= selection :typed) (selected-call v [0]) (list implementation 0))
          selector-var (if (= selection :typed) #'method-entry/selected-definition
                           #'method-entry/direct-selected-definition)
          original @selector-var
          observe! method-entry/observe-definition-input!
          changed? (atom false)
          inputs (atom {:static [] :values []})]
      (is (some? (original operation implementation 1))
          "both fixtures must reach a real admitted definition")
      (try
        (binding [method-entry/*emission-inputs* inputs]
          (with-redefs-fn
            {selector-var (fn [& arguments]
                            (let [selected (apply original arguments)]
                              (when (and selected (= :before-observation phase)
                                         (compare-and-set! changed? false true))
                                (alter-var-root source (constantly (fn [_] nil))))
                              selected))
             #'method-entry/observe-definition-input!
             (fn [v & arguments]
               (let [result (apply observe! v arguments)]
                 (when (and (identical? v source) (= :after-observation phase)
                            (compare-and-set! changed? false true))
                   (alter-var-root source (constantly (fn [_] nil))))
                 result))}
            #(if (= :before-observation phase)
               (do (is (= {:presence :unknown} (normal/result-fact call)))
                   (is (true? (method-entry/assert-definition-inputs-current! @inputs))))
               (is (= :stale-definition-proof-input
                      (try (normal/result-fact call) nil
                           (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
                   (pr-str [selection phase @changed?])))))
        (finally (alter-var-root source (constantly saved)))))))

(deftest selected-scalar-presence-is-owned-and-does-not-bypass-definition-checks
  (let [operation 'raster.numeric/sqrt
        signature (inference/registered-call-signature operation '[double])
        implementation (symbol (str (:mangled-sym signature) "-impl"))
        call (with-meta (list '.invk implementation 'x) {:raster.op/original operation})
        definition (method-entry/selected-definition operation implementation 1)
        source (:source-var definition) saved @source
        inputs (atom {:static [] :values []})]
    (is (some? definition))
    (is (nil? (scalar-body/checked-primitive-result call {'x :double}))
        "known operands must not bypass the selected producer's definition")
    (binding [method-entry/*emission-inputs* nil]
      (is (nil? (scalar-body/checked-selected-primitive-result call))))
    (binding [method-entry/*emission-inputs* inputs]
      (is (= {:type :double} (scalar-body/checked-selected-primitive-result call))))
    (is (contains? (:definition-proofs @inputs) source))
    (try
      (alter-var-root source (constantly (fn [_] nil)))
      (binding [method-entry/*emission-inputs* (atom {:static [] :values []})]
        (is (nil? (scalar-body/checked-selected-primitive-result call))))
      (is (= :stale-definition-proof-input
             (try (method-entry/assert-definition-inputs-current! @inputs) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (finally (alter-var-root source (constantly saved))))))

(deftest lexical-producer-facts-cross-selected-formals-without-type-assumptions
  (let [call (selected-call #'borrowed-array-with-count '[0 input])
        producer (normal/result-fact '(clojure.core/double-array 0))]
    (is (= {:presence :present :array-tag 'doubles} producer))
    (is (= producer (normal/result-fact call {'input producer})))
    (is (= {:presence :absent} (normal/result-fact call {'input {:presence :absent}})))
    (is (= {:presence :unknown} (normal/result-fact call)))
    (is (= {:presence :present}
           (normal/result-fact call {'input {:presence :present :array-tag 'floats}}))
        "an argument's different array carrier cannot certify the formal return")
    (is (= {:presence :unknown}
           (normal/result-fact (with-meta call {:raster.op/lexical-callee true})
                               {'input producer})))))

(deftest nil-tests-refine-only-their-real-lexical-argument
  (is (= {:presence :present}
         (normal/result-fact '(if (clojure.core/nil? input) [] input))))
  (is (= {:presence :absent}
         (normal/result-fact '(if (clojure.core/nil? input) input nil))))
  (doseq [test ['(foreign/nil? input)
                (with-meta '(clojure.core/nil? input) {:raster.op/lexical-callee true})]]
    (is (= {:presence :unknown}
           (normal/result-fact (list 'if test [] 'input)))))
  (is (= {:presence :unknown}
         (normal/result-fact '(if (clojure.core/nil? input) [] input)
                             {'clojure.core/nil? {:presence :present}}))))

(deftest aliases-transport-physical-array-producer-facts-not-result-hints
  (is (= {:presence :present :array-tag 'floats}
         (normal/result-fact '(let* [a (clojure.core/float-array 0) b a] b))))
  (is (= {:presence :unknown}
         (normal/result-fact (with-meta '(unknown-array) {:raster.type/tag 'floats}))))
  (is (= {:presence :present :array-tag 'floats}
         (normal/result-fact '(raster.par/pmap i 0 float i)))))

(deftest branch-and-formal-scopes-do-not-borrow-unrelated-array-facts
  (let [producer {:presence :present :array-tag 'doubles}
        call (selected-call #'borrowed-array-with-count '[0 unknown])]
    (is (= {:presence :unknown} (normal/result-fact call {'x producer})))
    (is (= {:presence :absent}
           (normal/result-fact '(if (clojure.core/nil? input) input nil)
                               {'input {:array-tag 'doubles}})))
    (is (= {:presence :present}
           (normal/result-fact '(if predicate (clojure.core/float-array 0)
                                   (clojure.core/double-array 0)))))
    (let [definition method-entry/selected-definition]
      (with-redefs [method-entry/selected-definition
                    (fn [& args] (some-> (apply definition args)
                                          (assoc :walked-body nil
                                                 :source-body '[(par/scan output a 0 i n float a)])))]
        (is (= {:presence :unknown}
               (normal/result-fact (selected-call #'borrowed-array-with-count '[0 input])
                                   {'input producer 'output producer})))))))

(deftm throwing-array [n :- Long] :- (Array double)
  (if (< n 0)
    (throw (ex-info "negative test input" {:n n}))
    (clojure.core/double-array n)))

(deftm throwing-array-guard [n :- Long] :- Long
  (let [out (throwing-array n)] (if (clojure.core/nil? out) 1 2)))

(deftest normal-return-lattice-and-lexical-roots
  (doseq [expression [false 0 [] '(clojure.core/float-array 0)
                      '(let* [out (clojure.core/double-array 2) alias out] alias)
                      '(let* [out (clojure.core/double-array 2) alias out out nil] alias)
                      '(do (opaque-effect) (clojure.core/float-array 0))
                      '(if condition (clojure.core/float-array 0) (clojure.core/double-array 1))
                      '(if condition (throw exception) (clojure.core/float-array 0))]]
    (is (= :present (normal/presence expression)) (pr-str expression)))
  (is (= :absent (normal/presence '(let* [out (clojure.core/double-array 2) out nil] out))))
  (is (= :unknown (normal/presence '(if condition (clojure.core/float-array 0) nil)))))

(deftest allocations-must-reach-the-actual-return
  (doseq [expression ['(let* [out (clojure.core/float-array 0)] (unknown out))
                      '(let* [out (clojure.core/float-array 0)] borrowed)
                      '(fake/float-array 0) '(float-array 0) '(fake/zeros-like input 0)
                      '((fn* [x] x) input)
                      (with-meta '(clojure.core/doubles nil) {:raster.type/tag 'doubles})
                      (with-meta '(clojure.core/float-array 0) {:raster.op/lexical-callee true})
                      (with-meta '(.invk fake/receiver 0) {:raster.op/original 'clojure.core/float-array})]]
    (is (= :unknown (normal/presence expression)) (pr-str expression))))

(deftest array-return-tags-do-not-establish-presence
  (is (= :unknown (normal/presence (selected-call #'borrowed-array '[x]))))
  (is (= :unknown (normal/presence (selected-call #'nullable-array '[flag])))))

(deftest selected-definition-correspondence-is-shared-not-a-result-assumption
  (let [call (selected-call #'allocated '[n])
        definition (method-entry/selected-definition
                    (:raster.op/original (meta call)) (second call) 1)]
    (is (var? (:source-var definition)))
    (is (= :present (normal/presence call)))
    (with-redefs-fn {(:source-var definition) (fn [& _] nil)}
      #(is (= :unknown (normal/presence call))))
    (with-redefs-fn {(:receiver-var definition) (fn [& _] nil)}
      #(is (= :unknown (normal/presence call))))
    (is (= :unknown (normal/presence '(allocated n))))))

(deftest analysis-does-not-run-effects-and-does-not-grant-ownership
  (let [counter (long-array [0]) call (selected-call #'effectful-allocation '[counter n])]
    ;; Realize the executable before the first proof observation. Calling it
    ;; below must not replace a lazy receiver in the middle of this owner's life.
    (core/ensure-bytecode-implementation! (ns-resolve *ns* (second call)))
    (is (= :present (normal/presence call)))
    (is (= 0 (aget counter 0)))
    (is (= 0 (alength (effectful-allocation counter 0))))
    (is (= 1 (aget counter 0)))
    (is (= :present (normal/presence call)))))

(deftest recursive-selected-definitions-decline-without-running
  (let [call (selected-call #'recursive-array '[n])
        definition (method-entry/selected-definition
                    (:raster.op/original (meta call)) (second call) 1)]
    (core/ensure-walked-body! (:source-var definition))
    (is (= :unknown (normal/presence call)))))

(deftest pure-maps-use-the-existing-materialized-return-contract
  (is (= :present (normal/presence '(raster.par/pmap i 0 float (unknown i)))))
  (is (= :unknown (normal/presence
                   (with-meta '(raster.par/pmap i 0 float (unknown i))
                     {:raster.op/lexical-callee true}))))
  (doseq [expression ['(raster.par/pmap i 0 float (unknown i) extra)
                      '(raster.par/pmap 0 0 float (unknown i))]]
    (is (= :unknown (normal/presence expression)))))

(deftest optional-materialization-declines-only-its-unsupported-domain
  (let [unsupported '(raster.par/pmap i 0 unknown-cast i)
        valid '(raster.par/pmap i 0 float i)]
    (is (= :unknown (normal/presence unsupported)))
    ;; The actual lowering still rejects this source; only optional proof declines.
    (is (= :unsupported-pure-map-materialization
           (try (materialize/materialize-pass unsupported nil) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (with-redefs [materialize/materialize-pass (fn [form _] {:form form})]
      (is (= :unknown (normal/presence valid))))
    (with-redefs [materialize/materialize-pass
                  (fn [form _] {:form (list 'do form)})]
      (is (= :unknown (normal/presence valid))))
    (with-redefs [materialize/materialize-pass
                  (fn [& _] (throw (ex-info "materializer invariant failed" {:reason :invariant})))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"materializer invariant failed"
                           (normal/presence valid))))))

(deftest selected-throwing-producers-must-retain-the-original-call
  (let [call (selected-call #'throwing-array '[n])
        compiled (pipeline/compile-aot #'throwing-array-guard :dtype :double)]
    (is (= :present (normal/presence call)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"negative test input"
                         (throwing-array-guard -1)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"negative test input" (compiled -1)))
    (is (= 2 (throwing-array-guard 0) (compiled 0)))))

(deftm allocation-guard [n :- Long] :- Long
  (let [out (clojure.core/double-array n)]
    (if (clojure.core/nil? out) 1 2)))

(deftm effectful-allocation-guard [counter :- (Array long) n :- Long] :- Long
  (let [out (effectful-allocation counter n)]
    (if (clojure.core/nil? out) 1 2)))

(deftest native-return-presence-retains-effects-and-changed-inputs
  (let [compiled (pipeline/compile-aot #'effectful-allocation-guard :target :c :dtype :double)
        reference (long-array [0]) actual (long-array [0])]
    (try
      (doseq [n [0 3 1]]
        (is (= 2 (effectful-allocation-guard reference n) (compiled actual n)))
        (is (= (vec reference) (vec actual))))
      (is (= [3] (vec actual)))
      (finally
        (when (instance? java.lang.AutoCloseable compiled)
          (.close ^java.lang.AutoCloseable compiled))))))

(deftest folding-presence-must-preserve-failing-allocation
  (let [source '(let* [out (clojure.core/double-array -1)]
                 (if (clojure.core/nil? out) 1 2))
        lowered (products/lower-local-products source)
        error-class (fn [f] (try (f) nil (catch Throwable e (class e))))]
    (is (= (second source) (second lowered)))
    (is (= NegativeArraySizeException
           (error-class #(eval source)) (error-class #(eval lowered))
           (error-class #(allocation-guard -1))))
    (doseq [target [:jvm :c]]
      (let [compiled (pipeline/compile-aot #'allocation-guard :target target :dtype :double)]
        (try
          (is (= NegativeArraySizeException (error-class #(compiled -1))))
          (is (= 2 (allocation-guard 0) (compiled 0)))
          (finally
            (when (instance? java.lang.AutoCloseable compiled)
              (.close ^java.lang.AutoCloseable compiled))))))))

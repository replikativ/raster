(ns raster.compiler.passes.scalar.normal-return-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.passes.scalar.normal-return :as normal]
            [raster.compiler.passes.scalar.local-products :as products]
            [raster.compiler.passes.scalar.dce :as dce]
            [raster.compiler.passes.parallel.materialize :as materialize]
            [raster.compiler.pipeline :as pipeline]))

(use-fixtures :each
  (fn [test!]
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

(deftm effectful-allocation [counter :- (Array long) n :- Long] :- (Array double)
  (let [out (clojure.core/double-array n)]
    (aset counter 0 (inc (aget counter 0)))
    out))

(deftm recursive-array [n :- Long] :- (Array double) (recursive-array n))

(deftm borrowed-array [x :- (Array double)] :- (Array double) x)

(deftm nullable-array [flag :- Boolean] :- (Array double)
  (if flag (clojure.core/double-array 0) nil))

(deftm borrowed-array-with-count [n :- Long x :- (Array double)] :- (Array double) x)

(defn- selected-call [v args]
  (let [m (meta v) operation (symbol (str (:ns m)) (str (:name m)))
        entry (first (get (core/methods-of v) (count args)))
        name (types/mangle (:name m) (:tags entry))]
    (with-meta (list* '.invk (symbol (str (:mangled-ns entry)) (str name "-impl")) args)
      {:raster.op/original operation})))

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
                     {:raster.op/lexical-callee true})))))

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

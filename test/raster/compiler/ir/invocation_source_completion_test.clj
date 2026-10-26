(ns raster.compiler.ir.invocation-source-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.parallel-program :as parallel-program]))

(defn- scalar
  ([] (scalar :int))
  ([dtype] (av/tensor {:dtype dtype :shape []})))
(defn- opaque [] (av/make {:kind :opaque :memory-space :host :ownership :borrowed}))

(defn- completion-node []
  (completion/complete-value
   (completion/source-integral-cast-contract :int {:unchecked-math false}) 'value))

(defn- fixture
  ([] (fixture :int false))
  ([dtype unchecked?]
  (let [node (fn []
               (if (= :float dtype)
                 (completion/complete-value
                  (completion/source-floating-cast-contract :float {:unchecked-math unchecked?}) 'value)
                 (completion-node)))
        nodes [(node) (node)]
        interface (types/ensure-fn-interface! '[Object])
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.OrderedSourceCompletion"
                 '[value] '[Object] 'Object nodes *ns* (:iface-name interface)
                 'ordered-source-completion)
        outline (if (= :float dtype) bytecode/compile-source-floating-completion-region
                    bytecode/compile-source-integral-completion-region)
        regions (mapv #(outline (:emission emitted) nodes %) nodes)
        bindings (mapv vector '[first-count second-count] nodes)
        plan (invocation/from-prefix
              {:id :ordered-source-counts
               :parameters '[value] :parameter-values {'value (opaque)}
               :bindings bindings
               :binding-values {'first-count (scalar dtype) 'second-count (scalar dtype)}
               :program-values {'first-count (scalar dtype) 'second-count (scalar dtype)}
               :program-inputs '[first-count second-count]
               :program-outputs '[second-count]})
        program (parallel-program/make
                 {:dialect :typed-parallel
                  :source (list 'let* (vec (mapcat identity bindings)) 'second-count)
                  :values {'first-count (scalar dtype) 'second-count (scalar dtype)}
                  :inputs '[first-count second-count] :outputs '[second-count]})
        executors (java.util.IdentityHashMap.)]
    (doseq [[step region] (map vector (:steps plan) regions)]
      (.put executors step region))
    {:plan plan :program program
     :execute {:source-completion
               (fn [step operands]
                 (let [region (.get executors step)]
                   (when-not region
                     (throw (ex-info "step is not owned by this prepared source program"
                                     {:reason :unowned-test-source-step})))
                   {:type dtype
                    :value (apply (:invoke region)
                                  (map #(get-in operands [% :source]) (:parameters region)))}))}})))

(defn- opaque-completion-plan []
  (let [boxed (completion/complete-value completion/object-input-contract 'value)
        checked (completion/complete-value
                 (completion/source-floating-cast-contract :float {:unchecked-math false}) 'alias)]
    (invocation/from-prefix
     {:id :opaque-source-completion
      :parameters '[value] :parameter-values {'value (scalar :float)}
      :bindings [['boxed boxed] ['alias 'boxed] ['rounded checked]]
      :binding-values {'boxed (opaque) 'alias (opaque) 'rounded (scalar :float)}
      :program-values {'rounded (scalar :float)}
      :program-inputs '[rounded] :program-outputs '[rounded]})))

(deftest opaque-source-completions-preserve-raw-reference-and-alias-identity
  (let [plan (opaque-completion-plan)
        [box alias checked] (:steps plan)
        hostile (proxy [Object] []
                  (equals [_] (throw (AssertionError. "opaque equality")))
                  (hashCode [] (throw (AssertionError. "opaque hash")))
                  (toString [] (throw (AssertionError. "opaque print"))))]
    (is (invocation/source-completion? box))
    (is (invocation/value-alias? alias))
    (is (invocation/source-completion? checked))
    (doseq [payload [nil hostile (float -0.0)]]
      (let [calls (atom 0)
            execute {:source-completion (fn [_ _] (swap! calls inc) payload)}
            initial (materialization/begin-materialization plan [(float 0.25)])
            boxed (materialization/materialize-next initial execute)
            aliased (materialization/materialize-next boxed execute)
            wrapper (get-in boxed [:values (:id box)])]
        (is (materialization/materialized-opaque? wrapper))
        (is (= (:id box) (:id wrapper)))
        (is (identical? payload (:source wrapper)))
        (is (identical? wrapper (get-in aliased [:values (:id alias)])))
        (is (identical? boxed (materialization/validate-state! boxed)))
        (is (identical? aliased (materialization/validate-state! aliased)))
        (materialization/prefix-bindings aliased)
        (is (= 1 @calls) "validation, prefix projection and aliases never repeat Object completion")
        (let [again (materialization/materialize-next initial execute)]
          (is (= 2 @calls) "explicitly replaying an earlier state performs a fresh completion")
          (is (not (identical? wrapper (get-in again [:values (:id box)])))))
        (doseq [forged [(assoc-in boxed [:values (:id box) :id] [:foreign-completion])
                        (assoc-in boxed [:values (:id box)] {:type :float :value (float 0.25)})
                        (assoc-in aliased [:values (:id alias)]
                                  (materialization/->MaterializedOpaque (:id wrapper) (:value wrapper) payload))]]
          (is (some? (try (materialization/validate-state! forged) nil
                          (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
        (is (= 2 @calls))))))

(deftest opaque-completion-then-checked-float-has-no-validation-reconversion
  (let [plan (opaque-completion-plan)
        [box alias checked] (:steps plan)
        calls (atom [])
        execute {:source-completion
                 (fn [step operands]
                   (swap! calls conj (:symbol step))
                   (if (identical? step box)
                     (get-in operands ['value :value])
                     {:type :float :value (float (get-in operands ['alias :source]))}))}
        initial (materialization/begin-materialization plan [(float -0.0)])
        states (take 4 (iterate #(materialization/materialize-next % execute) initial))
        final-state (last states)
        finished (materialization/finish-materialization final-state)
        wrapper (get-in finished [:values (:id box)])]
    (is (= '[boxed rounded] @calls))
    (is (= (Float/floatToRawIntBits (float -0.0))
           (Float/floatToRawIntBits (get-in finished [:program-scalars 'rounded :value]))))
    (is (identical? wrapper (get-in finished [:values (:id alias)])))
    (is (identical? finished (materialization/validate! finished)))
    (is (identical? final-state (materialization/validate-state! final-state)))
    (doseq [forged [(assoc-in finished [:values (:id box) :id] [:foreign-completion])
                    (assoc-in finished [:values (:id alias)]
                              (materialization/->MaterializedOpaque
                               (:id wrapper) (:value wrapper) (:source wrapper)))
                    (assoc-in finished [:values (:id checked) :value] (double -0.0))]]
      (is (some? (try (materialization/validate! forged) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (= '[boxed rounded] @calls))))

(deftest floating-completions-use-the-same-ordered-host-materialization-contract
  (doseq [unchecked? [false true]]
    (let [{:keys [plan execute]} (fixture :float unchecked?)
          calls (atom [])
          number (proxy [Number] []
                   (intValue [] (throw (AssertionError. "wrong conversion")))
                   (longValue [] (throw (AssertionError. "wrong conversion")))
                   (floatValue [] (swap! calls conj :float) (float 3.5))
                   (doubleValue [] (swap! calls conj :double) (double 1.0e40)))
          initial (materialization/begin-materialization plan [number])]
      (is (every? invocation/source-completion? (:steps plan)))
      (is (empty? @calls) "preparation does not execute the source Number conversion")
      (if unchecked?
        (let [first-state (materialization/materialize-next initial execute)
              _ (materialization/validate-state! first-state)
              final (materialization/materialize-next first-state execute)
              result (materialization/finish-materialization final)]
          (is (= [:float :float] @calls))
          (is (= {:type :float :value (float 3.5)}
                 (get-in result [:program-scalars 'second-count])))
          (is (instance? Float (get-in result [:program-scalars 'second-count :value]))))
        (do
          (is (thrown? IllegalArgumentException (materialization/materialize-next initial execute)))
          (is (= [:double] @calls))
          (is (zero? (:next-step initial)))
          (materialization/validate-state! initial)
          (is (= [:double] @calls) "validation does not replay the failed conversion"))))))

(deftest source-completion-executor-cannot-relabel-an-unconverted-result
  (doseq [[dtype value] [[:float (double 3.5)] [:int (long 3)]]]
    (let [{:keys [plan]} (fixture dtype false)
          initial (materialization/begin-materialization plan [(double 3.5)])
          execute {:source-completion (fn [_ _] {:type dtype :value value})}]
      (is (= :invocation-materialization-source-carrier
             (try (materialization/materialize-next initial execute) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
      (is (zero? (:next-step initial))))))

(deftest completed-source-carriers-are-revalidated-without-reexecution
  (let [{:keys [plan execute]} (fixture :float false)
        initial (materialization/begin-materialization plan [(double 0.25)])
        first-state (materialization/materialize-next initial execute)
        id (:id (first (:steps plan)))
        forged-state (assoc-in first-state [:values id :value] (double 0.25))
        finished (materialization/materialize plan [(double 0.25)] execute)
        forged-result (assoc-in finished [:values id :value] (double 0.25))
        forged-projection (assoc-in finished [:program-scalars 'second-count :value] (double 0.25))]
    (doseq [[validate value] [[materialization/validate-state! forged-state]
                              [materialization/validate! forged-result]
                              [materialization/validate! forged-projection]]]
      (is (= :invocation-materialization-source-carrier
             (try (validate value) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest source-completion-carrier-follows-explicit-value-aliases
  (let [{:keys [plan execute]} (fixture :float false)
        source (:id (last (:steps plan)))
        id [:source-float-alias]
        alias (invocation/->ValueAlias id 'aliased [{:symbol 'second-count :value source}]
                                       source (scalar :float))
        plan (-> plan
                 (update :steps conj alias)
                 (assoc-in [:values id] (scalar :float))
                 (update :bindings (fn [bindings]
                                     (mapv #(if (= 'second-count (:program-value %))
                                              (assoc % :invocation-value id) %) bindings))))
        initial (materialization/begin-materialization plan [(double 0.25)])
        final-state (nth (iterate #(materialization/materialize-next % execute) initial) 3)
        finished (materialization/finish-materialization final-state)]
    (is (instance? Float (get-in finished [:program-scalars 'second-count :value])))
    (doseq [[validate value]
            [[materialization/validate-state!
              (assoc-in final-state [:values id :value] (double 0.25))]
             [materialization/validate!
              (assoc-in finished [:values id :value] (double 0.25))]
             [materialization/validate!
              (assoc-in finished [:program-scalars 'second-count :value] (double 0.25))]]]
      (is (= :invocation-materialization-source-carrier
             (try (validate value) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest owned-completions-run-at-the-host-cursor-not-during-validation
  (let [{:keys [plan program execute]} (fixture)
        calls (atom 0)
        number (proxy [Number] []
                 (intValue [] (swap! calls inc))
                 (longValue [] (long (swap! calls inc)))
                 (floatValue [] (float (swap! calls inc)))
                 (doubleValue [] (double (swap! calls inc))))
        initial (materialization/begin-materialization plan [number])]
    (is (every? invocation/source-completion? (:steps plan)))
    (let [order (invocation/source-order plan program)
          emitted (assoc program :dialect :opencl-parallel)
          validation (emitted-program/validate-with-physical-results! emitted)]
      (is (= (mapv :id (:steps plan)) (vec (keep :invocation-step order))))
      (is (= :emitted-parallel-program-retained-validation
             (try (invocation-link/validate-ordered-source-actions!
                   {:program {:attributes {} :equations []}
                    :materialization initial :source-order order :source-realizations {}})
                  nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "a forged program cannot reach executor coverage admission")
      (is (= :invocation-ordered-source-action
             (try (invocation-link/validate-ordered-source-actions!
                   {:program emitted :retained-validation validation :projections (:projections validation)
                    :materialization initial :source-order order :source-realizations {}})
                  nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "the existing device executor cannot acquire source-action permission implicitly"))
    (is (not (identical? (get-in plan [:steps 0 :expression])
                         (get-in plan [:steps 1 :expression]))))
    (is (identical? number (get-in initial [:values (:id (first (:parameters plan))) :source])))
    (materialization/validate-state! initial)
    (materialization/prefix-bindings initial)
    (is (zero? @calls))
    (is (= :invocation-materialization-source-executor
           (try (materialization/materialize-next initial (fn [& _] (throw (AssertionError.))))
                nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (zero? @calls))
    (let [first-state (materialization/materialize-next initial execute)]
      (is (= 1 @calls))
      (materialization/validate-state! first-state)
      (is (= 1 @calls) "validating a completed source action never repeats it")
      (let [final-state (materialization/materialize-next first-state execute)
            result (materialization/finish-materialization final-state)]
        (is (= 2 @calls) "equal-looking source conversions remain distinct observations")
        (is (= {:type :int :value 1} (get-in result [:program-scalars 'first-count])))
        (is (= {:type :int :value 2} (get-in result [:program-scalars 'second-count])))
        (materialization/validate! result)
        (is (= 2 @calls))))
    ;; A new invocation replays source actions; successful earlier checks are not cached.
    (materialization/materialize plan [number] execute)
    (is (= 4 @calls))
    (let [step (first (:steps plan))]
      (is (= :unowned-test-source-step
             (try ((:source-completion execute) (assoc step :symbol 'copied) {}) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest conversion-failure-observes-earlier-host-writes-and-suppresses-the-suffix
  (let [{:keys [plan execute]} (fixture)
        memory (long-array 1) calls (atom [])
        number (proxy [Number] []
                 (intValue [] (throw (AssertionError. "unexpected overload")))
                 (longValue [] (throw (AssertionError. "unexpected overload")))
                 (floatValue [] (throw (AssertionError. "unexpected overload")))
                 (doubleValue [] (swap! calls conj (aget memory 0)) 4294967296.0))
        initial (materialization/begin-materialization plan [number])]
    (aset memory 0 17)
    (is (thrown? ArithmeticException (materialization/materialize-next initial execute)))
    (is (= [17] @calls))
    (is (zero? (:next-step initial)))
    (materialization/validate-state! initial)
    (is (= [17] @calls))))

(deftest eager-specialization-refuses-missing-source-execution-before-running-a-prefix
  (let [plan (invocation/from-prefix
              {:id :mixed-host-prefix :parameters '[seed value]
               :parameter-values {'seed (scalar) 'value (opaque)}
               :bindings [['n '(inc seed)] ['count (completion-node)]]
               :binding-values {'n (scalar) 'count (scalar)}
               :program-values {'count (scalar)}
               :program-inputs '[count] :program-outputs '[count]})
        calls (atom 0)
        evaluate (fn [& _] (swap! calls inc) {:type :int :value 2})]
    (is (= :invocation-materialization-source-executor
           (try (materialization/materialize plan [1 nil] evaluate) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (zero? @calls))
    ;; Incremental execution is deliberately local to its next action.
    (materialization/materialize-next
     (materialization/begin-materialization plan [1 nil]) evaluate)
    (is (= 1 @calls))))

(deftest opaque-host-values-need-no-equality-hashing-printing-or-numeric-conversion
  (let [plan (invocation/from-prefix
              {:id :opaque-host-alias :parameters '[value]
               :parameter-values {'value (opaque)}
               :bindings '[[alias value]] :binding-values {'alias (opaque)}
               :program-values {} :program-inputs [] :program-outputs []})
        hostile (proxy [Object] []
                  (equals [_] (throw (AssertionError. "equals")))
                  (hashCode [] (throw (AssertionError. "hashCode")))
                  (toString [] (throw (AssertionError. "toString"))))]
    (doseq [value [nil hostile]]
      (let [initial (materialization/begin-materialization plan [value])
            final-state (materialization/materialize-next initial nil)
            result (materialization/finish-materialization final-state)
            public (get-in result [:values (:id (first (:parameters plan)))])
            alias (get-in result [:values (:id (first (:steps plan)))])]
        (is (identical? value (:source public)))
        (is (identical? public alias))
        (materialization/validate-state! final-state)
        (materialization/validate! result)
        (let [redirected (assoc-in final-state [:values (:id (first (:steps plan)))]
                                   (assoc alias :id :redirected))
              failure (try (materialization/validate-state! redirected) nil
                           (catch clojure.lang.ExceptionInfo error error))]
          (is (= :invocation-materialization-state-producer (:reason (ex-data failure))))
          (is (string? (pr-str (ex-data failure)))
              "opaque failure diagnostics do not print a payload to prove inequality"))))))

(deftest opaque-values-and-undischarged-completions-do-not-acquire-numerical-permission
  (is (= :invocation-opaque-program-value
         (try (invocation/from-prefix
               {:id :opaque-escape :parameters '[value] :parameter-values {'value (opaque)}
                :bindings [] :binding-values {} :program-values {'value (opaque)}
                :program-inputs '[value] :program-outputs []})
              nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
  (is (= :invocation-scalar-operands
         (try (invocation/from-prefix
               {:id :opaque-numeric :parameters '[value] :parameter-values {'value (opaque)}
                :bindings '[[n (inc value)]] :binding-values {'n (scalar)}
                :program-values {'n (scalar)} :program-inputs '[n] :program-outputs []})
              nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

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

(defn- scalar [] (av/tensor {:dtype :int :shape []}))
(defn- opaque [] (av/make {:kind :opaque :memory-space :host :ownership :borrowed}))

(defn- completion-node []
  (completion/complete-value
   (completion/source-integral-cast-contract :int {:unchecked-math false}) 'value))

(defn- fixture []
  (let [nodes [(completion-node) (completion-node)]
        interface (types/ensure-fn-interface! '[Object])
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.OrderedSourceCompletion"
                 '[value] '[Object] 'Object nodes *ns* (:iface-name interface)
                 'ordered-source-completion)
        regions (mapv #(bytecode/compile-source-integral-completion-region
                        (:emission emitted) nodes %) nodes)
        bindings (mapv vector '[first-count second-count] nodes)
        plan (invocation/from-prefix
              {:id :ordered-source-counts
               :parameters '[value] :parameter-values {'value (opaque)}
               :bindings bindings
               :binding-values {'first-count (scalar) 'second-count (scalar)}
               :program-values {'first-count (scalar) 'second-count (scalar)}
               :program-inputs '[first-count second-count]
               :program-outputs '[second-count]})
        program (parallel-program/make
                 {:dialect :typed-parallel
                  :source (list 'let* (vec (mapcat identity bindings)) 'second-count)
                  :values {'first-count (scalar) 'second-count (scalar)}
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
                   {:type :int
                    :value (apply (:invoke region)
                                  (map #(get-in operands [% :source]) (:parameters region)))}))}}))

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

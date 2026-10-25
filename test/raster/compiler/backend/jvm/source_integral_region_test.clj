(ns raster.compiler.backend.jvm.source-integral-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- emit [tags node policy]
  (let [interface (types/ensure-fn-interface! tags)
        body [node]]
    (assoc (binding [*unchecked-math* policy]
             (core/emit-owned-bytecode!
              bytecode/compile-typed-impl! "raster.compiled.SourceIntegralRegion"
              '[value effects] tags 'Object body *ns* (:iface-name interface)
              'source-integral-region-test))
           :input-body body)))

(defn- outcome [invoke value]
  (let [effects (long-array 1)
        result (try [:value (invoke value effects)]
                    (catch Throwable error [:error (class error)]))]
    [result (vec effects)]))

(def ^:dynamic source-count-offset 2)

(deftest source-completion-outlining-preserves-real-ports-and-both-policies
  (doseq [operation [:int :long]
          frozen [false true :warn-on-boxed]
          [tag samples] [['long [0 4294967296 Long/MIN_VALUE Long/MAX_VALUE]]
                         ['double [0.75 Double/NaN Double/POSITIVE_INFINITY]]
                         ['Object [0 4294967296 nil true \a]]]]
    (let [operand-policy (if frozen false true)
          operand '(do (clojure.core/aset effects 0 (clojure.core/inc (clojure.core/aget effects 0))) value)
          contract (completion/source-integral-cast-contract operation {:unchecked-math frozen})
          node (completion/complete-value contract operand)
          {:keys [implementation emission input-body]} (emit [tag 'longs] node operand-policy)
          observed (bytecode/retained-input-completion-context emission input-body node)
          region (binding [*unchecked-math* (not operand-policy)]
                   (bytecode/compile-source-integral-completion-region emission input-body node))
          invoke (fn [value effects]
                   (apply (:invoke region)
                          (map {'value value 'effects effects} (:parameters region))))
          source (binding [*unchecked-math* operand-policy]
                   (eval (list 'fn [(with-meta 'value {:tag tag})
                                    (with-meta 'effects {:tag 'longs})]
                               (completion/project-to-jvm-source node))))]
      (is (= #{'value 'effects} (set (:parameters region))))
      (is (= {:unchecked-math operand-policy} (:operand-source-policy observed)))
      (is (= contract (:completion-contract region)))
      (is (= (case operation :int Integer/TYPE :long Long/TYPE)
             (.getReturnType ^java.lang.reflect.Method (:method region))))
      (doseq [value samples]
        (is (= (outcome source value) (outcome implementation value)
               (outcome invoke value)))))))

(deftest outlined-conversion-does-not-leak-its-policy-into-the-operand
  (doseq [operand-policy [false true :warn-on-boxed]]
    (let [node (completion/complete-value
                (completion/source-integral-cast-contract :long {:unchecked-math (not operand-policy)})
                '(clojure.core/int value))
          {:keys [implementation emission input-body]} (emit '[long longs] node operand-policy)
          region (binding [*unchecked-math* (not operand-policy)]
                   (bytecode/compile-source-integral-completion-region emission input-body node))
          invoke (fn [value _] ((:invoke region) value))]
      (is (= (outcome implementation 4294967296) (outcome invoke 4294967296)))
      (if operand-policy
        (is (= [[:value 0] [0]] (outcome invoke 4294967296)))
        (is (= [[:error ArithmeticException] [0]] (outcome invoke 4294967296)))))))

(deftest object-count-conversion-retains-user-number-effects
  (doseq [operation [:int :long] policy [false true :warn-on-boxed]]
    (let [operand '(do (clojure.core/aset effects 0 (clojure.core/inc (clojure.core/aget effects 0))) value)
          node (completion/complete-value
                (completion/source-integral-cast-contract operation {:unchecked-math policy}) operand)
          {:keys [implementation emission input-body]} (emit '[Object longs] node false)
          region (bytecode/compile-source-integral-completion-region emission input-body node)
          invoke (fn [value effects]
                   (apply (:invoke region)
                          (map {'value value 'effects effects} (:parameters region))))
          source (binding [*unchecked-math* policy]
                   (eval (list 'fn ['value (with-meta 'effects {:tag 'longs})]
                               (list (symbol "clojure.core" (name operation)) operand))))
          observe (fn [invoke]
                    (let [events (atom []) effects (long-array 1)
                          number (proxy [Number] []
                                   (intValue [] (swap! events conj :int) 7)
                                   (longValue [] (swap! events conj :long) 7)
                                   (floatValue [] (swap! events conj :float) (float 7))
                                   (doubleValue [] (swap! events conj :double) 7.0))]
                      {:result (invoke number effects) :effects (vec effects)
                       :number-methods @events}))
          expected (observe source)]
      (is (= [1] (:effects expected)))
      (is (seq (:number-methods expected)))
      (is (= expected (observe implementation) (observe invoke)))
      (is (not= :pure (effects/analyze-effect node))
          "an unqualified Object conversion is not an externally pure numeric operation"))))

(deftest nested-completions-cannot-substitute-for-the-audited-root
  (let [inner (completion/complete-value
               (completion/source-integral-cast-contract :int {:unchecked-math false}) 'value)
        node (completion/complete-value
              (completion/source-integral-cast-contract :long {:unchecked-math true}) inner)
        {:keys [implementation emission input-body]} (emit '[long longs] node false)
        region (bytecode/compile-source-integral-completion-region emission input-body node)]
    (is (= ['value] (:parameters region)))
    (is (= 3 ((:invoke region) 3)))
    (is (thrown? ArithmeticException ((:invoke region) 4294967296)))
    (is (thrown? ArithmeticException (implementation 4294967296 (long-array 1))))))

(deftest child-history-is-checked-independently-of-the-original-occurrence-query
  (let [node (completion/complete-value
              (completion/source-integral-cast-contract :int {:unchecked-math false})
              '(clojure.core/+ value raster.compiler.backend.jvm.source-integral-region-test/source-count-offset))
        {:keys [emission input-body]} (emit '[long longs] node false)
        observed (bytecode/retained-input-completion-context emission input-body node)
        region (bytecode/compile-source-integral-completion-region emission input-body node)
        offset source-count-offset]
    (is (= 3 ((:invoke region) 1)))
    (try
      (alter-var-root #'source-count-offset (constantly (inc offset)))
      ;; Isolate the child's own history check: the normal original query also
      ;; refuses this stale dependency, but cannot substitute for the child guard.
      (with-redefs [bytecode/retained-input-completion-context (fn [& _] observed)]
        (is (contains? #{:stale-definition-proof-input :stale-emission-inputs}
                       (try ((:invoke region) 1) nil
                            (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (finally (alter-var-root #'source-count-offset (constantly offset))))))

(deftest completion-outliner-refuses-copies-missing-history-and-changed-raw-carriers
  (let [node (completion/complete-value
              (completion/source-integral-cast-contract :int {:unchecked-math false}) 'value)
        {:keys [emission input-body]} (emit '[long longs] node false)
        observed (bytecode/retained-input-completion-context emission input-body node)
        token (:raster.compiler.backend.jvm.bytecode/completion-occurrence (meta (:source observed)))
        reason (fn [emission input-body node]
                 (try (bytecode/compile-source-integral-completion-region emission input-body node)
                      nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
    (is (= :unowned-source-integral-completion-region
           (reason emission input-body (with-meta (apply list node) (meta node)))))
    (is (= :unowned-source-integral-completion-region
           (reason emission (vec (seq input-body)) node)))
    (doseq [changed [(update emission :completion-contexts dissoc token)
                     (assoc-in emission [:completion-contexts token :status] :ambiguous)
                     (update-in emission [:completion-contexts token] dissoc :operand-input)]]
      (is (= :unowned-source-integral-completion-region (reason changed input-body node))))
    (is (= :source-integral-completion-replay-mismatch
           (reason (assoc-in emission [:completion-contexts token :input-carrier] :double)
                   input-body node)))
    (let [region (bytecode/compile-source-integral-completion-region emission input-body node)]
      (with-redefs [bytecode/retained-input-completion-context (fn [& _] nil)]
        (is (= :unowned-source-integral-completion-region
               (try ((:invoke region) 1) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

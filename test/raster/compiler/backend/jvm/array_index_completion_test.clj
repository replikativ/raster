(ns raster.compiler.backend.jvm.array-index-completion-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.value-completion :as completion])
  (:import [java.lang.reflect InvocationTargetException]))

(defn- invoke-method [method arguments]
  (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
       (catch InvocationTargetException error (throw (.getCause error)))))

(defn- observe [invoke index]
  (let [array (long-array [3]) effects (long-array 1)
        outcome (try [:value (invoke array index effects)]
                     (catch Throwable error [:throw (class error)]))]
    {:outcome outcome :array (vec array) :effects (vec effects)}))

(deftest frozen-source-integral-completions-preserve-overloads-policy-and-effect-order
  (doseq [mode [false true :warn-on-boxed]
          operation [:int :long]
          [carrier tag raw inputs]
          [[:long 'long 'value [0 4294967296 Long/MIN_VALUE Long/MAX_VALUE]]
           [:double 'double 'value [0.75 4294967296.0 Double/NaN
                                   Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
           [:int 'Object '(clojure.core/int value) [0 Integer/MIN_VALUE Integer/MAX_VALUE]]
           [:float 'Object '(clojure.core/unchecked-float value)
            [(float 0.75) Float/NaN Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY]]
           [:ref 'Object 'value [0 4294967296 0.75 \u0000 nil true (Object.)]]
           [:bool 'Object true [nil]]
           [:void 'Object '(java.util.Arrays/fill effects (clojure.core/long 4)) [nil]]]]
    (let [consumer-mode (if mode false true)
          policy {:unchecked-math mode}
          params [(with-meta 'value {:tag tag}) (with-meta 'effects {:tag 'longs})]
          operand (list 'do '(clojure.core/aset effects 0 1) raw)
          original (list (symbol "clojure.core" (name operation)) operand)
          retained (completion/complete-value
                    (completion/source-integral-cast-contract operation policy) operand)
          body (fn [conversion]
                 (list 'let* ['converted conversion]
                       '(clojure.core/aset effects 1 2) 'converted))
          source (binding [*unchecked-math* mode]
                   (eval (list 'fn params (body original))))
          projected (binding [*unchecked-math* consumer-mode]
                      (eval (list 'fn params
                                  (completion/project-to-jvm-source (body retained)))))
          artifact (binding [*unchecked-math* consumer-mode]
                     (bytecode/compile-specialized-class!
                      "raster.compiled.FrozenSourceIntegralCompletion"
                      [{:name 'run :params params :return-tag 'Object
                        :walked-body [(body retained)] :source-ns *ns*
                        :source-policy {:unchecked-math consumer-mode}}]))
          method (get-in artifact [:methods 'run])
          compiled #(invoke-method method [%1 %2])
          outcome (fn [run value]
                    (let [effects (long-array 2)]
                      {:outcome (try (let [result (run value effects)]
                                       [:value (class result) result])
                                     (catch Throwable error [:throw (class error)]))
                       :effects (vec effects)}))]
      (doseq [input inputs]
        (testing (str operation " frozen=" mode " consumer=" consumer-mode
                      " carrier=" carrier " input=" (pr-str input))
          (is (= (outcome source input) (outcome projected input)))
          (is (= (outcome source input) (outcome compiled input))))))))

(defn- store-body [loop?]
  `(clojure.core/aset ~'array ~'index
     (do (clojure.core/aset ~'effects 0 9)
         ~(if loop?
            '(loop* [iteration (clojure.core/int 0)]
               (if (clojure.core/< iteration 2)
                 (recur (clojure.core/unchecked-inc-int iteration)) 7))
            7))))

(deftest discarded-integral-conversion-still-demands-the-operands-value
  (doseq [retained? [false true]]
    (let [operand '(clojure.core/aset values 0 7)
          conversion (if retained?
                       (completion/complete-value
                        (completion/source-integral-cast-contract :int {:unchecked-math false})
                        operand)
                       (list 'clojure.core/int operand))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.DiscardedIntegralCompletion"
                    [{:name 'run :params [(with-meta 'values {:tag 'longs})]
                      :return-tag 'Object :walked-body [(list 'do conversion 99)]
                      :source-ns *ns* :source-policy {:unchecked-math false}}])
          values (long-array 1)]
      (is (= 99 (invoke-method (get-in artifact [:methods 'run]) [values])))
      (is (= [7] (vec values))))))

(deftest frozen-conversion-policy-does-not-rebind-its-operands-policy
  (doseq [[consumer frozen] [[true false] [false true]]]
    (let [operand '(clojure.core/int value)
          term (completion/complete-value
                (completion/source-integral-cast-contract :int {:unchecked-math frozen}) operand)
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.IntegralOperandPolicy"
                    [{:name 'run :params [(with-meta 'value {:tag 'long})]
                      :return-tag 'Object :walked-body [term] :source-ns *ns*
                      :source-policy {:unchecked-math consumer}}])
          run #(invoke-method (get-in artifact [:methods 'run]) [4294967296])]
      (if consumer
        (is (= 0 (run)) "inner unchecked conversion completes before the frozen checked cast")
        (is (thrown? ArithmeticException (run))
            "inner checked conversion fails before the frozen unchecked cast")))))

(defn- source-effect-loop [kind index bound statements]
  (if (= :map-void kind)
    (list 'raster.par/map-void! index bound (list* 'do statements))
    (list* kind [index bound] statements)))

(deftest effect-loop-count-and-ordinal-retain-the-source-long-domain
  (doseq [kind [:map-void 'dotimes 'clojure.core/dotimes]]
   (let [parameters [(with-meta 'events {:tag 'longs})
                    (with-meta 'captured {:tag 'objects})
                    (with-meta 'count {:tag 'long}) 'stop]
        body (source-effect-loop kind 'i 'count
                                 '((clojure.core/aset events 0 9)
                                   (clojure.core/aset captured 0 i)
                                   (throw stop)))
        source (binding [*unchecked-math* false] (eval (list 'fn parameters body)))
        artifact (bytecode/compile-specialized-class!
                  "raster.compiled.LongEffectMapCount"
                  [{:name 'run :params parameters :return-tag 'Object
                    :walked-body [body] :source-ns *ns*
                    :source-policy {:unchecked-math false}}])
        compiled (fn [& arguments] (invoke-method (get-in artifact [:methods 'run]) arguments))
        observe (fn [run count]
                  (let [events (long-array 1) captured (object-array 1)
                        stop (RuntimeException. "bounded first iteration")
                        outcome (try (run events captured count stop) :returned
                                     (catch Throwable error
                                       (if (identical? error stop) :stopped (class error))))]
                    {:outcome outcome :events (vec events)
                     :ordinal (aget captured 0) :carrier (class (aget captured 0))}))]
    (doseq [count [Long/MIN_VALUE -1 0 1 3 Integer/MAX_VALUE 2147483648 Long/MAX_VALUE]]
      (is (= (observe source count) (observe compiled count)) (str "count=" count)))
    (is (= {:outcome :stopped :events [9] :ordinal 0 :carrier Long}
           (observe compiled 2147483648))))))

(deftest effect-loop-completes-the-bound-once-and-returns-nil-after-iteration
  (doseq [kind [:map-void 'dotimes 'clojure.core/dotimes]]
   (let [parameters [(with-meta 'events {:tag 'longs})
                    (with-meta 'ordinals {:tag 'objects})]
        body (source-effect-loop
              kind 'i
              '(do (clojure.core/aset events 0 (clojure.core/inc (clojure.core/aget events 0))) 3)
              '((clojure.core/aset events (clojure.core/inc i)
                                  (clojure.core/inc (clojure.core/aget events (clojure.core/inc i))))
                (clojure.core/aset ordinals i (fn [] i))))
        source (binding [*unchecked-math* false] (eval (list 'fn parameters body)))
        artifact (bytecode/compile-specialized-class!
                  "raster.compiled.EffectMapBoundOnce"
                  [{:name 'run :params parameters :return-tag 'Object
                    :walked-body [body] :source-ns *ns* :source-policy {:unchecked-math false}}])
        compiled (fn [& arguments] (invoke-method (get-in artifact [:methods 'run]) arguments))
        observe (fn [run]
                  (let [events (long-array 4) ordinals (object-array 3)]
                    {:result (run events ordinals) :events (vec events)
                     :ordinals (mapv #(%) ordinals)
                     :carriers (mapv #(class (%)) ordinals)}))]
    (is (= {:result nil :events [1 1 1 1] :ordinals [0 1 2] :carriers [Long Long Long]}
           (observe compiled)))
    (is (= (observe source) (observe compiled))))))

(deftest private-counter-specialization-is-fresh-and-keeps-one-source-body
  (let [token (Object.)
        body (with-meta '(record i n) {:source-observation token})
        source (list 'raster.par/map-void! 'i 'n body)
        candidates (atom '[i n temporary-0 temporary-1 temporary-2 temporary-3 temporary-4])
        expanded (with-redefs [clojure.core/gensym
                              (fn [_]
                                (let [candidate (first @candidates)]
                                  (swap! candidates rest) candidate))]
                   (#'bytecode/expand-jvm-effect-map source))
        outer (take-nth 2 (second expanded))
        loop-form (nth expanded 2)
        counters (take-nth 2 (second loop-form))
        generated (set (concat outer counters))
        nodes (tree-seq coll? seq expanded)]
    (is (= 5 (count generated)))
    (is (not-any? generated '[i n]))
    (is (= 1 (count (filter #(identical? body %) nodes))))
    (is (= 1 (count (filter #(identical? token (:source-observation (meta %))) nodes))))
    (is (some #(and (seq? %) (= 'clojure.core/and (first %))) nodes))))

(deftest effect-loop-preserves-nested-recur-and-empty-body-semantics
  (doseq [kind [:map-void 'dotimes 'clojure.core/dotimes]
          statements ['() '((clojure.core/aset out i
                              (loop* [j (clojure.core/long 0)]
                                (if (clojure.core/< j 2)
                                  (recur (clojure.core/inc j)) j))))]]
    (let [parameters [(with-meta 'out {:tag 'longs})]
          body (source-effect-loop kind 'i 3 statements)
          source (eval (list 'fn parameters body))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.NestedEffectLoopRecurrence"
                    [{:name 'run :params parameters :return-tag 'Object
                      :walked-body [body] :source-ns *ns*
                      :source-policy {:unchecked-math false}}])
          compiled (fn [& arguments] (invoke-method (get-in artifact [:methods 'run]) arguments))
          observe (fn [run] (let [out (long-array 3)]
                             [(run out) (vec out)]))]
      (is (= (observe source) (observe compiled)))))
  (doseq [kind [:map-void 'dotimes 'clojure.core/dotimes]]
    (let [body (source-effect-loop kind 'i 3 '((recur (clojure.core/inc i))))]
      (is (thrown? Throwable (eval (list 'fn [] body))))
      (is (thrown? Throwable
            (bytecode/compile-specialized-class!
             "raster.compiled.InvalidEffectLoopRecurrence"
             [{:name 'run :params [] :return-tag 'Object
               :walked-body [body] :source-ns *ns*
               :source-policy {:unchecked-math false}}]))))))

(deftest recurrence-arity-cannot-silently-truncate-carried-values
  (doseq [[body expected actual]
          [['(loop* [a 0 b 0] (recur (clojure.core/inc a))) 2 1]
           ['(loop* [a 0] (recur (clojure.core/inc a) a)) 1 2]]]
    (is (thrown? clojure.lang.Compiler$CompilerException (eval (list 'fn [] body))))
    (let [failure (try
                    (bytecode/compile-specialized-class!
                     "raster.compiled.InvalidLoopCarryArity"
                     [{:name 'run :params [] :return-tag 'Object
                       :walked-body [body] :source-ns *ns*
                       :source-policy {:unchecked-math false}}])
                    nil
                    (catch clojure.lang.ExceptionInfo error (ex-data error)))]
      (is (= {:reason :bytecode-recur-arity :expected expected :actual actual}
             (select-keys failure [:reason :expected :actual]))))))

(deftest core-index-completion-matches-clojure-before-value-effects
  (doseq [policy [false true :warn-on-boxed]
          [tag inputs] [['long [0 1 -1 4294967296 Long/MIN_VALUE Long/MAX_VALUE]]
                        ['double [0.0 0.75 1.0 4294967296.0 Double/NaN
                                  Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
                        ['Object [0 4294967296 0.75 \u0000 nil true (Object.)]]]
          loop? [false true]]
    (let [parameters [(with-meta 'array {:tag 'longs})
                      (with-meta 'index {:tag tag})
                      (with-meta 'effects {:tag 'longs})]
          body (store-body loop?)
          source (binding [*unchecked-math* policy] (eval (list 'fn parameters body)))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.ArrayIndexCompletion"
                    [{:name 'run :params parameters :return-tag 'Object
                      :walked-body [body] :source-ns *ns*
                      :source-policy {:unchecked-math policy}}])
          method (get-in artifact [:methods 'run])
          compiled (fn [& arguments] (invoke-method method arguments))]
      (doseq [index inputs]
        (testing (str "policy=" policy " carrier=" tag " loop=" loop? " index=" (pr-str index))
          (is (= (observe source index) (observe compiled index))))))))

(deftest explicit-integral-casts-share-the-source-policy-and-actual-carrier-rule
  (doseq [policy [false true :warn-on-boxed]
          cast ['clojure.core/int 'clojure.core/long 'clojure.core/unchecked-int]
          [tag inputs] [['long [0 4294967296 Long/MAX_VALUE]]
                        ['double [0.75 4294967296.0 Double/NaN
                                  Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
                        ['Object [0 4294967296 0.75 \u0000 nil true (Object.)]]]]
    (let [parameters [(with-meta 'value {:tag tag})]
          body (list cast 'value)
          source (binding [*unchecked-math* policy] (eval (list 'fn parameters body)))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.SourceIntegralCompletion"
                    [{:name 'run :params parameters :return-tag 'Object
                      :walked-body [body] :source-ns *ns*
                      :source-policy {:unchecked-math policy}}])
          method (get-in artifact [:methods 'run])
          outcome (fn [f value]
                    (try (let [result (f value)] [:value (class result) result])
                         (catch Throwable error [:throw (class error)])))]
      (doseq [value inputs]
        (is (= (outcome source value)
               (outcome #(invoke-method method [%]) value))
            (str policy " " cast " " tag " " (pr-str value)))))))


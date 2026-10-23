(ns raster.compiler.backend.jvm.terminal-context-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.split :as split]))

(deftm conditional-object [xs :- (Array double) selected :- Long] :- Object
  (let [value (aget xs 0)]
    (if (= selected 1) (float value) nil)))

(deftm shadowed-value [xs :- (Array double)] :- Double
  (let [value (aget xs 0) value (aget xs 1)] value))

(def prefix-events (atom []))

(deftm observed-prefix-effect [value :- Double] :- Float
  (let [ignored (do (swap! prefix-events conj :prefix) (long 1))
        result (float value)] result))

(deftm nullable-object [input :- Object selected :- Long] :- Object
  (let [value input] (if (= selected 1) (float value) nil)))

(deftm typed-reference [input :- Object] :- java.lang.String
  (let [value input] value))

(deftm nested-tail [xs :- (Array double) selected :- Long] :- Object
  (let [value (aget xs 0)]
    (let [inside (+ value 1.0)] (if (= selected 1) (float inside) nil))))

(deftm forced-split-tail [xs :- (Array double) cnt :- Long] :- Double
  (let [sum (par/reduce acc 0.0 i cnt (+ acc (aget xs i)))] sum))

(deftm array-terminal [xs :- (Array double) index :- Long] :- Double
  (let [position index] (aget xs position)))

(deftm constant-tail [unused :- Double] :- Object
  (let [ignored unused] 1.5))

(deftm shadowed-callee [float :- Object input :- Double] :- Object
  (let [argument input] (float argument)))

(defn- emitted [v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns ^clojure.lang.Var source)
                             (symbol (str (:name (meta source)) "-impl")))]
    (core/ensure-bytecode-implementation! receiver)
    (:raster.core/bytecode-emission (meta receiver))))

(deftest tail-observation-keeps-object-context-and-actual-final-ports
  (let [emission (emitted #'conditional-object)
        scope (first (:body emission))
        record (bytecode/retained-terminal-context emission scope)
        binding (bytecode/retained-binding-context emission scope 0)]
    (is (some? record))
    (is (= '#{value selected} (set (keys (:ports record)))))
    (is (= (:after binding) (get-in record [:ports 'value])))
    (is (= :double (get-in record [:ports 'value :type])))
    (is (= :long (get-in record [:ports 'selected :type])))
    (is (= :ref (:emitted-result-type record)))
    (is (= "Ljava/lang/Object;" (get-in record [:owner :return-descriptor])))
    (is (true? (get-in record [:context :object-result?])))
    (is (false? (get-in record [:owner :result-context :preserve-source-call-results?])))
    (is (true? (get-in record [:context :preserve-source-call-results?]))
        "the actual Object region strengthens call-result preservation on entry")
    (is (= (dissoc record :owner) (get-in record [:owner :terminal-context])))))

(deftest tail-ports-use-the-last-shadowed-binding-not-its-spelling
  (let [emission (emitted #'shadowed-value)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)
        second-binding (bytecode/retained-binding-context emission scope 1)
        record (bytecode/retained-terminal-context emission scope)]
    (is (some? record))
    (is (= '#{value} (set (keys (:ports record)))))
    (is (= (:after second-binding) (get-in record [:ports 'value])))
    (is (not= (:after first-binding) (get-in record [:ports 'value])))
    (is (= :double (:emitted-result-type record)))
    (is (= "D" (get-in record [:owner :return-descriptor])))))

(deftest context-query-does-not-evaluate-the-source-prefix
  (reset! prefix-events [])
  (let [emission (emitted #'observed-prefix-effect)
        scope (first (:body emission))
        record (bytecode/retained-terminal-context emission scope)
        region (bytecode/compile-terminal-result-region emission scope)]
    (is (some? record))
    (is (empty? @prefix-events))
    (is (= '#{result} (set (keys (:ports record)))))
    (is (= :float (:emitted-result-type record)))
    (is (= "F" (get-in record [:owner :return-descriptor])))
    (is (= (Float/valueOf 2.0) ((:invoke region) (float 2.0))))
    (is (empty? @prefix-events) "outlining and executing the tail must not replay its prefix")
    (is (= (Float/valueOf 3.0) (observed-prefix-effect 3.0)))
    (is (= [:prefix] @prefix-events))))

(deftest copied-or-inconsistent-terminal-evidence-is-not-an-owner
  (let [emission (emitted #'conditional-object)
        scope (first (:body emission))
        token (::bytecode/binding-scope-occurrence (meta scope))
        key [:terminal-contexts token]]
    (is (nil? (bytecode/retained-terminal-context
               emission (with-meta (apply list scope) (meta scope)))))
    (is (nil? (bytecode/retained-terminal-context
               (assoc emission :body (with-meta (vec (:body emission)) {})) scope)))
    (doseq [changed [(assoc-in emission (conj key :status) :ambiguous)
                     (assoc-in emission (conj key :binding-count) 2)
                     (assoc-in emission (conj key :emitted-result-type) :float)
                     (assoc-in emission (conj key :ports 'value :slot) -1)
                     (assoc-in emission (conj key :context :object-result?) false)
                     (assoc-in emission (conj key :body)
                               [(with-meta (apply list (first (get-in emission (conj key :body)))) {})])]]
      (is (nil? (bytecode/retained-terminal-context changed scope))))))

(defn- result-shape [f]
  (try
    (let [value (f)]
      (cond
        (instance? Float value) [:float (Float/floatToRawIntBits value)]
        (instance? Double value) [:double (Double/doubleToRawLongBits value)]
        :else [:value value]))
    (catch Throwable error [:error (class error)])))

(defn- run-with [region arguments]
  (apply (:invoke region) (mapv arguments (:parameters region))))

(defn- conditional-object-oracle [^doubles xs ^long selected]
  ;; Keep an independent ordinary-Clojure oracle with the actual source context:
  ;; a typed array read produces a primitive Double, not an inferred result tag.
  (let [value (aget xs 0)] (if (= selected 1) (float value) nil)))

(deftest outlined-tail-keeps-lazy-checked-conversion-and-object-boxing
  (let [emission (emitted #'conditional-object)
        region (bytecode/compile-terminal-result-region emission (first (:body emission)))
        ^java.lang.reflect.Method method (:method region)]
    (is (identical? Object (.getReturnType method)))
    (is (= [Long/TYPE Double/TYPE] (vec (.getParameterTypes method))))
    (doseq [selected [0 1]
            value [1.5 16777217.0 -0.0 Double/NaN Double/POSITIVE_INFINITY
                   Double/NEGATIVE_INFINITY Double/MAX_VALUE]]
      (is (= (result-shape #(conditional-object-oracle (double-array [value]) selected))
             (result-shape #(conditional-object (double-array [value]) selected))
             (result-shape #(run-with region {'selected selected 'value value})))
          (str "selected=" selected ", value=" value)))
    (is (nil? (run-with region {'selected 0 'value Double/MAX_VALUE})))
    (is (instance? Float (run-with region {'selected 1 'value 1.5}))))
  (let [emission (emitted #'nullable-object)
        region (bytecode/compile-terminal-result-region emission (first (:body emission)))]
    (doseq [selected [0 1] value [nil 2.5 "not numeric"]]
      (is (= (result-shape #(if (= selected 1) (float value) nil))
             (result-shape #(nullable-object value selected))
             (result-shape #(run-with region {'selected selected 'value value})))))))

(deftest outlining-keeps-the-actual-reference-return-check
  (let [emission (emitted #'typed-reference)
        scope (first (:body emission))
        record (bytecode/retained-terminal-context emission scope)
        region (bytecode/compile-terminal-result-region emission scope)]
    (is (= :ref (:emitted-result-type record)))
    (is (= "Ljava/lang/String;" (get-in record [:owner :return-descriptor])))
    (is (identical? String (.getReturnType ^java.lang.reflect.Method (:method region))))
    (doseq [value [nil "text" 2.5]]
      (is (= (result-shape #(cast String value))
             (result-shape #(typed-reference value))
             (result-shape #((:invoke region) value)))))))

(deftest only-the-outer-scope-is-a-terminal-region-owner
  (let [emission (emitted #'nested-tail)
        scope (first (:body emission))
        inner (first (drop 2 scope))
        record (bytecode/retained-terminal-context emission scope)
        region (bytecode/compile-terminal-result-region emission scope)]
    (is (some? record))
    (is (= '#{value selected} (set (keys (:ports record)))))
    (is (nil? (bytecode/retained-terminal-context emission inner)))
    (is (= (Float/valueOf 3.5) (run-with region {'value 2.5 'selected 1})))
    (is (nil? (run-with region {'value 2.5 'selected 0})))))

(deftest terminal-outlining-refuses-stale-or-missing-owner-evidence
  (let [emission (emitted #'conditional-object)
        scope (first (:body emission))
        token (::bytecode/binding-scope-occurrence (meta scope))
        body-token (::bytecode/body-occurrence (meta (:body emission)))
        region (bytecode/compile-terminal-result-region emission scope)]
    (doseq [changed [(dissoc emission :terminal-contexts)
                     (assoc-in emission [:terminal-contexts token :status] :ambiguous)
                     (update emission :body-contexts dissoc body-token)
                     (assoc-in emission [:body-contexts body-token :status] :ambiguous)
                     (assoc-in emission [:body-contexts body-token :method] nil)
                     (assoc-in emission [:body-contexts body-token :return-descriptor] "D")]]
      (is (= :unowned-bytecode-terminal-region
             (try (bytecode/compile-terminal-result-region changed scope) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (let [dependency (:var (first (:dependency-snapshots emission)))]
      (is (var? dependency))
      (with-redefs-fn {dependency (fn [& _] :stale)}
        #(do
           (is (thrown? clojure.lang.ExceptionInfo
                        (bytecode/retained-terminal-context emission scope)))
           (is (thrown? clojure.lang.ExceptionInfo
                        (run-with region {'selected 1 'value 2.5}))))))
    (is (= (Float/valueOf 2.5) (run-with region {'selected 1 'value 2.5})))))

(deftest split-emission-does-not-authorize-an-original-unsplit-tail
  ;; TARGET-SIZE is a compile-time constant. Force the size decision through
  ;; its measurement function rather than redefining an already inlined Var.
  (with-redefs [split/estimate-form-size (constantly 1000000)]
    (let [emission (emitted #'forced-split-tail)
          scope (first (:body emission))]
      (is (nil? (bytecode/retained-body-context emission)))
      (is (nil? (bytecode/retained-terminal-context emission scope)))
      (is (= :unowned-bytecode-terminal-region
             (try (bytecode/compile-terminal-result-region emission scope) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest terminal-array-ports-retain-the-actual-verifier-class
  (let [emission (emitted #'array-terminal)
        region (bytecode/compile-terminal-result-region emission (first (:body emission)))
        array-class (class (double-array 0))]
    (is (identical? array-class (get-in region [:physical-ports 'xs :reference-class])))
    (is (= [array-class Long/TYPE]
           (vec (.getParameterTypes ^java.lang.reflect.Method (:method region)))))
    (doseq [[xs position] [[(double-array [2.0 7.5]) 1] [nil 0] [(double-array 0) 0]]]
      (is (= (result-shape #(aget xs position))
             (result-shape #(array-terminal xs position))
             (result-shape #(run-with region {'xs xs 'position position})))))))

(deftest empty-port-and-shadowed-call-tails-use-the-same-owner-boundary
  (let [emission (emitted #'constant-tail)
        region (bytecode/compile-terminal-result-region emission (first (:body emission)))]
    (is (empty? (:parameters region)))
    (is (= (result-shape #(constant-tail 2.0)) (result-shape #((:invoke region))))))
  (let [emission (emitted #'shadowed-callee)
        region (bytecode/compile-terminal-result-region emission (first (:body emission)))
        callback (fn [x] [:shadowed x])]
    (is (= '#{float argument} (set (:parameters region))))
    (is (= (shadowed-callee callback 2.5)
           (run-with region {'float callback 'argument 2.5})))))

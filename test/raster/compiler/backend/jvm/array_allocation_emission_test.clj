(ns raster.compiler.backend.jvm.array-allocation-emission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm fresh-doubles [n :- Long] :- (Array double) (double-array n))
(deftm fresh-floats [n :- Long] :- (Array float) (float-array n))
(deftm filled-doubles [n :- Long value :- Double] :- (Array double)
  (double-array n value))
(deftm literal-doubles [ignored :- Long] :- (Array double) (double-array [0.0 1.0]))
(deftm observable-count [events :- (Array long) n :- Long] :- (Array double)
  (double-array (do (aset events 0 (inc (aget events 0))) n)))

(defn- owned-allocation [v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))
        emission (core/ensure-current-owned-bytecode! receiver)
        allocations (keep (fn [call]
                            (when-let [observation (bytecode/retained-zero-array-allocation emission call)]
                              {:source call :observation observation}))
                          (vals (:invocation-catalog emission)))]
    {:emission emission :primal @receiver :allocations (vec allocations)}))

(deftest observed-zero-allocation-matches-actual-class-count-and-freshness
  (doseq [[v expected-class zero] [[#'fresh-doubles (class (double-array 0)) 0.0]
                                  [#'fresh-floats (class (float-array 0)) (float 0.0)]]]
    (let [{:keys [primal allocations]} (owned-allocation v)
          observation (:observation (first allocations))]
      (is (= 1 (count allocations)))
      (is (identical? expected-class (:array-class observation)))
      (is (true? (:fresh? observation)))
      (is (= :jvm-zero (:initialization observation)))
      (is (true? (:primitive-count? observation)))
      (is (vector? (:count-routes observation)))
      (is (seq (:count-routes observation)))
      (is (identical? (second (:expression observation))
                      (get-in observation [:count :operand])))
      (is (= :long (get-in observation [:count :input-type])))
      (is (= :int (get-in observation [:count :completed-type])))
      (is (= :raw-jvm-int (get-in observation [:count :completion])))
      (is (= :long (get-in observation [:count :ports 'n :type])))
      (doseq [n [0 1 3]]
        (let [left (primal n) right (primal n)]
          (is (not (identical? left right)))
          (is (identical? expected-class (class left)))
          (is (= (vec (repeat n zero)) (vec left)))))
      ;; Preserve what the actual bytecode does: its final count coercion is
      ;; l2i, not Clojure's potentially trapping checked int conversion.
      (is (= 1 (alength (primal 4294967297))))
      (is (thrown? NegativeArraySizeException (primal -1))))))

(deftest zero-allocation-observation-does-not-certify-fill-or-literal-routes
  (is (empty? (:allocations (owned-allocation #'filled-doubles))))
  (is (empty? (:allocations (owned-allocation #'literal-doubles)))))

(deftest fresh-storage-does-not-imply-a-pure-or-replayable-count
  (let [{:keys [primal allocations]} (owned-allocation #'observable-count)
        observation (:observation (first allocations))
        events (long-array 1)]
    (is (= 1 (count allocations)))
    (is (seq? (get-in observation [:count :operand])))
    (is (= #{'events 'n} (set (keys (get-in observation [:count :ports])))))
    (is (not-any? #(contains? observation %) [:pure? :replayable? :does-not-escape?]))
    (is (false? (:primitive-count? observation)))
    (is (= [] (vec (primal events 0))))
    (is (= 1 (aget events 0)))
    (is (thrown? NegativeArraySizeException (primal events -1)))
    (is (= 2 (aget events 0)))))

(deftest count-region-preserves-raw-completion-without-allocating-or-replaying
  (doseq [v [#'fresh-doubles #'fresh-floats]]
    (let [{:keys [emission allocations]} (owned-allocation v)
          original (:source (first allocations))
          region (bytecode/compile-array-count-region emission original)]
      (is (= ['n] (:parameters region)))
      (is (= Integer/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
      (is (= 3 ((:invoke region) 3)))
      (is (= 1 ((:invoke region) 4294967297)))
      (is (= -1 ((:invoke region) -1)) "the count region does not execute the allocation")
      (is (= :unowned-bytecode-array-count-region
             (try (bytecode/compile-array-count-region emission
                                                       (with-meta (apply list original) (meta original)))
                  nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
  (let [{:keys [emission allocations]} (owned-allocation #'observable-count)
        region (bytecode/compile-array-count-region emission (:source (first allocations)))
        events (long-array 1)
        arguments (mapv {'events events 'n 2} (:parameters region))]
    (is (= 2 (apply (:invoke region) arguments)))
    (is (= 1 (aget events 0)))))

(deftest allocation-evidence-stays-with-original-occurrence-and-historical-emission
  (let [{:keys [emission allocations]} (owned-allocation #'fresh-doubles)
        source (:source (first allocations))
        copy (with-meta (apply list source) (meta source))
        token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta source))]
    (is (some? (bytecode/retained-zero-array-allocation emission source)))
    (is (nil? (bytecode/retained-zero-array-allocation emission copy)))
    (is (nil? (bytecode/retained-zero-array-allocation
               (assoc-in emission [:array-allocation-contexts token :status] :ambiguous) source)))
    (is (= :unstable-emission-inputs
           (try
             (bytecode/retained-zero-array-allocation
              (update emission :completed-definition-revision inc) source)
             nil
             (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))))

(defn- fresh-count-emission [unchecked? operand]
  (let [node (list 'clojure.core/double-array operand)
        input-body [node]
        interface (types/ensure-fn-interface! '[long long])
        emitted (binding [*unchecked-math* unchecked?]
                  (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.FrozenArrayCount"
                   '[x y] '[long long] 'Object input-body
                   (the-ns 'raster.compiler.backend.jvm.array-allocation-emission-test)
                   (:iface-name interface) 'frozen-array-count))
        original (first (filter #(bytecode/retained-zero-array-allocation (:emission emitted) %)
                                (vals (get-in emitted [:emission :invocation-catalog]))))]
    (assoc emitted :input-body input-body :source node :original original)))

(defn- count-outcome [f & arguments]
  (try {:value (apply f arguments)} (catch Throwable failure {:error (class failure)})))

(deftest array-count-replay-retains-frozen-operand-policy-under-opposite-ambient-policy
  (doseq [unchecked? [false true]]
    (let [operand '(clojure.core/+ (clojure.core/int x) y)
          {:keys [implementation emission input-body source original]}
          (fresh-count-emission unchecked? operand)
          input (bytecode/retained-input-array-count-context emission input-body source)
          region (binding [*unchecked-math* (not unchecked?)]
                   (bytecode/compile-array-count-region emission original))]
      (is (= {:unchecked-math unchecked?} (:source-policy input)))
      (is (identical? operand (:source-original input)))
      (is (identical? (:operand input) (get-in input [:allocation :count :operand])))
      (is (= '[x y] (:parameters region)))
      (is (= {:unchecked-math unchecked?} (get-in region [:spec :source-policy])))
      (doseq [[x y] [[0 0] [3 2] [-1 0] [4294967297 0]]]
        (let [count-result (count-outcome (:invoke region) x y)
              allocated (count-outcome #(alength (implementation %1 %2)) x y)]
          (if-let [error (:error count-result)]
            (is (= error (:error allocated)))
            (if (neg? (:value count-result))
              (is (= NegativeArraySizeException (:error allocated)))
              (is (= count-result allocated))))))))
  (let [{:keys [emission original]} (fresh-count-emission false 'x)
        region (binding [*unchecked-math* true]
                 (bytecode/compile-array-count-region emission original))]
    (is (= 1 ((:invoke region) 4294967297)))
    (is (= -1 ((:invoke region) -1)) "the raw Int count region never allocates")))

(deftest array-count-origin-and-policy-cannot-be-copied-or-filled-from-ambient-state
  (let [{:keys [emission input-body source original]}
        (fresh-count-emission false '(clojure.core/+ x y))
        foreign (fresh-count-emission false '(clojure.core/+ x y))
        token (:raster.compiler.backend.jvm.bytecode/call-occurrence (meta original))
        constructions (atom 0)]
    (is (some? (bytecode/retained-input-array-count-context emission input-body source)))
    (is (nil? (bytecode/retained-input-array-count-context
               (assoc-in emission [:array-allocation-contexts token :status] :ambiguous)
               input-body source)))
    (is (nil? (bytecode/retained-input-array-count-context
               (update-in emission [:array-allocation-contexts token :count] dissoc :source-policy)
               input-body source)))
    (doseq [[owner body node]
            [[emission input-body (with-meta (apply list source) (meta source))]
             [emission (vec (seq input-body)) source]
             [(:emission foreign) input-body source]]]
      (is (nil? (bytecode/retained-input-array-count-context owner body node))))
    (with-redefs [bytecode/compile-specialized-class!
                  (fn [& _] (swap! constructions inc)
                    (throw (AssertionError. "unowned count reached class construction")))]
      (is (= :unclosed-bytecode-array-count-policy
             (try (bytecode/compile-array-count-region
                   (update-in emission [:array-allocation-contexts token :count] dissoc :source-policy)
                   original)
                  :accepted
                  (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (bytecode/compile-array-count-region
                    (update emission :completed-definition-revision inc) original)))
      (is (zero? @constructions)))))

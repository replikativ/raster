(ns raster.compiler.backend.jvm.array-allocation-emission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
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

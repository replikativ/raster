(ns raster.compiler.backend.jvm.binding-context-owner-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.set :as set]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm response [a :- Double k :- Double x :- Double] :- Double
  (n/* a (n// x (n/+ k x))))
(deftm array-tail [theta :- (Array double) x :- Double] :- Double
  (let [a (aget theta 0) k (aget theta 1)] (response a k x)))
(deftm shadowed [theta :- (Array double) x :- Double] :- Double
  (let [a (aget theta 0) a (aget theta 1)] (response a a x)))
(deftm inferred-binding [x :- Double] :- Double
  (let [power (n/* x x)] power))
(deftm weighted-reduction [xs :- (Array double) coefficient :- Double cnt :- Long] :- Double
  (let [gain (n/* coefficient coefficient)]
    (par/reduce acc 0.0 i cnt (n/+ acc (n/* gain (aget xs i))))))
(deftm float-reduced [xs :- (Array float) cnt :- Long] :- Float
  (par/reduce acc (float 0.0) i cnt (n/+ acc (aget xs i))))
(deftm initialized-reduction [xs :- (Array double) events :- (Array long) cnt :- Long] :- Double
  (par/reduce acc (double (do (aset events 0 (long 1)) (aget xs 0))) i cnt (n/+ acc 1.0)))

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- emitted [v]
  (let [source (core/resolve-deftm-var v)
        receiver (ns-resolve (.ns source) (symbol (str (:name (meta source)) "-impl")))]
    (core/ensure-bytecode-implementation! receiver)
    (:raster.core/bytecode-emission (meta receiver))))

(deftest sequential-binding-facts-come-from-the-actual-owner
  (let [emission (emitted #'array-tail)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)
        second-binding (bytecode/retained-binding-context emission scope 1)]
    (is (= 'a (:binder first-binding)))
    (is (= 'k (:binder second-binding)))
    (is (= :double (get-in first-binding [:after :type])))
    (is (= (:after first-binding) (get-in second-binding [:before 'a])))
    (is (= :ref (get-in first-binding [:before 'theta :type])))
    (is (identical? (class (double-array 0)) (get-in first-binding [:before 'theta :reference-class])))
    (is (nil? (bytecode/retained-binding-context emission scope -1)))
    (is (nil? (bytecode/retained-binding-context emission scope 2)))
    (is (nil? (bytecode/retained-binding-context
               emission (with-meta (apply list scope) (meta scope)) 0)))))

(deftest whole-body-entry-and-return-facts-belong-to-the-original-body
  (let [emission (emitted #'array-tail)
        context (bytecode/retained-body-context emission)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)]
    (is (identical? (:body emission) (:body context)))
    (is (= (:locals context) (:before first-binding)))
    (is (= :double (:return-stack-type context)))
    (is (= {:object-result? false :object-call-result? false
            :preserve-source-call-results? false :void-context false} (:result-context context)))
    (is (nil? (bytecode/retained-body-context
               (assoc emission :body (with-meta (into [] (:body emission)) (meta (:body emission)))))))))

(deftest same-spelling-is-not-a-binding-identity
  (let [emission (emitted #'shadowed)
        scope (first (:body emission))
        first-binding (bytecode/retained-binding-context emission scope 0)
        second-binding (bytecode/retained-binding-context emission scope 1)]
    (is (= (:binder first-binding) (:binder second-binding)))
    (is (not= (get-in first-binding [:after :slot]) (get-in second-binding [:after :slot])))
    (is (= (:after first-binding) (get-in second-binding [:before 'a])))
    (let [call (first (filter #(= 'raster.compiler.backend.jvm.binding-context-owner-test/response
                                  (:raster.op/original (meta %)))
                             (vals (:invocation-catalog emission))))
          context (bytecode/retained-invocation-context emission call)]
      (is (= (:after second-binding) (get-in context [:inputs 0 :ports 'a]))))))

(deftest new-emission-does-not-reuse-lexical-occurrence-authority
  (let [first-owner (bytecode/retain-invocation-occurrences '[(let* [x 1.0] x)])
        second-owner (bytecode/retain-invocation-occurrences (:body first-owner))]
    (is (= (:body first-owner) (:body second-owner)))
    (is (empty? (set/intersection (set (keys (:binding-catalog first-owner)))
                                         (set (keys (:binding-catalog second-owner))))))))

(deftest binding-layout-is-part-of-the-occurrence-proof
  (let [emission (emitted #'inferred-binding)
        scope (first (:body emission))
        record (bytecode/retained-binding-context emission scope 0)
        key (first (keys (:binding-contexts emission)))]
    (is (some? record))
    (is (= 'power (:binder record)))
    (is (= 1 (:binding-count record)))
    (is (nil? (bytecode/retained-binding-context
               (assoc-in emission [:binding-contexts key :binding-count] 2) scope 0)))
    (is (nil? (bytecode/retained-binding-context
               (assoc-in emission [:binding-contexts key :binder] 'different) scope 0)))))

(deftest reduction-observations-retain-the-actual-loop-and-backedge-carriers
  (let [emission (emitted #'weighted-reduction)
        source (first (vals (:reduction-catalog emission)))
        {:keys [expansion bound loop recur] :as observed}
        (bytecode/retained-reduction-emission emission source)]
    (is (some? observed))
    (is (identical? source (:source expansion)))
    (is (= [:int :double] (mapv #(get-in loop [:locals % :type]) (:variables loop))))
    (is (= [:int :double] (mapv #(get-in % [:after :type]) (:initializers loop))))
    (is (= [:int :double] (mapv :type (:values recur))))
    (is (= {:object-result? false :object-call-result? false
            :preserve-source-call-results? true :void-context false}
           (:context bound)))
    (is (every? #(= (:context bound) (:context %)) (:initializers loop)))
    (is (identical? (:label-id loop) (:label-id recur)))
    (is (= (:after bound) (get-in loop [:outer-locals (:binder bound)])))
    (is (= :ref (get-in loop [:outer-locals 'gain :type])))
    (is (identical? Object (get-in loop [:outer-locals 'gain :reference-class])))
    (is (not (contains? emission :reduction-labels)))
    (is (nil? (bytecode/retained-reduction-emission
               emission (with-meta (apply list source) (meta source)))))
    (is (nil? (bytecode/retained-reduction-emission
               (assoc-in emission [:reduction-contexts [(:witness expansion) :recur] :label-id] (Object.))
               source)))))

(deftest reduction-observations-do-not-assume-double-carries
  (let [emission (emitted #'float-reduced)
        source (first (vals (:reduction-catalog emission)))
        observed (bytecode/retained-reduction-emission emission source)]
    (is (some? observed))
    (is (= [:int :float] (mapv #(get-in observed [:loop :locals % :type])
                              (get-in observed [:loop :variables]))))))

(deftest observed-reduction-retains-count-before-initializer-order
  (doseq [cnt [-1 0 2]]
    (let [events (long-array [0])]
      (is (= (+ 3.0 (max 0 cnt)) (initialized-reduction (double-array [3.0]) events cnt)))
      (is (= [1] (vec events)))))
  (let [events (long-array [0])
        expected (try (int Long/MAX_VALUE) nil (catch Exception error (class error)))
        actual (try (initialized-reduction (double-array 0) events Long/MAX_VALUE) nil
                    (catch Exception error (class error)))]
    (is (= expected actual))
    (is (= [0] (vec events)))))

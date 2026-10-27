(ns raster.compiler.core.generic-source-return-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.normal-return :as normal]))

(defn- with-fresh-source [check]
  (let [scope (create-ns (gensym "raster.generic-source-return."))
        events (atom 0)]
    (try
      (intern scope 'events events)
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm allocate [input :- (Array float)] :- (Array float)
                 (swap! events inc)
                 (clojure.core/float-array (clojure.core/alength input))))
        (eval '(raster.core/deftm relay [input :- (Array float)] :- (Array float)
                 (allocate input))))
      (check {:scope scope :events events
              :operation (symbol (str (ns-name scope)) "relay")
              :allocator-operation (symbol (str (ns-name scope)) "allocate")})
      (finally (remove-ns (ns-name scope))))))

(defn- result-fact [operation presence]
  (normal/result-fact (list operation 'input)
                      {'input {:presence presence :array-tag 'floats}}))

(defn- owned-fact [operation presence]
  ;; A semantic demand can restart this callback. Capture only the successful
  ;; attempt's history, never an externally held discovery collector.
  (entry/call-with-definition-inputs
   #(let [fact (result-fact operation presence)]
      {:fact fact :inputs @entry/*emission-inputs*})))

(defn- registered-entry [scope name tags]
  (first (filter #(= tags (:tags %))
                 (get @(:raster.core/dispatch-table (meta (ns-resolve scope name)))
                      (count tags)))))

(deftest cold-generic-return-requires-the-real-semantic-preparation-owner
  (with-fresh-source
    (fn [{:keys [scope events operation allocator-operation]}]
      (is (= {:presence :unknown}
             (binding [entry/*emission-inputs* (atom {:static [] :values []})]
               (result-fact operation :present)))
          "an optional raw collector cannot walk or manufacture a frontend")
      (let [{:keys [fact inputs]} (owned-fact operation :present)
            allocator-source (core/resolve-deftm-var (find-var allocator-operation))
            receivers (mapv (fn [name]
                              (let [signature (inference/registered-call-signature
                                               (symbol (str (ns-name scope)) (str name)) '[floats])
                                    source (find-var (:mangled-sym signature))]
                                (ns-resolve scope (symbol (str (.sym source) "-impl")))))
                            '[allocate relay])]
        (is (= {:presence :present :array-tag 'floats} fact)
            "the transitive allocation body, not its nullable return annotation, proves presence")
        (is (contains? (:definition-proofs inputs) allocator-source))
        (is (true? (entry/assert-definition-inputs-current! inputs)))
        (is (seq receivers))
        (is (not-any? #(contains? (:definition-proofs inputs) %) receivers)
            "source-only facts do not acquire replaceable physical receiver snapshots")
        (is (every? #(nil? (:raster.core/bytecode-emission (meta %))) receivers)
            "semantic source preparation does not emit JVM numerical modules")
        (is (zero? @events) "preparation never executes either numerical body")
        (let [signature (inference/registered-call-signature allocator-operation '[floats])
              direct (entry/call-with-definition-inputs
                       #(normal/result-fact
                         (list (:mangled-sym signature) 'input)
                         {'input {:presence :unknown :array-tag 'floats}}))]
          (is (= {:presence :present :array-tag 'floats} direct)
              "a fixed mangled callable needs no nullable overload selection"))
        (doseq [name '[allocate relay]
                :let [method (registered-entry scope name '[floats])]]
          (is (some? (:typed-impl method)))
          (is (identical? (:target-fn method) (:typed-target-fn method))))
        (is (= [] (vec ((deref (ns-resolve scope 'relay)) (float-array 0)))))
        (is (= 1 @events))
        (is (true? (entry/assert-definition-inputs-current! inputs))
            "legitimate first JIT of the same original frontend preserves semantic facts")))))

(deftest nullable-array-facts-do-not-select-a-generic-reference-overload
  (with-fresh-source
    (fn [{:keys [scope events operation]}]
      ;; A real second nullable reference alternative makes the nil-dispatch
      ;; ambiguity substantive, even though the operand retains an array tag.
      (binding [*ns* scope]
        (eval '(raster.core/deftm relay [input :- (Array double)] :- Object nil)))
      (let [present (owned-fact operation :present)
            nullable (owned-fact operation :unknown)]
        (is (= {:presence :present :array-tag 'floats} (:fact present)))
        (is (= {:presence :unknown} (:fact nullable))))
      (is (zero? @events)))))

(deftest arbitrary-typed-dispatch-wrapper-cannot-inherit-the-original-return-fact
  (with-fresh-source
    (fn [{:keys [scope events operation]}]
      (let [{:keys [fact inputs]} (owned-fact operation :present)
            original (registered-entry scope 'relay '[floats])
            replacement (fn [& _] nil)]
        (is (= {:presence :present :array-tag 'floats} fact))
        ;; Public registration creates a genuine new dispatcher while keeping
        ;; the authentic typed implementation. Only the executed wrapper differs.
        (dispatch/register-method! (:target-fn original) 'relay '[floats] scope
                                   (:typed-impl original) (:typed-iface original) replacement)
        (let [changed (registered-entry scope 'relay '[floats])]
          (is (identical? (:typed-impl original) (:typed-impl changed)))
          (is (identical? replacement (:typed-target-fn changed)))
          (is (not (identical? (:target-fn changed) (:typed-target-fn changed)))))
        (is (= {:presence :unknown} (:fact (owned-fact operation :present))))
        (is (nil? ((deref (ns-resolve scope 'relay)) (float-array 0))))
        (is (= :stale-definition-proof-input
               (try (entry/assert-definition-inputs-current! inputs)
                    nil
                    (catch clojure.lang.ExceptionInfo failure (:reason (ex-data failure))))))
        (is (zero? @events))))))

(deftest parametric-specialization-registers-its-actual-published-source-callable
  (with-fresh-source
    (fn [{:keys [scope events]}]
      (binding [*ns* scope]
        (eval '(raster.core/deftm parametric-identity
                 (All [T] [input :- (Array T)] :- (Array T)
                   (do (swap! events inc) input)))))
      (let [operation (symbol (str (ns-name scope)) "parametric-identity")
            ;; This explicitly constructs a Float specialization, exercising
            ;; register-parametric-specialization!, not just eager deftm Double.
            signature (inference/specialize-call-signature operation '[floats])
            source (find-var (:mangled-sym signature))
            method (registered-entry scope 'parametric-identity '[floats])]
        (is (= '[floats] (:tags signature)))
        (is (some? (:typed-impl method)))
        (is (identical? @source (:target-fn method)))
        (is (identical? (:target-fn method) (:typed-target-fn method)))
        (is (zero? @events) "constructing the specialization never invokes its body")))))

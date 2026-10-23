(ns raster.compiler.backend.jvm.frontend-binding-region-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as numeric]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftm scalar-bindings [x :- Double y :- Double] :- Double
  (let [scaled (numeric/* x y)
        result (numeric// scaled y)]
    result))

(deftm effect-bindings [events :- (Array long) x :- Double] :- Double
  (let [first-value (do (aset events 0 (inc (aget events 0))) (numeric/* x 2.0))
        result (do (aset events 1 (inc (aget events 1))) (numeric/+ first-value x))]
    result))

(deftm reference-binding-checked [value :- Object unused :- Long] :- Object
  (let [^String text value] text))

(defn- captured
  ([] (scalar-bindings 3.0 2.0) (captured #'scalar-bindings))
  ([function]
  (entry/call-with-definition-inputs
   #(entry/original-source-frontend (core/resolve-deftm-var function)))))

(defn- prepare [frontend scope index]
  (entry/call-with-definition-inputs
   #(entry/prepare-frontend-binding-input! frontend scope index)))

(defn- reason [run]
  (try (run) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest binding-input-runs-only-the-requested-initializer
  (let [frontend (captured) scope (first (:body frontend))
        scaled (prepare frontend scope 0)
        quotient (prepare frontend scope 1)]
    (is (some? scaled))
    (is (some? quotient))
    (is (identical? scope (:source-input quotient)))
    (is (= 1 (:source-binding-index quotient)))
    (is (= Object (.getReturnType ^java.lang.reflect.Method (:method quotient))))
    (is (= 2 (count (:parameters scaled)) (count (:parameters quotient))))
    ;; `scaled` is an already-completed lexical input here. The original first
    ;; initializer is not rerun, and x is not an input to the second initializer.
    (is (not (some #{'x} (:parameters quotient))))
    (doseq [[x y] [[3.0 2.0] [-0.0 2.0] [1.0 0.0] [0.0 0.0]
                   [Double/MAX_VALUE 0.5]]]
      (let [first-value ((:invoke scaled) x y)
            arguments (mapv #(if (= 'y %) y first-value) (:parameters quotient))
            actual (apply (:invoke quotient) arguments)
            expected (scalar-bindings x y)]
        (is (or (and (Double/isNaN expected) (Double/isNaN actual))
                (= (Double/doubleToRawLongBits expected)
                   (Double/doubleToRawLongBits actual))))))
    (let [forbidden (fn [& _] (throw (AssertionError. "binding invocation refreshed source evidence")))]
      (with-redefs [entry/observe-definition-input! forbidden
                    entry/observe-frontend-source-identity! forbidden
                    bytecode/compile-input-binding-region forbidden]
        (is (= 6.0 ((:invoke scaled) 3.0 2.0)))))))

(deftest binding-input-requires-exact-owned-scope-and-index
  (let [frontend (captured) scope (first (:body frontend))
        owner (atom {:static [] :values []}) before @owner]
    (is (nil? (entry/prepare-frontend-binding-input! frontend scope 0)))
    (binding [entry/*emission-inputs* owner]
      (doseq [scope [(with-meta (apply list scope) (meta scope)) nil]]
        (is (nil? (entry/prepare-frontend-binding-input! frontend scope 0)))
        (is (identical? before @owner)))
      (doseq [index [-1 2 nil 0.5]]
        (is (nil? (entry/prepare-frontend-binding-input! frontend scope index)))
        (is (identical? before @owner))))))

(deftest binding-region-keeps-effects-exceptions-and-completed-inputs
  (effect-bindings (long-array 2) 3.0)
  (let [frontend (captured #'effect-bindings) scope (first (:body frontend))
        region (prepare frontend scope 1)
        events (long-array 2)
        values {'events events 'x 3.0 'first-value 17.0}
        arguments #(mapv values (:parameters region))]
    (is (some? region))
    (is (= 20.0 (apply (:invoke region) (arguments))))
    (is (= [0 1] (vec events)) "does not replay the earlier initializer")
    (is (= 20.0 (apply (:invoke region) (arguments))))
    (is (= [0 2] (vec events)) "executes the selected effect exactly once per invocation")
    (is (thrown? NullPointerException
                 (apply (:invoke region) (mapv (assoc values 'events nil) (:parameters region)))))
    (is (thrown? ArrayIndexOutOfBoundsException
                 (apply (:invoke region) (mapv (assoc values 'events (long-array 1))
                                               (:parameters region)))))))

(deftest binding-region-preserves-the-original-reference-check
  (reference-binding-checked "ready" 0)
  (let [frontend (captured #'reference-binding-checked) scope (first (:body frontend))
        region (prepare frontend scope 0)]
    (is (= String (get-in region [:binding-context :after :reference-class])))
    (is (= String (.getReturnType ^java.lang.reflect.Method (:method region))))
    (let [text (String. "ready")]
      (is (identical? text ((:invoke region) text))))
    (is (nil? ((:invoke region) nil)))
    (is (thrown? ClassCastException ((:invoke region) 3)))
    (is (thrown? ClassCastException (reference-binding-checked 3 0)))))

(deftest binding-region-retains-raw-source-history-after-preparation
  (let [frontend (captured) scope (first (:body frontend))
        region (prepare frontend scope 0)
        source (get-in frontend [:source-entry :source-var]) metadata (meta source)
        body (:raster.core/deftm-source-body metadata)]
    (try
      (alter-meta! source assoc :raster.core/deftm-source-body
                   (with-meta (mapv identity body) (meta body)))
      (is (= :stale-definition-proof-input (reason #((:guard region)))))
      (is (= :stale-definition-proof-input (reason #((:invoke region) 3.0 2.0))))
      (finally (reset-meta! source metadata)))
    (is (= 6.0 ((:invoke region) 3.0 2.0)))))

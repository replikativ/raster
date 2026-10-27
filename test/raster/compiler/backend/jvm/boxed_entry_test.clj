(ns raster.compiler.backend.jvm.boxed-entry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]))

(defn- emit [parameters tags body]
  (let [interface (types/ensure-fn-interface! tags)]
    (let [emitted (core/emit-owned-bytecode!
                   bytecode/compile-typed-impl! "raster.compiled.BoxedEntryTest"
                   parameters tags 'Object body
                   (the-ns 'raster.compiler.backend.jvm.boxed-entry-test)
                   (:iface-name interface) 'boxed-entry-test)]
      (assoc emitted :interface-class (:iface-class interface)))))

(defn- project [original arguments caller-parameters]
  (let [input-body (:input-body original)
        plan (generated/instantiate-boxed-entry-inputs original
                                                     (get-in original [:emission :params]) arguments)
        body [(list 'let* (vec (mapcat identity (:bindings plan)))
                    (util/subst-syms (:substitutions plan) (first input-body)))]
        emitted (emit caller-parameters (vec (repeat (count caller-parameters) 'Object)) body)]
    {:plan plan :function (:implementation emitted) :emission (:emission emitted) :input-body body}))

(defn- outcome [f]
  (try [:value (f)] (catch Throwable error [:error (class error)])))

(deftest boxed-completion-uses-the-same-parameter-helper-as-public-invoke
  (doseq [[tag samples]
          [['double [2.5 (float -0.0) Double/NaN nil true "wrong"]]
           ['float [Double/MAX_VALUE -0.0 (float 2.5) nil true]]
           ['long [2.9 Double/POSITIVE_INFINITY (bigint 9) nil true]]
           ['int [2.9 Long/MAX_VALUE Double/NaN nil true]]
           ['boolean [true false nil 1]]
           ['byte [129 2.9 nil true]]
           ['short [32768 2.9 nil true]]
           ['char [\a nil 97]]
           ['Object [nil true "same"]]
           ['floats [(float-array [1.0]) nil (double-array [1.0])]]]]
    (let [body ['x]
          emitted (emit '[x] [tag] body)
          original (assoc emitted :input-body (get-in emitted [:emission :body]))
          {:keys [function]} (project original '[value] '[value])
          baseline (:implementation original)]
      (is (some? function) (str tag))
      (doseq [sample samples]
        (let [left (outcome #(baseline sample)) right (outcome #(function sample))]
          ;; NaNs and signed zero need representation checks, not numeric =.
          (if (and (= :value (first left)) (number? (second left)))
            (do
              (is (= (class (second left)) (class (second right))) (str tag))
              (is (= (Double/doubleToRawLongBits (double (second left)))
                     (Double/doubleToRawLongBits (double (second right)))) (str tag)))
            (do (is (= (first left) (first right)) (str tag))
                (if (= :error (first left))
                  (is (= (second left) (second right)) (str tag))
                  (is (identical? (second left) (second right)) (str tag))))))))))

(deftest all-operands-evaluate-before-the-first-wrapper-conversion
  (let [body ['[x y]]
        emitted (emit '[x y] '[int double] body)
        original (assoc emitted :input-body (get-in emitted [:emission :body]))
        arguments '[(first-draw) (second-draw)]
        {:keys [function plan emission input-body]}
        (project original arguments '[first-draw second-draw])
        baseline (:implementation original)
        events (atom [])
        number (proxy [Number] []
                 (intValue [] (swap! events conj :int-value) 7)
                 (longValue [] (swap! events conj :long-value) 7)
                 (floatValue [] (swap! events conj :float-value) (float 7))
                 (doubleValue [] (swap! events conj :double-value) 7.0))
        first-draw #(do (swap! events conj :first-argument) number)
        second-draw #(do (swap! events conj :second-argument) number)]
    (is (= '[x y] (mapv :formal (:slots plan))))
    (is (= [0 1] (mapv :ordinal (:slots plan))))
    (is (= [Integer/TYPE Double/TYPE] (mapv :parameter-class (:slots plan))))
    (is (every? true? (map identical? arguments (map :original-argument (:slots plan)))))
    (is (= (:bindings plan) (into (:object-bindings plan) (:input-bindings plan))))
    (is (every? #(nil? (meta %)) (mapcat (juxt :object-binder :input-binder) (:slots plan))))
    (doseq [[_ node] (:bindings plan)]
      (let [observed (bytecode/retained-input-completion-context emission input-body node)]
        (is (some? observed))
        (is (identical? node (:source-input observed)))
        (is (= (second node) (:contract observed)))
        (is (nil? (bytecode/retained-input-completion-context
                   emission input-body (with-meta (apply list node) (meta node)))))))
    (is (= [7 7.0] (function first-draw second-draw)))
    (is (= [:first-argument :second-argument :int-value :double-value] @events))
    (reset! events [])
    (is (= [7 7.0] (baseline (first-draw) (second-draw))))
    (is (= [:first-argument :second-argument :int-value :double-value] @events))
    (let [failure (ex-info "second argument failed" {})]
      (is (identical? failure
                      (try (function (constantly nil) #(throw failure)) nil
                           (catch Throwable error error))))
      (is (= [:error ClassCastException] (outcome #(function (constantly true) (constantly 1.0)))))
      (is (= [:error NullPointerException] (outcome #(function (constantly nil) (constantly 1.0))))))))

(deftest wide-apply-to-preserves-the-same-entry-completions
  (let [parameters (mapv #(symbol (str "x" %)) (range 21))
        body [parameters]
        emitted (emit parameters (vec (repeat 21 'double)) body)
        original (assoc emitted :input-body (get-in emitted [:emission :body]))
        bridge (bytecode/retained-boxed-entry-bridge (:emission original))
        {:keys [function]} (project original parameters parameters)
        values (vec (repeat 21 (float 2.5)))]
    (is (some? bridge))
    (is (nil? (get-in bridge [:boxed-entry :invoke-method])))
    (is (= (apply (:implementation original) values) (apply function values)))
    (is (every? #(instance? Double %) (apply function values)))
    (is (= [:error NullPointerException] (outcome #(apply function (assoc values 20 nil)))))
    (is (= [:error clojure.lang.ArityException] (outcome #(apply function (pop values)))))))

(deftest bridge-and-input-plan-refuse-mismatched-evidence
  (let [body ['x] emitted (emit '[x] '[double] body)
        original (assoc emitted :input-body (get-in emitted [:emission :body]))
        emission (:emission original)
        parameters (:params emission)
        foreign (emit '[x] '[double] body)
        foreign-bridge (bytecode/retained-boxed-entry-bridge (:emission foreign))]
    (doseq [[field value] [[:plans []] [:apply-to-method nil] [:invoke-method nil]
                           [:apply-to-method (get-in foreign-bridge [:boxed-entry :apply-to-method])]
                           [:invoke-method (get-in foreign-bridge [:boxed-entry :invoke-method])]
                           [:parameter-carriers [Float/TYPE]]]]
      (let [changed (update emission :typed-return-bridges
                            (fn [bridges] (mapv #(assoc-in % [:boxed-entry field] value) bridges)))]
        (is (nil? (bytecode/retained-boxed-entry-bridge changed)))
        (is (thrown? clojure.lang.ExceptionInfo
                     (generated/instantiate-boxed-entry-inputs
                      (assoc original :emission changed) parameters '[value])))))
    (doseq [[parameters arguments] [['[other] '[value]] ['[x] []]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (generated/instantiate-boxed-entry-inputs original parameters arguments))))
    (doseq [changed [(assoc original :input-body (mapv identity (:input-body original)))
                     (assoc original :implementation identity)
                     (assoc original :interface-class Object)]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (generated/instantiate-boxed-entry-inputs changed parameters '[value]))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (generated/instantiate-boxed-entry-inputs original (mapv identity parameters) '[value])))
    (let [plan (generated/instantiate-boxed-entry-inputs original parameters '[value])]
      (is (every? completion/canonical-form? (map second (:bindings plan)))))))

(ns raster.compiler.backend.jvm.arithmetic-emission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.types :as types]))

(defn- emit [tags node]
  (let [parameters (vec (take (count tags) '[x y events]))
        body [node]
        interface (types/ensure-fn-interface! tags)]
    (assoc (core/emit-owned-bytecode!
            bytecode/compile-typed-impl! "raster.compiled.ArithmeticObservation"
            parameters tags 'Object body
            (the-ns 'raster.compiler.backend.jvm.arithmetic-emission-test) (:iface-name interface)
            'arithmetic-observation)
           :input-body body)))

(defn- observed [emission]
  (keep #(bytecode/retained-arithmetic emission %)
        (vals (:invocation-catalog emission))))

(defn- original-arithmetic-occurrence [emission]
  ;; The record's :expression is the actual post-macroexpansion emitter node.
  ;; Only the catalog node is an ORIGINAL query key; the retained token joins
  ;; the two. Never substitute equal syntax or the physical node for that key.
  (first (filter #(bytecode/retained-arithmetic emission %)
                 (vals (:invocation-catalog emission)))))

(deftest arithmetic-observes-actual-operand-completions-and-policy
  (doseq [[tags operator raw completed conversions overflow]
          [['[int int] "+" [:int :int] [:int :int] [] :wrap]
           ['[long long] "*" [:long :long] [:long :long] [] :trap]
           ['[long long] "/" [:long :long] [:double :double] [1 0] :ieee-754]
           ['[float int] "+" [:float :int] [:double :double] [1 0] :ieee-754]
           ['[int float] "+" [:int :float] [:float :float] [0] :ieee-754]
           ['[float float] "/" [:float :float] [:float :float] [] :ieee-754]]]
    (let [node (list (symbol "clojure.core" operator) 'x 'y)
          {:keys [emission input-body]} (emit tags node)
          records (vec (observed emission))
          record (first records)]
      (is (= 1 (count records)))
      (is (= operator (:operator record)))
      (is (= raw (mapv :emitted-type (:input-completions record))))
      (is (= completed (mapv :completed-type (:input-completions record))))
      (is (= (first completed) (:result-carrier record) (get-in record [:policy :carrier])))
      (is (= overflow (get-in record [:policy :overflow])))
      (is (= [0 1] (:evaluation-order record)))
      (is (= conversions (:completion-order record)))
      (doseq [[id input completion carrier] (map vector '[x y] (:inputs record)
                                                 (:input-completions record) raw)]
        (is (= id (:operand input)))
        (is (identical? (:operand input) (:operand completion)))
        (is (= carrier (get-in input [:ports id :type])))
        (is (map? (:context input))))
      (is (identical? node (:source-input
                           (bytecode/retained-input-arithmetic emission input-body node))))
      (is (not-any? #(contains? record %) [:pure? :total? :replayable? :range :nonthrowing])))))

(deftest nonbinary-and-generic-paths-grant-no-primitive-observation
  (doseq [[tags node] [['[Object Object] '(clojure.core/+ x y)]
                      ['[double Object] '(clojure.core/+ x y)]
                      ['[long long] '(clojure.core/- x)]
                      ['[long long] '(clojure.core/+ x y 1)]
                      ['[long long] '(clojure.core/unchecked-add x y)]
                      ['[long long] '(clojure.core/inc x)]
                      ['[long long] '(let* [f clojure.core/+] (f x y))]]]
    (let [{:keys [emission input-body]} (emit tags node)]
      (is (empty? (observed emission)) (pr-str node))
      (is (nil? (bytecode/retained-input-arithmetic emission input-body node))))))

(deftest primitive-instruction-observation-is-not-an-operand-effect-proof
  (let [node '(clojure.core/+
               (do (clojure.core/aset events 0 11) x)
               (do (clojure.core/aset events 1 22) y))
        {:keys [emission implementation input-body]} (emit '[long long longs] node)
        record (bytecode/retained-input-arithmetic emission input-body node)
        events (long-array 2)]
    (is (some? record))
    (is (= [#{'events 'x} #{'events 'y}]
           (mapv #(set (keys (:ports %))) (:inputs record))))
    (is (= :trap (get-in record [:policy :overflow])))
    (is (thrown? ArithmeticException (implementation Long/MAX_VALUE 1 events)))
    (is (= [11 22] (vec events)) "both original operands precede the arithmetic trap")
    (is (not (contains? record :pure?)))))

(deftest arithmetic-occurrences-refuse-copies-foreign-owners-and-repeated-emission
  (let [node '(clojure.core/+ x y)
        {:keys [emission input-body]} (emit '[long long] node)
        other (:emission (emit '[long long] node))
        original (original-arithmetic-occurrence emission)
        copy (with-meta (apply list original) (meta original))
        inputs (atom {:static [] :values []})]
    (is (some? (bytecode/retained-arithmetic emission original)))
    (is (nil? (bytecode/retained-arithmetic emission copy)))
    (is (nil? (bytecode/retained-arithmetic other original)))
    (is (nil? (bytecode/retained-input-arithmetic emission (mapv identity input-body) node)))
    (is (nil? (bytecode/retained-input-arithmetic emission input-body (apply list node))))
    ;; Actual duplicate emission, not a forged :ambiguous record.
    (binding [entry/*emission-inputs* inputs]
      (bytecode/compile-specialized-class!
       "raster.compiled.RepeatedArithmeticObservation"
       [{:name 'run :params [(with-meta 'x {:tag 'long}) (with-meta 'y {:tag 'long})]
         :return-tag 'Object :source-ns (the-ns 'raster.compiler.backend.jvm.arithmetic-emission-test)
         :walked-body [(list 'do original original)]}]))
    (let [records (vals (:arithmetic-contexts @inputs))]
      (is (= 1 (count records)))
      (is (= :ambiguous (:status (first records))))
      (is (= :repeated-emitted-arithmetic-context (:reason (first records)))))))

(def ^:dynamic arithmetic-offset 3)

(deftest arithmetic-retains-original-dependency-history
  (let [node '(clojure.core/+ x (clojure.core/long arithmetic-offset))
        {:keys [emission]} (emit '[long long] node)
        original (original-arithmetic-occurrence emission)]
    (is (some? original))
    (with-redefs [arithmetic-offset 5]
      (is (thrown? clojure.lang.ExceptionInfo
                   (bytecode/retained-arithmetic emission original))))))

(deftest frontend-arithmetic-authenticates-module-before-exposing-evidence
  (let [scope (create-ns (gensym "raster.arithmetic-observation-owner."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm add [x :- Double y :- Double] :- Double
                 (clojure.core/+ x y))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'add))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))]
        (core/ensure-current-owned-bytecode! receiver)
        (entry/call-with-definition-inputs
         (fn []
           (let [frontend (entry/original-source-frontend source)
                 node (first (:body frontend))]
             (with-redefs [core/ensure-current-owned-bytecode!
                           (fn [& _] (throw (AssertionError. "query must not prepare source")))
                           core/ensure-bytecode-implementation!
                           (fn [& _] (throw (AssertionError. "query must not force JIT")))]
               (let [record (bytecode/observed-frontend-arithmetic frontend node)]
                 (is (some? record))
                 (is (identical? node (:source-input record)))
                 (is (= :ieee-754 (get-in record [:policy :overflow]))))
               (doseq [foreign [(into {} (:emission frontend))
                                (assoc (:emission frontend) :arithmetic-contexts {})]]
                 (is (nil? (bytecode/observed-frontend-arithmetic
                            (assoc frontend :emission foreign) node))))
               (is (nil? (bytecode/observed-frontend-arithmetic frontend (apply list node)))))
             (with-redefs-fn {source (fn [& _] 0.0)}
               #(is (thrown? clojure.lang.ExceptionInfo
                             (bytecode/observed-frontend-arithmetic frontend node))))))))
      (finally (remove-ns (ns-name scope))))))

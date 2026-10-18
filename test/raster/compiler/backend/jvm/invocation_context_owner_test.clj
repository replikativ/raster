(ns raster.compiler.backend.jvm.invocation-context-owner-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]))

(deftm scalar-helper [x :- Double] :- Double (n/* x x))
(deftm duplicate-calls [x :- Double] :- Double
  (n/+ (scalar-helper x) (scalar-helper x)))
(deftm shadowed-calls [x :- Double] :- Double
  (let [left (scalar-helper x)
        x (n/+ x 1.0)
        right (scalar-helper x)]
    (n/+ left right)))

(defn- receiver [v]
  (let [source (core/resolve-deftm-var v)]
    (ns-resolve (.ns ^clojure.lang.Var source)
                (symbol (str (:name (meta source)) "-impl")))))

(defn- emission [v]
  (let [r (receiver v)]
    (core/ensure-bytecode-implementation! r)
    (:raster.core/bytecode-emission (meta r))))

(defn- helper-occurrences [emitted]
  (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/scalar-helper
              (:raster.op/original (meta %)))
          (vals (:invocation-catalog emitted))))

(deftest actual-emission-distinguishes-equal-call-occurrences
  (let [emitted (emission #'duplicate-calls)
        calls (vec (helper-occurrences emitted))
        records (mapv #(bytecode/retained-invocation-context emitted %) calls)]
    (is (= 18.0 (duplicate-calls 3.0)))
    (is (= 2 (count calls)))
    (is (= (first calls) (second calls)) "equal syntax is not occurrence identity")
    (is (not (identical? (first calls) (second calls))))
    (is (every? some? records))
    (doseq [record records]
      (is (= :global-typed-indy (:invocation-mode record)))
      (is (= Double/TYPE (:return-carrier record)))
      (is (= [Double/TYPE] (vec (.getParameterTypes ^java.lang.reflect.Method (:method record)))))
      (is (= [:double] (mapv :type (vals (get-in record [:inputs 0 :ports]))))))
    (is (nil? (bytecode/retained-invocation-context
               emitted (with-meta (apply list (first calls)) (meta (first calls))))))
    (is (nil? (bytecode/retained-invocation-context (emission #'shadowed-calls) (first calls))))))

(deftest actual-physical-ports-survive-shadowing-as-distinct-slots
  (let [emitted (emission #'shadowed-calls)
        records (mapv #(bytecode/retained-invocation-context emitted %) (helper-occurrences emitted))
        ports (mapv #(first (vals (get-in % [:inputs 0 :ports]))) records)]
    (is (= 25.0 (shadowed-calls 3.0)))
    (is (= 2 (count ports)))
    ;; Logical Double is not a physical ABI: the shadowing arithmetic result
    ;; is Object-carried, whereas the original parameter is primitive.
    (is (= {:double 1 :ref 1} (frequencies (map :type ports))))
    (is (= 2 (count (set (map :slot ports)))))))

(deftest annotations-are-private-fresh-and-quote-opaque
  (let [quoted '(quote (scalar-helper x))
        qualified-quoted '(clojure.core/quote (scalar-helper x))
        source [(list 'do quoted qualified-quoted '(scalar-helper x) '(scalar-helper x))]
        first-owner (bytecode/retain-invocation-occurrences source)
        second-owner (bytecode/retain-invocation-occurrences (:body first-owner))
        [first-body] (:body first-owner)]
    (is (identical? quoted (second first-body)))
    (is (identical? qualified-quoted (nth first-body 2)))
    (is (= 2 (count (:catalog first-owner))))
    (is (= 2 (count (:catalog second-owner))))
    (is (empty? (set/intersection (set (keys (:catalog first-owner)))
                                        (set (keys (:catalog second-owner))))))
    (is (= source (:body first-owner) (:body second-owner)))))

(deftest runtime-selected-local-inputs-do-not-claim-one-primitive-contract
  (let [call (first (helper-occurrences (emission #'duplicate-calls)))
        local (with-meta 'helper (meta (second call)))
        argument (nth call 2)
        owned (bytecode/retain-invocation-occurrences
               [(with-meta (list '.invk local argument) (meta call))])
        inputs (atom {:static [] :values []})
        compiled
    ;; A typed hint on a parameter does not prove that the receiver implements
    ;; the interface: emission has an IFn fallback with different conversions.
        (binding [entry/*emission-inputs* inputs]
          (bytecode/compile-specialized-class!
           "raster.compiled.GuardedInvocationContext"
           [{:name 'run :params [(with-meta local {:tag 'Object}) argument] :return-tag 'Object
             :walked-body (:body owned)
             :source-ns (the-ns 'raster.compiler.backend.jvm.invocation-context-owner-test)}]))
        method (get-in compiled [:methods 'run])]
    (is (= 4.0 (.invoke ^java.lang.reflect.Method method nil
                        (object-array [@(receiver #'scalar-helper) 2.0]))))
    (is (= 3.0 (.invoke ^java.lang.reflect.Method method nil
                        (object-array [(fn [x] (+ 1.0 x)) 2.0]))))
    (let [record (first (vals (:invocation-contexts @inputs)))]
      (is (= 1 (count (:invocation-contexts @inputs))))
      (is (= :unsupported (:status record)))
      (is (= :runtime-selected-input-context (:reason record)))
      (is (= :guarded-local-typed-interface (:invocation-mode record)))
      (is (= :ref (get-in record [:receiver-input :ports local :type])))
      (is (false? (get-in record [:receiver-input :ports local :verified]))))))

(deftest cloned-occurrence-evidence-must-agree-or-refuse
  (let [emitted (emission #'duplicate-calls)
        call (first (helper-occurrences emitted))
        ;; Reuse one occurrence intentionally; real macro expansion can clone
        ;; a source node. Retain its actual receiver hints, not reconstructed ones.
        argument (nth call 2)
        compile! (fn [body]
                   (let [inputs (atom {:static [] :values []})]
                     (binding [entry/*emission-inputs* inputs]
                       (bytecode/compile-specialized-class!
                        "raster.compiled.ClonedInvocationContext"
                        [{:name 'run :params [argument] :return-tag 'Object
                          :walked-body [body]
                          :source-ns (the-ns 'raster.compiler.backend.jvm.invocation-context-owner-test)}]))
                     (vals (:invocation-contexts @inputs))))]
    (let [records (compile! (list 'let* ['a call 'b call] 'b))]
      (is (= 1 (count records)))
      (is (= :ambiguous (:status (first records)))))
    (let [records (compile! (list 'let* ['a call] [call]))]
      (is (= 1 (count records)))
      (is (= :ambiguous (:status (first records))))
      (is (= :repeated-emitted-invocation-context (:reason (first records)))))))

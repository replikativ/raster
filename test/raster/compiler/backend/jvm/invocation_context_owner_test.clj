(ns raster.compiler.backend.jvm.invocation-context-owner-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [clojure.set :as set]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.method-entry :as entry]
            [raster.core :as core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]))

(deftm scalar-helper [x :- Double] :- Double (n/* x x))
(deftm duplicate-calls [x :- Double] :- Double
  (n/+ (scalar-helper x) (scalar-helper x)))
(deftm shadowed-calls [x :- Double] :- Double
  (let [left (scalar-helper x)
        x (n/+ x 1.0)
        right (scalar-helper x)]
    (n/+ left right)))
(deftm float-helper [x :- Float] :- Float x)
(deftm narrowing-call [x :- Double] :- Double (float-helper (float x)))
(deftm array-input [xs :- (Array double) index :- Long] :- Double
  (scalar-helper (arrays/aget xs index)))
(deftm let-array-input [xs :- (Array double) index :- Long] :- Double
  (let [copy xs] (scalar-helper (arrays/aget copy index))))

(use-fixtures :once
  (fn [run]
    ;; Materialize shared numerical helper dependencies before capturing caller
    ;; emissions. A later unrelated lazy JIT is a real historical invalidation,
    ;; not permission to refresh an already captured owner's certificate.
    (scalar-helper 1.0)
    (n/+ 1.0 1.0)
    (run)))

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

(deftest outlined-inputs-preserve-the-original-physical-lexical-abi
  (let [emitted (emission #'shadowed-calls)]
    (doseq [call (helper-occurrences emitted)]
      (let [region (bytecode/compile-invocation-input-region emitted call 0)
            original-port (first (vals (:physical-ports region)))
            parameter-class (first (.getParameterTypes ^java.lang.reflect.Method (:method region)))
            run (:invoke region)]
        (is (= (if (= :double (:type original-port)) Double/TYPE Object) parameter-class))
        (is (= (:physical-ports region) (get-in region [:spec ::bytecode/input-port-layout])))
        (doseq [value [1.5 -0.0 Double/POSITIVE_INFINITY Double/NaN]]
          (is (= (Double/doubleToRawLongBits value)
                 (Double/doubleToRawLongBits (run value)))))
        (when (= :ref (:type original-port))
          (is (thrown? NullPointerException (run nil)))
          (is (thrown? ClassCastException (run "not a number"))))))))

(deftest outlined-float-input-keeps-narrowing-and-original-operand
  (let [emitted (emission #'narrowing-call)
        call (first (filter #(= 'raster.compiler.backend.jvm.invocation-context-owner-test/float-helper
                                (:raster.op/original (meta %)))
                            (vals (:invocation-catalog emitted))))
        region (bytecode/compile-invocation-input-region emitted call 0)
        record (bytecode/retained-invocation-context emitted call)]
    (is (identical? (get-in record [:inputs 0 :operand])
                    (get-in region [:input :operand])))
    (is (= Float/TYPE (.getReturnType ^java.lang.reflect.Method (:method region))))
    (doseq [value [0.1 -0.0 Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN]]
      (is (= (Double/doubleToRawLongBits (narrowing-call value))
             (Double/doubleToRawLongBits (double ((:invoke region) value))))))))

(deftest outlining-refuses-copied-occurrences-and-stale-dependencies
  (let [emitted (emission #'duplicate-calls)
        call (first (helper-occurrences emitted))
        region (bytecode/compile-invocation-input-region emitted call 0)]
    (doseq [[node index] [[(with-meta (apply list call) (meta call)) 0]
                         [call -1] [call 1]]]
      (is (= :unowned-bytecode-input-region
             (try (bytecode/compile-invocation-input-region emitted node index)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (with-redefs-fn {(:var (first (:dependency-snapshots emitted))) (fn [& _] :stale)}
      #(is (thrown? clojure.lang.ExceptionInfo ((:invoke region) 2.0))))
    (is (= 2.0 ((:invoke region) 2.0)))))

(deftest outlined-array-input-restores-an-actual-verifier-carrier
  (doseq [v [#'array-input #'let-array-input]]
    (let [emitted (emission v)
          call (first (helper-occurrences emitted))
          region (bytecode/compile-invocation-input-region emitted call 0)
          classes (vec (.getParameterTypes ^java.lang.reflect.Method (:method region)))
          array-port (first (keep (fn [[id port]]
                                    (when (= :ref (:type port)) id)) (:physical-ports region)))
          args (fn [xs i] (mapv #(if (= array-port %) xs i) (:parameters region)))]
      (is (some #{(class (double-array 0))} classes))
      (is (= (class (double-array 0)) (get-in region [:physical-ports array-port :reference-class])))
      (is (= 7.5 (apply (:invoke region) (args (double-array [2.0 7.5]) 1))))
      (is (thrown? NullPointerException (apply (:invoke region) (args nil 0))))
      (is (thrown? ArrayIndexOutOfBoundsException
                   (apply (:invoke region) (args (double-array 0) 0))))
      (let [weakened (update (:spec region) :params
                             #(mapv (fn [p] (if (= p array-port) (vary-meta p assoc :tag 'Object) p)) %))]
        (is (= :bytecode-input-reference-carrier
               (try (bytecode/compile-specialized-class! "raster.compiled.WeakenedInput" [weakened])
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

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

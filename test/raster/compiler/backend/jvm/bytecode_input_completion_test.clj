(ns raster.compiler.backend.jvm.bytecode-input-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.value-completion :as completion]
            [raster.core :as core :refer [deftm]]))

(deftm retained-helper [x :- Double] :- Double x)
(deftm nullable-helper [x :- Double] :- Double (if (> x 0.0) x nil))
(deftm retained-caller [x :- Double] :- Double (retained-helper x))
(deftm nullable-caller [x :- Double] :- Double (nullable-helper x))

(defn- typed-helper-call [caller argument]
  (let [source (core/resolve-deftm-var caller)
        body (core/ensure-walked-body! source)
        call (first (filter #(and (seq? %) (= '.invk (first %))) (tree-seq coll? seq body)))]
    (assert call "test requires an actual walked typed invocation")
    (with-meta (list (first call) (second call) argument) (meta call))))

(defn- compiled
  ([params body] (compiled params body 'Object))
  ([params body return-tag]
  (let [{:keys [methods]}
        (bytecode/compile-specialized-class!
         "raster.compiled.InputCompletion"
         [{:name 'run :params params :return-tag return-tag
           :walked-body [body]
           :source-ns (the-ns 'raster.compiler.backend.jvm.bytecode-input-completion-test)}])
        method (get methods 'run)]
    (fn [& arguments]
      (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
           (catch java.lang.reflect.InvocationTargetException e (throw (.getCause e))))))))

(defn- outcome [f & arguments]
  (try {:value (apply f arguments)}
       (catch Throwable e {:error (class e)})))

(defn- input-contract [carrier]
  ;; These original methods return Object. The primitive-method discriminator
  ;; below retains its own distinct producer context explicitly.
  (completion/bytecode-floating-input-contract
   carrier (completion/bytecode-operand-context {:preserve-source-call-results? true})))

(deftest contexts-are-captured-from-the-original-input-producer
  (doseq [[return-tag expected] [['Object {:object-result? false :object-call-result? false
                                          :preserve-source-call-results? true :void-context false}]
                                ['double {:object-result? false :object-call-result? false
                                          :preserve-source-call-results? false :void-context false}]]]
    (let [seen (atom []) original @#'bytecode/emit-argument-value]
      (with-redefs-fn {#'bytecode/emit-argument-value
                      (fn [code expression carrier locals context]
                        (when (and (= 'input expression) (= Double/TYPE carrier))
                          (swap! seen conj (completion/bytecode-operand-context context)))
                        (original code expression carrier locals context))}
        #(compiled ['input] '(java.lang.Math/copySign input 1.0) return-tag))
      (is (= [expected] @seen)))))

(deftest original-bytecode-floating-input-and-retained-completion-agree
  (doseq [[carrier bits] [[Double/TYPE 'java.lang.Double/doubleToRawLongBits]
                          [Float/TYPE 'java.lang.Float/floatToRawIntBits]]]
    (let [contract (input-contract carrier)
          original (compiled ['input] (list bits 'input))
          retained (compiled ['input]
                             (list 'let* ['completed (completion/complete-value contract 'input)]
                                   (list bits 'completed)))]
      (doseq [input [nil false "not a number" (Object.)
                     7 (long 16777217) 0.1M -0.0 Double/NaN
                     Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/MAX_VALUE
                     (Float/valueOf (Float/intBitsToFloat 2143289410))]]
        (is (= (outcome original input) (outcome retained input))))
      ;; Physical primitive operands must use the same conversion instructions,
      ;; not be boxed first and reinterpreted by a source compiler.
      (doseq [tag ['double 'float 'long]]
        (let [parameter (with-meta 'input {:tag tag})
              a (compiled [parameter] (list bits 'input))
              b (compiled [parameter]
                          (list bits (completion/complete-value contract 'input)))]
          (doseq [input (case tag
                         double [-0.0 0.1 Double/MAX_VALUE Double/NaN]
                         float [(float -0.0) (float 0.1) Float/MAX_VALUE Float/NaN]
                         long [0 16777217 Long/MAX_VALUE])]
            (is (= (outcome a input) (outcome b input)))))))))

(deftest retained-conversion-happens-before-the-next-operand
  (let [first-input (completion/complete-value
                     (input-contract Double/TYPE) 'input)
        original (compiled ['input 'later] '(java.lang.Math/copySign input (clojure.core/aget later 0)))
        retained (compiled ['input 'later]
                           (list 'let* ['completed first-input]
                                 '(java.lang.Math/copySign completed (clojure.core/aget later 0))))]
    (doseq [input [nil false 2.0]]
      (is (= (outcome original input (double-array 0))
             (outcome retained input (double-array 0)))))
    (is (= {:error NullPointerException} (outcome retained nil (double-array 0))))
    (is (= {:error ArrayIndexOutOfBoundsException} (outcome retained 2.0 (double-array 0))))))

(deftest overload-selection-uses-the-completed-physical-carrier
  (doseq [[carrier cast] [[Double/TYPE 'clojure.core/double] [Float/TYPE 'clojure.core/float]]]
    (let [term (completion/complete-value (input-contract carrier) 'input)
          original (compiled ['input] (list 'java.lang.Math/max (list cast 'input) '(long 0)))
          direct (compiled ['input] (list 'java.lang.Math/max term '(long 0)))
          bound (compiled ['input] (list 'let* ['completed term] '(java.lang.Math/max completed (long 0))))]
      (doseq [input [0.5 1.5 -0.5]]
        (is (= (outcome original input) (outcome direct input) (outcome bound input)))))))

(deftest nested-helper-inputs-retain-the-original-conversion-context
  (let [operand (list 'clojure.core// (typed-helper-call #'retained-caller 'numerator)
                      (typed-helper-call #'retained-caller 'denominator))
        term (completion/complete-value
              (input-contract Double/TYPE) operand)
        original (compiled ['numerator 'denominator]
                           (list 'java.lang.Double/doubleToRawLongBits operand))
        bound (compiled ['numerator 'denominator]
                        (list 'let* ['completed term] '(java.lang.Double/doubleToRawLongBits completed)))
        object-result (compiled ['numerator 'denominator] [term])]
    (doseq [[numerator denominator] [[1.0 2.0] [1.0 0.0] [0.0 0.0] [nil 1.0]]]
      (let [expected (outcome original numerator denominator)
            materialized (outcome object-result numerator denominator)
            materialized-bits (if (:error materialized) materialized
                               {:value (Double/doubleToRawLongBits (first (:value materialized)))})]
        (is (= expected (outcome bound numerator denominator)))
        (is (= expected materialized-bits))))))

(deftest nested-nullable-helper-does-not-move-later-effects
  (let [operand (list 'clojure.core// (typed-helper-call #'nullable-caller 'numerator)
                      (list 'do '(swap! events conj :later) 1.0))
        term (completion/complete-value
              (input-contract Double/TYPE) operand)
        original (compiled ['numerator 'events] (list 'java.lang.Double/doubleToRawLongBits operand))
        retained (compiled ['numerator 'events] [term])]
    (doseq [input [-1.0 1.0]]
      (let [a (atom []) b (atom [])
            expected (outcome original input a)
            actual (outcome retained input b)]
        (is (= (:error expected) (:error actual)))
        (is (= @a @b))))))

(deftest object-consumption-does-not-change-a-primitive-call-input-region
  (let [params (mapv #(with-meta % {:tag 'double}) '[numerator denominator])
        left (typed-helper-call #'retained-caller 'numerator)
        right (typed-helper-call #'retained-caller 'denominator)
        operand (list 'clojure.core// left right)
        term (completion/complete-value
              (completion/bytecode-floating-input-contract Double/TYPE
               (completion/bytecode-operand-context {})) operand)
        original (compiled params (list 'java.lang.Math/copySign operand 1.0) 'double)
        retained (compiled params (list 'java.lang.Math/copySign (list 'clojure.core/nth [term] 0) 1.0) 'double)]
    (is (= 'double (:raster.type/ret-tag (meta (second left)))))
    (doseq [[n d] [[1.0 2.0] [1.0 0.0] [0.0 0.0]]]
      (let [a (outcome original n d) b (outcome retained n d)]
        (is (= (if (and (:value a) (Double/isNaN (:value a))) :nan a)
               (if (and (:value b) (Double/isNaN (:value b))) :nan b)))))))

(deftest source-and-bytecode-conversion-authorities-are-not-interchangeable
  (let [bytecode-term (completion/complete-value
                       (input-contract Float/TYPE) 'input)
        source-term (completion/complete-value
                     (completion/primitive-input-contract Float/TYPE {:unchecked-math false}) 'input)]
    (is (= :unsupported-value-completion-source-context
           (try (completion/project-to-jvm-source bytecode-term {:unchecked-math false})
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :unsupported-jvm-value-completion
           (try (compiled ['input] source-term)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :unsupported-value-completion
           (try (input-contract Long/TYPE)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [original-source (binding [*unchecked-math* false]
                            (eval '(fn [input] (java.lang.Float/floatToRawIntBits input))))
          original-bytecode (compiled ['input] '(java.lang.Float/floatToRawIntBits input))
          retained-bytecode (compiled ['input]
                                      (list 'java.lang.Float/floatToRawIntBits bytecode-term))]
      (is (= {:error IllegalArgumentException} (outcome original-source Double/MAX_VALUE)))
      (is (= {:value (Float/floatToRawIntBits Float/POSITIVE_INFINITY)}
             (outcome original-bytecode Double/MAX_VALUE)
             (outcome retained-bytecode Double/MAX_VALUE))))))

(deftest void-and-diverging-operands-do-not-invent-a-stack-value
  (let [contract (input-contract Double/TYPE)]
    ;; Java void calls are already materialized as nil by the original emitter;
    ;; the conversion must trap on that nil, not invent a primitive value.
    (let [original (compiled [] '(java.lang.Double/doubleToRawLongBits (java.lang.Thread/yield)))
          retained (compiled [] (completion/complete-value contract '(java.lang.Thread/yield)))]
      (is (= {:error NullPointerException} (outcome original) (outcome retained))))
    (is (= :void-bytecode-input-completion
           (try (with-redefs-fn {#'bytecode/emit-argument-value (fn [& _] :void)}
                  #(compiled [] (completion/complete-value contract 'input)))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [f (compiled ['failure] (completion/complete-value contract '(throw failure)))]
      (is (= {:error IllegalStateException} (outcome f (IllegalStateException. "stop")))))))

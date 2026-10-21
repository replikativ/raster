(ns raster.compiler.backend.jvm.bytecode-input-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.types :as types]
            [raster.core :as core :refer [deftm]]))

(deftm retained-helper [x :- Double] :- Double x)
(deftm nullable-helper [x :- Double] :- Double (if (> x 0.0) x nil))
(deftm retained-caller [x :- Double] :- Double (retained-helper x))
(deftm nullable-caller [x :- Double] :- Double (nullable-helper x))

(defonce input-calls (atom 0))
(defn input [value]
  (swap! input-calls inc)
  (+ (double value) 1.0))

(defn- typed-helper-call [caller argument]
  (let [source (core/resolve-deftm-var caller)
        body (core/ensure-walked-body! source)
        call (first (filter #(and (seq? %) (= '.invk (first %))) (tree-seq coll? seq body)))]
    (assert call "test requires an actual walked typed invocation")
    (with-meta (list (first call) (second call) argument) (meta call))))

(deftest outlined-input-does-not-capture-a-global-input-function
  (let [body (typed-helper-call #'retained-caller '(input value))
        interface (types/ensure-fn-interface! '[double])
        {:keys [implementation emission]}
        (#'core/emit-owned-bytecode!
         bytecode/compile-typed-impl! "raster.compiled.GlobalInputOperand"
         '[value] '[double] 'double [body]
         (the-ns 'raster.compiler.backend.jvm.bytecode-input-completion-test)
         (:iface-name interface) 'global-input-operand)
        original (first (:body emission))
        before @input-calls
        region (bytecode/compile-invocation-input-region emission original 0)
        method-name (.getName ^java.lang.reflect.Method (:method region))]
    (is (= before @input-calls) "outlining does not execute the operand")
    (is (not= "input" method-name))
    (is (= (str (get-in region [:spec :name])) method-name))
    ;; Do not run the old recursively self-capturing method after it fails the
    ;; name assertion: the failure is already established without stack growth.
    (when (not= "input" method-name)
      (doseq [value [-0.0 2.0 -3.5 Double/POSITIVE_INFINITY Double/NaN]]
        (let [before @input-calls
              expected (implementation value)
              actual ((:invoke region) value)]
          (is (= (+ before 2) @input-calls))
          (is (= (Double/doubleToRawLongBits expected) (Double/doubleToRawLongBits actual))))))))

(deftest outlined-method-name-avoids-qualified-and-bound-source-names
  (let [choices (atom ['taken 'bound 'fresh])
        allocate (fn [& _] (let [choice (first @choices)] (swap! choices subvec 1) choice))
        chosen (with-redefs [clojure.core/gensym allocate]
                 (#'bytecode/fresh-input-method-name
                  {:params '[parameter]
                   :walked-body '[(let* [bound 1] (other/taken bound))]}))]
    (is (= 'fresh chosen))
    (is (empty? @choices))))

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

(deftest mixed-floating-comparisons-retain-source-order
  (doseq [integer-tag ['int 'long]]
    (let [params [(with-meta 'i {:tag integer-tag}) (with-meta 'f {:tag 'float})]
          less (compiled params '(clojure.core/< f i))
          greater (compiled params '(clojure.core/> i f))
          integer (if (= integer-tag 'int) (int 16777217) (long 16777217))]
      (is (= (< (float 16777216) integer) (less integer (float 16777216))))
      (is (= (> integer (float 16777216)) (greater integer (float 16777216))))
      (is (false? (less integer Float/NaN)))
      (is (false? (greater integer Float/NaN)))
      (is (false? (less integer Float/POSITIVE_INFINITY)))
      (is (true? (greater integer Float/NEGATIVE_INFINITY)))))
  (let [params [(with-meta 'a {:tag 'float}) (with-meta 'b {:tag 'float})]
        less (compiled params '(clojure.core/< a b))]
    (doseq [[a b] [[(float -0.0) (float 0.0)]
                   [Float/NEGATIVE_INFINITY Float/POSITIVE_INFINITY]
                   [Float/NaN (float 0.0)] [(float 0.0) Float/NaN]]]
      (is (= (< a b) (less a b)))))
  (let [params [(with-meta 'i {:tag 'int}) (with-meta 'wide {:tag 'long})]
        less (compiled params '(clojure.core/< i wide))
        greater (compiled params '(clojure.core/> wide i))]
    (is (true? (less (int 1) (long 9007199254740993))))
    (is (true? (greater (int 1) (long 9007199254740993)))))
  (let [params [(with-meta 'a {:tag 'long}) (with-meta 'b {:tag 'long})]
        less (compiled params '(clojure.core/< a b))
        greater (compiled params '(clojure.core/> b a))]
    ;; Adjacent Longs collapse to the same Double; unlike int1 vs wide Long,
    ;; this actually detects accidental integral-to-floating promotion.
    (is (true? (less 9007199254740992 9007199254740993)))
    (is (true? (greater 9007199254740992 9007199254740993)))))

(defn- outcome [f & arguments]
  (try {:value (apply f arguments)}
       (catch Throwable e {:error (class e)})))

(defn- input-contract [carrier]
  ;; These original methods return Object. The primitive-method discriminator
  ;; below retains its own distinct producer context explicitly.
  (completion/bytecode-floating-input-contract
   carrier (completion/bytecode-operand-context {:preserve-source-call-results? true})))

(deftest outlined-compound-input-is-an-actual-bytecode-region-not-object-source
  (doseq [tags [['double 'double] ['Object 'Object]]]
    (let [operand (list 'clojure.core// (typed-helper-call #'retained-caller 'numerator)
                        (typed-helper-call #'retained-caller 'denominator))
          body (typed-helper-call #'retained-caller operand)
          interface (types/ensure-fn-interface! tags)
          {:keys [implementation emission]}
          (#'core/emit-owned-bytecode!
           bytecode/compile-typed-impl! "raster.compiled.CompoundRegionOriginal"
           '[numerator denominator] tags 'double [body]
           (the-ns 'raster.compiler.backend.jvm.bytecode-input-completion-test)
           (:iface-name interface) 'compound-input)
          original (first (:body emission))
          region (bytecode/compile-invocation-input-region emission original 0)]
      (is (= '[numerator denominator] (:parameters region)))
      (doseq [[n d expected] [[1.0 2.0 0.5] [1.0 0.0 Double/POSITIVE_INFINITY]
                              [0.0 0.0 Double/NaN] [-0.0 2.0 -0.0]]]
        (let [original (implementation n d)
              outlined ((:invoke region) n d)]
          (is (= (Double/doubleToRawLongBits original) (Double/doubleToRawLongBits outlined)))
          (if (Double/isNaN expected)
            (is (Double/isNaN outlined))
            (is (= (Double/doubleToRawLongBits expected) (Double/doubleToRawLongBits outlined)))))))))

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

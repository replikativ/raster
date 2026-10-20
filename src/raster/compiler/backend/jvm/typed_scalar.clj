(ns raster.compiler.backend.jvm.typed-scalar
  "Reference execution for closed, typed scalar regions.

   This is a JVM backend for TypedSOAC scalar SSA, not another type or function inference layer.
   Regions have already retained operand/result dtypes and passed the compiler's purity checker.
   The interpreter resolves their semantic operations through the defining namespace (or a Java
   static class) and coerces every local/result at its declared dtype boundary. It never evals a
   source form and keeps no private operation registry."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :backend :typed-scalar-jvm))))

(defn- typed-scalar?
  [value]
  (and (map? value) (keyword? (:type value)) (contains? value :value)))

(defn buffer-operand
  "Describe one shaped host/runtime buffer to a host-only typed scalar equation."
  [shape]
  (let [shape (vec shape)]
    (when-not (every? #(and (integer? %) (not (neg? %))) shape)
      (fail! :typed-scalar-buffer-shape "typed scalar buffer shape must be concrete"
             {:shape shape}))
    {::buffer true :shape shape}))

(defn- buffer-operand?
  [value]
  (and (map? value) (true? (::buffer value)) (vector? (:shape value))))

(defn- namespace-object
  [source-ns]
  (cond
    (instance? clojure.lang.Namespace source-ns) source-ns
    (symbol? source-ns)
    (or (find-ns source-ns)
        (fail! :typed-scalar-namespace "typed scalar source namespace is not loaded"
               {:source-ns source-ns}))
    :else
    (fail! :typed-scalar-namespace "typed scalar execution requires a namespace or namespace symbol"
           {:source-ns source-ns})))

(defn- coerce-value
  [declared value]
  (case (dtype/canon declared)
    :byte (byte value)
    :int (int value)
    :long (long value)
    :float (float value)
    :double (double value)
    :half (fail! :typed-scalar-half
                 "the JVM reference backend has no scalar FP16 value representation"
                 {:dtype declared :value value})))

(defn- resolve-class
  [source-ns operation]
  (when-let [owner (namespace operation)]
    (let [owner-symbol (symbol owner)
          resolved (or (ns-resolve source-ns owner-symbol)
                       (try (Class/forName owner false (.getContextClassLoader
                                                        (Thread/currentThread)))
                            (catch ClassNotFoundException _ nil))
                       (try (Class/forName (str "java.lang." owner) false
                                           (.getContextClassLoader (Thread/currentThread)))
                            (catch ClassNotFoundException _ nil)))]
      (when (class? resolved) resolved))))

(defn- resolve-callable
  [source-ns operation]
  (or (when (namespace operation)
        (try (requiring-resolve operation) (catch Throwable _ nil)))
      (ns-resolve source-ns operation)
      (ns-resolve 'clojure.core operation)))

(declare evaluate-expression)

(defn- invoke-operation
  [source-ns expression environment]
  (let [{:keys [dispatch? lexical-callee]} (descriptor/call-description expression)
        _ (when (or dispatch? lexical-callee)
            (fail! :typed-scalar-owned-source-required
                   "selected source calls require their original receiver and argument-completion context"
                   {:expression expression}))
        operation (descriptor/semantic-op expression)
        arguments (descriptor/call-args expression)]
    (when-not operation
      (fail! :typed-scalar-operation "typed scalar call has no retained semantic operation"
             {:expression expression}))
    (cond
      (descriptor/alength-op? operation)
      (let [[argument & extra] arguments
            value (evaluate-expression source-ns environment argument)]
        (when (or (nil? argument) (seq extra))
          (fail! :typed-scalar-alength "typed scalar alength requires one buffer operand"
                 {:expression expression}))
        (cond
          (buffer-operand? value) (first (:shape value))
          (and value (.isArray (class value))) (alength value)
          :else (fail! :typed-scalar-alength
                       "typed scalar alength operand has no host shape contract"
                       {:expression expression :operand value})))

      :else
      (case operation
        if (let [[predicate consequent alternate] arguments]
             (evaluate-expression source-ns environment
                                  (if (evaluate-expression source-ns environment predicate)
                                    consequent alternate)))
        and (loop [remaining arguments, result true]
              (if (seq remaining)
                (let [value (evaluate-expression source-ns environment (first remaining))]
                  (if value (recur (next remaining) value) value))
                result))
        or (loop [remaining arguments, result nil]
             (if (seq remaining)
               (let [value (evaluate-expression source-ns environment (first remaining))]
                 (if value value (recur (next remaining) value)))
               result))
        (let [values (mapv #(evaluate-expression source-ns environment %) arguments)]
          (if-let [callable (resolve-callable source-ns operation)]
            (apply callable values)
            (if-let [owner (resolve-class source-ns operation)]
              (clojure.lang.Reflector/invokeStaticMethod
               (.getName ^Class owner) (name operation) (to-array values))
              (fail! :typed-scalar-operation
                     "typed scalar semantic operation cannot be resolved by the JVM backend"
                     {:operation operation :expression expression
                      :source-ns (ns-name source-ns)}))))))))

(defn evaluate-expression
  "Interpret one proven-pure scalar expression in `environment`.

   Symbols name typed region operands/locals. Calls are dispatched by their retained semantic
   identity; unknown or effectful expressions fail loudly. A selected source call is not a
   semantic generic call. The optional compiler-owned source executor evaluates the WHOLE
   expression under its original namespace/policy and completion contexts, not pre-evaluated
   arguments. Its owner must prepare/cache the executable and guard its historical inputs."
  ([source-ns environment expression]
   (evaluate-expression source-ns environment expression nil))
  ([source-ns environment expression execute-source]
  (let [source-ns (namespace-object source-ns)]
    (cond
      (or (number? expression) (boolean? expression) (nil? expression)) expression
      (symbol? expression)
      (if (contains? environment expression)
        (get environment expression)
        (if-let [constant (numeric-constant/value expression)]
          (:value constant)
          (fail! :typed-scalar-unbound "typed scalar expression references an unbound SSA value"
                 {:symbol expression :available (set (keys environment))})))
      (seq? expression)
      (do
        (when-not (= :pure (effects/analyze-effect expression))
          (fail! :typed-scalar-effect "JVM scalar backend only executes proven-pure expressions"
                 {:expression expression :effect (effects/analyze-effect expression)}))
        (if execute-source
          (execute-source expression environment)
          (invoke-operation source-ns expression environment)))
      :else
      (fail! :typed-scalar-expression "JVM scalar backend cannot execute this typed scalar value"
             {:expression expression :type (type expression)})))))

(defn evaluate-array-size
  "Evaluate one admitted size-only primitive-array constructor operand exactly once.
   Core constructors use Number.intValue, not checked `int` or long launch-extent arithmetic.
   This performs neither allocation nor initialization and grants no native capacity proof."
  ([source-ns environment expression]
   (evaluate-array-size source-ns environment expression nil))
  ([source-ns environment expression execute-source]
  (let [source-value (if execute-source
                       (evaluate-expression source-ns environment expression execute-source)
                       (evaluate-expression source-ns environment expression))]
    (when-not (instance? Number source-value)
      (fail! :typed-scalar-array-size
             "size-only array realization requires a numeric constructor operand"
             {:expression expression :type (type source-value)}))
    (let [elements (.intValue ^Number source-value)]
      (when (neg? elements)
        (throw (NegativeArraySizeException. (str elements))))
      {:source-value source-value :elements elements}))))

(defn evaluate-region
  "Execute a canonical TypedSOAC lambda for ordered typed scalar operands.

   `result-dtypes` must align with the region's declared result expressions. The return value is
   an ordered vector of `{:type dtype :value value}` maps suitable for compiler/runtime ABIs.
   The optional source executor does not remove declared local/result conversions. Its producer
   must independently establish that these checkpoints correspond to original source completion."
  ([source-ns lambda operands result-dtypes]
   (evaluate-region source-ns lambda operands result-dtypes nil))
  ([source-ns lambda operands result-dtypes execute-source]
  (let [{:keys [parameters locals body-results]} (soac/lambda-parts lambda)
        operands (vec operands)
        result-dtypes (vec result-dtypes)]
    (when-not (= (count parameters) (count operands))
      (fail! :typed-scalar-arity "typed scalar operand count differs from its lambda"
             {:parameters parameters :operand-count (count operands)}))
    (when-not (every? #(or (typed-scalar? %) (buffer-operand? %)) operands)
      (fail! :typed-scalar-operand "typed scalar region requires typed scalar or buffer operands"
             {:operands operands}))
    (when-not (= (count body-results) (count result-dtypes))
      (fail! :typed-scalar-results "typed scalar result dtypes differ from its result arity"
             {:results (count body-results) :dtypes result-dtypes}))
    (let [initial (into {} (map (fn [parameter operand]
                                  [parameter (if (typed-scalar? operand)
                                               (:value operand)
                                               operand)])
                                parameters operands))
          environment
          (reduce (fn [environment {:keys [id dtype init]}]
                    (assoc environment id
                           (coerce-value dtype
                                         (evaluate-expression source-ns environment init execute-source))))
                  initial locals)]
      (mapv (fn [expression result-dtype]
              {:type (dtype/canon result-dtype)
               :value (coerce-value result-dtype
                                    (evaluate-expression source-ns environment expression execute-source))})
            body-results result-dtypes)))))

(defn evaluate-invocation-step
  "InvocationMaterialization evaluator for one closed ScalarCompute step."
  ([source-ns step operand-values]
   (evaluate-invocation-step source-ns step operand-values nil))
  ([source-ns step operand-values execute-source]
  (let [{:keys [parameters]} (soac/lambda-parts (:region step))
        operands (mapv (fn [parameter]
                         (or (get operand-values parameter)
                             (fail! :typed-scalar-invocation-operand
                                    "invocation scalar operand is missing"
                                    {:parameter parameter :available (set (keys operand-values))})))
                       parameters)]
    (first (evaluate-region source-ns (:region step) operands
                            [(get-in step [:value :dtype])] execute-source)))))

(defn evaluate-host-equation
  "EmittedParallelProgramCall evaluator for one host-only TypedSOAC equation.
   Buffer operands here are shape descriptions, not source arrays. An optional source executor
   must not pass those stand-ins to original array-taking calls without an audited projection."
  ([source-ns equation context]
   (evaluate-host-equation source-ns equation context nil))
  ([source-ns equation {:keys [operands values buffer-shapes]} execute-source]
  (let [algorithm (soac/validate! (:algorithm equation))
        operands (reduce-kv (fn [operands id value]
                              (assoc operands id
                                     (if-let [shape (get buffer-shapes id)]
                                       (buffer-operand shape)
                                       value)))
                            {} operands)
        environment
        (reduce
         (fn [environment algorithm-equation]
           (let [{:keys [kind captures lambda]} (soac/operation-parts algorithm-equation)
                 results (vec (nth algorithm-equation 2))]
             (when-not (= 'scalar kind)
               (fail! :typed-scalar-host-operation
                      "host-only equation contains a non-scalar TypedSOAC operation"
                      {:equation (:id equation) :kind kind}))
             (let [arguments (mapv (fn [capture]
                                     (or (get environment capture)
                                         (fail! :typed-scalar-host-operand
                                                "host scalar capture is unavailable"
                                                {:capture capture
                                                 :available (set (keys environment))})))
                                   captures)
                   produced (evaluate-region source-ns lambda arguments
                                             (mapv #(get-in values [% :dtype]) results) execute-source)]
               (into environment (map vector results produced)))))
         operands (soac/equations algorithm))]
    (select-keys environment (:results equation)))))

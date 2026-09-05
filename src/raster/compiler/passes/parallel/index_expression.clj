(ns raster.compiler.passes.parallel.index-expression
  "Lower already-verified integer index expressions to target-neutral KernelBody arithmetic.

  Schedules provide their own decline function so this small shared vocabulary does not decide
  whether a missing rule is a map, reduction, scan, or contraction coverage failure."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(def ^:private operators
  {'+ :add, 'clojure.core/+ :add, 'raster.numeric/+ :add
   '- :sub, 'clojure.core/- :sub, 'raster.numeric/- :sub
   '* :mul, 'clojure.core/* :mul, 'raster.numeric/* :mul
   'quot :floor-div, 'clojure.core/quot :floor-div
   'rem :mod, 'clojure.core/rem :mod
   'mod :mod, 'clojure.core/mod :mod
   'min :min, 'clojure.core/min :min
   'max :max, 'clojure.core/max :max})

(def ^:private casts
  '#{int long clojure.core/int clojure.core/long})

(def ^:private cast-dtypes
  {'int :int 'clojure.core/int :int
   'long :long 'clojure.core/long :long})

(def ^:private floating-casts
  '#{double clojure.core/double})

(def ^:private ceil-operators
  '#{Math/ceil java.lang.Math/ceil})

(def ^:private division-operators
  '#{/ clojure.core//})

(defn- unwrap-floating-cast
  [expression]
  (if (and (seq? expression)
           (contains? floating-casts (first expression))
           (= 2 (count expression)))
    (second expression)
    expression))

(defn- exact-positive-integer
  [value]
  (when (and (number? value)
             (pos? value)
             (== (double value) (Math/rint (double value)))
             (<= (double value) (double Long/MAX_VALUE)))
    (long value)))

(defn- floating-ceil-div
  "Recognize the legacy scheduling spelling `(Math/ceil (/ (double n) (double d)))`.

   KernelGrid predates the explicit launch algebra and still carries this exact source form.
   Lowering it here retains the source schedule while replacing host floating-point arithmetic
   with the checked integer `ceil-div` operation used by KernelBody and KernelLaunch."
  [expression]
  (when (and (seq? expression)
             (contains? ceil-operators (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (let [division (first (descriptor/call-args expression))]
      (when (and (seq? division)
                 (contains? division-operators (descriptor/semantic-op division))
                 (= 2 (count (descriptor/call-args division))))
        (let [[numerator divisor] (mapv unwrap-floating-cast
                                       (descriptor/call-args division))]
          (when-let [divisor (exact-positive-integer divisor)]
            [numerator divisor]))))))

(defn lower
  "Translate a typed source index expression whose symbols are members of `scope`.

   `decline!` is called as `(decline! rule message data)`.  This is normalization of explicit
   index arithmetic only; it does not infer types, layouts, or affine facts."
  [expression scope decline!]
  (let [expression (descriptor/unwrap-int-cast expression)]
    (cond
      (integer? expression) expression

      (instance? raster.compiler.ir.kernel_body.IndexExpr expression)
      (apply body/expression (:op expression)
             (map #(lower % scope decline!) (:arguments expression)))

      (instance? raster.compiler.ir.kernel_body.IndexCast expression)
      (body/index-cast (lower (:argument expression) scope decline!)
                       (:dtype expression) (:overflow expression))

      (symbol? expression)
      (if (contains? scope expression)
        expression
        (decline! :unbound-index-symbol
                  "portable index expression references an undeclared symbol"
                  {:expression expression :scope scope}))

      (and (seq? expression) (contains? casts (first expression))
           (= 2 (count expression)))
      (lower (second expression) scope decline!)

      (and (seq? expression)
           (descriptor/increment-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression))))
      (body/expression :add
                       (lower (first (descriptor/call-args expression)) scope decline!)
                       1)

      (and (seq? expression)
           (descriptor/decrement-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression))))
      (body/expression :sub
                       (lower (first (descriptor/call-args expression)) scope decline!)
                       1)

      (floating-ceil-div expression)
      (let [[numerator divisor] (floating-ceil-div expression)]
        (body/expression :ceil-div (lower numerator scope decline!) divisor))

      (seq? expression)
      (let [operator (get operators (descriptor/semantic-op expression))
            arguments (vec (descriptor/call-args expression))]
        (when-not (and operator (seq arguments)
                       (or (not= :sub operator) (= 2 (count arguments))))
          (decline! :index-expression
                    "portable index expression requires explicit integer arithmetic"
                    {:expression expression :operator (descriptor/semantic-op expression)}))
        (apply body/expression operator
               (map #(lower % scope decline!) arguments)))

      :else
      (decline! :index-expression
                "portable index expression has an unsupported value"
                {:expression expression :type (type expression)}))))

(defn lower-typed
  "Lower exact integral arithmetic while preserving its retained Typed Clojure width.

   `leaf-dtype` supplies the authoritative dtype for each scalar SSA leaf and `expected-dtype`
   is the typed scalar equation's result. Compound arithmetic requires its own retained dtype and
   must already be long; int arithmetic may wrap or trap before widening and therefore declines.
   The only implicit conversion this projection inserts is exact int-to-long leaf widening.
   This is the graph-launch counterpart of scalar KernelBody lowering, not source type inference."
  [expression scope leaf-dtype expected-dtype decline!]
  (letfn [(source-dtype [form]
            (or (some-> (or (:raster.type/tag (meta form)) (:tag (meta form)))
                        dtype/dtype-for-scalar-tag dtype/canon)
                (when (symbol? form) (some-> (leaf-dtype form) dtype/canon))
                (when (integer? form)
                  (if (<= Integer/MIN_VALUE form Integer/MAX_VALUE) :int :long))))
          (coerce [lowered source target form]
            (let [source (dtype/canon source)
                  target (dtype/canon target)]
              (cond
                (= source target) lowered
                (and (= :int source) (= :long target))
                (body/index-cast lowered :long :exact)
                :else
                (decline! :index-expression-dtype
                          "typed launch arithmetic permits only exact integral widening"
                          {:expression form :source source :target target}))))
          (lower* [form expected]
            (let [expected (dtype/canon expected)]
              (cond
                (integer? form)
                (coerce form (source-dtype form) expected form)

                (symbol? form)
                (if (contains? scope form)
                  (let [source (source-dtype form)]
                    (when-not (contains? #{:int :long} source)
                      (decline! :index-expression-dtype
                                "typed launch scalar requires an integral retained dtype"
                                {:expression form :dtype source}))
                    (coerce form source expected form))
                  (decline! :unbound-index-symbol
                            "portable index expression references an undeclared symbol"
                            {:expression form :scope scope}))

                (and (seq? form)
                     (contains? cast-dtypes (descriptor/semantic-op form))
                     (= 1 (count (descriptor/call-args form))))
                (let [target (get cast-dtypes (descriptor/semantic-op form))
                      argument (first (descriptor/call-args form))
                      source (source-dtype argument)
                      _ (when-not (contains? #{:int :long} source)
                          (decline! :index-expression-dtype
                                    "an exact index cast requires its argument's retained dtype"
                                    {:expression form :argument argument :dtype source}))
                      lowered (lower* argument source)]
                  (coerce lowered source target form))

                (seq? form)
                (let [operator (get operators (descriptor/semantic-op form))
                      arguments (vec (descriptor/call-args form))
                      result-type (source-dtype form)]
                  (when-not (and operator (seq arguments)
                                 (contains? #{:int :long} result-type)
                                 (or (not= :sub operator) (= 2 (count arguments))))
                    (decline! :index-expression
                              "typed launch expression requires explicit integral arithmetic"
                              {:expression form :operator (descriptor/semantic-op form)
                               :dtype result-type}))
                  ;; KernelLaunch is exact checked mathematical-integer algebra. An `int`
                  ;; operation is not: Raster's numerical overloads wrap, and checked host int
                  ;; operations can fail where a widened launch computation would succeed. Only
                  ;; an already-typed long operation may be replayed as storage/launch algebra.
                  (when (= :int result-type)
                    (decline! :index-expression-overflow
                              "int arithmetic cannot be projected into exact launch algebra"
                              {:expression form :operator (descriptor/semantic-op form)
                               :dtype result-type}))
                  (coerce
                   (apply body/expression operator
                          (map #(lower* % result-type) arguments))
                   result-type expected form))

                :else
                (decline! :index-expression
                          "portable typed index expression has an unsupported value"
                          {:expression form :type (type form)}))))]
    (let [expected (dtype/canon expected-dtype)]
      (when-not (contains? #{:int :long} expected)
        (decline! :index-expression-dtype
                  "typed launch result requires an integral retained dtype"
                  {:expression expression :dtype expected-dtype}))
      (lower* expression expected))))

(defn to-launch-expression
  "Project non-negative KernelBody index arithmetic into resolvable host launch IR.

   KernelBody and KernelLaunch deliberately use distinct expression records because one executes
   in a kernel and the other is resolved by the runtime binder. This conversion admits only the
   monotone extent vocabulary shared by both; subtraction and modulo remain kernel-local."
  [expression decline!]
  (cond
    (integer? expression) expression
    (or (symbol? expression) (keyword? expression)) (launch/runtime-value expression)

    (instance? raster.compiler.ir.kernel_body.IndexCast expression)
    (body/index-cast (to-launch-expression (:argument expression) decline!)
                     (:dtype expression) (:overflow expression))

    (instance? raster.compiler.ir.kernel_body.IndexExpr expression)
    (let [arguments (mapv #(to-launch-expression % decline!) (:arguments expression))]
      (case (:op expression)
        :add (apply launch/sum arguments)
        :mul (apply launch/product arguments)
        :floor-div (apply launch/floor-div arguments)
        :ceil-div (apply launch/ceil-div arguments)
        :min (apply launch/minimum arguments)
        :max (apply launch/maximum arguments)
        (decline! :launch-index-expression
                  "kernel index operation is not a non-negative launch extent"
                  {:expression expression :operation (:op expression)})))

    :else
    (decline! :launch-index-expression
              "kernel index value cannot be projected into launch IR"
              {:expression expression :type (type expression)})))

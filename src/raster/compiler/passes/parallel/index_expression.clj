(ns raster.compiler.passes.parallel.index-expression
  "Lower already-verified integer index expressions to target-neutral KernelBody arithmetic.

   Schedules provide their own decline function so this small shared vocabulary does not decide
   whether a missing rule is a map, reduction, scan, or contraction coverage failure."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
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

(defn lower
  "Translate a typed source index expression whose symbols are members of `scope`.

   `decline!` is called as `(decline! rule message data)`.  This is normalization of explicit
   index arithmetic only; it does not infer types, layouts, or affine facts."
  [expression scope decline!]
  (let [expression (descriptor/unwrap-int-cast expression)]
    (cond
      (integer? expression) expression

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
    (to-launch-expression (:argument expression) decline!)

    (instance? raster.compiler.ir.kernel_body.IndexExpr expression)
    (let [arguments (mapv #(to-launch-expression % decline!) (:arguments expression))]
      (case (:op expression)
        :add (apply launch/sum arguments)
        :mul (apply launch/product arguments)
        :floor-div (apply launch/floor-div arguments)
        :ceil-div (apply launch/ceil-div arguments)
        :min (apply launch/minimum arguments)
        (decline! :launch-index-expression
                  "kernel index operation is not a non-negative launch extent"
                  {:expression expression :operation (:op expression)})))

    :else
    (decline! :launch-index-expression
              "kernel index value cannot be projected into launch IR"
              {:expression expression :type (type expression)})))

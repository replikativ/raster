(ns raster.compiler.ir.extent-expression
  "Canonical equality for target-neutral integer extent expressions.

   Source S-expressions, KernelBody index expressions, and KernelLaunch records deliberately have
   different representations. This namespace normalizes only the small, checked extent algebra;
   it is not a general scalar simplifier and does not prove inequalities.")

(defn- record-name [value]
  (some-> value class .getSimpleName))

(defn- canonical-commutative [operator arguments]
  (let [arguments (mapcat (fn [argument]
                            (if (and (vector? argument) (= operator (first argument)))
                              (second argument)
                              [argument]))
                          arguments)
        identity (case operator :mul [:leaf 1] :add [:leaf 0] nil)
        arguments (if identity (remove #{identity} arguments) arguments)
        arguments (vec (sort-by pr-str arguments))]
    (case (count arguments)
      0 (or identity [operator []])
      1 (first arguments)
      [operator arguments])))

(declare canonical)

(defn operation [operator arguments]
  (let [arguments (mapv canonical arguments)]
    (case operator
      (:mul :add :min :max) (canonical-commutative operator arguments)
      [operator arguments])))

(defn canonical
  "Normalize source, KernelBody, and KernelLaunch integer extent spellings for equality only."
  [expression]
  (case (record-name expression)
    "RuntimeValue" (canonical (:value expression))
    "Product" (operation :mul (:factors expression))
    "Sum" (operation :add (:terms expression))
    "Minimum" (operation :min (:values expression))
    "Maximum" (operation :max (:values expression))
    "CeilDiv" (operation :ceil-div [(:value expression) (:divisor expression)])
    "FloorDiv" (operation :floor-div [(:value expression) (:divisor expression)])
    "AlignUp" (operation :align-up [(:value expression) (:alignment expression)])
    "IndexExpr" (operation (:op expression) (:arguments expression))
    "IndexCast" (canonical (:argument expression))
    (cond
      (and (seq? expression)
           (contains? '#{int long double clojure.core/int clojure.core/long
                         clojure.core/double}
                      (first expression))
           (= 2 (count expression)))
      (canonical (second expression))

      (seq? expression)
      (let [operator ({'* :mul 'clojure.core/* :mul
                       '+ :add 'clojure.core/+ :add
                       'min :min 'clojure.core/min :min
                       'max :max 'clojure.core/max :max
                       'quot :floor-div 'clojure.core/quot :floor-div}
                      (first expression))]
        (if operator
          (operation operator (rest expression))
          [:leaf expression]))

      :else [:leaf expression])))

(defn equivalent? [left right]
  (= (canonical left) (canonical right)))

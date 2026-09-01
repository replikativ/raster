(ns raster.compiler.passes.parallel.scalar-expression-body
  "Lower a typed scalar S-expression region to target-neutral KernelBody SSA.

   This is shared by ordinary maps and ordered fold-maps. The caller supplies authoritative
   storage/scalar dtypes, index lowering, participation, and its own structured decline function;
   this pass performs no source-level type or function inference."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]))

(def ^:private cast-heads
  {'byte :byte, 'clojure.core/byte :byte
   'int :int, 'clojure.core/int :int
   'long :long, 'clojure.core/long :long
   'float :float, 'clojure.core/float :float
   'double :double, 'clojure.core/double :double})

(defn inline-lets
  "Inline a top-level scalar let region while preserving its already-typed expression tree."
  [expression]
  (if (and (seq? expression) (contains? #{'let 'let* 'clojure.core/let} (first expression)))
    (let [[_ bindings result] expression]
      (reduce (fn [result [id init]] (util/subst-syms {id init} result))
              result (reverse (partition 2 bindings))))
    expression))

(defn make-lowerer
  "Build a scalar-expression lowerer.

   Returns `{:lower f}` where `f` accepts expression, expected dtype, and a map of typed local
   values. `decline!` is called as `(decline! rule message data)` so each owning schedule retains
   an honest, local coverage contract."
  [{:keys [array-types scalar-types arrays index-scope lower-index predicate id-prefix decline!]
    :or {id-prefix "scalar"}}]
  (let [canon-type #(if (= :predicate %) :predicate (dtype/canon %))
        counter (atom 0)
        fresh (fn [prefix] (symbol (str id-prefix "-" prefix "-" (swap! counter inc))))
        source-type
        (fn source-type [expression expected env]
          (let [expression (inline-lets expression)]
            (or
             (some-> (or (:raster.type/tag (meta expression)) (:tag (meta expression)))
                     dtype/dtype-for-scalar-tag)
             (cond
               (symbol? expression) (or (get env expression) (get scalar-types expression) expected)
               (number? expression) expected
               (descriptor/aget-call? expression)
               (or (get array-types (descriptor/aget-array-sym expression)) expected)
               (and (seq? expression) (contains? cast-heads (first expression)))
               (get cast-heads (first expression))
               (and (seq? expression)
                    (= :cmp (:kind (intrinsics/descriptor
                                    (intrinsics/canonical
                                     (descriptor/semantic-op expression))))))
               :predicate
               :else expected))))]
    (when-not (and (fn? lower-index) (fn? decline!))
      (throw (ex-info "scalar KernelBody lowering requires index and decline callbacks"
                      {:reason :raster/bug :lower-index lower-index :decline decline!})))
    (letfn [(lower [expression expected env]
              (let [expression (inline-lets expression)
                    expected (canon-type expected)]
                (cond
                  (number? expression)
                  {:operations [] :result (body/literal expression expected) :type expected}

                  (boolean? expression)
                  {:operations [] :result (body/literal expression :predicate) :type :predicate}

                  (symbol? expression)
                  (if-let [actual (or (get env expression) (get scalar-types expression))]
                    {:operations [] :result expression :type (canon-type actual)}
                    (decline! :unbound-scalar
                              "scalar expression references an undeclared value"
                              {:expression expression :environment (set (keys env))}))

                  (descriptor/aget-call? expression)
                  (let [array (descriptor/aget-array-sym expression)
                        arguments (vec (descriptor/call-args expression))
                        coordinate (last arguments)
                        array-type (some-> (get array-types array) dtype/canon)]
                    (when-not (and (= 2 (count arguments)) (contains? arrays array) array-type)
                      (decline! :indexed-load
                                "scalar loads require a declared typed stable tensor"
                                {:expression expression :array array :array-types array-types}))
                    (let [id (fresh "load")]
                      {:operations [(body/->ScalarLoad
                                     (body/value id array-type) array
                                     [(lower-index coordinate)] predicate
                                     (when predicate (body/literal 0 array-type)) :cached)]
                       :result id :type array-type}))

                  (and (seq? expression) (contains? cast-heads (first expression))
                       (= 2 (count expression)))
                  (let [target (dtype/canon (get cast-heads (first expression)))
                        source-expected (dtype/canon (source-type (second expression) target env))
                        lowered (lower (second expression) source-expected env)]
                    (if (= target (:type lowered))
                      lowered
                      (let [source (dtype/canon (:type lowered))
                            fp-source? (dtype/fp-dtype? source)
                            fp-target? (dtype/fp-dtype? target)
                            widening? (<= (dtype/bytes-of source) (dtype/bytes-of target))
                            [rounding overflow]
                            (cond
                              (and fp-source? fp-target? widening?) [:exact :exact]
                              (and fp-source? fp-target?) [:nearest-even :ieee]
                              (and (not fp-source?) (not fp-target?) widening?) [:exact :exact]
                              :else
                              (decline! :cast-policy
                                        "scalar cast has no portable exact policy"
                                        {:expression expression :source source :target target}))
                            id (fresh "cast")]
                        {:operations (conj (:operations lowered)
                                           (body/->ScalarCompute
                                            (body/value id target)
                                            (body/cast-expression (:result lowered) target
                                                                  rounding overflow)))
                         :result id :type target})))

                  (and (seq? expression) (= 'if (first expression)) (= 4 (count expression)))
                  (let [[_ condition then-expression else-expression] expression
                        condition (lower condition :predicate env)
                        then-value (lower then-expression expected env)
                        else-value (lower else-expression expected env)
                        result-type (:type then-value)
                        _ (when-not (= result-type (:type else-value) expected)
                            (decline! :branch-dtype
                                      "scalar branches must have one explicit result dtype"
                                      {:expression expression :then (:type then-value)
                                       :else (:type else-value) :expected expected}))
                        result (fresh "if")]
                    {:operations
                     (conj (vec (:operations condition))
                           (body/->IfRegion
                            (:result condition)
                            (conj (vec (:operations then-value))
                                  (body/->Yield [(:result then-value)]))
                            (conj (vec (:operations else-value))
                                  (body/->Yield [(:result else-value)]))
                            [(body/value result result-type)]))
                     :result result :type result-type})

                  (seq? expression)
                  (let [operator (intrinsics/canonical (descriptor/semantic-op expression))
                        intrinsic (intrinsics/descriptor operator)
                        arguments (vec (descriptor/call-args expression))]
                    (when-not (and intrinsic (= (:arity intrinsic) (count arguments)))
                      (decline! :scalar-expression
                                "scalar expression has no canonical intrinsic"
                                {:expression expression :operator operator}))
                    (let [comparison? (= :cmp (:kind intrinsic))
                          operand-type (if comparison?
                                         (dtype/canon
                                          (or (source-type (first arguments) :int env) :int))
                                         expected)
                          lowered (mapv #(lower % operand-type env) arguments)
                          result-type (if comparison? :predicate operand-type)
                          _ (when-not (every? #(= operand-type (:type %)) lowered)
                              (decline! :operand-dtype
                                        "scalar intrinsic operands require one dtype"
                                        {:expression expression :operand-type operand-type
                                         :actual (mapv :type lowered)}))
                          result (fresh "value")]
                      {:operations
                       (conj (vec (mapcat :operations lowered))
                             (body/->ScalarCompute
                              (body/value result result-type)
                              (body/scalar-expression operator result-type
                                                      (mapv :result lowered))))
                       :result result :type result-type}))

                  :else
                  (decline! :scalar-expression
                            "scalar expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      {:lower lower})))

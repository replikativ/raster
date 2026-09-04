(ns raster.compiler.passes.parallel.scalar-expression-body
  "Lower a typed scalar S-expression region to target-neutral KernelBody SSA.

   This is shared by ordinary maps and ordered fold-maps. The caller supplies authoritative
   storage/scalar dtypes, index lowering, participation, and its own structured decline function;
   this pass performs no source-level type or function inference."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.passes.parallel.patterns :as patterns]))

(def ^:private cast-heads
  {'byte :byte, 'clojure.core/byte :byte
   'int :int, 'clojure.core/int :int
   'long :long, 'clojure.core/long :long
   'float :float, 'clojure.core/float :float
   'double :double, 'clojure.core/double :double})

(defn- contains-indexed-load?
  [expression]
  (boolean (some descriptor/aget-call? (tree-seq coll? seq expression))))

(defn inline-lets
  "Inline a top-level scalar let region while preserving its already-typed expression tree."
  [expression]
  (if (and (seq? expression) (contains? #{'let 'let* 'clojure.core/let} (first expression)))
    (let [[_ bindings result] expression]
      (reduce (fn [result [id init]] (util/subst-syms {id init} result))
              result (reverse (partition 2 bindings))))
    expression))

(defn- match-ordered-loop
  "Recognize the canonical one-index/one-carry loop without authorizing reassociation."
  [expression]
  (or
   (when-let [matched (patterns/match-reduce-loop expression)]
     (assoc matched :inclusive? false
            :update-expr (or (:scoped-update-expr matched) (:update-expr matched))))
   (when-let [{:keys [kind index-sym acc-sym acc-init body-form bound]}
              (patterns/normalize-loop expression)]
     (when (and (= :reduce-loop kind)
                (= :le (descriptor/comparison-kind
                        (descriptor/semantic-op (second body-form)))))
       (let [[_ _test then-branch else-expr] body-form
             recur-form (patterns/find-recur-form then-branch)
             recur-args (vec (rest recur-form))
             index-update? #(= 1 (descriptor/affine-step % index-sym))
             [update-expr index-update]
             (cond
               (and (= 2 (count recur-args)) (index-update? (second recur-args)))
               [(first recur-args) (second recur-args)]

               (and (= 2 (count recur-args)) (index-update? (first recur-args)))
               [(second recur-args) (first recur-args)]

               :else [nil nil])]
         (when (and update-expr index-update)
           {:acc-sym acc-sym :acc-init acc-init :index-sym index-sym
            :bound-expr bound :else-expr else-expr :update-expr update-expr
            :inclusive? true}))))))

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
    (letfn [(cast-lowered [lowered target expression]
              (let [target (dtype/canon target)]
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
                          (and (not fp-source?) fp-target? (= :double target)
                               (<= (dtype/bytes-of source) 4)) [:exact :exact]
                          (and (not fp-source?) fp-target?)
                          [:nearest-even (if (= :half target) :ieee :exact)]
                          :else
                          (decline! :cast-policy
                                    "scalar cast has no portable rounding and overflow policy"
                                    {:expression expression :source source :target target}))
                        id (fresh "cast")]
                    {:operations (conj (:operations lowered)
                                       (body/->ScalarCompute
                                        (body/value id target)
                                        (body/cast-expression (:result lowered) target
                                                              rounding overflow)))
                     :result id :type target}))))

            (lower [expression expected env]
              (let [expression (inline-lets expression)
                    expected (canon-type expected)]
                (cond
                  (instance? raster.compiler.ir.kernel_body.Literal expression)
                  {:operations [] :result expression
                   :type (canon-type (:type expression))}

                  (and (= :predicate expected) (number? expression))
                  (if (contains? #{0 1} expression)
                    {:operations [] :result (body/literal (= 1 expression) :predicate)
                     :type :predicate}
                    (decline! :predicate-literal
                              "numeric predicates must use the canonical zero/one encoding"
                              {:expression expression}))

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
                    (let [coordinate-value
                          (when (contains-indexed-load? coordinate)
                            ;; Storage coordinates are address arithmetic. Keep their composed
                            ;; SSA in the KernelBody index width; narrowing a long launch index to
                            ;; an element dtype would either overflow or invent a cast policy.
                            (lower coordinate :long env))
                          coordinate-expression
                          (if coordinate-value
                            (:result coordinate-value)
                            (lower-index coordinate (set (keys env))))
                          id (fresh "load")]
                      {:operations (conj (vec (:operations coordinate-value))
                                         (body/->ScalarLoad
                                          (body/value id array-type) array
                                          [coordinate-expression] predicate
                                          (when predicate (body/literal 0 array-type)) :cached))
                       :result id :type array-type}))

                  (and (seq? expression) (contains? cast-heads (first expression))
                       (= 2 (count expression)))
                  (let [target (dtype/canon (get cast-heads (first expression)))
                        source-expected (dtype/canon (source-type (second expression) target env))
                        lowered (lower (second expression) source-expected env)]
                    (cast-lowered lowered target expression))

                  (and (seq? expression)
                       (= 'raster.numeric/oftype (descriptor/semantic-op expression))
                       (= 2 (count (descriptor/call-args expression))))
                  (let [value-expression (second (descriptor/call-args expression))
                        source-expected (canon-type (source-type value-expression expected env))
                        lowered (lower value-expression source-expected env)]
                    (cast-lowered lowered expected expression))

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

                  (and (seq? expression) (contains? #{'loop 'loop*} (first expression)))
                  (if-let [{:keys [acc-sym acc-init index-sym bound-expr else-expr update-expr
                                   inclusive?]}
                           (match-ordered-loop expression)]
                    (do
                      (when (or (patterns/contains-sym? bound-expr index-sym)
                                (patterns/contains-sym? bound-expr acc-sym)
                                (patterns/contains-sym? else-expr index-sym))
                        (decline! :ordered-loop-shape
                                  "ordered scalar loop requires an invariant upper bound and an exit over its carry"
                                  {:expression expression :bound bound-expr
                                   :index index-sym :accumulator acc-sym :exit else-expr}))
                      (let [loop-index (fresh "loop-index")
                            carry (fresh "loop-carry")
                            result (fresh "loop-result")
                            ;; `(loop [i 0 s init] (if (< i n) (recur (inc i) step) exit))` is the
                            ;; ordered fold followed by `exit` over the final carry. The carry keeps
                            ;; the step's own dtype so the exit's cast (e.g. `(double s)`) is an
                            ;; explicit conversion rather than a silent widening of the fold.
                            exit? (not= else-expr acc-sym)
                            carry-type (if exit?
                                         (canon-type (source-type update-expr expected env))
                                         expected)
                            initial (lower acc-init carry-type env)
                            loop-type (:type initial)
                            update (util/subst-syms {index-sym loop-index acc-sym carry}
                                                    update-expr)
                            lowered (lower update loop-type
                                           (assoc env loop-index :long carry loop-type))
                            _ (when-not (= loop-type (:type lowered))
                                (decline! :ordered-loop-dtype
                                          "ordered scalar loop carry dtype must remain invariant"
                                          {:expression expression :initial loop-type
                                           :update (:type lowered)}))
                            upper (lower-index bound-expr (set (keys env)))
                            upper (if inclusive?
                                    (body/expression :add upper
                                                     (body/index-cast 1 :long :exact))
                                    upper)
                            loop (body/->ForLoop
                                  (body/value loop-index :long)
                                  (body/index-cast 0 :long :exact)
                                  upper 1
                                  [(body/->LoopArg (body/value carry loop-type)
                                                   (:result initial))]
                                  (conj (vec (:operations lowered))
                                        (body/->Yield [(:result lowered)]))
                                  [(body/value result loop-type)]
                                  {:association :ordered})
                            exit (when exit?
                                   (lower (util/subst-syms {acc-sym result} else-expr)
                                          expected (assoc env result loop-type)))]
                        (if exit
                          {:operations (into (conj (vec (:operations initial)) loop)
                                             (:operations exit))
                           :result (:result exit) :type (:type exit)}
                          {:operations (conj (vec (:operations initial)) loop)
                           :result result :type loop-type})))
                    (decline! :ordered-loop-shape
                              "scalar loop is outside the canonical ordered carry form"
                              {:expression expression}))

                  (seq? expression)
                  (let [semantic-operation (descriptor/semantic-op expression)
                        operator (intrinsics/canonical semantic-operation)
                        intrinsic (intrinsics/descriptor operator)
                        arguments (vec (descriptor/call-args expression))]
                    (when-not intrinsic
                      (decline! :scalar-expression
                                "scalar expression has no canonical intrinsic"
                                {:expression expression :operator operator}))
                    (if (and (= 2 (:arity intrinsic)) (> (count arguments) 2)
                             (contains? #{:+ :* :- :div :min :max} operator))
                      ;; Preserve source evaluation order while spelling variadic scalar folds in
                      ;; the binary KernelBody vocabulary. This is normalization, not algebraic
                      ;; reassociation: `(- a b c)` becomes `(- (- a b) c)`.
                      (lower (reduce (fn [left right]
                                       (list (descriptor/semantic-op expression) left right))
                                     arguments)
                             expected env)
                      (do
                        (when-not (= (:arity intrinsic) (count arguments))
                          (decline! :scalar-expression
                                    "scalar expression has the wrong intrinsic arity"
                                    {:expression expression :operator operator
                                     :expected (:arity intrinsic)
                                     :actual (count arguments)}))
                        (let [comparison? (= :cmp (:kind intrinsic))
                              operand-type (if comparison?
                                             (dtype/canon
                                              (or (source-type (first arguments) :int env) :int))
                                             expected)
                              lowered (mapv #(cast-lowered
                                              (lower % operand-type env)
                                              operand-type expression)
                                            arguments)
                              result-type (if comparison? :predicate operand-type)
                              overflow (intrinsics/source-overflow-policy semantic-operation)
                              options (cond-> {} overflow (assoc :overflow overflow))
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
                                                          (mapv :result lowered)
                                                          options)))
                           :result result :type result-type}))))

                  :else
                  (decline! :scalar-expression
                            "scalar expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      {:lower lower})))

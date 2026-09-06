(ns raster.compiler.passes.parallel.scalar-expression-body
  "Lower a typed scalar S-expression region to target-neutral KernelBody SSA.

   This is shared by ordinary maps and ordered fold-maps. The caller supplies authoritative
   storage/scalar dtypes, index lowering, participation, and its own structured decline function;
   this pass performs no source-level type or function inference."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scalar-range :as scalar-range]
            [raster.compiler.passes.parallel.patterns :as patterns]))

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
   (when-let [matched (patterns/match-ordered-reduce-loop expression)]
     (assoc matched :inclusive? false
            :update-expr (:scoped-update-expr matched)))
   (when-let [{:keys [kind index-sym index-init index-slot acc-sym acc-init body-form bound]}
              (patterns/normalize-ordered-loop expression)]
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
         (when (and update-expr index-update
                    (= 'recur (first then-branch))
                    (patterns/ordered-unit-step? (nth recur-args index-slot) index-sym))
           {:acc-sym acc-sym :acc-init acc-init :index-sym index-sym :index-init index-init
            :bound-expr bound :else-expr else-expr :update-expr update-expr
            :inclusive? true}))))))

(defn make-lowerer
  "Build a scalar-expression lowerer.

   Returns `{:lower f :lower-region g}`. `f` accepts expression, expected dtype, and a map of typed
   local values. `g` accepts an ordered {:bindings [...] :results [...]} region, result dtypes,
   retained binding dtypes, and typed locals; it returns shared SSA operations and ordered results.
   Neither entry point stores tuple results: the schedule must commit all components together.
   `decline!` is called as `(decline! rule message data)` so each owning schedule retains an honest,
   local coverage contract."
  [{:keys [array-types scalar-types arrays index-scope lower-index predicate id-prefix decline!]
    :or {id-prefix "scalar"}}]
  (let [canon-type #(if (= :predicate %) :predicate (dtype/canon %))
        counter (atom 0)
        fresh (fn [prefix] (symbol (str id-prefix "-" prefix "-" (swap! counter inc))))
        ;; Lowered SSA names retain their exact interval here.  The public local environment still
        ;; carries only authoritative dtypes; this private table is a proof cache, not a type or
        ;; function registry, and a missing entry simply declines certification.
        value-ranges (atom {})
        remember-range! (fn [id range]
                          (when range (swap! value-ranges assoc id range))
                          range)
        known-range (fn [value type]
                      (or (get @value-ranges value) (scalar-range/for-dtype type)))
        retained-type (fn [expression]
                        (some-> (or (:raster.type/tag (meta expression))
                                    (:tag (meta expression)))
                                dtype/dtype-for-scalar-tag))
        source-type
        (fn source-type [expression expected env]
          (let [expression (inline-lets expression)]
            (or
             (retained-type expression)
             (cond
               (instance? raster.compiler.ir.kernel_body.Literal expression) (:type expression)
               (symbol? expression) (or (get env expression) (get scalar-types expression) expected)
               (number? expression) expected
               (descriptor/aget-call? expression)
               (or (get array-types (descriptor/aget-array-sym expression)) expected)
               (and (seq? expression) (descriptor/cast-op? (first expression)))
               (dtype/dtype-for-scalar-tag (descriptor/cast-result-tag (first expression)))
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
                        [rounding overflow]
                        ;; Keep this owner's existing device-wrap policy explicit. Reduction
                        ;; lowering must not inherit it merely by sharing conversion machinery.
                        (or (scalar-conversion/policy source target :wrap)
                            (decline! :cast-policy
                                      "scalar cast has no portable rounding and overflow policy"
                                      {:expression expression :source source :target target}))
                        id (fresh "cast")]
                    (let [range (when (scalar-range/contained-in-dtype? (:range lowered) target)
                                  (:range lowered))]
                      (remember-range! id (or range (scalar-range/for-dtype target)))
                      {:operations (conj (:operations lowered)
                                       (body/->ScalarCompute
                                        (body/value id target)
                                        (body/cast-expression (:result lowered) target
                                                              rounding overflow)))
                       :result id :type target :range (or range (scalar-range/for-dtype target))})))))

            (lower [expression expected env]
              (let [expression (inline-lets expression)
                    ;; Preserve floating rounding before consumer promotion. Integral contexts
                    ;; retain their existing owner policy: quantized loops still carry widened
                    ;; integers around narrower intrinsic results and need separate reconciliation.
                    expected (canon-type expected)
                    retained (when (seq? expression) (retained-type expression))
                    expected (if (and (dtype/fp-dtype? expected)
                                      retained (dtype/fp-dtype? retained))
                               retained expected)]
                (cond
                  (instance? raster.compiler.ir.kernel_body.Literal expression)
                  {:operations [] :result expression
                   :type (canon-type (:type expression))
                   :range (scalar-range/literal (:value expression) (:type expression))}

                  (and (= :predicate expected) (number? expression))
                  (if (contains? #{0 1} expression)
                    {:operations [] :result (body/literal (= 1 expression) :predicate)
                     :type :predicate}
                    (decline! :predicate-literal
                              "numeric predicates must use the canonical zero/one encoding"
                              {:expression expression}))

                  (number? expression)
                  (let [range (scalar-range/literal expression expected)]
                    {:operations [] :result (body/literal expression expected)
                     :type expected :range range})

                  (boolean? expression)
                  {:operations [] :result (body/literal expression :predicate) :type :predicate}

                  (symbol? expression)
                  (if-let [actual (or (get env expression) (get scalar-types expression))]
                    (let [type (canon-type actual)]
                      {:operations [] :result expression :type type
                       :range (known-range expression type)})
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
                      (let [range (scalar-range/for-dtype array-type)]
                        (remember-range! id range)
                        {:operations (conj (vec (:operations coordinate-value))
                                         (body/->ScalarLoad
                                          (body/value id array-type) array
                                          [coordinate-expression] predicate
                                          (when predicate (body/literal 0 array-type)) :cached))
                         :result id :type array-type :range range})))

                  (and (seq? expression) (descriptor/cast-op? (first expression))
                       (= 2 (count expression)))
                  (let [target (dtype/dtype-for-scalar-tag
                                (descriptor/cast-result-tag (first expression)))
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
                                   inclusive? index-init]}
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
                                  (body/index-cast index-init :long :exact)
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

                  ;; `inc`/`dec` are the closed-core index steppers: `(+ x 1)` / `(- x 1)` with the
                  ;; same operand dtype and overflow policy. Normalize before intrinsic lookup so
                  ;; the KernelBody vocabulary stays binary.
                  (and (seq? expression) (= 2 (count expression))
                       (contains? '#{inc clojure.core/inc} (first expression)))
                  (lower (list 'clojure.core/+ (second expression) 1) expected env)

                  (and (seq? expression) (= 2 (count expression))
                       (contains? '#{dec clojure.core/dec} (first expression)))
                  (lower (list 'clojure.core/- (second expression) 1) expected env)

                  ;; Unary subtraction is the existing negation intrinsic for floating values:
                  ;; spelling it as 0-x would lose the sign of zero. Integral negation instead
                  ;; uses the checked/wrapping subtraction machinery, including MIN_VALUE.
                  (and (seq? expression)
                       (= :- (intrinsics/canonical (descriptor/semantic-op expression)))
                       (= 1 (count (descriptor/call-args expression))))
                  (lower (if (dtype/fp-dtype? expected)
                           (list :neg (first (descriptor/call-args expression)))
                           (list (descriptor/semantic-op expression)
                                 (body/literal 0 expected)
                                 (first (descriptor/call-args expression))))
                         expected env)

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
                              ;; Typed source arithmetic has a semantic overflow contract: normal
                              ;; Clojure integral arithmetic is checked, while the explicitly
                              ;; `unchecked-*` forms wrap.  Retain that distinction in KernelBody
                              ;; rather than leaving a C-family emitter to choose signed overflow.
                              integral-arithmetic?
                              (and (contains? #{:byte :int :long} operand-type)
                                   (contains? #{:+ :- :*} operator))
                              operand-ranges (mapv :range lowered)
                              proven-range (when integral-arithmetic?
                                             (scalar-range/arithmetic operator operand-ranges))
                              overflow (when integral-arithmetic?
                                         (or (intrinsics/source-overflow-policy semantic-operation)
                                             (when (scalar-range/contained-in-dtype? proven-range operand-type)
                                               :no-overflow)
                                             :trap))
                              options (cond-> {} overflow (assoc :overflow overflow)
                                               (= :no-overflow overflow)
                                               (assoc :proof (assoc proven-range
                                                              :kind :typed-scalar-range)))
                              _ (when-not (every? #(= operand-type (:type %)) lowered)
                                  (decline! :operand-dtype
                                            "scalar intrinsic operands require one dtype"
                                            {:expression expression :operand-type operand-type
                                             :actual (mapv :type lowered)}))
                              result (fresh "value")]
                          (let [range (when (= :no-overflow overflow) proven-range)]
                            (remember-range! result (or range (scalar-range/for-dtype result-type)))
                            {:operations
                             (conj (vec (mapcat :operations lowered))
                                   (body/->ScalarCompute
                                    (body/value result result-type)
                                    (body/scalar-expression operator result-type
                                                            (mapv :result lowered)
                                                            options)))
                             :result result :type result-type
                             :range (or range (scalar-range/for-dtype result-type))})))))

                  :else
                  (decline! :scalar-expression
                            "scalar expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      {:lower lower
       :lower-region
       (fn [{:keys [bindings results]} result-types binding-types env]
         ;; Region locals are evaluated once, in source order. In particular, a product's
         ;; shared combine bindings must not be beta-expanded separately into every result.
         ;; Types are retained facts supplied by the caller, never guessed from a consumer.
         (when-not (and (vector? bindings) (even? (count bindings))
                        (every? symbol? (take-nth 2 bindings))
                        (= (count (take-nth 2 bindings))
                           (count (set (take-nth 2 bindings))))
                        (vector? results) (= (count results) (count result-types)))
           (decline! :scalar-region-shape
                     "scalar region requires distinct ordered bindings and typed results"
                     {:bindings bindings :results results :result-types result-types}))
         (let [checked-lower
               (fn [expression expected local-env]
                 (let [expected (canon-type expected)
                       lowered (lower expression expected local-env)]
                   (when-not (= expected (:type lowered))
                     (decline! :scalar-region-dtype
                               "scalar region value disagrees with its retained dtype"
                               {:expression expression :expected expected
                                :actual (:type lowered)}))
                   lowered))
               {:keys [operations substitutions environment]}
               (reduce
                (fn [{:keys [operations substitutions environment]} [id expression]]
                  (when-not (get binding-types id)
                    (decline! :scalar-region-binding-dtype
                              "scalar region binding lacks a retained dtype"
                              {:binding id :expression expression}))
                  (let [lowered (checked-lower (util/subst-syms substitutions expression)
                                               (get binding-types id) environment)
                        result (:result lowered)
                        ;; References to fresh SSA locals must carry the same retained type as
                        ;; their definition, including when they become storage coordinates.
                        ;; Index lowering must not rediscover this from an enclosing result.
                        ;; Half and internal predicate values have no JVM scalar tag. Their
                        ;; dtype stays in the existing typed SSA environment, not a guessed tag.
                        tag (when-not (= :predicate (:type lowered))
                              (:scalar-tag (dtype/info (:type lowered))))
                        result (if (and (symbol? result) tag)
                                 (with-meta result
                                   (assoc (meta result) :raster.type/tag tag))
                                 result)]
                    {:operations (into operations (:operations lowered))
                     :substitutions (assoc substitutions id result)
                     :environment (cond-> environment
                                    (symbol? result) (assoc result (:type lowered)))}))
                {:operations [] :substitutions {} :environment env}
                (partition 2 bindings))
               lowered-results
               (mapv (fn [expression expected]
                       (checked-lower (util/subst-syms substitutions expression)
                                      expected environment))
                     results result-types)]
           {:operations (into operations (mapcat :operations lowered-results))
            :results (mapv :result lowered-results)
            :types (mapv :type lowered-results)}))})))

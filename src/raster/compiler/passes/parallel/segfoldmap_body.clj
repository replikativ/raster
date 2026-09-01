(ns raster.compiler.passes.parallel.segfoldmap-body
  "Portable KernelBody schedule for ordered segmented fold-map operations.

   One work item owns one independent segment. Each fold is a sequential, loop-carried region in
   declared order; completed fold values feed later folds and the final dense map. This is the
   baseline schedule, not an attention or normalization implementation."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.contraction-body :as contraction-body]))

(def ^:private cast-heads
  {'byte :byte, 'clojure.core/byte :byte
   'int :int, 'clojure.core/int :int
   'long :long, 'clojure.core/long :long
   'float :float, 'clojure.core/float :float
   'double :double, 'clojure.core/double :double})

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data :reason :segfoldmap-kernel-body-declined
                                 :missing-rule rule))))

(defn declined? [exception]
  (= :segfoldmap-kernel-body-declined (:reason (ex-data exception))))

(defn- product-expression [values]
  (case (count values)
    0 1
    1 (first values)
    (apply body/expression :mul values)))

(defn- widen-index-expression
  "Make portable address arithmetic uniformly 64-bit without changing scalar ABI types."
  [expression value-types]
  (cond
    (integer? expression) (body/index-cast expression :long :exact)
    (symbol? expression)
    (if (= :long (dtype/canon (get value-types expression :int)))
      expression
      (body/index-cast expression :long :exact))
    (instance? raster.compiler.ir.kernel_body.IndexExpr expression)
    (apply body/expression (:op expression)
           (map #(widen-index-expression % value-types) (:arguments expression)))
    (instance? raster.compiler.ir.kernel_body.IndexCast expression)
    (if (= :long (dtype/canon (:dtype expression)))
      expression
      (body/index-cast expression :long :exact))
    :else expression))

(defn- inline-let
  [expression]
  (if (and (seq? expression) (contains? #{'let 'let* 'clojure.core/let} (first expression)))
    (let [[_ bindings result] expression]
      (reduce (fn [result [id init]] (util/subst-syms {id init} result))
              result (reverse (partition 2 bindings))))
    expression))

(defn- scalar-lowerer
  [{:keys [array-types scalar-types arrays index-scope lower-index active-mask]}]
  (let [counter (atom 0)
        fresh (fn [prefix] (symbol (str "foldmap-" prefix "-" (swap! counter inc))))
        source-type
        (fn source-type [expression expected env]
          (let [expression (inline-let expression)]
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
              :else expected)))]
    (letfn [(lower [expression expected env]
              (let [expression (inline-let expression)]
                (cond
                  (number? expression)
                  {:operations [] :result (body/literal expression expected) :type expected}

                  (boolean? expression)
                  {:operations [] :result (body/literal expression :predicate) :type :predicate}

                  (symbol? expression)
                  (if-let [actual (or (get env expression) (get scalar-types expression))]
                    {:operations [] :result expression :type actual}
                    (decline! :unbound-scalar
                              "fold-map scalar expression references an undeclared value"
                              {:expression expression :environment (set (keys env))}))

                  (descriptor/aget-call? expression)
                  (let [array (descriptor/aget-array-sym expression)
                        arguments (vec (descriptor/call-args expression))
                        coordinate (last arguments)
                        array-type (some-> (get array-types array) dtype/canon)]
                    (when-not (and (= 2 (count arguments)) (contains? arrays array) array-type)
                      (decline! :indexed-load
                                "fold-map loads require a declared typed stable tensor"
                                {:expression expression :array array :array-types array-types}))
                    (let [id (fresh "load")]
                      {:operations [(body/->ScalarLoad
                                     (body/value id array-type) array
                                     [(lower-index coordinate)] active-mask
                                     (body/literal 0 array-type) :cached)]
                       :result id :type array-type}))

                  (and (seq? expression) (contains? cast-heads (first expression))
                       (= 2 (count expression)))
                  (let [target (dtype/canon (get cast-heads (first expression)))
                        lowered (lower (second expression) target env)]
                    (if (= target (:type lowered))
                      lowered
                      (let [id (fresh "cast")]
                        {:operations (conj (:operations lowered)
                                           (body/->ScalarCompute
                                            (body/value id target)
                                            (body/cast-expression (:result lowered) target
                                                                  :exact :exact)))
                         :result id :type target})))

                  (and (seq? expression) (= 'if (first expression)) (= 4 (count expression)))
                  (let [[_ condition then-expression else-expression] expression
                        condition (lower condition :predicate env)
                        then-value (lower then-expression expected env)
                        else-value (lower else-expression expected env)
                        result-type (:type then-value)
                        _ (when-not (= result-type (:type else-value) expected)
                            (decline! :branch-dtype
                                      "fold-map branches must have one explicit scalar dtype"
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
                                "fold-map scalar expression has no canonical intrinsic"
                                {:expression expression :operator operator}))
                    (let [comparison? (= :cmp (:kind intrinsic))
                          operand-type (if comparison?
                                         (dtype/canon
                                          (or (source-type (first arguments) :int env) :int))
                                         (dtype/canon expected))
                          lowered (mapv #(lower % operand-type env) arguments)
                          result-type (if comparison? :predicate operand-type)
                          _ (when-not (every? #(= operand-type (:type %)) lowered)
                              (decline! :operand-dtype
                                        "fold-map intrinsic operands require one scalar dtype"
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
                            "fold-map scalar expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      {:lower lower})))

(defn lower
  "Apply the portable one-work-item-per-segment schedule to a SegFoldMap."
  [segfold {:keys [workgroup-size array-types scalar-types]
            :or {workgroup-size 256 array-types {} scalar-types {}}}]
  (when-not (instance? raster.compiler.ir.segop.SegFoldMap segfold)
    (throw (ex-info "fold-map KernelBody lowering requires SegFoldMap"
                    {:reason :raster/bug :operation segfold})))
  (let [space (:space segfold)
        segment-dims (segop/seg-space-segment-dims space)
        mapped-dim (segop/seg-space-reduced-dim space)
        _ (when (empty? segment-dims)
            (decline! :no-segments "fold-map requires at least one segment axis"
                      {:operation (:id segfold)}))
        _ (when-not (and (integer? workgroup-size) (pos? workgroup-size))
            (decline! :workgroup-size "fold-map workgroup size must be positive"
                      {:workgroup-size workgroup-size}))
        inputs (vec (sort-by name (:inputs segfold)))
        outputs (vec (:outputs segfold))
        scalars (vec (sort-by name (:scalars segfold)))
        output-dtypes (mapv dtype/canon (:dtypes segfold))
        default-dtype (or (first output-dtypes) :float)
        array-types (into {}
                          (map (fn [id]
                                 [id (dtype/canon (or (get array-types id)
                                                     (get array-types (symbol (name id)))
                                                     default-dtype))]))
                          (concat inputs outputs))
        declared-scalar-types scalar-types
        scalar-types (into {}
                           (map (fn [id]
                                  [id (dtype/canon
                                       (or (get declared-scalar-types id)
                                           (get declared-scalar-types (symbol (name id)))
                                           :int))]))
                           scalars)
        axis-symbols (set (concat (map :name segment-dims) [(:name mapped-dim)]))
        scheduled-indices (conj (set (map #(symbol (str "foldmap-index-" %))
                                          (range (count (:folds segfold)))))
                                'foldmap-map-index)
        index-scope (into (set/union axis-symbols scheduled-indices) scalars)
        index-value-types (merge (zipmap axis-symbols (repeat :long))
                                 (zipmap scheduled-indices (repeat :long))
                                 scalar-types)
        segment-count-source (segop/seg-space-num-segments-expr space)
        lower-index #(widen-index-expression
                      (contraction-body/lower-index % index-scope)
                      index-value-types)
        segment-count (lower-index segment-count-source)
        map-extent (lower-index (:bound mapped-dim))
        total-elements (body/expression :mul segment-count map-extent)
        segment-index 'foldmap-segment
        group-index 'foldmap-group
        local-index 'foldmap-lane
        active-mask :foldmap-active
        decomposition
        (mapv
         (fn [position {:keys [name bound]}]
           (let [following (subvec (vec segment-dims) (inc position))
                 divisor (product-expression (mapv #(lower-index (:bound %)) following))
                 quotient (if (= 1 divisor) segment-index
                              (body/expression :floor-div segment-index divisor))]
             (body/->IndexCompute name
                                  (body/expression :mod quotient (lower-index bound)))))
         (range) segment-dims)
        base-coordinate (body/expression :mul segment-index map-extent)
        parameters
        (vec (concat
              (map (fn [input]
                     (let [input-dtype (get array-types input)]
                       (body/->KernelParameter input :input input-dtype [total-elements] :global
                                               (layout/row-major [total-elements] input-dtype)
                                               :operand)))
                   inputs)
              (map (fn [output output-dtype]
                     (body/->KernelParameter output :output output-dtype [total-elements] :global
                                             (layout/row-major [total-elements] output-dtype)
                                             :result))
                   outputs output-dtypes)
              (map #(body/->KernelParameter % :scalar (get scalar-types %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_nseg :scalar :long [] nil nil :bound)]))
        base-env (merge (zipmap (map :name segment-dims) (repeat :long))
                        {(:index segfold) :long}
                        scalar-types)
        scalar-lower (scalar-lowerer {:array-types array-types :scalar-types scalar-types
                                      :arrays (set inputs) :index-scope index-scope
                                      :lower-index lower-index :active-mask active-mask})
        fold-state
        (reduce
         (fn [{:keys [operations env]} [ordinal fold]]
           (when (seq (:locals fold))
             (decline! :fold-locals
                       "the first portable fold-map schedule requires canonical expression-only folds"
                       {:fold ordinal :locals (:locals fold)}))
           (let [source-index (:index segfold)
                 loop-index (symbol (str "foldmap-index-" ordinal))
                 carry (symbol (str "foldmap-carry-" ordinal))
                 accumulator (:accumulator fold)
                 fold-dtype (dtype/canon (:dtype fold))
                 expression (util/subst-syms {source-index loop-index accumulator carry}
                                             (:step fold))
                 lowered ((:lower scalar-lower) expression fold-dtype
                          (assoc env loop-index :long carry fold-dtype))
                 loop (body/->ForLoop
                       (body/value loop-index :long)
                       (body/index-cast 0 :long :exact) (lower-index (:extent fold)) 1
                       [(body/->LoopArg (body/value carry fold-dtype)
                                       (body/literal (:identity fold) fold-dtype))]
                       (vec (concat (:operations lowered)
                                    [(body/->Yield [(:result lowered)])]))
                       [(body/value accumulator fold-dtype)]
                       {:association :ordered})]
             {:operations (conj operations loop)
              :env (assoc env accumulator fold-dtype)}))
         {:operations [] :env base-env}
         (map-indexed vector (:folds segfold)))
        map-index 'foldmap-map-index
        output-coordinate (body/expression :add base-coordinate map-index)
        map-operations
        (mapcat
         (fn [ordinal output output-dtype expression]
           (let [expression (util/subst-syms {(:index segfold) map-index} expression)
                 lowered ((:lower scalar-lower) expression output-dtype
                          (assoc (:env fold-state) map-index :long))]
             (concat (:operations lowered)
                     [(body/->ScalarStore output [output-coordinate]
                                          (:result lowered) active-mask)])))
         (range) outputs output-dtypes (:map-results segfold))
        final-loop
        (body/->ForLoop (body/value map-index :long)
                        (body/index-cast 0 :long :exact) map-extent 1 []
                        (vec (concat map-operations [(body/->Yield [])])) []
                        {:association :ordered :role :final-map})]
    {:kernel-body
     (body/make
      {:id [:segmented-fold-map (:id segfold) :portable-ordered]
       :parameters parameters
       :stable-reads (mapv body/stable-read inputs)
       :indices (vec (concat
                      [(body/->IndexBinding group-index :group 0)
                       (body/->IndexBinding local-index :local 0)
                       (body/->IndexCompute
                        segment-index
                        (body/index-cast
                         (body/expression :add
                                          (body/expression :mul group-index workgroup-size)
                                          local-index)
                         :long :exact))]
                      decomposition))
       :masks [(body/->Mask active-mask
                            [(body/predicate :lt segment-index '_nseg)])]
       :operations (conj (vec (:operations fold-state)) final-loop)
       :schedule {:strategy :one-work-item-per-segment
                  :association :ordered :workgroup-size workgroup-size
                  :fold-count (count (:folds segfold))}
       :launch (launch/spec {:workgroup-size [workgroup-size]
                             :group-count [(launch/ceil-div segment-count workgroup-size)]})
       :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                    :segop-id (:id segfold)}
       :attributes {:kind :portable-segmented-fold-map
                    :segment-count segment-count :map-extent map-extent
                    :no-write-alias true}})
     :arrays inputs :outputs outputs :scalars scalars
     :segment-count segment-count :map-extent map-extent
     :workgroup-size workgroup-size}))

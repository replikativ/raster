(ns raster.compiler.passes.parallel.segmap-body
  "Portable scalar/control schedule for a one-dimensional typed SegMap.

   One work item owns each logical element. Grid-stride virtualization remains explicit as a
   KernelBody ForLoop; scalar loads, computation, branches, and horizontally fused stores use the
   shared typed scalar-expression lowering boundary."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data :reason :segmap-kernel-body-declined
                                 :missing-rule rule))))

(defn declined?
  [exception]
  (= :segmap-kernel-body-declined (:reason (ex-data exception))))

(defn- widen-index-expression
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

(defn- scalar-forms
  [expression]
  (if (and (seq? expression) (= 'do (first expression)))
    (vec (rest expression))
    [expression]))

(defn- scalar-region
  "Separate the typed top-level local bindings from a SegMap result region.

   TypedSOAC materialization annotates every local ID with its scalar dtype. Keeping these
   bindings explicit lets KernelBody preserve sharing across several result stores."
  [expression]
  (if (and (seq? expression)
           (contains? #{'let 'let* 'clojure.core/let} (first expression)))
    (let [[_ bindings & body] expression]
      (when-not (= 1 (count body))
        (decline! :local-region-shape
                  "map scalar local region requires one result expression"
                  {:expression expression :body-count (count body)}))
      {:bindings (vec (partition 2 bindings)) :result (first body)})
    {:bindings [] :result expression}))

(defn lower
  "Apply a portable grid-stride scalar schedule to a typed one-dimensional SegMap."
  [segmap {:keys [workgroup-size array-types scalar-types]
           :or {workgroup-size 256 array-types {} scalar-types {}}}]
  (when-not (instance? raster.compiler.ir.segop.SegMap segmap)
    (throw (ex-info "map KernelBody lowering requires SegMap"
                    {:reason :raster/bug :operation segmap})))
  (let [space (:space segmap)
        dimensions (:dims space)
        _ (when-not (= 1 (count dimensions))
            (decline! :iteration-rank
                      "the first portable map schedule requires one flattened iteration axis"
                      {:operation (:id segmap) :dimensions dimensions}))
        _ (when-not (and (integer? workgroup-size) (pos? workgroup-size))
            (decline! :workgroup-size "map workgroup size must be positive"
                      {:workgroup-size workgroup-size}))
        index (:name (first dimensions))
        bound (:bound (first dimensions))
        inputs (vec (sort-by name (:inputs segmap)))
        primary-output (:out-sym segmap)
        outputs (if primary-output
                  (vec (concat (sort-by name (disj (:outputs segmap) primary-output))
                               [primary-output]))
                  (vec (sort-by name (:outputs segmap))))
        scalars (vec (sort-by name (:scalars segmap)))
        _ (when (seq (set/intersection (set inputs) (set outputs)))
            (decline! :inout-storage
                      "portable map stable reads must not alias writable results"
                      {:operation (:id segmap) :inputs inputs :outputs outputs}))
        default-dtype (dtype/canon (or (:dtype segmap) :float))
        array-dtype (fn [id]
                      (dtype/canon (or (get array-types id)
                                       (get array-types (symbol (name id)))
                                       default-dtype)))
        scalar-dtype (fn [id]
                       (dtype/canon (or (get scalar-types id)
                                        (get scalar-types (symbol (name id)))
                                        :int)))
        array-types (into {} (map (juxt identity array-dtype)) (concat inputs outputs))
        scalar-types (into {} (map (juxt identity scalar-dtype)) scalars)
        index-scope (conj (set scalars) index)
        index-types (assoc scalar-types index :long)
        lower-index (fn lower-index
                      ([expression] (lower-index expression #{}))
                      ([expression extra-scope]
                       (widen-index-expression
                        (index-expression/lower
                         expression (set/union index-scope extra-scope) decline!)
                        index-types)))
        lowerer (scalar-expression/make-lowerer
                 {:array-types array-types :scalar-types scalar-types
                  :arrays (set inputs) :index-scope index-scope
                  :lower-index lower-index :predicate :map-active
                  :id-prefix "map" :decline! decline!})
        {:keys [bindings result]} (scalar-region (:lambda segmap))
        base-environment (assoc scalar-types index :long)
        local-state
        (reduce
         (fn [{:keys [substitutions operations environment]} [id init]]
           (let [tag (or (:raster.type/tag (meta id)) (:tag (meta id)))
                 local-type (some-> tag dtype/dtype-for-scalar-tag dtype/canon)
                 _ (when-not local-type
                     (decline! :local-dtype
                               "map scalar local is missing its TypedSOAC dtype"
                               {:operation (:id segmap) :local id :metadata (meta id)}))
                 init (util/subst-syms substitutions init)
                 lowered ((:lower lowerer) init local-type environment)]
             {:substitutions (assoc substitutions id (:result lowered))
              :operations (into operations (:operations lowered))
              :environment (assoc environment (:result lowered) (:type lowered))}))
         {:substitutions {} :operations [] :environment base-environment}
         bindings)
        forms (scalar-forms (util/subst-syms (:substitutions local-state) result))
        explicit-forms (if primary-output (pop forms) forms)
        primary-form (when primary-output (peek forms))
        _ (when (and primary-output (empty? forms))
            (decline! :missing-result "map result has no scalar expression"
                      {:operation (:id segmap) :output primary-output}))
        environment (:environment local-state)
        lower-store
        (fn [form]
          (when-not (descriptor/aset-call? form)
            (decline! :explicit-store
                      "map side-effect region contains a non-store statement"
                      {:operation (:id segmap) :statement form}))
          (let [[array coordinate expression] (take-last 3 (descriptor/call-args form))
                array (descriptor/aset-array-sym form)
                _ (when-not (contains? (set outputs) array)
                    (decline! :explicit-store-target
                              "map store targets an undeclared result"
                              {:operation (:id segmap) :statement form :target array}))
                lowered ((:lower lowerer) expression (get array-types array) environment)]
            (concat (:operations lowered)
                    [(body/->ScalarStore array [(lower-index coordinate)]
                                         (:result lowered) :map-active)])))
        explicit-operations (vec (mapcat lower-store explicit-forms))
        primary-lowered (when primary-output
                          ((:lower lowerer) primary-form
                                            (get array-types primary-output) environment))
        scalar-operations
        (vec (concat (:operations local-state)
                     explicit-operations
                     (:operations primary-lowered)
                     (when primary-output
                       [(body/->ScalarStore primary-output [index]
                                            (:result primary-lowered) :map-active)])))
        group-index 'map-group
        local-index 'map-lane
        parameters
        (vec (concat
              (map #(body/->KernelParameter
                     % :input (get array-types %) ['_n_bound] :global
                     (layout/row-major ['_n_bound] (get array-types %)) :operand)
                   inputs)
              (map #(body/->KernelParameter
                     % :output (get array-types %) ['_n_bound] :global
                     (layout/row-major ['_n_bound] (get array-types %)) :result)
                   outputs)
              (map #(body/->KernelParameter % :scalar (get scalar-types %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))]
    {:kernel-body
     (body/make
      {:id [:segmap (:id segmap) :portable-one-item]
       :parameters parameters
       :stable-reads (mapv body/stable-read inputs)
       :indices [(body/->IndexBinding group-index :group 0)
                 (body/->IndexBinding local-index :local 0)
                 (body/->IndexCompute
                  index
                  (body/index-cast
                   (body/expression :add
                                    (body/expression :mul group-index workgroup-size)
                                    local-index)
                   :long :exact))]
       :masks [(body/->Mask
                :map-active
                [(body/predicate :lt index (body/index-cast '_n_bound :long :exact))])]
       :operations scalar-operations
       :schedule {:strategy :one-work-item-per-element :association :independent
                  :workgroup-size workgroup-size}
       :launch (launch/spec {:workgroup-size [workgroup-size]
                             :group-count [(launch/ceil-div bound workgroup-size)]})
       :provenance {:dialect :kernel-body :source-dialect :segmap
                    :segop-id (:id segmap)}
       :attributes {:kind :portable-segmap :extent bound :no-write-alias true}})
     :bound bound :inputs inputs :outputs outputs :scalars scalars}))

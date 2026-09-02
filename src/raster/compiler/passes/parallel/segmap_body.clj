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

(defn- contains-indexed-load?
  [expression]
  (boolean (some descriptor/aget-call? (tree-seq coll? seq expression))))

(defn- scalar-region
  "Read typed local SSA directly from a scheduled SegMap.

   Direct TypedSOAC lowering carries the authoritative local dtype beside each definition. The
   source-shaped lambda remains only for compatibility-created SegMaps; its binder metadata is
   projected once here and is never used to infer an arithmetic function or result type."
  [segmap]
  (if-let [{:keys [locals result] :as region} (:scalar-region segmap)]
    (do
      (when-not (and (vector? locals)
                     (every? #(and (symbol? (:id %)) (:dtype %) (contains? % :init)) locals)
                     (contains? region :result))
        (decline! :typed-local-region
                  "scheduled typed map carries a malformed scalar region"
                  {:operation (:id segmap) :scalar-region region}))
      {:locals locals :result result})
    (let [expression (:lambda segmap)]
      (if (and (seq? expression)
               (contains? #{'let 'let* 'clojure.core/let} (first expression)))
        (let [[_ bindings & body] expression]
          (when-not (= 1 (count body))
            (decline! :local-region-shape
                      "map scalar local region requires one result expression"
                      {:expression expression :body-count (count body)}))
          {:locals
           (mapv (fn [[id init]]
                   {:id id
                    :dtype (some-> (or (:raster.type/tag (meta id)) (:tag (meta id)))
                                   dtype/dtype-for-scalar-tag dtype/canon)
                    :init init})
                 (partition 2 bindings))
           :result (first body)})
        {:locals [] :result expression}))))

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
        output-set (set outputs)
        inout (set/intersection (set inputs) output-set)
        read-only-inputs (vec (remove inout inputs))
        scalars (vec (sort-by name (:scalars segmap)))
        explicit-certified-write?
        (and (nil? primary-output)
             (contains? #{:unique :reduce} (:write-conflict segmap)))
        _ (when (and (seq inout) (not explicit-certified-write?))
            (decline! :inout-storage
                      "portable dense map stable reads must not alias writable results"
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
        {:keys [locals result]} (scalar-region segmap)
        base-environment (assoc scalar-types index :long)
        local-state
        (reduce
         (fn [{:keys [substitutions operations environment]}
              {:keys [id init] local-dtype :dtype}]
           (let [local-type (some-> local-dtype dtype/canon)
                 _ (when-not local-type
                     (decline! :local-dtype
                               "map scalar local is missing its scheduled dtype"
                               {:operation (:id segmap) :local id
                                :scalar-region (:scalar-region segmap)}))
                 init (util/subst-syms substitutions init)
                 lowered ((:lower lowerer) init local-type environment)]
             {:substitutions (assoc substitutions id (:result lowered))
              :operations (into operations (:operations lowered))
              :environment (assoc environment (:result lowered) (:type lowered))}))
         {:substitutions {} :operations [] :environment base-environment}
         locals)
        forms (scalar-forms (util/subst-syms (:substitutions local-state) result))
        explicit-forms (if primary-output (pop forms) forms)
        primary-form (when primary-output (peek forms))
        _ (when (and primary-output (empty? forms))
            (decline! :missing-result "map result has no scalar expression"
                      {:operation (:id segmap) :output primary-output}))
        environment (:environment local-state)
        lower-store
        (fn [form]
          (let [atomic-add? (and (seq? form)
                                 (= 'raster.par/atomic-add!
                                    (descriptor/semantic-op form)))]
            (when-not (or (descriptor/aset-call? form) atomic-add?)
              (decline! :explicit-store
                        "map side-effect region contains a non-store statement"
                        {:operation (:id segmap) :statement form}))
            (let [[array coordinate expression] (take-last 3 (descriptor/call-args form))
                  array (if atomic-add? array (descriptor/aset-array-sym form))
                  _ (when-not (contains? (set outputs) array)
                      (decline! :explicit-store-target
                                "map store targets an undeclared result"
                                {:operation (:id segmap) :statement form :target array}))
                  coordinate-value (when (contains-indexed-load? coordinate)
                                     ((:lower lowerer) coordinate :long environment))
                  coordinate-expression (if coordinate-value
                                          (:result coordinate-value)
                                          (lower-index coordinate))
                  lowered ((:lower lowerer) expression (get array-types array) environment)]
              (concat (:operations coordinate-value)
                      (:operations lowered)
                      [(if atomic-add?
                         (body/->AtomicRMW array [coordinate-expression]
                                           (:result lowered) :+ :map-active)
                         (body/->ScalarStore array [coordinate-expression]
                                             (:result lowered) :map-active))]))))
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
                   read-only-inputs)
              (map #(body/->KernelParameter
                     % (if (contains? inout %) :inout :output)
                     (get array-types %) ['_n_bound] :global
                     (layout/row-major ['_n_bound] (get array-types %)) :result)
                   outputs)
              (map #(body/->KernelParameter % :scalar (get scalar-types %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))]
    {:kernel-body
     (body/make
      {:id [:segmap (:id segmap) :portable-one-item]
       :parameters parameters
       :stable-reads (mapv body/stable-read read-only-inputs)
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
     :bound bound :inputs read-only-inputs :outputs outputs :scalars scalars}))

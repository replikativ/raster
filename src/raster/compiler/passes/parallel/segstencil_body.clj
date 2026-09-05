(ns raster.compiler.passes.parallel.segstencil-body
  "Portable KernelBody schedule for certified one-dimensional boundary stencils."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data :reason :segstencil-kernel-body-declined
                                 :missing-rule rule :fallback :none))))

(defn declined?
  [exception]
  (= :segstencil-kernel-body-declined (:reason (ex-data exception))))

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

(defn lower
  "Select a one-item-per-element portable schedule for a proved Dirichlet stencil.

   The interior branch dominates all neighborhood loads. Boundary lanes yield the typed zero
   value and perform one final store, so no target emitter reconstructs boundary semantics."
  [stencil {:keys [workgroup-size array-types scalar-types]
            :or {workgroup-size 256 array-types {} scalar-types {}}}]
  (when-not (instance? raster.compiler.ir.segop.SegStencil stencil)
    (throw (ex-info "stencil KernelBody lowering requires SegStencil"
                    {:reason :raster/bug :operation stencil})))
  (let [dimensions (get-in stencil [:space :dims])
        _ (when-not (= 1 (count dimensions))
            (decline! :iteration-rank
                      "portable stencil requires one flattened iteration axis"
                      {:operation (:id stencil) :dimensions dimensions}))
        _ (when-not (and (= :dirichlet (:boundary stencil))
                         (= 1 (:radius stencil))
                         (= :no-write-alias (:aliasing stencil)))
            (decline! :boundary-contract
                      "portable stencil requires the certified radius-one Dirichlet contract"
                      {:operation (:id stencil) :radius (:radius stencil)
                       :boundary (:boundary stencil) :aliasing (:aliasing stencil)}))
        _ (when-not (and (integer? workgroup-size) (pos? workgroup-size))
            (decline! :workgroup-size "stencil workgroup size must be positive"
                      {:workgroup-size workgroup-size}))
        index (:name (first dimensions))
        bound (:bound (first dimensions))
        radius (:radius stencil)
        inputs (vec (sort-by name (:inputs stencil)))
        outputs (vec (sort-by name (:outputs stencil)))
        output (:out-sym stencil)
        scalars (vec (sort-by name (:scalars stencil)))
        _ (when-not (and (= 1 (count outputs)) (= output (first outputs))
                         (empty? (set/intersection (set inputs) (set outputs))))
            (decline! :storage-contract
                      "portable stencil requires one distinct physical result"
                      {:operation (:id stencil) :inputs inputs :outputs outputs :output output}))
        result-type (dtype/canon (:dtype stencil))
        array-dtype (fn [id]
                      (dtype/canon (or (get array-types id)
                                       (get array-types (symbol (name id)))
                                       result-type)))
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
                  :lower-index lower-index :predicate nil
                  :id-prefix "stencil" :decline! decline!})
        lowered ((:lower lowerer) (:lambda stencil) result-type
                                    (assoc scalar-types index :long))
        group-index 'stencil-group
        local-index 'stencil-lane
        right-limit 'stencil-right-limit
        left-interior 'stencil-left-interior
        right-interior 'stencil-right-interior
        right-result 'stencil-right-result
        result 'stencil-result
        parameters
        (vec
         (concat
          (map #(body/->KernelParameter
                 % :input (get array-types %) ['_n_bound] :global
                 (layout/row-major ['_n_bound] (get array-types %)) :operand)
               inputs)
          [(body/->KernelParameter
            output :output result-type ['_n_bound] :global
            (layout/row-major ['_n_bound] result-type) :result)]
          (map #(body/->KernelParameter % :scalar (get scalar-types %) [] nil nil :parameter)
               scalars)
          [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))]
    {:kernel-body
     (body/make
      {:id [:segstencil (:id stencil) :portable-one-item]
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
                :stencil-active
                [(body/predicate :lt index (body/index-cast '_n_bound :long :exact))])]
       :operations
       [(body/->ScalarCompute
         (body/value right-limit :long)
         (body/scalar-expression
          :- :long [(body/cast-expression '_n_bound :long :exact :exact)
                    (body/literal radius :long)]
          ;; `_n_bound` is a non-negative int extent and radius is a validated static stencil
          ;; radius, so this widened schedule-bound subtraction cannot overflow long.
          {:overflow :no-overflow}))
        (body/->ScalarCompute
         (body/value left-interior :predicate)
         (body/scalar-expression :le :predicate
                                 [(body/literal radius :long) index]))
        (body/->ScalarCompute
         (body/value right-interior :predicate)
         (body/scalar-expression :lt :predicate [index right-limit]))
        (body/->IfRegion
         left-interior
         [(body/->IfRegion
           right-interior
           (conj (vec (:operations lowered)) (body/->Yield [(:result lowered)]))
           [(body/->Yield [(body/literal 0 result-type)])]
           [(body/value right-result result-type)])
          (body/->Yield [right-result])]
         [(body/->Yield [(body/literal 0 result-type)])]
         [(body/value result result-type)])
        (body/->ScalarStore output [index] result :stencil-active)]
       :schedule {:strategy :one-work-item-per-element :association :independent
                  :workgroup-size workgroup-size :boundary :dirichlet :radius radius}
       :launch (launch/spec {:workgroup-size [workgroup-size]
                             :group-count [(launch/ceil-div '_n_bound workgroup-size)]})
       :provenance {:dialect :kernel-body :source-dialect :segstencil
                    :segop-id (:id stencil)}
       :attributes {:kind :portable-segstencil :extent bound
                    :boundary :dirichlet :radius radius :no-write-alias true}})
     :bound bound :inputs inputs :outputs outputs :scalars scalars}))

(defn schedule
  "Refine one SegStencil into a complete, target-neutral ScheduledKernelBody.

   The exact SegStencil remains the semantic source. The schedule fixes radius-one Dirichlet
   guards, stable neighborhood reads, one independent result store, scalar order, and any checked
   logical-to-physical bound conversion before target emission."
  [stencil {:keys [scalar-types] :as options}]
  (let [{:keys [kernel-body bound inputs outputs scalars]} (lower stencil options)
        parameters (:parameters kernel-body)
        arguments (mapv (fn [parameter]
                          (if (= '_n_bound (:id parameter)) bound (:id parameter)))
                        parameters)
        logical-bound-dtype
        (some-> (or (get scalar-types bound)
                    (get scalar-types (when (or (symbol? bound) (keyword? bound))
                                        (symbol (name bound)))))
                dtype/canon)
        scalar-bindings
        (mapv (fn [[parameter argument]]
                (let [kernel-dtype (dtype/canon (:dtype parameter))
                      logical-dtype (if (and (= '_n_bound (:id parameter))
                                             (contains? #{:int :long} logical-bound-dtype))
                                      logical-bound-dtype
                                      kernel-dtype)]
                  {:parameter (:id parameter)
                   :value argument
                   :dtype logical-dtype
                   :kernel-dtype kernel-dtype
                   :conversion (if (= logical-dtype kernel-dtype)
                                 :identity
                                 :checked-range)}))
              (filterv (fn [[parameter _]] (= :scalar (:kind parameter)))
                       (map vector parameters arguments)))]
    (scheduled-body/make
     {:source stencil
      :body kernel-body
      :arguments arguments
      :scalar-bindings scalar-bindings
      :effects {:kind :stencil
                :uses (scheduled-body/derive-uses kernel-body arguments)
                :boundary (:boundary stencil)
                :radius (:radius stencil)}
      :legality {:kind :segstencil-body-lowering
                 :iteration-rank 1
                 :boundary :dirichlet
                 :radius 1
                 :aliasing :no-write-alias}
      :numerics {:mode :exact :policy :same-scalar-evaluation-order}
      :provenance {:dialect :kernel-body :source-dialect :segstencil
                   :segop-id (:id stencil)}
      :attributes {:array-params (vec (concat inputs outputs))
                   :scalar-params scalars
                   :dtype (:dtype stencil)
                   :boundary (:boundary stencil)
                   :radius (:radius stencil)
                   :aliasing :no-write-alias}})))

(ns raster.compiler.passes.parallel.segmap-body
  "Portable scalar/control schedule for a one-dimensional typed SegMap.

   One work item owns each logical element. Grid-stride virtualization remains explicit as a
   KernelBody ForLoop; scalar loads, computation, branches, and horizontally fused stores use the
   shared typed scalar-expression lowering boundary."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
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
  (if-let [{:keys [locals result effects iteration-order] :as region} (:scalar-region segmap)]
    (do
      (when-not (and (vector? locals)
                     (every? #(and (symbol? (:id %)) (:dtype %) (contains? % :init)) locals)
                     (not= (contains? region :result) (contains? region :effects))
                     (or (not (contains? region :effects))
                         (and (vector? effects) (seq effects))))
        (decline! :typed-local-region
                  "scheduled typed map carries a malformed scalar region"
                  {:operation (:id segmap) :scalar-region region}))
      (cond-> {:locals locals}
        (contains? region :result) (assoc :result result)
        (contains? region :effects) (assoc :effects effects)
        (contains? region :iteration-order) (assoc :iteration-order iteration-order)))
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

(defn- same-element-inout?
  "Prove that every read of a writable result is owned by the same logical work item.

   A same-index read followed by a same-index write is ordinary private lane state, not a
   cross-lane alias hazard.  Shifted/indirect reads remain outside the portable SegMap schedule;
   they require a stencil, scatter, or an explicitly ordered schedule."
  [inout index expressions]
  (every? (fn [{:keys [sym idx]}]
            (or (not (contains? inout sym))
                (= index (descriptor/unwrap-int-cast idx))))
          (mapcat descriptor/aget-reads expressions)))

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
        write-conflicts (or (:write-conflicts segmap) {})
        reduction-destinations
        (into #{} (keep (fn [[destination conflict]]
                          (when (= :reduce (:kind conflict)) destination)))
              write-conflicts)
        inout (set/union (set/intersection (set inputs) output-set)
                         reduction-destinations)
        read-only-inputs (vec (remove inout inputs))
        scalars (vec (sort-by name (:scalars segmap)))
        {:keys [locals result effects iteration-order]} (scalar-region segmap)
        ordered-effects? (seq effects)
        sequential-effects? (and ordered-effects? (= :sequential iteration-order))
        explicit-certified-write?
        (and (nil? primary-output)
             (contains? #{:unique :reduce} (:write-conflict segmap)))
        same-element-inout?
        (same-element-inout? inout index (conj (mapv :init locals) result))
        _ (when (and (seq inout)
                     (not (or explicit-certified-write? ordered-effects?
                              same-element-inout?)))
            (decline! :inout-storage
                      "portable dense map writable results may only read their lane-owned element"
                      {:operation (:id segmap) :inputs inputs :outputs outputs}))
        default-dtype (dtype/canon (or (:dtype segmap) :float))
        array-dtype (fn [id]
                      (dtype/canon (or (get array-types id)
                                       (get array-types (symbol (name id)))
                                       default-dtype)))
        scalar-dtype (fn [id]
                       (if-let [declared (or (get scalar-types id)
                                             (get scalar-types (symbol (name id))))]
                         (dtype/canon declared)
                         ;; A guessed integer here silently zeroed a captured floating scalar
                         ;; once; the owner (params, program values or walker tags) must say.
                         (decline! :scalar-dtype
                                   "portable map scalar parameter has no declared dtype"
                                   {:operation (:id segmap) :scalar id
                                    :declared (vec (keys scalar-types))})))
        array-types (into {} (map (juxt identity array-dtype)) (concat inputs outputs))
        scalar-types (into {} (map (juxt identity scalar-dtype)) scalars)
        index-scope (conj (set scalars) index)
        index-types (assoc scalar-types index :long)
        lower-index (fn lower-index
                      ([expression] (lower-index expression #{}))
                      ([expression extra-scope]
                       (lower-index expression extra-scope {}))
                      ([expression extra-scope extra-types]
                       (widen-index-expression
                        (index-expression/lower
                         expression (set/union index-scope extra-scope) decline!)
                        (merge index-types extra-types))))
        lowerer (scalar-expression/make-lowerer
                 {:array-types array-types :scalar-types scalar-types
                  :arrays (set inputs) :index-scope index-scope
                  :lower-index lower-index :predicate :map-active
                  :id-prefix "map" :decline! decline!})
        base-environment (assoc scalar-types index :long)
        lower-locals
        (fn [locals base-environment]
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
         locals))
        local-state (lower-locals locals base-environment)
        forms (if ordered-effects?
                []
                (scalar-forms (util/subst-syms (:substitutions local-state) result)))
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
                                          (lower-index coordinate (set (keys environment))
                                                       environment))
                  lowered ((:lower lowerer) expression (get array-types array) environment)]
              (concat (:operations coordinate-value)
                      (:operations lowered)
                      [(if atomic-add?
                         (body/->AtomicRMW array [coordinate-expression]
                                           (:result lowered) :+ :map-active)
                         (body/->ScalarStore array [coordinate-expression]
                                             (:result lowered) :map-active))]))))
        explicit-operations (vec (mapcat lower-store explicit-forms))
        substitute-effect
        (fn substitute-effect [substitutions effect]
          (let [substitute #(util/subst-syms substitutions %)]
            (if-let [loop (:loop effect)]
              (assoc effect :loop
                     (-> loop
                         (update :extent substitute)
                         (update :locals (fn [locals] (mapv #(update % :init substitute) locals)))
                         (update :effects (fn [effects]
                                            (mapv #(substitute-effect substitutions %) effects)))))
              (reduce (fn [effect field] (update effect field substitute))
                      effect [:destination :destination-index :predicate :value]))))
        lower-effect
        (fn lower-effect
          [environment {:keys [destination conflict destination-index predicate value] :as effect}]
          (if-let [{loop-index :index loop-locals :locals loop-effects :effects
                    :keys [lower extent]} (:loop effect)]
            ;; A counted store loop lowers to an ordered ForLoop nested in the work item: its
            ;; locals are SSA values scoped to one iteration and its stores keep their own
            ;; per-destination contracts.
            (let [loop-state (lower-locals loop-locals (assoc environment loop-index :long))
                  inner (vec (mapcat #(lower-effect (:environment loop-state)
                                                    (substitute-effect
                                                     (:substitutions loop-state) %))
                                     loop-effects))]
              [(body/->ForLoop
                (body/value loop-index :long)
                (lower-index lower (set (keys environment)) environment)
                (lower-index extent (set (keys environment)) environment)
                1
                []
                (vec (concat (:operations loop-state) inner [(body/->Yield [])]))
                []
                {:association :ordered :source-order true})])
          (do
          (when-not (contains? output-set destination)
            (decline! :effect-destination
                      "ordered effect targets an undeclared result"
                      {:operation (:id segmap) :effect effect}))
          (let [coordinate-value (when (contains-indexed-load? destination-index)
                                   ((:lower lowerer) destination-index :long environment))
                coordinate-expression (if coordinate-value
                                        (:result coordinate-value)
                                        (lower-index destination-index
                                                     (set (keys environment)) environment))
                lowered-value ((:lower lowerer) value (get array-types destination) environment)
                operator (when (= :reduce (:kind conflict))
                           (intrinsics/canonical (:operator conflict)))
                _ (when (and (= :reduce (:kind conflict)) (nil? operator))
                    (decline! :effect-reduction-operator
                              "ordered reduction effect has no canonical scalar operator"
                              {:operation (:id segmap) :effect effect}))
                store (if (= :reduce (:kind conflict))
                        (body/->AtomicRMW destination [coordinate-expression]
                                          (:result lowered-value) operator :map-active)
                        (body/->ScalarStore destination [coordinate-expression]
                                            (:result lowered-value) :map-active))
                effect-operations (vec (concat (:operations coordinate-value)
                                               (:operations lowered-value) [store]))]
            (if (contains? #{true 1} predicate)
              effect-operations
              (let [lowered-predicate ((:lower lowerer) predicate :predicate environment)]
                (vec (concat (:operations lowered-predicate)
                             [(body/->IfRegion (:result lowered-predicate)
                                               (conj effect-operations (body/->Yield []))
                                               [(body/->Yield [])] [])]))))))))
        effect-operations
        (vec (mapcat #(lower-effect environment
                                    (substitute-effect (:substitutions local-state) %))
                     effects))
        primary-lowered (when primary-output
                          ((:lower lowerer) primary-form
                                            (get array-types primary-output) environment))
        scalar-operations
        (vec (concat (:operations local-state)
                     (if ordered-effects? effect-operations explicit-operations)
                     (:operations primary-lowered)
                     (when primary-output
                       [(body/->ScalarStore primary-output [index]
                                            (:result primary-lowered) :map-active)])))
        group-index 'map-group
        local-index 'map-lane
        launch-index (if sequential-effects? 'effect-launch index)
        scheduled-operations
        (if sequential-effects?
          [(body/->ForLoop
            (body/value index :long)
            (body/index-cast 0 :long :exact)
            (body/index-cast '_n_bound :long :exact)
            1
            []
            (conj scalar-operations (body/->Yield []))
            []
            {:association :ordered :source-order true})]
          scalar-operations)
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
                  launch-index
                  (body/index-cast
                   (body/expression :add
                                    (body/expression :mul group-index workgroup-size)
                                    local-index)
                   :long :exact))]
       :masks [(body/->Mask
                :map-active
                [(if sequential-effects?
                   (body/predicate :eq launch-index 0)
                   (body/predicate :lt index (body/index-cast '_n_bound :long :exact)))])]
       :operations scheduled-operations
       :schedule {:strategy (if sequential-effects?
                              :one-work-item-ordered-loop
                              :one-work-item-per-element)
                  :association (if sequential-effects? :ordered :independent)
                  :workgroup-size workgroup-size}
       :launch (if sequential-effects?
                 (launch/spec {:workgroup-size [1] :group-count [1]})
                 (launch/spec {:workgroup-size [workgroup-size]
                               :group-count [(launch/ceil-div bound workgroup-size)]}))
       :provenance {:dialect :kernel-body :source-dialect :segmap
                    :segop-id (:id segmap)}
       :attributes {:kind :portable-segmap :extent bound :no-write-alias true
                    :effect-iteration-order iteration-order}})
     :bound bound :inputs read-only-inputs :outputs outputs :scalars scalars}))

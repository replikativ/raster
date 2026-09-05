(ns raster.compiler.passes.parallel.segfoldmap-body
  "Portable KernelBody schedule for ordered segmented fold-map operations.

   One work item owns one independent segment. Each fold is a sequential, loop-carried region in
   declared order; completed fold values feed later folds and the final dense map. This is the
   baseline schedule, not an attention or normalization implementation."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data :reason :segfoldmap-kernel-body-declined
                                 :missing-rule rule :fallback :none))))

(defn declined? [exception]
  (= :segfoldmap-kernel-body-declined (:reason (ex-data exception))))

(defn- product-expression [values]
  (case (count values)
    0 1
    1 (first values)
    (apply body/expression :mul values)))

(defn- static-index-integer
  [expression]
  (cond
    (integer? expression) expression
    (instance? raster.compiler.ir.kernel_body.IndexCast expression)
    (static-index-integer (:argument expression))
    :else nil))

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

(defn lower
  "Apply the portable one-work-item-per-segment schedule to a SegFoldMap."
  [segfold {:keys [workgroup-size array-types scalar-types]
            :or {array-types {} scalar-types {}}}]
  (when-not (instance? raster.compiler.ir.segop.SegFoldMap segfold)
    (throw (ex-info "fold-map KernelBody lowering requires SegFoldMap"
                    {:reason :raster/bug :operation segfold})))
  (let [space (:space segfold)
        source-grid (:grid segfold)
        source-workgroup-size (:block-size source-grid)
        workgroup-size (or workgroup-size source-workgroup-size)
        segment-dims (segop/seg-space-segment-dims space)
        mapped-dim (segop/seg-space-reduced-dim space)
        _ (when (empty? segment-dims)
            (decline! :no-segments "fold-map requires at least one segment axis"
                      {:operation (:id segfold)}))
        _ (when-not (and (integer? workgroup-size) (pos? workgroup-size))
            (decline! :workgroup-size "fold-map workgroup size must be positive"
                      {:workgroup-size workgroup-size}))
        _ (when-not (= source-workgroup-size workgroup-size)
            (decline! :source-grid
                      "fold-map KernelBody must preserve its source KernelGrid block size"
                      {:source-grid source-grid :requested-workgroup-size workgroup-size}))
        _ (when-not (zero? (:shared-mem-bytes source-grid))
            (decline! :source-grid-shared-memory
                      "fold-map has no modeled workgroup allocation for source-grid shared memory"
                      {:source-grid source-grid}))
        inputs (vec (sort-by name (:inputs segfold)))
        outputs (vec (:outputs segfold))
        scalars (vec (sort-by name (:scalars segfold)))
        _ (when-not (= :no-write-alias (:aliasing segfold))
            (decline! :aliasing-contract
                      "fold-map requires a no-write-alias source contract"
                      {:operation (:id segfold) :aliasing (:aliasing segfold)}))
        _ (when (seq (set/intersection (set inputs) (set outputs)))
            (decline! :storage-contract
                      "fold-map stable inputs must be disjoint from its destinations"
                      {:operation (:id segfold) :inputs inputs :outputs outputs}))
        _ (when-not (= (count outputs) (count (distinct outputs)))
            (decline! :distinct-outputs
                      "fold-map destinations must have distinct identities"
                      {:operation (:id segfold) :outputs outputs}))
        _ (when-not (and (seq outputs) (seq (:dtypes segfold)) (seq (:map-results segfold))
                         (= (count outputs) (count (:dtypes segfold))
                            (count (:map-results segfold))))
            (decline! :result-contract
                      "fold-map requires aligned non-empty outputs, dtypes, and map results"
                      {:operation (:id segfold) :outputs outputs
                       :dtypes (:dtypes segfold) :map-results (:map-results segfold)}))
        _ (when-not (= (:extent segfold) (:bound mapped-dim))
            (decline! :mapped-extent
                      "fold-map semantic extent must equal its reduced-space bound"
                      {:operation (:id segfold) :extent (:extent segfold)
                       :reduced-bound (:bound mapped-dim)}))
        _ (when-let [[ordinal fold]
                     (first (remove (fn [[_ fold]] (= :ordered (:association fold)))
                                    (map-indexed vector (:folds segfold))))]
            (decline! :fold-association
                      "every fold-map accumulator must retain declared order"
                      {:operation (:id segfold) :fold ordinal
                       :association (:association fold)}))
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
                                'foldmap-map-index
                                'foldmap-segment)
        index-scope (into (set/union axis-symbols scheduled-indices) scalars)
        index-value-types (merge (zipmap axis-symbols (repeat :long))
                                 (zipmap scheduled-indices (repeat :long))
                                 scalar-types)
        segment-count-source (segop/seg-space-num-segments-expr space)
        lower-index (fn lower-index
                      ([expression] (lower-index expression #{}))
                      ([expression extra-scope]
                       (widen-index-expression
                        (index-expression/lower
                         expression (set/union index-scope extra-scope) decline!)
                        index-value-types)))
        segment-count (lower-index segment-count-source)
        map-extent (lower-index (:bound mapped-dim))
        total-elements (body/expression :mul segment-count map-extent)
        grid-group-count-index (lower-index (:num-blocks source-grid))
        grid-group-count (index-expression/to-launch-expression grid-group-count-index decline!)
        expected-grid-ceil (body/expression :ceil-div segment-count
                                            (lower-index workgroup-size))
        _ (when-not (and (instance? raster.compiler.ir.kernel_body.IndexExpr
                                    grid-group-count-index)
                         (= :min (:op grid-group-count-index))
                         (= 2 (count (:arguments grid-group-count-index)))
                         (some-> (first (:arguments grid-group-count-index))
                                 static-index-integer pos?)
                         (= expected-grid-ceil (second (:arguments grid-group-count-index))))
            (decline! :source-grid-count
                      "fold-map source KernelGrid must retain its capped ceil-div segment launch"
                      {:source-grid source-grid :segment-count segment-count
                       :expected-ceil expected-grid-ceil
                       :lowered-group-count grid-group-count-index}))
        segment-count-launch (index-expression/to-launch-expression segment-count decline!)
        _ (when-not (set/subset? (launch/expression-references grid-group-count)
                                 (launch/expression-references segment-count-launch))
            (decline! :source-grid-scalar-closure
                      "fold-map source KernelGrid references values outside its segment extent"
                      {:source-grid source-grid
                       :grid-scalars (launch/expression-references grid-group-count)
                       :segment-scalars (launch/expression-references segment-count-launch)}))
        segment-index 'foldmap-segment
        first-segment 'foldmap-first-segment
        group-index 'foldmap-group
        group-count 'foldmap-group-count
        local-index 'foldmap-lane
        ;; A grid-stride work item advances `segment-index` each iteration, so logical segment
        ;; axes must be recomputed *inside* that loop.  Keep them as named IndexCompute values:
        ;; ScalarExpr consumes typed SSA identities, never raw IndexExpr trees.
        segment-axis-computes
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
        scalar-lower (scalar-expression/make-lowerer
                      {:array-types array-types :scalar-types scalar-types
                       :arrays (set inputs) :index-scope index-scope
                       :lower-index lower-index :predicate nil
                       :id-prefix "foldmap" :decline! decline!})
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
         (fn [_ordinal output output-dtype expression]
           (let [expression (util/subst-syms {(:index segfold) map-index} expression)
                 lowered ((:lower scalar-lower) expression output-dtype
                                                (assoc (:env fold-state) map-index :long))]
             (concat (:operations lowered)
                     [(body/->ScalarStore output [output-coordinate]
                                          (:result lowered) nil)])))
         (range) outputs output-dtypes (:map-results segfold))
        final-loop
        (body/->ForLoop (body/value map-index :long)
                        (body/index-cast 0 :long :exact) map-extent 1 []
                        (vec (concat map-operations [(body/->Yield [])])) []
                        {:association :ordered :role :final-map})
        segment-step
        (body/expression
         :mul
         (body/index-cast group-count :long :exact)
         (body/index-cast workgroup-size :long :exact))
        segment-loop
        (body/->ForLoop
         (body/value segment-index :long) first-segment '_nseg segment-step []
         (vec (concat segment-axis-computes (:operations fold-state)
                      [final-loop (body/->Yield [])])) []
         {:association :independent :role :segment-grid-stride})]
    {:kernel-body
     (body/make
      {:id [:segmented-fold-map (:id segfold) :portable-ordered]
       :parameters parameters
       :stable-reads (mapv body/stable-read inputs)
       :indices [(body/->IndexBinding group-index :group 0)
                 (body/->IndexBinding group-count :group-count 0)
                 (body/->IndexBinding local-index :local 0)
                 (body/->IndexCompute
                  first-segment
                  (body/expression
                   :add
                   (body/expression :mul
                                    (body/index-cast group-index :long :exact)
                                    (body/index-cast workgroup-size :long :exact))
                   (body/index-cast local-index :long :exact)))]
       :masks []
       :operations [segment-loop]
       :schedule {:strategy :grid-stride-one-work-item-per-segment
                  :association :ordered :workgroup-size workgroup-size
                  :fold-count (count (:folds segfold))
                  :source-grid source-grid}
       :launch (launch/spec {:workgroup-size [workgroup-size]
                             :group-count [grid-group-count]
                             :shared-memory-bytes (:shared-mem-bytes source-grid)})
       :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                    :segop-id (:id segfold)}
       :attributes {:kind :portable-segmented-fold-map
                    :segment-count segment-count :map-extent map-extent
                    :source-grid source-grid
                    :grid-group-count grid-group-count-index
                    :no-write-alias true}})
     :arrays inputs :outputs outputs :scalars scalars
     :segment-count segment-count :map-extent map-extent
     :workgroup-size workgroup-size}))

(defn schedule
  "Refine one ordered SegFoldMap into a complete target-neutral ScheduledKernelBody.

   The exact SegFoldMap remains the semantic source. One-dimensional launch geometry assigns an
   independent work item to each segment, while all folds and final map results retain declaration
   order without reassociation. Stable reads and distinct writes are derived from the body rather
   than inferred by a target emitter."
  [segfold options]
  (let [{:keys [kernel-body segment-count inputs outputs scalars]} (lower segfold options)
        arguments (mapv (fn [parameter]
                          (if (= '_nseg (:id parameter)) segment-count (:id parameter)))
                        (:parameters kernel-body))]
    (scheduled-body/make
     {:source segfold
      :body kernel-body
      :arguments arguments
      :scalar-bindings (scheduled-body/derive-scalar-bindings kernel-body arguments)
      :effects {:kind :segmented-fold-map
                :uses (scheduled-body/derive-uses kernel-body arguments)
                :association :ordered}
      :legality {:kind :segfoldmap-body-lowering
                 :launch-rank 1
                 :segment-parallelism :grid-stride-independent
                 :association :ordered
                 :source-grid (:grid segfold)
                 :aliasing :no-write-alias}
      :numerics {:mode :exact
                 :policy :declaration-order
                 :reassociation :none}
      :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                   :segop-id (:id segfold)}
      :attributes {:array-params (vec (concat inputs outputs))
                   :scalar-params scalars
                   :dtype (first (:dtypes segfold))
                   :aliasing :no-write-alias}})))

(defn- record-name
  [value]
  (some-> value class .getSimpleName))

(defn- canonical-commutative
  [operator arguments]
  (let [arguments (mapcat (fn [argument]
                            (if (and (vector? argument) (= operator (first argument)))
                              (second argument)
                              [argument]))
                          arguments)]
    [operator (vec (sort-by pr-str arguments))]))

(declare canonical-extent)

(defn- canonical-operation
  [operator arguments]
  (let [arguments (mapv canonical-extent arguments)]
    (case operator
      (:mul :add :min :max) (canonical-commutative operator arguments)
      [operator arguments])))

(defn- canonical-extent
  "Normalize source, KernelBody, and KernelLaunch integer extent spellings for equality only."
  [expression]
  (case (record-name expression)
    "RuntimeValue" (canonical-extent (:value expression))
    "Product" (canonical-operation :mul (:factors expression))
    "Sum" (canonical-operation :add (:terms expression))
    "Minimum" (canonical-operation :min (:values expression))
    "CeilDiv" (canonical-operation :ceil-div [(:value expression) (:divisor expression)])
    "FloorDiv" (canonical-operation :floor-div [(:value expression) (:divisor expression)])
    "AlignUp" (canonical-operation :align-up [(:value expression) (:alignment expression)])
    "IndexExpr" (canonical-operation (:op expression) (:arguments expression))
    "IndexCast" (canonical-extent (:argument expression))
    (cond
      (and (seq? expression)
           (contains? '#{int long double clojure.core/int clojure.core/long
                         clojure.core/double}
                      (first expression))
           (= 2 (count expression)))
      (canonical-extent (second expression))

      (seq? expression)
      (let [operator ({'* :mul 'clojure.core/* :mul
                       '+ :add 'clojure.core/+ :add
                       'min :min 'clojure.core/min :min
                       'max :max 'clojure.core/max :max
                       'quot :floor-div 'clojure.core/quot :floor-div}
                      (first expression))]
        (if operator
          (canonical-operation operator (rest expression))
          [:leaf expression]))

      :else [:leaf expression])))

(defn- closed-derived-storage-scalars
  [kernel-graph closed-body]
  (if-not closed-body
    {}
    (let [host-prefix (vec (take-while #(true? (get-in % [:attributes :host-only]))
                                       (:equations closed-body)))
          numerical-equations (vec (drop (count host-prefix) (:equations closed-body)))]
      (when-not (= 1 (count numerical-equations))
        (throw (ex-info "fold-map storage proof requires exactly one numerical equation"
                        {:reason :segfoldmap-storage-proof
                         :host-prefix (mapv :id host-prefix)
                         :numerical-equations (mapv :id numerical-equations)})))
      (let [numerical (first numerical-equations)
            ;; `make` revalidates the complete SegOp program, the retained TypedSOAC boundary,
            ;; every buffer extent, and graph dataflow. Preserve only descriptive graph context
            ;; while reconstructing; it cannot contribute a scalar definition.
            expected-graph
            (equation-graph/make
             (:algorithm numerical) closed-body
             {:effects (:effects kernel-graph)
              :provenance (:provenance kernel-graph)
              :attributes (:attributes kernel-graph)})]
        (when-not (= expected-graph kernel-graph)
          (throw (ex-info "fold-map graph is not the exact projection of its retained equation body"
                          {:reason :segfoldmap-storage-proof
                           :expected expected-graph :actual kernel-graph})))
        (equation-graph/derived-scalar-expressions (:values closed-body) host-prefix)))))

(defn validate-against-node!
  "Close a fold-map refinement over its exact source grid and graph storage descriptions.

   ScheduledKernelBody proves source/effect/scalar closure generically. Fold-map additionally knows
   that every pointer is a dense `[segments, extent]` value and that the source KernelGrid is the
   complete launch schedule, so this validator can derive—not trust—those remaining obligations."
  ([scheduled node kernel-graph]
   (validate-against-node! scheduled node kernel-graph nil))
  ([scheduled node kernel-graph closed-body]
   (let [scheduled (scheduled-body/validate-against-node! scheduled node kernel-graph)
         source (:source scheduled)
         _ (when-not (instance? raster.compiler.ir.segop.SegFoldMap source)
             (throw (ex-info "fold-map storage closure requires an exact SegFoldMap source"
                             {:reason :segfoldmap-schedule-source :source source})))
         buffers (into {} (map (juxt :id identity))
                       (distinct (concat (:inputs kernel-graph)
                                         (:outputs kernel-graph)
                                         (:temporaries kernel-graph))))
         parameters (get-in scheduled [:body :parameters])
         bindings (into {} (map (fn [[parameter argument]] [(:id parameter) argument]))
                        (map vector parameters (:arguments scheduled)))
         derived-scalars (closed-derived-storage-scalars kernel-graph closed-body)
         expand-derived #(util/subst-syms derived-scalars %)
         expected-elements (canonical-extent
                            (expand-derived
                             (list '* (segop/seg-space-num-segments-expr (:space source))
                                   (:extent source))))]
     (doseq [[parameter argument] (map vector parameters (:arguments scheduled))
             :when (not= :scalar (:kind parameter))]
       (let [buffer (get buffers argument)
             parameter-elements
             (canonical-operation
              :mul (map #(-> %
                             (launch/rebind-expression bindings)
                             expand-derived)
                        (:shape parameter)))
             graph-elements (some-> buffer :elements canonical-extent)]
         (when-not (= (dtype/canon (:dtype parameter)) (some-> buffer :dtype dtype/canon))
           (throw (ex-info "fold-map KernelBody pointer dtype differs from its graph buffer"
                           {:reason :segfoldmap-storage-dtype :parameter (:id parameter)
                            :argument argument :parameter-dtype (:dtype parameter)
                            :graph-dtype (:dtype buffer)})))
         (when-not (and graph-elements
                        (= expected-elements parameter-elements graph-elements))
           (throw (ex-info "fold-map pointer extent differs across source, body, and graph"
                           {:reason :segfoldmap-storage-extent :parameter (:id parameter)
                            :argument argument :source expected-elements
                            :body parameter-elements :graph graph-elements})))))
     (let [array-types (into {} (map (juxt :id :dtype)) (vals buffers))
           scalar-types (into {} (map (juxt :id :dtype)) (:scalars kernel-graph))
           expected (schedule source {:array-types array-types :scalar-types scalar-types})]
       (when-not (= expected scheduled)
         (throw (ex-info "fold-map scheduled body differs from its exact source KernelGrid refinement"
                         {:reason :segfoldmap-schedule-source
                          :source-grid (:grid source)
                          :expected-body (:body expected)
                          :actual-body (:body scheduled)}))))
     scheduled)))

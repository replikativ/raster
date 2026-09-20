(ns raster.compiler.passes.parallel.segfoldmap-body
  "Portable KernelBody schedules for ordered and certified-reassociated segmented fold-map operations.

   One work item owns one independent segment. Each fold is a sequential, loop-carried region in
   declared order; completed fold values feed later folds and the final dense map. This is the
   baseline schedule, not an attention or normalization implementation."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
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

(defn- capped-ceiling-power-of-two
  [n cap]
  (loop [power 1]
    (if (or (>= power n) (>= power cap)) power (recur (* 2 power)))))

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

(defn- lower-ordered
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

(defn- schedule-ordered
  "Refine one ordered SegFoldMap into a complete target-neutral ScheduledKernelBody.

   The exact SegFoldMap remains the semantic source. One-dimensional launch geometry assigns an
   independent work item to each segment, while all folds and final map results retain declaration
   order without reassociation. Stable reads and distinct writes are derived from the body rather
   than inferred by a target emitter."
  [segfold options]
  (let [{:keys [kernel-body segment-count inputs outputs scalars]} (lower-ordered segfold options)
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

(defn- lower-cooperative
  "Assign one workgroup to each segment of a single certified fold-map.

   Lanes traverse the fold axis strided by workgroup size, combine their private partials through
   a portable shared-memory tree, then traverse the dense final map cooperatively. This is a
   schedule for the general SegFoldMap algebra; no normalization or tensor-library operation is
   recognized here."
  [segfold {:keys [workgroup-size array-types scalar-types]
            :or {array-types {} scalar-types {}}}]
  (when-not (instance? raster.compiler.ir.segop.SegFoldMap segfold)
    (throw (ex-info "cooperative fold-map lowering requires SegFoldMap"
                    {:reason :raster/bug :operation segfold})))
  (let [space (:space segfold)
        source-grid (:grid segfold)
        requested-workgroup-size workgroup-size
        segment-dims (segop/seg-space-segment-dims space)
        mapped-dim (segop/seg-space-reduced-dim space)
        folds (vec (:folds segfold))
        _ (when-not (= 1 (count folds))
            (decline! :cooperative-fold-count
                      "the first cooperative fold-map schedule requires exactly one fold"
                      {:operation (:id segfold) :fold-count (count folds)}))
        fold (first folds)
        source-workgroup-size (or (:block-size source-grid) 256)
        ;; Do not launch more lanes than a static fold can use. Dynamic folds retain the
        ;; hardware-selected size; an explicit caller override is a testing/tuning decision and
        ;; is preserved exactly. The next power of two keeps the shared tree structurally valid.
        static-fold-extent (some-> (:extent fold) numeric-constant/value :value)
        workgroup-size (or requested-workgroup-size
                           (if (and (integer? static-fold-extent) (pos? static-fold-extent))
                             (capped-ceiling-power-of-two static-fold-extent
                                                          source-workgroup-size)
                             source-workgroup-size))
        _ (when-not (= :implementation-defined (:association fold))
            (decline! :fold-association
                      "cooperative fold-map scheduling requires an explicit reassociation contract"
                      {:operation (:id segfold) :association (:association fold)}))
        _ (when (seq (:locals fold))
            (decline! :fold-locals
                      "the first cooperative fold-map schedule requires an expression-only fold"
                      {:operation (:id segfold) :locals (:locals fold)}))
        _ (when-not (and (integer? workgroup-size) (pos? workgroup-size)
                         (zero? (bit-and workgroup-size (dec workgroup-size))))
            (decline! :workgroup-size
                      "cooperative fold-map scheduling requires a positive power-of-two workgroup"
                      {:operation (:id segfold) :workgroup-size workgroup-size}))
        _ (when (empty? segment-dims)
            (decline! :no-segments "cooperative fold-map requires at least one segment axis"
                      {:operation (:id segfold)}))
        _ (when-not (= (:extent segfold) (:bound mapped-dim))
            (decline! :mapped-extent
                      "fold-map semantic extent must equal its reduced-space bound"
                      {:operation (:id segfold) :extent (:extent segfold)
                       :reduced-bound (:bound mapped-dim)}))
        inputs (vec (sort-by name (:inputs segfold)))
        outputs (vec (:outputs segfold))
        scalars (vec (sort-by name (:scalars segfold)))
        _ (when-not (= :no-write-alias (:aliasing segfold))
            (decline! :aliasing-contract
                      "cooperative fold-map requires a no-write-alias source contract"
                      {:operation (:id segfold) :aliasing (:aliasing segfold)}))
        _ (when (or (seq (set/intersection (set inputs) (set outputs)))
                    (not= (count outputs) (count (distinct outputs))))
            (decline! :storage-contract
                      "cooperative fold-map requires disjoint stable inputs and distinct outputs"
                      {:operation (:id segfold) :inputs inputs :outputs outputs}))
        _ (when-not (= (count outputs) (count (:dtypes segfold))
                       (count (:map-results segfold)))
            (decline! :result-contract
                      "fold-map outputs, dtypes, and map results must be aligned"
                      {:operation (:id segfold) :outputs outputs
                       :dtypes (:dtypes segfold) :map-results (:map-results segfold)}))
        output-dtypes (mapv dtype/canon (:dtypes segfold))
        default-dtype (or (first output-dtypes) :float)
        fold-dtype (dtype/canon (:dtype fold))
        _ (when-not (contains? #{:float :double :int :long} fold-dtype)
            (decline! :fold-dtype
                      "cooperative fold-map requires a scalar arithmetic accumulator"
                      {:operation (:id segfold) :dtype fold-dtype}))
        array-types (into {}
                          (map (fn [id]
                                 [id (dtype/canon (or (get array-types id)
                                                      (get array-types (symbol (name id)))
                                                      default-dtype))]))
                          (concat inputs outputs))
        scalar-types (into {}
                           (map (fn [id]
                                  [id (dtype/canon
                                       (or (get scalar-types id)
                                           (get scalar-types (symbol (name id)))
                                           :int))]))
                           scalars)
        segment-count-source (segop/seg-space-num-segments-expr space)
        source-index (:index segfold)
        reduce-index 'foldmap-reduce-index
        map-index 'foldmap-map-index
        segment-index 'foldmap-segment
        axis-symbols (set (concat (map :name segment-dims) [source-index]))
        index-scope (into (set/union axis-symbols
                                     #{reduce-index map-index segment-index})
                          scalars)
        index-value-types (merge (zipmap axis-symbols (repeat :long))
                                 {reduce-index :long map-index :long segment-index :long}
                                 scalar-types)
        lower-index (fn lower-index
                      ([expression] (lower-index expression #{}))
                      ([expression extra-scope]
                       (widen-index-expression
                        (index-expression/lower
                         expression (set/union index-scope extra-scope) decline!)
                        index-value-types)))
        segment-count (lower-index segment-count-source)
        map-extent (lower-index (:bound mapped-dim))
        fold-extent (lower-index (:extent fold))
        total-elements (body/expression :mul segment-count map-extent)
        group-count (index-expression/to-launch-expression segment-count decline!)
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
                   scalars)))
        base-env (merge (zipmap (map :name segment-dims) (repeat :long)) scalar-types)
        scalar-lower (scalar-expression/make-lowerer
                      {:array-types array-types :scalar-types scalar-types
                       :arrays (set inputs) :index-scope index-scope
                       :lower-index lower-index :predicate nil
                       :id-prefix "cooperative-foldmap" :decline! decline!})
        concrete-step (dialect/scalar-converts->source (:step fold))
        derived
        (try
          (scan/certify-reassociation
           {:acc (:accumulator fold) :init (:identity fold) :lambda concrete-step}
           fold-dtype)
          (catch clojure.lang.ExceptionInfo exception
            (decline! :certified-monoid
                      "cooperative fold-map could not rederive its scalar monoid"
                      {:operation (:id segfold) :certificate-error (ex-data exception)})))
        _ (when-not (scan/compatible-certificate? (:algebra fold) derived)
            (decline! :certified-monoid
                      "cooperative fold-map algebra disagrees with its scalar region"
                      {:operation (:id segfold) :declared (:algebra fold) :derived derived}))
        operator (intrinsics/canonical (:combine derived))
        _ (when-not (contains? #{:+ :* :min :max} operator)
            (decline! :cooperative-operator
                      "cooperative fold-map scheduling requires a supported scalar monoid"
                      {:operation (:id segfold) :operator operator}))
        identity (numeric-constant/literal-or-original (:identity fold))
        _ (when-not (number? identity)
            (decline! :literal-identity
                      "cooperative fold-map requires a literal scalar identity"
                      {:operation (:id segfold) :identity identity}))
        element (util/subst-syms {source-index reduce-index} (:element derived))
        element-lowered ((:lower scalar-lower) element fold-dtype
                         (assoc base-env reduce-index :long))
        lane-accumulator 'foldmap-lane-accumulator
        next-lane-accumulator 'foldmap-next-lane-accumulator
        lane-result 'foldmap-lane-result
        completed-fold (:accumulator fold)
        scratch 'foldmap-workgroup-scratch
        barrier (fn [] (body/->WorkgroupBarrier
                        :workgroup #{:workgroup} :acquire-release
                        (body/full-participation)))
        strides (take-while pos? (iterate #(quot % 2) (quot workgroup-size 2)))
        tree-stages
        (mapcat
         (fn [stride]
           (let [mask (keyword (str "foldmap-reduce-stride-" stride))
                 left (symbol (str "foldmap-tree-left-" stride))
                 right (symbol (str "foldmap-tree-right-" stride))
                 combined (symbol (str "foldmap-tree-combined-" stride))]
             [(body/->ScalarLoad (body/value left fold-dtype) scratch ['foldmap-lane]
                                 mask (body/literal identity fold-dtype) :cached)
              (body/->ScalarLoad
               (body/value right fold-dtype) scratch
               [(body/expression :add 'foldmap-lane stride)]
               mask (body/literal identity fold-dtype) :cached)
              (body/->ScalarCompute
               (body/value combined fold-dtype)
               (body/scalar-expression operator fold-dtype [left right]))
              (body/->ScalarStore scratch ['foldmap-lane] combined mask)
              (barrier)]))
         strides)
        lane-fold
        (body/->ForLoop
         (body/value reduce-index :long)
         (body/index-cast 'foldmap-lane :long :exact) fold-extent workgroup-size
         [(body/->LoopArg (body/value lane-accumulator fold-dtype)
                          (body/literal identity fold-dtype))]
         (vec (concat
               (:operations element-lowered)
               [(body/->ScalarCompute
                 (body/value next-lane-accumulator fold-dtype)
                 (body/scalar-expression
                  operator fold-dtype [lane-accumulator (:result element-lowered)]))
                (body/->Yield [next-lane-accumulator])]))
         [(body/value lane-result fold-dtype)]
         {:association :implementation-defined :role :cooperative-lane-fold})
        map-coordinate (body/expression :add base-coordinate map-index)
        map-operations
        (mapcat
         (fn [_ordinal output output-dtype expression]
           (let [expression (util/subst-syms {source-index map-index} expression)
                 lowered ((:lower scalar-lower) expression output-dtype
                                                (assoc base-env map-index :long
                                                       completed-fold fold-dtype))]
             (concat (:operations lowered)
                     [(body/->ScalarStore output [map-coordinate]
                                          (:result lowered) nil)])))
         (range) outputs output-dtypes (:map-results segfold))
        final-map
        (body/->ForLoop
         (body/value map-index :long)
         (body/index-cast 'foldmap-lane :long :exact) map-extent workgroup-size []
         (vec (concat map-operations [(body/->Yield [])])) []
         {:association :independent :role :cooperative-final-map})
        kernel-body
        (body/make
         {:id [:segmented-fold-map (:id segfold) :cooperative-workgroup]
          :parameters parameters
          :stable-reads (mapv body/stable-read inputs)
          :allocations [(body/->WorkgroupAllocation
                         scratch fold-dtype [workgroup-size]
                         (layout/row-major [workgroup-size] fold-dtype)
                         (dtype/bytes-of fold-dtype))]
          :indices (vec (concat
                         [(body/->IndexBinding 'foldmap-group :group 0)
                          (body/->IndexBinding 'foldmap-lane :local 0)
                          (body/->IndexCompute
                           segment-index
                           (body/index-cast 'foldmap-group :long :exact))]
                         segment-axis-computes))
          :masks (mapv (fn [stride]
                         (body/->Mask (keyword (str "foldmap-reduce-stride-" stride))
                                      [(body/predicate :lt 'foldmap-lane stride)]))
                       strides)
          :operations (vec (concat
                            [lane-fold
                             (body/->ScalarStore scratch ['foldmap-lane] lane-result nil)
                             (barrier)]
                            tree-stages
                            [(body/->ScalarLoad
                              (body/value completed-fold fold-dtype) scratch [0] nil nil :cached)
                             ;; Every lane must consume scratch[0] before the next use of the
                             ;; allocation. Keep this barrier even in the one-fold schedule so
                             ;; extending it to dependent certified folds cannot introduce a race.
                             (barrier)
                             final-map]))
          :schedule {:strategy :one-workgroup-per-segment
                     :association :certified
                     :workgroup-size workgroup-size
                     :reduction-operator operator
                     :mapped-traversal :strided}
          :launch (launch/spec
                   {:workgroup-size [workgroup-size]
                    :group-count [group-count]
                    :shared-memory-bytes (* workgroup-size (dtype/bytes-of fold-dtype))})
          :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                       :segop-id (:id segfold)}
          :attributes {:kind :cooperative-segmented-fold-map
                       :segment-count segment-count :map-extent map-extent
                       :no-write-alias true}})]
    {:kernel-body kernel-body :arrays inputs :outputs outputs :scalars scalars
     :segment-count segment-count :map-extent map-extent
     :workgroup-size workgroup-size :operator operator}))

(defn- schedule-cooperative
  [segfold options]
  (let [{:keys [kernel-body inputs outputs scalars operator]}
        (lower-cooperative segfold options)
        arguments (mapv :id (:parameters kernel-body))]
    (scheduled-body/make
     {:source segfold
      :body kernel-body
      :arguments arguments
      :scalar-bindings (scheduled-body/derive-scalar-bindings kernel-body arguments)
      :effects {:kind :segmented-fold-map
                :uses (scheduled-body/derive-uses kernel-body arguments)
                :association :implementation-defined}
      :legality {:kind :segfoldmap-body-lowering
                 :launch-rank 1
                 :segment-parallelism :one-workgroup-per-segment
                 :association :certified
                 :power-of-two-workgroup true
                 :aliasing :no-write-alias}
      :numerics {:mode :reassociated
                 :policy :certified-workgroup-tree
                 :rounding :implementation-defined
                 :accumulator-dtype (dtype/canon (:dtype (first (:folds segfold))))
                 :reassociation :implementation-defined}
      :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                   :segop-id (:id segfold)}
      :attributes {:array-params (vec (concat inputs outputs))
                   :scalar-params scalars
                   :dtype (first (:dtypes segfold))
                   :reduction-operator operator
                   :aliasing :no-write-alias}})))

(defn- cooperative-source?
  [segfold]
  (some #(= :implementation-defined (:association %)) (:folds segfold)))

(defn lower
  "Choose the exact ordered baseline or certified cooperative fold-map schedule."
  [segfold options]
  (if (cooperative-source? segfold)
    (lower-cooperative segfold options)
    (lower-ordered segfold options)))

(defn schedule
  "Refine one SegFoldMap without changing its per-fold numerical contracts."
  [segfold options]
  (if (cooperative-source? segfold)
    (schedule-cooperative segfold options)
    (schedule-ordered segfold options)))

(defn- record-name
  [value]
  (some-> value class .getSimpleName))

(defn- canonical-commutative
  [operator arguments]
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
    "Maximum" (canonical-operation :max (:values expression))
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
  [kernel-graph closed-algorithm closed-body]
  (when-not (= (some? closed-algorithm) (some? closed-body))
    (throw (ex-info "fold-map graph proof requires both algorithm and scheduled body"
                    {:reason :segfoldmap-storage-proof
                     :algorithm closed-algorithm :body closed-body})))
  (if-not closed-body
    {}
    (let [host-prefix (vec (take-while #(true? (get-in % [:attributes :host-only]))
                                       (:equations closed-body)))]
      (equation-graph/validate-projection! kernel-graph closed-algorithm closed-body)
      (equation-graph/derived-scalar-expressions (:values closed-body) host-prefix))))

(defn validate-against-node!
  "Close a fold-map refinement over its exact source grid and graph storage descriptions.

   ScheduledKernelBody proves source/effect/scalar closure generically. Fold-map additionally knows
   that every pointer is a dense `[segments, extent]` value and that the source KernelGrid is the
   complete launch schedule, so this validator can derive—not trust—those remaining obligations."
  ([scheduled node kernel-graph]
   (validate-against-node! scheduled node kernel-graph nil nil))
  ([scheduled node kernel-graph closed-algorithm closed-body]
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
         derived-scalars (closed-derived-storage-scalars
                          kernel-graph closed-algorithm closed-body)
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
                        (= expected-elements parameter-elements)
                        (or (= expected-elements graph-elements)
                            ;; An externally supplied array may have only a runtime capacity
                            ;; contract at compile time. The graph reconstruction above proves
                            ;; this exact sentinel came from the retained AbstractValue; binding
                            ;; checks the concrete capacity before launch.
                            (and (contains? #{:input :output :inout} (:role buffer))
                                 (= (list 'extent argument) (:elements buffer)))))
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

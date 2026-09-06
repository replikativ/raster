(ns raster.compiler.passes.parallel.segred-body
  "Schedule an ordinary scalar SegRed as target-neutral KernelBody data.

   This first production slice deliberately models the same portable workgroup tree used by the
   established OpenCL emitter: occupancy-capped workgroups, a per-lane sequential chunk, identity
   padding, workgroup scratch, a barrier-separated binary tree, and one partial result per
   workgroup. Scalar expressions become typed SSA operations; target emitters never recover either
   the algorithm or schedule from Clojure source spelling. Unsupported scalar regions decline
   explicitly; scheduled graphs have no source-shaped target fallback."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data
                                 :reason :segred-kernel-body-declined
                                 :missing-rule rule
                                 :fallback :none))))

(defn declined?
  [exception]
  (= :segred-kernel-body-declined (:reason (ex-data exception))))

(def ^:private scalar-phases #{:single :block-local :cross-block})

(defn scalar-workgroup-tree-schedule
  "Construct the canonical schedule carried by one ordinary scalar SegRed phase.

   This is shared by semantic-to-SegOp lowering and the zero-free contraction compatibility
   projection. The source operation therefore states the workgroup tree that KernelBody later
   verifies; KernelBody scheduling does not manufacture a second, implicit schedule."
  [operator grid phase]
  (let [operator (reduction/validate! operator)
        workgroup-size (:block-size grid)
        accumulator-dtype (dtype/canon (:dtype (first (:components operator))))
        shared-memory-bytes (* workgroup-size (dtype/bytes-of accumulator-dtype))
        terminal? (contains? #{:single :cross-block} phase)]
    (reduction/schedule
     {:strategy :scalar-workgroup-tree
      :workgroup-size workgroup-size
      :stages [:lane-fold :workgroup-tree (if terminal? :terminal-store :partial-store)]
      :tuning-space {}
      :numerical-mode (select-keys (if (scan/associative-scan? (:algebra operator))
                                     (:algebra operator)
                                     (first (:components (:algebra operator))))
                                   [:order :reassociation :overflow])
      :attributes {:phase phase
                   :group-count (:num-blocks grid)
                   :shared-memory-bytes shared-memory-bytes}})))

(defn- validate-scalar-segred!
  "Validate the complete semantic/schedule subset implemented by the portable workgroup tree."
  [segred out-sym array-types]
  (when-not (instance? raster.compiler.ir.segop.SegRed segred)
    (throw (ex-info "scalar reduction scheduling requires a SegRed"
                    {:reason :raster/bug :operation segred})))
  (let [operator (try
                   (reduction/validate! (:reduction segred))
                   (catch clojure.lang.ExceptionInfo exception
                     (decline! :scalar-product-reduction
                               "portable scalar SegRed requires a canonical ProductReduction"
                               {:operation (:id segred) :reduction (:reduction segred)
                                :reduction-error (ex-data exception)})))
        dimensions (segop/seg-space-segment-dims (:space segred))
        phase (:phase segred)
        outputs (vec (:outputs segred))
        component (first (:components operator))
        accumulator-dtype (dtype/canon (:dtype component))
        grid (:grid segred)
        workgroup-size (:block-size grid)
        num-blocks (get-in segred [:grid :num-blocks])
        shared-memory-bytes (get-in segred [:grid :shared-mem-bytes])
        schedule (try
                   (reduction/validate-schedule! (:schedule segred))
                   (catch clojure.lang.ExceptionInfo exception
                     (decline! :scalar-workgroup-tree-schedule
                               "portable scalar SegRed requires an explicit reduction schedule"
                               {:operation (:id segred) :schedule (:schedule segred)
                                :schedule-error (ex-data exception)})))
        result-region (get-in operator [:attributes :result-region])
        output (or out-sym (first outputs))
        declared-type (fn [id]
                        (some-> (or (get array-types id)
                                    (get array-types
                                         (when (or (symbol? id) (keyword? id))
                                           (symbol (name id)))))
                                dtype/canon))
        input-types (mapv (fn [input]
                            (or (declared-type input) accumulator-dtype))
                          (:inputs segred))
        output-type (or (declared-type output) accumulator-dtype)]
    (when-not (reduction/scalar? operator)
      (decline! :scalar-product-reduction
                "portable scalar SegRed scheduling requires one ProductReduction component"
                {:operation (:id segred) :components (count (:components operator))}))
    (when (seq dimensions)
      (decline! :zero-segment-dimensions
                "portable scalar SegRed cannot schedule a segmented result space"
                {:operation (:id segred) :segment-dimensions dimensions}))
    (when-not (nil? (:lambda segred))
      (decline! :fused-map-representation
                "scalar SegRed requires its element expression in ProductReduction, not map-lambda"
                {:operation (:id segred) :map-lambda (:lambda segred)}))
    (when-not (contains? scalar-phases phase)
      (decline! :scalar-reduction-phase
                "portable scalar SegRed has an unsupported reduction phase"
                {:operation (:id segred) :phase phase :supported scalar-phases}))
    (when-not (and (= 1 (count outputs)) (some? (first outputs)) (some? output))
      (decline! :single-physical-output
                "portable scalar SegRed requires one stable physical output identity"
                {:operation (:id segred) :outputs outputs :requested-output out-sym}))
    (when-not (= (first outputs) output)
      (decline! :physical-output-remap
                "scalar SegRed emission cannot silently replace its semantic output identity"
                {:operation (:id segred) :semantic-output (first outputs)
                 :requested-output out-sym}))
    (when-not (contains? #{:float :double} accumulator-dtype)
      (decline! :uniform-scalar-storage
                "portable scalar SegRed supports FP32 or FP64 storage and accumulation"
                {:operation (:id segred) :accumulator-dtype accumulator-dtype}))
    (when-not (and (= accumulator-dtype (dtype/canon (:dtype segred)))
                   (= accumulator-dtype output-type)
                   (every? #{accumulator-dtype} input-types))
      (decline! :uniform-scalar-storage
                "portable scalar SegRed requires uniform storage and accumulator dtypes"
                {:operation (:id segred) :segred-dtype (:dtype segred)
                 :accumulator-dtype accumulator-dtype :input-dtypes input-types
                 :output-dtype output-type}))
    (when-not (and (integer? workgroup-size) (pos? workgroup-size)
                   (zero? (bit-and workgroup-size (dec workgroup-size))))
      (decline! :workgroup-size
                "portable scalar SegRed requires a positive power-of-two workgroup"
                {:operation (:id segred) :workgroup-size workgroup-size}))
    (let [expected-level (if (= :block-local phase)
                           (segop/->SegLevel :block :virtual)
                           (segop/->SegLevel :block :none))
          expected-shared-memory (* workgroup-size (dtype/bytes-of accumulator-dtype))
          expected-schedule (scalar-workgroup-tree-schedule operator grid phase)]
      (when-not (= expected-level (:level segred))
        (decline! :phase-level
                  "scalar SegRed phase and execution level disagree"
                  {:operation (:id segred) :phase phase
                   :expected expected-level :actual (:level segred)}))
      (when-not (= expected-shared-memory shared-memory-bytes)
        (decline! :grid-shared-memory
                  "scalar SegRed grid shared memory differs from its workgroup tree"
                  {:operation (:id segred) :expected expected-shared-memory
                   :actual shared-memory-bytes}))
      (when-not (= expected-schedule schedule)
        (decline! :schedule-grid
                  "scalar SegRed schedule is not the exact canonical workgroup-tree schedule"
                  {:operation (:id segred) :phase phase :grid grid
                   :expected expected-schedule :schedule schedule})))
    (when (and (contains? #{:single :cross-block} phase) (not= 1 num-blocks))
      (decline! :terminal-phase-groups
                "terminal scalar SegRed phases must launch exactly one workgroup"
                {:operation (:id segred) :phase phase :num-blocks num-blocks}))
    (when (and (= :block-local phase) result-region)
      (decline! :nonterminal-result-transform
                "a completed reduction transform may run only in a terminal SegRed phase"
                {:operation (:id segred) :phase phase :result-region result-region}))
    (when-not (= phase (get-in operator [:attributes :physical-phase]))
      (decline! :physical-phase
                "scalar SegRed reduction must state the exact physical phase it inhabits"
                {:operation (:id segred) :phase phase
                 :reduction-physical-phase (get-in operator [:attributes :physical-phase])}))
    {:operator operator :component component :dtype accumulator-dtype
     :workgroup-size workgroup-size :phase phase :output output
     :result-region result-region :schedule schedule}))

(defn- retained-expression-dtype
  "Read the walker/TypedClojure result fact carried by a scalar expression.

   This is intentionally not an inference rule: if a retained tag is present but outside the
   portable scalar vocabulary, the scheduled reduction declines."
  [expression]
  (when (instance? clojure.lang.IObj expression)
    (when-let [tag (types/sym-type-tag expression)]
      (or (when (and (keyword? tag) (dtype/known? tag)) (dtype/canon tag))
          (dtype/dtype-for-scalar-tag tag)
          (decline! :scalar-result-dtype
                    "KernelBody reduction cannot project the retained scalar result type"
                    {:expression expression :retained-tag tag})))))

(defn- source-value-dtype
  [expression]
  (or (retained-expression-dtype expression)
      (when (number? expression)
        (some-> expression types/literal-tag dtype/dtype-for-scalar-tag))))

(defn- inline-scalar-bindings
  [expression]
  (if (and (seq? expression) (contains? #{'let 'let*} (first expression)))
    (let [[_ bindings & body] expression
          pairs (vec (partition 2 bindings))
          initializers (vec (take-nth 2 (rest bindings)))]
      (when-not (= 1 (count body))
        (decline! :multi-expression-let
                  "KernelBody scalar reduction requires a single-expression let region"
                  {:expression expression}))
      (when (some util/effectful? initializers)
        (decline! :effectful-scalar-binding
                  "KernelBody scalar reduction cannot inline effectful scalar bindings"
                  {:expression expression}))
      (doseq [[binding init] pairs
              :let [binding-dtype (some-> binding types/sym-type-tag
                                          dtype/dtype-for-scalar-tag)]
              :when (and binding-dtype (not= binding-dtype (source-value-dtype init)))]
        (decline! :typed-scalar-binding-conversion
                  "KernelBody reduction cannot erase a typed scalar binding conversion"
                  {:binding binding :binding-dtype binding-dtype :initializer init
                   :initializer-dtype (source-value-dtype init)}))
      (recur (util/subst-syms (util/binding-env bindings) (first body))))
    expression))

(defn scalar-plan
  "Project one scalar SegRed into an explicit combine operator, identity and element expression."
  [segred]
  (let [{:keys [acc init lambda]} (segop/scalar-reduce-op segred)
        expression (inline-scalar-bindings lambda)
        component (first (get-in segred [:reduction :components]))
        dtype (:dtype component)
        declared-algebra (get-in segred [:reduction :algebra])
        declared (if (scan/associative-scan? declared-algebra)
                   declared-algebra
                   (first (:components declared-algebra)))
        derived (try
                  (scan/certify-reassociation
                   {:acc acc :init init :lambda expression} dtype)
                  (catch clojure.lang.ExceptionInfo exception
                    (decline! :certified-monoid
                              "KernelBody reduction requires a certified typed monoid"
                              {:segred-id (:id segred) :certificate-error (ex-data exception)})))
        operator (intrinsics/canonical (:combine derived))]
    (when-not (scan/compatible-certificate? declared derived)
      (decline! :certified-monoid
                "scheduled reduction algebra disagrees with its concrete scalar region"
                {:segred-id (:id segred) :declared declared :derived derived}))
    ;; Retain the concrete neutral spelling after proving it equivalent to the typed registry
    ;; identity. KernelBody consumers need a literal, while the certificate remains the proof.
    {:operator operator :identity (constant/literal-or-original init) :element (:element derived)
     :accumulator acc}))

(defn capped-group-count
  "Construct the canonical non-empty occupancy-capped scalar-reduction grid."
  [cap bound workgroup-size]
  (when-not (and (integer? cap) (pos? cap))
    (decline! :launch-grid "scalar reduction occupancy cap must be positive"
              {:cap cap :bound bound :workgroup-size workgroup-size}))
  (launch/maximum 1 (launch/minimum cap (launch/ceil-div bound workgroup-size))))

(defn launch-group-count
  "Translate SegRed's exact occupancy cap into a non-empty inspectable launch expression.

   `compute-launch-params` predates KernelLaunch and represents the reduction grid as
   `(min occupancy-cap (int (Math/ceil (/ (double bound) block-size))))`.  Do not carry that
   executable host form into KernelBody: recognize the exact producer contract and rebuild it
   from the authoritative SegSpace bound and workgroup size. The outer maximum gives an empty
   reduction one workgroup, whose inactive lanes reduce to the certified identity."
  [grid-expression bound workgroup-size]
  (let [historical-groups
        (list 'int
              (list 'Math/ceil
                    (list '/ (list 'double bound) (double workgroup-size))))
        historical-cap
        (when (and (seq? grid-expression)
                   (contains? '#{min clojure.core/min} (first grid-expression))
                   (= 3 (count grid-expression))
                   (integer? (second grid-expression))
                   (pos? (second grid-expression))
                   (= historical-groups (nth grid-expression 2)))
          (second grid-expression))
        ;; KernelLaunch records are maps. Recognize only the exact canonical shape rather than
        ;; accepting an arbitrary expression that happens to agree with a forged body.
        canonical-cap
        (when (= "raster.compiler.ir.kernel_launch.Maximum"
                 (some-> grid-expression class .getName))
          (let [[one capped] (:values grid-expression)]
            (when (and (= 1 one)
                       (= "raster.compiler.ir.kernel_launch.Minimum"
                          (some-> capped class .getName)))
              (let [[cap groups] (:values capped)]
                  (when (and (integer? cap) (pos? cap)
                           (= groups (launch/ceil-div bound workgroup-size)))
                  cap)))))
        cap (or historical-cap canonical-cap)]
    (if cap
      (capped-group-count cap bound workgroup-size)
      (decline! :launch-grid
                "KernelBody scalar reduction requires its canonical occupancy-capped group count"
                {:grid-expression grid-expression :bound bound
                 :workgroup-size workgroup-size}))))

(defn- strip-index-cast
  [expression]
  (if (and (seq? expression)
           (contains? #{'int 'long 'clojure.core/int 'clojure.core/long} (first expression))
           (= 2 (count expression)))
    (second expression)
    expression))

(defn- widen-index-expression
  "Make certified contraction-coordinate arithmetic uniformly 64-bit."
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

(defn- cast-policy
  [source target]
  (let [source (dtype/canon source)
        target (dtype/canon target)]
    (when-not (= source target)
      (or (scalar-conversion/policy source target)
          (decline! :checked-scalar-cast
                    (if (dtype/fp-dtype? source)
                      "KernelBody reduction cannot preserve a checked floating-to-integral cast"
                      "KernelBody reduction cannot preserve a checked narrowing integral cast")
                    {:source-dtype source :target-dtype target})))))

(defn lower-element-operations
  "Lower a scalar reduction element into typed SSA. coordinate-lower may translate a verified
   source-level flat array index into KernelBody index arithmetic; without it, this retains the
   pointwise full-reduction contract. Mixed scalar arithmetic requires the walker/TypedClojure
   result dtype retained on that expression. The lowered element must already match the certified
   accumulator dtype; this pass never invents a final narrowing conversion. A typed region owner
   may provide `declared-result-dtype` for the outer expression only; nested calls still require
   their own retained facts. Admission remains here; accepted loads, conversions and arithmetic
   over typed child values use the same SSA builder as maps and ordered fold-maps."
  [expression {:keys [index coordinate dtype arrays array-types scalars scalar-types coordinate-lower
                      load-predicate load-other declared-result-dtype]}]
  (let [dtype (dtype/canon dtype)
        operations (atom [])
        lowerer (scalar-expression/make-lowerer
                 {:arrays (set arrays)
                  :array-types (into {} (map (fn [id] [id (dtype/canon (get array-types id dtype))]))
                                     arrays)
                  :scalar-types (into {} (map (fn [id] [id (dtype/canon (get scalar-types id dtype))]))
                                      scalars)
                  :source-region expression
                  ;; Only this adapter's already-approved coordinates reach the SSA builder.
                  :lower-index (fn [coordinate _] coordinate)
                  :predicate load-predicate
                  :load-other (fn [storage-dtype]
                                (if load-other
                                  (body/literal (:value load-other) storage-dtype)
                                  (body/literal 0 storage-dtype)))
                  :conversion-policy cast-policy :decline! decline! :id-prefix "element"})
        append! (fn [lowered]
                  (let [{:keys [result type]} lowered]
                    (swap! operations into (:operations lowered))
                    {:value result :dtype type}))]
    (letfn [(cast! [{:keys [value dtype] :as typed} target]
              (let [source (dtype/canon dtype)
                    target (dtype/canon target)]
                (if (= source target)
                  typed
                  (append! ((:cast lowerer) {:operations [] :result value :type source}
                            target nil)))))

            (lower [expression declared-dtype]
              (let [expression (inline-scalar-bindings expression)]
                (cond
                  (number? expression)
                  (if-let [literal-dtype (some-> expression types/literal-tag
                                                 dtype/dtype-for-scalar-tag)]
                    {:value (body/literal expression literal-dtype) :dtype literal-dtype}
                    (decline! :scalar-literal-dtype
                              "KernelBody reduction requires a primitive numeric literal"
                              {:expression expression :class (class expression)}))

                  (symbol? expression)
                  (if (contains? scalars expression)
                    {:value expression
                     :dtype (dtype/canon (get scalar-types expression dtype))}
                    (decline! :unbound-scalar
                              "KernelBody element expression references an undeclared scalar"
                              {:expression expression :scalars scalars}))

                  (descriptor/aget-call? expression)
                  (let [arguments (vec (descriptor/call-args expression))
                        array (descriptor/aget-array-sym expression)
                        source-coordinate (some-> (last arguments) strip-index-cast)
                        lowered-coordinate (if coordinate-lower
                                             (coordinate-lower source-coordinate)
                                             (when (= index source-coordinate) coordinate))]
                    (when-not (and (= 2 (count arguments))
                                   (contains? arrays array)
                                   lowered-coordinate)
                      (decline! :indexed-load
                                "KernelBody reduction cannot prove this array load coordinate"
                                {:expression expression :array array :coordinate source-coordinate
                                 :index index :arrays arrays}))
                    (append! ((:load lowerer) array [lowered-coordinate])))

                  (and (seq? expression) (descriptor/cast-op? (first expression))
                       (= 2 (count expression)))
                  (let [target (dtype/dtype-for-scalar-tag
                                (descriptor/cast-result-tag (first expression)))]
                    ;; Clojure's integral casts are checked. KernelBody can describe a trapping
                    ;; conversion, but the current C-family emitters intentionally reject it.
                    ;; Refuse the source construct here instead of silently changing it to the
                    ;; backend's wrap or saturate conversion.
                    (when (dtype/integral? target)
                      (decline! :checked-scalar-cast
                                "portable reduction lowering cannot yet emit checked integral casts"
                                {:expression expression :target-dtype target}))
                    (cast! (lower (second expression) nil) target))

                  (seq? expression)
                  (let [operator (intrinsics/canonical (descriptor/semantic-op expression))
                        intrinsic (intrinsics/descriptor operator)
                        arguments (vec (descriptor/call-args expression))
                        retained-dtype (or (retained-expression-dtype expression)
                                           (some-> declared-dtype dtype/canon))
                        _ (when-not retained-dtype
                            (decline! :scalar-result-dtype
                                      "scalar arithmetic requires its retained walker/TypedClojure result dtype"
                                      {:expression expression :operator operator}))
                        typed-inputs (mapv #(lower % nil) arguments)
                        result-dtype retained-dtype]
                    (when (dtype/integral? result-dtype)
                      (decline! :integral-scalar-arithmetic
                                "portable reduction value arithmetic requires explicit overflow semantics"
                                {:expression expression :operator operator
                                 :result-dtype result-dtype}))
                    (when-not (and intrinsic
                                   (= (:arity intrinsic) (count arguments))
                                   (intrinsics/accepts-scalar-dtype? operator result-dtype)
                                   (not= :cmp (:kind intrinsic)))
                      (decline! :scalar-expression
                                "KernelBody element expression contains an unsupported scalar operation"
                                {:expression expression :operator operator
                                 :result-dtype result-dtype}))
                    (append! ((:compute lowerer) operator result-dtype
                              (mapv (comp :value #(cast! % result-dtype)) typed-inputs) {})))

                  :else
                  (decline! :scalar-expression
                            "KernelBody element expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      (let [result (lower expression declared-result-dtype)]
        (when-not (= dtype (:dtype result))
          (decline! :element-result-dtype
                    "KernelBody reduction element must match its certified accumulator dtype"
                    {:expression expression :element-dtype (:dtype result)
                     :accumulator-dtype dtype}))
        {:operations @operations :result (:value result)}))))
(defn lower
  "Lower an eligible scalar SegRed to one verified portable workgroup-tree KernelBody.

  `array-types` and `scalar-types` are authoritative ABI facts. Tensor element storage and the
   accumulator remain uniform; integral scalar parameters may participate in index expressions."
  [segred out-sym & {:keys [dtype array-types scalar-types coordinate-proof]
                     :or {dtype :double array-types {} scalar-types {}}}]
  (let [{validated-dtype :dtype output :output result-region :result-region}
        (validate-scalar-segred! segred out-sym array-types)
        dtype (or validated-dtype (:dtype segred) dtype)
        index (:name (segop/seg-space-reduced-dim (:space segred)))
        bound (:bound (segop/seg-space-reduced-dim (:space segred)))
        ;; Direct compatibility-front-door reductions can arrive before target scheduling. Keep
        ;; the portable baseline explicit here; scheduled TypedSOAC operations retain their
        ;; descriptor-derived block size and capped group count.
        workgroup-size (or (get-in segred [:grid :block-size]) 256)
        arrays (vec (sort-by name (:inputs segred)))
        scalars (vec (sort-by name (:scalars segred)))
        scalar-dtype (fn [id] (or (get scalar-types id)
                                  (get scalar-types (symbol (name id))) dtype))
        array-dtype (fn [id] (or (get array-types id)
                                 (get array-types (symbol (name id))) dtype))
        bound-dimension '_n_bound
        _ (when-not (and (contains? #{:float :double} (dtype/canon dtype))
                         (integer? workgroup-size) (pos? workgroup-size)
                         (zero? (bit-and workgroup-size (dec workgroup-size)))
                         (every? #(= (dtype/canon dtype) (dtype/canon (array-dtype %))) arrays)
                         (every? #(dtype/known? (dtype/canon (scalar-dtype %))) scalars))
            (decline! :uniform-scalar-storage
                      "KernelBody scalar reduction requires a static power-of-two workgroup and uniform tensor storage"
                      {:segred-id (:id segred) :dtype dtype :bound bound
                       :workgroup-size workgroup-size :arrays arrays :scalars scalars}))
        {:keys [operator identity element]} (scalar-plan segred)
        contraction-coordinate-proof?
        (when coordinate-proof
          (let [view (when (contraction-facts/facts? coordinate-proof)
                       (contraction-facts/scalar-reduction-view coordinate-proof))
                [proof-index proof-bound] (:flat-contract-axis coordinate-proof)
                operand-ids (set (map :sym (:operands coordinate-proof)))]
            (when-not (and view
                           (zero? (:n-free coordinate-proof))
                           (= [index bound] [proof-index proof-bound])
                           (= (set arrays) operand-ids)
                           ;; Facts retain every load occurrence. Validate each occurrence rather
                           ;; than looking up by symbol, which would select only the first read of
                           ;; an array and could hide a later unproved coordinate.
                           (every? #(contraction-facts/operand-axis-map coordinate-proof %)
                                   (:operands coordinate-proof))
                           (= element (:element view))
                           (constant/equivalent? identity (:neutral view))
                           (= operator (intrinsics/canonical (:combine view)))
                           (= (dtype/canon dtype) (dtype/canon (:dtype view))))
              (decline! :contraction-coordinate-proof
                        "affine scalar-contraction coordinates disagree with retained facts"
                        {:segred-id (:id segred) :facts coordinate-proof
                         :index index :bound bound :arrays arrays
                         :operator operator :identity identity :element element}))
            true))
        _ (when (contains? #{:min :max} operator)
            (decline! :floating-minmax-semantics
                      "portable scalar reduction needs an explicit NaN and signed-zero policy for min/max"
                      {:segred-id (:id segred) :operator operator}))
        identity (constant/literal-or-original identity)
        _ (when-not (number? identity)
            (decline! :literal-identity
                      "KernelBody scalar reduction requires a numeric identity"
                      {:segred-id (:id segred) :identity identity}))
        element-index 'element-index
        lane-accumulator 'lane-accumulator
        next-lane-accumulator 'next-lane-accumulator
        lane-result 'lane-result
        {:keys [operations result]}
        (lower-element-operations
         element
         {:index index :coordinate element-index :dtype dtype
          ;; SegRed's certified scalar region declares the dtype of its outer element value. This
          ;; authorizes only that result; nested calls still need retained walker/TypedClojure
          ;; facts in lower-element-operations.
          :declared-result-dtype dtype
          :arrays (set arrays) :scalars (set scalars)
          :scalar-types (into {} (map (fn [id] [id (scalar-dtype id)])) scalars)
          :coordinate-lower
          (fn [source-coordinate]
            ;; The first complete vertical proves every input has at least `bound` elements.
            ;; That proves only the pointwise coordinate. Affine/gathered reads need an explicit
            ;; view extent plus a range proof; admitting `i+offset` from expression syntax would
            ;; turn an n-element capacity check into a false memory-safety certificate.
            (if contraction-coordinate-proof?
              ;; Verified contraction facts prove every operand's AxisMap and physical extent.
              (widen-index-expression
               (index-expression/lower
                (util/subst-syms {index element-index} source-coordinate)
                (conj (set scalars) element-index)
                decline!)
               (assoc scalar-types element-index :long))
              (when (= index source-coordinate) element-index)))})
        group-count (if (= :block-local (:phase segred))
                      (launch/rebind-expression
                       (launch-group-count (get-in segred [:grid :num-blocks])
                                           bound workgroup-size)
                       {bound bound-dimension})
                      1)
        scratch 'workgroup-reduction-scratch
        barrier (fn [] (body/->WorkgroupBarrier
                        :workgroup #{:workgroup} :acquire-release (body/full-participation)))
        tree-stages
        (mapcat
         (fn [stride]
           (let [mask (keyword (str "reduce-stride-" stride))
                 left (symbol (str "tree-left-" stride))
                 right (symbol (str "tree-right-" stride))
                 combined (symbol (str "tree-combined-" stride))]
             [(body/->ScalarLoad (body/value left dtype) scratch ['local-index]
                                 mask (body/literal identity dtype) :cached)
              (body/->ScalarLoad
               (body/value right dtype) scratch
               [(body/expression :add 'local-index stride)]
               mask (body/literal identity dtype) :cached)
              (body/->ScalarCompute
               (body/value combined dtype)
               (body/scalar-expression operator dtype [left right]))
              (body/->ScalarStore scratch ['local-index] combined mask)
              (barrier)]))
         (take-while pos? (iterate #(quot % 2) (quot workgroup-size 2))))
        final-value 'workgroup-result
        parameters
        (vec (concat
              (map #(body/->KernelParameter
                     % :input dtype ['_n_bound] :global
                     (layout/row-major ['_n_bound] dtype) :operand)
                   arrays)
              [(body/->KernelParameter output :output dtype [group-count] :global
                                       (layout/row-major [group-count] dtype) :result)]
              (map #(body/->KernelParameter % :scalar (scalar-dtype %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))
        _ (when (and result-region (seq (:operands result-region)))
            (decline! :full-reduction-result-operands
                      "full-reduction result transforms cannot address tensor operands"
                      {:segred-id (:id segred) :operands (:operands result-region)}))
        transformed
        (when result-region
          (scalar-region-lower/lower
           result-region
           {:accumulator final-value :accumulator-dtype dtype :store-dtype dtype
            :parameters (into {} (map (fn [parameter] [(:id parameter) parameter])) parameters)
            :coordinate-lower (fn [_]
                                (decline! :full-reduction-result-operands
                                          "full-reduction result transform has no tensor axes"
                                          {:segred-id (:id segred)}))
            :id-prefix "completed-reduction"}))
        stored-value (or (:result transformed) final-value)
        kernel-body
        (body/make
         {:id [:segred (:id segred) :workgroup-tree]
          :parameters parameters
          :stable-reads (mapv body/stable-read arrays)
          :allocations [(body/->WorkgroupAllocation
                         scratch dtype [workgroup-size]
                         (layout/row-major [workgroup-size] dtype)
                         (dtype/bytes-of dtype))]
          :indices [(body/->IndexBinding 'group-index :group 0)
                    (body/->IndexBinding 'group-count :group-count 0)
                    (body/->IndexBinding 'local-index :local 0)
                    ;; Widen schedule arithmetic before multiplying or adding. An int32 bound may
                    ;; equal Integer/MAX_VALUE; inactive final-group lanes must not overflow while
                    ;; merely forming the coordinate that the loop predicate excludes.
                    (body/->IndexCompute
                     'wide-bound (body/index-cast '_n_bound :long :exact))
                    (body/->IndexCompute
                     'wide-group-index (body/index-cast 'group-index :long :exact))
                    (body/->IndexCompute
                     'wide-group-count (body/index-cast 'group-count :long :exact))
                    (body/->IndexCompute
                     'wide-local-index (body/index-cast 'local-index :long :exact))
                    (body/->IndexCompute
                     'group-chunk
                     ;; Ceil-div is defined for zero, so an empty reduction reaches the tree with
                     ;; one inactive, identity-valued workgroup and never forms `n - 1`.
                     (body/expression :ceil-div 'wide-bound 'wide-group-count))
                    (body/->IndexCompute
                     'group-start (body/expression :mul 'wide-group-index 'group-chunk))
                    (body/->IndexCompute
                     'group-length
                     (body/expression :min 'group-chunk
                                      (body/expression :sub 'wide-bound 'group-start)))
                    (body/->IndexCompute
                     'global-index (body/expression :add 'group-start 'wide-local-index))
                    (body/->IndexCompute
                     'group-end (body/expression :add 'group-start 'group-length))]
          :masks (vec (concat
                       [(body/->Mask :lane-zero [(body/predicate :eq 'local-index 0)])]
                       (map (fn [stride]
                              (body/->Mask (keyword (str "reduce-stride-" stride))
                                           [(body/predicate :lt 'local-index stride)]))
                            (take-while pos?
                                        (iterate #(quot % 2) (quot workgroup-size 2))))))
          :operations (vec (concat
                            [(body/->ForLoop
                              (body/value element-index :long)
                              'global-index 'group-end workgroup-size
                              [(body/->LoopArg (body/value lane-accumulator dtype)
                                               (body/literal identity dtype))]
                              (vec (concat
                                    operations
                                    [(body/->ScalarCompute
                                      (body/value next-lane-accumulator dtype)
                                      (body/scalar-expression
                                       operator dtype [lane-accumulator result]))
                                     (body/->Yield [next-lane-accumulator])]))
                              [(body/value lane-result dtype)]
                              {})
                             (body/->ScalarStore scratch ['local-index] lane-result nil)
                             (barrier)]
                            tree-stages
                            [(body/->ScalarLoad (body/value final-value dtype) scratch [0]
                                                :lane-zero (body/literal identity dtype) :cached)]
                            (:operations transformed)
                            [(body/->ScalarStore output ['group-index] stored-value :lane-zero)]))
          :schedule {:strategy :workgroup-tree
                     :workgroup-size workgroup-size
                     :reduction-operator operator}
          :launch (launch/spec
                   {:workgroup-size [workgroup-size]
                    :group-count [group-count]
                    :shared-memory-bytes (* workgroup-size (dtype/bytes-of dtype))})
          :provenance {:dialect :kernel-body :source-dialect :segred
                       :segop-id (:id segred)
                       :algorithm-dialect (:algorithm-dialect segred)}
          :attributes {:kind :scalar-reduction :identity identity
                       :operator operator}})]
    {:kernel-body kernel-body
     :operator operator
     :identity identity
     :arrays arrays
     :scalars scalars
     :output output
     :bound bound
     :group-count group-count
     :result-region result-region}))

(defn- logical-bound-dtype
  [bound scalar-types]
  (let [scalar-dtype (fn [id]
                       (or (get scalar-types id)
                           (get scalar-types (when (or (symbol? id) (keyword? id))
                                               (symbol (name id))))))]
    (try
      (launch/typed-expression-dtype bound scalar-dtype)
      (catch clojure.lang.ExceptionInfo exception
        (decline! :phase-bound-dtype
                  "scalar SegRed phase bound lacks a complete checked integer type"
                  {:bound bound :scalar-types scalar-types
                   :type-error (ex-data exception)})))))

(defn schedule
  "Refine one scalar SegRed phase into a complete target-neutral ScheduledKernelBody.

   The exact SegRed is the semantic source. `_n_bound` is target-private int storage bound to the
   phase's exact logical extent; a wider public extent carries an explicit checked-range proof.
   Each artifact writes exactly its launch group count, while only terminal phases may apply the
   typed completed-result transform."
  ([segred options]
   (schedule segred nil options))
  ([segred out-sym {:keys [scalar-types] :as options}]
   (let [{:keys [kernel-body operator identity arrays scalars output bound group-count result-region]}
        (lower segred out-sym
               :dtype (:dtype options) :array-types (:array-types options)
               :scalar-types scalar-types :coordinate-proof (:coordinate-proof options))
        arguments (mapv (fn [parameter]
                          (if (= '_n_bound (:id parameter)) bound (:id parameter)))
                        (:parameters kernel-body))
        bound-dtype (logical-bound-dtype bound scalar-types)
        scalar-bindings
        (mapv (fn [[parameter argument]]
                (let [kernel-dtype (dtype/canon (:dtype parameter))
                      logical-dtype (if (= '_n_bound (:id parameter))
                                      bound-dtype kernel-dtype)]
                  {:parameter (:id parameter) :value argument
                   :dtype logical-dtype :kernel-dtype kernel-dtype
                   :conversion (if (= logical-dtype kernel-dtype)
                                 :identity :checked-range)}))
              (filterv (fn [[parameter _]] (= :scalar (:kind parameter)))
                       (map vector (:parameters kernel-body) arguments)))
        phase (:phase segred)
        output-elements (launch/rebind-expression group-count {'_n_bound bound})
        c-op ({:+ "+" :* "*"} operator)
        result-dtype (dtype/canon (or (:result-dtype result-region) (:dtype segred)))]
    (when-not c-op
      (decline! :certified-monoid
                "portable scalar reduction has no emitted combine spelling"
                {:operation (:id segred) :operator operator}))
    (scheduled-body/make
     {:source segred
      :body kernel-body
      :arguments arguments
      :scalar-bindings scalar-bindings
      :effects {:kind :scalar-reduction-phase
                :phase phase
                :uses (scheduled-body/derive-uses kernel-body arguments)}
      :legality {:kind :scalar-segred-workgroup-tree
                 :product-components 1
                 :segment-dimensions 0
                 :phase phase
                 :workgroup-size (get-in kernel-body [:schedule :workgroup-size])
                 :power-of-two-workgroup true
                 :terminal (contains? #{:single :cross-block} phase)
                 :certified-monoid (get-in segred [:reduction :algebra])
                 :identity identity
                 :uniform-dtype (dtype/canon (:dtype segred))}
      :numerics (cond-> {:mode :reassociated
                         :policy :certified-workgroup-tree
                         :rounding :implementation-defined
                         :accumulator-dtype (dtype/canon (:dtype segred))}
                  result-region
                  (assoc :result-transform
                         {:kind :typed-scalar-region
                          :policy :same-typed-ssa-evaluation-order
                          :input-dtype (dtype/canon (:dtype segred))
                          :result-dtype result-dtype}))
      :provenance {:dialect :kernel-body :source-dialect :segred
                   :segop-id (:id segred) :phase phase}
      :attributes {:array-params arrays
                   :scalar-params scalars
                   :strategy :scalar-workgroup-tree
                   :dtype (dtype/canon (:dtype segred))
                   :phase phase
                   :group-count output-elements
                   :output-elements output-elements
                   :physical-result output
                   ;; Compatibility staging still combines partials on the host and consumes these
                   ;; values. They are projections of the proved monoid, not emitter inference.
                   :identity-val identity
                   :c-op c-op}}))))

(defn validate-against-node!
  "Close scalar SegRed over its exact source, body, launch, and KernelGraph storage facts."
  [scheduled node kernel-graph]
  (let [scheduled (scheduled-body/validate-against-node! scheduled node kernel-graph)
        source (:source scheduled)
        _ (when-not (instance? raster.compiler.ir.segop.SegRed source)
            (decline! :schedule-source
                      "scalar reduction storage closure requires an exact SegRed source"
                      {:source source :node (:id node)}))
        semantic-output (first (:outputs source))
        physical-result (get-in scheduled [:attributes :physical-result])
        output-use (some #(when (and (= semantic-output (:buffer %))
                                     (contains? #{:write :read-write} (:access %)))
                            %)
                         (:uses node))
        output-buffer (some #(when (= (:buffer output-use) (:id %)) %)
                            (concat (:inputs kernel-graph) (:outputs kernel-graph)
                                    (:temporaries kernel-graph)))
        output-parameter (some #(when (= :result (:role %)) %)
                               (get-in scheduled [:body :parameters]))
        parameters (get-in scheduled [:body :parameters])
        arguments (:arguments scheduled)
        bindings (into {} (map (fn [[parameter argument]] [(:id parameter) argument]))
                       (map vector parameters arguments))
        buffers (into {} (map (juxt :id identity))
                      (distinct (concat (:inputs kernel-graph) (:outputs kernel-graph)
                                        (:temporaries kernel-graph))))
        bound-binding (some #(when (= '_n_bound (:parameter %)) %)
                            (:scalar-bindings scheduled))
        realized-shape (mapv #(launch/rebind-expression
                               % {'_n_bound (:value bound-binding)})
                             (:shape output-parameter))
        output-elements (get-in scheduled [:attributes :output-elements])
        realized-launch (scheduled-body/realized-launch scheduled)
        group-count (get-in realized-launch [:group-count 0])
        workgroup-size (get-in source [:grid :block-size])
        shared-memory-bytes (get-in source [:grid :shared-mem-bytes])]
    (doseq [[parameter argument] (map vector parameters arguments)
            :when (not= :scalar (:kind parameter))]
      (let [buffer (get buffers argument)
            realized-shape (mapv #(launch/rebind-expression % bindings)
                                 (:shape parameter))
            realized-elements (case (count realized-shape)
                                0 1
                                1 (first realized-shape)
                                (apply launch/product realized-shape))
            source-elements (if (= :result (:role parameter))
                              output-elements
                              (:bound (segop/seg-space-reduced-dim (:space source))))]
        (when-not (= (dtype/canon (:dtype parameter)) (some-> buffer :dtype dtype/canon))
          (decline! :storage-dtype
                    "scalar SegRed pointer dtype differs from its KernelGraph buffer"
                    {:node (:id node) :parameter (:id parameter) :argument argument
                     :parameter-dtype (:dtype parameter) :buffer buffer}))
        (when-not (and buffer
                       (= source-elements realized-elements (:elements buffer)))
          (decline! :storage-extent
                    "scalar SegRed pointer extent differs across source, body, and graph"
                    {:node (:id node) :parameter (:id parameter) :argument argument
                     :source-elements source-elements :body-elements realized-elements
                     :buffer buffer}))))
    (when-not (and (= 1 (count (:outputs source)))
                   (= semantic-output physical-result (:id output-parameter))
                   output-use output-buffer output-parameter bound-binding
                   (= [output-elements] realized-shape)
                   (= output-elements group-count)
                   (= output-elements (:elements output-buffer)))
      (decline! :output-elements
                "scalar SegRed output extent differs from its launch or KernelGraph storage"
                {:node (:id node) :output-use output-use :output-buffer output-buffer
                 :output-parameter output-parameter :realized-shape realized-shape
                 :output-elements output-elements :group-count group-count}))
    (when-not (and (= (:phase source) (get-in scheduled [:effects :phase]))
                   (= (:phase source) (get-in scheduled [:legality :phase]))
                   (= (:phase source) (get-in scheduled [:attributes :phase]))
                   (= workgroup-size (get-in scheduled [:body :schedule :workgroup-size]))
                   (= [workgroup-size] (:workgroup-size realized-launch))
                   (= shared-memory-bytes (:shared-memory-bytes realized-launch)))
      (decline! :schedule-projection
                "scalar SegRed certificate changes its source phase or workgroup-tree geometry"
                {:node (:id node) :phase (:phase source)
                 :effects (:effects scheduled) :legality (:legality scheduled)
                 :attributes (:attributes scheduled) :launch realized-launch
                 :grid (:grid source)}))
    (let [array-types (into {} (map (juxt :id :dtype)) (vals buffers))
          scalar-types (into {} (map (juxt :id :dtype)) (:scalars kernel-graph))
          expected (schedule source semantic-output
                             {:dtype (:dtype source)
                              :array-types array-types
                              :scalar-types scalar-types})]
      (when-not (= expected scheduled)
        (decline! :schedule-source
                  "scalar SegRed scheduled body is not the exact refinement of its source"
                  {:node (:id node) :source (:id source)
                   :expected-body (:body expected) :actual-body (:body scheduled)})))
    scheduled))

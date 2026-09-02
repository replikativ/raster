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
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
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

(defn- literal-value
  [value]
  (if (and (seq? value)
           (= 2 (count value))
           (descriptor/cast-op? (first value))
           (number? (second value)))
    (second value)
    (case value
      Double/POSITIVE_INFINITY Double/POSITIVE_INFINITY
      Double/NEGATIVE_INFINITY Double/NEGATIVE_INFINITY
      Float/POSITIVE_INFINITY Float/POSITIVE_INFINITY
      Float/NEGATIVE_INFINITY Float/NEGATIVE_INFINITY
      value)))

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
    {:operator operator :identity (literal-value init) :element (:element derived)
     :accumulator acc}))

(defn launch-group-count
  "Translate SegRed's historical capped-grid expression into inspectable launch IR.

   `compute-launch-params` predates KernelLaunch and represents the reduction grid as
   `(min occupancy-cap (int (Math/ceil (/ (double bound) block-size))))`.  Do not carry that
   executable host form into KernelBody: recognize the exact producer contract and rebuild it
   from the authoritative SegSpace bound and workgroup size."
  [grid-expression bound workgroup-size]
  (cond
    (and (integer? grid-expression) (pos? grid-expression))
    grid-expression

    (symbol? grid-expression)
    (launch/runtime-value grid-expression)

    (and (seq? grid-expression)
         (contains? '#{min clojure.core/min} (first grid-expression))
         (= 3 (count grid-expression))
         (integer? (second grid-expression))
         (pos? (second grid-expression)))
    (launch/minimum (second grid-expression)
                    (launch/ceil-div bound workgroup-size))

    :else
    (decline! :launch-grid
              "KernelBody scalar reduction requires an explicit or canonical capped SegRed grid"
              {:grid-expression grid-expression :bound bound
               :workgroup-size workgroup-size})))

(defn- strip-index-cast
  [expression]
  (if (and (seq? expression)
           (contains? #{'int 'long 'clojure.core/int 'clojure.core/long} (first expression))
           (= 2 (count expression)))
    (second expression)
    expression))

(defn- cast-policy
  [source target]
  (let [source (dtype/canon source)
        target (dtype/canon target)]
    (cond
      (= source target) nil

      (and (dtype/fp-dtype? source) (dtype/fp-dtype? target))
      (if (< (dtype/bytes-of source) (dtype/bytes-of target))
        [:exact :exact]
        [:nearest-even :ieee])

      (and (dtype/integral? source) (dtype/fp-dtype? target))
      (if (and (= :double target) (<= (dtype/bytes-of source) 4))
        [:exact :exact]
        [:nearest-even (if (= :half target) :ieee :exact)])

      (and (dtype/fp-dtype? source) (dtype/integral? target))
      (decline! :checked-scalar-cast
                "KernelBody reduction cannot preserve a checked floating-to-integral cast"
                {:source-dtype source :target-dtype target})

      (and (dtype/integral? source) (dtype/integral? target))
      (if (< (dtype/bytes-of source) (dtype/bytes-of target))
        [:exact :exact]
        (decline! :checked-scalar-cast
                  "KernelBody reduction cannot preserve a checked narrowing integral cast"
                  {:source-dtype source :target-dtype target}))

      :else
      (decline! :scalar-cast
                "KernelBody reduction has no explicit numerical cast policy"
                {:source-dtype source :target-dtype target}))))

(defn lower-element-operations
  "Lower a scalar reduction element into typed SSA. coordinate-lower may translate a verified
   source-level flat array index into KernelBody index arithmetic; without it, this retains the
   pointwise full-reduction contract. Mixed scalar arithmetic requires the walker/TypedClojure
   result dtype retained on that expression. The lowered element must already match the certified
   accumulator dtype; this pass never invents a final narrowing conversion. A typed region owner
   may provide `declared-result-dtype` for the outer expression only; nested calls still require
   their own retained facts."
  [expression {:keys [index coordinate dtype arrays scalars scalar-types coordinate-lower
                      load-predicate load-other declared-result-dtype]}]
  (let [dtype (dtype/canon dtype)
        operations (atom [])
        counter (atom 0)
        fresh (fn [prefix] (symbol (str prefix "-" (swap! counter inc))))
        emit! (fn [operation value value-dtype]
                (swap! operations conj operation)
                {:value value :dtype (dtype/canon value-dtype)})]
    (letfn [(cast! [{:keys [value dtype] :as typed} target]
              (let [source (dtype/canon dtype)
                    target (dtype/canon target)]
                (if (= source target)
                  typed
                  (let [[rounding overflow] (cast-policy source target)
                        result (fresh "element-cast")]
                    (emit! (body/->ScalarCompute
                            (body/value result target)
                            (body/cast-expression value target rounding overflow))
                           result target)))))

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
                    (let [result (fresh "element-load")]
                      (emit! (body/->ScalarLoad
                              (body/value result dtype) array [lowered-coordinate]
                              load-predicate
                              (when load-predicate
                                (or load-other (body/literal 0 dtype)))
                              :cached)
                             result dtype)))

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
                        input-dtypes (mapv :dtype typed-inputs)
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
                    (let [inputs (mapv (comp :value #(cast! % result-dtype)) typed-inputs)
                          result (fresh "element-value")]
                      (emit! (body/->ScalarCompute
                              (body/value result result-dtype)
                              (body/scalar-expression operator result-dtype inputs))
                             result result-dtype)))

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
  [segred out-sym & {:keys [dtype array-types scalar-types]
                     :or {dtype :double array-types {} scalar-types {}}}]
  (let [dtype (or (:dtype segred) dtype)
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
        bound-dimension (if (or (symbol? bound) (keyword? bound) (vector? bound))
                          (launch/runtime-value bound)
                          bound)
        _ (when-not (and (contains? #{:float :double} (dtype/canon dtype))
                         (integer? workgroup-size) (pos? workgroup-size)
                         (zero? (bit-and workgroup-size (dec workgroup-size)))
                         (launch/dimension-expression? bound-dimension)
                         (every? #(= (dtype/canon dtype) (dtype/canon (array-dtype %))) arrays)
                         (every? #(dtype/known? (dtype/canon (scalar-dtype %))) scalars))
            (decline! :uniform-scalar-storage
                      "KernelBody scalar reduction requires a static power-of-two workgroup and uniform tensor storage"
                      {:segred-id (:id segred) :dtype dtype :bound bound
                       :workgroup-size workgroup-size :arrays arrays :scalars scalars}))
        {:keys [operator identity element]} (scalar-plan segred)
        identity (literal-value identity)
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
            (index-expression/lower
             (util/subst-syms {index element-index} source-coordinate)
             (conj (set scalars) element-index)
             decline!))})
        output (or out-sym 'output)
        group-count (if-let [grid-expression (get-in segred [:grid :num-blocks])]
                      (launch-group-count grid-expression bound workgroup-size)
                      (launch/ceil-div bound-dimension workgroup-size))
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
              [(body/->KernelParameter output :output dtype ['partial-count] :global
                                       (layout/row-major ['partial-count] dtype) :result)]
              (map #(body/->KernelParameter % :scalar (scalar-dtype %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))
        result-region (get-in segred [:reduction :attributes :result-region])
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
                    (body/->IndexCompute
                     'group-chunk
                     (body/expression :ceil-div '_n_bound 'group-count))
                    (body/->IndexCompute
                     'group-start (body/expression :mul 'group-index 'group-chunk))
                    (body/->IndexCompute
                     'group-end
                     (body/expression :min '_n_bound
                                      (body/expression :add 'group-start 'group-chunk)))
                    (body/->IndexCompute
                     'global-index (body/expression :add 'group-start 'local-index))]
          :masks (vec (concat
                       [(body/->Mask :lane-zero [(body/predicate :eq 'local-index 0)])]
                       (map (fn [stride]
                              (body/->Mask (keyword (str "reduce-stride-" stride))
                                           [(body/predicate :lt 'local-index stride)]))
                            (take-while pos?
                                        (iterate #(quot % 2) (quot workgroup-size 2))))))
          :operations (vec (concat
                            [(body/->ForLoop
                              (body/value element-index :int)
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
     :bound bound}))

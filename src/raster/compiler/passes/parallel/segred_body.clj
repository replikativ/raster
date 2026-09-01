(ns raster.compiler.passes.parallel.segred-body
  "Schedule an ordinary scalar SegRed as target-neutral KernelBody data.

   This first production slice deliberately models the same portable workgroup tree used by the
   established OpenCL emitter: occupancy-capped workgroups, a per-lane sequential chunk, identity
   padding, workgroup scratch, a barrier-separated binary tree, and one partial result per
   workgroup. Scalar expressions become typed SSA operations; target emitters never recover either
   the algorithm or schedule from Clojure source spelling. Unsupported scalar regions decline
   structurally to the established emitter."
  (:require [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]))

(def ^:private associative-operators #{:+ :* :min :max})
(def ^:private cast-heads
  {'byte :byte, 'clojure.core/byte :byte
   'int :int, 'clojure.core/int :int
   'long :long, 'clojure.core/long :long
   'float :float, 'clojure.core/float :float
   'double :double, 'clojure.core/double :double})

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data
                                 :reason :segred-kernel-body-declined
                                 :missing-rule rule
                                 :fallback :verified-segred-opencl))))

(defn declined?
  [exception]
  (= :segred-kernel-body-declined (:reason (ex-data exception))))

(defn- inline-scalar-bindings
  [expression]
  (if (and (seq? expression) (contains? #{'let 'let*} (first expression)))
    (let [[_ bindings & body] expression
          initializers (vec (take-nth 2 (rest bindings)))]
      (when-not (= 1 (count body))
        (decline! :multi-expression-let
                  "KernelBody scalar reduction requires a single-expression let region"
                  {:expression expression}))
      (when (some util/effectful? initializers)
        (decline! :effectful-scalar-binding
                  "KernelBody scalar reduction cannot inline effectful scalar bindings"
                  {:expression expression}))
      (recur (util/subst-syms (util/binding-env bindings) (first body))))
    expression))

(defn- accumulator-use?
  [accumulator expression]
  (or (= accumulator expression)
      (and (seq? expression)
           (contains? (set (keys cast-heads)) (first expression))
           (= 2 (count expression))
           (= accumulator (second expression)))))

(defn scalar-plan
  "Project one scalar SegRed into an explicit combine operator, identity and element expression."
  [segred]
  (let [{:keys [acc init lambda]} (segop/scalar-reduce-op segred)
        expression (inline-scalar-bindings lambda)
        operator (when (seq? expression)
                   (intrinsics/canonical (descriptor/semantic-op expression)))
        arguments (vec (when (seq? expression) (descriptor/call-args expression)))]
    (when-not (and acc (some? init) (contains? associative-operators operator)
                   (= 2 (count arguments)))
      (decline! :scalar-combine
                "KernelBody reduction requires a binary associative scalar combine"
                {:segred-id (:id segred) :accumulator acc :identity init
                 :operator operator :arguments arguments :lambda lambda}))
    (let [[left right] arguments
          left-accumulator? (accumulator-use? acc left)
          right-accumulator? (accumulator-use? acc right)
          element (if left-accumulator? right left)]
      (when (= left-accumulator? right-accumulator?)
        (decline! :accumulator-position
                  "KernelBody reduction combine does not contain its accumulator exactly once"
                  {:segred-id (:id segred) :accumulator acc :arguments arguments}))
      {:operator operator :identity init :element element :accumulator acc})))

(defn- literal-value
  [value]
  (case value
    Double/POSITIVE_INFINITY Double/POSITIVE_INFINITY
    Double/NEGATIVE_INFINITY Double/NEGATIVE_INFINITY
    Float/POSITIVE_INFINITY Float/POSITIVE_INFINITY
    Float/NEGATIVE_INFINITY Float/NEGATIVE_INFINITY
    value))

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

(defn lower-element-operations
  "Lower a scalar reduction element into typed SSA. `coordinate-lower` may translate a verified
   source-level flat array index into KernelBody index arithmetic; without it, this retains the
   pointwise full-reduction contract. A predicate requires an explicit typed load fallback."
  [expression {:keys [index coordinate dtype arrays scalars coordinate-lower
                      load-predicate load-other]}]
  (let [operations (atom [])
        counter (atom 0)
        fresh (fn [prefix] (symbol (str prefix "-" (swap! counter inc))))
        emit! (fn [operation result]
                (swap! operations conj operation)
                result)]
    (letfn [(lower [expression]
              (let [expression (inline-scalar-bindings expression)]
                (cond
                  (number? expression)
                  (body/literal expression dtype)

                  (symbol? expression)
                  (if (contains? scalars expression)
                    expression
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
                             result)))

                  (and (seq? expression) (contains? cast-heads (first expression))
                       (= 2 (count expression)))
                  (let [target (get cast-heads (first expression))]
                    (when-not (= (dtype/canon target) (dtype/canon dtype))
                      (decline! :scalar-cast
                                "initial KernelBody reduction requires element casts to its dtype"
                                {:expression expression :source-dtype dtype :target-dtype target}))
                    (lower (second expression)))

                  (seq? expression)
                  (let [operator (intrinsics/canonical (descriptor/semantic-op expression))
                        descriptor (intrinsics/descriptor operator)
                        arguments (vec (descriptor/call-args expression))]
                    (when-not (and descriptor
                                   (= (:arity descriptor) (count arguments))
                                   (intrinsics/accepts-scalar-dtype? operator dtype)
                                   (not= :cmp (:kind descriptor)))
                      (decline! :scalar-expression
                                "KernelBody element expression contains an unsupported scalar operation"
                                {:expression expression :operator operator :dtype dtype}))
                    (let [inputs (mapv lower arguments)
                          result (fresh "element-value")]
                      (emit! (body/->ScalarCompute
                              (body/value result dtype)
                              (body/scalar-expression operator dtype inputs))
                             result)))

                  :else
                  (decline! :scalar-expression
                            "KernelBody element expression has an unsupported value"
                            {:expression expression :type (type expression)}))))]
      (let [result (lower expression)]
        {:operations @operations :result result}))))

(defn lower
  "Lower an eligible scalar SegRed to one verified portable workgroup-tree KernelBody.

   `array-types` and `scalar-types` are authoritative ABI facts. This vertical initially accepts
   the uniform-dtype storage contract already implemented by the scalar SegRed runtime."
  [segred out-sym & {:keys [dtype array-types scalar-types]
                     :or {dtype :double array-types {} scalar-types {}}}]
  (let [dtype (or (:dtype segred) dtype)
        index (:name (segop/seg-space-reduced-dim (:space segred)))
        bound (:bound (segop/seg-space-reduced-dim (:space segred)))
        workgroup-size (get-in segred [:grid :block-size])
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
                         (every? #(= (dtype/canon dtype) (dtype/canon (scalar-dtype %))) scalars))
            (decline! :uniform-scalar-storage
                      "KernelBody scalar reduction requires a static power-of-two workgroup and uniform scalar storage"
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
        (lower-element-operations element {:index index :coordinate element-index :dtype dtype
                                           :arrays (set arrays) :scalars (set scalars)})
        output (or out-sym 'output)
        group-count (launch-group-count (get-in segred [:grid :num-blocks])
                                        bound workgroup-size)
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

(ns raster.compiler.passes.parallel.indexed-weighted-reduction-body
  "Target-neutral scalar correctness schedule for indexed dense weighted reductions."
  (:require [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- lit [value type] (body/literal value type))
(defn- expr [op type & arguments] (body/scalar-expression op type arguments))
(defn- bounded-expr [op type & arguments]
  (body/scalar-expression op type arguments {:overflow :no-overflow}))
(defn- compute [id type expression]
  (body/->ScalarCompute (body/value id type) expression))
(defn- select [condition if-true if-false type]
  (expr :select type condition if-true if-false))
(defn- conjunction [& predicates]
  (reduce #(select %1 %2 (lit false :predicate) :predicate)
          (lit true :predicate) predicates))

(defn- lower-reference*
  "Lower one indexed edge-list reduction to KernelBody.

  One work-item owns one destination/feature. It retains the historical correctness schedule:
  ordered multiset traversal, private numerator/denominator, no edge-sized intermediates, and a
  NaN result for every active head component when any edge index is malformed. Unused row
  tails remain zero; malformed shapes produce NaN for every launched output."
  [plan {:keys [entities edges heads components total-dim] :as shape} workgroup-x dynamic?]
  (let [plan (swr/validate! plan)
        [q k v destination-indices source-indices] (:operands plan)
        output (:output plan)
        dtype (:accumulator-dtype plan)
        long-value (fn [value] (if dynamic? value (lit value :long)))
        active-width (if dynamic? 'active-width (* heads components))
        final-entity (if dynamic? 'final-entity (lit (dec entities) :long))
        bound (double (get-in plan [:score :arguments 2 :value]))
        epsilon (double (get-in plan [:normalization :epsilon]))
        scale (if dynamic? 'scale-value (/ 1.0 (Math/sqrt (double components))))
        group-x 'indexed-group-x
        group-y 'indexed-group-y
        lane-x 'indexed-lane-x
        feature 'feature
        destination 'destination
        edge 'edge
        x 'x
        dot-loop
        (body/->ForLoop
         (body/value x :long)
         (body/index-cast 0 :long :exact) (body/index-cast components :long :exact) 1
         [(body/->LoopArg (body/value 'dot-state dtype) (lit 0.0 dtype))]
         [(compute 'qk-component :long
                   (body/scalar-expression
                    :+ :long
                    [(body/scalar-expression
                      :* :long ['head (long-value components)]
                      {:overflow (if dynamic? :wrap :no-overflow)}) x]
                    {:overflow (if dynamic? :wrap :no-overflow)}))
          (body/->ScalarLoad (body/value 'q-element dtype) (:id q)
                             [destination 'qk-component] nil nil :cached)
          (body/->ScalarLoad (body/value 'k-element dtype) (:id k)
                             ['safe-source 'qk-component] nil nil :cached)
          (compute 'dot-product dtype (expr :* dtype 'q-element 'k-element))
          (compute 'dot-next dtype (expr :+ dtype 'dot-state 'dot-product))
          (body/->Yield ['dot-next])]
         [(body/value 'dot dtype)] {})
        edge-loop
        (body/->ForLoop
         (body/value edge :long)
         (body/index-cast 0 :long :exact) (body/index-cast edges :long :exact) 1
         [(body/->LoopArg (body/value 'valid-state :predicate) (lit true :predicate))
          (body/->LoopArg (body/value 'numerator-state dtype) (lit 0.0 dtype))
          (body/->LoopArg (body/value 'denominator-state dtype) (lit 0.0 dtype))]
         [(body/->ScalarLoad (body/value 'edge-destination :long) (:id destination-indices)
                             [edge] nil nil :cached)
          (body/->ScalarLoad (body/value 'edge-source :long) (:id source-indices)
                             [edge] nil nil :cached)
          (compute 'edge-destination-nonnegative :predicate
                   (expr :le :predicate (lit 0 :long) 'edge-destination))
          (compute 'edge-destination-bounded :predicate
                   (expr :lt :predicate 'edge-destination (long-value entities)))
          (compute 'edge-source-nonnegative :predicate
                   (expr :le :predicate (lit 0 :long) 'edge-source))
          (compute 'edge-source-bounded :predicate
                   (expr :lt :predicate 'edge-source (long-value entities)))
          (compute 'edge-valid :predicate
                   (conjunction 'edge-destination-nonnegative 'edge-destination-bounded
                                'edge-source-nonnegative 'edge-source-bounded))
          (compute 'valid-next :predicate
                   (select 'valid-state 'edge-valid (lit false :predicate) :predicate))
          (compute 'safe-source :long
                   (expr :min :long
                         (expr :max :long 'edge-source (lit 0 :long))
                         final-entity))
          (compute 'destination-match :predicate
                   (expr :eq :predicate 'edge-destination destination))
          (compute 'member-applies :predicate
                   (conjunction 'edge-valid 'destination-match))
          (body/->IfRegion
           'member-applies
           [(compute 'head :long (expr :quot :long feature (long-value components)))
            dot-loop
            (compute 'scaled dtype (expr :* dtype 'dot (if dynamic? scale (lit scale dtype))))
            (compute 'scaled-is-nan :predicate
                     (body/scalar-expression :isnan :predicate ['scaled]))
            (compute 'clamped dtype
                     (expr :min dtype (lit bound dtype)
                           (expr :max dtype (lit (- bound) dtype) 'scaled)))
            (compute 'score dtype (select 'scaled-is-nan 'scaled 'clamped dtype))
            (compute 'weight dtype (expr :exp dtype 'score))
            (body/->ScalarLoad (body/value 'value-element dtype) (:id v)
                               ['safe-source feature] nil nil :cached)
            (compute 'weighted-value dtype (expr :* dtype 'weight 'value-element))
            (compute 'numerator-updated dtype
                     (expr :+ dtype 'numerator-state 'weighted-value))
            (compute 'denominator-updated dtype
                     (expr :+ dtype 'denominator-state 'weight))
            (body/->Yield ['numerator-updated 'denominator-updated])]
           [(body/->Yield ['numerator-state 'denominator-state])]
           [(body/value 'numerator-next dtype)
            (body/value 'denominator-next dtype)])
          (body/->Yield ['valid-next 'numerator-next 'denominator-next])]
         [(body/value 'final-valid :predicate)
          (body/value 'final-numerator dtype)
          (body/value 'final-denominator dtype)] {})
        denominator-zero (compute 'denominator-zero :predicate
                                  (expr :eq :predicate 'final-denominator (lit 0.0 dtype)))
        quotient (compute 'normalized-value dtype
                          (expr :div dtype 'final-numerator
                                (expr :+ dtype 'final-denominator (lit epsilon dtype))))
        result (compute 'valid-result dtype
                        (select 'denominator-zero (lit 0.0 dtype) 'normalized-value dtype))
        active-operations
        [(compute 'feature-active :predicate
                  (expr :lt :predicate feature (if dynamic? active-width
                                                    (lit active-width :long))))
         (body/->IfRegion
          'feature-active
          [edge-loop denominator-zero quotient result
           (compute 'final-result dtype
                    (select 'final-valid 'valid-result (lit Double/NaN dtype) dtype))
           (body/->ScalarStore (:id output) [destination feature] 'final-result nil)
           (body/->Yield [])]
          [(body/->ScalarStore (:id output) [destination feature] (lit 0.0 dtype) nil)
           (body/->Yield [])]
          [])
         (body/->Yield [])]
        dynamic-derived
        (when dynamic?
          [(compute 'entities-positive :predicate
                    (expr :lt :predicate (lit 0 :long) entities))
           (compute 'edges-nonnegative :predicate
                    (expr :le :predicate (lit 0 :long) edges))
           (compute 'heads-positive :predicate
                    (expr :lt :predicate (lit 0 :long) heads))
           (compute 'components-positive :predicate
                    (expr :lt :predicate (lit 0 :long) components))
           (compute 'safe-components :long
                    (expr :max :long components (lit 1 :long)))
           (compute 'head-layout-fits :predicate
                    (expr :le :predicate heads (expr :quot :long total-dim 'safe-components)))
           (compute 'shape-valid :predicate
                    (conjunction 'entities-positive 'edges-nonnegative 'heads-positive
                                 'components-positive 'head-layout-fits))
           (compute 'active-width :long
                    (body/scalar-expression :* :long [heads components] {:overflow :wrap}))
           (compute 'final-entity :long
                    (body/scalar-expression
                     :- :long [(expr :max :long entities (lit 1 :long)) (lit 1 :long)]
                     {:overflow :wrap}))
           (compute 'components-fp dtype
                    (body/cast-expression components dtype :nearest-even :exact))
           (compute 'scale-value dtype
                    (expr :div dtype (lit 1.0 dtype) (expr :sqrt dtype 'components-fp)))])
        feature-operations
        (if dynamic?
          [(body/->IfRegion
            'shape-valid active-operations
            [(body/->ScalarStore (:id output) [destination feature] (lit Double/NaN dtype) nil)
             (body/->Yield [])]
            [])
           (body/->Yield [])]
          active-operations)
        scalar-parameters
        (when dynamic?
          (mapv #(body/->KernelParameter % :scalar :long [] nil nil :shape)
                [entities edges total-dim heads components 'output_elements]))]
    (body/make
     {:id [:indexed-weighted-reduction-body (:id plan) shape]
      :parameters (vec (concat
                        [(body/->KernelParameter
                          (:id q) :input dtype [entities total-dim] :global
                          (layout/row-major [entities total-dim] dtype) :query)
                         (body/->KernelParameter
                          (:id k) :input dtype [entities total-dim] :global
                          (layout/row-major [entities total-dim] dtype) :key)
                         (body/->KernelParameter
                          (:id v) :input dtype [entities total-dim] :global
                          (layout/row-major [entities total-dim] dtype) :value)
                         (body/->KernelParameter
                          (:id destination-indices) :input :long [edges] :global
                          (layout/row-major [edges] :long) :destination-indices)
                         (body/->KernelParameter
                          (:id source-indices) :input :long [edges] :global
                          (layout/row-major [edges] :long) :source-indices)
                         (body/->KernelParameter
                          (:id output) :output dtype [entities total-dim] :global
                          (layout/row-major [entities total-dim] dtype) :result)]
                        scalar-parameters))
      :stable-reads (mapv body/stable-read (swr/ordered-input-ids plan))
      :indices [(body/->IndexBinding group-x :group 0)
                (body/->IndexBinding lane-x :local 0)
                (body/->IndexBinding group-y :group 1)
                (body/->IndexCompute destination
                                     (body/index-cast group-y :long :exact))
                (body/->IndexCompute feature
                                     (body/index-cast
                                      (body/expression :add
                                                       (body/expression :mul group-x workgroup-x)
                                                       lane-x)
                                      :long :exact))]
      :masks []
      :operations
      (vec (concat dynamic-derived
       [(compute 'feature-bounded :predicate
                (expr :lt :predicate feature (long-value total-dim)))
       (body/->IfRegion
        'feature-bounded
        feature-operations
        [(body/->Yield [])]
        [])]))
      :schedule {:strategy :indexed-segmented-reduction-reference
                 :membership-traversal :ordered-edge-list
                 :score-reuse :per-output-component}
      :launch (launch/spec
               {:workgroup-size [workgroup-x 1]
                :group-count (if dynamic?
                               [(launch/ceil-div (launch/runtime-value total-dim) workgroup-x)
                                (launch/runtime-value entities)]
                               [(long (quot (+ total-dim (dec workgroup-x)) workgroup-x))
                                entities])})
      :provenance {:dialect :kernel-body
                   :semantic-op :segmented-weighted-reduction
                   :algebra-plan-id (:id plan)
                   :lowering :indexed-reference-kernel-body}
      :attributes {:storage-kind :indexed-dense-values
                   :membership-kind :edge-list-by-destination
                   :duplicate-policy :multiset
                   :dynamic-shape? dynamic?}})))

(defn lower-reference
  "Lower a statically specialized indexed edge-list correctness schedule."
  [plan shape workgroup-x]
  (lower-reference* plan shape workgroup-x false))

(defn lower-dynamic-reference
  "Lower the same correctness schedule with ordered int64 runtime extents."
  [plan workgroup-x]
  (lower-reference* plan
                    {:entities 'n_entities :edges 'n_edges :heads 'n_heads
                     :components 'n_components :total-dim 'total_dim}
                    workgroup-x true))

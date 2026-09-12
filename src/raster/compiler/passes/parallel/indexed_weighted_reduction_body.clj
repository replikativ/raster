(ns raster.compiler.passes.parallel.indexed-weighted-reduction-body
  "Target-neutral reference and subgroup schedules for indexed dense weighted reductions."
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

(defn lower-dynamic-score-reuse
  "One subgroup owns a destination/head/component tile. Edge traversal is uniform; each
  lane accumulates a strided dot fragment, then the subgroup shares one score. No collective
  occurs inside the lane-varying dot loop or the guarded value update."
  [plan width]
  (let [plan (swr/validate! plan)
        [q k v dst src] (mapv :id (:operands plan))
        out (get-in plan [:output :id])
        dtype (:accumulator-dtype plan)
        zero (lit 0.0 dtype)
        nan (lit Double/NaN dtype)
        integer-op (fn [op & args]
                     (body/scalar-expression op :long args {:overflow :wrap}))
        load (fn [id type buffer coords]
               (body/->ScalarLoad (body/value id type) buffer coords nil nil :cached))
        yield (fn [& xs] (body/->Yield (vec xs)))
        bound (double (get-in plan [:score :arguments 2 :value]))
        epsilon (double (get-in plan [:normalization :epsilon]))
        dot-loop
        (body/->ForLoop
         (body/value 'x :long) (body/index-cast 'lane :long :exact)
         (body/index-cast 'n_components :long :exact) width
         [(body/->LoopArg (body/value 'dot-state dtype) zero)]
         [(compute 'qk-component :long
                   (integer-op :+ (integer-op :* 'head 'n_components) 'x))
          (load 'q-element dtype q ['destination 'qk-component])
          (load 'k-element dtype k ['safe-source 'qk-component])
          (compute 'dot-next dtype
                   (expr :+ dtype 'dot-state (expr :* dtype 'q-element 'k-element)))
          (yield 'dot-next)]
         [(body/value 'partial-dot dtype)] {})
        edge-loop
        (body/->ForLoop
         (body/value 'edge :long) (body/index-cast 0 :long :exact)
         (body/index-cast 'n_edges :long :exact) 1
         [(body/->LoopArg (body/value 'valid-state :predicate) 'shape-valid)
          (body/->LoopArg (body/value 'numerator-state dtype) zero)
          (body/->LoopArg (body/value 'denominator-state dtype) zero)]
         [(load 'edge-destination :long dst ['edge])
          (load 'edge-source :long src ['edge])
          (compute 'edge-valid :predicate
                   (conjunction (expr :le :predicate (lit 0 :long) 'edge-destination)
                                (expr :lt :predicate 'edge-destination 'n_entities)
                                (expr :le :predicate (lit 0 :long) 'edge-source)
                                (expr :lt :predicate 'edge-source 'n_entities)))
          (compute 'valid-next :predicate (conjunction 'valid-state 'edge-valid))
          (compute 'visible :predicate
                   (conjunction 'valid-next
                                (expr :eq :predicate 'edge-destination 'destination)))
          (compute 'safe-source :long
                   (expr :min :long (expr :max :long 'edge-source (lit 0 :long))
                         'final-entity))
          (body/->IfRegion 'visible [dot-loop (yield 'partial-dot)] [(yield zero)]
                           [(body/value 'local-dot dtype)])
          (body/->Collective (body/value 'dot dtype) :reduce :subgroup width
                             'local-dot :+ nil (body/full-participation)
                             :implementation-defined)
          (compute 'score-owner :predicate
                   (conjunction 'visible (expr :eq :predicate 'lane-long (lit 0 :long))))
          (body/->IfRegion
           'score-owner
           [(compute 'scaled dtype (expr :* dtype 'dot 'scale-value))
            (compute 'scaled-is-nan :predicate (expr :isnan :predicate 'scaled))
            (compute 'clamped dtype
                     (expr :min dtype (lit bound dtype)
                           (expr :max dtype (lit (- bound) dtype) 'scaled)))
            (compute 'score dtype (select 'scaled-is-nan 'scaled 'clamped dtype))
            (compute 'local-weight-value dtype (expr :exp dtype 'score))
            (yield 'local-weight-value)]
           [(yield zero)] [(body/value 'local-weight dtype)])
          (body/->Collective (body/value 'weight dtype) :broadcast :subgroup width
                             'local-weight nil 0 (body/full-participation) nil)
          (compute 'denominator-next dtype (expr :+ dtype 'denominator-state 'weight))
          (compute 'value-active :predicate (conjunction 'active 'visible))
          (body/->IfRegion
           'value-active
           [(load 'value-element dtype v ['safe-source 'feature])
            (compute 'numerator-updated dtype
                     (expr :+ dtype 'numerator-state (expr :* dtype 'weight 'value-element)))
            (yield 'numerator-updated)]
           [(yield 'numerator-state)] [(body/value 'numerator-next dtype)])
          (yield 'valid-next 'numerator-next 'denominator-next)]
         [(body/value 'final-valid :predicate) (body/value 'final-numerator dtype)
          (body/value 'final-denominator dtype)] {})
        fp-parameter (fn [id kind role]
                       (body/->KernelParameter id kind dtype ['n_entities 'total_dim] :global
                                               (layout/row-major ['n_entities 'total_dim] dtype) role))]
    (body/make
     {:id [:indexed-score-reuse-body (:id plan) width]
      :parameters (into [(fp-parameter q :input :query) (fp-parameter k :input :key)
                         (fp-parameter v :input :value)
                         (body/->KernelParameter dst :input :long ['n_edges] :global
                                                 (layout/row-major ['n_edges] :long) :destination-indices)
                         (body/->KernelParameter src :input :long ['n_edges] :global
                                                 (layout/row-major ['n_edges] :long) :source-indices)
                         (fp-parameter out :output :result)]
                        (map #(body/->KernelParameter % :scalar :long [] nil nil :shape)
                             '[n_entities n_edges total_dim n_heads n_components output_elements]))
      :stable-reads (mapv body/stable-read [q k v dst src])
      :indices [(body/->IndexBinding 'lane :lane 0)
                (body/->IndexBinding 'tile-index :group 0)
                (body/->IndexBinding 'head-index :group 1)
                (body/->IndexBinding 'destination-index :group 2)
                (body/->IndexCompute 'lane-long (body/index-cast 'lane :long :exact))
                (body/->IndexCompute 'tile (body/index-cast 'tile-index :long :exact))
                (body/->IndexCompute 'head (body/index-cast 'head-index :long :exact))
                (body/->IndexCompute 'destination (body/index-cast 'destination-index :long :exact))]
      :operations
      [(compute 'component :long (integer-op :+ (integer-op :* 'tile (lit width :long)) 'lane-long))
       (compute 'feature :long (integer-op :+ (integer-op :* 'head 'n_components) 'component))
       (compute 'active :predicate
                (conjunction (expr :lt :predicate 'component 'n_components)
                             (expr :lt :predicate 'feature 'total_dim)))
       (compute 'safe-components :long (expr :max :long 'n_components (lit 1 :long)))
       (compute 'shape-valid :predicate
                (conjunction (expr :lt :predicate (lit 0 :long) 'n_entities)
                             (expr :le :predicate (lit 0 :long) 'n_edges)
                             (expr :lt :predicate (lit 0 :long) 'n_heads)
                             (expr :lt :predicate (lit 0 :long) 'n_components)
                             (expr :le :predicate 'n_heads
                                   (expr :quot :long 'total_dim 'safe-components))))
       (compute 'final-entity :long (integer-op :- (expr :max :long 'n_entities (lit 1 :long)) (lit 1 :long)))
       (compute 'components-fp dtype (body/cast-expression 'n_components dtype :nearest-even :exact))
       (compute 'scale-value dtype (expr :div dtype (lit 1.0 dtype) (expr :sqrt dtype 'components-fp)))
       edge-loop
       (body/->IfRegion
        'active
        [(compute 'denominator-zero :predicate (expr :eq :predicate 'final-denominator zero))
         (compute 'normalized-value dtype
                  (expr :div dtype 'final-numerator (expr :+ dtype 'final-denominator (lit epsilon dtype))))
         (compute 'nonempty-result dtype (select 'denominator-zero zero 'normalized-value dtype))
         (compute 'result dtype (select 'final-valid 'nonempty-result nan dtype))
         (body/->ScalarStore out ['destination 'feature] 'result nil) (yield)]
        [(yield)] [])
       (compute 'tail-owner :predicate
                (conjunction (expr :eq :predicate 'head (lit 0 :long))
                             (expr :eq :predicate 'tile (lit 0 :long))))
       (body/->IfRegion
        'tail-owner
        [(compute 'tail-start :long
                  (integer-op :+ (integer-op :* 'n_heads 'n_components) 'lane-long))
         (compute 'tail-value dtype (select 'shape-valid zero nan dtype))
         (body/->ForLoop (body/value 'tail :long) (body/index-cast 'tail-start :long :exact)
                         (body/index-cast 'total_dim :long :exact) width []
                         [(body/->ScalarStore out ['destination 'tail] 'tail-value nil) (yield)] [] {})
         (yield)] [(yield)] [])]
      :schedule {:strategy :indexed-segmented-reduction-subgroup-score-reuse
                 :subgroup-size width :membership-traversal :ordered-edge-list
                 :score-reuse :component-tile}
      :launch (launch/spec {:workgroup-size [width 1 1]
                            :group-count [(launch/ceil-div 'n_components width)
                                          (launch/runtime-value 'n_heads)
                                          (launch/runtime-value 'n_entities)]})
      :provenance {:dialect :kernel-body :semantic-op :segmented-weighted-reduction
                   :algebra-plan-id (:id plan) :lowering :indexed-score-reuse-kernel-body}
      :attributes {:dynamic-shape? true :membership-kind :edge-list-by-destination}})))

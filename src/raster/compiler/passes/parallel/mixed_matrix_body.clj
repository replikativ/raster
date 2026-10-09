(ns raster.compiler.passes.parallel.mixed-matrix-body
  "Target-neutral matrix stage body and certificate construction.
   Target discovery, instruction spelling and artifact emission remain in the backend."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.layout-stage :as layout-stage]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.layout-transform-schedule :as layout-schedule]
            [raster.compiler.passes.parallel.mixed-matrix-schedule :as mixed-schedule]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]))

(defn ^:no-doc matrix-dimension-parameters
  [m n k reserved]
  (if (and (every? #(or (symbol? %) (keyword? %)) [m n k])
           (= 3 (count (set [m n k])))
           (empty? (set/intersection (set [m n k]) (set reserved))))
    [m n k]
    (contraction-schedule/allocate-dimension-parameters reserved)))

(defn ^:no-doc scheduled-matrix-body
  "Build one canonical f16 matrix KernelBody without selecting a target spelling."
  [{:keys [kernel-name id a b c m n k dimension-parameters axis-symbols tile result-dtype provenance
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes epilogue input-value-regions input-layouts scalar-math]
    :or {result-dtype :float provenance {}}}]
  (let [dimension-parameters
        (or dimension-parameters
            (matrix-dimension-parameters
             m n k
             (concat [a b c]
                     (map :id additional-parameters)
                     (map :sym (:operands epilogue))
                     (map :sym (:scalars epilogue)))))]
    (contraction-schedule/matrix-body
     {:id (or id [:gemm kernel-name])
      :row a :col b :out c
      :dimensions [m n k]
      :dimension-parameters dimension-parameters
      :axis-symbols (or axis-symbols ['i 'j 'l])
      :tile tile
      :bindings {:row a :col b}
      :epilogue epilogue
      :scalar-math scalar-math
      :input-value-regions (or input-value-regions {})
      :input-layouts (or input-layouts {})
      :result-dtype result-dtype
      :additional-parameters additional-parameters
      :additional-indices additional-indices
      :buffer-shapes buffer-shapes
      :buffer-views buffer-views
      :operation-buffers operation-buffers
      :k-range k-range
      :launch-group-count launch-group-count
      :attributes attributes
      :provenance (merge {:dialect :gemm :lowering :scheduled-matrix} provenance)})))

(defn ^:no-doc split-k-matrix-spec
  [{:keys [kernel-name id a b c m n k kc splits axis-symbols tile provenance input-value-regions]}]
  (let [[M N K :as dimension-parameters]
        (matrix-dimension-parameters m n k [a b c kc splits])
        body-M (if (number? m) m M)
        body-N (if (number? n) n N)
        body-K (if (number? k) k K)
        z 'k-slice
        c-view 'split-result-view
        k-lower (kbody/expression :mul z kc)
        k-upper (kbody/expression :min (kbody/expression :add k-lower kc) body-K)]
    {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
     :dimension-parameters dimension-parameters
     :axis-symbols axis-symbols
     :tile tile :result-dtype :float :provenance provenance :input-value-regions input-value-regions
     :additional-parameters [(kbody/->KernelParameter kc :scalar :int [] nil nil :schedule)
                             (kbody/->KernelParameter splits :scalar :int [] nil nil :schedule)]
     :additional-indices [(kbody/->IndexBinding z :group 2)]
     :buffer-shapes {c [splits body-M body-N]}
     :buffer-views [{:id c-view :buffer c
                     :element-offset (kbody/leading-slice-offset z [body-M body-N])
                     :shape [body-M body-N]}]
     :operation-buffers {c c-view}
     :k-range [k-lower k-upper]
     :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value body-N) (:block-n tile))
                          (klaunch/ceil-div (klaunch/runtime-value body-M) (:block-m tile))
                          (klaunch/runtime-value splits)]
     :attributes {:grid-z {:index z :extent splits :purpose :reduction-partition}}}))

(defn ^:no-doc batched-matrix-spec
  [{:keys [kernel-name id a b c m n k batch axis-symbols tile provenance batching epilogue
           input-value-regions input-layouts]
    :or {batching {:row true :col true}}}]
  (let [z 'slab
        ;; MatrixBody parameters are SSA identities, not semantic expressions.  Keep M/N/K
        ;; distinct even when two runtime dimensions are the same compiler value (square
        ;; attention scores are the common case); graph binding maps these identities back to
        ;; m/n/k.  Buffer views must reference this body-local scope, not the outer aliases.
        [M N K :as dimension-parameters]
        (contraction-schedule/allocate-dimension-parameters [a b c batch z])
        a-view 'batch-lhs-view
        b-view 'batch-rhs-view
        c-view 'batch-result-view
        row-batched? (get batching :row true)
        col-batched? (get batching :col true)
        row-slice-layout (get input-layouts a)
        col-slice-layout (get input-layouts b)
        col-transposed? (= [1 0] (:perm col-slice-layout))
        a-shape (if row-batched? [batch M K] [M K])
        b-shape (if col-batched?
                  (if col-transposed? [batch N K] [batch K N])
                  ;; The shared parameter retains its logical K/N shape; its permutation
                  ;; already describes the physical N/K storage. Batched parents instead use
                  ;; physical slice shape because the logical permutation lives on the view.
                  [K N])
        buffer-views
        (cond-> [{:id c-view :buffer c
                  :element-offset (kbody/leading-slice-offset z [M N]) :shape [M N]}]
          row-batched?
          (conj {:id a-view :buffer a
                 :element-offset (kbody/leading-slice-offset z [M K]) :shape [M K]
                 :layout (some-> row-slice-layout (assoc :shape [M K]))})
          col-batched?
          (conj {:id b-view :buffer b
                 :element-offset (kbody/leading-slice-offset z [K N]) :shape [K N]
                 :layout (some-> col-slice-layout (assoc :shape [K N]))}))
        operation-buffers
        (cond-> {c c-view}
          row-batched? (assoc a a-view)
          col-batched? (assoc b b-view))]
    {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
     :dimension-parameters dimension-parameters
     :axis-symbols axis-symbols
     :tile tile :result-dtype :float :provenance provenance
     :epilogue epilogue :input-value-regions input-value-regions
     ;; A batched operand's rank-2 permutation belongs to its selected slice. Shared operands
     ;; remain rank 2 and keep the layout on the parent parameter.
     :input-layouts (cond-> input-layouts
                      row-batched? (dissoc a)
                      col-batched? (dissoc b))
     :additional-parameters [(kbody/->KernelParameter batch :scalar :int [] nil nil :schedule)]
     :additional-indices [(kbody/->IndexBinding z :group 2)]
     :buffer-shapes {a a-shape b b-shape c [batch M N]}
     :buffer-views buffer-views
     :operation-buffers operation-buffers
     :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value N) (:block-n tile))
                          (klaunch/ceil-div (klaunch/runtime-value M) (:block-m tile))
                          (klaunch/runtime-value batch)]
     :attributes {:grid-z {:index z :extent batch :purpose :independent-slices}
                  :batching batching}}))

(defn schedule-matrix
  "Construct the target-neutral scheduled certificate for one closed matrix stage specification."
  [{:keys [argument-values source-operation phase scalar-types]
    :or {argument-values {}}
    :as spec}]
  (let [kernel-body (scheduled-matrix-body spec)
        dimension-values (get-in kernel-body [:attributes :dimension-values])
        arguments (mapv (fn [{:keys [id role]}]
                          (cond
                            (contains? argument-values id) (get argument-values id)
                            (= :dimension role) (get dimension-values id)
                            :else id))
                        (:parameters kernel-body))
        uses (scheduled-body/derive-uses kernel-body arguments)
        scheduled
        (scheduled-body/make
         {:source (or source-operation
                      (throw (ex-info "matrix artifact requires its exact scheduled stage"
                                      {:reason :matrix-stage-source :id (:id spec)
                                       :phase phase})))
          :body kernel-body
          :arguments arguments
          :scalar-bindings (scheduled-body/derive-scalar-bindings kernel-body arguments scalar-types)
          :effects {:kind :tensor-contraction-stage :uses uses}
          :legality {:kind :matrix-instruction-tiling
                     :scheduled-body (:id kernel-body)}
          :numerics (cond-> {:mode :reassociated :policy :tiled-contraction
                             :rounding :nearest-even :accumulator-dtype :float}
                      (contains? spec :scalar-math)
                      (assoc :scalar-math (numerics/validate-scalar-math-policy! (:scalar-math spec)))
                      (seq (:epilogue source-operation))
                      (assoc :result-transform
                             {:kind :typed-scalar-region
                              :policy :same-typed-ssa-evaluation-order
                              :input-dtype :float
                              :result-dtype (:result-dtype source-operation)}))
          :provenance {:semantic-op :contraction :lowering :gemm-graph :phase phase}
          :attributes (cond-> {:strategy phase
                               ;; Temporary compatibility projection; the body schedule is the
                               ;; authority and target/device tests use this flattened view.
                               :tile (:schedule kernel-body)
                               :accumulator-dtype :float}
                        (get-in kernel-body [:attributes :batching])
                        (assoc :batched? true
                               :batching (get-in kernel-body [:attributes :batching])))} )]
    scheduled))

(defn matrix-stage-spec
  "Project one validated matrix stage into its closed target-neutral body specification."
  ([stage phase scalar-types]
   (matrix-stage-spec stage phase scalar-types {}))
  ([stage phase scalar-types caller-options]
  (let [{stage-id :id a :lhs b :rhs c :result
         [m n k] :dimensions axis-symbols :axis-symbols reduction :reduction epilogue :epilogue
         batching :batching schedule :schedule input-value-regions :input-value-regions
         input-layouts :input-layouts}
        (matrix-stage/validate! stage)
        tile (:tile schedule)
        _ (when-not (and (= :matrix-instruction-tiling (:kind schedule))
                         (= :half (:operand-dtype stage))
                         (= :float (:accumulator-dtype stage))
                         (= :float (:result-dtype stage)))
            (throw (ex-info "GEMM matrix emitter does not implement the scheduled numerical form"
                            {:reason :gemm-stage-emission-unsupported :stage stage-id
                             :schedule schedule
                             :operand-dtype (:operand-dtype stage)
                             :accumulator-dtype (:accumulator-dtype stage)
                             :result-dtype (:result-dtype stage)})))
        _ (when-not (map? tile)
            (throw (ex-info "scheduled matrix stage does not close its emission choices"
                            {:reason :gemm-stage-emission-open :stage stage-id
                             :missing :tile})))
        split-k? (= :split-k (:kind reduction))
        kc (:chunk reduction)
        splits (:partitions reduction)
        emit-args {:id stage-id
                   :a a :b b :c c :m m :n n :k k
                   :axis-symbols axis-symbols
                   :tile tile :result-dtype (:result-dtype stage)
                   :epilogue epilogue
                   :input-value-regions input-value-regions
                   :input-layouts input-layouts
                   :phase phase
                   :source-operation stage
                   :provenance {:operation-id stage-id :phase phase}}]
    (merge (select-keys caller-options [:scalar-math])
     (assoc (cond
       batching
       (assoc (batched-matrix-spec
               (assoc emit-args
                      :batch (:extent batching)
                      :batching {:row (:lhs batching) :col (:rhs batching)}))
              :phase phase :source-operation stage)

       split-k?
       (assoc (split-k-matrix-spec
               (assoc emit-args :kc :k-chunk :splits :splits))
              :phase phase :source-operation stage
              :argument-values {:k-chunk kc :splits splits})

       :else emit-args)
           :scalar-types scalar-types)))))

(defn schedule-matrix-stage
  "Rebuild the complete scheduled certificate without target discovery or emission."
  ([stage scalar-types]
   (schedule-matrix-stage stage scalar-types {}))
  ([stage scalar-types caller-options]
   (schedule-matrix (matrix-stage-spec stage :matrix-contract scalar-types caller-options))))

(defn ^:no-doc make-schedule
  [{:keys [source body arguments effects legality numerics phase
           provenance attributes scalar-types]
    :or {effects {:kind :tensor-contraction-stage}
         provenance {} attributes {}}}]
  (let [uses (scheduled-body/derive-uses body arguments)
        scheduled
        (scheduled-body/make
         {:source source
          :body body
          :arguments arguments
          :scalar-bindings (scheduled-body/derive-scalar-bindings body arguments scalar-types)
          :effects (assoc effects :uses uses)
          :legality legality
          :numerics (cond-> numerics
                      (:source-arithmetic source)
                      (assoc :source-arithmetic (:source-arithmetic source)))
          :provenance (merge {:semantic-op :contraction
                              :lowering :gemm-graph :phase phase}
                             provenance)
          :attributes (merge {:strategy phase} attributes)})]
    scheduled))

(defn schedule-cast
  "Project the admitted dense FP32-to-FP16 layout stage, without target spelling."
  [stage phase scalar-types]
  (let [{stage-id :id in :input out :output input-shape :input-shape policy :policy}
        (layout-stage/validate! stage)
        elements (first input-shape)
        vector-width (:vector-width policy)
        _ (when-not (and (= :float (:input-dtype stage)) (= :half (:output-dtype stage))
                         (= :nearest-even (:rounding policy)) (= :ieee (:overflow policy)))
            (throw (ex-info "GEMM cast emitter does not implement the scheduled representation"
                            {:reason :gemm-stage-emission-unsupported :stage stage-id
                             :input-dtype (:input-dtype stage)
                             :output-dtype (:output-dtype stage) :policy policy})))
        _ (when-not (and (integer? vector-width) (pos? vector-width))
            (throw (ex-info "scheduled layout cast does not close its emission choices"
                            {:reason :gemm-stage-emission-open :stage stage-id
                             :missing :vector-width})))
        kernel-body
        (layout-schedule/cast-body
         {:id stage-id :input in :output out
          :source-dtype :float :destination-dtype :half :vector-width vector-width
          :extent-dtype (klaunch/typed-expression-dtype elements scalar-types)
          :rounding :nearest-even :overflow :ieee})]
    (make-schedule
     {:source stage
      :body kernel-body :arguments [in out elements]
      :scalar-types scalar-types
      :effects {:kind :layout-transform-stage}
      :legality {:kind :dense-affine-cast :vector-width vector-width}
      :numerics {:mode :bounded-error :policy :f32-to-f16-storage
                 :rounding :nearest-even :accumulator-dtype :half
                 :error-model {:kind :ieee-f16-conversion :overflow :ieee}}
      :phase phase
      :attributes {:vector-width vector-width :from :float :to :half
                   :rounding :nearest-even :overflow :ieee
                   :cacheable-transform? true}})))

(defn schedule-transpose
  "Project the admitted bit-preserving rank-two half transpose."
  [stage phase scalar-types]
  (let [{stage-id :id in :input out :output
         [rows cols] :input-shape} (layout-stage/validate! stage)
        _ (when-not (and (= :half (:input-dtype stage)) (= :half (:output-dtype stage))
                         (= [1 0] (get-in stage [:policy :permutation])))
            (throw (ex-info "GEMM transpose emitter does not implement the scheduled representation"
                            {:reason :gemm-stage-emission-unsupported :stage stage-id
                             :input-dtype (:input-dtype stage)
                             :output-dtype (:output-dtype stage) :policy (:policy stage)})))
        kernel-body
        (layout-schedule/transpose-body
         {:id stage-id :input in :output out :element-dtype :half
          :row-extent-dtype (klaunch/typed-expression-dtype rows scalar-types)
          :column-extent-dtype (klaunch/typed-expression-dtype cols scalar-types)})]
    (make-schedule
     {:source stage
      :body kernel-body :arguments [in out rows cols]
      :scalar-types scalar-types
      :effects {:kind :layout-transform-stage}
      :legality {:kind :bijective-affine-permutation :permutation [1 0]}
      :numerics {:mode :exact :policy :bit-preserving-permutation}
      :phase phase
      :attributes {:layout :transpose :dtype :half :cacheable-transform? true}})))

(defn schedule-combine
  "Project the canonical generated split-K combine stage; not an arbitrary reduction lowering."
  [operation partials c mn splits scalar-types]
  (let [{body :body} (mixed-schedule/split-k-combine-plan
                    (:id operation)
                    {'mn (klaunch/typed-expression-dtype mn scalar-types)
                     'splits (klaunch/typed-expression-dtype splits scalar-types)})]
    (make-schedule
     {:source operation :body body
      :arguments [partials c mn splits mn]
      :scalar-types scalar-types
      :effects {:kind :tensor-contraction-stage}
      :legality {:kind :portable-contraction :purpose :split-k-combine}
      :numerics {:mode :reassociated :policy :sequential-segment-fold
                 :rounding :nearest-even :accumulator-dtype :float}
      :phase :split-k-combine
      :attributes {:accumulator-dtype :float :semantic-op :contraction}})))

(defn ^:no-doc split-combine-values
  [operation]
  (let [operation (if (segop/seg-red? operation)
                    operation
                    (throw (ex-info "split combine emission requires a SegRed stage"
                                    {:reason :gemm-stage-lowering :operation operation})))
        partials (segop/operation-inputs operation)
        outputs (segop/operation-outputs operation)
        dimensions (get-in operation [:space :dims])
        mn (get-in dimensions [0 :bound])
        splits (get-in dimensions [1 :bound])]
    (when-not (and (= 1 (count partials)) (= 1 (count outputs))
                   (some? mn) (some? splits))
      (throw (ex-info "split combine stage does not close its storage and reduction geometry"
                      {:reason :gemm-stage-emission-open :stage (:id operation)
                       :inputs partials :outputs outputs :mn mn :splits splits})))
    {:partials (first partials) :output (first outputs) :mn mn :splits splits}))

(defn schedule-for-node
  "Rebuild one reference certificate from a closed mixed-matrix stage graph.

   This does not certify algorithm equivalence or whole-graph writes. A certification caller
   must first compare the complete graph with independent typed reconstruction; generic stage
   validation alone does not establish canonical reduction geometry."
  ([node graph]
   (schedule-for-node node graph {}))
  ([node graph caller-options]
  (let [graph (kgraph/validate! graph)
        _ (when-not (some #{node} (:nodes graph))
            (throw (ex-info "mixed matrix reference node is not in its stage graph"
                            {:reason :mixed-matrix-body-node :node (:id node)})))
        operation (:operation node)
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars graph))
        phase (last (:id node))
        scheduled (cond
                    (matrix-stage/matrix-stage? operation)
                    (schedule-matrix-stage operation scalar-types caller-options)

                    (layout-stage/layout-stage? operation)
                    (case (:operation operation)
                      :cast (schedule-cast operation phase scalar-types)
                      :transpose (schedule-transpose operation phase scalar-types))

                    (segop/seg-red? operation)
                    (let [{:keys [partials output mn splits]} (split-combine-values operation)]
                      (schedule-combine operation partials output mn splits scalar-types))

                    :else (throw (ex-info "mixed matrix stage has no reference body projection"
                                         {:reason :mixed-matrix-body-operation
                                          :node (:id node) :operation operation})))]
    (scheduled-body/validate-against-node! scheduled node graph)
    scheduled)))

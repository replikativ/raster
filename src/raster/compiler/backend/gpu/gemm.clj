(ns raster.compiler.backend.gpu.gemm
  "Compiler-owned executable schedules for dense GEMM.

   The public call is uniformly `(A B C M N K)` over f32 resident buffers. A schedule may be one
   scalar kernel or a graph containing conversion, layout conversion, matrix contraction, and
   split-K combination. All mixed-precision scratch and derived scheduling scalars are private to
   the graph; callers never bind them and runtimes never reconstruct the algorithm from `:gemm`."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.kernel-body-target :as kernel-body-target]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.backend.gpu.layout-transform :as layout-emitter]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.intel-block-io :as block-io]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.layout-stage :as layout-stage]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.scheduled-graph-refinement :as graph-refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.mixed-matrix-schedule :as mixed-schedule]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]))

(def ^:private default-min-split-chunk 1024)
(def ^:private default-max-splits 64)

(defn- identifier
  [value]
  (let [text (str/replace (str value) #"[^A-Za-z0-9_]" "_")]
    (if (re-find #"^[A-Za-z_]" text) text (str "k_" text))))

(defn- emitted-node
  [id artifact uses dependencies]
  (let [scheduled (kart/attribute artifact :scheduled-kernel-body)]
    (when-not (scheduled-body/scheduled-kernel-body? scheduled)
      (throw (ex-info "production GEMM graph node requires a scheduled-body certificate"
                      {:reason :gemm-scheduled-body :node id})))
    (mixed-schedule/stage-node id artifact uses (mapv :value (:scalar-bindings scheduled)) dependencies)))

(defn- emit-scheduled-body-artifact
  [{:keys [kernel-name source body arguments effects legality numerics phase target-dialect
           parameter-names provenance attributes scalar-types]
    :or {target-dialect :opencl-intel effects {:kind :tensor-contraction-stage}
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
          :numerics numerics
          :provenance (merge {:semantic-op :contraction
                              :lowering :gemm-graph :phase phase}
                             provenance)
          :attributes (merge {:strategy phase} attributes)})]
    (kernel-body-target/emit-artifact
     (c-emit/c-symbol kernel-name) scheduled target-dialect
     {:parameter-names parameter-names})))

(defn- scalar-contraction-facts
  [variant]
  (let [[a-index b-index]
        (case variant
          :nn ['(+ (* i k) l) '(+ (* l n) j)]
          :nt ['(+ (* i k) l) '(+ (* j k) l)]
          :tn ['(+ (* l m) i) '(+ (* l n) j)]
          :tt ['(+ (* l m) i) '(+ (* j k) l)])]
    (contraction-facts/from-components
     {:out 'C :free-axes '[[i m] [j n]] :contract-axes '[[l k]] :dtype :float
      :body (list '* (list 'aget 'A a-index) (list 'aget 'B b-index))})))

(defn- portable-scalar-matrix-plan
  [variant stage-id]
  (let [facts (scalar-contraction-facts variant)
        operation (contract-lower/contraction-facts->segred facts :id stage-id)
        planned (contraction-schedule/plan-portable-body
                 facts operation {}
                 {:array-types {'A :float 'B :float 'C :float}
                  :scalar-types {'m :int 'n :int 'k :int}})]
    (when-not (:ok planned)
      (throw (ex-info "matrix product did not admit the portable contraction schedule"
                      {:reason :raster/bug :variant variant :plan planned})))
    {:operation operation :body (:body planned) :plan planned}))

(defn emit-portable-scalar-matrix-kernel
  "Lower a dynamic f32 NN/NT/TN/TT matrix product through the portable contraction schedule."
  ([kernel-name variant]
   (emit-portable-scalar-matrix-kernel kernel-name variant :opencl-intel))
  ([kernel-name variant target-dialect]
   (let [kernel-body (:body (portable-scalar-matrix-plan
                             variant [:direct-gemm kernel-name]))]
     {:source
      (kernel-body-opencl/emit-scalar-kernel
       kernel-name kernel-body
       {:target-dialect target-dialect
        :parameter-names {'A "A" 'B "B" 'C "C" 'k "k" 'm "m" 'n "n"
                          '_nseg "_nseg"}})
      :kernel-body kernel-body
      :workgroup-size 256})))

(defn- scalar-graph
  [{:keys [id a b c m n k variant] :as spec}]
  (let [{:keys [abi arguments]} (mixed-schedule/public-outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (mixed-schedule/extents spec)
        prefix (identifier (str id "_scalar"))
        kernel-name (str prefix "_gemm")
        stage-id [:gemm id :f32-scalar :contract]
        {:keys [operation body]} (portable-scalar-matrix-plan variant stage-id)
        gemm (emit-scheduled-body-artifact
              {:kernel-name kernel-name :source operation :body body
               :arguments [a b c k m n c-elements]
               :effects {:kind :tensor-contraction-stage}
               :legality {:kind :portable-contraction :variant variant}
               :numerics {:mode :reassociated :policy :sequential-segment-fold
                          :rounding :nearest-even :accumulator-dtype :float}
               :phase :scalar-gemm
               :attributes {:variant variant :semantic-op :contraction}
               :parameter-names {'A "A" 'B "B" 'C "C" 'k "k" 'm "m" 'n "n"
                                 '_nseg "_nseg"}})]
    (kgraph/make
     {:inputs [(mixed-schedule/graph-buffer a :float a-elements :input)
               (mixed-schedule/graph-buffer b :float b-elements :input)]
      :outputs [(mixed-schedule/graph-buffer c :float c-elements :output)]
      :scalars (kgraph/interface-scalars abi arguments)
      :nodes [(emitted-node stage-id gemm
                            [(mixed-schedule/value-use a :read) (mixed-schedule/value-use b :read)
                             (mixed-schedule/value-use c :write)] [])]
      :abi abi :arguments arguments
      :effects (mixed-schedule/effects spec)
      :provenance {:semantic-op :contraction :variant variant :lowering :scalar-gemm}
      :attributes {:strategy :f32-scalar :variant variant :precision :f32}})))

(defn- convert-artifact
  [kernel-name stage phase target-dialect scalar-types]
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
        kernel-name (c-emit/c-symbol kernel-name)
        kernel-body
        (layout-emitter/cast-body
         {:id stage-id :input in :output out
          :source-dtype :float :destination-dtype :half :vector-width vector-width
          :extent-dtype (klaunch/typed-expression-dtype elements scalar-types)
          :rounding :nearest-even :overflow :ieee})]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name
      :source stage
      :body kernel-body :arguments [in out elements]
      :scalar-types scalar-types
      :effects {:kind :layout-transform-stage}
      :legality {:kind :dense-affine-cast :vector-width vector-width}
      :numerics {:mode :bounded-error :policy :f32-to-f16-storage
                 :rounding :nearest-even :accumulator-dtype :half
                 :error-model {:kind :ieee-f16-conversion :overflow :ieee}}
      :phase phase
      :target-dialect target-dialect
      :attributes {:vector-width vector-width :from :float :to :half
                   :rounding :nearest-even :overflow :ieee
                   :cacheable-transform? true}
      :parameter-names {in "input" out "output" :layout-elements "n"}})))

(defn- transpose-artifact
  [kernel-name stage phase target-dialect scalar-types]
  (let [{stage-id :id in :input out :output
         [rows cols] :input-shape} (layout-stage/validate! stage)
        _ (when-not (and (= :half (:input-dtype stage)) (= :half (:output-dtype stage))
                         (= [1 0] (get-in stage [:policy :permutation])))
            (throw (ex-info "GEMM transpose emitter does not implement the scheduled representation"
                            {:reason :gemm-stage-emission-unsupported :stage stage-id
                             :input-dtype (:input-dtype stage)
                             :output-dtype (:output-dtype stage) :policy (:policy stage)})))
        kernel-name (c-emit/c-symbol kernel-name)
        kernel-body
        (layout-emitter/transpose-body
         {:id stage-id :input in :output out :element-dtype :half
          :row-extent-dtype (klaunch/typed-expression-dtype rows scalar-types)
          :column-extent-dtype (klaunch/typed-expression-dtype cols scalar-types)})]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name
      :source stage
      :body kernel-body :arguments [in out rows cols]
      :scalar-types scalar-types
      :effects {:kind :layout-transform-stage}
      :legality {:kind :bijective-affine-permutation :permutation [1 0]}
      :numerics {:mode :exact :policy :bit-preserving-permutation}
      :phase phase
      :target-dialect target-dialect
      :attributes {:layout :transpose :dtype :half :cacheable-transform? true}
      :parameter-names {in "input" out "output"
                        :layout-rows "rows" :layout-cols "cols"}})))

(defn- matrix-dimension-parameters
  [m n k reserved]
  (if (and (every? #(or (symbol? %) (keyword? %)) [m n k])
           (= 3 (count (set [m n k])))
           (empty? (set/intersection (set [m n k]) (set reserved))))
    [m n k]
    (contraction-schedule/allocate-dimension-parameters reserved)))

(defn- scheduled-matrix-body
  "Build one canonical f16 matrix KernelBody without selecting a target spelling."
  [{:keys [kernel-name id a b c m n k dimension-parameters axis-symbols tile result-dtype provenance
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes epilogue input-value-regions input-layouts]
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

(defn emit-scheduled-matrix-kernel
  "Build and directly lower one canonical f16 matrix contraction.

  This is the shared compiler entry for graph-owned, legacy-plan, and resident direct/tiled GEMM
  front doors. Caller identities remain the KernelBody ABI identities; OpenCL parameter spelling
  is solely a target concern. Optional views, hardware indices, and K bounds are explicit schedule
  values used by the split-K and batched wrappers below. An optional epilogue becomes a typed
  ScalarSSARegion on every store and is lowered as part of the body."
  [{:keys [kernel-name id a b c m n k dimension-parameters axis-symbols tile result-dtype provenance
           target-dialect
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes parameter-names epilogue input-value-regions
           input-layouts]
    :or {result-dtype :float provenance {} target-dialect :opencl-intel}}]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        kernel-body (scheduled-matrix-body
                     {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
                      :dimension-parameters dimension-parameters :axis-symbols axis-symbols :tile tile
                      :result-dtype result-dtype :provenance provenance
                      :additional-parameters additional-parameters
                      :additional-indices additional-indices :buffer-shapes buffer-shapes
                      :buffer-views buffer-views :operation-buffers operation-buffers
                      :k-range k-range :launch-group-count launch-group-count
                      :attributes attributes :epilogue epilogue
                      :input-value-regions input-value-regions :input-layouts input-layouts})
        emitted (matrix-target/emit-matrix-kernel
                 kernel-name kernel-body target-dialect {:parameter-names parameter-names})]
    (assoc emitted
           :kernel-name kernel-name
           :workgroup-size (get-in kernel-body [:launch :workgroup-size]))))

(defn- split-k-matrix-spec
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
     :attributes {:grid-z {:index z :extent splits :purpose :reduction-partition}}
     :parameter-names {kc "KC" splits "splits"}}))

(defn emit-scheduled-split-k-kernel
  "Lower a grid-Z partition of the K reduction into disjoint f32 output views."
  [spec]
  (emit-scheduled-matrix-kernel (split-k-matrix-spec spec)))

(defn- batched-matrix-spec
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
                  (if col-transposed? [N K] [K N]))
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
                  :batching batching}
     :parameter-names {batch "batch"}}))

(defn emit-scheduled-batched-matrix-kernel
  "Lower independent dense matrix slabs as grid-Z-selected contiguous buffer views.

   `batching` states whether each operand carries the leading batch axis.  A false entry denotes a
   stable broadcast operand (most commonly shared model weights), so its view has zero batch
   offset instead of materializing a repeated tensor."
  [spec]
  (emit-scheduled-matrix-kernel (batched-matrix-spec spec)))

(defn- emit-scheduled-matrix-artifact
  [{:keys [kernel-name target-dialect parameter-names argument-values source-operation phase scalar-types]
    :or {target-dialect :opencl-intel argument-values {}}
    :as spec}]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        kernel-body (scheduled-matrix-body spec)
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
    (kernel-body-target/emit-artifact
     kernel-name scheduled target-dialect {:parameter-names parameter-names})))

(defn- gemm-artifact
  [stage kernel-name phase target-dialect scalar-types]
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
        emit-args {:kernel-name kernel-name
                   :id stage-id
                   :a a :b b :c c :m m :n n :k k
                   :axis-symbols axis-symbols
                   :tile tile :result-dtype (:result-dtype stage)
                   :epilogue epilogue
                   :input-value-regions input-value-regions
                   :input-layouts input-layouts
                   :phase phase
                   :target-dialect target-dialect
                   :source-operation stage
                   :provenance {:operation-id stage-id :phase phase}}]
    (emit-scheduled-matrix-artifact
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
            :scalar-types scalar-types :target-dialect target-dialect))))

(defn emit-split-k-combine-kernel
  "Lower C[i] = sum_s partials[s, i] through the generic portable contraction schedule."
  ([kernel-name] (emit-split-k-combine-kernel kernel-name :opencl-intel))
  ([kernel-name target-dialect]
   (let [kernel-name (c-emit/c-symbol kernel-name)
         kernel-body (:body (mixed-schedule/split-k-combine-plan [:direct-split-k-combine kernel-name]))
         source (kernel-body-opencl/emit-scalar-kernel
                 kernel-name kernel-body
                {:target-dialect target-dialect
                 :parameter-names {'partials "partials" 'C "C"
                                   'mn "mn" 'splits "splits" '_nseg "_nseg"}})]
     {:kernel-name kernel-name :source source :kernel-body kernel-body :workgroup-size 256})))

(defn- combine-artifact
  [kernel-name operation partials c mn splits target-dialect scalar-types]
  (let [{body :body} (mixed-schedule/split-k-combine-plan
                    (:id operation)
                    {'mn (klaunch/typed-expression-dtype mn scalar-types)
                     'splits (klaunch/typed-expression-dtype splits scalar-types)})]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name :source operation :body body
      :arguments [partials c mn splits mn]
      :scalar-types scalar-types
      :effects {:kind :tensor-contraction-stage}
      :legality {:kind :portable-contraction :purpose :split-k-combine}
      :numerics {:mode :reassociated :policy :sequential-segment-fold
                 :rounding :nearest-even :accumulator-dtype :float}
      :phase :split-k-combine
      :target-dialect target-dialect
      :attributes {:accumulator-dtype :float :semantic-op :contraction}
      :parameter-names {'partials "partials" 'C "C"
                        'mn "mn" 'splits "splits" '_nseg "_nseg"}})))

(defn split-factor-strategy
  "Stable strategy identity for one explicit split-K candidate."
  [factor]
  (when-not (and (integer? factor) (> (long factor) 1))
    (throw (ex-info "split factor must be an integer greater than one"
                    {:split-factor factor})))
  (keyword (str "xmx-split-k-" factor)))

(defn- split-combine-values
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

(defn- emit-stage-artifact
  [target-dialect prefix scalar-types {:keys [operation] :as node}]
  (let [phase (last (:id node))]
    (cond
      (layout-stage/layout-stage? operation)
      (case (:operation operation)
        :cast (convert-artifact (str prefix "_" (name phase)) operation
                                phase target-dialect scalar-types)
        :transpose (transpose-artifact (str prefix "_" (name phase)) operation
                                       phase target-dialect scalar-types))

      (matrix-stage/matrix-stage? operation)
      (gemm-artifact operation (str prefix "_" (name phase))
                     :matrix-contract target-dialect scalar-types)

      (segop/seg-red? operation)
      (let [{:keys [partials output mn splits]} (split-combine-values operation)]
        (combine-artifact (str prefix "_" (name phase)) operation
                          partials output mn splits target-dialect scalar-types))

      :else
      (throw (ex-info "GEMM stage has no ScheduledKernelBody lowering"
                      {:reason :gemm-stage-lowering :node (:id node)
                       :operation operation})))))

(defn emit-scheduled-stage-graph
  "Emit an already scheduled mixed-precision GEMM graph.

   Every physical choice is recovered from its validated stage operations. Only target spelling
   and entry-point naming remain emission inputs. When supplied, `refinement` must retain this
   exact graph rather than a boundary-compatible reconstruction."
  [stage-graph {:keys [target-dialect prefix refinement]
                :or {target-dialect :opencl-intel prefix "scheduled_gemm"}}]
  (let [stage-graph (kgraph/validate! stage-graph)
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars stage-graph))
        _ (when (and refinement
                     (not= stage-graph (graph-refinement/scheduled-graph refinement)))
            (throw (ex-info "GEMM emission refinement does not retain the exact scheduled graph"
                            {:reason :gemm-emission-refinement})))
        emitted
        (kgraph/map-operations
         stage-graph
         (fn [node]
           (let [artifact (emit-stage-artifact target-dialect prefix scalar-types node)
                 scheduled (kart/attribute artifact :scheduled-kernel-body)]
             (scheduled-body/validate-against-node! scheduled node stage-graph)
             (scheduled-body/validate-artifact-projection! scheduled artifact)
             artifact)))
        emitted (cond-> emitted
                  refinement
                  (assoc-in [:attributes :scheduled-graph-refinement] refinement))]
    (when-not (kgraph/dataflow-equivalent? stage-graph emitted)
      (throw (ex-info "GEMM target emission changed scheduled graph dataflow"
                      {:reason :gemm-emission-dataflow
                       :scheduled (kgraph/dataflow-contract stage-graph)
                       :emitted (kgraph/dataflow-contract emitted)})))
    (kexec/validate! emitted)))

(defn- xmx-graph
  [spec]
  (when-let [{:keys [graph refinement]} (mixed-schedule/plan spec)]
    (emit-scheduled-stage-graph
     graph {:target-dialect (get spec :target-dialect :opencl-intel)
            :prefix (identifier (str (:id spec) "_" (name (get-in graph [:attributes :strategy]))))
            :refinement refinement})))

(defn- matrix-input-fusion-target? [{:keys [variant target-dialect]}]
  (and (contains? #{:nn :nt} variant)
       (= :opencl-intel (or target-dialect :opencl-intel))))

(defn tile-input-strategy
  "Stable strategy identity for a finite-search tile-local matrix schedule. Every physical tile
   axis that can alter emitted code or launch geometry participates in the identity; the analytic
   default keeps the existing concise `:xmx-direct-tile-inputs` name."
  [{:keys [block-m block-n block-k sg-m sg-n num-stages]}]
  (keyword
   (format "xmx-direct-tile-inputs-bm%d-bn%d-sm%d-sn%d-bk%d-s%d"
           block-m block-n sg-m sg-n block-k (or num-stages 3))))

(defn dynamic-lhs-strategy
  "Stable strategy identity for a finite-search tile whose activation cast is tile-local while
   the RHS representation remains a cacheable materialized graph value."
  [{:keys [block-m block-n block-k sg-m sg-n num-stages]}]
  (keyword
   (format "xmx-direct-dynamic-lhs-bm%d-bn%d-sm%d-sn%d-bk%d-s%d"
           block-m block-n sg-m sg-n block-k (or num-stages 3))))

(defn direct-tile-strategy
  "Stable strategy identity for a finite-search materialized-input matrix schedule. The analytic
   tile retains the concise `:xmx-direct` identity; finite alternatives use this identity so the
   measured selector can choose tile geometry independently of input-fusion policy."
  [{:keys [block-m block-n block-k sg-m sg-n num-stages]}]
  (keyword
   (format "xmx-direct-bm%d-bn%d-sm%d-sn%d-bk%d-s%d"
           block-m block-n sg-m sg-n block-k (or num-stages 3))))

(defn- matrix-input-fusion-alternative [spec]
  (when (matrix-input-fusion-target? spec)
    (xmx-graph (assoc spec :split-k? false :fuse-tile-inputs? true
                     :vector-width (get spec :vector-width 4)
                     :strategy (or (:strategy spec) :xmx-direct-tile-inputs)))))

(defn- matrix-dynamic-lhs-alternative [spec]
  (when (matrix-input-fusion-target? spec)
    (xmx-graph (assoc spec :split-k? false :fuse-lhs-cast? true
                     :vector-width (get spec :vector-width 4)
                     :strategy (or (:strategy spec) :xmx-direct-dynamic-lhs)))))

(defn emit-matrix-dynamic-lhs-alternative
  "Emit an Intel direct-matrix candidate that converts dynamic FP32 activations in tile loads
   while retaining the typed RHS conversion/layout graph for ordinary constant hoisting."
  [{:keys [variant target-dialect] :as spec}]
  (when-not (matrix-input-fusion-target? spec)
    (throw (ex-info "dynamic-LHS matrix fusion requires Intel direct NN/NT storage"
                    {:reason :matrix-dynamic-lhs-target :variant variant :target target-dialect})))
  (or (matrix-dynamic-lhs-alternative spec)
      (throw (ex-info "dynamic-LHS matrix fusion obligations are not satisfied"
                      {:reason :matrix-dynamic-lhs-ineligible :id (:id spec)}))))

(defn emit-matrix-direct-alternative
  "Emit one explicit materialized-input Intel direct-matrix candidate for finite tile search.
   This is the same representation policy as `:xmx-direct`; only checked tile geometry differs."
  [{:keys [variant target-dialect] :as spec}]
  (when-not (matrix-input-fusion-target? spec)
    (throw (ex-info "finite direct-matrix search requires Intel NN/NT storage"
                    {:reason :matrix-direct-tile-target :variant variant
                     :target target-dialect})))
  (or (xmx-graph (assoc spec :split-k? false
                         :vector-width (get spec :vector-width 4)
                         :strategy (or (:strategy spec)
                                       (direct-tile-strategy (:tile spec)))))
      (throw (ex-info "finite direct-matrix schedule obligations are not satisfied"
                      {:reason :matrix-direct-tile-ineligible :id (:id spec)}))))

(defn emit-matrix-input-fusion-alternative
  "Emit an explicit Intel direct-matrix candidate with tile-local FP32→FP16 inputs.
   Binding requires physical A/C disjointness; runtime admission checks the concrete ranges.
   Retains the ordinary public ABI and original semantic
   refinement source. Unsupported layouts/targets fail closed; no implicit fallback."
  [{:keys [variant target-dialect] :as spec}]
  (when-not (matrix-input-fusion-target? spec)
    (throw (ex-info "matrix input fusion requires Intel direct NN/NT storage"
                    {:reason :matrix-input-fusion-target :variant variant :target target-dialect})))
  (or (matrix-input-fusion-alternative spec)
      (throw (ex-info "matrix input cast fusion obligations are not satisfied"
                      {:reason :matrix-input-fusion-ineligible :id (:id spec)}))))

(defn emit-batched-matrix-alternative
  "Emit the target-neutral batched mixed-matrix schedule and its existing checked selector."
  [{:keys [id batch m n k batching]
    :or {batching {:row true :col true}}
    :as spec}]
  (let [{:keys [graph refinement]} (mixed-schedule/plan-batched spec)
        graph (emit-scheduled-stage-graph
               graph {:target-dialect (get spec :target-dialect :opencl-intel)
                      :prefix (identifier (str id "_xmx_batched"))
                      :refinement refinement})]
    {:graph graph
     :refinement refinement
     :selector
     {:kind :runtime-expression-cases
      :cases (block-io/fallback-cases
              (into (graph-call/direct-scalar-range-preconditions graph)
                    (block-io/matrix-preconditions
                     m n k (cond-> []
                             (get batching :row true) (conj (klaunch/product m k))
                             (get batching :col true) (conj (klaunch/product k n)))))
              :f32-scalar)
      :default :xmx-batched}}))

(defn requested-splits
  "Build the generic occupancy expression used by both selection and split-K storage/launches."
  [{:keys [m n k tile fill-workgroups target-fill-multiple min-split-chunk max-splits]
    :or {min-split-chunk default-min-split-chunk
         max-splits default-max-splits
         target-fill-multiple 4}}]
  (let [{:keys [block-m block-n]} tile
        output-workgroups (klaunch/product (klaunch/ceil-div m block-m)
                                           (klaunch/ceil-div n block-n))
        target-workgroups (* target-fill-multiple fill-workgroups)
        ;; Exactly zero once the unsplit output grid fills the machine, one otherwise. This keeps
        ;; the historic "never split a filling GEMM" policy gate inside the same checked arithmetic
        ;; IR without introducing an opaque conditional callback.  Clamp the final request to one:
        ;; every emitted alternative must remain concretely bindable for offline autotuning, and a
        ;; split graph cannot use zero as the divisor of its private K chunk.  One is the canonical
        ;; executable representation of "do not split"; the selector still chooses split-K only
        ;; when this expression is at least two.
        starved (klaunch/minimum 1 (klaunch/floor-div (dec fill-workgroups)
                                                      output-workgroups))]
    (klaunch/maximum
     1
     (klaunch/product
      starved
      (klaunch/minimum (klaunch/ceil-div target-workgroups output-workgroups)
                       (klaunch/floor-div k min-split-chunk)
                       max-splits)))))

(defn mixed-dpas-schedule
  "Return the target-derived schedule facts for the current mixed f16×f16→f32 DPAS graph.

   A nil result is an honest target decline.  CUDA MMA and HIP MFMA will be separate target
   lowering rows over the same typed contraction; an arbitrary `:matrix` capability must never be
   emitted with Intel DPAS source."
  [desc requested-tile]
  (let [{:keys [family m n k subgroup]} (:matrix desc)
        backend (:backend desc)]
    (when (and (contains? #{:ze :ocl :opencl} backend)
               (= :dpas family) (= [8 16 16] [m n k])
               (= 16 subgroup)
               (contains? (hardware/supported-subgroup-sizes desc) (long subgroup)))
      (let [tile (or requested-tile (hardware/gemm-tile-for desc))
            workgroup-size (* (quot (:block-m tile) (:sg-m tile))
                              (quot (:block-n tile) (:sg-n tile))
                              subgroup)]
        {:tile tile
         :fill-workgroups (hardware/fill-workgroups desc workgroup-size)
         :matrix (:matrix desc)}))))

(defn- validate-split-factors!
  [split-factors]
  (when-not (and (vector? split-factors)
                 (= (count split-factors) (count (set split-factors)))
                 (every? #(and (integer? %) (> (long %) 1)) split-factors))
    (throw (ex-info "split-factor candidates must be unique integers greater than one"
                    {:split-factors split-factors})))
  split-factors)

(defn emit-matrix-alternatives
  "Emit direct and, when algebraically valid, split-K matrix graph schedules.

   This is the schedule contribution used by a typed contraction dispatch; it does not invent a
   scalar fallback or a new semantic operation.  A result transform is fused into the direct
   matrix store.  Split-K is withheld until the final combine can own that transform exactly—
   applying it independently to partial sums would be a silent algebraic error."
  [{:keys [id a b c m n k variant tile fill-workgroups vector-width epilogue split-factors]
    :or {vector-width 4 split-factors []}
    :as spec}]
  (when-not (contains? #{:nn :nt :tn :tt} variant)
    (throw (ex-info "matrix alternatives require :nn, :nt, :tn, or :tt variant"
                    {:id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:m m] [:n n] [:k k]
                         [:tile tile] [:fill-workgroups fill-workgroups]]]
    (when (nil? value)
      (throw (ex-info "matrix alternatives are missing a required field"
                      {:field field :spec spec}))))
  (let [split-factors (validate-split-factors! split-factors)
        spec (assoc spec :vector-width vector-width)
        split-expression (requested-splits spec)
        xmx-spec (assoc spec :requested-splits split-expression)
        direct (xmx-graph (assoc xmx-spec :split-k? false))
        dynamic-lhs (matrix-dynamic-lhs-alternative xmx-spec)
        fused-input (matrix-input-fusion-alternative xmx-spec)
        split? (not (seq epilogue))
        split (when split? (xmx-graph (assoc xmx-spec :split-k? true)))
        explicit-splits
        (when split?
          (mapv (fn [factor]
                  (xmx-graph (assoc spec
                                    :split-k? true
                                    :strategy (split-factor-strategy factor)
                                    :requested-splits factor)))
                split-factors))
        alignment-cases
        (block-io/fallback-cases (block-io/matrix-preconditions m n k) :f32-scalar)
        selector {:kind :runtime-expression-cases
                  :cases (cond-> alignment-cases
                           split? (conj {:expression split-expression :op :>= :value 2
                                         :strategy :xmx-split-k}))
                  :default :xmx-direct}]
    {:alternatives (cond-> [direct]
                     split (conj split)
                     (seq explicit-splits) (into explicit-splits)
                     dynamic-lhs (conj dynamic-lhs)
                     fused-input (conj fused-input))
     :selector selector
     :split-factor-schedules
     (into {} (map (fn [factor] [(split-factor-strategy factor) factor]))
           (if split? split-factors []))
     :result-transform-split-decline
     (when-not split? {:reason :split-k-result-transform-not-lowered})}))

(defn emit-executable
  "Emit the resident GEMM schedule as one graph or a checked runtime dispatch.

   Required keys: :id, :a/:b/:c, :m/:n/:k compiler values, :variant, :precision, :tile,
   :fill-workgroups. The mixed-precision dispatch handles the XMX pitch gate and low-occupancy
   split-K choice entirely through generic expression cases."
  [{:keys [id a b c m n k variant precision tile fill-workgroups vector-width epilogue]
    :or {vector-width 4}
    :as spec}]
  (when-not (contains? #{:nn :nt :tn :tt} variant)
    (throw (ex-info "GEMM executable requires :nn, :nt, :tn, or :tt variant"
                    {:id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:m m] [:n n] [:k k]
                         [:precision precision] [:tile tile] [:fill-workgroups fill-workgroups]]]
    (when (nil? value)
      (throw (ex-info "GEMM executable is missing a required field" {:field field :spec spec}))))
  (let [spec (assoc spec :vector-width vector-width)
        scalar (scalar-graph spec)]
    (case precision
      :f32-scalar scalar
      :mixed-f16-f32
      (let [_ (when (seq epilogue)
                (throw (ex-info
                        "standalone GEMM executable cannot manufacture its scalar fallback for a result transform"
                        {:reason :result-transform-requires-typed-contraction :id id})))
            {:keys [alternatives selector split-factor-schedules]}
            (emit-matrix-alternatives spec)]
        (kdispatch/make
         {:id (str id)
          :alternatives (into [scalar] alternatives)
          :default-strategy :xmx-direct
          :selector selector
          :provenance {:semantic-op :contraction :variant variant :lowering :gemm-schedule}
          :attributes {:tile tile :precision precision :hardware-aware? true
                       :split-factor-schedules split-factor-schedules}}))
      (throw (ex-info "unsupported GEMM precision" {:id id :precision precision})))))

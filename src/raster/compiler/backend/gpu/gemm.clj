(ns raster.compiler.backend.gpu.gemm
  "Compiler-owned executable schedules for dense GEMM.

   The public call is uniformly `(A B C M N K)` over f32 resident buffers. A schedule may be one
   scalar kernel or a graph containing conversion, layout conversion, matrix contraction, and
   split-K combination. All mixed-precision scratch and derived scheduling scalars are private to
   the graph; callers never bind them and runtimes never reconstruct the algorithm from `:gemm`."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.kernel-body-target :as kernel-body-target]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.backend.gpu.layout-transform :as layout-emitter]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.layout-stage :as layout-stage]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]))

(def ^:private default-min-split-chunk 1024)
(def ^:private default-max-splits 64)

(defn- identifier
  [value]
  (let [text (str/replace (str value) #"[^A-Za-z0-9_]" "_")]
    (if (re-find #"^[A-Za-z_]" text) text (str "k_" text))))

(defn- graph-buffer
  [id dtype elements role]
  (kgraph/buffer id dtype elements :device role))

(defn- value-use
  [buffer access]
  (kgraph/->ValueUse buffer access))

(defn- node
  [id operation uses dependencies]
  (let [scheduled (kart/attribute operation :scheduled-kernel-body)]
    (when-not (scheduled-body/scheduled-kernel-body? scheduled)
      (throw (ex-info "production GEMM graph node requires a scheduled-body certificate"
                      {:reason :gemm-scheduled-body :node id})))
    (kgraph/->ScheduledKernel
     id operation (vec uses)
     (reduce into #{} (map (comp klaunch/expression-references :value)
                           (:scalar-bindings scheduled)))
     (vec dependencies))))

(defn- epilogue-interface
  [epilogue]
  (vec
   (concat
    (for [{:keys [sym dtype] :or {dtype :float}} (:operands epilogue)]
      [(kabi/slot sym :input dtype :c-name (name sym) :role :epilogue)
       sym])
    (for [{:keys [sym dtype] :or {dtype :float}} (:scalars epilogue)]
      [(kabi/slot sym :scalar dtype :c-name (name sym) :role :epilogue)
       sym]))))

(defn- epilogue-buffer-specs
  [epilogue]
  (mapv (fn [{:keys [sym dtype map] :or {dtype :float}}]
          (let [shape (axis-map/shape map)
                elements (if (seq shape) (apply klaunch/product shape) 1)]
            {:id sym :dtype dtype :elements elements}))
        (:operands epilogue)))

(defn- outer-interface
  [{:keys [a b c m n k epilogue]}]
  (let [base [[(kabi/slot a :input :float :c-name "A" :role :lhs) a]
              [(kabi/slot b :input :float :c-name "B" :role :rhs) b]
              [(kabi/slot c :output :float :c-name "C" :role :result) c]
              [(kabi/slot m :scalar :int :c-name "M" :role :extent) m]
              [(kabi/slot n :scalar :int :c-name "N" :role :extent) n]
              [(kabi/slot k :scalar :int :c-name "K" :role :extent) k]]
        interface (into base (epilogue-interface epilogue))]
    {:abi (mapv first interface)
     :arguments (mapv second interface)}))

(defn- batched-outer-interface
  [{:keys [a b c batch m n k]}]
  {:abi [(kabi/slot a :input :float :c-name "A" :role :lhs)
         (kabi/slot b :input :float :c-name "B" :role :rhs)
         (kabi/slot c :output :float :c-name "C" :role :result)
         (kabi/slot batch :scalar :int :c-name "batch" :role :extent)
         (kabi/slot m :scalar :int :c-name "M" :role :extent)
         (kabi/slot n :scalar :int :c-name "N" :role :extent)
         (kabi/slot k :scalar :int :c-name "K" :role :extent)]
   :arguments [a b c batch m n k]})

(defn- public-outer-interface
  [spec]
  (let [{:keys [abi arguments]} (outer-interface spec)]
    (kgraph/public-interface abi arguments)))

(defn- public-batched-outer-interface
  [spec]
  (let [{:keys [abi arguments]} (batched-outer-interface spec)]
    (kgraph/public-interface abi arguments)))

(defn- extents
  [{:keys [m n k variant]}]
  {:a-elements (if (contains? #{:tn :tt} variant)
                 (klaunch/product k m) (klaunch/product m k))
   :b-elements (if (contains? #{:nt :tt} variant)
                 (klaunch/product n k) (klaunch/product k n))
   :c-elements (klaunch/product m n)})

(defn- effects
  [{:keys [a b c epilogue]}]
  {:kind :tensor-contraction
   :reads (into [a b] (map :sym) (:operands epilogue))
   :writes [c]})

(defn- emit-scheduled-body-artifact
  [{:keys [kernel-name source body arguments effects legality numerics phase target-dialect
           parameter-names provenance attributes]
    :or {target-dialect :opencl-intel effects {:kind :tensor-contraction-stage}
         provenance {} attributes {}}}]
  (let [uses (scheduled-body/derive-uses body arguments)
        scheduled
        (scheduled-body/make
         {:source source
          :body body
          :arguments arguments
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

(defn- scalar-contraction-form
  [variant]
  (case variant
    :nn '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* i k) l))
                                  (aget B (+ (* l n) j))))
    :nt '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* i k) l))
                                  (aget B (+ (* j k) l))))
    :tn '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* l m) i))
                                  (aget B (+ (* l n) j))))
    :tt '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* l m) i))
                                  (aget B (+ (* j k) l))))))

(defn- portable-scalar-matrix-plan
  [variant stage-id]
  (let [form (scalar-contraction-form variant)
        facts (contraction-facts/contraction-facts form :dtype :float)
        operation (contract-lower/contract-form->segred
                   form :id stage-id :dtype :float :facts facts)
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
  (let [{:keys [abi arguments]} (public-outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (extents spec)
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
     {:inputs [(graph-buffer a :float a-elements :input)
               (graph-buffer b :float b-elements :input)]
      :outputs [(graph-buffer c :float c-elements :output)]
      :scalars (kgraph/interface-scalars abi arguments)
      :nodes [(node stage-id gemm
                    [(value-use a :read) (value-use b :read) (value-use c :write)] [])]
      :abi abi :arguments arguments
      :effects (effects spec)
      :provenance {:semantic-op :contraction :variant variant :lowering :scalar-gemm}
      :attributes {:strategy :f32-scalar :variant variant :precision :f32}})))

(defn- convert-artifact
  [kernel-name stage-id in out elements vector-width phase]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        kernel-body
        (layout-emitter/cast-body
         {:kernel-name kernel-name :input in :output out
          :source-dtype :float :destination-dtype :half :vector-width vector-width
          :rounding :nearest-even :overflow :ieee})]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name
      :source (layout-stage/make
               {:id stage-id :operation :cast :input in :output out
                :input-shape [elements] :output-shape [elements]
                :input-dtype :float :output-dtype :half
                :policy {:rounding :nearest-even :overflow :ieee}})
      :body kernel-body :arguments [in out elements]
      :effects {:kind :layout-transform-stage}
      :legality {:kind :dense-affine-cast :vector-width vector-width}
      :numerics {:mode :bounded-error :policy :f32-to-f16-storage
                 :rounding :nearest-even :accumulator-dtype :half
                 :error-model {:kind :ieee-f16-conversion :overflow :ieee}}
      :phase phase
      :attributes {:vector-width vector-width :from :float :to :half
                   :rounding :nearest-even :overflow :ieee
                   :cacheable-transform? true}
      :parameter-names {in "input" out "output" :layout-elements "n"}})))

(defn- transpose-artifact
  [kernel-name stage-id in out rows cols phase]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        kernel-body
        (layout-emitter/transpose-body
         {:kernel-name kernel-name :input in :output out :element-dtype :half})]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name
      :source (layout-stage/make
               {:id stage-id :operation :transpose :input in :output out
                :input-shape [rows cols] :output-shape [cols rows]
                :input-dtype :half :output-dtype :half
                :policy {:permutation [1 0]}})
      :body kernel-body :arguments [in out rows cols]
      :effects {:kind :layout-transform-stage}
      :legality {:kind :bijective-affine-permutation :permutation [1 0]}
      :numerics {:mode :exact :policy :bit-preserving-permutation}
      :phase phase
      :attributes {:layout :transpose :dtype :half :cacheable-transform? true}
      :parameter-names {in "input" out "output"
                        :layout-rows "rows" :layout-cols "cols"}})))

(defn- scheduled-matrix-body
  "Build one canonical f16 matrix KernelBody without selecting a target spelling."
  [{:keys [kernel-name id a b c m n k dimension-parameters tile result-dtype provenance
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes epilogue]
    :or {result-dtype :float provenance {}}}]
  (let [dimension-parameters
        (or dimension-parameters
            (if (and (every? #(or (symbol? %) (keyword? %)) [m n k])
                     (= 3 (count (set [m n k]))))
              [m n k]
              ['M 'N 'K]))]
    (contraction-schedule/matrix-body
     {:id (or id [:gemm kernel-name])
      :row a :col b :out c
      :dimensions [m n k]
      :dimension-parameters dimension-parameters
      :axis-symbols ['i 'j 'l]
      :tile tile
      :bindings {:row a :col b}
      :epilogue epilogue
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
  ScalarRegion on every store and is lowered as part of the body."
  [{:keys [kernel-name id a b c m n k dimension-parameters tile result-dtype provenance
           target-dialect
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes parameter-names epilogue]
    :or {result-dtype :float provenance {} target-dialect :opencl-intel}}]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        kernel-body (scheduled-matrix-body
                     {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
                      :dimension-parameters dimension-parameters :tile tile
                      :result-dtype result-dtype :provenance provenance
                      :additional-parameters additional-parameters
                      :additional-indices additional-indices :buffer-shapes buffer-shapes
                      :buffer-views buffer-views :operation-buffers operation-buffers
                      :k-range k-range :launch-group-count launch-group-count
                      :attributes attributes :epilogue epilogue})
        emitted (matrix-target/emit-matrix-kernel
                 kernel-name kernel-body target-dialect {:parameter-names parameter-names})]
    (assoc emitted
           :kernel-name kernel-name
           :workgroup-size (get-in kernel-body [:launch :workgroup-size]))))

(defn- split-k-matrix-spec
  [{:keys [kernel-name id a b c m n k kc splits tile provenance]}]
  (let [z 'k-slice
        c-view 'split-result-view
        k-lower (kbody/expression :mul z kc)
        k-upper (kbody/expression :min (kbody/expression :add k-lower kc) k)]
    {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
     :tile tile :result-dtype :float :provenance provenance
     :additional-parameters [(kbody/->KernelParameter kc :scalar :int [] nil nil :schedule)
                             (kbody/->KernelParameter splits :scalar :int [] nil nil :schedule)]
     :additional-indices [(kbody/->IndexBinding z :group 2)]
     :buffer-shapes {c [splits m n]}
     :buffer-views [{:id c-view :buffer c
                     :element-offset (kbody/expression :mul z m n)
                     :shape [m n]}]
     :operation-buffers {c c-view}
     :k-range [k-lower k-upper]
     :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value n) (:block-n tile))
                          (klaunch/ceil-div (klaunch/runtime-value m) (:block-m tile))
                          (klaunch/runtime-value splits)]
     :attributes {:grid-z {:index z :extent splits :purpose :reduction-partition}}
     :parameter-names {kc "KC" splits "splits"}}))

(defn emit-scheduled-split-k-kernel
  "Lower a grid-Z partition of the K reduction into disjoint f32 output views."
  [spec]
  (emit-scheduled-matrix-kernel (split-k-matrix-spec spec)))

(defn- batched-matrix-spec
  [{:keys [kernel-name id a b c m n k batch tile provenance batching]
    :or {batching {:row true :col true}}}]
  (let [z 'slab
        a-view 'batch-lhs-view
        b-view 'batch-rhs-view
        c-view 'batch-result-view
        row-batched? (get batching :row true)
        col-batched? (get batching :col true)
        a-shape (if row-batched? [batch m k] [m k])
        b-shape (if col-batched? [batch k n] [k n])
        buffer-views
        (cond-> [{:id c-view :buffer c
                  :element-offset (kbody/expression :mul z m n) :shape [m n]}]
          row-batched?
          (conj {:id a-view :buffer a
                 :element-offset (kbody/expression :mul z m k) :shape [m k]})
          col-batched?
          (conj {:id b-view :buffer b
                 :element-offset (kbody/expression :mul z k n) :shape [k n]}))
        operation-buffers
        (cond-> {c c-view}
          row-batched? (assoc a a-view)
          col-batched? (assoc b b-view))]
    {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
     :tile tile :result-dtype :float :provenance provenance
     :additional-parameters [(kbody/->KernelParameter batch :scalar :int [] nil nil :schedule)]
     :additional-indices [(kbody/->IndexBinding z :group 2)]
     :buffer-shapes {a a-shape b b-shape c [batch m n]}
     :buffer-views buffer-views
     :operation-buffers operation-buffers
     :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value n) (:block-n tile))
                          (klaunch/ceil-div (klaunch/runtime-value m) (:block-m tile))
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
  [{:keys [kernel-name target-dialect parameter-names argument-values source-operation phase]
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
  [{:keys [id m n k tile epilogue]} stage-id kernel-name a b c split-k? kc splits phase]
  (let [reduction (if split-k?
                    (let [slice 'k-slice
                          lower (kbody/expression :mul slice kc)]
                      {:kind :split-k :slice slice :chunk kc :partitions splits
                       :range [lower (kbody/expression
                                      :min (kbody/expression :add lower kc) k)]})
                    {:kind :full :range [0 k]})
        stage (matrix-stage/make
               {:id stage-id
                :lhs a :rhs b :result c :dimensions [m n k]
                :reduction reduction
                :result-shape (if split-k? [splits m n] [m n])
                :epilogue (when-not split-k? epilogue)})
        emit-args {:kernel-name kernel-name
                   :id stage-id
                   :a a :b b :c c :m m :n n :k k
                   :tile tile :result-dtype :float
                   :epilogue (when-not split-k? epilogue)
                   :phase phase
                   :source-operation stage
                   :provenance {:operation-id id :phase phase}}]
    (emit-scheduled-matrix-artifact
     (if split-k?
       (assoc (split-k-matrix-spec
               (assoc emit-args :kc :k-chunk :splits :splits))
              :phase phase :source-operation stage
              :argument-values {:k-chunk kc :splits splits})
       emit-args))))

(defn- split-k-combine-plan
  [stage-id]
  (let [form '(raster.par/contract C [[i mn]] [[s splits]]
                                   (clojure.core/aget
                                    partials (clojure.core/+ (clojure.core/* s mn) i)))
        facts (contraction-facts/contraction-facts form :dtype :float)
        operation (contract-lower/contract-form->segred
                   form :id stage-id :dtype :float :facts facts)
        planned (contraction-schedule/plan-portable-body
                 facts operation {}
                 {:array-types {'partials :float 'C :float}
                  :scalar-types {'mn :int 'splits :int}})
        _ (when-not (:ok planned)
            (throw (ex-info "split-K combination did not admit the portable contraction schedule"
                            {:reason :raster/bug :plan planned})))]
    {:operation operation :body (:body planned) :plan planned}))

(defn emit-split-k-combine-kernel
  "Lower C[i] = sum_s partials[s, i] through the generic portable contraction schedule."
  ([kernel-name] (emit-split-k-combine-kernel kernel-name :opencl-intel))
  ([kernel-name target-dialect]
   (let [kernel-name (c-emit/c-symbol kernel-name)
         kernel-body (:body (split-k-combine-plan [:direct-split-k-combine kernel-name]))
         source (kernel-body-opencl/emit-scalar-kernel
                 kernel-name kernel-body
                {:target-dialect target-dialect
                 :parameter-names {'partials "partials" 'C "C"
                                   'mn "mn" 'splits "splits" '_nseg "_nseg"}})]
     {:kernel-name kernel-name :source source :kernel-body kernel-body :workgroup-size 256})))

(defn- combine-artifact
  [kernel-name stage-id partials c mn splits]
  (let [{:keys [operation body]} (split-k-combine-plan stage-id)]
    (emit-scheduled-body-artifact
     {:kernel-name kernel-name :source operation :body body
      :arguments [partials c mn splits mn]
      :effects {:kind :tensor-contraction-stage}
      :legality {:kind :portable-contraction :purpose :split-k-combine}
      :numerics {:mode :reassociated :policy :sequential-segment-fold
                 :rounding :nearest-even :accumulator-dtype :float}
      :phase :split-k-combine
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

(defn- xmx-graph
  [{:keys [id a b c m n k variant tile vector-width requested-splits split-k? epilogue
           strategy]
    :as spec}]
  (let [{:keys [abi arguments]} (public-outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (extents spec)
        epilogue-buffers (epilogue-buffer-specs epilogue)
        strategy (or strategy (if split-k? :xmx-split-k :xmx-direct))
        prefix (identifier (str id "_" (name strategy)))
        a16 [:gemm id strategy :a16]
        b16 [:gemm id strategy :b16]
        at16 [:gemm id strategy :at16]
        bt16 [:gemm id strategy :bt16]
        partials [:gemm id strategy :partials]
        final-a (if (contains? #{:tn :tt} variant) at16 a16)
        final-b (if (contains? #{:nt :tt} variant) bt16 b16)
        kc (when split-k?
             (klaunch/align-up (klaunch/ceil-div k requested-splits) (:block-k tile)))
        splits (when split-k? (klaunch/ceil-div k kc))
        partial-elements (when split-k? (klaunch/product splits m n))
        convert-a-id [:gemm id strategy :convert-a]
        convert-b-id [:gemm id strategy :convert-b]
        transpose-a-id [:gemm id strategy :transpose-a]
        transpose-b-id [:gemm id strategy :transpose-b]
        contract-id [:gemm id strategy :contract]
        convert-a (convert-artifact (str prefix "_convert_a") convert-a-id
                                    a a16 a-elements vector-width
                                    :convert-a)
        convert-b (convert-artifact (str prefix "_convert_b") convert-b-id
                                    b b16 b-elements vector-width
                                    :convert-b)
        transpose-a (when (contains? #{:tn :tt} variant)
                      (transpose-artifact (str prefix "_transpose_a") transpose-a-id
                                          a16 at16 k m
                                          :transpose-a))
        transpose-b (when (contains? #{:nt :tt} variant)
                      (transpose-artifact (str prefix "_transpose_b") transpose-b-id
                                          b16 bt16 n k
                                          :transpose-b))
        contract-output (if split-k? partials c)
        contract (gemm-artifact spec contract-id (str prefix "_contract")
                                final-a final-b contract-output
                                split-k? kc splits :matrix-contract)
        combine (when split-k?
                  (combine-artifact (str prefix "_combine")
                                    [:gemm id strategy :combine]
                                    partials c c-elements splits))
        nodes (cond->
               [(node convert-a-id convert-a [(value-use a :read) (value-use a16 :write)] [])
                (node convert-b-id convert-b [(value-use b :read) (value-use b16 :write)] [])]
                transpose-a
                (conj (node transpose-a-id transpose-a
                            [(value-use a16 :read) (value-use at16 :write)] [convert-a-id]))
                transpose-b
                (conj (node transpose-b-id transpose-b
                            [(value-use b16 :read) (value-use bt16 :write)] [convert-b-id]))
                true
                (conj (node contract-id contract
                            (into [(value-use final-a :read) (value-use final-b :read)
                                   (value-use contract-output :write)]
                                  (map #(value-use (:id %) :read))
                                  epilogue-buffers)
                            [(if transpose-a transpose-a-id convert-a-id)
                             (if transpose-b transpose-b-id convert-b-id)]))
                combine
                (conj (node [:gemm id strategy :combine] combine
                            [(value-use partials :read) (value-use c :write)] [contract-id])))
        temporaries (cond-> [(graph-buffer a16 :half a-elements :temporary)
                             (graph-buffer b16 :half b-elements :temporary)]
                      transpose-a (conj (graph-buffer at16 :half a-elements :temporary))
                      transpose-b (conj (graph-buffer bt16 :half b-elements :temporary))
                      split-k? (conj (graph-buffer partials :float partial-elements :temporary)))]
    (kgraph/make
     {:inputs (into [(graph-buffer a :float a-elements :input)
                    (graph-buffer b :float b-elements :input)]
                   (map #(graph-buffer (:id %) (:dtype %) (:elements %) :input))
                   epilogue-buffers)
      :outputs [(graph-buffer c :float c-elements :output)]
      :temporaries temporaries
      :scalars (kgraph/interface-scalars abi arguments)
      :nodes nodes
      :abi abi :arguments arguments
      :effects (effects spec)
      :provenance {:semantic-op :contraction :variant variant :lowering :xmx-gemm}
      :attributes {:strategy strategy :variant variant :precision :mixed-f16-f32
                   :tile tile :requested-splits requested-splits}})))

(defn- batched-gemm-artifact
  [{:keys [id a b c batch m n k tile batching]}]
  (let [kernel-name (str (identifier (str id "_xmx_batched")) "_contract")
        stage (matrix-stage/make
               {:id [:gemm id :xmx-batched :contract]
                :lhs a :rhs b :result c :dimensions [m n k]
                :batching {:extent batch
                           :lhs (get batching :row true)
                           :rhs (get batching :col true)}
                :reduction {:kind :full :range [0 k]}
                :result-shape [batch m n]})]
    (emit-scheduled-matrix-artifact
     (assoc (batched-matrix-spec
             {:kernel-name kernel-name
              :id [:gemm id :xmx-batched]
              :a a :b b :c c :m m :n n :k k :batch batch :batching batching
              :tile tile
              :provenance {:operation-id id :phase :matrix-contract}})
            :phase :matrix-contract
            :source-operation stage))))

(defn emit-batched-matrix-alternative
  "Emit one compiler-owned matrix schedule for a leading batch of dense NN contractions.

   The input and result tensors remain ordinary contiguous f32 values.  Flat layout adapters
   convert both operands once, then a grid-Z-selected matrix KernelBody interprets them as
   [batch,M,K], [batch,K,N], and [batch,M,N] views.  The return value deliberately is not a
   standalone dispatch: the originating typed contraction supplies its general fallback and this
   schedule contributes the alignment selector that chooses between them."
  [{:keys [id a b c batch m n k variant tile vector-width batching]
    :or {vector-width 4 batching {:row true :col true}}
    :as spec}]
  (when-not (= :nn variant)
    (throw (ex-info "batched matrix schedule currently requires canonical NN storage"
                    {:reason :batched-matrix-layout-not-lowered
                     :id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:batch batch]
                         [:m m] [:n n] [:k k] [:tile tile]]]
    (when (nil? value)
      (throw (ex-info "batched matrix schedule is missing a required field"
                      {:reason :raster/bug :field field :spec spec}))))
  (let [{:keys [abi arguments]} (public-batched-outer-interface spec)
        a-elements (if (get batching :row true)
                     (klaunch/product batch m k)
                     (klaunch/product m k))
        b-elements (if (get batching :col true)
                     (klaunch/product batch k n)
                     (klaunch/product k n))
        c-elements (klaunch/product batch m n)
        prefix (identifier (str id "_xmx_batched"))
        a16 [:gemm id :xmx-batched :a16]
        b16 [:gemm id :xmx-batched :b16]
        convert-a-id [:gemm id :xmx-batched :convert-a]
        convert-b-id [:gemm id :xmx-batched :convert-b]
        contract-id [:gemm id :xmx-batched :contract]
        convert-a (convert-artifact (str prefix "_convert_a") convert-a-id
                                    a a16 a-elements vector-width
                                    :convert-a)
        convert-b (convert-artifact (str prefix "_convert_b") convert-b-id
                                    b b16 b-elements vector-width
                                    :convert-b)
        contract (batched-gemm-artifact
                  (assoc spec :a a16 :b b16 :c c :batching batching))
        graph
        (kgraph/make
         {:inputs [(graph-buffer a :float a-elements :input)
                   (graph-buffer b :float b-elements :input)]
          :outputs [(graph-buffer c :float c-elements :output)]
          :temporaries [(graph-buffer a16 :half a-elements :temporary)
                        (graph-buffer b16 :half b-elements :temporary)]
          :scalars (kgraph/interface-scalars abi arguments)
          :nodes [(node convert-a-id convert-a
                        [(value-use a :read) (value-use a16 :write)] [])
                  (node convert-b-id convert-b
                        [(value-use b :read) (value-use b16 :write)] [])
                  (node contract-id contract
                        [(value-use a16 :read) (value-use b16 :read)
                         (value-use c :write)]
                        [convert-a-id convert-b-id])]
          :abi abi :arguments arguments
          :effects (effects spec)
          :provenance {:semantic-op :contraction
                       :variant :nn
                       :lowering :batched-xmx-gemm}
          :attributes {:strategy :xmx-batched
                       :variant :nn
                       :batched? true
                       :batching batching
                       :precision :mixed-f16-f32
                       :tile tile}})]
    {:graph graph
     :selector
     {:kind :runtime-expression-cases
      :cases [{:expression n :op :< :value 8 :strategy :f32-scalar}
              {:expression k :op :< :value 8 :strategy :f32-scalar}
              {:expression (kbody/expression :mod n 8)
               :op :> :value 0 :strategy :f32-scalar}
              {:expression (kbody/expression :mod k (get-in tile [:matrix :k]))
               :op :> :value 0 :strategy :f32-scalar}]
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
        ;; the historic "never split a filling GEMM" legality/policy gate inside the same checked
        ;; arithmetic IR without introducing an opaque conditional callback.
        starved (klaunch/minimum 1 (klaunch/floor-div (dec fill-workgroups)
                                                      output-workgroups))]
    (klaunch/product
     starved
     (klaunch/minimum (klaunch/ceil-div target-workgroups output-workgroups)
                      (klaunch/floor-div k min-split-chunk)
                      max-splits))))

(defn mixed-dpas-schedule
  "Return the target-derived schedule facts for the current mixed f16×f16→f32 DPAS graph.

   A nil result is an honest target decline.  CUDA MMA and HIP MFMA will be separate target
   lowering rows over the same typed contraction; an arbitrary `:matrix` capability must never be
   emitted with Intel DPAS source."
  [desc requested-tile]
  (let [{:keys [family m n k subgroup]} (:matrix desc)
        backend (:backend desc)]
    (when (and (contains? #{:ze :opencl} backend)
               (= :dpas family) (= [8 16 16] [m n k])
               (contains? #{8 16} subgroup)
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
        [{:expression n :op :< :value 8 :strategy :f32-scalar}
         {:expression k :op :< :value 8 :strategy :f32-scalar}
         {:expression (kbody/expression :mod n 8)
          :op :> :value 0 :strategy :f32-scalar}
         {:expression (kbody/expression :mod k (get-in tile [:matrix :k]))
          :op :> :value 0 :strategy :f32-scalar}]
        selector {:kind :runtime-expression-cases
                  :cases (cond-> alignment-cases
                           split? (conj {:expression split-expression :op :>= :value 2
                                         :strategy :xmx-split-k}))
                  :default :xmx-direct}]
    {:alternatives (cond-> [direct]
                     split (conj split)
                     (seq explicit-splits) (into explicit-splits))
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

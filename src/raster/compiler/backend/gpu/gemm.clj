(ns raster.compiler.backend.gpu.gemm
  "Compiler-owned executable schedules for dense GEMM.

   The public call is uniformly `(A B C M N K)` over f32 resident buffers. A schedule may be one
   scalar kernel or a graph containing conversion, layout conversion, matrix contraction, and
   split-K combination. All mixed-precision scratch and derived scheduling scalars are private to
   the graph; callers never bind them and runtimes never reconstruct the algorithm from `:gemm`."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.emitted-graph-interface :as emitted-interface]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.kernel-body-target :as kernel-body-target]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
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
            [raster.compiler.passes.parallel.mixed-matrix-candidate :as mixed-candidate]
            [raster.compiler.passes.parallel.mixed-matrix-body :as mixed-body]
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
  [{:keys [kernel-name target-dialect parameter-names]
    :or {target-dialect :opencl-intel} :as spec}]
  (kernel-body-target/emit-artifact
   (c-emit/c-symbol kernel-name) (mixed-body/make-schedule spec) target-dialect
   {:parameter-names parameter-names}))

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
  [kernel-name stage phase target-dialect scalar-types caller-options]
  (kernel-body-target/emit-artifact
   (c-emit/c-symbol kernel-name) (mixed-body/schedule-cast stage phase scalar-types)
   target-dialect
   (merge {:parameter-names {(:input stage) "input" (:output stage) "output" :layout-elements "n"}}
          (select-keys caller-options [:target-descriptor :scalar-math]))))

(defn- transpose-artifact
  [kernel-name stage phase target-dialect scalar-types caller-options]
  (kernel-body-target/emit-artifact
   (c-emit/c-symbol kernel-name) (mixed-body/schedule-transpose stage phase scalar-types)
   target-dialect
   (merge {:parameter-names {(:input stage) "input" (:output stage) "output"
                            :layout-rows "rows" :layout-cols "cols"}}
          (select-keys caller-options [:target-descriptor :scalar-math]))))



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
        kernel-body (mixed-body/scheduled-matrix-body
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


(defn emit-scheduled-split-k-kernel
  "Lower a grid-Z partition of the K reduction into disjoint f32 output views."
  [spec]
  (emit-scheduled-matrix-kernel
   (assoc (mixed-body/split-k-matrix-spec spec)
          :parameter-names {(:kc spec) "KC" (:splits spec) "splits"})))


(defn emit-scheduled-batched-matrix-kernel
  "Lower independent dense matrix slabs as grid-Z-selected contiguous buffer views.

   `batching` states whether each operand carries the leading batch axis.  A false entry denotes a
   stable broadcast operand (most commonly shared model weights), so its view has zero batch
   offset instead of materializing a repeated tensor."
  [spec]
  (emit-scheduled-matrix-kernel
   (assoc (mixed-body/batched-matrix-spec spec)
          :parameter-names {(:batch spec) "batch"})))

(defn- emit-scheduled-matrix-artifact
  [{:keys [kernel-name target-dialect parameter-names]
    :or {target-dialect :opencl-intel} :as spec}]
  (let [kernel-name (c-emit/c-symbol kernel-name)
        scheduled (mixed-body/schedule-matrix (assoc spec :kernel-name kernel-name))]
    (kernel-body-target/emit-artifact
     kernel-name scheduled target-dialect
     (merge {:parameter-names parameter-names}
            (select-keys spec [:target-descriptor :scalar-math])))))

(defn- gemm-artifact
  [stage kernel-name phase target-dialect scalar-types caller-options]
  (let [spec (mixed-body/matrix-stage-spec stage phase scalar-types)
        parameter-names (cond
                          (:batching stage) {(get-in stage [:batching :extent]) "batch"}
                          (= :split-k (get-in stage [:reduction :kind]))
                          {:k-chunk "KC" :splits "splits"}
                          :else nil)]
    (emit-scheduled-matrix-artifact
     (merge (assoc spec :kernel-name kernel-name :target-dialect target-dialect
                        :parameter-names parameter-names)
            (select-keys caller-options [:scalar-math :target-descriptor])))))

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
  [kernel-name operation partials c mn splits target-dialect scalar-types caller-options]
  (kernel-body-target/emit-artifact
   (c-emit/c-symbol kernel-name)
   (mixed-body/schedule-combine operation partials c mn splits scalar-types)
   target-dialect
   (merge {:parameter-names {'partials "partials" 'C "C" 'mn "mn" 'splits "splits" '_nseg "_nseg"}}
          (select-keys caller-options [:target-descriptor :scalar-math]))))

(defn split-factor-strategy
  "Stable strategy identity for one explicit split-K candidate."
  [factor]
  (when-not (and (integer? factor) (> (long factor) 1))
    (throw (ex-info "split factor must be an integer greater than one"
                    {:split-factor factor})))
  (keyword (str "xmx-split-k-" factor)))


(defn- emit-stage-artifact
  [target-dialect prefix scalar-types {:keys [operation] :as node} caller-options]
  (let [phase (last (:id node))]
    (cond
      (layout-stage/layout-stage? operation)
      (case (:operation operation)
        :cast (convert-artifact (str prefix "_" (name phase)) operation
                                phase target-dialect scalar-types caller-options)
        :transpose (transpose-artifact (str prefix "_" (name phase)) operation
                                       phase target-dialect scalar-types caller-options))

      (matrix-stage/matrix-stage? operation)
      (gemm-artifact operation (str prefix "_" (name phase))
                     :matrix-contract target-dialect scalar-types caller-options)

      (segop/seg-red? operation)
      (let [{:keys [partials output mn splits]} (mixed-body/split-combine-values operation)]
        (combine-artifact (str prefix "_" (name phase)) operation
                          partials output mn splits target-dialect scalar-types caller-options))

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
                :or {target-dialect :opencl-intel prefix "scheduled_gemm"} :as opts}]
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
           (let [artifact (emit-stage-artifact target-dialect prefix scalar-types node
                                               (select-keys opts [:scalar-math :target-descriptor]))
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
    (kexec/validate!
     (if (some? (:abi emitted))
       emitted
       (emitted-interface/finalize!
        emitted (c-dialect/target (c-dialect/resolve! target-dialect)) scalar-types)))))

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
  (mixed-candidate/target-schedule desc requested-tile))

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

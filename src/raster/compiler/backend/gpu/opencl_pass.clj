(ns raster.compiler.backend.gpu.opencl-pass
  "Unified OpenCL pipeline pass.

   Walks S-expressions, replaces par forms with Level Zero kernel
   invocation markers. Uses SegOp-based codegen for generic map/reduce,
   delegates to par_opencl generators for specialized forms.

   This is the GPU counterpart of simd-pass — both consume the same
   par form vocabulary but produce different target code."
  (:require [clojure.set :as set]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.soac-lower]
            [raster.compiler.backend.gpu.segop-opencl :as segop-cl]
            [raster.compiler.passes.parallel.contract-route :as croute]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-fuse :as swr-fuse]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-route :as swr-route]
            [raster.compiler.backend.gpu.par-opencl :as legacy]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [raster.runtime.hardware :as hw]))

;; Buffer semantics of the emitted kernel-invoke marker: invoke-registered-kernel
;; WRITES its `out` arg in place (arg 2 of [kname inputs out scalars n]) and
;; RETURNS it, so the binding sym is a pure alias of the out buffer. Declared in
;; the registry (not pattern-matched in passes) so resolve-alength's call-through
;; alias tracking resolves `(alength <invoke-binding>)` to the out buffer's alloc
;; size — a shape horizontal fusion produces when a later map's bound reads the
;; length of a fused map's result.
(descriptor/register-op-descriptor!
 'raster.gpu.ze-runtime/invoke-registered-kernel
 {:buffer {:allocates? false :in-place-arg 2}})

;; ================================================================
;; Single source of declared GPU param types (shared by BOTH compile entries:
;; raster.gpu.core/compile-deftm-internal! and the pipeline's pass-backend).
;; ================================================================

(defn derive-param-types
  "Declared scalar + array element types for the GPU emitter, read from a deftm's params + tags
  (the typed-dispatch system already knows these — we read them instead of letting the emitter
  guess from parameter names). Scalar params: long/int→:int, double/float→:float. Array params:
  the tag's element dtype, with float-family (float/double) mapped to the KERNEL dtype (a single-
  precision kernel reads float buffers regardless of a parametric (All [T]) param's default).
  Returns {:scalar-types {sym kw} :array-types {sym kw}}, attached as form metadata that
  opencl-pass reads. ONE derivation, both compile paths."
  [params tags dtype]
  (when (and params tags)
    {:scalar-types (into {} (keep (fn [[p t]]
                                    (case (dtype/dtype-for-scalar-tag t)
                                      (:long :int) [p :int]
                                      (:double :float) [p :float]
                                      nil))
                                  (map vector params tags)))
     :array-types (into {} (keep (fn [[p t]]
                                   (when-let [dt (dtype/dtype-for-array-tag t)]
                                     [p (if (#{:float :double} dt) dtype dt)]))
                                 (map vector params tags)))}))

(def ^:private segop-id-counter (atom 0))

(def ^:private fatal-reasons
  "A violated invariant is not \"the SegOp path does not cover this form\". Letting one fall through
   to the legacy generator would run the pipeline on with the bug intact."
  #{:raster/fatal :raster/bug})

(defn- segop-attempt
  "Run a SegOp lowering `thunk`, returning the SegOp or nil and recording a structured decline.
   The caller decides whether that decline is an allowed scheduling fallback or an illegal
   operation."
  [stats kind form dtype fallback thunk]
  ;; Only structured conversion refusals are legal fallbacks. An implementation exception is not
  ;; evidence that alternate codegen is legal, and must escape instead of becoming a decline.
  (let [r (try {:ok (thunk)} (catch clojure.lang.ExceptionInfo e {:err e}))]
    (cond
      (:err r)
      (let [e (:err r)]
        (when (contains? fatal-reasons (:reason (ex-data e))) (throw e))
        (swap! stats update :segop-declined (fnil conj [])
               {:kind kind :op (when (seq? form) (first form))
                :dtype (or dtype :double)
                :reason (or (:reason (ex-data e)) :no-lowering-rule)
                :message (.getMessage e)
                :fallback fallback})
        nil)

      (nil? (:ok r))
      (do (swap! stats update :segop-declined (fnil conj [])
                 {:kind kind :op (when (seq? form) (first form))
                  :dtype (or dtype :double)
                  :reason :lowering-produced-nothing
                  :fallback fallback})
          nil)

      :else (:ok r))))

(def ^:dynamic *bound-segops*
  "The SegOp records owned by the ParallelProgram equation currently being transformed, or nil
   when this backend was called directly on an S-expression.

   This is the seam that makes the SegOp boundary REAL instead of nominal. `segop-lower` lowered
   every par form with the real target device. The equation is consumed here; re-lowering remains
   a counted compatibility path for callers that invoke opencl-pass directly."
  nil)

(defn- take-bound-segop
  "The precomputed SegOp of `kind` for the current binding, if segop-lower produced one."
  [stats kind pred]
  (when-let [so (first (filter pred *bound-segops*))]
    (swap! stats update :segop-reused (fnil inc 0))
    so))

(defn- par->segmap
  "The SegMap for a par/map! form: the one `segop-lower` already computed for this binding when
   available; otherwise re-lowered here with the REAL `device-id` (it was `:ze:0` hardcoded) and
   counted as `:segop-relowered`. SegMap is a full conversion: a missing rule is an illegal
   operation, never permission to select a second emitter."
  [stats form dtype device-id]
  (or (take-bound-segop stats :segmap #(instance? raster.compiler.ir.segop.SegMap %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (or (segop-attempt stats :segmap form dtype :none
                             #(let [par-info (par/extract-par-map-info form)
                                    soac (raster.compiler.ir.soac/par-form->soac
                                          (:out par-info) form (swap! segop-id-counter inc))]
                                (first (raster.compiler.passes.parallel.soac-lower/lower-soac
                                        soac (or device-id :ze:0) :dtype (or dtype :double)))))
              (let [decline (last (:segop-declined @stats))]
                (throw (ex-info "par/map! remains illegal after full SegOp conversion"
                                {:reason :illegal-op-remains
                                 :op (first form)
                                 :source form
                                 :missing-rule (:reason decline)
                                 :target-dialect :segop
                                 :decline decline
                                 :fallback :none})))))))

(defn- par->segred
  "The SegRed for a par/reduce form. Reduction is a FULL conversion: absence is an illegal
   operation at the SegOp boundary, never permission to select a second emitter."
  [stats form dtype device-id]
  (or (take-bound-segop stats :segred #(instance? raster.compiler.ir.segop.SegRed %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (or (segop-attempt stats :segred form dtype :none
                             #(let [sym (gensym "red_")
                                    soac (raster.compiler.ir.soac/par-form->soac
                                          sym form (swap! segop-id-counter inc)
                                          :dtype (or dtype :double))]
                                (first (raster.compiler.passes.parallel.soac-lower/lower-soac
                                        soac (or device-id :ze:0) :dtype (or dtype :double)))))
              (let [decline (last (:segop-declined @stats))]
                (throw (ex-info "par/reduce remains illegal after full SegOp conversion"
                                {:reason :illegal-op-remains
                                 :op (first form)
                                 :source form
                                 :missing-rule (:reason decline)
                                 :target-dialect :segop
                                 :decline decline
                                 :fallback :none})))))))

(defn- maybe-compile-spirv
  "Optionally compile OpenCL C to SPIR-V."
  [kernel compile-spirv? device-id]
  (if compile-spirv?
    (assoc kernel :spv-bytes
           (legacy/compile-kernel-to-spirv (:source kernel) :device-id device-id))
    kernel))

(def ^:private contraction-marker-unexpressed-fields
  "Descriptor work that is outside a single kernel launch.  Kernel arguments, including lift and
   epilogue values, are carried by the ABI; pre-steps require a scheduler to emit another launch."
  [:pre-steps])

(defn- ensure-contraction-marker-expressible!
  "Refuse routed work that is not a kernel argument.  Lift and epilogue arguments are represented
   by the ordered ABI; a pre-step is a separate operation and still needs explicit scheduling."
  [descriptor]
  (let [unsupported (filterv #(seq (get descriptor %))
                             contraction-marker-unexpressed-fields)]
    (when (seq unsupported)
      (throw (ex-info (str "contraction: invoke marker cannot express descriptor fields "
                           (pr-str unsupported))
                      {:reason :raster/fatal
                       :missing-rule :contraction-pre-step-scheduling
                       :target 'raster.gpu.ze-runtime/invoke-registered-contraction!
                       :strategy (:strategy descriptor)
                       :unsupported-fields unsupported
                       :unsupported-values (select-keys descriptor unsupported)
                       :fallback :none})))
    descriptor))

(defn- emit-map-invocation
  "Render the compatibility map marker from the emitter-authored ABI and ordered values."
  [{:keys [kernel-name abi arguments]}]
  (let [arguments (kabi/validate-arguments! abi arguments)
        pairs (mapv vector abi arguments)
        result-pairs (filterv #(= :result (:role (first %))) pairs)
        bound-pairs (filterv #(= :bound (:role (first %))) pairs)]
    (when-not (= 1 (count result-pairs))
      (throw (ex-info "map kernel ABI must identify exactly one :result slot"
                      {:kernel-name kernel-name :abi abi})))
    (when-not (= 1 (count bound-pairs))
      (throw (ex-info "map kernel ABI must identify exactly one :bound slot"
                      {:kernel-name kernel-name :abi abi})))
    (let [[_ out] (first result-pairs)
          [_ bound] (first bound-pairs)
          inputs (mapv second
                       (filter (fn [[slot _]]
                                 (and (not= :scalar (:kind slot))
                                      (not= :result (:role slot))))
                               pairs))
          scalars (mapv second
                        (filter (fn [[slot _]]
                                  (and (= :scalar (:kind slot))
                                       (not= :bound (:role slot))))
                                pairs))]
      (list 'raster.gpu.ze-runtime/invoke-registered-kernel
            kernel-name inputs out scalars bound))))

(defn- emit-reduction-invocation
  "Render a SegRed marker as one complete ordered ABI value vector. `result` is the caller-owned
   resident buffer for reduce-into, or nil for a host-scalar reduction whose staging runtime owns
   the temporary partial-results buffer. No input/scalar/bound positions are reconstructed here."
  [{:keys [kernel-name abi arguments]} result]
  (let [arguments (kabi/validate-arguments! abi arguments)
        arguments (mapv (fn [slot value]
                          (if (= :result (:role slot)) result value))
                        abi arguments)]
    (kabi/validate-reduction-arguments! abi arguments)
    (list 'raster.gpu.ze-runtime/invoke-registered-reduction-kernel
          kernel-name arguments)))

(defn- emit-map-void-invocation
  "Render the staging marker from the artifact's logical argument projection. The marker remains
   useful to the host evaluator, but it no longer owns or reconstructs an argument convention."
  [artifact]
  (let [plan (kcall/logical-argument-plan artifact)
        pointer-values (mapv :value (filterv :pointer? plan))
        scalar-entries (filterv (complement :pointer?) plan)
        bound-entries (filterv #(= :bound (:role (first (:slots %)))) scalar-entries)
        user-scalars (filterv #(not= :bound (:role (first (:slots %)))) scalar-entries)]
    (when-not (= 1 (count bound-entries))
      (throw (ex-info "map-void artifact must identify exactly one :bound scalar"
                      {:kernel-name (:kernel-name artifact) :plan plan})))
    (list 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel
          (:kernel-name artifact)
          pointer-values
          (mapv :value user-scalars)
          (:value (first bound-entries)))))

(defn opencl-pass
  "Pipeline pass: walk S-expression, replace par forms with GPU kernel invocations.

   Uses full SegOp conversion for par/map! and par/reduce.
   Delegates to specialized generators for stencil, scatter,
   scan, rng-fill, active-ids, reduce-by-key, compound-kernel, map-void).

   Returns {:form new-form :stats {:ze-maps N :ze-reduces N :fallback N}
            :kernels [{:kernel-name :source ...} ...]}

   Options:
     :device-id     — target Level Zero device (default :ze:0)
     :dtype         — :double or :float (default :double)
     :min-elements  — minimum elements for GPU (default 4096)
     :compile-spirv? — compile to SPIR-V now (default false)"
  [form & {:keys [device-id dtype min-elements compile-spirv? scalar-types array-types schedule]
           :or {device-id :ze:0 dtype :double min-elements 4096
                compile-spirv? false}}]
  ;; DECLARED types from derive-param-types (opts) override the name-heuristic fallback in the
  ;; kernel generators — e.g. `features` (Long→int) and `gain-offset` (Double→float, whose name
  ;; would otherwise misfire the "offset"→int heuristic). Form-meta types are the base.
  (let [parallel-program (when (parallel-program/parallel-program? form)
                           (parallel-program/validate! form segop/segop-node?))
        source-form (if parallel-program (parallel-program/source-form parallel-program) form)
        ;; Typed values supply dtypes; scheduled SegOps supply ABI roles. Logical rank cannot choose
        ;; pass-by-value versus buffer because a rank-zero result may be resident in either form.
        program-types
        (when (and parallel-program
                   (= :typed-soac (get-in parallel-program [:provenance :source-dialect])))
          (let [operations (mapcat :operations (:equations parallel-program))
                arrays (set (mapcat #(concat (or (:inputs %) #{}) (or (:outputs %) #{}))
                                    operations))
                scalars (set (mapcat #(or (:scalars %) #{}) operations))
                overlap (set/intersection arrays scalars)]
            (when (seq overlap)
              (throw (ex-info "scheduled values have contradictory buffer and scalar ABI roles"
                              {:reason :parallel-program-parameter-role-conflict
                               :values overlap})))
            {:scalar-types (parallel-program/declared-value-types parallel-program scalars)
             :array-types (parallel-program/declared-value-types parallel-program arrays)}))
        top-scalar-types (merge (or (:scalar-types program-types) {})
                                (or (:scalar-types (meta source-form)) {})
                                (or (:scalar-types (meta form)) {}) scalar-types)
        top-array-types (merge (or (:array-types program-types) {})
                               (or (:array-types (meta source-form)) {})
                               (or (:array-types (meta form)) {}) array-types)
        stats (atom {:ze-maps 0 :ze-reduces 0 :ze-compounds 0 :ze-contracts 0
                     :ze-structured-reductions 0 :kernel-graphs 0
                     :fallback 0})
        kernels (atom [])
        dispatches (atom [])
        target-desc (delay
                      (try ((requiring-resolve
                             'raster.compiler.core.hardware/descriptor-for)
                            device-id)
                           (catch Throwable _ {:device-type :gpu})))

        register-kernel!
        (fn [kernel stat-key]
          (let [_ (when (kart/kernel-artifact? kernel) (kart/validate! kernel))
                k (maybe-compile-spirv kernel compile-spirv? device-id)]
            (swap! stats update stat-key inc)
            (swap! kernels conj k)
            k))

        emit-scheduled-graph!
        (fn [scheduled]
          ;; KernelDispatch is already the verified selection value above both a single artifact
          ;; and a graph.  A one-alternative dispatch therefore gives scheduled equations the same
          ;; registry/selection seam as tuned GEMM without introducing a second graph registry.
          ;; Scan is the first graph target-lowering; later graph families extend this backend
          ;; dispatcher rather than adding source/runtime conventions per operation.
          (let [emitted0 (segop-cl/generate-kernel-graph
                          scheduled :scalar-types top-scalar-types)
                emitted (-> emitted0
                            (kgraph/map-operations
                             (fn [node]
                               (maybe-compile-spirv (:operation node)
                                                    compile-spirv? device-id)))
                            (assoc-in [:attributes :strategy] :scheduled-graph)
                            kgraph/validate!)
                dispatch (kdispatch/make
                          {:id (str "scheduled-graph-"
                                    (Integer/toUnsignedString (hash emitted) 16))
                           :alternatives [emitted]
                           :default-strategy :scheduled-graph
                           :selector {:kind :fixed-strategy :strategy :scheduled-graph}
                           :provenance {:pass :opencl :source-dialect :segop}
                           :attributes {:operation-family :scheduled-graph}})]
            (swap! kernels into (mapv :operation (:nodes emitted)))
            (swap! dispatches conj dispatch)
            (swap! stats update :kernel-graphs inc)
            ;; Staging and resident extraction consume the same registered dispatch and complete
            ;; ordered executable ABI. The device id chooses only the runtime; it is not part of
            ;; the executable's semantic call interface.
            (list 'raster.compiler.pipeline/invoke-scheduled-executable!
                  device-id (:id dispatch) (vec (:arguments emitted)))))

        transform
        (fn transform [form]
          (cond
            ;; === Proven structured segmented weighted reduction ===
            (swr-fuse/marker? form)
            (let [plan (swr-fuse/marker-plan form dtype)
                  strategy (get-in schedule [:segmented-weighted-reduction :strategy] :reference)]
              (if (= :auto strategy)
                (if-let [dispatch0 (swr-route/dynamic-dispatch plan @target-desc schedule)]
                  (let [artifacts (mapv #(maybe-compile-spirv % compile-spirv? device-id)
                                        (:alternatives dispatch0))
                        dispatch (kdispatch/validate! (assoc dispatch0 :alternatives artifacts))]
                    (swap! stats update :ze-structured-reductions inc)
                    (swap! kernels into artifacts)
                    (swap! dispatches conj dispatch)
                    (list 'raster.gpu.ze-runtime/invoke-registered-contraction-dispatch!
                          (:id dispatch)
                          (:kernel-name (kdispatch/default-alternative dispatch))
                          (vec (:arguments (kdispatch/default-alternative dispatch)))))
                  (let [routed (swr-route/route-dynamic! plan @target-desc)
                        artifact (register-kernel! (:artifact routed)
                                                   :ze-structured-reductions)]
                    (list 'raster.gpu.ze-runtime/invoke-registered-contraction!
                          (:kernel-name artifact) (vec (:arguments artifact)))))
                (let [wanted (case strategy
                               :reference :indexed-segmented-reduction-reference
                               :subgroup-score-reuse
                               :indexed-segmented-reduction-subgroup-score-reuse)
                      routed-set (swr-route/route-dynamic-candidates! plan @target-desc)
                      routed (some #(when (= wanted (:strategy %)) %) (:candidates routed-set))
                      _ (when-not routed
                          (throw (ex-info "pinned segmented weighted-reduction schedule is unavailable"
                                          {:reason :segmented-weighted-reduction-pinned-schedule-unavailable
                                           :requested strategy
                                           :available (mapv :strategy (:candidates routed-set))
                                           :declines (:declines routed-set)})))
                      artifact (register-kernel! (:artifact routed)
                                                 :ze-structured-reductions)]
                  ;; A pinned schedule remains a plain single-entry marker. Runtime dispatch is
                  ;; introduced only by :auto and therefore never overrides an explicit policy.
                  (list 'raster.gpu.ze-runtime/invoke-registered-contraction!
                        (:kernel-name artifact) (vec (:arguments artifact))))))

            ;; === Compound kernel ===
            (and (seq? form) (= 'raster.compiler/compound-kernel (first form)))
            (let [[_ metadata original-dotimes] form
                  strategy (get-in metadata [:execution :strategy])]
              (case strategy
                :local
                (let [k (register-kernel!
                         (legacy/generate-compound-local-kernel metadata
                                                                :dtype dtype :device-id device-id)
                         :ze-compounds)]
                  (legacy/emit-compound-kernel-invocation metadata [k]))
                :global
                (transform original-dotimes)))

            ;; === par/map! — SegOp path ===
            (par/par-map-form? form)
            (let [{:keys [bound out]} (par/extract-par-map-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map! form))
                (let [segmap (par->segmap stats form dtype device-id)
                      kernel (segop-cl/generate-segmap-kernel
                              segmap out
                              :dtype dtype :scalar-types top-scalar-types
                              :array-types (merge top-array-types (:array-types (meta form))))
                      k (register-kernel! kernel :ze-maps)]
                  (emit-map-invocation k))))

            ;; === par/contract — tensor contraction, routed through the DPAS legality gate ===
            ;; The routing brain (contract-route) chooses the hardware-optimal kernel: DPAS/XMX
            ;; tensorize when legal (canonical f16 matmul, pitch-aligned) → peak (byte-identical
            ;; to the hand GEMM front door); otherwise the portable register-tiled kernel (any
            ;; dtype, arbitrary dims). Emits one artifact-backed marker containing only ABI values.
            (croute/par-contract-form? form)
            (let [out-sym (second form)
                  ;; pass the REAL descriptor: route-contraction fed `(or desc {})` into
                  ;; derive-gemm-tile, so the "hardware-derived" tile was derived from an empty map
                  ;; — Arc constants on every device. device-id is already in scope here.
                  ;; CONSUME the SegContract segop-lower recorded (its facts were derived and
                  ;; verified once, at the boundary) — the same take-bound-segop seam map/reduce
                  ;; use. Without one (door C, no pass run) route from the form and count it.
                  bound-sc (take-bound-segop stats :segcontract
                                             #(instance? raster.compiler.ir.segop.SegContract %))
                  _ (when-not bound-sc (swap! stats update :segop-relowered (fnil inc 0)))
                  r (ensure-contraction-marker-expressible!
                     (croute/route-contraction
                      form :dtype dtype
                      :facts (:facts bound-sc)
                      :operation-id (:id bound-sc)
                      ;; The resolved, feasibility-checked schedule is the single source of tile
                      ;; geometry.  Previously this option reached opencl-pass but contractions
                      ;; ignored it and re-derived a default, so a pinned/autotuned :tile changed
                      ;; neither the KernelBody nor emitted source.
                      :tile (:tile schedule)
                      :desc (try ((requiring-resolve 'raster.compiler.core.hardware/descriptor-for)
                                  device-id)
                                 (catch Throwable _ nil))))
                  ;; Every routed single-entry leaf is now a complete executable artifact. Routing
                  ;; diagnostics and tensorization facts live in its attributes/provenance rather
                  ;; than being flattened into the runtime registry namespace.
                  kernel (:artifact r)
                  k (register-kernel! kernel
                                      :ze-contracts)]
              ;; A full reduction (0 free axes) has its own two-phase launch protocol + a
              ;; host-side final combine — emit the reduction invoke, not the 2-D contraction one.
              (if (= :reduction (:invoke r))
                (emit-reduction-invocation k nil)
                ;; The staging marker carries only ordered values. Output extent and complete 2-D
                ;; launch live in the artifact; resident extraction never reconstructs geometry.
                (list 'raster.gpu.ze-runtime/invoke-registered-contraction!
                      (:kernel-name k)
                      (vec (:arguments r)))))

            ;; === par/reduce-into — resident SegRed writing a caller-supplied 1-elem buffer ===
            ;; Same SegRed kernel as par/reduce (it already has an `output` param), but the
            ;; resident invoke carries the output buffer (4-arg) so it stays device-resident
            ;; instead of round-tripping a host scalar. Emitted by the reduce-fusion pass.
            (par/par-reduce-into-form? form)
            (let [{:keys [out-buf reduce-form]} (par/extract-par-reduce-into-info form)
                  segred (par->segred stats reduce-form dtype device-id)
                  kernel (segop-cl/generate-segred-kernel
                          segred out-buf :dtype dtype :scalar-types top-scalar-types
                          :array-types top-array-types)
                  k (register-kernel! kernel :ze-reduces)]
              (emit-reduction-invocation k out-buf))

            ;; === par/reduce — SegOp path ===
            (par/par-reduce-form? form)
            (let [{:keys [bound]} (par/extract-par-reduce-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-reduce form))
                (let [segred (par->segred stats form dtype device-id)
                      kernel (segop-cl/generate-segred-kernel
                              segred nil :dtype dtype :scalar-types top-scalar-types
                              :array-types top-array-types)
                      k (register-kernel! kernel :ze-reduces)]
                  (emit-reduction-invocation k nil))))

            ;; Typed segmented product reduction. The portable schedule is one deterministic
            ;; workgroup tree per segment; mixed result components remain separate ABI buffers.
            (par/par-product-reduce-form? form)
            (let [segred (par->segred stats form dtype device-id)
                  kernel (segop-cl/generate-product-reduction-kernel
                          segred :scalar-types top-scalar-types :array-types top-array-types)
                  k (register-kernel! kernel :ze-reduces)]
              (emit-map-void-invocation k))

            ;; === Specialized forms — delegate to legacy generators ===

            ;; par/map-void!
            (par/par-map-void-form? form)
            (let [{:keys [bound]} (par/extract-par-map-void-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map-void! form))
                (let [kernel (legacy/generate-par-map-void-kernel form
                                                                  :dtype dtype :device-id device-id
                                                                  :array-types top-array-types
                                                                  :scalar-types top-scalar-types)
                      k (register-kernel! kernel :ze-maps)]
                  (emit-map-void-invocation k))))

            ;; par/scan-exclusive
            (par/par-scan-exclusive-form? form)
            (let [{:keys [bound]} (par/extract-par-scan-exclusive-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-scan-exclusive form))
                (let [kernel (legacy/generate-par-scan-exclusive-kernel form
                                                                        :dtype dtype :device-id device-id)
                      k (maybe-compile-spirv kernel compile-spirv? device-id)
                      block-entry (assoc k :kernel-name (:block-kernel-name k))
                      prop-entry (assoc k :kernel-name (:prop-kernel-name k))]
                  (swap! stats update :ze-maps inc)
                  (swap! kernels conj block-entry)
                  (swap! kernels conj prop-entry)
                  (let [{:keys [out]} (par/extract-par-scan-exclusive-info form)]
                    (list 'raster.gpu.ze-runtime/invoke-registered-scan-exclusive-kernel
                          (:block-kernel-name k) (:prop-kernel-name k)
                          (vec (:input-arrays k)) out bound)))))

            ;; par/rng-fill!
            (par/par-rng-fill-form? form)
            (let [{:keys [seeds n base-seed]} (par/extract-par-rng-fill-info form)
                  k (register-kernel!
                     (legacy/generate-par-rng-fill-kernel :device-id device-id)
                     :ze-maps)]
              (list 'raster.gpu.ze-runtime/invoke-registered-rng-fill-kernel
                    (:kernel-name k) seeds n base-seed))

            ;; par/active-ids!
            (par/par-active-ids-form? form)
            (let [{:keys [ids n-active n-agents base-seed]} (par/extract-par-active-ids-info form)
                  k (register-kernel!
                     (legacy/generate-par-active-ids-kernel :device-id device-id)
                     :ze-maps)]
              (list 'raster.gpu.ze-runtime/invoke-registered-active-ids-kernel
                    (:kernel-name k) ids n-active n-agents base-seed))

            ;; par/stencil!
            (par/par-stencil-form? form)
            (let [{:keys [bound]} (par/extract-par-stencil-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-stencil! form))
                (let [kernel (legacy/generate-par-stencil-kernel form
                                                                 :dtype dtype :device-id device-id
                                                                 :scalar-types top-scalar-types)
                      k (register-kernel! kernel :ze-maps)
                      {:keys [out]} (par/extract-par-stencil-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-kernel
                        (:kernel-name k) (vec (:array-params k))
                        out (vec (:scalar-params k)) bound))))

            ;; par/scatter!
            (par/par-scatter-form? form)
            (let [{:keys [n]} (par/extract-par-scatter-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-scatter! form))
                (let [k (register-kernel!
                         (legacy/generate-par-scatter-kernel form
                                                             :dtype dtype :device-id device-id)
                         :ze-maps)
                      {:keys [out src index stride]} (par/extract-par-scatter-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-scatter-kernel
                        (:kernel-name k) out src index n (when stride stride)))))

            ;; par/gather — out[e*stride+d] = src[index[e]*stride+d]. Writes every
            ;; output element once (no atomics/accumulation), so it lowers to a map-void
            ;; kernel and binds through the existing resident map path.
            (par/par-gather-form? form)
            (let [{:keys [n]} (par/extract-par-gather-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-gather! form))
                (let [k (register-kernel!
                         (legacy/generate-par-gather-kernel form
                                                            :dtype dtype :device-id device-id)
                         :ze-maps)]
                  (emit-map-void-invocation k))))

            ;; par/reduce-by-key
            (par/par-reduce-by-key-form? form)
            (let [{:keys [n]} (par/extract-par-reduce-by-key-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-reduce-by-key form))
                (let [k (register-kernel!
                         (legacy/generate-par-reduce-by-key-kernel form
                                                                   :dtype dtype :device-id device-id)
                         :ze-reduces)
                      {:keys [out keys vals]} (par/extract-par-reduce-by-key-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-reduce-by-key-kernel
                        (:kernel-name k) out keys vals n))))

            ;; par/butterfly!
            (par/par-butterfly-form? form)
            (let [{:keys [half]} (par/extract-par-butterfly-info form)]
              (if (and (number? half) (< half min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-butterfly! form))
                ;; No dedicated GPU kernel yet — fall back to scalar expansion
                (do (swap! stats update :fallback inc)
                    (par/expand-par-butterfly! form))))

            ;; === Structural recursion (PRESERVE metadata) ===
            ;; A bare (apply list head ...) rebuild drops the form's metadata — in particular
            ;; :raster.op/original on a devirtualized .invk (e.g. blas/dgemm!), which downstream
            ;; GEMM recognition (parse-gpu-step, gpu_plan) reads. Re-attach the original meta.

            (and (seq? form) (contains? #{'let 'let*} (first form)))
            (let [[let-sym bindings & body-exprs] form
                  pairs (partition 2 bindings)
                  new-bindings (vec (mapcat (fn [[sym expr]]
                                              (let [scheduled
                                                    (when parallel-program
                                                      (parallel-program/kernel-graph-for-binding
                                                       parallel-program sym expr))]
                                                [sym (if scheduled
                                                       (emit-scheduled-graph! scheduled)
                                                       (binding [*bound-segops*
                                                                 (when parallel-program
                                                                   (parallel-program/operations-for-binding
                                                                    parallel-program sym expr))]
                                                         (transform expr)))]))
                                            pairs))
                  new-body (mapv (fn [expr]
                                   (binding [*bound-segops*
                                             (when parallel-program
                                               (parallel-program/operations-for-source
                                                parallel-program expr))]
                                     (transform expr)))
                                 body-exprs)]
              (with-meta (apply list let-sym new-bindings new-body) (meta form)))

            (and (seq? form) (= 'do (first form)))
            (with-meta (apply list 'do (mapv transform (rest form))) (meta form))

            (seq? form)
            (let [head (first form)
                  rebuilt (if (symbol? head)
                            (apply list head (mapv transform (rest form)))
                            (mapv transform form))]
              (if (and (instance? clojure.lang.IObj rebuilt) (meta form))
                (with-meta rebuilt (meta form))
                rebuilt))

            :else form))]
    {:form (transform source-form)
     :stats @stats
     :kernels @kernels
     :dispatches @dispatches}))

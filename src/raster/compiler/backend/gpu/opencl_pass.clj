(ns raster.compiler.backend.gpu.opencl-pass
  "Unified OpenCL pipeline pass.

   Walks S-expressions, replaces par forms with Level Zero kernel
   invocation markers. Uses SegOp-based codegen for generic map/reduce,
   delegates to par_opencl generators for specialized forms.

   This is the GPU counterpart of simd-pass — both consume the same
   par form vocabulary but produce different target code."
  (:require [raster.compiler.core.util :as util]
            [clojure.set :as set]
            [clojure.walk :as walk]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower-pass]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
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
(descriptor/register-op-descriptor!
 'raster.gpu.ocl-runtime/invoke-registered-kernel
 {:buffer {:allocates? false :in-place-arg 2}})

;; ================================================================
;; Single source of declared GPU param types (shared by BOTH compile entries:
;; raster.gpu.core/compile-deftm-internal! and the pipeline's pass-backend).
;; ================================================================

(def ^:private integral-arithmetic-heads
  "clojure.core integer index/dimension arithmetic: the fixpoint census exempts such bindings from
   type tags because their result is a long whenever every operand is a long."
  '#{clojure.core/* clojure.core/+ clojure.core/- clojure.core/quot clojure.core/rem
     clojure.core/mod clojure.core/inc clojure.core/dec clojure.core/max clojure.core/min})

(defn- strip-scalar-cast [x]
  (if (and (seq? x) (= 2 (count x))
           (contains? '#{long clojure.core/long int clojure.core/int} (first x)))
    (recur (second x))
    x))

(defn- integral-expression?
  "An expression whose value is provably an integral scalar from declared facts: an integer
   literal, an explicit integral cast, a symbol declared integral in `env`, or integer arithmetic
   over such operands."
  [expression env]
  (cond
    (integer? expression) true
    (symbol? expression) (contains? #{:int :long} (get env expression))
    (and (seq? expression) (= 2 (count expression))
         (contains? '#{long clojure.core/long int clojure.core/int} (first expression)))
    true
    (and (seq? expression) (contains? integral-arithmetic-heads (first expression)))
    (every? #(integral-expression? % env) (rest expression))
    :else false))

(defn- counted-loop-index-dtype
  "The integral dtype of a loop counter: a `dotimes` binder, or the `loop*` binder that the
   loop's own exit test compares against a bound. The fact is the loop's structure, not the
   binder's name; accumulators and other carries are left to their tags."
  [form binder init]
  (cond
    (contains? '#{dotimes clojure.core/dotimes} (first form)) :long
    (and (form/loop-head? (first form)) (integer? init))
    (let [body (nth form 2 nil)
          test (when (and (seq? body) (contains? '#{if clojure.core/if} (first body)))
                 (second body))]
      (when (and (seq? test)
                 (contains? '#{< <= clojure.core/< clojure.core/<=} (first test))
                 (= binder (strip-scalar-cast (second test))))
        :long))
    :else nil))

(defn binder-scalar-types
  "Walker-stamped scalar dtypes of every `let`/`loop`/`dotimes` binder in `form`, projected the
   way `derive-param-types` projects declared params: integral tags keep their width, floating
   tags take the kernel dtype. Untagged integer-arithmetic locals (which the fixpoint census
   exempts from tags) are typed from their operands' declared dtypes. A hoisted host scalar that
   a kernel captures therefore has the same declared dtype on the host and in the kernel ABI; no
   emitter guesses it from its name."
  [form kernel-dtype declared]
  (let [kernel-dtype (dtype/canon (or kernel-dtype :float))
        found (atom (or declared {}))]
    (walk/prewalk
     (fn [x]
       (when (and (seq? x)
                  (symbol? (first x))
                  (or (form/let-head? (first x)) (form/loop-head? (first x))
                      (contains? '#{dotimes clojure.core/dotimes} (first x)))
                  (vector? (second x)))
         (doseq [[binder init] (partition 2 (second x))]
           (when (symbol? binder)
             (when-let [inferred (or (some-> (types/sym-type-tag binder) dtype/dtype-for-scalar-tag)
                                     (counted-loop-index-dtype x binder init)
                                     (when (integral-expression? init @found) :long))]
               (swap! found assoc binder
                      (if (dtype/fp-dtype? inferred) kernel-dtype inferred))))))
       x)
     form)
    (apply dissoc @found (keys (or declared {})))))

(defn derive-param-types
  "Declared scalar + array element types for the GPU emitter, read from a deftm's params + tags
  (the typed-dispatch system already knows these — we read them instead of letting the emitter
  guess from parameter names). Scalar params: long/int→:int, floating scalars specialize to the
  selected kernel dtype. Array params: the tag's element dtype, with float-family (float/double)
  mapped to the KERNEL dtype (a single-precision kernel reads float buffers regardless of a
  parametric (All [T]) param's default).
  Returns {:scalar-types {sym kw} :array-types {sym kw}}, attached as form metadata that
  opencl-pass reads. ONE derivation, both compile paths."
  [params tags dtype]
  (when (and params tags)
    {:scalar-types (into {} (keep (fn [[p t]]
                                    (case (dtype/dtype-for-scalar-tag t)
                                      (:long :int) [p :int]
                                      (:double :float) [p dtype]
                                      nil))
                                  (map vector params tags)))
     :array-types (into {} (keep (fn [[p t]]
                                   (when-let [dt (dtype/dtype-for-array-tag t)]
                                     [p (if (#{:float :double} dt) dtype dt)]))
                                 (map vector params tags)))}))

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

(def ^:dynamic *bound-algorithm*
  "The validated TypedSOAC program associated with `*bound-segops*`. Target scheduling reads the
   algorithm rather than recovering semantic facts from the host expression being replaced."
  nil)

(defn- take-bound-segop
  "The precomputed SegOp of `kind` for the current binding, if segop-lower produced one."
  [stats kind pred]
  (when-let [so (first (filter pred *bound-segops*))]
    (swap! stats update :segop-reused (fnil inc 0))
    so))

(defn- source->segmap
  "The SegMap for a direct parallel source form: the one `segop-lower` already computed when
   available; otherwise re-lowered here with the REAL `device-id` (it was `:ze:0` hardcoded) and
   counted as `:segop-relowered`. SegMap is a full conversion: a missing rule is an illegal
   operation, never permission to select a second emitter."
  [stats result-id form dtype device-id scalar-types array-types]
  (or (take-bound-segop stats :segmap #(instance? raster.compiler.ir.segop.SegMap %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (or (segop-attempt stats :segmap form dtype :none
                             #(let [scheduled
                                    (segop-lower-pass/schedule-single-operation
                                     result-id form
                                     {:target-device (or device-id :ze:0)
                                      :dtype (or dtype :double)
                                      :scalar-types scalar-types
                                      :array-types array-types})]
                                (first (:operations scheduled))))
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
  [stats form dtype device-id scalar-types array-types]
  (or (take-bound-segop stats :segred #(instance? raster.compiler.ir.segop.SegRed %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (or (segop-attempt stats :segred form dtype :none
                             #(let [sym (gensym "red_")
                                    scheduled
                                    (segop-lower-pass/schedule-single-operation
                                     sym form {:target-device (or device-id :ze:0)
                                               :dtype (or dtype :double)
                                               :scalar-types scalar-types
                                               :array-types array-types})]
                                (first (:operations scheduled))))
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
  "Render the target-specific compatibility map marker from the emitter-authored ABI and ordered
   values."
  [{:keys [kernel-name abi arguments]} device-id]
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
      (list (if (and device-id (.startsWith (name device-id) "ocl"))
              'raster.gpu.ocl-runtime/invoke-registered-kernel
              'raster.gpu.ze-runtime/invoke-registered-kernel)
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

(defn- resident-single-reduction
  "Refine a scalar reduction to the explicit one-workgroup phase required by reduce-into.

   The output identity is changed here, at the semantic operation boundary, rather than passed as
   an emitter-only alias. One group folds the complete extent, so the caller-owned one-element
   result is the scheduled output rather than an undersized partial buffer."
  [operation output]
  (let [dtype (dtype/canon (:dtype operation))
        workgroup-size (get-in operation [:grid :block-size])
        grid (segop/->KernelGrid 1 workgroup-size
                                (* workgroup-size (dtype/bytes-of dtype)))
        operator (-> (:reduction operation)
                     (update :components
                             (fn [components]
                               (mapv #(assoc % :result output) components)))
                     (assoc-in [:attributes :physical-phase] :single))]
    (assoc operation
           :level (segop/->SegLevel :block :none)
           :reduction operator
           :lambda nil
           :outputs #{output}
           :grid grid
           :phase :single
           :schedule (segred-body/scalar-workgroup-tree-schedule
                      operator grid :single))))

(defn- emit-map-void-invocation
  "Render the staging marker from the artifact's logical argument projection. The marker remains
   useful to the host evaluator, but it no longer owns or reconstructs an argument convention."
  [artifact device-id]
  (let [plan (kcall/logical-argument-plan artifact)
        pointer-values (mapv :value (filterv :pointer? plan))
        scalar-entries (filterv (complement :pointer?) plan)
        bound-entries (filterv #(= :bound (:role (first (:slots %)))) scalar-entries)
        user-scalars (filterv #(not= :bound (:role (first (:slots %)))) scalar-entries)]
    (when-not (= 1 (count bound-entries))
      (throw (ex-info "map-void artifact must identify exactly one :bound scalar"
                      {:kernel-name (:kernel-name artifact) :plan plan})))
    (list (if (and device-id (.startsWith (name device-id) "ocl"))
            'raster.gpu.ocl-runtime/invoke-registered-map-void-kernel
            'raster.gpu.ze-runtime/invoke-registered-map-void-kernel)
          (:kernel-name artifact)
          pointer-values
          (mapv :value user-scalars)
          (:value (first bound-entries)))))

(defn opencl-pass
  "Pipeline pass: walk S-expression, replace par forms with GPU kernel invocations.

   Uses full SegOp conversion for par/map! and par/reduce.
   Certified effect-only map-void forms consume their scheduled TypedSOAC SegMap; unsupported
   bodies and the remaining stencil, scatter, active-id and key-reduction forms retain
   explicit compatibility generators.

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
  (let [supplied-program0 (when (parallel-program/parallel-program? form)
                            (parallel-program/validate! form segop/segop-node?))
        retained-operations (mapcat :operations (:equations supplied-program0))
        retained-buffer-ids
        (set (mapcat #(concat (segop/operation-inputs %)
                              (segop/operation-outputs %))
                     retained-operations))
        retained-equation-ids
        (set (mapcat #(concat (:operands %) (:results %)) (:equations supplied-program0)))
        ;; Host-only scalar SSA has no SegOp by construction. Classify every equation-boundary
        ;; value not named by a physical buffer role as scalar; this preserves compound-extent
        ;; leaves across compatibility re-entry while rank-zero resident buffers remain buffers.
        retained-scalar-ids
        (set/union (set (mapcat segop/operation-scalars retained-operations))
                   (set/difference retained-equation-ids retained-buffer-ids))
        retained-role-overlap (set/intersection retained-buffer-ids retained-scalar-ids)
        _ (when (seq retained-role-overlap)
            (throw (ex-info "retained schedule gives values contradictory ABI roles"
                            {:reason :parallel-program-parameter-role-conflict
                             :values retained-role-overlap})))
        retained-scalar-types
        (when supplied-program0
          (parallel-program/known-value-types supplied-program0 retained-scalar-ids))
        retained-array-types
        (when supplied-program0
          (parallel-program/known-value-types supplied-program0 retained-buffer-ids))
        scalar-type-conflicts
        (into {}
              (keep (fn [[id retained]]
                      (when-let [provided (get scalar-types id)]
                        (when-not (= (dtype/canon retained) (dtype/canon provided))
                          [id {:retained retained :provided provided}]))))
              retained-scalar-types)
        array-type-conflicts
        (into {}
              (keep (fn [[id retained]]
                      (when-let [provided (get array-types id)]
                        (when-not (= (dtype/canon retained) (dtype/canon provided))
                          [id {:retained retained :provided provided}]))))
              retained-array-types)
        _ (when (or (seq scalar-type-conflicts) (seq array-type-conflicts))
            (throw (ex-info "caller type facts contradict the retained scheduled program"
                            {:reason :parallel-program-type-conflict
                             :scalar-conflicts scalar-type-conflicts
                             :array-conflicts array-type-conflicts})))
        ;; A compatibility ParallelProgram predates the direct TypedSOAC value identities used by
        ;; graph storage. Re-enter once through its retained source instead of preserving a second
        ;; backend contract whose operations and SSA results can name different physical values.
        compatibility-program? (and supplied-program0
                                    (some? (:source supplied-program0))
                                    (not= :typed-soac
                                          (get-in supplied-program0
                                                  [:provenance :source-dialect])))
        supplied-program (when-not compatibility-program? supplied-program0)
        direct-mini-program?
        (and (nil? supplied-program)
             (or (par/par-rng-fill-form? form)
                 (and (par/par-gather-form? form)
                      (:stride (par/extract-par-gather-info form)))
                 (and (par/par-scatter-form? form)
                      (:stride (par/extract-par-scatter-info form)))
                 (par/par-reduce-form? form)
                 ;; A compound source extent normalizes to a preceding typed scalar equation. It
                 ;; cannot be projected as a closed SegMap without losing that SSA definition.
                 (and (par/par-map-form? form)
                      (not (soac-dialect/extent?
                            (:bound (par/extract-par-map-info form)))))))
        direct-schedule
        (cond
          compatibility-program?
          (segop-lower-pass/schedule-source-program
           (:source supplied-program0)
           {:target-device device-id :dtype dtype
            ;; A compatibility re-entry may need to rebuild structure, but it may never discard
            ;; types already retained by the prior middle-end boundary.
            :scalar-types (merge scalar-types retained-scalar-types)
            :array-types (merge array-types retained-array-types)})

          (and (nil? supplied-program) (form/binding-form? form))
          (segop-lower-pass/schedule-source-program
           form {:target-device device-id :dtype dtype
                 :scalar-types scalar-types :array-types array-types})

          direct-mini-program?
          (segop-lower-pass/schedule-single-program
           (gensym "direct_indexed_result_") form
           {:target-device device-id :dtype dtype
            :scalar-types scalar-types :array-types array-types}))
        direct-program (:program direct-schedule)
        parallel-program (or supplied-program direct-program)
        source-form (if parallel-program (parallel-program/source-form parallel-program) form)
        ;; Typed values supply dtypes; scheduled SegOps supply ABI roles. Logical rank cannot choose
        ;; pass-by-value versus buffer because a rank-zero result may be resident in either form.
        program-types
        (when parallel-program
          (let [operations (mapcat :operations (:equations parallel-program))
                arrays (set (mapcat #(concat (segop/operation-inputs %)
                                             (segop/operation-outputs %))
                                    operations))
                scalars (set (mapcat segop/operation-scalars operations))
                overlap (set/intersection arrays scalars)
                typed? (= :typed-soac (get-in parallel-program [:provenance :source-dialect]))]
            (when (and typed? (seq overlap))
              (throw (ex-info "scheduled values have contradictory buffer and scalar ABI roles"
                              {:reason :parallel-program-parameter-role-conflict
                               :values overlap})))
            ;; Every scheduled program carries declared value dtypes, whichever pass produced
            ;; it. A typed program must declare every ABI value; a compatibility program
            ;; contributes the dtypes it does declare, so a captured host scalar such as an
            ;; AD-generated loss scale keeps its walker-stamped dtype instead of falling to the
            ;; SegMap emitter's integer default (which silently zeroed such scalars).
            (if typed?
              {:scalar-types (parallel-program/declared-value-types parallel-program scalars)
               :array-types (parallel-program/declared-value-types parallel-program arrays)}
              {:scalar-types (parallel-program/known-value-types parallel-program scalars)
               :array-types (parallel-program/known-value-types parallel-program arrays)})))
        top-scalar-types (merge (binder-scalar-types
                                 source-form dtype
                                 (merge (or (:scalar-types program-types) {})
                                        (or (:scalar-types (meta source-form)) {})
                                        (or (:scalar-types (meta form)) {}) scalar-types))
                                (or (:scalar-types program-types) {})
                                (or (:scalar-types (meta source-form)) {})
                                (or (:scalar-types (meta form)) {}) scalar-types)
        top-array-types (merge (or (:array-types program-types) {})
                               (or (:array-types (meta source-form)) {})
                               (or (:array-types (meta form)) {}) array-types)
        stats (atom (cond-> {:ze-maps 0 :ze-reduces 0 :ze-compounds 0 :ze-contracts 0
                             :ze-structured-reductions 0 :kernel-graphs 0
                             :fallback 0}
                      direct-schedule
                      (assoc :direct-scheduling (:stats direct-schedule))))
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
        (fn emit-scheduled-graph!
          ([scheduled] (emit-scheduled-graph! scheduled nil false))
          ([scheduled stat-key host-scalar-result?]
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
            (when stat-key (swap! stats update stat-key inc))
            ;; Staging and resident extraction consume the same registered dispatch and complete
            ;; ordered executable ABI. The device id chooses only the runtime; it is not part of
            ;; the executable's semantic call interface.
            (let [invocation-arguments
                  (if host-scalar-result?
                    (let [result-indexes
                          (keep-indexed (fn [index slot]
                                          (when (= :result (:role slot)) index))
                                        (:abi emitted))
                          _ (when-not (= 1 (count result-indexes))
                              (throw (ex-info
                                      "host scalar graph requires exactly one result ABI slot"
                                      {:reason :scheduled-host-scalar-result
                                       :result-indexes (vec result-indexes)
                                       :abi (:abi emitted)})))
                          result-index (first result-indexes)
                          result-slot (nth (:abi emitted) result-index)
                          result-id (nth (:arguments emitted) result-index)
                          result-buffer (some #(when (= result-id (:id %)) %)
                                              (:outputs emitted))
                          constructor ({:float 'float-array :double 'double-array}
                                       (dtype/canon (:dtype result-slot)))]
                      (when-not (and result-buffer (= 1 (:elements result-buffer)) constructor)
                        (throw (ex-info
                                "scheduled host scalar result must be one FP32/FP64 graph element"
                                {:reason :scheduled-host-scalar-result
                                 :slot result-slot :buffer result-buffer})))
                      (assoc (vec (:arguments emitted)) result-index (list constructor 1)))
                    (vec (:arguments emitted)))
                  invocation
                  (list 'raster.compiler.pipeline/invoke-scheduled-executable!
                        device-id (:id dispatch) invocation-arguments)]
              (if host-scalar-result?
                (list 'clojure.core/aget invocation 0)
                invocation)))))

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
            (let [{:keys [bound]} (par/extract-par-map-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map! form))
                (let [segmap (source->segmap stats
                                             (:out (par/extract-par-map-info form)) form
                                             dtype device-id top-scalar-types top-array-types)
                      ;; Physical storage and effects come exclusively from the scheduled SegMap.
                      ;; A dense map has one logical result; an offset map is an explicit-store
                      ;; unique scatter and therefore uses the effect-only ABI/marker.
                      kernel (segop-cl/generate-scheduled-segmap-kernel
                              segmap
                              :dtype dtype :scalar-types top-scalar-types
                              :array-types (merge top-array-types (:array-types (meta form))))
                      k (register-kernel! kernel :ze-maps)
                      result-count (count (filter #(= :result (:role %)) (:abi k)))]
                  ;; The compatibility map marker models exactly one returned buffer. A fused
                  ;; multi-output map and an effect-only scatter are both already fully described
                  ;; by the artifact ABI, so use the general resident-effect marker for them.
                  (if (= 1 result-count)
                    (emit-map-invocation k device-id)
                    (emit-map-void-invocation k device-id)))))

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
                  ;; CONSUME the contraction-specialized SegRed recorded by typed lowering. Its
                  ;; algorithm remains attached to the ParallelProgram equation, so routing can
                  ;; derive verified facts without hiding them in a second operation record.
                  ;; SegContract remains only for the compatibility front door; without either
                  ;; scheduled operation (door C), route from the form and count it.
                  bound-sr (take-bound-segop
                            stats :segred-contraction
                            #(and (instance? raster.compiler.ir.segop.SegRed %)
                                  (= :contraction (:phase %))
                                  (= :hardware-contraction-candidates
                                     (get-in % [:schedule :strategy]))))
                  bound-sc (when-not bound-sr
                             (take-bound-segop
                              stats :segcontract
                              #(instance? raster.compiler.ir.segop.SegContract %)))
                  bound-operation (or bound-sr bound-sc)
                  typed-algorithm
                  (when bound-sr
                    (or *bound-algorithm*
                        (throw (ex-info
                                "typed contraction SegRed lacks its algorithm"
                                {:reason :typed-contraction-algorithm
                                 :operation (:id bound-sr)}))))
                  _ (when-not bound-operation
                      (swap! stats update :segop-relowered (fnil inc 0)))
                  target-desc (try ((requiring-resolve
                                     'raster.compiler.core.hardware/descriptor-for)
                                    device-id)
                                   (catch Throwable _ nil))
                  typed-dispatch
                  (when bound-sr
                    (try
                      ;; The scheduled SegRed carries its equation's dtype; a double contraction
                      ;; inside a float program routes at its own precision, not the program's.
                      (croute/route-typed-contraction-dispatch
                       typed-algorithm bound-sr
                       :dtype (:dtype bound-sr) :tile (:tile schedule) :desc target-desc
                       :precision (:precision schedule))
                      (catch clojure.lang.ExceptionInfo exception
                        (let [reason (:reason (ex-data exception))]
                          (if (contains? #{:typed-contraction-dispatch-invoke-protocol}
                                         reason)
                            (do
                              (swap! stats update-in [:typed-contraction-dispatch-declines reason]
                                     (fnil inc 0))
                              nil)
                            (throw exception))))))]
              (if typed-dispatch
                (let [measured-selector
                      (get-in schedule [:typed-contraction :measured-selectors
                                        (:id typed-dispatch)])
                      typed-dispatch
                      (if measured-selector
                        (-> (kdispatch/with-selector typed-dispatch measured-selector)
                            (assoc-in [:attributes :selection] :measured-fixed))
                        typed-dispatch)
                      dispatch
                      (-> typed-dispatch
                          (update :alternatives
                                  (fn [alternatives]
                                    (mapv
                                     (fn [graph]
                                       (kgraph/map-operations
                                        graph
                                        (fn [node]
                                          (maybe-compile-spirv (:operation node)
                                                               compile-spirv? device-id))))
                                     alternatives)))
                          kdispatch/validate!)
                      artifacts (mapv :operation (mapcat :nodes (:alternatives dispatch)))
                      executable (kdispatch/default-alternative dispatch)]
                  (swap! kernels into artifacts)
                  (swap! dispatches conj dispatch)
                  (swap! stats update :ze-contracts inc)
                  (swap! stats update :kernel-graphs inc)
                  (list 'raster.compiler.pipeline/invoke-scheduled-executable!
                        device-id (:id dispatch) (vec (:arguments executable))))
                (let [r (ensure-contraction-marker-expressible!
                         (if bound-sr
                           (croute/route-typed-contraction
                            typed-algorithm bound-sr
                            :dtype (:dtype bound-sr) :tile (:tile schedule) :desc target-desc
                            :precision (:precision schedule))
                           (croute/route-contraction
                            ;; A compatibility equation routes from its verified facts. Without a
                            ;; scheduled operation, the direct backend door still consumes the form.
                            (when-not bound-sc form) :dtype dtype
                            :facts (:facts bound-sc)
                            :operation-id (:id bound-sc)
                            ;; The resolved, feasibility-checked schedule is the single source of
                            ;; tile geometry. Previously this option reached opencl-pass but
                            ;; contractions ignored it and re-derived a default.
                            :tile (:tile schedule) :desc target-desc)))
                      ;; A non-dispatchable leaf retains its complete executable artifact and
                      ;; specialized invocation protocol.
                      kernel (:artifact r)
                      k (register-kernel! kernel :ze-contracts)]
                  (if (= :reduction (:invoke r))
                    (emit-reduction-invocation k nil)
                    (list 'raster.gpu.ze-runtime/invoke-registered-contraction!
                          (:kernel-name k)
                          (vec (:arguments r)))))))

            ;; === par/reduce-into — resident terminal SegRed writing a caller-owned scalar ===
            ;; Unlike the occupancy-capped partial phase used by host-scalar reduction, this is an
            ;; explicit one-group terminal refinement. The generic executable path preserves that
            ;; launch for staging and resident replay alike. Emitted by the reduce-fusion pass.
            (par/par-reduce-into-form? form)
            (let [{:keys [out-buf reduce-form]} (par/extract-par-reduce-into-info form)
                  segred (resident-single-reduction
                          (par->segred stats reduce-form dtype device-id
                                      top-scalar-types top-array-types)
                          out-buf)
                  kernel (segop-cl/generate-segred-kernel
                          segred out-buf :dtype dtype :scalar-types top-scalar-types
                          :array-types top-array-types)
                  k (register-kernel! kernel :ze-reduces)
                  dispatch (kdispatch/make
                            {:id (str "resident-single-reduction-"
                                      (Integer/toUnsignedString (hash k) 16))
                             :alternatives [k]
                             :default-strategy :scalar-workgroup-tree
                             :selector {:kind :fixed-strategy
                                        :strategy :scalar-workgroup-tree}
                             :provenance {:pass :opencl
                                          :source-dialect :segred
                                          :operation (:id segred)}
                             :attributes {:operation-family :scalar-reduction
                                          :resident-result true}})]
              (swap! dispatches conj dispatch)
              ;; Unlike the nil-result host-scalar marker, this path is executable both through
              ;; staging and resident extraction. Both consume the same KernelCall geometry.
              (list 'raster.compiler.pipeline/invoke-scheduled-executable!
                    device-id (:id dispatch) (vec (:arguments k))))

            ;; === par/reduce — SegOp path ===
            (par/par-reduce-form? form)
            (throw (ex-info
                    "scalar reduction must enter GPU emission as a complete TypedSOAC graph"
                    {:reason :scalar-reduction-requires-typed-graph
                     :source form :target-dialect :kernel-graph :fallback :none}))

            ;; Typed segmented product reduction. The portable schedule is one deterministic
            ;; workgroup tree per segment; mixed result components remain separate ABI buffers.
            (par/par-product-reduce-form? form)
            (let [segred (par->segred stats form dtype device-id
                                      top-scalar-types top-array-types)
                  kernel (segop-cl/generate-product-reduction-kernel
                          segred :scalar-types top-scalar-types :array-types top-array-types)
                  k (register-kernel! kernel :ze-reduces)]
              (emit-map-void-invocation k device-id))

            ;; Ordered segmented fold-map. The source spelling is only a host fallback; GPU
            ;; emission must consume the bound TypedSOAC SegFoldMap and its verified KernelBody.
            (par/par-segmented-fold-map-form? form)
            (if-let [scheduled (take-bound-segop
                                stats :segfoldmap
                                #(and (instance? raster.compiler.ir.segop.SegFoldMap %)
                                      (= :typed-soac (:algorithm-dialect %))))]
              (let [kernel (segop-cl/generate-segfoldmap-kernel
                            scheduled
                            :workgroup-size (or (get-in scheduled [:grid :block-size]) 256)
                            :scalar-types top-scalar-types
                            :array-types top-array-types)
                    k (register-kernel! kernel :ze-maps)]
                (emit-map-void-invocation k device-id))
              (throw (ex-info "GPU fold-map source has no verified TypedSOAC schedule"
                              {:reason :unscheduled-segmented-fold-map
                               :target-dialect :opencl :form form})))

            ;; === Specialized forms — delegate to legacy generators ===

            ;; par/map-void!
            (par/par-map-void-form? form)
            (let [{:keys [bound]} (par/extract-par-map-void-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map-void! form))
                (let [scheduled (take-bound-segop
                                 stats :segmap
                                 #(and (instance? raster.compiler.ir.segop.SegMap %)
                                       (= :typed-soac (:algorithm-dialect %))))
                      kernel (if scheduled
                               (segop-cl/generate-scheduled-segmap-kernel
                                scheduled
                                :dtype (:dtype scheduled)
                                :scalar-types top-scalar-types
                                :array-types top-array-types)
                               (legacy/generate-par-map-void-kernel
                                form :dtype dtype :device-id device-id
                                :array-types top-array-types
                                :scalar-types top-scalar-types))
                      k (register-kernel! kernel :ze-maps)]
                  (emit-map-void-invocation k device-id))))

            ;; par/scan-exclusive
            (par/par-scan-exclusive-form? form)
            (throw (ex-info "GPU exclusive scan must enter the backend as a scheduled TypedSOAC graph"
                            {:reason :exclusive-scan-requires-typed-schedule
                             :source form :target-dialect :kernel-graph
                             :fallback :none}))

            ;; par/rng-fill! is a source convenience for an ordinary typed map. This branch only
            ;; projects the scheduled artifact back into the direct backend's compatibility call.
            (par/par-rng-fill-form? form)
            (if-let [scheduled (take-bound-segop
                                stats :segmap
                                #(and (instance? raster.compiler.ir.segop.SegMap %)
                                      (= :typed-soac (:algorithm-dialect %))))]
              (let [kernel (segop-cl/generate-scheduled-segmap-kernel
                            scheduled :dtype (:dtype scheduled)
                            :scalar-types top-scalar-types :array-types top-array-types)
                    k (register-kernel! kernel :ze-maps)]
                (emit-map-invocation k device-id))
              (throw (ex-info "GPU RNG fill source has no verified TypedSOAC map schedule"
                              {:reason :unscheduled-rng-fill
                               :target-dialect :kernel-body :form form :fallback :none})))

            ;; par/active-ids!
            (par/par-active-ids-form? form)
            (let [{:keys [ids n-active n-agents base-seed]} (par/extract-par-active-ids-info form)
                  k (register-kernel!
                     (legacy/generate-par-active-ids-kernel)
                     :ze-maps)]
              (list 'raster.gpu.ze-runtime/invoke-registered-active-ids-kernel
                    (:kernel-name k) ids n-active n-agents base-seed))

            ;; par/stencil!
            (par/par-stencil-form? form)
            (if-let [scheduled (take-bound-segop
                                stats :segstencil
                                #(and (instance? raster.compiler.ir.segop.SegStencil %)
                                      (= :typed-soac (:algorithm-dialect %))))]
              (let [kernel (segop-cl/generate-segstencil-kernel-body
                            scheduled :scalar-types top-scalar-types
                            :array-types top-array-types)
                    k (register-kernel! kernel :ze-maps)]
                (emit-map-invocation k device-id))
              (throw (ex-info "GPU stencil source has no verified TypedSOAC schedule"
                              {:reason :unscheduled-stencil
                               :target-dialect :opencl
                               :form form})))

            ;; par/scatter!
            (par/par-scatter-form? form)
            (let [{:keys [out index n stride]} (par/extract-par-scatter-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-scatter! form))
                (if stride
                  ;; The typed route hoists n*stride into a scalar equation. Direct source callers
                  ;; cannot yet carry that equation beside one returned operation, so retain the
                  ;; explicit compatibility generator rather than emitting an unbound extent.
                  (let [k (register-kernel!
                           (legacy/generate-par-scatter-kernel
                            form :dtype dtype :device-id device-id)
                           :ze-maps)]
                    (list 'raster.gpu.ze-runtime/invoke-registered-scatter-kernel
                          (:kernel-name k) out (:src (par/extract-par-scatter-info form))
                          index n stride))
                  (let [array-types (assoc top-array-types index :int)
                        segmap (source->segmap stats (gensym "scatter_result_") form
                                              dtype device-id top-scalar-types array-types)
                        kernel (segop-cl/generate-scheduled-segmap-kernel
                                segmap :dtype dtype :scalar-types top-scalar-types
                                :array-types array-types)
                        k (register-kernel! kernel :ze-maps)]
                    (list 'do (emit-map-void-invocation k device-id) out)))))

            ;; par/gather — out[e*stride+d] = src[index[e]*stride+d]. Writes every
            ;; output element once (no atomics/accumulation), so it lowers to a map-void
            ;; kernel and binds through the existing resident map path.
            (par/par-gather-form? form)
            (let [{:keys [out index n stride]} (par/extract-par-gather-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-gather! form))
                (if stride
                  (let [k (register-kernel!
                           (legacy/generate-par-gather-kernel
                            form :dtype dtype :device-id device-id)
                           :ze-maps)]
                    (emit-map-void-invocation k device-id))
                  (let [array-types (assoc top-array-types index :int)
                        segmap (source->segmap stats (gensym "gather_result_") form
                                              dtype device-id top-scalar-types array-types)
                        kernel (segop-cl/generate-scheduled-segmap-kernel
                                segmap :dtype dtype :scalar-types top-scalar-types
                                :array-types array-types)
                        k (register-kernel! kernel :ze-maps)]
                    ;; A flat gather is an ordinary dense map with one stable indirect read.
                    (emit-map-invocation k device-id)))))

            ;; par/reduce-by-key
            (par/par-reduce-by-key-form? form)
            (let [{:keys [out keys n]} (par/extract-par-reduce-by-key-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-reduce-by-key form))
                (let [array-types (assoc top-array-types keys :int)
                      segmap (source->segmap stats (gensym "reduce_by_key_result_") form
                                            dtype device-id top-scalar-types array-types)
                      kernel (segop-cl/generate-scheduled-segmap-kernel
                              segmap :dtype dtype :scalar-types top-scalar-types
                              :array-types array-types)
                      k (register-kernel! kernel :ze-reduces)]
                  (list 'do (emit-map-void-invocation k device-id) out))))

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
            ;; A bare (apply list head ...) rebuild drops source and type metadata on devirtualized
            ;; calls. Re-attach it: compatibility admission and diagnostics must never infer a
            ;; different semantic operation merely because structural recursion rebuilt a list.

            (and (seq? form) (contains? #{'let 'let*} (first form)))
            (let [[let-sym bindings & body-exprs] form
                  pairs (partition 2 bindings)
                  new-bindings (vec (mapcat (fn [[sym expr]]
                                              (let [equation
                                                    (when parallel-program
                                                      (parallel-program/equation-for-binding
                                                       parallel-program sym expr))
                                                    scheduled
                                                    (let [typed-equation?
                                                          (and (:algorithm equation)
                                                               (= :typed-soac
                                                                  (get-in parallel-program
                                                                          [:provenance
                                                                           :source-dialect])))
                                                          scalar-reduction?
                                                          (and (seq (:operations equation))
                                                               (every?
                                                                #(and (instance?
                                                                       raster.compiler.ir.segop.SegRed %)
                                                                      (reduction/scalar?
                                                                       (:reduction %)))
                                                                (:operations equation)))]
                                                      (if (and typed-equation?
                                                               (or scalar-reduction?
                                                                   (get-in equation
                                                                           [:attributes
                                                                            :kernel-graph])))
                                                        (:graph
                                                         (equation-graph/make-for-equation
                                                          parallel-program equation))
                                                        (get-in equation
                                                                [:attributes :kernel-graph])))]
                                                [sym (if scheduled
                                                       (do
                                                         (when (and equation
                                                                    (every? #(instance?
                                                                              raster.compiler.ir.segop.SegRed %)
                                                                            (:operations equation)))
                                                           (swap! stats update :segop-reused
                                                                  (fnil inc 0)))
                                                         (emit-scheduled-graph!
                                                          scheduled
                                                          (when (and equation
                                                                     (every? #(instance?
                                                                               raster.compiler.ir.segop.SegRed %)
                                                                             (:operations equation)))
                                                            :ze-reduces)
                                                          (boolean
                                                           (and equation
                                                                (= [] (get-in parallel-program
                                                                              [:values
                                                                               (first
                                                                                (:results
                                                                                 equation))
                                                                               :shape]))))))
                                                       (binding [*bound-segops*
                                                                 (when parallel-program
                                                                   (parallel-program/operations-for-binding
                                                                    parallel-program sym expr))
                                                                 *bound-algorithm*
                                                                 (when parallel-program
                                                                   (parallel-program/algorithm-for-binding
                                                                    parallel-program sym expr))]
                                                         (transform expr)))]))
                                            pairs))
                  new-body (mapv (fn [expr]
                                   (let [equation
                                         (when parallel-program
                                           (parallel-program/equation-for-source
                                            parallel-program expr))
                                         ;; The compatibility discoverer gives value-producing
                                         ;; body expressions an internal result identity rather
                                         ;; than a source binding. Its exact algorithm/SegOps can
                                         ;; therefore be graph-validated directly even though the
                                         ;; enclosing program predates typed provenance.
                                         typed-equation?
                                         (and (:algorithm equation)
                                              (or (= :typed-soac
                                                     (get-in parallel-program
                                                             [:provenance :source-dialect]))
                                                  (= :body (first (:site equation)))))
                                         scalar-reduction?
                                         (and (seq (:operations equation))
                                              (every?
                                               #(and (instance?
                                                      raster.compiler.ir.segop.SegRed %)
                                                     (reduction/scalar?
                                                      (:reduction %)))
                                               (:operations equation)))]
                                     (if (and typed-equation? scalar-reduction?)
                                       (do
                                         (swap! stats update :segop-reused (fnil inc 0))
                                         (emit-scheduled-graph!
                                          (:graph
                                           (equation-graph/make-for-equation
                                            parallel-program equation))
                                          :ze-reduces
                                          (boolean
                                           (= [] (get-in parallel-program
                                                         [:values (first (:results equation))
                                                          :shape])))))
                                       (binding [*bound-segops* (:operations equation)
                                                 *bound-algorithm*
                                                 (when parallel-program
                                                   (parallel-program/algorithm-for-source
                                                    parallel-program expr))]
                                         (transform expr)))))
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
    ;; Every declared value id is a local of the compiled form, even one spelled like a
    ;; `clojure.core` name (a parameter called `seq`): free-symbol analysis inside the emitters
    ;; must read it as a kernel scalar, never as the core function.
    {:form (binding [util/*shadowing-locals*
                     (set (concat (keys top-scalar-types) (keys top-array-types)
                                  (keys (:values parallel-program))))]
             (transform source-form))
     :stats @stats
     :kernels @kernels
     :dispatches @dispatches}))

(ns raster.compiler.equation-first
  "Public, allocation-free compilation of a deftm through Raster's equation-first TypedSOAC path.

   `compile` retains the semantic program, selected schedule, emitted target program, and kernel
   artifacts as ordinary immutable compiler values. `lower` specializes the public invocation
   contract against ordered arguments and returns the sole physical composition boundary: a
   validated LinkPlan. Runtime allocation begins only in raster.gpu.link/instantiate!.

   This is the migration boundary for direct TypedSOAC programs. It does not consult the legacy
   resident descriptor extractor or reconstruct ABI/storage facts from emitted source."
  (:refer-clojure :exclude [compile])
  (:require [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.gpu.parallel-program-c-family :as program-c-family]
            [raster.compiler.backend.gpu.target :as gpu-target]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.kernel-dispatch :as kernel-dispatch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.schedule :as gpu-schedule]
            [raster.core :as rcore]))

(defrecord EquationFirstCompilation
           [id function target dtype source-ns options semantic scheduled emitted kernels stats])

(defn equation-first-compilation?
  [value]
  (instance? EquationFirstCompilation value))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :compiler :equation-first))))

(defn- source-namespace-symbol
  [f-var]
  (let [metadata (meta f-var)]
    (or (:raster.core/deftm-source-ns metadata)
        (some-> (:ns metadata) ns-name)
        (fail! :equation-first-source-namespace
               "equation-first compilation requires the deftm defining namespace"
               {:function f-var}))))

(defn- function-symbol
  [f-var]
  (let [{:keys [ns name]} (meta f-var)]
    (if (and ns name)
      (symbol (str (ns-name ns)) (str name))
      (fail! :equation-first-function "equation-first compilation requires a deftm Var"
             {:function f-var}))))

(defn- compiler-options
  [f-var target requested-dtype options]
  (let [metadata (meta f-var)
        parameters (pipeline/clean-params (pipeline/get-params f-var requested-dtype))
        declared-tags (:raster.core/deftm-tags metadata)
        effective-dtype (or requested-dtype (dtype/infer-dtype-from-tags declared-tags) :double)
        source-ns-symbol (source-namespace-symbol f-var)
        source-ns (or (find-ns source-ns-symbol)
                      (fail! :equation-first-source-namespace
                             "deftm defining namespace is not loaded"
                             {:function f-var :source-ns source-ns-symbol}))
        param-env (pipeline/build-param-env f-var effective-dtype)
        ;; Parametric deftm wrappers do not themselves retain dispatch tags; get-params and
        ;; build-param-env resolve the same dtype specialization used by get-walked-body.
        tags (mapv param-env parameters)
        parameter-types (opencl-pass/derive-param-types parameters tags effective-dtype)]
    ;; A device reduction consumed by a later equation is storage, not a host ABI scalar.
    ;; Realize the retained cross-equation use before scheduling/emission, as the resident
    ;; compiler entry does; no binder may repair this by downloading a device result.
    (merge {:resident-reductions? true}
           options
           {:dtype effective-dtype
            :segmented-plans? true
            :target-device target
            :active-params parameters
            :public-parameters parameters
            :source-ns source-ns
            :array-types (:array-types parameter-types)
            :scalar-types (:scalar-types parameter-types)}
           (when param-env {:param-env param-env}))))

(defn- target-source-dialect
  "Select source spelling from the same hardware descriptor used by scheduling.

   A missing optional descriptor still has a conservative family default: Level Zero is the
   explicit Intel path, while generic OpenCL uses portable subgroup spelling."
  [target backend descriptor]
  (or (try
        (some-> descriptor gpu-target/kernel-body-c-dialect)
        (catch Throwable _ nil))
      (case (device/device-type target)
        :ze :opencl-intel
        :ocl :opencl-portable
        :cuda :cuda
        :hip :hip
        (case backend
          :cuda :cuda
          :hip :hip
          nil))))

(defn ^:no-doc validate-target-description!
  "Check the identity of a captured compilation target before cache lookup or scheduling."
  [target descriptor]
  (when-not (= target (:device-id descriptor))
    (fail! :equation-first-target-description
           "captured target description does not match the compilation target"
           {:target target :descriptor-target (:device-id descriptor)}))
  descriptor)

(defn- dispatch-reduction-emissions
  "Join two independently emitted schedules over one retained semantic equation spine.

   This runs after TypedSOAC construction: it neither repeats source analysis nor recognizes
   a model operation from target code. Non-reduction equations must be identical."
  [function-id reference subgroup]
  (let [reference-program (emitted-program/validate! (:program reference))
        subgroup-program (emitted-program/validate! (:program subgroup))
        ref-equations (:equations reference-program)
        subgroup-equations (:equations subgroup-program)]
    (when-not (and (= (dissoc reference-program :equations)
                      (dissoc subgroup-program :equations))
                   (= (count ref-equations) (count subgroup-equations)))
      (fail! :equation-dispatch-program-spine
             "reduction alternatives changed the surrounding emitted program" {}))
    (let [equations
          (mapv
           (fn [reference-equation subgroup-equation]
             (if (swr/plan? (:algorithm reference-equation))
               (let [_ (when-not (= (dissoc reference-equation :operations)
                                    (dissoc subgroup-equation :operations))
                         (fail! :equation-dispatch-equation-spine
                                "reduction alternatives changed their semantic equation" {}))
                     candidates (mapv (comp first :operations)
                                      [reference-equation subgroup-equation])
                     selection (kernel-dispatch/make
                                {:id (str function-id "/reduction-" (:id reference-equation))
                                 :alternatives (mapv :graph candidates)
                                 :default-strategy :indexed-segmented-reduction-reference
                                 :selector {:kind :fixed-strategy
                                            :strategy :indexed-segmented-reduction-subgroup-score-reuse}})
                     operation (equation-dispatch/make
                                candidates selection
                                {:permitted-modes #{:exact :reassociated}})]
                 (assoc reference-equation :operations [operation]))
               (do
                 (when-not (= reference-equation subgroup-equation)
                   (fail! :equation-dispatch-other-equation
                          "reduction scheduling changed an unrelated emitted equation"
                          {:equation (:id reference-equation)}))
                 reference-equation)))
           ref-equations subgroup-equations)
          dispatch-count (count (filter (comp equation-dispatch/emitted-equation-dispatch?
                                              first :operations)
                                        equations))
          _ (when (zero? dispatch-count)
              (fail! :equation-dispatch-no-reduction
                     "reassociated reduction dispatch requires a segmented reduction" {}))
          program (emitted-program/validate!
                   (assoc reference-program :equations equations))]
      {:program program
       :kernels (vec (distinct (concat (:kernels reference) (:kernels subgroup))))
       :stats (assoc (:stats reference)
                     :reduction-dispatches dispatch-count
                     :alternative-emission (:stats subgroup))})))

(defn compile
  "Compile one deftm Var into an immutable equation-first target program.

   `:target` may select Level Zero/OpenCL, CUDA, or HIP. CUDA/HIP compilation is hardware-free:
   it produces verified target source artifacts but does not require an installed runtime or a
   physical device. Unsupported target families and uncovered KernelBody graph families fail
   rather than borrowing the compatibility backend. `:dtype` defaults from retained deftm tags.

   The three-argument form consumes an already captured target descriptor, allowing cache identity
   and compilation to share one immutable snapshot. Its device identity must match `:target`."
  ([f-var] (compile f-var {}))
  ([f-var options] (compile f-var options nil))
  ([f-var {:keys [target dtype] :or {target :ze:0} :as options} captured-target]
   (when-not (var? f-var)
     (fail! :equation-first-function "equation-first compilation requires a deftm Var"
            {:function f-var :actual (type f-var)}))
   (when (dispatch/host-only? f-var)
     (fail! :equation-first-host-only
            "equation-first GPU compilation was requested for an explicitly host-only deftm"
            {:function (function-symbol f-var) :target target}))
   (let [resolved-var (or (rcore/resolve-deftm-var f-var {:dtype dtype :ambiguity :throw}) f-var)
         _ (when (dispatch/host-only? resolved-var)
             (fail! :equation-first-host-only
                    "equation-first specialization resolved to an explicitly host-only method"
                    {:function (function-symbol resolved-var) :target target :dtype dtype}))
         target-descriptor (validate-target-description!
                            target (or captured-target (hardware/descriptor-for target)))
         resolved-schedule (gpu-schedule/compilation-schedule target-descriptor options)
         dispatch-reassociated?
         (= :dispatch-reassociated
            (get-in resolved-schedule [:segmented-weighted-reduction :strategy]))
         reference-schedule
         (if dispatch-reassociated?
           (assoc-in resolved-schedule [:segmented-weighted-reduction :strategy] :reference)
           resolved-schedule)
         compiler-options (compiler-options f-var target dtype
                                            (-> options
                                                (dissoc :target :gemm-precision)
                                                (assoc :schedule reference-schedule
                                                       :target-descriptor target-descriptor)))
         walked (pipeline/get-walked-body f-var (:dtype compiler-options))
         source (if (= 1 (count walked)) (first walked) (list* 'do walked))
         represented-source (pipeline/run-passes
                             source pipeline/gpu-resident-pre-soa-passes compiler-options)
         semantic-candidate (pipeline/run-passes
                             represented-source pipeline/gpu-semantic-post-soa-passes
                             compiler-options :write-read-fused)
         semantic (case (:dialect semantic-candidate)
                    :typed-parallel (if (get-in semantic-candidate [:attributes :invocation-plan])
                                      semantic-candidate
                                      (structured-route/promote-program
                                       semantic-candidate compiler-options))
                    :typed-soac (structured-route/promote-program
                                 semantic-candidate compiler-options)
                    semantic-candidate)
         _ (when-not (= :typed-parallel (:dialect semantic))
             (fail! :equation-first-coverage
                    "deftm is outside the direct TypedSOAC/structured-control vertical"
                    {:function (function-symbol f-var)
                     :dialect (:dialect semantic) :fallback :none}))
         scheduled (structured-route/schedule-program semantic compiler-options)
         backend (device/select-runtime-backend target true nil)
         target-dialect (target-source-dialect target backend target-descriptor)
         reference-emission
         (if target-dialect
           (program-c-family/emit-program
            scheduled (assoc compiler-options :target-dialect target-dialect))
           (fail! :equation-first-target-emitter
                  "equation-first scheduled program has no public source emitter for this target"
                  {:target target :backend backend :fallback :none}))
         emission
         (if dispatch-reassociated?
           (let [subgroup-options
                 (assoc-in compiler-options
                           [:schedule :segmented-weighted-reduction :strategy]
                           :subgroup-score-reuse)
                 subgroup-scheduled (structured-route/schedule-program
                                     semantic subgroup-options)
                 subgroup-emission (program-c-family/emit-program
                                    subgroup-scheduled
                                    (assoc subgroup-options :target-dialect target-dialect))]
             (dispatch-reduction-emissions
              (function-symbol f-var) reference-emission subgroup-emission))
           reference-emission)
         invocation-plan (some-> semantic :attributes :invocation-plan invocation/validate!)
         _ (when-not invocation-plan
             (fail! :equation-first-invocation
                    "equation-first semantic program has no retained public invocation plan"
                    {:function (function-symbol f-var)}))
         source-ns-symbol (source-namespace-symbol f-var)
         id [::compilation (function-symbol f-var) target (:dtype compiler-options)]]
     (->EquationFirstCompilation
      id (function-symbol f-var) target (:dtype compiler-options) source-ns-symbol
      (-> compiler-options
          (assoc :schedule resolved-schedule)
          (assoc :source-ns source-ns-symbol)
          (dissoc :values))
      semantic scheduled (:program emission) (:kernels emission)
      {:semantic {:dialect (:dialect semantic)
                  :equations (count (:equations semantic))}
       :schedule {:dialect (:dialect scheduled)
                  :equations (count (:equations scheduled))}
       :emission (:stats emission)
       :fallback :none}))))

(defn lower
  "Specialize a compiled equation-first program against ordered public arguments.

   Returns a validated, allocation-free LinkPlan. Public buffers retain their stable host source
   identity until instantiation; scalar prefix and host-only equations execute through the same
   typed JVM reference backend."
  [compilation arguments]
  (when-not (equation-first-compilation? compilation)
    (fail! :equation-first-compilation "lower requires an EquationFirstCompilation"
           {:actual (type compilation)}))
  (let [source-ns (:source-ns compilation)
        invocation-plan (get-in compilation [:semantic :attributes :invocation-plan])
        materialized
        (materialization/materialize
         invocation-plan (vec arguments)
         (partial scalar/evaluate-invocation-step source-ns))
        buffer-shapes (into {} (map (fn [[id buffer]] [id (:shape buffer)]))
                            (:program-buffers materialized))
        evaluate-host (fn [equation context]
                        (scalar/evaluate-host-equation
                         source-ns equation (assoc context :buffer-shapes buffer-shapes)))]
    (invocation-link/lower
     materialized (:emitted compilation) (:target compilation)
     evaluate-host)))

(defn compile-link-plan
  "Convenience composition of `compile` and `lower`; still performs no runtime allocation."
  [f-var arguments options]
  (lower (compile f-var options) arguments))

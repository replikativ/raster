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
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-dispatch :as kernel-dispatch]
            [raster.compiler.ir.kernel-executable :as kernel-executable]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-route :as swr-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.scalar.soa-lower :as soa-lower]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.schedule :as gpu-schedule]
            [raster.core :as rcore]))

(defrecord EquationFirstCompilation
           [id function target dtype source-ns options semantic scheduled emitted kernels stats])

(def ^:dynamic *lower-observer*
  "Internal timing observer for invocation materialization and LinkPlan construction."
  nil)

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

(defn ^:no-doc physical-function
  "Resolve only the numerical source specialization, preserving the caller's logical boundary."
  [f-var requested-dtype]
  (let [source (or (:raster.params/flat-var (meta f-var)) f-var)]
    (or (rcore/resolve-deftm-var source {:dtype requested-dtype :ambiguity :throw}) source)))

(defn ^:no-doc parameter-representation
  "Declared aggregate representation, shared by compilation and its cache identity."
  [f-var requested-dtype]
  (let [source-var (physical-function f-var requested-dtype)
        parameters (pipeline/clean-params (pipeline/get-params source-var requested-dtype))
        param-env (pipeline/build-param-env source-var requested-dtype)
        specs (mapv (fn [sym] {:sym sym :tag (get param-env sym)}) parameters)
        env (soa-lower/soa-param-env specs {:mixed-products? true})
        aggregate-projection (soa-lower/parameter-projection
                              specs env (the-ns (source-namespace-symbol source-var)))
        physical-by-binding (group-by :binding (:physical-parameters aggregate-projection))
        metadata (meta f-var)
        treedefs (:raster.params/treedefs metadata)
        original (:raster.params/original-args metadata)
        public (or (:raster.params/public-args metadata) original)
        tree-projection
        (when (seq treedefs)
          (let [labels (zipmap original public)]
            {:public-parameters public
             :trees (into {} (map (fn [[root td]] [(get labels root) (:spec td)])) treedefs)
             :physical-parameters
             (vec (mapcat (fn [root]
                            (if-let [td (get treedefs root)]
                              (mapv (fn [{:keys [sym path]}]
                                      {:symbol sym :binding (get labels root) :path path
                                       :tag (get param-env sym)}) (:leaves td))
                              (or (get physical-by-binding root)
                                  [{:symbol root :binding root}]))) original))}))]
    {:params specs :soa-env env
     :projection (or tree-projection aggregate-projection)}))

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
        parameter-types (opencl-pass/derive-param-types parameters tags effective-dtype options)]
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
            :return-tag (:raster.core/return-tag metadata)
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
   a model operation from target code. Non-reduction semantic boundaries must be identical;
   their independently emitted local SSA is discarded in favor of the reference emission."
  [function-id schedule target-descriptor reference subgroup]
  (let [reference-program (emitted-program/validate! (:program reference))
        subgroup-program (emitted-program/validate! (:program subgroup))
        ref-equations (:equations reference-program)
        subgroup-equations (:equations subgroup-program)
        measured-selectors (get-in schedule
                                   [:segmented-weighted-reduction :measured-selectors] {})]
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
                     plan (:algorithm reference-equation)
                     candidates (mapv (comp first :operations)
                                      [reference-equation subgroup-equation])
                     id (str function-id "/reduction-" (:id reference-equation))
                     reference-strategy :indexed-segmented-reduction-reference
                     subgroup-strategy :indexed-segmented-reduction-subgroup-score-reuse
                     subgroup-size (long (:subgroup-size target-descriptor))
                     multiple (long (get-in schedule
                                            [:segmented-weighted-reduction
                                             :score-reuse-subgroup-multiple]))
                     measured-selector (get measured-selectors id)
                     selection (cond->
                                (kernel-dispatch/make
                                 {:id id
                                  :alternatives (mapv :graph candidates)
                                  :default-strategy reference-strategy
                                  :selector {:kind :runtime-scalar-threshold
                                             :argument (get-in plan [:value :components])
                                             :threshold (*' subgroup-size multiple)
                                             :at-least subgroup-strategy
                                             :otherwise reference-strategy}
                                  :attributes
                                  {:algebra :segmented-weighted-reduction
                                   :algebra-key (swr/algebra-key plan)
                                   :tuning (update (swr-route/tuning-contract plan id)
                                                   :numerical-mode assoc
                                                   :permitted-modes #{:exact :reassociated})
                                   :selection (if measured-selector
                                                :measured-runtime-shape
                                                :analytic-runtime-shape)}})
                                 measured-selector
                                 (kernel-dispatch/with-selector measured-selector))
                     operation (equation-dispatch/make
                                candidates selection
                                {:permitted-modes #{:exact :reassociated}})]
                 (assoc reference-equation :operations [operation]))
               (do
                 (when-not (= (dissoc reference-equation :operations)
                              (dissoc subgroup-equation :operations))
                   (fail! :equation-dispatch-other-equation
                          "reduction scheduling changed an unrelated semantic equation"
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
                   (assoc reference-program :equations equations))
          kernels (vec
                   (mapcat (fn [equation]
                             (let [operation (first (:operations equation))]
                               (mapcat #(map :operation (get-in % [:graph :nodes]))
                                       (cond
                                         (equation-dispatch/emitted-equation-dispatch? operation)
                                         (:alternatives operation)
                                         operation [operation]
                                         :else []))))
                           equations))]
      {:program program
       ;; Enumerate retained alternatives, not discarded independent emissions. Unrelated
       ;; schedules may generate fresh local SSA/kernel names without changing their semantic
       ;; boundary; neither those unused artifacts nor their private bindings belong here.
       :kernels kernels
       :stats (assoc (:stats reference)
                     :emission-routes (frequencies (map kernel-artifact/emission-route kernels))
                     :reduction-dispatches dispatch-count
                     :alternative-emission (:stats subgroup))})))

(defn- dispatch-contraction-emissions
  "Admit explicitly authorized contraction candidates without rescheduling unrelated equations."
  [function-id reference options]
  (let [mixed? (= :dispatch-mixed-matrix
                   (get-in options [:schedule :typed-contraction :strategy]))
        declines (atom {})
        measured-selectors (get-in options [:schedule :typed-contraction :measured-selectors] {})
        consumed (atom #{})
        admitted (atom 0)
        new-kernels (atom [])
        equations
        (mapv
         (fn [equation]
           (let [operation (first (:operations equation))]
             (if-not (emitted-equation/emitted-equation? operation)
               equation
               (let [planned ((if mixed?
                                program-c-family/emit-mixed-contraction-alternative
                                program-c-family/emit-register-contraction-alternative)
                              operation options)]
                 (if-not (:ok planned)
                   (do
                     (swap! declines update-in
                            [(if (= :not-single-plain-fp32-contraction (:reason planned))
                               :screen :admission)
                             (:reason planned)]
                            (fnil inc 0))
                     equation)
                   (let [portable (assoc-in operation [:graph :attributes :strategy]
                                            :sequential-segments)
                         candidate (:candidate planned)
                         alternatives [portable candidate]
                         candidate-strategy (kernel-dispatch/alternative-strategy (:graph candidate))
                         numerical-policy (if mixed?
                                            {:permitted-modes #{:exact :approximate-model}
                                             :permitted-models [(:numerical-model planned)]}
                                            {:permitted-modes #{:exact :reassociated}})
                         id (str function-id "/contraction-" (:id equation))
                         interface (mapv #(select-keys % [:kind :dtype :kernel-dtype :role
                                                         :aliasing :alignment])
                                         (kernel-executable/abi (:graph portable)))
                         measured-selector (get measured-selectors id)
                         _ (when (= :none (:fallback measured-selector))
                             (fail! :equation-first-contraction-pinning-unsupported
                                    "public contraction dispatch cannot yet prune pinned alternatives"
                                    {:dispatch-id id :selector measured-selector}))
                         selection
                         (cond->
                          (kernel-dispatch/make
                           {:id id
                            :alternatives (mapv :graph alternatives)
                            :default-strategy :sequential-segments
                            :selector {:kind :fixed-strategy :strategy candidate-strategy}
                            :attributes
                            {:selection (if measured-selector
                                          :supplied-selector
                                          (if mixed? :explicit-mixed-candidate :explicit-register-candidate))
                             :numerical-mode (if mixed? :approximate-model :reassociated)
                             :tuning {:schedule-path [:typed-contraction :measured-selectors]
                                      :schedule-key id
                                      :numerical-mode
                                      (cond-> (assoc numerical-policy :precision
                                                     (if mixed? :mixed-f16-f32 :f32))
                                        (not mixed?)
                                        (assoc :multiply-add
                                               (get-in options [:schedule :typed-contraction :multiply-add]
                                                       :decomposed)))
                                      :layout {:external-interface interface}}}})
                           measured-selector (kernel-dispatch/with-selector measured-selector))
                         certified (equation-dispatch/make
                                    alternatives selection
                                    numerical-policy)]
                     (swap! admitted inc)
                     (swap! new-kernels into (map :operation (get-in candidate [:graph :nodes])))
                     (when measured-selector (swap! consumed conj id))
                     (assoc equation :operations [certified])))))))
         (get-in reference [:program :equations]))
        _ (when-let [unconsumed (seq (remove @consumed (keys measured-selectors)))]
            (fail! :equation-first-contraction-selector-unconsumed
                   "measured contraction selectors do not name admitted public dispatches"
                   {:dispatch-ids (vec (sort unconsumed)) :consumed (vec (sort @consumed))
                    :declines @declines}))
        kernels (into (:kernels reference) @new-kernels)]
    {:program (emitted-program/validate! (assoc (:program reference) :equations equations))
     :kernels kernels
     :stats (assoc (:stats reference)
                   :emission-routes (frequencies (map kernel-artifact/emission-route kernels))
                   :contraction-dispatches @admitted
                   :contraction-candidate-declines (get @declines :admission {})
                   :contraction-screen-declines (get @declines :screen {}))}))

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
   (let [resolved-var (physical-function f-var dtype)
         _ (when (dispatch/host-only? resolved-var)
             (fail! :equation-first-host-only
                    "equation-first specialization resolved to an explicitly host-only method"
                    {:function (function-symbol resolved-var) :target target :dtype dtype}))
         target-descriptor (validate-target-description!
                            target (or captured-target (hardware/descriptor-for target)))
         resolved-schedule (gpu-schedule/compilation-schedule target-descriptor options)
         _ (when (and (= :dispatch-mixed-matrix
                         (get-in resolved-schedule [:typed-contraction :strategy]))
                      (not= :mixed-f16-f32 (get-in options [:schedule :precision])))
             (fail! :equation-first-matrix-consent
                    "mixed matrix dispatch requires an explicit :schedule :precision :mixed-f16-f32"
                    {:function (function-symbol resolved-var) :fallback :none}))
         _ (when (and (= :dispatch-mixed-matrix
                         (get-in resolved-schedule [:typed-contraction :strategy]))
                      (or (not= :default (get-in resolved-schedule [:typed-contraction :matrix-tiles]))
                          (seq (get-in resolved-schedule [:typed-contraction :split-factors]))))
             (fail! :equation-first-matrix-candidate-space
                    "public mixed matrix dispatch currently admits one full-K tile, not a tile space or split-K"
                    {:function (function-symbol resolved-var) :fallback :none}))
         _ (when (and (seq (get-in resolved-schedule [:typed-contraction :measured-selectors]))
                      (not (contains? #{:dispatch-register-tiled :dispatch-mixed-matrix}
                                      (get-in resolved-schedule [:typed-contraction :strategy]))))
             (fail! :equation-first-contraction-selector-unsupported
                    "equation-first compilation cannot yet consume measured contraction selectors"
                    {:function (function-symbol resolved-var) :target target
                     :schedule-path [:typed-contraction :measured-selectors]
                     :fallback :none}))
         dispatch-reassociated?
         (= :dispatch-reassociated
            (get-in resolved-schedule [:segmented-weighted-reduction :strategy]))
         dispatch-contractions?
         (contains? #{:dispatch-register-tiled :dispatch-mixed-matrix}
                    (get-in resolved-schedule [:typed-contraction :strategy]))
         reference-schedule
         (cond-> resolved-schedule
           dispatch-reassociated?
           (assoc-in [:segmented-weighted-reduction :strategy] :reference)
           dispatch-contractions?
           (assoc-in [:typed-contraction :strategy] :portable)
           dispatch-contractions?
           (assoc-in [:typed-contraction :measured-selectors] {}))
         compiler-options (compiler-options resolved-var target dtype
                                            (-> options
                                                (dissoc :target :gemm-precision)
                                                (assoc :schedule reference-schedule
                                                       :target-descriptor target-descriptor)))
         walked (pipeline/get-walked-body resolved-var (:dtype compiler-options))
         source (if (= 1 (count walked)) (first walked) (list* 'do walked))
         represented-source (pipeline/run-passes
                             source pipeline/gpu-resident-pre-soa-passes compiler-options)
         representation (parameter-representation f-var (:dtype compiler-options))
         param-specs (:params representation)
         represented (soa-lower/soa-lower represented-source param-specs (:soa-env representation))
         projection (:projection representation)
         physical-parameters (mapv :sym (:params represented))
         physical-types (when projection
                          (opencl-pass/derive-param-types physical-parameters
                                                         (mapv :tag (:params represented))
                                                         (:dtype compiler-options) compiler-options))
         compiler-options (cond-> compiler-options
                            projection (assoc :active-params physical-parameters
                                              :public-parameters physical-parameters
                                              :array-types (:array-types physical-types)
                                              :scalar-types (:scalar-types physical-types)))
         semantic-candidate (pipeline/run-passes
                             (:body represented) pipeline/gpu-semantic-post-soa-passes
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
         semantic (cond-> semantic
                    projection (assoc-in [:attributes :invocation-plan :attributes :parameter-projection]
                                         projection))
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
              (function-symbol f-var) resolved-schedule target-descriptor
              reference-emission subgroup-emission))
           reference-emission)
         emission
         (if dispatch-contractions?
           (dispatch-contraction-emissions
            (function-symbol f-var) emission
            (assoc compiler-options :target-dialect target-dialect
                   :schedule resolved-schedule))
           emission)
         invocation-plan (some-> semantic :attributes :invocation-plan invocation/validate!)
         _ (when-not invocation-plan
             (fail! :equation-first-invocation
                    "equation-first semantic program has no retained public invocation plan"
                    {:function (function-symbol f-var)}))
         source-ns-symbol (source-namespace-symbol f-var)
         id (cond-> [::compilation (function-symbol f-var) target (:dtype compiler-options)]
              (:preserve-declared-array-storage? compiler-options)
              (conj {:preserve-declared-array-storage? true}))]
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

(defn- lower-in-context
  "Specialize a compiled equation-first program against ordered public arguments.

   Returns a validated, allocation-free LinkPlan. Public buffers retain their stable host source
   identity until instantiation; scalar prefix and host-only equations execute through the same
   typed JVM reference backend.
   Retained-validation is an internal capability path: it must be exact-owner
   static evidence. Invocation materialization and final LinkPlan validation remain fresh."
  [compilation arguments project retained-validation caller-options]
  (when-not (equation-first-compilation? compilation)
    (fail! :equation-first-compilation "lower requires an EquationFirstCompilation"
           {:actual (type compilation)}))
  (let [started (System/nanoTime)
        source-ns (:source-ns compilation)
        invocation-plan (get-in compilation [:semantic :attributes :invocation-plan])
        materialized
        (materialization/materialize
         invocation-plan (vec arguments)
         (partial scalar/evaluate-invocation-step source-ns))
        materialization-ns (- (System/nanoTime) started)
        buffer-shapes (into {} (map (fn [[id buffer]] [id (:shape buffer)]))
                            (:program-buffers materialized))
        evaluate-host (fn [equation context]
                        (scalar/evaluate-host-equation
                         source-ns equation (assoc context :buffer-shapes buffer-shapes)))
        construction-started (System/nanoTime)
        projection-ns (volatile! 0)
        project (fn [plan]
                (let [started (System/nanoTime)
                      projected (project plan)]
                  (vswap! projection-ns + (- (System/nanoTime) started))
                  projected))
        result (if (nil? caller-options)
                 (invocation-link/lower
                  materialized (:emitted compilation) (:target compilation)
                  evaluate-host project retained-validation)
                 (invocation-link/lower
                  materialized (:emitted compilation) (:target compilation)
                  evaluate-host project retained-validation caller-options))]
    (when *lower-observer*
      (*lower-observer* {:materialization-ns materialization-ns
                         :link-plan-construction-ns
                         (- (System/nanoTime) construction-started @projection-ns)}))
    result))

(defn- lower-for-request
  [compilation arguments project retained-validation caller-options]
  (link-plan/without-validation-context
   #(lower-in-context compilation arguments project retained-validation caller-options)))

(defn lower
  "Specialize under independently supplied caller math intent, not compilation metadata.

   Legacy arities retain default intent. Retained static evidence must belong to this exact
   emitted program and request; materialization and final LinkPlan obligations remain fresh."
  ([compilation arguments]
   (:plan (lower-for-request compilation arguments (fn [plan] {:plan plan}) nil nil)))
  ([compilation arguments project]
   (lower-for-request compilation arguments project nil nil))
  ([compilation arguments project retained-validation]
   (lower-for-request compilation arguments project retained-validation nil))
  ([compilation arguments project retained-validation caller-options]
   (lower-for-request compilation arguments project retained-validation caller-options)))

(defn compile-link-plan
  "Compose `compile` and `lower` without runtime allocation, preserving explicit math intent."
  [f-var arguments options]
  (let [compilation (compile f-var options)]
    (if (contains? options :scalar-math)
      (:plan (lower compilation arguments (fn [plan] {:plan plan}) nil options))
      (lower compilation arguments))))

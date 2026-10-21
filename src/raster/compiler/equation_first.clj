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
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.stage-result :as stage-result]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-dispatch :as kernel-dispatch]
            [raster.compiler.ir.kernel-executable :as kernel-executable]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.passes.parallel.segmented-weighted-reduction-route :as swr-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-frontend :as typed-frontend]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
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
        frontend (pipeline/deftm-frontend-facts source-var requested-dtype)
        param-env (:param-env frontend)
        specs (:param-specs frontend)
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
  (when (contains? options :resident-outputs)
    (fail! :equation-first-resident-outputs
           "explicit resident outputs are available only for source-free TypedSOAC compilation"
           {:function (function-symbol f-var)}))
  (let [frontend (pipeline/deftm-frontend-facts f-var requested-dtype)
        parameters (:parameters frontend)
        effective-dtype (:effective-dtype frontend)
        source-ns-symbol (source-namespace-symbol f-var)
        source-ns (or (find-ns source-ns-symbol)
                      (fail! :equation-first-source-namespace
                             "deftm defining namespace is not loaded"
                             {:function f-var :source-ns source-ns-symbol}))
        param-env (:param-env frontend)
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
            :terminal-host-results? true
            :segmented-plans? true
            :target-device target
            :active-params parameters
            :public-parameters parameters
            :source-ns source-ns
            :source-policy (:source-policy frontend)
            :return-tag (:return-tag frontend)
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

(defn- validate-emitted-program-for-request
  [program caller-options]
  (if (contains? caller-options :scalar-math)
    (emitted-program/validate! program (select-keys caller-options [:scalar-math]))
    (emitted-program/validate! program)))

(defn- make-equation-dispatch-for-request
  [candidates selection numerical-policy caller-options]
  (if (contains? caller-options :scalar-math)
    (equation-dispatch/make candidates selection numerical-policy
                            (select-keys caller-options [:scalar-math]))
    (equation-dispatch/make candidates selection numerical-policy)))

(defn- dispatch-reduction-emissions
  "Join two independently emitted schedules over one retained semantic equation spine.

   This runs after TypedSOAC construction: it neither repeats source analysis nor recognizes
   a model operation from target code. Non-reduction semantic boundaries must be identical;
   their independently emitted local SSA is discarded in favor of the reference emission."
  [function-id schedule target-descriptor reference subgroup caller-options]
  (let [reference-program (validate-emitted-program-for-request (:program reference) caller-options)
        subgroup-program (validate-emitted-program-for-request (:program subgroup) caller-options)
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
                     operation (make-equation-dispatch-for-request
                                candidates selection
                                {:permitted-modes #{:exact :reassociated}} caller-options)]
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
          program (validate-emitted-program-for-request
                   (assoc reference-program :equations equations) caller-options)
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
                         certified (make-equation-dispatch-for-request
                                    alternatives selection
                                    numerical-policy options)]
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
    {:program (validate-emitted-program-for-request
               (assoc (:program reference) :equations equations) options)
     :kernels kernels
     :stats (assoc (:stats reference)
                   :emission-routes (frequencies (map kernel-artifact/emission-route kernels))
                   :contraction-dispatches @admitted
                   :contraction-candidate-declines (get @declines :admission {})
                   :contraction-screen-declines (get @declines :screen {}))}))

(defn- compilation-policy
  "Resolve one immutable target/schedule policy shared by source and source-free compilation."
  [function-id target options captured-target]
  (let [target-descriptor (validate-target-description!
                           target (or captured-target (hardware/descriptor-for target)))
        resolved-schedule (gpu-schedule/compilation-schedule target-descriptor options)
        _ (when (and (= :dispatch-mixed-matrix
                        (get-in resolved-schedule [:typed-contraction :strategy]))
                     (not= :mixed-f16-f32 (get-in options [:schedule :precision])))
            (fail! :equation-first-matrix-consent
                   "mixed matrix dispatch requires an explicit :schedule :precision :mixed-f16-f32"
                   {:function function-id :fallback :none}))
        _ (when (and (= :dispatch-mixed-matrix
                        (get-in resolved-schedule [:typed-contraction :strategy]))
                     (or (not= :default
                               (get-in resolved-schedule [:typed-contraction :matrix-tiles]))
                         (seq (get-in resolved-schedule
                                      [:typed-contraction :split-factors]))))
            (fail! :equation-first-matrix-candidate-space
                   "public mixed matrix dispatch currently admits one full-K tile, not a tile space or split-K"
                   {:function function-id :fallback :none}))
        _ (when (and (seq (get-in resolved-schedule
                                  [:typed-contraction :measured-selectors]))
                     (not (contains? #{:dispatch-register-tiled :dispatch-mixed-matrix}
                                     (get-in resolved-schedule
                                             [:typed-contraction :strategy]))))
            (fail! :equation-first-contraction-selector-unsupported
                   "equation-first compilation cannot yet consume measured contraction selectors"
                   {:function function-id :target target
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
          (assoc-in [:typed-contraction :measured-selectors] {}))]
    {:target-descriptor target-descriptor
     :resolved-schedule resolved-schedule
     :reference-schedule reference-schedule
     :dispatch-reassociated? dispatch-reassociated?
     :dispatch-contractions? dispatch-contractions?}))

(defn- compile-semantic
  "Schedule and emit one exact typed semantic program without source or ABI reconstruction."
  [{:keys [id function-id target dtype source-ns source-var retained-frontend options compiler-options]
    :as context}
   semantic]
  (when-not (= :typed-parallel (:dialect semantic))
    (fail! :equation-first-coverage
           "program is outside the direct TypedSOAC/structured-control vertical"
           {:function function-id :dialect (:dialect semantic) :fallback :none}))
  (let [terminal (get-in semantic [:attributes :invocation-plan :attributes :terminal-host-result])
        terminal-owner (when terminal (source-owner/compile-terminal-result semantic terminal source-var
                                                                           retained-frontend))
        {:keys [target-descriptor resolved-schedule
                dispatch-reassociated? dispatch-contractions?]} context
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
                subgroup-scheduled (structured-route/schedule-program semantic subgroup-options)
                subgroup-emission (program-c-family/emit-program
                                   subgroup-scheduled
                                   (assoc subgroup-options :target-dialect target-dialect))]
            (dispatch-reduction-emissions
             function-id resolved-schedule target-descriptor
             reference-emission subgroup-emission options))
          reference-emission)
        emission
        (if dispatch-contractions?
          (dispatch-contraction-emissions
           function-id emission
           (assoc compiler-options :target-dialect target-dialect
                  :schedule resolved-schedule))
          emission)
        invocation-plan (some-> semantic :attributes :invocation-plan invocation/validate!)
        ordered-owner (when (and source-var retained-frontend
                                 (source-owner/ordered-source-required? semantic (:program emission)))
                        (source-owner/compile-ordered-source
                         semantic (:program emission) source-var retained-frontend
                         (boolean (stage-result/requires-observation? (:program emission)))))
        eager-owner (when (and source-var
                              (source-owner/eager-evaluation-required? semantic (:program emission)))
                      (source-owner/compile-eager-evaluation
                       semantic (:program emission) source-var retained-frontend ordered-owner))]
    (when-not invocation-plan
      (fail! :equation-first-invocation
             "equation-first semantic program has no retained public invocation plan"
             {:function function-id}))
    (cond-> (->EquationFirstCompilation
     id function-id target dtype source-ns
     (cond-> (-> compiler-options
                 (assoc :schedule resolved-schedule)
                 (dissoc :values))
       source-ns (assoc :source-ns source-ns)
       (contains? options :resident-outputs)
       (assoc :resident-outputs (:resident-outputs options)))
     semantic scheduled (:program emission) (:kernels emission)
     {:semantic {:dialect (:dialect semantic)
                 :equations (count (:equations semantic))}
      :schedule {:dialect (:dialect scheduled)
                 :equations (count (:equations scheduled))}
      :emission (:stats emission)
      :fallback :none})
      terminal-owner (assoc :terminal-source-owner terminal-owner)
      eager-owner (assoc :eager-source-owner eager-owner)
      ordered-owner (assoc :ordered-source-owner ordered-owner))))

(def ^:private typed-compilation-options
  #{:target :schedule :scalar-math :resident-outputs})

(defn- validate-resident-output-demand!
  [typed-program resident-outputs]
  (when-not (vector? resident-outputs)
    (fail! :equation-first-resident-output-demand
           "resident-outputs must be an ordered vector of public TypedSOAC output identities"
           {:resident-outputs resident-outputs}))
  (let [outputs (soac/outputs typed-program)
        output-set (set outputs)]
    (when-not (and (= (count resident-outputs) (count (distinct resident-outputs)))
                   (every? soac/value-id? resident-outputs)
                   (every? output-set resident-outputs))
      (fail! :equation-first-resident-output-demand
             "resident-outputs must contain distinct exact public TypedSOAC outputs"
             {:resident-outputs resident-outputs :outputs outputs})))
  resident-outputs)

(defn- validate-resident-output-compilation!
  [compilation resident-outputs]
  (doseq [result resident-outputs]
    (let [scheduled-value
          (structured-route/validated-public-resident-output (:scheduled compilation) result)
          emitted-value (get-in compilation [:emitted :values result])]
      (when-not (and scheduled-value
                     (some #{result} (get-in compilation [:emitted :outputs]))
                     (semantic/equivalent? scheduled-value emitted-value))
        (fail! :equation-first-resident-output-representation
               "requested resident output did not retain its certified emitted representation"
               {:result result :target (:target compilation)}))))
  compilation)

(defn- scalar-region-local-dtypes
  "Collect declared locals only through validated TypedSOAC region structure."
  [lambda]
  (letfn [(expression-dtypes [expression]
            (cond
              (soac/scalar-convert-form? expression)
              (expression-dtypes (:operand (soac/scalar-convert-parts expression)))

              (soac/while-component-form? expression)
              (let [{:keys [condition update]} (soac/while-fold-parts (second expression))]
                (into (lambda-dtypes condition) (lambda-dtypes update)))

              (soac/product-component-form? expression)
              (lambda-dtypes (:lambda (soac/product-fold-parts (second expression))))

              (soac/scalar-fold-form? expression)
              (lambda-dtypes (:lambda (soac/scalar-fold-parts expression)))

              (and (seq? expression) (= 'quote (first expression))) []
              (map? expression) (mapcat expression-dtypes (mapcat identity expression))
              (coll? expression) (mapcat expression-dtypes expression)
              :else []))
          (lambda-dtypes [candidate]
            (let [{:keys [locals body-results effect-results]} (soac/lambda-parts candidate)]
              (into (mapv :dtype locals)
                    (mapcat expression-dtypes
                            (concat (map :init locals) body-results effect-results)))))]
    (lambda-dtypes lambda)))

(defn- typed-operation-local-dtypes
  [equation]
  (let [{:keys [attributes folds] :as parts} (soac/operation-parts equation)
        result-transform (:result-transform attributes)
        lambdas (concat (keep parts [:lambda :element-lambda :combine-lambda :map-lambda])
                        (map :lambda folds)
                        (when (and result-transform (soac/result-transform? result-transform))
                          [(:lambda result-transform)]))]
    (mapcat scalar-region-local-dtypes lambdas)))

(defn- typed-compilation-dtype
  "Derive one floating carrier from retained AbstractValues and scalar local declarations."
  [typed-program]
  (let [facts (soac/facts typed-program)
        value-dtypes
        (keep (fn [[_ value]]
                (av/validate! value)
                (when (= :tensor (:kind value)) (:dtype value)))
              (:values facts))
        local-dtypes (mapcat typed-operation-local-dtypes (soac/equations typed-program))
        floating (->> (concat value-dtypes local-dtypes)
                      (keep #(when (and (keyword? %) (dtype/known? %)) (dtype/canon %)))
                      (filter #(get-in dtype/dtype-info [% :fp?]))
                      distinct vec)]
    (when-not (and (= 1 (count floating))
                   (contains? #{:float :double} (first floating)))
      (fail! :equation-first-typed-dtype
             "source-free compilation requires one retained Float or Double carrier"
             {:floating-dtypes floating}))
    (first floating)))

(defn- typed-compiler-options
  [typed-program target carrier options target-descriptor reference-schedule]
  (let [facts (soac/facts typed-program)
        inputs (:inputs facts)
        values (:values facts)
        input-values (select-keys values inputs)
        array-types (into {} (keep (fn [[id value]]
                                     (when (and (= :tensor (:kind value))
                                                (seq (:shape value)))
                                       [id (:dtype value)]))) input-values)
        scalar-types (into {} (keep (fn [[id value]]
                                      (when (and (= :tensor (:kind value))
                                                 (empty? (:shape value)))
                                        [id (:dtype value)]))) input-values)]
    (merge {:resident-reductions? true
            :dtype carrier
            :segmented-plans? true
            :target-device target
            :target-descriptor target-descriptor
            :active-params inputs
            :public-parameters inputs
            :array-types array-types
            :scalar-types scalar-types
            :schedule reference-schedule}
           (select-keys options [:scalar-math]))))

(defn compile-typed
  "Compile one exact pure TypedSOAC value through the ordinary immutable compilation tail.

   The TypedSOAC owner supplies its complete logical interface. No deftm frontend, type inference,
   source namespace, persistent artifact identity, Prepared cache, or runtime session is created.
   Target, schedule and scalar-math policy are caller-selectable. `:resident-outputs` may demand an
   ordered subset of exact public output IDs whose direct output-only reductions must remain
   certified one-element device buffers. Independently audited reductions captured by later
   equations are represented as internal resident buffers before equation promotion. Plain host
   arrays satisfy owned public inputs; retained borrowed inputs still require a later same-session
   module owner."
  ([typed-program] (compile-typed typed-program {}))
  ([typed-program {:keys [target] :or {target :ze:0} :as options}]
   (let [unsupported (seq (remove typed-compilation-options (keys options)))]
     (when unsupported
       (fail! :equation-first-typed-options
              "source-free compilation accepts only target, schedule, scalar-math and resident-output policy"
              {:unsupported (vec (sort unsupported))
               :allowed (vec (sort typed-compilation-options))})))
   (let [resident-demand? (contains? options :resident-outputs)
         resident-outputs (if resident-demand?
                            (do (soac/validate! typed-program)
                                (validate-resident-output-demand!
                                 typed-program (:resident-outputs options)))
                            [])
         options (cond-> options resident-demand? (assoc :resident-outputs resident-outputs))
         options (cond-> options
                   (contains? options :scalar-math)
                   (update :scalar-math numerics/validate-scalar-math-policy!))
         ;; Promotion splits the complete TypedSOAC into scheduled equation components. Run the
         ;; shared reduction-only transaction while producer/use/facet authority is still whole;
         ;; otherwise a reduction result captured by a later map would become a physical buffer in
         ;; its producer component but remain a scalar in its consumer component. An explicit
         ;; public representation demand adds stricter roots to this same transaction.
         [compiled-program _resident-stats]
         (resident/realize-source-free-reductions typed-program resident-outputs)
         promoted (structured-route/promote-source-free-algorithm compiled-program)
         carrier (typed-compilation-dtype typed-program)
         source-fingerprint (semantic/fingerprint typed-program)
         function-id [::typed-algorithm source-fingerprint]
         policy (compilation-policy function-id target options nil)
         compiler-options (typed-compiler-options
                           typed-program target carrier options (:target-descriptor policy)
                           (:reference-schedule policy))
         explicit-policy (select-keys options [:schedule :scalar-math :resident-outputs])
         id [::typed-compilation source-fingerprint target carrier explicit-policy]]
     (validate-resident-output-compilation!
      (compile-semantic
       (merge policy
              {:id id :function-id function-id :target target :dtype carrier :source-ns nil
               :options options :compiler-options compiler-options})
       promoted)
      resident-outputs))))

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
   (method-entry/call-with-definition-inputs
    (fn []
   (when (contains? options :resident-outputs)
     (fail! :equation-first-resident-outputs
            "explicit resident outputs are available only for source-free TypedSOAC compilation"
            {:function (when (var? f-var) (function-symbol f-var))}))
   (when-not (var? f-var)
     (fail! :equation-first-function "equation-first compilation requires a deftm Var"
            {:function f-var :actual (type f-var)}))
   (when (dispatch/host-only? f-var)
     (fail! :equation-first-host-only
            "equation-first GPU compilation was requested for an explicitly host-only deftm"
            {:function (function-symbol f-var) :target target}))
   (let [options (cond-> options
                   (contains? options :scalar-math)
                   (update :scalar-math numerics/validate-scalar-math-policy!))
         resolved-var (physical-function f-var dtype)
         _ (when (dispatch/host-only? resolved-var)
             (fail! :equation-first-host-only
                    "equation-first specialization resolved to an explicitly host-only method"
                    {:function (function-symbol resolved-var) :target target :dtype dtype}))
         function-id (function-symbol resolved-var)
         _ (rcore/prepare-deftm-frontend! resolved-var)
         _ (method-entry/observe-definition-input! resolved-var (:ns (meta resolved-var))
                                                   (:name (meta resolved-var)))
         policy (compilation-policy function-id target options captured-target)
         {:keys [target-descriptor reference-schedule]} policy
         compiler-options (compiler-options resolved-var target dtype
                                            (-> options
                                                (dissoc :target :gemm-precision)
                                                (assoc :schedule reference-schedule
                                                       :target-descriptor target-descriptor)))
         retained-frontend (pipeline/retained-frontend-source
                            f-var resolved-var (:dtype compiler-options) compiler-options)
         walked (or (when retained-frontend
                      (pipeline/normalize-source-bindings
                       (:body retained-frontend) (:raster.core/deftm-params (meta resolved-var))))
                    (pipeline/get-walked-body resolved-var (:dtype compiler-options)))
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
         ;; Keep the actual normalized admission refusal. Reanalyzing the
         ;; original compatibility form can report already-normalized bindings
         ;; as additional gaps and needlessly repeats source analysis.
         semantic-trace (pipeline/run-passes-diagnostic
                         (:body represented) pipeline/gpu-semantic-post-soa-passes
                         compiler-options :write-read-fused)
         semantic-candidate (:form semantic-trace)
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
                     :dialect (:dialect semantic) :fallback :none
                     :coverage (or (get-in semantic-trace [:stats :soac-fused :typed-soac-declined])
                                   (typed-frontend/coverage-decline semantic compiler-options))}))
         semantic (cond-> semantic
                    projection (assoc-in [:attributes :invocation-plan :attributes :parameter-projection]
                                         projection))
         source-ns-symbol (source-namespace-symbol f-var)
         id (cond-> [::compilation (function-symbol f-var) target (:dtype compiler-options)]
              (:preserve-declared-array-storage? compiler-options)
              (conj {:preserve-declared-array-storage? true}))]
     (compile-semantic
      (merge policy
             {:id id :function-id (function-symbol f-var) :target target
              :dtype (:dtype compiler-options) :source-ns source-ns-symbol
              :source-var resolved-var :retained-frontend retained-frontend
              :options options :compiler-options compiler-options})
      semantic))))))

(defn- lower-in-context
  "Specialize a compiled equation-first program against ordered public arguments.

   Returns a validated, allocation-free LinkPlan. Public buffers retain their stable host source
   identity until instantiation. Flat source-owned scalar prefixes use their prepared original
   executors; shared host equations reuse those results. The ownerless and aggregate reference
   subsets retain their existing typed JVM evaluator and grant no ordered-source capability.
   Retained-validation is an internal capability path: it must be exact-owner
   static evidence. Invocation materialization and final LinkPlan validation remain fresh."
  [compilation arguments project retained-validation caller-options]
  (stage-result/refuse-unobserved! compilation :lower-equation-compilation)
  (when-not (equation-first-compilation? compilation)
    (fail! :equation-first-compilation "lower requires an EquationFirstCompilation"
           {:actual (type compilation)}))
  (let [started (System/nanoTime)
        source-ns (:source-ns compilation)
        invocation-plan (invocation/validate! (get-in compilation [:semantic :attributes :invocation-plan]))
        _ (when (get-in invocation-plan [:attributes :terminal-host-result])
            (fail! :explicit-host-result-required
                   "terminal host observation requires the ordered host-result executor" {}))
        ordered-host-actions (invocation/ordered-host-actions invocation-plan (:semantic compilation))
        _ (when (seq ordered-host-actions)
            (fail! :ordered-source-execution-required
                   "host actions after numerical work require source-ordered execution, not eager materialization"
                   {:actions (mapv #(select-keys % [:position :binding :invocation-step :equation])
                                   ordered-host-actions)}))
        _ (when (:ordered-source-owner compilation)
          (source-owner/assert-ordered-compilation-source! compilation))
        eager-context (source-owner/assert-eager-evaluation! compilation)
        source-ns (or (:source-ns eager-context) source-ns)
        owned-prefix
        (when (= :owned-flat (:mode eager-context))
          (let [initial (invocation-link/begin-ordered-realization compilation arguments caller-options)]
            ;; Qualify the whole executor set before any scalar source work.
            ;; No missing executor may fall back after an earlier check ran.
            (source-owner/assert-ordered-executor-coverage! compilation initial arguments)
            initial))
        unsupported-host-evaluation
        (fn [& _]
          (fail! :equation-first-source-free-host-evaluation
                 "source-free compilation cannot evaluate a host invocation step or equation"
                 {:function (:function compilation)}))
        evaluate-step (cond
                        owned-prefix (partial source-owner/evaluate-ordered-invocation-step compilation)
                        source-ns (partial scalar/evaluate-invocation-step source-ns)
                        :else unsupported-host-evaluation)
        materialized
        (materialization/materialize
         invocation-plan (vec arguments)
         evaluate-step)
        materialization-ns (- (System/nanoTime) started)
        buffer-shapes (into {} (map (fn [[id buffer]] [id (:shape buffer)]))
                            (:program-buffers materialized))
        shared-host-steps (when owned-prefix
                            (into {} (keep (fn [entry]
                                            (when (and (:equation entry) (:invocation-step entry))
                                              [(:equation entry) (:invocation-step entry)])))
                                  (:source-order owned-prefix)))
        evaluate-host (cond
                        owned-prefix
                        (fn [equation _]
                          (source-owner/assert-ordered-compilation-source! compilation)
                          (let [step (get shared-host-steps (:id equation))
                                result (get-in materialized [:values step])]
                            (when-not (and step (= 1 (count (:results equation))) result)
                              (fail! :equation-first-owned-host-result
                                     "owned host equation lacks its completed invocation result"
                                     {:equation (:id equation)}))
                            {(first (:results equation)) result}))
                        source-ns
                        (fn [equation context]
                          (scalar/evaluate-host-equation
                           source-ns equation (assoc context :buffer-shapes buffer-shapes)))
                        :else unsupported-host-evaluation)
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
    (when (:ordered-source-owner compilation)
      (source-owner/assert-ordered-compilation-source! compilation))
    (source-owner/assert-eager-evaluation! compilation)
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
      (:plan (lower compilation arguments (fn [plan] {:plan plan}) nil
                    (select-keys options [:scalar-math])))
      (lower compilation arguments))))

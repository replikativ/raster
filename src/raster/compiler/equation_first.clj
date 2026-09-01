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
            [raster.compiler.backend.gpu.parallel-program-opencl :as program-opencl]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.pipeline :as pipeline]))

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
    (merge options
           {:dtype effective-dtype
            :target-device target
            :active-params parameters
            :public-parameters parameters
            :source-ns source-ns
            :array-types (:array-types parameter-types)
            :scalar-types (:scalar-types parameter-types)}
           (when param-env {:param-env param-env}))))

(defn compile
  "Compile one deftm Var into an immutable equation-first target program.

   Options currently require `:target` in the Level Zero/OpenCL family. CUDA/HIP source emitters
   will plug in at this same scheduled-program boundary; unsupported target families fail rather
   than borrowing the compatibility backend. `:dtype` defaults from the deftm's retained tags."
  ([f-var] (compile f-var {}))
  ([f-var {:keys [target dtype] :or {target :ze:0} :as options}]
   (when-not (var? f-var)
     (fail! :equation-first-function "equation-first compilation requires a deftm Var"
            {:function f-var :actual (type f-var)}))
   (let [compiler-options (compiler-options f-var target dtype
                                            (dissoc options :target))
         walked (pipeline/get-walked-body f-var (:dtype compiler-options))
         source (if (= 1 (count walked)) (first walked) (list* 'do walked))
         semantic (pipeline/run-passes source pipeline/gpu-resident-pre-soa-passes
                                       compiler-options)
         _ (when-not (= :typed-parallel (:dialect semantic))
             (fail! :equation-first-coverage
                    "deftm is outside the direct TypedSOAC/structured-control vertical"
                    {:function (function-symbol f-var)
                     :dialect (:dialect semantic) :fallback :none}))
         scheduled (structured-route/schedule-program semantic compiler-options)
         backend (device/select-runtime-backend target true nil)
         emission
         (case backend
           :opencl (program-opencl/emit-program scheduled compiler-options)
           (fail! :equation-first-target-emitter
                  "equation-first scheduled program has no public source emitter for this target"
                  {:target target :backend backend :fallback :none}))
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
         (partial scalar/evaluate-invocation-step source-ns))]
    (invocation-link/lower
     materialized (:emitted compilation) (:target compilation)
     (partial scalar/evaluate-host-equation source-ns))))

(defn compile-link-plan
  "Convenience composition of `compile` and `lower`; still performs no runtime allocation."
  [f-var arguments options]
  (lower (compile f-var options) arguments))

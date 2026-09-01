(ns raster.compiler.ir.kernel-artifact
  "A verified, single-entry target kernel as a compiler value.

   A KernelArtifact is the boundary between target emission and registration.  It keeps the
   target module, ordered ABI, compiler argument values, launch contract and provenance together;
   callers must not reconstruct any of these from a marker arity or parameter-name convention.

   This is intentionally the single-kernel value.  Multi-kernel algorithms (scan, staged
   contraction, etc.) belong in the scheduled kernel graph and contain KernelArtifacts as nodes."
  (:require [clojure.string :as str]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-launch :as klaunch]))

(defrecord KernelArtifact
           [kernel-name   ;; emitted entry-point string
            target        ;; target module dialect (:opencl-c, :cuda-c, or :hip-cpp)
            source        ;; target module source containing kernel-name
            abi           ;; ordered physical signature
            arguments     ;; compiler values in the identical order
            launch        ;; checked launch contract, not runtime handles
            temporaries   ;; explicit scheduled temporaries (vector)
            effects       ;; target effects not already implied by ABI
            provenance    ;; semantic/scheduled origin for explain/profile
            attributes])  ;; emitter properties (dtype, combine, phases, ...)

(defn kernel-artifact?
  "Recognize emitted artifacts across Typed Clojure's child DynamicClassLoaders."
  [x]
  (and x (= "raster.compiler.ir.kernel_artifact.KernelArtifact"
            (.getName (class x)))))

(defn validate!
  "Validate `artifact` and return it unchanged. Validation includes the emitted target C-family
   signature, so no unverified KernelArtifact can enter a runtime registry."
  [artifact]
  (when-not (kernel-artifact? artifact)
    (throw (ex-info "kernel artifact must be a KernelArtifact value"
                    {:artifact artifact :actual (type artifact)})))
  (let [{:keys [kernel-name target source abi arguments launch temporaries effects provenance
                attributes]} artifact]
    (when-not (and (string? kernel-name) (not (str/blank? kernel-name)))
      (throw (ex-info "kernel artifact requires a non-blank entry point"
                      {:kernel-name kernel-name})))
    (when-not (keyword? target)
      (throw (ex-info "kernel artifact requires a keyword target dialect"
                      {:kernel-name kernel-name :target target})))
    (when-not (and (string? source) (not (str/blank? source)))
      (throw (ex-info "kernel artifact requires a non-blank target module"
                      {:kernel-name kernel-name :target target})))
    (kabi/validate-arguments! abi arguments)
    (kabi/validate-alias-contracts! abi arguments #(or (identical? %1 %2) (= %1 %2)))
    (case target
      (:opencl-c :cuda-c :hip-cpp)
      (kabi/validate-source-signature! target kernel-name source abi)
      (throw (ex-info "kernel artifact target has no module verifier"
                      {:kernel-name kernel-name :target target})))
    (try
      (klaunch/validate-spec! launch)
      (catch clojure.lang.ExceptionInfo e
        (throw (ex-info "kernel artifact has an invalid launch contract"
                        {:kernel-name kernel-name :launch launch}
                        e))))
    (when-not (vector? temporaries)
      (throw (ex-info "kernel artifact temporaries must be an ordered vector"
                      {:kernel-name kernel-name :temporaries temporaries})))
    (doseq [[k value] [[:effects effects] [:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (throw (ex-info (str "kernel artifact " k " must be a map")
                        {:kernel-name kernel-name k value}))))
    artifact))

(defn make
  "Construct and verify a KernelArtifact.  Optional sections default to empty values, but launch
   is mandatory because an entry point without a launch contract is not executable kernel IR."
  [{:keys [kernel-name target source abi arguments launch temporaries effects provenance attributes]
    :or {target :opencl-c temporaries [] effects {} provenance {} attributes {}}}]
  (validate! (->KernelArtifact kernel-name target source abi arguments launch
                               temporaries effects provenance attributes)))

(defn certify-scheduled-operation
  "Bind a target artifact to the complete immutable scheduled operation it implements."
  [artifact scheduled-operation]
  (when (nil? scheduled-operation)
    (throw (ex-info "an emitted artifact requires its scheduled operation certificate"
                    {:reason :kernel-artifact-scheduled-operation})))
  (validate! (assoc-in (validate! artifact)
                       [:provenance :scheduled-operation]
                       scheduled-operation)))

(defn launch-value
  "Read one value from the artifact's launch contract."
  [artifact k]
  (get (:launch (validate! artifact)) k))

(defn attribute
  "Read one emitter attribute without flattening it into the runtime registry namespace."
  [artifact k]
  (get (:attributes (validate! artifact)) k))

(defn emission-route
  "Return the stable route used to produce an artifact.

   New emitters set `:emission-route` explicitly. Existing verified artifacts retain their
   provenance dialect as an observable compatibility class until they are migrated; only an
   artifact with neither fact is reported as unclassified. This accessor intentionally accepts
   artifact-shaped maps so pure compiler-report tests need not construct target source."
  [artifact]
  (or (get artifact :emission-route)
      (get-in artifact [:attributes :emission-route])
      (get-in artifact [:provenance :dialect])
      :unclassified))

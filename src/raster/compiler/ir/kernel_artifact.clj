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
            target        ;; target module dialect, currently :opencl-c
            source        ;; target module source containing kernel-name
            abi           ;; ordered physical signature
            arguments     ;; compiler values in the identical order
            launch        ;; checked launch contract, not runtime handles
            temporaries   ;; explicit scheduled temporaries (vector)
            effects       ;; target effects not already implied by ABI
            provenance    ;; semantic/scheduled origin for explain/profile
            attributes])  ;; emitter properties (dtype, combine, phases, ...)

(defn kernel-artifact? [x] (instance? KernelArtifact x))

(defn validate!
  "Validate `artifact` and return it unchanged.  Validation includes the emitted OpenCL
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
    (case target
      :opencl-c (kabi/validate-source-signature! kernel-name source abi)
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

(defn launch-value
  "Read one value from the artifact's launch contract."
  [artifact k]
  (get (:launch (validate! artifact)) k))

(defn attribute
  "Read one emitter attribute without flattening it into the runtime registry namespace."
  [artifact k]
  (get (:attributes (validate! artifact)) k))

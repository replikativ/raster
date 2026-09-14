(ns raster.compiler.backend.gpu.par-opencl
  "OpenCL backend for declarative parallel primitives.

  Pipeline pass that replaces par forms with Level Zero kernel invocations.
  Analogous to par_cuda.clj but targets Intel GPUs via OpenCL C → SPIR-V → Level Zero.

  Pipeline: par forms → OpenCL C source → SPIR-V (via ocloc, cached)
            → (raster.ze/invoke-kernel ...) S-expression markers

  Usage:
    (opencl-pass form :device-id :ze:0)
    ;; Returns {:form new-form :stats {...} :kernels [...]}"
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.ir.par :as par]
            [raster.runtime.hardware :as hw]
            [raster.compiler.core.hardware :as chw]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.passes.scalar.soa-lower :as sl]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [clojure.string :as str]
            [clojure.walk :as walk]))

;; ================================================================
;; OpenCL-specific helpers (expression emission delegated to c-emit)
;; ================================================================

(def ^:private rstr-dp4a-helper
  "Portable int8 4-way dot-accumulate helper SOURCE — canonical home is the
   intrinsics registry (:dp4a :c-helper-src); this def just reads it."
  (:c-helper-src (intrinsics/descriptor 'dp4a)))

(defn- body-uses-dp4a?
  "True if the kernel body calls the dp4a int8-dot primitive — ANY spelling, including the
   walker's devirtualized `(.invk dp4a_m_…-impl …)`. Registry-classified via `semantic-op`,
   NOT a literal `#{'dp4a 'par/dp4a 'raster.par/dp4a}` head set: that set was blind to the
   `.invk` form, so once `par/dp4a` became a deftm (so the census could see its type) the
   helper was no longer prepended and every kernel using it failed OpenCL compile with
   `use of undeclared identifier 'rstr_dp4a'`. Same defect class as the #93 aget matchers."
  [body]
  (boolean (some (fn [f] (and (seq? f) (= 'raster.par/dp4a (descriptor/semantic-op f))))
                 (tree-seq coll? seq body))))
;; collect-gpu-fn-calls, generate-c-helper, tag->ctype-helper now in ce/

(defn generate-par-map-void-kernel
  "Generate an OpenCL C void-map kernel from a raster.par/map-void! form.
  No output array — body is executed for side effects (aset, atomic-add!).

  array-types: {sym -> :float|:int|:long|:double} for mixed-type params.
  Written arrays get __global TYPE* (not const restrict).

  Returns one verified KernelArtifact. Its ABI and :arguments are physical; logical SoA bindings
  are a projection of the ABI's contiguous `:binding` groups."
  [form & {:keys [dtype kernel-name-prefix array-types scalar-types]
           :or {dtype :float kernel-name-prefix "par_map_void"
                array-types {} scalar-types {}}}]
  (let [info (par/extract-par-map-void-info form)
        {:keys [idx]} info
        ;; Normalize devirtualized array prims (.invk aget_m_T-impl …) back to aget/aset heads so
        ;; array detection + element typing + emit recognize them (else arrays→scalar, aget→broken
        ;; helper). The forward-pass devirtualizes raster.arrays/aget; Path A skips those passes.
        body (ce/normalize-array-prims (:body info))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ctype (get codegen/opencl-type-map dtype "float")
        ;; Auto-collect array types from walker :tag metadata, merge with explicit
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        ;; SoA expansion: find symbols with SoA type tags (shared body-tag detector)
        soa-expansions (sl/collect-soa-env body)
        array-syms (ce/collect-arrays-in-body body)
        written-syms (ce/collect-written-arrays body)
        read-syms (ce/collect-read-arrays body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        all-arr-params (vec (sort-by name array-syms))
        ;; Partition: SoA arrays vs plain arrays
        soa-arr-params   (filterv #(contains? soa-expansions (symbol (name %))) all-arr-params)
        plain-arr-params (filterv #(not (contains? soa-expansions (symbol (name %)))) all-arr-params)
        scl-params (vec (sort-by name scalar-syms))
        ;; Per-array C type: use array-types map, fall back to dtype
        arr-type (fn [s]
                   (let [t (get array-types s (get array-types (symbol (name s)) dtype))]
                     (get codegen/opencl-type-map t default-ctype)))
        written? (fn [s] (or (contains? written-syms s)
                             (contains? written-syms (symbol (name s)))))
        read? (fn [s] (or (contains? read-syms s)
                          (contains? read-syms (symbol (name s)))))
        pointer-role (fn [s]
                       (cond
                         (and (written? s) (read? s)) :inout
                         (written? s) :effect
                         :else :operand))
        ;; Plain array params
        plain-arr-param-str (str/join ", "
                                      (map (fn [s]
                                             (let [ct (arr-type s)]
                                               (if (written? s)
                                                 (str "__global " ct "* " (ce/c-symbol s))
                                                 (str "__global const " ct "* restrict " (ce/c-symbol s)))))
                                           plain-arr-params))
        ;; SoA arrays: expand each SoA symbol into N flat __global pointers
        soa-arr-param-str (str/join ", "
                                    (mapcat (fn [s]
                                              (let [soa-info (get soa-expansions (symbol (name s)))
                                                    fields   (:fields soa-info)
                                                    writable (written? s)]
                                                (map (fn [{:keys [name element-tag]}]
                                                       (let [ct       (get ce/element-tag->c element-tag "float")
                                                             flat-sym (str (ce/c-symbol s) "_" name)]
                                                         (if writable
                                                           (str "__global " ct "* " flat-sym)
                                                           (str "__global const " ct "* restrict " flat-sym))))
                                                     fields)))
                                            soa-arr-params))
        ;; Infer scalar types: check explicit scalar-types map, name heuristic, or metadata
        scalar-dtype #(ce/scalar-parameter-dtype % scalar-types dtype)
        scl-type #(get codegen/opencl-type-map (scalar-dtype %) default-ctype)
        scalar-var-types (into {} (map (juxt identity scl-type)) scl-params)
        element-dtype (fn [tag]
                        (case tag
                          double :double
                          float :float
                          long :long
                          int :int
                          byte :byte))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        all-params (str/join ", "
                             (remove empty?
                                     [plain-arr-param-str soa-arr-param-str scl-param-str "int _n_bound"]))
        ;; The ABI describes the PHYSICAL C signature. An SoA contributes one pointer per field,
        ;; each linked back to the single logical caller binding via :binding. This lets source
        ;; validation and driver binding share one record without making the marker flatten a
        ;; GpuSoA itself.
        abi (kabi/validate!
             (vec
              (concat
               (map (fn [s]
                      (kabi/slot s (if (written? s) :output :input)
                                 (get array-types s (get array-types (symbol (name s)) dtype))
                                 :c-name (ce/c-symbol s)
                                 :role (pointer-role s)))
                    plain-arr-params)
               (mapcat (fn [s]
                         (map (fn [{field-name :name element-tag :element-tag}]
                                (let [field-sym (sl/field-arr-sym (symbol (name s)) field-name)]
                                  (kabi/slot field-sym (if (written? s) :output :input)
                                             (element-dtype element-tag)
                                             :c-name (ce/c-symbol field-sym)
                                             :binding s
                                             :field field-name
                                             :role (pointer-role s))))
                              (:fields (get soa-expansions (symbol (name s))))))
                       soa-arr-params)
               (map (fn [s]
                      (kabi/slot s :scalar (scalar-dtype s)
                                 :c-name (ce/c-symbol s) :role :parameter))
                    scl-params)
               [(kabi/slot '_n_bound :scalar :int :role :bound)])))
        ;; SROA: scalar-replace value-type access so the body has only per-field
        ;; plain array ops (no struct typedef / SoA aget-aset left for the C emitter).
        lowered-body (sl/lower-body soa-expansions body)
        adapted-body (ce/adapt-casts-for-dtype lowered-body dtype)
        ;; Per-field array symbols introduced by lowering (e.g. os_re, os_im),
        ;; replacing the SoA base names in the array-sym tracking set.
        soa-field-syms (set (mapcat (fn [s]
                                      (let [info (get soa-expansions (symbol (name s)))]
                                        (map (fn [{nm :name}]
                                               (sl/field-arr-sym (symbol (name s)) nm))
                                             (:fields info))))
                                    soa-arr-params))
        all-arr-syms (into (set (map #(symbol (name %)) plain-arr-params)) soa-field-syms)
        ;; Seed *int-vars* with the work-item index and the DECLARED-int scalar params
        ;; (e.g. `features`, `rows`), so index arithmetic like `(* idx features)` infers an
        ;; integer C type for the offset local rather than defaulting to the float scalar-type
        ;; (which yields non-integer array subscripts in per-row kernels with inner loops).
        int-scalar-syms (into #{idx} (filter #(contains? #{:int :long :byte}
                                                         (scalar-dtype %)) scl-params))
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* default-ctype
                           ce/*scalar-var-types* scalar-var-types
                           ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                   (ce/emit-stmt adapted-body idx all-arr-syms "idx"))
        ;; Affine-index vectorization (shared c_emit): float4/float2 grid-stride loop
        ;; guarded by a runtime divisibility check, else the scalar loop. nil ⇒ not
        ;; provably vectorizable ⇒ scalar loop below.
        loop-region (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* default-ctype
                              ce/*scalar-var-types* scalar-var-types
                              ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                      (ce/emit-vectorized-elementwise-loop
                       adapted-body idx all-arr-syms "idx" body-str {:n-bound "_n_bound"}))
        ;; Detect if body uses float atomic-add (needs CAS helper function)
        needs-float-atomic? (let [found (atom false)]
                              (walk/postwalk
                               (fn [form]
                                 (when (and (seq? form)
                                            (contains? #{'raster.par/atomic-add! 'par/atomic-add!} (first form))
                                            (>= (count form) 4))
                                   (let [arr (second form)
                                         tag (when (symbol? arr) (:tag (meta arr)))]
                                     (when (contains? #{'floats 'float} tag)
                                       (reset! found true))))
                                 form)
                               body)
                              @found)
        ;; Collect GPU-inlinable deftm helper functions referenced in body
        gpu-helpers (ce/collect-gpu-fn-calls body)
        helper-sources (str/join "\n" (map (comp :source ce/generate-c-helper) gpu-helpers))
        ;; Enable fp64 whenever double appears in the emitted kernel or its helpers — a float
        ;; kernel can still carry double from (double ...) casts or raster.numeric helpers, and
        ;; using double WITHOUT the extension is undefined (garbage) on the GPU.
        needs-fp64? (or (str/includes? body-str "double")
                        (str/includes? helper-sources "double"))
        source (str (codegen/extension-pragmas dtype (when needs-fp64? :double))
                    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                    helper-sources
                    (when needs-float-atomic? ce/opencl-atomic-add-float-helper)
                    ;; registry-owned intrinsic definitions (rstr_dp4a …), defined iff called
                    (ce/intrinsic-helper-sources body-str)
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    "
                    (or loop-region
                        (str "for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                             "        " body-str "\n"
                             "    }"))
                    "\n}\n")
        _ (kabi/validate-source-signature! kernel-name source abi)
        binding-params (kabi/pointer-binding-names abi)
        bound (:bound info)
        physical-arguments
        (mapv (fn [slot]
                (if (= :bound (:role slot)) bound (:name slot)))
              abi)]
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments physical-arguments
      :launch (klaunch/spec
               {:workgroup-size [256]
                :group-count [(klaunch/ceil-div bound 256)]})
      :temporaries []
      :effects {:kind :side-effect-map}
      :provenance {:dialect :map-void}
      :attributes {:array-params binding-params
                   :soa-expansions soa-expansions
                   :scalar-params scl-params
                   :written-arrays written-syms
                   :array-types array-types
                   :dtype dtype}})))


;; ================================================================
;; SPIR-V compilation (cached)
;; ================================================================

(def ^:private spirv-cache
  "Lazy SPIR-V cache instance."
  (delay (spirv-cache/make-cache)))

(defn compile-kernel-to-spirv
  "Compile OpenCL C source to SPIR-V bytes, using the cache.
  Resolves device-id (e.g. :ze:0) to ocloc device hex (e.g. 0x64a0)."
  [^String cl-source & {:keys [device-id]}]
  (let [device-hex (when device-id
                     (try (get-in (hw/device device-id)
                                  [:capabilities :device-id-hex])
                          (catch Exception _ nil)))]
    (spirv-cache/get-or-compile
     @spirv-cache
     cl-source
     (fn [src] (spirv-cache/compile-opencl-to-spirv src
                                                    :device device-hex))
     (or device-hex (when device-id (name device-id))))))
;; ================================================================
;; Pipeline pass
;; ================================================================

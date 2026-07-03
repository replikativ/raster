(ns raster.compiler.backend.gpu.par-opencl
  "OpenCL backend for declarative parallel primitives.

  Pipeline pass that replaces par forms with Level Zero kernel invocations.
  Analogous to par_cuda.clj but targets Intel GPUs via OpenCL C → SPIR-V → Level Zero.

  Pipeline: par forms → OpenCL C source → SPIR-V (via ocloc, cached)
            → (raster.ze/invoke-kernel ...) S-expression markers

  Usage:
    (opencl-pass form :device-id :ze:0)
    ;; Returns {:form new-form :stats {...} :kernels [...]}"
  (:require [raster.compiler.ir.par :as par]
            [raster.runtime.hardware :as hw]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.passes.scalar.soa-lower :as sl]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [clojure.string :as str]
            [clojure.walk :as walk]))

;; ================================================================
;; OpenCL-specific helpers (expression emission delegated to c-emit)
;; ================================================================

(def ^:private atomic-add-float-helper
  "OpenCL CAS-loop for float atomic-add (no native float atomic_add in OpenCL 1.x)."
  "float atomic_add_float(volatile __global float *addr, float val) {
    int old = *(volatile __global int*)addr;
    int assumed;
    do {
        assumed = old;
        old = atomic_cmpxchg((volatile __global int*)addr, assumed,
                              as_int(as_float(assumed) + val));
    } while (old != assumed);
    return as_float(old);
}\n")

(def ^:private rstr-dp4a-helper
  "Portable int8 4-way dot-accumulate. `as_char4` reinterprets each int32's 4 bytes
  (little-endian: .x = low byte) as signed int8 lanes; the OpenCL compiler (Intel IGC,
  NVIDIA, AMD) pattern-matches this idiom to a hardware dp4a instruction."
  "inline int rstr_dp4a(int a, int b, int acc) {
    char4 va = as_char4(a);
    char4 vb = as_char4(b);
    return acc + (int)va.x*(int)vb.x + (int)va.y*(int)vb.y
               + (int)va.z*(int)vb.z + (int)va.w*(int)vb.w;
}\n")

(defn- body-uses-dp4a?
  "True if the kernel body calls the dp4a int8-dot primitive (any spelling)."
  [body]
  (let [found (atom false)]
    (walk/postwalk
     (fn [form]
       (when (and (seq? form)
                  (contains? #{'dp4a 'par/dp4a 'raster.par/dp4a} (first form)))
         (reset! found true))
       form)
     body)
    @found))

;; Kernel generators
;; ================================================================

(defn generate-par-map-kernel
  "Generate an OpenCL C kernel from a raster.par/map! form.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix scalar-types]
           :or {dtype :double kernel-name-prefix "par_map"
                scalar-types {}}}]
  (let [info (par/extract-par-map-info form)
        {:keys [out idx cast body]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        array-syms (ce/collect-arrays-in-body body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        arr-params (vec (sort-by name array-syms))
        scl-params (vec (sort-by name scalar-syms))
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        scl-type (fn [s] (ce/scalar-native-type s scalar-types ctype))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param (str "__global " ctype "* restrict out")
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param scl-param-str "int _n_bound"]))
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype]
                   (ce/emit-expr adapted-body idx (set (map #(symbol (name %)) arr-params))))
        cast-str (if cast (str "(" (name cast) ")(" body-str ")") body-str)
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                    "        out[idx] = " cast-str ";\n"
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :out-param out
     :dtype dtype}))

;; collect-gpu-fn-calls, generate-c-helper, tag->ctype-helper now in ce/

(defn generate-par-map-void-kernel
  "Generate an OpenCL C void-map kernel from a raster.par/map-void! form.
  No output array — body is executed for side effects (aset, atomic-add!).

  array-types: {sym -> :float|:int|:long|:double} for mixed-type params.
  Written arrays get __global TYPE* (not const restrict).

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :written-arrays #{syms} :dtype kw}."
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
        use-fp64? (= dtype :double)
        ;; Auto-collect array types from walker :tag metadata, merge with explicit
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        ;; SoA expansion: find symbols with SoA type tags (shared body-tag detector)
        soa-expansions (sl/collect-soa-env body)
        array-syms (ce/collect-arrays-in-body body)
        written-syms (ce/collect-written-arrays body)
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
        scl-type (fn [s] (ce/scalar-native-type s scalar-types default-ctype))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        all-params (str/join ", "
                             (remove empty?
                                     [plain-arr-param-str soa-arr-param-str scl-param-str "int _n_bound"]))
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
        int-scalar-syms (into #{idx} (filter #(= "int" (scl-type %)) scl-params))
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* default-ctype
                           ce/*idx-sym* idx
                           ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                   (ce/emit-stmt adapted-body idx all-arr-syms "idx"))
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
        needs-fp64? (or use-fp64?
                        (str/includes? body-str "double")
                        (str/includes? helper-sources "double"))
        source (str (when needs-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                    helper-sources
                    (when needs-float-atomic? atomic-add-float-helper)
                    (when (body-uses-dp4a? body) rstr-dp4a-helper)
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                    "        " body-str "\n"
                    "    }\n"
                    "}\n")]
    {:kernel-name    kernel-name
     :source         source
     :array-params   all-arr-params
     :soa-expansions soa-expansions
     :scalar-params  scl-params
     :written-arrays written-syms
     :array-types    array-types
     :dtype          dtype}))

(defn generate-par-scan-exclusive-kernel
  "Generate OpenCL Blelloch exclusive scan kernels from a par/scan-exclusive form.

  Uses work-efficient Blelloch algorithm (O(n) work, O(log n) span).
  Generates two kernels in one source:
    1. Block-level exclusive scan + block totals
    2. Block-offset propagation

  The body expression (op acc element-expr) is decomposed: element-expr is loaded
  per-element, and op is used as the combining operation in the scan.

  Returns {:block-kernel-name str :prop-kernel-name str :source str
           :input-arrays [syms] :output-array sym
           :scan-dtype kw :identity-num number :combine-op sym
           :block-size int :workgroup-size int}."
  [form & {:keys [dtype device-id kernel-name-prefix]
           :or {kernel-name-prefix "par_scan_excl"}}]
  (let [info (par/extract-par-scan-exclusive-info form)
        {:keys [out init idx cast body]} info
        ;; Determine scan dtype from cast
        scan-dtype (case cast
                     int :int
                     long :long
                     float :float
                     double :double
                     (or dtype :int))
        ctype (get {:int "int" :long "long" :float "float" :double "double"}
                   scan-dtype "int")
        use-fp64? (= scan-dtype :double)
        ;; Extract combining op and element expression (same pattern as reduce)
        [let-bindings inner-body]
        (if (and (seq? body) (contains? #{'let* 'let} (first body)))
          (let [[_ binds & bdy] body]
            [(vec (partition 2 binds)) (last bdy)])
          [[] body])
        raw-op (when (and (seq? inner-body) (>= (count inner-body) 3))
                 (first inner-body))
        op (if (contains? #{'+ '* 'Math/max 'Math/min} raw-op) raw-op '+)
        element-expr (when (and (seq? inner-body) (>= (count inner-body) 3))
                       (nth inner-body 2))
        ;; Collect input arrays from body (exclude output array)
        array-syms (ce/collect-arrays-in-body body)
        input-arrays (disj array-syms out (symbol (name out)))
        arr-params (vec (sort-by name input-arrays))
        ;; Identity value
        identity-num (cond
                       (and (seq? init) (= 2 (count init))
                            (contains? #{'int 'long 'float 'double} (first init)))
                       (second init)
                       (number? init) init
                       :else 0)
        identity-str (cond
                       (= scan-dtype :float)
                       (str identity-num "f")
                       :else
                       (str identity-num))
        ;; Kernel names
        block-kn (str kernel-name-prefix "_blocks_" (gensym ""))
        prop-kn (str kernel-name-prefix "_prop_" (gensym ""))
        ;; Workgroup/block sizes (power of 2)
        wg-size (if device-id
                  (hw/optimal-workgroup-size device-id 256 :reduction? true)
                  256)
        block-size (* 2 wg-size) ;; Each thread handles 2 elements
        ;; OpenCL combine op
        combine-str (condp = op '+ "+" '* "*" 'Math/max "fmax" 'Math/min "fmin" "+")
        combine (ce/combine-fn combine-str (ce/fn-style-reduction-op? op))
        ;; Emit element expression as OpenCL C
        array-sym-names (set (map #(symbol (name %)) arr-params))
        adapted-element (when element-expr (ce/adapt-casts-for-dtype element-expr scan-dtype))
        adapted-let-bindings (mapv (fn [[s v]] [s (ce/adapt-casts-for-dtype v scan-dtype)])
                                   let-bindings)
        element-str (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* ctype]
                      (if adapted-element
                        (if (seq adapted-let-bindings)
                          (let [let-body (list* 'let*
                                                (vec (mapcat identity adapted-let-bindings))
                                                [adapted-element])]
                            (ce/emit-expr let-body idx array-sym-names "idx"))
                          (ce/emit-expr adapted-element idx array-sym-names "idx"))
                        "input[idx]"))
        ;; Kernel params
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        half-block (/ block-size 2)
        ;; Generate source with two kernels
        source (str
                (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                "\n"
                 ;; Kernel 1: Block-level exclusive scan (Blelloch)
                "__kernel void " block-kn "(\n"
                (when (seq arr-params) (str "    " arr-param-str ",\n"))
                "    __global " ctype "* output,\n"
                "    __global " ctype "* block_sums,\n"
                "    int _n_bound)\n"
                "{\n"
                "    __local " ctype " sdata[" block-size "];\n"
                "    int tid = get_local_id(0);\n"
                "    int block_id = get_group_id(0);\n"
                "    int block_offset = block_id * " block-size ";\n"
                "    int ai = 2 * tid;\n"
                "    int bi = 2 * tid + 1;\n"
                "    int idx;\n"
                "\n"
                 ;; Load 2 elements per thread (apply element expression)
                "    idx = block_offset + ai;\n"
                "    sdata[ai] = (idx < _n_bound) ? " element-str " : " identity-str ";\n"
                "    idx = block_offset + bi;\n"
                "    sdata[bi] = (idx < _n_bound) ? " element-str " : " identity-str ";\n"
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n"
                 ;; Up-sweep (reduce phase)
                "    int offset = 1;\n"
                "    for (int d = " half-block "; d > 0; d >>= 1) {\n"
                "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                "        if (tid < d) {\n"
                "            int a_idx = offset * (2 * tid + 1) - 1;\n"
                "            int b_idx = offset * (2 * tid + 2) - 1;\n"
                "            sdata[b_idx] = " (combine "sdata[b_idx]" "sdata[a_idx]") ";\n"
                "        }\n"
                "        offset <<= 1;\n"
                "    }\n"
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n"
                 ;; Save block total, set identity for down-sweep
                "    if (tid == 0) {\n"
                "        block_sums[block_id] = sdata[" (dec block-size) "];\n"
                "        sdata[" (dec block-size) "] = " identity-str ";\n"
                "    }\n"
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n"
                 ;; Down-sweep
                "    for (int d = 1; d < " block-size "; d <<= 1) {\n"
                "        offset >>= 1;\n"
                "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                "        if (tid < d) {\n"
                "            int a_idx = offset * (2 * tid + 1) - 1;\n"
                "            int b_idx = offset * (2 * tid + 2) - 1;\n"
                "            " ctype " t = sdata[a_idx];\n"
                "            sdata[a_idx] = sdata[b_idx];\n"
                "            sdata[b_idx] = " (combine "sdata[b_idx]" "t") ";\n"
                "        }\n"
                "    }\n"
                "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n"
                 ;; Write output
                "    if (block_offset + ai < _n_bound) output[block_offset + ai] = sdata[ai];\n"
                "    if (block_offset + bi < _n_bound) output[block_offset + bi] = sdata[bi];\n"
                "}\n"
                "\n"
                 ;; Kernel 2: Propagate block offsets
                "__kernel void " prop-kn "(\n"
                "    __global " ctype "* output,\n"
                "    __global const " ctype "* block_offsets,\n"
                "    int _n_bound)\n"
                "{\n"
                "    int tid = get_local_id(0);\n"
                "    int block_id = get_group_id(0);\n"
                "    int base = block_id * " block-size ";\n"
                "    int ai = base + 2 * tid;\n"
                "    int bi = base + 2 * tid + 1;\n"
                "    " ctype " off = block_offsets[block_id];\n"
                "    if (ai < _n_bound) output[ai] = " (combine "output[ai]" "off") ";\n"
                "    if (bi < _n_bound) output[bi] = " (combine "output[bi]" "off") ";\n"
                "}\n")]
    {:block-kernel-name block-kn
     :prop-kernel-name prop-kn
     :source source
     :input-arrays arr-params
     :output-array out
     :scan-dtype scan-dtype
     :identity-num identity-num
     :combine-op op
     :block-size block-size
     :workgroup-size wg-size
     :dtype (or dtype scan-dtype)}))

(defn generate-par-reduce-kernel
  "Generate an OpenCL C reduction kernel from a raster.par/reduce form.
  Uses local memory tree reduction with subgroup optimization.
  Fuses the reduce body expression directly into the kernel.

  Returns {:kernel-name str :source str :op sym :identity-val num :dtype kw
           :array-params [syms] :workgroup-size int}."
  [form & {:keys [dtype device-id kernel-name-prefix]
           :or {dtype :double kernel-name-prefix "par_reduce"}}]
  (let [info (par/extract-par-reduce-info form)
        {:keys [init idx bound body]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        ;; Detect reduction op: the body is (op acc element-expr)
        ;; We need to extract the element expression and the combining op
        [let-bindings inner-body]
        (if (and (seq? body) (contains? #{'let* 'let} (first body)))
          (let [[_ binds & bdy] body]
            [(vec (partition 2 binds)) (last bdy)])
          [[] body])
        raw-op (when (and (seq? inner-body) (>= (count inner-body) 3))
                 (first inner-body))
        op (if (contains? #{'+ '- '* 'Math/max 'Math/min} raw-op) raw-op '+)
        ;; Extract element expression (second arg to combining op, after acc)
        ;; e.g. (+ acc (* (- (aget pred i) (aget X i)) (- (aget pred i) (aget X i))))
        ;; → element-expr is the second arg
        element-expr (when (and (seq? inner-body) (>= (count inner-body) 3))
                       (nth inner-body 2))
        ;; Collect arrays used in the full body
        array-syms (ce/collect-arrays-in-body body)
        arr-params (vec (sort-by name array-syms))
        identity-val (or init 0.0)
        ;; Workgroup size for local memory
        wg-size (if device-id
                  (hw/optimal-workgroup-size device-id (or bound 65536) :reduction? true)
                  256)
        reduce-op-str (condp = op
                        '+ "+"
                        '* "*"
                        'Math/max "fmax"
                        'Math/min "fmin"
                        "+")
        combine (ce/combine-fn reduce-op-str (ce/fn-style-reduction-op? op))
        ;; Generate per-array kernel params
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str
                                      (str "__global " ctype "* restrict output")
                                      "int _n_bound"]))
        ;; Emit the element expression as OpenCL C
        array-sym-names (set (map #(symbol (name %)) arr-params))
        ;; Adapt casts for target dtype (double→float when :float)
        adapted-element (when element-expr (ce/adapt-casts-for-dtype element-expr dtype))
        adapted-let-bindings (mapv (fn [[s v]] [s (ce/adapt-casts-for-dtype v dtype)]) let-bindings)
        element-str (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* ctype]
                      (if adapted-element
                        (ce/emit-expr adapted-element idx array-sym-names "i")
                        "input[i]"))
        ;; Handle let bindings in body by wrapping element expression
        full-element-str (binding [ce/*emit-config* ce/opencl-config
                                   ce/*scalar-type* ctype]
                           (if (seq adapted-let-bindings)
                             (let [let-body (list* 'let*
                                                   (vec (mapcat identity adapted-let-bindings))
                                                   [(or adapted-element 'input)])]
                               (ce/emit-expr let-body idx array-sym-names "i"))
                             element-str))
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    __local " ctype " sdata[" wg-size "];\n"
                    "    int tid = get_local_id(0);\n"
                    "    " ctype " val = " (ce/normalize-identity-val identity-val) ";\n"
                    "    int stride = get_global_size(0);\n"
                    "    int i = get_global_id(0);\n"
                    ;; No unrolling for fused body (expressions can be complex)
                    "    for (; i < _n_bound; i += stride) {\n"
                    "        val = " (combine "val" full-element-str) ";\n"
                    "    }\n"
                    "    sdata[tid] = val;\n"
                    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                    ;; Tree reduction
                    "    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
                    "        if (tid < s) {\n"
                    "            sdata[tid] = " (combine "sdata[tid]" "sdata[tid + s]") ";\n"
                    "        }\n"
                    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                    "    }\n"
                    "    if (tid == 0) output[get_group_id(0)] = sdata[0];\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :op op
     :identity-val identity-val
     :array-params arr-params
     :dtype dtype
     :workgroup-size wg-size}))

;; ================================================================
;; Row-softmax kernel generator
;; ================================================================

(defn generate-row-softmax-kernel
  "Generate an OpenCL row-softmax kernel.
  Returns {:kernel-name str :source str :dtype kw :workgroup-size int}."
  [& {:keys [dtype kernel-name-prefix workgroup-size]
      :or {dtype :double kernel-name-prefix "row_softmax" workgroup-size 256}}]
  (let [kernel-name (str kernel-name-prefix "_" (gensym ""))]
    {:kernel-name kernel-name
     :source (codegen/emit-row-softmax-kernel kernel-name dtype
                                              :workgroup-size workgroup-size)
     :dtype dtype
     :workgroup-size workgroup-size}))

;; ================================================================
;; GroupNorm kernel generator
;; ================================================================

(defn generate-group-norm-kernel
  "Generate an OpenCL GroupNorm kernel.
  Returns {:kernel-name str :source str :dtype kw :workgroup-size int}."
  [& {:keys [dtype kernel-name-prefix workgroup-size eps]
      :or {dtype :double kernel-name-prefix "group_norm" workgroup-size 256 eps 1e-5}}]
  (let [kernel-name (str kernel-name-prefix "_" (gensym ""))]
    {:kernel-name kernel-name
     :source (codegen/emit-group-norm-kernel kernel-name dtype
                                             :workgroup-size workgroup-size :eps eps)
     :dtype dtype
     :workgroup-size workgroup-size}))

;; ================================================================
;; Scatter-reduce kernel generator
;; ================================================================

(defn generate-scatter-reduce-kernel
  "Generate an OpenCL scatter-reduce kernel for graph attention.
  Returns {:kernel-name str :source str :variant kw :dtype kw}."
  [& {:keys [dtype variant with-weights? kernel-name-prefix]
      :or {dtype :double variant :atomic with-weights? true
           kernel-name-prefix "scatter_reduce"}}]
  (let [kernel-name (str kernel-name-prefix "_" (gensym ""))]
    {:kernel-name kernel-name
     :source (codegen/emit-scatter-reduce-kernel
              kernel-name dtype variant
              :with-weights? with-weights?)
     :variant variant
     :dtype dtype}))

(defn generate-scatter-reduce-scalar-kernel
  "Generate a scalar scatter-reduce kernel.
  Returns {:kernel-name str :source str :dtype kw}."
  [& {:keys [dtype kernel-name-prefix]
      :or {dtype :double kernel-name-prefix "scatter_reduce_scalar"}}]
  (let [kernel-name (str kernel-name-prefix "_" (gensym ""))]
    {:kernel-name kernel-name
     :source (codegen/emit-scatter-reduce-scalar-kernel kernel-name dtype)
     :dtype dtype}))

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
;; RNG fill kernel codegen (splitmix64)
;; ================================================================

(defn generate-par-active-ids-kernel
  "Generate an OpenCL kernel for raster.par/active-ids! (splitmix64 random index generation).
  Each work-item i computes: ids[i] = splitmix64(base_seed + i * golden_ratio) mod n_agents.
  Fully parallel — no inter-thread communication needed.

  Returns {:kernel-name str :source str :array-params ['ids]
           :scalar-params [{:name 'n-active :type :int} {:name 'n-agents :type :long}
                           {:name 'base-seed :type :long}]
           :active-ids? true}."
  [& {:keys [kernel-name device-id]
      :or {kernel-name (str "par_active_ids_" (gensym ""))}}]
  (let [src (str "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
                 "__kernel void " kernel-name
                 "(__global int* ids, int n_active, long n_agents, long base_seed) {\n"
                 "  int i = get_global_id(0);\n"
                 "  if (i >= n_active) return;\n"
                 "  ulong state = (ulong)base_seed + (ulong)(long)i * (ulong)0x9e3779b97f4a7c15L;\n"
                 "  state ^= state >> 30;\n"
                 "  state *= (ulong)0xbf58476d1ce4e5b9L;\n"
                 "  state ^= state >> 27;\n"
                 "  state *= (ulong)0x94d049bb133111ebL;\n"
                 "  state ^= state >> 31;\n"
                 "  ids[i] = (int)((long)(state & 0x7FFFFFFFFFFFFFFFL) % n_agents);\n"
                 "}\n")
        spv-bytes (when device-id
                    (compile-kernel-to-spirv src :device-id device-id))]
    {:kernel-name kernel-name
     :source src
     :spv-bytes spv-bytes
     :array-params ['ids]
     :scalar-params [{:name 'n-active :type :int} {:name 'n-agents :type :long}
                     {:name 'base-seed :type :long}]
     :active-ids? true}))

(defn generate-par-rng-fill-kernel
  "Generate an OpenCL kernel for raster.par/rng-fill! (splitmix64 per-element).
  Each work-item computes: seeds[i] = splitmix64(base_seed + i * golden_ratio_64).
  Fully parallel — no inter-thread communication needed.

  Returns {:kernel-name str :source str :array-params ['seeds]
           :scalar-params [{:name 'n :type :int} {:name 'base-seed :type :long}]
           :rng-fill? true}."
  [& {:keys [kernel-name device-id]
      :or {kernel-name (str "par_rng_fill_" (gensym ""))}}]
  (let [src (str "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
                 "__kernel void " kernel-name
                 "(__global long* seeds, int _n_bound, long base_seed) {\n"
                 "  int i = get_global_id(0);\n"
                 "  if (i >= _n_bound) return;\n"
                 "  ulong state = (ulong)base_seed + (ulong)(long)i * (ulong)0x9e3779b97f4a7c15L;\n"
                 "  state ^= state >> 30;\n"
                 "  state *= (ulong)0xbf58476d1ce4e5b9L;\n"
                 "  state ^= state >> 27;\n"
                 "  state *= (ulong)0x94d049bb133111ebL;\n"
                 "  state ^= state >> 31;\n"
                 "  seeds[i] = (long)state;\n"
                 "}\n")
        spv-bytes (when device-id
                    (compile-kernel-to-spirv src :device-id device-id))]
    {:kernel-name kernel-name
     :source src
     :spv-bytes spv-bytes
     :array-params ['seeds]
     :scalar-params [{:name 'n :type :int} {:name 'base-seed :type :long}]
     :rng-fill? true}))

;; ================================================================
;; Stencil kernel generator
;; ================================================================

(defn generate-par-stencil-kernel
  "Generate an OpenCL C kernel from a raster.par/stencil! form.

  Stencil body reads from in-arrays within a neighborhood of radius.
  Boundary elements are zero (Dirichlet). Interior elements computed from body.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :out-param sym :dtype kw :radius int}."
  [form & {:keys [dtype kernel-name-prefix scalar-types]
           :or {dtype :double kernel-name-prefix "par_stencil"
                scalar-types {}}}]
  (let [info (par/extract-par-stencil-info form)
        {:keys [out radius boundary cast idx body]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        ;; Collect arrays from body (includes in-arrays references)
        array-syms (ce/collect-arrays-in-body body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        arr-params (vec (sort-by name array-syms))
        scl-params (vec (sort-by name scalar-syms))
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        scl-type (fn [s] (ce/scalar-native-type s scalar-types ctype))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param-str (str "__global " ctype "* " (ce/c-symbol out))
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param-str scl-param-str "int _n_bound"]))
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype]
                   (ce/emit-expr adapted-body idx (set (map #(symbol (name %)) arr-params))))
        cast-str (if cast (str "(" (name cast) ")(" body-str ")") body-str)
        out-c (ce/c-symbol out)
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                    (case boundary
                      :dirichlet
                      (str "        if (idx < " radius " || idx >= _n_bound - " radius ") {\n"
                           "            " out-c "[idx] = 0.0;\n"
                           "        } else {\n"
                           "            " out-c "[idx] = " cast-str ";\n"
                           "        }\n")
                      ;; Default: no boundary handling (caller ensures valid range)
                      (str "        " out-c "[idx] = " cast-str ";\n"))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :out-param out
     :radius radius
     :dtype dtype}))

;; ================================================================
;; Scatter kernel generator
;; ================================================================

(defn generate-par-scatter-kernel
  "Generate an OpenCL C kernel from a raster.par/scatter! form.
  Scatter-add: output[index[i]] += src[i] (using atomics for correctness).

  For unstrided: one work-item per source element.
  For strided: one work-item per source element, inner loop over stride.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [{:name sym :type kw} ...] :dtype kw :strided? bool}."
  [form & {:keys [dtype kernel-name-prefix]
           :or {dtype :float kernel-name-prefix "par_scatter"}}]
  (let [info (par/extract-par-scatter-info form)
        {:keys [out src index stride]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "float")
        use-fp64? (= dtype :double)
        strided? (some? stride)
        out-c (ce/c-symbol out)
        src-c (ce/c-symbol src)
        index-c (ce/c-symbol index)
        ;; Scatter needs float atomic CAS for float types, native atomic for int
        needs-float-atomic? (contains? #{:float :double} dtype)
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                    (when needs-float-atomic? atomic-add-float-helper)
                    "__kernel void " kernel-name
                    "(__global " ctype "* " out-c
                    ", __global const " ctype "* restrict " src-c
                    ", __global const int* restrict " index-c
                    (if strided?
                      (str ", int _n_bound, int stride")
                      ", int _n_bound")
                    ") {\n"
                    "    for (int i = get_global_id(0); i < _n_bound; i += get_global_size(0)) {\n"
                    "        int dst_idx = " index-c "[i];\n"
                    (if strided?
                      (str "        for (int d = 0; d < stride; d++) {\n"
                           "            int src_pos = i * stride + d;\n"
                           "            int dst_pos = dst_idx * stride + d;\n"
                           (if needs-float-atomic?
                             (str "            atomic_add_float(&" out-c "[dst_pos], " src-c "[src_pos]);\n")
                             (str "            atomic_add(&" out-c "[dst_pos], " src-c "[src_pos]);\n"))
                           "        }\n")
                      (if needs-float-atomic?
                        (str "        atomic_add_float(&" out-c "[dst_idx], " src-c "[i]);\n")
                        (str "        atomic_add(&" out-c "[dst_idx], " src-c "[i]);\n")))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params [out src index]
     :scalar-params (if strided?
                      [{:name 'n :type :int} {:name 'stride :type :int}]
                      [{:name 'n :type :int}])
     :strided? strided?
     :dtype dtype}))

;; ================================================================
;; Reduce-by-key kernel generator
;; ================================================================

(defn generate-par-reduce-by-key-kernel
  "Generate an OpenCL C kernel from a raster.par/reduce-by-key form.
  Segmented reduction: output[keys[i]] op= values[i] using atomics.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [{:name sym :type kw} ...] :dtype kw}."
  [form & {:keys [dtype kernel-name-prefix]
           :or {dtype :float kernel-name-prefix "par_reduce_by_key"}}]
  (let [info (par/extract-par-reduce-by-key-info form)
        {:keys [out keys vals]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "float")
        use-fp64? (= dtype :double)
        out-c (ce/c-symbol out)
        keys-c (ce/c-symbol keys)
        vals-c (ce/c-symbol vals)
        needs-float-atomic? (contains? #{:float :double} dtype)
        ;; Only + is supported for atomic reduce-by-key (most common case)
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                    (when needs-float-atomic? atomic-add-float-helper)
                    "__kernel void " kernel-name
                    "(__global " ctype "* " out-c
                    ", __global const int* restrict " keys-c
                    ", __global const " ctype "* restrict " vals-c
                    ", int _n_bound) {\n"
                    "    for (int i = get_global_id(0); i < _n_bound; i += get_global_size(0)) {\n"
                    "        int k = " keys-c "[i];\n"
                    (if needs-float-atomic?
                      (str "        atomic_add_float(&" out-c "[k], " vals-c "[i]);\n")
                      (str "        atomic_add(&" out-c "[k], " vals-c "[i]);\n"))
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params [out keys vals]
     :scalar-params [{:name 'n :type :int}]
     :dtype dtype}))

;; ================================================================
;; Compound kernel codegen
;; ================================================================

(defn- emit-compound-phase-body
  "Emit OpenCL C for a single phase inside a compound kernel.
  Returns a string of C statements."
  [phase idx-var array-syms ctype]
  (let [{:keys [type out body idx]} phase
        out-name (ce/c-symbol out)
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype]
                   (ce/emit-expr body idx (set (map #(symbol (name %)) array-syms)) idx-var))]
    (case type
      :stencil
      (let [n-var "_n_bound"]
        (str "    // stencil → " out-name "\n"
             "    if (" idx-var " == 0 || " idx-var " == " n-var " - 1) {\n"
             "        " out-name "[" idx-var "] = 0.0;\n"
             "    } else {\n"
             "        " out-name "[" idx-var "] = " body-str ";\n"
             "    }\n"
             "    barrier(CLK_LOCAL_MEM_FENCE);\n"))

      :map
      (str "    // map → " out-name "\n"
           "    " out-name "[" idx-var "] = " body-str ";\n"
           "    barrier(CLK_LOCAL_MEM_FENCE);\n")

      ;; reduce — not emitted in :local compound (stays on host)
      :reduce
      (str "    // reduce (host-side) — skipped in compound kernel\n"))))

(defn generate-compound-local-kernel
  "Generate a single OpenCL C kernel for a compound kernel (:local strategy).
  All scratch arrays become __local, inputs/outputs are __global.

  Returns {:kernel-name str :source str :array-params [syms]
           :scalar-params [syms] :dtype kw}."
  [metadata & {:keys [dtype kernel-name-prefix device-id]
               :or {dtype :double kernel-name-prefix "compound_local"}}]
  (let [{:keys [inputs outputs scratch phases scalars]} metadata
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        ;; All arrays involved in phases
        all-arrays (concat inputs outputs scratch)
        ;; Parameter lists
        global-inputs (vec inputs)
        global-outputs (vec outputs)
        ;; Build kernel params
        input-params-str (str/join ", "
                                   (map (fn [s] (str "__global const " ctype "* restrict "
                                                     (ce/c-symbol s) "_global"))
                                        global-inputs))
        output-params-str (str/join ", "
                                    (map (fn [s] (str "__global " ctype "* restrict "
                                                      (ce/c-symbol s) "_global"))
                                         global-outputs))
        scalar-params-str (str/join ", "
                                    (map (fn [s] (str ctype " " (ce/c-symbol s)))
                                         scalars))
        all-params (str/join ", "
                             (remove empty?
                                     [input-params-str output-params-str scalar-params-str
                                      "int nsteps" "int _n_bound"]))
        ;; __local declarations — use fixed max size since OpenCL C
        ;; requires compile-time constants for __local array sizes.
        ;; _n_bound param + if(i>=_n_bound) return handles actual bounds.
        max-wg (if device-id
                 (try (or (get-in (hw/device device-id)
                                  [:capabilities :max-workgroup-size])
                          1024)
                      (catch Exception _ 1024))
                 1024)
        local-decls (str/join "\n"
                              (map (fn [s]
                                     (str "    __local " ctype " " (ce/c-symbol s) "[" max-wg "];"))
                                   all-arrays))
        ;; Copy inputs AND outputs from __global to __local
        ;; (outputs are read+write, e.g. u is both input and output)
        copy-in (str/join "\n"
                          (map (fn [s]
                                 (str "    " (ce/c-symbol s) "[i] = " (ce/c-symbol s) "_global[i];"))
                               (concat global-inputs global-outputs)))
        ;; Copy outputs from __local to __global
        copy-out (str/join "\n"
                           (map (fn [s]
                                  (str "    " (ce/c-symbol s) "_global[i] = " (ce/c-symbol s) "[i];"))
                                global-outputs))
        ;; Phase bodies (only map and stencil — reduce stays on host)
        non-reduce-phases (filter #(not= :reduce (:type %)) phases)
        ;; Each phase must be separated by a barrier so all work items
        ;; complete phase N before any work item starts phase N+1.
        ;; Without barriers, stencil/map phases that read __local arrays
        ;; written by a prior phase see stale values from other work items.
        phase-bodies (str/join "\n        barrier(CLK_LOCAL_MEM_FENCE);\n"
                               (map #(emit-compound-phase-body % "i"
                                                               (set (map symbol (map name all-arrays)))
                                                               ctype)
                                    non-reduce-phases))
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    int i = get_local_id(0);\n"
                    "    if (i >= _n_bound) return;\n"
                    local-decls "\n"
                    "    // Copy inputs to __local\n"
                    copy-in "\n"
                    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                    "    // Time-stepping loop\n"
                    "    for (int step = 0; step < nsteps; step++) {\n"
                    phase-bodies
                    "    }\n"
                    "    // Write back outputs\n"
                    copy-out "\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params (vec (concat global-inputs global-outputs))
     :scalar-params (vec scalars)
     :dtype dtype
     :strategy :local}))

(defn generate-compound-global-kernels
  "Generate per-phase OpenCL C kernels for a compound kernel (:global strategy).
  Each phase gets its own kernel. The time loop stays on the host.

  Returns a vector of kernel descriptors."
  [metadata & {:keys [dtype kernel-name-prefix]
               :or {dtype :double kernel-name-prefix "compound_global"}}]
  (let [{:keys [phases]} metadata
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)]
    (vec
     (keep-indexed
      (fn [idx phase]
        (when (not= :reduce (:type phase))
          (let [{:keys [type out body inputs]} phase
                kernel-name (str kernel-name-prefix "_phase" idx "_" (gensym ""))
                in-arrays (vec (sort-by name inputs))
                all-arrays-in-body (ce/collect-arrays-in-body body)
                scalar-syms (ce/collect-scalars-in-body body (:idx phase) all-arrays-in-body)
                scl-params (vec (sort-by name scalar-syms))
                arr-param-str (str/join ", "
                                        (map (fn [s] (str "__global const " ctype "* restrict "
                                                          (ce/c-symbol s)))
                                             in-arrays))
                out-param (str "__global " ctype "* restrict " (ce/c-symbol out))
                scl-param-str (str/join ", "
                                        (map (fn [s] (str ctype " " (ce/c-symbol s)))
                                             scl-params))
                all-params (str/join ", "
                                     (remove empty?
                                             [arr-param-str out-param scl-param-str "int _n_bound"]))
                body-str (binding [ce/*emit-config* ce/opencl-config
                                   ce/*scalar-type* ctype]
                           (ce/emit-expr body (:idx phase)
                                         (set (map #(symbol (name %)) in-arrays))))
                kernel-body
                (case type
                  :stencil
                  (str "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                       "        if (idx == 0 || idx == _n_bound - 1) {\n"
                       "            " (ce/c-symbol out) "[idx] = 0.0;\n"
                       "        } else {\n"
                       "            " (ce/c-symbol out) "[idx] = " body-str ";\n"
                       "        }\n"
                       "    }\n")

                  :map
                  (str "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                       "        " (ce/c-symbol out) "[idx] = " body-str ";\n"
                       "    }\n"))
                source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                            "__kernel void " kernel-name
                            "(" all-params ") {\n"
                            kernel-body
                            "}\n")]
            {:kernel-name kernel-name
             :source source
             :array-params in-arrays
             :scalar-params scl-params
             :out-sym out
             :dtype dtype
             :phase-idx idx
             :phase-type type
             :strategy :global})))
      phases))))

(defn emit-compound-kernel-invocation
  "Emit S-expression for compound kernel invocation.
  For :local — single kernel call wrapping the dotimes.
  For :global — host-side dotimes calling per-phase kernels."
  [metadata kernels]
  (let [{:keys [execution trip-count-bound inputs outputs scratch phases]} metadata
        strategy (:strategy execution)
        ;; Use first available array for size — outputs then inputs then scratch
        size-array (or (first outputs) (first inputs) (first scratch))
        n-expr (if size-array
                 (list 'clojure.core/alength size-array)
                 (:parallel-bound execution))]
    (case strategy
      :local
      (let [kernel (first kernels)
            {:keys [kernel-name scalar-params]} kernel]
        (list 'raster.gpu.ze-runtime/invoke-compound-kernel
              kernel-name
              (vec (concat inputs outputs))
              (vec scalar-params)
              trip-count-bound
              n-expr))

      :global
      ;; Host-side loop with per-phase kernel launches
      ;; Allocate device buffers, copy in, loop, copy out
      (let [phase-kernels (filter #(not= :reduce (:type (nth phases (:phase-idx %)))) kernels)]
        (list 'raster.gpu.ze-runtime/invoke-compound-global
              (mapv :kernel-name phase-kernels)
              (vec inputs)
              (vec outputs)
              (vec scratch)
              (mapv (fn [k] {:arrays (:array-params k)
                             :out (:out-sym k)
                             :scalars (:scalar-params k)})
                    phase-kernels)
              trip-count-bound
              n-expr)))))

;; ================================================================
;; Pipeline pass
;; ================================================================

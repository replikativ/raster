(ns raster.compiler.backend.gpu.opencl-codegen
  "S-expression to OpenCL C source emission.

  Converts walked S-expressions into OpenCL C kernel source code.
  Adapted from cuda_codegen.clj with OpenCL-specific kernel boilerplate:
  - `__kernel void` instead of `extern \"C\" __global__ void`
  - `get_global_id(0)` instead of `blockIdx.x * blockDim.x + threadIdx.x`
  - `__global` address space qualifiers
  - `barrier(CLK_LOCAL_MEM_FENCE)` instead of `__syncthreads()`
  - `intel_sub_group_shuffle_down` for Intel GPU subgroup ops

  Expression emission delegates to c_emit.clj shared utilities.

  Usage:
    (emit-elementwise-kernel \"relu\" :double '(fmax 0.0 x))
    (emit-reduction-kernel \"sum\" :double '+ 0.0)
    (kernel-launch-config 100000 :device-id :ze:0)"
  (:require [clojure.string :as str]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.backend.gpu.c-emit :as ce]))

;; ================================================================
;; OpenCL-specific op map (extends shared op-map)
;; ================================================================

(def opencl-op-map
  "Maps Clojure/Java op symbols to OpenCL C equivalents.
  OpenCL C math functions are identical to C99 — same as CUDA."
  ce/op-map)

(def opencl-type-map
  "Maps Raster type keywords to OpenCL C types. Derived from the single faceted
   dtype/native-types. OpenCL 'long' is 64-bit (unlike CUDA's 'long long')."
  (dtype/backend-types :opencl))

;; ================================================================
;; Expression emission (delegates to shared)
;; ================================================================

(defn emit-expr
  "Convert an S-expression to an OpenCL C expression string."
  [expr]
  (ce/emit-expr expr nil #{} "idx"))

;; ================================================================
;; OpenCL pragmas
;; ================================================================

(defn extension-pragmas
  "OpenCL extension-enable pragmas required by the given dtypes.

  Derived from the dtype-info `:needs-pragma` facet (:double → cl_khr_fp64,
  :half → cl_khr_fp16 — new dtypes get their pragma for free). Emits one
  guarded `#pragma OPENCL EXTENSION <ext> : enable` block per distinct
  required extension; nil dtypes are ignored. Returns \"\" when none needed."
  [& dtypes]
  (->> dtypes
       (keep dtype/needs-pragma-for)
       distinct
       (map (fn [ext]
              (let [e (name ext)]
                (str "#if defined(" e ")\n"
                     "#pragma OPENCL EXTENSION " e " : enable\n"
                     "#endif\n"))))
       (apply str)))

(def ^:private subgroup-pragma
  "Enable Intel subgroup extensions for shuffle ops."
  "#if defined(cl_khr_subgroups)\n#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n#elif defined(cl_intel_subgroups)\n#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n#endif\n")

;; ================================================================
;; Kernel templates
;; ================================================================

(defn emit-elementwise-kernel
  "Generate an OpenCL C element-wise kernel with grid-stride loop.

  kernel-name: string name for the kernel
  dtype: :double or :float
  body-expr: S-expression for element transform, using 'x' for input element
  n-arrays: number of input arrays (default 1)

  Returns an OpenCL C source string."
  [kernel-name dtype body-expr & {:keys [n-arrays] :or {n-arrays 1}}]
  (let [ctype (get opencl-type-map dtype "double")
        var-names ["x" "y" "z" "w"]
        array-params (str/join ", "
                               (map (fn [i] (str "__global const " ctype "* restrict arr" i))
                                    (range n-arrays)))
        reads (str/join "\n"
                        (map (fn [i]
                               (str "        " ctype " " (get var-names i (str "v" i))
                                    " = arr" i "[idx];"))
                             (range n-arrays)))]
    (str (extension-pragmas dtype)
         "__kernel void " kernel-name "("
         array-params ", __global " ctype "* restrict out, int n) {\n"
         "    for (int idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n"
         reads "\n"
         "        out[idx] = " (emit-expr body-expr) ";\n"
         "    }\n"
         "}\n")))

(defn emit-fused-kernel
  "Generate a fused OpenCL C kernel from a chain of element-wise ops.

  kernel-name: string name
  dtype: :double or :float
  ops: seq of {:op op-sym, :result-sym sym, :arg-syms [syms...]}

  Returns OpenCL C source string."
  [kernel-name dtype ops]
  (let [ctype (get opencl-type-map dtype "double")
        body-lines (mapv (fn [{:keys [op result-sym arg-syms]}]
                           (str "        " ctype " "
                                (emit-expr result-sym)
                                " = " (emit-expr (cons op arg-syms)) ";"))
                         ops)]
    (str (extension-pragmas dtype)
         "__kernel void " kernel-name
         "(__global const double* restrict in, __global double* restrict out, int n) {\n"
         "    for (int idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n"
         "        " ctype " x = in[idx];\n"
         (str/join "\n" body-lines) "\n"
         "        out[idx] = "
         (emit-expr (:result-sym (last ops))) ";\n"
         "    }\n"
         "}\n")))

(defn emit-reduction-kernel
  "Generate an OpenCL C parallel reduction kernel.
  Uses local memory tree reduction with Intel subgroup shuffle optimization.

  kernel-name: string name
  dtype: :double or :float
  op: '+, '*, 'Math/max, 'Math/min
  identity-val: identity element for the reduction op
  workgroup-size: compile-time workgroup size (required for local memory sizing)

  Returns OpenCL C source string."
  [kernel-name dtype op identity-val & {:keys [workgroup-size] :or {workgroup-size 256}}]
  (let [ctype (get opencl-type-map dtype "double")
        reduce-op (condp = op
                    '+ "+"
                    '* "*"
                    'Math/max "fmax"
                    'Math/min "fmin"
                    "+")
        combine (ce/combine-fn reduce-op (ce/fn-style-reduction-op? op))]
    (str (extension-pragmas dtype)
         subgroup-pragma
         "__kernel void " kernel-name
         "(__global const " ctype "* restrict input, "
         "__global " ctype "* restrict output, int n) {\n"
         "    __local " ctype " sdata[" workgroup-size "];\n"
         "    int tid = get_local_id(0);\n"
         "    " ctype " val = " (ce/normalize-identity-val identity-val) ";\n"
         "    int stride = get_global_size(0);\n"
         "    int i = get_global_id(0);\n"
         ;; 4x unrolled grid-stride accumulation
         "    for (; i + 3 * stride < n; i += 4 * stride) {\n"
         "        val = " (combine "val" "input[i]") ";\n"
         "        val = " (combine "val" "input[i + stride]") ";\n"
         "        val = " (combine "val" "input[i + 2 * stride]") ";\n"
         "        val = " (combine "val" "input[i + 3 * stride]") ";\n"
         "    }\n"
         "    for (; i < n; i += stride) {\n"
         "        val = " (combine "val" "input[i]") ";\n"
         "    }\n"
         "    sdata[tid] = val;\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         ;; Tree reduction in local memory
         "    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
         "        if (tid < s) {\n"
         "            sdata[tid] = " (combine "sdata[tid]" "sdata[tid + s]") ";\n"
         "        }\n"
         "        barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    }\n"
         "    if (tid == 0) output[get_group_id(0)] = sdata[0];\n"
         "}\n")))

(defn emit-par-map-kernel
  "Generate an OpenCL C kernel from a raster.par/map! body expression.

  kernel-name: string name for the kernel
  dtype: :double or :float
  body-expr: S-expression for element transform
  array-params: seq of array symbol names used as inputs
  scalar-params: seq of scalar symbol names

  Returns OpenCL C source string."
  [kernel-name dtype body-expr array-params scalar-params]
  (let [ctype (get opencl-type-map dtype "double")
        arr-params-str (str/join ", "
                                 (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s)))
                                      array-params))
        scalar-params-str (str/join ", "
                                    (map (fn [s] (str ctype " " (ce/c-symbol s)))
                                         scalar-params))
        all-params (str/join ", "
                             (remove empty?
                                     [arr-params-str
                                      (str "__global " ctype "* restrict out")
                                      scalar-params-str
                                      "int n"]))]
    (str (extension-pragmas dtype)
         "__kernel void " kernel-name "("
         all-params ") {\n"
         "    for (int idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n"
         ;; Array reads
         (str/join ""
                   (map-indexed (fn [i s]
                                  (str "        " ctype " " (ce/c-symbol s)
                                       "_val = " (ce/c-symbol s) "[idx];\n"))
                                array-params))
         "        out[idx] = " (emit-expr body-expr) ";\n"
         "    }\n"
         "}\n")))

(defn emit-par-reduce-kernel
  "Generate an OpenCL C parallel reduction kernel from a raster.par/reduce form.

  kernel-name: string name
  dtype: :double or :float
  op: reduction operation (+, *, Math/max, Math/min)
  identity-val: identity element

  Returns OpenCL C source string."
  [kernel-name dtype op identity-val & opts]
  (apply emit-reduction-kernel kernel-name dtype op identity-val opts))

;; ================================================================
;; Axpy kernel (in-place weight update)
;; ================================================================

(defn emit-axpy-kernel
  "Generate an OpenCL axpy kernel: y[i] += alpha * x[i].
  Used for SGD weight updates on GPU with persistent DeviceBuffers.

  kernel-name: string name
  dtype: :double or :float (default :float)
  Returns OpenCL C source string."
  [kernel-name & {:keys [dtype] :or {dtype :float}}]
  (let [ctype (get opencl-type-map dtype "float")]
    (str (extension-pragmas dtype)
         "__kernel void " kernel-name
         "(__global " ctype "* restrict y,"
         " __global const " ctype "* restrict x,"
         " " ctype " alpha, int n) {\n"
         "    for (int i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
         "        y[i] += alpha * x[i];\n"
         "    }\n"
         "}\n")))

;; ================================================================

(defn emit-silu-kernel
  "Generate a fused SiLU kernel: y = x * sigmoid(x) = x / (1 + exp(-x))."
  [kernel-name dtype]
  (emit-elementwise-kernel kernel-name dtype '(/ x (+ 1.0 (exp (- x))))))

;; ================================================================
;; Non-square XMX GEMM kernel — tile-parametric generator
;; ================================================================



(defn gemm-launch-config
  "Compute 2D launch config for GEMM kernel.
  Returns {:workgroup-size [256 1] :group-count [gc-n gc-m]}
  where 256 = 16 subgroups × 16 work-items."
  [m n]
  (let [gc-m (int (Math/ceil (/ (double m) 128.0)))
        gc-n (int (Math/ceil (/ (double n) 128.0)))]
    {:workgroup-size [256 1]
     :group-count [gc-n gc-m]}))

;; ================================================================
;; Kernel launch config (Level Zero)
;; ================================================================

(def ^:private min-ze-elements
  "Minimum elements before Level Zero kernel is worthwhile."
  4096)

(defn kernel-launch-config
  "Compute workgroup/group-count dimensions for n elements on Level Zero.
  Returns {:workgroup-size int, :group-count int, :local-mem int} or nil if n too small.

  Options:
    :device-id    — Level Zero device keyword (e.g. :ze:0)
    :reduction?   — if true, uses larger workgroups and allocates local memory
    :min-elements — override minimum element threshold"
  [n & {:keys [device-id reduction? min-elements]
        :or {reduction? false min-elements min-ze-elements}}]
  (when (>= n min-elements)
    (if device-id
      (try
        (require 'raster.runtime.hardware 'raster.compiler.core.hardware)
        ((resolve 'raster.runtime.hardware/init!))
        (let [desc ((resolve 'raster.compiler.core.hardware/descriptor-for) device-id)
              hw-wg ((resolve 'raster.compiler.core.hardware/workgroup-size)
                     desc n :reduction? reduction?)
              hw-gc ((resolve 'raster.compiler.core.hardware/group-count)
                     desc n hw-wg :reduction? reduction?)
              local-mem (if reduction? (* hw-wg 8) 0)]
          {:workgroup-size hw-wg :group-count hw-gc :local-mem local-mem})
        (catch Exception _
          (let [wg (if reduction? 256 256)
                gc (int (Math/ceil (/ (double n) (double wg))))]
            {:workgroup-size wg :group-count gc :local-mem (if reduction? (* wg 8) 0)})))
      (let [wg (if reduction? 256 256)
            gc (int (Math/ceil (/ (double n) (double wg))))]
        {:workgroup-size wg :group-count gc :local-mem (if reduction? (* wg 8) 0)}))))

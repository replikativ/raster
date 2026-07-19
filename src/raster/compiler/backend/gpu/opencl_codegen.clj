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
            [raster.compiler.core.layout :as layout]
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
;; XMX Matrix Multiply kernel (Intel Xe Matrix Extensions)
;; ================================================================

;; ================================================================
;; Row-softmax kernel (attention score normalization)
;; ================================================================

(defn emit-row-softmax-kernel
  "Generate an OpenCL row-wise softmax kernel.
  Input: [n_rows, row_len] row-major array.
  Output: [n_rows, row_len] with softmax applied per row.

  Algorithm: 3-pass (max, exp+sum, normalize) with cooperative reduction.
  One workgroup per row, subgroup cooperative for max and sum.

  kernel-name: string name
  dtype: :double or :float
  Options:
    :workgroup-size — threads per row (default 256)"
  [kernel-name dtype & {:keys [workgroup-size] :or {workgroup-size 256}}]
  (let [ctype (get opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        neg-inf (if use-fp64? "-1.0e308" "-1.0e38f")]
    (str (extension-pragmas dtype)
         subgroup-pragma
         "\n"
         "// Row-wise softmax: one workgroup per row\n"
         "__kernel void " kernel-name "(\n"
         "    __global const " ctype "* restrict input,\n"
         "    __global " ctype "* restrict output,\n"
         "    int n_rows, int row_len) {\n"
         "    __local " ctype " sdata[" workgroup-size "];\n"
         "    int row = get_group_id(0);\n"
         "    if (row >= n_rows) return;\n"
         "    int tid = get_local_id(0);\n"
         "    int offset = row * row_len;\n"
         "\n"
         "    // Pass 1: find row max\n"
         "    " ctype " max_val = " neg-inf ";\n"
         "    for (int j = tid; j < row_len; j += " workgroup-size ") {\n"
         "        max_val = fmax(max_val, input[offset + j]);\n"
         "    }\n"
         "    sdata[tid] = max_val;\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    for (int s = " (/ workgroup-size 2) "; s > 0; s >>= 1) {\n"
         "        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);\n"
         "        barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    }\n"
         "    " ctype " row_max = sdata[0];\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "\n"
         "    // Pass 2: exp(x - max) and accumulate sum\n"
         "    " ctype " sum_val = 0;\n"
         "    for (int j = tid; j < row_len; j += " workgroup-size ") {\n"
         "        " ctype " e = exp(input[offset + j] - row_max);\n"
         "        output[offset + j] = e;\n"
         "        sum_val += e;\n"
         "    }\n"
         "    sdata[tid] = sum_val;\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    for (int s = " (/ workgroup-size 2) "; s > 0; s >>= 1) {\n"
         "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
         "        barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    }\n"
         "    " ctype " inv_sum = 1 / sdata[0];\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "\n"
         "    // Pass 3: normalize\n"
         "    for (int j = tid; j < row_len; j += " workgroup-size ") {\n"
         "        output[offset + j] *= inv_sum;\n"
         "    }\n"
         "}\n")))

;; ================================================================
;; GroupNorm kernel (channel normalization)
;; ================================================================

(defn emit-group-norm-kernel
  "Generate an OpenCL GroupNorm kernel.
  Input: [n, channels] (flattened from [batch, channels, spatial]).
  Gamma, Beta: [channels] (affine parameters).
  num_groups groups, each group has channels/num_groups channels.

  Two-pass: compute group stats, then normalize + scale/shift.
  One workgroup per (sample, group) pair.

  kernel-name: string name
  dtype: :double or :float
  Options:
    :workgroup-size — threads per group normalization (default 256)
    :eps           — epsilon for stability (default 1e-5)"
  [kernel-name dtype & {:keys [workgroup-size eps] :or {workgroup-size 256 eps 1e-5}}]
  (let [ctype (get opencl-type-map dtype "double")
        eps-str (str eps)]
    (str (extension-pragmas dtype)
         "\n"
         "// GroupNorm: normalize within groups of channels\n"
         "// Layout: x[batch, channels], gamma[channels], beta[channels]\n"
         "__kernel void " kernel-name "(\n"
         "    __global const " ctype "* restrict input,\n"
         "    __global const " ctype "* restrict gamma,\n"
         "    __global const " ctype "* restrict beta,\n"
         "    __global " ctype "* restrict output,\n"
         "    int batch_size, int channels, int num_groups) {\n"
         "    __local " ctype " sdata[" (* 2 workgroup-size) "];\n"  ;; [mean_part | var_part]
         "    int pair_id = get_group_id(0);\n"  ;; (sample, group) pair
         "    int sample = pair_id / num_groups;\n"
         "    int group = pair_id % num_groups;\n"
         "    if (sample >= batch_size) return;\n"
         "    int tid = get_local_id(0);\n"
         "    int group_size = channels / num_groups;\n"
         "    int ch_start = group * group_size;\n"
         "    int base = sample * channels;\n"
         "\n"
         "    // Pass 1: compute group mean\n"
         "    " ctype " sum = 0;\n"
         "    for (int c = tid; c < group_size; c += " workgroup-size ") {\n"
         "        sum += input[base + ch_start + c];\n"
         "    }\n"
         "    sdata[tid] = sum;\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    for (int s = " (/ workgroup-size 2) "; s > 0; s >>= 1) {\n"
         "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
         "        barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    }\n"
         "    " ctype " mean = sdata[0] / (" ctype ")(group_size);\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "\n"
         "    // Pass 2: compute group variance\n"
         "    " ctype " var_sum = 0;\n"
         "    for (int c = tid; c < group_size; c += " workgroup-size ") {\n"
         "        " ctype " diff = input[base + ch_start + c] - mean;\n"
         "        var_sum += diff * diff;\n"
         "    }\n"
         "    sdata[tid] = var_sum;\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    for (int s = " (/ workgroup-size 2) "; s > 0; s >>= 1) {\n"
         "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
         "        barrier(CLK_LOCAL_MEM_FENCE);\n"
         "    }\n"
         "    " ctype " inv_std = 1 / sqrt(sdata[0] / (" ctype ")(group_size) + " eps-str ");\n"
         "    barrier(CLK_LOCAL_MEM_FENCE);\n"
         "\n"
         "    // Pass 3: normalize + affine\n"
         "    for (int c = tid; c < group_size; c += " workgroup-size ") {\n"
         "        int ch = ch_start + c;\n"
         "        " ctype " x_hat = (input[base + ch] - mean) * inv_std;\n"
         "        output[base + ch] = gamma[ch] * x_hat + beta[ch];\n"
         "    }\n"
         "}\n")))

;; ================================================================
;; SiLU elementwise kernel (fused activation)
;; ================================================================
;; Transpose kernel
;; ================================================================

(defn- render-c-index
  "Render a layout->offset index S-expression to C infix (matches the hand-written kernel strings:
   `(+ (* i cols) j)` → \"i * cols + j\"). Index arithmetic only (+ and *, symbols/ints)."
  [e]
  (cond
    (symbol? e) (name e)
    (number? e) (str e)
    (seq? e)    (let [[op a b] e]
                  (str (render-c-index a) " " (case op + "+" * "*" (str op)) " " (render-c-index b)))
    :else       (str e)))

(defn emit-transpose-kernel
  "Generate an OpenCL 2D matrix transpose kernel. LAYOUT-DRIVEN: the read/write indices are computed
   from the layout facet (`layout/layout->offset`), not hand-written — a transpose is exactly a
   `convert_layout` between a row-major input [rows,cols] and its transposed output [cols,rows]. The
   emitted string is byte-identical to the former hand-written `out[j*rows+i] = in[i*cols+j]`; this
   is the first place a layout DRIVES codegen (the Stage-1 transpose-as-convert seam, proven here).

  dtype: :double | :float | :half (default :float). Returns OpenCL C source string."
  [kernel-name & {:keys [dtype] :or {dtype :float}}]
  (let [ctype  (if (= dtype :half) "half" (get opencl-type-map dtype "float"))
        in-l   (layout/row-major '[rows cols] dtype)          ;; input row-major
        out-l  (layout/row-major '[cols rows] dtype)          ;; output row-major, transposed extents
        in-idx  (render-c-index (layout/layout->offset in-l  '[i j]))   ;; "i * cols + j"
        out-idx (render-c-index (layout/layout->offset out-l '[j i]))]  ;; "j * rows + i"
    (str (extension-pragmas dtype)
         "__kernel void " kernel-name
         "(__global const " ctype "* restrict in,"
         " __global " ctype "* restrict out,"
         " int rows, int cols) {\n"
         "    int gid = get_global_id(0);\n"
         "    int total = rows * cols;\n"
         "    for (int idx = gid; idx < total; idx += get_global_size(0)) {\n"
         "        int i = idx / cols;\n"
         "        int j = idx % cols;\n"
         "        out[" out-idx "] = in[" in-idx "];\n"
         "    }\n"
         "}\n")))

;; ================================================================
;; Scalar (non-XMX) GEMM kernel — small-N fallback
;; ================================================================

(defn emit-gemm-scalar-kernel
  "Generate a plain scalar (non-XMX) f32 GEMM kernel for output-column dims too
  small for the XMX 2D-block path: Intel 2D-block IO requires a >=16-byte pitch,
  which at fp16 means N>=8 — N in {2,4,6} reads a garbage B tile. This kernel
  reads the f32 resident buffers DIRECTLY (no f16 convert/transpose expansion),
  one work-item per C element, grid-stride over m*n, scalar k-loop.

    :nn  C[m,n] = A[m,k] * B[k,n]
    :nt  C[m,n] = A[m,k] * B[n,k]^T   (B stored row-major [n,k])
    :tn  C[m,n] = A[k,m]^T * B[k,n]   (A stored row-major [k,m])

  kernel-name: string name. variant: :nn | :nt | :tn (default :nn).
  Returns OpenCL C source string."
  [kernel-name & {:keys [variant] :or {variant :nn}}]
  (let [a-idx (case variant :tn "A[p * m + i]" "A[i * k + p]")
        b-idx (case variant :nt "B[j * k + p]" "B[p * n + j]")]
    (str "__kernel void " kernel-name
         "(__global const float* restrict A,"
         " __global const float* restrict B,"
         " __global float* restrict C,"
         " int m, int n, int k) {\n"
         "    int total = m * n;\n"
         "    for (int idx = get_global_id(0); idx < total; idx += get_global_size(0)) {\n"
         "        int i = idx / n;\n"
         "        int j = idx % n;\n"
         "        float acc = 0.0f;\n"
         "        for (int p = 0; p < k; ++p) acc += " a-idx " * " b-idx ";\n"
         "        C[idx] = acc;\n"
         "    }\n"
         "}\n")))

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
;; Scatter-reduce kernel (graph attention core)
;; ================================================================

(defn emit-scatter-reduce-kernel
  "Generate an OpenCL scatter-reduce kernel for graph attention.
  out[dst[e]] += f(src_data[src[e]]) for each edge e.

  Two variants:
    :atomic  — uses atomic_add, works for any edge ordering
    :sorted  — edges sorted by dst, one workgroup per segment (no atomics)

  For :atomic with FP16, emulates via CAS loop on ushort.

  kernel-name: string name
  dtype: :double or :float
  variant: :atomic (default) or :sorted
  Options:
    :with-weights? — if true, multiply by weight array: out[dst[e]] += w[e] * src[src[e]*stride+d]
    :d-model      — feature dimension (inner loop over d)
    :workgroup-size — for :sorted variant"
  [kernel-name dtype variant
   & {:keys [with-weights? d-model workgroup-size]
      :or {with-weights? true d-model nil workgroup-size 256}}]
  (let [ctype (get opencl-type-map dtype "double")
        use-fp64? (= dtype :double)]
    (case variant
      :atomic
      (str (extension-pragmas dtype)
           "#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable\n"
           "\n"
           "// Scatter-reduce (atomic): out[dst[e]] += w[e] * data[src[e]*stride+d]\n"
           "__kernel void " kernel-name "(\n"
           "    __global const " ctype "* restrict data,\n"    ;; source node features [n_nodes, d_model]
           "    __global const int* restrict src_edges,\n"     ;; [n_edges]
           "    __global const int* restrict dst_edges,\n"     ;; [n_edges]
           (when with-weights?
             (str "    __global const " ctype "* restrict weights,\n"))
           "    __global " ctype "* restrict out,\n"           ;; [n_nodes, d_model]
           "    int n_edges, int stride) {\n"   ;; stride = d_model
           "    for (int e = get_global_id(0); e < n_edges; e += get_global_size(0)) {\n"
           "        int src = src_edges[e];\n"
           "        int dst = dst_edges[e];\n"
           (if with-weights?
             (str "        " ctype " w = weights[e];\n"
                  "        for (int d = 0; d < stride; d++) {\n"
                  "            " ctype " val = w * data[src * stride + d];\n")
             (str "        for (int d = 0; d < stride; d++) {\n"
                  "            " ctype " val = data[src * stride + d];\n"))
           ;; Atomic add — use native for double/float
           (if use-fp64?
             ;; CAS-based atomic add for double
             (str "            // CAS-based atomic add for double\n"
                  "            __global volatile long* addr = (__global volatile long*)(out + dst * stride + d);\n"
                  "            long old_val = *addr;\n"
                  "            long new_val;\n"
                  "            do {\n"
                  "                new_val = as_long(as_double(old_val) + val);\n"
                  "            } while (atom_cmpxchg(addr, old_val, new_val) != old_val);\n")
             ;; Native atomic for float (available on Xe2)
             (str "            atomic_add(out + dst * stride + d, val);\n"))
           "        }\n"
           "    }\n"
           "}\n")

      :sorted
      ;; Sorted-segment: edges grouped by dst, each workgroup processes one dst node
      (str (extension-pragmas dtype)
           "\n"
           "// Scatter-reduce (sorted): edges sorted by dst, one workgroup per dst segment\n"
           "__kernel void " kernel-name "(\n"
           "    __global const " ctype "* restrict data,\n"
           "    __global const int* restrict src_edges,\n"
           "    __global const int* restrict dst_edges,\n"
           (when with-weights?
             (str "    __global const " ctype "* restrict weights,\n"))
           "    __global " ctype "* restrict out,\n"
           "    __global const int* restrict seg_offsets,\n"   ;; [n_nodes+1] CSR-style
           "    int n_nodes, int stride) {\n"
           "    int node = get_group_id(0);\n"
           "    if (node >= n_nodes) return;\n"
           "    int seg_start = seg_offsets[node];\n"
           "    int seg_end = seg_offsets[node + 1];\n"
           "    int tid = get_local_id(0);\n"
           "    // Each thread handles a subset of the feature dimensions\n"
           "    for (int d = tid; d < stride; d += get_local_size(0)) {\n"
           "        " ctype " acc = 0;\n"
           "        for (int e = seg_start; e < seg_end; e++) {\n"
           "            int src = src_edges[e];\n"
           (if with-weights?
             (str "            acc += weights[e] * data[src * stride + d];\n")
             (str "            acc += data[src * stride + d];\n"))
           "        }\n"
           "        out[node * stride + d] = acc;\n"
           "    }\n"
           "}\n"))))

(defn emit-scatter-reduce-scalar-kernel
  "Generate scatter-reduce for scalar values (no feature dim loop).
  out[dst[e]] += value[e]. Used for normalization Z accumulation."
  [kernel-name dtype]
  (let [ctype (get opencl-type-map dtype "double")
        use-fp64? (= dtype :double)]
    (str (extension-pragmas dtype)
         (when use-fp64?
           "#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics : enable\n")
         "\n"
         "// Scatter-reduce scalar: out[dst[e]] += values[e]\n"
         "__kernel void " kernel-name "(\n"
         "    __global const " ctype "* restrict values,\n"
         "    __global const int* restrict dst_edges,\n"
         "    __global " ctype "* restrict out,\n"
         "    int n_edges) {\n"
         "    for (int e = get_global_id(0); e < n_edges; e += get_global_size(0)) {\n"
         "        int dst = dst_edges[e];\n"
         (if use-fp64?
           (str "        __global volatile long* addr = (__global volatile long*)(out + dst);\n"
                "        long old_val = *addr;\n"
                "        long new_val;\n"
                "        do {\n"
                "            new_val = as_long(as_double(old_val) + values[e]);\n"
                "        } while (atom_cmpxchg(addr, old_val, new_val) != old_val);\n")
           (str "        atomic_add(out + dst, values[e]);\n"))
         "    }\n"
         "}\n")))

;; ================================================================
;; Non-square XMX GEMM kernel — tile-parametric generator
;; ================================================================

(defn emit-gemm-tiled
  "Tile-PARAMETRIC XMX GEMM generator (C = A·B, FP16 in, FP32 accumulate). The tile geometry
   (workgroup BLOCK_M×BLOCK_N, per-subgroup SG_M×SG_N, K-step BLOCK_K, prefetch depth) is COMPUTED,
   not hardcoded; the DPAS instruction shape (M_i×N_i×K_i) + subgroup size come from the :matrix
   descriptor. The Arc default (128×128, 32×32, K 32, DPAS 8×16×16, subgroup 16) reproduces the
   original hand-unrolled kernel bit-for-bit (validated over plain/split-k/batched paths); varying
   the tile is bit-invariant since each output element's K-reduction is the same DPAS sequence. The
   Intel DPAS builtins are fixed here; a different vendor's matrix instruction (WGMMA/MFMA) is a
   separate emitter keyed by :matrix :family (the one genuine fork). Handles arbitrary M,N,K with
   full boundary checking; supports alpha/beta, c-dtype, split-k? (grid-z partials) and batched?
   (grid-z slabs)."
  [kernel-name & {:keys [block-m block-n sg-m sg-n block-k matrix alpha beta c-dtype split-k? batched? prefetch]
                  :or {block-m 128 block-n 128 sg-m 32 sg-n 32 block-k 32
                       matrix {:m 8 :n 16 :k 16 :subgroup 16}
                       alpha 1.0 beta 0.0 c-dtype :half split-k? false batched? false prefetch 3}}]
  (let [{mi :m ni :n ki :k sg :subgroup} matrix]
    (doseq [[nm a b] [["sg-m/M_i" sg-m mi] ["sg-n/N_i" sg-n ni] ["block-m/sg-m" block-m sg-m]
                      ["block-n/sg-n" block-n sg-n] ["block-k/K_i" block-k ki]]]
      (when-not (zero? (rem a b))
        (throw (ex-info (str "emit-gemm-tiled: " nm " not divisible (" a "/" b ")") {:tile [block-m block-n sg-m sg-n block-k] :matrix matrix}))))
    (when (and split-k? (not= beta 0.0))
      (throw (ex-info "split-k GEMM requires beta = 0" {:beta beta})))
    (when (and split-k? batched?)
      (throw (ex-info "batched and split-k both claim grid-z — mutually exclusive" {})))
    (let [nms (quot sg-m mi) nns (quot sg-n ni)      ;; M/N subtiles per subgroup
          ncols (quot block-n sg-n)                   ;; subgroup columns
          ksteps (quot block-k ki)
          c-dtype (if split-k? :float c-dtype)
          c-type (if (= c-dtype :float) "float" "half")
          store-cast (if (= c-dtype :float) "" "(half)")
          kend (if split-k? "k_hi" "K")
          accT (str "float" mi)
          ms (range nms) ns (range nns)
          amul (fn [m] (if (zero? m) "m_base" (str "m_base+" (* m mi))))
          kstep (fn [kpos]
                  (str "        { int pk = " kpos " + " (* prefetch ki) ";\n"
                       "          if (pk < " kend ") {\n"
                       (apply str (for [m ms] (str "            intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(pk, " (amul m) "));\n")))
                       "          } }\n"
                       (apply str (for [n ns] (str "        bp" n " = as_int8(intel_subgroup_block_read_transform_u16_k16((__global void*)B, b_wb, K, b_pb, (int2)(n_base" n ", " kpos ")));\n")))
                       (apply str (for [m ms] (str "        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" kpos ", " (amul m) "), &a" m ");\n")))
                       (apply str (for [m ms] (str "        sa" m " = as_short8(a" m ");\n")))
                       (apply str (for [m ms n ns] (str "        acc" m n " = intel_sub_group_f16_f16_matrix_mad_k16(sa" m ", bp" n ", acc" m n ");\n")))))]
      (str
       "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
       "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n\n"
       "// Tiled GEMM (parametric): WG " block-m "x" block-n ", SG " sg-m "x" sg-n ", K " block-k
       ", DPAS " mi "x" ni "x" ki ", sg " sg "\n"
       "__attribute__((intel_reqd_sub_group_size(" sg ")))\n"
       "__kernel void " kernel-name "(\n"
       "    __global const half* restrict A,\n    __global const half* restrict B,\n"
       "    __global " c-type "* restrict C,\n    int M, int N, int K"
       (when split-k? ", int KC") (when (not= alpha 1.0) ", float alpha") (when (not= beta 0.0) ", float beta")
       ") {\n"
       "    int sg_id = get_sub_group_id();\n    int sg_lid = get_sub_group_local_id();\n"
       "    int sg_row = sg_id / " ncols ";\n    int sg_col = sg_id % " ncols ";\n"
       "    int m_base = get_group_id(1) * " block-m " + sg_row * " sg-m ";\n"
       (apply str (for [n ns] (str "    int n_base" n " = get_group_id(0) * " block-n " + sg_col * " sg-n (when (pos? n) (str " + " (* n ni))) ";\n")))
       "    if (m_base >= M || n_base0 >= N) return;\n"
       (when batched?
         "    long slab = get_group_id(2);\n    A += slab*(long)M*(long)K;\n    B += slab*(long)K*(long)N;\n    C += slab*(long)M*(long)N;\n")
       (when split-k?
         "    int k_lo = get_group_id(2) * KC;\n    if (k_lo >= K) return;\n    int k_hi = k_lo + KC; if (k_hi > K) k_hi = K;\n    C += (long)get_group_id(2) * (long)M * (long)N;\n")
       (apply str (for [m ms] (str "    " accT " " (str/join ", " (for [n ns] (str "acc" m n "=0.0f"))) ";\n")))
       "    int a_wb = K * 2, a_pb = K * 2;\n    int b_wb = N * 2, b_pb = N * 2;\n"
       "    ushort8 " (str/join ", " (for [m ms] (str "a" m))) ";\n"
       "    short8 " (str/join ", " (for [m ms] (str "sa" m))) ";\n"
       "    int8 " (str/join ", " (for [n ns] (str "bp" n))) ";\n"
       (apply str (for [p (range prefetch)]
                    (str "    if (" (when split-k? "k_lo + ") (* p ki) " < " kend ") {\n"
                         (apply str (for [m ms] (str "        intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" (when split-k? "k_lo + ") (* p ki) ", " (amul m) "));\n")))
                         "    }\n")))
       "    int k = " (if split-k? "k_lo" "0") ";\n"
       "    for (; k + " (dec block-k) " < " kend "; k += " block-k ") {\n"
       (apply str (for [ks (range ksteps)] (kstep (if (zero? ks) "k" (str "k + " (* ks ki))))))
       "    }\n"
       "    for (; k < " kend "; k += " ki ") {\n"
       (kstep "k")
       "    }\n"
       (apply str
              (for [m ms i (range mi)]
                (str "    { int row = m_base + " (+ (* m mi) i) ";\n      if (row < M) {\n"
                     (apply str (for [n ns]
                                  (str "        { int col = n_base" n " + sg_lid;\n          if (col < N) "
                                       (if (not= beta 0.0)
                                         (str "{ float old=(float)C[row*N+col]; C[row*N+col] = " store-cast "(" (when (not= alpha 1.0) "alpha * ") "acc" m n ".s" i " + beta*old); }\n")
                                         (str "C[row*N+col] = " store-cast "(" (when (not= alpha 1.0) "alpha * ") "acc" m n ".s" i ");\n"))
                                       "        }\n")))
                     "      }\n    }\n")))
       "}\n"))))

(defn emit-gemm-splitk-reduce-kernel
  "Second stage of a split-k GEMM: C[i] = Σ_s partials[s·MN + i], i < MN = M·N.
  One work-item per OUTPUT element (grid-stride), each summing `splits` f32
  partials — MN work items, so the combine itself has full occupancy whenever the
  GEMM's output is non-tiny. f32 throughout (the partials are f32 accumulators)."
  [kernel-name]
  (str "__kernel void " kernel-name "(\n"
       "    __global const float* restrict partials,\n"
       "    __global float* restrict C,\n"
       "    int mn, int splits) {\n"
       "    for (int i = get_global_id(0); i < mn; i += get_global_size(0)) {\n"
       "        float acc = 0.0f;\n"
       "        for (int s = 0; s < splits; ++s) acc += partials[(long)s * (long)mn + i];\n"
       "        C[i] = acc;\n"
       "    }\n"
       "}\n"))

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

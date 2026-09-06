(ns raster.compiler.reference.legacy-nn-opencl
  "Test-only source oracles for retired, unreferenced NN kernel generators.
   These preserve historical source-shape checks, not production schedules or performance claims."
  (:require [raster.compiler.backend.gpu.opencl-codegen :as codegen
             :refer [opencl-type-map extension-pragmas]]))

(def ^:private subgroup-pragma @#'codegen/subgroup-pragma)

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


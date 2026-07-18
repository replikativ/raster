(ns raster.compiler.backend.gpu.par-hip
  "HIP/CUDA kernel emission for raster.par elementwise forms — the forcing-function backend
   that proves 'CUDA/HIP is a descriptor + emitter, not a pipeline fork' (.internal/cuda_hip_plan.md).

   The EXPRESSION layer is vendor-neutral: the kernel body is emitted by the shared `c-emit`
   under `hip-config` — the SAME emitter par_opencl uses, byte-for-byte the same body. Only the
   thin per-target WRAPPER differs from OpenCL:

     OpenCL                                   CUDA / HIP
     __kernel void NAME(...)                  extern \"C\" __global__ void NAME(...)
     __global const T* restrict X             const T* __restrict__ X
     __global T* restrict out                 T* __restrict__ out
     get_global_id(0)                         (blockIdx.x*blockDim.x + threadIdx.x)
     get_global_size(0)                       (gridDim.x*blockDim.x)
     #pragma OPENCL EXTENSION cl_khr_fp64     (none — double is native)

   HIP and CUDA share ONE source dialect (both are CUDA-C); the target differs only at
   compile-gate / launch time (arch string, compiler binary). This ns emits the source and
   records the target; the compile-gate (test/raster/compiler/backend/gpu/hip_compile_gate_test)
   proves legality via nvcc/hipcc with no device.

   SCOPE: elementwise map! / map-void! — the clearly-portable cases with zero vendor builtins.
   Reductions (portable local-mem tree reduce), the runtime FFM layer, and the per-vendor f16
   MATRIX kernel are follow-ups (the matrix kernel is the one genuine fork, post-S6)."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.par :as par]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [clojure.string :as str]))

(def cuda-type-map
  "Raster dtype → CUDA-C native type. The shared :c facet already matches CUDA-C (64-bit is
   `long long`, unlike OpenCL's `long`). :half maps to `__half` (needs <cuda_fp16.h>/<hip_fp16.h>)
   — elementwise float/double/int need no header."
  (assoc (dtype/backend-types :c) :half "__half"))

(defn- ctype [dtype] (get cuda-type-map dtype (get cuda-type-map :double)))

(def preamble
  "Portable HIP/CUDA compilation-unit preamble — prepend ONCE per module. HIP (clang) needs the
   runtime header for the kernel-launch builtins (blockIdx/blockDim/threadIdx/gridDim); nvcc
   injects them implicitly, so the include is guarded to hipcc and skipped under CUDA."
  "#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)\n#include <hip/hip_runtime.h>\n#endif\n")

;; The device-index idiom shared by every CUDA-C grid-stride loop.
(def ^:private grid-stride-init "blockIdx.x * blockDim.x + threadIdx.x")
(def ^:private grid-stride-step "gridDim.x * blockDim.x")

(defn- emit-body
  "Emit a par-form body as a CUDA-C expression string via the shared emitter under hip-config."
  [body idx arr-params dtype]
  (binding [ce/*emit-config* ce/hip-config
            ce/*scalar-type* (ctype dtype)]
    (ce/emit-expr (ce/adapt-casts-for-dtype body dtype)
                  idx (set (map #(symbol (name %)) arr-params)))))

(defn generate-par-map-kernel
  "CUDA/HIP kernel from a raster.par/map! form. Mirrors par_opencl/generate-par-map-kernel with
   the CUDA-C wrapper. Returns {:kernel-name :source :array-params :scalar-params :dtype :target}."
  [form & {:keys [dtype kernel-name-prefix scalar-types target]
           :or {dtype :double kernel-name-prefix "par_map" scalar-types {} target :cuda}}]
  (let [info        (par/extract-par-map-info form)
        {:keys [idx cast body]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ct          (ctype dtype)
        array-syms  (ce/collect-arrays-in-body body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        arr-params  (vec (sort-by name array-syms))
        scl-params  (vec (sort-by name scalar-syms))
        arr-param-str (str/join ", " (map (fn [s] (str "const " ct "* __restrict__ " (ce/c-symbol s)))
                                          arr-params))
        scl-type    (fn [s] (ce/scalar-native-type s scalar-types ct))
        scl-param-str (str/join ", " (map (fn [s] (str (scl-type s) " " (ce/c-symbol s))) scl-params))
        out-param   (str ct "* __restrict__ out")
        all-params  (str/join ", " (remove empty? [arr-param-str out-param scl-param-str "int _n_bound"]))
        body-str    (emit-body body idx arr-params dtype)
        cast-str    (if cast (str "(" (name cast) ")(" body-str ")") body-str)
        source      (str "extern \"C\" __global__ void " kernel-name
                         "(" all-params ") {\n"
                         "    for (int idx = " grid-stride-init "; idx < _n_bound; idx += " grid-stride-step ") {\n"
                         "        out[idx] = " cast-str ";\n"
                         "    }\n"
                         "}\n")]
    {:kernel-name kernel-name :source source
     :array-params arr-params :scalar-params scl-params :dtype dtype :target target}))

(defn generate-par-map-void-kernel
  "CUDA/HIP kernel from a raster.par/map-void! form (side-effecting, no output array). Written
   arrays get `T*` (mutable), read-only arrays `const T* __restrict__`."
  [form & {:keys [dtype kernel-name-prefix array-types scalar-types target]
           :or {dtype :float kernel-name-prefix "par_map_void" array-types {} scalar-types {}
                target :cuda}}]
  (let [info        (par/extract-par-map-void-info form)
        idx         (:idx info)
        body        (ce/normalize-array-prims (:body info))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        default-ct  (ctype dtype)
        meta-types  (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        array-syms  (ce/collect-arrays-in-body body)
        written-syms (ce/collect-written-arrays body)
        scalar-syms (ce/collect-scalars-in-body body idx array-syms)
        arr-params  (vec (sort-by name array-syms))
        scl-params  (vec (sort-by name scalar-syms))
        arr-ct      (fn [s] (ctype (get array-types s (get array-types (symbol (name s)) dtype))))
        written?    (fn [s] (or (contains? written-syms s) (contains? written-syms (symbol (name s)))))
        arr-param-str (str/join ", "
                                (map (fn [s]
                                       (let [c (arr-ct s)]
                                         (if (written? s)
                                           (str c "* " (ce/c-symbol s))
                                           (str "const " c "* __restrict__ " (ce/c-symbol s)))))
                                     arr-params))
        scl-type    (fn [s] (ce/scalar-native-type s scalar-types default-ct))
        scl-param-str (str/join ", " (map (fn [s] (str (scl-type s) " " (ce/c-symbol s))) scl-params))
        all-params  (str/join ", " (remove empty? [arr-param-str scl-param-str "int _n_bound"]))
        body-str    (emit-body body idx arr-params dtype)
        source      (str "extern \"C\" __global__ void " kernel-name
                         "(" all-params ") {\n"
                         "    for (int idx = " grid-stride-init "; idx < _n_bound; idx += " grid-stride-step ") {\n"
                         "        " body-str ";\n"
                         "    }\n"
                         "}\n")]
    {:kernel-name kernel-name :source source
     :array-params arr-params :scalar-params scl-params
     :written-arrays written-syms :dtype dtype :target target}))

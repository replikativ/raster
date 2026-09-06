(ns raster.compiler.reference.indexed-transfer-opencl
  "Frozen test-only source oracles for indexed transfers. Production uses TypedSOAC/KernelBody."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-launch :as klaunch]))

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
        strided? (some? stride)
        out-c (ce/c-symbol out)
        src-c (ce/c-symbol src)
        index-c (ce/c-symbol index)
        ;; Scatter needs float atomic CAS for float types, native atomic for int
        needs-float-atomic? (contains? #{:float :double} dtype)
        source (str (codegen/extension-pragmas dtype)
                    "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                    (when needs-float-atomic? ce/opencl-atomic-add-float-helper)
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

(defn generate-par-gather-kernel
  "Generate an OpenCL C kernel from a raster.par/gather form.
  Gather: out[e*stride+d] = src[index[e]*stride+d] — one work-item per gathered pair
  e (inner loop over stride). A gather writes every output element exactly once (no
  atomics, no accumulation), so it uses the map-void calling convention
  (out, src, index, [stride,] int _n_bound); resident execution binds its artifact through
  KernelCall — `out` is just another effect pointer.

  Returns one verified KernelArtifact using the same side-effect map contract."
  [form & {:keys [dtype kernel-name-prefix]
           :or {dtype :float kernel-name-prefix "par_gather"}}]
  (let [info (par/extract-par-gather-info form)
        {:keys [out src index stride n]} info
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "float")
        strided? (some? stride)
        out-c (ce/c-symbol out)
        src-c (ce/c-symbol src)
        index-c (ce/c-symbol index)
        source (str (codegen/extension-pragmas dtype)
                    "__kernel void " kernel-name
                    "(__global " ctype "* " out-c
                    ", __global const " ctype "* restrict " src-c
                    ", __global const int* restrict " index-c
                    (if strided? ", int stride, int _n_bound" ", int _n_bound")
                    ") {\n"
                    "    for (int e = get_global_id(0); e < _n_bound; e += get_global_size(0)) {\n"
                    "        int src_idx = " index-c "[e];\n"
                    (if strided?
                      (str "        for (int d = 0; d < stride; d++) {\n"
                           "            " out-c "[e * stride + d] = " src-c "[src_idx * stride + d];\n"
                           "        }\n")
                      (str "        " out-c "[e] = " src-c "[src_idx];\n"))
                    "    }\n"
                    "}\n")
        scalar-params (if strided? ['stride] [])
        abi (kabi/validate!
             (vec (concat
                   [(kabi/slot out :output dtype :c-name (ce/c-symbol out) :role :effect)
                    (kabi/slot src :input dtype :c-name (ce/c-symbol src) :role :operand)
                    (kabi/slot index :input :int :c-name (ce/c-symbol index) :role :operand)]
                   (when strided?
                     [(kabi/slot 'stride :scalar :int :role :parameter)])
                   [(kabi/slot '_n_bound :scalar :int :role :bound)])))]
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments (vec (concat [out src index] (when strided? [stride]) [n]))
      :launch (klaunch/spec
               {:workgroup-size [256]
                :group-count [(klaunch/ceil-div n 256)]})
      :temporaries []
      :effects {:kind :side-effect-map}
      :provenance {:dialect :gather}
      :attributes {:array-params [out src index]
                   :scalar-params scalar-params
                   :written-arrays [out]
                   :strided? strided?
                   :dtype dtype}})))


(ns raster.compiler.backend.gpu.kernel-body-c-dialect
  "Thin target spellings for the scalar/control KernelBody C-family emitter.

  A dialect never selects a schedule. It only spells verified storage types, hardware indices,
  entry points and subgroup shuffles for one target language. CUDA and HIP intentionally share
  the same structural lowering while retaining distinct collective intrinsics and module targets."
  (:require [raster.compiler.core.dtype :as dtype]))

(def ^:private dialects
  {:opencl-intel {:id :opencl-intel :target :opencl-c :family :opencl
                  :association :implementation-defined}
   :opencl-portable {:id :opencl-portable :target :opencl-c :family :opencl
                     :association :implementation-defined}
   :cuda {:id :cuda :target :cuda-c :family :cuda
          :association :explicit-shuffle-down-tree}
   :hip {:id :hip :target :hip-cpp :family :hip
         :association :explicit-shuffle-down-tree}})

(defn resolve!
  "Resolve and validate a target spelling descriptor."
  [id]
  (or (get dialects id)
      (throw (ex-info "KernelBody C-family target dialect is unsupported"
                      {:reason :kernel-body-c-target-dialect
                       :dialect id :supported (set (keys dialects))}))))

(defn target [dialect] (:target dialect))
(defn family [dialect] (:family dialect))
(defn collective-association [dialect] (:association dialect))
(defn opencl? [dialect] (= :opencl (family dialect)))

(defn type-name
  [dialect type]
  (if (= :predicate type)
    "bool"
    (let [type (dtype/canon type)]
      (cond
        (opencl? dialect) (dtype/ctype :opencl type)
        (= :half type) "__half"
        :else (dtype/ctype :c type)))))

(defn unsigned-type-name
  [dialect type]
  (if (opencl? dialect)
    ({:byte "uchar" :int "uint" :long "ulong"} (dtype/canon type))
    ({:byte "uint8_t" :int "unsigned int" :long "unsigned long long"}
     (dtype/canon type))))

(defn parameter-declaration
  [dialect parameter c-name]
  (if (= :scalar (:kind parameter))
    (str (type-name dialect (:dtype parameter)) " " c-name)
    (do
      (when-not (= :global (:memory-space parameter))
        (throw (ex-info "C-family scalar lowering only supports global kernel storage"
                        {:reason :kernel-body-c-memory-space
                         :dialect (:id dialect) :parameter parameter})))
      (if (opencl? dialect)
        (str "__global " (when (= :input (:kind parameter)) "const ")
             (type-name dialect (:dtype parameter)) "* " c-name)
        (str (when (= :input (:kind parameter)) "const ")
             (type-name dialect (:dtype parameter)) "* __restrict__ " c-name)))))

(def ^:private axis-name ["x" "y" "z"])

(defn- linear-thread-id []
  "((int)threadIdx.x + (int)blockDim.x * ((int)threadIdx.y + (int)blockDim.y * (int)threadIdx.z))")

(defn index-binding
  [dialect source axis subgroup-width]
  (if (opencl? dialect)
    (case source
      :group (str "get_group_id(" axis ")")
      :local (str "get_local_id(" axis ")")
      :subgroup "get_sub_group_id()"
      :lane "get_sub_group_local_id()")
    (case source
      :group (str "blockIdx." (nth axis-name axis))
      :local (str "threadIdx." (nth axis-name axis))
      :subgroup (str "(" (linear-thread-id) " / " subgroup-width ")")
      :lane (str "(" (linear-thread-id) " % " subgroup-width ")"))))

(defn subgroup-attribute
  [dialect subgroup-width uses-subgroups?]
  (when uses-subgroups?
    (case (:id dialect)
      :opencl-intel
      (str "__attribute__((intel_reqd_sub_group_size(" subgroup-width ")))\n")
      :opencl-portable
      (str "__attribute__((reqd_sub_group_size(" subgroup-width ")))\n")
      (:cuda :hip) "")))

(defn preamble
  [dialect {:keys [uses-half? uses-double? uses-subgroups?]}]
  (if (opencl? dialect)
    (str (when uses-half? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n")
         (when uses-double? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
         (when uses-subgroups?
           (str "#if defined(cl_khr_subgroups)\n"
                "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
                "#elif defined(cl_intel_subgroups)\n"
                "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
                "#endif\n"))
         (when (or uses-half? uses-double? uses-subgroups?) "\n"))
    (str "#include <stdint.h>\n#include <math.h>\n"
         (case (:id dialect)
           :cuda "#include <cuda_fp16.h>\n"
           :hip (str "#include <hip/hip_runtime.h>\n"
                     "#include <hip/hip_fp16.h>\n"))
         "\n")))

(defn entry-prefix
  [dialect]
  (if (opencl? dialect) "__kernel void " "extern \"C\" __global__ void "))

(defn workgroup-arena-declaration
  "Spell one statically sized, explicitly aligned workgroup-memory arena."
  [dialect name bytes alignment]
  (case (:id dialect)
    (:opencl-intel :opencl-portable)
    (let [[backing-type backing-bytes]
          ({1 ["uchar" 1] 2 ["ushort" 2] 4 ["uint" 4] 8 ["ulong" 8] 16 ["uint4" 16]}
           alignment)]
      (str "__local " backing-type " " name "[" (quot bytes backing-bytes) "];"))
    :cuda
    (str "__shared__ __align__(" alignment ") unsigned char " name "[" bytes "];")
    :hip
    (str "__shared__ __attribute__((aligned(" alignment "))) unsigned char " name "[" bytes "];")))

(defn workgroup-pointer-declaration
  "Spell a typed pointer into a verified byte offset of the workgroup arena."
  [dialect type name arena byte-offset]
  (if (opencl? dialect)
    (str "__local " (type-name dialect type) "* " name " = (__local "
         (type-name dialect type) "*)((__local uchar*)" arena " + " byte-offset ");")
    (str (type-name dialect type) "* " name " = reinterpret_cast<"
         (type-name dialect type) "*>(" arena " + " byte-offset ");")))

(defn workgroup-barrier
  "Spell the verified full-workgroup acquire/release barrier."
  [dialect]
  (if (opencl? dialect) "barrier(CLK_LOCAL_MEM_FENCE);" "__syncthreads();"))

(defn broadcast-expression
  [dialect input source-lane width]
  (case (:id dialect)
    (:opencl-intel :opencl-portable)
    (str "sub_group_broadcast(" input ", " source-lane ")")
    :cuda
    (str "__shfl_sync(__activemask(), " input ", " source-lane ", " width ")")
    :hip
    (str "__shfl(" input ", " source-lane ", " width ")")))

(defn shuffle-down-expression
  [dialect input distance width]
  (case (:id dialect)
    :cuda
    (str "__shfl_down_sync(__activemask(), " input ", " distance ", " width ")")
    :hip
    (str "__shfl_down(" input ", " distance ", " width ")")
    (throw (ex-info "OpenCL subgroup reductions use target builtins, not explicit shuffles"
                    {:reason :kernel-body-c-shuffle-dialect :dialect (:id dialect)}))))

(defn opencl-reduction-builtin
  [operator]
  ({:+ "sub_group_reduce_add"
    :min "sub_group_reduce_min"
    :max "sub_group_reduce_max"} operator))

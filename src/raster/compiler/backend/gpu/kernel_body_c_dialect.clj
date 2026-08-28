(ns raster.compiler.backend.gpu.kernel-body-c-dialect
  "Thin target spellings for the scalar/control KernelBody C-family emitter.

  A dialect never selects a schedule. It only spells verified storage types, hardware indices,
  entry points and subgroup shuffles for one target language. CUDA and HIP intentionally share
  the same structural lowering while retaining distinct collective intrinsics and module targets."
  (:require [clojure.string :as str]
            [raster.compiler.core.dtype :as dtype]))

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

(defn async-copy-mode
  "Report the physical lowering used for the target-neutral async dependency contract."
  [dialect]
  (case (:id dialect)
    (:opencl-intel :opencl-portable) :native-events
    :cuda (if (and (vector? (:compute-capability dialect))
                   (not (neg? (compare (:compute-capability dialect) [8 0]))))
            :native-cp-async
            :synchronous-cooperative)
    :hip :synchronous-cooperative))

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
      :group-count (str "get_num_groups(" axis ")")
      :local (str "get_local_id(" axis ")")
      :subgroup "get_sub_group_id()"
      :lane "get_sub_group_local_id()")
    (case source
      :group (str "blockIdx." (nth axis-name axis))
      :group-count (str "gridDim." (nth axis-name axis))
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

(defn async-copy
  "Spell one workgroup-cooperative contiguous global-to-workgroup copy."
  [dialect {:keys [name source destination elements element-bytes transfer-bytes workgroup-width
                   overlap]}]
  (case (async-copy-mode dialect)
    :native-events
    (str "event_t " name " = async_work_group_copy(" destination ", " source ", "
         elements ", (event_t)0);")

    :native-cp-async
    (let [offset (str name "_byte_offset")
          shared-address (str name "_shared_address")
          native-copy
          (str "for (int " offset " = (" (linear-thread-id) ") * " transfer-bytes
               "; " offset " < " (* elements element-bytes) "; " offset " += "
               workgroup-width " * " transfer-bytes ") {\n"
               "  unsigned int " shared-address
               " = (unsigned int)__cvta_generic_to_shared((void*)((unsigned char*)"
               destination " + " offset "));\n"
               "  asm volatile(\"cp.async.ca.shared.global [%0], [%1], " transfer-bytes
               ";\\n\" :: \"r\"(" shared-address "), \"l\"((const unsigned char*)"
               source " + " offset "));\n"
               "}")
          fallback-offset (str name "_fallback_element_offset")
          fallback
          (str "for (int " fallback-offset " = " (linear-thread-id) "; " fallback-offset
               " < " elements "; " fallback-offset " += " workgroup-width ") {\n"
               "  " destination "[" fallback-offset "] = " source "[" fallback-offset "];\n"
               "}")]
      (if (= :required overlap)
        native-copy
        (str "if ((((uintptr_t)" source " | (uintptr_t)" destination ") & "
             (dec transfer-bytes) ") == 0) {\n"
             (str/replace native-copy #"(?m)^" "  ") "\n} else {\n"
             (str/replace fallback #"(?m)^" "  ") "\n}")))

    :synchronous-cooperative
    (let [offset (str name "_element_offset")]
      (str "for (int " offset " = " (linear-thread-id) "; " offset " < " elements
           "; " offset " += " workgroup-width ") {\n"
           "  " destination "[" offset "] = " source "[" offset "];\n"
           "}"))))

(defn async-commit
  [dialect group]
  (case (async-copy-mode dialect)
    :native-cp-async "asm volatile(\"cp.async.commit_group;\\n\" ::);"
    :native-events (str "/* OpenCL event group " group " committed at issue. */")
    :synchronous-cooperative (str "/* synchronous async group " group " committed. */")))

(defn async-wait
  [dialect event-names pending-groups]
  (case (async-copy-mode dialect)
    :native-events
    (str/join "\n" (map #(str "wait_group_events(1, &" % ");") event-names))

    :native-cp-async
    (str "asm volatile(\"cp.async.wait_group " pending-groups ";\\n\" ::);")

    :synchronous-cooperative
    "/* synchronous cooperative copies are complete before this wait. */"))

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

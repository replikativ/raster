(ns raster.dl.qlinear-gpu
  "Resident-weight GPU execution of the int8-MAC quantized linear (raster.dl.qlinear)
  — the :gpu compute-profile for qlinear-i8. The SAME Q4_0 weight and the SAME kernel
  descriptor as the CPU path, lowered to OpenCL (cq/emit-qmatmul-opencl) and run on a
  concrete device via Level-Zero.

  Residency model (the point of 'load onto the GPU'): weights are uploaded ONCE into a
  resident arena bound to a CONCRETE device-id (:ze:0 here; multiple GPUs = multiple
  arenas — the device-type :gpu placement tag is abstract, the device instance is this
  binding) and reused across every decode step; only the activation crosses per call.
  Each resident weight also owns reusable input/output scratch (allocated once, written
  per call) so there is no per-call allocation. On this unified-memory Arc the arena
  uses zero-copy shared allocations (alloc-shared).

  Scope: Level-Zero + unified memory. Follow-ons: route allocation through gpu.core for
  backend-generality (OpenCL/CUDA, discrete device alloc + upload); quantize the
  activation ON the GPU (currently CPU-serial, copied into the resident input scratch).
  This is device-orchestration glue (defn, like the gpu runtimes), not numeric compute —
  the numeric kernel is the validated OpenCL source in raster...cpu.quant."
  (:require [raster.compiler.backend.cpu.quant :as cq]
            [raster.compiler.backend.gpu.par-opencl :as pocl]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign MemorySegment]))

;; device-id -> {:module :kernel}. The kernel is identical across weights (it reads
;; in/out as scalar args), so one compiled kernel per device serves the whole model.
(defonce ^:private kernel-cache (atom {}))

(def ^:private kernel-name "qlinear_i8_q4_gpu")

(defn resident-kernel
  "Compile + load the Q4_0 OpenCL kernel for device-id once; cache module+kernel."
  [device-id]
  (or (get @kernel-cache device-id)
      (let [spv (pocl/compile-kernel-to-spirv (cq/emit-qmatmul-opencl kernel-name cq/q4-0)
                                              :device-id device-id)
            mod (ze/load-module! spv)
            entry {:module mod :kernel (ze/create-kernel mod kernel-name)}]
        (swap! kernel-cache assoc device-id entry)
        entry)))

(defn- shared-of
  "alloc-shared n-bytes and copy the first n-bytes of a heap array into it."
  [arr ^long n-bytes]
  (let [s (ze/alloc-shared n-bytes)]
    (MemorySegment/copy (MemorySegment/ofArray arr) 0 s 0 n-bytes)
    s))

(defn upload-weight!
  "Upload one Q4_0 weight (row-major {wq,ws} from cq/quantize-weight-q4) into a resident
  arena on device-id, with reusable activation/output scratch. The returned handle is
  reused across all subsequent qlinear-i8-gpu calls for this weight — weights never move."
  [device-id wq ws in out]
  (let [in (int in) out (int out) nb (quot in 32)]
    (resident-kernel device-id)               ;; ensure kernel compiled/cached
    {:device device-id :in in :out out
     :b-wq (shared-of wq (* out (quot in 2)))  ;; resident weight (uploaded once)
     :b-ws (shared-of ws (* out nb 4))
     :b-xq (ze/alloc-shared in)                ;; reusable input/output scratch
     :b-xs (ze/alloc-shared (* nb 4))
     :b-xsum (ze/alloc-shared (* nb 4))
     :b-y (ze/alloc-shared (* out 4))}))

(defn qlinear-i8-gpu
  "Run the resident-weight quantized linear on the GPU: int8-quantize x (CPU-serial),
  copy into the resident input scratch, launch the kernel, read y back. Weights stay
  resident on the device across calls; matches the CPU qlinear-i8 to f32 precision."
  ^floats [W x]
  (let [{:keys [device in out b-wq b-ws b-xq b-xs b-xsum b-y]} W
        in (int in) out (int out) nb (quot in 32)
        {:keys [xq xs xsum]} (cq/quantize-act-i8 x 1 in)]
    (MemorySegment/copy (MemorySegment/ofArray xq) 0 b-xq 0 in)
    (MemorySegment/copy (MemorySegment/ofArray xs) 0 b-xs 0 (* nb 4))
    (MemorySegment/copy (MemorySegment/ofArray xsum) 0 b-xsum 0 (* nb 4))
    (let [wgs 64 groups (quot (+ out (dec wgs)) wgs)
          y (float-array out)]
      (ze/launch! (:kernel (resident-kernel device)) groups wgs
                  [b-xq b-xs b-xsum b-wq b-ws b-y
                   {:type :int :value in} {:type :int :value out}])
      (MemorySegment/copy b-y 0 (MemorySegment/ofArray y) 0 (* out 4))
      y)))

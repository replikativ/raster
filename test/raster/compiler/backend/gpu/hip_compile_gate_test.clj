(ns raster.compiler.backend.gpu.hip-compile-gate-test
  "The HIP/CUDA COMPILE-GATE — the validation-ladder bottom rung (.internal/cuda_hip_plan.md §65).

   Proves raster's portable elementwise kernels (emitted by par_hip, body = the shared c-emit)
   are LEGAL CUDA-C and HIP by running them through nvcc and hipcc — pure host compilers, NO
   device needed. This is what makes 'CUDA/HIP is a descriptor + emitter, not a fork' concrete:
   the same expression layer that passes the OpenCL suite compiles unmodified on both vendors.

   Honest-shipping posture: these kernels are COMPILE-validated (legality), not yet
   EXECUTION-validated (that needs chipStar-on-Arc / real NVIDIA/AMD HW). The gate skips
   cleanly when a compiler is absent, but records a FAILING assertion if emission itself breaks."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.shell :as sh]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.par-hip :as hip]))

;; ── compiler probes ──────────────────────────────────────────────────────────────
(defn- on-path? [bin]
  (try (zero? (:exit (sh/sh "sh" "-c" (str "command -v " bin)))) (catch Throwable _ false)))

(def ^:private nvcc?  (delay (on-path? "nvcc")))
(def ^:private hipcc? (delay (on-path? "hipcc")))

(def ^:private scratch
  (doto (io/file (System/getProperty "java.io.tmpdir") (str "raster-hip-gate-" (System/nanoTime)))
    (.mkdirs)))

(defn- write-src! [basename source]
  (let [f (io/file scratch basename)]
    (spit f (str hip/preamble source))
    (.getAbsolutePath f)))

(defn- nvcc-gate
  "nvcc → PTX for sm_89. PTX generation exercises the full front+middle end (parse, type-check,
   device-code-gen) with no GPU. Returns {:exit :out :err}."
  [src-path]
  (sh/sh "nvcc" "-ptx" "-arch=sm_89" "-o" (str src-path ".ptx") src-path))

(defn- hipcc-gate
  "hipcc syntax+codegen check for RDNA3 (gfx1100). -fsyntax-only is the lightest legality gate."
  [src-path]
  (sh/sh "hipcc" "--offload-arch=gfx1100" "-fsyntax-only" "-x" "hip" src-path))

;; ── the kernels under gate: portable elementwise ─────────────────────────────────
(def ^:private kernels
  "[label form opts] — elementwise forms spanning arith, scalar broadcast, device math, float,
   and a side-effecting void map (the clearly-portable cases, zero vendor builtins)."
  [["add-double"   '(raster.par/map! out i n double (+ (aget a i) (aget b i)))          {}]
   ["scalar-mul"   '(raster.par/map! out i n double (* alpha (aget a i)))               {}]
   ["device-math"  '(raster.par/map! out i n double (Math/sin (aget a i)))              {}]
   ["add-float"    '(raster.par/map! out i n float  (+ (aget a i) (aget b i)))          {:dtype :float}]
   ["fma-float"    '(raster.par/map! out i n float  (+ (* (aget a i) (aget b i)) (aget c i))) {:dtype :float}]])

(def ^:private void-kernels
  [["axpy-void" '(raster.par/map-void! i n (raster.arrays/aset y i (+ (aget y i) (* alpha (aget x i))))) {:dtype :float}]])

(defn- emit-all
  "Emit every gated kernel; return [{:label :source :name} …]. Runs with no device."
  []
  (concat
   (for [[label form opts] kernels]
     (let [k (apply hip/generate-par-map-kernel form (mapcat identity opts))]
       {:label label :source (:source k) :name (:kernel-name k)}))
   (for [[label form opts] void-kernels]
     (let [k (apply hip/generate-par-map-void-kernel form (mapcat identity opts))]
       {:label label :source (:source k) :name (:kernel-name k)}))))

;; ════════════════════════════════════════════════════════════════════════════════
;; Emission is device-free and ALWAYS runs (a broken emitter fails loud here)
;; ════════════════════════════════════════════════════════════════════════════════

(deftest par-hip-emits-cuda-c
  (testing "every elementwise kernel emits a CUDA-C __global__ with the grid-stride idiom"
    (doseq [{:keys [label source]} (emit-all)]
      (is (str/includes? source "extern \"C\" __global__ void") (str label ": __global__ wrapper"))
      (is (str/includes? source "blockIdx.x * blockDim.x + threadIdx.x") (str label ": grid-stride index"))
      (is (not (str/includes? source "__kernel")) (str label ": no OpenCL __kernel"))
      (is (not (str/includes? source "get_global_id")) (str label ": no OpenCL builtins")))))

;; ════════════════════════════════════════════════════════════════════════════════
;; The compile-gate: nvcc (CUDA) + hipcc (HIP). Skips cleanly if a compiler is absent.
;; ════════════════════════════════════════════════════════════════════════════════

(deftest cuda-compile-gate
  (if-not @nvcc?
    (println "  [HIP GATE] nvcc absent — CUDA compile-gate skipped (emission still validated above)")
    (doseq [{:keys [label source name]} (emit-all)]
      (let [src (write-src! (str "cuda_" label "_" name ".cu") source)
            {:keys [exit err]} (nvcc-gate src)]
        (is (zero? exit) (str label ": nvcc -ptx -arch=sm_89 must succeed.\n" err))))))

(deftest hip-compile-gate
  (if-not @hipcc?
    (println "  [HIP GATE] hipcc absent — HIP compile-gate skipped (emission still validated above)")
    (doseq [{:keys [label source name]} (emit-all)]
      (let [src (write-src! (str "hip_" label "_" name ".hip") source)
            {:keys [exit err]} (hipcc-gate src)]
        (is (zero? exit) (str label ": hipcc --offload-arch=gfx1100 -fsyntax-only must succeed.\n" err))))))

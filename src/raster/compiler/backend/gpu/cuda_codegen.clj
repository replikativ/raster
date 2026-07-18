(ns raster.compiler.backend.gpu.cuda-codegen
  "CUDA-C codegen — the NVIDIA `:mma` (Tensor Core) fork of the tile-parametric GEMM. The tile
   vocabulary is vendor-neutral (block-m/block-n/sg-m/sg-n/block-k × matrix shape, identical to the
   Intel :dpas generator in opencl-codegen); only the emitted LANGUAGE and matrix INSTRUCTION differ.
   This generator emits CUDA C using the WMMA fragment API (nvcuda::wmma), which the NVIDIA toolchain
   lowers to HMMA Tensor Core instructions on cc≥7.0.

   T5 scope + honesty: on a machine with nvcc but NO NVIDIA GPU, this is validated to the strongest
   available gate — it COMPILES (nvcc → SASS) and SELECTS Tensor Cores (HMMA in the SASS). NUMERICAL
   correctness needs a real GPU (deferred, like the Intel kernel's on-device gate). The MVP emits the
   f32-accumulator (:float output), tile-aligned path (M/N/K multiples of the tile — ragged-tile
   boundary handling via SMEM zero-pad is the on-device follow-up). SMEM double-buffering (the true
   num_stages knob for this family) and WGMMA (sm_90 async warp-group) are separate emitters."
  (:require [clojure.string :as str]))

(defn emit-gemm-tiled-cuda
  "Tile-parametric CUDA WMMA GEMM (C = A·B, f16 in, f32 accumulate, row-major). Same tile map as the
   Intel emit-gemm-tiled; `matrix` is the WMMA fragment shape {:m 16 :n 16 :k 16 :subgroup 32}. Each
   WARP computes an sg-m×sg-n output tile as an (sg-m/16)×(sg-n/16) grid of 16×16 accumulator
   fragments; the block is a (block-m/sg-m)×(block-n/sg-n) grid of warps. K-loop steps by 16
   (WMMA k), unrolled block-k/16 per iteration. Emits :float (f32-accumulator) output."
  [kernel-name & {:keys [block-m block-n sg-m sg-n block-k matrix c-dtype]
                  :or {block-m 64 block-n 64 sg-m 32 sg-n 32 block-k 16
                       matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32} c-dtype :float}}]
  (let [{mi :m ni :n ki :k sg :subgroup :or {mi 16 ni 16 ki 16 sg 32}} matrix]
    (doseq [[nm a b] [["sg-m/M_i" sg-m mi] ["sg-n/N_i" sg-n ni] ["block-m/sg-m" block-m sg-m]
                      ["block-n/sg-n" block-n sg-n] ["block-k/K_i" block-k ki]]]
      (when-not (zero? (rem a b))
        (throw (ex-info (str "emit-gemm-tiled-cuda: " nm " not divisible (" a "/" b ")") {:matrix matrix}))))
    (when (not= c-dtype :float)
      (throw (ex-info "emit-gemm-tiled-cuda MVP emits :float (f32 accumulator); :half epilogue is a follow-up"
                      {:c-dtype c-dtype})))
    (let [nms (quot sg-m mi) nns (quot sg-n ni)      ;; fragments per warp (M, N)
          ncols (quot block-n sg-n)                   ;; warp columns
          ksteps (quot block-k ki)
          ms (range nms) ns (range nns)
          warps-per-block (* (quot block-m sg-m) (quot block-n sg-n))
          frag (fn [role & [layout]] (str "wmma::fragment<wmma::" role ", " mi ", " ni ", " ki ", "
                                          (if (= role "accumulator") "float" (str "half, wmma::" layout))
                                          ">"))]
      (str
       "#include <mma.h>\n"
       "using namespace nvcuda;\n\n"
       "// Tiled WMMA GEMM (parametric): block " block-m "x" block-n ", warp-tile " sg-m "x" sg-n
       ", K " block-k ", frag " mi "x" ni "x" ki ", warp " sg ", warps/block " warps-per-block "\n"
       "extern \"C\" __global__ void " kernel-name "(\n"
       "    const half* __restrict__ A, const half* __restrict__ B,\n"
       "    float* __restrict__ C, int M, int N, int K) {\n"
       "  int warpId = threadIdx.x / 32;\n"
       "  int warp_row = warpId / " ncols ";\n"
       "  int warp_col = warpId % " ncols ";\n"
       "  int m_base = blockIdx.y * " block-m " + warp_row * " sg-m ";\n"
       "  int n_base = blockIdx.x * " block-n " + warp_col * " sg-n ";\n"
       ;; accumulator fragments
       (apply str (for [m ms n ns] (str "  " (frag "accumulator") " acc" m "_" n ";\n")))
       (apply str (for [m ms n ns] (str "  wmma::fill_fragment(acc" m "_" n ", 0.0f);\n")))
       "  " (frag "matrix_a" "row_major") " " (str/join ", " (for [m ms] (str "a" m))) ";\n"
       "  " (frag "matrix_b" "row_major") " " (str/join ", " (for [n ns] (str "b" n))) ";\n"
       "  for (int k = 0; k < K; k += " block-k ") {\n"
       (apply str
              (for [ks (range ksteps)]
                (let [koff (* ks ki)]
                  (str
                   (apply str (for [m ms] (str "    wmma::load_matrix_sync(a" m ", A + (m_base + " (* m mi) ") * K + k + " koff ", K);\n")))
                   (apply str (for [n ns] (str "    wmma::load_matrix_sync(b" n ", B + (k + " koff ") * N + n_base + " (* n ni) ", N);\n")))
                   (apply str (for [m ms n ns] (str "    wmma::mma_sync(acc" m "_" n ", a" m ", b" n ", acc" m "_" n ");\n")))))))
       "  }\n"
       (apply str (for [m ms n ns]
                    (str "  wmma::store_matrix_sync(C + (m_base + " (* m mi) ") * N + n_base + " (* n ni)
                         ", acc" m "_" n ", N, wmma::mem_row_major);\n")))
       "}\n"))))

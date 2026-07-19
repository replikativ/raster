(ns raster.runtime.hardware-catalogue
  "Predefined hardware specifications for common GPUs and CPUs.

  Used by raster.runtime.hardware to fill gaps when hardware cannot be auto-detected
  (e.g. developing CUDA code on a CPU-only machine). Detected values
  override catalogue entries; catalogue fills gaps.

  Source: official NVIDIA specs, AMD product pages, Intel ARK.")

;; ================================================================
;; GPU catalogue
;; ================================================================

(def gpu-catalogue
  "Predefined GPU specs keyed by model name substring.
  Values are capability maps merged into detected device info."
  {"NVIDIA H100"
   {:compute-capability [9 0]
    :sm-count 132
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 85899345920
    :memory-bandwidth-gb-s 3350.0
    :peak-flops-dp 33.5e12
    :peak-flops-sp 67.0e12}

   "NVIDIA A100"
   {:compute-capability [8 0]
    :sm-count 108
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 85899345920
    :memory-bandwidth-gb-s 2039.0
    :peak-flops-dp 9.7e12
    :peak-flops-sp 19.5e12}

   "NVIDIA RTX 4090"
   {:compute-capability [8 9]
    :sm-count 128
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 25769803776
    :memory-bandwidth-gb-s 1008.0
    :peak-flops-dp 1.29e12
    :peak-flops-sp 82.6e12}

   "NVIDIA RTX 4080"
   {:compute-capability [8 9]
    :sm-count 76
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 17179869184
    :memory-bandwidth-gb-s 716.8
    :peak-flops-dp 0.78e12
    :peak-flops-sp 48.7e12}

   "NVIDIA RTX 3090"
   {:compute-capability [8 6]
    :sm-count 82
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 25769803776
    :memory-bandwidth-gb-s 936.2
    :peak-flops-dp 0.556e12
    :peak-flops-sp 35.6e12}

   "NVIDIA RTX 3080"
   {:compute-capability [8 6]
    :sm-count 68
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 10737418240
    :memory-bandwidth-gb-s 760.3
    :peak-flops-dp 0.465e12
    :peak-flops-sp 29.8e12}

   "NVIDIA V100"
   {:compute-capability [7 0]
    :sm-count 80
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 34359738368
    :memory-bandwidth-gb-s 900.0
    :peak-flops-dp 7.8e12
    :peak-flops-sp 15.7e12}

   "NVIDIA T4"
   {:compute-capability [7 5]
    :sm-count 40
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 17179869184
    :memory-bandwidth-gb-s 320.0
    :peak-flops-dp 0.254e12
    :peak-flops-sp 8.1e12}

   "NVIDIA L4"
   {:compute-capability [8 9]
    :sm-count 58
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 25769803776
    :memory-bandwidth-gb-s 300.0
    :peak-flops-dp 0.24e12
    :peak-flops-sp 30.3e12}

   "NVIDIA A10"
   {:compute-capability [8 6]
    :sm-count 72
    :warp-size 32
    :max-threads-per-block 1024
    :shared-memory-per-block 49152
    :registers-per-block 65536
    :global-memory-bytes 25769803776
    :memory-bandwidth-gb-s 600.0
    :peak-flops-dp 0.312e12
    :peak-flops-sp 31.2e12}

   ;; Intel GPUs (Level Zero / OpenCL)

   "Intel(R) Arc(TM) Graphics"  ;; Lunar Lake integrated (Arc 140V)
   {:device-type :level-zero
    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}  ;; XMX DPAS (f16 in, f32 accum)
    :total-eus 64
    :threads-per-eu 8
    :simd-width 16
    :subgroup-sizes [16 32]
    :max-workgroup-size 1024
    :shared-local-memory 131072       ;; 128 KB SLM
    :cache-l2 8388608                 ;; 8 MB L2 (Xe2 LLC) — the residency threshold for the locality
                                      ;; cost term: a working set ≤ this stays WARM, above it goes cold
                                      ;; (measured: proj operands ~7.5 MB fit → 70% peak; 24-pair cold
                                      ;; set ~194 MB evicts → 20%, the 3.5× dram-cold-penalty).
    :global-memory-bytes 15946350592  ;; ~14.9 GB shared LPDDR5x
    :memory-bandwidth-gb-s 89.6      ;; LPDDR5x 8533 MT/s × 128-bit
    :peak-flops-dp 249.6e9           ;; 64 EUs × 1950 MHz × 2 FMA
    :peak-flops-sp 3993.6e9          ;; 64 × 1950 × 2 × 16 SIMD
    ;; XMX DPAS f16 = 8× the FP32 vector rate on Xe2 (int8 = 16× → ~64 TOPS, matching
    ;; Intel's Arc 140V spec). WITHOUT this the descriptor's :f16 peak falls back to the
    ;; f32 value (~4 TF), so balance-for(:f16) computes the WRONG roofline ridge and the
    ;; AM misclassifies every XMX-GEMM as memory-bound. Measured: emit-gemm-tiled hits
    ;; 13.8 TF at 2048³ = ~43% of this peak (~66% of oneDNN) — the B1 headroom.
    :peak-flops-hp 31948.8e9}        ;; 8 × peak-flops-sp (Xe2 XMX f16:FP32 = 8:1)

   "Arc A770"
   {:device-type :level-zero
    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}  ;; XMX DPAS (f16 in, f32 accum)
    :total-eus 512
    :threads-per-eu 8
    :simd-width 16
    :subgroup-sizes [16 32]
    :max-workgroup-size 1024
    :shared-local-memory 65536        ;; 64 KB SLM
    :global-memory-bytes 17179869184  ;; 16 GB GDDR6
    :memory-bandwidth-gb-s 560.0
    :peak-flops-dp 1.23e12
    :peak-flops-sp 19.66e12}

   "Arc B580"
   {:device-type :level-zero
    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}  ;; XMX DPAS (f16 in, f32 accum)
    :total-eus 320
    :threads-per-eu 8
    :simd-width 16
    :subgroup-sizes [16 32]
    :max-workgroup-size 1024
    :shared-local-memory 65536        ;; 64 KB SLM
    :global-memory-bytes 12884901888  ;; 12 GB GDDR6
    :memory-bandwidth-gb-s 456.0
    :peak-flops-dp 0.77e12
    :peak-flops-sp 12.29e12}

   "Data Center GPU Max"  ;; Intel PVC (Ponte Vecchio)
   {:device-type :level-zero
    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}  ;; XMX DPAS (f16 in, f32 accum)
    :total-eus 512
    :threads-per-eu 8
    :simd-width 16
    :subgroup-sizes [16 32]
    :max-workgroup-size 1024
    :shared-local-memory 131072       ;; 128 KB SLM
    :global-memory-bytes 85899345920  ;; 80 GB HBM2e (per tile)
    :memory-bandwidth-gb-s 1600.0     ;; HBM2e per tile
    :peak-flops-dp 22.2e12
    :peak-flops-sp 44.4e12}})

;; ================================================================
;; CPU architecture defaults
;; ================================================================

(def cpu-simd-defaults
  "Default SIMD widths by CPU architecture when Vector API is unavailable.
  Keys are os.arch values."
  {"amd64"   {:simd-width 4 :simd-width-float 8}   ;; AVX2 baseline
   "x86_64"  {:simd-width 4 :simd-width-float 8}
   "aarch64" {:simd-width 2 :simd-width-float 4}   ;; NEON
   "arm64"   {:simd-width 2 :simd-width-float 4}})

;; ================================================================
;; Lookup helpers
;; ================================================================

(defn find-gpu-spec
  "Find GPU spec from catalogue by matching name substring.
  When multiple catalogue entries match, returns the longest match
  (most specific). E.g. 'NVIDIA A100' matches 'NVIDIA A100' not 'NVIDIA A10'.
  Returns the capability map or nil."
  [^String device-name]
  (when device-name
    (let [upper (.toUpperCase device-name)
          matches (filter (fn [[k _]]
                            (.contains upper (.toUpperCase ^String k)))
                          gpu-catalogue)]
      (when (seq matches)
        ;; Pick the longest matching key (most specific)
        (second (apply max-key (comp count first) matches))))))

(defn cpu-defaults-for-arch
  "Get default CPU SIMD widths for the given architecture string."
  [^String arch]
  (get cpu-simd-defaults (or arch "amd64")
       {:simd-width 4 :simd-width-float 8}))

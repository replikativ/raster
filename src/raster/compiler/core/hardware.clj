(ns raster.compiler.core.hardware
  "HardwareDescriptor — the resource budget a schedule is planned against.

  The top-down, hardware-aware planner (see .internal/hardware_aware_planner.md)
  treats scheduling as ALLOCATING an algorithm's iteration space onto a hardware
  resource budget. This namespace is that budget, as data, plus the ANALYTIC
  derivations TVM/Halide leave hard-coded or to search (evidenced by reading both:
  Halide's autoscheduler literally carries `// TODO: 16 should be the number of
  vector registers in the architecture` and never wires it). We wire them.

  The descriptor is the minimal 5 fields — strictly the union of Halide's Target
  (`vector_bits`, ISA feature bits) and its autoscheduler `ArchParams`
  (`last_level_cache_size`, `balance`) plus the register count Halide leaves hard-
  coded — sufficient to derive {lane-width, lane-axis, register-block, regime}:

    :vector-bits           SIMD register width in bits (AVX-512 512, AVX2 256, NEON 128)
    :has-native-dot-reduce true if a widening int dot accumulates horizontally for free
                           (x86 AVX-VNNI vpdpbusd, ARM dotprod sdot, GPU dp4a)
    :num-vector-registers  architectural vector register count (zmm 32, ymm 16, NEON 32)
    :llc-bytes             last-level cache size (roofline knee)
    :balance               load-cost / arith-cost at LLC (Halide ArchParams.balance)

  All derivations are pure functions of (descriptor, problem) — no search, no
  autotuner. `*descriptor*` is the active target; defaults to the detected host.

  ONE record across backends (no duplicate hardware tables). The GPU runtimes
  (raster.gpu.ze-runtime / ocl-runtime) already query raw per-device capabilities
  for runtime DISPATCH (:simd-width, :integrated?, :total-eus, :max-work-group-size,
  :global-mem-bytes). That extraction stays single-sourced there; a GPU target
  descriptor is PROJECTED from it via `from-gpu-device-info` rather than re-queried.
  So the planner consumes one descriptor shape for :cpu and :gpu; the runtime owns
  the FFI queries. (Full unification of the two records is a later merge.)")

;; ---------------------------------------------------------------------------
;; Element widths — bytes per scalar of a raster dtype keyword.
;; ---------------------------------------------------------------------------

(def ^:private dtype-bytes
  {:double 8 :float 4 :long 8 :int 4 :short 2 :byte 1
   :f64 8 :f32 4 :i64 8 :i32 4 :i16 2 :i8 1})

(defn bytes-of
  "Bytes per scalar of a dtype keyword (e.g. :float -> 4, :byte -> 1)."
  [dt]
  (or (dtype-bytes dt)
      (throw (ex-info "Unknown dtype for hardware width" {:dtype dt}))))

;; ---------------------------------------------------------------------------
;; Host detection
;; ---------------------------------------------------------------------------

(defn detect-vector-bits
  "Probe the running JVM's preferred SIMD width in bits via the Vector API's
   preferred f32 species length (4 lanes -> AVX2/256, 8 -> AVX-512/512, 2 -> NEON/
   wasm 128). Returns nil if the incubator module is unavailable."
  []
  (try
    (let [lanes (.length ^jdk.incubator.vector.VectorSpecies
                         (jdk.incubator.vector.FloatVector/SPECIES_PREFERRED))]
      (* (int lanes) 32))
    (catch Throwable _ nil)))

(defn host-descriptor
  "The descriptor for the running host. Vector width is probed; the register count
   follows the width (AVX-512 -> 32 zmm, else 16 ymm); native int-dot-reduce is
   assumed on x86-64 (the -march=native C build selects vpdpbusd, falling back to
   maddubs); LLC/balance use Halide's CPU defaults. Override any field via `with`."
  []
  (let [vb (or (detect-vector-bits) 256)
        x86? (let [a (System/getProperty "os.arch" "")]
               (or (.contains a "amd64") (.contains a "x86")))]
    {:device-type           :cpu
     :vector-bits           vb
     :has-native-dot-reduce (boolean x86?)
     :num-vector-registers  (if (>= vb 512) 32 16)
     :llc-bytes             (* 16 1024 1024)
     :balance               40}))

(defn from-gpu-device-info
  "Project a GPU runtime device-info map (as returned by raster.gpu.ze-runtime /
   ocl-runtime — :simd-width, :integrated?, :total-eus, :global-mem-bytes, ...) into
   a HardwareDescriptor so the planner consumes ONE record shape across :cpu/:gpu.
   The runtime owns the FFI extraction; this is a pure projection, never a re-query.
   A GPU's int dot-reduce is dp4a/DPAS (assumed present on the dl-targeted parts);
   the SIMD width is the subgroup × dtype lanes, modelled here as simd-width*32 bits
   for the f32 lane analogue. Memory/registers map to the device's global mem and a
   nominal GRF count. (Refined as the GPU planner path lands.)"
  [{:keys [simd-width integrated? global-mem-bytes] :as dev}]
  {:device-type           :gpu
   :vector-bits           (* (long (or simd-width 16)) 32)
   :has-native-dot-reduce true
   :num-vector-registers  128                ;; GRF-backed; nominal until profiled
   :llc-bytes             (long (or global-mem-bytes (* 1024 1024 1024)))
   :balance               60                 ;; GPU is more bandwidth-tolerant than CPU
   :integrated?           (boolean integrated?)
   :gpu-device-info       dev})

(def ^:dynamic *descriptor*
  "The active hardware target the planner schedules against."
  (host-descriptor))

(defn with
  "Return a copy of the active (or given) descriptor with overrides merged. Handy
   for cross-targeting (e.g. (with {:vector-bits 512}) to plan for AVX-512)."
  ([overrides] (merge *descriptor* overrides))
  ([desc overrides] (merge desc overrides)))

;; ---------------------------------------------------------------------------
;; Analytic derivations
;; ---------------------------------------------------------------------------

(defn natural-lanes
  "SIMD lane count for a dtype on this target: vector-bits / (8 * bytes-per-elem).
   Halide `natural_vector_size`. f32 -> 8 on AVX-512, 4 on AVX2."
  [desc dt]
  (quot (long (:vector-bits desc)) (* 8 (bytes-of dt))))

(defn column-tile
  "NC — the output-column tile width for a SIMD GEMV whose float accumulators live
   in the vector lanes (one column per lane, no horizontal reduce). This is the
   interleave width the weight LAYOUT must be packed to (repack-stream's NC): a
   hardware-derived number, not a hard-coded 8. = f32 lanes (AVX2 8, AVX-512 16,
   NEON 4)."
  [desc]
  (natural-lanes desc :float))

(defn register-block
  "How many independent accumulator TILES to keep live in vector registers at once
   — the register-blocking factor Halide hard-codes (12/16) and TVM passes as a
   literal. Analytic: with `regs-per-tile` vector registers consumed per tile
   (accumulators + live operands), at most floor(num-vector-registers / regs-per-
   tile) fit; clamp to >=1 and to `latency-target` (independent chains needed to
   hide FMA latency — beyond that, more tiles waste registers). Defaults model the
   stream-GEMV: 1 accumulator + ~1 operand reg per tile, ~4 chains to hide FMA."
  ([desc] (register-block desc {}))
  ([desc {:keys [regs-per-tile latency-target] :or {regs-per-tile 2 latency-target 4}}]
   (max 1 (min (long latency-target)
               (quot (long (:num-vector-registers desc)) (long regs-per-tile))))))

(defn roofline-regime
  "Memory-bound vs compute-bound by arithmetic intensity vs machine balance.
   AI = flops/bytes; AI < balance -> :memory-bound (stream schedule); else
   :compute-bound (tile schedule). Generalizes classify-regime: a decode GEMV
   (flops ~= bytes, AI ~= 1 << balance) is memory-bound; a prefill GEMM reusing
   each weight M times raises AI above balance."
  [desc {:keys [flops bytes]}]
  (if (< (/ (double flops) (double (max 1 bytes))) (double (:balance desc)))
    :memory-bound
    :compute-bound))

(defn reduce-intrinsic
  "The native widening-int-dot reduce instruction available on this target, or
   :scalar if none — the per-target leaf the int8-MAC seam lowers to. (Width and
   exact opcode are resolved at emit time against the lane count.)"
  [desc]
  (if (:has-native-dot-reduce desc) :dpbusd :scalar))

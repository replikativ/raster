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

  ONE detection source (no duplicate hardware tables). raster.runtime.hardware is the
  established device registry — it auto-detects CPU (Vector-API SIMD width, cache sizes,
  cores) and GPU (Level-Zero/OpenCL caps, subgroup size, EUs, bandwidth) and exposes a
  catalogue for cross-compilation. THIS namespace does NOT re-detect: `descriptor-for`
  PROJECTS a runtime.hardware device into the planner's descriptor shape and adds the
  compiler's analytic fields; the derivations below operate on that. So the runtime owns
  the probing; the compiler owns the schedule math, over one record for :cpu and :gpu."
  (:require [raster.runtime.hardware :as rt]))

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

(defn descriptor-for
  "Project a runtime.hardware device into the planner's HardwareDescriptor — the SINGLE
   detection source (runtime.hardware owns the probing; this adds the compiler's analytic
   fields). CPU and GPU via one path (subsumes the old from-gpu-device-info). The
   register count follows the width (AVX-512 -> 32 zmm, else 16 ymm; GPU is GRF-backed);
   native int-dot-reduce is x86 vpdpbusd / GPU dp4a; LLC comes from the detected L3 (or
   GPU global mem); balance uses Halide-style defaults."
  [device-id]
  (let [caps   (:capabilities (rt/device device-id))
        gpu?   (not= (rt/device-type device-id) :cpu)
        flanes (long (if gpu? (rt/subgroup-size device-id)
                         (rt/simd-lanes device-id :float)))   ;; f32 lanes / subgroup
        vbits  (* flanes 32)]
    (cond-> {:device-type (if gpu? :gpu :cpu)
             :device-id   device-id
             :vector-bits vbits
             :has-native-dot-reduce
             (if gpu? true
                 (let [a (str (:arch caps))]
                   (boolean (or (.contains a "amd64") (.contains a "x86")))))
             :num-vector-registers (if gpu? 128 (if (>= vbits 512) 32 16))
             :llc-bytes   (or (:cache-l3 caps)
                              (when gpu? (:global-memory-bytes caps))
                              (* 16 1024 1024))
             :balance     (if gpu? 60 40)}
      gpu? (assoc :integrated? (boolean (:integrated? caps))
                  :max-workgroup-size (long (or (:max-workgroup-size caps) 256)))
      ;; MACHINE WIDTH — work-items in flight when the device is full: EUs x hw threads per
      ;; EU x SIMD lanes. Every GPU launch geometry is a function of this and the problem
      ;; shape, so it belongs on the descriptor rather than as a literal at each launch site
      ;; (it was 8192, hardcoded, in three of them). Arc 140V: 64 x 8 x 16 = 8192.
      ;;
      ;; Set ONLY when the device actually reported all three. Defaulting a missing EU count
      ;; to 1 would silently yield machine-lanes=16 — split-k would never fire and every
      ;; elementwise kernel would mis-vectorize, with no error anywhere. A missing value
      ;; leaves the key ABSENT, and the derivations below fall back to a documented default.
      (and gpu? (:total-eus caps) (:threads-per-eu caps)
           (or (:simd-width caps) (rt/subgroup-size device-id)))
      (assoc :machine-lanes (* (long (:total-eus caps))
                               (long (:threads-per-eu caps))
                               (long (or (:simd-width caps) (rt/subgroup-size device-id))))))))

(defn host-descriptor
  "The descriptor for the running host CPU (projected from runtime.hardware :cpu:0)."
  []
  (descriptor-for :cpu:0))

;; Host detection is deferred to first use (runtime.hardware/init! probes GPUs over FFI)
;; — not paid at load. Production callers pass an explicit descriptor; this is the default.
(defonce ^:private host-desc (delay (host-descriptor)))

(def ^:dynamic *descriptor*
  "The active hardware target the planner schedules against; nil = the detected host
   (resolved lazily via `active-descriptor`). Bind to cross-target."
  nil)

(defn active-descriptor
  "The descriptor in force: the bound *descriptor*, else the (lazily detected) host."
  []
  (or *descriptor* @host-desc))

(defn with
  "Return a copy of the active (or given) descriptor with overrides merged. Handy
   for cross-targeting (e.g. (with {:vector-bits 512}) to plan for AVX-512)."
  ([overrides] (merge (active-descriptor) overrides))
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

;; ---------------------------------------------------------------------------
;; GPU launch geometry — a schedule is a function of (shape, machine width),
;; never of the model. These are the two decisions every GPU launch site needs.
;; ---------------------------------------------------------------------------

(defn fill-workgroups
  "Workgroups of `wg` work-items needed to fill the device once = machine-lanes / wg.
   A launch below this count leaves EUs idle and is OCCUPANCY-bound — no amount of
   inner-loop tuning helps it. Arc 140V at wg=256: 8192/256 = 32."
  [desc wg]
  (max 1 (quot (long (:machine-lanes desc 8192)) (long wg))))

(defn stream-vector-width
  "Vector width (1 | 2 | 4) for an ELEMENTWISE/streaming kernel over `n` elements.

   Two forces pull against each other, and the machine width is what balances them:
     - WIDER is better per lane: this iGPU's elementwise kernels are REQUEST-RATE bound,
       not DRAM bound (a scalar f32 copy tops out ~53 GB/s while an f64 copy over the same
       work-item count reaches ~78 — same requests, twice the bytes). float4 quarters the
       requests per lane at identical DRAM traffic.
     - WIDER costs work-items: n/w of them. Drop below `machine-lanes` and EUs go idle,
       which is exactly how a naive `always float4` LOSES on a mid-sized kernel (n=16384
       scalar = 16384 work-items = full; float4 = 4096 = HALF the machine idle).

   So: take the widest w whose work-item count still fills the machine, cap at 4, and
   fall back to scalar for anything too small to fill it at all (those kernels are
   launch-bound; vectorizing them only removes lanes)."
  [desc n]
  (let [lanes (long (:machine-lanes desc 8192))
        n (long n)]
    (cond
      (>= (quot n 4) lanes) 4
      (>= (quot n 2) lanes) 2
      :else 1)))

(defn reduction-accumulators
  "Independent vector accumulators to keep live in a SIMD REDUCTION to hide FMA latency
   (the register-blocking facet for a reduce, vs `register-block` for a GEMM tile).
   Wider machines hide more latency: ≥8 dtype-lanes -> 8, else 4. Hardware-aware via the
   descriptor's lane count for the element type — the one place this policy lives (the
   JVM segop-simd reducer consults this rather than its own lanes check)."
  [desc elem-type]
  (if (>= (natural-lanes desc elem-type) 8) 8 4))

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

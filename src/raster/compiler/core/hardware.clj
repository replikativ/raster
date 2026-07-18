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

(defn- merge-measured
  "Overlay a device's :measured microbench layer (raster.runtime.microbench) onto the probed/
   analytic descriptor: measured bandwidth / peak-flops / launch-overhead OVERRIDE the guess, the
   whole map is kept under :measured for inspection, and its provenance is carried. This is the
   feedback edge — measurement improving the readable model, not a cache beside it."
  [desc device-id]
  (if-let [m (rt/measured-for device-id)]
    (cond-> (assoc desc :measured m)
      (:bandwidth-bytes-s m)  (assoc :bandwidth-bytes-s (:bandwidth-bytes-s m))
      (:peak-flops m)         (update :peak-flops merge (:peak-flops m))
      (:launch-overhead-ns m) (assoc :launch-overhead-ns (:launch-overhead-ns m))
      (:provenance m)         (update :provenance merge (:provenance m)))
    desc))

(defn- build-descriptor
  "Project a runtime.hardware device into the planner's HardwareDescriptor — the SINGLE
   detection source (runtime.hardware owns the probing; this adds the compiler's analytic
   fields). CPU and GPU via one path (subsumes the old from-gpu-device-info). The
   register count follows the width (AVX-512 -> 32 zmm, else 16 ymm; GPU is GRF-backed);
   native int-dot-reduce is x86 vpdpbusd / GPU dp4a; LLC comes from the detected L3 (or
   GPU global mem); balance uses Halide-style defaults. The :measured layer is overlaid by
   descriptor-for (below)."
  [device-id]
  (let [caps   (:capabilities (rt/device device-id))
        gpu?   (not= (rt/device-type device-id) :cpu)
        flanes (long (if gpu? (rt/subgroup-size device-id)
                         (rt/simd-lanes device-id :float)))   ;; f32 lanes / subgroup
        vbits  (* flanes 32)
        ;; PERFORMANCE quantities — bandwidth / per-dtype peak-flops / cache hierarchy — projected
        ;; from whatever the probe or the shipped catalogue supplied (runtime.hardware merges the
        ;; catalogue into caps). These are the roofline inputs (balance-for / roofline-time-ns).
        ;; A CPU probe carries no bandwidth/flops (not statically knowable) → they are simply
        ;; ABSENT here and the analytic fns fall back to :balance; the :measured layer (Phase 1)
        ;; fills them from a microbenchmark and OVERWRITES the analytic guess.
        bw-gb  (:memory-bandwidth-gb-s caps)
        peak-flops (into {} (remove (comp nil? val))
                         {:f32 (:peak-flops-sp caps) :float (:peak-flops-sp caps)
                          :f64 (:peak-flops-dp caps) :double (:peak-flops-dp caps)
                          :f16 (:peak-flops-hp caps) :half (:peak-flops-hp caps)})
        cache  (into {} (remove (comp nil? val))
                     {:l1 (:cache-l1 caps) :l2 (:cache-l2 caps) :l3 (:cache-l3 caps)
                      :slm (when gpu? (:shared-local-memory caps))})
        ;; GPU occupancy inputs the launch-geometry derivations (workgroup-size / group-count /
        ;; block-size / grid-size) read — present-only, projected from caps. Intel Level-Zero uses
        ;; :total-eus/:threads-per-eu; NVIDIA CUDA uses :sm-count/:max-warps-per-sm/:max-blocks-per-sm.
        gpu-geo (when gpu?
                  (into {} (remove (comp nil? val))
                        {:total-eus (:total-eus caps) :threads-per-eu (:threads-per-eu caps)
                         :sm-count (:sm-count caps) :fpus-per-core (:fpus-per-core caps)
                         :max-warps-per-sm (:max-warps-per-sm caps) :max-blocks-per-sm (:max-blocks-per-sm caps)
                         :registers-per-block (:registers-per-block caps)
                         :shared-memory-per-block (:shared-memory-per-block caps)}))]
    (cond-> {:device-type (if gpu? :gpu :cpu)
             :device-id   device-id
             :vector-bits vbits
             :has-native-dot-reduce
             (if gpu? true
                 ;; x86 has a widening int-dot-reduce ONLY with AVX-VNNI (vpdpbusd). Prefer the
                 ;; probed feature set (correct: a pre-VNNI Haswell is x86 yet lacks vpdpbusd);
                 ;; fall back to the coarse arch heuristic only when features weren't probed.
                 (let [feats (:simd-features caps)
                       a (str (:arch caps))]
                   (if feats
                     (boolean (some feats [:avx-vnni :avx512-vnni :vnni]))
                     (boolean (or (.contains a "amd64") (.contains a "x86"))))))
             :num-vector-registers (if gpu? 128 (if (>= vbits 512) 32 16))
             :llc-bytes   (or (:cache-l3 caps)
                              (when gpu? (:global-memory-bytes caps))
                              (* 16 1024 1024))
             :balance     (if gpu? 60 40)}
      bw-gb           (assoc :bandwidth-bytes-s (* (double bw-gb) 1e9))
      (seq peak-flops) (assoc :peak-flops peak-flops)
      (seq cache)     (assoc :cache cache)
      gpu? (assoc :integrated? (boolean (:integrated? caps))
                  :subgroup-size flanes
                  ;; GRF budget per LANE = grf-regs/thread × reg-bytes / subgroup-size. INTEL Xe/Xe2
                  ;; grf128 model: 128 GRF registers/thread of 32 B (256-bit); per SIMD lane that is
                  ;; 128×32/subgroup. Arc 140V (subgroup 16): 256 B/lane — the budget the schedule
                  ;; feasibility gate (raster.gpu.schedule/feasible?) charges the accumulator +
                  ;; register staging against.
                  ;; NOTE (vendor-specificity, HIP/CUDA follow-up): 128×32 is the Intel-Xe-grf128
                  ;; constant. NVIDIA (255 regs × 4 B / warp 32 ≈ 1020 B/lane) and AMD VGPR budgets
                  ;; differ ~8× — this value is only correct for Intel targets until it is derived
                  ;; from per-vendor caps (with descriptor :vendor/:arch). The gate must not be
                  ;; trusted for a non-Intel descriptor yet. (:num-vector-registers above is a
                  ;; CPU-shaped placeholder; this is the Intel GPU value.)
                  :grf-bytes-per-lane (long (quot (* 128 32) (max 1 flanes)))
                  :max-workgroup-size (long (or (:max-workgroup-size caps) 256)))
      ;; MATRIX UNIT (systolic array) — the hardware matrix-multiply shape {:family :m :n :k
      ;; :subgroup}, e.g. Intel XMX DPAS 8×16×16. A catalogue fact (not always probeable); the
      ;; GEMM tile generator (derive-gemm-tile) reads it to size the accumulator tile + K-unroll,
      ;; and the emitter keys its instruction family off :family (:dpas vs a WGMMA/MFMA fork).
      (and gpu? (:matrix caps)) (assoc :matrix (:matrix caps))
      (seq gpu-geo) (merge gpu-geo)
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

(defn descriptor-for
  "The HardwareDescriptor for `device-id`: the probed/analytic model (build-descriptor) with the
   :measured microbench layer overlaid (measurement OVERRIDES the guess, tagged in :provenance).
   This is the single entry the planner reads — probe ⊕ catalogue ⊕ measured ⊕ derived."
  [device-id]
  (merge-measured (build-descriptor device-id) device-id))

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

   These kernels are REQUEST-RATE bound on an iGPU, not DRAM bound: a scalar f32 copy tops
   out ~53 GB/s while an f64 copy over the SAME work-item count reaches ~78 — same requests,
   twice the bytes, more bandwidth. Widening to w elements per work-item cuts the memory
   requests per lane w-fold at identical DRAM traffic.

   The obvious counter-force is that w elements per item means n/w ITEMS, so a wide vector
   on a small kernel leaves EUs idle. That suggests 'take the widest w that still fills the
   machine' — and MEASUREMENT SAYS THAT RULE IS WRONG. On the gemma layer's f32→f16 casts
   (median n=16384 on a 8192-lane part) float4 leaves HALF the machine idle (4096 items =
   16 workgroups of the 32 that fill it) and still beat float2 at full occupancy, at both
   B=1 and B=16. A 16-byte request carries more memory-level parallelism per thread than the
   idle EUs cost: half a machine issuing wide requests beats a full machine issuing narrow
   ones.

   So: go as WIDE as possible, and only fall back when the launch gets genuinely small — the
   floor is a quarter of the machine's work-items, below which a kernel is launch-bound and
   removing its lanes helps nothing."
  [desc n]
  (let [lanes (long (:machine-lanes desc 8192))
        floor (max 1 (quot lanes 4))                ;; smallest work-item count worth widening
        n (long n)]
    (cond
      (>= (quot n 4) floor) 4
      (>= (quot n 2) floor) 2
      :else 1)))

(defn reduction-accumulators
  "Independent vector accumulators to keep live in a SIMD REDUCTION to hide FMA latency
   (the register-blocking facet for a reduce, vs `register-block` for a GEMM tile).
   Wider machines hide more latency: ≥8 dtype-lanes -> 8, else 4. Hardware-aware via the
   descriptor's lane count for the element type — the one place this policy lives (the
   JVM segop-simd reducer consults this rather than its own lanes check)."
  [desc elem-type]
  (if (>= (natural-lanes desc elem-type) 8) 8 4))

;; ---------------------------------------------------------------------------
;; Roofline — the analytic performance model (XLA gpu_performance_model shape).
;; Every input is a descriptor field, so it improves transparently as the
;; :measured layer (Phase 1) overwrites the probed/catalogue/default constants.
;; ---------------------------------------------------------------------------

(defn peak-flops-for
  "Peak FLOP/s for `dtype` on this target (from probe/catalogue/measured), or nil if unknown.
   f16/half falls back to f32 when the target reports no separate half-rate."
  [desc dtype]
  (let [pf (:peak-flops desc)]
    (or (get pf dtype)
        (case dtype
          (:float :f32) (:f32 pf)
          (:double :f64) (:f64 pf)
          (:half :f16) (or (:f16 pf) (:f32 pf))   ;; no half-rate reported → f32 ceiling (conservative)
          nil))))

(defn balance-for
  "The roofline RIDGE (flops/byte) for `dtype`: peak-flops(dtype) / bandwidth — the arithmetic
   intensity above which a kernel is compute-bound. PRECISION-DEPENDENT (the f16 ridge is far above
   the f64 ridge on a mixed-rate GPU), derived from the descriptor's bandwidth+peak-flops when both
   are present; falls back to the scalar :balance default only when they are absent. Fixes the bug
   where a single :balance constant misclassifies every non-default-precision kernel."
  [desc dtype]
  (let [pf (peak-flops-for desc dtype)
        bw (:bandwidth-bytes-s desc)]
    (if (and pf bw (pos? (double bw)))
      (/ (double pf) (double bw))
      (double (:balance desc 40)))))

(defn roofline-regime
  "Memory-bound vs compute-bound by arithmetic intensity vs the ridge. AI = flops/bytes;
   AI < ridge -> :memory-bound (stream schedule), else :compute-bound (tile schedule).
   The 3-arg form uses the PRECISION-AWARE ridge (`balance-for`); the 2-arg form keeps the legacy
   scalar :balance (back-compat for callers with no dtype)."
  ([desc {:keys [flops bytes]}]
   (if (< (/ (double flops) (double (max 1 bytes))) (double (:balance desc)))
     :memory-bound :compute-bound))
  ([desc {:keys [flops bytes]} dtype]
   (if (< (/ (double flops) (double (max 1 bytes))) (balance-for desc dtype))
     :memory-bound :compute-bound)))

(def ^:private memory-compute-parallelism
  "Overlap fraction of the compute and memory ceilings (XLA kMemoryComputeParallelism): exec is a
   SOFT roofline elbow ≈ max(compute,mem), not their sum."
  0.95)

(def ^:private default-launch-overhead-ns
  "Per-kernel dispatch tax (XLA kKernelLaunchOverhead = 1µs) — the term that makes fusion pay for
   itself analytically (fewer kernels ⇒ less tax). Overwritten by a measured empty-kernel ping."
  1000.0)

(defn roofline-time-ns
  "Analytic runtime estimate in ns (XLA overlap roofline). compute = flops/peak-flops(dtype);
   mem = bytes/bandwidth; exec = compute + mem − 0.95·min(compute,mem) + n-kernels·launch-tax.
   Returns nil when the descriptor has no bandwidth/peak-flops yet (nothing probed or measured) —
   the caller then has no estimate rather than a fabricated one. This is the ranker the schedule
   search (Phase 2) prices candidates with; measurement (Phase 1) sharpens every term."
  [desc {:keys [flops bytes dtype n-kernels launch-overhead-ns]
         :or {dtype :f32 n-kernels 1}}]
  (let [pf (peak-flops-for desc dtype)
        bw (:bandwidth-bytes-s desc)]
    (when (and pf bw (pos? (double pf)) (pos? (double bw)))
      (let [compute-ns (* (/ (double flops) (double pf)) 1e9)
            mem-ns     (* (/ (double bytes) (double bw)) 1e9)
            overlap    (* memory-compute-parallelism (min compute-ns mem-ns))
            tax        (* (long n-kernels)
                          (double (or launch-overhead-ns (:launch-overhead-ns desc) default-launch-overhead-ns)))]
        (+ compute-ns mem-ns (- overlap) tax)))))

(defn reduce-intrinsic
  "The native widening-int-dot reduce instruction available on this target, or
   :scalar if none — the per-target leaf the int8-MAC seam lowers to. (Width and
   exact opcode are resolved at emit time against the lane count.)"
  [desc]
  (if (:has-native-dot-reduce desc) :dpbusd :scalar))

;; ---------------------------------------------------------------------------
;; Occupancy-scored launch geometry — the SINGLE implementation. These read the
;; descriptor (probe ⊕ catalogue ⊕ measured); runtime.hardware/optimal-* delegate
;; here so there is one occupancy model, not two. The math is occupancy-maximizing
;; over subgroup/warp-multiple candidates (Halide-style enumeration).
;; ---------------------------------------------------------------------------

(defn- enumerate-multiples
  "Candidate sizes = multiples of `unit` up to `max-size` (powers of 2, plus ×3 — the
   Halide tiling enumeration), filtered to exact multiples of `unit`."
  [unit max-size]
  (let [pow2s (take-while #(<= % max-size) (iterate #(* 2 %) unit))
        pow2x3 (filter #(<= % max-size) (map #(* 3 %) pow2s))]
    (filter #(zero? (rem % unit)) (sort (distinct (concat pow2s pow2x3))))))

(defn workgroup-size
  "Intel Level-Zero occupancy-optimal workgroup size for `n` elements. EU model:
   EUs × threads-per-EU ÷ subgroup = max concurrent subgroups; score candidates by how
   well they fill the EUs (reductions prefer larger groups). Reads :subgroup-size,
   :max-workgroup-size, :total-eus, :threads-per-eu off the descriptor."
  [desc n & {:keys [reduction?] :or {reduction? false}}]
  (let [sg-size (long (:subgroup-size desc 16))
        max-wg  (long (:max-workgroup-size desc 1024))
        threads-per-eu (long (:threads-per-eu desc 8))
        ;; powers-of-2 multiples of the subgroup (the Level-Zero workgroup enumeration; block-size's
        ;; CUDA path additionally tries ×3 tiles — kept distinct to preserve the prior behavior).
        candidates (take-while #(<= % max-wg) (iterate #(* 2 %) sg-size))
        scored (map (fn [wg]
                      (let [subgroups-per-wg (quot wg sg-size)
                            wgs-per-eu (max 1 (quot threads-per-eu subgroups-per-wg))
                            occupancy (min 1.0 (/ (* wgs-per-eu subgroups-per-wg sg-size)
                                                  (* threads-per-eu 1.0)))
                            score (if reduction? (* occupancy (Math/log (double wg))) occupancy)]
                        {:workgroup-size wg :score score}))
                    candidates)]
    (:workgroup-size (apply max-key :score scored))))

(defn group-count
  "Level-Zero group count (grid) for `n` and `wg`: capped at EU-count × waves (grid-stride)."
  [desc n wg & {:keys [reduction?] :or {reduction? false}}]
  (let [eus (long (:total-eus desc 64))
        waves (if reduction? 1 2)
        needed (int (Math/ceil (/ (double n) (double wg))))]
    (min needed (* eus waves))))

(defn derive-gemm-tile
  "The default XMX-GEMM tile for a descriptor — {:block-m :block-n :sg-m :sg-n :block-k :matrix},
   the argument map for raster...opencl-codegen/emit-gemm-tiled. DERIVED, not hardcoded:

     - per-subgroup accumulator tile (sg-m×sg-n) is GRF-BOUND: it holds sg-m·sg-n/subgroup f32
       accumulators per lane, so sg-m·sg-n ≤ grf-bytes-per-lane·subgroup/4. Take the largest square
       tile under that cap, rounded DOWN to the matrix M_i/N_i granularity.
     - workgroup tile is `wg-subgroups` (default 16 = a 4×4 subgroup grid) copies of the sg-tile.
     - K-unroll is 2× the matrix K (double-buffered DPAS depth).

   On the Arc 140V descriptor (matrix 8×16×16, GRF 256 B/lane, subgroup 16) this yields exactly the
   hand-tuned 128×128 / 32×32 / K32 config — the T1 default — with zero magic numbers. Other Intel
   parts (different GRF/subgroup) get a correctly-rescaled tile from the same rule. `opts` may pass
   :wg-subgroups to override the workgroup subgroup count (the occupancy knob T3 autotunes)."
  ([desc] (derive-gemm-tile desc {}))
  ([desc opts]
   (let [{:keys [m n k subgroup] :or {m 8 n 16 k 16 subgroup 16}} (:matrix desc)
         grf     (long (:grf-bytes-per-lane desc 256))
         sg      (long subgroup)
         acc-cap (quot (* grf sg) 4)                 ;; max sg-m·sg-n (f32 accumulators)
         side    (long (Math/sqrt (double acc-cap))) ;; largest square side
         sg-m    (* m (max 1 (quot side m)))         ;; round down to M_i multiple
         sg-n    (* n (max 1 (quot side n)))         ;; round down to N_i multiple
         wg-sub  (long (:wg-subgroups opts 16))
         side-sg (max 1 (long (Math/sqrt (double wg-sub))))]
     {:block-m (* sg-m side-sg)
      :block-n (* sg-n side-sg)
      :sg-m sg-m :sg-n sg-n
      :block-k (* 2 k)
      :matrix (:matrix desc {:family :dpas :m m :n n :k k :subgroup subgroup})})))

(defn gemm-tile-candidates
  "A CURATED set of valid GEMM tiles for `desc` — the autotuner's search space for the tile axis.
   Not a free ×2/÷2 product (that explodes and yields non-divisible / register-spilling tiles):
   the per-subgroup accumulator tile stays at the GRF-bound derived size, and only the WORKGROUP
   block (occupancy) and K-unroll (pipeline depth) vary — the two axes with real, measurable
   headroom. Every candidate keeps block divisible by the subgroup tile and workgroup ≤
   max-workgroup-size, so all are feasible by construction."
  [desc]
  (let [base (derive-gemm-tile desc)
        {:keys [sg-m sg-n block-k matrix]} base
        max-wg (long (:max-workgroup-size desc 1024))
        sg     (long (:subgroup matrix 16))
        wg-ok? (fn [{:keys [block-m block-n]}]
                 (<= (* (quot (long block-m) (long sg-m)) (quot (long block-n) (long sg-n)) sg) max-wg))]
    (->> [base
          (assoc base :block-m sg-m :block-n sg-n)                    ;; 1 subgroup — max workgroups (small problems)
          (assoc base :block-m (* 2 sg-m) :block-n (* 2 sg-n))        ;; 4 subgroups
          (assoc base :block-m (* 2 (:block-m base)))                 ;; taller M block
          (assoc base :block-n (* 2 (:block-n base)))                 ;; wider N block
          (assoc base :block-k (* 2 block-k))                         ;; deeper K unroll
          (assoc base :block-k (max (long (:k matrix 16)) (quot block-k 2)))] ;; shallower K
         (filter wg-ok?)
         distinct
         vec)))

(defn block-size
  "NVIDIA CUDA occupancy-optimal block size for `n`. warp-multiple candidates scored by
   occupancy against :max-warps-per-sm/:max-blocks-per-sm. Falls back to 256/512."
  [desc n & {:keys [reduction?] :or {reduction? false}}]
  (let [ws (long (:subgroup-size desc 32))          ;; warp size
        max-tpb (long (:max-workgroup-size desc 1024))
        max-warps (long (:max-warps-per-sm desc 64))
        max-blocks (long (:max-blocks-per-sm desc 32))
        candidates (enumerate-multiples ws max-tpb)]
    (if (> (count candidates) 1)
      (:block-size
       (apply max-key :score
              (map (fn [bs]
                     (let [warps-per-block (quot bs ws)
                           blocks-per-sm (min max-blocks (quot max-warps warps-per-block))
                           occupancy (/ (double (* blocks-per-sm warps-per-block)) max-warps)
                           score (if reduction? (* occupancy (Math/log (double bs))) occupancy)]
                       {:block-size bs :score score}))
                   candidates)))
      (let [preferred (if reduction? 512 256)]
        (* ws (quot (min max-tpb (max ws preferred)) ws))))))

(defn grid-size
  "NVIDIA CUDA grid size for `n` and `block`: blocks-per-SM × SM-count × waves, capped at needed."
  [desc n block & {:keys [reduction?] :or {reduction? false}}]
  (let [sm (long (:sm-count desc 108))
        ws (long (:subgroup-size desc 32))
        max-warps (long (:max-warps-per-sm desc 64))
        max-blocks (long (:max-blocks-per-sm desc 32))
        warps-per-block (quot block ws)
        blocks-per-sm (min max-blocks (quot max-warps warps-per-block))
        waves (if reduction? 1 2)]
    (min (int (Math/ceil (/ (double n) block))) (* blocks-per-sm sm waves))))

(defn launch-config
  "CUDA launch config {:block-size :grid-size :shared-mem} for `n`, or nil when `n` is below
   `min-elements` (too small to be worth a GPU launch). shared-mem = block × 8 for reductions."
  [desc n & {:keys [reduction? min-elements] :or {reduction? false min-elements 1024}}]
  (when (>= n min-elements)
    (let [bs (block-size desc n :reduction? reduction?)
          gs (grid-size desc n bs :reduction? reduction?)]
      {:block-size bs :grid-size gs :shared-mem (if reduction? (* bs 8) 0)})))

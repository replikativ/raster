(ns raster.gpu.schedule
  "The schedule layer (S6) — schedule-as-DATA plus the register-budget feasibility gate.

   A `Schedule` is an EDN map attached to a compiled artifact (rides on the `Compiled`
   record's `:schedule` field). It unifies the four scattered representations of today's
   tuning knobs — the `gemm-schedule` pure fn, the `:gemm-precision` descriptor field, the
   `*gemm-splitk-*` dynamic vars, and the hardcoded launch geometry — into ONE inspectable,
   pinnable structure, and adds the piece all three reference compilers (Halide anderson2021,
   oneDNN, Triton) have and raster lacked: a register-budget feasibility gate that rejects an
   infeasible schedule at COMPILE time instead of discovering it as a measurement-time spill.

   PR-1 scope (see .internal/schedule_layer_design.md §4): the representation + the gate + the
   ONE robust axis (precision) wired end-to-end. Launch-geometry unification (the 256-literal
   kill), split-k promotion, and the autotuner are later increments. The dead inner-loop knobs
   (`:grf`, `:stage`) have a HOME here as opt-in, default-off data policed by the gate — never
   an untracked emitter branch.

   Three-stage mechanism (per axis): descriptor-derived default → optional data override →
   (deferred) autotuner. `derive-default` and `feasible?` both take the HardwareDescriptor as an
   ARGUMENT (probed OR supplied) so target-aware compilation is 'pass a different descriptor',
   not a retrofit — the Intel-GRF vs AMD-VGPR budget flows through the same gate."
  (:refer-clojure :exclude [resolve])
  (:require [raster.compiler.core.hardware :as hw]))

;; ================================================================
;; Stage 1 — derive-default
;; ================================================================

(def ^:private default-precision
  "The one POLICY axis. Training default: mixed-precision f16 GEMM inputs, f32 accumulate —
   the measured-robust 2.33× backward speedup at grads cosine 1.000."
  :f16-xmx)

(defn- machine-params
  "The cost/legality numbers the schedule carries in :meta for inspection + the gate."
  [desc]
  {:machine-lanes      (:machine-lanes desc 8192)
   :grf-bytes-per-lane (:grf-bytes-per-lane desc 256)
   :subgroup           (:subgroup-size desc (:subgroup desc 16))})

(defn derive-default
  "Stage 1: (program-steps × HardwareDescriptor) → Schedule. Replaces hardcodes with analytic
   descriptor values. PR-1 wires the precision axis; :grf/:stage are the dead-knob homes,
   default OFF (:grf128 / :stage :none) so the emitter never sees a dead default on.

   opts: {:precision :f16-xmx|:f32-scalar}  — policy override of the training default."
  ([desc] (derive-default nil desc {}))
  ([steps desc] (derive-default steps desc {}))
  ([steps desc {:keys [precision]}]
   {:precision (or precision default-precision)
    :tile  (hw/derive-gemm-tile desc)          ;; axis 9 — the GEMM tile (T2 derivation), autotunable
    :grf   {:mode :grf128}                     ;; axis 8a — OPT-IN, default OFF (gate-consumed by feasible?)
    :stage {:space :none}                      ;; axis 7a — staging space, gate-consumed by feasible?
    ;; axis 10 — RESIDENCY/placement: which operands stay cache/on-chip resident (WARM) across use vs
    ;; are DRAM first-touch (COLD). The locality cost term (roofline warm/cold split) reads this via
    ;; schedule-cost-ns. Default all :dram = conservative (matches the flat-bytes behavior); a
    ;; fusion / cross-layer-resident schedule flips operands to :resident. The graph-level extension
    ;; for stationary LLMs (operand resident across the recorded decode sequence) lives here.
    :residency {:a :dram :b :dram :c :dram}
    :meta  {:target (:device-id desc)
            :machine-params (machine-params desc)
            :derived-by :raster.gpu.schedule/derive-default
            :overrides #{}
            :autotuned-at nil}}))

;; ================================================================
;; Stage 2 — resolve (data override deep-merged onto the default)
;; ================================================================

(defn- deep-merge
  [a b]
  (cond
    (and (map? a) (map? b)) (merge-with deep-merge a b)
    (some? b) b
    :else a))

(def ^:private valid-precisions #{:f16-xmx :f32-scalar})
(def ^:private valid-stage-spaces #{:none :slm :l3 :register})
(def ^:private valid-grf-modes #{:grf128 :grf256})

(defn resolve
  "Stage 2: deep-merge a user `override` schedule onto the derived default, recording the pinned
   top-level keys in :meta :overrides. `:gemm-precision` is deprecated sugar for `:precision`.
   A user `:meta` is NOT allowed to clobber the derived machine-params (stripped before merge).
   Throws on a `{:gemm-precision X :precision Y}` conflict rather than silently letting the
   deprecated key win."
  [derived override]
  (if (nil? override)
    derived
    (let [gp (:gemm-precision override)
          _ (when (and gp (:precision override) (not= gp (:precision override)))
              (throw (ex-info (str "schedule/resolve: conflicting :gemm-precision " gp
                                   " and :precision " (:precision override)
                                   " — pass one (prefer :precision; :gemm-precision is deprecated sugar)")
                              {:gemm-precision gp :precision (:precision override)})))
          override  (cond-> (dissoc override :meta)   ;; user :meta never clobbers machine-params
                      gp (-> (dissoc :gemm-precision) (assoc :precision gp)))
          pinned    (set (keys override))
          merged    (deep-merge derived override)]
      (update-in merged [:meta :overrides] (fnil into #{}) pinned))))

;; ================================================================
;; Stage 3 — the register-budget feasibility gate (axis 8b)
;; ================================================================

;; The byte model charges KERNEL-TILE properties against the per-lane GRF budget — NOT fractions
;; of the budget (an earlier draft did, which coupled the accumulator to the budget and made the
;; gate unable to detect an accumulator that spills on a smaller budget). The accumulator and the
;; staged-operand panel are fixed footprints of the emitted XMX tile; the budget is the device's.
(def ^:private xmx-f16-acc-bytes-per-lane
  "Fallback per-lane accumulator footprint of the f16 XMX tile when the schedule carries no explicit
   :tile: 8 float8 = 8×8×4 = 256 B/lane (the Arc default). A CONSTANT of the kernel tile (independent
   of the device's GRF budget). With an explicit :tile the gate uses `tile-acc-bytes-per-lane`
   instead — the ACTUAL emitted tile, so a bigger tile on a smaller-GRF device is caught."
  256)

(defn- tile-acc-bytes-per-lane
  "Per-lane f16 accumulator footprint of an explicit GEMM :tile — sg-m·sg-n f32 accumulators spread
   across the subgroup = (sg-m·sg-n/subgroup)·4 B/lane. This is what the emitter actually allocates
   in registers, so the gate charges the real tile (not the 256 constant). nil if no tile."
  [schedule]
  (when-let [{:keys [sg-m sg-n matrix]} (:tile schedule)]
    (let [sg (long (:subgroup matrix 16))]
      (quot (* (long sg-m) (long sg-n) 4) sg))))

(def ^:private operand-panel-bytes-per-lane
  "Per-lane footprint of ONE register-staged operand panel (a quarter of the f16 accumulator tile).
   Register double-buffering both operands is 4 panels = 256 B/lane — on top of the 256 B/lane
   accumulator that is the measured grf128 spill."
  64)

(defn- grf-budget-bytes-per-lane
  "The per-lane GRF budget for the schedule's GRF mode. grf256 doubles the register file — so it
   genuinely buys headroom the accumulator+staging can use (the accumulator does NOT scale with it)."
  [schedule desc]
  (let [base (long (:grf-bytes-per-lane desc 256))]
    (if (= :grf256 (get-in schedule [:grf :mode]))
      (* 2 base)
      base)))

(defn- acc-bytes-per-lane
  "Accumulator footprint per lane by precision. Prefers the ACTUAL emitted :tile (T2/T3 —
   tile-acc-bytes-per-lane); falls back to the Arc-default constant when no tile is pinned. f16-xmx
   uses the full XMX accumulator tile; :f32-scalar a quarter of it (no wide MMA accumulator)."
  [schedule]
  (let [full (or (tile-acc-bytes-per-lane schedule) xmx-f16-acc-bytes-per-lane)]
    (case (:precision schedule)
      :f16-xmx    full
      :f32-scalar (quot full 4)
      full)))

(defn- register-staged-bytes-per-lane
  "GRF charge of the staging schedule. ONLY :space :register consumes the register file; :slm /
   :l3 / :none stage into SLM/L3 and cost nothing against the GRF budget (the measured lesson: the
   winning staging space is L3/SLM, never register). Charge = Σcopies × depth × operand-panel —
   `:depth` (the staging ring depth) is honored, so deeper staging costs proportionally more."
  [schedule]
  (let [{:keys [space copies depth]} (:stage schedule)]
    (if (= space :register)
      (let [n-copies (+ (long (get copies :a 0)) (long (get copies :b 0)))]
        (* n-copies (long (or depth 1)) operand-panel-bytes-per-lane))
      0)))

(defn feasible?
  "The register-budget feasibility gate: (Schedule × HardwareDescriptor) → true | (throw).
   Called after `resolve`, before emission, so the emitter never sees an infeasible schedule.
   Requires `acc + register-staged ≤ GRF-budget` per lane. Fail-loud (raster-native); a
   `minimize()` fallback is deferred until an autotuner generates candidates.

   Anchors (grf128, 256 B/lane): the default f16-xmx GEMM with :stage :none is 256 ≤ 256 → OK;
   register double-buffering both operands is 256 + 4×64 = 512 > 256 → REJECTED at compile time
   (precisely the measurement-time spill, turned into a loud rejection)."
  [schedule desc]
  ;; validate the resolved schedule BEFORE pricing it — an unmodeled precision / stage-space / grf
  ;; mode (e.g. a :bf16 typo, or `:regsiter`) must FAIL LOUD here, not slip past into a silent XMX
  ;; bind (the #{:f16-xmx :f32-scalar} kwarg check upstream only sees the sugar, not a :schedule).
  (let [prec  (:precision schedule)
        space (get-in schedule [:stage :space] :none)
        grf   (get-in schedule [:grf :mode] :grf128)]
    (when-not (valid-precisions prec)
      (throw (ex-info (str "schedule: unknown :precision " (pr-str prec) " — expected " valid-precisions)
                      {:precision prec})))
    (when-not (valid-stage-spaces space)
      (throw (ex-info (str "schedule: unknown :stage :space " (pr-str space) " — expected " valid-stage-spaces)
                      {:space space})))
    (when-not (valid-grf-modes grf)
      (throw (ex-info (str "schedule: unknown :grf :mode " (pr-str grf) " — expected " valid-grf-modes)
                      {:grf grf}))))
  (let [budget (grf-budget-bytes-per-lane schedule desc)
        acc    (acc-bytes-per-lane schedule)
        staged (register-staged-bytes-per-lane schedule)
        req    (+ acc staged)]
    (when (> req budget)
      (throw (ex-info (str "schedule/feasible?: register budget exceeded — " req
                           " B/lane required > " budget " B/lane available"
                           " (grf mode " (get-in schedule [:grf :mode] :grf128) ", precision "
                           (:precision schedule) ", stage " (:stage schedule) ")."
                           " Move staging to :slm/:l3, drop copies/depth, or opt into :grf256.")
                      {:required-bytes-per-lane req
                       :budget-bytes-per-lane budget
                       :acc-bytes-per-lane acc
                       :register-staged-bytes-per-lane staged
                       :schedule schedule})))
    true))

;; ================================================================
;; Analytic cost — the LOCALITY roofline consumes the :residency axis
;; ================================================================

(defn schedule-cost-ns
  "Predicted device time (ns) for a GEMM `shape` {:m :n :k :dtype} under `schedule` on `desc`, via
   the locality roofline. Splits operand bytes into WARM (schedule :residency :resident → cache/
   on-chip) and COLD (:dram → first-touch), so a schedule that keeps operands resident (fusion,
   cross-layer residency) is priced faster — the capability the flat roofline lacked. This is the
   function the cost-guided chooser / autotune seed rank schedules with. Returns nil when the
   descriptor carries no bandwidth/peak-flops (the roofline abstains rather than fabricating)."
  [schedule {:keys [m n k dtype] :or {dtype :f16}} desc]
  (let [res      (:residency schedule)
        elem-b   (case dtype (:f16 :half) 2 (:f32 :float) 4 (:f64 :double) 8 2)
        ;; row-major operand footprints: A[m×k] B[k×n] C[m×n]
        ob       {:a (* (long m) (long k) elem-b) :b (* (long k) (long n) elem-b) :c (* (long m) (long n) elem-b)}
        warm?    (fn [o] (= :resident (get res o :dram)))
        cold     (reduce + 0 (for [o [:a :b :c] :when (not (warm? o))] (ob o)))
        warm     (reduce + 0 (for [o [:a :b :c] :when (warm? o)] (ob o)))]
    (hw/roofline-time-ns desc {:flops (* 2 (long m) (long n) (long k))
                               :cold-bytes cold :warm-bytes warm :dtype dtype})))

;; ================================================================
;; Convenience — the whole stage-1→2→3 pipeline
;; ================================================================

(defn schedule-for
  "Derive → resolve → gate in one call. Returns the feasible resolved Schedule (or throws at
   the gate). `override` may be nil. `policy` carries derive-default opts (e.g. {:precision …})."
  ([steps desc] (schedule-for steps desc nil nil))
  ([steps desc override] (schedule-for steps desc override nil))
  ([steps desc override policy]
   (let [sched (resolve (derive-default steps desc (or policy {})) override)]
     (feasible? sched desc)
     sched)))

(defn schedule-for-device
  "schedule-for against the descriptor PROBED for `device-id` — the live-device convenience the
   compiler uses. (Target-aware compilation for an ABSENT target passes a supplied descriptor to
   `schedule-for` instead.)"
  [steps device-id override policy]
  (schedule-for steps (hw/descriptor-for device-id) override policy))

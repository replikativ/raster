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
    :grf   {:mode :grf128}                     ;; axis 8a — OPT-IN, default OFF
    :stage {:space :none}                      ;; axis 7a — DEAD family, default :none
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

(defn resolve
  "Stage 2: deep-merge a user `override` schedule onto the derived default, recording the pinned
   top-level keys in :meta :overrides. `:gemm-precision` is deprecated sugar: an override of
   `{:gemm-precision p}` writes `{:precision p}`. Returns the resolved Schedule."
  [derived override]
  (if (nil? override)
    derived
    (let [override  (if-let [p (:gemm-precision override)]
                      (-> override (dissoc :gemm-precision) (assoc :precision p))
                      override)
          pinned    (set (remove #{:meta} (keys override)))
          merged    (deep-merge derived override)]
      (update-in merged [:meta :overrides] (fnil into #{}) pinned))))

;; ================================================================
;; Stage 3 — the register-budget feasibility gate (axis 8b)
;; ================================================================

(defn- grf-budget-bytes-per-lane
  "The per-lane GRF budget for the schedule's GRF mode. grf256 doubles the file."
  [schedule desc]
  (let [base (long (:grf-bytes-per-lane desc 256))]
    (if (= :grf256 (get-in schedule [:grf :mode]))
      (* 2 base)
      base)))

(defn- acc-bytes-per-lane
  "Accumulator footprint per lane by precision. The f16-xmx accumulator (8 float8 = 256 B/lane
   at grf128) FILLS the register file — the design's core finding, and why ANY register staging
   on top of it spills. :f32-scalar uses a quarter of the file."
  [schedule budget]
  (case (:precision schedule)
    :f16-xmx    budget
    :f32-scalar (quot budget 4)
    budget))

(defn- register-staged-bytes-per-lane
  "GRF charge of the staging schedule. ONLY :space :register consumes the register file; :slm /
   :l3 / :none stage into SLM/L3 and cost nothing against the GRF budget (the design's measured
   lesson: the winning staging space is L3/SLM, never register). Each register-staged operand
   copy is a quarter-file operand panel."
  [schedule budget]
  (let [{:keys [space copies]} (:stage schedule)]
    (if (= space :register)
      (let [n-copies (+ (long (get copies :a 0)) (long (get copies :b 0)))]
        (* n-copies (quot budget 4)))
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
  (let [budget (grf-budget-bytes-per-lane schedule desc)
        acc    (acc-bytes-per-lane schedule budget)
        staged (register-staged-bytes-per-lane schedule budget)
        req    (+ acc staged)]
    (when (> req budget)
      (throw (ex-info (str "schedule/feasible?: register budget exceeded — " req
                           " B/lane required > " budget " B/lane available"
                           " (grf mode " (get-in schedule [:grf :mode] :grf128) ", precision "
                           (:precision schedule) ", stage " (:stage schedule) ")."
                           " Move staging to :slm/:l3, drop copies, or (if deep-K) opt into :grf256.")
                      {:required-bytes-per-lane req
                       :budget-bytes-per-lane budget
                       :acc-bytes-per-lane acc
                       :register-staged-bytes-per-lane staged
                       :schedule schedule})))
    true))

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

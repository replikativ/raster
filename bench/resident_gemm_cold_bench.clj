(ns resident-gemm-cold-bench
  "Cadence-controlled COLD benchmark for the resident XMX GEMM (Level-Zero / Intel Xe).

   This is the durable measurement METHODOLOGY salvaged from the GEMM-tiling episode —
   the harness that caught two 'wins' as pure measurement artifacts. It is the intended
   STANDARD for any future resident-GEMM perf claim. It deliberately carries NO tiling
   schedule knobs (see 'Dropped dead levers' below): it measures a kernel config honestly;
   it does not advocate one.

   ── The two confounds this harness controls ──────────────────────────────────────
   1. CADENCE / CLOCK CONFOUND. A GPU boost-clocks under sustained load and throttles
      when idle. A harness that records ONE launch, resets, reads its timestamp, then
      records the NEXT launch gives each kernel a different clock/thermal history — a
      faster-looking config may simply have been sampled while the clock was higher.
      FIX: one recorded graph = `launches` (default 24) back-to-back GEMM launches
      (barriers serialize them). The GPU stays boosted across the whole sequence, so
      every launch runs under the SAME drifting clock — only cache temperature differs.
      And when comparing configs, all configs are replayed ROUND-ROBIN each round, so
      each is sampled ~ms apart under the same clock/thermal state (kills the
      sequential-config confound where config A is timed cold and config B timed warm).

   2. WARM-OPERAND CONFOUND. Reusing ONE resident A/B pair across replays leaves the
      operands L3-resident, so the benchmark measures an all-cache-hit steady state that
      never occurs in training (where each step's activations are first-touch DRAM
      reads). A schedule that only hides COLD DRAM-load latency then looks worthless warm
      and a genuinely-cold win looks like noise.
      FIX: COLD = each of the `launches` launches reads a DISTINCT buffer pair
      (total footprint >> L3 -> every read is a first-touch DRAM read = a realistic
      per-step activation). WARM = one shared pair (L3-resident) kept only as a control
      to expose the gap. Report the COLD number for any training-relevant claim.

   Reproducibility on the reference machine (Arc 140V, Xe2/Lunar Lake): ±2%.

   ── Dropped dead levers ───────────────────────────────────────────────────────────
   The originating episode swept {grf128,grf256}×{prefetch-b?,k-double-buffer?}. Under
   this cold harness NONE of them beat the plain baseline on the shapes they were
   proposed to gate — they were warm-harness artifacts. Per the repo's no-dead-flags
   policy they are NOT resurrected here: prefetch-b?/k-double-buffer? do not exist in the
   emitter on main, and this harness's default kernel is the plain baseline.

   The one surviving knob is `:large-grf?` (the -ze-opt-large-register-file BUILD flag),
   kept ONLY as an opt-in EXPERIMENTAL comparand so a NEW schedule idea can be A/B'd
   through the same rigorous method. It is OFF by default and auto-gated NOWHERE.

   ── Use ──────────────────────────────────────────────────────────────────────────
     (require 'resident-gemm-cold-bench :reload)
     (resident-gemm-cold-bench/cold-bench {:name \"proj\" :m 1024 :k 640 :n 2048})
     ;; A/B an experimental comparand against the baseline (both under the cold method):
     (resident-gemm-cold-bench/compare-configs
       {:name \"proj\" :m 1024 :k 640 :n 2048}
       [{:label \"baseline\"}
        {:label \"large-grf\" :large-grf? true}])"
  (:require [raster.gpu.ze-runtime :as ze]
            [raster.compiler.backend.gpu.opencl-codegen :as cg]
            [raster.compiler.support.spirv-cache :as spv]
            [raster.runtime.hardware :as hw]))

(def ^:private device-hex
  (delay (get-in (hw/device :ze:0) [:capabilities :device-id-hex])))

(def ^:private I32 java.lang.foreign.ValueLayout/JAVA_INT)

(defn- fa ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (* 0.5 (.nextGaussian r))))) a))

(defn- median [xs]
  (let [v (vec (sort xs)) n (count v)]
    (if (odd? n) (nth v (quot n 2))
        (/ (+ (nth v (dec (quot n 2))) (nth v (quot n 2))) 2.0))))

(defn- pctl [xs p]
  (let [v (vec (sort xs)) n (count v)]
    (nth v (min (dec n) (int (* p n))))))

(defn- gflops [ms m n k] (/ (* 2.0 m k n) (* ms 1.0e6)))

;; ── kernel build/cache ───────────────────────────────────────────────────────────
;; A config is a map. The only recognized key is the EXPERIMENTAL, opt-in, default-off
;; :large-grf? (the -ze-opt-large-register-file build flag). Everything else is the
;; plain baseline resident GEMM (:c-dtype :half — matches bind-registered-gemm!'s
;; default; f16 C write is half the DRAM traffic of f32, reproducing the resident path).
(def ^:private kern-cache (atom {}))
(defn- tile-tag [{:keys [block-m block-n block-k]}]
  (str "b" block-m "x" block-n "k" block-k))
(defn build-kernel!
  "Compile+cache the resident GEMM kernel for `config`.
   Returns {:module :kname :tile}. config keys:
     :tile        a derive-gemm-tile map {:block-m :block-n :sg-m :sg-n :block-k :matrix}
                  → builds the TILE-PARAMETRIC emit-gemm-tiled kernel (B1 tile comparison).
                  Absent → the plain hand emit-gemm-nonsquare-kernel baseline (block 128).
     :large-grf?  EXPERIMENTAL build-flag comparand (default false, gated nowhere)."
  [{:keys [large-grf? tile] :as _config}]
  (let [ck [large-grf? tile]]
    (or (get @kern-cache ck)
        (let [kname (str "gemm_cold_" (if tile (tile-tag tile) (if large-grf? "grf256" "grf128")))
              src   (if tile
                      (apply cg/emit-gemm-tiled kname :c-dtype :half :prefetch (:num-stages tile 3)
                             (mapcat identity (select-keys tile [:block-m :block-n :sg-m :sg-n :block-k :matrix])))
                      ;; no tile → emit-gemm-tiled at its DEFAULT geometry (block 128, wg 256 —
                      ;; matches the historical baseline; the old hand emit-gemm-nonsquare-kernel
                      ;; was removed when the tile-parametric generator replaced it in T1).
                      (cg/emit-gemm-tiled kname :c-dtype :half))
              spirv (spv/compile-opencl-to-spirv src :device @device-hex)
              _ (when large-grf?
                  (throw (ex-info ":large-grf? unsupported — the -ze-opt-large-register-file build-flag path was removed from load-module!; re-add flag plumbing to compile/load to use it" {})))
              module (ze/load-module! spirv :spirv)
              entry {:module module :kname kname :tile tile}]
          (swap! kern-cache assoc ck entry)
          entry))))

;; ── resident buffer plumbing ───────────────────────────────────────────────────────
;; `tile` (or nil for the plain block-128 baseline) drives the workgroup size and grid,
;; mirroring bind-registered-gemm-tiled!: wg = n-subgroups × subgroup, grid = ceil(n/bn) ×
;; ceil(m/bm). A wrong wg/grid for the tile geometry = garbage output, so this MUST match
;; the tile the kernel was emitted for.
(defn- bind-gemm! [module kname a16 b16 c m n k tile]
  (let [kh (ze/create-kernel-fresh module kname)
        m (long m) n (long n) k (long k)
        {:keys [block-m block-n sg-m sg-n matrix] :or {block-m 128 block-n 128 sg-m 32 sg-n 32}} (or tile {})
        sg (long (:subgroup matrix 16))
        wg (if tile (* (quot (long block-m) (long sg-m)) (quot (long block-n) (long sg-n)) sg) 256)
        args [(:segment a16) (:segment b16) (:segment c)
              {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]
        bnd (ze/bind-kernel-2d! kh [wg 1] args)
        gc ^java.lang.foreign.MemorySegment (:gc-seg bnd)]
    (.set gc I32 0 (int (Math/ceil (/ (double n) (double block-n)))))
    (.set gc I32 4 (int (Math/ceil (/ (double m) (double block-m)))))
    (.set gc I32 8 (int 1))
    bnd))

;; Cached random host arrays (values are irrelevant to timing; DISTINCT DEVICE buffers
;; are what makes cold reads cold). Avoids regenerating millions of gaussians per pair.
(def ^:private src-cache (atom {}))
(defn- src ^floats [n] (or (get @src-cache n)
                           (let [a (fa n (+ 9 n))] (swap! src-cache assoc n a) a)))

(defn- alloc-pair [m n k]
  (let [a16 (ze/buffer-of-floats-as-half (src (* m k)))
        b16 (ze/buffer-of-floats-as-half (src (* k n)))
        c   (ze/make-buffer (* m n) :half)]
    {:a a16 :b b16 :c c}))

(defn- free-pair [{:keys [a b c]}] (doseq [x [a b c]] (ze/free-buffer! x)))

(defn- record-seq-graph
  "Record ONE graph of `launches` back-to-back GEMM launches (barriers serialize them).
   `pair-of` maps launch index -> a buffer pair. Identical launch cadence for warm/cold;
   the GPU stays boosted across the whole sequence, so only cache temperature differs."
  [module kname m n k launches pair-of tile]
  (ze/record-graph!
   (mapv (fn [i]
           (let [p (pair-of i)]
             {:bound (bind-gemm! module kname (:a p) (:b p) (:c p) m n k tile)
              :kernel-name kname}))
         (range launches))
   {:profile? true}))

;; ── the interleaved cold/warm measurement (THE method) ─────────────────────────────
(defn compare-configs
  "Cadence-controlled, interleaved COLD-vs-WARM comparison of `configs` on one GEMM shape.

   For each config, builds a COLD graph (`launches` distinct buffer pairs, every read a
   first-touch DRAM read) and a WARM control graph (one shared L3-resident pair) over
   SHARED buffers, then replays ALL graphs ROUND-ROBIN each round so every config is
   sampled under the same drifting clock/thermal state. Reports COLD and WARM median
   TFLOPS relative to the first config (the baseline). The COLD column is the honest,
   training-relevant number; WARM is the confound control.

   shape:   {:name :m :k :n}
   configs: vector of config maps (see build-kernel!); FIRST is the baseline.
            default [{:label \"baseline\"}]
   opts:    :launches (24) :reps (10) :warmup (6)"
  ([shape] (compare-configs shape [{:label "baseline"}]))
  ([{:keys [m n k name]} configs & {:keys [launches reps warmup]
                                    :or {launches 24 reps 10 warmup 6}}]
   (let [pairs   (mapv (fn [_] (alloc-pair m n k)) (range launches)) ;; shared cold pairs
         wpair   (alloc-pair m n k)                                  ;; shared warm pair
         entries (vec (map-indexed
                       (fn [i c]
                         (let [{:keys [module kname tile]} (build-kernel! c)
                               label (or (:label c) (str "cfg" i))]
                           {:label label
                            :cold (record-seq-graph module kname m n k launches (fn [j] (nth pairs j)) tile)
                            :warm (record-seq-graph module kname m n k launches (constantly wpair) tile)}))
                       configs))
         acc (atom (into {} (for [e entries] [(:label e) {:cold [] :warm []}])))]
     ;; interleaved warmup — every graph, several rounds, for thermal/clock steady state
     (dotimes [_ warmup]
       (doseq [e entries]
         (ze/replay-graph! (:cold e)) (ze/reset-graph-events! (:cold e))
         (ze/replay-graph! (:warm e)) (ze/reset-graph-events! (:warm e))))
     ;; interleaved timed rounds
     (dotimes [_ reps]
       (doseq [e entries]
         (ze/replay-graph! (:cold e))
         (let [ks (:kernels (ze/read-graph-timestamps! (:cold e)))]
           (swap! acc update-in [(:label e) :cold] into (map :ms ks)))
         (ze/replay-graph! (:warm e))
         (let [ks (:kernels (ze/read-graph-timestamps! (:warm e)))]
           ;; skip launch 0-1 of warm (cold first-touch of the shared pair)
           (swap! acc update-in [(:label e) :warm] into (map :ms (drop 2 ks))))))
     (doseq [e entries] (ze/destroy-graph! (:cold e)) (ze/destroy-graph! (:warm e)))
     (doseq [p pairs] (free-pair p)) (free-pair wpair)
     ;; report
     (println (format "\n===== COLD BENCH  SHAPE %s (M%d K%d N%d) reps=%d launches=%d ====="
                      name m k n reps launches))
     (println (format "%-16s | %8s %9s %9s | %8s %9s %9s | %8s"
                      "config" "cold-TF" "cMed-ms" "cP75-ms" "warm-TF" "wMed-ms" "wP25-ms" "vs-base"))
     (let [rows (for [e entries
                      :let [cs (get-in @acc [(:label e) :cold])
                            ws (get-in @acc [(:label e) :warm])
                            cmed (median cs) wmed (median ws)]]
                  {:label (:label e)
                   :cold-tf (/ (gflops cmed m n k) 1000.0) :cold-med cmed :cold-p75 (pctl cs 0.75)
                   :warm-tf (/ (gflops wmed m n k) 1000.0) :warm-med wmed :warm-p25 (pctl ws 0.25)
                   :n (count cs)})
           base-tf (:cold-tf (first rows))]
       (doseq [r rows]
         (println (format "%-16s | %8.2f %9.4f %9.4f | %8.2f %9.4f %9.4f | %+7.1f%%"
                          (:label r) (:cold-tf r) (:cold-med r) (:cold-p75 r)
                          (:warm-tf r) (:warm-med r) (:warm-p25 r)
                          (* 100.0 (- (/ (:cold-tf r) base-tf) 1.0)))))
       (vec rows)))))

(defn cold-bench
  "Cold benchmark of the BASELINE resident GEMM on one shape. Convenience over
   compare-configs with a single baseline config. Returns the baseline row map
   {:label :cold-tf :cold-med :cold-p75 :warm-tf :warm-med :warm-p25 :n}."
  [shape & {:as opts}]
  (first (apply compare-configs shape [{:label "baseline"}] (apply concat opts))))

;; Reference shapes from the gemma FFN (the GEMM-tiling episode's targets). Not a claim —
;; just a convenient standard sweep to exercise the harness.
(def reference-shapes
  [{:name "proj      " :m 1024 :k 640  :n 2048}
   {:name "gate      " :m 1024 :k 640  :n 1024}
   {:name "down-proj " :m 1024 :k 2048 :n 640}])

(defn run-reference
  "Run the baseline cold bench across reference-shapes."
  []
  (mapv cold-bench reference-shapes))

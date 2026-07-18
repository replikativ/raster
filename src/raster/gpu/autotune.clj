(ns raster.gpu.autotune
  "Phase 2: measurement-driven schedule autotuning — the feedback loop the analytic model seeds and
   measurement closes. Coordinate-descent (Inductor's algorithm) from the derive-default schedule,
   priced by a kernel benchmark, cached to disk keyed by (op × shape-class × descriptor-signature ×
   version). The analytic model PROPOSES the seed; measurement DISPOSES the winner; the winner is
   cached so tuning is a one-time cost per (op,shape,machine).

   The `cost-fn` is supplied by the caller: on the resident GPU path it compiles+binds+runs the
   kernel under a candidate schedule and times it with the device-event profiler (do_bench); an
   infeasible candidate returns +Inf and self-prunes. This ns owns the SEARCH + CACHE; the kernel
   bench is the integration seam (lands with the resident/Compiled path)."
  (:require [raster.gpu.schedule :as sched]
            [raster.compiler.core.hardware :as hw]
            [clojure.java.io :as io])
  (:import [java.io File]))

(def autotune-version 1)

;; ================================================================
;; Coordinate descent (Inductor) — greedy, one neighbourhood step at a time
;; ================================================================

(defn coordinate-descent
  "Greedy local search from `seed` (a config map). `neighbors` : {field → (current-value → [candidate
   values])} — the ×2/÷2 (block/split-k) and two-valued (precision) moves. `cost-fn` : config → number
   (lower is better; Double/POSITIVE_INFINITY for an infeasible candidate → self-prunes). A move is
   accepted only if it improves by more than `threshold` (0.1% — rejects measurement noise, Inductor's
   floor). Memoized so each config is priced once. Returns {:config :cost :evals}."
  [seed neighbors cost-fn & {:keys [threshold max-iters] :or {threshold 0.001 max-iters 100}}]
  (let [pget   (fn [m k] (if (vector? k) (get-in m k) (get m k)))
        passoc (fn [m k v] (if (vector? k) (assoc-in m k v) (assoc m k v)))
        cache (atom {})
        cost  (fn [c] (if (contains? @cache c) (@cache c)
                          (let [v (double (cost-fn c))] (swap! cache assoc c v) v)))]
    (loop [config seed best (cost seed) i 0]
      (let [moves (for [[field vals-fn] neighbors
                        v (vals-fn (pget config field))
                        :let [c (passoc config field v)]
                        :when (not= c config)]
                    [c (cost c)])
            [bc bcost] (when (seq moves) (apply min-key second moves))]
        (if (and bc (< i max-iters) (< bcost (* best (- 1.0 threshold))))
          (recur bc bcost (inc i))
          {:config config :cost best :evals (count @cache)})))))

;; ================================================================
;; The default schedule neighbourhood
;; ================================================================

(defn schedule-neighbors
  "The moves coordinate-descent explores over an S6 Schedule. Two WIRED axes:
     :precision — selects the XMX-f16 vs scalar GEMM binder.
     :tile      — the GEMM tile (T2/T3), a CURATED per-descriptor candidate list
                  (hw/gemm-tile-candidates) that drives emit-gemm-tiled +
                  bind-registered-gemm-tiled! (real, priced kernels).
   :grf / :stage stay schema-only until the emitter consumes them (the S6 review found nothing
   reads them at emission) — searching an un-wired knob wastes measurements on an identical kernel.

   NB split-k factor (~10× on deep-k occupancy-bound shapes, gemm_autotune_test) is a separate
   GEMM-level occupancy axis measured directly there; promoting it to an S6 field is the next step."
  [desc]
  {:precision (fn [_] [:f16-xmx :f32-scalar])
   :tile      (fn [_] (hw/gemm-tile-candidates desc))})

;; ================================================================
;; Disk cache — keyed by (op × shape-class × descriptor-signature × version)
;; ================================================================

(defn descriptor-signature
  "Perf-relevant identity of a descriptor: the fields that change the optimal schedule (INCLUDING
   the measured bandwidth/flops — a re-calibrated machine gets a fresh autotune, XLA's discipline)."
  [descriptor]
  (select-keys descriptor [:device-id :vendor :arch :machine-lanes :grf-bytes-per-lane
                           :bandwidth-bytes-s :peak-flops :subgroup-size]))

(defn cache-key [op-key shape-class descriptor]
  (pr-str [op-key shape-class (descriptor-signature descriptor) autotune-version]))

(defn- cache-file ^File [key-str]
  (io/file (System/getProperty "user.home") ".raster" "autotune"
           (str (format "%08x" (hash key-str)) ".edn")))

(defn cache-get [key-str]
  (let [f (cache-file key-str)]
    (when (.exists f)
      (try (let [{:keys [key config]} (read-string (slurp f))]
             (when (= key key-str) config))
           (catch Exception _ nil)))))

(defn cache-put! [key-str config]
  (let [f (cache-file key-str)]
    (io/make-parents f)
    (let [tmp (File/createTempFile "atune" ".edn" (.getParentFile f))]
      (spit tmp (pr-str {:key key-str :config config}))
      (.renameTo tmp f))
    config))

;; ================================================================
;; The autotune entry — cache-or-search
;; ================================================================

(defn autotune-schedule
  "The best Schedule for (op-key × shape-class × descriptor): from the disk cache, or found by
   coordinate-descent from the derive-default seed priced by `cost-fn` and then cached. `cost-fn`
   takes a candidate Schedule and returns its measured cost (lower better; +Inf = infeasible).
   opts: :neighbors (override), :force? (ignore cache), :steps (program steps for derive-default)."
  [op-key shape-class descriptor cost-fn & {:keys [neighbors force? steps]}]
  (let [k (cache-key op-key shape-class descriptor)]
    (or (when-not force? (cache-get k))
        (let [seed (sched/derive-default steps descriptor)
              result (coordinate-descent seed (or neighbors (schedule-neighbors descriptor)) cost-fn)]
          (cache-put! k (:config result))
          (:config result)))))

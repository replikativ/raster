(ns raster.abm-bench
  "FIRMS ABM benchmark: CPU sequential, CPU parallel, and GPU.

  Run with Valhalla JDK:
    source valhalla-env.sh
    OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \\
      -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \\
      -J--enable-native-access=ALL-UNNAMED \\
      -M:bench:valhalla -m raster.abm-bench

  Options:
    --all             Run all scales (1K to 10M)
    --gpu             Include GPU benchmarks (requires Level Zero)
    --cpu-only        Skip GPU, run CPU sequential + parallel only
    <N>               Benchmark at N agents (e.g. 1000000)

  Results written to bench/results/abm_firms_<date>.txt

  C++11 reference (separate process, same machine):
    FIRMS_N=10000000 bash bench/comparison/firms_c11.sh

  JAX / Julia / etc. do not have comparable ABM frameworks — the C++11
  FIRMS implementation is the canonical reference."
  (:require [raster.abm.firms :as firms]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.time LocalDateTime]
           [java.time.format DateTimeFormatter]))

;; ================================================================
;; Bench harness
;; ================================================================

(defn- nanos->ms [^long nanos] (/ (double nanos) 1e6))

(defn- bench-period
  "Time one period execution. Returns elapsed-ms."
  [thunk]
  (let [t0 (System/nanoTime)
        _  (thunk)
        t1 (System/nanoTime)]
    (nanos->ms (- t1 t0))))

(defn- summarize [times-ms]
  (let [sorted (sort times-ms)
        n      (count sorted)]
    {:best   (first sorted)
     :median (nth sorted (quot n 2))
     :p90    (nth sorted (int (* n 0.9)))
     :mean   (/ (reduce + times-ms) n)
     :times  (vec sorted)}))

;; ================================================================
;; CPU sequential benchmark
;; ================================================================

(defn bench-cpu-seq
  "Benchmark CPU sequential ABM for n-agents.
   Returns {:n :best :median :p90 :mean :times}."
  [n-agents & {:keys [warmup timed] :or {warmup 2 timed 5}}]
  (println (format "  CPU-seq  n=%,12d  initializing..." n-agents))
  (flush)
  (let [config (firms/default-config (long n-agents))
        [config agents firms rng] (firms/init-simulation config)]
    ;; warmup
    (dotimes [t warmup]
      (firms/run-period! config agents firms rng t))
    (System/gc)
    (Thread/sleep 500)
    ;; measure
    (let [times (mapv (fn [i]
                        (bench-period
                         #(firms/run-period! config agents firms rng (+ warmup i))))
                      (range timed))
          result (assoc (summarize times) :n n-agents :variant :cpu-seq)]
      (println (format "           median=%8.1f ms  best=%8.1f ms  p90=%8.1f ms"
                       (:median result) (:best result) (:p90 result)))
      result)))

;; ================================================================
;; CPU parallel benchmark
;; ================================================================

(defn bench-cpu-par
  "Benchmark CPU parallel ABM for n-agents.
   Returns {:n :best :median :p90 :mean :times}."
  [n-agents & {:keys [warmup timed] :or {warmup 2 timed 5}}]
  (println (format "  CPU-par  n=%,12d  initializing..." n-agents))
  (flush)
  (let [config (firms/default-config (long n-agents))
        [config agents firms rng] (firms/init-simulation config)]
    (dotimes [t warmup]
      (firms/run-period-parallel! config agents firms rng t))
    (System/gc)
    (Thread/sleep 500)
    (let [times (mapv (fn [i]
                        (bench-period
                         #(firms/run-period-parallel! config agents firms rng (+ warmup i))))
                      (range timed))
          result (assoc (summarize times) :n n-agents :variant :cpu-par)]
      (println (format "           median=%8.1f ms  best=%8.1f ms  p90=%8.1f ms"
                       (:median result) (:best result) (:p90 result)))
      result)))

;; ================================================================
;; GPU benchmark (requires Level Zero)
;; ================================================================

(defn- gpu-available? []
  (try
    (require 'raster.gpu.ze-runtime)
    (let [qfn (resolve 'raster.gpu.ze-runtime/query-devices)]
      (and qfn (seq (qfn))))
    (catch Exception _ false)))

(defn bench-gpu
  "Benchmark GPU-fused ABM for n-agents.
   Requires Level Zero. Returns {:n :best :median :p90 :mean :times}."
  [n-agents & {:keys [device-id warmup timed]
               :or   {device-id :ze:0 warmup 3 timed 5}}]
  (require 'raster.abm.firms.gpu)
  (let [bench-fn (resolve 'raster.abm.firms.gpu/bench-gpu!)]
    (println (format "  GPU      n=%,12d  compiling + allocating..." n-agents))
    (flush)
    (let [raw (bench-fn n-agents {:device-id device-id :warmup warmup :timed timed})
          result (assoc (summarize (:times raw)) :n n-agents :variant :gpu)]
      (println (format "           median=%8.1f ms  best=%8.1f ms  p90=%8.1f ms"
                       (:median result) (:best result) (:p90 result)))
      result)))

;; ================================================================
;; Comparison table
;; ================================================================

(def ^:private c11-reference-ms
  "C++11 FIRMS reference times (ms/period) from bench/comparison/firms_c11.sh.
   Run `FIRMS_N=<n> bash bench/comparison/firms_c11.sh` to measure on your machine.
   Set to nil to omit from table."
  nil  ;; populate after running firms_c11.sh, e.g.:
  ;; {1000000 260.0, 10000000 3000.0}
  )

(defn- format-comparison-table
  "Format comparison table from benchmark results."
  [results]
  (let [header  (format "%-12s  %10s  %10s  %10s  %10s  %12s"
                        "n-agents" "CPU-seq" "CPU-par" "GPU" "C++11-ref" "GPU speedup")
        sep     (apply str (repeat 80 "-"))
        by-n    (group-by :n results)]
    (str header "\n" sep "\n"
         (str/join "\n"
                   (for [n (sort (keys by-n))]
                     (let [rs     (get by-n n)
                           cpu-s  (first (filter #(= :cpu-seq (:variant %)) rs))
                           cpu-p  (first (filter #(= :cpu-par (:variant %)) rs))
                           gpu-r  (first (filter #(= :gpu (:variant %)) rs))
                           c11    (get c11-reference-ms n)
                           fmt    (fn [r] (if r (format "%8.1f ms" (:median r)) "       n/a"))
                           c11-s  (if c11 (format "%8.1f ms" c11) "       n/a")
                           speedup (when (and gpu-r cpu-p)
                                     (format "%7.1fx"
                                             (/ (:median cpu-p) (:median gpu-r))))]
                       (format "%-12s  %10s  %10s  %10s  %10s  %12s"
                               (format "%,d" n) (fmt cpu-s) (fmt cpu-p) (fmt gpu-r)
                               c11-s (or speedup "n/a"))))))))

;; ================================================================
;; Output
;; ================================================================

(defn- results-path []
  (let [date (.format (LocalDateTime/now) (DateTimeFormatter/ofPattern "yyyy-MM-dd"))]
    (str "bench/results/abm_firms_" date ".txt")))

(defn- write-results!
  "Write formatted results to bench/results/."
  [table-str scales gpu?]
  (let [path (results-path)
        jdk  (System/getProperty "java.version")
        cpus (.. Runtime getRuntime availableProcessors)
        content (str "FIRMS ABM Benchmark\n"
                     (apply str (repeat 60 "=")) "\n"
                     (format "Date: %s\n" (LocalDateTime/now))
                     (format "JDK:  %s\n" jdk)
                     (format "CPUs: %d\n" cpus)
                     (format "GPU:  %s\n" (if gpu? "Level Zero (Intel)" "n/a"))
                     (format "Scales: %s\n\n" (str/join ", " (map #(format "%,d" %) scales)))
                     "All times: median ms/period\n\n"
                     table-str "\n\n"
                     "Notes:\n"
                     "  - CPU-seq: single-threaded sequential (run-period!)\n"
                     "  - CPU-par: parallel agent decisions (run-period-parallel!)\n"
                     "  - GPU: Level Zero fused pipeline (run-period-gpu-fused!)\n"
                     "  - C++11-ref: g++ -O3 -march=native (bench/comparison/firms_c11.sh)\n"
                     "  - GPU speedup = CPU-par median / GPU median\n")]
    (io/make-parents path)
    (spit path content)
    (println (format "\nResults written to %s" path))))

;; ================================================================
;; Runner
;; ================================================================

(defn run-benchmark
  "Run ABM benchmarks at given scales.
   Options:
     :gpu?    include GPU benchmarks (default: auto-detect)
     :scales  vector of [n-agents warmup timed] triples"
  [& {:keys [gpu? scales]
      :or   {scales [[1000     3 5]
                     [10000    3 5]
                     [100000   2 5]
                     [1000000  2 3]
                     [10000000 1 3]]}}]
  (let [gpu?    (if (some? gpu?) gpu? (gpu-available?))
        n-list  (mapv first scales)]
    (println "\n=== FIRMS ABM Benchmark ===")
    (println (format "JDK: %s  |  CPUs: %d  |  GPU: %s"
                     (System/getProperty "java.version")
                     (.. Runtime getRuntime availableProcessors)
                     (if gpu? "yes" "no")))
    (println)

    (let [results (vec
                   (mapcat
                    (fn [[n warmup timed]]
                      (println (format "\n--- n = %,d agents ---" n))
                      (let [cpu-s (bench-cpu-seq n :warmup warmup :timed timed)
                            cpu-p (bench-cpu-par n :warmup warmup :timed timed)
                            gpu-r (when gpu?
                                    (try (bench-gpu n :warmup warmup :timed timed)
                                         (catch Exception e
                                           (println (format "  GPU      FAILED: %s" (.getMessage e)))
                                           nil)))]
                        (cond-> [cpu-s cpu-p]
                          gpu-r (conj gpu-r))))
                    scales))
          table (format-comparison-table results)]
      (println "\n\n=== Summary ===")
      (println table)
      (write-results! table n-list gpu?)
      results)))

;; ================================================================
;; Script entry point
;; ================================================================

(defn -main [& args]
  (let [args (vec args)]
    (cond
      (some #{"--all"} args)
      (run-benchmark :gpu? (not (some #{"--cpu-only"} args)))

      (some #{"--cpu-only"} args)
      (run-benchmark :gpu? false)

      (some #{"--gpu"} args)
      (run-benchmark :gpu? true)

      (and (seq args) (re-matches #"\d+" (first args)))
      (let [n (Long/parseLong (first args))
            gpu? (not (some #{"--cpu-only"} args))]
        (run-benchmark :gpu? gpu? :scales [[n 2 5]]))

      :else
      (run-benchmark)))

  (shutdown-agents))

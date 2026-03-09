(ns raster.abm.firms-bench-test
  "Benchmark the Firms ABM at multiple scales.
   Measures ms/period with per-phase breakdown.
   Results written to /tmp/abm_bench.txt."
  (:require [raster.abm.firms :as firms]
            [raster.abm.firms.membership :as mem]
            [raster.random :as rnd]))

(defn- nanos->ms [^long nanos] (/ (double nanos) 1e6))

(defn bench-phase
  "Time a thunk, return [result elapsed-ms]."
  [thunk]
  (let [t0 (System/nanoTime)
        r  (thunk)
        t1 (System/nanoTime)]
    [r (nanos->ms (- t1 t0))]))

(defn bench-period!
  "Run one period with per-phase timing. Returns map of phase->ms."
  [^raster.abm.firms.FirmsConfig config
   ^raster.abm.firms.AgentSoA agents
   ^raster.abm.firms.FirmSoA firms
   ^java.util.SplittableRandom rng
   period]
  (let [n          (.n-agents config)
        n-friends  (.n-friends config)
        act-rate   (.activation-rate config)
        n-active   (int (Math/round (* act-rate n)))
        max-firms  (* n 2)
        ;; Extract arrays
        theta      (.theta agents)
        endowment  (.endowment agents)
        effort     (.effort agents)
        current-firm (.firm agents)
        friends    (.friends agents)
        s-cache    (.cache agents)
        param-a    (.a firms)
        param-b    (.b firms)
        param-beta (.beta firms)
        total-effort (.te firms)
        firm-size  (.n-workers firms)
        alive      (.alive firms)
        output     (.output firms)
        members    (.members firms)
        member-offsets (.offsets firms)
        ;; Generate per-agent seeds
        rng-seeds  (let [seeds (long-array n)]
                     (dotimes [i n]
                       (aset ^longs seeds i (.nextLong rng)))
                     seeds)
        ;; Allocate queue arrays
        q-agent-ids  (int-array (max 1 (int n-active)))
        q-types      (int-array (max 1 (int n-active)))
        q-targets    (int-array (max 1 (int n-active)))
        q-efforts    (float-array (max 1 (int n-active)))
        q-count      (int-array 1)
        ;; Phase 1: Agent decisions
        [_ t-decide]
        (bench-phase
         (fn []
           (dotimes [_ n-active]
             (let [agent-id (.nextInt rng 0 n)]
               (firms/agent-decide-impl!
                agent-id theta endowment effort current-firm friends
                s-cache
                param-a param-b param-beta total-effort output
                firm-size alive
                q-agent-ids q-types q-targets q-efforts q-count
                n-friends)))))
        ;; Phase 2: Execute decisions
        cnt (aget ^ints q-count 0)
        [_ t-execute]
        (bench-phase
         (fn []
           (mem/execute-decisions-impl!
            q-agent-ids q-types q-targets q-efforts
            cnt effort current-firm total-effort firm-size alive
            param-a param-b param-beta output
            max-firms rng-seeds)))
        ;; Phase 3: Rebuild CSR
        [_ t-rebuild]
        (bench-phase
         (fn []
           (mem/rebuild-csr! current-firm firm-size alive
                             members member-offsets n max-firms)))
        ;; Phase 4: Produce output
        [_ t-produce]
        (bench-phase
         (fn []
           (firms/produce-output-impl! param-a param-b param-beta total-effort
                                       output alive max-firms)))
        ;; Phase 5: Distribute income
        [_ t-distribute]
        (bench-phase
         (fn []
           (firms/distribute-income-impl! (.income agents) output firm-size alive
                                          members member-offsets max-firms)))]
    {:decide    t-decide
     :execute   t-execute
     :rebuild   t-rebuild
     :produce   t-produce
     :distribute t-distribute
     :total     (+ t-decide t-execute t-rebuild t-produce t-distribute)}))

(defn run-bench
  "Benchmark at given scale. Returns vector of per-period timing maps."
  [n-agents n-periods n-warmup n-measured]
  (let [config (firms/->FirmsConfig n-agents 4 4 0.04 0
                                    (+ n-warmup n-measured) 42 n-agents)
        [config agents firms rng] (firms/init-simulation config)]
    ;; Warmup
    (dotimes [t n-warmup]
      (firms/run-period! config agents firms rng t))
    ;; Measured
    (vec
     (for [t (range n-warmup (+ n-warmup n-measured))]
       (bench-period! config agents firms rng t)))))

(defn median [xs]
  (let [sorted (sort xs)
        n (count sorted)]
    (nth sorted (quot n 2))))

(defn format-results [n-agents results]
  (let [phases [:decide :execute :rebuild :produce :distribute :total]
        medians (into {} (for [p phases]
                           [p (median (map p results))]))]
    (str (format "n=%,d agents (%d periods measured):\n" n-agents (count results))
         (clojure.string/join
          "\n"
          (for [p phases]
            (format "  %-12s %8.2f ms" (name p) (get medians p))))
         "\n")))

(defn run-all-benchmarks
  "Run benchmarks at multiple scales. Write results to /tmp/abm_bench.txt."
  []
  (let [scales [[1000     10 2 5]    ;; [n-agents, n-periods, warmup, measured]
                [10000    10 2 5]
                [100000   10 2 5]
                [1000000  10 1 3]
                [10000000 10 1 2]]
        sb (StringBuilder.)]
    (.append sb (format "Firms ABM Benchmark — %s\n" (java.time.LocalDateTime/now)))
    (.append sb (str (repeat 60 \=) "\n\n"))
    (doseq [[n-agents n-periods n-warmup n-measured] scales]
      (println (format "Benchmarking n=%,d agents..." n-agents))
      (let [results (run-bench n-agents n-periods n-warmup n-measured)
            text    (format-results n-agents results)]
        (println text)
        (.append sb text)
        (.append sb "\n")))
    (spit "/tmp/abm_bench.txt" (str sb))
    (println "Results written to /tmp/abm_bench.txt")))

;; Entry point
(defn -main [& _args]
  (run-all-benchmarks))

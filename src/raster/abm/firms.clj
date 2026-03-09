(ns raster.abm.firms
  "Firms ABM — agent-based model of endogenous firm formation.

   Port of FIRMS_C11 (120M agent scale) using SoA flat arrays,
   CSR firm membership, and SplittableRandom for deterministic
   parallel RNG.

   All numerical functions use deftm for walker type inference.
   Value types (AgentSoA, FirmSoA) bundle arrays so deftm can
   take the full simulation state without hitting param limits.
   defn used only for initialization and stats collection."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.arrays :refer [aget aset]]
            [raster.abm.firms.econ :as econ]
            [raster.abm.firms.membership :as mem]
            [raster.random :as rnd])
  (:import [java.util SplittableRandom]))

(declare produce-output-impl!)

;; ================================================================
;; Value Types — SoA bundles for deftm parameter passing
;; ================================================================

(defvalue AgentSoA
  [effort :- (Array float),
   income :- (Array float),
   theta :- (Array float),
   endowment :- (Array float),
   firm :- (Array int),          ;; current firm index (-1 = unemployed)
   friends :- (Array int),       ;; flat-packed int[n * n-friends]
   cache :- (Array float)])      ;; singleton cache: interleaved [e0 u0 e1 u1 ...]

(defvalue FirmSoA
  [a :- (Array float),           ;; production param a
   b :- (Array float),           ;; production param b
   beta :- (Array float),        ;; production param beta
   te :- (Array float),          ;; total effort
   output :- (Array float),
   n-workers :- (Array int),     ;; firm size (n-workers)
   alive :- (Array int),
   members :- (Array int),       ;; CSR values
   offsets :- (Array int)])      ;; CSR row pointers (member-offsets)

;; ================================================================
;; Configuration
;; ================================================================

(defrecord FirmsConfig
           [^int n-agents
            ^int n-friends          ;; friends per agent
            ^int n-alt-firms        ;; alternative firms to evaluate
            ^double activation-rate ;; fraction of agents activated per period
            ^int compensation-mode  ;; 0=equal-shares
            ^int n-periods
            ^long rng-seed
            ^int max-firms])        ;; firm slot capacity; = n-agents (leave-before-allocate ensures reuse)

(defn default-config
  "Default config matching C11 defaults."
  ([] (default-config 1000))
  ([n-agents]
   (->FirmsConfig n-agents 4 4 0.04 0 2500 42 n-agents)))

;; ================================================================
;; Initialization helpers (deftm)
;; ================================================================

(deftm compute-singleton-cache!
  "Pre-compute optimal singleton effort and utility for all agents.
   Writes to interleaved cache: cache[2*i] = effort, cache[2*i+1] = utility."
  [singleton-cache :- (Array float),
   theta :- (Array float), endowment :- (Array float),
   param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), current-firm :- (Array int),
   n :- Long] :- Void
  (dotimes [i n]
    (let [fi   (aget current-firm i)
          a    (double (aget param-a fi))
          b    (double (aget param-b fi))
          beta (double (aget param-beta fi))
          th   (double (aget theta i))
          end  (double (aget endowment i))
          e    (econ/solve-singleton-effort a b beta th end)
          Y    (econ/production a b beta e)
          u    (econ/log-utility th Y end e)
          idx  (int (* 2 i))]
      (aset singleton-cache idx (float e))
      (aset singleton-cache (inc idx) (float u)))))

(deftm init-efforts-from-singleton!
  "Set initial efforts from singleton cache and accumulate total-effort."
  [effort :- (Array float), singleton-cache :- (Array float),
   current-firm :- (Array int), total-effort :- (Array float),
   n :- Long] :- Void
  (dotimes [i n]
    (let [e  (double (aget singleton-cache (int (* 2 i))))
          fi (aget current-firm i)]
      (aset effort i (float e))
      (aset total-effort fi (float (+ (double (aget total-effort fi)) e))))))

;; ================================================================
;; Initialization — returns [config AgentSoA FirmSoA rng]
;; ================================================================

(defn init-simulation
  "Initialize a complete simulation state.
   Returns [config agents firms rng] where agents is AgentSoA
   and firms is FirmSoA."
  ([] (init-simulation (default-config)))
  ([^FirmsConfig config]
   (let [n         (.n-agents config)
         nf        (.n-friends config)
         seed      (.rng-seed config)
         max-firms (.max-firms config)
         rng       (SplittableRandom. seed)
         ;; Agent arrays
         effort     (float-array n)
         income     (float-array n)
         theta      (float-array n)
         endowment  (float-array n)
         current-firm (int-array n)
         friends    (int-array (* n nf))
         s-cache    (float-array (* 2 n))
         ;; Firm arrays
         param-a    (float-array max-firms)
         param-b    (float-array max-firms)
         param-beta (float-array max-firms)
         total-effort (float-array max-firms)
         output     (float-array max-firms)
         firm-size  (int-array max-firms)
         alive      (int-array max-firms)
         members    (int-array n)
         member-offsets (int-array (inc max-firms))]
     ;; Init agent parameters
     (dotimes [i n]
       (clojure.core/aset ^floats theta i (float (.nextDouble rng 0.0 1.0)))
       (clojure.core/aset ^floats endowment i (float 1.0))
       (clojure.core/aset ^floats effort i (float 0.0))
       (clojure.core/aset ^floats income i (float 0.0))
       (clojure.core/aset ^ints current-firm i (int i)))
     ;; Init friends (random, excluding self)
     (dotimes [i n]
       (dotimes [j nf]
         (loop []
           (let [friend (.nextInt rng 0 n)]
             (if (== friend i)
               (recur)
               (clojure.core/aset ^ints friends (+ (* i nf) j) (int friend)))))))
     ;; Init firm parameters (one per agent initially)
     (dotimes [i n]
       (clojure.core/aset ^floats param-a i (float (.nextDouble rng 0.1 1.5)))
       (clojure.core/aset ^floats param-b i (float (.nextDouble rng 0.1 1.5)))
       (clojure.core/aset ^floats param-beta i (float (.nextDouble rng 1.0 1.99)))
       (clojure.core/aset ^floats total-effort i (float 0.0))
       (clojure.core/aset ^floats output i (float 0.0))
       (clojure.core/aset ^ints firm-size i (int 1))
       (clojure.core/aset ^ints alive i (int 1)))
     ;; Build initial CSR (each firm = 1 member)
     (dotimes [i n]
       (clojure.core/aset ^ints members i (int i)))
     (dotimes [i (inc n)]
       (clojure.core/aset ^ints member-offsets i (int i)))
     (let [agents (->AgentSoA effort income theta endowment current-firm
                              friends s-cache)
           firms  (->FirmSoA param-a param-b param-beta total-effort output
                             firm-size alive members member-offsets)]
       ;; Compute singleton cache
       (compute-singleton-cache! s-cache theta endowment
                                 param-a param-b param-beta current-firm n)
       ;; Set initial efforts
       (init-efforts-from-singleton! effort s-cache current-firm
                                     total-effort n)
       ;; Compute initial output
       (produce-output-impl! param-a param-b param-beta total-effort
                             output alive max-firms)
       [config agents firms rng]))))

;; ================================================================
;; Production and Distribution (deftm)
;; ================================================================

(deftm produce-output-impl!
  "Compute production for all alive firms."
  [param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), total-effort :- (Array float),
   output :- (Array float), alive :- (Array int),
   max-firms :- Long] :- Void
  (dotimes [fi max-firms]
    (when (== 1 (aget alive fi))
      (let [E (double (aget total-effort fi))]
        (aset output fi
              (float (if (> E 0.0)
                       (econ/production (double (aget param-a fi))
                                        (double (aget param-b fi))
                                        (double (aget param-beta fi)) E)
                       0.0)))))))

(deftm distribute-income-impl!
  "Distribute output to workers via CSR (equal shares)."
  [income :- (Array float), output :- (Array float),
   firm-size :- (Array int), alive :- (Array int),
   members :- (Array int), member-offsets :- (Array int),
   max-firms :- Long] :- Void
  (dotimes [fi max-firms]
    (when (== 1 (aget alive fi))
      (let [Y    (double (aget output fi))
            n    (aget firm-size fi)
            wage (float (if (> n 0) (/ Y (double n)) 0.0))
            start (aget member-offsets fi)
            end   (aget member-offsets (inc fi))]
        (loop [j start]
          (when (< j end)
            (aset income (aget members j) wage)
            (recur (unchecked-add-int j 1))))))))

;; ================================================================
;; Agent Decision (deftm)
;; ================================================================

(deftm agent-decide-impl!
  "Single agent evaluates options and queues decision.
   Inline NR for current firm, cached output for alternatives."
  [agent-id :- Long,
   ;; Agent arrays
   theta :- (Array float), endowment :- (Array float),
   effort :- (Array float), current-firm :- (Array int),
   friends :- (Array int),
   s-cache :- (Array float),
   ;; Firm arrays
   param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), total-effort :- (Array float),
   output :- (Array float),
   firm-size :- (Array int), alive :- (Array int),
   ;; Decision queue arrays
   q-agent-ids :- (Array int), q-types :- (Array int),
   q-targets :- (Array int), q-efforts :- (Array float),
   q-count :- (Array int),
   ;; Config
   n-friends :- Long] :- Long
  (let [fi    (aget current-firm agent-id)
        th    (double (aget theta agent-id))
        end   (double (aget endowment agent-id))
        old-e (double (aget effort agent-id))
        fsize (aget firm-size fi)
        a     (double (aget param-a fi))
        b     (double (aget param-b fi))
        beta  (double (aget param-beta fi))
        tot-e (double (aget total-effort fi))
        others-e (Math/max (- tot-e old-e) 0.0)
        sc-idx (int (* 2 agent-id))
        ;; Evaluate current firm
        cur-e (if (== fsize 1)
                (double (aget s-cache sc-idx))
                ;; Inline Newton-Raphson for team case
                (let [e-min 0.001
                      e-max (- end 0.001)]
                  (loop [guess (* th end) iter (int 0)]
                    (if (>= iter 10)
                      (Math/max e-min (Math/min e-max guess))
                      (let [e    (Math/max e-min (Math/min e-max guess))
                            E    (+ e others-e)
                            Ebm2 (Math/pow E (- beta 2.0))
                            Ebm1 (* Ebm2 E)
                            Eb   (* Ebm1 E)
                            Y    (+ (* a E) (* b Eb))
                            MP   (+ a (* b beta Ebm1))
                            dMP  (* b beta (- beta 1.0) Ebm2)
                            leis (- end e)
                            fe   (- (/ (* th MP) Y)
                                    (/ (- 1.0 th) leis))
                            fpe  (- (/ (* th (- (* dMP Y) (* MP MP)))
                                       (* Y Y))
                                    (/ (- 1.0 th)
                                       (* leis leis)))]
                        (if (or (< (Math/abs fe) 1e-6)
                                (== fpe 0.0))
                          e
                          (let [step  (/ fe fpe)
                                step  (Math/max (- (/ e-max 2.0))
                                                (Math/min (/ e-max 2.0) step))
                                new-e (- e step)]
                            (if (< (Math/abs step) 0.001)
                              (Math/max e-min (Math/min e-max new-e))
                              (recur new-e (unchecked-add-int iter 1))))))))))
        cur-u (if (== fsize 1)
                (double (aget s-cache (inc sc-idx)))
                ;; Compute utility from converged effort (one pow call)
                (let [E    (+ cur-e others-e)
                      Ebm2 (Math/pow E (- beta 2.0))
                      Y    (+ (* a E) (* b (* (* Ebm2 E) E)))
                      wage (/ Y (double fsize))
                      leis (- end cur-e)]
                  (if (> leis 1e-300)
                    (+ (* th (Math/log wage))
                       (* (- 1.0 th) (Math/log leis)))
                    (* th (Math/log wage)))))
        friend-base (* agent-id n-friends)]
    ;; Loop over friends' firms using cached output, then compare startup
    (loop [j (int 0)
           best-fi   (int fi)
           best-e    cur-e
           best-u    cur-u
           best-type (int 0)]
      (if (>= j n-friends)
        ;; Done evaluating friends — compare with startup and enqueue
        (let [su-u (double (aget s-cache (inc sc-idx)))]
          (if (> su-u best-u)
            (mem/enqueue! q-agent-ids q-types q-targets q-efforts q-count
                          agent-id 2 -1 (aget s-cache sc-idx))
            (mem/enqueue! q-agent-ids q-types q-targets q-efforts q-count
                          agent-id (long best-type) (long best-fi)
                          (float best-e))))
        ;; Evaluate friend's firm using cached output (no pow calls)
        (let [fid    (aget friends (+ friend-base j))
              alt-fi (aget current-firm fid)]
          (if (or (== alt-fi fi)
                  (== (aget alive alt-fi) 0))
            (recur (unchecked-add-int j 1)
                   best-fi best-e best-u best-type)
            (let [af-Y    (double (aget output alt-fi))
                  af-size (aget firm-size alt-fi)]
              (if (or (<= af-Y 0.0) (<= af-size 0))
                (recur (unchecked-add-int j 1)
                       best-fi best-e best-u best-type)
                (let [af-te   (double (aget total-effort alt-fi))
                      ;; Wage if I join: Y / (N+1)
                      af-wage (/ af-Y (double (inc af-size)))
                      ;; Estimate effort: average effort in firm
                      af-e    (/ af-te (double af-size))
                      af-e    (Math/max 0.001 (Math/min (- end 0.001) af-e))
                      af-leis (- end af-e)
                      af-u    (if (> af-leis 1e-300)
                                (+ (* th (Math/log (Math/max 1e-300 af-wage)))
                                   (* (- 1.0 th) (Math/log af-leis)))
                                (* th (Math/log (Math/max 1e-300 af-wage))))]
                  (if (> af-u best-u)
                    (recur (unchecked-add-int j 1)
                           alt-fi af-e af-u (int 1))
                    (recur (unchecked-add-int j 1)
                           best-fi best-e best-u best-type)))))))))))

;; ================================================================
;; Statistics (deftm)
;; ================================================================

(deftm compute-stats-impl
  "Compute mean effort and income."
  [effort :- (Array float), income :- (Array float), n :- Long] :- (Array double)
  (let [result (double-array 2)]
    (loop [i (int 0) se 0.0 si 0.0]
      (if (>= i n)
        (do (aset result 0 (/ se (double n)))
            (aset result 1 (/ si (double n)))
            result)
        (recur (unchecked-add-int i 1)
               (+ se (double (aget effort i)))
               (+ si (double (aget income i))))))))

(deftm count-alive-firms
  "Count alive firms and find max firm size."
  [alive :- (Array int), firm-size :- (Array int),
   max-firms :- Long] :- (Array long)
  (let [result (long-array 2)]
    (loop [f (int 0) cnt (int 0) maxsz (int 0)]
      (if (>= f max-firms)
        (do (aset result 0 (long cnt))
            (aset result 1 (long maxsz))
            result)
        (if (== 1 (aget alive f))
          (recur (unchecked-add-int f 1)
                 (unchecked-add-int cnt 1)
                 (int (clojure.core/max maxsz (int (aget firm-size f)))))
          (recur (unchecked-add-int f 1) cnt maxsz))))))

(defn collect-stats
  "Collect period statistics from AgentSoA and FirmSoA."
  [^AgentSoA agents ^FirmSoA firms n max-firms period queue-count]
  (let [s  (compute-stats-impl (.effort agents) (.income agents) n)
        fc (count-alive-firms (.alive firms) (.n-workers firms) max-firms)]
    {:period      period
     :n-alive     (clojure.core/aget ^longs fc 0)
     :max-size    (clojure.core/aget ^longs fc 1)
     :mean-effort (clojure.core/aget ^doubles s 0)
     :mean-income (clojure.core/aget ^doubles s 1)
     :queue-count queue-count}))

;; ================================================================
;; Simulation Period — deftm core (all flat arrays, no records)
;; ================================================================

(deftm run-period-impl!
  "Execute one simulation period. Takes AgentSoA and FirmSoA value types.
   Returns queue count (Long) for stats collection."
  [rng :- Object,
   agents :- AgentSoA, firms :- FirmSoA,
   n-agents :- Long, n-friends :- Long,
   n-active :- Long, max-firms :- Long] :- Long
  (let [;; Extract arrays from value types
        theta       (.theta agents)
        endowment   (.endowment agents)
        effort      (.effort agents)
        income      (.income agents)
        current-firm (.firm agents)
        friends     (.friends agents)
        s-cache     (.cache agents)
        param-a     (.a firms)
        param-b     (.b firms)
        param-beta  (.beta firms)
        total-effort (.te firms)
        output      (.output firms)
        firm-size   (.n-workers firms)
        alive       (.alive firms)
        members     (.members firms)
        member-offsets (.offsets firms)
        ;; Allocate decision queue arrays
        q-agent-ids  (int-array n-active)
        q-types      (int-array n-active)
        q-targets    (int-array n-active)
        q-efforts    (float-array n-active)
        q-count      (int-array 1)
        ;; Generate per-agent seeds for startup firm allocation
        rng-seeds    (long-array n-agents)]
    (rnd/fill-seeds! rng rng-seeds n-agents)

    ;; 1. Agent decisions (random activation)
    (dotimes [_ n-active]
      (let [agent-id (rnd/rand-int rng n-agents)]
        (agent-decide-impl!
         agent-id theta endowment effort current-firm friends
         s-cache
         param-a param-b param-beta total-effort output
         firm-size alive
         q-agent-ids q-types q-targets q-efforts q-count
         n-friends)))

    ;; 2. Execute decisions (serial)
    (let [cnt (long (aget q-count 0))]
      (mem/execute-decisions-impl!
       q-agent-ids q-types q-targets q-efforts
       cnt effort current-firm total-effort firm-size alive
       param-a param-b param-beta output
       max-firms rng-seeds)

      ;; 3. Rebuild CSR
      (mem/rebuild-csr! current-firm firm-size alive
                        members member-offsets n-agents max-firms)

      ;; 4. Produce output
      (produce-output-impl! param-a param-b param-beta total-effort
                            output alive max-firms)

      ;; 5. Distribute income (equal shares)
      (distribute-income-impl! income output firm-size alive
                               members member-offsets max-firms)

      cnt)))

;; ================================================================
;; Simulation Period (sequential) — thin wrapper
;; ================================================================

(defn run-period!
  "Execute one simulation period (sequential). Returns stats map."
  [^FirmsConfig config ^AgentSoA agents ^FirmSoA firms
   ^SplittableRandom rng period]
  (let [n          (.n-agents config)
        n-friends  (.n-friends config)
        act-rate   (.activation-rate config)
        n-active   (int (Math/round (* act-rate n)))
        max-firms  (long (.max-firms config))
        cnt (run-period-impl!
             rng agents firms
             n n-friends (long (clojure.core/max 1 n-active)) max-firms)]
    (collect-stats agents firms n max-firms period cnt)))

;; ================================================================
;; Simulation Period (parallel decision phase)
;; ================================================================

(defn run-period-parallel!
  "Execute one simulation period with parallel agent decisions.
   Agent decisions are read-only on shared state — each thread
   writes to its own queue arrays, which are then executed serially."
  [^FirmsConfig config ^AgentSoA agents ^FirmSoA firms
   ^SplittableRandom rng period]
  (let [n          (.n-agents config)
        n-friends  (.n-friends config)
        act-rate   (.activation-rate config)
        n-active   (int (Math/round (* act-rate n)))
        max-firms  (long (.max-firms config))
        ;; Pre-select activated agents (sequential, deterministic)
        active-ids (let [ids (int-array n-active)]
                     (dotimes [i n-active]
                       (clojure.core/aset ^ints ids i (int (.nextInt rng 0 n))))
                     ids)
        ;; Generate per-agent seeds for startup firm allocation
        rng-seeds  (let [seeds (long-array n)]
                     (dotimes [i n]
                       (clojure.core/aset ^longs seeds i (.nextLong rng)))
                     seeds)
        ;; Extract arrays (shared, read-only during decisions)
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
        ;; Parallel decision phase: chunk agents across threads
        n-threads  (.. Runtime getRuntime availableProcessors)
        chunk-size (clojure.core/max 1 (int (Math/ceil (/ (double n-active) n-threads))))
        futures    (vec
                    (for [chunk-start (range 0 n-active chunk-size)]
                      (let [chunk-end (int (clojure.core/min n-active
                                                             (+ chunk-start chunk-size)))
                            chunk-n   (- chunk-end chunk-start)
                            ;; Thread-local queue arrays
                            q-aids    (int-array (clojure.core/max 1 chunk-n))
                            q-types   (int-array (clojure.core/max 1 chunk-n))
                            q-targets (int-array (clojure.core/max 1 chunk-n))
                            q-efforts (float-array (clojure.core/max 1 chunk-n))
                            q-count   (int-array 1)]
                        (future
                          (dotimes [i chunk-n]
                            (let [agent-id (clojure.core/aget ^ints active-ids
                                                              (+ chunk-start i))]
                              (agent-decide-impl!
                               agent-id theta endowment effort current-firm friends
                               s-cache
                               param-a param-b param-beta total-effort output
                               firm-size alive
                               q-aids q-types q-targets q-efforts q-count
                               n-friends)))
                          ;; Return the queue arrays + count
                          [q-aids q-types q-targets q-efforts
                           (clojure.core/aget ^ints q-count 0)]))))
        ;; Collect all thread-local queues
        queues     (mapv deref futures)
        total-cnt  (long (reduce + 0 (map #(nth % 4) queues)))]

    ;; 2. Execute all queued decisions serially
    (doseq [[q-aids q-types q-targets q-efforts cnt] queues]
      (when (> (long cnt) 0)
        (mem/execute-decisions-impl!
         q-aids q-types q-targets q-efforts
         (long cnt) effort current-firm total-effort firm-size alive
         param-a param-b param-beta output
         max-firms rng-seeds)))

    ;; 3. Rebuild CSR
    (mem/rebuild-csr! current-firm firm-size alive
                      members member-offsets n max-firms)

    ;; 4. Produce output
    (produce-output-impl! param-a param-b param-beta total-effort
                          output alive max-firms)

    ;; 5. Distribute income (equal shares)
    (distribute-income-impl! (.income agents) output firm-size alive
                             members member-offsets max-firms)

    ;; 6. Collect stats
    (collect-stats agents firms n max-firms period total-cnt)))

;; ================================================================
;; Full Simulation
;; ================================================================

(defn run-simulation
  "Run the full simulation. Returns vector of per-period stats.
   Set :parallel? true for parallel decision phase."
  ([] (run-simulation (default-config)))
  ([config] (run-simulation config false))
  ([^FirmsConfig config parallel?]
   (let [[config agents firms rng] (init-simulation config)
         n-periods (.n-periods config)
         run-fn    (if parallel? run-period-parallel! run-period!)]
     (loop [t 0 stats (transient [])]
       (if (>= t n-periods)
         (persistent! stats)
         (let [s (run-fn config agents firms rng t)
               stats' (conj! stats s)]
           (when (or (zero? t) (zero? (rem t 100)) (== t (dec n-periods)))
             (println (format "Period %d: firms=%d effort=%.4f income=%.4f max-size=%d"
                              t (:n-alive s) (:mean-effort s)
                              (:mean-income s) (:max-size s))))
           (recur (inc t) stats')))))))

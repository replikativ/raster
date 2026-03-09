(ns raster.abm.firms.phases
  "Backend-agnostic ABM phases using par primitives.

   Each phase is expressed using raster.par/map-void!, raster.par/scan,
   and raster.par/atomic-add! so the compiler pipeline can target them
   to GPU kernels or SIMD. On CPU, the par macros expand to sequential
   loops giving identical results.

   The execute-decisions phase remains serial (3% of time, needs
   firm slot allocation). All other phases are parallel.

   Usage (CPU fallback — par macros expand to dotimes):
     (run-period! rng agents firms n-agents n-friends n-active max-firms)"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.par :as par]
            [raster.abm.firms.econ :as econ]
            [raster.abm.firms.membership :as mem]
            [raster.abm.firms :as firms]
            [raster.random :as rnd])
  (:import [raster.abm.firms AgentSoA FirmSoA]))

;; ================================================================
;; Phase 1: produce-output-par! — per-firm production
;; ================================================================

(deftm produce-output-par!
  "Compute production Y = a*E + b*E^β for all firms.
   Uses par/map-void! for GPU compilation."
  [param-a :- (Array float),
   param-b :- (Array float),
   param-beta :- (Array float),
   total-effort :- (Array float),
   output :- (Array float),
   alive :- (Array int),
   max-firms :- Long] :- Void
  (par/map-void! fi max-firms
                 (if (== 1 (aget alive fi))
                   (let [E (double (aget total-effort fi))]
                     (aset output fi
                           (float (if (> E 0.0)
                                    (+ (* (double (aget param-a fi)) E)
                                       (* (double (aget param-b fi))
                                          (Math/pow E (double (aget param-beta fi)))))
                                    0.0))))
                   (aset output fi (float 0.0)))))

;; ================================================================
;; Phase 2: distribute-income-par! — per-firm income distribution
;; ================================================================

(deftm distribute-income-par!
  "Distribute output to workers via CSR (equal shares).
   Uses par/map-void! for GPU compilation."
  [income :- (Array float), output :- (Array float),
   firm-size :- (Array int), alive :- (Array int),
   members :- (Array int), member-offsets :- (Array int),
   max-firms :- Long] :- Void
  (par/map-void! fi max-firms
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
;; Phase 3: rebuild-csr-par! — 5 sub-phases
;; ================================================================

(deftm zero-firm-sizes!
  "Zero all firm sizes. Sub-phase 1 of CSR rebuild."
  [firm-size :- (Array int), max-firms :- Long] :- Void
  (par/map-void! f max-firms
                 (aset firm-size f (int 0))))

(deftm histogram-firms!
  "Count agents per firm using atomic adds. Sub-phase 2."
  [firm-size :- (Array int), current-firm :- (Array int),
   n :- Long] :- Void
  (par/map-void! i n
                 (when (>= (aget current-firm i) 0)
                   (par/atomic-add! firm-size (aget current-firm i) (int 1)))))

(deftm update-alive!
  "Set alive flag based on firm size. Sub-phase 4."
  [alive :- (Array int), firm-size :- (Array int),
   max-firms :- Long] :- Void
  (par/map-void! f max-firms
                 (aset alive f (int (if (> (aget firm-size f) 0) 1 0)))))

(deftm fill-members-par!
  "Fill CSR members array using atomic position counters. Sub-phase 5."
  [members :- (Array int), pos :- (Array int),
   member-offsets :- (Array int), current-firm :- (Array int),
   n :- Long] :- Void
  ;; First zero position counters (reuse pos array)
  (par/map-void! f (alength pos)
                 (aset pos f (int 0)))
  ;; Fill members with atomic position allocation
  (par/map-void! i n
                 (when (>= (aget current-firm i) 0)
                   (let [f (aget current-firm i)
                         p (par/atomic-add! pos f (int 1))]
                     (aset members (+ (aget member-offsets f) p) (int i))))))

(deftm rebuild-csr-par!
  "Rebuild CSR membership using par primitives.
   Returns count of alive firms."
  [current-firm :- (Array int), firm-size :- (Array int),
   alive :- (Array int), members :- (Array int),
   member-offsets :- (Array int), n :- Long, max-firms :- Long] :- Long
  ;; Sub-phase 1: Zero sizes
  (zero-firm-sizes! firm-size max-firms)
  ;; Sub-phase 2: Histogram
  (histogram-firms! firm-size current-firm n)
  ;; Sub-phase 3: Exclusive prefix sum → offsets
  (par/scan-exclusive member-offsets acc (int 0) f max-firms int
                      (+ acc (aget firm-size f)))
  ;; Sub-phase 4: Update alive
  (update-alive! alive firm-size max-firms)
  ;; Sub-phase 5: Fill members
  (let [pos (int-array max-firms)]
    (fill-members-par! members pos member-offsets current-firm n)
    ;; Count alive
    (loop [f (int 0) cnt (int 0)]
      (if (>= f max-firms)
        (long cnt)
        (recur (unchecked-add-int f 1)
               (if (== 1 (aget alive f))
                 (unchecked-add-int cnt 1)
                 cnt))))))

;; ================================================================
;; Phase 3b: csr-scan-par! and startup-scan-par! — GPU-resident prefix sums
;; ================================================================

(deftm csr-scan-par!
  "Exclusive prefix sum of firm-size → member-offsets (n+1 elements).
   Compiled by opencl-pass to Blelloch scan kernels; output DeviceBuffer
   avoids the firm-size download + member-offsets upload round-trip."
  [member-offsets :- (Array int), firm-size :- (Array int),
   max-firms :- Long] :- Void
  (par/scan-exclusive member-offsets acc (int 0) f max-firms int
                      (+ acc (aget firm-size f))))

(deftm startup-scan-par!
  "Exclusive prefix sum of not-alive → free-offsets (max-firms+1 elements).
   Used for compacting dead firm slots for STARTUP decisions."
  [free-offsets :- (Array int), not-alive :- (Array int),
   max-firms :- Long] :- Void
  (par/scan-exclusive free-offsets acc (int 0) f max-firms int
                      (+ acc (aget not-alive f))))

;; ================================================================
;; Phase 3b.5: generate-active-ids-par! — GPU-side active agent selection
;; ================================================================

(deftm generate-active-ids-par!
  "Fill active-ids with splitmix64-derived random indices in [0, n-agents).
   Compiled by opencl-pass to a parallel GPU kernel.
   Eliminates 1.6MB active-ids upload per period (at n=10M, 10% activation)."
  [active-ids :- (Array int), n-active :- Long,
   n-agents :- Long, base-seed :- Long] :- Void
  (par/active-ids! active-ids n-active n-agents base-seed))

;; ================================================================
;; Phase 3c: fill-rng-seeds-par! — GPU-side RNG seed generation
;; ================================================================

(deftm fill-rng-seeds-par!
  "Fill rng-seeds with splitmix64 seeds derived from base-seed.
   Compiled by opencl-pass to a parallel GPU kernel.
   Eliminates 80MB rng-seeds upload per period."
  [rng-seeds :- (Array long), n :- Long, base-seed :- Long] :- Void
  (par/rng-fill! rng-seeds n base-seed))

;; ================================================================
;; Phase 4: agent-decide-par! — parallel agent decisions
;; ================================================================

(deftm agent-decide-par!
  "All activated agents evaluate options and queue decisions.
   Uses par/map-void! over active agents. Each agent is independent:
   reads shared state, writes to queue arrays via atomic-add!.

   Inlines the NR solver and friend evaluation loop."
  [active-ids :- (Array int), n-active :- Long,
   agents :- AgentSoA, firms :- FirmSoA,
   ;; Decision queue arrays (written via atomic-add!/aset)
   agent-ids :- (Array int), decision-types :- (Array int),
   target-firms :- (Array int), new-efforts :- (Array float),
   q-count :- (Array int),
   ;; Config
   n-friends :- Long] :- Void
  (let [theta       (.theta agents)
        endowment   (.endowment agents)
        effort      (.effort agents)
        current-firm (.firm agents)
        friends     (.friends agents)
        s-cache     (.cache agents)
        param-a     (.a firms)
        param-b     (.b firms)
        param-beta  (.beta firms)
        total-effort (.te firms)
        output      (.output firms)
        firm-size   (.n-workers firms)
        alive       (.alive firms)]
    (par/map-void! k n-active
                   (let [agent-id (aget active-ids k)
                         fi    (aget current-firm agent-id)
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
          ;; Evaluate current firm effort
                         cur-e (if (== fsize 1)
                                 (double (aget s-cache sc-idx))
                                 (econ/solve-effort a b beta others-e th end))
          ;; Current firm utility
                         cur-u (if (== fsize 1)
                                 (double (aget s-cache (inc sc-idx)))
                                 (let [E    (+ cur-e others-e)
                                       Ebm2 (Math/pow E (- beta 2.0))
                                       Y    (+ (* a E) (* b (* (* Ebm2 E) E)))
                                       wage (/ Y (double fsize))]
                                   (econ/log-utility th wage end cur-e)))
                         friend-base (* agent-id n-friends)]
      ;; Loop over friends' firms, compare with startup, enqueue
                     (loop [j (int 0)
                            best-fi   (int fi)
                            best-e    cur-e
                            best-u    cur-u
                            best-type (int 0)]
                       (if (>= j n-friends)
          ;; Done evaluating friends — compare with startup and enqueue
                         (let [su-u (double (aget s-cache (inc sc-idx)))
                               final-type (if (> su-u best-u) (int 2) best-type)
                               final-fi   (if (> su-u best-u) (int -1) best-fi)
                               final-e    (if (> su-u best-u) (aget s-cache sc-idx) (float best-e))]
                           (par/collect! q-count
                                         agent-ids (int agent-id)
                                         decision-types final-type
                                         target-firms final-fi
                                         new-efforts final-e))
          ;; Evaluate friend's firm
                         (let [fid    (aget friends (+ friend-base j))
                               alt-fi (aget current-firm fid)]
                           (if (== alt-fi fi)
                             (recur (unchecked-add-int j 1)
                                    best-fi best-e best-u best-type)
                             (if (== (aget alive alt-fi) 0)
                               (recur (unchecked-add-int j 1)
                                      best-fi best-e best-u best-type)
                               (let [af-Y    (double (aget output alt-fi))
                                     af-size (aget firm-size alt-fi)]
                                 (if (<= af-Y 0.0)
                                   (recur (unchecked-add-int j 1)
                                          best-fi best-e best-u best-type)
                                   (if (<= af-size 0)
                                     (recur (unchecked-add-int j 1)
                                            best-fi best-e best-u best-type)
                                     (let [af-te   (double (aget total-effort alt-fi))
                                           af-wage (/ af-Y (double (inc af-size)))
                                           raw-af-e (/ af-te (double af-size))
                                           af-e    (Math/max 0.001 (Math/min (- end 0.001) raw-af-e))
                                           af-leis (- end af-e)
                                           af-u    (if (> af-leis 1e-300)
                                                     (+ (* th (Math/log (Math/max 1e-300 af-wage)))
                                                        (* (- 1.0 th) (Math/log af-leis)))
                                                     (* th (Math/log (Math/max 1e-300 af-wage))))]
                                       (if (> af-u best-u)
                                         (recur (unchecked-add-int j 1)
                                                alt-fi af-e af-u (int 1))
                                         (recur (unchecked-add-int j 1)
                                                best-fi best-e best-u best-type)))))))))))))))

;; ================================================================
;; Phase 5: GPU execute-decisions — parallel phases
;; ================================================================

(deftm execute-stay-switch-par!
  "Execute STAY(0) and SWITCH(1) decisions in parallel on GPU.
   Uses float atomic-add for total-effort (CAS loop in OpenCL)."
  [agent-ids :- (Array int), decision-types :- (Array int),
   target-firms :- (Array int), new-efforts :- (Array float),
   effort :- (Array float), current-firm :- (Array int),
   total-effort :- (Array float), firm-size :- (Array int),
   alive :- (Array int), cnt :- Long] :- Void
  (par/map-void! q cnt
                 (let [ai (aget agent-ids q)
                       dtype (aget decision-types q)
                       tgt (aget target-firms q)
                       new-e (aget new-efforts q)
                       old-fi (aget current-firm ai)
                       old-e (aget effort ai)]
                   (case (int dtype)
                     0 ;; STAY
                     (do (aset effort ai new-e)
                         (when (>= old-fi (int 0))
                           (par/atomic-add! total-effort old-fi (float (- (float new-e) (float old-e))))))
                     1 ;; SWITCH
                     (when (and (>= tgt (int 0)) (== (int 1) (aget alive tgt)))
                       (when (>= old-fi (int 0))
                         (par/atomic-add! total-effort old-fi (float (- (float 0.0) (float old-e))))
                         (par/atomic-add! firm-size old-fi (int -1)))
                       (aset current-firm ai (int tgt))
                       (aset effort ai new-e)
                       (par/atomic-add! total-effort tgt new-e)
                       (par/atomic-add! firm-size tgt (int 1)))
        ;; default: skip STARTUP (handled separately)
                     nil))))

(deftm count-startups-par!
  "Count STARTUP decisions and collect their queue indices into startup-indices.
   Uses atomic counter for parallel collection."
  [decision-types :- (Array int), startup-indices :- (Array int),
   startup-count :- (Array int), cnt :- Long] :- Void
  (par/map-void! q cnt
                 (when (== (aget decision-types q) (int 2))
                   (par/collect! startup-count
                                 startup-indices (int q)))))

(deftm build-not-alive-par!
  "Build not-alive flags: 1 where firm is dead, 0 where alive."
  [alive :- (Array int), not-alive :- (Array int),
   max-firms :- Long] :- Void
  (par/map-void! f max-firms
                 (aset not-alive f (int (if (== (aget alive f) (int 0)) 1 0)))))

(deftm scatter-free-slots-par!
  "Scatter dead firm indices into compacted free-slots array using prefix-sum offsets."
  [not-alive :- (Array int), free-offsets :- (Array int),
   free-slots :- (Array int), max-firms :- Long] :- Void
  (par/map-void! f max-firms
                 (when (== (aget not-alive f) (int 1))
                   (aset free-slots (aget free-offsets f) (int f)))))

(deftm execute-startups-par!
  "Execute STARTUP decisions in parallel, using pre-computed free firm slots."
  [agent-ids :- (Array int), startup-indices :- (Array int),
   new-efforts :- (Array float),
   effort :- (Array float), current-firm :- (Array int),
   total-effort :- (Array float), firm-size :- (Array int),
   alive :- (Array int), param-a :- (Array float), param-b :- (Array float),
   param-beta :- (Array float), output :- (Array float),
   free-slots :- (Array int), rng-seeds :- (Array long),
   n-startups :- Long] :- Void
  (par/map-void! s n-startups
                 (let [q (aget startup-indices s)
                       ai (aget agent-ids q)
                       new-e (aget new-efforts q)
                       old-fi (aget current-firm ai)
                       old-e (aget effort ai)
                       new-fi (aget free-slots s)
                       seed (aget rng-seeds ai)
                       a-new (float (+ 0.1 (* 1.4 (/ (float (int (bit-and seed (long 0xFFFF)))) 65536.0))))
                       b-new (float (+ 0.1 (* 1.4 (/ (float (int (bit-and (unsigned-bit-shift-right seed 16) (long 0xFFFF)))) 65536.0))))
                       beta-new (float (+ 1.0 (* 0.99 (/ (float (int (bit-and (unsigned-bit-shift-right seed 32) (long 0xFFFF)))) 65536.0))))]
      ;; Init new firm
                   (aset alive new-fi (int 1))
                   (aset param-a new-fi a-new)
                   (aset param-b new-fi b-new)
                   (aset param-beta new-fi beta-new)
                   (aset total-effort new-fi new-e)
                   (aset output new-fi (float 0.0))
                   (aset firm-size new-fi (int 1))
      ;; Leave old firm
                   (when (>= old-fi (int 0))
                     (par/atomic-add! total-effort old-fi (float (- (float 0.0) (float old-e))))
                     (par/atomic-add! firm-size old-fi (int -1)))
      ;; Join new
                   (aset current-firm ai (int new-fi))
                   (aset effort ai new-e))))

(deftm update-alive-post-decisions-par!
  "Set alive=0 for firms with size<=0 after decision execution."
  [alive :- (Array int), firm-size :- (Array int),
   max-firms :- Long] :- Void
  (par/map-void! f max-firms
                 (when (<= (aget firm-size f) (int 0))
                   (aset alive f (int 0)))))

;; ================================================================
;; Orchestrator: run-period!
;; ================================================================

(deftm run-period!
  "Execute one simulation period using par primitives.
   Agent decisions are parallelized via map-void! with atomic queue writes.
   Execute-decisions remains serial.
   Returns queue count for stats."
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
        ;; Generate per-agent seeds and pre-select activated agents
        rng-seeds    (long-array n-agents)
        active-ids   (int-array n-active)]
    (rnd/fill-seeds! rng rng-seeds n-agents)
    ;; Pre-select activated agents (sequential, deterministic)
    (dotimes [i n-active]
      (aset active-ids i (int (rnd/rand-int rng n-agents))))

    ;; 1. Agent decisions (parallel via map-void!)
    (agent-decide-par!
     active-ids n-active
     agents firms
     q-agent-ids q-types q-targets q-efforts q-count
     n-friends)

    ;; 2. Execute decisions (serial — needs firm slot allocation)
    (let [cnt (long (aget q-count 0))]
      (mem/execute-decisions-impl!
       q-agent-ids q-types q-targets q-efforts
       cnt effort current-firm total-effort firm-size alive
       param-a param-b param-beta output
       max-firms rng-seeds)

      ;; 3. Rebuild CSR (par sub-phases)
      (rebuild-csr-par! current-firm firm-size alive
                        members member-offsets n-agents max-firms)

      ;; 4. Produce output (par)
      (produce-output-par! param-a param-b param-beta total-effort
                           output alive max-firms)

      ;; 5. Distribute income (par)
      (distribute-income-par! income output firm-size alive
                              members member-offsets max-firms)

      cnt)))

;; ================================================================
;; Public API (thin wrapper)
;; ================================================================

(defn run-period
  "Execute one simulation period using par primitives. Returns stats map."
  [config ^AgentSoA agents ^FirmSoA firms rng period]
  (let [n         (long (.n-agents config))
        n-friends (long (.n-friends config))
        act-rate  (.activation-rate config)
        n-active  (int (Math/round (* act-rate n)))
        max-firms (long (.max-firms config))
        cnt (run-period!
             rng agents firms
             n n-friends (long (clojure.core/max 1 n-active)) max-firms)]
    (firms/collect-stats agents firms n max-firms period cnt)))

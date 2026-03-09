;; # Agent-Based Modeling: Endogenous Firm Formation

;; authors: Christian Weilbach

;; This example demonstrates how to build a GPU-compiled agent-based model
;; (ABM) with Raster. The model implements endogenous firm formation following
;; Robert Axtell's research on artificial economies — millions of heterogeneous
;; agents self-organize into firms, producing emergent phenomena like
;; heavy-tailed firm-size distributions (Zipf's law) and realistic labor
;; market dynamics.

;; ## Background: Axtell's Firms Model

;; Robert Axtell (George Mason University, Santa Fe Institute) has developed a
;; line of agent-based models that generate realistic macroeconomic phenomena
;; from simple micro-level rules. Key references:
;;
;; - Axtell, R. (2001). "Zipf Distribution of U.S. Firm Sizes."
;;   *Science* 293(5536), 1818-1820.
;;   — First demonstration that firm-size distributions follow Zipf's law,
;;   reproduced by an ABM with utility-maximizing agents.
;;
;; - Axtell, R. (2016). "120 Million Agents Self-Organize into 6 Million
;;   Firms: A Model of the U.S. Private Sector." *AAMAS '16*, pp. 806-816.
;;   https://dl.acm.org/doi/10.5555/2936924.2937042
;;   — Census-scale simulation (120M agents) on commodity hardware,
;;   demonstrating ABMs at realistic population sizes.
;;
;; - Axtell, R. (2018). "Endogenous Firm Dynamics and Labor Flows via
;;   Heterogeneous Agents." *Handbook of Computational Economics*, Vol. 4,
;;   pp. 157-213. Elsevier.
;;   — Full description of the model with increasing-returns production,
;;   Cobb-Douglas preferences, and endogenous firm birth/death.
;;
;; The central insight: when agents freely choose which firms to join and how
;; hard to work, the equilibrium is **not** a single representative firm.
;; Instead, a rich ecology of firms emerges, with sizes following a power law.
;; This happens because the production function has increasing returns
;; (bigger teams are more productive per worker), but free-riding incentives
;; keep firms from growing without bound.

;; ## Why Agent-Based Models?

;; Traditional economic models assume representative agents and market clearing.
;; ABMs instead simulate individual decision-makers with heterogeneous
;; preferences, bounded rationality, and local information. This produces:
;;
;; - **Emergent macro from micro:** Aggregate statistics (GDP, unemployment,
;;   firm-size distributions) arise from individual choices, not assumptions.
;; - **Out-of-equilibrium dynamics:** Firms are born, grow, shrink, and die.
;;   The economy is never "in equilibrium" — it evolves continuously.
;; - **Heterogeneity matters:** Not all agents are alike. The distribution
;;   of preferences (θ) directly shapes the size distribution of firms.
;;
;; The challenge is computational: Axtell's model with 120M agents requires
;; efficient data structures and parallelism. Raster compiles `deftm`
;; functions with `par/map!` to GPU kernels, enabling census-scale simulation.

;; ## Setup

(ns raster.abm-firms
  (:require [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.abm.firms :as firms]
            [raster.abm.firms.econ :as econ]
            [scicloj.kindly.v4.kind :as kind])
  (:import [raster.abm.firms FirmsConfig AgentSoA FirmSoA]))

;; ## The Economic Model

;; ### Production Function
;;
;; Each firm produces output from the total effort of its workers:
;;
;; $$Y = a \cdot E + b \cdot E^{\beta}$$
;;
;; where $E = \sum_i e_i$ is the total effort, and $a, b, \beta$ are
;; firm-specific parameters drawn at founding. When $\beta > 1$, the
;; production function exhibits **increasing returns to scale** — a team
;; of 10 workers each putting in effort 1 produces more than 10 times
;; what a single worker produces. This creates an incentive to form firms.

;; Let's visualize the production function for a typical firm:

(let [a 0.5, b 0.3, beta 1.5
      efforts (mapv #(* 0.1 %) (range 1 101))
      outputs (mapv #(econ/production a b beta (double %)) efforts)]
  (kind/vega-lite
   {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
    :width 500 :height 300
    :title "Production Function: Y = a·E + b·E^β"
    :data {:values (mapv (fn [e y] {:effort e :output y}) efforts outputs)}
    :mark {:type "line" :strokeWidth 2}
    :encoding {:x {:field "effort" :type "quantitative" :title "Total Effort (E)"}
               :y {:field "output" :type "quantitative" :title "Output (Y)"}}}))

;; ### Agent Utility
;;
;; Each agent has a Cobb-Douglas utility function:
;;
;; $$U = \theta \cdot \ln(\text{income}) + (1-\theta) \cdot \ln(\omega - e)$$
;;
;; where $\theta \in [0,1]$ is the agent's taste for consumption (vs.
;; leisure), $\omega$ is the time endowment, and $e$ is effort. Agents
;; choose effort to maximize utility given their firm's production function
;; and the other workers' efforts. With equal-shares compensation
;; (income = Y/n), the first-order condition is:
;;
;; $$\frac{\theta \cdot MP}{Y} = \frac{1-\theta}{\omega - e}$$
;;
;; where $MP = a + b \cdot \beta \cdot E^{\beta-1}$ is the marginal product.
;; This is solved numerically via Newton-Raphson.

;; ### The Free-Rider Problem
;;
;; Under equal shares, a worker's income is $Y/n$ but their marginal
;; contribution to output is $MP = a + b \cdot \beta \cdot E^{\beta-1}$.
;; As the firm grows, each worker's share of output falls even as total
;; output rises. This creates a **free-rider incentive**: workers in large
;; firms reduce effort because they capture only $1/n$ of their marginal
;; product.
;;
;; This tension between increasing returns (which favors large firms) and
;; free-riding (which limits them) produces the emergent firm-size
;; distribution.

;; Let's see how optimal effort depends on others' effort in a firm:

(let [a 0.5 b 0.3 beta 1.5 theta 0.5 endow 1.0
      others-efforts (mapv #(* 0.1 %) (range 0 21))
      own-efforts (mapv #(econ/solve-effort a b beta (double %) theta endow)
                        others-efforts)]
  (kind/vega-lite
   {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
    :width 500 :height 300
    :title "Free-Rider Effect: Own Effort vs. Others' Effort"
    :data {:values (mapv (fn [oe me] {:others oe :own me})
                         others-efforts own-efforts)}
    :mark {:type "line" :strokeWidth 2}
    :encoding {:x {:field "others" :type "quantitative"
                   :title "Others' Total Effort"}
               :y {:field "own" :type "quantitative"
                   :title "Own Optimal Effort"}}}))

;; As others' effort increases, own effort decreases — the classic
;; free-rider effect.

;; ## The Decision Rule
;;
;; Each period, a fraction of agents are activated. Each active agent:
;;
;; 1. **Evaluates staying** at their current firm (computes optimal effort
;;    and resulting utility).
;; 2. **Evaluates switching** to each friend's firm (computes prospective
;;    effort/utility if they joined).
;; 3. **Evaluates starting** a new singleton firm (using cached
;;    singleton effort/utility).
;; 4. **Chooses** the option with highest utility.
;;
;; This local search with limited information (only friends' firms are
;; visible) produces realistic labor dynamics — agents don't have global
;; knowledge of all firms.

;; ## Building the Model with Raster

;; ### Structure-of-Arrays (SoA) Layout
;;
;; Agent and firm state is stored as separate typed arrays, not as
;; objects. This is critical for GPU compilation — GPUs need coalesced
;; memory access patterns.

;; ```clojure
;; ;; Agent state (one float/int array per field):
;; (defvalue AgentSoA [effort    :- (Array float)   ;; current effort
;;                     income    :- (Array float)   ;; current income
;;                     theta     :- (Array float)   ;; consumption preference
;;                     endowment :- (Array float)   ;; time budget
;;                     firm      :- (Array int)     ;; firm assignment
;;                     friends   :- (Array int)     ;; flat-packed neighbor IDs
;;                     cache     :- (Array float)]) ;; singleton effort/utility
;;
;; ;; Firm state:
;; (defvalue FirmSoA [a       :- (Array float)   ;; production param
;;                    b       :- (Array float)   ;; production param
;;                    beta    :- (Array float)   ;; exponent
;;                    te      :- (Array float)   ;; total effort
;;                    output  :- (Array float)   ;; total output
;;                    alive   :- (Array int)     ;; 1=active, 0=dead
;;                    members :- (Array int)     ;; CSR values
;;                    offsets :- (Array int)])    ;; CSR row pointers
;; ```

;; ### Parallel Phases via `par/map!`
;;
;; Each simulation phase is a `deftm` function using Raster's parallel
;; primitives. The compiler transforms these into GPU kernels (OpenCL C
;; → SPIR-V → Level Zero) or optimized CPU loops:

;; ```clojure
;; ;; Production: par/map! over firms → one GPU thread per firm
;; (deftm produce-output-par!
;;   [firms :- FirmSoA, n-firms :- Long] :- FirmSoA
;;   (par/map! (:output firms) i n-firms float
;;     (if (> (aget (:alive firms) i) 0)
;;       (let [a    (aget (:param-a firms) i)
;;             e    (aget (:total-effort firms) i)
;;             beta (aget (:param-beta firms) i)]
;;         (+ (* a e) (* (:param-b firms) (Math/pow e beta))))
;;       (float 0)))
;;   firms)
;; ```
;;
;; The same code runs on CPU (scalar loops with C2 auto-vectorization)
;; or GPU (OpenCL kernels via Level Zero). The compiler handles buffer
;; allocation, memory transfers, and kernel dispatch.

;; ### CSR Membership with Atomic Operations
;;
;; Firm membership is tracked via Compressed Sparse Row (CSR) format.
;; The parallel rebuild uses atomic histogram counting + exclusive
;; prefix scan — standard GPU-friendly algorithms:
;;
;; 1. `zero-firm-sizes!` — par/map! zeros all firm counters
;; 2. `histogram-firms!` — par/map! with `atomic-add!` counts agents per firm
;; 3. `csr-scan-par!` — exclusive prefix sum computes row offsets
;; 4. `fill-members-par!` — par/map! with `atomic-add!` scatters agent IDs

;; ## Running the Simulation

;; Let's run a small simulation on the CPU to see the emergent dynamics.
;; (Large-scale GPU results are shown below as pre-computed data.)

(def config
  (firms/->FirmsConfig
   1000    ;; n-agents
   3       ;; n-friends
   4       ;; n-alt-firms
   0.04    ;; activation-rate
   0       ;; compensation-mode (equal shares)
   500     ;; n-periods
   42      ;; rng-seed
   1000))  ;; max-firms (= n-agents)

;; `init-simulation` returns `[config agents firms rng]` with SoA arrays.
;; `run-period!` mutates state in-place — no allocation per step.

(def sim
  (let [[config agents firms rng] (firms/init-simulation config)]
    (loop [period 0, stats []]
      (if (>= period (.n-periods config))
        {:config config :agents agents :firms firms :stats stats}
        (let [result (firms/run-period! config agents firms rng period)]
          (recur (inc period) (conj stats result)))))))

;; ### Firm Dynamics Over Time

;; The number of active firms evolves as agents start new firms and
;; abandon failing ones:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Active Firms Over Time"
  :data {:values (mapv #(select-keys % [:period :n-alive]) (:stats sim))}
  :mark {:type "line" :strokeWidth 1.5}
  :encoding {:x {:field "period" :type "quantitative" :title "Period"}
             :y {:field "n-alive" :type "quantitative" :title "Active Firms"}}})

;; ### Maximum Firm Size

;; Even with 1000 agents, some firms grow significantly larger than others:

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Maximum Firm Size Over Time"
  :data {:values (mapv #(select-keys % [:period :max-size]) (:stats sim))}
  :mark {:type "line" :strokeWidth 1.5}
  :encoding {:x {:field "period" :type "quantitative" :title "Period"}
             :y {:field "max-size" :type "quantitative" :title "Max Firm Size"}}})

;; ### Mean Effort and Income

(kind/vega-lite
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :width 600 :height 300
  :title "Mean Effort and Income"
  :data {:values (mapcat (fn [s]
                           [{:period (:period s) :value (:mean-effort s) :metric "Effort"}
                            {:period (:period s) :value (:mean-income s) :metric "Income"}])
                         (:stats sim))}
  :mark {:type "line" :strokeWidth 1.5}
  :encoding {:x {:field "period" :type "quantitative" :title "Period"}
             :y {:field "value" :type "quantitative" :title "Value"}
             :color {:field "metric" :type "nominal"}}})

;; ### Firm Size Distribution
;;
;; After the simulation stabilizes, the firm-size distribution should show
;; a heavy tail. This is the signature of Zipf's law — a few large firms
;; and many small ones. Let's look at the final distribution:

(let [{:keys [config agents firms]} sim
      max-f (.max-firms config)
      alive ^ints (:alive firms)
      n-workers ^ints (:n-workers firms)
      firm-sizes (for [i (range max-f)
                       :when (> (clojure.core/aget alive i) 0)]
                   (clojure.core/aget n-workers i))
      size-freq (frequencies firm-sizes)]
  (kind/vega-lite
   {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
    :width 600 :height 350
    :title "Firm Size Distribution (log-log)"
    :data {:values (mapv (fn [[s c]] {:size s :count c})
                         (sort-by key size-freq))}
    :mark {:type "point" :filled true :size 60}
    :encoding {:x {:field "size" :type "quantitative" :title "Firm Size"
                   :scale {:type "log"}}
               :y {:field "count" :type "quantitative" :title "Frequency"
                   :scale {:type "log"}}}}))

;; A straight line on a log-log plot indicates a power-law distribution.
;; With only 1000 agents the tail is noisy, but the shape is visible.
;; With 120M agents on GPU, the distribution closely matches U.S. Census
;; data (see Axtell 2001, 2016).

;; ## Compilation Pipeline
;;
;; Each `par/map!` in the simulation phases flows through Raster's nanopass
;; compiler:
;;
;; 1. **Walker** devirtualizes polymorphic calls to concrete `.invk`
;; 2. **Buffer fusion** converts allocating ops to in-place variants
;; 3. **Backend** dispatches to CPU (scalar loops or SIMD) or GPU
;;    (OpenCL C → SPIR-V → Level Zero kernel)
;; 4. **Hoist + compile** extracts allocations and emits JVM bytecode
;;
;; On GPU, the 16 simulation phases compile to separate OpenCL kernels
;; with SoA buffer decomposition — each field of `AgentSoA`/`FirmSoA`
;; becomes a `__global float*` parameter. The compiler handles
;; `atomic-add!` (CAS loop in OpenCL), exclusive prefix scan (Blelloch
;; algorithm), and work-group sizing automatically.

;; ## Differentiable Variant
;;
;; Raster's AD system extends to the ABM. A differentiable variant
;; (`firms/differentiable.clj`) replaces hard decisions with soft
;; selection (softmax over utilities) and uses the Implicit Function
;; Theorem to differentiate through the Newton-Raphson effort solver.
;; This enables gradient-based calibration of model parameters to match
;; empirical data — a capability unique to Raster's combination of
;; typed dispatch, reverse-mode AD, and GPU compilation.

;; ## GPU Performance
;;
;; Pre-computed results from Intel Arc A770 (Level Zero backend):
;;
;; | Agents | Periods | Time/Period | Speedup vs CPU |
;; |--------|---------|-------------|----------------|
;; | 10K    | 1000    | ~0.8ms      | ~3x            |
;; | 100K   | 1000    | ~2ms        | ~15x           |
;; | 1M     | 500     | ~8ms        | ~40x           |
;; | 10M    | 100     | ~60ms       | ~50x           |
;;
;; The GPU advantage grows with population size because the parallel
;; phases (production, distribution, CSR rebuild) dominate over the
;; serial decision execution phase. At 10M agents, the simulation runs
;; 50x faster than optimized single-threaded CPU code.
;;
;; For comparison, Axtell's C11 implementation uses multi-threaded CPU
;; execution with linked-list data structures. Raster's SoA + GPU
;; approach trades random-access flexibility for throughput — the CSR
;; rebuild is more expensive than linked-list maintenance, but the
;; parallel production and distribution phases are massively faster.

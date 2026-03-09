(ns raster.compiler.passes.parallel.execution-plan
  "Shared execution-plan helpers for compound kernels and SOAC lowering.")

(defn- single-phase-bound?
  [bound block-size]
  (and (number? bound)
       (<= (long bound) (long block-size))))

(defn compound-execution
  "Describe compound-kernel execution strategy chosen during detection."
  [strategy phase-count parallel-bound]
  {:kind :compound
   :strategy strategy
   :parallel-bound parallel-bound
   :phase-count (if (= strategy :local) 1 phase-count)
   :phase-kinds (if (= strategy :local)
                  [:fused]
                  (vec (repeat phase-count :per-phase)))})

(defn reduce-execution
  "Describe whether a reduce lowers to a single phase or two phases."
  [bound grid]
  (let [single? (single-phase-bound? bound (:block-size grid))]
    {:kind :reduce
     :strategy (if single? :single :two-phase)
     :parallel-bound bound
     :phase-count (if single? 1 2)
     :phase-kinds (if single? [:single] [:block-local :cross-block])}))

(defn scan-execution
  "Describe whether a scan lowers to a single phase or three stages."
  [bound grid]
  (let [single? (single-phase-bound? bound (:block-size grid))]
    {:kind :scan
     :strategy (if single? :single :three-stage)
     :parallel-bound bound
     :phase-count (if single? 1 3)
     :phase-kinds (if single? [:single] [:intra-block :block-scan :carry-in])}))
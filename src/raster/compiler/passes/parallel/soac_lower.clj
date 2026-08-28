(ns raster.compiler.passes.parallel.soac-lower
  "SOAC/Screma → SegOp lowering with hardware-aware decisions.

  Transforms high-level SOAC combinators into concrete GPU execution
  plans (SegOps) based on device capabilities:

  Map  → single SegMap with grid-stride loop
  Reduce → single-phase (small n) or two-phase (large n):
           Phase 1: block-local shared-memory tree reduction
           Phase 2: cross-block reduction of partial results
  Scan → single-phase (n ≤ block-size) or three-stage (large n):
         Stage 1: intra-block workgroup scan
         Stage 2: scan of block totals
         Stage 3: carry-in combination (SegMap)"
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.execution-plan :as execution-plan]))

(declare lower-reduce)

(defn typed-reduce-program?
  "Whether a validated one-equation TypedSOAC program is the scalar reduction vertical currently
   accepted by SegRed lowering."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'reduce (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn lower-typed-reduce
  "Lower one functional TypedSOAC reduction to SegRed without inspecting its source spelling.

   TypedSOAC keeps element values and captures as lexical lambda parameters. SegRed's temporary
   scalar-region adapter still spells element reads as `aget`; this conversion is a mechanical
   projection from the validated parameter layout, not a second analysis of the host form."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-reduce-program? program)
      (throw (ex-info "typed SegRed lowering requires one reduce equation"
                      {:reason :typed-soac-reduce-subset :program program})))
    (let [equation (first (soac-dialect/equations program))
          [_ equation-id results operation] equation
          [_ attributes arrays captures lambda] operation
          [_ _ body-results] lambda
          {:keys [accumulators elements capture-parameters]}
          (soac-dialect/parameter-layout equation)
          _ (when-not (and (= 1 (count results))
                           (= 1 (count accumulators))
                           (= 1 (count body-results))
                           (symbol? (first results)))
              (throw (ex-info "initial typed SegRed vertical supports one symbolic result"
                              {:reason :typed-soac-reduce-subset
                               :equation equation-id :results results
                               :accumulators accumulators})))
          index (:index attributes)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array index)])
                     elements arrays))
          step-result (util/subst-syms substitutions (first body-results))
          result (first results)
          accumulator (first accumulators)
          accumulator-dtype (or (first (:dtypes attributes)) dtype :double)
          operator (reduction/scalar
                    {:accumulator accumulator
                     :neutral (first (:identities attributes))
                     :dtype accumulator-dtype
                     :result result
                     :index index
                     :step-result step-result
                     :algebra (or (first (:algebra attributes)) {})
                     :attributes {:source :typed-soac :equation equation-id}})
          node (cond-> (soac/->SoacReduce equation-id result operator []
                                          (:extent attributes) (set arrays) results (set captures))
                 accumulator-dtype (assoc :elem-type accumulator-dtype))]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id)
            (lower-reduce node device-id :dtype accumulator-dtype)))))

;; ================================================================
;; Lowering helpers
;; ================================================================

(defn- soac-outputs*
  [soac]
  (or (soac/soac-outputs soac) (:outputs soac)))

(defn- phase-grid
  [segop-type device-id bound-expr dtype]
  (segop/compute-launch-params segop-type device-id bound-expr :dtype dtype))

(defn- scan-grid
  "A scan needs one workgroup per contiguous block. Unlike map/reduce it cannot cap the grid and
   recover coverage with a grid-stride loop because block prefixes are ordered dataflow."
  [device-id bound-expr dtype]
  (let [planned (phase-grid :scan device-id bound-expr dtype)
        block-size (:block-size planned)]
    (segop/->KernelGrid
     (kernel-launch/ceil-div bound-expr block-size)
     block-size
     (:shared-mem-bytes planned))))

(defn- single-block-grid
  [grid]
  (segop/->KernelGrid 1 (:block-size grid) (:shared-mem-bytes grid)))

(defn- floor-power-of-two
  [n]
  (loop [power 1]
    (if (<= (* 2 power) n)
      (recur (* 2 power))
      power)))

(defn- product-grid
  "Constrain a product reduction's workgroup by both the target's thread limit and its SLM
   budget. Each component has an independent local array, so charging only the primary dtype
   would make mixed products legal on paper while overcommitting local memory at emission."
  [device-id planned reduction]
  (let [descriptor (hardware/descriptor-for device-id)
        bytes-per-lane (reduce + (map (comp dtype/bytes-of :dtype) (:components reduction)))
        slm-budget (long (or (get-in descriptor [:cache :slm])
                             (:shared-memory-per-block descriptor)
                             65536))
        max-by-slm (max 1 (quot slm-budget bytes-per-lane))
        max-workgroup (long (:max-workgroup-size descriptor 1024))
        workgroup-size (floor-power-of-two
                        (max 1 (min (long (:block-size planned))
                                    max-workgroup
                                    max-by-slm)))]
    {:grid (segop/->KernelGrid (:num-blocks planned)
                               workgroup-size
                               (* workgroup-size bytes-per-lane))
     :slm-budget slm-budget
     :bytes-per-lane bytes-per-lane}))

(defn- screma-map-lambda
  [soac]
  (when (soac/screma? soac)
    (:map-lambda soac)))

(defn- reduction-info
  [soac]
  (if (soac/soac-reduce? soac)
    (:reduction soac)
    (first (:reduces soac))))

(defn- scan-op-info
  [soac]
  (if (soac/soac-scan? soac)
    {:acc (:acc soac) :init (:init soac)
     :lambda (:lambda soac) :out (:out soac)}
    (first (:scans soac))))

;; ================================================================
;; Map lowering
;; ================================================================

(defn lower-map
  "Lower a SoacMap/Screma-map to a single SegMap with grid-stride virtualization."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type soac) dtype)
        bound (:bound soac)
        idx (soac/soac-idx soac)
        space (segop/make-seg-space idx bound)
        level (segop/->SegLevel :thread :virtual)
        grid (phase-grid :map device-id bound dtype)
        out-sym (:sym soac)
        cast-fn (:cast-fn soac)]
    [(segop/->SegMap (:id soac) space level
                     (:lambda soac)
                     (:inputs soac) (soac-outputs* soac)
                     (:scalars soac) grid
                     dtype out-sym cast-fn)]))

;; ================================================================
;; Reduce lowering — single or two-phase
;; ================================================================

(defn lower-reduce
  "Lower a SoacReduce/Screma-reduce to SegRed SegOps.

  Decision criteria:
    - n ≤ block-size → single-phase (one block does everything)
    - n > block-size → two-phase:
        Phase 1 (:block-local): each block reduces its chunk via
                shared-memory tree reduction + warp shuffle.
                Output: per-block partial results array.
        Phase 2 (:cross-block): single block reduces partials.

  Returns a vector of SegRed records (1 or 2 elements)."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [reduction (reduction-info soac)
        product? (some? (:combine reduction))
        dtype (or (first (map :dtype (:components reduction))) (:elem-type soac) dtype)
        bound (:bound soac)
        idx (soac/soac-idx soac)
        map-lambda (screma-map-lambda soac)
        space (segop/make-seg-space-nd
               (conj (mapv (fn [[name axis-bound]] {:name name :bound axis-bound})
                           (or (:segment-axes soac) []))
                     {:name idx :bound bound}))
        planned-grid (phase-grid :reduce device-id bound dtype)
        product-grid-info (when product? (product-grid device-id planned-grid reduction))
        grid-1 (or (:grid product-grid-info) planned-grid)
        product-schedule
        (when product?
          (let [workgroup-size (:block-size grid-1)
                candidates (filterv #(<= % workgroup-size) [32 64 128 256 512 1024])]
            (reduction/schedule
             {:strategy :segmented-workgroup-tree
              :workgroup-size workgroup-size
              :stages [:lane-fold :workgroup-tree :segment-store]
              :tuning-space {:workgroup-size candidates
                             :elements-per-lane [:runtime-stride]}
              :numerical-mode (select-keys (:algebra reduction)
                                           [:order :reassociation :overflow])
              :attributes {:scratch :workgroup-local
                           :component-dtypes (reduction/dtypes reduction)
                           :scratch-bytes-per-lane (:bytes-per-lane product-grid-info)
                           :slm-budget (:slm-budget product-grid-info)}})))
        execution (execution-plan/reduce-execution bound grid-1)]
    (if product?
      [(segop/->SegRed (:id soac) space (segop/->SegLevel :block :virtual)
                       reduction map-lambda (:inputs soac)
                       (set (filter symbol? (or (:outputs soac) []))) (:scalars soac)
                       grid-1 :product product-schedule dtype)]
      (case (:strategy execution)
        :single
        [(segop/->SegRed (:id soac)
                         space
                         (segop/->SegLevel :block :none)
                         reduction
                         map-lambda
                         (:inputs soac)
                         (or (soac-outputs* soac) #{(:sym soac)})
                         (:scalars soac)
                         (single-block-grid grid-1)
                         :single nil
                         dtype)]

        :two-phase
        (let [level-1 (segop/->SegLevel :block :virtual)
              phase-1 (segop/->SegRed (:id soac) space level-1
                                      reduction map-lambda
                                      (:inputs soac)
                                      (or (soac-outputs* soac) #{(:sym soac)})
                                      (:scalars soac)
                                      grid-1 :block-local nil
                                      dtype)
              partials-sym (gensym "partials_")
              phase-2-idx (gensym "j_")
              phase-2-space (segop/make-seg-space phase-2-idx (:num-blocks grid-1))
              grid-2 (single-block-grid grid-1)
              level-2 (segop/->SegLevel :block :none)
              phase-2 (segop/->SegRed (+ (:id soac) 1000)
                                      phase-2-space level-2
                                      reduction nil
                                      #{partials-sym} #{(:sym soac)}
                                      #{} grid-2 :cross-block nil
                                      dtype)]
          [phase-1 phase-2])))))

;; ================================================================
;; Scan lowering — single or three-stage
;; ================================================================

(defn lower-scan
  "Lower a SoacScan/Screma-scan to SegScan SegOps.

  Decision criteria:
    - n ≤ block-size → single-phase intra-block workgroup scan
    - n > block-size → three-stage:
        Stage 1 (:intra-block): scan within each block,
                 last element = block total
        Stage 2 (:block-scan): ordered scan of block totals
        Stage 3 (:carry-in): combine carry-in with each element (SegMap)

  Returns a vector of SegScan/SegMap records."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type soac) dtype)
        bound (:bound soac)
        idx (:idx soac)
        raw-scan-op (scan-op-info soac)
        map-lambda (screma-map-lambda soac)
        _ (when map-lambda
            (throw (ex-info "fused scan/map needs an explicit scheduled scan epilogue"
                            {:reason :scan-fused-map-unimplemented
                             :scan-op raw-scan-op :map-lambda map-lambda})))
        scan-facts (scan/certify raw-scan-op dtype)
        scan-op (assoc raw-scan-op :algebra scan-facts)
        space (segop/make-seg-space idx bound)
        grid-1 (scan-grid device-id bound dtype)
        execution (execution-plan/scan-execution bound grid-1)]
    (case (:strategy execution)
      :single
      [(segop/->SegScan (:id soac)
                        space
                        (segop/->SegLevel :block :none)
                        scan-op
                        map-lambda
                        (:inputs soac)
                        (soac-outputs* soac)
                        (:scalars soac)
                        (single-block-grid grid-1)
                        :single
                        dtype)]

      :three-stage
      (let [level-1 (segop/->SegLevel :block :virtual)
            totals-sym (gensym "block_totals_")
            stage-1 (segop/->SegScan (:id soac) space level-1
                                     scan-op map-lambda
                                     (:inputs soac)
                                     (conj (soac-outputs* soac) totals-sym)
                                     (:scalars soac)
                                     grid-1 :intra-block
                                     dtype)
            stage-2-idx (gensym "k_")
            stage-2-space (segop/make-seg-space stage-2-idx (:num-blocks grid-1))
            grid-2 (single-block-grid grid-1)
            level-2 (segop/->SegLevel :block :none)
            stage-2 (segop/->SegScan (+ (:id soac) 2000)
                                     stage-2-space level-2
                                     scan-op nil
                                     #{totals-sym} #{totals-sym}
                                     #{} grid-2 :block-scan
                                     dtype)
            carry-idx (gensym "ci_")
            carry-space (segop/make-seg-space carry-idx bound)
            grid-3 (phase-grid :map device-id bound dtype)
            level-3 (segop/->SegLevel :thread :virtual)
            out-sym (or (:out scan-op) (first (soac-outputs* soac)))
            block-idx-expr (list 'clojure.core/quot carry-idx (:block-size grid-1))
            combine (:combine scan-facts)
            carry-lambda (list 'if (list '> block-idx-expr 0)
                               (list combine
                                     (list 'aget totals-sym
                                           (list 'clojure.core/- block-idx-expr 1))
                                     (list 'aget out-sym carry-idx))
                               (list 'aget out-sym carry-idx))
            stage-3 (segop/->SegMap (+ (:id soac) 3000)
                                    carry-space level-3
                                    carry-lambda
                                    #{out-sym totals-sym}
                                    #{out-sym}
                                    #{} grid-3
                                    dtype out-sym nil)]
        [stage-1 stage-2 stage-3]))))

(defn scan-kernel-graph
  "Turn an already lowered scan into a verified scheduled graph.

   This consumes the exact SegOps returned by `lower-scan`; it must not lower a second time because
   the block-totals buffer identity is generated during decomposition."
  ([soac segops]
   (scan-kernel-graph soac segops {}))
  ([soac segops {:keys [array-types] :or {array-types {}}}]
   (let [external (set/union (or (:inputs soac) #{}) (or (soac-outputs* soac) #{}))
         used (reduce set/union #{}
                      (map #(set/union (or (segop/segop-inputs %) #{})
                                       (or (segop/segop-outputs %) #{}))
                           segops))
         temporary-ids (set/difference used external)
         block-stage (some #(when (= :block-scan (:phase %)) %) segops)
         temporary-elements (when block-stage
                              (:bound (segop/seg-space-reduced-dim (:space block-stage))))
         dtype (or (:elem-type soac) (:dtype (first segops)) :double)
         temporaries (into {}
                           (map (fn [id]
                                  [id {:dtype dtype :elements temporary-elements
                                       :memory-space :device}]))
                           temporary-ids)
         buffer-specs (into {}
                            (map (fn [id]
                                   [id {:dtype (or (get array-types id)
                                                   (get array-types (symbol (name id)))
                                                   dtype)
                                        :elements (:bound soac)}]))
                            external)]
     (kernel-graph/from-segops
      segops
      {:inputs (or (:inputs soac) #{})
       :outputs (or (soac-outputs* soac) #{})
       :temporaries temporaries
       :buffer-specs buffer-specs
       :dtype dtype
       :effects {:memory-order :dependency-ordered}
       :provenance {:dialect :segop :algorithm :scan :soac-id (:id soac)}
       :attributes {:strategy (if (= 1 (count segops)) :single :three-stage)
                    :scan-algebra (get-in (first segops) [:scan-op :algebra])}}))))

(defn scan-soac?
  "True when a SOAC/Screma node selects scan decomposition."
  [node]
  (or (soac/soac-scan? node)
      (and (soac/screma? node) (seq (:scans node)) (empty? (:reduces node)))))

;; ================================================================
;; Unified lowering dispatch
;; ================================================================

(defn lower-soac
  "Lower a SOAC/Screma node to one or more SegOps.
  Returns a vector of SegOp records.

  Dispatches on SOAC type:
    SoacMap/Screma-pure-map → [SegMap]
    SoacReduce/Screma-reduce → [SegRed SegRed] (two-phase)
    SoacScan/Screma-scan → [SegScan SegScan SegMap] (three-stage)"
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type soac) dtype)]
    (cond
      (nil? soac)
      (throw (ex-info "Cannot lower nil: the preceding conversion produced no SOAC node"
                      {:reason :no-soac-node :target-dialect :segop}))

      ;; Screma dispatch based on contents
      (soac/screma? soac)
      (cond
        (and (empty? (:scans soac)) (empty? (:reduces soac)) (:map-lambda soac))
        (lower-map soac device-id :dtype dtype)

        (and (empty? (:scans soac)) (seq (:reduces soac)))
        (lower-reduce soac device-id :dtype dtype)

        (and (seq (:scans soac)) (empty? (:reduces soac)))
        (lower-scan soac device-id :dtype dtype)

        :else
        (throw (ex-info "Complex Screma not yet supported for lowering" {:soac soac})))

      ;; Direct SOAC dispatch
      ;; a contraction: record the facts for this target; the backend routes from them
      (soac/contract? soac)
      [(segop/->SegContract (:id soac) (:facts soac) dtype device-id)]

      (soac/soac-map? soac)
      (lower-map soac device-id :dtype dtype)

      (soac/soac-reduce? soac)
      (lower-reduce soac device-id :dtype dtype)

      (soac/soac-scan? soac)
      (lower-scan soac device-id :dtype dtype)

      :else
      (throw (ex-info "Unsupported SOAC type for lowering" {:soac soac})))))

(defn lower-soac-nodes
  "Lower all SOAC nodes from a fusion graph to SegOps.
  Returns a map from original SOAC id to [SegOp ...] vector."
  [nodes-map device-id & {:keys [dtype] :or {dtype :double}}]
  (reduce-kv
   (fn [acc id node]
     (if (or (soac/soac? node) (soac/contract? node))
       (assoc acc id (lower-soac node device-id :dtype dtype))
       acc))
   {} nodes-map))

(ns raster.compiler.passes.parallel.soac-lower
  "SOAC/Screma → SegOp lowering with hardware-aware decisions.

  Transforms high-level SOAC combinators into concrete GPU execution
  plans (SegOps) based on device capabilities:

  Map  → single SegMap with grid-stride loop
  Reduce → single-phase (small n) or two-phase (large n):
           Phase 1: block-local shared-memory tree reduction
           Phase 2: cross-block reduction of partial results
  Scan → single-phase (n ≤ block-size) or three-stage (large n):
         Stage 1: intra-block Blelloch scan
         Stage 2: scan of block totals
         Stage 3: carry-in addition (SegMap)"
  (:require [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.execution-plan :as execution-plan]))

;; ================================================================
;; Lowering helpers
;; ================================================================

(defn- soac-outputs*
  [soac]
  (or (soac/soac-outputs soac) (:outputs soac)))

(defn- phase-grid
  [segop-type device-id bound-expr dtype]
  (segop/compute-launch-params segop-type device-id bound-expr :dtype dtype))

(defn- single-block-grid
  [grid]
  (segop/->KernelGrid 1 (:block-size grid) (:shared-mem-bytes grid)))

(defn- screma-map-lambda
  [soac]
  (when (instance? raster.compiler.ir.soac.Screma soac)
    (:map-lambda soac)))

(defn- reduce-op-info
  [soac]
  (if (instance? raster.compiler.ir.soac.SoacReduce soac)
    {:acc (:acc soac) :init (:init soac) :lambda (:lambda soac)}
    (first (:reduces soac))))

(defn- scan-op-info
  [soac]
  (if (instance? raster.compiler.ir.soac.SoacScan soac)
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
        idx (:idx soac)
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
  (let [dtype (or (:elem-type soac) dtype)
        bound (:bound soac)
        idx (:idx soac)
        reduce-op (reduce-op-info soac)
        map-lambda (screma-map-lambda soac)
        space (segop/make-seg-space idx bound)
        grid-1 (phase-grid :reduce device-id bound dtype)
        execution (execution-plan/reduce-execution bound grid-1)]
    (case (:strategy execution)
      :single
      [(segop/->SegRed (:id soac)
                       space
                       (segop/->SegLevel :block :none)
                       reduce-op
                       map-lambda
                       (:inputs soac)
                       (or (soac-outputs* soac) #{(:sym soac)})
                       (:scalars soac)
                       (single-block-grid grid-1)
                       :single
                       dtype)]

      :two-phase
      (let [level-1 (segop/->SegLevel :block :virtual)
            phase-1 (segop/->SegRed (:id soac) space level-1
                                    reduce-op map-lambda
                                    (:inputs soac)
                                    (or (soac-outputs* soac) #{(:sym soac)})
                                    (:scalars soac)
                                    grid-1 :block-local
                                    dtype)
            partials-sym (gensym "partials_")
            phase-2-idx (gensym "j_")
            phase-2-space (segop/make-seg-space phase-2-idx (:num-blocks grid-1))
            grid-2 (single-block-grid grid-1)
            level-2 (segop/->SegLevel :block :none)
            phase-2 (segop/->SegRed (+ (:id soac) 1000)
                                    phase-2-space level-2
                                    reduce-op nil
                                    #{partials-sym} #{(:sym soac)}
                                    #{} grid-2 :cross-block
                                    dtype)]
        [phase-1 phase-2]))))

;; ================================================================
;; Scan lowering — single or three-stage
;; ================================================================

(defn lower-scan
  "Lower a SoacScan/Screma-scan to SegScan SegOps.

  Decision criteria:
    - n ≤ block-size → single-phase intra-block Blelloch scan
    - n > block-size → three-stage:
        Stage 1 (:intra-block): Blelloch scan within each block,
                 last element = block total
        Stage 2 (:block-scan): scan of block totals (recursive)
        Stage 3 (:carry-in): add carry-in to each element (SegMap)

  Returns a vector of SegScan/SegMap records."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type soac) dtype)
        bound (:bound soac)
        idx (:idx soac)
        scan-op (scan-op-info soac)
        map-lambda (screma-map-lambda soac)
        space (segop/make-seg-space idx bound)
        grid-1 (phase-grid :scan device-id bound dtype)
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
            stage-1 (segop/->SegScan (:id soac) space level-1
                                     scan-op map-lambda
                                     (:inputs soac)
                                     (soac-outputs* soac)
                                     (:scalars soac)
                                     grid-1 :intra-block
                                     dtype)
            totals-sym (gensym "block_totals_")
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
            carry-lambda (list 'if (list '> block-idx-expr 0)
                               (list '+ (list 'aget out-sym carry-idx)
                                     (list 'aget totals-sym
                                           (list 'clojure.core/- block-idx-expr 1)))
                               (list 'aget out-sym carry-idx))
            stage-3 (segop/->SegMap (+ (:id soac) 3000)
                                    carry-space level-3
                                    carry-lambda
                                    #{out-sym totals-sym}
                                    #{out-sym}
                                    #{} grid-3
                                    dtype out-sym nil)]
        [stage-1 stage-2 stage-3]))))

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
      ;; Screma dispatch based on contents
      (instance? raster.compiler.ir.soac.Screma soac)
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
      (instance? raster.compiler.ir.soac.SoacMap soac)
      (lower-map soac device-id :dtype dtype)

      (instance? raster.compiler.ir.soac.SoacReduce soac)
      (lower-reduce soac device-id :dtype dtype)

      (instance? raster.compiler.ir.soac.SoacScan soac)
      (lower-scan soac device-id :dtype dtype)

      :else
      (throw (ex-info "Unsupported SOAC type for lowering" {:soac soac})))))

(defn lower-soac-nodes
  "Lower all SOAC nodes from a fusion graph to SegOps.
  Returns a map from original SOAC id to [SegOp ...] vector."
  [nodes-map device-id & {:keys [dtype] :or {dtype :double}}]
  (reduce-kv
   (fn [acc id node]
     (if (soac/soac? node)
       (assoc acc id (lower-soac node device-id :dtype dtype))
       acc))
   {} nodes-map))

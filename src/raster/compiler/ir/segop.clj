(ns raster.compiler.ir.segop
  "SegOp GPU IR — hardware-aware kernel representation.

  Lowered from SOAC/Screma nodes, SegOps encode the concrete
  GPU execution strategy: thread/block mapping, multi-phase
  decomposition (for reductions and scans), and launch parameters.

  Follows Futhark's SegOp hierarchy:
    SegMap  — element-wise parallel map (grid-stride loop)
    SegRed  — parallel reduction (single or two-phase)
    SegScan — parallel prefix scan (single or three-stage)

  Each SegOp carries a KernelGrid with pre-computed launch config
  based on raster.runtime.hardware device properties."
  (:require [raster.runtime.hardware :as hw]
            [raster.compiler.core.hardware :as chw]))

;; ================================================================
;; Segmented space and grid config
;; ================================================================

(defrecord SegSpace
           [flat-idx    ;; symbol — linear thread index
            dims])      ;; [{:name sym :bound expr} ...]

(defrecord KernelGrid
           [num-blocks  ;; expr — grid dimension (number of blocks)
            block-size  ;; int — threads per block
            shared-mem-bytes]) ;; int — dynamic shared memory per block

(defrecord SegLevel
           [level       ;; :thread | :block
            virt])      ;; :none | :virtual (grid-stride virtualization)

;; ================================================================
;; SegOp records
;; ================================================================

(defrecord SegMap
           [id          ;; int — from original SOAC id
            space       ;; SegSpace
            level       ;; SegLevel
            lambda      ;; expr — kernel body S-expression
            inputs      ;; #{sym} — array symbols read
            outputs     ;; #{sym} — array symbols written
            scalars     ;; #{sym} — scalar parameters
            grid        ;; KernelGrid
            dtype        ;; :double | :float — element type
            out-sym      ;; sym — output array symbol (for aset target)
            cast-fn])    ;; sym | nil — cast function for aset (e.g. 'float)

(defrecord SegRed
           [id          ;; int
            space       ;; SegSpace
            level       ;; SegLevel
            reduce-op   ;; {:acc sym :init expr :lambda expr}
            lambda      ;; expr — map body before reduction (or nil for direct)
            inputs      ;; #{sym}
            outputs     ;; #{sym}
            scalars     ;; #{sym}
            grid        ;; KernelGrid
            phase       ;; :single | :block-local | :cross-block
            dtype])      ;; :double | :float — element type

(defrecord SegScan
           [id          ;; int
            space       ;; SegSpace
            level       ;; SegLevel
            scan-op     ;; {:acc sym :init expr :lambda expr :out sym}
            lambda      ;; expr — map body before scan (or nil)
            inputs      ;; #{sym}
            outputs     ;; #{sym}
            scalars     ;; #{sym}
            grid        ;; KernelGrid
            phase       ;; :single | :intra-block | :block-scan | :carry-in
            dtype])      ;; :double | :float — element type

;; ================================================================
;; Predicates
;; ================================================================

(defn segop?
  "Check if a value is a SegOp record."
  [x]
  (or (instance? SegMap x)
      (instance? SegRed x)
      (instance? SegScan x)))

(defn segop-grid
  "Get the KernelGrid from a SegOp."
  [segop]
  (:grid segop))

(defn segop-inputs
  "Get the input array symbols from a SegOp."
  [segop]
  (:inputs segop))

(defn segop-outputs
  "Get the output array symbols from a SegOp."
  [segop]
  (:outputs segop))

;; ================================================================
;; Launch parameter computation
;; ================================================================

(defn compute-launch-params
  "Compute optimal KernelGrid for a SegOp given device and problem size.

  Uses raster.runtime.hardware for device-specific tuning:
    - SM count for grid size
    - Warp size for block size rounding
    - Shared memory limits

  Returns a KernelGrid. For reduction/scan, shared-mem is
  block-size * element-size bytes."
  [segop-type device-id n-expr & {:keys [dtype] :or {dtype :double}}]
  (hw/init!)
  (let [desc (chw/descriptor-for device-id)
        config (chw/launch-config desc 65536 :reduction? (not= segop-type :map))
        block-size (or (:block-size config) 256)
        elem-size (case dtype :double 8 :float 4 :long 8 :int 4 8)
        shared-mem (case segop-type
                     :map 0
                     :reduce (* block-size elem-size)
                     :scan (* block-size elem-size)
                     0)
        ;; Grid cap for the grid-stride loop: enough workgroups to fill the machine a few waves
        ;; over — derived from the descriptor's machine width (EU/SM count), not a CUDA sm-count
        ;; literal that read 80 for an Intel Level-Zero device.
        max-blocks (* (chw/fill-workgroups desc block-size) 4)
        grid-expr (list 'min max-blocks
                        (list 'int (list 'Math/ceil
                                         (list '/ (list 'double n-expr) (double block-size)))))]
    (->KernelGrid grid-expr block-size shared-mem)))

(defn make-seg-space-nd
  "Create an N-D SegSpace from an ordered [{:name sym :bound expr} ...] vector,
  OUTER→INNER (Futhark convention). Segmentation is positional: for a segmented
  reduce/map the INNERMOST dim is the reduced/mapped axis and all outer dims are
  segments. A 1-D space is the degenerate single-dim case."
  [dims-vec]
  (->SegSpace (gensym "tid_") (vec dims-vec)))

(defn make-seg-space
  "Create a SegSpace for a 1D problem."
  [idx-sym bound-expr]
  (make-seg-space-nd [{:name idx-sym :bound bound-expr}]))

;; --- Positional dim accessors (single source of the Futhark convention) ---
(defn seg-space-dims [space] (:dims space))

(defn seg-space-1d?
  "True when the space has a single dimension (the current/degenerate case)."
  [space]
  (= 1 (count (:dims space))))

(defn seg-space-reduced-dim
  "The innermost (reduced/mapped) dim — {:name :bound}. This is what the 1-D
  consumers historically read as `(-> :space :dims first)`."
  [space]
  (last (:dims space)))

(defn seg-space-segment-dims
  "The outer (segment) dims — all but the innermost. Empty vector for a 1-D space."
  [space]
  (vec (butlast (:dims space))))

(defn seg-space-num-segments-expr
  "Expr for the number of segments = product of the outer dim bounds; 1 (int) for
  a 1-D space."
  [space]
  (let [segs (seg-space-segment-dims space)]
    (if (empty? segs)
      1
      (reduce (fn [acc d] (list '* acc (:bound d))) (:bound (first segs)) (rest segs)))))

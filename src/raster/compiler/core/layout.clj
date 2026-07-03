(ns raster.compiler.core.layout
  "Layout-as-schedule — the weight layout a SIMD int-dot kernel requires, DERIVED
  from (instruction, dtype, hardware) rather than hand-written.

  Triton is the reference (studied /home/.../triton): a tensor layout is a
  type-encoding attribute; the dot/MMA instruction is chosen and the OPERAND layout
  is derived from it (opIdx + parent + dtype->kWidth); a `convert_layout` op
  materializes any layout change. raster has had ONE layout — the Q4 stream-gemv
  interleave — hand-written twice over: `repack-stream` (scatter) and
  `emit-qmatmul-stream` (the matching gather), agreeing only by convention, with the
  tile width `NC=8` hard-coded in both. This namespace makes the layout an explicit
  DESCRIPTOR both sides read, with `NC` derived from the hardware (= the f32 lane
  count: AVX2 8, AVX-512 16, NEON 4), so the convention can no longer drift.

  Scope today: the `:q4-col-interleaved` layout the stream-gemv dpbusd kernel needs —
  output column i in accumulator lane i (no permute). The descriptor + index function
  generalize the hand interleave (NC, k-group, nibble lo/hi split) so the same code
  serves AVX2/AVX-512/NEON. Future layouts (row-major tile-gemm, GPU dp4a) add a
  `:kind`; a cost-priced `convert_layout` op (Triton's RemoveLayoutConversions) is the
  later generalization once there is more than one layout to choose between."
  (:require [raster.compiler.core.hardware :as hw]))

;; ---------------------------------------------------------------------------
;; Layout descriptor — derived from (instruction, dtype, hardware)
;; ---------------------------------------------------------------------------

(def q4-format
  "The layout-relevant fields of the Q4_0 quant format (a subset of the full format
   descriptor in cpu.quant): 4-bit weights, 32-element blocks, two columns per byte via
   lo/hi nibble. Other formats (8-bit direct, 5-bit, ...) are sibling maps."
  {:block 32 :weight-bits 4 :pack :nibble-interleaved})

(defn quant-stream-layout
  "The weight layout a quantized stream-gemv dpbusd kernel requires, DERIVED from the
  hardware descriptor + the quant FORMAT (block, weight-bits, pack) — not hard-coded to
  Q4. The dpbusd tile puts NC output columns in the lanes and reduces over the input
  axis, so the weights are column-interleaved by NC. The PACK parameterizes precision:
  :nibble-interleaved (4-bit) packs two columns per byte (columns [0,NC/2) -> low nibble,
  [NC/2,NC) -> high nibble), so the structure (half, bytes-per-igroup) follows from the
  bit width. NC (= f32 lanes) is the column tile; k-group (=4) is the dpbusd int8 4-way
  lane-step. 8-bit/5-bit add a :pack branch.

  Returns {:kind :tile :block :k-group :bits :pack :half :bytes-per-igroup :igroups}."
  ([desc] (quant-stream-layout desc q4-format))
  ([desc {:keys [block weight-bits pack] :or {block 32 weight-bits 4 pack :nibble-interleaved}}]
   (assert (contains? #{:nibble-interleaved :byte-direct} pack)
           (str "quant-stream-layout: pack " pack " not implemented (5-bit, ...)"))
   (let [nc (hw/column-tile desc)            ;; output-column tile = f32 lanes
         k-group 4                           ;; dpbusd int8 4-way lane-step
         nibble? (= pack :nibble-interleaved)
         half (when nibble? (quot nc 2))]    ;; columns sharing a byte via lo/hi nibble (4-bit)
     {:kind            (if nibble? :q4-col-interleaved :q8-col-interleaved)
      :tile            nc
      :block           block
      :k-group         k-group
      :bits            weight-bits
      :pack            pack
      :half            half
      :igroups         (quot block k-group)  ;; dpbusd steps per block (8 for block 32)
      ;; bytes per input-group: nibble packs 2 columns/byte (half*k-group);
      ;; byte-direct is one column-element per byte (nc*k-group).
      :bytes-per-igroup (if nibble? (* half k-group) (* nc k-group))})))

;; ---------------------------------------------------------------------------
;; The layout function — physical byte index for a (group, block, igroup, byte)
;; ---------------------------------------------------------------------------

(defn wqi-len  [{:keys [tile igroups bytes-per-igroup]} out in]
  (* (quot out tile) (quot in 32) igroups bytes-per-igroup))
(defn wsi-len  [{:keys [tile]} out in] (* (quot out tile) (quot in 32) tile))

;; ---------------------------------------------------------------------------
;; Repack — generate the interleaved layout from the descriptor (was the
;; hand-written repack-stream; now the single source the kernel's gather mirrors).
;; ---------------------------------------------------------------------------

(defn repack
  "Row-major quantized weight {wq, ws[out*nb]} -> the descriptor's interleaved layout.
  Output column i lands in accumulator lane i (no shuffles). Generated from the
  layout descriptor; byte-identical to the AVX2 (NC=8) hand repack-stream for Q4, and
  the generalization to NC=16/4 and :byte-direct (8-bit) follows from the descriptor.
  `out` must be a multiple of :tile (caller pads). One-time weight-load cost.
  Source wq: :nibble-interleaved = Q4_0 rows (byte[out*in/2], lo=w[k]/hi=w[k+16]);
  :byte-direct = one unsigned byte per weight (byte[out*in])."
  [layout ^bytes wq ^floats ws out in]
  (let [{:keys [tile block k-group half igroups bytes-per-igroup pack]} layout
        nb (quot (int in) (int block))
        ng (quot (int out) (int tile))
        halfw (quot (int in) 2)
        wqi (byte-array (wqi-len layout out in))
        wsi (float-array (wsi-len layout out in))
        ;; nibble of (output row, block b, element e in 0..block-1) from Q4_0 source
        src-nib (fn [row b e]
                  (let [bv (bit-and (aget wq (+ (* (int row) halfw) (* (int b) (quot block 2))
                                                (if (< e 16) e (- e 16)))) 0xFF)]
                    (if (< e 16) (bit-and bv 0xF) (bit-shift-right bv 4))))]
    (dotimes [gi ng]
      (dotimes [b nb]
        (dotimes [ig igroups]
          (dotimes [j bytes-per-igroup]
            (let [idx (+ (* gi nb bytes-per-igroup igroups) (* b bytes-per-igroup igroups)
                         (* ig bytes-per-igroup) j)
                  e (+ (* ig k-group) (mod j k-group))]
              (if (= pack :byte-direct)
                ;; byte j = column (gi*tile + j/k-group), element e — direct copy
                (let [row (+ (* gi tile) (quot j k-group))]
                  (aset wqi idx (aget wq (+ (* row (int in)) (* b block) e))))
                ;; byte j packs column (gi*tile + j/k-group) low nibble and
                ;; column (that + half) high nibble, for element e.
                (let [base (+ (* gi tile) (quot j k-group))
                      nlo (src-nib base b e)
                      nhi (src-nib (+ base half) b e)]
                  (aset wqi idx (unchecked-byte (bit-or nlo (bit-shift-left nhi 4)))))))))
        (dotimes [r tile]
          (aset wsi (+ (* gi nb tile) (* b tile) r)
                (aget ws (+ (* (+ (* gi tile) r) nb) b))))))
    {:wqi wqi :wsi wsi}))

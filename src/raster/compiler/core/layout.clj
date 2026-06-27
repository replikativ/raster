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

(defn q4-stream-layout
  "The weight layout the Q4_0 stream-gemv dpbusd kernel requires, derived from the
  hardware descriptor. The dpbusd tile puts NC output columns in the lanes and
  reduces over the input axis; the weights must therefore be column-interleaved by
  NC with the Q4 nibbles split lo/hi (columns [0,NC/2) -> low nibble, [NC/2,NC) ->
  high nibble) and grouped k-group=4 elements per dpbusd step.

  Returns {:kind :tile :block :k-group :bits :half :bytes-per-igroup :igroups}.
  `out` must be a multiple of `:tile` (caller pads)."
  ([] (q4-stream-layout hw/*descriptor*))
  ([desc]
   (let [nc (hw/column-tile desc)            ;; output-column tile = f32 lanes
         block 32                            ;; Q4_0 block size
         k-group 4]                          ;; elements per dpbusd lane-step
     {:kind            :q4-col-interleaved
      :tile            nc
      :block           block
      :k-group         k-group
      :bits            4
      :half            (quot nc 2)           ;; columns sharing a byte via lo/hi nibble
      :igroups         (quot block k-group)  ;; dpbusd steps per block (8 for block 32)
      :bytes-per-igroup (* (quot nc 2) k-group)})))

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
  "Row-major Q4_0 {wq[out*in/2], ws[out*nb]} -> the descriptor's interleaved layout.
  Output column i lands in accumulator lane i (no shuffles). Generated from the
  layout descriptor; byte-identical to the AVX2 (NC=8) hand repack-stream, and the
  generalization to NC=16/4 follows from the descriptor. `out` must be a multiple of
  :tile (caller pads). One-time weight-load cost."
  [layout ^bytes wq ^floats ws out in]
  (let [{:keys [tile block k-group half igroups bytes-per-igroup]} layout
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
            ;; byte j packs column (gi*tile + j/k-group) low nibble and
            ;; column (that + half) high nibble, for element ig*k-group + j%k-group.
            (let [base (+ (* gi tile) (quot j k-group))
                  e (+ (* ig k-group) (mod j k-group))
                  nlo (src-nib base b e)
                  nhi (src-nib (+ base half) b e)]
              (aset wqi (+ (* gi nb bytes-per-igroup igroups) (* b bytes-per-igroup igroups)
                           (* ig bytes-per-igroup) j)
                    (unchecked-byte (bit-or nlo (bit-shift-left nhi 4)))))))
        (dotimes [r tile]
          (aset wsi (+ (* gi nb tile) (* b tile) r)
                (aget ws (+ (* (+ (* gi tile) r) nb) b))))))
    {:wqi wqi :wsi wsi}))

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

;; ===========================================================================
;; General layout facet (Triton-style, REDUCED model: stride + permutation +
;; broadcast-mask + a named tile — NO GF(2) linear-layout algebra). A layout is
;; plain DATA (a map with a :kind), consistent with schedule-as-data. The GF(2)
;; core is deferred to the single boundary that needs it: a swizzled shared-memory
;; layout (see .internal/layout_inference_design.md §"the one tension").
;; ===========================================================================

(defn dtype-bits [dtype]
  (case dtype
    (:half :f16 :bf16 :float16) 16
    (:float :f32) 32
    (:double :f64) 64
    (:byte :int8 :i8 :uint8) 8
    (:int :i32 :int32) 32
    32))

;; --- dense strided layouts (row/col-major, transpose views) ---

(defn row-major
  "Contiguous row-major layout of `shape` (identity perm — the default everywhere today)."
  ([shape] (row-major shape :float))
  ([shape dtype] {:kind :row-major :rank (count shape) :shape (vec shape)
                  :perm (vec (range (count shape))) :dtype dtype}))

(defn col-major [shape dtype]
  (assoc (row-major shape dtype) :kind :col-major :perm (vec (reverse (range (count shape))))))

(defn transpose-layout
  "A layout viewing `l` transposed — flip the physical major order (:perm). The ONLY thing a
   transpose changes; the data does not move (that's what makes a transpose potentially free)."
  [l] (update l :perm (comp vec reverse)))

(defn row-major-unit?
  "Is this the trivial default — identity perm, no explicit strides, dense? The emitter
   short-circuits on this to the exact pre-layout index expression (byte-identity)."
  [l]
  (and (map? l)
       (contains? #{:row-major nil} (:kind l))
       (nil? (:strides l))
       (let [p (:perm l)] (or (nil? p) (= p (vec (range (count p))))))))

(defn resolve-strides
  "Per-LOGICAL-axis element strides from (shape, perm), row-major within the permuted order:
   the innermost physical axis (perm[n-1]) has stride 1, perm[k] has the product of physical
   extents after k. Numeric when shape is numeric, else an S-expression (clojure.core * — index
   arithmetic stays clojure.core per project convention). An explicit :strides overrides."
  [{:keys [shape perm strides]}]
  (or strides
      (let [n     (count shape)
            perm  (or perm (vec (range n)))
            pshp  (mapv shape perm)
            mul   (fn [a b] (cond (= a 1) b (= b 1) a
                                  (and (number? a) (number? b)) (* a b) :else (list '* a b)))
            pstr  (loop [k (dec n) acc 1 out (vec (repeat n 1))]
                    (if (< k 0) out
                        (recur (dec k) (mul acc (nth pshp k)) (assoc out k acc))))]
        (reduce (fn [m k] (assoc m (nth perm k) (nth pstr k)))
                (vec (repeat n 1)) (range n)))))

(defn layout->offset
  "Physical element offset of logical `coords` under `layout` — Σ coord_i · stride_i, as a numeric
   value or an S-expression. Zero-stride (broadcast) axes drop out; unit strides are bare."
  [layout coords]
  (let [st (resolve-strides layout)
        terms (keep (fn [[c s]]
                      (cond (= s 0) nil (= s 1) c
                            (and (number? c) (number? s)) (* c s)
                            :else (list '* c s)))
                    (map vector coords st))]
    (case (count terms)
      0 0
      1 (first terms)
      (reduce (fn [a b] (if (and (number? a) (number? b)) (+ a b) (list '+ a b))) terms))))

(defn offset->coords
  "Inverse of layout->offset for a DENSE strided layout (numeric strides) — recover logical coords
   by divmod in descending-stride order. A bijection only for dense row/col-major/transpose layouts
   (the cases where delinearizing a flat index is meaningful)."
  [layout offset]
  (let [st (resolve-strides layout)
        order (sort-by #(- (long (nth st %))) (range (count st)))]
    (:coords (reduce (fn [{:keys [rem coords]} ax]
                       (let [s (long (nth st ax))]
                         {:rem (mod rem s) :coords (assoc coords ax (quot rem s))}))
                     {:rem offset :coords (vec (repeat (count st) 0))} order))))

;; --- tensor-core / coalesced layouts (carry tile + thread-map, not dense strides) ---

(defn blocked-coalesced
  "A coalesced memory layout: consecutive lanes read consecutive elements along the fastest axis,
   size-per-thread clamped to a 128-bit vector load."
  [shape dtype desc]
  {:kind :blocked :rank (count shape) :shape (vec shape)
   :perm (vec (range (count shape))) :dtype dtype
   :thread-map {:order (vec (range (count shape)))
                :size-per-thread (max 1 (quot 128 (dtype-bits dtype)))
                :threads-per-warp (long (:subgroup-size desc 16))
                :broadcast #{}}})

(defn mma-frag
  "The MMA/matrix result-fragment layout for a hardware `matrix` unit ({:family :m :n :k :subgroup})."
  [matrix dtype]
  {:kind :mma-frag :dtype dtype :matrix matrix
   :tile {:m (:m matrix) :n (:n matrix) :k (:k matrix) :subgroup (:subgroup matrix)}})

(defn dot-operand
  "The MMA INPUT (A/B) layout — derived from the MMA fragment it feeds (Triton opIdx + parent +
   kWidth). op-idx 0 = A, 1 = B."
  [op-idx parent k-width dtype]
  {:kind :dot-operand :op-idx op-idx :parent parent :k-width k-width :dtype dtype})

(defn derive-layout
  "The layout an operand of `instr` (a semantic op keyword) requires — enumerated dispatch from
   (instruction, dtype, hardware descriptor), NOT a search. opts: {:op-idx :shape}."
  [instr dtype desc & {:keys [op-idx shape] :or {op-idx 0}}]
  (case instr
    :elementwise    (row-major shape dtype)
    :coalesced-load (blocked-coalesced shape dtype desc)
    :mma-acc        (mma-frag (:matrix desc) dtype)
    :dot-operand    (dot-operand op-idx (mma-frag (:matrix desc) dtype)
                                 (max 1 (quot 32 (dtype-bits dtype))) dtype)
    :transpose      (transpose-layout (row-major shape dtype))
    (row-major shape dtype)))

(defn quant-layout->facet
  "Lift the CPU quant layout (quant-stream-layout) into the general facet shape, so the Q4 path and
   the general path are ONE datatype (proves the generalization; the index-fn stays `repack`)."
  [q]
  (assoc q :general :quant :index-fn :repack))

;; --- convert cost model (Triton's trichotomy in the reduced model) ---

(defn- perm-crosses-coalescing-axis?
  "Does converting src→dst change which logical axis is innermost (the coalescing axis)? If so the
   data must cross lanes/warps; if only the within-lane order changes it's a register shuffle."
  [src dst]
  (let [ps (or (:perm src) (vec (range (:rank src 1))))
        pd (or (:perm dst) (vec (range (:rank dst 1))))]
    (not= (last ps) (last pd))))

(defn register-local?
  "A conversion that only reorders elements WITHIN a lane's registers — free (no shuffle)."
  [src dst]
  (and (= (:kind src) (:kind dst))
       (= (:perm src) (:perm dst))
       (= (:thread-map src) (:thread-map dst))))

(defn lane-local?
  "A conversion resolvable by a warp/subgroup shuffle (crosses lanes but not warps) — cheap. Reduced
   proxy: the coalescing axis is unchanged but the intra-tile thread-map differs."
  [src dst]
  (and (not (register-local? src dst))
       (not (perm-crosses-coalescing-axis? src dst))))

(defn convert-cost
  "Cost class of a layout conversion (the highest-leverage idea from Triton, reduced): register-local
   → FREE register shuffle; lane-local → warp shuffle; else → shared-memory round-trip (priced in
   bytes, and re-entering schedule/feasible? when the register-fragment footprint changes)."
  [src dst _desc]
  (cond
    (register-local? src dst) {:class :register-shuffle :bytes 0}
    (lane-local? src dst)     {:class :warp-shuffle :bytes 0}
    :else                     (let [n (reduce (fn [a b] (if (and (number? a) (number? b)) (* a b) (list '* a b)))
                                              1 (:shape dst (:shape src [1])))
                                    b (dtype-bits (:dtype dst (:dtype src :float)))]
                                {:class :shared-roundtrip
                                 :bytes (if (number? n) (* n (quot b 8)) (list '* n (quot b 8)))})))

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

(ns raster.quant.ggml
  "ggml-faithful block formats: quantizers that reproduce llama.cpp's bytes.

  A format here is ggml's own block layout (`block_q8_0`, `block_q5_0`, ...),
  held as a row-major byte array of blocks, together with the quantizer and
  dequantizer that ggml's reference code defines. The quantizers follow
  `quantize_row_*_ref` in ggml/src/ggml-quants.c operation by operation,
  because `llama-quantize` uses exactly those when no importance matrix is
  given; `test/resources/ggml_oracle` holds llama.cpp's output for the same
  inputs, and the tests require byte equality.

  Float semantics are C's single precision. A float product or quotient is
  computed in double and rounded once to float, which is the correctly rounded
  single-precision result. FP16 conversion rounds to nearest even, as F16C and
  ggml's scalar fallback do. `roundf` rounds halves away from zero, a C cast
  truncates toward zero, and ggml's `nearest_int` rounds halves to even; each
  is spelled out where ggml uses it.

  Kernel layouts are separate: a GPU kernel consumes a lossless repacking of
  these blocks, so the bytes stay the single source of truth, the same one a
  GGUF file carries."
  (:import [java.nio ByteBuffer ByteOrder]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Scalar semantics
;; ---------------------------------------------------------------------------

(defn fp32->fp16-bits
  "GGML_FP32_TO_FP16: round to nearest even, as an unsigned 16-bit pattern."
  ^long [^double x]
  (bit-and 0xFFFF (long (Float/floatToFloat16 (float x)))))

(defn fp16-bits->fp32
  "GGML_FP16_TO_FP32."
  ^double [^long bits]
  (double (Float/float16ToFloat (unchecked-short bits))))

(defn- fp16-round
  "The float value a float takes after a round trip through FP16."
  ^double [^double x]
  (fp16-bits->fp32 (fp32->fp16-bits x)))

(defn- f*
  "Single-precision product."
  ^double [^double a ^double b]
  (double (float (* a b))))

(defn- f-div
  "Single-precision quotient."
  ^double [^double a ^double b]
  (double (float (/ a b))))

(defn- f+
  "Single-precision sum."
  ^double [^double a ^double b]
  (double (float (+ a b))))

(defn roundf
  "C `roundf`: nearest integer, halves away from zero."
  ^long [^double x]
  (let [r (Math/floor (+ (Math/abs x) 0.5))]
    ;; |x| + 0.5 is exact in double for every float, so floor gives roundf's value.
    (long (if (neg? x) (- r) r))))

(defn nearest-int
  "ggml `nearest_int`: adds 12582912.0f and reads the mantissa, which rounds
  halves to even for |x| <= 4194303."
  ^long [^double x]
  (long (Math/rint (double (float x)))))

(defn- c-int-cast
  "A C float-to-integer conversion: truncation toward zero."
  ^long [^double x]
  (long x))

;; ---------------------------------------------------------------------------
;; Byte helpers (little-endian, as ggml writes on every supported host)
;; ---------------------------------------------------------------------------

(defn- put-u16! [^bytes out ^long offset ^long v]
  (aset out offset (unchecked-byte (bit-and v 0xFF)))
  (aset out (inc offset) (unchecked-byte (bit-and (bit-shift-right v 8) 0xFF))))

(defn- get-u16 ^long [^bytes in ^long offset]
  (bit-or (bit-and (long (aget in offset)) 0xFF)
          (bit-shift-left (bit-and (long (aget in (inc offset))) 0xFF) 8)))

(defn- put-u32! [^bytes out ^long offset ^long v]
  (dotimes [i 4]
    (aset out (+ offset i) (unchecked-byte (bit-and (bit-shift-right v (* 8 i)) 0xFF)))))

(defn- get-u32 ^long [^bytes in ^long offset]
  (reduce (fn [^long acc ^long i]
            (bit-or acc (bit-shift-left (bit-and (long (aget in (+ offset i))) 0xFF) (* 8 i))))
          0 (range 4)))

;; ---------------------------------------------------------------------------
;; Formats
;; ---------------------------------------------------------------------------

(def formats
  "ggml block geometry: elements per block and bytes per block."
  {:q8_0 {:block 32 :bytes 34}
   :q5_0 {:block 32 :bytes 22}
   :q4_K {:block 256 :bytes 144}
   :q6_K {:block 256 :bytes 210}
   ;; activation-only: the vec_dot_type of q4_K and q6_K
   :q8_K {:block 256 :bytes 292}})

(defn row-bytes
  "Bytes of one quantized row of `n` elements in `format`."
  ^long [format ^long n]
  (let [{:keys [block bytes]} (or (get formats format)
                                  (throw (ex-info "Unknown ggml format" {:format format})))]
    (when-not (zero? (rem n (long block)))
      (throw (ex-info "Row length is not a whole number of blocks"
                      {:format format :n n :block block})))
    (* (quot n (long block)) (long bytes))))

;; q8_0: ggml_half d; int8 qs[32].

(defn- quantize-block-q8-0! [^floats x ^long xo ^bytes out ^long o]
  (let [amax (loop [j 0 m 0.0]
               (if (< j 32)
                 (let [v (Math/abs (double (aget x (+ xo j))))]
                   ;; MAX(a, b) = a > b ? a : b
                   (recur (inc j) (if (> m v) m v)))
                 m))
        d (f-div amax 127.0)
        id (if (zero? d) 0.0 (f-div 1.0 d))]
    (put-u16! out o (fp32->fp16-bits d))
    (dotimes [j 32]
      (aset out (+ o 2 j) (unchecked-byte (roundf (f* (aget x (+ xo j)) id)))))))

(defn- dequantize-block-q8-0! [^bytes in ^long o ^floats y ^long yo]
  (let [d (fp16-bits->fp32 (get-u16 in o))]
    (dotimes [j 32]
      (aset y (+ yo j) (float (f* (double (aget in (+ o 2 j))) d))))))

;; q5_0: ggml_half d; uint8 qh[4]; uint8 qs[16]. Low nibble of qs[j] is element
;; j, high nibble element j+16; bit j of qh is element j's fifth bit.

(defn- quantize-block-q5-0! [^floats x ^long xo ^bytes out ^long o]
  (let [[_ mx] (loop [j 0 amax 0.0 mx 0.0]
                 (if (< j 32)
                   (let [v (double (aget x (+ xo j)))
                         a (Math/abs v)]
                     ;; strict: the first element of largest magnitude wins
                     (if (< amax a) (recur (inc j) a v) (recur (inc j) amax mx)))
                   [amax mx]))
        d (f-div (double mx) -16.0)
        id (if (zero? d) 0.0 (f-div 1.0 d))
        qh (loop [j 0 qh 0]
             (if (< j 16)
               (let [x0 (f* (aget x (+ xo j)) id)
                     x1 (f* (aget x (+ xo 16 j)) id)
                     ;; MIN(31, (int8_t)(x + 16.5f)), then stored as uint8
                     xi0 (bit-and (min 31 (c-int-cast (f+ x0 16.5))) 0xFF)
                     xi1 (bit-and (min 31 (c-int-cast (f+ x1 16.5))) 0xFF)]
                 (aset out (+ o 6 j)
                       (unchecked-byte (bit-or (bit-and xi0 0x0F)
                                               (bit-shift-left (bit-and xi1 0x0F) 4))))
                 (recur (inc j)
                        (bit-or qh
                                (bit-shift-left (bit-shift-right (bit-and xi0 0x10) 4) j)
                                (bit-shift-left (bit-shift-right (bit-and xi1 0x10) 4) (+ j 16)))))
               qh))]
    (put-u16! out o (fp32->fp16-bits d))
    (put-u32! out (+ o 2) qh)))

(defn- dequantize-block-q5-0! [^bytes in ^long o ^floats y ^long yo]
  (let [d (fp16-bits->fp32 (get-u16 in o))
        qh (get-u32 in (+ o 2))]
    (dotimes [j 16]
      (let [q (bit-and (long (aget in (+ o 6 j))) 0xFF)
            xh0 (bit-shift-left (bit-and (bit-shift-right qh j) 1) 4)
            xh1 (bit-and (bit-shift-right qh (+ j 12)) 0x10)
            x0 (- (bit-or (bit-and q 0x0F) xh0) 16)
            x1 (- (bit-or (bit-shift-right q 4) xh1) 16)]
        (aset y (+ yo j) (float (f* (double x0) d)))
        (aset y (+ yo 16 j) (float (f* (double x1) d)))))))

;; ---------------------------------------------------------------------------
;; K-quants: shared scale searches
;; ---------------------------------------------------------------------------

(defn- f-
  "Single-precision difference."
  ^double [^double a ^double b]
  (double (float (- a b))))

(def ^:private ^:const float-0-1
  "The C literal 0.1f."
  (double (float 0.1)))

(def ^:private ^:const group-max-eps
  "GROUP_MAX_EPS, 1e-15f."
  (double (float 1.0e-15)))

(defn- make-qkx2-quants!
  "ggml make_qkx2_quants with use_mad = false. Writes codes for x[xo, xo+n)
  into L[lo, lo+n) and returns [scale the-min], where x ~ scale*L - the-min."
  [^floats x xo n nmax ^doubles weights ^longs L lo rmin rdelta nstep]
  (let [xo (long xo) n (long n) nmax (long nmax) lo (long lo)
        rmin (double rmin) rdelta (double rdelta) nstep (long nstep)
        x0 (double (aget x xo))
        w0 (aget weights 0)
        [mn mx sum-w sum-x]
        (loop [i 1 mn x0 mx x0 sum-w w0 sum-x (f* w0 x0)]
          (if (< i n)
            (let [xi (double (aget x (+ xo i)))
                  w (aget weights i)]
              (recur (inc i) (if (< xi mn) xi mn) (if (> xi mx) xi mx)
                     (f+ sum-w w) (f+ sum-x (f* w xi))))
            [mn mx sum-w sum-x]))
        mn (double (if (> (double mn) 0.0) 0.0 mn))
        mx (double mx) sum-w (double sum-w) sum-x (double sum-x)]
    (if (== mx mn)
      (do (dotimes [i n] (aset L (+ lo i) 0))
          [0.0 (- mn)])
      (let [iscale (f-div (double nmax) (f- mx mn))
            scale (f-div 1.0 iscale)
            best-error
            (loop [i 0 err 0.0]
              (if (< i n)
                (let [xi (double (aget x (+ xo i)))
                      l (max 0 (min nmax (nearest-int (f* iscale (f- xi mn)))))
                      diff (f- (f+ (f* scale (double l)) mn) xi)]
                  (aset L (+ lo i) l)
                  (recur (inc i) (f+ err (f* (aget weights i) (f* diff diff)))))
                err))]
        (if (< nstep 1)
          [scale (- mn)]
          (let [laux (long-array n)]
            (loop [is 0 best-error best-error scale scale mn mn]
              (if (> is nstep)
                [scale (- mn)]
                (let [iscale (f-div (f+ (f+ rmin (f* rdelta (double is))) (double nmax)) (f- mx mn))
                      [sum-l sum-l2 sum-xl]
                      (loop [i 0 sl 0.0 sl2 0.0 sxl 0.0]
                        (if (< i n)
                          (let [xi (double (aget x (+ xo i)))
                                l (max 0 (min nmax (nearest-int (f* iscale (f- xi mn)))))
                                w (aget weights i)
                                wl (f* w (double l))]
                            (aset laux i l)
                            (recur (inc i) (f+ sl wl) (f+ sl2 (f* wl (double l))) (f+ sxl (f* wl xi))))
                          [sl sl2 sxl]))
                      sum-l (double sum-l) sum-l2 (double sum-l2) sum-xl (double sum-xl)
                      D (f- (f* sum-w sum-l2) (f* sum-l sum-l))]
                  (if (> D 0.0)
                    (let [this-scale (f-div (f- (f* sum-w sum-xl) (f* sum-x sum-l)) D)
                          this-min (f-div (f- (f* sum-l2 sum-x) (f* sum-l sum-xl)) D)
                          [this-scale this-min] (if (> this-min 0.0)
                                                  [(f-div sum-xl sum-l2) 0.0]
                                                  [this-scale this-min])
                          this-scale (double this-scale) this-min (double this-min)
                          cur-error (loop [i 0 err 0.0]
                                      (if (< i n)
                                        (let [diff (f- (f+ (f* this-scale (double (aget laux i))) this-min)
                                                       (double (aget x (+ xo i))))]
                                          (recur (inc i) (f+ err (f* (aget weights i) (f* diff diff)))))
                                        err))]
                      (if (< cur-error best-error)
                        (do (dotimes [i n] (aset L (+ lo i) (aget laux i)))
                            (recur (inc is) cur-error this-scale this-min))
                        (recur (inc is) best-error scale mn)))
                    (recur (inc is) best-error scale mn)))))))))))

(defn- make-qx-quants!
  "ggml make_qx_quants with rmse_type 1 and no quant weights: the search
  quantize_row_q6_K_ref uses. Writes codes (offset by nmax) for x[xo, xo+n)
  into L[lo, lo+n) and returns the scale."
  [^floats x xo n nmax ^longs L lo]
  (let [xo (long xo) n (long n) nmax (long nmax) lo (long lo)
        [amax mx] (loop [i 0 amax 0.0 mx 0.0]
                    (if (< i n)
                      (let [xi (double (aget x (+ xo i)))
                            ax (Math/abs xi)]
                        (if (> ax amax) (recur (inc i) ax xi) (recur (inc i) amax mx)))
                      [amax mx]))
        amax (double amax) mx (double mx)]
    (if (< amax group-max-eps)
      (do (dotimes [i n] (aset L (+ lo i) 0)) 0.0)
      (let [clamp (fn ^long [^long l] (max (- nmax) (min (dec nmax) l)))
            sums (fn [^double iscale]
                   (loop [i 0 sumlx 0.0 suml2 0.0]
                     (if (< i n)
                       (let [xi (double (aget x (+ xo i)))
                             l (clamp (nearest-int (f* iscale xi)))
                             w (f* xi xi)]
                         (recur (inc i)
                                (f+ sumlx (f* (f* w xi) (double l)))
                                (f+ suml2 (f* (f* w (double l)) (double l)))))
                       [sumlx suml2])))
            fill! (fn [^double iscale]
                    (dotimes [i n]
                      (aset L (+ lo i)
                            (+ nmax (long (clamp (nearest-int (f* iscale (double (aget x (+ xo i)))))))))))
            iscale (f-div (double (- nmax)) mx)
            _ (fill! iscale)
            [sumlx suml2] (sums iscale)
            sumlx (double sumlx) suml2 (double suml2)
            scale (if (zero? suml2) 0.0 (f-div sumlx suml2))]
        (loop [is -9 scale scale best (f* scale sumlx)]
          (cond
            (> is 9) scale
            (zero? is) (recur (inc is) scale best)
            :else
            (let [iscale (f-div (- (f+ (double nmax) (f* float-0-1 (double is)))) mx)
                  [sumlx suml2] (sums iscale)
                  sumlx (double sumlx) suml2 (double suml2)]
              (if (and (> suml2 0.0) (> (f* sumlx sumlx) (f* best suml2)))
                (let [scale (f-div sumlx suml2)]
                  (fill! iscale)
                  (recur (inc is) scale (f* scale sumlx)))
                (recur (inc is) scale best)))))))))

;; q4_K: ggml_half d; ggml_half dmin; uint8 scales[12]; uint8 qs[128].

(defn- scale-min-k4
  "ggml get_scale_min_k4: the 6-bit scale and min of sub-block j."
  [^bytes in ^long so ^long j]
  (let [q (fn ^long [^long i] (bit-and (long (aget in (+ so i))) 0xFF))]
    (if (< j 4)
      [(bit-and (q j) 63) (bit-and (q (+ j 4)) 63)]
      [(bit-or (bit-and (q (+ j 4)) 0xF) (bit-shift-left (bit-shift-right (q (- j 4)) 6) 4))
       (bit-or (bit-shift-right (q (+ j 4)) 4) (bit-shift-left (bit-shift-right (q j) 6) 4))])))

(defn- quantize-block-q4-K! [^floats x ^long xo ^bytes out ^long o]
  (let [L (long-array 256)
        weights (double-array 32)
        scales (double-array 8)
        mins (double-array 8)
        [max-scale max-min]
        (loop [j 0 max-scale 0.0 max-min 0.0]
          (if (< j 8)
            (let [base (+ xo (* 32 j))
                  sum-x2 (loop [l 0 s 0.0]
                           (if (< l 32)
                             (let [v (double (aget x (+ base l)))] (recur (inc l) (f+ s (f* v v))))
                             s))
                  av-x (double (float (Math/sqrt (f-div sum-x2 32.0))))]
              (dotimes [l 32] (aset weights l (f+ av-x (Math/abs (double (aget x (+ base l)))))))
              (let [[scale the-min] (make-qkx2-quants! x base 32 15 weights L (* 32 j)
                                                       -1.0 float-0-1 20)
                    scale (double scale) the-min (double the-min)]
                (aset scales j scale)
                (aset mins j the-min)
                (recur (inc j) (if (> scale max-scale) scale max-scale)
                       (if (> the-min max-min) the-min max-min))))
            [max-scale max-min]))
        max-scale (double max-scale) max-min (double max-min)
        inv-scale (if (> max-scale 0.0) (f-div 63.0 max-scale) 0.0)
        inv-min (if (> max-min 0.0) (f-div 63.0 max-min) 0.0)
        so (+ o 4)]
    (dotimes [i 12] (aset out (+ so i) (byte 0)))
    (dotimes [j 8]
      (let [ls (min 63 (bit-and (nearest-int (f* inv-scale (aget scales j))) 0xFF))
            lm (min 63 (bit-and (nearest-int (f* inv-min (aget mins j))) 0xFF))
            put! (fn [^long i ^long v] (aset out (+ so i) (unchecked-byte v)))
            get (fn ^long [^long i] (bit-and (long (aget out (+ so i))) 0xFF))]
        (if (< j 4)
          (do (put! j ls) (put! (+ j 4) lm))
          (do (put! (+ j 4) (bit-or (bit-and ls 0xF) (bit-shift-left (bit-and lm 0xF) 4)))
              (put! (- j 4) (bit-or (get (- j 4)) (bit-shift-left (bit-shift-right ls 4) 6)))
              (put! j (bit-or (get j) (bit-shift-left (bit-shift-right lm 4) 6)))))))
    (put-u16! out o (fp32->fp16-bits (f-div max-scale 63.0)))
    (put-u16! out (+ o 2) (fp32->fp16-bits (f-div max-min 63.0)))
    (let [d (fp16-bits->fp32 (get-u16 out o))
          dmin (fp16-bits->fp32 (get-u16 out (+ o 2)))]
      (dotimes [j 8]
        (let [[sc m] (scale-min-k4 out so j)
              dj (f* d (double sc))]
          ;; a zero scale keeps the codes make_qkx2_quants chose
          (when-not (zero? dj)
            (let [dm (f* dmin (double m))]
              (dotimes [ii 32]
                (aset L (+ (* 32 j) ii)
                      (max 0 (min 15 (nearest-int (f-div (f+ (double (aget x (+ xo (* 32 j) ii))) dm)
                                                         dj)))))))))))
    (dotimes [g 4]
      (dotimes [l 32]
        (aset out (+ o 16 (* 32 g) l)
              (unchecked-byte (bit-or (aget L (+ (* 64 g) l))
                                      (bit-shift-left (aget L (+ (* 64 g) l 32)) 4))))))))

(defn- dequantize-block-q4-K! [^bytes in ^long o ^floats y ^long yo]
  (let [d (fp16-bits->fp32 (get-u16 in o))
        dmin (fp16-bits->fp32 (get-u16 in (+ o 2)))
        so (+ o 4)]
    (dotimes [g 4]
      (let [[sc1 m1] (scale-min-k4 in so (* 2 g))
            [sc2 m2] (scale-min-k4 in so (inc (* 2 g)))
            d1 (f* d (double sc1)) mm1 (f* dmin (double m1))
            d2 (f* d (double sc2)) mm2 (f* dmin (double m2))]
        (dotimes [l 32]
          (let [q (bit-and (long (aget in (+ o 16 (* 32 g) l))) 0xFF)]
            (aset y (+ yo (* 64 g) l) (float (f- (f* d1 (double (bit-and q 0xF))) mm1)))
            (aset y (+ yo (* 64 g) 32 l) (float (f- (f* d2 (double (bit-shift-right q 4))) mm2)))))))))

;; q6_K: uint8 ql[128]; uint8 qh[64]; int8 scales[16]; ggml_half d (offset 208).

(defn- quantize-block-q6-K! [^floats x ^long xo ^bytes out ^long o]
  (let [L (long-array 256)
        scales (double-array 16)
        [max-scale max-abs-scale]
        (loop [ib 0 max-scale 0.0 max-abs 0.0]
          (if (< ib 16)
            (let [scale (double (make-qx-quants! x (+ xo (* 16 ib)) 16 32 L (* 16 ib)))
                  abs-scale (Math/abs scale)]
              (aset scales ib scale)
              (if (> abs-scale max-abs)
                (recur (inc ib) scale abs-scale)
                (recur (inc ib) max-scale max-abs)))
            [max-scale max-abs]))
        max-scale (double max-scale)]
    (if (< (double max-abs-scale) group-max-eps)
      (do (dotimes [i 210] (aset out (+ o i) (byte 0)))
          (put-u16! out (+ o 208) (fp32->fp16-bits 0.0)))
      (let [iscale (f-div -128.0 max-scale)]
        (put-u16! out (+ o 208) (fp32->fp16-bits (f-div 1.0 iscale)))
        (dotimes [ib 16]
          (aset out (+ o 192 ib) (unchecked-byte (min 127 (nearest-int (f* iscale (aget scales ib)))))))
        (let [d (fp16-bits->fp32 (get-u16 out (+ o 208)))]
          (dotimes [j 16]
            (let [dj (f* d (double (aget out (+ o 192 j))))]
              (when-not (zero? dj)
                (dotimes [ii 16]
                  (aset L (+ (* 16 j) ii)
                        (+ 32 (max -32 (min 31 (nearest-int (f-div (double (aget x (+ xo (* 16 j) ii)))
                                                                   dj)))))))))))
        (dotimes [h 2]
          (let [lb (* 128 h) qlo (+ o (* 64 h)) qho (+ o 128 (* 32 h))]
            (dotimes [l 32]
              (let [q1 (aget L (+ lb l)) q2 (aget L (+ lb l 32))
                    q3 (aget L (+ lb l 64)) q4 (aget L (+ lb l 96))]
                (aset out (+ qlo l) (unchecked-byte (bit-or (bit-and q1 0xF) (bit-shift-left (bit-and q3 0xF) 4))))
                (aset out (+ qlo 32 l) (unchecked-byte (bit-or (bit-and q2 0xF) (bit-shift-left (bit-and q4 0xF) 4))))
                (aset out (+ qho l)
                      (unchecked-byte (bit-or (bit-shift-right q1 4)
                                              (bit-shift-left (bit-shift-right q2 4) 2)
                                              (bit-shift-left (bit-shift-right q3 4) 4)
                                              (bit-shift-left (bit-shift-right q4 4) 6))))))))))))

(defn- dequantize-block-q6-K! [^bytes in ^long o ^floats y ^long yo]
  (let [d (fp16-bits->fp32 (get-u16 in (+ o 208)))]
    (dotimes [h 2]
      (let [qlo (+ o (* 64 h)) qho (+ o 128 (* 32 h)) sco (+ o 192 (* 8 h)) yb (+ yo (* 128 h))
            u (fn ^long [^long i] (bit-and (long (aget in i)) 0xFF))]
        (dotimes [l 32]
          (let [is (quot l 16)
                ql0 (u (+ qlo l)) ql1 (u (+ qlo 32 l)) qh (u (+ qho l))
                q1 (- (bit-or (bit-and ql0 0xF) (bit-shift-left (bit-and qh 3) 4)) 32)
                q2 (- (bit-or (bit-and ql1 0xF) (bit-shift-left (bit-and (bit-shift-right qh 2) 3) 4)) 32)
                q3 (- (bit-or (bit-shift-right ql0 4) (bit-shift-left (bit-and (bit-shift-right qh 4) 3) 4)) 32)
                q4 (- (bit-or (bit-shift-right ql1 4) (bit-shift-left (bit-and (bit-shift-right qh 6) 3) 4)) 32)
                sc (fn ^double [^long k] (double (aget in (+ sco is k))))]
            (aset y (+ yb l) (float (f* (f* d (sc 0)) (double q1))))
            (aset y (+ yb 32 l) (float (f* (f* d (sc 2)) (double q2))))
            (aset y (+ yb 64 l) (float (f* (f* d (sc 4)) (double q3))))
            (aset y (+ yb 96 l) (float (f* (f* d (sc 6)) (double q4))))))))))

;; q8_K: float d; int8 qs[256]; int16 bsums[16]. Activation format only; its
;; scale is a full float and negative when the largest-magnitude value is positive.

(defn- quantize-block-q8-K! [^floats x ^long xo ^bytes out ^long o]
  (let [[amax mx] (loop [j 0 amax 0.0 mx 0.0]
                    (if (< j 256)
                      (let [v (double (aget x (+ xo j)))
                            a (Math/abs v)]
                        (if (> a amax) (recur (inc j) a v) (recur (inc j) amax mx)))
                      [amax mx]))
        bb (.order (ByteBuffer/wrap out) ByteOrder/LITTLE_ENDIAN)]
    (if (zero? (double amax))
      (do (.putFloat bb (int o) (float 0.0))
          (dotimes [i 288] (aset out (+ o 4 i) (byte 0))))
      (let [iscale (f-div -127.0 (double mx))]
        (dotimes [j 256]
          (aset out (+ o 4 j) (unchecked-byte (min 127 (nearest-int (f* iscale (double (aget x (+ xo j)))))))))
        (dotimes [j 16]
          (let [sum (reduce + (map #(long (aget out (+ o 4 (* 16 j) (long %)))) (range 16)))]
            (.putShort bb (int (+ o 260 (* 2 j))) (short sum))))
        (.putFloat bb (int o) (float (f-div 1.0 iscale)))))))

(defn- dequantize-block-q8-K! [^bytes in ^long o ^floats y ^long yo]
  (let [d (double (.getFloat (.order (ByteBuffer/wrap in) ByteOrder/LITTLE_ENDIAN) (int o)))]
    (dotimes [j 256]
      (aset y (+ yo j) (float (f* d (double (aget in (+ o 4 j)))))))))

(def ^:private block-fns
  {:q8_0 [quantize-block-q8-0! dequantize-block-q8-0!]
   :q5_0 [quantize-block-q5-0! dequantize-block-q5-0!]
   :q4_K [quantize-block-q4-K! dequantize-block-q4-K!]
   :q6_K [quantize-block-q6-K! dequantize-block-q6-K!]
   :q8_K [quantize-block-q8-K! dequantize-block-q8-K!]})

(defn quantize
  "Quantize `nrows` rows of `n-per-row` floats from `x` into ggml `format`
  blocks, as ggml_quantize_chunk does without an importance matrix."
  ^bytes [format ^floats x ^long n-per-row ^long nrows]
  (when-not (= (alength x) (* n-per-row nrows))
    (throw (ex-info "Input length does not match the shape"
                    {:length (alength x) :n-per-row n-per-row :nrows nrows})))
  (let [row (row-bytes format n-per-row)
        {:keys [block bytes]} (get formats format)
        block (long block) bytes (long bytes)
        [quantize-block!] (get block-fns format)
        out (byte-array (* row nrows))]
    (dotimes [b (quot (* n-per-row nrows) block)]
      (quantize-block! x (* b block) out (* b bytes)))
    out))

(defn dequantize
  "Expand ggml `format` blocks back to floats, as the type's to_float does."
  ^floats [format ^bytes blocks ^long n]
  (let [{:keys [block bytes]} (or (get formats format)
                                  (throw (ex-info "Unknown ggml format" {:format format})))
        block (long block) bytes (long bytes)
        [_ dequantize-block!] (get block-fns format)
        y (float-array n)]
    (when-not (= (alength blocks) (* (quot n block) bytes))
      (throw (ex-info "Block bytes do not match the element count"
                      {:format format :bytes (alength blocks) :n n})))
    (dotimes [b (quot n block)]
      (dequantize-block! blocks (* b bytes) y (* b block)))
    y))

;; ---------------------------------------------------------------------------
;; Dot products: ggml's scalar generic vec_dot
;; ---------------------------------------------------------------------------

(defn- q8-0-block [^bytes b ^long o]
  {:d (fp16-bits->fp32 (get-u16 b o))
   :q (long-array (map #(long (aget b (+ o 2 (long %)))) (range 32)))})

(defn- q5-0-codes
  "Signed codes q-16 of the q5_0 block at o, in element order."
  ^longs [^bytes b ^long o]
  (let [qh (get-u32 b (+ o 2))
        out (long-array 32)]
    (dotimes [j 16]
      (let [q (bit-and (long (aget b (+ o 6 j))) 0xFF)]
        (aset out j (- (bit-or (bit-and q 0x0F) (bit-shift-left (bit-and (bit-shift-right qh j) 1) 4)) 16))
        (aset out (+ 16 j) (- (bit-or (bit-shift-right q 4) (bit-and (bit-shift-right qh (+ j 12)) 0x10)) 16))))
    out))

(defn- q8-K-block [^bytes b ^long o]
  (let [bb (.order (ByteBuffer/wrap b) ByteOrder/LITTLE_ENDIAN)]
    {:d (double (.getFloat bb (int o)))
     :q (long-array (map #(long (aget b (+ o 4 (long %)))) (range 256)))
     :bsums (long-array (map #(long (.getShort bb (int (+ o 260 (* 2 (long %)))))) (range 16)))}))

(defn- q4-K-codes
  "Unsigned 4-bit codes of the q4_K block at o, in element order."
  ^longs [^bytes b ^long o]
  (let [out (long-array 256)]
    (dotimes [g 4]
      (dotimes [l 32]
        (let [q (bit-and (long (aget b (+ o 16 (* 32 g) l))) 0xFF)]
          (aset out (+ (* 64 g) l) (bit-and q 0xF))
          (aset out (+ (* 64 g) 32 l) (bit-shift-right q 4)))))
    out))

(defn- q6-K-codes
  "Signed codes q-32 of the q6_K block at o, in element order."
  ^longs [^bytes b ^long o]
  (let [out (long-array 256)
        u (fn ^long [^long i] (bit-and (long (aget b i)) 0xFF))]
    (dotimes [h 2]
      (let [qlo (+ o (* 64 h)) qho (+ o 128 (* 32 h)) base (* 128 h)]
        (dotimes [l 32]
          (let [ql0 (u (+ qlo l)) ql1 (u (+ qlo 32 l)) qh (u (+ qho l))]
            (aset out (+ base l) (- (bit-or (bit-and ql0 0xF) (bit-shift-left (bit-and qh 3) 4)) 32))
            (aset out (+ base 32 l) (- (bit-or (bit-and ql1 0xF) (bit-shift-left (bit-and (bit-shift-right qh 2) 3) 4)) 32))
            (aset out (+ base 64 l) (- (bit-or (bit-shift-right ql0 4) (bit-shift-left (bit-and (bit-shift-right qh 4) 3) 4)) 32))
            (aset out (+ base 96 l) (- (bit-or (bit-shift-right ql1 4) (bit-shift-left (bit-and (bit-shift-right qh 6) 3) 4)) 32))))))
    out))

(defn- fma-f
  "Single-precision fused multiply-add: a*b + c rounded once."
  ^double [^double a ^double b ^double c]
  (double (Math/fma (float a) (float b) (float c))))

(defn- lane-sums
  "ggml's eight int32 lanes: element i of every 8-element chunk adds
  scale(chunk) * w[i] * x[i] into lane (i mod 8)."
  ^longs [^longs w ^longs x scale-of-chunk]
  (let [lanes (long-array 8)]
    (dotimes [c (quot (alength w) 8)]
      (let [scale (long (scale-of-chunk c))]
        (dotimes [l 8]
          (let [i (+ (* 8 c) l)]
            (aset lanes l (+ (aget lanes l) (* scale (* (aget w i) (aget x i)))))))))
    lanes))

(defn vec-dot
  "ggml's scalar generic vec_dot of one `weight-format` row with its activation
  row, both as ggml blocks, over `n` elements.

  Integer accumulation is exact, as in C. Float operations follow the C
  expression order. `contract?` makes the float steps a C compiler may fuse
  under -ffp-contract (q4_K's `sums[l] += d*aux32[l]` and `sumf -= dmin*sumi`)
  single-rounded fused multiply-adds, as a build with FMA does."
  ([weight-format ^bytes w ^bytes x n] (vec-dot weight-format w x n {}))
  ([weight-format ^bytes w ^bytes x n {:keys [contract?]}]
   (let [n (long n)]
     (case weight-format
       :q8_0
       (loop [ib 0 sumf 0.0]
         (if (< ib (quot n 32))
           (let [wb (q8-0-block w (* ib 34))
                 xb (q8-0-block x (* ib 34))
                 wd (double (:d wb)) xd (double (:d xb))
                 ^longs wq (:q wb) ^longs xq (:q xb)
                 sumi (areduce wq i acc 0 (+ acc (* (aget wq i) (aget xq i))))]
             (recur (inc ib) (f+ sumf (f* (double sumi) (f* wd xd)))))
           sumf))

       :q5_0
       (loop [ib 0 sumf 0.0]
         (if (< ib (quot n 32))
           (let [^longs wq (q5-0-codes w (* ib 22))
                 wd (fp16-bits->fp32 (get-u16 w (* ib 22)))
                 xb (q8-0-block x (* ib 34))
                 xd (double (:d xb))
                 ^longs xq (:q xb)
                 sumi (areduce wq i acc 0 (+ acc (* (aget wq i) (aget xq i))))]
             (recur (inc ib) (f+ sumf (f* (f* wd xd) (double sumi)))))
           sumf))

       (:q4_K :q6_K)
       (let [q4? (= :q4_K weight-format)
             wbytes (if q4? 144 210)
             sums (double-array 8)
             sumf (loop [i 0 sumf 0.0]
                    (if (< i (quot n 256))
                      (let [wo (* i wbytes)
                            xb (q8-K-block x (* i 292))
                            xd (double (:d xb))
                            ^longs xq (:q xb)
                            ^longs bsums (:bsums xb)
                            codes (if q4? (q4-K-codes w wo) (q6-K-codes w wo))
                            scale-min (when q4? (mapv #(scale-min-k4 w (+ wo 4) %) (range 8)))
                            lanes (lane-sums codes xq
                                             (if q4?
                                               ;; one 6-bit scale per 32 elements (4 chunks)
                                               (fn [c] (first (nth scale-min (quot (long c) 4))))
                                               ;; one int8 scale per 16 elements (2 chunks)
                                               (fn [c] (aget w (+ wo 192 (quot (long c) 2))))))
                            wd (fp16-bits->fp32 (get-u16 w (if q4? wo (+ wo 208))))
                            d (f* wd xd)]
                        (dotimes [l 8]
                          (aset sums l (if (and contract? q4?)
                                         (fma-f d (double (aget lanes l)) (aget sums l))
                                         (f+ (aget sums l) (f* d (double (aget lanes l)))))))
                        (if q4?
                          (let [sumi (reduce + (map #(* (aget bsums (long %))
                                                        (long (second (nth scale-min (quot (long %) 2)))))
                                                    (range 16)))
                                dmin (f* (fp16-bits->fp32 (get-u16 w (+ wo 2))) xd)]
                            (recur (inc i) (if contract?
                                             (fma-f (- dmin) (double sumi) sumf)
                                             (f- sumf (f* dmin (double sumi))))))
                          (recur (inc i) sumf)))
                      sumf))]
         (areduce sums l acc (double sumf) (f+ acc (aget sums l))))))))

;; ---------------------------------------------------------------------------
;; Kernel layouts: lossless decodings of ggml blocks for the GPU dot kernels
;; ---------------------------------------------------------------------------

(defn- pack-words
  "Pack signed or unsigned byte-range codes four per int32, little-endian."
  ^ints [^longs codes]
  (let [out (int-array (quot (alength codes) 4))]
    (dotimes [w (alength out)]
      (aset out w (unchecked-int
                   (bit-or (bit-and (aget codes (* 4 w)) 0xFF)
                           (bit-shift-left (bit-and (aget codes (+ (* 4 w) 1)) 0xFF) 8)
                           (bit-shift-left (bit-and (aget codes (+ (* 4 w) 2)) 0xFF) 16)
                           (bit-shift-left (bit-and (aget codes (+ (* 4 w) 3)) 0xFF) 24)))))
    out))

(defn- concat-longs ^longs [parts]
  (let [out (long-array (reduce + (map #(alength ^longs %) parts)))]
    (reduce (fn [^long o ^longs p] (System/arraycopy p 0 out o (alength p)) (+ o (alength p)))
            0 parts)
    out))

(defn kernel-layout
  "Decode `nrows` rows of ggml `format` blocks (each `n` elements) into the
  arrays the GPU dot kernels read. Every value is exact: FP16 scales become the
  floats they denote, packed 6-bit scales become ints, and codes are packed
  four int8 per int32 word in element order.

    :q8_0, :q5_0  {:q int[n/4 per row] :d float[n/32 per row]}
    :q8_K         {:q int[...] :d float[n/256 per row] :bsums int[16 per block]}
    :q4_K         {:q int[...] :d float[...] :dmin float[...] :sc int[8 per block] :m int[8 per block]}
    :q6_K         {:q int[...] :d float[...] :sc int[16 per block]}"
  [format ^bytes blocks n nrows]
  (let [n (long n) nrows (long nrows)
        {:keys [block bytes]} (get formats format)
        block (long block) bytes (long bytes)
        nblocks (* nrows (quot n block))
        offsets (map #(* bytes (long %)) (range nblocks))]
    (case format
      (:q8_0 :q5_0)
      {:q (pack-words (concat-longs (map #(if (= format :q8_0)
                                             (:q (q8-0-block blocks %))
                                             (q5-0-codes blocks %))
                                          offsets)))
       :d (float-array (map #(fp16-bits->fp32 (get-u16 blocks %)) offsets))}

      :q8_K
      (let [bs (map #(q8-K-block blocks %) offsets)]
        {:q (pack-words (concat-longs (map :q bs)))
         :d (float-array (map :d bs))
         :bsums (int-array (mapcat #(seq ^longs (:bsums %)) bs))})

      :q4_K
      {:q (pack-words (concat-longs (map #(q4-K-codes blocks %) offsets)))
       :d (float-array (map #(fp16-bits->fp32 (get-u16 blocks %)) offsets))
       :dmin (float-array (map #(fp16-bits->fp32 (get-u16 blocks (+ (long %) 2))) offsets))
       :sc (int-array (mapcat (fn [o] (map #(first (scale-min-k4 blocks (+ (long o) 4) %)) (range 8))) offsets))
       :m (int-array (mapcat (fn [o] (map #(second (scale-min-k4 blocks (+ (long o) 4) %)) (range 8))) offsets))}

      :q6_K
      {:q (pack-words (concat-longs (map #(q6-K-codes blocks %) offsets)))
       :d (float-array (map #(fp16-bits->fp32 (get-u16 blocks (+ (long %) 208))) offsets))
       :sc (int-array (mapcat (fn [o]
                                (map (fn [i] (long (aget blocks (+ (long o) 192 (long i)))))
                                     (range 16)))
                              offsets))})))

(defn read-f32
  "Little-endian float32 values from a byte array."
  ^floats [^bytes data]
  (let [bb (.order (ByteBuffer/wrap data) ByteOrder/LITTLE_ENDIAN)
        out (float-array (quot (alength data) 4))]
    (dotimes [i (alength out)] (aset out i (.getFloat bb (* 4 i))))
    out))

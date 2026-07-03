(ns raster.dl.qlinear-k
  "Composable K-quant GEMV kernels — the SAME deftm compiles to CPU-C (gcc auto-vectorizes
  the dpbusd-able inner loop) and, via the shared c_emit, to OpenCL/GPU. The scalar per-row
  dot mirrors the validated references (raster.compiler.backend.cpu.quant/q4k-q8k-dot,
  q6k-q8k-dot); the fused 8-col-lanes peak SIMD is the orthogonal #27 optimization serving
  every format. Q4_K (asymmetric, 8×32) + Q6_K (symmetric, 16×16) over the format registry.

  Weight matrix [out×in] is quantized per-row (each row independent); for in a multiple of
  256 each row spans whole super-blocks, so the flat per-row arrays index as (row, block)."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.par :as par]))

;; ---- q8_K activation quantizer (the GPU-resident enabler) ----
;; float activation x[in] → q8_K for the dp4a matmul, ON the GPU: per 256-super-block
;; d = max|x|/127; int8 quants packed 4-per-int32 element-order (xp); xs = d; bsums = per
;; 32-sub-block int sum. Output feeds qmatmul-q4k-dp4a! directly (no host round-trip), so
;; norm→quant→matmul chains run entirely on-device in a graph.
;;
;; 2-PHASE PARALLEL (was: one work-item per super-block, 512-deep serial — the same
;; occupancy disease as the old decode attention, measured ~30µs/launch at nsb=3):
;; phase 1 computes per-32-sub-block |max| (nsb*8 items, 32-deep), phase 2 folds the 8
;; sibling submaxes (float max is EXACT under reassociation → d is bit-identical to the
;; serial scan) and quantizes its 32 elements (nsb*8 items, ~40-deep). `submax` is an
;; nsb*8 float scratch. The 8 same-value xs writes per super-block are benign.
(deftm quant-act-q8k-gpu!
  [x :- (Array float), xp :- (Array int), xs :- (Array float), bsums :- (Array int),
   submax :- (Array float), nsb :- Long] :- Void
  (do
    (par/map-void! si (* (long nsb) 8)
      (let [base (* (long si) 32)
            m (loop [k 0 m 0.0]
                (if (< k 32)
                  (let [v (ra/aget x (+ base k))]
                    (recur (inc k) (max m (max v (- v)))))
                  m))]
        (ra/aset submax si (float m))))
    (par/map-void! sj (* (long nsb) 8)
      (let [sb (quot sj 8)
            j (rem sj 8)
            mx (loop [k 0 m 0.0]
                 (if (< k 8)
                   (recur (inc k) (max m (ra/aget submax (+ (* (long sb) 8) k))))
                   m))
            d (/ (double mx) 127.0)
            id (if (> (double d) 0.0) (/ 1.0 (double d)) 0.0)
            sidx (+ (* (long sb) 8) j)
            wbase (+ (* (long sb) 64) (* j 8))
            ebase (+ (* (long sb) 256) (* j 32))]
        (ra/aset xs sb (float d))
        (let [bs (loop [w 0 s 0]
                   (if (< w 8)
                     ;; |x·id| ≤ 127 by construction (id = 127/max|x|), so round lands in
                     ;; int8 range without an explicit clamp; the (long ...) cast types q.
                     (let [e (+ ebase (* w 4))
                           q0 (long (Math/round (* (double (ra/aget x e)) id)))
                           q1 (long (Math/round (* (double (ra/aget x (+ e 1))) id)))
                           q2 (long (Math/round (* (double (ra/aget x (+ e 2))) id)))
                           q3 (long (Math/round (* (double (ra/aget x (+ e 3))) id)))
                           word (bit-or (bit-and q0 0xFF)
                                        (bit-shift-left (bit-and q1 0xFF) 8)
                                        (bit-shift-left (bit-and q2 0xFF) 16)
                                        (bit-shift-left (bit-and q3 0xFF) 24))]
                       (ra/aset xp (+ wbase w) (int word))
                       (recur (inc w) (+ s q0 q1 q2 q3)))
                     s))]
          (ra/aset bsums sidx (int bs)))))))

(deftm qmatmul-q4k-composable!
  "Q4_K GEMV into y[o] for o in [o-start, o-start+o-count): per super-block
  d_act·[Σ_sub (da·aq)·dpbusd + (db·bq)·bsum], the asymmetric K dot. Weight arrays are the
  whole matrix (wq nibbles, da/db super-scales, aq/bq 6-bit sub scale/min); activation is
  q8_K (xq int8, xs super-scale, bsums per 32-sub-block)."
  [xq :- (Array byte), xs :- (Array float), bsums :- (Array int),
   wq :- (Array byte), da :- (Array float), db :- (Array float),
   aq :- (Array byte), bq :- (Array byte),
   y :- (Array float), in :- Long, out :- Long, o-start :- Long, o-count :- Long] :- (Array float)
  (let [nsb (quot (long in) 256) nsub (quot (long in) 32) wrow (quot (long in) 2)]
    (dotimes [oi (long o-count)]
      (let [o (+ (long o-start) oi)
            wb (* o wrow) sbb (* o nsb) subb (* o nsub)
            acc (loop [sb 0 a 0.0]
                  (if (< sb nsb)
                    (let [sbase (* sb 256)
                          dav (double (ra/aget da (+ sbb sb)))
                          dbv (double (ra/aget db (+ sbb sb)))
                          dact (double (ra/aget xs sb))
                          ssum (loop [j 0 s 0.0]
                                 (if (< j 8)
                                   (let [base (+ sbase (* j 32)) sidx (+ (* sb 8) j)
                                         aj (* dav (long (ra/aget aq (+ subb sidx))))
                                         bj (* dbv (long (ra/aget bq (+ subb sidx))))
                                         dp (loop [k 0 d 0]
                                              (if (< k 16)
                                                (let [bv (bit-and (long (ra/aget wq (+ wb (quot base 2) k))) 0xFF)]
                                                  (recur (inc k)
                                                         (+ d (* (bit-and bv 0xF) (long (ra/aget xq (+ base k))))
                                                            (* (bit-shift-right bv 4) (long (ra/aget xq (+ base k 16)))))))
                                                d))]
                                     (recur (inc j) (+ s (* aj (double dp)) (* bj (double (ra/aget bsums sidx))))))
                                   s))]
                      (recur (inc sb) (+ a (* dact ssum))))
                    a))]
        (ra/aset y o (float acc))))
    y))

(deftm qmatmul-q6k-composable!
  "Q6_K GEMV into y[o]: per super-block d·d_act·Σ_sub sc·(dpbusd - 32·bsum), the symmetric
  K dot (no min term; the signed 6-bit weight via the unsigned+zp(32) fold). Weight arrays:
  wq unsigned-6-bit bytes, sc int8 sub-scale, ds super-scale; activation q8_K with 16-sub bsums."
  [xq :- (Array byte), xs :- (Array float), bsums :- (Array int),
   wq :- (Array byte), sc :- (Array byte), ds :- (Array float),
   y :- (Array float), in :- Long, out :- Long, o-start :- Long, o-count :- Long] :- (Array float)
  (let [nsb (quot (long in) 256) nsub (quot (long in) 16)]
    (dotimes [oi (long o-count)]
      (let [o (+ (long o-start) oi)
            wb (* o (long in)) sbb (* o nsb) subb (* o nsub)
            acc (loop [sb 0 a 0.0]
                  (if (< sb nsb)
                    (let [sbase (* sb 256)
                          dv (double (ra/aget ds (+ sbb sb)))
                          dact (double (ra/aget xs sb))
                          ssum (loop [j 0 s 0.0]
                                 (if (< j 16)
                                   (let [base (+ sbase (* j 16)) sidx (+ (* sb 16) j)
                                         scj (long (ra/aget sc (+ subb sidx)))
                                         dp (loop [k 0 d 0]
                                              (if (< k 16)
                                                (recur (inc k)
                                                       (+ d (* (bit-and (long (ra/aget wq (+ wb base k))) 0xFF)
                                                               (long (ra/aget xq (+ base k))))))
                                                d))
                                         folded (- dp (* 32 (long (ra/aget bsums sidx))))]
                                     (recur (inc j) (+ s (* scj (double folded)))))
                                   s))]
                      (recur (inc sb) (+ a (* dv dact ssum))))
                    a))]
        (ra/aset y o (float acc))))
    y))

;; ---- dp4a int8-GEMV core (the format-agnostic hardware-accelerated path) ----
;; int8×int8 GEMV over int32-packed lanes: y[o] = Σ_w dp4a(wp[o*kw+w], xp[w]). par/dp4a
;; lowers to the portable rstr_dp4a helper which the OpenCL/C compiler pattern-matches to
;; a hardware dp4a (Intel/CUDA __dp4a, AMD sdot4). This is the reusable MAC core the quant
;; formats feed unpacked int8 into; validated exact vs scalar on the Arc.
(deftm i8gemv-dp4a!
  [wp :- (Array int), xp :- (Array int), y :- (Array float),
   kw :- Long, out :- Long] :- Void
  (par/map-void! o out
                 (let [base (* (long o) (long kw))
                       acc (loop [k 0 a 0]
                             (if (< k (long kw))
                               (recur (inc k) (par/dp4a (ra/aget wp (+ base k)) (ra/aget xp k) a))
                               a))]
                   (ra/aset y o (float acc)))))

;; ---- GPU shape: par/map-void! over output rows (one work-item per row) ----
;; The GPU form the opencl-pass turns into a kernel (work-item o computes y[o]). Same K
;; dot as the composable CPU deftm; the GPU int8 path needs byte buffers / int32-packing
;; (dp4a) — see #26. This is the work-item-mapped twin of qmatmul-q4k-composable!.
(deftm qmatmul-q4k-gpu!
  [xq :- (Array byte), xs :- (Array float), bsums :- (Array int),
   wq :- (Array byte), da :- (Array float), db :- (Array float),
   aq :- (Array byte), bq :- (Array byte),
   y :- (Array float), in :- Long, out :- Long] :- Void
  (par/map-void! o out
                 (let [nsb (quot (long in) 256) nsub (quot (long in) 32) wrow (quot (long in) 2)
                       wb (* (long o) wrow) sbb (* (long o) nsb) subb (* (long o) nsub)
                       acc (loop [sb 0 a 0.0]
                             (if (< sb nsb)
                               (let [sbase (* sb 256)
                                     dav (double (ra/aget da (+ sbb sb)))
                                     dbv (double (ra/aget db (+ sbb sb)))
                                     dact (double (ra/aget xs sb))
                                     ssum (loop [j 0 s 0.0]
                                            (if (< j 8)
                                              (let [base (+ sbase (* j 32)) sidx (+ (* sb 8) j)
                                                    aj (* dav (long (ra/aget aq (+ subb sidx))))
                                                    bj (* dbv (long (ra/aget bq (+ subb sidx))))
                                                    dp (loop [k 0 d 0]
                                                         (if (< k 16)
                                                           (let [bv (bit-and (long (ra/aget wq (+ wb (quot base 2) k))) 0xFF)]
                                                             (recur (inc k)
                                                                    (+ d (* (bit-and bv 0xF) (long (ra/aget xq (+ base k))))
                                                                       (* (bit-shift-right bv 4) (long (ra/aget xq (+ base k 16)))))))
                                                           d))]
                                                (recur (inc j) (+ s (* aj (double dp)) (* bj (double (ra/aget bsums sidx))))))
                                              s))]
                                 (recur (inc sb) (+ a (* dact ssum))))
                               a))]
                   (ra/aset y o (float acc)))))

;; ---- Q4_K dp4a GPU kernel (the hardware-accelerated path) ----
;; Same scale/min fold as qmatmul-q4k-gpu!, but the 32-element sub-block dot is computed
;; with par/dp4a over int32-packed lanes. Weights uploaded as int32 (wq nibbles reinterpreted
;; little-endian): one int read yields 4 low nibbles (wi & 0x0F0F0F0F → elements k..k+3) AND
;; 4 high nibbles ((wi>>4) & 0x0F0F0F0F → elements k+16..k+19), the llama.cpp mmvq trick.
;; Activation xq packed element-order to int32 (xp). Nibbles 0..15 are positive int8, so the
;; signed dp4a equals the unsigned-nibble × signed-act dpbusd of the composable kernel.
(deftm qmatmul-q4k-dp4a!
  [xp :- (Array int), xs :- (Array float), bsums :- (Array int),
   wp :- (Array int), da :- (Array float), db :- (Array float),
   aq :- (Array byte), bq :- (Array byte),
   y :- (Array float), in :- Long, out :- Long] :- Void
  (par/map-void! o out
                 (let [nsb (quot (long in) 256)
                       wiw (quot (long in) 8)           ; weight int32 words per row (in/2 bytes / 4)
                       wb (* (long o) wiw)
                       sbb (* (long o) nsb)
                       subb (* (long o) (quot (long in) 32))
                       acc (loop [sb 0 a 0.0]
                             (if (< sb nsb)
                               (let [dav (double (ra/aget da (+ sbb sb)))
                                     dbv (double (ra/aget db (+ sbb sb)))
                                     dact (double (ra/aget xs sb))
                                     wsb (+ wb (* sb 32))   ; 32 int words / super-block
                                     xsb (* sb 64)          ; 64 act words / super-block
                                     ssum (loop [j 0 s 0.0]
                                            (if (< j 8)
                                              (let [sidx (+ (* sb 8) j)
                                                    aj (* dav (long (ra/aget aq (+ subb sidx))))
                                                    bj (* dbv (long (ra/aget bq (+ subb sidx))))
                                                    wj (+ wsb (* j 4)) xj (+ xsb (* j 8))
                                                    dp (loop [r 0 d 0]
                                                         (if (< r 4)
                                                           (let [wi (ra/aget wp (+ wj r))
                                                                 lo (bit-and wi 0x0F0F0F0F)
                                                                 hi (bit-and (bit-shift-right wi 4) 0x0F0F0F0F)
                                                                 d1 (par/dp4a lo (ra/aget xp (+ xj r)) d)
                                                                 d2 (par/dp4a hi (ra/aget xp (+ xj 4 r)) d1)]
                                                             (recur (inc r) d2))
                                                           d))]
                                                (recur (inc j) (+ s (* aj (double dp)) (* bj (double (ra/aget bsums sidx))))))
                                              s))]
                                 (recur (inc sb) (+ a (* dact ssum))))
                               a))]
                   (ra/aset y o (float acc)))))

;; Q6_K work-item-per-row twin of qmatmul-q6k-composable! (symmetric K dot, unsigned+zp32).
(deftm qmatmul-q6k-gpu!
  [xq :- (Array byte), xs :- (Array float), bsums :- (Array int),
   wq :- (Array byte), sc :- (Array byte), ds :- (Array float),
   y :- (Array float), in :- Long, out :- Long] :- Void
  (par/map-void! o out
                 (let [nsb (quot (long in) 256) nsub (quot (long in) 16)
                       wb (* (long o) (long in)) sbb (* (long o) nsb) subb (* (long o) nsub)
                       acc (loop [sb 0 a 0.0]
                             (if (< sb nsb)
                               (let [sbase (* sb 256)
                                     dv (double (ra/aget ds (+ sbb sb)))
                                     dact (double (ra/aget xs sb))
                                     ssum (loop [j 0 s 0.0]
                                            (if (< j 16)
                                              (let [base (+ sbase (* j 16)) sidx (+ (* sb 16) j)
                                                    scj (long (ra/aget sc (+ subb sidx)))
                                                    dp (loop [k 0 d 0]
                                                         (if (< k 16)
                                                           (recur (inc k)
                                                                  (+ d (* (bit-and (long (ra/aget wq (+ wb base k))) 0xFF)
                                                                          (long (ra/aget xq (+ base k))))))
                                                           d))
                                                    folded (- dp (* 32 (long (ra/aget bsums sidx))))]
                                                (recur (inc j) (+ s (* scj (double folded)))))
                                              s))]
                                 (recur (inc sb) (+ a (* dv dact ssum))))
                               a))]
                   (ra/aset y o (float acc)))))

;; ---- Q6_K dp4a GPU kernel ----
;; Symmetric K dot via par/dp4a. Q6_K weights are full bytes (no nibble split), so they pack
;; directly to int32 element-order (wp) like the activation (xp); the 16-element sub-block dot
;; is 4 dp4a calls. 6-bit weights are 0..63 (positive int8), so signed dp4a == the composable's
;; unsigned-weight × signed-act dpbusd. The zp32 fold (dp − 32·bsum) and per-subblock int8
;; scale stay as in qmatmul-q6k-gpu!.
(deftm qmatmul-q6k-dp4a!
  [xp :- (Array int), xs :- (Array float), bsums :- (Array int),
   wp :- (Array int), sc :- (Array byte), ds :- (Array float),
   y :- (Array float), in :- Long, out :- Long] :- Void
  (par/map-void! o out
                 (let [nsb (quot (long in) 256)
                       niw (quot (long in) 4)          ; int32 words per row (in bytes / 4)
                       wb (* (long o) niw)
                       sbb (* (long o) nsb)
                       subb (* (long o) (quot (long in) 16))
                       acc (loop [sb 0 a 0.0]
                             (if (< sb nsb)
                               (let [dv (double (ra/aget ds (+ sbb sb)))
                                     dact (double (ra/aget xs sb))
                                     wsb (+ wb (* sb 64))   ; 64 int words / super-block (256 elems / 4)
                                     xsb (* sb 64)
                                     ssum (loop [j 0 s 0.0]
                                            (if (< j 16)
                                              (let [sidx (+ (* sb 16) j)
                                                    scj (long (ra/aget sc (+ subb sidx)))
                                                    wj (+ wsb (* j 4)) xj (+ xsb (* j 4))  ; 4 words / 16-sub-block
                                                    dp (loop [r 0 d 0]
                                                         (if (< r 4)
                                                           (recur (inc r) (par/dp4a (ra/aget wp (+ wj r)) (ra/aget xp (+ xj r)) d))
                                                           d))
                                                    folded (- dp (* 32 (long (ra/aget bsums sidx))))]
                                                (recur (inc j) (+ s (* scj (double folded)))))
                                              s))]
                                 (recur (inc sb) (+ a (* dv dact ssum))))
                               a))]
                   (ra/aset y o (float acc)))))

(ns raster.dl.qlinear-k
  "Composable K-quant GEMV kernels — the SAME deftm compiles to CPU-C (gcc auto-vectorizes
  the dpbusd-able inner loop) and, via the shared c_emit, to OpenCL/GPU. The scalar per-row
  dot mirrors the validated references (raster.compiler.backend.cpu.quant/q4k-q8k-dot,
  q6k-q8k-dot); the fused 8-col-lanes peak SIMD is the orthogonal #27 optimization serving
  every format. Q4_K (asymmetric, 8×32) + Q6_K (symmetric, 16×16) over the format registry.

  Weight matrix [out×in] is quantized per-row (each row independent); for in a multiple of
  256 each row spans whole super-blocks, so the flat per-row arrays index as (row, block)."
  (:require [raster.core :refer [deftm]]
            [raster.arrays :as ra]))

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

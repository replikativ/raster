(ns raster.compiler.backend.cpu.quant
  "Quantized matmul codegen for the native CPU backend — the principled,
  no-hardcoded-kernel design distilled from Futhark / MLIR / ggml / XLA:

    qmatmul = par/map rows · par/reduce blocks ·
                scale · widening-i8-dot(act_i8, unpack(W, FORMAT-DESCRIPTOR))

  Three composable parts, NOT N hand-written kernels:
   - FORMAT = DATA: a layout descriptor (block size, weight bits, packing,
     zero-point, symmetric?) drives the unpack codegen. Q4_0 is one data row.
   - The MAC = ONE primitive `widening-i8-dot`, lowered per ISA (`:maddubs` AVX2 /
     `:scalar` fallback; `:vnni`/`:dpas` later). The single irreducible seam.
   - SCHEDULE: deferred per-block scale, float-vector accumulation, one reduce/row.

  The emitter assembles these → the same optimal C llama.cpp hand-writes per
  format×ISA, from one descriptor + one primitive."
  (:require [clojure.string :as str]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.backend.cpu.codegen :as cpu]))

;; ---- FORMAT DESCRIPTORS (data) — one row per quant level ----
;; The unifying trick: encode weights UNSIGNED with a zero-point (q in [0,2^bits),
;; zp = 2^(bits-1)), so the kernel's u8*s8 dpbusd + (-zp*Σact) fold reconstructs
;; dot = d_w*d_x*(dpbusd(q_w,q_x) - zp*Σact) for ANY bit width — 4-bit and 8-bit
;; share one MAC core (only the unpack: nibble vs byte). The legacy _0 family is a
;; clean parametric {bits, pack}; K-quants (Q4_K…) need per-format unpack/scale-decode.
(def q4-0
  "Symmetric 4-bit, block 32, two nibbles/byte interleaved (lo=w[k], hi=w[k+16]),
  zero-point 8. (= GGUF Q4_0 quant level.)"
  {:name "q4_0" :block 32 :weight-bits 4 :pack :nibble-interleaved
   :zero-point 8 :symmetric true :act-type :i8})

(def q8-0
  "Symmetric 8-bit, block 32, one unsigned byte/weight, zero-point 128. raster's
  dpbusd-compatible 8-bit (unsigned-offset, not GGUF's signed Q8_0 — functionally
  equivalent, same accuracy). The clean parametric proof: same kernel as q4-0 modulo
  byte-direct unpack."
  {:name "q8_0" :block 32 :weight-bits 8 :pack :byte-direct
   :zero-point 128 :symmetric true :act-type :i8})

;; ---- encode side: ONE parametric quantizer (Q4_0 is the default data row) ----
(defn quantize-weight
  "f32 weight (len % block = 0) → {:wq byte[] :ws float[]}: unsigned-offset symmetric
  quantization parameterized by the FORMAT (weight-bits, zero-point, pack). Per block,
  d = max/(-zp); the unsigned weight q = clamp(round(w/d)+zp, 0, 2^bits-1). :nibble-
  interleaved packs two weights/byte (w[k] low, w[k+half] high); :byte-direct one/byte.
  The bit width changes only the pack + range; the dpbusd kernel core is shared."
  [^floats w {:keys [block weight-bits zero-point pack]}]
  (let [len (alength w) blk (long block) nb (quot len blk) half (quot blk 2)
        zp (long zero-point) qmax (dec (bit-shift-left 1 (long weight-bits)))
        ws (float-array nb)
        wq (byte-array (if (= pack :nibble-interleaved) (quot len 2) len))]
    (dotimes [b nb]
      (let [base (* b blk)
            mx (loop [j 0 a 0.0 v 0.0]
                 (if (< j blk)
                   (let [x (aget w (+ base j)) ax (Math/abs x)]
                     (if (> ax a) (recur (inc j) ax x) (recur (inc j) a v)))
                   v))
            d (/ mx (double (- zp))) id (if (zero? d) 0.0 (/ 1.0 d))
            q (fn [x] (max 0 (min qmax (+ (Math/round (* (double x) id)) zp))))]
        (aset ws b (float d))
        (if (= pack :nibble-interleaved)
          (dotimes [k half]
            (aset wq (+ (* b half) k)
                  (unchecked-byte (bit-or (q (aget w (+ base k)))
                                          (bit-shift-left (q (aget w (+ base k half))) 4)))))
          (dotimes [k blk]
            (aset wq (+ base k) (unchecked-byte (q (aget w (+ base k)))))))))
    {:wq wq :ws ws}))

(defn quantize-weight-q4
  "f32 weight → Q4_0 {:wq byte[len/2] :ws float[len/32]}. Thin wrapper over the
  parametric quantize-weight with the q4-0 format."
  [^floats w]
  (quantize-weight w q4-0))

;; ---- K-quants: super-block format (the registry case the legacy params don't cover) ----
;; A K-quant is a SUPER-BLOCK of 256 split into sub-blocks (32 here), each with its OWN
;; scale + min, those per-sub-block scales/mins themselves quantized to 6-bit via two
;; super-block fp scales. Asymmetric: x ≈ (d·aq_j)·q + (dmin·bq_j). This is the same
;; block→scale→int-MAC skeleton, but the unpack + scale-decode are format-specific (the
;; "registry" not the flat {bits,pack} params). raster's own layout (not GGUF-byte-exact;
;; GGUF compat is a loader concern) — proves the pipeline generalizes to super-blocks.
(def q4-K
  "Asymmetric 4-bit super-block: 256 elems / 8 sub-blocks of 32; per-sub-block 6-bit
  scale (aq, unsigned) + 6-bit min (bq, signed); two fp super-scales d (of scales) and
  dmin (of mins). Reconstruct x = (d·aq)·q + (dmin·bq)."
  {:name "q4_K" :super-block 256 :subblock 32 :n-subblocks 8
   :weight-bits 4 :scale-bits 6 :has-dmin true :pack :nibble-interleaved :act-type :i8-K})

(defn quantize-weight-q4k
  "f32 weight (len % super-block = 0) → the q4-K super-block format:
   {:wq byte[len/2] (nibbles, w[k] low / w[k+16] high per sub-block)
    :da float[nsb] :db float[nsb]   (super-block scale-of-scales / scale-of-mins)
    :aq byte[nsub] :bq byte[nsub]}  (per-sub-block 6-bit scale 0..63 / min -32..31).
   Weights are quantized against the QUANTIZED sub-scales so encode/decode are consistent."
  [^floats w {:keys [super-block subblock]}]
  (let [len (alength w) sbk (long super-block) sub (long subblock)
        subs-per (quot sbk sub) nsb (quot len sbk) nsub (quot len sub) half (quot sub 2)
        wq (byte-array (quot len 2))
        da (float-array nsb) db (float-array nsb)
        aq (byte-array nsub) bq (byte-array nsub)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk)
            as (double-array subs-per) bs (double-array subs-per)]
        ;; 1. per sub-block scale a = (max-min)/15 and min b
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub))
                lo (loop [i 1 m (aget w base)] (if (< i sub) (recur (inc i) (min m (aget w (+ base i)))) m))
                hi (loop [i 1 m (aget w base)] (if (< i sub) (recur (inc i) (max m (aget w (+ base i)))) m))]
            (aset as j (/ (- hi lo) 15.0)) (aset bs j lo)))
        ;; 2. super-block scales-of-scales, quantize sub scales/mins to 6-bit
        (let [amax (areduce as i m 0.0 (max m (aget as i)))
              bmax (areduce bs i m 0.0 (max m (Math/abs (aget bs i))))
              dav (/ amax 63.0) dbv (/ bmax 31.0)
              ida (if (zero? dav) 0.0 (/ 1.0 dav))
              idb (if (zero? dbv) 0.0 (/ 1.0 dbv))]
          (aset da sb (float dav)) (aset db sb (float dbv))
          (dotimes [j subs-per]
            (let [aqj (max 0 (min 63 (Math/round (* (aget as j) ida))))
                  bqj (max -32 (min 31 (Math/round (* (aget bs j) idb))))
                  a' (* dav aqj) b' (* dbv bqj)
                  ia' (if (zero? a') 0.0 (/ 1.0 a'))
                  base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
              (aset aq sidx (unchecked-byte aqj))
              (aset bq sidx (unchecked-byte bqj))
              ;; 3. quantize weights against the QUANTIZED sub scale/min (consistency)
              (dotimes [k half]
                (let [q0 (max 0 (min 15 (Math/round (* (- (aget w (+ base k)) b') ia'))))
                      q1 (max 0 (min 15 (Math/round (* (- (aget w (+ base k half)) b') ia'))))]
                  (aset wq (+ (quot base 2) k)
                        (unchecked-byte (bit-or q0 (bit-shift-left q1 4)))))))))))
    {:wq wq :da da :db db :aq aq :bq bq}))

(defn dequant-q4k
  "q4-K super-block → f32[len]: x = (da·aq)·q + (db·bq) per sub-block. The inverse of
  quantize-weight-q4k (raster's own layout); the reference the fused kernel matches."
  [{:keys [wq da db aq bq]} {:keys [super-block subblock]} len]
  (let [out (float-array len) sbk (long super-block) sub (long subblock)
        subs-per (quot sbk sub) nsb (quot len sbk) half (quot sub 2)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) dav (aget da sb) dbv (aget db sb)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                a' (* dav (aget aq sidx))   ;; aq in [0,63] (positive byte)
                b' (* dbv (aget bq sidx))]  ;; bq signed
            (dotimes [k half]
              (let [bv (bit-and (long (aget wq (+ (quot base 2) k))) 0xFF)]
                (aset out (+ base k)      (float (+ (* a' (bit-and bv 0xF)) b')))
                (aset out (+ base k half) (float (+ (* a' (bit-shift-right bv 4)) b')))))))))
    out))

(defn quantize-act-q8k
  "f32 activation [in] (in % super-block = 0) → q8_K for the K-quant dot: per super-block
  int8 quants + scale d=max/127, plus bsums[nsub] = per-sub-block int sum of the quants
  (the K min-term needs Σ activation per sub-block, precomputed)."
  [^floats x in {:keys [super-block subblock]}]
  (let [sbk (long super-block) sub (long subblock)
        nsb (quot (long in) sbk) nsub (quot (long in) sub) subs-per (quot sbk sub)
        xq (byte-array in) xs (float-array nsb) bsums (int-array nsub)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk)
            mx (loop [i 0 m 0.0] (if (< i sbk) (recur (inc i) (max m (Math/abs (aget x (+ sbase i))))) m))
            d (/ mx 127.0) id (if (zero? d) 0.0 (/ 1.0 d))]
        (aset xs sb (float d))
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
            (aset bsums sidx (int (loop [k 0 s 0]
                                    (if (< k sub)
                                      (let [qv (max -127 (min 127 (Math/round (* (aget x (+ base k)) id))))]
                                        (aset xq (+ base k) (unchecked-byte qv)) (recur (inc k) (+ s (int qv))))
                                      s))))))))
    {:xq xq :xs xs :bsums bsums}))

(defn q4k-q8k-dot
  "Reference dot of ONE Q4_K weight row · a q8_K activation — the K-quant compute the
  fused kernel will SIMD-ize. Per super-block: d_act·[Σ_sub (d·aq)·dpbusd + (dmin·bq)·bsum],
  where dpbusd = Σ_i q_w·q_act (the int8-MAC, dpbusd-able) and bsum is the precomputed
  per-sub-block activation sum. Equals dot(dequant weight, dequant act) exactly."
  [{:keys [wq da db aq bq]} {:keys [xq xs bsums]} in {:keys [super-block subblock]}]
  (let [sbk (long super-block) sub (long subblock) half (quot sub 2)
        nsb (quot (long in) sbk) subs-per (quot sbk sub)]
    (loop [sb 0 acc 0.0]
      (if (< sb nsb)
        (let [sbase (* sb sbk) dav (aget da sb) dbv (aget db sb) dact (double (aget xs sb))
              sbsum (loop [j 0 s 0.0]
                      (if (< j subs-per)
                        (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                              a' (* dav (aget aq sidx)) b' (* dbv (aget bq sidx))
                              dp (loop [k 0 d 0]
                                   (if (< k half)
                                     (let [bv (bit-and (long (aget wq (+ (quot base 2) k))) 0xFF)]
                                       (recur (inc k)
                                              (+ d (* (bit-and bv 0xF) (long (aget xq (+ base k))))
                                                 (* (bit-shift-right bv 4) (long (aget xq (+ base k half)))))))
                                     d))]
                          (recur (inc j) (+ s (* a' (double dp)) (* b' (double (aget bsums sidx))))))
                        s))]
          (recur (inc sb) (+ acc (* dact sbsum))))
        acc))))

;; ---- Q6_K: the 2nd K-quant — a DIFFERENT variant, to confirm the registry
;; generalizes across K-quants (not just one). Q6_K is SYMMETRIC (d only, no min/dmin),
;; 16×16 sub-blocks (vs Q4_K's 8×32), int8 sub-scales (vs Q4_K's 6-bit-packed), 6-bit
;; weights. Reuses the SAME skeleton + the SAME q8_K activation (parameterized to 16-elem
;; sub-blocks). The signed 6-bit weight rides the unsigned+zero-point trick (zp=32) so the
;; dpbusd core is shared. (raster stores 6-bit/byte; GGUF's 4+2 plane is a packing detail.)
(def q6-K
  "Symmetric 6-bit super-block: 256 / 16 sub-blocks of 16; per-sub-block int8 scale,
  one fp super-scale d, no min. x = (d·sc)·(q-32), q unsigned 6-bit [0,63]."
  {:name "q6_K" :super-block 256 :subblock 16 :n-subblocks 16
   :weight-bits 6 :scale-bits 8 :has-dmin false :zero-point 32
   :pack :byte-direct :act-type :i8-K})

(defn quantize-weight-q6k
  "f32 weight → q6-K: {:wq byte[len] (unsigned 6-bit) :sc byte[nsub] (int8 sub-scale)
   :ds float[nsb] (super-scale)}. Symmetric: per sub-block a=max|x|/32, super d=max(a)/127."
  [^floats w {:keys [super-block subblock zero-point]}]
  (let [len (alength w) sbk (long super-block) sub (long subblock) zp (long zero-point)
        subs-per (quot sbk sub) nsb (quot len sbk) nsub (quot len sub) qmax 63
        wq (byte-array len) sc (byte-array nsub) ds (float-array nsb)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) as (double-array subs-per)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub))
                mx (loop [i 0 m 0.0] (if (< i sub) (recur (inc i) (max m (Math/abs (aget w (+ base i))))) m))]
            (aset as j (/ mx 32.0))))
        (let [amax (areduce as i m 0.0 (max m (aget as i)))
              dv (/ amax 127.0) idv (if (zero? dv) 0.0 (/ 1.0 dv))]
          (aset ds sb (float dv))
          (dotimes [j subs-per]
            (let [scj (max 0 (min 127 (Math/round (* (aget as j) idv))))
                  a' (* dv scj) ia' (if (zero? a') 0.0 (/ 1.0 a'))
                  base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)]
              (aset sc sidx (unchecked-byte scj))
              (dotimes [k sub]
                (aset wq (+ base k)
                      (unchecked-byte (max 0 (min qmax (+ (Math/round (* (aget w (+ base k)) ia')) zp)))))))))))
    {:wq wq :sc sc :ds ds}))

(defn dequant-q6k
  "q6-K → f32[len]: x = (ds·sc)·(q-zp) per sub-block."
  [{:keys [wq sc ds]} {:keys [super-block subblock zero-point]} len]
  (let [out (float-array len) sbk (long super-block) sub (long subblock) zp (long zero-point)
        subs-per (quot sbk sub) nsb (quot len sbk)]
    (dotimes [sb nsb]
      (let [sbase (* sb sbk) dv (aget ds sb)]
        (dotimes [j subs-per]
          (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                a' (* dv (long (aget sc sidx)))]
            (dotimes [k sub]
              (aset out (+ base k)
                    (float (* a' (- (bit-and (long (aget wq (+ base k))) 0xFF) zp)))))))))
    out))

(defn q6k-q8k-dot
  "Reference dot of ONE Q6_K weight row · q8_K activation: per super-block d·d_act·Σ_sub
  sc·(dpbusd - zp·bsum). Symmetric (no min term); the signed 6-bit weight via the unsigned
  +zp(32) fold reuses the shared dpbusd core. Equals dot(dequant, dequant) exactly."
  [{:keys [wq sc ds]} {:keys [xq xs bsums]} in {:keys [super-block subblock zero-point]}]
  (let [sbk (long super-block) sub (long subblock) zp (long zero-point)
        nsb (quot (long in) sbk) subs-per (quot sbk sub)]
    (loop [sb 0 acc 0.0]
      (if (< sb nsb)
        (let [sbase (* sb sbk) dv (double (aget ds sb)) dact (double (aget xs sb))
              ssum (loop [j 0 s 0.0]
                     (if (< j subs-per)
                       (let [base (+ sbase (* j sub)) sidx (+ (* sb subs-per) j)
                             scj (long (aget sc sidx))
                             dp (loop [k 0 d 0]
                                  (if (< k sub)
                                    (recur (inc k) (+ d (* (bit-and (long (aget wq (+ base k))) 0xFF)
                                                           (long (aget xq (+ base k))))))
                                    d))
                             folded (- dp (* zp (long (aget bsums sidx))))]
                         (recur (inc j) (+ s (* scj (double folded)))))
                       s))]
          (recur (inc sb) (+ acc (* dv dact ssum))))
        acc))))

(defn- quantize-act-blocks!
  "Quantize the 32-element blocks [r0, r0+cnt) of activation x into xq/xs/xsum
  (per-block symmetric int8, d=max/127, with the per-block sum for the -8 fold).
  Blocks are independent — this is the parallel work unit."
  [^floats x ^bytes xq ^floats xs ^ints xsum r0 cnt]
  (dotimes [i (long cnt)]
    (let [r (+ (long r0) i) base (* r 32)
          mx (loop [j 0 a 0.0] (if (< j 32) (recur (inc j) (max a (Math/abs (aget x (+ base j))))) a))
          d (/ mx 127.0) id (if (zero? d) 0.0 (/ 1.0 d))]
      (aset xs r (float d))
      (aset xsum r (int (loop [k 0 s 0]
                          (if (< k 32)
                            (let [q (max -127 (min 127 (Math/round (* (aget x (+ base k)) id))))]
                              (aset xq (+ base k) (unchecked-byte q)) (recur (inc k) (+ s (int q))))
                            s)))))))

(defn quantize-act-i8
  "f32 activation row(s) [M*in] → {:xq byte[M*in] :xs float[M*nb] :xsum int[M*nb]}
  per-block (32) symmetric int8, d=max/127, with per-block sum for the -8 correction."
  [^floats x M in]
  (let [nb (quot in 32)
        xq (byte-array (alength x)) xs (float-array (* M nb)) xsum (int-array (* M nb))]
    (quantize-act-blocks! x xq xs xsum 0 (* M nb))
    {:xq xq :xs xs :xsum xsum}))

;; ---- per-ISA pieces of the widening-i8-dot primitive (the one seam) ----
;; :maddubs names the x86 SIMD lowering, but the emitted `_i8dot` is a PORTABLE
;; macro cascade: under -march=native the host's best int8-MAC is selected at
;; compile time — AVX-VNNI `dpbusd` (1 instr) where available, baseline AVX2
;; `maddubs+madd` otherwise. No runtime CPUID needed for the JIT-on-host model.
;; (Other ISAs — NEON sdot, SVE — are added as more #elif arms; same descriptor,
;; same schedule, different MAC spelling. A shipped fat binary would multi-version.)
(def ^:private stream-gemv-layout
  (layout/quant-stream-layout {:vector-bits 256} q4-0)) ; AVX2 NC=8, format-derived

(defn repack-stream
  "Row-major Q4_0 {wq[out*in/2], ws[out*nb]} → the :stream-gemv interleaved layout,
  where output column i lands in accumulator lane i (no shuffles, no permute). Single
  source: delegates to the descriptor-driven core.layout/repack, which the composable
  x8 GEMV (raster.quant.kernels/qmatmul-q4-x8!) mirrors. out must be a multiple of
  NC=8. One-time cost."
  [^bytes wq ^floats ws out in]
  (layout/repack stream-gemv-layout wq ws out in))

(definterface IParTask (^void runSlice [^int wid ^int n]))

(def ^:private pool-threads
  (max 1 (or (some-> (System/getenv "RASTER_DECODE_THREADS") Integer/parseInt) 1)))

;; Workers spin-poll for new work this many iterations, THEN park (sleep). During
;; active decode, dispatches arrive faster than this window, so workers stay spinning
;; (low latency); when decode stops/idle they park — no idle core-burn, and robust
;; under contention (a busy machine doesn't get 4 cores pinned by a dormant pool).
(def ^:private spin-limit 100000)

(defonce ^:private the-pool
  (delay
    (let [n     pool-threads
          gen   (java.util.concurrent.atomic.AtomicLong. 0)
          arrive (java.util.concurrent.atomic.AtomicLong. 0)
          task  (java.util.concurrent.atomic.AtomicReference.)
          threads (object-array (dec n))]   ;; worker handles, for unpark
      (dotimes [w (dec n)]
        (let [wid (inc w)
              th (Thread.
                  (fn []
                    (let [g (long-array 1)]
                      (loop []
                        (let [cur (.get gen)]
                          (if (== cur (aget g 0))
                            ;; no work: spin a bounded window, then park until unparked.
                            ;; park()/unpark() permits make this lost-wakeup-safe (an
                            ;; unpark before park returns immediately).
                            (do (loop [s 0]
                                  (when (and (< s spin-limit) (== (.get gen) (aget g 0)))
                                    (Thread/onSpinWait) (recur (inc s))))
                                (when (== (.get gen) (aget g 0))
                                  (java.util.concurrent.locks.LockSupport/park))
                                (recur))
                            (do (aset g 0 cur)
                                (.runSlice ^IParTask (.get task) wid n)
                                (.incrementAndGet arrive)
                                (recur))))))))]
          (aset threads w th)
          (doto th (.setDaemon true) (.setName (str "raster-decode-" wid)) (.start))))
      {:gen gen :arrive arrive :task task :n n :threads threads})))

(defn run-par!
  "Run `t` across the persistent pool: worker i of n runs (.runSlice t i n). The
  caller is worker 0; it busy-waits on the spin-barrier until the n-1 workers finish."
  [^IParTask t]
  (let [{:keys [^java.util.concurrent.atomic.AtomicLong gen
                ^java.util.concurrent.atomic.AtomicLong arrive
                ^java.util.concurrent.atomic.AtomicReference task n
                ^objects threads]} @the-pool]
    (if (== 1 (int n))
      (.runSlice t 0 1)
      (do (.set arrive 0)
          (.set task t)            ; volatile store, visible before the gen bump
          (.incrementAndGet gen)   ; full fence — releases the spinning workers
          (dotimes [w (dec (int n))]   ; wake any parked workers
            (java.util.concurrent.locks.LockSupport/unpark (aget threads w)))
          (.runSlice t 0 (int n))  ; main = worker 0
          (let [need (dec (long n))]
            (while (< (.get arrive) need) (Thread/onSpinWait)))))))

(defn run-par-fn!
  "Public pooling entry for consumers (e.g. the decode loop's attention): runs
  (f wid n) on every worker of the persistent pool, f writing a disjoint slice.
  Serial at 1 thread."
  [f]
  (if (== 1 (int pool-threads))
    (f 0 1)
    (run-par! (reify IParTask (runSlice [_ wid n] (f wid n))))))

(defn quantize-act-i8-par
  "Pooled M=1 quantize-act-i8: the nb blocks are independent, so split them across
  the SAME persistent pool the matmul uses — the otherwise-idle workers do the
  quantization instead of spinning. Falls back to serial at 1 thread."
  [^floats x in]
  (let [nb (quot (int in) 32)
        xq (byte-array (int in)) xs (float-array nb) xsum (int-array nb)]
    (if (== 1 (int pool-threads))
      (quantize-act-blocks! x xq xs xsum 0 nb)
      (run-par!
       (reify IParTask
         (runSlice [_ wid n]
           (let [chunk (quot (+ nb (dec (int n))) (int n))
                 r0 (* (int wid) chunk)
                 cnt (max 0 (min chunk (- nb r0)))]
             (when (pos? cnt) (quantize-act-blocks! x xq xs xsum r0 cnt)))))))
    {:xq xq :xs xs :xsum xsum}))

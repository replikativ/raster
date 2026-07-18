(ns raster.runtime.microbench
  "Stationary-machine hardware CALIBRATION — measures what a probe can't report (memory bandwidth,
   effective peak-flops, kernel launch overhead, cache knees) and fills the descriptor's :measured
   layer. The analytic model (raster.compiler.core.hardware) proposes; measurement here disposes and
   OVERWRITES the readable descriptor constants (the loop XLA/Triton/Inductor leave open — they cache
   measurements beside a stale model; raster feeds them back into the model).

   Methodology (do-bench): warmup, iteration count derived from a TIME budget, cache-flush between
   timed iterations, report MIN (autotune) and MEDIAN (report). The stationary-machine assumption is
   ASSERTED, not hoped: the coefficient of variation across samples must be below a threshold, else
   the measurement is flagged non-stationary (thermal/contention) and the field stays low-confidence.

   Provenance: every :measured field is tagged :measured; a caller reading the descriptor always
   knows whether a number is measured-and-stationary or a probed/catalogue/analytic guess."
  (:require [raster.runtime.hardware :as rt]))

;; ================================================================
;; do-bench — the measurement primitive
;; ================================================================

(defn do-bench
  "Time `thunk` under the do_bench discipline. Returns
     {:median-ns :min-ns :mean-ns :cv :n :stationary?}.
   opts: :warmup-ms (25) :budget-ms (100) :flush-fn (nil) :cv-threshold (0.05)."
  [thunk & {:keys [warmup-ms budget-ms flush-fn cv-threshold]
            :or {warmup-ms 25 budget-ms 100 cv-threshold 0.05}}]
  ;; warmup — let the JIT compile + clocks spin up (Halide's thermal-settle, in-band)
  (let [warm-end (+ (System/nanoTime) (* (long warmup-ms) 1000000))]
    (while (< (System/nanoTime) warm-end) (thunk)))
  ;; estimate single-call cost → derive an iteration count that fits the time budget
  (let [t0 (System/nanoTime)
        _  (dotimes [_ 5] (thunk))
        est (/ (double (- (System/nanoTime) t0)) 5.0)
        n  (max 3 (min 100000 (long (/ (* (double budget-ms) 1.0e6) (max 1.0 est)))))
        times (double-array n)]
    (dotimes [i n]
      (when flush-fn (flush-fn))
      (let [s (System/nanoTime)]
        (thunk)
        (aset times i (double (- (System/nanoTime) s)))))
    (let [sorted (double-array (sort (seq times)))
          med  (aget sorted (quot n 2))
          mn   (aget sorted 0)
          mean (/ (areduce times i acc 0.0 (+ acc (aget times i))) (double n))
          var  (/ (areduce times i acc 0.0 (let [d (- (aget times i) mean)] (+ acc (* d d)))) (double n))
          cv   (if (pos? mean) (/ (Math/sqrt var) mean) 0.0)]
      {:median-ns med :min-ns mn :mean-ns mean :cv cv :n n
       :stationary? (< cv (double cv-threshold))})))

;; ================================================================
;; Cache-flush scratch (evict the LLC between timed iterations)
;; ================================================================

(defn make-flush [llc-bytes]
  (let [n (max 1 (quot (long (* 2 llc-bytes)) 8))
        scratch (double-array n)]
    (fn [] (java.util.Arrays/fill scratch 0.0))))

;; ================================================================
;; CPU memory bandwidth — STREAM triad
;; ================================================================

(defn measure-cpu-bandwidth
  "Single-thread STREAM-triad DRAM bandwidth (bytes/s): a[i] = b[i] + s·c[i] over arrays sized WELL
   ABOVE the LLC so every element misses cache. STREAM convention: 3 arrays × 8 B = 24 B/element
   moved (2 reads + 1 write). Returns the do-bench stats + :bandwidth-bytes-s."
  [& {:keys [llc-bytes] :or {llc-bytes (* 32 1024 1024)}}]
  (let [n (int (max 4000000 (* 4 (quot (long llc-bytes) 8))))  ;; each array ≫ LLC
        a (double-array n) b (double-array n) c (double-array n)]
    (dotimes [i n] (aset ^doubles b i (double i)) (aset ^doubles c i (double (* 2 i))))
    (let [scale 3.0
          triad (fn [] (let [^doubles a a ^doubles b b ^doubles c c]
                         (dotimes [i n]
                           (aset a i (+ (aget b i) (* scale (aget c i)))))))
          r (do-bench triad :budget-ms 200 :cv-threshold 0.08)
          bytes-moved (* 3.0 (double n) 8.0)
          bw (/ bytes-moved (/ (:median-ns r) 1.0e9))]
      (assoc r :bandwidth-bytes-s bw :bytes-moved bytes-moved :n-elements n))))

;; ================================================================
;; Calibration — run the benches, store the :measured layer
;; ================================================================

(defn calibrate-cpu!
  "Measure the CPU (:cpu:0) and store a :measured map in the hardware registry, so descriptor-for
   projects it (measured OVERRIDES catalogue/analytic). MEASURE-ONCE: reuses a matching on-disk
   calibration unless :force? is set; a fresh measurement is persisted (keyed by device signature ×
   version). Returns the :measured map."
  [& {:keys [device-id force?] :or {device-id :cpu:0}}]
  (rt/init!)
  (or (when-not force? (rt/load-calibration! device-id))
      (let [caps (:capabilities (rt/device device-id))
            llc  (or (:cache-l3 caps) (* 32 1024 1024))
            bw   (measure-cpu-bandwidth :llc-bytes llc)
            measured {:bandwidth-bytes-s (:bandwidth-bytes-s bw)
                      :provenance {:bandwidth-bytes-s (if (:stationary? bw) :measured :measured-noisy)}
                      :bench {:bandwidth (dissoc bw :n)}}]
        (rt/set-measured! device-id measured)
        (rt/save-calibration! device-id measured)
        measured)))

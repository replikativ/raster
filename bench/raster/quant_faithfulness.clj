(ns raster.quant-faithfulness
  "Faithfulness + perf regression harness for the quantized Q4_0 GEMV.

  The original job of this harness — gate moving the quantized GEMV OUT of a
  hand-written C string emitter INTO raster's composable IR — is DONE: the
  composable `qmatmul-q4-x8!` (lowered via compile-aot-c :simd? true, the csimd
  int->float widening column fold) matches the retired hand kernel bit-exact and
  within noise on latency (#27), and the hand-string emitter has been deleted.

  So this is now a REGRESSION harness for the production composable path: every
  shape is checked against an independent scalar dequant-dot oracle (correctness —
  a quantized matmul that is 'fast but wrong' is worthless) and timed (perf — the
  composable lowering must keep reproducing optimal machine code, not regress).
  The row-major `qmatmul-q4-composable` is included as the scalar-fold reference
  the SIMD x8 path must stay well ahead of.  Run: (run) or (run :shapes ...)."
  (:require [raster.compiler.backend.cpu.quant :as cq]
            [raster.quant.kernels :as k]))

(def decode-shapes
  "[in out label] — real decode matmul shapes across model sizes (in%32=0, out%8=0)."
  [[640 1024  "270m attn-q"]
   [2048 640  "270m ffn-down"]
   [1152 6912 "1b ffn-up"]
   [2048 11008 "3b ffn-up"]
   [4096 4096 "square-4k"]])

;; ---- candidates: each (fn [wf in out] -> (fn [^floats x] -> ^floats y)) ----

(defn cand-x8
  "Production path: composable qmatmul-q4-x8! (compile-aot-c :simd? true), pool-driven
  over 8-column groups on the interleaved repack-stream layout. This is what
  raster.quant.op/qlinear-i8 uses."
  [wf in out]
  (let [{:keys [wq ws]} (cq/quantize-weight-q4 wf)
        {:keys [wqi wsi]} (cq/repack-stream wq ws out in)
        drive (k/make-x8-c-gemv)]
    (fn [x]
      (let [{:keys [xq xs xsum]} (cq/quantize-act-i8-par x in)]
        (drive xq xs xsum wqi wsi in out)))))

(defn cand-composable-rowmajor
  "Row-major composable GEMV (scalar-fold reference) — the SIMD x8 path must stay
  well ahead of this."
  [wf in out]
  (let [{:keys [wq ws]} (cq/quantize-weight-q4 wf)
        drive (k/make-composable-c-gemv)]
    (fn [x]
      (let [{:keys [xq xs xsum]} (cq/quantize-act-i8-par x in)]
        (drive xq xs xsum wq ws in out)))))

(def candidates {:x8 cand-x8 :rowmajor cand-composable-rowmajor})

;; ---- correctness oracle: independent scalar dequant-dot (double) ----

(defn scalar-ref
  "Independent reference: y[o] = sum_b xs[b]*ws[o,b] * sum_i xq_i*(nibble-8). Over the
  first `rows` output rows only (full lm-head would be too slow for a correctness spot)."
  [^bytes wq ^floats ws ^bytes xq ^floats xs in rows]
  (let [nb (quot (int in) 32)]
    (mapv (fn [o]
            (let [wrow (* o (quot (int in) 2))]
              (loop [b 0 acc 0.0]
                (if (< b nb)
                  (let [s (loop [k 0 ss 0]
                            (if (< k 16)
                              (let [bv (bit-and (aget wq (+ wrow (* b 16) k)) 0xFF)]
                                (recur (inc k)
                                       (+ ss (* (aget xq (+ (* b 32) k)) (- (bit-and bv 0xF) 8))
                                            (* (aget xq (+ (* b 32) k 16)) (- (bit-shift-right bv 4) 8)))))
                              ss))]
                    (recur (inc b) (+ acc (* (* (double (aget xs b)) (double (aget ws (+ (* o nb) b)))) s))))
                  acc))))
          (range rows))))

;; ---- timing ----

(defn- median-us [f n warm]
  (dotimes [_ warm] (f))
  (let [ts (sort (mapv (fn [_] (let [t (System/nanoTime)] (f) (- (System/nanoTime) t))) (range n)))]
    (/ (double (nth ts (quot n 2))) 1000.0)))

(defn run
  "Correctness (max |diff| vs scalar-ref over the first min(out,512) rows) + median
  latency per candidate on each shape. Returns rows; also prints a table.
  tol = f32 round-off tolerance for correctness."
  [& {:keys [shapes cands tol reps] :or {shapes decode-shapes cands candidates tol 1.0e-3}}]
  (println (format "%-14s %-9s %8s  %10s  %s" "shape" "cand" "us" "maxdiff" "correct?"))
  (vec
   (for [[in out lbl] shapes
         :let [wf (let [a (float-array (* out in))] (dotimes [i (* out in)] (aset a i (float (- (rand) 0.5)))) a)
               x  (let [a (float-array in)] (dotimes [i in] (aset a i (float (- (rand) 0.5)))) a)
               rows (min out 512)
               {:keys [wq ws]} (cq/quantize-weight-q4 wf)
               {:keys [xq xs]} (cq/quantize-act-i8 x 1 in)
               yref (scalar-ref wq ws xq xs in rows)
               n (or reps (if (> out 100000) 15 60))]
         [nm prep] cands]
     (let [f (prep wf in out)
           y (vec (f x))
           dscal (reduce max 0.0 (map (fn [a b] (Math/abs (- (double a) (double b)))) yref (take rows y)))
           us (median-us #(f x) n 25)
           ok (< dscal tol)]
       (println (format "%-14s %-9s %8.0f  %10.6f  %s" lbl (name nm) us (float dscal) ok))
       {:shape lbl :cand nm :us (Math/round us) :maxdiff (float dscal) :correct? ok}))))

(defn -main [& _]
  (run))

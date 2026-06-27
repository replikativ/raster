(ns raster.quant-faithfulness
  "Differential FAITHFULNESS harness for the quantized matmul.

  The hard gate for moving the quantized GEMV from a hand-written string emitter into
  raster's composable IR: ANY candidate lowering (the current hand kernel, the GPU
  kernel, or a future par-form / widening-i8-dot composable kernel) is measured here
  against two oracles on the REAL decode matmul shapes:

    - CORRECTNESS oracle: an independent scalar Clojure dequant-dot (double precision).
      A candidate must be bit-faithful (f32 round-off only) — a quantized matmul that is
      'fast but wrong' is worthless.
    - PERFORMANCE oracle: the handwritten CPU kernel (raster.dl.qlinear/qlinear-i8, the
      maddubs/VNNI spin-pool stream-gemv). A composable lowering is only worth adopting
      if it is within noise of this — top performance is a hard requirement, so the
      compiler must reproduce the optimal machine code, not regress it.

  A candidate is `(fn [wf in out] -> (fn [^floats x] -> ^floats y))` — it prepares its
  own weight layout (repack-stream, row-major, packed-IR, ...) then runs. Register a new
  composable candidate and it is held to the same bar. Run: (run) or (run :shapes ...)."
  (:require [raster.dl.qlinear :as ql]
            [raster.dl.qlinear-gpu :as qg]
            [raster.compiler.backend.cpu.quant :as cq]))

(def decode-shapes
  "[in out label] — real decode matmul shapes across model sizes (in%32=0, out%8=0)."
  [[640 1024  "270m attn-q"]
   [2048 640  "270m ffn-down"]
   [640 262144 "270m lm-head"]
   [1152 6912 "1b ffn-up"]
   [2048 11008 "3b ffn-up"]
   [4096 14336 "7b ffn-up"]
   [14336 4096 "7b ffn-down"]])

;; ---- candidates: each prepares its layout, returns a (fn [x] -> y) ----

(defn cand-cpu-hand
  "The handwritten CPU kernel — the performance oracle (maddubs/VNNI spin-pool)."
  [wf in out]
  (let [{:keys [wq ws]} (cq/quantize-weight-q4 wf)
        {:keys [wqi wsi]} (cq/repack-stream wq ws out in)]
    (fn [x] (ql/qlinear-i8 x wqi wsi in out))))

(defn cand-gpu
  "The GPU OpenCL kernel, resident weights on :ze:0 (different backend; informational)."
  [wf in out]
  (let [{:keys [wq ws]} (cq/quantize-weight-q4 wf)
        W (qg/upload-weight! :ze:0 wq ws in out)]
    (fn [x] (qg/qlinear-i8-gpu W x))))

(defn cand-cpu-composable
  "The composable-IR Q4 GEMV (raster.dl.qlinear-composable), lowered via compile-aot
  :target :c and pool-driven over output-row slices — the candidate we are closing to
  hand-kernel parity. Row-major weight (no repack-stream); activation int8-quantized per
  call. Measures the composable path vs the hand kernel to localize the residual gap."
  [wf in out]
  (let [{:keys [wq ws]} (cq/quantize-weight-q4 wf)
        drive ((requiring-resolve 'raster.dl.qlinear-composable/make-composable-c-gemv))]
    (fn [x]
      (let [{:keys [xq xs xsum]} (cq/quantize-act-i8-par x in)]
        (drive xq xs xsum wq ws in out)))))

(def candidates
  "name -> candidate prep fn. Composable par-form lowerings register here."
  {:cpu-hand       cand-cpu-hand
   :cpu-composable cand-cpu-composable
   :gpu            cand-gpu})

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
  "Run the faithfulness comparison. Returns a vector of per-(shape,candidate) rows:
  {:shape :cand :maxdiff-vs-scalar :maxdiff-vs-hand :us :x-vs-hand :correct? :faithful?}.
  faithful? = correct (f32 round-off) AND within 15% of the hand kernel's latency."
  [& {:keys [shapes cands tol perf-slack]
      :or {shapes decode-shapes cands candidates tol 1.0e-3 perf-slack 1.15}}]
  (vec
   (for [[in out lbl] shapes]
     (let [wf (let [a (float-array (* out in))]
                (dotimes [i (* out in)] (aset a i (float (- (rand) 0.5)))) a)
           x  (let [a (float-array in)] (dotimes [i in] (aset a i (float (- (rand) 0.5)))) a)
           ;; correctness oracle over the first min(out,512) rows
           rows (min out 512)
           {:keys [wq ws]} (cq/quantize-weight-q4 wf)
           {:keys [xq xs]} (cq/quantize-act-i8 x 1 in)
           yref (scalar-ref wq ws xq xs in rows)
           reps (if (> out 100000) 15 60)
           prepared (into {} (map (fn [[nm prep]] [nm (prep wf in out)]) cands))
           y-hand (vec ((:cpu-hand prepared) x))
           hand-us (median-us (fn [] ((:cpu-hand prepared) x)) reps 25)
           per-cand
           (into {}
                 (for [[nm f] prepared]
                   (let [y (vec (f x))
                         dscal (reduce max 0.0 (map (fn [a b] (Math/abs (- (double a) (double b))))
                                                    yref (take rows y)))
                         dhand (reduce max 0.0 (map (fn [a b] (Math/abs (- (double a) (double b))))
                                                    y-hand y))
                         us (median-us (fn [] (f x)) reps 25)
                         correct? (< dscal tol)
                         faithful? (and correct? (<= us (* perf-slack hand-us)))]
                     [nm {:maxdiff-vs-scalar (float dscal) :maxdiff-vs-hand (float dhand)
                          :us (Math/round us) :x-vs-hand (Math/round (/ us hand-us))
                          :correct? correct? :faithful? faithful?}])))]
       {:shape [in out] :label lbl :hand-us (Math/round hand-us) :cands per-cand}))))

(defn -main [& _]
  (doseq [{:keys [label shape hand-us cands]} (run)]
    (println (format "%-16s %-12s hand=%dus" label (pr-str shape) hand-us))
    (doseq [[nm r] cands]
      (println (format "    %-10s %6dus  %2dx-hand  diff(scalar)=%.2g correct=%s faithful=%s"
                       (name nm) (:us r) (:x-vs-hand r) (:maxdiff-vs-scalar r)
                       (:correct? r) (:faithful? r))))))

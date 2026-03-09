(ns raster.mlp-assembly
  "Dump C2 native assembly and timing for the MLP 784-128-10 train step.

  Produces:
    <outdir>/c2_helpers.txt          — helper list with source expressions
    <outdir>/c2_bytecode.txt         — JVM bytecode for all helpers
    <outdir>/c2_native_<helper>.txt  — native x86 assembly per helper
    <outdir>/c2_timing.txt           — micro-benchmark timing (µs/step)
    <outdir>/c2_pipeline_stats.txt   — SIMD stats, fusion stats, hoisted count
    <outdir>/c2_pipeline_stages.txt  — full pipeline stage dump

  Run (Valhalla JDK):
    source valhalla-env.sh
    OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \\
      -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \\
      -J--enable-native-access=ALL-UNNAMED \\
      -J--add-modules=jdk.incubator.vector \\
      -M:bench -m raster.mlp-assembly [outdir]"
  (:require [raster.core :refer [deftm]]
            [raster.nn :as nn]
            [raster.dl.optim :as optim]
            [raster.arrays :as arrays]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pipeline]
            [raster.tooling.inspect :as inspect]
            [clojure.pprint :as pp]
            [clojure.string :as str])
  (:import [java.io File]))

;; ================================================================
;; Train step — identical to mnist_bench.clj
;; ================================================================

(deftm train-step
  [W1 :- (Array double), b1 :- (Array double),
   W2 :- (Array double), b2 :- (Array double),
   x :- (Array double), y :- (Array double),
   lr :- Double] :- Double
  (let [vg ((rev/value+grad (var nn/loss-fn)) W1 b1 W2 b2 x y)
        loss (nth vg 0)]
    (optim/sgd-step! W1 (nth vg 1) (arrays/alength W1) lr)
    (optim/sgd-step! b1 (nth vg 2) (arrays/alength b1) lr)
    (optim/sgd-step! W2 (nth vg 3) (arrays/alength W2) lr)
    (optim/sgd-step! b2 (nth vg 4) (arrays/alength b2) lr)
    loss))

(defn- spit-file [outdir filename content]
  (let [path (str outdir "/" filename)]
    (spit path content)
    (println (str "  wrote " path))))

(defn -main [& args]
  (let [outdir (or (first args) "/tmp/raster_vs_xla")
        _ (.mkdirs (File. ^String outdir))
        _ (print "Compiling... ") _ (flush)
        t0 (System/currentTimeMillis)
        f (pipeline/compile-aot #'train-step :simd? true)
        _ (println (format "done (%.1fs)" (/ (- (System/currentTimeMillis) t0) 1000.0)))
        diag (inspect/compiled-diagnostics #'train-step)
        stages (pipeline/show-pipeline #'train-step)]

    ;; ============================================================
    ;; 1. Pipeline stats
    ;; ============================================================
    (println "Dumping pipeline stats...")
    (spit-file outdir "c2_pipeline_stats.txt"
      (str/join "\n"
        [(str "SIMD stats: " (:backend-applied-stats stages))
         (str "SOAC fusion: " (:soac-fused-stats stages))
         (str "Buffer fusion: " (:buffer-fused-stats stages))
         (str "DCE: " (:dce-cleaned-stats stages))
         (str "Hoisted buffers: " (:hoisted-count diag))
         (str "Helpers: " (count (:helper-map diag)))]))

    ;; ============================================================
    ;; 2. Helper list with source expressions
    ;; ============================================================
    (println "Dumping helper source expressions...")
    (spit-file outdir "c2_helpers.txt"
      (with-out-str
        (doseq [[name info] (sort-by key (:helper-map diag))]
          (println (format "=== %s === (return: %s, params: %d)"
                     name (:return-tag info) (count (:params info))))
          (println "Params:" (pr-str (mapv #(vector % (:tag (meta %))) (:params info))))
          (println)
          (pp/pprint (:expr info))
          (println "\n"))))

    ;; ============================================================
    ;; 3. Bytecode for all helpers
    ;; ============================================================
    (println "Dumping bytecode...")
    (when-let [bytes (:statics-bytes diag)]
      (spit-file outdir "c2_bytecode.txt"
        (with-out-str
          (doseq [[name _] (sort-by key (:helper-map diag))]
            (println (str "=== " name " ==="))
            (inspect/disassemble bytes {:method name})
            (println "\n")))))

    ;; ============================================================
    ;; 4. Native assembly per helper (requires hsdis + warmup)
    ;; ============================================================
    (println "Warming up for native assembly...")
    (let [W1 (double-array (* 784 128))
          b1 (double-array 128)
          W2 (double-array (* 128 10))
          b2 (double-array 10)
          x (double-array 784)
          y (let [a (double-array 10)] (aset a 3 1.0) a)]
      ;; 20K warmup to trigger C2 compilation of all helpers
      (dotimes [_ 20000] (f W1 b1 W2 b2 x y 0.01))

      ;; Dump native assembly per helper
      (println "Dumping native assembly...")
      (when-let [bytes (:statics-bytes diag)]
        (let [cf-name (str/replace (.getName (class @#'train-step)) "$W" "")]
          (try
            (let [cls (Class/forName cf-name)]
              (doseq [[name _] (sort-by key (:helper-map diag))]
                (let [asm (with-out-str
                            (try (inspect/native-disassemble cls name)
                                 (catch Exception e
                                   (println (str "  " (.getMessage e))))))]
                  (when (seq (str/trim asm))
                    (spit-file outdir (str "c2_native_" name ".txt") asm)))))
            (catch Exception e
              (println (str "  native disassembly not available: " (.getMessage e)
                            "\n  (install hsdis for native assembly)"))))))

      ;; ============================================================
      ;; 5. Micro-benchmark (matches JAX protocol)
      ;; ============================================================
      (println "Benchmarking...")
      (let [ts (long-array 200)]
        (dotimes [i 200]
          (let [t0 (System/nanoTime)]
            (dotimes [_ 200]
              (f W1 b1 W2 b2 x y 0.01))
            (aset ts i (- (System/nanoTime) t0))))
        (let [times (sort (map #(/ (double %) 200000.0) ts))
              median (nth times 100)
              p10 (nth times 20)
              p90 (nth times 180)
              minimum (first times)]
          (spit-file outdir "c2_timing.txt"
            (str/join "\n"
              [(format "Raster AOT+SIMD, Valhalla JDK, float64, MLP 784-128-10, SGD lr=0.01")
               (format "Protocol: 20000 warmup, 200 batches of 200 steps")
               ""
               (format "median:  %.1f µs" median)
               (format "p10:     %.1f µs" p10)
               (format "p90:     %.1f µs" p90)
               (format "min:     %.1f µs" minimum)]))
          (println (format "\n  Raster MLP f64: median=%.1f µs, min=%.1f µs" median minimum)))))

    ;; ============================================================
    ;; 6. Full pipeline stage dump (fixpointed, backend-applied)
    ;; ============================================================
    (println "Dumping pipeline stages...")
    (spit-file outdir "c2_pipeline_stages.txt"
      (with-out-str
        (println "=== fixpointed (after inlining + lowering + DCE) ===")
        (pp/pprint (:fixpointed stages))
        (println "\n=== backend-applied (after SIMD pass) ===")
        (pp/pprint (:backend-applied stages))
        (println "\n=== mem-merged (final form) ===")
        (pp/pprint (:mem-merged stages))))

    (println "\nDone. Compare with:")
    (println (str "  diff " outdir "/c2_timing.txt " outdir "/xla_timing.txt"))
    (println (str "  # Look for vmulpd/vaddpd (AVX2) in c2_native_*.txt"))
    (println (str "  # Look for fused_computation blocks in xla_fusion_summary.txt"))))

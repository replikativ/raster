(ns specbr (:require [raster.umap.spectral :as sp]))

(defn- npy-doff [^bytes bs]
  (let [bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN))
        hlen (do (.position bb 8) (bit-and (int (.getShort bb)) 0xFFFF))]
    [bb (+ 10 hlen)]))

(defn read-f64 ^doubles [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        [bb doff] (npy-doff bs)
        n (quot (- (alength bs) doff) 8) out (double-array n)]
    (.position bb (int doff)) (dotimes [i n] (aset out i (.getDouble bb))) out))

(defn read-i32 ^ints [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        [bb doff] (npy-doff bs)
        n (quot (- (alength bs) doff) 4) out (int-array n)]
    (.position bb (int doff)) (dotimes [i n] (aset out i (.getInt bb))) out))

(defn median ^double [xs] (let [s (vec (sort xs)) c (count s)] (nth s (quot c 2))))

;; [tag n data-dim umap-ms]  (umap-ms from spectral_bench.py output)
;; Latest with Lanczos early-termination (median ms, single machine):
;;   swiss_1000 30/25 0.86x  swiss_2000 88/84 0.96x  swiss_5000 378/380 1.01x
;;   mnist_1000 19/42 2.21x  mnist_2000 64/85 1.32x  mnist_5000 80/77 0.97x
;; (before early-termination mnist_5000 was 880ms = 0.09x — fixed-256-iter Lanczos)
(def CASES [["swiss_1000" 1000 3  25.35] ["swiss_2000" 2000 3  84.44] ["swiss_5000" 5000 3  379.83]
            ["mnist_1000" 1000 50 42.40] ["mnist_2000" 2000 50 84.90] ["mnist_5000" 5000 50 77.48]])
(def G "/tmp/umap_gold")
(def REPS 7)

(println (format "%-12s %6s %8s %4s %11s %11s %8s" "case" "n" "edges" "comp" "raster(ms)" "umap(ms)" "speedup"))
(doseq [[tag n dd umap-ms] CASES]
  (let [head (read-i32 (str G "/sp_bench_head_" tag ".npy"))
        tail (read-i32 (str G "/sp_bench_tail_" tag ".npy"))
        w    (read-f64 (str G "/sp_bench_w_" tag ".npy"))
        X    (read-f64 (str G "/sp_bench_X_" tag ".npy"))
        ne   (alength w)]
    (sp/spectral-init head tail w n 2 X dd)   ; warm up (lazy-JIT compile)
    (let [ts (vec (for [_ (range REPS)]
                    (let [t0 (System/nanoTime)]
                      (sp/spectral-init head tail w n 2 X dd)
                      (/ (- (System/nanoTime) t0) 1.0e6))))
          rms (median ts)]
      (println (format "%-12s %6d %8d %4s %11.2f %11.2f %7.2fx"
                       tag n ne "-" rms umap-ms (/ umap-ms rms))))))

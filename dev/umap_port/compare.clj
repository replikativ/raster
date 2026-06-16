(ns compare-driver
  "Run raster UMAP on a dataset, measure approx-kNN recall vs the exact sample,
  and dump the embedding (.npy) + metrics for the python eval/plot step.

  Dataset selected via the DS env var (mnist|fashion|...). Reads C-order .npy via
  the fortran-aware npy loader. Run:
    DS=fashion clojure -M:valhalla -e '(load-file \"dev/umap_port/compare.clj\")'"
  (:require [raster.umap.fit :as fit]
            [raster.spatial.nndescent :as nnd]
            [clojure.set :as set]))
(load-file "/tmp/raster-bc/dev/umap_port/npy.clj")

(def OUT "/tmp/umap_gold")
(def DS (or (System/getenv "DS") "fashion"))

(defn f32-copy ^floats [^doubles X]
  (let [n (alength X) a (float-array n)] (dotimes [i n] (aset a i (float (aget X i)))) a))

(defn knn-recall
  "Mean recall of raster's approx kNN vs the exact sample. idx: int[n*k] (self-incl).
  knnq: int[q] query rows. knne: int[q*K] exact neighbors (self-incl, K cols)."
  [^ints idx ^ints knnq ^ints knne k K]
  (let [k (long k) K (long K) q (alength knnq)]
    (/ (reduce + (for [qi (range q)]
                   (let [p (aget knnq qi)
                         rset (set (for [c (range k) :let [v (aget idx (+ (* p k) c))] :when (not= v p)] v))
                         eset (set (for [c (range K) :let [v (aget knne (+ (* qi K) c))] :when (not= v p)] v))]
                     (/ (double (count (set/intersection rset eset))) (max 1 (count eset))))))
       (double q))))

(let [X    (npy/read-f64 (str OUT "/" DS "_X.npy"))
      meta (npy/load-npy (str OUT "/" DS "_X.npy"))
      [n dim] (:shape meta)
      n (long n) dim (long dim)
      y    (npy/read-i32 (str OUT "/" DS "_y.npy"))
      knnq (npy/read-i32 (str OUT "/" DS "_knnq.npy"))
      knne (npy/read-i32 (str OUT "/" DS "_knne.npy"))
      k 15 K 15
      ;; raster approx kNN recall (separate run on a fresh f32 copy — cosine-knn mutates X)
      t-knn (System/nanoTime)
      kres (nnd/cosine-knn (f32-copy X) n dim k)
      knn-secs (/ (- (System/nanoTime) t-knn) 1.0e9)
      recall (knn-recall (:idx kres) knnq knne k K)
      ;; full raster UMAP fit (spectral for small n, random for large — fit :auto)
      t0 (System/nanoTime)
      res (fit/fit (f32-copy X) n dim :k k :init :auto :seed 42)
      fit-secs (/ (- (System/nanoTime) t0) 1.0e9)
      emb (:emb res)]
  (npy/save-f64-2d (str OUT "/" DS "_emb_raster.npy") emb n 2)
  (spit (str OUT "/" DS "_raster_meta.edn")
        (pr-str {:ds DS :n n :dim dim :k k :init (:init res)
                 :knn-recall recall :knn-secs knn-secs :fit-secs fit-secs}))
  (spit (str OUT "/" DS "_raster_summary.txt")
        (format "[%s] n=%d dim=%d  raster fit=%.1fs (knn %.1fs, init=%s)  approx-kNN recall=%.4f"
                DS n dim fit-secs knn-secs (name (:init res)) recall)))
(println (slurp (str OUT "/" DS "_raster_summary.txt")))

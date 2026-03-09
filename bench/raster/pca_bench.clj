(ns raster.pca-bench
  (:require [raster.linalg.pca :as pca]
            [raster.linalg.lapack :as lapack]))

(defn bench-pca [m n k label]
  (let [rng (java.util.Random. 42)
        X (double-array (* m n))
        _ (dotimes [i (* m n)] (aset X i (.nextGaussian rng)))
        iters (max 3 (min 20 (int (/ 500 (max m n)))))
        time-fn (fn [f]
                  (f) ;; warmup
                  (let [t0 (System/nanoTime)]
                    (dotimes [_ iters] (f))
                    (/ (/ (- (System/nanoTime) t0) 1e6) iters)))]
    (let [t-full (time-fn #(pca/pca-full (aclone X) m n k))
          t-eigh (time-fn #(pca/pca-covariance-eigh (aclone X) m n k))
          t-rand (time-fn #(pca/pca-randomized (aclone X) m n k 10 4))]
      (println (format "%s [%dx%d, k=%d]:  full=%.1fms  eigh=%.1fms  rand=%.1fms"
                 label m n k t-full t-eigh t-rand)))))

(defn -main []
  (lapack/available?)  ;; triggers thread optimization
  (println "OpenBLAS threads:" (lapack/get-num-threads))
  (println "")
  (bench-pca 100 50 10 "Small")
  (bench-pca 500 100 10 "Medium-1")
  (bench-pca 1000 500 10 "Medium-2")
  (bench-pca 5000 100 10 "Tall")
  (bench-pca 1000 500 50 "Med-k50")
  (bench-pca 10000 1000 10 "Large")
  (bench-pca 10000 100 10 "Large-tall"))

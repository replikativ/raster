(ns raster.umap.knn
  "Nearest-neighbor search for the UMAP/EVoC pipeline.

  `knn-brute!` is an exact O(n^2 d) brute-force kNN — the baseline used by
  UMAP itself for small inputs (n < 4096). For each row it keeps the k smallest
  squared distances via insertion into a sorted per-row buffer, then sqrt's only
  the k survivors. Output matches umap.umap_.nearest_neighbors: row 0 of each
  point is itself (distance 0), neighbors sorted ascending by distance.

  (NN-descent for large n comes later; this is the validation baseline.)"
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == sqrt]]))

;; X: flat double[n*dim], row-major. Fills:
;;   out-idx : int[n*k]    neighbor indices (ascending distance, self first)
;;   out-dst : double[n*k] euclidean distances
(deftm knn-brute!
  [X :- (Array double) n :- Long dim :- Long k :- Long
   out-idx :- (Array int) out-dst :- (Array double)]
  :- (Array double)
  (dotimes [i n]
    (let [ib (* i k)
          xib (* i dim)]
      ;; init the per-row buffer: distances = +inf, indices = -1
      (dotimes [t k]
        (aset out-dst (+ ib t) 1.0e308)
        (aset out-idx (+ ib t) (int -1)))
      ;; scan every candidate j, insert by squared distance
      (dotimes [j n]
        (let [xjb (* j dim)
              d2 (loop [d 0 acc 0.0]
                   (if (< d dim)
                     (recur (inc d)
                            (let [df (- (aget X (+ xib d)) (aget X (+ xjb d)))]
                              (+ acc (* df df))))
                     acc))]
          (when (< d2 (aget out-dst (+ ib (- k 1))))
            ;; insertion sort step: shift larger entries right, drop d2 in place
            (loop [p (- k 1)]
              (if (and (> p 0) (> (aget out-dst (+ ib (- p 1))) d2))
                (do (aset out-dst (+ ib p) (aget out-dst (+ ib (- p 1))))
                    (aset out-idx (+ ib p) (aget out-idx (+ ib (- p 1))))
                    (recur (- p 1)))
                (do (aset out-dst (+ ib p) d2)
                    (aset out-idx (+ ib p) (int j))))))))
      ;; convert the k kept squared distances to euclidean
      (dotimes [t k]
        (aset out-dst (+ ib t) (sqrt (aget out-dst (+ ib t)))))))
  out-dst)

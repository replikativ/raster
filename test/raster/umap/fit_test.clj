(ns raster.umap.fit-test
  "End-to-end UMAP fit: well-separated high-dim Gaussian blobs must map to
  well-separated 2D clusters. UMAP layout is chaotically sensitive, so we test
  at the quality level (2D nearest-neighbor label agreement), not coordinates."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.umap.fit :as fit]))

(defn- blobs
  "n points in `dim` dims drawn from `nc` well-separated Gaussian blobs.
  Returns [X(flat n*dim) labels(int[n])]."
  [n dim nc seed]
  (let [r (java.util.Random. seed)
        centers (vec (for [_ (range nc)]
                       (let [c (double-array dim)]
                         (dotimes [d dim] (aset c d (* 20.0 (.nextGaussian r)))) c)))
        X (double-array (* n dim))
        y (int-array n)]
    (dotimes [i n]
      (let [c (mod i nc) ^doubles ctr (nth centers c)]
        (aset y i (int c))
        (dotimes [d dim]
          (aset X (+ (* i dim) d) (+ (aget ctr d) (.nextGaussian r))))))
    [X y]))

(defn- nn-label-agreement
  "Fraction of points whose nearest 2D-embedding neighbor shares its label."
  [^doubles emb ^ints y n]
  (/ (double
       (reduce + (for [i (range n)]
                   (let [xi (* 2 i)
                         best (loop [j 0 bd Double/MAX_VALUE bj -1]
                                (if (< j n)
                                  (if (= j i)
                                    (recur (inc j) bd bj)
                                    (let [dx (- (aget emb xi) (aget emb (* 2 j)))
                                          dy (- (aget emb (+ xi 1)) (aget emb (+ (* 2 j) 1)))
                                          d (+ (* dx dx) (* dy dy))]
                                      (if (< d bd) (recur (inc j) d j) (recur (inc j) bd bj))))
                                  bj))]
                     (if (= (aget y best) (aget y i)) 1 0)))))
     n))

(deftest fit-separates-blobs
  (testing "UMAP fit maps well-separated high-dim blobs to separated 2D clusters"
    (let [n 300 dim 20 nc 3
          [X y] (blobs n dim nc 7)
          {:keys [emb] :as res} (fit/fit X n dim :k 15 :seed 42)
          agree (nn-label-agreement emb y n)]
      (is (= (* n 2) (count emb)) "embedding is n*2 doubles")
      (is (= :spectral (:init res)) "small n uses spectral init")
      (is (> agree 0.95) (str "2D NN-label agreement " agree " should exceed 0.95")))))

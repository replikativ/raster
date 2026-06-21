(ns raster.spatial.nndescent-test
  "Approximate cosine kNN (NN-descent) must agree with exact brute force at
  n > the brute/approx threshold. Regression guard for the double-conversion bug
  where finalize! AND cosine-knn both applied neg-cos->log2!, mapping every real
  neighbor distance to the 1e308 'far' sentinel (recall collapsed to ~0)."
  (:require [clojure.test :refer [deftest testing is]]
            [clojure.set :as set]
            [raster.spatial.nndescent :as nnd]
            [raster.knn]))

(defn- blobs
  "n points in `dim` dims from `nc` separated Gaussian blobs; flat double[n*dim]."
  ^doubles [n dim nc seed]
  (let [r (java.util.Random. seed)
        centers (vec (for [_ (range nc)]
                       (double-array (repeatedly dim #(* 10.0 (.nextGaussian r))))))
        X (double-array (* n dim))]
    (dotimes [i n]
      (let [^doubles c (nth centers (mod i nc))]
        (dotimes [d dim] (aset X (+ (* i dim) d) (+ (aget c d) (.nextGaussian r))))))
    X))

(deftest nndescent-recall-above-threshold
  (testing "NN-descent (n>4096) finds real neighbors and matches brute force"
    (let [n 4500 dim 32 k 15            ; n>4096 forces the approximate path
          X (blobs n dim 8 7)
          {ai :idx ad :dst} (nnd/cosine-knn (aclone X) n dim k)
          {bi :idx}         (nnd/cosine-knn (aclone X) n dim k :threshold (inc n))
          finite (count (for [i (range (* n k)) :when (< (aget ^doubles ad i) 1.0e300)] i))
          recall (/ (double (reduce + (for [i (range n)]
                                        (count (set/intersection
                                                (set (for [t (range k)] (aget ^ints ai (+ (* i k) t))))
                                                (set (for [t (range k)] (aget ^ints bi (+ (* i k) t)))))))))
                    (* n k))]
      ;; every slot must be a real neighbor (not the 1e308 sentinel)
      (is (> finite (* 0.99 n k))
          (str "only " finite "/" (* n k) " finite distances — neighbors not populated"))
      ;; approximate recall should be high on well-separated blobs. The
      ;; reverse-augmented local-join (exploring 2-hop via reverse neighbours, not
      ;; just forward) lifted recall from ~0.66 to ~0.90 on hard high-dim data and
      ;; ~0.99 here; guard against regressing to the forward-only plateau.
      (is (> recall 0.95) (str "NN-descent recall vs brute " recall " should exceed 0.95")))))

(deftest euclidean-nndescent-recall
  (testing "euclidean NN-descent (n>threshold) matches exact brute euclidean kNN"
    (let [n 5000 dim 20 k 15
          X (blobs n dim 10 3)
          {ai :idx ad :dst} (nnd/euclidean-knn (aclone X) n dim k)
          bi (let [oi (int-array (* n k)) od (double-array (* n k))]
               (raster.knn/knn-brute! (aclone X) n dim k oi od) oi)
          recall (/ (double (reduce + (for [i (range n)]
                                        (count (set/intersection
                                                (set (for [t (range k)] (aget ^ints ai (+ (* i k) t))))
                                                (set (for [t (range k)] (aget ^ints bi (+ (* i k) t)))))))))
                    (* n k))]
      ;; euclidean distances must be non-negative and self at 0
      (is (= 0.0 (aget ^doubles ad 0)) "self distance should be 0")
      (is (> recall 0.9) (str "euclidean NN-descent recall " recall " should exceed 0.9")))))

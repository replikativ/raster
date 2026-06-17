(ns raster.spatial.kdtree-test
  "Self-contained tests: KD-tree kNN is exact vs brute-force; RP-tree forest leaf
  coverage; NN-descent recall (dense + quantized byte)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.spatial.kdtree :as kd]
            [raster.spatial.rptree]
            [raster.spatial.nndescent]
            [raster.knn :as knn]))

(defn- rand-data [n dim seed]
  (let [r (java.util.Random. seed) a (double-array (* n dim))]
    (dotimes [i (* n dim)] (aset a i (.nextGaussian r)))
    a))

(deftest kdtree-knn-exact-vs-brute
  (testing "KD-tree kNN returns the same neighbor sets + distances as brute-force"
    (let [n 300 dim 3 k 8 X (rand-data n dim 1)
          bi (int-array (* n k)) bd (double-array (* n k))
          _ (knn/knn-brute! X n dim k bi bd)
          kq (kd/knn-query (kd/build X n dim) k)
          ki (:idx kq) kdst (:dst kq)
          set-ok (count (filter (fn [i]
                    (= (set (map #(aget ki (+ (* i k) %)) (range k)))
                       (set (map #(aget bi (+ (* i k) %)) (range k))))) (range n)))
          ;; kdtree dst is squared; brute is euclidean
          dmax (reduce max 0.0
                 (for [i (range n)]
                   (let [ks (sort (map #(aget kdst (+ (* i k) %)) (range k)))
                         bs (sort (map #(let [e (aget bd (+ (* i k) %))] (* e e)) (range k)))]
                     (reduce max 0.0 (map (fn [a b] (Math/abs (- a b))) ks bs)))))]
      (is (= n set-ok) "every row's neighbor set matches brute-force")
      (is (< dmax 1e-9) "squared distances match"))))

;; (Boruvka-MST-equals-Prim moved to the evoc-rstr library's test suite with the
;; clustering code.)

;; ================================================================
;; RP-tree forest + NN-descent (approximate kNN)
;; ================================================================

(deftest rptree-leaves-cover-all-points
  (testing "RP-tree leaf ranges partition every point exactly once"
    (let [n 400 dim 5 leaf 30
          X (let [r (java.util.Random. 1) a (double-array (* n dim))]
              (dotimes [i (* n dim)] (aset a i (.nextGaussian r))) a)
          idx (int-array n) hp (double-array dim) stk (int-array (* 4 n))
          ls (int-array n) le (int-array n)
          rng (long-array [1 2 3])
          nl (raster.spatial.rptree/build-rptree-leaves! X n dim leaf idx rng hp stk ls le)
          seen (boolean-array n)
          total (reduce + (for [lf (range nl)]
                            (do (doseq [p (range (aget ls lf) (aget le lf))]
                                  (aset seen (aget idx p) true))
                                (- (aget le lf) (aget ls lf)))))]
      (is (= n total) "leaf ranges cover all n points")
      (is (every? identity (seq seen)) "every point appears in a leaf"))))

(deftest nndescent-recall-vs-brute
  (testing "NN-descent achieves high recall vs exact brute-force cosine kNN"
    (let [n 1500 dim 8 k 12
          X (let [r (java.util.Random. 3) a (double-array (* n dim))]
              (dotimes [i (* n dim)] (aset a i (.nextGaussian r))) a)
          Xn (double-array (seq X))
          _ (raster.knn/l2-normalize! Xn n dim)
          bi (int-array (* n k)) bd (double-array (* n k))
          _ (raster.knn/knn-brute-cosine! Xn n dim k bi bd)
          ni (:idx (raster.spatial.nndescent/nn-descent (double-array (seq X)) n dim k))
          recall (/ (reduce + (for [i (range n)]
                     (count (clojure.set/intersection
                              (set (for [t (range k)] (aget ni (+ (* i k) t))))
                              (set (for [t (range k)] (aget bi (+ (* i k) t))))))))
                    (double (* n k)))]
      (is (> recall 0.9) (str "recall " recall " should exceed 0.9")))))

(defn- byte-blobs
  "n quantized byte points in `dim` dims, drawn from `nc` Gaussian blobs.
  Each cluster center is a random byte vector; points jitter around it."
  [n dim nc seed]
  (let [r (java.util.Random. seed)
        centers (vec (for [_ (range nc)]
                       (let [c (byte-array dim)]
                         (dotimes [d dim] (aset c d (byte (- (.nextInt r 256) 128)))) c)))
        X (byte-array (* n dim))]
    (dotimes [i n]
      (let [^bytes c (nth centers (mod i nc))]
        (dotimes [d dim]
          (let [v (+ (aget c d) (* 8 (.nextGaussian r)))]
            (aset X (+ (* i dim) d)
                  (byte (max -128 (min 127 (Math/round v)))))))))
    X))

(deftest nndescent-bytes-recall-vs-brute
  (testing "Quantized byte NN-descent (int8 + uint8) achieves high recall vs brute-force"
    (let [n 300 dim 32 k 12
          ;; pure-Clojure brute (calls byte-dist from Clojure — no deftm inlining)
          brute (fn [^bytes data mode]
                  (let [oi (int-array (* n k))]
                    (dotimes [i n]
                      (let [ds (sort-by second
                                 (map (fn [j] [j (raster.spatial.nndescent/byte-dist data i j dim mode)])
                                      (range n)))]
                        (dotimes [t k] (aset oi (+ (* i k) t) (int (first (nth ds t)))))))
                    oi))
          rec (fn [ni bi]
                (/ (reduce + (for [i (range n)]
                     (count (clojure.set/intersection
                              (set (for [t (range k)] (aget ni (+ (* i k) t))))
                              (set (for [t (range k)] (aget bi (+ (* i k) t))))))))
                   (double (* n k))))]
      (doseq [mode [0 1]]                       ;; 0 = int8 (signed inner), 1 = uint8 (bit-Jaccard)
        (let [X (byte-blobs n dim 6 (+ 2 mode))
              bi (brute X mode)
              ni (:idx (raster.spatial.nndescent/nn-descent-bytes X n dim k mode))
              r (rec ni bi)]
          (is (> r 0.85) (str "mode " mode " recall " r " should exceed 0.85")))))))

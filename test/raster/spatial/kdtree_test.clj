(ns raster.spatial.kdtree-test
  "Self-contained tests: KD-tree kNN is exact vs brute-force; Boruvka MST is a
  valid minimum spanning tree (total weight == dense Prim)."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.spatial.kdtree :as kd]
            [raster.umap.knn :as knn]
            [raster.cluster.mst :as mst]
            [raster.cluster.boruvka :as bor]))

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

(deftest boruvka-mst-equals-prim
  (testing "Boruvka MST has identical total weight to dense Prim (valid min tree)"
    (let [n 250 dim 2 ms 5 X (rand-data n dim 2)
          p (mst/mutual-reachability-mst X n dim ms)
          b (bor/boruvka-mst X n dim ms)
          tot (fn [m] (reduce + 0.0 (map #(aget ^doubles (:w m) %) (range (alength ^doubles (:w m))))))]
      (is (= (clojure.core/dec n) (alength ^doubles (:w b))) "n-1 edges")
      (is (< (Math/abs (- (tot p) (tot b))) 1e-6) "same total weight as Prim"))))

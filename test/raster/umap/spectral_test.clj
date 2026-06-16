(ns raster.umap.spectral-test
  "Spectral initialization: the matrix-free Lanczos path (large n) and the dense
  eigh path (tiny n) must both produce a valid bottom-of-Laplacian embedding.
  We test at the structural level — the first non-trivial eigenvector (Fiedler
  vector) must separate two weakly-coupled graph components by sign — since the
  eigenbasis is only defined up to sign/rotation within degenerate subspaces."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.umap.spectral :as sp]))

(defn- components
  "Run the union-find connected-components kernels on an undirected edge list.
  Returns [n-components labels-vec]."
  [edges n]
  (let [bidir (mapcat (fn [[i j]] [[i j] [j i]]) edges)
        head (int-array (map first bidir))
        tail (int-array (map second bidir))
        ne (alength head)
        parent (int-array n)
        _ (sp/connected-components! head tail parent n ne)
        label (int-array n)
        remap (int-array n)
        c (long (sp/densify-labels! parent label remap n))]
    [c (vec label)]))

;; Ported from scipy/sparse/csgraph/tests/test_connected_components.py
;; (the weak-connection cases — raster's graph is symmetric, so union-find over
;; the edge list is exactly scipy's connection='weak'/undirected).
(deftest connected-components-weak-test
  (testing "test_weak_connections: edge 0-1, vertex 2 isolated -> 2 components"
    (let [[c labels] (components [[0 1]] 3)]
      (is (= c 2))
      (is (= (vec (sort labels)) [0 0 1]))))
  (testing "test_weak_connections2: {0,1} and {2,3,4,5} -> 2 components"
    (let [[c labels] (components [[1 0] [2 3] [3 4] [5 4]] 6)]
      (is (= c 2))
      (is (= (vec (sort labels)) [0 0 1 1 1 1]))))
  (testing "test_fully_connected_graph: K4 -> 1 component"
    (let [[c _] (components (for [i (range 4) j (range 4) :when (not= i j)] [i j]) 4)]
      (is (= c 1)))))

(defn- two-block-graph
  "Two rings of `half` vertices (blocks A=[0,half), B=[half,2*half)) joined by a
  single weak bridge edge. The Fiedler vector separates A from B by sign.
  Returns [head tail weights]."
  [half]
  (let [edges (java.util.ArrayList.)
        add (fn [i j w] (.add edges [i j w]) (.add edges [j i w]))]
    (doseq [b [0 half]]
      (dotimes [i half]
        (add (+ b i) (+ b (mod (inc i) half)) 1.0)))   ; ring within each block
    (add 0 half 0.01)                                  ; weak bridge A<->B
    (let [n (count edges)]
      [(int-array (map #(nth % 0) edges))
       (int-array (map #(nth % 1) edges))
       (double-array (map #(nth % 2) edges))])))

(defn- fiedler-separates? [^doubles emb half]
  ;; Column 0 of the embedding is the Fiedler vector. Each block should be
  ;; sign-consistent and the two blocks opposite. Count majority sign per block.
  (let [dim 2
        block-sign (fn [lo hi]
                     (let [s (reduce + (for [i (range lo hi)]
                                         (if (>= (aget emb (* i dim)) 0.0) 1 -1)))]
                       (Integer/signum (int s))))]
    (not= (block-sign 0 half) (block-sign half (* 2 half)))))

(deftest lanczos-path-fiedler-test
  (testing "matrix-free Lanczos (n > dense-cutoff) separates two blocks"
    (let [half 200 n (* 2 half)
          [h t w] (two-block-graph half)
          emb (sp/spectral-init h t w n 2)]
      (is (= (alength emb) (* n 2)))
      (is (fiedler-separates? emb half)
          "Fiedler vector should give opposite signs to the two blocks"))))

(deftest dense-path-fiedler-test
  (testing "dense eigh (n <= dense-cutoff) separates two blocks"
    (let [half 50 n (* 2 half)
          [h t w] (two-block-graph half)
          emb (sp/spectral-init h t w n 2)]
      (is (= (alength emb) (* n 2)))
      (is (fiedler-separates? emb half)
          "Fiedler vector should give opposite signs to the two blocks"))))

(defn- two-disconnected-rings
  "Two rings of `half` vertices with NO bridge — two connected components.
  Returns [head tail weights]."
  [half]
  (let [edges (java.util.ArrayList.)
        add (fn [i j] (.add edges [i j 1.0]) (.add edges [j i 1.0]))]
    (doseq [b [0 half]] (dotimes [i half] (add (+ b i) (+ b (mod (inc i) half)))))
    [(int-array (map #(nth % 0) edges))
     (int-array (map #(nth % 1) edges))
     (double-array (map #(nth % 2) edges))]))

(defn- centroid [^doubles emb lo hi]
  (let [dim 2 c (double-array dim)]
    (doseq [i (range lo hi)] (dotimes [d dim] (aset c d (+ (aget c d) (aget emb (+ (* i dim) d))))))
    (dotimes [d dim] (aset c d (/ (aget c d) (double (- hi lo)))))
    c))

(defn- ring-clusters-on-circle
  "K disconnected ring-clusters of `per` vertices each, whose data-space centroids
  lie on a circle (cluster k at angle 2πk/K). Returns [head tail weights X data-dim]."
  [k-clusters per]
  (let [dd 2 n (* k-clusters per)
        x (double-array (* n dd))
        h (java.util.ArrayList.) t (java.util.ArrayList.) w (java.util.ArrayList.)]
    (dotimes [k k-clusters]
      (let [cx (* 10.0 (Math/cos (* 2 Math/PI (/ k (double k-clusters)))))
            cy (* 10.0 (Math/sin (* 2 Math/PI (/ k (double k-clusters)))))]
        (dotimes [i per]
          (let [v (+ (* k per) i)]
            (aset x (* v dd) (+ cx (* 0.3 (Math/cos (* 13.0 i)))))
            (aset x (+ (* v dd) 1) (+ cy (* 0.3 (Math/sin (* 7.0 i)))))
            (let [j (+ (* k per) (mod (inc i) per))]
              (.add h v) (.add t j) (.add w 1.0)
              (.add h j) (.add t v) (.add w 1.0))))))
    [(int-array h) (int-array t) (double-array w) x dd]))

(deftest tier-c-data-centroid-topology-test
  (testing "with many components (c > 2*dim), data-centroid meta-placement recovers
            the component topology: clusters whose centroids are circular neighbours
            map to neighbouring embedding centers"
    (let [k-clusters 6 per 30 n (* k-clusters per)
          [h t w x dd] (ring-clusters-on-circle k-clusters per)
          emb (sp/spectral-init h t w n 2 x dd)         ; 7-arg: Tier C enabled
          center (fn [k] (centroid emb (* k per) (* (inc k) per)))
          centers (mapv center (range k-clusters))
          dist (fn [a b] (let [ca (centers a) cb (centers b)]
                           (Math/sqrt (+ (Math/pow (- (aget ca 0) (aget cb 0)) 2)
                                         (Math/pow (- (aget ca 1) (aget cb 1)) 2)))))]
      (is (= (alength emb) (* n 2)))
      (is (not (some #(Double/isNaN %) (seq emb))) "no NaN")
      ;; each cluster's nearest other center must be one of its ring neighbours
      (doseq [k (range k-clusters)]
        (let [others (remove #{k} (range k-clusters))
              nearest (apply min-key #(dist k %) others)
              ring-nbrs #{(mod (dec k) k-clusters) (mod (inc k) k-clusters)}]
          (is (contains? ring-nbrs nearest)
              (str "cluster " k " nearest center is " nearest
                   ", expected a ring neighbour " ring-nbrs)))))))

(deftest multi-component-separation-test
  (testing "two disconnected components are placed at distinct meta-centers,
            spread well apart relative to each component's internal radius"
    (let [half 200 n (* 2 half)
          [h t w] (two-disconnected-rings half)
          emb (sp/spectral-init h t w n 2)
          ca (centroid emb 0 half)
          cb (centroid emb half n)
          dist (Math/sqrt (+ (Math/pow (- (aget ca 0) (aget cb 0)) 2)
                             (Math/pow (- (aget ca 1) (aget cb 1)) 2)))
          ;; max radius of component A about its centroid
          rad (reduce max (for [i (range half)]
                            (Math/sqrt (+ (Math/pow (- (aget emb (* i 2)) (aget ca 0)) 2)
                                          (Math/pow (- (aget emb (+ (* i 2) 1)) (aget ca 1)) 2)))))]
      (is (= (alength emb) (* n 2)))
      ;; umap sizes each component radius to data_range = half the center gap, so
      ;; components touch but don't overlap: radius <= ~half the center distance.
      (is (<= rad (* 0.55 dist))
          (str "component radius " rad " should stay within ~half the center gap " dist))
      (is (not (some #(Double/isNaN %) (seq emb))) "no NaN in embedding"))))

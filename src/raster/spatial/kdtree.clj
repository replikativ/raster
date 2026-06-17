(ns raster.spatial.kdtree
  "Array-based KD-tree (general spatial index) — ports EVoC's numba_kdtree.
  Complete-binary-tree layout (node i -> children 2i+1, 2i+2). All node data is
  flat arrays held in a KDTree defvalue (mirrors linalg.sparse's CSRMatrix).
  Build = median split on the widest-spread axis via quickselect; kNN query =
  box-pruned descent with a bounded max-heap. Recursion is replaced by explicit
  array stacks (deftm has no recursion). Distances are rdist = squared euclidean.

  The query is exact for any valid median partition, so the tree structure need
  not bit-match EVoC — only the returned neighbors must (validated vs brute-force)."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == min max])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == min max]]))

(defvalue KDTree (All [T])
  [data :- (Array double) idx :- (Array int)
   idx-start :- (Array int) idx-end :- (Array int)
   lower :- (Array double) upper :- (Array double) is-leaf :- (Array int)
   n :- Long dim :- Long n-nodes :- Long])

;; squared euclidean between query row at offset qb and data point pidx
(deftm rdist [data :- (Array double) qb :- Long pidx :- Long dim :- Long] :- Double
  (let [pb (* pidx dim)]
    (loop [j 0 acc 0.0]
      (if (< j dim)
        (recur (inc j) (let [df (- (aget data (+ qb j)) (aget data (+ pb j)))] (+ acc (* df df))))
        acc))))

;; lower bound (rdist) from query point to node's bounding box
(deftm point-box-lb
  [lower :- (Array double) upper :- (Array double) data :- (Array double)
   node :- Long qb :- Long dim :- Long] :- Double
  (let [nb (* node dim)]
    (loop [j 0 acc 0.0]
      (if (< j dim)
        (let [pj (aget data (+ qb j))
              lj (aget lower (+ nb j)) uj (aget upper (+ nb j))
              dlo (if (< pj lj) (- lj pj) 0.0)
              dhi (if (> pj uj) (- pj uj) 0.0)
              d (+ dlo dhi)]
          (recur (inc j) (+ acc (* d d))))
        acc))))

;; bounded max-heap push (mirrors simple_heap_push): insert if p < current max.
(deftm heap-push! [hp :- (Array double) hi :- (Array int) p :- Double idx :- Long size :- Long] :- Long
  (if (>= p (aget hp 0))
    0
    (let [fin (loop [i 0]
                (let [ic1 (+ (* 2 i) 1) ic2 (+ ic1 1)]
                  (if (>= ic1 size)
                    i
                    (let [isw (if (>= ic2 size)
                                (if (> (aget hp ic1) p) ic1 -1)
                                (if (>= (aget hp ic1) (aget hp ic2))
                                  (if (< p (aget hp ic1)) ic1 -1)
                                  (if (< p (aget hp ic2)) ic2 -1)))]
                      (if (== isw -1)
                        i
                        (do (aset hp i (aget hp isw)) (aset hi i (aget hi isw)) (recur isw)))))))]
      (aset hp fin p) (aset hi fin (int idx))
      1)))

;; widest-spread axis over idx[start,end)
(deftm find-split-dim [data :- (Array double) idx :- (Array int) dim :- Long start :- Long end :- Long] :- Long
  (loop [j 0 best 0 best-spread -1.0]
    (if (< j dim)
      (let [mn (loop [i start mn 1.0e308 mx -1.0e308]
                 (if (< i end)
                   (let [v (aget data (+ (* (long (aget idx i)) dim) j))]
                     (recur (inc i) (min mn v) (max mx v)))
                   (- mx mn)))]
        (if (> mn best-spread) (recur (inc j) j mn) (recur (inc j) best best-spread)))
      best)))

;; quickselect: partition idx[left,right) so idx[nth] holds the axis-median,
;; smaller-by-axis on the left. Lomuto partition, recurse the side holding nth.
(deftm nth-element! [data :- (Array double) idx :- (Array int) axis :- Long dim :- Long
                     left0 :- Long right0 :- Long nth :- Long] :- Long
  (loop [left left0 right (- right0 1)]
    (if (< left right)
      (let [mid (+ left (quot (- right left) 2))
            piv (aget data (+ (* (long (aget idx mid)) dim) axis))
            _ (let [t (aget idx mid)] (aset idx mid (aget idx right)) (aset idx right (int t)))
            store (loop [i left s left]
                    (if (< i right)
                      (if (< (aget data (+ (* (long (aget idx i)) dim) axis)) piv)
                        (do (let [t (aget idx i)] (aset idx i (aget idx s)) (aset idx s (int t)))
                            (recur (inc i) (inc s)))
                        (recur (inc i) s))
                      s))
            _ (let [t (aget idx store)] (aset idx store (aget idx right)) (aset idx right (int t)))]
        (if (== store nth)
          nth
          (if (< nth store) (recur left (- store 1)) (recur (+ store 1) right))))
      nth)))

;; Build the tree in place (explicit stack of (node,start,end) frames).
(deftm build-kdtree!
  [data :- (Array double) n :- Long dim :- Long n-nodes :- Long
   idx :- (Array int) idx-start :- (Array int) idx-end :- (Array int)
   lower :- (Array double) upper :- (Array double) is-leaf :- (Array int) stk :- (Array int)]
  :- (Array int)
  (dotimes [i n] (aset idx i (int i)))
  (dotimes [t n-nodes] (aset is-leaf t (int 0)))
  (aset stk 0 (int 0)) (aset stk 1 (int 0)) (aset stk 2 (int n))
  (loop [sp 1]
    (when (> sp 0)
      (let [sp1 (- sp 1)
            base (* sp1 3)
            node (long (aget stk base))
            start (long (aget stk (+ base 1)))
            end (long (aget stk (+ base 2)))
            nb (* node dim)]
        (dotimes [j dim] (aset lower (+ nb j) 1.0e308) (aset upper (+ nb j) -1.0e308))
        (loop [ii start]
          (when (< ii end)
            (let [pb (* (long (aget idx ii)) dim)]
              (dotimes [j dim]
                (let [v (aget data (+ pb j))]
                  (when (< v (aget lower (+ nb j))) (aset lower (+ nb j) v))
                  (when (> v (aget upper (+ nb j))) (aset upper (+ nb j) v)))))
            (recur (inc ii))))
        (aset idx-start node (int start)) (aset idx-end node (int end))
        (if (or (>= (+ (* 2 node) 1) n-nodes) (< (- end start) 2))
          (do (aset is-leaf node (int 1)) (recur sp1))
          (let [axis (find-split-dim data idx dim start end)
                mid (+ start (quot (- end start) 2))
                _ (nth-element! data idx axis dim start end mid)
                lb (* sp1 3) rb (* sp 3)]
            (aset stk lb (int (+ (* 2 node) 1))) (aset stk (+ lb 1) (int start)) (aset stk (+ lb 2) (int mid))
            (aset stk rb (int (+ (* 2 node) 2))) (aset stk (+ rb 1) (int mid)) (aset stk (+ rb 2) (int end))
            (recur (+ sp1 2)))))))
  idx)

;; kNN query for every point. Fills out-idx/out-dst[n*k] with the k nearest
;; (rdist, unsorted heap order; self included). Box-pruned, nearer-child-first.
(deftm tree-query-knn!
  [data :- (Array double) idx :- (Array int) idx-start :- (Array int) idx-end :- (Array int)
   lower :- (Array double) upper :- (Array double) is-leaf :- (Array int)
   n :- Long dim :- Long n-nodes :- Long k :- Long
   out-idx :- (Array int) out-dst :- (Array double)
   heap-p :- (Array double) heap-i :- (Array int) stk-node :- (Array int) stk-lb :- (Array double)]
  :- (Array double)
  (dotimes [q n]
    (let [qb (* q dim)]
      (dotimes [t k] (aset heap-p t 1.0e308) (aset heap-i t (int -1)))
      (aset stk-node 0 (int 0)) (aset stk-lb 0 0.0)
      (loop [sp 1]
        (when (> sp 0)
          (let [sp1 (- sp 1)
                node (long (aget stk-node sp1))
                lb (aget stk-lb sp1)]
            (if (> lb (aget heap-p 0))
              (recur sp1)
              (if (== (aget is-leaf node) 1)
                (do (loop [ii (long (aget idx-start node))]
                      (when (< ii (long (aget idx-end node)))
                        (let [pidx (long (aget idx ii))
                              d (rdist data qb pidx dim)]
                          (when (< d (aget heap-p 0)) (heap-push! heap-p heap-i d pidx k)))
                        (recur (inc ii))))
                    (recur sp1))
                (let [left (+ (* 2 node) 1) right (+ left 1)
                      lbl (point-box-lb lower upper data left qb dim)
                      lbr (point-box-lb lower upper data right qb dim)]
                  (if (<= lbl lbr)
                    (do (aset stk-node sp1 (int right)) (aset stk-lb sp1 lbr)
                        (aset stk-node sp (int left)) (aset stk-lb sp lbl))
                    (do (aset stk-node sp1 (int left)) (aset stk-lb sp1 lbl)
                        (aset stk-node sp (int right)) (aset stk-lb sp lbr)))
                  (recur (+ sp1 2))))))))
      (dotimes [t k]
        (aset out-dst (+ (* q k) t) (aget heap-p t))
        (aset out-idx (+ (* q k) t) (aget heap-i t)))))
  out-dst)

;; --- thin glue ---
(defn n-levels-nodes [n leaf-size]
  (let [lv (clojure.core/inc (int (Math/floor (/ (Math/log (clojure.core/max 1.0 (/ (double (clojure.core/dec n)) (double leaf-size)))) (Math/log 2.0)))))]
    (clojure.core/dec (int (Math/pow 2 lv)))))

(defn build [^doubles data n dim & {:keys [leaf-size] :or {leaf-size 40}}]
  (let [n (long n) dim (long dim)
        n-nodes (n-levels-nodes n leaf-size)
        idx (int-array n) idx-start (int-array n-nodes) idx-end (int-array n-nodes)
        lower (double-array (clojure.core/* n-nodes dim)) upper (double-array (clojure.core/* n-nodes dim))
        is-leaf (int-array n-nodes) stk (int-array (clojure.core/* 3 n-nodes))]
    (build-kdtree! data n dim n-nodes idx idx-start idx-end lower upper is-leaf stk)
    (->KDTree data idx idx-start idx-end lower upper is-leaf n dim n-nodes)))

(defn knn-query [^KDTree t k]
  (let [n (.-n t) dim (.-dim t) k (long k)
        out-idx (int-array (clojure.core/* n k)) out-dst (double-array (clojure.core/* n k))
        heap-p (double-array k) heap-i (int-array k)
        stk-node (int-array (clojure.core/+ 4 (clojure.core/* 2 (.-n-nodes t)))) ; generous
        stk-lb (double-array (clojure.core/+ 4 (clojure.core/* 2 (.-n-nodes t))))]
    (tree-query-knn! (.-data t) (.-idx t) (.-idx-start t) (.-idx-end t)
                     (.-lower t) (.-upper t) (.-is-leaf t) n dim (.-n-nodes t) k
                     out-idx out-dst heap-p heap-i stk-node stk-lb)
    {:idx out-idx :dst out-dst}))

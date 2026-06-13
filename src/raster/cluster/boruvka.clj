(ns raster.cluster.boruvka
  "Boruvka MST over mutual-reachability distance via the KD-tree — ports EVoC's
  parallel_boruvka. The O(n log n) replacement for the dense O(n²) Prim in
  cluster.mst. Reuses raster.spatial.kdtree (build + box-bound + rdist) and
  raster.cluster.tree's union-find (ds-find!/ds-union!).

  Per round: component-aware query finds each point's nearest neighbor in a
  DIFFERENT component (mutual-reach max(rdist, core_i, core_j), pruned by the
  component's best so far); merge keeps each component's cheapest outgoing edge,
  unions them; repeat until one component. Starts from singletons (the kNN-init
  is only an iteration-count optimization). Edge weights stored as sqrt (actual
  euclidean mutual reachability), matching EVoC + cluster.mst/prim-mst!."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == max])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == max sqrt]]
            [raster.spatial.kdtree :as kd]
            [raster.cluster.tree :as ctree]))

;; core[i] = rdist to the min_samples-th neighbor = max of the k-nearest heap row.
(deftm knn-row-max! [dst :- (Array double) n :- Long k :- Long core :- (Array double)] :- (Array double)
  (dotimes [i n]
    (let [ib (* i k)]
      (aset core i (loop [t 0 m 0.0] (if (< t k) (recur (inc t) (max m (aget dst (+ ib t)))) m)))))
  core)

;; point_components[i] = find(i); then node_components bottom-up (a node carries a
;; component id iff its whole subtree is that one component, else -1).
(deftm update-component-vectors!
  [ds-parent :- (Array int) point-comp :- (Array int) node-comp :- (Array int)
   idx :- (Array int) idx-start :- (Array int) idx-end :- (Array int) is-leaf :- (Array int)
   n :- Long n-nodes :- Long] :- (Array int)
  (dotimes [i n] (aset point-comp i (int (ctree/ds-find! ds-parent (long i)))))
  (dotimes [t n-nodes]
    (let [node (- (- n-nodes 1) t)]
      (if (== (aget is-leaf node) 1)
        (let [is (long (aget idx-start node)) ie (long (aget idx-end node))
              cand (aget point-comp (aget idx is))
              same (loop [j (+ is 1)]
                     (if (< j ie)
                       (if (== (aget point-comp (aget idx j)) cand) (recur (inc j)) 0)
                       1))]
          (aset node-comp node (if (== same 1) (int cand) (int -1))))
        (let [left (+ (* 2 node) 1) right (+ left 1)]
          (if (and (== (aget node-comp left) (aget node-comp right))
                   (not (== (aget node-comp left) -1)))
            (aset node-comp node (aget node-comp left))
            (aset node-comp node (int -1)))))))
  point-comp)

;; Component-aware nearest-neighbor query for every point (serial; later points in
;; a component benefit from earlier ones via the shared comp-nn pruning array).
;; Fills cand-dist[q]/cand-idx[q] = q's cheapest mutual-reach edge to another component.
(deftm component-query!
  [data :- (Array double) idx :- (Array int) idx-start :- (Array int) idx-end :- (Array int)
   lower :- (Array double) upper :- (Array double) is-leaf :- (Array int)
   n :- Long dim :- Long
   core :- (Array double) point-comp :- (Array int) node-comp :- (Array int)
   cand-dist :- (Array double) cand-idx :- (Array int) comp-nn :- (Array double)
   stk-node :- (Array int) stk-lb :- (Array double)] :- (Array double)
  (dotimes [q n]
    (let [qb (* q dim) cc (long (aget point-comp q)) ccore (aget core q)]
      (aset stk-node 0 (int 0)) (aset stk-lb 0 (kd/point-box-lb lower upper data 0 qb dim))
      (loop [sp 1]
        (when (> sp 0)
          (let [sp1 (- sp 1) node (long (aget stk-node sp1)) lb (aget stk-lb sp1)]
            (cond
              (> lb (aget cand-dist q)) (recur sp1)
              (or (> lb (aget comp-nn cc)) (> ccore (aget comp-nn cc))) (recur sp1)
              (== (aget node-comp node) cc) (recur sp1)
              (== (aget is-leaf node) 1)
              (do (loop [ii (long (aget idx-start node))]
                    (when (< ii (long (aget idx-end node)))
                      (let [pidx (long (aget idx ii))]
                        (when (and (not (== (aget point-comp pidx) cc))
                                   (< (aget core pidx) (aget comp-nn cc)))
                          (let [d (max (kd/rdist data qb pidx dim) (max ccore (aget core pidx)))]
                            (when (< d (aget cand-dist q))
                              (aset cand-dist q d) (aset cand-idx q (int pidx))
                              (when (< d (aget comp-nn cc)) (aset comp-nn cc d))))))
                      (recur (inc ii))))
                  (recur sp1))
              :else
              (let [left (+ (* 2 node) 1) right (+ left 1)
                    lbl (kd/point-box-lb lower upper data left qb dim)
                    lbr (kd/point-box-lb lower upper data right qb dim)]
                (if (<= lbl lbr)
                  (do (aset stk-node sp1 (int right)) (aset stk-lb sp1 lbr)
                      (aset stk-node sp (int left)) (aset stk-lb sp lbl))
                  (do (aset stk-node sp1 (int left)) (aset stk-lb sp1 lbl)
                      (aset stk-node sp (int right)) (aset stk-lb sp lbr)))
                (recur (+ sp1 2)))))))))
  cand-dist)

;; kNN-init: each point's candidate edge to its NEAREST denser-or-equal neighbor
;; (mutual-reach weight). Pre-merging these collapses many components before the
;; expensive Boruvka rounds (initialize_boruvka_from_knn). knn-dst is rdist.
(deftm init-candidates!
  [knn-idx :- (Array int) knn-dst :- (Array double) core :- (Array double) n :- Long k :- Long
   cand-to :- (Array int) cand-w :- (Array double)] :- (Array int)
  (dotimes [i n]
    (let [ci (aget core i)]
      (loop [t 0 b -1 bd 1.0e308]
        (if (< t k)
          (let [kk (long (aget knn-idx (+ (* i k) t)))
                d (aget knn-dst (+ (* i k) t))]
            (if (and (not (== kk i)) (>= ci (aget core kk)) (< d bd))
              (recur (inc t) kk d)
              (recur (inc t) b bd)))
          (do (aset cand-to i (int b))
              (aset cand-w i (if (>= b 0) (max ci bd) 1.0e308))
              b)))))
  cand-to)

;; --- orchestrator (glue: round loop + per-component best-edge merge + union-find) ---
(defn boruvka-mst
  "Mutual-reachability MST of `emb` (flat n*dim) via KD-tree Boruvka.
   Returns {:from :to :w :core} (n-1 edges, w = sqrt mutual reach), matching cluster.mst."
  [^doubles emb n dim min-samples]
  (let [n (long n) dim (long dim) ms (long min-samples)
        t (kd/build emb n dim)
        data (.-data t) idx (.-idx t) is (.-idx-start t) ie (.-idx-end t)
        lo (.-lower t) up (.-upper t) lf (.-is-leaf t) nn (.-n-nodes t)
        ;; core distances
        kq (kd/knn-query t (clojure.core/inc ms))
        core (double-array n)
        _ (knn-row-max! (:dst kq) n (clojure.core/inc ms) core)
        ;; union-find + component vectors
        dsp (int-array n) dsr (int-array n)
        _ (dotimes [i n] (aset dsp i (int i)))
        pc (int-array n) ncv (int-array nn)
        cand-d (double-array n) cand-i (int-array n) comp-nn (double-array n)
        stk-node (int-array (clojure.core/+ 4 (clojure.core/* 2 nn)))
        stk-lb (double-array (clojure.core/+ 4 (clojure.core/* 2 nn)))
        ef (int-array (clojure.core/dec n)) et (int-array (clojure.core/dec n)) ew (double-array (clojure.core/dec n))
        ;; kNN-init: pre-merge each point to its nearest denser neighbor
        cand-to (int-array n) cand-w (double-array n)
        _ (init-candidates! (:idx kq) (:dst kq) core n (clojure.core/inc ms) cand-to cand-w)
        ne0 (loop [i 0 e 0]
              (if (clojure.core/< i n)
                (let [to (clojure.core/aget cand-to i)]
                  (if (clojure.core/>= to 0)
                    (let [fc (ctree/ds-find! dsp (long i)) tc (ctree/ds-find! dsp (long to))]
                      (if (clojure.core/not= fc tc)
                        (do (ctree/ds-union! dsp dsr (long i) (long to))
                            (aset ef e (int i)) (aset et e (int to))
                            (aset ew e (Math/sqrt (clojure.core/aget cand-w i)))
                            (recur (clojure.core/inc i) (clojure.core/inc e)))
                        (recur (clojure.core/inc i) e)))
                    (recur (clojure.core/inc i) e)))
                e))
        _ (update-component-vectors! dsp pc ncv idx is ie lf n nn)]
    (loop [n-comp (clojure.core/- (long n) (long ne0)) ne (long ne0) guard 0]
      (if (clojure.core/and (clojure.core/> n-comp 1) (clojure.core/< guard 64))
        (do
          ;; reset per-round buffers
          (dotimes [i n] (aset cand-d i 1.0e308) (aset cand-i i (int -1)))
          (dotimes [i n] (aset comp-nn i 1.0e308))
          (component-query! data idx is ie lo up lf n dim core pc ncv cand-d cand-i comp-nn stk-node stk-lb)
          ;; best outgoing edge per component
          (let [bd (double-array n) bp (int-array n)]
            (dotimes [i n] (aset bd i 1.0e308) (aset bp i (int -1)))
            (dotimes [i n]
              (let [c (clojure.core/aget pc i)]
                (when (clojure.core/< (clojure.core/aget cand-d i) (clojure.core/aget bd c))
                  (clojure.core/aset bd c (clojure.core/aget cand-d i))
                  (clojure.core/aset bp c (int i)))))
            ;; union + emit
            (let [ne2 (loop [c 0 e ne]
                        (if (clojure.core/< c n)
                          (let [pi (clojure.core/aget bp c)]
                            (if (clojure.core/>= pi 0)
                              (let [j (clojure.core/aget cand-i pi)
                                    fc (ctree/ds-find! dsp (long pi)) tc (ctree/ds-find! dsp (long j))]
                                (if (clojure.core/not= fc tc)
                                  (do (ctree/ds-union! dsp dsr (long pi) (long j))
                                      (aset ef e (int pi)) (aset et e (int j))
                                      (aset ew e (Math/sqrt (clojure.core/aget bd c)))
                                      (recur (clojure.core/inc c) (clojure.core/inc e)))
                                  (recur (clojure.core/inc c) e)))
                              (recur (clojure.core/inc c) e)))
                          e))]
              (update-component-vectors! dsp pc ncv idx is ie lf n nn)
              (recur (clojure.core/- (long n) (clojure.core/- ne2 0)
                                     ) ne2 (clojure.core/inc guard)))))
        ;; done — trim to ne edges
        {:from (java.util.Arrays/copyOf ef (int ne))
         :to (java.util.Arrays/copyOf et (int ne))
         :w (java.util.Arrays/copyOf ew (int ne))
         :core core}))))

(ns raster.spatial.rptree
  "Random-projection tree (forest) for approximate-NN initialization — ports
  EVoC's float RP-tree (float_random_projection_split / make_float_tree). Splits
  points by a random hyperplane (between two random points), recursively, until
  leaves of <= leaf-size. We collect LEAF RANGES over a reordered index array
  (no complete-tree node indexing — RP-trees are unbalanced), which is all the
  NN-descent init needs (each leaf's points seed each other's candidate lists).

  General spatial structure (sibling to kdtree); reuses raster.umap/tau-rand-int!."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == mod])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == mod sqrt]]
            [raster.umap :as u]))

;; Partition idx[start,end) by a random hyperplane; collect leaf ranges.
;; Reorders `idx`; writes leaf-start[]/leaf-end[]; returns the leaf count.
;; rng-state is a long[3] (mutated); hp is a dim-length work buffer; stk holds
;; (start,end) frames (2 ints each).
(deftm build-rptree-leaves!
  (All [T] [data :- (Array T) n :- Long dim :- Long leaf-size :- Long
   idx :- (Array int) rng-state :- (Array long) hp :- (Array double) stk :- (Array int)
   leaf-start :- (Array int) leaf-end :- (Array int)] :- Long
  (dotimes [i n] (aset idx i (int i)))
  (aset stk 0 (int 0)) (aset stk 1 (int n))
  (loop [sp 1 nl 0]
    (if (> sp 0)
      (let [sp1 (- sp 1) base (* sp1 2)
            start (long (aget stk base)) end (long (aget stk (+ base 1)))
            sz (- end start)]
        (if (<= sz leaf-size)
          (do (aset leaf-start nl (int start)) (aset leaf-end nl (int end)) (recur sp1 (inc nl)))
          (let [li (mod (u/tau-rand-int! rng-state 0) sz)
                ri0 (mod (u/tau-rand-int! rng-state 0) sz)
                ri (if (== li ri0) (mod (+ ri0 1) sz) ri0)
                left (long (aget idx (+ start li)))
                right (long (aget idx (+ start ri)))
                ;; hyperplane = normalize(data[left] - data[right])
                _ (let [nrm (loop [d 0 s 0.0]
                              (if (< d dim)
                                (let [v (double (- (aget data (+ (* left dim) d)) (aget data (+ (* right dim) d))))]
                                  (aset hp d v) (recur (inc d) (+ s (* v v))))
                                s))
                        r (sqrt nrm)
                        inv (/ 1.0 (if (< r 1.0e-8) 1.0 r))]
                    (dotimes [d dim] (aset hp d (* (aget hp d) inv))))
                ;; Lomuto partition: margin>0 -> left (front)
                split (loop [i start s start]
                        (if (< i end)
                          (let [pidx (long (aget idx i))
                                margin (loop [d 0 acc 0.0]
                                         (if (< d dim)
                                           (recur (inc d) (+ acc (* (aget hp d) (aget data (+ (* pidx dim) d)))))
                                           acc))
                                cls (if (< (Math/abs margin) 1.0e-8)
                                      (mod (u/tau-rand-int! rng-state 0) 2)
                                      (if (> margin 0.0) 0 1))]
                            (if (== cls 0)
                              (do (let [t (aget idx i)] (aset idx i (aget idx s)) (aset idx s (int t)))
                                  (recur (inc i) (inc s)))
                              (recur (inc i) s)))
                          s))
                split2 (if (or (== split start) (== split end)) (+ start (quot sz 2)) split)]
            (aset stk base (int start)) (aset stk (+ base 1) (int split2))
            (aset stk (* sp 2) (int split2)) (aset stk (+ (* sp 2) 1) (int end))
            (recur (+ sp1 2) nl))))
      nl))))

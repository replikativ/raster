(ns raster.cluster.embed
  "EVoC node embedding — ports node_embedding_epoch. Structurally identical to
  UMAP's optimize-layout! (per-edge attractive + negative-sampled SGD over the
  fuzzy graph), but with EVoC's grad coefficients and a deterministic uint32-
  arithmetic negative-sample index (not Tausworthe):

    attractive: dist=√d², gc = (-2·noise·dist - 2)/(2·d² - 0.5·dist + 1)   [no clip]
    negative  : k = ((n+p)·i·rng) mod n_vertices (uint32 wrap); if d²>1e-2:
                gc = 4/((1+0.25·d²)·d²); grad clipped to [-4,4]

  Uses a local uclip (gradient clip to [-4,4]). The whole multi-epoch solve compiles to one method."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= == mod bit-and])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == sqrt mod bit-and]]))

;; Clamp gradient into [-4, 4] (umap.layouts.clip) — local to keep clustering
;; independent of the UMAP layer (umap-rstr).
(deftm uclip [v :- Double] :- Double
  (if (> v 4.0) 4.0 (if (< v -4.0) -4.0 v)))

(deftm node-embedding-layout!
  [emb :- (Array double) head :- (Array int) tail :- (Array int)
   eps :- (Array double) epn :- (Array double) eons :- (Array double) eonsn :- (Array double)
   rng-vals :- (Array long)
   noise :- Double init-alpha :- Double n-vertices :- Long dim :- Long n-epochs :- Long]
  :- (Array double)
  (let [n-edges (alength eps)]
    (dotimes [ep n-epochs]
      (let [epd (double ep)
            rng (aget rng-vals ep)
            alpha (if (== ep 0)
                    init-alpha
                    (* init-alpha (- 1.0 (/ (double (dec ep)) (double n-epochs)))))]
        (dotimes [i n-edges]
          (when (<= (aget eons i) epd)
            (let [j (long (aget head i))
                  k (long (aget tail i))
                  jb (* j dim) kb (* k dim)
                  d2 (loop [d 0 acc 0.0]
                       (if (< d dim)
                         (recur (inc d)
                                (let [df (- (aget emb (+ jb d)) (aget emb (+ kb d)))]
                                  (+ acc (* df df))))
                         acc))]
              ;; attractive (only when d2>0; no clip)
              (when (> d2 0.0)
                (let [dist (sqrt d2)
                      gc (/ (- (* (* -2.0 noise) dist) 2.0)
                            (+ (- (* 2.0 d2) (* 0.5 dist)) 1.0))]
                  (dotimes [d dim]
                    (let [cd (aget emb (+ jb d)) od (aget emb (+ kb d))
                          ga (* (* gc (- cd od)) alpha)]
                      (aset emb (+ jb d) (+ cd ga))
                      (aset emb (+ kb d) (- od ga))))))
              (aset eons i (+ (aget eons i) (aget eps i)))
              ;; negative samples: deterministic uint32-arith index
              (let [n-neg (long (/ (- epd (aget eonsn i)) (aget epn i)))]
                (dotimes [p n-neg]
                  ;; numba evaluates ((n+p)*i*rng) in int64 (fits, no 32-bit wrap)
                  (let [kk (mod (* (* (+ ep (long p)) i) rng) n-vertices)
                        kb2 (* kk dim)
                        d2n (loop [d 0 acc 0.0]
                              (if (< d dim)
                                (recur (inc d)
                                       (let [df (- (aget emb (+ jb d)) (aget emb (+ kb2 d)))]
                                         (+ acc (* df df))))
                                acc))]
                    (when (> d2n 0.01)
                      (let [gcn (/ 4.0 (* (+ 1.0 (* 0.25 d2n)) d2n))]
                        (dotimes [d dim]
                          (let [cd (aget emb (+ jb d)) od (aget emb (+ kb2 d))
                                g (uclip (* gcn (- cd od)))]
                            (aset emb (+ jb d) (+ cd (* g alpha)))))))))
                (aset eonsn i (+ (aget eonsn i) (* (double n-neg) (aget epn i))))))))))
    emb))

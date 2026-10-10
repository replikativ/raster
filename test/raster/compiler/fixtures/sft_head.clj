(ns raster.compiler.fixtures.sft-head
  "SFT loss/head-cotangent witness from finetune-rstr 9e9ba5d6.
   Numerical bodies are retained, only helper namespace qualification changes.
   No private dependency or legacy host orchestration is loaded by this fixture."
  (:refer-clojure :exclude [aget aset alength])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength alloc-like]]
            [raster.math :as m] [raster.numeric :as n]
            [raster.dl.nn :as nn] [raster.par]))

(defn inputs
  "Fresh dense inputs; rows are distinct, including the zero-weight padding position."
  ([] (inputs 0))
  ([batch]
   (let [[rows targets weights] (nth [[[0 2] [1 4] [0.5 0.5]]
                                     [[2 1] [3 2] [1.0 0.0]]
                                     [[1 0] [0 6] [0.25 0.75]]] batch)]
     {:y (float-array (map #(+ (* 0.125 (- (mod % 7) 3))
                               (* 0.03125 batch (- (mod % 3) 1))) (range 24)))
      :rows (long-array rows) :targets (long-array targets) :w (float-array weights)
      :E (float-array (map #(* 0.03125 (- (mod (+ (* % 3) (quot % 8)) 13) 6)) (range 64)))
      :wf (float-array (map #(* 0.0625 (- (mod % 5) 2)) (range 8)))
      :lse (float-array 2) :nll (float-array 2) :g (float-array 24)
      :seq-len 3 :m 2 :d 8 :vocab 8 :chunks 2 :eps 1.0e-6})))

(defn arguments [{:keys [y rows targets w E wf lse nll g seq-len m d vocab chunks eps]}]
  [y rows targets w E wf lse nll g seq-len m d vocab chunks eps])

(defn oracle
  "Independent Double-coordinate softmax loss and RMSNorm input adjoint.
   This does not call Raster normalization, linear algebra, loss or AD. Distinct selected
   rows are the fixture's scatter contract; unselected and zero-weight rows remain zero."
  [{:keys [y rows targets w E wf seq-len m d vocab eps]}]
  (let [g (double-array (* seq-len d))
        nll (double-array m)]
    (dotimes [b m]
      (let [row (int (nth rows b))
            x (mapv #(double (nth y (+ (* row d) %))) (range d))
            inv (/ 1.0 (Math/sqrt (+ eps (/ (reduce + 0.0 (map #(* % %) x)) d))))
            gains (mapv #(inc (double %)) wf)
            h (mapv #(* (nth x %) (nth gains %) inv) (range d))
            logits (mapv (fn [o]
                           (reduce + 0.0 (for [j (range d)]
                                           (* (nth h j) (double (nth E (+ (* o d) j)))))))
                         (range vocab))
            maximum (apply max logits)
            exps (mapv #(Math/exp (- % maximum)) logits)
            denominator (reduce + 0.0 exps)
            weight (double (nth w b))
            target (int (nth targets b))
            dl (mapv #(* weight (- (/ (nth exps %) denominator)
                                   (if (= % target) 1.0 0.0))) (range vocab))
            dh (mapv (fn [j]
                       (reduce + 0.0 (for [o (range vocab)]
                                       (* (nth dl o) (double (nth E (+ (* o d) j)))))))
                     (range d))
            mean-dot (/ (reduce + 0.0 (map #(* %1 %2 %3) x dh gains)) d)]
        (aset nll b (* weight (- (+ maximum (Math/log denominator)) (nth logits target))))
        (dotimes [j d]
          (aset g (+ (* row d) j)
                (* inv (- (* (nth dh j) (nth gains j))
                          (* (nth x j) inv inv mean-dot)))))))
    {:nll nll :g g}))

(deftm
 gather-rows
 (All
  [T]
  [y :- (Array T) rows :- (Array long) m :- Long d :- Long]
  :-
  (Array T)
  (let
   [out (alloc-like y (* m d))]
   (raster.par/map-void!
    t
    (clojure.core/* m d)
    (let
     [i (quot t d) j (rem t d) src (clojure.core/+ (clojure.core/* (aget rows i) d) j)]
     (aset out t (aget y src))))
   out)))
(deftm
 zero-buf!
 (All [T] [g :- (Array T) n :- Long] :- Void (raster.par/map-void! t n (aset g t 0.0))))
(deftm
 scatter-rows!
 (All
  [T]
  [dh :- (Array T) rows :- (Array long) g :- (Array T) m :- Long d :- Long]
  :-
  Void
  (raster.par/map-void!
   t
   (clojure.core/* m d)
   (let
    [i (quot t d) j (rem t d) dst (clojure.core/+ (clojure.core/* (aget rows i) d) j)]
    (aset g dst (aget dh t))))))
(deftm
 chunk-max
 (All
  [T]
  [logits :- (Array T) m :- Long vocab :- Long chunks :- Long]
  :-
  (Array T)
  (let
   [pm
    (alloc-like logits (* m chunks))
    csz
    (quot (clojure.core/+ vocab (clojure.core/- chunks 1)) chunks)]
   (raster.par/map-void!
    t
    (clojure.core/* m chunks)
    (let
     [i
      (quot t chunks)
      c
      (rem t chunks)
      off
      (clojure.core/* i vocab)
      start
      (clojure.core/* c csz)
      e0
      (clojure.core/+ start csz)
      end
      (if (< e0 vocab) e0 vocab)
      mx
      (loop
       [k start mm -1.0E38]
       (if (< k end) (recur (inc k) (n/max mm (aget logits (clojure.core/+ off k)))) mm))]
     (aset pm t mx)))
   pm)))
(deftm
 row-max
 (All
  [T]
  [pm :- (Array T) m :- Long chunks :- Long]
  :-
  (Array T)
  (let
   [rmax (alloc-like pm m)]
   (raster.par/map-void!
    i
    m
    (let
     [off
      (clojure.core/* i chunks)
      mx
      (loop
       [c 0 mm -1.0E38]
       (if (< c chunks) (recur (inc c) (n/max mm (aget pm (clojure.core/+ off c)))) mm))]
     (aset rmax i mx)))
   rmax)))
(deftm
 chunk-sumexp
 (All
  [T]
  [logits :- (Array T) rmax :- (Array T) m :- Long vocab :- Long chunks :- Long]
  :-
  (Array T)
  (let
   [ps
    (alloc-like logits (* m chunks))
    csz
    (quot (clojure.core/+ vocab (clojure.core/- chunks 1)) chunks)]
   (raster.par/map-void!
    t
    (clojure.core/* m chunks)
    (let
     [i
      (quot t chunks)
      c
      (rem t chunks)
      off
      (clojure.core/* i vocab)
      start
      (clojure.core/* c csz)
      e0
      (clojure.core/+ start csz)
      end
      (if (< e0 vocab) e0 vocab)
      mi
      (aget rmax i)
      s
      (loop
       [k start acc 0.0]
       (if
        (< k end)
        (recur (inc k) (n/+ acc (m/exp (n/- (aget logits (clojure.core/+ off k)) mi))))
        acc))]
     (aset ps t s)))
   ps)))
(deftm
 ce-rows-finish!
 (All
  [T]
  [psum
   :-
   (Array T)
   rmax
   :-
   (Array T)
   logits
   :-
   (Array T)
   targets
   :-
   (Array long)
   w
   :-
   (Array T)
   lse
   :-
   (Array T)
   nll
   :-
   (Array T)
   m
   :-
   Long
   vocab
   :-
   Long
   chunks
   :-
   Long]
  :-
  Void
  (raster.par/map-void!
   i
   m
   (let
    [off
     (clojure.core/* i chunks)
     s
     (loop
      [c 0 acc 0.0]
      (if (< c chunks) (recur (inc c) (n/+ acc (aget psum (clojure.core/+ off c)))) acc))
     l
     (n/+ (aget rmax i) (m/log s))]
    (aset lse i l)
    (aset
     nll
     i
     (n/*
      (aget w i)
      (n/- l (aget logits (clojure.core/+ (clojure.core/* i vocab) (aget targets i))))))))))
(deftm
 ce-rows-dlogits
 (All
  [T]
  [logits :- (Array T) w :- (Array T) lse :- (Array T) m :- Long vocab :- Long]
  :-
  (Array T)
  (let
   [d (alloc-like logits (* m vocab))]
   (raster.par/map-void!
    t
    (clojure.core/* m vocab)
    (let
     [i (quot t vocab) p (m/exp (n/- (aget logits t) (aget lse i)))]
     (aset d t (n/* p (aget w i)))))
   d)))
(deftm
 ce-rows-subtract-onehot!
 (All
  [T]
  [d :- (Array T) targets :- (Array long) w :- (Array T) m :- Long vocab :- Long]
  :-
  Void
  (raster.par/map-void!
   i
   m
   (let
    [idx (clojure.core/+ (clojure.core/* i vocab) (aget targets i))]
    (aset d idx (n/- (aget d idx) (aget w i)))))))
(deftm
 head-step!
 [y
  :-
  (Array float)
  rows
  :-
  (Array long)
  targets
  :-
  (Array long)
  w
  :-
  (Array float)
  E
  :-
  (Array float)
  wf
  :-
  (Array float)
  lse
  :-
  (Array float)
  nll
  :-
  (Array float)
  g
  :-
  (Array float)
  seq-len
  :-
  Long
  m
  :-
  Long
  d
  :-
  Long
  vocab
  :-
  Long
  chunks
  :-
  Long
  eps
  :-
  Double]
 :-
 (Array float)
 (let
  [hsel
   (raster.compiler.fixtures.sft-head/gather-rows y rows m d)
   hn
   (raster.dl.nn/rms-norm hsel wf m d eps 1.0)
   logits
   (raster.dl.nn/linear-nb hn E m d vocab)]
  (let
   [pmax
    (raster.compiler.fixtures.sft-head/chunk-max logits m vocab chunks)
    rmax
    (raster.compiler.fixtures.sft-head/row-max pmax m chunks)
    psum
    (raster.compiler.fixtures.sft-head/chunk-sumexp logits rmax m vocab chunks)]
   (raster.compiler.fixtures.sft-head/ce-rows-finish!
    psum
    rmax
    logits
    targets
    w
    lse
    nll
    m
    vocab
    chunks))
  (let
   [dlog (raster.compiler.fixtures.sft-head/ce-rows-dlogits logits w lse m vocab)]
   (raster.compiler.fixtures.sft-head/ce-rows-subtract-onehot! dlog targets w m vocab)
   (let
    [dhn
     (raster.dl.nn/linear-dx dlog E m d vocab)
     dhsel
     (raster.dl.nn/rms-norm-backward-dx dhn hsel wf m d eps 1.0)]
    (raster.compiler.fixtures.sft-head/zero-buf! g (clojure.core/* seq-len d))
    (raster.compiler.fixtures.sft-head/scatter-rows! dhsel rows g m d)
    g))))

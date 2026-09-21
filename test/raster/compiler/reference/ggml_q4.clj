(ns raster.compiler.reference.ggml-q4
  "Retired serial Q4_K device algorithm retained only as an independent performance reference."
  (:require [clojure.walk]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
            [raster.par :as par]))

(defn- lane-form [lane]
  (let [named #(symbol (str % "-l" lane))
        chunk (named "chunk")
        acc (named "acc")
        word (named "word")
        weight (named "weight")
        activation (named "activation")]
    (list 'loop [chunk 0 acc 0]
          (list 'if (list '< chunk 8)
                (list 'let [word (list '+ (list '* chunk 8) lane)
                            weight (list 'ra/aget 'wq (list '+ 'ww word))
                            activation (list 'ra/aget 'xq (list '+ 'xw word))]
                      (list 'recur (list 'inc chunk)
                            (list '+ acc
                                  (list '* (list 'ra/aget 'wsc (list '+ 'scb chunk))
                                        (list 'par/dp4a weight activation 0)))))
                acc))))

(defn- expand-lanes [template]
  (clojure.walk/postwalk
   (fn [form]
     (if (and (seq? form) (= 'lane (first form)))
       (lane-form (second form))
       form))
   template))

(defmacro ^:private define-serial-q4 []
  (expand-lanes
   '(deftm serial-qdot-q4-K-rows!
      [xq :- (Array int), xd :- (Array float), xbs :- (Array int),
       wq :- (Array int), wd :- (Array float), wdmin :- (Array float),
       wsc :- (Array int), wm :- (Array int),
       y :- (Array float), in :- Long, out :- Long, nrows :- Long] :- Void
      (par/map-void! ro (* nrows out)
                     (let [row (quot ro out)
                           o (rem ro out)
                           nb (quot in 256)
                           acc
                           (loop [b 0 s0 (float 0.0) s1 (float 0.0)
                                  s2 (float 0.0) s3 (float 0.0)
                                  s4 (float 0.0) s5 (float 0.0)
                                  s6 (float 0.0) s7 (float 0.0)
                                  sumf (float 0.0)]
                             (if (< b nb)
                               (let [wb (+ (* o nb) b)
                                     xb (+ (* row nb) b)
                                     xw (* xb 64)
                                     ww (* wb 64)
                                     scb (* wb 8)
                                     d (* (ra/aget wd wb) (ra/aget xd xb))
                                     dmin (* (ra/aget wdmin wb) (ra/aget xd xb))
                                     sumi (loop [j 0 s 0]
                                            (if (< j 16)
                                              (recur (inc j)
                                                     (+ s (* (ra/aget xbs (+ (* xb 16) j))
                                                             (ra/aget wm (+ scb (quot j 2))))))
                                              s))
                                     l0 (lane 0) l1 (lane 1) l2 (lane 2) l3 (lane 3)
                                     l4 (lane 4) l5 (lane 5) l6 (lane 6) l7 (lane 7)
                                     p0 (* d (float l0)) p1 (* d (float l1))
                                     p2 (* d (float l2)) p3 (* d (float l3))
                                     p4 (* d (float l4)) p5 (* d (float l5))
                                     p6 (* d (float l6)) p7 (* d (float l7))
                                     pm (* dmin (float sumi))]
                                 (recur (inc b)
                                        (+ s0 p0) (+ s1 p1) (+ s2 p2) (+ s3 p3)
                                        (+ s4 p4) (+ s5 p5) (+ s6 p6) (+ s7 p7)
                                        (- sumf pm)))
                               (+ (+ (+ (+ (+ (+ (+ (+ sumf s0) s1) s2) s3)
                                             s4) s5) s6) s7)))]
                       (ra/aset y ro acc))))))

(define-serial-q4)

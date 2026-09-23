(ns raster.dl.attention-reference
  "Host-only attention oracles. This namespace is outside the production compiler corpus."
  (:refer-clojure :exclude [aget aset + - * /])
  (:require [raster.arrays :refer [aget aset alloc-like]]
            [raster.core :refer [deftm]]
            [raster.math :as m]
            [raster.numeric :as n :refer [+ - * /]]))

(deftm gqa-decode
  "Sequential single-query GQA/MQA oracle used only for numerical kernel comparisons.

  Keeping this as a test-namespace generic preserves the exact float/double specialization and
  intermediate rounding of the retired production reference without making it a production
  compiler input."
  (All [T]
       [q :- (Array T) k :- (Array T) v :- (Array T)
        cache-len :- Long n-q :- Long n-kv :- Long
        head-dim :- Long scale :- Double] :- (Array T)
       (let [out (alloc-like q (* n-q head-dim))
             group (quot n-q n-kv)
             neg-inf (n/neg-inf-val (aget q 0))]
         (dotimes [hq n-q]
           (let [hkv (quot hq group)
                 qb (* hq (int head-dim))
                 sc (alloc-like q cache-len)
                 _ (dotimes [j cache-len]
                     (let [kb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))
                           dot (loop [d 0 acc 0.0]
                                 (if (< d head-dim)
                                   (recur (inc d)
                                          (+ acc (* (aget q (+ qb d))
                                                    (aget k (+ kb d)))))
                                   acc))]
                       (aset sc j (* dot scale))))
                 mx (loop [j 0 mm neg-inf]
                      (if (< j cache-len)
                        (recur (inc j) (n/max mm (aget sc j)))
                        mm))
                 sum (loop [j 0 s 0.0]
                       (if (< j cache-len)
                         (let [e (m/exp (- (aget sc j) mx))]
                           (aset sc j e)
                           (recur (inc j) (+ s e)))
                         s))
                 inv (/ 1.0 sum)
                 ob (* hq (int head-dim))]
             (dotimes [d head-dim]
               (aset out (+ ob d)
                     (loop [j 0 a 0.0]
                       (if (< j cache-len)
                         (let [kvb (+ (* j (* n-kv head-dim)) (* hkv (int head-dim)))]
                           (recur (inc j)
                                  (+ a (* (* (aget sc j) inv)
                                          (aget v (+ kvb d))))))
                         a))))))
         out)))

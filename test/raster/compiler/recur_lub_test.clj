(ns raster.compiler.recur-lub-test
  "Recur-LUB fixpoint regression: a loop var's type is LUB(init, recur args),
   not the init's class. Before the walker rewalk-fixpoint, an integer-literal
   accumulator carrying doubles was stamped `long`, and the stamp-driven
   (long acc) cast TRUNCATED the running value every iteration — silent
   numerical corruption in BOTH the interpreted and compiled paths:
   (loop [acc 0] (+ acc 1.5)) over 3 elements summed 3.5 instead of 4.5.

   Surfaced by the A2 bytecode stamp-vs-LUB differential census (the
   'stamped long but derived :double' class)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.compiler.pipeline :as pl]
            [clojure.walk :as w]))

(deftm lub-sum-k [x :- (Array double) n :- Long] :- Double
  ;; acc deliberately initialized with the INTEGER literal 0 — the natural
  ;; spelling that used to truncate
  (loop [i 0 acc 0]
    (if (< i n)
      (recur (inc i) (+ acc (ra/aget x i)))
      (double acc))))

(deftest integer-init-accumulator-does-not-truncate
  (let [xs (double-array [1.5 1.5 1.5])]
    (testing "interpreted path"
      (is (= 4.5 (lub-sum-k xs 3))))
    (testing "compiled path"
      (is (= 4.5 ((pl/compile-aot #'lub-sum-k) xs 3))))))

(deftest loop-var-stamp-is-the-lub
  (testing "walked IR stamps the accumulator with the widened type"
    (let [wb ((deref #'pl/get-walked-body) #'lub-sum-k :double)
          tags (volatile! {})]
      (w/postwalk
       (fn [f]
         (when (and (seq? f) (= 'loop* (first f)))
           (doseq [[sym _] (partition 2 (second f))]
             (vswap! tags assoc (symbol (re-find #"^[a-z-]+" (name sym)))
                     (:raster.type/tag (meta sym)))))
         f)
       wb)
      (is (= 'double (get @tags 'acc))
          "acc must be widened long→double by the recur-LUB fixpoint")
      (is (= 'long (get @tags 'i))
          "the genuine integer counter stays long"))))

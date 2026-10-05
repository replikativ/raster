(ns raster.ad.compiled-value-grad-test
  "value+grad :compile? builds the runtime gradient with the compile-aot
  pipeline: same values as the evaluated body, compiled; compiled code reads
  var roots at each call; scalar adjoints accumulate without dispatch."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.arrays :as ra]
            [raster.math :as rm]
            [raster.numeric :as n]
            [raster.compiler.pipeline :as pipeline]))

(deftm logistic-lp [b0 :- Double, b1 :- Double, b2 :- Double,
                    xs :- (Array double), ys :- (Array double), cnt :- Long] :- Double
  (loop [i 0 acc (n/* -0.5 (n/+ (n/* b0 b0) (n/+ (n/* b1 b1) (n/* b2 b2))))]
    (if (< i cnt)
      (let [eta (n/+ b0 (n/+ (n/* b1 (ra/aget xs (* 2 i))) (n/* b2 (ra/aget xs (inc (* 2 i))))))]
        (recur (inc i) (n/+ acc (n/- (n/* (ra/aget ys i) eta) (rm/log (n/+ 1.0 (rm/exp eta)))))))
      acc)))

(def ^:private cnt 200)
(def ^:private xs (let [r (java.util.Random. 1)] (double-array (repeatedly (* 2 cnt) #(.nextGaussian r)))))
(def ^:private ys (let [r (java.util.Random. 2)] (double-array (repeatedly cnt #(if (< (.nextDouble r) 0.5) 1.0 0.0)))))

(defn- close? [a b] (<= (Math/abs (- (double a) (double b))) (* 1e-12 (max 1.0 (Math/abs (double b))))))

(deftest compile?-matches-the-evaluated-gradient
  (let [reference (rev/value+grad #'logistic-lp :wrt [0 1 2])
        compiled (rev/value+grad #'logistic-lp :wrt [0 1 2] :compile? true)]
    (is (not (::rev/compiled? (meta reference))))
    (is (::rev/compiled? (meta compiled)))
    (doseq [[b0 b1 b2] [[0.0 0.0 0.0] [0.1 -0.2 0.3] [1.5 2.0 -1.0]]]
      (let [r (reference b0 b1 b2 xs ys cnt)
            c (compiled b0 b1 b2 xs ys cnt)]
        (testing [b0 b1 b2]
          (is (every? true? (map close? (take 4 c) (take 4 r))) (str c " vs " r)))))))

(def ^:dynamic *offset* 0.0)
(defn offset-of [x] (+ (double x) 1.0))

(deftm calls-a-var [x :- Double] :- Double
  (n/+ x (double (offset-of x))))

(deftest compiled-code-reads-var-roots-at-each-call
  (let [f (pipeline/compile-aot #'calls-a-var)]
    (is (== 5.0 (f 2.0)))
    (with-redefs [offset-of (fn [x] (* 10.0 (double x)))]
      (is (== 22.0 (f 2.0)) "a redefinition is seen through the var constant"))
    (is (== 5.0 (f 2.0)))))

(deftest scalar-adjoints-accumulate-as-addition
  (is (== 3.5 (rev/grad-acc 1.25 2.25)))
  (is (nil? (rev/grad-acc nil nil)))
  (is (== 2.0 (rev/grad-acc nil 2.0)))
  (is (= [3.0 4.0] (vec (rev/grad-acc 1.0 (double-array [2.0 3.0]))))))

(deftm scaled-sum [x :- (Array double), k :- Double] :- Double
  (let [n (alength x) tmp (double-array n)]
    (dotimes [i n] (aset tmp i (* k (aget x i))))
    (loop [i 0 acc 0.0] (if (< i n) (recur (inc i) (+ acc (aget tmp i))) acc))))

(defn- concurrent-results
  "Each of four threads calls f on its own args many times; returns the
  distinct results each thread saw."
  [f arg-lists]
  (let [runs (doall (for [args arg-lists]
                      (future (into #{} (repeatedly 2000 #(let [r (apply f args)]
                                                            (if (sequential? r)
                                                              (mapv (fn [v] (if (number? v) v (vec v))) r)
                                                              r)))))))]
    (mapv deref runs)))

(deftest reentrant-compiled-functions-allocate-per-call
  (let [f (pipeline/compile-aot #'scaled-sum :reentrant? true)
        arg-lists (for [k (range 4)] [(double-array (range (+ 100 k))) (double (inc k))])]
    (is (= (mapv #(hash-set (apply scaled-sum %)) arg-lists)
           (concurrent-results f arg-lists))
        "concurrent calls with different shapes each see their own result")))

(deftest compiled-value+grad-is-reentrant
  (let [vg (rev/value+grad #'logistic-lp :wrt [0 1 2] :compile? true)
        arg-lists (for [k (range 4)] [(* 0.1 k) (- 0.2 (* 0.1 k)) 0.3 xs ys (- cnt (* 10 k))])
        expected (mapv #(let [r (apply vg %)] (hash-set (vec (take 4 r)))) arg-lists)]
    (is (= expected (mapv (fn [s] (into #{} (map #(vec (take 4 %))) s))
                          (concurrent-results vg arg-lists))))))

(ns raster.types.int8-test
  "Int8 as a Julia-style bit-type with a WIDENING orientation: construction wraps
   to signed 8-bit, widening sign-extends into the int tower, and Int8 mixed with a
   wider type promotes to that type. Arrays stay (Array byte); the quantized dot is
   int8×int8 → i32 accumulate (exact on every backend)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.types.int8 :as i8 :refer [int8 int8->long]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.numeric :as rn]
            [raster.compiler.pipeline :as pl]))

;; int8 widening dot over byte[] storage — the quantized inner product.
(deftm i8dot [x :- (Array byte), y :- (Array byte), n :- Long] :- Long
  (loop [i 0 acc 0]
    (if (clojure.core/< (long i) (long n))
      (recur (clojure.core/inc (long i))
             (clojure.core/+ (long acc)
                             (clojure.core/* (int8->long (int8 (ra/aget x i)))
                                             (int8->long (int8 (ra/aget y i))))))
      acc)))

(defn- gen [n f] (byte-array (map #(unchecked-byte (f %)) (range n))))
(defn- ref-dot [^bytes x ^bytes y n] (reduce + (map (fn [a b] (* (long a) (long b))) x y)))

(deftest int8-construction-and-widening
  (testing "construction wraps to signed 8-bit; widening sign-extends"
    (is (= 42 (int8->long (int8 42))))
    (is (= -7 (int8->long (int8 -7))))
    (is (= -56 (int8->long (int8 200))) "200 wraps to -56 (like Julia Int8)")
    (is (= 127 (int8->long (int8 127))))
    (is (= -128 (int8->long (int8 128))) "128 wraps to -128")))

(deftest int8-promotes-up
  (testing "Int8 mixed with a wider type widens to that type (never stays int8)"
    (is (= 15 (rn/+ (int8 10) (long 5))))
    (is (= -3 (rn/+ (int8 -8) (long 5))))
    (is (= 30 (rn/* (int8 6) (long 5))))))

(deftest int8-widening-dot-interpreted
  (testing "int8×int8 → i64 accumulate matches reference (interpreted)"
    (doseq [n [1 8 33 100]]
      (let [x (gen n #(- (* 3 %) 5)) y (gen n #(- 7 (* 2 %)))]
        (is (= (ref-dot x y n) (i8dot x y n)) (str "n=" n))))))

(deftest int8-widening-dot-compiled
  (testing "the int8 dot compiles (AOT bytecode) and stays correct"
    (let [f (pl/compile-aot #'i8dot)]
      (doseq [n [8 64]]
        (let [x (gen n #(- (* 3 %) 5)) y (gen n #(- 7 (* 2 %)))]
          (is (= (ref-dot x y n) (f x y n)) (str "compiled n=" n)))))))

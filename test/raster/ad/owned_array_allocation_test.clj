(ns raster.ad.owned-array-allocation-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as p]
            [raster.ad.reverse :as reverse]
            [raster.compiler.pipeline :as pipeline]))

(deftm unused-double-storage [x :- Double size :- Long] :- Double
  (let [scratch (double-array size)] (n/* x x)))
(deftm unused-float-storage [x :- Double size :- Long] :- Double
  (let [scratch (float-array size)] (n/* x x)))
(deftm read-fresh-storage [x :- Double size :- Long] :- Double
  (let [scratch (double-array size) zero (aget scratch 0)] (n/+ (n/* x x) zero)))
(deftm active-count [x :- Double] :- Double
  (let [scratch (double-array (long x))] (n/* x x)))
(deftm effectful-count [x :- Double events :- (Array long) size :- Long] :- Double
  (let [scratch (double-array (do (aset events 0 (inc (aget events 0))) size))] (n/* x x)))
(deftm double-storage-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-double-storage :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm float-storage-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-float-storage :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))

(use-fixtures :once (fn [run] (n/* 1.0 1.0) (n/+ 1.0 1.0) (run)))

(defn- compile-owned [v]
  (#'reverse/compile-prepared-value+grad (reverse/prepare-owned-bytecode-value+grad v [0])))

(deftest unused-allocation-keeps-size-completion-and-failure-before-result
  (doseq [v [#'unused-double-storage #'unused-float-storage]]
    (let [gradient (compile-owned v)]
      (doseq [x [0.5 -1.2] size [0 1 3]]
        (let [[value derivative inactive] (gradient x size)]
          (is (= (* x x) value))
          (is (= (* 2.0 x) derivative))
          (is (nil? inactive))))
      (is (thrown? NegativeArraySizeException (gradient 2.0 -1)))
      (is (= [4.0 4.0 nil] (gradient 2.0 4294967296))
          "preserve original raw Int count completion instead of checked int"))))

(deftest fresh-array-reads-remain-visible-and-preserve-bounds-failures
  (let [gradient (compile-owned #'read-fresh-storage)]
    (is (= [9.0 6.0 nil] (gradient 3.0 1)))
    (is (thrown? ArrayIndexOutOfBoundsException (gradient 3.0 0)))
    (is (thrown? NegativeArraySizeException (gradient 3.0 -1)))))

(deftest invoked-jvm-aot-preserves-unused-allocation-failures
  (doseq [v [#'double-storage-gradient #'float-storage-gradient]]
    (let [compiled (pipeline/compile-aot v)]
      (is (= [9.0 6.0] (vec (compiled 3.0 0))))
      (is (thrown? NegativeArraySizeException (compiled 3.0 -1)))
      (is (= [9.0 6.0] (vec (compiled 3.0 4294967296)))))))

(deftest allocation-freshness-does-not-authorize-active-or-effectful-counts
  (doseq [v [#'active-count #'effectful-count]]
    (let [failure (try (reverse/prepare-owned-bytecode-value+grad v [0]) nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :ad-unclosed-bytecode-body-region (:reason failure)))
      (is (= :array-allocation (:phase failure))))))

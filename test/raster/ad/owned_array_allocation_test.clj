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
(deftm unused-storage-read [x :- Double size :- Long] :- Double
  (let [scratch (double-array size) ignored (aget scratch 0)] (n/* x x)))
(deftm unused-read-helper [x :- Double size :- Long] :- Double
  (unused-storage-read x size))
(deftm read-before-second-allocation [x :- Double size :- Long later-size :- Long] :- Double
  (let [scratch (double-array size)
        ignored (aget scratch 0)
        later (float-array later-size)]
    (n/* x x)))
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
(deftm unused-read-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-storage-read :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm helper-read-gradient [x :- Double size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'unused-read-helper :wrt [0]) x size)]
    (p/map [i 2] (double (nth result i)))))
(deftm read-before-allocation-gradient [x :- Double size :- Long later-size :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'read-before-second-allocation :wrt [0]) x size later-size)]
    (p/map [i 2] (double (nth result i)))))

(use-fixtures :once (fn [run] (n/* 1.0 1.0) (n/+ 1.0 1.0) (run)))

(defn- compile-owned [v]
  (#'reverse/compile-prepared-value+grad (reverse/prepare-owned-bytecode-value+grad v [0])))

(deftest source-obligations-survive-executable-traversal
  (let [read (with-meta '(clojure.core/aget storage 0)
               {:raster.effect/preserve-evaluation true :line 17})
        form (with-meta (list 'let* ['unused read] 1.0) {:line 16})
        traversed (#'reverse/postwalk-code identity form)
        quoted (list 'quote read)]
    (is (= (meta form) (meta traversed)))
    (is (= (meta read) (meta (second (second traversed)))))
    (is (identical? quoted (#'reverse/postwalk-code
                            (fn [node] (throw (ex-info "quoted data visited" {}))) quoted)))
    (is (= (meta read)
           (meta (second (second (#'reverse/binaryize-core-arithmetic form))))))
    (let [lexical (with-meta '(double-array size) {:raster.op/lexical-callee true})
          retained (#'reverse/retain-source-memory-evaluations lexical)]
      (is (= (meta lexical) (meta retained)))
      (is (not (:raster.effect/preserve-evaluation (meta retained)))))))

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

(deftest unused-source-reads-retain-bounds-and-evaluation-order
  (let [read-runtime (compile-owned #'unused-storage-read)
        ordered-runtime (compile-owned #'read-before-second-allocation)
        read-aot (pipeline/compile-aot #'unused-read-gradient)
        helper-aot (pipeline/compile-aot #'helper-read-gradient)
        ordered-aot (pipeline/compile-aot #'read-before-allocation-gradient)]
    (doseq [[read expected] [[unused-storage-read 9.0]
                            [read-runtime [9.0 6.0 nil]]
                            [read-aot [9.0 6.0]]
                            [helper-aot [9.0 6.0]]]]
      (is (= expected (let [value (read 3.0 1)]
                        (if (number? value) value (vec value)))))
      (is (thrown? ArrayIndexOutOfBoundsException (read 3.0 0)))
      (is (thrown? NegativeArraySizeException (read 3.0 -1)))
      (is (thrown? ArrayIndexOutOfBoundsException (read 3.0 4294967296))
          "raw JVM Int count completion still wraps to zero"))
    (doseq [[ordered expected] [[read-before-second-allocation 9.0]
                               [ordered-runtime [9.0 6.0 nil nil]]
                               [ordered-aot [9.0 6.0]]]]
      (is (= expected (let [value (ordered 3.0 1 0)]
                        (if (number? value) value (vec value)))))
      (is (thrown? ArrayIndexOutOfBoundsException (ordered 3.0 0 -1))
          "the original read must fail before the later negative allocation")
      (is (thrown? NegativeArraySizeException (ordered 3.0 1 -1)))
      (is (thrown? NegativeArraySizeException (ordered 3.0 -1 0))))))

(deftest allocation-freshness-does-not-authorize-active-or-effectful-counts
  (doseq [v [#'active-count #'effectful-count]]
    (let [failure (try (reverse/prepare-owned-bytecode-value+grad v [0]) nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :ad-unclosed-bytecode-body-region (:reason failure)))
      (is (= :array-allocation (:phase failure))))))

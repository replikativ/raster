(ns raster.ad.loop-terminal-tape-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]))

(def terminal-draws (java.util.concurrent.atomic.AtomicLong. 0))
(defn next-terminal! []
  (double (.incrementAndGet ^java.util.concurrent.atomic.AtomicLong terminal-draws)))

(deftm terminal-draw [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt) (recur (inc i) (n/* acc x))
        (n/* acc (double (next-terminal!))))))

(deftm terminal-observation [x :- Double, ys :- (Array double), cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt) (recur (inc i) (n/* acc x)) (n/* acc (aget ys 0)))))

(deftm terminal-parameter [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt) (recur (inc i) (n/* acc x)) x)))

(deftm terminal-lazy-parameter [x :- Double, y :- Double, flag :- Long, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (recur (inc i) (n/* acc x))
      (if (= flag 1) (n/* acc y) (n/* acc (double (next-terminal!)))))))

(deftm terminal-shadowed-loop [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (recur (inc i) (n/* acc x))
      (loop [i 1 acc acc]
        (if (< i 3) (recur (inc i) (n/* acc x)) (n/* acc x))))))

(deftm terminal-statements [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (recur (inc i) (n/* acc x))
      (let [v (let [w (n/* acc x)] w)]
        (next-terminal!)
        (next-terminal!)
        (n/* v x)))))

(deftm retained-free-terminal [x :- Double, counts :- (Array long), cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (let [tick (aset counts 0 (inc (aget counts 0)))]
        (recur (inc i) (n/* acc x)))
      x)))

(deftm terminal-tail-let [x :- Double, cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (recur (inc i) (n/* acc x))
      (let [v (n/* acc x)]
        (next-terminal!)
        (let [w (n/* v x)] (next-terminal!) w)))))

(deftm nested-effectful-carry [x :- Double, counts :- (Array long), cnt :- Long] :- Double
  (loop [i 1 acc x]
    (if (< i cnt)
      (recur (inc i)
             (loop [j 1 carry acc]
               (if (< j 3)
                 (let [tick (aset counts 0 (inc (aget counts 0)))]
                   (recur (inc j) (n/* carry x)))
                 carry)))
      acc)))

(deftest terminal-primal-and-pullback-share-one-draw
  (let [gradient (rev/value+grad #'terminal-draw :wrt [0])]
    (doseq [[cnt value derivative] [[1 2.0 1.0] [3 8.0 12.0]]]
      (.set ^java.util.concurrent.atomic.AtomicLong terminal-draws 0)
      (is (= [value derivative nil] (gradient 2.0 cnt)))
      (is (= 1 (.get ^java.util.concurrent.atomic.AtomicLong terminal-draws))))))

(deftest terminal-observation-read-is-taped-not-repeated
  (doseq [[cnt value dx dys] [[1 6.0 3.0 2.0] [3 24.0 36.0 8.0]]]
    (let [ys (double-array [3.0])
          [actual pullback] (rev/vjp #'terminal-observation 2.0 ys cnt)]
      (is (= value actual))
      (aset ys 0 7.0)
      (doseq [seed [1.0 2.0]]
        (let [[actual-dx actual-dys dcnt] (pullback seed)]
          (is (= (* seed dx) actual-dx))
          (is (= [(* seed dys)] (vec actual-dys)))
          (is (nil? dcnt))))
      (is (= [7.0] (vec ys))))))

(deftest a-free-terminal-parameter-is-not-a-carried-result
  (let [gradient (rev/value+grad #'terminal-parameter :wrt [0])]
    (doseq [cnt [1 3]]
      (is (= 2.0 (terminal-parameter 2.0 cnt)))
      (is (= [2.0 1.0 nil] (gradient 2.0 cnt))))))

(deftest terminal-pullback-preserves-lazy-exits-and-free-parameters
  (let [gradient (rev/value+grad #'terminal-lazy-parameter :wrt [0 1])]
    (doseq [[cnt value dx dy] [[1 6.0 3.0 2.0] [3 24.0 36.0 8.0]]]
      (.set ^java.util.concurrent.atomic.AtomicLong terminal-draws 0)
      (is (= [value dx dy nil nil] (gradient 2.0 3.0 1 cnt)))
      (is (zero? (.get ^java.util.concurrent.atomic.AtomicLong terminal-draws))))
    (.set ^java.util.concurrent.atomic.AtomicLong terminal-draws 0)
    (is (= [8.0 12.0 0.0 nil nil] (gradient 2.0 3.0 0 3)))
    (is (= 1 (.get ^java.util.concurrent.atomic.AtomicLong terminal-draws)))))

(deftest terminal-pullback-captures-final-carries-without-rewriting-inner-binders
  (let [gradient (rev/value+grad #'terminal-shadowed-loop :wrt [0])]
    (doseq [[cnt value derivative] [[1 16.0 32.0] [3 64.0 192.0]]]
      (is (= value (terminal-shadowed-loop 2.0 cnt)))
      (is (= [value derivative nil] (gradient 2.0 cnt))))))

(deftest terminal-statements-execute-once-in-source-order
  (let [gradient (rev/value+grad #'terminal-statements :wrt [0])]
    (doseq [[cnt value derivative] [[1 8.0 12.0] [3 32.0 80.0]]]
      (.set ^java.util.concurrent.atomic.AtomicLong terminal-draws 0)
      (is (= value (terminal-statements 2.0 cnt)))
      (is (= 2 (.get ^java.util.concurrent.atomic.AtomicLong terminal-draws)))
      (.set ^java.util.concurrent.atomic.AtomicLong terminal-draws 0)
      (is (= [value derivative nil] (gradient 2.0 cnt)))
      (is (= 2 (.get ^java.util.concurrent.atomic.AtomicLong terminal-draws))))))

(deftest a-retained-effectful-loop-may-return-a-free-parameter
  (let [gradient (rev/value+grad #'retained-free-terminal :wrt [0])]
    (doseq [cnt [1 3]]
      (let [counts (long-array 1)]
        (is (= [2.0 1.0 nil nil] (gradient 2.0 counts cnt)))
        (is (= [(dec cnt)] (vec counts)))))))

(deftest existing-nested-effectful-carries-remain-supported
  (let [counts (long-array 1)
        gradient (rev/value+grad #'nested-effectful-carry :wrt [0])]
    (is (= 32.0 (nested-effectful-carry 2.0 counts 3)))
    (is (= [4] (vec counts)))
    (aset counts 0 0)
    (is (= [32.0 80.0 nil nil] (gradient 2.0 counts 3)))
    (is (= [4] (vec counts)))))

(deftest unsupported-terminal-lexical-region-remains-explicit
  ;; This exact shape also refuses before the terminal-tape repair. Do not
  ;; silently drop its prefix statements to make a nested tail look supported.
  (is (= :ad-active-unlinearized-region
         (try (rev/value+grad #'terminal-tail-let :wrt [0]) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

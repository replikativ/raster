(ns raster.compiler.passes.parallel.dot-reduce-test
  "Unit tests for the backend-neutral two-array dot-product reduction matcher.
   Covers bare clojure.core arithmetic, devirtualized (.invk …) raster.numeric
   arithmetic, widening-cast multiplicands (int8 MAC shape), and rejections."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.patterns :as patterns]))

(defn- invk
  "Build a devirtualized (.invk impl args…) form carrying the :raster.op/original
   metadata the walker attaches, exactly as the matcher sees it post-walk."
  [orig & args]
  (with-meta (apply list '.invk 'impl args)
    {:raster.op/original orig}))

(deftest bare-dot-test
  (testing "plain clojure.core dot product matches"
    (let [form '(loop [acc 0.0 i 0]
                  (if (< i n)
                    (recur (+ acc (* (aget a i) (aget b i))) (inc i))
                    acc))
          r (patterns/match-dot-reduce-loop form)]
      (is (some? r))
      (is (= 'acc (:acc-sym r)))
      (is (= 0.0 (:acc-init r)))
      (is (= 'a (:array-a r)))
      (is (= 'b (:array-b r)))
      (is (= 'n (:bound-expr r)))
      (is (nil? (:cast-a r)))
      (is (nil? (:cast-b r))))))

(deftest devirtualized-dot-test
  (testing "raster.numeric +/* arriving as (.invk …) matches via semantic-op"
    (let [prod (invk 'raster.numeric/* '(aget a i) '(aget b i))
          upd  (invk 'raster.numeric/+ 'acc prod)
          form (list 'loop ['acc 0.0 'i 0]
                     (list 'if '(< i n)
                           (list 'recur upd '(inc i))
                           'acc))
          r (patterns/match-dot-reduce-loop form)]
      (is (some? r))
      (is (= 'raster.numeric/+ (:op r)))
      (is (= 'raster.numeric/* (:mul-op r)))
      (is (= 'a (:array-a r)))
      (is (= 'b (:array-b r))))))

(deftest widening-int8-dot-test
  (testing "widening-cast multiplicands (int8 MAC shape) carry cast info"
    (let [form '(loop [acc 0 i 0]
                  (if (< i n)
                    (recur (+ acc (* (long (aget w i)) (long (aget x i)))) (inc i))
                    acc))
          r (patterns/match-dot-reduce-loop form)]
      (is (some? r))
      (is (= 'w (:array-a r)))
      (is (= 'x (:array-b r)))
      (is (= 'long (:cast-a r)))
      (is (= 'long (:cast-b r))))))

(deftest acc-arg-order-test
  (testing "matches regardless of which recur arg is the index step"
    (let [form '(loop [acc 0.0 i 0]
                  (if (< i n)
                    (recur (inc i) (+ acc (* (aget a i) (aget b i))))
                    acc))
          r (patterns/match-dot-reduce-loop form)]
      (is (some? r))
      (is (= 'a (:array-a r))))))

(deftest rejects-single-array-reduce-test
  (testing "a one-array fold (sum) is NOT a dot product"
    (let [form '(loop [acc 0.0 i 0]
                  (if (< i n)
                    (recur (+ acc (aget a i)) (inc i))
                    acc))]
      (is (nil? (patterns/match-dot-reduce-loop form)))))
  (testing "a non-aget multiplicand is rejected"
    (let [form '(loop [acc 0.0 i 0]
                  (if (< i n)
                    (recur (+ acc (* (aget a i) scalar)) (inc i))
                    acc))]
      (is (nil? (patterns/match-dot-reduce-loop form))))))

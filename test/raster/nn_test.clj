(ns raster.nn-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :as core]
            [raster.nn :as nn]))

;; ================================================================
;; Helpers
;; ================================================================

(defn- approx=
  "True if a and b are within eps."
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) eps)))

(defn- arr-approx=
  "True if all elements of double arrays a and b are within eps."
  ([a b] (arr-approx= a b 1e-6))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) eps))))))

;; ================================================================
;; double[] overload tests
;; ================================================================

(deftest dense-doubles-test
  (testing "Dense layer forward with double[]"
    ;; W is row-major: 2 rows, 3 cols
    ;; Row 0: [1 3 5], Row 1: [2 4 6]
    (let [W (double-array [1 3 5 2 4 6])
          x (double-array [1 2 3])
          b (double-array [0.1 0.2])
          y (nn/dense W x b)]
      (is (instance? (Class/forName "[D") y))
      (is (= 2 (alength ^doubles y)))
      ;; y[0] = 1*1 + 3*2 + 5*3 + 0.1 = 22.1
      ;; y[1] = 2*1 + 4*2 + 6*3 + 0.2 = 28.2
      (is (approx= 22.1 (aget ^doubles y 0)))
      (is (approx= 28.2 (aget ^doubles y 1))))))

(deftest relu-doubles-test
  (testing "ReLU forward with double[]"
    (let [x (double-array [-2 -1 0 1 2])
          y (nn/relu x)]
      (is (instance? (Class/forName "[D") y))
      (is (approx= 0.0 (aget ^doubles y 0)))
      (is (approx= 0.0 (aget ^doubles y 1)))
      (is (approx= 0.0 (aget ^doubles y 2)))
      (is (approx= 1.0 (aget ^doubles y 3)))
      (is (approx= 2.0 (aget ^doubles y 4))))))

(deftest softmax-doubles-test
  (testing "Softmax with double[] sums to 1"
    (let [x (double-array [1 2 3])
          s (nn/softmax x)
          sum (loop [i 0 acc 0.0]
                (if (< i 3)
                  (recur (inc i) (+ acc (aget ^doubles s i)))
                  acc))]
      (is (approx= 1.0 sum))))
  (testing "Softmax ordering preserved"
    (let [x (double-array [1 2 3])
          s (nn/softmax x)]
      (is (< (aget ^doubles s 0) (aget ^doubles s 1)))
      (is (< (aget ^doubles s 1) (aget ^doubles s 2))))))

(defn- reference-softmax [values]
  (if (empty? values)
    []
    (let [maximum (apply max values)
          weights (mapv #(Math/exp (- (double %) (double maximum))) values)
          total (reduce + weights)]
      (mapv #(/ % total) weights))))

(deftest softmax-retained-source-has-a-valid-typed-identity
  ;; Execute the retained expression with the ordinary Clojure compiler, too.
  ;; An unchecked backend d2f must not hide a checked `(float ##-Inf)` seed.
  (doseq [[dtype make-array] [[:float float-array] [:double double-array]]]
    (let [implementation (core/resolve-deftm-var #'nn/softmax {:dtype dtype})
          metadata (meta implementation)
          source (binding [*ns* (:ns metadata)]
                   (eval (list* 'fn (:raster.core/deftm-params metadata)
                                (:raster.core/deftm-walked-body metadata))))]
      (doseq [values [[] [0.0 1.0] [1000.0 1001.0]]]
        (let [input (make-array values)
              output (source input)]
          (is (= (class input) (class output)))
          (is (= (count values) (count output)))
          (is (every? true? (map #(approx= %1 %2 2e-6)
                                (reference-softmax (vec input)) output)))))))
  (testing "checked source narrowing remains checked"
    (let [narrow (fn [^double value] (float value))]
      (is (thrown? IllegalArgumentException (narrow Double/NEGATIVE_INFINITY))))))

(deftest softmax-typed-identities-and-changed-inputs
  (doseq [[dtype make-array tolerance]
          [[:float float-array 2e-6] [:double double-array 1e-12]]
          values [[] [0.0] [0.0 1.0 2.0] [1000.0 1001.0 1002.0]
                  [-1002.0 -1001.0 -1000.0] [10000.0 9999.0 -10000.0]]]
    (testing (str dtype " " values)
      (let [input (make-array values)
            before (vec input)
            output (nn/softmax input)
            expected (reference-softmax before)]
        (is (= (class input) (class output)))
        (is (= before (vec input)) "softmax must not mutate its input")
        (is (= (count expected) (count output)))
        (doseq [[wanted actual] (map vector expected output)]
          (is (Double/isFinite (double actual)))
          (is (approx= wanted actual tolerance)))
        (when (seq expected)
          (is (approx= 1.0 (reduce + output) tolerance))))))
  (doseq [make-array [float-array double-array]]
    (let [input (make-array [0.0 1.0 2.0])
          before (vec (nn/softmax input))]
      (if (instance? (class (float-array 0)) input)
        (aset ^floats input 0 (float 4.0))
        (aset ^doubles input 0 (double 4.0)))
      (let [after (vec (nn/softmax input))]
        (is (not= before after))
        (is (every? true? (map #(approx= %1 %2 2e-6)
                              (reference-softmax (vec input)) after)))))))

(deftest cross-entropy-doubles-test
  (testing "Cross-entropy with double[]"
    (let [p (double-array [0.1 0.7 0.2])
          t (double-array [0 1 0])]
      (is (approx= (- (Math/log 0.7)) (nn/cross-entropy p t))))))

(deftest softmax-cross-entropy-doubles-test
  (let [logits (double-array [1.0 2.0 3.0])
        target (double-array [0.0 1.0 0.0])
        [loss probabilities] (nn/softmax-cross-entropy logits target)]
    (is (approx= (- (Math/log (aget ^doubles probabilities 1))) loss))
    (is (approx= 1.0 (reduce + probabilities)))))

(deftest dense-backward-db-returns-an-independent-copy
  (let [gradient (double-array [1.0 2.0 3.0])
        copied (nn/dense-backward-db gradient)]
    (is (= (vec gradient) (vec copied)))
    (is (not (identical? gradient copied)))
    (aset ^doubles copied 0 9.0)
    (is (= 1.0 (aget ^doubles gradient 0)))))

(deftest allocating-dense-weight-gradient-is-row-major
  (is (= [3.0 4.0 5.0 6.0 8.0 10.0]
         (vec (nn/dense-backward-dW (double-array [1.0 2.0])
                                    (double-array [3.0 4.0 5.0]))))))

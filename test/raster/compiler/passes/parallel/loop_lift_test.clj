(ns raster.compiler.passes.parallel.loop-lift-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.par :as par]
            [raster.compiler.ir.par :as ir.par]
            [raster.compiler.passes.parallel.loop-lift :as loop-lift]))

;; ================================================================
;; Map pattern detection
;; ================================================================

(deftest detect-dotimes-map-simple
  (testing "Simple dotimes + aset is detected as par/map!"
    (let [form '(let* [x (dotimes [i n]
                           (clojure.core/aset out i (double (* (aget a i) 2.0))))]
                      x)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:maps-detected stats)))
      (is (= 0 (:reduces-detected stats)))
      ;; The dotimes should have been replaced with raster.par/map!
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (is (and (seq? expr) (= 'raster.par/map! (first expr))))))))

(deftest detect-dotimes-map-no-cast
  (testing "dotimes + aset without cast wrapper is detected"
    (let [form '(let* [x (dotimes [i n]
                           (clojure.core/aset out i (* (aget a i) (aget b i))))]
                      x)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:maps-detected stats))))))

(deftest detect-let-wrapped-dotimes
  (testing "let-wrapped dotimes (with int bound) is detected"
    (let [form '(let* [x (let [n_ (int bound)]
                           (dotimes [i n_]
                             (clojure.core/aset out i (double (+ (aget a i) 1.0)))))]
                      x)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:maps-detected stats)))
      ;; Bound should be the original 'bound', not 'n_'
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (when (and (seq? expr) (= 'raster.par/map! (first expr)))
          (let [[_ _out _idx bound-expr _cast _body] expr]
            (is (= 'bound bound-expr))))))))

(deftest allow-same-index-self-read-map
  (testing "dotimes reading output at same index (SGD pattern) IS detected as parallel"
    (let [form '(let* [x (dotimes [i n]
                           ;; Reads 'out[i]' and writes 'out[i]' — element-wise safe
                           (clojure.core/aset out i (double (+ (aget out i) 1.0))))]
                      x)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:maps-detected stats))))))

(deftest reject-cross-index-raw-hazard-map
  (testing "dotimes reading output at different index is NOT detected"
    (let [form '(let* [x (dotimes [i n]
                           ;; Reads 'out[0]' (not 'out[i]') — true RAW hazard
                           (clojure.core/aset out i (double (+ (aget out 0) 1.0))))]
                      x)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats))))))

(deftest reject-non-elementwise-map
  (testing "dotimes with non-element-wise body (aset inside body) is not detected"
    (let [form '(let* [x (dotimes [i n]
                           ;; Inner aset = side effect in body = not element-wise
                           (clojure.core/aset tmp i (double 0.0))
                           (clojure.core/aset out i (double (aget a i))))]
                      x)
          ;; multi-body dotimes won't match our single-aset pattern
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats))))))

;; ================================================================
;; Reduce pattern detection
;; ================================================================

(deftest detect-loop-reduce-simple
  (testing "Simple loop/recur accumulation is detected as par/reduce"
    (let [form '(let* [result (loop [i 0 acc 0.0]
                                (if (< i n)
                                  (recur (inc i) (+ acc (aget a i)))
                                  acc))]
                      result)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats)))
      (is (= 1 (:reduces-detected stats)))
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (is (and (seq? expr) (= 'raster.par/reduce (first expr))))))))

(deftest detect-let-wrapped-reduce
  (testing "let-wrapped loop reduce is detected"
    (let [form '(let* [result (let [n_ (int bound)]
                                (loop [i 0 acc 0.0]
                                  (if (< i n_)
                                    (recur (inc i) (+ acc (* (aget a i) (aget b i))))
                                    acc)))]
                      result)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:reduces-detected stats)))
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (when (and (seq? expr) (= 'raster.par/reduce (first expr)))
          (let [[_ _acc _init _idx bound-expr _body] expr]
            (is (= 'bound bound-expr))))))))

(deftest reject-loop-with-wrong-init
  (testing "Loop starting at non-zero is not detected"
    (let [form '(let* [result (loop [i 1 acc 0.0]
                                (if (< i n)
                                  (recur (inc i) (+ acc (aget a i)))
                                  acc))]
                      result)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:reduces-detected stats))))))

;; ================================================================
;; Scan pattern detection
;; ================================================================

(deftest detect-loop-scan-simple
  (testing "Simple loop/recur with aset is detected as par/scan"
    (let [form '(let* [result (loop [i 0 acc 0.0]
                                (if (< i n)
                                  (let [acc (+ acc (aget a i))]
                                    (clojure.core/aset out i acc)
                                    (recur (inc i) acc))
                                  out))]
                      result)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats)))
      (is (= 0 (:reduces-detected stats)))
      (is (= 1 (:scans-detected stats)))
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (is (and (seq? expr) (= 'raster.par/scan (first expr))))))))

(deftest detect-loop-scan-with-cast
  (testing "Scan with cast wrapper (double) is detected"
    (let [form '(let* [result (loop [i 0 acc 0.0]
                                (if (< i n)
                                  (let [acc (+ acc (aget a i))]
                                    (clojure.core/aset out i (double acc))
                                    (recur (inc i) acc))
                                  out))]
                      result)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:scans-detected stats))))))

(deftest detect-let-wrapped-scan
  (testing "let-wrapped scan loop is detected"
    (let [form '(let* [result (let [n_ (int bound)]
                                (loop [i 0 acc 0.0]
                                  (if (< i n_)
                                    (let [acc (+ acc (aget a i))]
                                      (clojure.core/aset out i acc)
                                      (recur (inc i) acc))
                                    out)))]
                      result)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:scans-detected stats)))
      ;; Bound should be unwrapped to 'bound'
      (let [bindings (second form)
            [_ expr] (take 2 bindings)]
        (when (and (seq? expr) (= 'raster.par/scan (first expr)))
          (let [[_ _out _acc _init _idx bound-expr _cast _body] expr]
            (is (= 'bound bound-expr))))))))

(deftest roundtrip-scan-expand-redetect
  (testing "expand-par-forms -> loop lift recovers scan pattern"
    (let [original '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i)))
          expanded (ir.par/expand-par-forms original)
          let-form (list 'let* ['result expanded] 'result)
          {:keys [stats]} (loop-lift/lift-parallel-forms let-form)]
      (is (= 1 (:scans-detected stats))))))

;; ================================================================
;; Round-trip: expand -> redetect
;; ================================================================

(deftest roundtrip-map-expand-redetect
  (testing "expand-par-forms produces int-counted loop* for C2 vectorization"
    ;; Expanded par forms now use loop* with int counters (not dotimes with long).
    ;; The loop lifter recognizes dotimes patterns from user code, not pipeline-
    ;; expanded forms, so roundtrip detection is not expected.
    (let [original '(raster.par/map! out i n double (* (aget a i) 2.0))
          expanded (ir.par/expand-par-forms original)]
      (is (some #(and (seq? %) (= 'loop* (first %)))
                (tree-seq seq? seq expanded))))))

(deftest roundtrip-reduce-expand-redetect
  (testing "expand-par-forms produces int-counted loop* for C2 vectorization"
    (let [original '(raster.par/reduce acc 0.0 i n (+ acc (aget a i)))
          expanded (ir.par/expand-par-forms original)]
      (is (some #(and (seq? %) (= 'loop* (first %)))
                (tree-seq seq? seq expanded))))))

;; ================================================================
;; No false positives
;; ================================================================

(deftest no-detection-on-plain-let
  (testing "Plain let* with no dotimes/loop is unchanged"
    (let [form '(let* [a 1 b (+ a 2)] b)
          {:keys [stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats)))
      (is (= 0 (:reduces-detected stats))))))

(deftest no-detection-on-non-let
  (testing "Non-let* forms pass through unchanged"
    (let [form '(+ 1 2)
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 0 (:maps-detected stats)))
      (is (= 0 (:reduces-detected stats)))
      (is (= '(+ 1 2) form)))))

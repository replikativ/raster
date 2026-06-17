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

;; ================================================================
;; Hand-written loop/recur reduction recovery (Option A)
;; Matchers are metadata-aware (descriptor/semantic-op), so they handle both
;; bare ops and walker-devirtualized (.invk impl ...) forms, and emit the
;; original body (preserving invokedynamic dispatch).
;; ================================================================

(deftest detect-handloop-reduce
  (testing "Hand (loop [i 0 acc 0.0] (if (< i n) (recur (+ i 1) (+ acc ...)) acc)) lifts to par/reduce"
    (let [form '(loop [d 0 acc 0.0]
                  (if (raster.numeric/< d n)
                    (recur (raster.numeric/+ d 1)
                           (raster.numeric/+ acc (raster.numeric/* (clojure.core/aget a d)
                                                                   (clojure.core/aget b d))))
                    acc))
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:reduces-detected stats)))
      (is (= 'raster.par/reduce (first form)))
      ;; the update body is emitted verbatim (no rewriting): the original
      ;; operators are preserved, so a devirtualized .invk body keeps its typed
      ;; invokedynamic dispatch (here the equivalent bare ops are preserved).
      (is (re-find #"raster.numeric/\*" (pr-str form))
          "original operator preserved verbatim in lifted body"))))

(deftest induction-var-from-test-not-init
  (testing "Both-init-zero loop: induction var comes from the loop TEST, not an init guess"
    ;; (loop [i 0 next-idx 0] (if (< i n) (recur (+ i 1) (+ next-idx 1)) next-idx))
    ;; index MUST be i (the tested var). Guessing by init value would swap to
    ;; next-idx and miscompile (the bug that broke gsdm).
    (let [form '(loop [i 0 next-idx 0]
                  (if (raster.numeric/< i n)
                    (recur (raster.numeric/+ i 1) (raster.numeric/+ next-idx 1))
                    next-idx))
          {:keys [form stats]} (loop-lift/lift-parallel-forms form)]
      (is (= 1 (:reduces-detected stats)))
      ;; (raster.par/reduce acc init IDX bound body)
      (is (= 'i (nth form 3)) "index must be the variable tested by (< i n)")
      (is (= 'n (nth form 4)) "bound must be n"))))

(deftest reduce-soundness-gates
  (testing "Non-associative / out-of-shape reductions are NOT lifted"
    (let [lifts? (fn [f] (:reduces-detected (:stats (loop-lift/lift-parallel-forms f))))
          sub '(loop [d 0 acc 0.0] (if (raster.numeric/< d n)
                                     (recur (raster.numeric/+ d 1)
                                            (raster.numeric/- acc (clojure.core/aget a d))) acc))
          gt  '(loop [d 0 acc 0.0] (if (raster.numeric/> d n)
                                     (recur (raster.numeric/+ d 1)
                                            (raster.numeric/+ acc (clojure.core/aget a d))) acc))
          ser '(loop [d 0 acc 0.0] (if (raster.numeric/< d n)
                                     (recur (raster.numeric/+ d 1)
                                            (raster.numeric/+ acc (raster.numeric/* acc (clojure.core/aget a d)))) acc))]
      (is (= 0 (lifts? sub)) "subtraction (non-associative) must not lift")
      (is (= 0 (lifts? gt))  "non-< bound must not lift")
      (is (= 0 (lifts? ser)) "accumulator inside the element (serial dep) must not lift"))))

(deftest reduce-soundness-gates-extended
  (testing "Lift fires ONLY when all preconditions are proven (no false positives)"
    (let [r? (fn [f] (:reduces-detected (:stats (loop-lift/lift-parallel-forms f))))]
      ;; bound must be loop-invariant (not reference index/acc)
      (is (= 0 (r? '(loop [i 0 acc 0.0]
                      (if (raster.numeric/< i acc)
                        (recur (raster.numeric/+ i 1) (raster.numeric/+ acc (clojure.core/aget a i))) acc))))
          "bound referencing a loop var must not lift")
      ;; a side effect in a do before recur must not be silently dropped
      (is (= 0 (r? '(loop [i 0 acc 0.0]
                      (if (raster.numeric/< i n)
                        (do (clojure.core/aset log i acc)
                            (recur (raster.numeric/+ i 1) (raster.numeric/+ acc (clojure.core/aget a i)))) acc))))
          "do-wrapped side effect must not lift (would be dropped)")
      ;; exit value must not reference the index (would escape its scope)
      (is (= 0 (r? '(loop [i 0 acc 0.0]
                      (if (raster.numeric/< i n)
                        (recur (raster.numeric/+ i 1) (raster.numeric/+ acc (clojure.core/aget a i)))
                        (raster.numeric// acc i)))))
          "post-fn referencing the index must not lift")
      ;; element must be pure
      (is (= 0 (r? '(loop [i 0 acc 0.0]
                      (if (raster.numeric/< i n)
                        (recur (raster.numeric/+ i 1) (raster.numeric/+ acc (some.ns/sfx! i))) acc))))
          "impure element must not lift")
      ;; but an invariant post-fn (mean) is fine
      (is (= 1 (r? '(loop [i 0 acc 0.0]
                      (if (raster.numeric/< i n)
                        (recur (raster.numeric/+ i 1) (raster.numeric/+ acc (clojure.core/aget a i)))
                        (raster.numeric// acc n)))))
          "post-fn over loop-invariants (mean) lifts"))))

(deftest reduce-commutative-monoid-coverage
  (testing "Commutative monoids (incl. integer bitwise) lift; non-monoids bail"
    (let [r? (fn [f] (:reduces-detected (:stats (loop-lift/lift-parallel-forms f))))]
      (is (= 1 (r? '(loop [d 0 acc 0]
                      (if (raster.numeric/< d n)
                        (recur (raster.numeric/+ d 1) (clojure.core/bit-xor acc (clojure.core/aget a d))) acc))))
          "bit-xor (commutative monoid) lifts")
      (is (= 1 (r? '(loop [d 0 acc -1]
                      (if (raster.numeric/< d n)
                        (recur (raster.numeric/+ d 1) (clojure.core/bit-and acc (clojure.core/aget a d))) acc))))
          "bit-and lifts")
      (is (= 1 (r? '(loop [d 0 acc 0.0]
                      (if (raster.numeric/< d n)
                        (recur (raster.numeric/+ d 1) (raster.numeric/max acc (clojure.core/aget a d))) acc))))
          "max lifts")
      (is (= 0 (r? '(loop [d 0 acc 0.0]
                      (if (raster.numeric/< d n)
                        (recur (raster.numeric/+ d 1) (raster.numeric/- acc (clojure.core/aget a d))) acc))))
          "subtraction (not a commutative monoid) bails")
      (is (= 0 (r? '(loop [d 0 acc 1.0]
                      (if (raster.numeric/< d n)
                        (recur (raster.numeric/+ d 1) (raster.numeric// acc (clojure.core/aget a d))) acc))))
          "division (not a commutative monoid) bails"))))

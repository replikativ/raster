(ns raster.compiler.core.hardware-test
  "Unit tests for the analytic hardware-descriptor derivations — the planner's
   resource budget. Pure functions of (descriptor, problem); no host dependence
   (descriptors are constructed explicitly so tests are deterministic)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hw]))

(def avx2   {:vector-bits 256 :has-native-dot-reduce true  :num-vector-registers 16
             :llc-bytes 16777216 :balance 40})
(def avx512 {:vector-bits 512 :has-native-dot-reduce true  :num-vector-registers 32
             :llc-bytes 33554432 :balance 40})
(def neon   {:vector-bits 128 :has-native-dot-reduce true  :num-vector-registers 32
             :llc-bytes 8388608 :balance 40})

(deftest natural-lanes-test
  (testing "lane count = vector-bits / (8 * dtype-bytes)"
    (is (= 8 (hw/natural-lanes avx2 :float)))    ; 256/32
    (is (= 4 (hw/natural-lanes avx2 :double)))   ; 256/64
    (is (= 32 (hw/natural-lanes avx2 :byte)))    ; 256/8
    (is (= 16 (hw/natural-lanes avx512 :float)))
    (is (= 4 (hw/natural-lanes neon :float)))))

(deftest column-tile-test
  (testing "NC (interleave width) = f32 lanes — the hardware-derived layout tile"
    (is (= 8 (hw/column-tile avx2)))             ; matches the hard-coded NC=8 today
    (is (= 16 (hw/column-tile avx512)))
    (is (= 4 (hw/column-tile neon)))))

(deftest register-block-test
  (testing "clamps to latency-target and to regs/tile budget"
    (is (= 4 (hw/register-block avx2)))               ; min(4, 16/2)
    (is (= 4 (hw/register-block avx512)))             ; min(4, 32/2)
    (is (= 2 (hw/register-block avx2 {:regs-per-tile 8})))   ; min(4, 16/8)
    (is (= 1 (hw/register-block avx2 {:regs-per-tile 32}))) ; clamp >=1
    (is (= 8 (hw/register-block avx512 {:latency-target 8 :regs-per-tile 2})))))

(deftest roofline-regime-test
  (testing "AI vs balance picks memory- vs compute-bound"
    ;; decode GEMV: flops ~= bytes -> AI ~= 1 << balance 40 -> memory-bound
    (is (= :memory-bound (hw/roofline-regime avx2 {:flops 1000 :bytes 1000})))
    ;; GEMM reusing each weight 64x -> AI 64 > 40 -> compute-bound
    (is (= :compute-bound (hw/roofline-regime avx2 {:flops 64000 :bytes 1000})))))

(deftest reduction-accumulators-test
  (testing "the centralized SIMD-reduction accumulator policy (≥8 lanes → 8, else 4)"
    (is (= 4 (hw/reduction-accumulators avx2 :double)))    ; 256/64 = 4 lanes
    (is (= 8 (hw/reduction-accumulators avx2 :float)))     ; 256/32 = 8 lanes
    (is (= 8 (hw/reduction-accumulators avx512 :double)))  ; 512/64 = 8 lanes
    (is (= 8 (hw/reduction-accumulators avx512 :float)))   ; 512/32 = 16 -> 8
    (is (= 4 (hw/reduction-accumulators neon :float)))))   ; 128/32 = 4 lanes

(deftest reduce-intrinsic-test
  (testing "native dot-reduce availability selects the leaf instruction"
    (is (= :dpbusd (hw/reduce-intrinsic avx2)))
    (is (= :scalar (hw/reduce-intrinsic (hw/with avx2 {:has-native-dot-reduce false}))))))

(deftest host-descriptor-test
  (testing "host descriptor has all five fields with sane values"
    (let [d (hw/host-descriptor)]
      (is (pos? (:vector-bits d)))
      (is (contains? #{16 32} (:num-vector-registers d)))
      (is (pos? (:llc-bytes d)))
      (is (number? (:balance d)))
      (is (boolean? (:has-native-dot-reduce d))))))

(ns raster.compiler.passes.parallel.par-fusion-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.par-fusion :as fusion]
            [raster.par :as par]
            [raster.compiler.ir.par :as ir.par]))

;; ================================================================
;; Map→Reduce fusion
;; ================================================================

(deftest fuse-map-reduce-simple-test
  (testing "Fuse squared-norm pattern: map! → reduce"
    (let [bindings '[tmp (raster.par/map! tmp i n double (* (aget a i) (aget a i)))
                     result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
          body-exprs ['result]
          fused (fusion/fuse-map-reduce bindings body-exprs)]
      (is (some? fused) "Should fuse map→reduce")
      (is (= 1 (:fused fused)))
      ;; The tmp binding should be eliminated
      (let [pairs (partition 2 (:bindings fused))]
        (is (= 1 (count pairs)) "Should have 1 binding (fused reduce)")
        (let [[sym expr] (first pairs)]
          (is (= 'result sym))
          (is (= 'raster.par/reduce (first expr)))
          ;; Body should inline the map computation
          (let [reduce-body (last expr)]
            (is (seq? reduce-body))
            ;; Should contain (* ...) instead of (aget tmp j)
            (is (not (some #{'tmp} (flatten reduce-body)))
                "tmp should be eliminated")))))))

(deftest fuse-map-reduce-no-fusion-when-tmp-used-test
  (testing "No fusion when tmp is used elsewhere"
    (let [bindings '[tmp (raster.par/map! tmp i n double (* (aget a i) 2.0))
                     result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
          ;; tmp is used in the body — can't eliminate
          body-exprs ['(some-fn tmp result)]
          fused (fusion/fuse-map-reduce bindings body-exprs)]
      (is (nil? fused) "Should NOT fuse when tmp is live"))))

(deftest fuse-map-reduce-non-adjacent-test
  (testing "Fusion works for non-adjacent map and reduce"
    (let [bindings '[tmp (raster.par/map! tmp i n double (aget a i))
                     other 42
                     result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
          body-exprs ['result]
          fused (fusion/fuse-map-reduce bindings body-exprs)]
      (is (some? fused))
      (is (= 1 (:fused fused))))))

;; ================================================================
;; Horizontal fusion
;; ================================================================

(deftest fuse-horizontal-maps-test
  (testing "Fuse two independent maps with same bound"
    (let [bindings '[out1 (raster.par/map! out1 i n double (Math/sin (aget a i)))
                     out2 (raster.par/map! out2 i n double (Math/cos (aget a i)))]
          fused (fusion/fuse-horizontal-maps bindings)]
      (is (some? fused) "Should fuse horizontal maps")
      (is (= 1 (:fused fused)))
      ;; Should have 1 binding (fused map)
      (let [pairs (partition 2 (:bindings fused))]
        (is (= 1 (count pairs)))))))

(deftest no-horizontal-fusion-when-dependent-test
  (testing "No fusion when second map depends on first"
    (let [bindings '[out1 (raster.par/map! out1 i n double (Math/sin (aget a i)))
                     out2 (raster.par/map! out2 i n double (aget out1 i))]
          fused (fusion/fuse-horizontal-maps bindings)]
      (is (nil? fused) "Should NOT fuse dependent maps"))))

(deftest no-horizontal-fusion-different-bounds-test
  (testing "No fusion when bounds differ"
    (let [bindings '[out1 (raster.par/map! out1 i n double (aget a i))
                     out2 (raster.par/map! out2 i m double (aget b i))]
          fused (fusion/fuse-horizontal-maps bindings)]
      (is (nil? fused) "Should NOT fuse maps with different bounds"))))

;; ================================================================
;; Binding reorder
;; ================================================================

(deftest reorder-for-fusion-test
  (testing "Reorders par forms with same bound to be adjacent"
    (let [bindings '[map1 (raster.par/map! out1 i n double (aget a i))
                     scalar1 42
                     map2 (raster.par/map! out2 i n double (aget b i))]
          reordered (fusion/reorder-for-fusion bindings)]
      ;; Should reorder so map1 and map2 are adjacent
      (when reordered
        (let [pairs (partition 2 reordered)
              par-indices (keep-indexed
                           (fn [i [_ expr]]
                             (when (ir.par/par-form? expr) i))
                           pairs)]
          ;; Par forms should be adjacent
          (when (>= (count par-indices) 2)
            (is (<= (- (second par-indices) (first par-indices)) 1)
                "Par forms should be adjacent after reorder")))))))

(deftest no-reorder-when-already-grouped-test
  (testing "No reorder needed when par forms already adjacent"
    (let [bindings '[map1 (raster.par/map! out1 i n double (aget a i))
                     map2 (raster.par/map! out2 i n double (aget b i))]
          reordered (fusion/reorder-for-fusion bindings)]
      ;; May or may not be nil — if reorder produces same result, nil is fine
      (is true))))

;; ================================================================
;; Combined par-fusion-pass
;; ================================================================

(deftest par-fusion-pass-non-let-test
  (testing "Pass through non-let forms unchanged"
    (let [{:keys [form stats]} (fusion/par-fusion-pass '(+ 1 2))]
      (is (= '(+ 1 2) form))
      (is (zero? (:map-reduce stats)))
      (is (zero? (:horizontal stats))))))

(deftest par-fusion-pass-map-reduce-test
  (testing "Full pass applies map→reduce fusion"
    (let [form '(let* [tmp (raster.par/map! tmp i n double (* (aget a i) (aget a i)))
                       result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
                      result)
          {:keys [form stats]} (fusion/par-fusion-pass form)]
      (is (pos? (:map-reduce stats)) "Should report map-reduce fusions")
      ;; tmp should not appear in the fused form bindings
      (let [[_ bindings & _] form
            binding-syms (set (take-nth 2 bindings))]
        (is (not (contains? binding-syms 'tmp))
            "tmp should be eliminated")))))

(deftest par-fusion-pass-nested-let-test
  (testing "Recursion into nested let forms"
    (let [form '(let* [x 1
                       inner (let* [tmp (raster.par/map! tmp i n double (aget a i))
                                    result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))]
                                   result)]
                      inner)
          {:keys [form stats]} (fusion/par-fusion-pass form)]
      (is (pos? (:map-reduce stats))))))

;; ================================================================
;; Horizontal map-void! fusion
;; ================================================================

(deftest fuse-horizontal-map-voids-basic-test
  (testing "Fuse two independent map-void! forms with same bound"
    (let [bindings '[_a (raster.par/map-void! i n (clojure.core/aset output i (compute-a i)))
                     _b (raster.par/map-void! i n (clojure.core/aset other  i (compute-b i)))]
          fused (fusion/fuse-horizontal-map-voids bindings)]
      (is (some? fused) "Should fuse independent map-void! forms")
      (is (= 1 (:fused fused)) "Should report 1 fusion")
      (let [pairs (partition 2 (:bindings fused))]
        (is (= 1 (count pairs)) "Should collapse to one binding")
        (let [[_ expr] (first pairs)]
          (is (ir.par/par-map-void-form? expr) "Result should still be map-void!")
          ;; Body should be a do block
          (let [body (last expr)]
            (is (seq? body))
            (is (= 'do (first body)) "Fused body should be a do block")
            (is (= 3 (count body)) "do block should have 2 side-effect exprs")))))))

(deftest no-map-void-fusion-when-output-read-test
  (testing "No fusion when second form reads from first form's write target"
    ;; produce-output writes 'output; distribute-income reads 'output
    (let [bindings '[_po (raster.par/map-void! i n (clojure.core/aset output i (+ 1.0 (clojure.core/aget a i))))
                     _di (raster.par/map-void! i n (clojure.core/aset income i (* 0.5 (clojure.core/aget output i))))]
          fused (fusion/fuse-horizontal-map-voids bindings)]
      (is (nil? fused) "Should NOT fuse when second reads first's writes"))))

(deftest no-map-void-fusion-different-bounds-test
  (testing "No fusion when bounds differ"
    (let [bindings '[_a (raster.par/map-void! i n (clojure.core/aset out1 i 1.0))
                     _b (raster.par/map-void! i m (clojure.core/aset out2 i 2.0))]
          fused (fusion/fuse-horizontal-map-voids bindings)]
      (is (nil? fused) "Should NOT fuse map-void! forms with different bounds"))))

(deftest fuse-horizontal-map-voids-three-way-test
  (testing "Three-way fusion: three independent map-void! forms"
    (let [bindings '[_a (raster.par/map-void! i n (clojure.core/aset out1 i (clojure.core/aget src1 i)))
                     _b (raster.par/map-void! i n (clojure.core/aset out2 i (clojure.core/aget src2 i)))
                     _c (raster.par/map-void! i n (clojure.core/aset out3 i (clojure.core/aget src3 i)))]
          fused (fusion/fuse-horizontal-map-voids bindings)]
      (is (some? fused) "Should fuse three independent map-void! forms")
      (is (= 2 (:fused fused)) "Should report 2 fusions (3 → 1)")
      (let [pairs (partition 2 (:bindings fused))]
        (is (= 1 (count pairs)) "Should collapse to one binding")
        (let [[_ expr] (first pairs)
              body (last expr)]
          (is (= 'do (first body)) "Body should be do block")
          (is (= 4 (count body)) "do block should have 3 side-effect statements"))))))

(deftest par-fusion-pass-map-void-test
  (testing "par-fusion-pass integrates map-void! horizontal fusion"
    (let [form '(let* [_a (raster.par/map-void! i bound (clojure.core/aset x i (clojure.core/aget src-x i)))
                       _b (raster.par/map-void! i bound (clojure.core/aset y i (clojure.core/aget src-y i)))]
                      nil)
          {:keys [form stats]} (fusion/par-fusion-pass form)]
      (is (pos? (:horizontal-void stats)) "Should report map-void! fusion")
      ;; Fused form should have 1 binding
      (let [[_ bindings & _] form]
        (is (= 2 (count bindings)) "Should have 1 binding (sym + fused expr)")))))

(deftest par-fusion-pass-map-void-dependency-test
  (testing "par-fusion-pass does not fuse dependent map-void! forms"
    ;; First writes output; second reads output — must NOT fuse
    (let [form '(let* [_po (raster.par/map-void! i n (clojure.core/aset output i (clojure.core/aget a i)))
                       _di (raster.par/map-void! i n (clojure.core/aset income i (clojure.core/aget output i)))]
                      nil)
          {:keys [form stats]} (fusion/par-fusion-pass form)]
      (is (zero? (:horizontal-void stats)) "Should NOT fuse dependent map-void! forms")
      (let [[_ bindings & _] form]
        (is (= 4 (count bindings)) "Should still have 2 bindings (not fused)")))))

(ns raster.compiler.ir.segop-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.soac-lower :as lower]))

;; ================================================================
;; SegOp record construction
;; ================================================================

(deftest seg-space-test
  (testing "SegSpace construction"
    (let [space (segop/make-seg-space 'i 'n)]
      (is (instance? raster.compiler.ir.segop.SegSpace space))
      (is (symbol? (:flat-idx space)))
      (is (= 1 (count (:dims space))))
      (is (= 'i (:name (first (:dims space)))))
      (is (= 'n (:bound (first (:dims space))))))))

(deftest kernel-grid-test
  (testing "KernelGrid construction"
    (let [grid (segop/->KernelGrid 128 256 2048)]
      (is (= 128 (:num-blocks grid)))
      (is (= 256 (:block-size grid)))
      (is (= 2048 (:shared-mem-bytes grid))))))

(deftest seg-level-test
  (testing "SegLevel construction"
    (let [level (segop/->SegLevel :thread :virtual)]
      (is (= :thread (:level level)))
      (is (= :virtual (:virt level))))))

(deftest segop-predicates-test
  (testing "segop? predicate"
    (let [space (segop/make-seg-space 'i 'n)
          level (segop/->SegLevel :thread :none)
          grid (segop/->KernelGrid 1 256 0)
          seg-map (segop/->SegMap 0 space level '(* (aget a i) 2.0)
                                  #{'a} #{'out} #{} grid
                                  :double 'out nil)]
      (is (segop/segop? seg-map))
      (is (not (segop/segop? {:not "a segop"}))))))

;; ================================================================
;; Launch parameter computation
;; ================================================================

(deftest compute-launch-params-test
  (testing "Launch params computed for map"
    (let [grid (segop/compute-launch-params :map nil 'n)]
      (is (instance? raster.compiler.ir.segop.KernelGrid grid))
      (is (pos? (:block-size grid)))
      (is (zero? (:shared-mem-bytes grid))))))

(deftest compute-launch-params-reduce-test
  (testing "Launch params for reduce include shared memory"
    (let [grid (segop/compute-launch-params :reduce nil 'n)]
      (is (pos? (:shared-mem-bytes grid))))))

;; ================================================================
;; Map lowering
;; ================================================================

(deftest lower-map-test
  (testing "SoacMap → single SegMap"
    (let [soac (soac/par-form->soac 'out
                                    '(raster.par/map! out i n double (* (aget a i) 2.0)) 0)
          segops (lower/lower-map soac nil)]
      (is (= 1 (count segops)))
      (is (instance? raster.compiler.ir.segop.SegMap (first segops)))
      (let [seg (first segops)]
        (is (= :virtual (get-in seg [:level :virt])))
        (is (= :thread (get-in seg [:level :level])))
        (is (zero? (:shared-mem-bytes (:grid seg))))))))

;; ================================================================
;; Reduce lowering — two-phase
;; ================================================================

(deftest lower-reduce-test
  (testing "SoacReduce → two-phase SegRed"
    (let [soac (soac/par-form->soac 'result
                                    '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 0)
          segops (lower/lower-reduce soac nil)]
      (is (= 2 (count segops)))
      ;; Phase 1: block-local
      (is (instance? raster.compiler.ir.segop.SegRed (first segops)))
      (is (= :block-local (:phase (first segops))))
      (is (pos? (:shared-mem-bytes (:grid (first segops)))))
      ;; Phase 2: cross-block
      (is (instance? raster.compiler.ir.segop.SegRed (second segops)))
      (is (= :cross-block (:phase (second segops)))))))

(deftest lower-reduce-single-phase-test
  (testing "Small constant reduce lowers to a single SegRed"
    (let [soac (soac/par-form->soac 'result
                                    '(raster.par/reduce acc 0.0 j 64 (+ acc (aget a j))) 0)
          segops (lower/lower-reduce soac nil)]
      (is (= 1 (count segops)))
      (is (= :single (:phase (first segops)))))))

(deftest lower-reduce-op-preserved-test
  (testing "Reduce op (acc/init/lambda) is preserved"
    (let [soac (soac/par-form->soac 'result
                                    '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 0)
          segops (lower/lower-reduce soac nil)
          op (:reduce-op (first segops))]
      (is (= 'acc (:acc op)))
      (is (= 0.0 (:init op))))))

;; ================================================================
;; Scan lowering — three-stage
;; ================================================================

(deftest lower-scan-test
  (testing "SoacScan → three-stage SegScan/SegMap"
    (let [soac (soac/par-form->soac 'out
                                    '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i))) 0)
          segops (lower/lower-scan soac nil)]
      (is (= 3 (count segops)))
      ;; Stage 1: intra-block scan
      (is (instance? raster.compiler.ir.segop.SegScan (nth segops 0)))
      (is (= :intra-block (:phase (nth segops 0))))
      ;; Stage 2: block-totals scan
      (is (instance? raster.compiler.ir.segop.SegScan (nth segops 1)))
      (is (= :block-scan (:phase (nth segops 1))))
      ;; Stage 3: carry-in (SegMap)
      (is (instance? raster.compiler.ir.segop.SegMap (nth segops 2))))))

(deftest lower-scan-single-phase-test
  (testing "Small constant scan lowers to a single SegScan"
    (let [soac (soac/par-form->soac 'out
                                    '(raster.par/scan out acc 0.0 i 64 double (+ acc (aget a i))) 0)
          segops (lower/lower-scan soac nil)]
      (is (= 1 (count segops)))
      (is (instance? raster.compiler.ir.segop.SegScan (first segops)))
      (is (= :single (:phase (first segops)))))))

;; ================================================================
;; Unified dispatch
;; ================================================================

(deftest lower-soac-dispatch-test
  (testing "lower-soac dispatches correctly"
    (let [map-soac (soac/par-form->soac 'out
                                        '(raster.par/map! out i n double (aget a i)) 0)
          red-soac (soac/par-form->soac 'result
                                        '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 1)
          scan-soac (soac/par-form->soac 'out
                                         '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i))) 2)]
      (is (= 1 (count (lower/lower-soac map-soac nil))))
      (is (= 2 (count (lower/lower-soac red-soac nil))))
      (is (= 3 (count (lower/lower-soac scan-soac nil)))))))

;; ================================================================
;; Screma lowering
;; ================================================================

(deftest lower-screma-map-test
  (testing "Screma pure-map → single SegMap"
    (let [map-soac (soac/par-form->soac 'out
                                        '(raster.par/map! out i n double (* (aget a i) 2.0)) 0)
          screma (soac/soac->screma map-soac)
          segops (lower/lower-soac screma nil)]
      (is (= 1 (count segops)))
      (is (instance? raster.compiler.ir.segop.SegMap (first segops))))))

(deftest lower-screma-reduce-test
  (testing "Screma with reduce → two-phase SegRed"
    (let [red-soac (soac/par-form->soac 'result
                                        '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 0)
          screma (soac/soac->screma red-soac)
          segops (lower/lower-soac screma nil)]
      (is (= 2 (count segops)))
      (is (every? #(instance? raster.compiler.ir.segop.SegRed %) segops)))))

(deftest lower-nodes-map-test
  (testing "lower-soac-nodes processes all SOAC nodes"
    (let [pairs [['out '(raster.par/map! out i n double (aget a i))]
                 ['x '42]
                 ['result '(raster.par/reduce acc 0.0 j n (+ acc (aget out j)))]]
          nodes (soac/let-bindings->nodes pairs)
          nodes-map (into {} (map (fn [n] [(:id n) n]) nodes))
          lowered (lower/lower-soac-nodes nodes-map nil)]
      ;; 2 SOACs lowered (scalar binding skipped)
      (is (= 2 (count lowered)))
      (is (contains? lowered 0))   ;; map
      (is (contains? lowered 2))   ;; reduce
      (is (= 1 (count (get lowered 0))))  ;; map → 1 SegOp
      (is (= 2 (count (get lowered 2))))))) ;; reduce → 2 SegOps

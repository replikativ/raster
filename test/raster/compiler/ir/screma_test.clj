(ns raster.compiler.ir.screma-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]))

;; ================================================================
;; Screma construction from SOAC types
;; ================================================================

(deftest screma-from-map-test
  (testing "SoacMap → Screma: pure map with map-lambda"
    (let [map-node (soac/par-form->soac 'out
                                        '(raster.par/map! out i n double (* (aget a i) 2.0)) 0)
          screma (soac/soac->screma map-node)]
      (is (instance? raster.compiler.ir.soac.Screma screma))
      (is (= '(* (aget a i) 2.0) (:map-lambda screma)))
      (is (empty? (:scans screma)))
      (is (empty? (:reduces screma)))
      (is (= 'double (:cast-fn screma))))))

(deftest screma-from-reduce-test
  (testing "SoacReduce → Screma: pure reduce with one reduce op"
    (let [red-node (soac/par-form->soac 'result
                                        '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 0)
          screma (soac/soac->screma red-node)]
      (is (instance? raster.compiler.ir.soac.Screma screma))
      (is (nil? (:map-lambda screma)))
      (is (empty? (:scans screma)))
      (is (= 1 (count (:reduces screma))))
      (let [r (first (:reduces screma))]
        (is (= 'acc (:acc r)))
        (is (= 0.0 (:init r)))))))

(deftest screma-from-scan-test
  (testing "SoacScan → Screma: pure scan with one scan op"
    (let [scan-node (soac/par-form->soac 'out
                                         '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i))) 0)
          screma (soac/soac->screma scan-node)]
      (is (instance? raster.compiler.ir.soac.Screma screma))
      (is (nil? (:map-lambda screma)))
      (is (= 1 (count (:scans screma))))
      (is (empty? (:reduces screma))))))

;; ================================================================
;; Screma → par form round-trip
;; ================================================================

(deftest screma-roundtrip-map-test
  (testing "Screma round-trip for pure map"
    (let [map-node (soac/par-form->soac 'out
                                        '(raster.par/map! out i n double (* (aget a i) 2.0)) 0)
          screma (soac/soac->screma map-node)
          par-form (soac/screma->par-form screma)]
      (is (= 'raster.par/map! (first par-form)))
      (is (= 'out (second par-form))))))

(deftest screma-roundtrip-reduce-test
  (testing "Screma round-trip for pure reduce"
    (let [red-node (soac/par-form->soac 'result
                                        '(raster.par/reduce acc 0.0 j n (+ acc (aget a j))) 0)
          screma (soac/soac->screma red-node)
          par-form (soac/screma->par-form screma)]
      (is (= 'raster.par/reduce (first par-form)))
      (is (= 'acc (second par-form))))))

(deftest screma-roundtrip-scan-test
  (testing "Screma round-trip for pure scan"
    (let [scan-node (soac/par-form->soac 'out
                                         '(raster.par/scan out acc 0.0 i n double (+ acc (aget a i))) 0)
          screma (soac/soac->screma scan-node)
          par-form (soac/screma->par-form screma)]
      (is (= 'raster.par/scan (first par-form)))
      (is (= 'out (second par-form))))))

;; ================================================================
;; Silently-ignored-information family: a Map+Reduce screma whose map
;; was NOT inlined into the reduce lambda must fail loud — emitting only
;; the reduce (as the arm used to) would drop the map computation.
;; ================================================================

(deftest screma-map-reduce-unfused-throws
  (testing "a reduce screma still carrying a :map-lambda is rejected, not silently reduce-only"
    (let [screma (soac/->Screma 1 'r 'i 'n #{'a 'tmp} #{'r} #{}
                                nil [] [{:acc 'acc :init 0.0
                                         :lambda '(+ acc (aget tmp i))}]
                                '(* (aget a i) 2.0))]
      (is (thrown-with-msg?
           Exception #"was not inlined into the reduce lambda"
           (soac/screma->par-form screma)))))
  (testing "a properly-fused reduce (no map-lambda) still converts"
    (let [screma (soac/->Screma 1 'r 'i 'n #{'a} #{'r} #{}
                                nil [] [{:acc 'acc :init 0.0
                                         :lambda '(+ acc (aget a i))}]
                                nil)]
      (is (= 'raster.par/reduce (first (soac/screma->par-form screma)))))))

;; ================================================================
;; Screma composition
;; ================================================================

(deftest compose-map-map-test
  (testing "Map→Map composition: producer body inlined into consumer"
    (let [prod (soac/soac->screma
                (soac/par-form->soac 'tmp
                                     '(raster.par/map! tmp i n double (* (aget a i) 2.0)) 0))
          cons (soac/soac->screma
                (soac/par-form->soac 'out
                                     '(raster.par/map! out i n double (+ (aget tmp i) 1.0)) 1))
          composed (soac/screma-compose prod cons 'tmp)]
      ;; Should have map-lambda, no reduces/scans
      (is (some? (:map-lambda composed)))
      (is (empty? (:reduces composed)))
      (is (empty? (:scans composed)))
      ;; Input should be 'a', not 'tmp'
      (is (contains? (:inputs composed) 'a))
      (is (not (contains? (:inputs composed) 'tmp))))))

(deftest compose-map-reduce-test
  (testing "Map→Reduce composition: map body inlined into reduce"
    (let [prod (soac/soac->screma
                (soac/par-form->soac 'tmp
                                     '(raster.par/map! tmp i n double (* (aget a i) (aget a i))) 0))
          cons (soac/soac->screma
                (soac/par-form->soac 'result
                                     '(raster.par/reduce acc 0.0 j n (+ acc (aget tmp j))) 1))
          composed (soac/screma-compose prod cons 'tmp)]
      ;; Should have one reduce, no map-lambda
      (is (= 1 (count (:reduces composed))))
      ;; Input should be 'a', not 'tmp'
      (is (contains? (:inputs composed) 'a))
      (is (not (contains? (:inputs composed) 'tmp))))))

(deftest compose-map-scan-test
  (testing "Map→Scan composition: map body inlined into scan"
    (let [prod (soac/soac->screma
                (soac/par-form->soac 'tmp
                                     '(raster.par/map! tmp i n double (* (aget a i) 2.0)) 0))
          cons (soac/soac->screma
                (soac/par-form->soac 'out
                                     '(raster.par/scan out acc 0.0 i n double (+ acc (aget tmp i))) 1))
          composed (soac/screma-compose prod cons 'tmp)]
      ;; Should have one scan
      (is (= 1 (count (:scans composed))))
      (is (empty? (:reduces composed)))
      ;; Input should be 'a'
      (is (contains? (:inputs composed) 'a))
      (is (not (contains? (:inputs composed) 'tmp))))))

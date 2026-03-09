(ns raster.compiler.passes.parallel.compound-detect-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.compound-detect :as cd]))

;; ================================================================
;; Phase extraction
;; ================================================================

(deftest extract-phases-test
  (testing "Extracts map and stencil phases from a body"
    (let [body '(do
                  (raster.par/stencil! k1 [u] 1 :dirichlet double i__1 n
                                       (* alpha (* inv-dx2 (+ (aget u (- i__1 1))
                                                              (* -2.0 (aget u i__1))
                                                              (aget u (+ i__1 1))))))
                  (raster.par/map! tmp i__2 n double
                                   (+ (aget u i__2) (* half-dt (aget k1 i__2))))
                  (raster.par/map! u i__3 n double
                                   (+ (aget u i__3) (* dt6 (aget k1 i__3)))))
          phases (#'cd/extract-phases body)]
      (is (= 3 (count phases)))
      (is (= :stencil (:type (first phases))))
      (is (= :map (:type (second phases))))
      (is (= :map (:type (nth phases 2))))
      (is (= 'k1 (:out (first phases))))
      (is (= 'tmp (:out (second phases))))
      (is (= 'u (:out (nth phases 2)))))))

;; ================================================================
;; Array classification
;; ================================================================

(deftest classify-arrays-test
  (testing "Classifies inputs, outputs, and scratch arrays"
    (let [phases [{:type :stencil :out 'k1 :inputs #{'u}}
                  {:type :map :out 'tmp :inputs #{'u 'k1}}
                  {:type :map :out 'u :inputs #{'u 'k1 'k2 'k3 'k4}}]
          ;; u is used after the loop (for reduce)
          live-after #{'u 'target}
          result (#'cd/classify-arrays phases live-after)]
      (is (contains? (:outputs result) 'u))
      (is (contains? (:scratch result) 'k1))
      (is (contains? (:scratch result) 'tmp)))))

;; ================================================================
;; Strategy selection
;; ================================================================

(deftest select-strategy-test
  (testing "Small n selects :local"
    (is (= :local (cd/select-strategy 64 5 nil))))

  (testing "Large n selects :global"
    (is (= :global (cd/select-strategy 4096 5 nil))))

  (testing "Many scratch arrays force :global even with small n"
    ;; 256 elements * 100 scratch arrays * 8 bytes = 204800 > 49152 (75% of 65536)
    (is (= :global (cd/select-strategy 256 100 nil))))

  (testing "Boundary: n exactly at max-workgroup-size"
    ;; 1024 elements, 5 scratch: 1024*5*8=40960 < 49152 (75% of 65536)
    (is (= :local (cd/select-strategy 1024 5 nil))))

  (testing "Just over max-workgroup-size selects :global"
    (is (= :global (cd/select-strategy 1025 5 nil)))))

;; ================================================================
;; Full detection pass
;; ================================================================

(deftest compound-detect-basic-test
  (testing "Detects compound kernel in dotimes with ≥2 par forms"
    (let [form '(let* [n 64
                       u (double-array n)
                       k1 (double-array n)
                       tmp (double-array n)]
                      (dotimes [step 100]
                        (raster.par/stencil! k1 [u] 1 :dirichlet double i__1 n
                                             (* alpha (* inv-dx2 (+ (aget u (- i__1 1))
                                                                    (* -2.0 (aget u i__1))
                                                                    (aget u (+ i__1 1))))))
                        (raster.par/map! tmp i__2 n double
                                         (+ (aget u i__2) (* half-dt (aget k1 i__2))))
                        (raster.par/map! u i__3 n double
                                         (+ (aget u i__3) (* dt6 (aget k1 i__3)))))
                      u)
          result (cd/compound-detect-pass form {})
          detected-form (:form result)]
      (is (= 1 (get-in result [:stats :compound-kernels])))
      ;; The dotimes should be wrapped in compound-kernel marker
      (is (some #(and (seq? %) (= 'raster.compiler/compound-kernel (first %)))
                (tree-seq seq? seq detected-form))))))

(deftest no-compound-single-par-test
  (testing "Single par form in dotimes is NOT a compound candidate"
    (let [form '(dotimes [i 100]
                  (raster.par/map! out j n double (+ (aget a j) 1.0)))
          result (cd/compound-detect-pass form {})]
      (is (= 0 (get-in result [:stats :compound-kernels]))))))

(deftest compound-metadata-test
  (testing "Compound kernel metadata contains correct arrays and phases"
    (let [form '(let* [u (double-array 64)]
                      (dotimes [step 10]
                        (raster.par/map! k1 i n double (aget u i))
                        (raster.par/map! u i n double (+ (aget u i) (aget k1 i))))
                      (aget u 0))
          result (cd/compound-detect-pass form {})
          ;; Find the compound-kernel form
          compound (first (filter #(and (seq? %) (= 'raster.compiler/compound-kernel (first %)))
                                  (tree-seq seq? seq (:form result))))
          metadata (second compound)]
      (is (some? metadata))
      (is (contains? #{:local :global} (get-in metadata [:execution :strategy])))
      (is (= :compound (get-in metadata [:execution :kind])))
      (is (vector? (:phases metadata)))
      (is (= 2 (count (:phases metadata))))
      (is (= :map (:type (first (:phases metadata)))))
      (is (= :map (:type (second (:phases metadata))))))))

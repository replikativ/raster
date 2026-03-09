(ns raster.compiler.passes.parallel.speculative-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.speculative :as spec]
            [raster.compiler.passes.scalar.pe :as pe]))

;; ================================================================
;; Shape signature tests
;; ================================================================

(deftest shape-signature-test
  (testing "Empty extractors produce empty sig"
    (is (= [] (spec/shape-signature [1 2 3] []))))

  (testing "Shape extraction with mock extractors"
    (let [extractors [{:param 'x :tag 'Vector
                       :extractor (fn [x] [(count x)])}]
          args [[1 2 3]]]
      (is (= [3] (spec/shape-signature args extractors))))))

(deftest make-shape-extractors-test
  (testing "Scalar params have no extractors"
    (let [params '[x y]
          tags   ['double 'double]
          result (spec/make-shape-extractors params tags)]
      (is (empty? result))))

  (testing "Unknown tags have no extractors"
    (let [params '[x]
          tags   ['Object]
          result (spec/make-shape-extractors params tags)]
      (is (empty? result)))))

;; ================================================================
;; Speculative wrapper tests
;; ================================================================

(deftest wrap-speculative-no-extractors-test
  (testing "No extractors: always use generic"
    (let [call-count (atom 0)
          generic-fn (fn [& args]
                       (swap! call-count inc)
                       (apply + args))
          wrapped (spec/wrap-speculative
                   generic-fn nil
                   {:extractors [] :threshold 3})]
      (is (= 6 (wrapped 1 2 3)))
      (is (= 10 (wrapped 4 6)))
      (is (= 2 @call-count)))))

(deftest wrap-speculative-with-extractors-test
  (testing "With extractors: records observations and falls back to generic"
    (let [call-count (atom 0)
          generic-fn (fn [& args]
                       (swap! call-count inc)
                       (reduce + (first args)))
          extractors [{:param 'x :tag 'Vector
                       :extractor (fn [x] [(count x)])}]
          wrapped (spec/wrap-speculative
                   generic-fn nil
                   {:extractors extractors
                    :threshold 3
                     ;; No compile-fn, so no specialization
                    :compile-fn nil})]
      ;; Should use generic for all calls
      (is (= 6 (wrapped [1 2 3])))
      (is (= 10 (wrapped [4 6])))
      (is (= 2 @call-count)))))

(deftest wrap-speculative-specialization-test
  (testing "After threshold stable calls, specialization is triggered"
    (let [generic-calls (atom 0)
          special-calls (atom 0)
          generic-fn (fn [& args]
                       (swap! generic-calls inc)
                       42)
          compile-count (atom 0)
          compile-fn (fn [known-dims]
                       (swap! compile-count inc)
                       ;; Return a "specialized" fn
                       (fn [& args]
                         (swap! special-calls inc)
                         99))
          extractors [{:param 'x :tag 'Vector
                       :extractor (fn [x] [(count x)])}]
          wrapped (spec/wrap-speculative
                   generic-fn nil
                   {:extractors extractors
                    :threshold 2
                    :compile-fn compile-fn})]
      ;; First 2 calls (threshold=2): generic
      (is (= 42 (wrapped [1 2 3])))
      (is (= 42 (wrapped [1 2 3])))
      ;; Specialization triggered in background after threshold reached.
      ;; 3rd call may see either generic (42) or specialized (99)
      ;; depending on whether the future completed — don't assert a
      ;; specific value here to avoid a race.
      (let [transitional (wrapped [1 2 3])]
        (is (contains? #{42 99} transitional)))
      ;; Wait for specialization to complete
      (Thread/sleep 200)
      ;; After specialization, should use specialized
      (is (= 99 (wrapped [1 2 3])))
      (is (>= @compile-count 1)))))

;; ================================================================
;; PE specialize-dimensions tests
;; ================================================================

(deftest specialize-dimensions-test
  (testing "Replaces dimension queries with constants"
    (let [body '(let* [buf (make-array (raster.arrays/alength x))]
                      buf)
          known-dims {'(raster.arrays/alength x) 784}
          result (pe/specialize-dimensions body known-dims)]
      (is (some? result))
      (is (= '(let* [buf (make-array 784)] buf) result))))

  (testing "Leaves non-matching exprs untouched"
    (let [body '(+ x y)
          known-dims {'(raster.arrays/alength z) 100}
          result (pe/specialize-dimensions body known-dims)]
      (is (= '(+ x y) result))))

  (testing "Handles nested forms"
    (let [body '(let* [n (raster.dl.tensor/shape-dim W 0)
                       buf (make-array n)]
                      buf)
          known-dims {'(raster.dl.tensor/shape-dim W 0) 128}
          result (pe/specialize-dimensions body known-dims)]
      (is (= '(let* [n 128 buf (make-array n)] buf) result)))))

;; ================================================================
;; Concurrent safety test
;; ================================================================

(deftest concurrent-calls-test
  (testing "Concurrent calls during specialization don't deadlock"
    (let [generic-fn (fn [& args] 42)
          compile-fn (fn [_dims]
                       (Thread/sleep 100)
                       (fn [& args] 99))
          extractors [{:param 'x :tag 'Vector
                       :extractor (fn [x] [(count x)])}]
          wrapped (spec/wrap-speculative
                   generic-fn nil
                   {:extractors extractors
                    :threshold 1
                    :compile-fn compile-fn})
          results (atom [])]
      ;; Launch concurrent calls
      (let [futures (doall (for [_ (range 10)]
                             (future (swap! results conj (wrapped [1 2 3])))))]
        (doseq [f futures] @f))
      ;; All calls should return either 42 or 99
      (is (every? #{42 99} @results)))))

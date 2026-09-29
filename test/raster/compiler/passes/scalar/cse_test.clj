(ns raster.compiler.passes.scalar.cse-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.cse :as cse]))

(deftest basic-cse-test
  (testing "duplicate expressions become aliases"
    (let [form '(let* [a (Math/sin x)
                       b (Math/sin x)
                       c (+ a b)]
                      c)
          {:keys [form stats]} (cse/cse-let form)]
      (is (= 1 (:cse-aliases stats)))
      ;; b should now be aliased to a
      (let [[_ bindings _] form
            pairs (partition 2 bindings)]
        (is (= 'a (second (second pairs))))))))

(deftest no-cse-for-impure-test
  (testing "side-effecting calls are not CSE'd"
    (let [form '(let* [a (reset! buf 1.0)
                       b (reset! buf 1.0)]
                      (+ a b))
          {:keys [stats]} (cse/cse-let form)]
      (is (= 0 (:cse-aliases stats))))))

(deftest fresh-allocations-are-removable-but-not-commonable
  (testing "devirtualized pure allocators retain distinct mutable identities"
    (let [allocation
          (with-meta
            '(.invk raster.arrays/zeros-like_m_floats_long-impl exemplar n)
            {:raster.op/original 'raster.arrays/zeros-like
             :raster.type/tag 'floats :tag 'floats})
          form (list 'let* ['q allocation 'k allocation] '[q k])
          {:keys [form stats]} (cse/cse-let form)]
      (is (= 0 (:cse-aliases stats)))
      (is (= allocation (second (second (partition 2 (second form)))))))))

(deftest no-cse-for-different-exprs-test
  (testing "different expressions stay independent"
    (let [{:keys [stats]} (cse/cse-let '(let* [a (Math/sin x) b (Math/cos x)] (+ a b)))]
      (is (= 0 (:cse-aliases stats))))))

(deftest multiple-duplicates-test
  (testing "multiple duplicates are all caught"
    (let [form '(let* [a (Math/sin x)
                       b (Math/sin x)
                       c (Math/cos x)
                       d (Math/cos x)
                       e (Math/sin x)]
                      (+ a b c d e))
          {:keys [stats]} (cse/cse-let form)]
      (is (= 3 (:cse-aliases stats))))))

(deftest non-let-passthrough-test
  (testing "non-let forms pass through unchanged"
    (let [form '(do (+ 1 2))
          {:keys [form stats]} (cse/cse-let form)]
      (is (= '(do (+ 1 2)) form))
      (is (= 0 (:cse-aliases stats))))))

(deftest metadata-stripped-for-matching-test
  (testing "type hint metadata doesn't prevent matching"
    (let [form '(let* [a (Math/sin ^double x)
                       b (Math/sin x)]
                      (+ a b))
          {:keys [stats]} (cse/cse-let form)]
      (is (= 1 (:cse-aliases stats))))))

(deftest cse-body-substitution-preserves-lexical-values
  (doseq [source ['(let* [x (Math/sin 0.5) y (Math/sin 0.5)]
                     (let* [x 2.0] y))
                  '(let* [count (Math/sin 0.5) y (Math/sin 0.5)]
                     (let* [count 2.0] y))
                  '(let* [x (Math/sin 0.5) y (Math/sin 0.5)]
                     ((fn* [y] y) 7.0))
                  '(let* [x (Math/sin 0.5) y (Math/sin 0.5)]
                     (quote y))]]
    (let [{:keys [form stats]} (cse/cse-let source)]
      (is (= 1 (:cse-aliases stats)))
      (is (= (eval source) (eval form)) (str form)))))

(deftest vector-projections-do-not-cross-shadowing-body-scopes
  (doseq [source ['(let* [x 3.0 values [x]]
                     (let* [x 9.0] (clojure.core/nth values 0)))
                  '(let* [values [3.0]]
                     (let* [values [9.0]] (clojure.core/nth values 0)))]]
    (let [optimized (:form (cse/cse-let source))]
      (is (= (eval source) (eval optimized)) (str optimized)))))

(deftest vector-projection-retains-evaluate-once-semantics
  (doseq [source ['(let* [counter (atom 0) values [(swap! counter inc)]]
                     (clojure.core/nth values 0))
                  '(let* [counter (atom 0) values [(swap! counter inc)]
                          result (clojure.core/nth values 0)] result)
                  '(let* [storage (double-array [3.0]) values [(aget storage 0)]
                          ignored (aset storage 0 9.0)]
                     (clojure.core/nth values 0))
                  '(let* [values [clojure.core/*print-length*]]
                     (binding [clojure.core/*print-length* 17]
                       (clojure.core/nth values 0)))
                  '(let* [values [count] count 7]
                     (clojure.core/nth values 0))
                  '(let* [x 3 values [x] x 9]
                     (clojure.core/nth values 0))
                  '(let* [values [3] values (vector 9)]
                     (clojure.core/nth values 0))]]
    (let [optimized (:form (cse/cse-let source))]
      (is (= (eval source) (eval optimized)) (str optimized)))))

(deftest evaluated-local-vector-projection-still-folds
  (let [source '(let* [counter (atom 0) saved (swap! counter inc) values [saved]]
                  (clojure.core/nth values 0))
        optimized (:form (cse/cse-let source))]
    (is (= 'saved (last optimized)))
    (is (= (eval source) (eval optimized)))))

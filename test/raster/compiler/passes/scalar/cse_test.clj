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

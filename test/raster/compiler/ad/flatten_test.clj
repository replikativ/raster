(ns raster.compiler.ad.flatten-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ad.flatten :as flat]))

;; ================================================================
;; extract-ad-structure tests
;; ================================================================

(deftest extract-ad-structure-test
  (testing "parses standard AD output"
    (let [form '(let* [a (* x y)
                       result__1 a]
                      [result__1
                       (fn* [dy__rad]
                            (let* [d_a__2 dy__rad
                                   dg_x__3 (raster.numeric/* d_a__2 y)
                                   dg_y__4 (raster.numeric/* d_a__2 x)
                                   d_x__5 dg_x__3
                                   d_y__6 dg_y__4]
                                  [d_x__5 d_y__6]))])
          parsed (#'flat/extract-ad-structure form)]
      (is (some? parsed))
      (is (= 'result__1 (:result-sym parsed)))
      (is (= 'dy__rad (:dy-sym parsed)))
      (is (= '[d_x__5 d_y__6] (:param-adj-syms parsed)))))

  (testing "returns nil for non-AD forms"
    (is (nil? (#'flat/extract-ad-structure '(+ x y))))
    (is (nil? (#'flat/extract-ad-structure '(let* [a 1] a))))))

(deftest canonicalize-ad-form-test
  (testing "wraps raw AD vector output in canonical let*"
    (let [raw '[x (fn* [dy] [dy])]
          result (flat/canonicalize-ad-form raw)]
      (is (seq? result))
      (is (= 'let* (first result)))
      (is (= [] (second result)))
      (is (= raw (nth (vec result) 2)))))

  (testing "canonicalizes let input to let*"
    (let [form '(let [a x]
                  [a (fn* [dy] [dy])])
          result (flat/canonicalize-ad-form form)]
      (is (= 'let* (first result)))))

  (testing "returns nil for non-AD forms"
    (is (nil? (flat/canonicalize-ad-form '(+ x y))))))

;; ================================================================
;; flatten-ad-form tests
;; ================================================================

(deftest flatten-ad-form-test
  (testing "flattens AD output into single let*"
    (let [form '(let* [a (raster.numeric/* x y)
                       result__1 a]
                      [result__1
                       (fn* [dy__rad]
                            (let* [d_a__2 dy__rad
                                   dg_x__3 (raster.numeric/* d_a__2 y)
                                   dg_y__4 (raster.numeric/* d_a__2 x)]
                                  [dg_x__3 dg_y__4]))])
          result (flat/flatten-ad-form form)]
      (is (some? result))
      (let [{:keys [form result-sym param-adj-syms]} result
            [_ bindings & _] form]
        ;; Should have fwd + [dy 1.0] + rev bindings
        (is (= 'let* (first form)))
        ;; dy__rad should be bound to 1.0 somewhere in bindings
        (let [pairs (partition 2 bindings)]
          (is (some (fn [[s e]] (and (= s 'dy__rad) (= e 1.0))) pairs)
              "dy__rad should be bound to 1.0"))
        ;; result-sym should be the body
        (is (= 'result__1 result-sym))
        ;; param-adj-syms from the pullback return
        (is (= '[dg_x__3 dg_y__4] param-adj-syms))))))

(deftest flatten-ad-form-canonicalizes-let-test
  (testing "canonicalizes let input to let* output"
    (let [form '(let [a (raster.numeric/* x y)
                      result__1 a]
                  [result__1
                   (fn* [dy__rad]
                        (let [d_a__2 dy__rad
                              dg_x__3 (raster.numeric/* d_a__2 y)
                              dg_y__4 (raster.numeric/* d_a__2 x)]
                          [dg_x__3 dg_y__4]))])
          result (flat/flatten-ad-form form)]
      (is (some? result))
      (is (= 'let* (first (:form result)))))))

;; ================================================================
;; flatten-for-gradient tests
;; ================================================================

(deftest flatten-for-gradient-test
  (testing "flattens AD output into value+gradient form"
    (let [ad-form '(let* [a (raster.numeric/* x y)
                          result__1 a]
                         [result__1
                          (fn* [dy__rad]
                               (let* [d_a__2 dy__rad
                                      dg_x__3 (raster.numeric/* d_a__2 y)
                                      dg_y__4 (raster.numeric/* d_a__2 x)]
                                     [dg_x__3 dg_y__4]))])
          result (flat/flatten-for-gradient ad-form)]
      (is (some? result))
      (let [{:keys [form result-sym param-adj-syms stats]} result]
        ;; Form should be a flat let*
        (is (= 'let* (first form)))
        ;; Body should be [result grad1 grad2]
        (let [body (nth (vec form) 2)]
          (is (vector? body))
          (is (= 3 (count body)))
          (is (= 'result__1 (first body))))
        ;; result-sym
        (is (= 'result__1 result-sym))
        ;; param-adj-syms
        (is (= '[dg_x__3 dg_y__4] param-adj-syms))
        ;; stats
        (is (pos? (:bindings-flattened stats))))))

  (testing "returns nil for non-AD form"
    (is (nil? (flat/flatten-for-gradient '(+ x y))))))


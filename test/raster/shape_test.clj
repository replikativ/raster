(ns raster.shape-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.dispatch]))

;; ================================================================
;; Compound tag utilities
;; ================================================================

(deftest compound-tag-test
  (testing "compound-tag? recognizes compound tags"
    (is (types/compound-tag? '(Matrix 32 64)))
    (is (types/compound-tag? '(DenseArray 1024)))
    (is (not (types/compound-tag? 'double)))
    (is (not (types/compound-tag? nil))))

  (testing "compound-tag-base extracts base type"
    (is (= 'Matrix (types/compound-tag-base '(Matrix 32 64))))
    (is (= 'double (types/compound-tag-base 'double))))

  (testing "compound-tag-params extracts shape params"
    (is (= [32 64] (types/compound-tag-params '(Matrix 32 64))))
    (is (nil? (types/compound-tag-params 'double)))))

;; ================================================================
;; Mangle with compound tags
;; ================================================================

(deftest mangle-compound-test
  (testing "mangle handles compound tags"
    (is (= 'matmul_m_Matrix_32_64_Matrix_64_16
           (types/mangle 'matmul ['(Matrix 32 64) '(Matrix 64 16)]))))

  (testing "mangle still handles simple tags"
    (is (= 'foo_m_long_double
           (types/mangle 'foo ['long 'double])))))

;; ================================================================
;; tag->check-class with compound tags
;; ================================================================

(deftest tag-check-class-compound-test
  (testing "tag->check-class resolves compound tag via base"
    ;; java.lang.String is a known class
    (is (= String (types/tag->check-class '(String 42))))
    ;; Simple tags still work
    (is (= Long (types/tag->check-class 'long)))))

;; ================================================================
;; annotation->tag with compound tags
;; ================================================================

(deftest annotation-tag-compound-test
  (testing "annotation->tag recognizes compound parametric annotations"
    (is (= '(Matrix 32 64)
           (types/annotation->tag '(Matrix 32 64) 'x)))
    (is (= '(DenseArray 1024)
           (types/annotation->tag '(DenseArray 1024) 'x))))

  (testing "annotation->tag still handles simple annotations"
    (is (= 'long (types/annotation->tag 'Long 'x)))
    (is (= 'doubles (types/annotation->tag '(Array double) 'x)))))

;; ================================================================
;; Shaped protocol and dispatch
;; ================================================================

(defrecord TestMatrix [rows cols data]
  types/Shaped
  (shape [_] [rows cols]))

(deftest shaped-dispatch-test
  (testing "shape-matches? checks shape parameters"
    (let [m (->TestMatrix 32 64 nil)]
      (is (#'raster.compiler.core.dispatch/shape-matches? '(TestMatrix 32 64) m))
      (is (not (#'raster.compiler.core.dispatch/shape-matches? '(TestMatrix 32 32) m)))
      (is (#'raster.compiler.core.dispatch/shape-matches? 'TestMatrix m) "simple tag always matches"))))

;; ================================================================
;; Shape rule registry
;; ================================================================

(deftest shape-rule-registry-test
  (testing "register and apply shape rules"
    ;; Register matmul rule
    (types/register-shape-rule! 'matmul
                                (fn [arg-tags]
                                  (let [[a-tag b-tag] arg-tags]
                                    (when (and (types/compound-tag? a-tag) (types/compound-tag? b-tag))
                                      (let [a-base (types/compound-tag-base a-tag)
                                            b-base (types/compound-tag-base b-tag)
                                            a-params (types/compound-tag-params a-tag)
                                            b-params (types/compound-tag-params b-tag)]
                                        (cond
                ;; Matrix × Vector
                                          (and (= 'Matrix a-base) (= 'Vector b-base)
                                               (= (second a-params) (first b-params)))
                                          (list 'Vector (first a-params))
                ;; Matrix × Matrix
                                          (and (= 'Matrix a-base) (= 'Matrix b-base)
                                               (= (second a-params) (first b-params)))
                                          (list 'Matrix (first a-params) (second b-params))
                                          :else nil))))))

    ;; Matrix × Vector
    (is (= '(Vector 64)
           (types/infer-shape-from-rules 'matmul ['(Matrix 64 25) '(Vector 25)])))

    ;; Matrix × Matrix
    (is (= '(Matrix 64 10)
           (types/infer-shape-from-rules 'matmul ['(Matrix 64 25) '(Matrix 25 10)])))

    ;; Dimension mismatch → nil
    (is (nil? (types/infer-shape-from-rules 'matmul ['(Matrix 64 25) '(Vector 10)])))

    ;; Simple (non-compound) tags → nil
    (is (nil? (types/infer-shape-from-rules 'matmul ['double 'double]))))

  (testing "qualified symbol lookup strips namespace"
    (types/register-shape-rule! 'transpose
                                (fn [arg-tags]
                                  (let [[a-tag] arg-tags]
                                    (when (and (types/compound-tag? a-tag) (= 'Matrix (types/compound-tag-base a-tag)))
                                      (let [[m n] (types/compound-tag-params a-tag)]
                                        (list 'Matrix n m))))))
    ;; Qualified symbol should resolve via base name
    (is (= '(Matrix 25 64)
           (types/infer-shape-from-rules 'raster.linalg.core/transpose ['(Matrix 64 25)]))))

  (testing "chained shape inference"
    ;; matmul(A, B) then matmul(C, v) chains correctly
    (let [ab-shape (types/infer-shape-from-rules 'matmul ['(Matrix 64 25) '(Matrix 25 10)])]
      (is (= '(Vector 64)
             (types/infer-shape-from-rules 'matmul [ab-shape '(Vector 10)]))))))

;; ================================================================
;; TC Value propagation (if TC is available)
;; ================================================================

(deftest tc-value-propagation-test
  (testing "TC extensions load and propagate Values"
    (require 'raster.compiler.core.tc-extensions)
    (let [check-fn (requiring-resolve 'clojure.core.typed/check-form-info)]
      ;; (* 32 32) should infer (t/Val 1024)
      (let [result (check-fn '(clojure.core/* 32 32) {})
            ret-type (-> result :ret :t)]
        (is (some? ret-type) "TC should return a type")
        (is (= 1024 (:val ret-type)) "* should propagate Value types"))
      ;; Chaining: (+ (* 3 4) 10)
      (let [result (check-fn '(clojure.core/+ (clojure.core/* 3 4) 10) {})
            ret-type (-> result :ret :t)]
        (is (= 22 (:val ret-type)) "+ and * should chain Value propagation")))))

(ns raster.ad.activity-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.ir.core :as ir]
            [raster.ad.activity :as act]))

;; ================================================================
;; Helper
;; ================================================================

(defn analyze [form active-params]
  (ir/with-fresh-tables
    (binding [act/*activity-table* (atom {})]
      (let [node (ir/sexp->ir form)]
        (act/analyze-activity! node active-params)
        {:node node
         :active? (act/active? node)}))))

;; ================================================================
;; Basic activity propagation
;; ================================================================

(deftest literal-always-const-test
  (testing "Numeric literals are constant"
    (is (not (:active? (analyze 42 #{}))))
    (is (not (:active? (analyze 3.14 #{}))))))

(deftest active-param-test
  (testing "Symbol in active set is active"
    (is (:active? (analyze 'x #{'x}))))
  (testing "Symbol NOT in active set is const"
    (is (not (:active? (analyze 'y #{'x}))))))

;; ================================================================
;; Let binding propagation
;; ================================================================

(deftest let-propagation-test
  (testing "Active param flows through let binding"
    (is (:active? (analyze '(let* [a x] a) #{'x}))))
  (testing "Constant let binding stays const"
    (is (not (:active? (analyze '(let* [a 3.0] a) #{'x})))))
  (testing "Active computation in let"
    (is (:active? (analyze '(let* [a (* x 2.0)] a) #{'x}))))
  (testing "Inactive binding with active body"
    (is (:active? (analyze '(let* [a 3.0] x) #{'x}))))
  (testing "Mixed bindings"
    (is (:active? (analyze '(let* [a 3.0
                                   b (* a x)]
                                  b)
                           #{'x})))))

;; ================================================================
;; Call / Invk propagation
;; ================================================================

(deftest call-propagation-test
  (testing "Call with active arg is active"
    (is (:active? (analyze '(Math/sin x) #{'x}))))
  (testing "Call with only const args is const"
    (is (not (:active? (analyze '(Math/sin 3.0) #{'x})))))
  (testing "Binary call with one active arg"
    (is (:active? (analyze '(* x 3.0) #{'x}))))
  (testing "Binary call with both active"
    (is (:active? (analyze '(* x y) #{'x 'y})))))

(deftest invk-propagation-test
  (testing ".invk with active arg is active"
    (is (:active? (analyze '(.invk _star__m_double_double-impl x 3.0) #{'x}))))
  (testing ".invk with only const args is const"
    (is (not (:active? (analyze '(.invk _star__m_double_double-impl 2.0 3.0) #{'x}))))))

;; ================================================================
;; If propagation
;; ================================================================

(deftest if-propagation-test
  (testing "If with active then branch"
    (is (:active? (analyze '(if p x 0.0) #{'x}))))
  (testing "If with active else branch"
    (is (:active? (analyze '(if p 0.0 x) #{'x}))))
  (testing "If with both branches const"
    (is (not (:active? (analyze '(if p 1.0 2.0) #{'x})))))
  (testing "If where condition is active but branches aren't"
    ;; The condition being active doesn't make the result active
    ;; unless a branch uses an active value
    (is (not (:active? (analyze '(if x 1.0 2.0) #{'x}))))))

;; ================================================================
;; Do propagation
;; ================================================================

(deftest do-propagation-test
  (testing "Do is active if last form is active"
    (is (:active? (analyze '(do 42 x) #{'x}))))
  (testing "Do is const if last form is const"
    (is (not (:active? (analyze '(do x 42) #{'x}))))))

;; ================================================================
;; Vector propagation
;; ================================================================

(deftest vector-propagation-test
  (testing "Vector with active element"
    (is (:active? (analyze '[x 1.0] #{'x}))))
  (testing "Vector with only const elements"
    (is (not (:active? (analyze '[1.0 2.0] #{'x}))))))

;; ================================================================
;; Recur propagation
;; ================================================================

(deftest recur-propagation-test
  (testing "Recur with active arg"
    ;; Just test the recur form itself
    (ir/with-fresh-tables
      (binding [act/*activity-table* (atom {})]
        (let [node (ir/sexp->ir '(recur (+ i 1) (+ acc x)))]
          (act/propagate-forward! node {'x :active 'i :const 'acc :active})
          (is (act/active? node)))))))

;; ================================================================
;; Backward pruning (useful-activity)
;; ================================================================

(deftest backward-pruning-test
  (testing "Unused active binding is pruned"
    ;; a = (* x 2) is active but unused -> should be pruned
    (ir/with-fresh-tables
      (binding [act/*activity-table* (atom {})]
        (let [node (ir/sexp->ir '(let* [a (* x 2.0)] 42.0))]
          (act/analyze-activity! node #{'x})
          ;; The let result is const (body is literal 42.0)
          (is (not (act/active? node)))))))
  (testing "Used active binding is kept"
    (ir/with-fresh-tables
      (binding [act/*activity-table* (atom {})]
        (let [node (ir/sexp->ir '(let* [a (* x 2.0)] a))]
          (act/analyze-activity! node #{'x})
          (is (act/active? node)))))))

;; ================================================================
;; Realistic examples
;; ================================================================

(deftest lorenz-activity-test
  (testing "Lorenz dx = sigma * (y - x) with active x, y, z"
    (is (:active? (analyze '(let* [diff (- y x)
                                   dx (* sigma diff)]
                                  dx)
                           #{'x 'y 'z}))))
  (testing "Lorenz with sigma as only active param"
    (is (:active? (analyze '(let* [diff (- y x)
                                   dx (* sigma diff)]
                                  dx)
                           #{'sigma})))))

(deftest polynomial-activity-test
  (testing "f(x) = a*x^2 + b*x + c with x active"
    (is (:active? (analyze '(let* [x2 (* x x)
                                   ax2 (* a x2)
                                   bx (* b x)
                                   result (+ ax2 (+ bx c))]
                                  result)
                           #{'x}))))
  (testing "f(x) = a*x^2 + b*x + c with a,b active but x const"
    (is (:active? (analyze '(let* [x2 (* x x)
                                   ax2 (* a x2)
                                   bx (* b x)
                                   result (+ ax2 (+ bx c))]
                                  result)
                           #{'a 'b})))))

(deftest chain-rule-activity-test
  (testing "sin(cos(x)) — chain of active"
    (is (:active? (analyze '(let* [a (Math/cos x)
                                   b (Math/sin a)]
                                  b)
                           #{'x}))))
  (testing "sin(cos(3.0)) — constant chain"
    (is (not (:active? (analyze '(let* [a (Math/cos 3.0)
                                        b (Math/sin a)]
                                       b)
                                #{'x}))))))

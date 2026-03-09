(ns raster.compiler.ir-test
  (:require [clojure.test :refer [deftest testing is are]]
            [raster.compiler.ir.core :as ir]))

;; ================================================================
;; Round-trip: sexp -> IR -> sexp = identity
;; ================================================================

(deftest literal-round-trip-test
  (testing "numeric literals"
    (is (= 42 (ir/ir->sexp (ir/sexp->ir 42))))
    (is (= 3.14 (ir/ir->sexp (ir/sexp->ir 3.14)))))
  (testing "string literal"
    (is (= "hello" (ir/ir->sexp (ir/sexp->ir "hello")))))
  (testing "keyword"
    (is (= :foo (ir/ir->sexp (ir/sexp->ir :foo)))))
  (testing "boolean"
    (is (= true (ir/ir->sexp (ir/sexp->ir true))))
    (is (= false (ir/ir->sexp (ir/sexp->ir false)))))
  (testing "nil"
    (is (nil? (ir/ir->sexp (ir/sexp->ir nil))))))

(deftest symbol-round-trip-test
  (testing "simple symbol"
    (is (= 'x (ir/ir->sexp (ir/sexp->ir 'x)))))
  (testing "qualified symbol"
    (is (= 'raster.numeric/+ (ir/ir->sexp (ir/sexp->ir 'raster.numeric/+))))))

(deftest call-round-trip-test
  (testing "simple call"
    (is (= '(+ x y) (ir/ir->sexp (ir/sexp->ir '(+ x y))))))
  (testing "nested call"
    (is (= '(+ (* a b) (- c d))
           (ir/ir->sexp (ir/sexp->ir '(+ (* a b) (- c d)))))))
  (testing "qualified call"
    (is (= '(Math/sin x) (ir/ir->sexp (ir/sexp->ir '(Math/sin x)))))))

(deftest let-round-trip-test
  (testing "let*"
    (is (= '(let* [a 1 b 2] (+ a b))
           (ir/ir->sexp (ir/sexp->ir '(let* [a 1 b 2] (+ a b)))))))
  (testing "nested let"
    (is (= '(let* [a 1] (let* [b 2] (+ a b)))
           (ir/ir->sexp (ir/sexp->ir '(let* [a 1] (let* [b 2] (+ a b)))))))))

(deftest if-round-trip-test
  (testing "if with else"
    (is (= '(if p x y) (ir/ir->sexp (ir/sexp->ir '(if p x y))))))
  (testing "if without else"
    (is (= '(if p x) (ir/ir->sexp (ir/sexp->ir '(if p x)))))))

(deftest do-round-trip-test
  (testing "do form"
    (is (= '(do a b c) (ir/ir->sexp (ir/sexp->ir '(do a b c)))))))

(deftest invk-round-trip-test
  (testing ".invk call"
    (is (= '(.invk some-impl x y)
           (ir/ir->sexp (ir/sexp->ir '(.invk some-impl x y)))))))

(deftest fn-round-trip-test
  (testing "anonymous fn"
    (is (= '(fn ([x] (* x x)))
           (ir/ir->sexp (ir/sexp->ir '(fn [x] (* x x)))))))
  (testing "named fn"
    (is (= '(fn my-fn ([x] (* x x)))
           (ir/ir->sexp (ir/sexp->ir '(fn my-fn [x] (* x x))))))))

(deftest recur-round-trip-test
  (testing "recur"
    (is (= '(recur (+ i 1) acc)
           (ir/ir->sexp (ir/sexp->ir '(recur (+ i 1) acc)))))))

(deftest throw-round-trip-test
  (testing "throw"
    (is (= '(throw (new Exception "err"))
           (ir/ir->sexp (ir/sexp->ir '(throw (new Exception "err"))))))))

(deftest new-round-trip-test
  (testing "new"
    (is (= '(new String "hello")
           (ir/ir->sexp (ir/sexp->ir '(new String "hello")))))))

(deftest quote-round-trip-test
  (testing "quote"
    (is (= '(quote foo) (ir/ir->sexp (ir/sexp->ir '(quote foo)))))))

(deftest vector-round-trip-test
  (testing "vector"
    (is (= [1 2 3] (ir/ir->sexp (ir/sexp->ir [1 2 3]))))
    (is (= ['x '(+ 1 2)] (ir/ir->sexp (ir/sexp->ir ['x '(+ 1 2)]))))))

(deftest map-round-trip-test
  (testing "map"
    (is (= {:a 1 :b 2} (ir/ir->sexp (ir/sexp->ir {:a 1 :b 2}))))))

(deftest set-round-trip-test
  (testing "set"
    (is (= #{1 2 3} (ir/ir->sexp (ir/sexp->ir #{1 2 3}))))))

;; ================================================================
;; Complex walked-form round-trips
;; ================================================================

(deftest complex-round-trip-test
  (testing "Lorenz-like body"
    (let [form '(let* [dx (* sigma (- y x))
                       dy (- (* x (- rho z)) y)
                       dz (- (* x y) (* beta z))]
                      (do (aset du 0 dx)
                          (aset du 1 dy)
                          (aset du 2 dz)))]
      (is (= form (ir/ir->sexp (ir/sexp->ir form))))))

  (testing "if with arithmetic"
    (let [form '(if (< x 0.0)
                  (- x)
                  (* x x))]
      (is (= form (ir/ir->sexp (ir/sexp->ir form))))))

  (testing "loop with recur"
    (let [form '(loop [i 0 acc 0.0]
                  (if (< i n)
                    (recur (+ i 1) (+ acc (aget arr i)))
                    acc))]
      (is (= form (ir/ir->sexp (ir/sexp->ir form)))))))

;; ================================================================
;; IR node types
;; ================================================================

(deftest node-types-test
  (testing "literal creates TLiteral"
    (is (instance? raster.compiler.ir.core.TLiteral (ir/sexp->ir 42))))
  (testing "symbol creates TLocal"
    (is (instance? raster.compiler.ir.core.TLocal (ir/sexp->ir 'x))))
  (testing "call creates TCall"
    (is (instance? raster.compiler.ir.core.TCall (ir/sexp->ir '(+ 1 2)))))
  (testing "let creates TLet"
    (is (instance? raster.compiler.ir.core.TLet (ir/sexp->ir '(let* [a 1] a)))))
  (testing "if creates TIf"
    (is (instance? raster.compiler.ir.core.TIf (ir/sexp->ir '(if true 1 2)))))
  (testing ".invk creates TInvk"
    (is (instance? raster.compiler.ir.core.TInvk (ir/sexp->ir '(.invk impl x))))))

;; ================================================================
;; Side-table operations
;; ================================================================

(deftest side-table-test
  (ir/with-fresh-tables
    (testing "type table"
      (let [node (ir/sexp->ir 'x)]
        (ir/set-type! node 'double)
        (is (= 'double (ir/get-type node)))))
    (testing "purity table"
      (let [node (ir/sexp->ir '(+ 1 2))]
        (ir/set-purity! node :pure)
        (is (= :pure (ir/get-purity node)))))
    (testing "const table"
      (let [node (ir/sexp->ir 42)]
        (ir/set-const! node 42)
        (is (= 42 (ir/get-const node)))))))

;; ================================================================
;; Walk-ir transformation
;; ================================================================

(deftest walk-ir-test
  (testing "identity walk preserves structure"
    (let [form '(let* [a 1] (+ a 2))
          ir-node (ir/sexp->ir form)
          walked (ir/walk-ir identity ir-node)]
      (is (= form (ir/ir->sexp walked)))))
  (testing "transform all literals"
    (let [form '(+ 1 2)
          ir-node (ir/sexp->ir form)
          doubled (ir/walk-ir
                   (fn [node]
                     (if (and (instance? raster.compiler.ir.core.TLiteral node)
                              (number? (:value node)))
                       (assoc node :value (* 2 (:value node)))
                       node))
                   ir-node)]
      (is (= '(+ 2 4) (ir/ir->sexp doubled))))))

;; ================================================================
;; Collect locals
;; ================================================================

(deftest collect-locals-test
  (testing "simple expression"
    (is (= #{'x 'y} (ir/collect-locals (ir/sexp->ir '(+ x y))))))
  (testing "let binds remove from free vars"
    (is (= #{'y} (ir/collect-locals (ir/sexp->ir '(let* [x 1] (+ x y)))))))
  (testing "nested scope"
    (is (= #{'z} (ir/collect-locals
                  (ir/sexp->ir '(let* [x 1] (let* [y 2] (+ x y z)))))))))

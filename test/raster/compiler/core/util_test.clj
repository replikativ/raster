(ns raster.compiler.core.util-test
  "Tests for free variable analysis — correctness of scoping across all binding forms."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.core.util :as util]))

;; ================================================================
;; free-syms: scoped free variable collection
;; ================================================================

(deftest free-syms-simple-test
  (testing "bare symbols"
    (is (= #{'x} (util/free-syms 'x)))
    (is (= #{} (util/free-syms 42)))
    (is (= #{} (util/free-syms 'clojure.core/+))))
  (testing "simple call — clojure.core vars are not free"
    (is (= #{'x 'y} (util/free-syms '(+ x y)))
        "bare + resolves to clojure.core/+, not a local")
    (is (= #{'x 'y} (util/free-syms '(clojure.core/+ x y)))
        "qualified head is not a local variable")
    (is (= #{'my-fn 'x} (util/free-syms '(my-fn x)))
        "unknown bare symbol is free (could be a local fn)"))
  (testing "dot-prefixed symbols are method calls, not variables"
    (is (= #{'x 'y} (util/free-syms '(.add x y)))
        ".add is a method call, not a free variable")
    (is (= #{} (util/free-syms '.invk))
        "bare .invk is a method symbol, not free"))
  (testing "cast intrinsics are not free"
    (is (= #{} (util/free-syms 'double))
        "double is clojure.core/double, not a local")
    (is (= #{} (util/free-syms 'long))
        "long is clojure.core/long, not a local")
    (is (= #{'x} (util/free-syms '(double x)))
        "only x is free in (double x)")))

(deftest free-syms-let-binding-test
  (testing "let* binding scopes correctly"
    (is (= #{'x} (util/free-syms '(let* [a x] a)))
        "a is bound, x is free")
    (is (= #{'x 'y} (util/free-syms '(let* [a x b y] (+ a b))))
        "a and b are bound; x and y are free; + is clojure.core"))
  (testing "sequential binding visibility"
    (is (= #{'x} (util/free-syms '(let* [a x b a] b)))
        "second binding can see first; only x is free")
    (is (= #{'x} (util/free-syms '(let* [a x b (+ a 1)] b)))
        "init of b sees a from prior binding; + is clojure.core")))

(deftest free-syms-loop-test
  (testing "loop* scopes vars into body"
    (is (= #{'n} (util/free-syms '(loop* [i 0] (if (< i n) (recur (+ i 1)) i))))
        "i is bound by loop; n is free; <, + are clojure.core")))

(deftest free-syms-dotimes-test
  (testing "dotimes scopes index"
    (is (= #{'n 'arr 'v} (util/free-syms '(dotimes [i n] (aset arr i v))))
        "i is bound, n/arr/v are free; aset is clojure.core")))

(deftest free-syms-fn-test
  (testing "fn* params are local"
    (is (= #{'y} (util/free-syms '(fn* [x] (+ x y))))
        "x is param (bound), y is free; + is clojure.core")
    (is (= #{} (util/free-syms '(fn* [x y] (+ x y))))
        "x y bound by params, + is clojure.core")))

(deftest free-syms-try-catch-test
  (testing "catch binds exception variable"
    (is (= #{'foo 'x 'bar} (util/free-syms (list 'try '(foo x) (list 'catch 'Exception 'e '(bar e)))))
        "x/foo/bar are free, e is bound by catch")))

(deftest free-syms-nested-test
  (testing "nested let + fn"
    (is (= #{'outer} (util/free-syms '(let* [a outer] (fn* [x] (+ a x)))))
        "outer is free, a bound by let, x by fn, + is clojure.core"))
  (testing "deeply nested"
    (is (= #{'z} (util/free-syms
                  (list 'let* '[a 1]
                        (list 'let* '[b a]
                              (list 'fn* '[x]
                                    (list 'loop* '[i 0]
                                          '(if (< i x) (recur (+ i z)) b)))))))
        "z is free — a, b bound by let, x by fn, i by loop; <, + are clojure.core")))

;; ================================================================
;; free-syms: par form scoping
;; ================================================================

(deftest free-syms-par-map-test
  (testing "par/map! scopes idx into body"
    (let [result (util/free-syms '(raster.par/map! out i n double (+ (aget a i) 1.0)))]
      (is (contains? result 'out) "out is free")
      (is (contains? result 'n) "bound expr is free")
      (is (contains? result 'a) "array in body is free")
      (is (not (contains? result 'i)) "idx is scoped — not free")
      (is (not (contains? result 'double)) "double is clojure.core, not free")
      (is (not (contains? result '+)) "+ is clojure.core, not free")
      (is (not (contains? result 'aget)) "aget is clojure.core, not free"))))

(deftest free-syms-par-reduce-test
  (testing "par/reduce scopes acc and idx into body"
    (is (= #{'a 'n}
           (util/free-syms '(raster.par/reduce acc 0.0 i n (+ acc (aget a i)))))
        "acc and i are scoped; a, n free; +, aget are clojure.core")))

(deftest free-syms-par-map-void-test
  (testing "par/map-void! scopes idx into body"
    (is (= #{'n 'arr 'v}
           (util/free-syms '(raster.par/map-void! i n (aset arr i v))))
        "i is scoped, n/arr/v are free; aset is clojure.core")))


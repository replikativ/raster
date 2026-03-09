(ns raster.numeric-test
  (:refer-clojure :exclude [+ - * / zero? min max abs])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.numeric :refer [+ - * / zero one similar abs sign clamp
                                    min max zero?
                                    add! sub! scale! axpy!]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.inference :refer [register-typed-macro!]]))

;; ================================================================
;; Polymorphic arithmetic
;; ================================================================

(deftest addition-test
  (testing "long + long"
    (is (= 3 (+ 1 2))))
  (testing "double + double"
    (is (= 3.0 (+ 1.0 2.0))))
  (testing "mixed long + double"
    (is (= 3.0 (+ 1 2.0)))
    (is (= 3.0 (+ 1.0 2)))))

(deftest subtraction-test
  (testing "unary negation"
    (is (= -5 (- 5)))
    (is (= -3.0 (- 3.0))))
  (testing "binary subtraction"
    (is (= 3 (- 10 7)))
    (is (= 0.5 (- 1.5 1.0)))))

(deftest multiplication-test
  (testing "long * long"
    (is (= 12 (* 3 4))))
  (testing "double * double"
    (is (= 6.0 (* 2.0 3.0))))
  (testing "mixed"
    (is (= 6.0 (* 2 3.0)))))

(deftest division-test
  (testing "double / double"
    (is (< (Math/abs (clojure.core/- (/ 10.0 3.0) 3.333333333)) 0.001))))

;; ================================================================
;; Variadic arithmetic (reducible)
;; ================================================================

(deftest variadic-addition-test
  (testing "3-arg +"
    (is (= 6 (+ 1 2 3)))
    (is (= 6.0 (+ 1.0 2.0 3.0))))
  (testing "4-arg +"
    (is (= 10 (+ 1 2 3 4))))
  (testing "5-arg +"
    (is (= 15 (+ 1 2 3 4 5))))
  (testing "mixed types fold correctly"
    (is (= 6.0 (+ 1 2.0 3)))))

(deftest variadic-multiplication-test
  (testing "3-arg *"
    (is (= 24 (* 2 3 4)))
    (is (= 24.0 (* 2.0 3.0 4.0))))
  (testing "4-arg *"
    (is (= 120 (* 2 3 4 5)))))

;; ================================================================
;; Generic constructors
;; ================================================================

(deftest zero-test
  (testing "scalar zeros"
    (is (= 0 (zero 42)))
    (is (= 0.0 (zero 3.14))))
  (testing "array zero"
    (let [a (double-array [1 2 3])
          z (zero a)]
      (is (= 3 (alength z)))
      (is (every? #(== 0.0 %) (seq z))))))

(deftest one-test
  (is (= 1 (one 42)))
  (is (= 1.0 (one 3.14))))

(deftest similar-test
  (testing "creates array of same size"
    (let [a (double-array [1 2 3])
          s (similar a)]
      (is (= 3 (alength s)))
      (is (not (identical? a s))))))

;; ================================================================
;; Array arithmetic (allocating)
;; ================================================================

(deftest array-addition-test
  (testing "array + array"
    (is (= [5.0 7.0 9.0]
           (vec (+ (double-array [1 2 3]) (double-array [4 5 6])))))))

(deftest array-subtraction-test
  (testing "array - array"
    (is (= [3.0 3.0 3.0]
           (vec (- (double-array [4 5 6]) (double-array [1 2 3]))))))
  (testing "negate array"
    (is (= [-1.0 -2.0 -3.0]
           (vec (- (double-array [1 2 3])))))))

(deftest array-scalar-mul-test
  (testing "scalar * array"
    (is (= [2.0 4.0 6.0]
           (vec (* 2.0 (double-array [1 2 3]))))))
  (testing "array * scalar"
    (is (= [3.0 6.0 9.0]
           (vec (* (double-array [1 2 3]) 3.0))))))

(deftest array-scalar-div-test
  (testing "array / scalar"
    (is (= [2.0 4.0 6.0]
           (vec (/ (double-array [10 20 30]) 5.0))))))

;; ================================================================
;; In-place array operations
;; ================================================================

(deftest in-place-ops-test
  (testing "add!"
    (is (= [5.0 7.0 9.0]
           (vec (add! (double-array 3) (double-array [1 2 3]) (double-array [4 5 6]))))))
  (testing "sub!"
    (is (= [3.0 3.0 3.0]
           (vec (sub! (double-array 3) (double-array [4 5 6]) (double-array [1 2 3]))))))
  (testing "scale!"
    (is (= [2.0 4.0 6.0]
           (vec (scale! (double-array 3) 2.0 (double-array [1 2 3]))))))
  (testing "axpy!"
    (is (= [9.0 12.0 15.0]
           (vec (axpy! (double-array 3) (double-array [1 2 3]) 2.0 (double-array [4 5 6])))))))

;; ================================================================
;; broadcast and muladd in deftm
;; ================================================================

(deftm euler-step [u :- (Array double), k :- (Array double), dt :- Double]
  :- (Array double)
  (broadcast [u k] (+ u (* dt k))))

(deftest broadcast-test
  (testing "euler step via broadcast"
    (let [u (double-array [1.0 2.0 3.0])
          k (double-array [-0.1 -0.2 -0.3])
          result (euler-step u k 0.5)]
      (is (= [0.95 1.9 2.85] (vec result))))))

(deftm fma-step [u :- (Array double), k :- (Array double), dt :- Double]
  :- (Array double)
  (muladd (broadcast [u k] (+ u (* dt k)))))

(deftest muladd-broadcast-test
  (testing "broadcast + muladd combined"
    (let [u (double-array [1.0 2.0 3.0])
          k (double-array [-0.1 -0.2 -0.3])
          result (fma-step u k 0.5)]
      (is (= [0.95 1.9 2.85] (vec result))))))

(deftm long-axpy [a :- (Array long), b :- (Array long), c :- Long]
  :- (Array long)
  (broadcast [a b] (+ a (* c b))))

(deftest broadcast-long-array-test
  (testing "broadcast with long arrays uses long cast"
    (is (= [21 42 63]
           (vec (long-axpy (long-array [1 2 3]) (long-array [10 20 30]) 2))))))

(deftm muladd-scalar [a :- Double, b :- Double, c :- Double, d :- Double] :- Double
  (muladd (+ a (* b c) (* d c))))

(deftest muladd-scalar-test
  (testing "multi-arg muladd folds to nested fma"
    ;; 1.0 + 2.0*3.0 + 4.0*3.0 = 1 + 6 + 12 = 19
    (is (== 19.0 (muladd-scalar 1.0 2.0 3.0 4.0)))))

;; ================================================================
;; User-defined typed macros
;; ================================================================

;; Custom typed macro: negate-arrays! — negate all arrays in scope
;; Expand-fn receives type-env: {sym → {:tag, :fn-info, :element}}
(register-typed-macro! 'negate-arrays!
                       (fn [form type-env]
                         (let [[_ & syms] form
                               array-tags #{'doubles 'longs 'ints 'floats}
                               array-syms (filter #(contains? array-tags (get-in type-env [% :tag])) syms)
                               n-sym (gensym "n__")
                               i-sym (gensym "i__")
                               first-arr (first array-syms)]
                           (list 'let [n-sym (list 'alength first-arr)]
                                 (list 'dotimes [i-sym n-sym]
                                       (cons 'do
                                             (for [s array-syms]
                                               (list 'aset s i-sym (list 'double (list '- (list 'aget s i-sym)))))))
                                 first-arr))))

(deftm negate-both! [a :- (Array double), b :- (Array double)] :- (Array double)
  (negate-arrays! a b))

(deftest custom-typed-macro-test
  (testing "user-defined typed macro expands with type env"
    (let [a (double-array [1 2 3])
          b (double-array [4 5 6])
          result (negate-both! a b)]
      (is (= [-1.0 -2.0 -3.0] (vec result)))
      (is (= [-4.0 -5.0 -6.0] (vec b))))))

;; ================================================================
;; reduce! typed macro
;; ================================================================

(deftm dot-product [a :- (Array double), b :- (Array double)] :- Double
  (reduce! [acc 0.0] [a b] (+ acc (* a b))))

(deftm weighted-sum-of-squares [a :- (Array double), w :- (Array double)] :- Double
  (muladd (reduce! [acc 0.0] [a w] (+ acc (* w a a)))))

(deftest reduce-typed-macro-test
  (testing "reduce! dot product"
    (is (== 32.0 (dot-product (double-array [1 2 3]) (double-array [4 5 6])))))
  (testing "reduce! with muladd"
    ;; 2*1^2 + 3*2^2 + 4*3^2 = 2+12+36 = 50
    (is (== 50.0 (weighted-sum-of-squares (double-array [1 2 3]) (double-array [2 3 4]))))))

;; ================================================================
;; Strided/view broadcast (offset + length)
;; ================================================================

(deftm add-subrange [a :- (Array double),
                     b :- (Array double), off :- Long, len :- Long]
  :- (Array double)
  (broadcast [a b] :offset off :length len (+ a b)))

(deftm dot-subrange [a :- (Array double), b :- (Array double),
                     off :- Long, len :- Long]
  :- Double
  (reduce! [acc 0.0] [a b] :offset off :length len (+ acc (* a b))))

(deftest strided-broadcast-test
  (testing "broadcast with offset and length"
    (let [a   (double-array [0 0 0 1 2 3 0 0 0 0])
          b   (double-array [0 0 0 4 5 6 0 0 0 0])
          result (add-subrange a b 3 3)]
      ;; Only elements 3-5 should be written
      (is (= [0.0 0.0 0.0 5.0 7.0 9.0 0.0 0.0 0.0 0.0] (vec result)))))
  (testing "reduce! with offset and length"
    ;; dot product of elements 2-4: 3*6 + 4*7 + 5*8 = 18+28+40 = 86
    (let [a (double-array [1 2 3 4 5 6 7])
          b (double-array [4 5 6 7 8 9 0])]
      (is (== 86.0 (dot-subrange a b 2 3))))))

;; ================================================================
;; Number tower: abs, sign, clamp, min, max, zero?
;; ================================================================

(deftest abs-test
  (testing "Long abs"
    (is (= 5 (abs -5)))
    (is (= 5 (abs 5)))
    (is (= 0 (abs 0))))
  (testing "Double abs"
    (is (= 3.14 (abs -3.14)))
    (is (= 3.14 (abs 3.14)))))

(deftest sign-test
  (testing "Long sign"
    (is (= 1 (sign 42)))
    (is (= -1 (sign -42)))
    (is (= 0 (sign 0))))
  (testing "Double sign"
    (is (= 1.0 (sign 3.14)))
    (is (= -1.0 (sign -3.14)))
    (is (= 0.0 (sign 0.0)))))

(deftest clamp-test
  (testing "Long clamp"
    (is (= 5 (clamp 3 5 10)))
    (is (= 7 (clamp 7 5 10)))
    (is (= 10 (clamp 15 5 10))))
  (testing "Double clamp"
    (is (= 0.0 (clamp -1.0 0.0 1.0)))
    (is (= 0.5 (clamp 0.5 0.0 1.0)))
    (is (= 1.0 (clamp 2.0 0.0 1.0)))))

(deftest min-max-test
  (testing "Long min/max"
    (is (= 3 (min 3 7)))
    (is (= 7 (max 3 7))))
  (testing "Double min/max"
    (is (= 1.5 (min 1.5 2.5)))
    (is (= 2.5 (max 1.5 2.5)))))

(deftest zero-pred-test
  (testing "Long zero?"
    (is (zero? 0))
    (is (not (zero? 1))))
  (testing "Double zero?"
    (is (zero? 0.0))
    (is (not (zero? 1.0)))))

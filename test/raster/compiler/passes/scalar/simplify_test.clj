(ns raster.compiler.passes.scalar.simplify-test
  "Strict raw simplification: purity is not numerical or dispatch evidence."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.simplify :as simp]))

(deftest unknown-operands-retain-numerical-and-dispatch-boundaries
  (doseq [form '[(+ x 0) (+ x 0.0) (+ 0 x) (+ 0.0 x)
                (- x 0) (- x 0.0) (- x x) (- x)
                (* x 0) (* x 0.0) (* 0 x) (* 0.0 x)
                (* x 1) (* x 1.0) (* 1 x) (* 1.0 x) (* x)
                (/ x 1) (/ x 1.0) (/ 0 x) (/ 0.0 x)
                (Math/pow x 0) (Math/pow x 1) (Math/pow x 2.0)
                (Math/pow x 0.5) (Math/pow x -1.0)
                (raster.numeric/+ 1 2) (raster.numeric/* 1.0 2.0)
                (raster.numeric// 1 2) (raster.numeric/pow 2.0 3.0)]
          simplify [simp/simplify-1 simp/simplify simp/simplify-derivative]]
    (is (= form (simplify form)) (str "No type/value witness for " form))))

(defn- same-result? [a b]
  (and (= (class a) (class b))
       (cond
         (instance? Double a)
         (or (and (Double/isNaN a) (Double/isNaN b))
             (= (Double/doubleToRawLongBits a) (Double/doubleToRawLongBits b)))
         (instance? Float a)
         (or (and (Float/isNaN a) (Float/isNaN b))
             (= (Float/floatToRawIntBits a) (Float/floatToRawIntBits b)))
         :else (= a b))))

(deftest literal-core-arithmetic-preserves-carrier-and-value
  (doseq [op ['clojure.core/+ 'clojure.core/- 'clojure.core/* 'clojure.core//]
          [a b] [[2 3] [2.0 3.0] [2 3.0] [1/2 3/4]
                 [-0.0 1.0] [-0.0 -1.0] [-0.0 -0.0]
                 [Double/NaN 0.0] [Double/POSITIVE_INFINITY 0.0]
                 [(float -0.0) (float 1.0)] [(float 2.0) (float 3.0)]]
          :when (not (and (= op 'clojure.core//) (zero? b)))]
    (let [form (list op a b)
          expected (apply (ns-resolve 'clojure.core op) [a b])]
      (is (same-result? expected (simp/simplify-1 form)) (str form))))
  (is (same-result? 1/2 (simp/simplify-1 '(clojure.core// 1 2))))
  (is (same-result? 2 (simp/simplify-1 '(clojure.core// 6 3))))
  (is (same-result? -0.0 (simp/simplify-1 '(clojure.core/- 0.0))))
  (is (= '(clojure.core// 1 0) (simp/simplify-1 '(clojure.core// 1 0)))))

(deftest exceptional-values-survive-symbolic-simplification
  (doseq [form '[(* x 0.0) (* 0.0 x) (- x x) (+ x 0.0)
                (+ 0.0 x) (/ 0.0 x) (Math/pow x 0.5)]
          x [Double/NaN Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY
             -0.0 0.0 -2.0 2.0 (float -0.0) Float/NaN (float 2.0)]
          simplify [simp/simplify-1 simp/simplify simp/simplify-derivative]]
    (let [run (fn [body] ((eval (list 'fn '[x] body)) x))]
      (is (same-result? (run form) (run (simplify form))) (str form " at " x)))))

(deftest checked-literal-traps-retain-source-order
  (doseq [trap '[(clojure.core/- -9223372036854775808)
                (clojure.core/+ 9223372036854775807 1)
                (clojure.core/* 9223372036854775807 2)
                (clojure.core// 1 0)]]
    (is (= trap (simp/simplify-1 trap)))
    (let [form (list 'do '(swap! counter inc) trap)
          counter (atom 0)
          run (eval (list 'fn '[counter] (simp/simplify form)))]
      (is (thrown? ArithmeticException (run counter)))
      (is (= 1 @counter) "The preceding observable write still occurs before the trap"))))

(deftest math-literals-and-recursive-folding
  (doseq [form '[(Math/pow -0.0 0.5) (Math/pow 2.0 3.0)
                (Math/sin 0.0) (Math/cos 0.0) (Math/exp 1.0)
                (Math/sqrt 9.0) (Math/sqrt -0.0) (Math/abs -5.0)
                (Math/min -0.0 0.0) (Math/max -0.0 0.0)
                (Math/atan2 -0.0 -1.0)]]
    (is (same-result? (eval form) (simp/simplify-1 form)) (str form)))
  (is (= 24.0 (simp/simplify '(clojure.core/* (clojure.core/+ 3.0 5.0)
                                                          (clojure.core/- 4.0 1.0)))))
  (is (= '(let* [a 7.0] (+ a 0))
         (simp/simplify '(let* [a (clojure.core/+ 3.0 4.0)] (+ a 0))))))

(deftest unresolved-heads-and-overloads-are-not-guessed
  (doseq [form '[(+ 1 2) (/ 1 2) (Math/min 1 2) (Math/max 1 2)
                (Math/abs -1) (Math/abs -9223372036854775808)]]
    (is (= form (simp/simplify-1 form))))
  (let [form '(let [+ (fn [a b] 99)] (+ 1 2))]
    (is (= 99 (eval (simp/simplify form)))))
  (is (= 'x (simp/simplify-symbolic-derivative '(+ (* 1 x) (* 0 y)))))
  (is (= '(+ (* 1 x) (* 0 y)) (simp/simplify-derivative '(+ (* 1 x) (* 0 y))))))

(deftest checked-effects-and-selected-calls-survive
  (doseq [form '[(* (clojure.core/int x) 0) (* 0 (clojure.core/int x))
                (- (clojure.core/int x) (clojure.core/int x))
                (/ 0 (clojure.core/int x))
                (Math/pow (clojure.core/int x) 0)
                (Math/pow (clojure.core/int x) 2)
                (* (swap! counter inc) 0)
                (.invk some_custom_fn x y)
                (.invk some_star__m_double_double-impl x 1)]]
    (is (= form (simp/simplify form))))
  (let [form (with-meta '(.invk impl__plus x 0)
               {:raster.op/original 'raster.numeric/+})
        result (simp/simplify form)]
    (is (= form result))
    (is (= (meta form) (meta result)))))

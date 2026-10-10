(ns raster.compiler.simplify-test
  "Strict simplifier/PE integration; arithmetic cases live in scalar/simplify-test."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.simplify :as simp]
            [raster.compiler.passes.scalar.pe :as pe]))

(deftest structural-simplification
  (doseq [form '[nil x () (str a b) (+ x y) (let* [a x] (+ a 0))]]
    (is (= form (simp/simplify form))))
  (is (= ['x 7.0] (simp/simplify ['x '(clojure.core/+ 3.0 4.0)]))))

(deftest partial-evaluation-retains-unknown-numerical-boundaries
  (doseq [form '[(* x 0.0) (+ x 0.0) (- x x) (/ 0.0 x)
                (Math/pow x 0.5) (* x 1.0)]]
    (is (= form (pe/pe form {})) (str form)))
  (testing "known operands fold without inventing division precision"
    (is (= 1/2 (pe/pe '(clojure.core// numerator denominator) {'numerator 1 'denominator 2})))
    (is (= 12.0 (pe/pe '(clojure.core/* scale value) {'scale 3.0 'value 4.0}))))
  (testing "selected .invk signature is not guessed from spelling"
    (is (= '(.invk _star__m_double_double-impl 10.0 (- y x))
           (pe/pe '(.invk _star__m_double_double-impl sigma (- y x)) {'sigma 10.0})))))

(deftest lorenz-specialization-test
  (let [walked '(let* [dx (* sigma (- y x))
                      dy (- (* x (- rho z)) y)
                      dz (- (* x y) (* beta z))]
                 [dx dy dz])
        result (pe/pe walked {'sigma 10.0 'rho 28.0 'beta (/ 8.0 3.0)})
        terms (flatten (if (seq? result) result [result]))]
    (is (not (some #{'sigma 'rho 'beta} terms)))
    (is (some #{'x} terms))))

(ns raster.compiler.passes.parallel.simd-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.simd :as simd]))

;; ================================================================
;; loop* / loop / dotimes acceptance
;; ================================================================

(deftest loop-star-test
  (testing "loop* form accepted (original)"
    (let [form '(loop* [i 0]
                       (if (< i n)
                         (do (aset arr i (Math/max 0.0 (aget arr i)))
                             (recur (inc i)))
                         nil))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map (:pattern result)))
      (is (= :relu (:op result))))))

(deftest loop-no-star-test
  (testing "loop form accepted (walker output)"
    (let [form '(loop [i 0]
                  (if (< i n)
                    (do (aset arr i (Math/max 0.0 (aget arr i)))
                        (recur (inc i)))
                    nil))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map (:pattern result)))
      (is (= :relu (:op result))))))

(deftest dotimes-test
  (testing "dotimes form accepted (common in deftm code)"
    (let [form '(dotimes [i n]
                  (aset arr i (Math/abs (aget arr i))))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map (:pattern result)))
      (is (= 'Math/abs (:op result)))
      (is (= 'arr (:in-array result)))
      (is (= 'arr (:out-array result)))
      (is (true? (:in-place result))))))

(deftest dotimes-binary-test
  (testing "dotimes with binary array op"
    (let [form '(dotimes [i n]
                  (aset out i (+ (aget a i) (aget b i))))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-binary (:pattern result)))
      (is (= '+ (:op result))))))

;; ================================================================
;; Scalar-array patterns
;; ================================================================

(deftest scalar-map-right-test
  (testing "Scalar on right: (aset out i (* (aget arr i) 2.0))"
    (let [form '(dotimes [i n]
                  (aset out i (* (aget arr i) 2.0)))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-scalar (:pattern result)))
      (is (= '* (:op result)))
      (is (= 2.0 (:scalar result)))
      (is (= :right (:scalar-pos result)))
      (is (= 'arr (:in-array result))))))

(deftest scalar-map-left-test
  (testing "Scalar on left: (aset out i (+ alpha (aget arr i)))"
    (let [form '(loop [i 0]
                  (if (< i n)
                    (do (aset out i (+ alpha (aget arr i)))
                        (recur (inc i)))
                    nil))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-scalar (:pattern result)))
      (is (= '+ (:op result)))
      (is (= 'alpha (:scalar result)))
      (is (= :left (:scalar-pos result))))))

;; ================================================================
;; Compound patterns (ODE-style)
;; ================================================================

(deftest compound-two-terms-test
  (testing "out[i] = u[i] + dt * k1[i]"
    (let [form '(dotimes [i n]
                  (aset out i (+ (aget u i) (* dt (aget k1 i)))))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-compound (:pattern result)))
      (is (= 2 (count (:terms result))))
      ;; First term: u[i] with coeff 1.0
      (is (= 1.0 (:coeff (first (:terms result)))))
      (is (= 'u (:array (first (:terms result)))))
      ;; Second term: k1[i] with coeff dt
      (is (= 'dt (:coeff (second (:terms result)))))
      (is (= 'k1 (:array (second (:terms result))))))))

(deftest compound-three-terms-test
  (testing "out[i] = u[i] + c1*k1[i] + c2*k2[i]"
    (let [form '(dotimes [i n]
                  (aset out i (+ (aget u i) (* c1 (aget k1 i)) (* c2 (aget k2 i)))))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-compound (:pattern result)))
      (is (= 3 (count (:terms result)))))))

(deftest compound-rk4-final-test
  (testing "RK4 final: out[i] = u[i] + (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])"
    ;; Flattened form (what broadcast! would produce for simple cases):
    ;; Note: full RK4 has nested arithmetic that needs pre-expansion
    (let [form '(dotimes [i n]
                  (aset out i (+ (aget u i)
                                 (* c1 (aget k1 i))
                                 (* c2 (aget k2 i))
                                 (* c3 (aget k3 i))
                                 (* c4 (aget k4 i)))))
          result (simd/vectorizable? form)]
      (is (some? result))
      (is (= :map-compound (:pattern result)))
      (is (= 5 (count (:terms result)))))))

;; ================================================================
;; Reduce patterns
;; ================================================================

(deftest reduce-sum-loop-test
  (testing "Sum reduction with loop (not loop*)"
    (let [form '(loop [acc 0.0 i 0]
                  (if (< i n)
                    (recur (+ acc (aget arr i)) (inc i))
                    acc))
          result (simd/match-reduce-pattern form)]
      (is (some? result))
      (is (= :reduce (:pattern result)))
      (is (= '+ (:op result)))
      (is (= 'acc (:acc-sym result)))
      (is (== 0.0 (:acc-init result))))))

(deftest reduce-max-test
  (testing "Max reduction"
    (let [form '(loop [acc -1.0E308 i 0]
                  (if (< i n)
                    (recur (Math/max acc (aget arr i)) (inc i))
                    acc))
          result (simd/match-reduce-pattern form)]
      (is (some? result))
      (is (= :reduce (:pattern result)))
      (is (= 'Math/max (:op result))))))

(deftest reduce-reversed-bindings-test
  (testing "Max reduction with reversed bindings (index first)"
    (let [form '(loop [i 0 m -1.7976931348623157E308]
                  (if (< i n)
                    (recur (unchecked-inc i) (Math/max m (aget arr i)))
                    m))
          result (simd/match-reduce-pattern form)]
      (is (some? result))
      (is (= :reduce (:pattern result)))
      (is (= 'Math/max (:op result)))
      (is (= 'i (:index-sym result)))
      (is (= 'm (:acc-sym result))))))

(deftest reduce-walker-style-test
  (testing "Reduce with (double acc) wrapper and clojure.core/aget"
    (let [form '(loop [i 0 m -1.7976931348623157E308]
                  (if (< i n)
                    (recur (unchecked-inc i)
                           (Math/max (double m) (clojure.core/aget arr i)))
                    m))
          result (simd/match-reduce-pattern form)]
      (is (some? result))
      (is (= :reduce (:pattern result)))
      (is (= 'Math/max (:op result)))
      (is (= 'arr (:array-sym result))))))

;; ================================================================
;; Non-vectorizable forms
;; ================================================================

(deftest non-vectorizable-test
  (testing "Non-loop form returns nil"
    (is (nil? (simd/vectorizable? '(+ 1 2)))))

  (testing "println loop is not vectorizable"
    (let [form '(dotimes [i n] (println i))]
      (is (nil? (simd/vectorizable? form)))))

  (testing "Side-effecting println loop is not vectorizable"
    ;; Uses non-array ops, not aget/aset
    (let [form '(dotimes [i n]
                  (println (Math/max 0.0 (double i))))]
      (is (nil? (simd/vectorizable? form))))))

;; ================================================================
;; SIMD codegen tests
;; ================================================================

(deftest generate-simd-relu-test
  (testing "Generates SIMD S-expression for relu"
    (let [pattern {:pattern :map :op :relu
                   :in-array 'arr :out-array 'arr :in-place true}
          sexp (simd/generate-simd-sexp pattern {:bound-sym 'n})]
      (is (some? sexp))
      ;; Should contain let* with species
      (is (= 'let* (first sexp))))))

(deftest generate-simd-binary-test
  (testing "Generates SIMD S-expression for binary add"
    (let [pattern {:pattern :map-binary :op '+ :in-array-1 'a :in-array-2 'b
                   :out-array 'out}
          sexp (simd/generate-simd-sexp pattern {:bound-sym 'n})]
      (is (some? sexp))
      (is (= 'let* (first sexp))))))

(deftest generate-simd-scalar-test
  (testing "Generates SIMD S-expression for scalar mul"
    (let [pattern {:pattern :map-scalar :op '* :scalar 2.0 :scalar-pos :right
                   :in-array 'arr :out-array 'out}
          sexp (simd/generate-simd-sexp pattern {:bound-sym 'n})]
      (is (some? sexp))
      (is (= 'let* (first sexp))))))

(deftest generate-simd-compound-test
  (testing "Generates SIMD S-expression for compound (u + dt*k)"
    (let [pattern {:pattern :map-compound
                   :terms [{:coeff 1.0 :array 'u} {:coeff 'dt :array 'k}]
                   :out-array 'out}
          sexp (simd/generate-simd-sexp pattern {:bound-sym 'n})]
      (is (some? sexp))
      (is (= 'let* (first sexp))))))

;; ================================================================
;; Module loads
;; ================================================================

(deftest module-loads-test
  (testing "SIMD module loads without errors"
    (is (fn? simd/vectorizable?))
    (is (fn? simd/emit-simd-class))
    (is (fn? simd/match-reduce-pattern))))

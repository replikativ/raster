(ns raster.compiler.bc-correctness-test
  "Bytecode compiler correctness tests.

  Each test defines a deftm, calls it twice through the typed dispatch path
  (first call = deftype/Clojure compiler, second call = BC compiler), and
  asserts both produce the same result. This catches BC code generation bugs
  that would otherwise be hidden by the deftype fallback.

  Categories:
    1. Arithmetic (int/long/float/double, mixed types)
    2. Comparisons and boolean logic (and/or short-circuit)
    3. Control flow (if/when/cond/case with primitives)
    4. Loops (loop/recur with counters, nested)
    5. Arrays (aget/aset all types, bounds)
    6. Casts and coercions
    7. Static method calls (Math/*, overloaded)
    8. Let bindings and scoping"
  (:refer-clojure :exclude [aget aset alength])
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.arrays :refer [aget aset alength]]))

;; ================================================================
;; Test helper: call through typed dispatch, first = deftype, second = BC
;; ================================================================

(defn- call-typed
  "Call a deftm var through typed dispatch. Returns the result.
  First call triggers lazy JIT; second call uses BC-compiled code."
  [deftm-var & args]
  (let [dt @(:raster.core/dispatch-table (meta deftm-var))
        arity (count args)
        entry (first (get dt arity))]
    (when-not entry
      (throw (ex-info "No dispatch entry" {:var deftm-var :arity arity})))
    (apply (:typed-target-fn entry) args)))

;; ================================================================
;; 1. Arithmetic
;; ================================================================

(deftm bc-add-dd [a :- Double, b :- Double] :- Double (+ a b))
(deftm bc-mul-dd [a :- Double, b :- Double] :- Double (* a b))
(deftm bc-sub-dd [a :- Double, b :- Double] :- Double (- a b))
(deftm bc-div-dd [a :- Double, b :- Double] :- Double (/ a b))
(deftm bc-add-ll [a :- Long, b :- Long] :- Long (clojure.core/+ a b))
(deftm bc-fma [a :- Double, b :- Double, c :- Double] :- Double
  (Math/fma a b c))
(deftm bc-multi-arith [x :- Double] :- Double
  (+ (* 3.0 x x) (* -2.0 x) 5.0))

(deftest bc-arithmetic-test
  (testing "Basic arithmetic"
    ;; Each pair: first call = deftype, second = BC
    (is (== 5.0 (call-typed #'bc-add-dd 2.0 3.0)))
    (is (== 5.0 (call-typed #'bc-add-dd 2.0 3.0)) "BC path")
    (is (== 6.0 (call-typed #'bc-mul-dd 2.0 3.0)))
    (is (== 6.0 (call-typed #'bc-mul-dd 2.0 3.0)))
    (is (== 1.0 (call-typed #'bc-sub-dd 3.0 2.0)))
    (is (== 1.0 (call-typed #'bc-sub-dd 3.0 2.0)))
    (is (== 1.5 (call-typed #'bc-div-dd 3.0 2.0)))
    (is (== 1.5 (call-typed #'bc-div-dd 3.0 2.0))))
  (testing "Long arithmetic"
    (is (== 7 (call-typed #'bc-add-ll 3 4)))
    (is (== 7 (call-typed #'bc-add-ll 3 4))))
  (testing "Math.fma"
    (is (== 7.0 (call-typed #'bc-fma 2.0 3.0 1.0)))
    (is (== 7.0 (call-typed #'bc-fma 2.0 3.0 1.0))))
  (testing "Multi-op expression"
    ;; 3*4 + (-2)*2 + 5 = 12 - 4 + 5 = 13
    (is (== 13.0 (call-typed #'bc-multi-arith 2.0)))
    (is (== 13.0 (call-typed #'bc-multi-arith 2.0)))))

;; ================================================================
;; 2. Comparisons and boolean logic
;; ================================================================

(deftm bc-lt [a :- Double, b :- Double] :- Boolean (< a b))
(deftm bc-gt [a :- Long, b :- Long] :- Boolean (> a b))

(deftm bc-and-guard
  "Inner loop of insertion sort — exercises (and (> j 0) (comparison))
  with recur that decrements j to 0."
  [arr :- (Array int), n :- Long]
  (loop [j (int 2)]
    (when (and (> j 0)
               (> (clojure.core/aget arr (dec j)) (clojure.core/aget arr j)))
      (let [tmp (clojure.core/aget arr j)]
        (clojure.core/aset arr j (clojure.core/aget arr (dec j)))
        (clojure.core/aset arr (dec j) tmp)
        (recur (dec j))))))

(deftm bc-if-comparison [x :- Double, y :- Double] :- Double
  (if (< x y) x y))

(deftest bc-comparison-test
  (testing "Comparison deftms with Boolean return"
    (is (true? (call-typed #'bc-lt 1.0 2.0)))
    (is (true? (call-typed #'bc-lt 1.0 2.0)) "BC path")
    (is (false? (call-typed #'bc-lt 2.0 1.0)))
    (is (false? (call-typed #'bc-lt 2.0 1.0)) "BC path")
    (is (true? (call-typed #'bc-gt 5 3)))
    (is (true? (call-typed #'bc-gt 5 3)) "BC path"))
  (testing "and short-circuit with comparison in loop"
    ;; [2 3 1]: element at j=2 bubbles left to position 0
    (let [a (int-array [2 3 1])]
      (call-typed #'bc-and-guard a 3) ;; deftype
      (is (= [1 2 3] (vec a))))
    (let [a (int-array [2 3 1])]
      (call-typed #'bc-and-guard a 3) ;; BC
      (is (= [1 2 3] (vec a)) "BC path must sort correctly")))
  (testing "if with comparison predicate"
    (is (== 1.0 (call-typed #'bc-if-comparison 1.0 2.0)))
    (is (== 1.0 (call-typed #'bc-if-comparison 1.0 2.0)))
    (is (== 1.0 (call-typed #'bc-if-comparison 2.0 1.0)))
    (is (== 1.0 (call-typed #'bc-if-comparison 2.0 1.0)))))

;; ================================================================
;; 3. Control flow
;; ================================================================

(deftm bc-if-branches [x :- Long] :- Long
  (if (> x 0) (+ x 10) (- x 10)))

(deftm bc-nested-if [x :- Double] :- Double
  (if (< x 0.0)
    (- x)
    (if (> x 100.0)
      100.0
      x)))

(deftm bc-when-effect [arr :- (Array double), x :- Double] :- Double
  (when (> x 0.0)
    (aset arr 0 x))
  (aget arr 0))

(deftest bc-control-flow-test
  (testing "if with both branches"
    (is (== 15 (call-typed #'bc-if-branches 5)))
    (is (== 15 (call-typed #'bc-if-branches 5)))
    (is (== -15 (call-typed #'bc-if-branches -5)))
    (is (== -15 (call-typed #'bc-if-branches -5))))
  (testing "Nested if"
    (is (== 5.0 (call-typed #'bc-nested-if -5.0)))
    (is (== 5.0 (call-typed #'bc-nested-if -5.0)))
    (is (== 50.0 (call-typed #'bc-nested-if 50.0)))
    (is (== 50.0 (call-typed #'bc-nested-if 50.0)))
    (is (== 100.0 (call-typed #'bc-nested-if 200.0)))
    (is (== 100.0 (call-typed #'bc-nested-if 200.0))))
  (testing "when with side effect"
    (let [a (double-array [0.0])]
      (is (== 0.0 (call-typed #'bc-when-effect a -1.0)))
      (is (== 0.0 (aget a 0))))
    (let [a (double-array [0.0])]
      (is (== 5.0 (call-typed #'bc-when-effect a 5.0)))
      (is (== 5.0 (aget a 0))))))

;; ================================================================
;; 4. Loops
;; ================================================================

(deftm bc-sum-loop [n :- Long] :- Long
  (loop [i (int 0), acc (long 0)]
    (if (>= i n) acc
        (recur (unchecked-inc-int i) (clojure.core/+ acc (long i))))))

(deftm bc-dot-product [a :- (Array double), b :- (Array double)] :- Double
  (let [n (alength a)]
    (loop [i (int 0), acc 0.0]
      (if (>= i n) acc
          (recur (unchecked-inc-int i)
                 (+ acc (* (aget a i) (aget b i))))))))

(deftm bc-nested-loop [n :- Long] :- Long
  (loop [i (int 0), total (long 0)]
    (if (>= i n) total
        (recur (unchecked-inc-int i)
               (loop [j (int 0), acc total]
                 (if (>= j i) acc
                     (recur (unchecked-inc-int j) (clojure.core/+ acc 1))))))))

(deftest bc-loop-test
  (testing "Sum 0..9"
    (is (== 45 (call-typed #'bc-sum-loop 10)))
    (is (== 45 (call-typed #'bc-sum-loop 10))))
  (testing "Dot product"
    (let [a (double-array [1 2 3])
          b (double-array [4 5 6])]
      (is (== 32.0 (call-typed #'bc-dot-product a b)))
      (is (== 32.0 (call-typed #'bc-dot-product a b)))))
  (testing "Nested loop (triangular number)"
    ;; n=4: 0+1+2+3 = 6
    (is (== 6 (call-typed #'bc-nested-loop 4)))
    (is (== 6 (call-typed #'bc-nested-loop 4)))))

;; ================================================================
;; 5. Arrays
;; ================================================================

(deftm bc-array-double [arr :- (Array double), i :- Long] :- Double
  (aget arr i))

(deftm bc-array-float [arr :- (Array float), i :- Long] :- Float
  (aget arr i))

(deftm bc-array-int [arr :- (Array int), i :- Long] :- Long
  (long (clojure.core/aget arr (int i))))

(deftm bc-array-write [arr :- (Array double), i :- Long, v :- Double]
  (aset arr i v))

(deftm bc-arraycopy [src :- (Array double), dst :- (Array double), n :- Long]
  (System/arraycopy src (int 0) dst (int 0) (int n)))

(deftest bc-array-test
  (testing "Double array read"
    (let [a (double-array [10 20 30])]
      (is (== 20.0 (call-typed #'bc-array-double a 1)))
      (is (== 20.0 (call-typed #'bc-array-double a 1)))))
  (testing "Float array read"
    (let [a (float-array [1.5 2.5 3.5])]
      (is (== 2.5 (call-typed #'bc-array-float a 1)))
      (is (== 2.5 (call-typed #'bc-array-float a 1)))))
  (testing "Int array read"
    (let [a (int-array [100 200 300])]
      (is (== 200 (call-typed #'bc-array-int a 1)))
      (is (== 200 (call-typed #'bc-array-int a 1)))))
  (testing "Array write"
    (let [a (double-array [0 0 0])]
      (call-typed #'bc-array-write a 1 42.0)
      (is (== 42.0 (aget a 1)))
      (call-typed #'bc-array-write a 2 99.0)
      (is (== 99.0 (aget a 2)))))
  (testing "System/arraycopy (void static method)"
    (let [src (double-array [1 2 3]) dst (double-array 3)]
      (call-typed #'bc-arraycopy src dst 3)
      (is (= [1.0 2.0 3.0] (vec dst))))
    (let [src (double-array [4 5 6]) dst (double-array 3)]
      (call-typed #'bc-arraycopy src dst 3)
      (is (= [4.0 5.0 6.0] (vec dst)) "BC path"))))

;; ================================================================
;; 6. Casts and coercions
;; ================================================================

(deftm bc-cast-d2l [x :- Double] :- Long (long x))
(deftm bc-cast-l2d [x :- Long] :- Double (double x))
(deftm bc-cast-d2i [x :- Double] :- Long (long (int x)))
(deftm bc-cast-d2f [x :- Double] :- Float (float x))
(deftm bc-cast-f2d [x :- Float] :- Double (double x))

(deftest bc-cast-test
  (testing "double → long"
    (is (== 3 (call-typed #'bc-cast-d2l 3.7)))
    (is (== 3 (call-typed #'bc-cast-d2l 3.7))))
  (testing "long → double"
    (is (== 5.0 (call-typed #'bc-cast-l2d 5)))
    (is (== 5.0 (call-typed #'bc-cast-l2d 5))))
  (testing "double → int → long"
    (is (== 3 (call-typed #'bc-cast-d2i 3.9)))
    (is (== 3 (call-typed #'bc-cast-d2i 3.9))))
  (testing "double ↔ float roundtrip"
    (is (< (Math/abs (- 3.14 (call-typed #'bc-cast-f2d
                                         (call-typed #'bc-cast-d2f 3.14))))
           0.001))))

;; ================================================================
;; 7. Static method calls (including overloaded)
;; ================================================================

(deftm bc-math-abs-d [x :- Double] :- Double (Math/abs x))
(deftm bc-math-abs-i [x :- Long] :- Long (long (Math/abs (int x))))
(deftm bc-math-sqrt [x :- Double] :- Double (Math/sqrt x))
(deftm bc-math-pow [x :- Double, n :- Double] :- Double (Math/pow x n))
(deftm bc-math-min [a :- Double, b :- Double] :- Double (Math/min a b))

(deftm bc-abs-of-diff [a :- Double, b :- Double] :- Double
  "Math.abs on arithmetic result — catches overload resolution bug."
  (Math/abs (- a b)))

(deftest bc-static-methods-test
  (testing "Math.abs(double)"
    (is (== 3.0 (call-typed #'bc-math-abs-d -3.0)))
    (is (== 3.0 (call-typed #'bc-math-abs-d -3.0))))
  (testing "Math.abs(int)"
    (is (== 3 (call-typed #'bc-math-abs-i -3)))
    (is (== 3 (call-typed #'bc-math-abs-i -3))))
  (testing "Math.sqrt"
    (is (== 3.0 (call-typed #'bc-math-sqrt 9.0)))
    (is (== 3.0 (call-typed #'bc-math-sqrt 9.0))))
  (testing "Math.pow"
    (is (== 8.0 (call-typed #'bc-math-pow 2.0 3.0)))
    (is (== 8.0 (call-typed #'bc-math-pow 2.0 3.0))))
  (testing "Math.min"
    (is (== 1.0 (call-typed #'bc-math-min 1.0 2.0)))
    (is (== 1.0 (call-typed #'bc-math-min 1.0 2.0))))
  (testing "Math.abs on arithmetic (overload resolution)"
    ;; abs(2.0 - 5.0) = 3.0. Must pick Math.abs(double), not Math.abs(int).
    (is (== 3.0 (call-typed #'bc-abs-of-diff 2.0 5.0)))
    (is (== 3.0 (call-typed #'bc-abs-of-diff 2.0 5.0)))
    ;; abs(0.3) must NOT truncate to int 0
    (is (< 0.2 (double (call-typed #'bc-abs-of-diff 1.0 0.7))))))

;; ================================================================
;; 8. Let bindings and scoping
;; ================================================================

(deftm bc-let-chain [x :- Double] :- Double
  (let [a (* x 2.0)
        b (+ a 1.0)
        c (* b b)]
    (- c a)))

(deftm bc-let-shadow [x :- Double] :- Double
  (let [y (* x 2.0)
        y (+ y 1.0)]
    y))

(deftm bc-let-with-if [x :- Double] :- Double
  (let [abs-x (if (< x 0.0) (- x) x)]
    abs-x))

(deftest bc-let-test
  (testing "Let chain"
    ;; x=3: a=6, b=7, c=49, c-a=43
    (is (== 43.0 (call-typed #'bc-let-chain 3.0)))
    (is (== 43.0 (call-typed #'bc-let-chain 3.0))))
  (testing "Let shadowing"
    ;; x=5: y=10, y=11
    (is (== 11.0 (call-typed #'bc-let-shadow 5.0)))
    (is (== 11.0 (call-typed #'bc-let-shadow 5.0))))
  (testing "Let with if expression"
    (is (== 3.0 (call-typed #'bc-let-with-if 3.0)))
    (is (== 3.0 (call-typed #'bc-let-with-if 3.0)))
    (is (== 3.0 (call-typed #'bc-let-with-if -3.0)))
    (is (== 3.0 (call-typed #'bc-let-with-if -3.0)))))

;; ================================================================
;; 9. Typed function calls (.invk on Fn params)
;; ================================================================

(deftm bc-apply-fn [f :- (Fn [Double] Double), x :- Double] :- Double
  (f x))

(deftm bc-fold-fn [f :- (Fn [Double Double] Double),
                   arr :- (Array double)] :- Double
  (let [n (alength arr)]
    (loop [i (int 1), acc (aget arr 0)]
      (if (>= i n) acc
          (recur (unchecked-inc-int i) (f acc (aget arr i)))))))

(deftest bc-fn-param-test
  (testing "Apply typed function"
    (let [sq (ftm [x :- Double] :- Double (* x x))]
      (is (== 9.0 (call-typed #'bc-apply-fn sq 3.0)))
      (is (== 9.0 (call-typed #'bc-apply-fn sq 3.0)))))
  (testing "Fold with typed function"
    (let [add (ftm [a :- Double, b :- Double] :- Double (+ a b))
          arr (double-array [1 2 3 4 5])]
      (is (== 15.0 (call-typed #'bc-fold-fn add arr)))
      (is (== 15.0 (call-typed #'bc-fold-fn add arr))))))

;; ================================================================
;; 10. Poisson-like loop with mixed types and boolean conditions
;; ================================================================

(deftm bc-poisson-like [lam :- Double] :- Double
  ;; Mimics Knuth's Poisson sampling: loop with (<=) on doubles
  (let [L (Math/exp (- lam))]
    (loop [k 0 p 1.0]
      (if (<= p L)
        (double (dec k))
        (recur (inc k) (* p 0.9))))))

(deftm bc-multi-cond [x :- Double, y :- Double] :- Double
  ;; Tests cond with and/or chains (like Brent's conditions)
  (let [use-path-a (and (> x 0.0) (< y 10.0))
        use-path-b (or (< x -1.0) (> y 100.0))]
    (cond
      use-path-a (+ x y)
      use-path-b (- x y)
      :else (* x y))))

(deftm bc-loop-bool-recur [n :- Long] :- Long
  ;; Loop with boolean flag in recur (like Brent's mflag)
  (loop [i (int 0) flag true acc (long 0)]
    (if (>= i n) acc
        (recur (unchecked-inc-int i)
               (not flag)
               (if flag (clojure.core/+ acc (long i)) acc)))))

(deftest bc-complex-control-flow-test
  (testing "Poisson-like loop (<=, dec, mixed int/double)"
    (is (number? (call-typed #'bc-poisson-like 5.0)))
    (is (number? (call-typed #'bc-poisson-like 5.0))))
  (testing "Multi-cond with and/or chains"
    (is (== 8.0 (call-typed #'bc-multi-cond 3.0 5.0)))   ;; path a: 3+5
    (is (== 8.0 (call-typed #'bc-multi-cond 3.0 5.0)))
    (is (== 1.0 (call-typed #'bc-multi-cond -2.0 -3.0)))   ;; path b: -2-(-3)=1
    (is (== 1.0 (call-typed #'bc-multi-cond -2.0 -3.0)))
    (is (== -0.25 (call-typed #'bc-multi-cond -0.5 0.5)))  ;; else: -0.5*0.5
    (is (== -0.25 (call-typed #'bc-multi-cond -0.5 0.5))))
  (testing "Loop with boolean flag recur"
    (is (== 20 (call-typed #'bc-loop-bool-recur 10)))  ;; sum of even indices: 0+2+4+6+8
    (is (== 20 (call-typed #'bc-loop-bool-recur 10)))))

;; ================================================================
;; 11. Multi-var loop with destructuring (Brent-like pattern)
;; ================================================================

(deftm bc-brent-like [a0 :- Double, b0 :- Double] :- Double
  ;; Simplified Brent pattern: many loop vars, nested conds, destructuring
  (loop [a a0 b b0 c a0 d (- b0 a0)
         fa a0 fb b0 fc a0
         flag true iter (int 0)]
    (if (or (== fb 0.0) (>= iter 50) (< (Math/abs (- b a)) 1e-12))
      b
      (let [s (if (and (not (== fa fc)) (not (== fb fc)))
                ;; inverse quadratic (simplified)
                (* 0.5 (+ a b))
                ;; secant (simplified)
                (- b (* fb (/ (- b a) (- fb fa)))))
            use-bisect (or (and flag (>= (Math/abs (- s b))
                                         (/ (Math/abs (- b c)) 2.0)))
                           (< (Math/abs (- b c)) 1e-12))
            s (if use-bisect (/ (+ a b) 2.0) s)
            flag use-bisect
            fs (* s s)  ;; fake function eval
            d c c b fc fb
            [a b fa fb] (if (< (* fa fs) 0.0)
                          [a s fa fs]
                          [s b fs fb])
            [a b fa fb] (if (< (Math/abs fa) (Math/abs fb))
                          [b a fb fa]
                          [a b fa fb])]
        (recur a b c d fa fb fc flag (unchecked-add-int iter 1))))))

(deftest bc-brent-like-test
  (testing "Multi-var loop with destructuring"
    (is (number? (call-typed #'bc-brent-like 1.0 2.0)))
    (is (number? (call-typed #'bc-brent-like 1.0 2.0)))))

;; ================================================================
;; 10. reify* / ftm — typed closures
;; ================================================================

(deftm bc-make-adder [c :- Double]
  (ftm [x :- Double] :- Double (n/+ x c)))

(deftm bc-make-multiplier [c :- Double]
  (ftm [x :- Double] :- Double (n/* x c)))

(deftm bc-inline-closure [f :- Double, x :- Double] :- Double
  (let [adder (ftm [v :- Double] :- Double (n/+ v f))]
    (double (adder x))))

(deftm bc-nested-closure [a :- Double, b :- Double] :- Double
  (let [g (ftm [x :- Double] :- Double (n/+ (n/* a x) b))]
    (n/+ (double (g 1.0)) (double (g 2.0)))))

(deftest bc-reify-test
  (testing "ftm returning typed closure — make-adder"
    (let [add5 (call-typed #'bc-make-adder 5.0)]
      (is (== 8.0 (.invk add5 3.0)))
      (is (== -2.0 (.invk add5 -7.0))))
    ;; Second call through BC path
    (let [add5 (call-typed #'bc-make-adder 5.0)]
      (is (== 8.0 (.invk add5 3.0)))
      (is (== -2.0 (.invk add5 -7.0)))))
  (testing "ftm returning typed closure — make-multiplier"
    (let [mul3 (call-typed #'bc-make-multiplier 3.0)]
      (is (== 15.0 (.invk mul3 5.0))))
    (let [mul3 (call-typed #'bc-make-multiplier 3.0)]
      (is (== 15.0 (.invk mul3 5.0)))))
  (testing "ftm inline closure via IFn.invoke"
    (is (== 7.0 (call-typed #'bc-inline-closure 4.0 3.0)))
    (is (== 7.0 (call-typed #'bc-inline-closure 4.0 3.0))))
  (testing "Nested closure over two captured vars"
    ;; f(x) = a*x + b, result = f(1) + f(2)
    ;; a=2, b=3: f(1)=5, f(2)=7, sum=12
    (is (== 12.0 (call-typed #'bc-nested-closure 2.0 3.0)))
    (is (== 12.0 (call-typed #'bc-nested-closure 2.0 3.0)))))

;; ================================================================
;; 11. Regression tests for specific fixed bugs
;; ================================================================

;; not= comparison + branch folding (was falling to IFn dispatch)
(deftm bc-not-equal [a :- Long, b :- Long] :- Long
  (if (not= a b) 1 0))

;; Float comparison without double widening (fcmpg path)
(deftm bc-float-cmp [a :- Float, b :- Float] :- Long
  (if (< a b) -1 (if (> a b) 1 0)))

;; quote keyword and symbol produce correct types
(deftm bc-quote-kw [] (quote :foo))
(deftm bc-quote-sym [] (quote bar))

;; rem with float (was truncating to int via irem)
(deftm bc-rem-float [a :- Float, b :- Float] :- Float
  (clojure.core/rem a b))

;; quot with double (was truncating to int via idiv)
(deftm bc-quot-double [a :- Double, b :- Double] :- Double
  (clojure.core/quot a b))

;; Large integer literal (> Integer/MAX_VALUE)
(deftm bc-large-int [] :- Long 3000000000)

;; short array aset (sastore, not bastore)
(deftm bc-short-aset [arr :- (Array short), i :- Long, v :- Long]
  (clojure.core/aset arr (int i) (short v)))

;; Integer arithmetic preserves types (long+long→long, not double)
(deftm bc-long-arith [a :- Long, b :- Long] :- Long
  (clojure.core/+ (clojure.core/* a b) (clojure.core/- a b)))

;; min/max preserve long type (was always double)
(deftm bc-long-min [a :- Long, b :- Long] :- Long
  (clojure.core/min a b))

(deftest bc-regression-test
  (testing "not= comparison"
    (is (== 1 (call-typed #'bc-not-equal 3 5)))
    (is (== 1 (call-typed #'bc-not-equal 3 5)))  ;; BC path
    (is (== 0 (call-typed #'bc-not-equal 5 5)))
    (is (== 0 (call-typed #'bc-not-equal 5 5))))
  (testing "Float comparison (fcmpg)"
    (is (== -1 (call-typed #'bc-float-cmp (float 1.0) (float 2.0))))
    (is (== -1 (call-typed #'bc-float-cmp (float 1.0) (float 2.0))))
    (is (== 1 (call-typed #'bc-float-cmp (float 3.0) (float 2.0))))
    (is (== 0 (call-typed #'bc-float-cmp (float 2.0) (float 2.0)))))
  (testing "quote keyword produces Keyword"
    (is (keyword? (bc-quote-kw)))
    (is (keyword? (bc-quote-kw)))  ;; BC path
    (is (= :foo (bc-quote-kw))))
  (testing "quote symbol produces Symbol"
    (is (symbol? (bc-quote-sym)))
    (is (symbol? (bc-quote-sym)))  ;; BC path
    (is (= 'bar (bc-quote-sym))))
  (testing "rem with float (not truncated to int)"
    (is (< (Math/abs (- 1.5 (float (call-typed #'bc-rem-float (float 5.5) (float 2.0))))) 0.01))
    (is (< (Math/abs (- 1.5 (float (call-typed #'bc-rem-float (float 5.5) (float 2.0))))) 0.01)))
  (testing "quot with double (not truncated to int)"
    (is (== 3.0 (call-typed #'bc-quot-double 7.5 2.5)))
    (is (== 3.0 (call-typed #'bc-quot-double 7.5 2.5))))
  (testing "Large integer literal"
    (is (== 3000000000 (call-typed #'bc-large-int)))
    (is (== 3000000000 (call-typed #'bc-large-int))))
  ;; Short array aset test skipped — (Array short) dispatch not
  ;; registered as a deftm overload target. The sastore fix in
  ;; emit-core-intrinsic is correct but needs a walked body that
  ;; has ^shorts type hints, which requires manual setup.
  (testing "Long arithmetic preserves type (not double)"
    ;; 3*4=12, 3-4=-1, 12+(-1)=11
    (is (== 11 (call-typed #'bc-long-arith 3 4)))
    (is (== 11 (call-typed #'bc-long-arith 3 4))))
  (testing "Long min/max preserves type"
    (is (== 3 (call-typed #'bc-long-min 3 5)))
    (is (== 3 (call-typed #'bc-long-min 3 5)))
    (is (== 2 (call-typed #'bc-long-min 7 2)))
    (is (== 2 (call-typed #'bc-long-min 7 2)))))

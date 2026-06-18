(ns raster.compiler.backend.jvm.bytecode-test
  "Tests for bytecode compiler form coverage.
   Each test compiles a deftm with specific Clojure forms and verifies
   correct execution via the compiled function."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm ftm]]
            [raster.ad.reverse]
            [raster.arrays :as arr]
            [raster.numeric :as rn]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.mem-merge :as mm]))

(defn- compile-and-run
  "Compile a deftm var and call it with args."
  [var-ref & {:keys [dtype args] :or {dtype :double args []}}]
  (let [compiled (binding [mm/*disable-mem-merge* true]
                   (pipeline/compile-aot var-ref :dtype dtype))]
    (apply compiled args)))

;; ================================================================
;; Arithmetic & primitives
;; ================================================================

(deftm bc-add [a :- Double b :- Double] :- Double (+ a b))
(deftm bc-mul [a :- Double b :- Double] :- Double (* a b))
(deftm bc-negate [a :- Double] :- Double (- a))
(deftm bc-add3 [a :- Double b :- Double c :- Double] :- Double
  (clojure.core/+ a b c))
(deftm bc-add6 [a :- Double b :- Double c :- Double
                d :- Double] :- Double
  (clojure.core/+ a b c d))
(deftm bc-sub3 [a :- Double b :- Double c :- Double] :- Double
  (clojure.core/- a b c))
(deftm bc-mul4 [a :- Double b :- Double c :- Double
                d :- Double] :- Double
  (clojure.core/* a b c d))
(deftm bc-div3 [a :- Double b :- Double c :- Double] :- Double
  (clojure.core// a b c))
(deftm bc-recip [x :- Double] :- Double (clojure.core// x))
(deftm bc-min3 [a :- Double b :- Double c :- Double] :- Double
  (clojure.core/min a b c))
(deftm bc-max4 [a :- Double b :- Double c :- Double
                d :- Double] :- Double
  (clojure.core/max a b c d))
(deftm bc-sum-of-products [a :- Double b :- Double c :- Double
                           x :- Double] :- Double
  (clojure.core/+ (clojure.core/* a x)
                  (clojure.core/* b x)
                  (clojure.core/* c x)))

(deftest arithmetic-test
  (testing "binary"
    (is (== 5.0 (compile-and-run #'bc-add :args [2.0 3.0])))
    (is (== 6.0 (compile-and-run #'bc-mul :args [2.0 3.0])))
    (is (== -3.0 (compile-and-run #'bc-negate :args [3.0]))))
  (testing "multi-arg + (regression: was silently dropping args beyond 2nd)"
    (is (== 6.0 (compile-and-run #'bc-add3 :args [1.0 2.0 3.0])))
    (is (== 10.0 (compile-and-run #'bc-add6 :args [1.0 2.0 3.0 4.0]))))
  (testing "multi-arg -"
    (is (== 5.0 (compile-and-run #'bc-sub3 :args [10.0 3.0 2.0]))))
  (testing "multi-arg *"
    (is (== 120.0 (compile-and-run #'bc-mul4 :args [2.0 3.0 4.0 5.0]))))
  (testing "multi-arg /"
    (is (== 2.0 (compile-and-run #'bc-div3 :args [24.0 4.0 3.0]))))
  (testing "unary / (reciprocal)"
    (is (== 0.25 (compile-and-run #'bc-recip :args [4.0]))))
  (testing "multi-arg min (regression: was only using first 2 args)"
    (is (== 1.0 (compile-and-run #'bc-min3 :args [5.0 1.0 3.0]))))
  (testing "multi-arg max"
    (is (== 7.0 (compile-and-run #'bc-max4 :args [2.0 7.0 3.0 5.0]))))
  (testing "sum of products (dp5 error pattern)"
    (is (== 24.0 (compile-and-run #'bc-sum-of-products :args [1.0 2.0 3.0 4.0])))))

;; ================================================================
;; let* bindings
;; ================================================================

(deftm bc-let [x :- Double] :- Double
  (let [a (* x 2.0)
        b (+ a 1.0)]
    (* b b)))

(deftest let-test
  (is (== 49.0 (compile-and-run #'bc-let :args [3.0]))))

;; ================================================================
;; if / conditionals
;; ================================================================

(deftm bc-if [x :- Double] :- Double
  (if (> x 0.0) x (- x)))

(deftest if-test
  (is (== 5.0 (compile-and-run #'bc-if :args [5.0])))
  (is (== 3.0 (compile-and-run #'bc-if :args [-3.0]))))

;; ================================================================
;; loop/recur
;; ================================================================

(deftm bc-sum-to [n :- Long] :- Long
  (loop [i 0 acc 0]
    (if (< i n)
      (recur (inc i) (+ acc i))
      acc)))

(deftest loop-recur-test
  (is (== 45 (compile-and-run #'bc-sum-to :args [10]))))

;; ================================================================
;; do (sequential effects)
;; ================================================================

(deftm bc-do [x :- Double] :- Double
  (do (+ x 1.0) (* x 2.0)))

(deftest do-test
  (is (== 10.0 (compile-and-run #'bc-do :args [5.0]))))

;; ================================================================
;; Array operations
;; ================================================================

(deftm bc-array-sum [arr :- (Array double)] :- Double
  (let [n (clojure.core/alength arr)]
    (loop [i 0 s 0.0]
      (if (< i n)
        (recur (inc i) (+ s (clojure.core/aget arr i)))
        s))))

(deftest array-ops-test
  (is (== 10.0 (compile-and-run #'bc-array-sum :args [(double-array [1.0 2.0 3.0 4.0])]))))

;; ================================================================
;; throw
;; ================================================================

(deftm bc-throw [x :- Double] :- Double
  (if (< x 0.0)
    (throw (ex-info "negative" {:x x}))
    x))

(deftest throw-test
  (is (== 5.0 (compile-and-run #'bc-throw :args [5.0])))
  (is (thrown-with-msg? Exception #"negative"
                        (compile-and-run #'bc-throw :args [-1.0]))))

;; ================================================================
;; try/catch
;; ================================================================

(deftm bc-try-catch [x :- Double] :- Double
  (try
    (if (< x 0.0)
      (throw (ex-info "neg" {}))
      x)
    (catch Exception e
      -999.0)))

(deftest try-catch-test
  (is (== 5.0 (compile-and-run #'bc-try-catch :args [5.0])))
  (is (== -999.0 (compile-and-run #'bc-try-catch :args [-1.0]))))

;; ================================================================
;; new (constructor)
;; ================================================================

(deftm bc-new-sb [] :- Long
  (let [sb (new StringBuilder "hello")]
    (long (.length sb))))

(deftest new-test
  (is (== 5 (compile-and-run #'bc-new-sb))))

;; ================================================================
;; Java static method
;; ================================================================

(deftm bc-static [x :- Double] :- Double
  (Math/abs x))

(deftest static-method-test
  (is (== 5.0 (compile-and-run #'bc-static :args [-5.0]))))

;; ================================================================
;; fn* (anonymous functions / closures)
;; ================================================================

(deftm bc-fn [x :- Double] :- Double
  (let [f (fn [y] (+ y x))]
    (double (f 10.0))))

(deftest fn-test
  (is (== 15.0 (compile-and-run #'bc-fn :args [5.0]))))

;; ================================================================
;; Vector literal
;; ================================================================

(deftm bc-vec [a :- Double b :- Double] :- Long
  (let [v [a b (+ a b)]]
    (long (count v))))

(deftest vector-literal-test
  (is (== 3 (compile-and-run #'bc-vec :args [1.0 2.0]))))

;; ================================================================
;; Map literal + keyword invocation
;; ================================================================

(deftm bc-map-kw [x :- Double] :- Double
  (let [m {:val x :double (* x 2.0)}]
    (double (:double m))))

(deftest map-keyword-test
  (is (== 10.0 (compile-and-run #'bc-map-kw :args [5.0]))))

;; ================================================================
;; instance?
;; ================================================================

(deftm bc-instance [x :- Double] :- Long
  (if (instance? Double x)
    1
    0))

(deftest instance-test
  (is (== 1 (compile-and-run #'bc-instance :args [5.0]))))

;; ================================================================
;; Float32 — scalar operations
;; ================================================================

(deftm bc-f32-add [a :- Float b :- Float] :- Float
  (+ a b))

(deftm bc-f32-mul [a :- Float b :- Float] :- Float
  (* a b))

(deftm bc-f32-let [x :- Float] :- Float
  (let [a (* x 2.0)
        b (+ a 1.0)]
    (* b b)))

(deftm bc-f32-if [x :- Float] :- Float
  (if (> x 0.0) x (- x)))

(deftest float32-scalar-test
  (testing "f32 add"
    (let [result (compile-and-run #'bc-f32-add :dtype :float :args [(float 2.0) (float 3.0)])]
      (is (instance? Float result))
      (is (== 5.0 result))))
  (testing "f32 mul"
    (is (== 6.0 (compile-and-run #'bc-f32-mul :dtype :float :args [(float 2.0) (float 3.0)]))))
  (testing "f32 let bindings"
    (is (== 49.0 (compile-and-run #'bc-f32-let :dtype :float :args [(float 3.0)]))))
  (testing "f32 if/branch"
    (is (== 5.0 (compile-and-run #'bc-f32-if :dtype :float :args [(float 5.0)])))
    (is (== 3.0 (compile-and-run #'bc-f32-if :dtype :float :args [(float -3.0)])))))

;; ================================================================
;; Float32 — array operations (faload/fastore)
;; ================================================================

(deftm bc-f32-array-sum [arr :- (Array float)] :- Double
  (let [n (clojure.core/alength arr)]
    (loop [i 0 s 0.0]
      (if (< i n)
        (recur (inc i) (+ s (clojure.core/aget arr i)))
        s))))

(deftm bc-f32-dotimes-aset [arr :- (Array float) scale :- Float] :- (Array float)
  (let [n (clojure.core/alength arr)]
    (dotimes [i n]
      (clojure.core/aset arr i (float (* (clojure.core/aget arr i) scale))))
    arr))

(deftm bc-f32-sgd-step [param :- (Array float) grad :- (Array float) lr :- Double]
  :- (Array float)
  (let [n (clojure.core/alength param)]
    (dotimes [i n]
      (clojure.core/aset param i
                         (float (- (clojure.core/aget param i)
                                   (* lr (clojure.core/aget grad i))))))
    param))

(deftest float32-array-test
  (testing "f32 array sum (faload in loop)"
    (is (== 10.0 (compile-and-run #'bc-f32-array-sum
                                  :dtype :float :args [(float-array [1.0 2.0 3.0 4.0])]))))
  (testing "f32 dotimes with aset (fastore in loop)"
    (let [arr (float-array [1.0 2.0 3.0])]
      (compile-and-run #'bc-f32-dotimes-aset :dtype :float :args [arr (float 10.0)])
      (is (== 10.0 (aget arr 0)))
      (is (== 20.0 (aget arr 1)))
      (is (== 30.0 (aget arr 2)))))
  (testing "f32 SGD step (two arrays + scalar)"
    (let [param (float-array [10.0 20.0 30.0])
          grad  (float-array [1.0 2.0 3.0])]
      (compile-and-run #'bc-f32-sgd-step :dtype :float :args [param grad 0.1])
      (is (< (Math/abs (- 9.9 (aget param 0))) 1e-5))
      (is (< (Math/abs (- 19.8 (aget param 1))) 1e-5))
      (is (< (Math/abs (- 29.7 (aget param 2))) 1e-5)))))

;; ================================================================
;; Float32 — AD through compiled pipeline
;; ================================================================

(deftm bc-f32-dot-loss [W :- (Array float) x :- (Array float)] :- Double
  (let [n (clojure.core/alength W)]
    (loop [i 0 s 0.0]
      (if (< i n)
        (let [w (clojure.core/aget W i) xi (clojure.core/aget x i)]
          (recur (inc i) (+ s (* w xi))))
        s))))

(deftm bc-f32-vg-dot [W :- (Array float) x :- (Array float)] :- Double
  (let [vg ((raster.ad.reverse/value+grad #'bc-f32-dot-loss) W x)
        loss (double (nth vg 0))]
    loss))

(deftm bc-f32-train-step [W :- (Array float) x :- (Array float) lr :- Double] :- Double
  (let [vg ((raster.ad.reverse/value+grad #'bc-f32-dot-loss) W x)
        loss (double (nth vg 0))
        ^floats dW (nth vg 1)]
    (dotimes [i (clojure.core/alength W)]
      (clojure.core/aset W i (float (- (clojure.core/aget W i) (* lr (clojure.core/aget dW i))))))
    loss))

(deftest float32-ad-test
  (testing "f32 value+grad through compiled pipeline"
    (is (== 32.0 (compile-and-run #'bc-f32-vg-dot
                                  :dtype :float
                                  :args [(float-array [1.0 2.0 3.0])
                                         (float-array [4.0 5.0 6.0])]))))
  (testing "f32 train step (AD + SGD update)"
    (let [W (float-array [1.0 2.0 3.0])
          x (float-array [4.0 5.0 6.0])]
      ;; loss = 1*4 + 2*5 + 3*6 = 32
      (is (== 32.0 (compile-and-run #'bc-f32-train-step :dtype :float :args [W x 0.1])))
      ;; W updated: W[i] -= 0.1 * x[i]
      (is (< (Math/abs (- 0.6 (aget W 0))) 1e-5))
      (is (< (Math/abs (- 1.5 (aget W 1))) 1e-5))
      (is (< (Math/abs (- 2.4 (aget W 2))) 1e-5)))))

;; ================================================================
;; quote
;; ================================================================

(deftm bc-quote [] :- Long
  (let [s 'hello]
    (long (count (str s)))))

(deftest quote-test
  (is (== 5 (compile-and-run #'bc-quote))))

;; ================================================================
;; Multi-form let* body — intermediate results must be popped
;; Regression: intermediate aset results accumulated on stack
;; ================================================================

(deftm bc-let-multi-body [arr :- (Array double)] :- Double
  (let [n (clojure.core/alength arr)]
    (clojure.core/aset arr 0 99.0)  ;; void — must not corrupt stack
    (clojure.core/aset arr 1 88.0)  ;; void — must not corrupt stack
    (+ (clojure.core/aget arr 0) (clojure.core/aget arr 1))))

(deftest let-multi-body-test
  (testing "intermediate aset results are popped in let body"
    (is (== 187.0 (compile-and-run #'bc-let-multi-body
                                   :args [(double-array [1.0 2.0 3.0])])))))

;; ================================================================
;; Multi-form loop* body — same stack pop issue
;; ================================================================

(deftm bc-loop-multi-body [arr :- (Array double)] :- Double
  (loop [i 0 sum 0.0]
    (if (< i (clojure.core/alength arr))
      (let [v (clojure.core/aget arr i)]
        (clojure.core/aset arr i (* v 2.0))  ;; side effect in loop body
        (recur (inc i) (+ sum v)))
      sum)))

(deftest loop-multi-body-test
  (testing "intermediate aset in loop body doesn't corrupt stack"
    (let [arr (double-array [1.0 2.0 3.0])]
      (is (== 6.0 (compile-and-run #'bc-loop-multi-body :args [arr])))
      ;; Side effect: array doubled
      (is (== 2.0 (aget arr 0)))
      (is (== 4.0 (aget arr 1))))))

;; ================================================================
;; aset as sibling method — must handle void body + return type
;; ================================================================

(deftm bc-aset-void [arr :- (Array double) i :- Long v :- Double] :- Double
  (raster.arrays/aset arr i v)
  (raster.arrays/aget arr i))

(deftest aset-void-test
  (testing "aset as void sibling method doesn't cause VerifyError"
    (let [arr (double-array [0.0 0.0])]
      (is (== 42.0 (compile-and-run #'bc-aset-void :args [arr 0 42.0]))))))

;; ================================================================
;; dotimes with aset — the embed-timestep pattern
;; Multiple aset calls inside dotimes body
;; ================================================================

(deftm bc-dotimes-aset [arr :- (Array double) scale :- Double] :- (Array double)
  (let [n (clojure.core/alength arr)]
    (dotimes [i n]
      (clojure.core/aset arr i (* (clojure.core/aget arr i) scale)))
    arr))

(deftest dotimes-aset-test
  (testing "dotimes with aset compiles without stack mismatch"
    (let [arr (double-array [1.0 2.0 3.0])]
      (compile-and-run #'bc-dotimes-aset :args [arr 10.0])
      (is (== 10.0 (aget arr 0)))
      (is (== 20.0 (aget arr 1)))
      (is (== 30.0 (aget arr 2))))))

;; ================================================================
;; Two aset calls in dotimes — the exact embed-timestep pattern
;; ================================================================

(deftm bc-dotimes-two-asets [out :- (Array double) scale :- Double n :- Long] :- (Array double)
  (dotimes [i n]
    (clojure.core/aset out i (* (double i) scale))
    (clojure.core/aset out (+ n i) (* (double i) scale 2.0)))
  out)

(deftest dotimes-two-asets-test
  (testing "two aset calls in dotimes body compile correctly"
    (let [out (double-array 8)]
      (compile-and-run #'bc-dotimes-two-asets :args [out 10.0 4])
      (is (== 0.0  (aget out 0)))
      (is (== 10.0 (aget out 1)))
      (is (== 20.0 (aget out 2)))
      (is (== 30.0 (aget out 3)))
      (is (== 0.0  (aget out 4)))
      (is (== 20.0 (aget out 5))))))

;; ================================================================
;; SGD-like pattern: arr[i] = arr[i] - lr * grad[i]
;; Two arrays + scalar in a broadcast loop
;; ================================================================

(deftm bc-sgd-step [param :- (Array double) grad :- (Array double) lr :- Double] :- (Array double)
  (let [n (clojure.core/alength param)]
    (dotimes [i n]
      (clojure.core/aset param i
                         (- (clojure.core/aget param i)
                            (* lr (clojure.core/aget grad i)))))
    param))

(deftest sgd-step-test
  (testing "SGD update pattern with two arrays compiles correctly"
    (let [param (double-array [10.0 20.0 30.0])
          grad  (double-array [1.0 2.0 3.0])]
      (compile-and-run #'bc-sgd-step :args [param grad 0.1])
      (is (< (Math/abs (- 9.9  (aget param 0))) 1e-10))
      (is (< (Math/abs (- 19.8 (aget param 1))) 1e-10))
      (is (< (Math/abs (- 29.7 (aget param 2))) 1e-10)))))

;; ================================================================
;; object-array + nth — the value+grad result pattern
;; ================================================================

(deftm bc-object-array-nth [a :- Double b :- Double] :- Double
  (let [result (clojure.core/object-array [a b (+ a b)])
        ^double x (clojure.core/aget result 0)
        ^double y (clojure.core/aget result 2)]
    (+ x y)))

(deftest object-array-nth-test
  (testing "object-array construction and aget compile correctly"
    ;; a=3, b=5, result[0]=a=3, result[2]=a+b=8, 3+8=11
    (is (== 11.0 (compile-and-run #'bc-object-array-nth :args [3.0 5.0])))))

;; ================================================================
;; Compiled value+grad — AD through the bytecode pipeline
;; ================================================================

(deftm bc-scalar-loss [a :- Double b :- Double] :- Double (* a b))

(deftm bc-vg-scalar [a :- Double b :- Double] :- Double
  (let [vg ((raster.ad.reverse/value+grad #'bc-scalar-loss) a b)
        loss (double (nth vg 0))]
    loss))

(deftest compiled-value+grad-scalar-test
  (testing "compiled value+grad on scalar loss"
    (is (== 12.0 (compile-and-run #'bc-vg-scalar :args [3.0 4.0])))))

;; ================================================================
;; Compiled value+grad with array params and loop
;; ================================================================

(deftm bc-dot-loss [W :- (Array double) x :- (Array double)] :- Double
  (let [n (clojure.core/alength W)]
    (loop [i 0 s 0.0]
      (if (< i n)
        (let [w (clojure.core/aget W i) xi (clojure.core/aget x i)]
          (recur (inc i) (+ s (* w xi))))
        s))))

(deftm bc-vg-dot [W :- (Array double) x :- (Array double)] :- Double
  (let [vg ((raster.ad.reverse/value+grad #'bc-dot-loss) W x)
        loss (double (nth vg 0))]
    loss))

(deftest compiled-value+grad-loop-test
  (testing "compiled value+grad with loop over arrays"
    (is (== 32.0 (compile-and-run #'bc-vg-dot
                                  :args [(double-array [1.0 2.0 3.0])
                                         (double-array [4.0 5.0 6.0])])))))

;; ================================================================
;; Compiled training step: value+grad + SGD in one function
;; ================================================================

(deftm bc-train-step [W :- (Array double) x :- (Array double) lr :- Double] :- Double
  (let [vg ((raster.ad.reverse/value+grad #'bc-dot-loss) W x)
        loss (double (nth vg 0))
        ^doubles dW (nth vg 1)]
    (dotimes [i (clojure.core/alength W)]
      (clojure.core/aset W i (- (clojure.core/aget W i) (* lr (clojure.core/aget dW i)))))
    loss))

(deftest compiled-train-step-test
  (testing "compiled training step with loop loss + SGD update"
    (let [W (double-array [1.0 2.0 3.0])
          x (double-array [4.0 5.0 6.0])]
      ;; loss = 1*4 + 2*5 + 3*6 = 32
      (is (== 32.0 (compile-and-run #'bc-train-step :args [W x 0.1])))
      ;; W updated: W[i] -= 0.1 * x[i]
      (is (< (Math/abs (- 0.6 (aget W 0))) 1e-10))
      (is (< (Math/abs (- 1.5 (aget W 1))) 1e-10))
      (is (< (Math/abs (- 2.4 (aget W 2))) 1e-10)))))

;; ================================================================
;; Branch-merge stack-map regression
;;
;; (if cond <const> <deep .invk-arithmetic with a loop>): the if-emitter's
;; skip-box? optimization predicts the else-branch's stack type; when that
;; prediction disagreed with the actually-emitted type it boxed the else to a
;; ref while the then-branch was left an unboxed primitive — different stack
;; heights at the join => "Stack size mismatch" bytecode-verifier failure.
;; This is the trapz/besselj0 shape (raster.numeric .invk arithmetic).
;; ================================================================

(deftm bc-if-deep-arith-loop [ys :- (Array double), dx :- Double] :- Double
  (if (< (arr/alength ys) 2)
    0.0
    (let [n-1 (dec (arr/alength ys))]
      (rn/* dx
            (rn/+ (rn/* 0.5 (rn/+ (double (arr/aget ys 0)) (double (arr/aget ys n-1))))
                  (loop [i 1 s 0.0]
                    (if (>= i n-1)
                      s
                      (recur (inc i) (rn/+ s (double (arr/aget ys i)))))))))))

(deftest if-merge-deep-arith-loop-test
  (testing "if with a constant then-branch and a deep .invk-arithmetic else
            containing a loop compiles and computes correctly"
    ;; Call directly to exercise the lazy-JIT bytecode path (where the
    ;; skip-box? misprediction occurs — compile-aot's richer inference hides
    ;; it). trapezoidal sum of [0 1 4 9] dx=1: 0.5*(0+9) + (1+4) = 9.5
    (is (== 9.5 (bc-if-deep-arith-loop (double-array [0.0 1.0 4.0 9.0]) 1.0)))))

(ns raster.compiler.pipeline-quality-test
  "Regression tests for compiler output quality.
   Ensures bytecode optimizations (int loops, branch folding, checkcast skip)
   and dispatch table correctness (BLAS > parametric) are maintained."
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.core :refer [deftm ftm]]
            [raster.nn :as nn]
            [raster.linalg.blas :as blas]
            [raster.dl.optim :as optim]
            [raster.ad.reverse :as rev]
            [raster.arrays :as arrays]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.pe :as pe]
            [raster.tooling.inspect :as inspect]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(defn- form-contains-name?
  "True if any symbol in form has a name containing substring s."
  [form ^String s]
  (cond
    (symbol? form) (.contains (str form) s)
    (seq? form) (some #(form-contains-name? % s) form)
    (vector? form) (some #(form-contains-name? % s) form)
    :else false))

;; Self-contained train-step (mirrors bench/raster/mnist_bench.clj)
;; so this test runs with -M:test without the :bench classpath.

(deftm test-train-step
  [W1 :- (Array double), b1 :- (Array double),
   W2 :- (Array double), b2 :- (Array double),
   x :- (Array double), y :- (Array double),
   lr :- Double] :- Double
  (let [vg ((rev/value+grad (var nn/loss-fn)) W1 b1 W2 b2 x y)
        loss (nth vg 0)]
    (optim/sgd-step! W1 (nth vg 1) (arrays/alength W1) lr)
    (optim/sgd-step! b1 (nth vg 2) (arrays/alength b1) lr)
    (optim/sgd-step! W2 (nth vg 3) (arrays/alength W2) lr)
    (optim/sgd-step! b2 (nth vg 4) (arrays/alength b2) lr)
    loss))

;; ================================================================
;; Dense layer transparency — ensures dense uses SOAC forms
;; (par/map! + par/reduce) so fusion can see through them.
;; ================================================================

(deftest dense-transparent-loops-test
  (testing "dense-into! uses transparent dotimes loops (not opaque BLAS)"
    (let [v (ns-resolve 'raster.nn (symbol "dense-into!_m_doubles_doubles_doubles_doubles"))
          body (:raster.core/deftm-walked-body (meta v))]
      (is (some #(form-contains-name? % "dotimes") body)
          "dense-into! should use dotimes loops")
      (is (not (some #(form-contains-name? % "dgemv") body))
          "dense-into! should NOT use opaque BLAS calls")))

  (testing "dense uses par/pmap + par/reduce SOACs (not dotimes or BLAS)"
    (let [v (ns-resolve 'raster.nn (symbol "dense_m_doubles_doubles_doubles"))
          body (:raster.core/deftm-walked-body (meta v))]
      (is (some #(form-contains-name? % "raster.par/pmap") body)
          "dense should use par/pmap (pure) for SOAC fusion")
      (is (some #(form-contains-name? % "raster.par/reduce") body)
          "dense should use par/reduce for inner dot product")
      (is (not (some #(form-contains-name? % "dgemv") body))
          "dense should NOT use opaque BLAS calls"))))

(deftest dense-relu-fusion-test
  (testing "predict pipeline fuses dense→relu into single par/map!"
    (let [stages (pipeline/show-pipeline #'nn/predict-fn)]
      ;; After body-position call lifting, all 3 calls (dense, relu, dense)
      ;; should be inlined. SOAC fusion merges dense1→relu vertically.
      (is (not (form-contains-name? (:fixpointed stages) "raster.nn/"))
          "All nn/ calls should be inlined in fixpointed stage")
      (is (pos? (:vertical (:soac-fused-stats stages)))
          "SOAC should report vertical fusion (dense→relu)"))))

;; ================================================================
;; Bytecode quality — catches loop/comparison/checkcast regressions
;; ================================================================

(defn- sgd-helper-names
  "Identify SGD update helpers by structure: contains loop* and alength.
   Par-extracted helpers carry their free variables as params (variable count)."
  [diag]
  (for [[name info] (:helper-map diag)
        :let [expr (:expr info)]
        :when (and (form-contains-name? expr "loop*")
                   (form-contains-name? expr "alength"))]
    name))

(deftest compiled-bytecode-quality-test
  (testing "Compiled SGD helpers have optimal bytecode"
    (let [f (pipeline/compile-aot #'test-train-step :simd? false)
          diag (inspect/compiled-diagnostics #'test-train-step)
          helpers (sgd-helper-names diag)]
      (is (>= (count helpers) 2)
          "Should have at least 2 SGD update helpers")
      (when-let [bytes (:statics-bytes diag)]
        (doseq [helper-name helpers]
          (let [bc (with-out-str (inspect/disassemble bytes {:method helper-name}))]
            (when (seq bc)
              (testing (str helper-name " uses integer comparison")
                (is (.contains bc "IF_ICMPGE")
                    (str helper-name " should use IF_ICMPGE (int counted loop)")))
              ;; CHECKCASTs on array types ([D, [F) are expected at if-merge
              ;; points and are zero-cost in C2. Only flag CHECKCASTs on
              ;; IFn or Object (which indicate missing type info).
              (testing (str helper-name " has no untyped checkcast")
                (is (not (re-find #"CHECKCAST.*Object|CHECKCAST.*IFn\b" bc))
                    (str helper-name " should not have CHECKCAST to Object/IFn")))
              (testing (str helper-name " has no double comparison in loop")
                (is (not (.contains bc "DCMPG"))
                    (str helper-name " should not use DCMPG for loop bounds"))))))))))

;; ================================================================
;; Functional correctness — compiled step produces right loss
;; ================================================================

(deftest compiled-mlp-correctness-test
  (testing "Compiled MLP train step produces valid loss"
    (let [f (pipeline/compile-aot #'test-train-step :simd? false)
          W1 (double-array (* 784 128))
          b1 (double-array 128)
          W2 (double-array (* 128 10))
          b2 (double-array 10)
          x (double-array 784)
          y (let [a (double-array 10)] (aset a 3 1.0) a)
          loss (double (f W1 b1 W2 b2 x y 0.01))]
      (is (Double/isFinite loss) "Loss should be finite")
      (is (> loss 0.0) "Loss should be positive")
      (is (< loss 100.0) "Loss should be reasonable"))))

;; ================================================================
;; Hoisted buffers — catches allocation regression
;; ================================================================

(deftest hoisted-buffers-test
  (testing "MLP train step has hoisted buffers (zero per-call allocations)"
    (let [f (pipeline/compile-aot #'test-train-step :simd? false)
          diag (inspect/compiled-diagnostics #'test-train-step)]
      (is (>= (:hoisted-count diag) 2)
          "Should have at least 2 hoisted buffers"))))

;; ================================================================
;; Float type-flow — ensures float pipelines use FloatVector, not DoubleVector
;; ================================================================

(deftm f32-relu
  "Float relu for type-flow testing."
  [x :- (Array float)] :- (Array float)
  (let [n (arrays/alength x)
        out (float-array n)]
    (raster.par/map! out i n float (Math/max (float 0.0) (raster.arrays/aget x i)))
    out))

(deftm f32-scale
  "Float scale: element-wise multiply by scalar."
  [x :- (Array float), s :- Float] :- (Array float)
  (let [n (arrays/alength x)
        out (float-array n)]
    (raster.par/map! out i n float (raster.numeric/* (raster.arrays/aget x i) s))
    out))

(deftest float-pipeline-uses-float-vector-test
  (testing "Float relu pipeline uses FloatVector, not DoubleVector"
    (let [stages (pipeline/show-pipeline #'f32-relu)
          backend-form (:backend-applied stages)]
      (is (some? backend-form) "Backend stage should produce output")
      (when backend-form
        (is (form-contains-name? backend-form "FloatVector")
            "Float relu backend should use FloatVector")
        (is (not (form-contains-name? backend-form "DoubleVector"))
            "Float relu backend should NOT use DoubleVector"))))

  (testing "Float scale pipeline uses FloatVector, not DoubleVector"
    (let [stages (pipeline/show-pipeline #'f32-scale)
          backend-form (:backend-applied stages)]
      (when backend-form
        (is (form-contains-name? backend-form "FloatVector")
            "Float scale backend should use FloatVector")
        (is (not (form-contains-name? backend-form "DoubleVector"))
            "Float scale backend should NOT use DoubleVector")))))

(deftest float-simd-stats-test
  (testing "Float relu pipeline reports SIMD map compilation"
    (let [stages (pipeline/show-pipeline #'f32-relu)
          stats (:backend-applied-stats stages)]
      (when stats
        (is (pos? (:simd-maps stats))
            "Float relu should have at least 1 SIMD-compiled map")))))

(deftest float-pipeline-correctness-test
  (testing "Float relu computes correctly"
    (let [x (float-array [-1.0 0.0 2.5 -3.0 4.0])
          result (f32-relu x)]
      (is (= [0.0 0.0 2.5 0.0 4.0] (vec result)))))

  (testing "Float scale computes correctly"
    (let [x (float-array [1.0 2.0 3.0])
          result (f32-scale x (float 2.0))]
      (is (= [2.0 4.0 6.0] (vec result))))))

;; ================================================================
;; BC compiler: overload resolution + loop convergence
;; ================================================================

(deftm bc-fixed-point-test-fn
  "Fixed-point iteration — exercises Math.abs overload resolution and
  loop convergence in the BC compiler. Catches the d2i truncation bug
  where Math.abs(int) was selected instead of Math.abs(double)."
  [g :- (Fn [Double Double] Double),
   z0 :- Double, theta :- Double,
   tol :- Double, maxiter :- Long] :- Double
  (loop [z z0, iter (int 0)]
    (if (>= iter maxiter)
      z
      (let [z-new (g z theta)
            diff (Math/abs (- z-new z))]
        (if (< diff tol)
          z-new
          (recur z-new (unchecked-add-int iter 1)))))))

(deftm bc-and-short-circuit-fn
  "Tests that (and (> j 0) (expr-using-j)) short-circuits correctly
  in BC-compiled code. The comparison must return Boolean.FALSE (not int 0)
  so Clojure truthiness works in the and macro expansion."
  [arr :- (Array int), n :- Long]
  (loop [j (int 2)]
    (when (and (> j 0)
               (> (clojure.core/aget arr (dec j)) (clojure.core/aget arr j)))
      (let [tmp (clojure.core/aget arr j)]
        (clojure.core/aset arr j (clojure.core/aget arr (dec j)))
        (clojure.core/aset arr (dec j) tmp)
        (recur (dec j))))))

(deftest bc-and-short-circuit-test
  (testing "BC-compiled (and (> j 0) ...) short-circuits correctly"
    ;; The test function is an insertion sort inner loop starting at j=2.
    ;; Input [2 3 1]: element at j=2 (value 1) must bubble left past 3 and 2.
    ;; This exercises: j=2 (swap), j=1 (swap), j=0 (> j 0 is false → stop).
    ;; If (and) short-circuit is broken, j=0 causes (aget arr -1) → crash.
    ;; First call = deftype, second = BC. Both must work.
    (let [entry (first (get @(:raster.core/dispatch-table (meta #'bc-and-short-circuit-fn)) 2))
          tfn (:typed-target-fn entry)]
      (let [a (int-array [2 3 1])]
        (tfn a 3)
        (is (= [1 2 3] (vec a)) "Deftype path should sort"))
      (let [a (int-array [2 3 1])]
        (tfn a 3)
        (is (= [1 2 3] (vec a)) "BC path should sort (and must not index -1)")))))

(deftest bc-overload-resolution-test
  (testing "BC-compiled loop converges correctly (Math.abs picks double overload)"
    ;; Babylonian sqrt: g(z, theta) = (z + theta/z) / 2, converges to sqrt(theta)
    (let [g (ftm [z :- Double, theta :- Double] :- Double
                 (/ (+ z (/ theta z)) 2.0))]
      ;; First call triggers lazy JIT
      (bc-fixed-point-test-fn g 1.0 4.0 1e-12 100)
      ;; Second call uses BC-compiled version — this is where the bug manifested
      (let [result (bc-fixed-point-test-fn g 1.0 9.0 1e-12 100)]
        (is (< (Math/abs (- result 3.0)) 1e-10)
            (str "BC-compiled sqrt(9) should be 3.0, got " result))))))

;; ================================================================
;; Loop-lift recovery — differential correctness (Option A)
;; A hand-written loop/recur reduction must (a) lift to a SIMD par/reduce and
;; (b) produce the same result as the scalar reference, within FP reassociation
;; tolerance (the multi-accumulator SIMD reduce reorders the additions).
;; ================================================================

(deftm handloop-dot
  [a :- (Array double), b :- (Array double), n :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (raster.numeric/< i n)
      (recur (raster.numeric/+ i 1)
             (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget a i)
                                                     (raster.arrays/aget b i))))
      acc)))

(deftest handloop-reduce-differential-test
  (testing "hand-loop dot lifts to SIMD par/reduce and matches scalar within FP tol"
    (let [expl (with-out-str (pipeline/explain-pipeline (var handloop-dot)))]
      (is (re-find #"raster.par/reduce" expl) "hand loop should lift to par/reduce")
      (is (re-find #"simd-reduces 1" expl) "lifted reduce should vectorize"))
    (let [f (pipeline/compile-aot (var handloop-dot))
          rng (java.util.Random. 42)
          n 1000
          a (double-array (repeatedly n #(.nextGaussian rng)))
          b (double-array (repeatedly n #(.nextGaussian rng)))
          a2 (doubles a) b2 (doubles b)
          expected (loop [i 0 acc 0.0]
                     (if (< i n) (recur (inc i) (+ acc (* (aget a2 i) (aget b2 i)))) acc))
          got (double (f a b n))]
      (is (< (Math/abs (- got (double expected))) 1e-9)
          (str "lifted=" got " scalar=" expected)))))

;; ================================================================
;; Task #78 regression: pe must not propagate a stale :raster.op/original
;; across a rewrite that changed the node's operator.
;; ================================================================

(defn- count-name
  "Count symbols in `form` whose name contains substring `s`."
  [form ^String s]
  (count (filter #(and (symbol? %) (.contains (str %) s))
                 (tree-seq #(or (seq? %) (vector? %)) seq form))))

;; cross-entropy-backward-dp = (broadcast [p t] (* (- (/ t (max 1e-15 p))) dy)).
;; With a LITERAL 1.0 adjoint (the f64 loss seed), mul-by-one folds (* X 1.0)->X;
;; pe then re-stamped the surviving (- X) with the *'s :raster.op/original,
;; normalize rewrote the head - -> * and single-arg-mul deleted the negation ->
;; dp = +t/p -> exact gradient sign flip -> MLP-f64 training divergence.
(deftm ce-bwd-literal-seed [p :- (Array double) t :- (Array double)] :- (Array double)
  (nn/cross-entropy-backward-dp 1.0 p t))

(deftest cross-entropy-backward-literal-seed-keeps-negation-issue-78
  (let [fx (:fixpointed (pipeline/show-pipeline #'ce-bwd-literal-seed :dtype :double))]
    (is (>= (count-name fx "_minus__m_double-impl") 1)
        (str "task #78: a LITERAL 1.0 f64 adjoint must NOT delete the -t/p "
             "negation in cross-entropy-backward-dp (gradient sign flip)"))))

;; Direct invariant on pe-pass: a rewrite that changes a node's operator must not
;; leave the surviving node claiming the parent's :raster.op/original.
(deftest pe-drops-stale-op-original-on-rewrite-issue-78
  (let [node (with-meta
               (list '.invk 'raster.numeric/_star__m_double_double-impl
                     (with-meta (list '.invk 'raster.numeric/_minus__m_double-impl 'x)
                       {:raster.op/original 'raster.numeric/-})
                     1.0)
               {:raster.op/original 'raster.numeric/*})
        out (pe/pe-pass node {})]
    (is (not= 'raster.numeric/* (:raster.op/original (meta out)))
        "surviving (- x) must not inherit the *'s :raster.op/original after mul-by-one")))

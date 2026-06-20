(ns raster.compiler.backend.wasm.emit-test
  "Track A — differential + validity tests for the JVM-side wasm backend.
   Emits .wasm from a real deftm's post-pass IR, then uses Chicory (a pure-JVM
   WebAssembly runtime) to (a) PARSE it — which validates the module, malformed
   bytes throw — and (b) EXECUTE it, asserting results match a reference.
   No node / native deps; runs in `clojure -M:test`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.math]
            [raster.arrays]
            [raster.compiler.pipeline :as pl])
  (:import [com.dylibso.chicory.wasm Parser]
           [com.dylibso.chicory.runtime Instance]))

(deftm saxpy-k! [a :- Double, x :- (Array double), y :- (Array double), n :- Long] :- Long
  (loop [i 0]
    (if (clojure.core/< (long i) (long n))
      (do (raster.arrays/aset y (long i)
                              (raster.numeric/+ (raster.numeric/* a (raster.arrays/aget x i))
                                                (raster.arrays/aget y i)))
          (recur (clojure.core/inc (long i))))
      n)))

(deftm poly-k [x :- Double] :- Double
  (raster.numeric/+ (raster.numeric/* x x) x))

(deftm dot-k [x :- (Array double), y :- (Array double), n :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (clojure.core/< (long i) (long n))
      (recur (clojure.core/inc (long i))
             (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget x i)
                                                     (raster.arrays/aget y i))))
      acc)))

(deftm sq-k [x :- Double] :- Double (raster.numeric/* x x))

(deftm sumsq-k [a :- (Array double), n :- Long] :- Double
  (loop [i 0 acc 0.0]
    (if (clojure.core/< (long i) (long n))
      (recur (clojure.core/inc (long i)) (raster.numeric/+ acc (sq-k (raster.arrays/aget a i))))
      acc)))

(deftm relu-k! [x :- (Array double), out :- (Array double), n :- Long] :- nil
  (dotimes [i n]
    (when (clojure.core/> (raster.arrays/aget x i) 0.0)
      (raster.arrays/aset out i (raster.arrays/aget x i)))))

(deftm dot-f32-k [x :- (Array float), y :- (Array float), n :- Long] :- Float
  (loop [i 0 acc (float 0.0)]
    (if (clojure.core/< (long i) (long n))
      (recur (clojure.core/inc (long i))
             (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget x i)
                                                     (raster.arrays/aget y i))))
      acc)))

(deftm scaled-sum-k [x :- (Array double), n :- Long, s :- Double] :- Double
  (let [half (raster.numeric/* s 0.5)]
    (loop [i 0 acc 0.0]
      (if (clojure.core/< (long i) (long n))
        (recur (clojure.core/inc (long i))
               (raster.numeric/+ acc (raster.numeric/* half (raster.arrays/aget x i))))
        acc))))

(defn- instantiate [^bytes wasm]
  (-> (Instance/builder (Parser/parse wasm)) (.build)))

(deftest saxpy-emits-valid-and-correct-wasm
  ;; Exercises the wired pipeline entry: deftm → compile-wasm → .wasm bytes.
  (let [m (pl/compile-wasm #'saxpy-k! :name "saxpy")
        inst (instantiate (:bytes m))      ; parse = validate; throws on malformed
        mem  (.memory inst)
        saxpy (.export inst "saxpy")
        n 1000, a 3.0]
    (testing "module shape"
      (is (= [:f64 :i32 :i32 :i32] (:param-types m)))
      (is (pos? (count (:bytes m)))))
    (testing "executes saxpy correctly vs reference (Chicory)"
      (dotimes [i n]
        (.writeF64 mem (* 8 i) (double i))
        (.writeF64 mem (* 8 (+ n i)) (* 0.5 (double i))))
      (.apply saxpy (long-array [(Double/doubleToRawLongBits a) 0 (* 8 n) n]))
      (let [max-err (reduce (fn [mx i]
                              (max mx (Math/abs (- (.readDouble mem (* 8 (+ n i)))
                                                   (+ (* a i) (* 0.5 i))))))
                            0.0 (range n))]
        (is (< max-err 1e-9) (str "saxpy max-err=" max-err))))))

(deftest poly-scalar-kernel
  (testing "scalar f64 kernel returns its value"
    (let [m (pl/compile-wasm #'poly-k :name "poly")
          inst (instantiate (:bytes m))
          r (.apply (.export inst "poly") (long-array [(Double/doubleToRawLongBits 3.0)]))]
      (is (= [:f64] (:result-types m)))
      (is (< (Math/abs (- (Double/longBitsToDouble (aget r 0)) 12.0)) 1e-12)))))

(deftest dot-reduction-kernel
  (testing "multi-var loop reduction returns f64 accumulator"
    (let [m (pl/compile-wasm #'dot-k :name "dot")
          inst (instantiate (:bytes m))
          mem (.memory inst)
          n 1000]
      (is (= [:f64] (:result-types m)))
      (dotimes [i n] (.writeF64 mem (* 8 i) (double i)) (.writeF64 mem (* 8 (+ n i)) 2.0))
      (let [r (.apply (.export inst "dot") (long-array [0 (* 8 n) n]))
            got (Double/longBitsToDouble (aget r 0))
            exp (reduce + (map #(* (double %) 2.0) (range n)))]
        (is (< (Math/abs (- got exp)) 1e-6) (str "dot=" got " exp=" exp))))))

(deftest sumsq-inlined-deftm-kernel
  (testing "deftm→deftm call (sq-k) inlined → let* in value position"
    (let [m (pl/compile-wasm #'sumsq-k :name "sumsq")
          inst (instantiate (:bytes m))
          mem (.memory inst)
          n 100]
      (is (= [:f64] (:result-types m)))
      (dotimes [i n] (.writeF64 mem (* 8 i) (double i)))
      (let [r (.apply (.export inst "sumsq") (long-array [0 n]))
            got (Double/longBitsToDouble (aget r 0))
            exp (reduce + (map #(* (double %) (double %)) (range n)))]
        (is (< (Math/abs (- got exp)) 1e-6) (str "sumsq=" got " exp=" exp))))))

;; Step 2: deftm→deftm inlining of an `if`-bodied helper that takes a primitive
;; byte array. Exercises (a) ensure-walked-body! on a cold callee, (b) :branch
;; bodies being inlinable, (c) byte-array params substituted directly (no aliased
;; let* binding that would break the emitter's array :elems lookup).
(deftm blut-k [tbl :- (Array byte), id :- Long] :- Long
  (if (clojure.core/and (clojure.core/>= id 0) (clojure.core/<= id 7))
    (long (raster.arrays/aget tbl (int id)))
    0))

(deftm blut-call-k [tbl :- (Array byte), id :- Long] :- Long
  (clojure.core/+ (blut-k tbl id) (blut-k tbl (clojure.core/inc id))))

(deftest deftm-call-bytearray-inline-kernel
  (testing "if-bodied helper taking (Array byte) inlines without array aliasing"
    (let [m    (pl/compile-wasm #'blut-call-k :name "blut")
          inst (instantiate (:bytes m))
          mem  (.memory inst)
          tbl  (vec (for [i (range 8)] (* i 3)))      ; 0,3,6,…21
          ref  (fn [id] (+ (if (and (>= id 0) (<= id 7)) (nth tbl id) 0)
                           (if (and (>= (inc id) 0) (<= (inc id) 7)) (nth tbl (inc id)) 0)))]
      (is (= [:i32] (:result-types m)))
      (.write mem 0 (byte-array (map byte tbl)))
      (doseq [id [0 3 6 7 8]]
        (let [got (aget (.apply (.export inst "blut") (long-array [0 id])) 0)]
          (is (= (ref id) got) (str "blut id=" id)))))))

(deftest relu-dotimes-when-void-kernel
  (testing "dotimes + when + void function + f64 comparison"
    (let [m (pl/compile-wasm #'relu-k! :name "relu")
          inst (instantiate (:bytes m))
          mem (.memory inst)
          n 1000]
      (is (= [] (:result-types m)) "void kernel has no result")
      (dotimes [i n] (.writeF64 mem (* 8 i) (- (double i) 500.0)) (.writeF64 mem (* 8 (+ n i)) 0.0))
      (.apply (.export inst "relu") (long-array [0 (* 8 n) n]))
      (let [bad (count (filter (fn [i]
                                 (let [xi (- (double i) 500.0)
                                       got (.readDouble mem (* 8 (+ n i)))]
                                   (> (Math/abs (- got (max xi 0.0))) 1e-9)))
                               (range n)))]
        (is (zero? bad) (str bad " relu mismatches"))))))

(deftest dot-f32-kernel
  (testing "f32 reduction: f32 element loads, f32 arith, f32 result"
    (let [m (pl/compile-wasm #'dot-f32-k :name "dotf" :dtype :float)
          inst (instantiate (:bytes m))
          mem (.memory inst)
          n 1000]
      (is (= [:f32] (:result-types m)))
      (dotimes [i n] (.writeF32 mem (* 4 i) (float i)) (.writeF32 mem (* 4 (+ n i)) (float 2.0)))
      (let [r (.apply (.export inst "dotf") (long-array [0 (* 4 n) n]))
            got (Float/intBitsToFloat (unchecked-int (aget r 0)))
            exp (reduce + (map #(* (float %) 2.0) (range n)))]
        (is (< (Math/abs (- got exp)) 1.0) (str "dot-f32=" got " exp=" exp))))))

(deftest scaled-sum-let-binding-kernel
  (testing "let* binding (local alloc) wrapping a loop"
    (let [m (pl/compile-wasm #'scaled-sum-k :name "ss")
          inst (instantiate (:bytes m))
          mem (.memory inst)
          n 1000]
      (dotimes [i n] (.writeF64 mem (* 8 i) (double i)))
      (let [r (.apply (.export inst "ss") (long-array [0 n (Double/doubleToRawLongBits 4.0)]))
            got (Double/longBitsToDouble (aget r 0))
            exp (* 2.0 (reduce + (range n)))]  ; half=2.0 → 2*Σi
        (is (< (Math/abs (- got exp)) 1e-6) (str "scaled-sum=" got " exp=" exp))))))

;; ── SIMD128 (opt-in) ──────────────────────────────────────────────────────
;; Chicory cannot EXECUTE v128, so here we assert the SIMD module is structurally
;; valid (Parser/parse validates — malformed/ill-typed bytes throw) and actually
;; vectorized (contains the 0xfd SIMD prefix), and that the scalar build of the
;; same kernel still runs correctly. Execution of the v128 path is validated on
;; node/V8 (cljs-sandbox; f64 vadd + saxpy across sizes incl. odd remainders).
(deftm vadd-simd! [a :- (Array double), b :- (Array double), out :- (Array double), n :- Long] :- nil
  (dotimes [i n]
    (raster.arrays/aset out i (raster.numeric/+ (raster.arrays/aget a i) (raster.arrays/aget b i)))))

(defn- has-v128? [m] (boolean (some #(= (bit-and (long %) 0xff) 0xfd) (seq (:bytes m)))))

(deftest simd-elementwise-f64-vectorizes-and-validates
  (testing "f64 elementwise map: SIMD build is valid v128, scalar build runs"
    (let [simd   (pl/compile-wasm #'vadd-simd! :name "vadd" :wasm-simd? true)
          scalar (pl/compile-wasm #'vadd-simd! :name "vadd")]
      ;; SIMD module is structurally valid wasm and actually used v128
      (is (some? (Parser/parse (:bytes simd))))
      (is (has-v128? simd) "vectorizer emitted v128 instructions")
      (is (not (has-v128? scalar)) "scalar build has no v128")
      ;; functional reference via the scalar build (Chicory can run it)
      (let [inst (instantiate (:bytes scalar)) mem (.memory inst) n 100]
        (dotimes [i n] (.writeF64 mem (* 8 i) (double i)) (.writeF64 mem (* 8 (+ n i)) (double (* 10 i))))
        (.apply (.export inst "vadd") (long-array [0 (* 8 n) (* 8 (* 2 n)) n]))
        (let [bad (count (filter #(not= (.readDouble mem (* 8 (+ (* 2 n) %))) (double (* 11 %))) (range n)))]
          (is (zero? bad) (str bad " vadd mismatches")))))))

(deftm vaddf-simd! [a :- (Array float), b :- (Array float), out :- (Array float), n :- Long] :- nil
  (dotimes [i n]
    (raster.arrays/aset out i (raster.numeric/+ (raster.arrays/aget a i) (raster.arrays/aget b i)))))

(deftest f32-elementwise-scalar-runs
  (testing "f32 elementwise map (scalar): f32.load/f32.add/f32.store, no f64 demote"
    ;; regression for type-aware (float …) casts — previously emitted f32.demote_f64
    ;; over an already-f32 value → validation failure.
    (let [m (pl/compile-wasm #'vaddf-simd! :name "vaddf" :dtype :float)
          inst (instantiate (:bytes m)) mem (.memory inst) n 100]
      (dotimes [i n] (.writeF32 mem (* 4 i) (float i)) (.writeF32 mem (* 4 (+ n i)) (float (* 10 i))))
      (.apply (.export inst "vaddf") (long-array [0 (* 4 n) (* 4 (* 2 n)) n]))
      (let [bad (count (filter #(not= (.readFloat mem (* 4 (+ (* 2 n) %))) (float (* 11 %))) (range n)))]
        (is (zero? bad) (str bad " f32 vadd mismatches"))))))

(deftest f32-elementwise-simd-vectorizes-and-validates
  (testing "f32 elementwise map: SIMD build is valid v128 (f32x4), 4-wide"
    (let [simd (pl/compile-wasm #'vaddf-simd! :name "vaddf" :dtype :float :wasm-simd? true)]
      (is (some? (Parser/parse (:bytes simd))))
      (is (has-v128? simd) "f32 vectorizer emitted v128 instructions"))))

;; ── value-type boundary (defvalue-in / defvalue-out deftm → wasm) ───────────
;; A scalar value-type param expands to per-field scalar params; a (->Type …)
;; return becomes a wasm multi-value return. This is the per-element value-type
;; path that lets defvalue flow over the cljs↔wasm boundary (cross-platform games).
(raster.core/defvalue WV2 [x :- Double, y :- Double])

(deftm wv2-move [s :- WV2, dx :- Double, dy :- Double] :- WV2
  (->WV2 (raster.numeric/+ (.x s) dx) (raster.numeric/+ (.y s) dy)))

(deftest value-type-in-out-multivalue
  (testing "value-type param → per-field scalars; value-type return → multi-value"
    (let [m (pl/compile-wasm #'wv2-move :name "wv2move")]
      (is (= [:f64 :f64 :f64 :f64] (:param-types m)))   ; s.x s.y dx dy
      (is (= [:f64 :f64] (:result-types m)))            ; (->WV2 …) multi-value
      (let [inst (instantiate (:bytes m))
            r (.apply (.export inst "wv2move")
                      (long-array [(Double/doubleToRawLongBits 1.0) (Double/doubleToRawLongBits 2.0)
                                   (Double/doubleToRawLongBits 10.0) (Double/doubleToRawLongBits 20.0)]))]
        (is (= 2 (alength r)))
        (is (= 11.0 (Double/longBitsToDouble (aget r 0))))
        (is (= 22.0 (Double/longBitsToDouble (aget r 1))))))))

;; Transcendentals — wasm has no sin/cos/exp opcode; they lower to an inline
;; polynomial (backend.wasm.transcendental): sin/cos floor-reduce + degree-9
;; minimax; exp = x/1024 + Taylor + 10 squarings; tan = sin/cos. tan/cos exercise
;; the infer-vt fix (a let* used as a `/` operand must infer its own tail's vt).
(deftm sin-k [x :- Double] :- Double (raster.math/sin x))
(deftm cos-k [x :- Double] :- Double (raster.math/cos x))
(deftm tan-k [x :- Double] :- Double (raster.math/tan x))
(deftm exp-k [x :- Double] :- Double (raster.math/exp x))
(deftm log-k [x :- Double] :- Double (raster.math/log x))
(deftm pow-k [x :- Double, y :- Double] :- Double (raster.math/pow x y))
(deftm fma-k [x :- Double, y :- Double, z :- Double] :- Double (raster.math/fma x y z))

(defn- call1 [inst nm x]
  (Double/longBitsToDouble (aget (.apply (.export inst nm) (long-array [(Double/doubleToRawLongBits x)])) 0)))
(defn- calln [inst nm xs]
  (Double/longBitsToDouble (aget (.apply (.export inst nm) (long-array (map #(Double/doubleToRawLongBits %) xs))) 0)))

(deftest transcendentals-inline-polynomial
  (testing "sin/cos/tan/exp/log/pow/fma emit valid wasm matching Math (poly accuracy)"
    (let [si (instantiate (:bytes (pl/compile-wasm #'sin-k :name "sin")))
          ci (instantiate (:bytes (pl/compile-wasm #'cos-k :name "cos")))
          ti (instantiate (:bytes (pl/compile-wasm #'tan-k :name "tan")))
          ei (instantiate (:bytes (pl/compile-wasm #'exp-k :name "exp")))
          li (instantiate (:bytes (pl/compile-wasm #'log-k :name "log")))
          pi (instantiate (:bytes (pl/compile-wasm #'pow-k :name "pow")))
          fi (instantiate (:bytes (pl/compile-wasm #'fma-k :name "fma")))]
      (doseq [x [0.0 0.5 1.0 2.0 3.0 -1.0 -2.5 5.0 10.0 -7.3]]
        (is (< (Math/abs (- (call1 si "sin" x) (Math/sin x))) 5e-5) (str "sin " x))
        (is (< (Math/abs (- (call1 ci "cos" x) (Math/cos x))) 5e-5) (str "cos " x))
        (is (< (Math/abs (/ (- (call1 ei "exp" x) (Math/exp x)) (Math/exp x))) 1e-9) (str "exp " x)))
      ;; tan away from its poles
      (doseq [x [0.0 0.5 1.0 -1.0 2.0 3.0 -2.5]]
        (is (< (Math/abs (- (call1 ti "tan" x) (Math/tan x))) 1e-4) (str "tan " x)))
      ;; log (x>0)
      (doseq [x [0.001 0.5 1.0 2.0 2.718281828 10.0 100.0 1000.0]]
        (is (< (Math/abs (- (call1 li "log" x) (Math/log x))) 1e-9) (str "log " x)))
      ;; pow (x>0) and fma — multi-arg poly lowering
      (doseq [[x y] [[2.0 10.0] [2.0 0.5] [10.0 3.0] [1.5 2.5] [3.0 -2.0] [100.0 0.25]]]
        (is (< (Math/abs (/ (- (calln pi "pow" [x y]) (Math/pow x y)) (Math/pow x y))) 1e-9)
            (str "pow " x " " y)))
      (doseq [[x y z] [[2.0 3.0 1.0] [-1.5 4.0 0.5] [10.0 0.1 -2.0]]]
        (is (< (Math/abs (- (calln fi "fma" [x y z]) (+ (* x y) z))) 1e-12) (str "fma " x y z))))))

;; Broader elementary-math set — all lower to wasm via composition/polynomial
;; (backend.wasm.transcendental), exercising the emitter's generality: value-
;; position if (atan reduction, atan2 quadrants, cbrt/signum sign), abs/sqrt/min/max.
(deftm asin-k [x :- Double] :- Double (raster.math/asin x))
(deftm atan-k [x :- Double] :- Double (raster.math/atan x))
(deftm atan2-k [y :- Double, x :- Double] :- Double (raster.math/atan2 y x))
(deftm sinh-k [x :- Double] :- Double (raster.math/sinh x))
(deftm tanh-k [x :- Double] :- Double (raster.math/tanh x))
(deftm asinh-k [x :- Double] :- Double (raster.math/asinh x))
(deftm cbrt-k [x :- Double] :- Double (raster.math/cbrt x))
(deftm log10-k [x :- Double] :- Double (raster.math/log10 x))
(deftm exp2-k [x :- Double] :- Double (raster.math/exp2 x))
(deftm hypot-k [x :- Double, y :- Double] :- Double (raster.math/hypot x y))
(deftm clamp-k [x :- Double, lo :- Double, hi :- Double] :- Double (raster.math/clamp x lo hi))
(deftm ceil-k [x :- Double] :- Double (raster.math/ceil x))
(deftm trunc-k [x :- Double] :- Double (raster.math/trunc x))
(deftm signum-k [x :- Double] :- Double (raster.math/signum x))

(deftest elementary-math-inline
  (testing "inverse/hyperbolic/log-family/rounding/sign all emit valid wasm vs Math"
    (let [as (instantiate (:bytes (pl/compile-wasm #'asin-k :name "asin")))
          at (instantiate (:bytes (pl/compile-wasm #'atan-k :name "atan")))
          a2 (instantiate (:bytes (pl/compile-wasm #'atan2-k :name "atan2")))
          sh (instantiate (:bytes (pl/compile-wasm #'sinh-k :name "sinh")))
          th (instantiate (:bytes (pl/compile-wasm #'tanh-k :name "tanh")))
          ah (instantiate (:bytes (pl/compile-wasm #'asinh-k :name "asinh")))
          cb (instantiate (:bytes (pl/compile-wasm #'cbrt-k :name "cbrt")))
          lt (instantiate (:bytes (pl/compile-wasm #'log10-k :name "log10")))
          e2 (instantiate (:bytes (pl/compile-wasm #'exp2-k :name "exp2")))
          hy (instantiate (:bytes (pl/compile-wasm #'hypot-k :name "hypot")))
          cl (instantiate (:bytes (pl/compile-wasm #'clamp-k :name "clamp")))
          ce (instantiate (:bytes (pl/compile-wasm #'ceil-k :name "ceil")))
          tr (instantiate (:bytes (pl/compile-wasm #'trunc-k :name "trunc")))
          sg (instantiate (:bytes (pl/compile-wasm #'signum-k :name "signum")))]
      (doseq [x [-0.9 -0.3 0.0 0.4 0.9]]
        (is (< (Math/abs (- (call1 as "asin" x) (Math/asin x))) 5e-5) (str "asin " x)))
      (doseq [x [-5.0 -1.0 0.3 3.0 20.0]]
        (is (< (Math/abs (- (call1 at "atan" x) (Math/atan x))) 5e-5) (str "atan " x)))
      ;; atan2 across all quadrants + axes
      (doseq [[y x] [[1.0 1.0] [1.0 -1.0] [-1.0 -1.0] [-1.0 1.0] [1.0 0.0] [-1.0 0.0]]]
        (is (< (Math/abs (- (calln a2 "atan2" [y x]) (Math/atan2 y x))) 5e-5) (str "atan2 " y " " x)))
      (doseq [x [-3.0 -0.5 0.0 0.5 3.0]]
        (is (< (Math/abs (- (call1 sh "sinh" x) (Math/sinh x))) 1e-9) (str "sinh " x))
        (is (< (Math/abs (- (call1 th "tanh" x) (Math/tanh x))) 1e-9) (str "tanh " x))
        (is (< (Math/abs (- (call1 ah "asinh" x) (Math/log (+ x (Math/sqrt (+ (* x x) 1.0)))))) 1e-9) (str "asinh " x)))
      ;; cbrt incl. negative and the zero guard
      (doseq [x [27.0 -8.0 0.0 1000.0 -0.001]]
        (is (< (Math/abs (- (call1 cb "cbrt" x) (Math/cbrt x))) 1e-6) (str "cbrt " x)))
      (doseq [x [0.5 10.0 1000.0]]
        (is (< (Math/abs (- (call1 lt "log10" x) (Math/log10 x))) 1e-9) (str "log10 " x)))
      (doseq [x [-3.0 0.0 10.0]]
        (is (< (Math/abs (/ (- (call1 e2 "exp2" x) (Math/pow 2.0 x)) (Math/pow 2.0 x))) 1e-9) (str "exp2 " x)))
      (is (< (Math/abs (- (calln hy "hypot" [3.0 4.0]) 5.0)) 1e-9))
      (is (= 3.0 (calln cl "clamp" [5.0 0.0 3.0])))
      (is (= 0.0 (calln cl "clamp" [-1.0 0.0 3.0])))
      (is (= 2.0 (calln cl "clamp" [2.0 0.0 3.0])))
      (is (= 3.0 (call1 ce "ceil" 2.1)))
      (is (= -2.0 (call1 ce "ceil" -2.1)))
      (is (= 2.0 (call1 tr "trunc" 2.9)))
      (is (= -2.0 (call1 tr "trunc" -2.9)))
      (is (= -1.0 (call1 sg "signum" -5.0)))
      (is (= 1.0 (call1 sg "signum" 5.0)))
      (is (= 0.0 (call1 sg "signum" 0.0))))))

;; R5b wasm emitter extensions (driven by raster.noise): integer bitwise ops +
;; shifts, value-position `case` (gradient-hash dispatch), unary minus, and a
;; `loop` in value position (a reduction bound in a let).
(deftm bitops-k [a :- Long, b :- Long] :- Long
  (let [x (bit-and a b) y (bit-or a b) z (bit-xor a b)
        s (bit-shift-left a 2) r (bit-shift-right b 1)]
    (clojure.core/+ x (clojure.core/+ y (clojure.core/+ z (clojure.core/+ s r))))))

(deftm grad-case-k [h :- Long, x :- Double, y :- Double] :- Double
  (case (int (bit-and h 3))
    0 (raster.numeric/+ x y)
    1 (raster.numeric/+ (raster.numeric/- x) y)
    2 (raster.numeric/- x y)
    3 (raster.numeric/- (raster.numeric/- x) y)))

(deftm vloop-k [n :- Long] :- Double
  (let [s (loop [i 0 acc 0.0]
            (if (clojure.core/= (long i) (long n)) acc
                (recur (clojure.core/inc (long i)) (raster.numeric/+ acc 1.0))))]
    (raster.numeric/* s 2.0)))

(deftest wasm-bitwise-case-vloop
  (testing "bitwise + shifts"
    (let [inst (instantiate (:bytes (pl/compile-wasm #'bitops-k :name "bit")))]
      (doseq [[a b] [[12 10] [255 3] [7 1]]]
        (let [got (aget (.apply (.export inst "bit") (long-array [a b])) 0)
              exp (+ (bit-and a b) (bit-or a b) (bit-xor a b) (bit-shift-left a 2) (bit-shift-right b 1))]
          (is (= exp got) (str "bitops " a " " b))))))
  (testing "value-position case + unary minus (grad2d shape) vs reference"
    (let [inst (instantiate (:bytes (pl/compile-wasm #'grad-case-k :name "gc")))
          ref (fn [h x y] (case (bit-and h 3) 0 (+ x y) 1 (+ (- x) y) 2 (- x y) 3 (- (- x) y)))]
      (doseq [h [0 1 2 3 7 10]]
        (let [got (Double/longBitsToDouble (aget (.apply (.export inst "gc")
                                                         (long-array [h (Double/doubleToRawLongBits 0.5) (Double/doubleToRawLongBits 0.7)])) 0))]
          (is (< (Math/abs (- got (ref h 0.5 0.7))) 1e-12) (str "grad-case h=" h))))))
  (testing "loop in value position (let-bound reduction)"
    (let [inst (instantiate (:bytes (pl/compile-wasm #'vloop-k :name "vl")))]
      (doseq [n [0 1 5 17]]
        (let [got (Double/longBitsToDouble (aget (.apply (.export inst "vl") (long-array [n])) 0))]
          (is (= (* 2.0 n) got) (str "vloop n=" n)))))))

;; Generic macroexpansion (Step 1): the wasm backend macroexpands at the last
;; moment, so arbitrary core macros (cond / when / when-let) lower to the ~12
;; special forms with no per-macro emitter handler. `dotimes` is kept un-expanded
;; for the SIMD vectorizer.
(deftm classify-k [x :- Double] :- Double
  (cond
    (clojure.core/< x 0.0) -1.0
    (clojure.core/< x 1.0) 0.0
    :else 1.0))

(deftm clamp-neg-k! [a :- (Array double), n :- Long] :- nil
  ;; when inside dotimes body (effect position) → if; dotimes itself stays un-expanded
  (dotimes [i n]
    (when (clojure.core/< (raster.arrays/aget a i) 0.0)
      (raster.arrays/aset a i 0.0))))

(deftest wasm-macroexpand-cond-when
  (testing "cond → nested-if value"
    (let [inst (instantiate (:bytes (pl/compile-wasm #'classify-k :name "cl")))
          ref  (fn [x] (cond (< x 0.0) -1.0 (< x 1.0) 0.0 :else 1.0))]
      (doseq [x [-2.5 -0.0 0.5 0.999 1.0 7.0]]
        (let [got (Double/longBitsToDouble
                   (aget (.apply (.export inst "cl") (long-array [(Double/doubleToRawLongBits x)])) 0))]
          (is (= (ref x) got) (str "classify x=" x))))))
  (testing "when inside dotimes → conditional store (void kernel)"
    (let [inst (instantiate (:bytes (pl/compile-wasm #'clamp-neg-k! :name "cn")))
          mem  (.memory inst)
          xs   [-1.0 2.0 -3.5 0.0 4.0]
          n    (count xs)]
      (dotimes [i n] (.writeF64 mem (* 8 i) (double (nth xs i))))
      (.apply (.export inst "cn") (long-array [0 n]))
      (doseq [i (range n)]
        (is (= (max 0.0 (nth xs i)) (.readDouble mem (* 8 i))) (str "clamp i=" i))))))

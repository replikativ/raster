(ns raster.compiler.backend.wasm.emit-test
  "Track A — differential + validity tests for the JVM-side wasm backend.
   Emits .wasm from a real deftm's post-pass IR, then uses Chicory (a pure-JVM
   WebAssembly runtime) to (a) PARSE it — which validates the module, malformed
   bytes throw — and (b) EXECUTE it, asserting results match a reference.
   No node / native deps; runs in `clojure -M:test`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric]
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

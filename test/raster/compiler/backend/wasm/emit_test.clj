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

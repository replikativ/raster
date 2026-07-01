(ns raster.compiler.backend.cpu.aot-test
  "Differential tests for the monolithic CPU-C AOT backend (compile-aot :target :c).
   Each deftm is compiled to a single native C function via clang + Panama and its
   output asserted against the lazy-JIT reference. Guarded on clang availability so
   `clojure -M:test` skips cleanly where no C compiler is present.

   Covers the compiler-quality fixes landed alongside fused norm+quant:
   - multi-output kernels (a deftm returning [q scales] -> two native buffers)
   - int8/byte array output (per-array element types)
   - block reductions (per-block max-abs) as nested loops
   - index arithmetic bound to a name (base = b*32) typed long, not float
   - Math/round half-up semantics (floor(x+0.5)) matching Java on negative .5"
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm reduce!]]
            [raster.numeric :as rn]
            [raster.arrays :as ra]
            [raster.math]
            [raster.compiler.backend.cpu.aot :as aot]))

(defn- clang-available? []
  (try
    (let [cc (or (System/getenv "RASTER_CC") "clang")
          p (-> (ProcessBuilder. ^java.util.List [cc "--version"])
                (.redirectErrorStream true) (.start))]
      (.waitFor p) (zero? (.exitValue p)))
    (catch Exception _ false)))

;; ---- kernels under test (proper raster style: rn/ arithmetic, ra/ arrays) ----

;; int8 output: scaled int8 (narrowing store), exercises byte buffers. Double
;; dtype so the result is bit-exact to the lazy-JIT reference (f32 mode computes
;; in float by design and is tolerance-matched, not bit-exact — tested separately).
(deftm toi8-k [x :- (Array double) n :- Long] :- (Array byte)
  (let [out (clojure.core/byte-array n)]
    (dotimes [i n] (ra/aset out i (long (rn/* (ra/aget x i) 100.0))))
    out))

;; per-block (B=32) symmetric int8 quant returning BOTH q (int8) and scales (double).
;; Exercises: block reduction (max-abs), bound index (base), Math/round, multi-output.
(deftm quant-k [x :- (Array double) nblk :- Long]
  (let [d (clojure.core/* nblk 32)
        q (clojure.core/byte-array d)
        scales (clojure.core/double-array nblk)]
    (dotimes [b nblk]
      (let [base (clojure.core/* b 32)
            mx (loop [j 0 a 0.0]
                 (if (clojure.core/< j 32)
                   (recur (clojure.core/inc j)
                          (rn/max a (rn/abs (ra/aget x (clojure.core/+ base j)))))
                   a))
            s (rn// mx 127.0)
            id (rn// 1.0 s)]
        (ra/aset scales b s)
        (dotimes [k 32]
          (ra/aset q (clojure.core/+ base k)
                   (long (rn/max -127.0
                                 (rn/min 127.0
                                         (double (Math/round (rn/* (ra/aget x (clojure.core/+ base k)) id))))))))))
    [q scales]))

(deftest cpu-c-int8-output
  (when (clang-available?)
    (testing "double -> int8 narrowing store matches lazy-JIT bit-exact"
      (let [n 16
            x (double-array (map #(clojure.core/- (/ (double %) 10.0) 0.8) (range n)))
            cfn (aot/compile-aot-c #'toi8-k :double)]
        (is (= (seq (toi8-k x n)) (seq (cfn x n))))))))

(deftest cpu-c-multi-output-blockquant
  (when (clang-available?)
    (testing "norm+blockquant returns [q scales]; both bit-exact (double path)"
      (let [x (double-array (map #(clojure.core/- (/ (double %) 7.0) 4.0) (range 64)))
            [rq rs] (quant-k x 2)
            cfn (aot/compile-aot-c #'quant-k :double)
            result (cfn x 2)]
        (is (vector? result) "multi-output kernel returns a vector of buffers")
        (let [[cq cs] result]
          (is (= (seq rq) (seq cq)) "int8 quantized output bit-exact")
          (is (= (seq rs) (seq cs)) "double per-block scales bit-exact"))))))

;; ---- composed deftms must fuse into ONE C function (the rms-norm ∘ quant case) ----

(deftm rmsnorm-k [x :- (Array double) w :- (Array double) n :- Long eps :- Double] :- (Array double)
  (let [out (clojure.core/double-array n)
        ms (loop [i 0 s 0.0] (if (clojure.core/< i n)
                               (recur (clojure.core/inc i) (rn/+ s (rn/* (ra/aget x i) (ra/aget x i))))
                               (rn// s (double n))))
        inv (rn// 1.0 (rn/sqrt (rn/+ ms eps)))]
    (dotimes [i n] (ra/aset out i (rn/* (rn/* (ra/aget x i) inv) (ra/aget w i))))
    out))

;; reuse quant-k (above) as the quantizer; compose: normalize then block-quantize.
(deftm norm-quant-k [x :- (Array double) w :- (Array double) n :- Long nblk :- Long eps :- Double]
  (quant-k (rmsnorm-k x w n eps) nblk))

(deftest cpu-c-composed-deftms-fuse
  (when (clang-available?)
    (testing "rms-norm ∘ quant fuses into ONE C function; double eps param; tag-typed locals"
      (let [n 64 nblk 2 eps 1.0e-6
            x (double-array (map #(clojure.core/- (/ (double %) 9.0) 3.0) (range n)))
            w (double-array (repeat n 1.3))
            [rq rs] (norm-quant-k x w n nblk eps)
            cfn (aot/compile-aot-c #'norm-quant-k :double)
            src (:c-source (meta cfn))
            [cq cs] (cfn x w n nblk eps)]
        (is (= 1 (count (re-seq (re-pattern "void ") src)))
            "the composition is a single fused C function, not two")
        (is (clojure.string/includes? src "double eps")
            "the Double scalar param is declared double, not int")
        (is (= (seq rs) (seq cs)) "per-block scales bit-exact")
        ;; q is within 1 ULP of the reference: -ffast-math reassociation in the
        ;; reduction can shift a value across a rounding boundary (inference-grade,
        ;; same as llama.cpp), so allow an off-by-one, never more.
        (is (every? #(<= (Math/abs (- (int (first %)) (int (second %)))) 1)
                    (map vector rq cq))
            "int8 output within 1 ULP of the lazy-JIT reference")))))

(deftest cpu-c-round-half-up
  (when (clang-available?)
    (testing "Math/round is half-up (floor(x+0.5)): negative .5 matches Java, not C round()"
      ;; pick values so y/d_b lands on a negative half-integer (round(-63.5): Java -63)
      (let [x (double-array (map #(clojure.core/- (/ (double %) 7.0) 4.0) (range 64)))
            [rq _] (quant-k x 2)
            cfn (aot/compile-aot-c #'quant-k :double)
            [cq _] (cfn x 2)]
        (is (= (seq rq) (seq cq))
            "no off-by-one from round-half-away-from-zero on negatives")))))

;; ---- #27: explicit C-SIMD reduction (compile-aot-c :simd? true) ----

;; rms-norm with the variance reduction expressed as a reduce! SOAC (par/reduce),
;; so :simd? true PRESERVES it and emits an AVX2 __m256 FMA loop (via csimd)
;; instead of a scalar loop left to clang auto-vec. Array output + reduction
;; intermediate = the real target shape (rms-norm / quant-GEMV).
(deftm rmsnorm-red [x :- (Array double) w :- (Array double) n :- Long eps :- Double] :- (Array double)
  (let [out (clojure.core/double-array n)
        s (reduce! [acc 0.0] [x] (rn/+ acc (rn/* x x)))
        ms (rn// s (double n))
        inv (rn// 1.0 (rn/sqrt (rn/+ ms eps)))]
    (dotimes [i n] (ra/aset out i (rn/* (rn/* (ra/aget x i) inv) (ra/aget w i))))
    out))

(deftest cpu-c-simd-reduction
  (when (clang-available?)
    (testing ":simd? true emits an __m256 FMA reduction and matches the scalar path + interpreter"
      (let [f-simd (aot/compile-aot-c #'rmsnorm-red :double :simd? true)
            f-scal (aot/compile-aot-c #'rmsnorm-red :double)]
        (is (re-find #"_mm256_fmadd_pd" (:c-source (meta f-simd)))
            ":simd? true lowers the reduce! to a vector FMA loop")
        (is (not (re-find #"_mm256_fmadd_pd" (:c-source (meta f-scal))))
            ":simd? false stays scalar (clang auto-vec only)")
        (doseq [n [16 64 257 1000]]
          (let [eps 1.0e-6
                x (double-array (map #(clojure.core/- (/ (double %) 9.0) 3.0) (range n)))
                w (double-array (repeat n 1.3))
                ref (rmsnorm-red x w n eps)
                os  (f-simd x w n eps)
                oc  (f-scal x w n eps)]
            (is (every? true? (map #(< (Math/abs (clojure.core/- %1 %2)) 1e-9) os ref))
                (str "n=" n " simd == interpreter"))
            (is (every? true? (map #(< (Math/abs (clojure.core/- %1 %2)) 1e-9) os oc))
                (str "n=" n " simd == scalar-C"))))))))

;; DP5 Lorenz ODE benchmark — all JVM implementations
;;
;; Run with Valhalla JDK:
;;   source valhalla-env.sh
;;   OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \
;;     -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
;;     -J--enable-native-access=ALL-UNNAMED -M:valhalla \
;;     bench/raster/ode_bench.clj
;;
;; Compares four implementations of the same algorithm:
;;   1. Clojure — type-hinted, no boxing, idiomatic numeric Clojure
;;   2. Raster lazy JIT — deftm with BC compilation on first call
;;   3. Raster compile-aot — full AOT inlining into one method
;;   4. Java — pure Java reference (run separately, see dev/DP5Java.java)

(require '[raster.core :refer [deftm ftm]])
(require '[raster.numeric :as n])
(require '[raster.arrays :refer [alength aclone acopy!]])
(require '[raster.ode.core :as ode])
(require '[raster.compiler.pipeline :as pipeline])

;; ================================================================
;; Benchmark harness
;; ================================================================

(defn bench
  "Benchmark f with n-warmup warmup calls and n-measure measurements.
  Returns {:best :p10 :median :p90 :mean} in microseconds."
  [f n-warmup n-measure]
  (dotimes [_ n-warmup] (f))
  (System/gc)
  (Thread/sleep 2000)
  (let [times (long-array n-measure)]
    (dotimes [i n-measure]
      (let [t0 (System/nanoTime)]
        (f)
        (aset times i (- (System/nanoTime) t0))))
    (java.util.Arrays/sort times)
    (let [us #(/ (double (aget times %)) 1000.0)]
      {:best   (us 0)
       :p10    (us (int (* n-measure 0.1)))
       :median (us (int (* n-measure 0.5)))
       :p90    (us (int (* n-measure 0.9)))
       :mean   (/ (reduce + (map #(/ (double %) 1000.0) times)) n-measure)})))

(defn report [label result]
  (println (format "  %-35s best:%5.0f  p10:%5.0f  med:%5.0f  p90:%5.0f  mean:%5.0f"
                   label (:best result) (:p10 result) (:median result)
                   (:p90 result) (:mean result)))
  result)

;; ================================================================
;; 1. Clojure type-hinted DP5
;; ================================================================

(defn clj-lorenz! [^doubles du ^doubles u ^double t]
  (let [x (aget u 0) y (aget u 1) z (aget u 2)
        sigma 10.0 rho 28.0 beta (/ 8.0 3.0)]
    (aset du 0 (* sigma (- y x)))
    (aset du 1 (- (* x (- rho z)) y))
    (aset du 2 (- (* x y) (* beta z)))))

;; DP5 Butcher tableau
(def ^:const ^double dp5-a21 0.2)
(def ^:const ^double dp5-a31 0.075)  (def ^:const ^double dp5-a32 0.225)
(def ^:const ^double dp5-a41 (/ 44.0 45.0))  (def ^:const ^double dp5-a42 (/ -56.0 15.0))  (def ^:const ^double dp5-a43 (/ 32.0 9.0))
(def ^:const ^double dp5-a51 (/ 19372.0 6561.0))  (def ^:const ^double dp5-a52 (/ -25360.0 2187.0))
(def ^:const ^double dp5-a53 (/ 64448.0 6561.0))  (def ^:const ^double dp5-a54 (/ -212.0 729.0))
(def ^:const ^double dp5-a61 (/ 9017.0 3168.0))  (def ^:const ^double dp5-a62 (/ -355.0 33.0))
(def ^:const ^double dp5-a63 (/ 46732.0 5247.0))  (def ^:const ^double dp5-a64 (/ 49.0 176.0))  (def ^:const ^double dp5-a65 (/ -5103.0 18656.0))
(def ^:const ^double dp5-b1 (/ 35.0 384.0))  (def ^:const ^double dp5-b3 (/ 500.0 1113.0))
(def ^:const ^double dp5-b4 (/ 125.0 192.0))  (def ^:const ^double dp5-b5 (/ -2187.0 6784.0))  (def ^:const ^double dp5-b6 (/ 11.0 84.0))
(def ^:const ^double dp5-e1 (/ 71.0 57600.0))  (def ^:const ^double dp5-e3 (/ -71.0 16695.0))
(def ^:const ^double dp5-e4 (/ 71.0 1920.0))  (def ^:const ^double dp5-e5 (/ -17253.0 339200.0))
(def ^:const ^double dp5-e6 (/ 22.0 525.0))  (def ^:const ^double dp5-e7 (/ -1.0 40.0))
(def ^:const ^double dp5-c2 0.2)  (def ^:const ^double dp5-c3 0.3)  (def ^:const ^double dp5-c4 0.8)
(def ^:const ^double dp5-c5 (/ 8.0 9.0))

(defn clj-dp5-step! [f ^doubles k1 ^doubles k2 ^doubles k3 ^doubles k4
                     ^doubles k5 ^doubles k6 ^doubles k7 ^doubles tmp ^doubles utilde
                     ^doubles u t dt atol rtol]
  (let [t (double t) dt (double dt) atol (double atol) rtol (double rtol)
        n (int (alength u))]
    (f k1 u t)
    (dotimes [i n] (aset tmp i (+ (aget u i) (* dt dp5-a21 (aget k1 i)))))
    (f k2 tmp (+ t (* dp5-c2 dt)))
    (dotimes [i n] (aset tmp i (+ (aget u i) (* dt (Math/fma dp5-a32 (aget k2 i) (* dp5-a31 (aget k1 i)))))))
    (f k3 tmp (+ t (* dp5-c3 dt)))
    (dotimes [i n] (aset tmp i (+ (aget u i) (* dt (Math/fma dp5-a43 (aget k3 i) (Math/fma dp5-a42 (aget k2 i) (* dp5-a41 (aget k1 i))))))))
    (f k4 tmp (+ t (* dp5-c4 dt)))
    (dotimes [i n] (aset tmp i (+ (aget u i) (* dt (Math/fma dp5-a54 (aget k4 i) (Math/fma dp5-a53 (aget k3 i) (Math/fma dp5-a52 (aget k2 i) (* dp5-a51 (aget k1 i)))))))))
    (f k5 tmp (+ t (* dp5-c5 dt)))
    (dotimes [i n] (aset tmp i (+ (aget u i) (* dt (Math/fma dp5-a65 (aget k5 i) (Math/fma dp5-a64 (aget k4 i) (Math/fma dp5-a63 (aget k3 i) (Math/fma dp5-a62 (aget k2 i) (* dp5-a61 (aget k1 i))))))))))
    (f k6 tmp (+ t dt))
    (dotimes [i n] (aset utilde i (+ (aget u i) (* dt (Math/fma dp5-b6 (aget k6 i) (Math/fma dp5-b5 (aget k5 i) (Math/fma dp5-b4 (aget k4 i) (Math/fma dp5-b3 (aget k3 i) (* dp5-b1 (aget k1 i))))))))))
    (f k7 utilde (+ t dt))
    (let [err (loop [i (int 0) acc 0.0]
                (if (>= i n) acc
                    (let [ei (* dt (+ (* dp5-e1 (aget k1 i)) (* dp5-e3 (aget k3 i)) (* dp5-e4 (aget k4 i))
                                      (* dp5-e5 (aget k5 i)) (* dp5-e6 (aget k6 i)) (* dp5-e7 (aget k7 i))))
                          sc (+ atol (* rtol (Math/max (Math/abs (aget u i)) (Math/abs (aget utilde i)))))
                          r (/ ei sc)]
                      (recur (unchecked-inc-int i) (Math/fma r r acc)))))]
      (Math/sqrt (/ err (double n))))))

(defn clj-solve-dp5 [f ^doubles u0 t0 tf dt0 atol rtol]
  (let [t0 (double t0) tf (double tf) dt0 (double dt0)
        atol (double atol) rtol (double rtol)
        n (long (alength u0))
        u (aclone u0) u-save (double-array n)
        k1 (double-array n) k2 (double-array n) k3 (double-array n)
        k4 (double-array n) k5 (double-array n) k6 (double-array n)
        k7 (double-array n) tmp (double-array n) utilde (double-array n)]
    (loop [t t0, dt dt0, nreject (int 0)]
      (if (>= t tf)
        (aclone u)
        (let [dt-actual (Math/min dt (- tf t))
              _ (System/arraycopy u 0 u-save 0 n)
              err (double (clj-dp5-step! f k1 k2 k3 k4 k5 k6 k7 tmp utilde u t dt-actual atol rtol))]
          (if (<= err 1.0)
            (let [_ (System/arraycopy utilde 0 u 0 n)
                  factor (Math/min 5.0 (Math/max 0.2 (* 0.9 (Math/pow err -0.2))))]
              (recur (+ t dt-actual) (* dt-actual factor) nreject))
            (let [factor (Math/max 0.2 (* 0.9 (Math/pow err -0.2)))
                  _ (System/arraycopy u-save 0 u 0 n)]
              (recur t (* dt-actual factor) (unchecked-inc-int nreject)))))))))

;; ================================================================
;; 2. Raster deftm (lazy JIT)
;; ================================================================

(deftm solve-dp5-deftm
  [f :- (Fn [(Array double) (Array double) Double]),
   u0 :- (Array double),
   t0 :- Double, tf :- Double, dt0 :- Double,
   atol :- Double, rtol :- Double]
  (let [n-dim (alength u0)
        cache (ode/make-cache (ode/->DP5 atol rtol) u0)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)]
    (acopy! u0 0 u 0 n-dim)
    (loop [t t0, dt-loop dt0, nreject (int 0)]
      (if (>= t tf)
        (aclone u)
        (let [dt-actual (n/min dt-loop (- tf t))
              _ (acopy! u 0 u-save 0 n-dim)
              err (double (ode/dp5-step! cache f u t dt-actual))]
          (if (<= err 1.0)
            (let [_ (acopy! utilde 0 u 0 n-dim)
                  t-new (+ t dt-actual)
                  factor (n/min 5.0 (n/max 0.2 (* 0.9 (n/pow err -0.2))))]
              (recur t-new (* dt-actual factor) nreject))
            (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.2)))
                  _ (acopy! u-save 0 u 0 n-dim)]
              (recur t (* dt-actual factor) (unchecked-inc-int nreject)))))))))

;; ================================================================
;; 3. Raster compile-aot (AOT inlining)
;; ================================================================

(deftm solve-dp5-aot
  [f :- (Fn [(Array double) (Array double) Double]),
   u0 :- (Array double),
   t0 :- Double, tf :- Double, dt0 :- Double,
   atol :- Double, rtol :- Double]
  (let [n-dim (alength u0)
        cache (ode/make-cache (ode/->DP5 atol rtol) u0)
        u (n/similar u0)
        u-save (n/similar u0)
        utilde (:utilde cache)]
    (acopy! u0 0 u 0 n-dim)
    (loop [t t0, dt-loop dt0, nreject (int 0)]
      (if (>= t tf)
        (aclone u)
        (let [dt-actual (n/min dt-loop (- tf t))
              _ (acopy! u 0 u-save 0 n-dim)
              err (double (ode/dp5-step! cache f u t dt-actual))]
          (if (<= err 1.0)
            (let [_ (acopy! utilde 0 u 0 n-dim)
                  t-new (+ t dt-actual)
                  factor (n/min 5.0 (n/max 0.2 (* 0.9 (n/pow err -0.2))))]
              (recur t-new (* dt-actual factor) nreject))
            (let [factor (n/max 0.2 (* 0.9 (n/pow err -0.2)))
                  _ (acopy! u-save 0 u 0 n-dim)]
              (recur t (* dt-actual factor) (unchecked-inc-int nreject)))))))))

;; ================================================================
;; Run benchmarks
;; ================================================================

(deftm lorenz-ftm [du :- (Array double), u :- (Array double), t :- Double]
    (let [x (aget u 0) y (aget u 1) z (aget u 2)
          sigma 10.0 rho 28.0 beta (/ 8.0 3.0)]
      (aset du 0 (* sigma (- y x)))
      (aset du 1 (- (* x (- rho z)) y))
      (aset du 2 (- (* x y) (* beta z)))))

(def u0 (double-array [1.0 0.0 0.0]))

(def n-warmup 10000)
(def n-measure 100)

;; Correctness check (Lorenz is chaotic — verify finite, reasonable magnitude)
(let [r-clj (clj-solve-dp5 clj-lorenz! u0 0.0 100.0 0.01 1e-8 1e-6)
      r-deftm (solve-dp5-deftm lorenz-ftm u0 0.0 100.0 0.01 1e-8 1e-6)]
  (println "Clojure result:" (mapv #(format "%.6f" %) r-clj))
  (println "Raster result: " (mapv #(format "%.6f" %) r-deftm))
  (assert (every? #(and (Double/isFinite %) (< (Math/abs %) 100)) r-clj) "Clojure result not finite")
  (assert (every? #(and (Double/isFinite %) (< (Math/abs %) 100)) r-deftm) "Raster result not finite"))

(println)
(println "=== DP5 Lorenz attractor, t=[0,100], atol=1e-8, rtol=1e-6 ===")
(println (format "    %d warmup iterations, %d measurements" n-warmup n-measure))
(println (format "    JDK: %s" (System/getProperty "java.version")))
(println)

;; 1. Clojure
(report "Clojure (type-hinted)"
  (bench #(clj-solve-dp5 clj-lorenz! u0 0.0 100.0 0.01 1e-8 1e-6)
         n-warmup n-measure))

;; 2. Raster lazy JIT
(report "Raster lazy JIT (deftm)"
  (bench #(solve-dp5-deftm lorenz-ftm u0 0.0 100.0 0.01 1e-8 1e-6)
         n-warmup n-measure))

;; 3. Raster compile-aot
(let [compiled-fn (pipeline/compile-aot #'solve-dp5-aot :simd? false)]
  (report "Raster compile-aot"
    (bench #(compiled-fn lorenz-ftm u0 0.0 100.0 0.01 1e-8 1e-6)
           n-warmup n-measure)))

(println)
(println "  Reference (separate JVM, same hardware):")
(println "    Java DP5:  ~420µs  (dev/DP5Java.java)")
(println "    Julia DP5: ~583µs  (DifferentialEquations.jl)")

(shutdown-agents)

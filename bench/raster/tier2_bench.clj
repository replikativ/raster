;; Tier 2 Comparison Benchmark — Java vs deftm vs AOT vs Julia
;;
;; Run with Valhalla JDK:
;;   source valhalla-env.sh
;;   OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \
;;     -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
;;     -J--enable-native-access=ALL-UNNAMED \
;;     -J-XX:ReservedCodeCacheSize=512m -J-Xmx8g \
;;     -M:bench:valhalla bench/raster/tier2_bench.clj

(require '[raster.core :refer [deftm ftm]])
(require '[raster.numeric :as n])
(require '[raster.arrays :refer [alength aclone acopy!]])
(require '[raster.par :as par])
(require '[raster.compiler.pipeline :as pipeline])

;; ================================================================
;; Harness (same as tier 1)
;; ================================================================

(defn warmup! [f ms]
  (let [deadline (+ (System/nanoTime) (* ms 1000000))]
    (loop [elapsed 0.0]
      (let [t0 (System/nanoTime)] (f) (let [e (/ (- (System/nanoTime) t0) 1e6)]
        (if (< (System/nanoTime) deadline) (recur e) e))))))

(defn bench [f per-call-ms]
  (let [n (max 5 (min 50 (int (/ 5000.0 (max 0.01 per-call-ms)))))
        _ (System/gc) _ (Thread/sleep 200)
        times (long-array n)]
    (dotimes [i n]
      (let [t0 (System/nanoTime)] (f) (aset times i (- (System/nanoTime) t0))))
    (java.util.Arrays/sort times)
    (/ (double (aget times (int (* n 0.5)))) 1e6)))

(defn row [label java deftm aot julia]
  (println (format "  %-35s %8.2f  %8.2f  %8.2f  %8.2f  %5.2fx  %5.2fx  %5.2fx"
                   label java deftm aot julia
                   (/ deftm java) (/ aot java) (/ julia java))))

(defn row-us [label java deftm aot julia]
  (println (format "  %-35s %8.1f  %8.1f  %8.1f  %8.1f  %5.2fx  %5.2fx  %5.2fx"
                   label java deftm aot julia
                   (/ deftm java) (/ aot java) (/ julia java))))

;; ================================================================
;; 1. Jacobi-2D (SOR relaxation)
;; ================================================================

;; Java baseline (primitive type hints)
(defn java-jacobi-2d! [^doubles u ^doubles tmp ^long nn ^long niter]
  (dotimes [_ niter]
    (loop [j (int 1)]
      (when (< j (dec nn))
        (loop [i (int 1)]
          (when (< i (dec nn))
            (aset tmp (+ (* i nn) j)
                  (* 0.25 (+ (aget u (+ (* (dec i) nn) j))
                             (aget u (+ (* (inc i) nn) j))
                             (aget u (+ (* i nn) (dec j)))
                             (aget u (+ (* i nn) (inc j))))))
            (recur (inc i))))
        (recur (inc j))))
    (loop [j (int 1)]
      (when (< j (dec nn))
        (loop [i (int 1)]
          (when (< i (dec nn))
            (aset u (+ (* i nn) j) (aget tmp (+ (* i nn) j)))
            (recur (inc i))))
        (recur (inc j))))))

;; deftm version (plain loops — lazy JIT)
(deftm tm-jacobi-2d! [u :- (Array double), tmp :- (Array double),
                      nn :- Long, niter :- Long]
  (dotimes [_ niter]
    (loop [j (int 1)]
      (when (< j (dec nn))
        (loop [i (int 1)]
          (when (< i (dec nn))
            (aset tmp (+ (* i nn) j)
                  (* 0.25 (+ (aget u (+ (* (dec i) nn) j))
                             (aget u (+ (* (inc i) nn) j))
                             (aget u (+ (* i nn) (dec j)))
                             (aget u (+ (* i nn) (inc j))))))
            (recur (inc i))))
        (recur (inc j))))
    (loop [j (int 1)]
      (when (< j (dec nn))
        (loop [i (int 1)]
          (when (< i (dec nn))
            (aset u (+ (* i nn) j) (aget tmp (+ (* i nn) j)))
            (recur (inc i))))
        (recur (inc j))))))

;; par/map! offset version — row-wise SIMD via AOT
(deftm tm-jacobi-2d-par! [u :- (Array double), tmp :- (Array double),
                          nn :- Long, niter :- Long]
  (let [cols (- nn 2)]
    (dotimes [_ niter]
      (loop [j (int 1)]
        (when (< j (dec nn))
          (let [base (+ (* j nn) 1)]
            (par/map! tmp i cols :offset base double
              (* 0.25 (+ (aget u (+ (* (dec j) nn) 1 i))
                         (aget u (+ (* (inc j) nn) 1 i))
                         (aget u (+ (* j nn) i))
                         (aget u (+ (* j nn) 2 i))))))
          (recur (inc j))))
      (loop [j (int 1)]
        (when (< j (dec nn))
          (let [base (+ (* j nn) 1)]
            (par/map! u i cols :offset base double
              (aget tmp (+ base i))))
          (recur (inc j)))))))

;; ================================================================
;; 2. Lotka-Volterra ODE (RK4 fixed-step)
;; ================================================================

;; Java baseline
(defn java-lotka-volterra! [^doubles du ^doubles u ^doubles p ^double t]
  (let [a (aget p 0) b (aget p 1) c (aget p 2) d (aget p 3)]
    (aset du 0 (- (* a (aget u 0)) (* b (aget u 0) (aget u 1))))
    (aset du 1 (+ (- (* c (aget u 1))) (* d (aget u 0) (aget u 1))))))

;; >4 args: use (double ...) casts inside body instead of ^double hints on params
(defn java-rk4-step! [f u du k2 k3 k4 tmp p t dt]
  (let [^doubles u u ^doubles du du ^doubles k2 k2 ^doubles k3 k3
        ^doubles k4 k4 ^doubles tmp tmp ^doubles p p
        t (double t) dt (double dt)
        nn (clojure.core/alength u)]
    (f du u p t)
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* 0.5 dt (aget du i)))))
    (f k2 tmp p (+ t (* 0.5 dt)))
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* 0.5 dt (aget k2 i)))))
    (f k3 tmp p (+ t (* 0.5 dt)))
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* dt (aget k3 i)))))
    (f k4 tmp p (+ t dt))
    (dotimes [i nn]
      (aset u i (+ (aget u i)
                   (* (/ dt 6.0)
                      (+ (aget du i) (* 2.0 (aget k2 i))
                         (* 2.0 (aget k3 i)) (aget k4 i))))))))

(defn java-solve-rk4 [f u0 p t0 tf dt]
  (let [^doubles u0 u0 ^doubles p p
        t0 (double t0) tf (double tf) dt (double dt)
        u (clojure.core/aclone u0)
        nn (clojure.core/alength u0)
        du (double-array nn) k2 (double-array nn) k3 (double-array nn)
        k4 (double-array nn) tmp (double-array nn)]
    (loop [t t0]
      (when (< t tf)
        (let [dt-actual (Math/min dt (- tf t))]
          (java-rk4-step! f u du k2 k3 k4 tmp p t dt-actual)
          (recur (+ t dt-actual)))))
    u))

;; deftm version
(deftm tm-lotka-volterra! [du :- (Array double), u :- (Array double),
                           p :- (Array double), t :- Double]
  (let [a (aget p 0) b (aget p 1) c (aget p 2) d (aget p 3)]
    (aset du 0 (- (* a (aget u 0)) (* b (aget u 0) (aget u 1))))
    (aset du 1 (+ (- (* c (aget u 1))) (* d (aget u 0) (aget u 1))))))

(deftm tm-rk4-step! [f :- (Fn [(Array double) (Array double) (Array double) Double]),
                     u :- (Array double), du :- (Array double),
                     k2 :- (Array double), k3 :- (Array double),
                     k4 :- (Array double), tmp :- (Array double),
                     p :- (Array double), t :- Double, dt :- Double]
  (let [nn (alength u)]
    (f du u p t)
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* 0.5 dt (aget du i)))))
    (f k2 tmp p (+ t (* 0.5 dt)))
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* 0.5 dt (aget k2 i)))))
    (f k3 tmp p (+ t (* 0.5 dt)))
    (dotimes [i nn] (aset tmp i (+ (aget u i) (* dt (aget k3 i)))))
    (f k4 tmp p (+ t dt))
    (dotimes [i nn]
      (aset u i (+ (aget u i)
                   (* (/ dt 6.0)
                      (+ (aget du i) (* 2.0 (aget k2 i))
                         (* 2.0 (aget k3 i)) (aget k4 i))))))))

(deftm tm-solve-rk4 [f :- (Fn [(Array double) (Array double) (Array double) Double]),
                     u0 :- (Array double), p :- (Array double),
                     t0 :- Double, tf :- Double, dt :- Double] :- (Array double)
  (let [u (aclone u0)
        nn (alength u0)
        du (double-array nn) k2 (double-array nn) k3 (double-array nn)
        k4 (double-array nn) tmp (double-array nn)]
    (loop [t t0]
      (when (< t tf)
        (let [dt-actual (Math/min dt (- tf t))]
          (tm-rk4-step! f u du k2 k3 k4 tmp p t dt-actual)
          (recur (+ t dt-actual)))))
    u))

;; ================================================================
;; 3. FFT (Radix-2 Cooley-Tukey)
;; ================================================================

;; Java baseline
(defn java-fft-radix2! [^doubles re ^doubles im]
  (let [nn (clojure.core/alength re)]
    (when (> nn 1)
      (loop [i (int 0) j (int 0)]
        (when (< i (dec nn))
          (when (< i j)
            (let [tr (aget re i) ti (aget im i)]
              (aset re i (aget re j)) (aset im i (aget im j))
              (aset re j tr) (aset im j ti)))
          (let [new-j (loop [m (int (bit-shift-right nn 1)) j j]
                        (if (and (>= m 1) (>= j m))
                          (recur (bit-shift-right m 1) (- j m))
                          (+ j m)))]
            (recur (inc i) new-j))))
      (loop [len (int 2)]
        (when (<= len nn)
          (let [half (bit-shift-right len 1)
                angle (/ (* -2.0 Math/PI) (double len))]
            (loop [start (int 0)]
              (when (< start nn)
                (loop [k (int 0) wr 1.0 wi 0.0]
                  (when (< k half)
                    (let [idx1 (+ start k) idx2 (+ start k half)
                          ur (aget re idx1) ui (aget im idx1)
                          vr (- (* wr (aget re idx2)) (* wi (aget im idx2)))
                          vi (+ (* wr (aget im idx2)) (* wi (aget re idx2)))]
                      (aset re idx1 (+ ur vr)) (aset im idx1 (+ ui vi))
                      (aset re idx2 (- ur vr)) (aset im idx2 (- ui vi))
                      (let [cos-a (Math/cos (* angle (double (inc k))))
                            sin-a (Math/sin (* angle (double (inc k))))]
                        (recur (inc k) cos-a sin-a)))))
                (recur (+ start len)))))
          (recur (* len 2)))))))

;; deftm version (plain loops — lazy JIT)
(deftm tm-fft-radix2! [re :- (Array double), im :- (Array double)]
  (let [nn (alength re)]
    (when (> nn 1)
      (loop [i (int 0) j (int 0)]
        (when (< i (dec nn))
          (when (< i j)
            (let [tr (aget re i) ti (aget im i)]
              (aset re i (aget re j)) (aset im i (aget im j))
              (aset re j tr) (aset im j ti)))
          (let [new-j (loop [m (int (bit-shift-right nn 1)) j j]
                        (if (and (>= m 1) (>= j m))
                          (recur (bit-shift-right m 1) (- j m))
                          (+ j m)))]
            (recur (inc i) new-j))))
      (loop [len (int 2)]
        (when (<= len nn)
          (let [half (bit-shift-right len 1)
                angle (/ (* -2.0 Math/PI) (double len))]
            (loop [start (int 0)]
              (when (< start nn)
                (loop [k (int 0) wr 1.0 wi 0.0]
                  (when (< k half)
                    (let [idx1 (+ start k) idx2 (+ start k half)
                          ur (aget re idx1) ui (aget im idx1)
                          vr (- (* wr (aget re idx2)) (* wi (aget im idx2)))
                          vi (+ (* wr (aget im idx2)) (* wi (aget re idx2)))]
                      (aset re idx1 (+ ur vr)) (aset im idx1 (+ ui vi))
                      (aset re idx2 (- ur vr)) (aset im idx2 (- ui vi))
                      (let [cos-a (Math/cos (* angle (double (inc k))))
                            sin-a (Math/sin (* angle (double (inc k))))]
                        (recur (inc k) cos-a sin-a)))))
                (recur (+ start len)))))
          (recur (* len 2)))))))

;; par/butterfly! version — pre-computed twiddle factors + SIMD via AOT
(deftm tm-fft-butterfly! [re :- (Array double), im :- (Array double)]
  (let [nn (alength re)]
    (when (> nn 1)
      ;; Bit-reversal permutation (sequential)
      (loop [i (int 0) j (int 0)]
        (when (< i (dec nn))
          (when (< i j)
            (let [tr (aget re i) ti (aget im i)]
              (aset re i (aget re j)) (aset im i (aget im j))
              (aset re j tr) (aset im j ti)))
          (let [new-j (loop [m (int (bit-shift-right nn 1)) j j]
                        (if (and (>= m 1) (>= j m))
                          (recur (bit-shift-right m 1) (- j m))
                          (+ j m)))]
            (recur (inc i) new-j))))
      ;; Butterfly stages with pre-computed twiddle tables
      (loop [len (int 2)]
        (when (<= len nn)
          (let [half (int (bit-shift-right len 1))
                angle (/ (* -2.0 Math/PI) (double len))
                wr (double-array half)
                wi (double-array half)]
            ;; Pre-compute twiddle factors for this stage
            (dotimes [k half]
              (aset wr k (Math/cos (* angle (double k))))
              (aset wi k (Math/sin (* angle (double k)))))
            ;; Apply butterfly at each block
            (loop [start (int 0)]
              (when (< start nn)
                (par/butterfly! re im k half wr wi start)
                (recur (+ start len)))))
          (recur (* len 2)))))))

;; ================================================================
;; 4. DGEMM (naive dense matmul)
;; ================================================================

;; Java baseline — explicit int loops for primitive array access
(defn java-matmul! [^doubles c ^doubles a ^doubles b ^long nn]
  (let [n (int nn)]
    (loop [i (int 0)]
      (when (< i n)
        (loop [j (int 0)]
          (when (< j n)
            (let [s (loop [k (int 0) s 0.0]
                      (if (>= k n) s
                          (recur (unchecked-inc-int k)
                                 (+ s (* (aget a (+ (* i n) k))
                                         (aget b (+ (* k n) j)))))))]
              (aset c (+ (* i n) j) s))
            (recur (unchecked-inc-int j))))
        (recur (unchecked-inc-int i))))))

;; deftm version
(deftm tm-matmul! [c :- (Array double), a :- (Array double),
                   b :- (Array double), nn :- Long]
  (dotimes [i nn]
    (dotimes [j nn]
      (let [s (loop [k (int 0) s 0.0]
                (if (>= k nn) s
                    (recur (inc k) (+ s (* (aget a (+ (* i nn) k))
                                           (aget b (+ (* k nn) j)))))))]
        (aset c (+ (* i nn) j) s)))))

;; ================================================================
;; Run comparison
;; ================================================================

(println)
(println "=== Tier 2 Comparison: Java vs deftm vs AOT vs Julia ===")
(println (format "    JDK: %s" (System/getProperty "java.version")))
(println)
(println (format "  %-35s %8s  %8s  %8s  %8s  %6s  %6s  %6s"
                 "" "Java" "deftm" "AOT" "Julia" "dtm/J" "AOT/J" "Jul/J"))
(println (apply str (repeat 106 "─")))

;; --- Jacobi-2D ---
;; Julia measured on same machine (julia_tier2_bench.jl)
;; AOT uses par/map! offset version for row-wise SIMD
(def aot-jacobi (pipeline/compile-aot #'tm-jacobi-2d-par!))
(doseq [[nn niter julia-ms] [[100 100 0.55] [256 100 4.72] [512 50 15.39]]]
  (let [make-arrays (fn [] (let [u (double-array (* nn nn))
                                 _ (dotimes [j nn] (aset u j 1.0))
                                 tmp (clojure.core/aclone u)]
                             [u tmp]))
        ms-java (do (warmup! #(let [[u t] (make-arrays)] (java-jacobi-2d! u t nn niter)) 3000)
                    (bench #(let [[u t] (make-arrays)] (java-jacobi-2d! u t nn niter))
                           (warmup! #(let [[u t] (make-arrays)] (java-jacobi-2d! u t nn niter)) 100)))
        ms-deftm (do (warmup! #(let [[u t] (make-arrays)] (tm-jacobi-2d! u t nn niter)) 3000)
                     (bench #(let [[u t] (make-arrays)] (tm-jacobi-2d! u t nn niter))
                            (warmup! #(let [[u t] (make-arrays)] (tm-jacobi-2d! u t nn niter)) 100)))
        ms-aot (do (warmup! #(let [[u t] (make-arrays)] (aot-jacobi u t nn niter)) 3000)
                   (bench #(let [[u t] (make-arrays)] (aot-jacobi u t nn niter))
                          (warmup! #(let [[u t] (make-arrays)] (aot-jacobi u t nn niter)) 100)))]
    (row (format "Jacobi-2D N=%d iter=%d" nn niter) ms-java ms-deftm ms-aot julia-ms)))

;; --- Lotka-Volterra ODE ---
;; Julia measured on same machine
(println)
(let [p (double-array [1.5 1.0 3.0 1.0])
      u0 (double-array [1.0 1.0])]
  ;; Correctness check
  (let [result (tm-solve-rk4 tm-lotka-volterra! u0 p 0.0 10.0 0.01)]
    (println (format "  Lotka-Volterra u(10) = [%.6f, %.6f]" (aget result 0) (aget result 1))))
  ;; AOT
  (def aot-solve (pipeline/compile-aot #'tm-solve-rk4))
  (let [aot-result (aot-solve tm-lotka-volterra! u0 p 0.0 10.0 0.01)]
    (println (format "  AOT u(10)            = [%.6f, %.6f]" (aget ^doubles aot-result 0) (aget ^doubles aot-result 1))))
  ;; Benchmark
  (doseq [[dt julia-us] [[0.01 52.0] [0.001 515.0]]]
    (let [julia-ms (/ julia-us 1000.0)
          ms-java (do (warmup! #(java-solve-rk4 java-lotka-volterra! u0 p 0.0 10.0 dt) 3000)
                      (bench #(java-solve-rk4 java-lotka-volterra! u0 p 0.0 10.0 dt)
                             (warmup! #(java-solve-rk4 java-lotka-volterra! u0 p 0.0 10.0 dt) 100)))
          ms-deftm (do (warmup! #(tm-solve-rk4 tm-lotka-volterra! u0 p 0.0 10.0 dt) 3000)
                       (bench #(tm-solve-rk4 tm-lotka-volterra! u0 p 0.0 10.0 dt)
                              (warmup! #(tm-solve-rk4 tm-lotka-volterra! u0 p 0.0 10.0 dt) 100)))
          ms-aot (do (warmup! #(aot-solve tm-lotka-volterra! u0 p 0.0 10.0 dt) 3000)
                     (bench #(aot-solve tm-lotka-volterra! u0 p 0.0 10.0 dt)
                            (warmup! #(aot-solve tm-lotka-volterra! u0 p 0.0 10.0 dt) 100)))]
      (row (format "RK4 Lotka-Volterra dt=%.3f" dt) ms-java ms-deftm ms-aot julia-ms))))

;; --- FFT ---
;; Julia measured on same machine (hand-written radix-2)
;; AOT uses par/butterfly! with pre-computed twiddle tables
(println)
(def aot-fft (pipeline/compile-aot #'tm-fft-butterfly!))
(doseq [[logn julia-us] [[10 15.0] [14 357.0] [20 41235.0]]]
  (let [nn (bit-shift-left 1 logn)
        julia-ms (/ julia-us 1000.0)
        rng (java.util.Random. 42)
        re0 (double-array (repeatedly nn #(.nextGaussian rng)))
        im0 (double-array (repeatedly nn #(.nextGaussian rng)))
        ms-java (do (warmup! #(let [re (clojure.core/aclone re0) im (clojure.core/aclone im0)]
                                 (java-fft-radix2! re im)) 3000)
                    (bench #(let [re (clojure.core/aclone re0) im (clojure.core/aclone im0)]
                              (java-fft-radix2! re im))
                           (warmup! #(let [re (clojure.core/aclone re0) im (clojure.core/aclone im0)]
                                       (java-fft-radix2! re im)) 100)))
        ms-deftm (do (warmup! #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                                  (tm-fft-radix2! re im)) 3000)
                     (bench #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                               (tm-fft-radix2! re im))
                            (warmup! #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                                        (tm-fft-radix2! re im)) 100)))
        ms-aot (do (warmup! #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                                (aot-fft re im)) 3000)
                   (bench #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                             (aot-fft re im))
                          (warmup! #(let [re (raster.arrays/aclone re0) im (raster.arrays/aclone im0)]
                                      (aot-fft re im)) 100)))]
    (row (format "FFT N=2^%d (%d)" logn nn) ms-java ms-deftm ms-aot julia-ms)))

;; --- DGEMM (naive) ---
;; Julia naive matmul measured on same machine (not BLAS)
(println)
(def aot-matmul (pipeline/compile-aot #'tm-matmul!))
(doseq [[nn julia-ms] [[100 0.645] [500 89.691] [1000 874.763]]]
  (let [rng (java.util.Random. 42)
        a (double-array (repeatedly (* nn nn) #(.nextGaussian rng)))
        b (double-array (repeatedly (* nn nn) #(.nextGaussian rng)))
        c (double-array (* nn nn))
        ms-java (do (warmup! #(java-matmul! c a b nn) (if (> nn 500) 500 3000))
                    (bench #(java-matmul! c a b nn)
                           (warmup! #(java-matmul! c a b nn) 100)))
        ms-deftm (do (warmup! #(tm-matmul! c a b nn) (if (> nn 500) 500 3000))
                     (bench #(tm-matmul! c a b nn)
                            (warmup! #(tm-matmul! c a b nn) 100)))
        ms-aot (do (warmup! #(aot-matmul c a b nn) (if (> nn 500) 500 3000))
                   (bench #(aot-matmul c a b nn)
                          (warmup! #(aot-matmul c a b nn) 100)))]
    (row (format "Naive matmul N=%d" nn) ms-java ms-deftm ms-aot julia-ms)))

;; --- DGEMM (BLAS) ---
;; Julia BLAS matmul measured on same machine (A * B dispatches to OpenBLAS)
(println)
(require '[raster.linalg.blas :as blas])
(doseq [[nn julia-ms] [[100 0.061] [500 2.713] [1000 21.254]]]
  (let [rng (java.util.Random. 42)
        a (double-array (repeatedly (* nn nn) #(.nextGaussian rng)))
        b (double-array (repeatedly (* nn nn) #(.nextGaussian rng)))
        c (double-array (* nn nn))
        ms-blas (do (warmup! #(blas/dgemm! a b c nn nn nn 1.0 0.0) 3000)
                    (bench #(blas/dgemm! a b c nn nn nn 1.0 0.0)
                           (warmup! #(blas/dgemm! a b c nn nn nn 1.0 0.0) 100)))]
    (row (format "BLAS dgemm N=%d" nn) ms-blas ms-blas ms-blas julia-ms)))

(println)
(println "  Java = Clojure with primitive type hints (identical code)")
(println "  deftm = Raster lazy JIT (walker + bytecode compiler, plain loops)")
(println "  AOT = compile-aot (full pipeline + par combinators)")
(println "  AOT Jacobi uses par/map! offset (row-wise SIMD)")
(println "  AOT FFT uses par/butterfly! (pre-computed twiddle + SIMD)")
(println "  BLAS = OpenBLAS via Panama FFI (raster.linalg.blas)")
(println "  Julia = same machine, BenchmarkTools.jl median")

(shutdown-agents)

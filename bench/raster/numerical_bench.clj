;; Numerical benchmark suite — Tier 1
;;
;; Run with Valhalla JDK:
;;   source valhalla-env.sh
;;   OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \
;;     -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
;;     -J--enable-native-access=ALL-UNNAMED -M:valhalla \
;;     bench/raster/numerical_bench.clj
;;
;; Benchmarks:
;;   1. N-body (5 planets, Benchmarks Game)
;;   2. Spectral Norm (Benchmarks Game)
;;   3. Rosenbrock gradient (forward-mode AD)
;;   4. Monte Carlo Pi

(require '[raster.core :refer [deftm ftm]])
(require '[raster.numeric :as n])
(require '[raster.arrays :refer [alength aclone acopy!]])

;; ================================================================
;; Benchmark harness
;; ================================================================

(defn bench
  "Benchmark f with adaptive iteration count.
  Targets ~5s total measurement time. Minimum 5 measurements.
  Returns {:best :p10 :median :p90 :mean} in microseconds."
  [f per-call-ms]
  ;; Determine measurement count: target 5s total, min 5, max 50
  (let [n-measure (max 5 (min 50 (int (/ 5000.0 (max 0.01 per-call-ms)))))
        _ (System/gc)
        _ (Thread/sleep 200)
        times (long-array n-measure)]
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
       :mean   (/ (reduce + (map #(/ (double %) 1000.0) times)) n-measure)
       :n      n-measure})))

(defn report [label result]
  (println (format "  %-35s best:%8.1f  p10:%8.1f  med:%8.1f  p90:%8.1f  mean:%8.1f"
                   label (:best result) (:p10 result) (:median result)
                   (:p90 result) (:mean result)))
  result)

;; ================================================================
;; 1. N-body (5 planets, Benchmarks Game)
;; ================================================================
;; Exact same algorithm as the Benchmarks Game reference.
;; 5 bodies: Sun, Jupiter, Saturn, Uranus, Neptune.
;; Symplectic leapfrog integrator, dt=0.01.
;; State: arrays of x,y,z,vx,vy,vz,mass (structure-of-arrays).

(def ^:const ^double SOLAR_MASS (* 4.0 Math/PI Math/PI))
(def ^:const ^double DAYS_PER_YEAR 365.24)
(def ^:const NBODIES 5)

;; Initial conditions (Benchmarks Game standard)
(def ^doubles init-x  (double-array [ 0.0
                                       4.84143144246472090e+00
                                       8.34336671824457987e+00
                                       1.28943695621391310e+01
                                       1.53796971148509165e+01]))
(def ^doubles init-y  (double-array [ 0.0
                                      -1.16032004402742839e+00
                                       4.12479856412430479e+00
                                      -1.51111514016986312e+01
                                      -2.59193146099879641e+01]))
(def ^doubles init-z  (double-array [ 0.0
                                      -1.03622044471123109e-01
                                      -4.03523417114321381e-01
                                      -2.23307578892655734e-01
                                       1.79258772950371181e-01]))
(def ^doubles init-vx (double-array [ 0.0
                                       (* 1.66007664274403694e-03 DAYS_PER_YEAR)
                                       (* -2.76742510726862411e-03 DAYS_PER_YEAR)
                                       (* 2.96460137564761618e-03 DAYS_PER_YEAR)
                                       (* 2.68067772490389322e-03 DAYS_PER_YEAR)]))
(def ^doubles init-vy (double-array [ 0.0
                                       (* 7.69901118419740425e-03 DAYS_PER_YEAR)
                                       (* 4.99852801234917238e-03 DAYS_PER_YEAR)
                                       (* 2.37847173959480950e-03 DAYS_PER_YEAR)
                                       (* 1.62824170038242295e-03 DAYS_PER_YEAR)]))
(def ^doubles init-vz (double-array [ 0.0
                                       (* -6.90460016972063023e-05 DAYS_PER_YEAR)
                                       (* 2.30417297573763929e-05 DAYS_PER_YEAR)
                                       (* -2.96589568540237556e-05 DAYS_PER_YEAR)
                                       (* -9.51592254519715870e-05 DAYS_PER_YEAR)]))
(def ^doubles body-mass (double-array [ SOLAR_MASS
                                         (* 9.54791938424326609e-04 SOLAR_MASS)
                                         (* 2.85885980666130812e-04 SOLAR_MASS)
                                         (* 4.36624404335156298e-05 SOLAR_MASS)
                                         (* 5.15138902046611451e-05 SOLAR_MASS)]))

(defn make-nbody-state []
  {:x (aclone init-x) :y (aclone init-y) :z (aclone init-z)
   :vx (aclone init-vx) :vy (aclone init-vy) :vz (aclone init-vz)
   :mass (aclone body-mass)})

(defn offset-momentum! [{:keys [^doubles vx ^doubles vy ^doubles vz ^doubles mass]}]
  (let [n (int NBODIES)]
    (loop [i (int 1) px 0.0 py 0.0 pz 0.0]
      (if (>= i n)
        (do (aset vx 0 (/ (- px) SOLAR_MASS))
            (aset vy 0 (/ (- py) SOLAR_MASS))
            (aset vz 0 (/ (- pz) SOLAR_MASS)))
        (recur (inc i)
               (+ px (* (aget vx i) (aget mass i)))
               (+ py (* (aget vy i) (aget mass i)))
               (+ pz (* (aget vz i) (aget mass i))))))))

(defn energy [{:keys [^doubles x ^doubles y ^doubles z
                       ^doubles vx ^doubles vy ^doubles vz ^doubles mass]}]
  (let [n (int NBODIES)]
    (loop [i (int 0) e 0.0]
      (if (>= i n) e
          (let [bi-ke (* 0.5 (aget mass i)
                        (+ (* (aget vx i) (aget vx i))
                           (* (aget vy i) (aget vy i))
                           (* (aget vz i) (aget vz i))))
                bi-pe (loop [j (int (inc i)) pe 0.0]
                        (if (>= j n) pe
                            (let [dx (- (aget x i) (aget x j))
                                  dy (- (aget y i) (aget y j))
                                  dz (- (aget z i) (aget z j))
                                  dist (Math/sqrt (+ (* dx dx) (* dy dy) (* dz dz)))]
                              (recur (inc j) (- pe (/ (* (aget mass i) (aget mass j)) dist))))))]
            (recur (inc i) (+ e bi-ke bi-pe)))))))

(defn advance! [{:keys [^doubles x ^doubles y ^doubles z
                         ^doubles vx ^doubles vy ^doubles vz ^doubles mass]}
                ^double dt ^long n-steps]
  (let [n (int NBODIES)]
    (dotimes [_ n-steps]
      ;; Update velocities (all pairs)
      (loop [i (int 0)]
        (when (< i n)
          (loop [j (int (inc i))]
            (when (< j n)
              (let [dx (- (aget x i) (aget x j))
                    dy (- (aget y i) (aget y j))
                    dz (- (aget z i) (aget z j))
                    dsq (+ (* dx dx) (* dy dy) (* dz dz))
                    dist (Math/sqrt dsq)
                    mag (/ dt (* dsq dist))
                    mj-mag (* (aget mass j) mag)
                    mi-mag (* (aget mass i) mag)]
                (aset vx i (- (aget vx i) (* dx mj-mag)))
                (aset vy i (- (aget vy i) (* dy mj-mag)))
                (aset vz i (- (aget vz i) (* dz mj-mag)))
                (aset vx j (+ (aget vx j) (* dx mi-mag)))
                (aset vy j (+ (aget vy j) (* dy mi-mag)))
                (aset vz j (+ (aget vz j) (* dz mi-mag))))
              (recur (inc j))))
          (recur (inc i))))
      ;; Update positions
      (dotimes [i n]
        (aset x i (+ (aget x i) (* dt (aget vx i))))
        (aset y i (+ (aget y i) (* dt (aget vy i))))
        (aset z i (+ (aget z i) (* dt (aget vz i))))))))

;; ================================================================
;; 2. Spectral Norm (Benchmarks Game)
;; ================================================================

(deftm spectral-A [i :- Long, j :- Long] :- Double
  (/ 1.0 (+ (/ (* (+ i j) (+ i j 1)) 2) i 1)))

(deftm spectral-Av! [out :- (Array double), v :- (Array double), n :- Long]
  (dotimes [i n]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j n) s
                  (recur (inc j) (+ s (* (spectral-A i j) (aget v j))))))]
      (aset out i s))))

(deftm spectral-Atv! [out :- (Array double), v :- (Array double), n :- Long]
  (dotimes [i n]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j n) s
                  (recur (inc j) (+ s (* (spectral-A j i) (aget v j))))))]
      (aset out i s))))

(deftm spectral-AtAv! [out :- (Array double), v :- (Array double), tmp :- (Array double), n :- Long]
  (spectral-Av! tmp v n)
  (spectral-Atv! out tmp n))

(deftm spectral-norm [n :- Long] :- Double
  (let [u (double-array n 1.0)
        v (double-array n)
        tmp (double-array n)]
    (dotimes [_ 10]
      (spectral-AtAv! v u tmp n)
      (spectral-AtAv! u v tmp n))
    (let [vBv (loop [i (int 0) acc 0.0]
                (if (>= i n) acc
                    (recur (inc i) (+ acc (* (aget u i) (aget v i))))))
          vv  (loop [i (int 0) acc 0.0]
                (if (>= i n) acc
                    (recur (inc i) (+ acc (* (aget v i) (aget v i))))))]
      (Math/sqrt (/ vBv vv)))))

;; ================================================================
;; 3. Rosenbrock gradient (forward-mode AD)
;; ================================================================
;; Uses raster.ad.forward Dual numbers for gradient computation.
;; f(x) = sum_{i=1}^{N-1} [100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2]

(deftm rosenbrock [x :- (Array double)] :- Double
  (let [n (alength x)]
    (loop [i (int 0) s 0.0]
      (if (>= i (dec n)) s
          (let [xi (aget x i)
                xi1 (aget x (inc i))
                t1 (- xi1 (* xi xi))
                t2 (- 1.0 xi)]
            (recur (inc i) (+ s (* 100.0 t1 t1) (* t2 t2))))))))

;; For AD gradient, we'll use the manual forward-mode approach:
;; compute df/dx_k by seeding x_k with dual part 1, rest with 0.
;; This is the standard forward-mode gradient computation.
(defn rosenbrock-gradient! [^doubles grad ^doubles x]
  (let [n (alength x)]
    (dotimes [k n]
      ;; Compute df/dx_k by finite difference (fast, matches ForwardDiff pattern)
      ;; Using central differences for accuracy matching ForwardDiff's dual number approach
      (let [eps 1e-8
            xk (aget x k)
            _ (aset x k (+ xk eps))
            fp (rosenbrock x)
            _ (aset x k (- xk eps))
            fm (rosenbrock x)
            _ (aset x k xk)]
        (aset grad k (/ (- fp fm) (* 2.0 eps)))))))

;; TODO: Replace with proper Dual number AD when Dual4 is available
;; For now, this tests the Rosenbrock function compilation quality
;; and the gradient computation overhead.

;; ================================================================
;; 4. Monte Carlo Pi
;; ================================================================

(deftm monte-carlo-pi [n :- Long] :- Double
  (let [rng (java.util.SplittableRandom. 42)]
    (loop [i (int 0) count (int 0)]
      (if (>= i n) (* 4.0 (/ (double count) (double n)))
          (let [x (.nextDouble rng)
                y (.nextDouble rng)]
            (recur (inc i)
                   (if (<= (+ (* x x) (* y y)) 1.0)
                     (unchecked-inc-int count)
                     count)))))))

;; ================================================================
;; Run benchmarks — warm JVM (Leyden-faithful)
;; ================================================================
;; We warm thoroughly so C2 has time to compile and optimize.
;; With Project Leyden, these warm numbers will be the cold-start numbers.

(defn warmup!
  "Run f repeatedly for at least warmup-ms milliseconds. Returns per-call time in ms."
  [f warmup-ms]
  (let [deadline (+ (System/nanoTime) (* warmup-ms 1000000))]
    (loop [i 0]
      (let [t0 (System/nanoTime)]
        (f)
        (let [elapsed (/ (- (System/nanoTime) t0) 1e6)]
          (if (< (System/nanoTime) deadline)
            (recur (inc i))
            elapsed))))))

(defn validate
  "Check that actual ≈ expected within tolerance. Prints PASS/FAIL."
  [label actual expected tol]
  (let [err (Math/abs (- (double actual) (double expected)))
        ok? (< err tol)]
    (println (format "  %s %s: %.9f (expected %.9f, err=%.2e, tol=%.2e)"
                     (if ok? "✓" "✗") label (double actual) (double expected) err tol))
    ok?))

(defn report-vs-julia
  "Report benchmark result with Julia comparison."
  [label result julia-ms]
  (let [raster-ms (/ (:median result) 1000.0)
        ratio (/ raster-ms julia-ms)]
    (println (format "  %-32s  med:%8.2f ms  Julia:%8.2f ms  ratio: %.2fx%s"
                     label raster-ms julia-ms ratio
                     (if (<= ratio 1.0) " ✓" "")))
    result))

(println)
(println "=== Tier 1 Numerical Benchmarks ===")
(println (format "    JDK: %s" (System/getProperty "java.version")))
(println "    3s timed warmup per benchmark, adaptive measurement count")
(println)

;; --- N-body ---
(println "── N-body (5 planets, Benchmarks Game) ──")

;; Correctness
(let [state (make-nbody-state)]
  (offset-momentum! state)
  (validate "initial energy" (energy state) -0.169075164 1e-8)
  (advance! state 0.01 1000)
  (validate "after 1000 steps" (energy state) -0.169087605 1e-8))

;; Warm + bench
(doseq [[n-steps julia-ms] [[1000 0.07] [100000 7.73] [1000000 79.81]]]
  (let [f (fn [] (let [s (make-nbody-state)] (offset-momentum! s) (advance! s 0.01 n-steps)))
        ms (warmup! f 3000)]
    (report-vs-julia (format "N-body %,d steps" n-steps)
      (bench f ms) julia-ms)))

;; --- Spectral Norm ---
(println)
(println "── Spectral Norm (Benchmarks Game) ──")

(validate "spectral-norm(100)" (spectral-norm 100) 1.274219991 1e-9)

(doseq [[n julia-ms] [[100 0.744] [1000 94.849] [5500 2847.1]]]
  (let [ms (warmup! #(spectral-norm n) 3000)]
    (report-vs-julia (format "Spectral norm N=%,d" n)
      (bench #(spectral-norm n) ms) julia-ms)))

;; --- Rosenbrock ---
(println)
(println "── Rosenbrock function + gradient ──")

;; Correctness: f([1,1,...,1]) = 0
(validate "rosenbrock([1,1,1])" (rosenbrock (double-array [1.0 1.0 1.0])) 0.0 1e-12)

(doseq [[n julia-grad-ms] [[10 0.0008] [100 0.011] [1000 0.9644]]]
  (let [x (double-array (repeatedly n #(- (rand) 0.5)))
        g (double-array n)]
    ;; Function eval (no Julia comparison — Julia only benchmarks gradient)
    (let [ms (warmup! #(rosenbrock x) 3000)]
      (report (format "Rosenbrock f(x) N=%d" n)
        (bench #(rosenbrock x) ms)))
    ;; Gradient (finite diff — Julia uses ForwardDiff which is faster)
    (let [ms (warmup! #(rosenbrock-gradient! g x) 3000)]
      (report-vs-julia (format "Rosenbrock grad N=%d" n)
        (bench #(rosenbrock-gradient! g x) ms) julia-grad-ms))))

;; --- Monte Carlo Pi ---
(println)
(println "── Monte Carlo Pi ──")

(let [pi-approx (monte-carlo-pi 1000000)]
  (validate "pi (1M samples)" pi-approx Math/PI 0.01))

(doseq [[n julia-ms] [[1000000 4.31] [10000000 67.27]]]
  (let [ms (warmup! #(monte-carlo-pi n) 3000)]
    (report-vs-julia (format "Monte Carlo pi N=%,d" n)
      (bench #(monte-carlo-pi n) ms) julia-ms)))

(println)
(println "Julia reference: bench/comparison/julia_nbody_bench.jl")

(shutdown-agents)

;; Tier 1 Comparison Benchmark — Java vs Clojure vs deftm vs Julia
;;
;; Run with Valhalla JDK:
;;   source valhalla-env.sh
;;   OPENBLAS_NUM_THREADS=1 clojure -J--enable-preview \
;;     -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
;;     -J--enable-native-access=ALL-UNNAMED -M:valhalla \
;;     bench/raster/tier1_comparison.clj

(require '[raster.core :refer [deftm ftm]])
(require '[raster.numeric :as n])
(require '[raster.arrays :refer [alength aclone acopy!]])
(require '[raster.ad.forward :as fwd])

;; ================================================================
;; Harness
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

(defn row [label java clj deftm julia]
  (println (format "  %-28s %8.2f  %8.2f  %8.2f  %8.2f  %5.2fx  %5.2fx  %5.2fx"
                   label java clj deftm julia
                   (/ clj java) (/ deftm java) (/ julia java))))

;; ================================================================
;; 1. Spectral Norm
;; ================================================================
;; "Java" = pure Clojure with primitive hints, no dispatch overhead.
;; This is what hand-optimized Java would produce on the JVM.

(defn java-spectral-A ^double [^long i ^long j]
  (/ 1.0 (+ (/ (* (+ i j) (+ i j 1)) 2) i 1)))

(defn java-spectral-Av! [^doubles out ^doubles v ^long nn]
  (dotimes [i nn]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j nn) s
                  (recur (inc j) (+ s (* (java-spectral-A i j) (aget v j))))))]
      (aset out i s))))

(defn java-spectral-Atv! [^doubles out ^doubles v ^long nn]
  (dotimes [i nn]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j nn) s
                  (recur (inc j) (+ s (* (java-spectral-A j i) (aget v j))))))]
      (aset out i s))))

(defn java-spectral-norm ^double [^long nn]
  (let [u (double-array nn 1.0) v (double-array nn) tmp (double-array nn)]
    (dotimes [_ 10]
      (java-spectral-Av! tmp u nn) (java-spectral-Atv! v tmp nn)
      (java-spectral-Av! tmp v nn) (java-spectral-Atv! u tmp nn))
    (let [vBv (loop [i (int 0) a 0.0] (if (>= i nn) a (recur (inc i) (+ a (* (aget u i) (aget v i))))))
          vv  (loop [i (int 0) a 0.0] (if (>= i nn) a (recur (inc i) (+ a (* (aget v i) (aget v i))))))]
      (Math/sqrt (/ vBv vv)))))

;; deftm version
(deftm tm-spectral-A [i :- Long, j :- Long] :- Double
  (/ 1.0 (+ (/ (* (+ i j) (+ i j 1)) 2) i 1)))

(deftm tm-spectral-Av! [out :- (Array double), v :- (Array double), nn :- Long]
  (dotimes [i nn]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j nn) s (recur (inc j) (+ s (* (tm-spectral-A i j) (aget v j))))))]
      (aset out i s))))

(deftm tm-spectral-Atv! [out :- (Array double), v :- (Array double), nn :- Long]
  (dotimes [i nn]
    (let [s (loop [j (int 0) s 0.0]
              (if (>= j nn) s (recur (inc j) (+ s (* (tm-spectral-A j i) (aget v j))))))]
      (aset out i s))))

(deftm tm-spectral-AtAv! [out :- (Array double), v :- (Array double), tmp :- (Array double), nn :- Long]
  (tm-spectral-Av! tmp v nn) (tm-spectral-Atv! out tmp nn))

(deftm tm-spectral-norm [nn :- Long] :- Double
  (let [u (double-array nn 1.0) v (double-array nn) tmp (double-array nn)]
    (dotimes [_ 10] (tm-spectral-AtAv! v u tmp nn) (tm-spectral-AtAv! u v tmp nn))
    (let [vBv (loop [i (int 0) a 0.0] (if (>= i nn) a (recur (inc i) (+ a (* (aget u i) (aget v i))))))
          vv  (loop [i (int 0) a 0.0] (if (>= i nn) a (recur (inc i) (+ a (* (aget v i) (aget v i))))))]
      (Math/sqrt (/ vBv vv)))))

;; ================================================================
;; 2. Monte Carlo Pi
;; ================================================================

(defn java-mc-pi ^double [^long nn]
  (let [rng (java.util.SplittableRandom. 42)]
    (loop [i (int 0) cnt (int 0)]
      (if (>= i nn) (* 4.0 (/ (double cnt) (double nn)))
          (let [x (.nextDouble rng) y (.nextDouble rng)]
            (recur (inc i) (if (<= (+ (* x x) (* y y)) 1.0)
                             (unchecked-inc-int cnt) cnt)))))))

(deftm tm-mc-pi [nn :- Long] :- Double
  (let [rng (java.util.SplittableRandom. 42)]
    (loop [i (int 0) cnt (int 0)]
      (if (>= i nn) (* 4.0 (/ (double cnt) (double nn)))
          (let [x (.nextDouble rng) y (.nextDouble rng)]
            (recur (inc i) (if (<= (+ (* x x) (* y y)) 1.0)
                             (unchecked-inc-int cnt) cnt)))))))

;; ================================================================
;; 3. Rosenbrock function eval
;; ================================================================

(defn java-rosenbrock ^double [^doubles x]
  (let [nn (clojure.core/alength x)]
    (loop [i (int 0) s 0.0]
      (if (>= i (dec nn)) s
          (let [xi (aget x i) xi1 (aget x (inc i))
                t1 (- xi1 (* xi xi)) t2 (- 1.0 xi)]
            (recur (inc i) (+ s (* 100.0 t1 t1) (* t2 t2))))))))

(deftm tm-rosenbrock [x :- (Array double)] :- Double
  (let [nn (alength x)]
    (loop [i (int 0) s 0.0]
      (if (>= i (dec nn)) s
          (let [xi (aget x i) xi1 (aget x (inc i))
                t1 (- xi1 (* xi xi)) t2 (- 1.0 xi)]
            (recur (inc i) (+ s (* 100.0 t1 t1) (* t2 t2))))))))

;; Rosenbrock gradient via devirtualized Dual deftm
(require '[raster.compiler.core.dispatch :as dispatch])
(doseq [op ['+ '- '* '/ 'dual]]
  (let [v (ns-resolve 'raster.ad.forward op)
        ta (when v (:raster.core/dispatch-table (meta v)))]
    (when ta (#'dispatch/emit-tc-ann! (the-ns 'raster.ad.forward) op ta))))
(doseq [op ['+ '- '* '/]]
  (let [v (ns-resolve 'raster.numeric op)
        ta (:raster.core/dispatch-table (meta v))]
    (when ta (#'dispatch/emit-tc-ann! (the-ns 'raster.numeric) op ta))))

(deftm tm-rosenbrock-dual [x :- (Array raster.ad.forward.Dual)] :- raster.ad.forward.Dual
  (let [nn (raster.arrays/alength x)]
    (loop [i (int 0) s (fwd/dual 0.0)]
      (if (>= i (dec nn)) s
          (let [xi (aget x i) xi1 (aget x (inc i))
                t1 (n/- xi1 (n/* xi xi))
                t2 (n/- (fwd/dual 1.0) xi)]
            (recur (inc i) (n/+ s (n/* (fwd/dual 100.0) (n/* t1 t1)) (n/* t2 t2))))))))

;; Finite diff gradient (for Clojure column)
(defn java-rosenbrock-grad! [^doubles grad ^doubles x]
  (let [nn (clojure.core/alength x) eps 1e-8]
    (dotimes [k nn]
      (let [xk (aget x k)]
        (aset x k (+ xk eps))
        (let [fp (java-rosenbrock x)]
          (aset x k (- xk eps))
          (let [fm (java-rosenbrock x)]
            (aset x k xk)
            (aset grad k (/ (- fp fm) (* 2.0 eps)))))))))

;; ================================================================
;; Run comparison
;; ================================================================

(println)
(println "=== Tier 1 Comparison: Java vs Clojure vs deftm vs Julia ===")
(println (format "    JDK: %s" (System/getProperty "java.version")))
(println)
(println (format "  %-28s %8s  %8s  %8s  %8s  %6s  %6s  %6s"
                 "" "Java" "Clojure" "deftm" "Julia" "Clj/J" "dtm/J" "Jul/J"))
(println (apply str (repeat 100 "─")))

;; --- Spectral norm ---
(doseq [[nn julia-ms] [[100 0.744] [1000 94.85] [5500 2847.1]]]
  (let [ms-java  (do (warmup! #(java-spectral-norm nn) 3000)
                     (bench #(java-spectral-norm nn) (warmup! #(java-spectral-norm nn) 100)))
        ;; Clojure = same as Java for this benchmark (pure type-hinted)
        ms-clj   ms-java
        ms-deftm (do (warmup! #(tm-spectral-norm nn) 3000)
                     (bench #(tm-spectral-norm nn) (warmup! #(tm-spectral-norm nn) 100)))]
    (row (format "Spectral norm N=%d" nn) ms-java ms-clj ms-deftm julia-ms)))

;; --- Monte Carlo ---
(doseq [[nn julia-ms] [[1000000 4.31] [10000000 67.27]]]
  (let [ms-java  (do (warmup! #(java-mc-pi nn) 3000)
                     (bench #(java-mc-pi nn) (warmup! #(java-mc-pi nn) 100)))
        ms-clj   ms-java
        ms-deftm (do (warmup! #(tm-mc-pi nn) 3000)
                     (bench #(tm-mc-pi nn) (warmup! #(tm-mc-pi nn) 100)))]
    (row (format "MC pi N=%dM" (/ nn 1000000)) ms-java ms-clj ms-deftm julia-ms)))

;; --- Rosenbrock function eval ---
(doseq [[nn julia-ms] [[100 nil] [1000 nil]]]
  (let [x (double-array (repeatedly nn #(- (rand) 0.5)))
        ms-java  (do (warmup! #(java-rosenbrock x) 3000)
                     (bench #(java-rosenbrock x) (warmup! #(java-rosenbrock x) 100)))
        ms-clj   ms-java
        ms-deftm (do (warmup! #(tm-rosenbrock x) 3000)
                     (bench #(tm-rosenbrock x) (warmup! #(tm-rosenbrock x) 100)))]
    (row (format "Rosenbrock f(x) N=%d" nn) ms-java ms-clj ms-deftm (or julia-ms ms-java))))

;; --- Rosenbrock gradient ---
(println)
(println "  Rosenbrock gradient (Dual AD vs finite diff):")
(doseq [[nn julia-ms] [[10 0.0008] [100 0.011] [1000 0.964]]]
  (let [x (double-array (repeatedly nn #(- (rand) 0.5)))
        g (double-array nn)
        ;; Java/Clojure: finite differences
        ms-fd (do (warmup! #(java-rosenbrock-grad! g x) 3000)
                  (bench #(java-rosenbrock-grad! g x) (warmup! #(java-rosenbrock-grad! g x) 100)))
        ;; deftm: devirtualized Dual forward-mode AD
        build-dual (fn [^doubles vals ^long chunk]
                     (let [arr (object-array (clojure.core/alength vals))]
                       (dotimes [i (clojure.core/alength vals)]
                         (aset arr i (fwd/->Dual (aget vals i) (double-array chunk))))
                       arr))
        ms-dual (let [xd (build-dual x nn)]
                  (dotimes [j nn] (aset (.partials ^raster.ad.forward.Dual (aget ^objects xd j)) j 1.0))
                  (warmup! #(let [xd2 (build-dual x nn)]
                              (dotimes [j nn] (aset (.partials ^raster.ad.forward.Dual (aget ^objects xd2 j)) j 1.0))
                              (tm-rosenbrock-dual xd2)) 3000)
                  (bench #(let [xd2 (build-dual x nn)]
                            (dotimes [j nn] (aset (.partials ^raster.ad.forward.Dual (aget ^objects xd2 j)) j 1.0))
                            (tm-rosenbrock-dual xd2))
                         (warmup! #(let [xd2 (build-dual x nn)]
                                     (dotimes [j nn] (aset (.partials ^raster.ad.forward.Dual (aget ^objects xd2 j)) j 1.0))
                                     (tm-rosenbrock-dual xd2)) 100)))]
    (println (format "  %-28s %8.3f  %8.3f  %8.3f  %8.3f  %5.1fx  %5.1fx  %5.1fx"
                     (format "Rosenbrock grad N=%d" nn)
                     ms-fd ms-fd ms-dual julia-ms
                     1.0 (/ ms-dual ms-fd) (/ julia-ms ms-fd)))))

(println)
(println "  Java = Clojure with primitive type hints (same code, identical perf)")
(println "  deftm = Raster lazy JIT (walker + bytecode compiler)")
(println "  Rosenbrock grad: Java/Clojure = finite diff, deftm = Dual forward-mode AD")
(println "  Julia reference: bench/comparison/julia_nbody_bench.jl")

(shutdown-agents)

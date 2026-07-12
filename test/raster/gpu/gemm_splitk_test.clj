(ns raster.gpu.gemm-splitk-test
  "SPLIT-K GEMM: the schedule fix for a GEMM whose (M,N) tiling cannot fill the machine.

   The XMX GEMM launches ceil(N/128) x ceil(M/128) workgroups of 256 work items. A GEMM
   with a small output and a huge K — the tied-embedding backward
   dx[13,640] = dlogits[13,262144] · E[262144,640] is the motivating one — launches 5
   workgroups of the ~32 that fill an Arc 140V, each doing a k=262144 serial reduction.
   Splitting the k-reduction over a third grid dimension multiplies the workgroup count
   at CONSTANT DRAM traffic; a second kernel sums the per-chunk partials.

   Asserts:
     • the split-k GEMM agrees with the plain XMX GEMM (same f16 inputs, f32 accumulate)
       to f32 summation-order noise, across ragged m/n/k and several split counts —
       including a k-range that does not divide evenly and an m below one 8-row DPAS tile;
     • the split-k DECISION (raster.gpu.core's policy knobs) leaves machine-filling GEMMs
       alone and splits the low-occupancy ones."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gpu-grad-parity :as gp]))

(defn- rnd ^floats [n seed]
  (let [a (float-array n) r (java.util.Random. (long seed))]
    (dotimes [i n] (aset a i (float (* 0.05 (.nextGaussian r)))))
    a))

(defn- rel-l1 [^floats x ^floats y]
  (let [n (alength x)]
    (loop [i 0 num 0.0 den 0.0]
      (if (< i n)
        (recur (inc i)
               (+ num (Math/abs (- (double (aget x i)) (double (aget y i)))))
               (+ den (Math/abs (double (aget y i)))))
        (/ num (max den 1.0e-30))))))

(deftest split-k-matches-plain-xmx-gemm
  (if-not @gp/gpu-available?
    (println "  [SKIP] gemm split-k: no Level Zero GPU")
    (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          make-buffer   (ns-resolve ze 'make-buffer)
          upload!       (ns-resolve ze 'array->buffer!)
          download      (ns-resolve ze 'buffer->array)
          free!         (ns-resolve ze 'free-buffer!)
          record-graph! (ns-resolve ze 'record-graph!)
          replay!       (ns-resolve ze 'replay-graph!)
          destroy!      (ns-resolve ze 'destroy-graph!)
          convert!      (ns-resolve ze 'bind-registered-convert!)
          gemm!         (ns-resolve ze 'bind-registered-gemm!)
          splitk!       (ns-resolve ze 'bind-registered-gemm-splitk!)
          reduce!       (ns-resolve ze 'bind-registered-splitk-reduce!)]
      (doseq [[m n k splits kc]
              [[13 640 8192 8 1024]     ;; the LM-head dx shape (small k, same geometry)
               [13 640 8192 26 320]     ;; k/splits not a multiple of kc*splits
               [7 640 4096 4 1024]      ;; m below one 8-row DPAS tile
               [33 200 1000 3 352]      ;; ragged m, n and k; last chunk clipped
               [128 256 4096 8 512]]]
        (let [A (rnd (* m k) 1) B (rnd (* k n) 2)
              af (make-buffer (* m k) :float) bf (make-buffer (* k n) :float)
              a16 (make-buffer (* m k) :half) b16 (make-buffer (* k n) :half)
              c-plain (make-buffer (* m n) :float)
              c-split (make-buffer (* m n) :float)
              parts (make-buffer (* splits m n) :float)]
          (upload! af A)
          (upload! bf B)
          (let [g (record-graph! [{:bound (convert! af a16 (* m k))}
                                  {:bound (convert! bf b16 (* k n))}
                                  {:bound (gemm! a16 b16 c-plain m n k :float)}
                                  {:bound (splitk! a16 b16 parts m n k kc splits)}
                                  {:bound (reduce! parts c-split (* m n) splits)}])]
            (replay! g)
            (let [P (download c-plain) S (download c-split)
                  err (rel-l1 S P)]
              (testing (str "m=" m " n=" n " k=" k " splits=" splits)
                (is (< err 1.0e-4)
                    (str "split-k vs plain XMX gemm rel-L1 " err))))
            (destroy! g))
          (doseq [b [af bf a16 b16 c-plain c-split parts]] (free! b)))))))

(deftest split-k-policy-only-fires-on-low-occupancy-gemms
  (require 'raster.gpu.core)
  (let [core (find-ns 'raster.gpu.core)
        fill @(ns-resolve core '*gemm-splitk-fill-wgs*)
        target @(ns-resolve core '*gemm-splitk-target-wgs*)
        min-chunk @(ns-resolve core '*gemm-splitk-min-chunk*)
        max-splits @(ns-resolve core '*gemm-splitk-max-splits*)
        ;; the same decision bind-program! makes (kept here as an executable spec of
        ;; the policy — the binder's copy is a closure over the step's m/n/k)
        decide (fn [m n k]
                 (let [base (* (Math/ceil (/ (double n) 128.0))
                               (Math/ceil (/ (double m) 128.0)))]
                   (if (>= base (double fill))
                     1
                     (let [want (long (Math/ceil (/ (double target) base)))
                           cap (quot (long k) (long min-chunk))
                           s (min want cap (long max-splits))]
                       (if (< s 2)
                         1
                         (let [kc (* 32 (quot (+ (quot (long k) s) 31) 32))]
                           (quot (+ (long k) kc -1) kc)))))))]
    (testing "the tied-embedding backward (5 workgroups, k=262144) is split"
      (is (> (decide 13 640 262144) 1)))
    (testing "a GEMM that already fills the machine is NOT split"
      (is (= 1 (decide 13 262144 640)))   ;; the head's forward logits: 2048 workgroups
      (is (= 1 (decide 640 2048 64))))    ;; a :tn weight-gradient: 80 workgroups
    (testing "a low-occupancy GEMM with a SHORT k is not split (chunks would be tiny)"
      (is (= 1 (decide 64 640 512))))
    (testing "split count stays within the cap"
      (is (<= (decide 13 640 (* 1024 1024)) max-splits)))))

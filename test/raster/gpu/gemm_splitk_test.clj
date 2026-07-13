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

(deftest vectorized-convert-is-bit-exact
  ;; The f32→f16 operand cast is VECTORIZED (w elements per work-item, w from the hardware
  ;; descriptor). `vstore_halfN` rounds to nearest-even — the same rounding as the `(half)`
  ;; cast it replaced — so the result must be BIT-IDENTICAL to the scalar reference, not
  ;; merely close. The interesting cases are the ones a vectorized kernel gets wrong:
  ;; an n that is not a multiple of w (the tail loop) and an n below w (no vector iteration
  ;; at all, and a group count that must not round down to zero).
  (if-not @gp/gpu-available?
    (println "  [SKIP] vectorized convert: no Level Zero GPU")
    (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          make-buffer (ns-resolve ze 'make-buffer)
          upload!     (ns-resolve ze 'array->buffer!)
          download    (ns-resolve ze 'buffer->array)
          free!       (ns-resolve ze 'free-buffer!)
          record!     (ns-resolve ze 'record-graph!)
          replay!     (ns-resolve ze 'replay-graph!)
          destroy!    (ns-resolve ze 'destroy-graph!)
          convert!    (ns-resolve ze 'bind-registered-convert!)]
      (doseq [n [1 2 3 4 5 7 8 33 255 1023 4096 40961]
              w [1 2 4]]
        (let [a  (rnd n (+ n w))
              af (make-buffer n :float)
              h  (make-buffer n :half)]
          (upload! af a)
          (let [g (record! [{:bound (convert! af h n w)}])]
            (replay! g) (destroy! g))
          ;; buffer->array on a :half buffer returns the RAW FP16 BITS (a short[]) — exactly
          ;; what a bit-exactness check wants. Float/floatToFloat16 is the JVM's own RTE
          ;; conversion, so the two shorts must be equal, bit for bit.
          (let [^shorts got (download h)]
            (is (every? true?
                        (for [i (range n)]
                          (= (Float/floatToFloat16 (aget ^floats a i))
                             (aget got i))))
                (str "vectorized convert w=" w " n=" n " is not bit-exact")))
          (free! af) (free! h))))))

(deftest split-k-policy-only-fires-on-low-occupancy-gemms
  ;; Calls the BINDER'S OWN decision (gpu.core/gemm-schedule) — this test used to
  ;; re-implement the policy by hand, which meant it could not catch a regression in the
  ;; copy that actually schedules the GEMMs. One policy, one caller, one spec.
  (require 'raster.gpu.core)
  (let [core (find-ns 'raster.gpu.core)
        schedule @(ns-resolve core 'gemm-schedule)
        max-splits @(ns-resolve core '*gemm-splitk-max-splits*)
        fill 32                                   ;; the Arc 140V's 8192 lanes / 256-item wg
        decide (fn [m n k] (first (schedule m n k fill)))]
    (testing "the tied-embedding backward (5 workgroups, k=262144) is split"
      (is (> (decide 13 640 262144) 1)))
    (testing "a GEMM that already fills the machine is NOT split"
      (is (= 1 (decide 13 262144 640)))   ;; the head's forward logits: 2048 workgroups
      (is (= 1 (decide 640 2048 64))))    ;; a :tn weight-gradient: 80 workgroups
    (testing "a low-occupancy GEMM with a SHORT k is not split (chunks would be tiny)"
      (is (= 1 (decide 64 640 512))))
    (testing "split count stays within the cap"
      (is (<= (decide 13 640 (* 1024 1024)) max-splits)))
    (testing "the k-chunk is a multiple of the 32-wide K-unroll and covers all of k"
      (let [[s kc] (schedule 13 640 262144 fill)]
        (is (zero? (mod kc 32)))
        (is (>= (* s kc) 262144))))
    (testing "the schedule follows the HARDWARE, not the shape alone"
      ;; the same GEMM on a machine that only needs 4 workgroups to fill is NOT starved,
      ;; so it is not split — the fill count is an input, not a constant.
      (is (> (decide 13 640 262144) 1))
      (is (= 1 (first (schedule 13 640 262144 4)))))))

(deftest hardware-derived-launch-geometry
  ;; The machine width is a HardwareDescriptor field, not a literal at each launch site.
  (require 'raster.compiler.core.hardware)
  (let [hw (find-ns 'raster.compiler.core.hardware)
        fill-wgs @(ns-resolve hw 'fill-workgroups)
        vec-width @(ns-resolve hw 'stream-vector-width)
        arc {:machine-lanes 8192}]                ;; 64 EU x 8 threads x 16 lanes
    (testing "fill-workgroups = machine-lanes / workgroup-size"
      (is (= 32 (fill-wgs arc 256)))
      (is (= 64 (fill-wgs arc 128)))
      (is (= 16 (fill-wgs {:machine-lanes 4096} 256))))
    (testing "stream-vector-width takes the widest vector that STILL FILLS the machine"
      ;; the whole trap: float4 on a mid-sized kernel removes work-items faster than it
      ;; saves memory requests, and half the EUs go idle.
      (is (= 4 (vec-width arc (* 8192 4))))       ;; 32768 → 8192 float4 items = full
      (is (= 2 (vec-width arc (* 8192 2))))       ;; 16384 → 8192 float2 items = full
      (is (= 1 (vec-width arc 8192)))             ;; 8192  → scalar is already exactly full
      (is (= 1 (vec-width arc 1024))))            ;; too small to fill at any width
    (testing "a wider machine needs a bigger n before it pays to vectorize"
      (is (= 1 (vec-width {:machine-lanes 65536} 32768)))
      (is (= 4 (vec-width {:machine-lanes 65536} (* 65536 4)))))))

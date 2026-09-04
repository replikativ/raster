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
     • the compiler-emitted split-k schedule expression leaves machine-filling GEMMs alone
       and splits the low-occupancy ones."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.typed-matrix-device-support :as support]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]))

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
    (gp/gpu-skip! "gemm split-k")
    (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          record-graph! (ns-resolve ze 'record-graph!)
          replay!       (ns-resolve ze 'replay-graph!)
          destroy!      (ns-resolve ze 'destroy-graph!)
          splitk!       (ns-resolve ze 'bind-registered-gemm-splitk!)
          reduce!       (ns-resolve ze 'bind-registered-splitk-reduce!)]
      (doseq [[m n k splits kc]
              [[13 640 8192 8 1024]     ;; the LM-head dx shape (small k, same geometry)
               [13 640 8192 26 320]     ;; k/splits not a multiple of kc*splits
               [7 640 4096 4 1024]      ;; m below one 8-row DPAS tile
               [33 200 1000 3 352]      ;; ragged m, n and k; last chunk clipped
               [128 256 4096 8 512]]]
        (let [A (rnd (* m k) 1) B (rnd (* k n) 2)
              scheduled (support/dense-dispatch :ze:0)
              direct (support/matrix-artifact scheduled :xmx-direct)]
          (gpu/with-gpu-session [session :ze:0]
            (gpu/alloc! session {:a16 [:half (* m k) (support/half-array A)]
                                 :b16 [:half (* k n) (support/half-array B)]
                                 :c-plain [:float (* m n) nil]
                                 :c-split [:float (* m n) nil]
                                 :parts [:float (* splits m n) nil]})
            (let [direct-handle
                  (gpu/bind-kernel-executable!
                   session [:typed-direct m n k] direct
                   [:a16 :b16 :c-plain
                    {:type :int :value m}
                    {:type :int :value n}
                    {:type :int :value k}])]
              (try
                (gpu/run-kernel-graph! session direct-handle)
                ;; Explicit split factors remain a legacy benchmark surface until split count is
                ;; represented as a finite compiler schedule candidate in the next increment.
                (let [g (record-graph!
                         [{:bound (splitk! (gpu/buffer session :a16)
                                           (gpu/buffer session :b16)
                                           (gpu/buffer session :parts)
                                           m n k kc splits)
                           :kernel-name "gemm_nonsquare_splitk"}
                          {:bound (reduce! (gpu/buffer session :parts)
                                           (gpu/buffer session :c-split)
                                           (* m n) splits)
                           :kernel-name "splitk_reduce"}])]
                  (try
                    (replay! g)
                    (let [plain (gpu/download session :c-plain)
                          split (gpu/download session :c-split)
                          error (rel-l1 split plain)]
                      (testing (str "m=" m " n=" n " k=" k " splits=" splits)
                        (is (< error 1.0e-4)
                            (str "split-k vs compiler-derived direct matrix rel-L1 " error))))
                    (finally
                      (destroy! g))))
                (finally
                  (gpu/release-kernel-graph! session direct-handle))))))))))

(deftest unrolled-layout-convert-is-bit-exact
  ;; The f32→f16 operand cast schedules w statically unrolled, lane-owned elements per work-item.
  ;; Its explicit nearest-even IEEE conversion must be BIT-IDENTICAL to the scalar reference, not
  ;; merely close. The interesting cases are a non-multiple tail and an extent below w, where the
  ;; launch still needs one workgroup and every inactive unrolled element must remain masked.
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "unrolled layout convert")
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
  ;; Realize the compiler IR expression used by both KernelDispatch selection and graph-private
  ;; split storage. There is no binder copy of this policy anymore.
  (let [tile (hardware/gemm-tile-for nil)
        fill 32                                   ;; the Arc 140V's 8192 lanes / 256-item wg
        decide (fn [m n k & [fill-workgroups]]
                 (launch/resolve-expression
                  {} (gemm/requested-splits
                      {:m m :n n :k k :tile tile
                       :fill-workgroups (or fill-workgroups fill)})))]
    (testing "the tied-embedding backward (5 workgroups, k=262144) is split"
      (is (> (decide 13 640 262144) 1)))
    (testing "a GEMM that already fills the machine is NOT split"
      (is (< (decide 13 262144 640) 2))   ;; the head's forward logits: 2048 workgroups
      (is (< (decide 640 2048 64) 2)))    ;; a :tn weight-gradient: 80 workgroups
    (testing "a low-occupancy GEMM with a SHORT k is not split (chunks would be tiny)"
      (is (< (decide 64 640 512) 2)))
    (testing "split count stays within the cap"
      (is (<= (decide 13 640 (* 1024 1024)) 64)))
    (testing "the k-chunk is a multiple of the 32-wide K-unroll and covers all of k"
      (let [s (decide 13 640 262144)
            kc (launch/resolve-expression
                {} (launch/align-up (launch/ceil-div 262144 s) (:block-k tile)))]
        (is (zero? (mod kc 32)))
        (is (>= (* s kc) 262144))))
    (testing "the schedule follows the HARDWARE, not the shape alone"
      ;; the same GEMM on a machine that only needs 4 workgroups to fill is NOT starved,
      ;; so it is not split — the fill count is an input, not a constant.
      (is (> (decide 13 640 262144) 1))
      (is (< (decide 13 640 262144 4) 2)))))

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
    (testing "stream-vector-width goes as WIDE as it can down to a quarter-machine floor"
      ;; measured: float4 at n=16384 leaves half the EUs idle (16 workgroups of 32) and STILL
      ;; beats float2 at full occupancy — a 16-byte request carries more memory-level
      ;; parallelism per thread than the idle lanes cost. Wide beats full.
      (is (= 4 (vec-width arc 32768)))            ;; 8192 float4 items
      (is (= 4 (vec-width arc 16384)))            ;; 4096 items = HALF the machine, still w4
      (is (= 4 (vec-width arc 8192)))             ;; 2048 items = the quarter-machine floor
      (is (= 2 (vec-width arc 4096)))             ;; below it at w4 → step down
      (is (= 1 (vec-width arc 1024))))            ;; launch-bound: widening only removes lanes
    (testing "the floor scales with the machine — a wider part needs a bigger n"
      (is (= 1 (vec-width {:machine-lanes 65536} 8192)))
      (is (= 4 (vec-width {:machine-lanes 65536} 65536))))))

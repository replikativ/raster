(ns raster.compiler.backend.gpu.gemm-tiled-test
  "Correctness of the tile-parametric XMX GEMM generator (raster...opencl-codegen/emit-gemm-tiled),
   which replaced the hand-unrolled kernel. Two guards:

     (1) ABSOLUTE correctness — the resident GEMM (default Arc tile, launched through the production
         bind-registered-gemm! path) matches an independent CPU f32-accumulate reference over
         f16-rounded inputs. This catches a miscompiled tile that produces zeros/garbage (the exact
         failure mode a silent GPU miscompile would show).

     (2) CROSS-TILE bit-invariance — a non-default tile (block 64×64, block-k 64) produces output
         BIT-IDENTICAL to the default tile. Every output element's K-reduction is the same DPAS
         sequence regardless of how the tile assigns elements to subgroups/workgroups, so a correct
         generator is bit-invariant across tile geometry. This is the guard the T2/T3 autotune work
         (which VARIES the tile) rests on — a tile bug that changes results at all trips it.

   Device-free structural checks live in raster.dl.gsdm-test/gemm-tiled-kernel-test."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gpu-grad-parity :as gp]))

(defn- cpu-gemm-ref
  "C[m×n] = A[m×k]·B[k×n], f16-rounded inputs, f64 accumulate → float (the independent oracle)."
  [^floats A ^floats B m n k]
  (let [h  (fn [x] (Float/float16ToFloat (Float/floatToFloat16 (float x))))
        C  (float-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (let [acc (loop [p 0 s 0.0]
                    (if (< p k)
                      (recur (inc p) (+ s (* (double (h (aget A (+ (* i k) p))))
                                             (double (h (aget B (+ (* p n) j)))))))
                      s))]
          (aset C (+ (* i n) j) (float acc)))))
    C))

(defn- rms [a b] (Math/sqrt (/ (reduce + (map (fn [x y] (let [d (- (double x) (double y))] (* d d))) a b))
                               (double (count a)))))

(deftest tiled-gemm-matches-cpu-and-is-tile-invariant
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "tile-parametric GEMM: CPU-reference correctness + cross-tile bit-invariance")
    (let [ze   (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          sv   (requiring-resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv)
          emit (requiring-resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-tiled)
          r    (fn [s] (ns-resolve ze s))
          _    (@(r 'ensure-init!))                 ;; MUST init before reading device-hex (else :device nil → generic compile, no DPAS)
          I32  @(r 'I32)
          dev  (:device-id-hex @@(r 'state))
          f16  @(r 'buffer-of-floats-as-half)  MB @(r 'make-buffer)
          LM   @(r 'load-module!)  CK @(r 'create-kernel-fresh)  BK2 @(r 'bind-kernel-2d!)
          RG   @(r 'record-graph!) RP @(r 'replay-graph!) DG @(r 'destroy-graph!)
          BA   @(r 'buffer->array) FB @(r 'free-buffer!)  breg @(r 'bind-registered-gemm!)
          seti (fn [seg off v] (.set ^java.lang.foreign.MemorySegment seg I32 (int off) (int v)))
          ;; launch an arbitrary tiled kernel source at (m,n,k) with an explicit (block-m,block-n) grid
          run  (fn [src kname a16 b16 m n block-m block-n k]
                 (let [mod (LM (sv src :device dev)) kh (CK mod kname) c (MB (* m n) :float)
                       bnd (BK2 kh [256 1] [(:segment a16) (:segment b16) (:segment c)
                                            {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}])
                       gc (:gc-seg bnd)]
                   (seti gc 0 (Math/ceil (/ (double n) (double block-n))))
                   (seti gc 4 (Math/ceil (/ (double m) (double block-m))))
                   (seti gc 8 1)
                   (let [g (RG [{:bound bnd :kernel-name kname}])] (RP g) (let [o (BA c)] (DG g) (FB c) o))))
          rng  (java.util.Random. 7)  m 64 n 128 k 256
          mk   (fn [s] (let [a (float-array s)] (dotimes [i s] (aset a i (float (* 0.1 (.nextGaussian rng))))) a))
          A    (mk (* m k))  B (mk (* k n))
          a16  (f16 A)  b16 (f16 B)]
      (try
        (testing "(1) resident GEMM (production path) matches the CPU f32 reference"
          (let [c (MB (* m n) :float)
                ;; :float output arity (7-arg) — the 6-arg default is :half and would write
                ;; 2-byte halfs into this 4-byte float buffer (→ garbage read back as float).
                g (RG [{:bound (breg a16 b16 c m n k :float) :kernel-name "gemm_nonsquare_float"}])
                _ (RP g)  gpu (BA c)  ref (cpu-gemm-ref A B m n k)
                err (rms gpu ref)]
            (DG g) (FB c)
            (is (< err 1.0e-2) (str "resident GEMM vs CPU reference RMS=" err " (miscompile → zeros/garbage would blow this up)"))
            (is (pos? (count (filter #(> (Math/abs (double %)) 0.05) gpu))) "output is non-trivial (not all zeros)")))
        (testing "(2) a non-default tile is bit-identical to the default tile"
          (let [dflt (run (emit "g_def" :c-dtype :float) "g_def" a16 b16 m n 128 128 k)
                t64  (run (emit "g64" :c-dtype :float :block-m 64 :block-n 64) "g64" a16 b16 m n 64 64 k)
                tk64 (run (emit "gk64" :c-dtype :float :block-k 64) "gk64" a16 b16 m n 128 128 k)]
            (is (= (seq dflt) (seq t64))  "block 64×64 tile ≡ default (bit-identical)")
            (is (= (seq dflt) (seq tk64)) "block-k 64 tile ≡ default (bit-identical)")))
        (finally (FB a16) (FB b16))))))

(deftest fused-epilogue-gemm-binder
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "fused-epilogue GEMM (bind-registered-gemm-epilogue!): C = A·B + bias")
    (let [ze   (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
          hw   (do (require 'raster.compiler.core.hardware) (find-ns 'raster.compiler.core.hardware))
          r    (fn [s] @(ns-resolve ze s))
          _    ((r 'ensure-init!))
          f16  (r 'buffer-of-floats-as-half) MB (r 'make-buffer) RG (r 'record-graph!)
          RP (r 'replay-graph!) DG (r 'destroy-graph!) BA (r 'buffer->array) FB (r 'free-buffer!)
          bind (r 'bind-registered-gemm-epilogue!)
          tile ((ns-resolve hw 'derive-gemm-tile) ((ns-resolve hw 'descriptor-for) :ze:0))
          m 64 n 128 k 256 rng (java.util.Random. 3)
          mk (fn [s] (let [a (float-array s)] (dotimes [i s] (aset a i (float (* 0.1 (.nextGaussian rng))))) a))
          A (mk (* m k)) B (mk (* k n)) bias (mk n)
          a16 (f16 A) b16 (f16 B) biasbuf (f16 bias) c (MB (* m n) :float)
          epi {:key :bias :params ", __global const half* restrict bias"
               :fn (fn [acc _row col] (str acc " + (float)bias[" col "]")) :operands [biasbuf]}]
      (try
        (RP (RG [{:bound (bind a16 b16 c m n k :float tile epi) :kernel-name "gemm_epi_bias_float"}]))
        (let [gpu (BA c) h (fn [x] (Float/float16ToFloat (Float/floatToFloat16 (float x)))) ref (float-array (* m n))]
          (dotimes [i m] (dotimes [j n]
            (let [s (loop [p 0 acc 0.0] (if (< p k) (recur (inc p) (+ acc (* (double (h (aget A (+ (* i k) p)))) (double (h (aget B (+ (* p n) j))))))) acc))]
              (aset ref (+ (* i n) j) (float (+ s (double (h (aget bias j)))))))))
          (let [maxrel (reduce max 0.0 (map (fn [a b] (/ (Math/abs (- (double a) (double b))) (+ 0.05 (Math/abs (double b))))) gpu ref))]
            (is (< maxrel 1.0e-3) (str "fused C=A·B+bias must match CPU ref within f16 tol (max-rel " maxrel ")"))))
        (finally (FB a16) (FB b16) (FB biasbuf) (FB c))))))

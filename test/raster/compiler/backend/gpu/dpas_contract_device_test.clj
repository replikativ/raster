(ns raster.compiler.backend.gpu.dpas-contract-device-test
  "W1 step 4: ON-DEVICE numeric validation of the DPAS/XMX-tensorized contraction emitter
   (Arc GPU). Proves the SOAC path (par/contract → segmented SegRed → dpas-contraction-legal?
   → emit-gemm-nonsquare-kernel body) produces a NUMERICALLY CORRECT peak kernel: its f16
   output matches a CPU matmul on the SAME f16-rounded inputs (f32 accumulate) within f16
   tolerance. Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- f16 ^double [^double x]
  ;; round a value through IEEE-754 half, matching what the GPU reads/writes
  (double (Float/float16ToFloat (Float/floatToFloat16 (float x)))))

(defn- ref-matmul-f16
  "CPU matmul on f16-rounded inputs, f64 accumulate — the reference the DPAS kernel targets."
  [^doubles A ^doubles B m k n]
  (let [C (double-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (loop [l 0 acc 0.0]
                (if (< l k)
                  (recur (inc l) (+ acc (* (f16 (aget A (+ (* i k) l)))
                                           (f16 (aget B (+ (* l n) j))))))
                  acc)))))
    C))

(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- run-dpas
  "Emit the DPAS-tensorized kernel for m×n×k from the SOAC segred, launch on device (f16),
   return {:gpu :cpu}."
  [m k n]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        make-buffer (ns-resolve ze 'make-buffer)
        buf-of-halfs (ns-resolve ze 'buffer-of-floats-as-half)
        buf->doubles (ns-resolve ze 'buffer->double-array)
        launch-2d! (ns-resolve ze 'launch-2d!)
        ;; small, zero-centered, bounded operands: the accumulated matmul stays well within
        ;; f16 range (max ~65504) so the f16 OUTPUT store doesn't overflow (f32 acc is fine).
        Ad (double-array (map #(* 0.1 (- (double (mod % 7)) 3.0)) (range (* m k))))
        Bd (double-array (map #(* 0.1 (- (double (mod % 5)) 2.0)) (range (* k n))))
        Af (float-array (map float Ad))
        Bf (float-array (map float Bd))
        sr (cl/contract-form->segred (matmul-form m n k) :dtype :half)
        {:keys [kernel-name source array-params tensorized]}
        (sco/generate-dpas-contraction-kernel sr 'C)
        _ (assert tensorized "DPAS gate should accept the canonical matmul")
        _ (assert (= '[A B] array-params) (str "row/col binding order, got " array-params))
        _ (register! kernel-name {:source source :dtype :half})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        abuf (buf-of-halfs Af)
        bbuf (buf-of-halfs Bf)
        cbuf (make-buffer (* m n) :half)
        gcx (long (Math/ceil (/ (double n) 128.0)))
        gcy (long (Math/ceil (/ (double m) 128.0)))
        args [(:segment abuf) (:segment bbuf) (:segment cbuf)
              {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]]
    (launch-2d! kernel-handle [256 1] [gcx gcy] args)
    {:gpu (vec (buf->doubles cbuf))
     :cpu (vec (ref-matmul-f16 Ad Bd m k n))}))

(defn- rel-close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true?
               (map (fn [a b]
                      (let [a (double a) b (double b)
                            denom (max 1.0 (Math/abs b))]
                        (< (/ (Math/abs (- a b)) denom) tol)))
                    xs ys))))

(deftest dpas-contraction-matches-cpu-on-device
  (if-not @gpu?
    (println "[skip] dpas-contraction-device: no GPU device available")
    (do
      (testing "tile-divisible dims (256×128×512)"
        (let [{:keys [gpu cpu]} (run-dpas 256 128 512)]
          (is (rel-close? gpu cpu 2.0e-2)
              (str "divisible: GPU " (take 4 gpu) " vs CPU " (take 4 cpu)))))
      (testing "M-unaligned, pitch-aligned partial tiles (130×96×72) — boundary path"
        ;; M=130 (not a tile multiple) exercises the store row-guard; N=72,K=96 keep the
        ;; operand pitches 16-byte aligned (N%8==0, K%8==0) as the DPAS block-reads require.
        (let [{:keys [gpu cpu]} (run-dpas 130 96 72)]
          (is (rel-close? gpu cpu 2.0e-2)
              (str "boundary: GPU " (take 4 gpu) " vs CPU " (take 4 cpu))))))))

(deftest dpas-gate-discriminates
  (testing "canonical matmul tensorizes; transpose/f64 fall back"
    (let [nn (cl/contract-form->segred (matmul-form 64 64 64) :dtype :half)
          tn (cl/contract-form->segred
              (list 'raster.par/contract 'C [['i 64] ['j 64]] [['l 64]]
                    (list '* (list 'aget 'A (list '+ (list '* 'i 64) 'l))
                          (list 'aget 'B (list '+ (list '* 'j 64) 'l)))) ; B transposed
              :dtype :half)
          f64 (cl/contract-form->segred (matmul-form 64 64 64) :dtype :double)]
      (is (:tensorized (sco/generate-dpas-contraction-kernel nn 'C)))
      (is (= :non-canonical-orientation
             (:reason (sco/generate-dpas-contraction-kernel tn 'C))))
      (is (= :dtype-not-dpas
             (:reason (sco/generate-dpas-contraction-kernel f64 'C :dtype :double))))
      (is (str/includes? (:source (sco/generate-dpas-contraction-kernel nn 'C))
                         "matrix_mad_k16"))))
  (testing "pitch-unaligned dims are REJECTED (would silently miscompile → fall back)"
    ;; N%8≠0 (B pitch = N·2 not 16-byte aligned) and K%8≠0 (A pitch) are device-verified to
    ;; corrupt ~80% of outputs; the gate must reject so the caller uses the register-tiled path.
    (let [n-bad (cl/contract-form->segred (matmul-form 128 70 128) :dtype :half)  ; N=70
          k-bad (cl/contract-form->segred (matmul-form 128 128 124) :dtype :half)] ; K=124
      (is (= :n-pitch-unaligned (:reason (sco/generate-dpas-contraction-kernel n-bad 'C))))
      (is (= :k-pitch-unaligned (:reason (sco/generate-dpas-contraction-kernel k-bad 'C))))
      (is (false? (:tensorized (sco/generate-dpas-contraction-kernel n-bad 'C)))))))

;; ── Q2: EPILOGUE FUSION — fold bias+activation into the peak kernel's store slot ───────
;; emit-gemm-tiled exposes a store-splice hook; the contraction path now drives it from a
;; DOMAIN-AGNOSTIC spec (an expression over the accumulator + operands carrying axis-maps), so
;; bias / activation / residual / dequant are one mechanism and compose by nesting. The win is a
;; whole elementwise kernel and a DRAM round-trip of C removed.
(defn- silu ^double [^double x] (/ x (+ 1.0 (Math/exp (- x)))))

(deftest epilogue-fusion-matches-unfused-two-pass
  (if-not @gpu?
    (println "[skip] epilogue-fusion-device: no GPU")
    (let [ze (find-ns 'raster.gpu.ze-runtime)
          register! (ns-resolve ze 'register-kernel!)
          ensure! (ns-resolve ze 'ensure-kernel-loaded!)
          make-buffer (ns-resolve ze 'make-buffer)
          halfs (ns-resolve ze 'buffer-of-floats-as-half)
          arr->buf! (ns-resolve ze 'array->buffer!)
          launch-2d! (ns-resolve ze 'launch-2d!)
          buf->d (ns-resolve ze 'buffer->double-array)
          am (requiring-resolve 'raster.compiler.ir.axis-map/of-axes)
          M 128 K 64 N 128
          Ad (double-array (map #(* 0.05 (- (double (mod % 7)) 3.0)) (range (* M K))))
          Bd (double-array (map #(* 0.05 (- (double (mod % 5)) 2.0)) (range (* K N))))
          biasd (float-array (map #(float (* 0.01 (- (mod % 9) 4))) (range N)))
          sr (cl/contract-form->segred (matmul-form M N K) :dtype :half)
          ep {:acc 'acc
              :expr (list 'silu_f (list 'raster.numeric/+ 'acc (list 'aget 'bias 'j)))
              :operands [{:sym 'bias :map (am [['j N]]) :dtype :float}]
              :helpers "inline float silu_f(float x){ return x / (1.0f + native_exp(-x)); }"}
          run (fn [k extra-args out]
                (register! (:kernel-name k) {:source (:source k) :dtype :half})
                (let [{:keys [kernel-handle]} (ensure! (:kernel-name k))
                      abuf (halfs (float-array (map float Ad)))
                      bbuf (halfs (float-array (map float Bd)))
                      args (into [(:segment abuf) (:segment bbuf) (:segment out)
                                  {:type :int :value M} {:type :int :value N} {:type :int :value K}]
                                 extra-args)]
                  (launch-2d! kernel-handle (:workgroup k)
                              [(long (Math/ceil (/ (double N) (:block-n (:tile k)))))
                               (long (Math/ceil (/ (double M) (:block-m (:tile k)))))]
                              args)
                  (vec (buf->d out))))
          plain-k (sco/generate-dpas-contraction-kernel sr 'C)
          fused-k (sco/generate-dpas-contraction-kernel sr 'C :epilogue ep)
          bias-buf (arr->buf! (make-buffer N :float) biasd)
          unfused (run plain-k [] (make-buffer (* M N) :half))
          fused   (run fused-k [(:segment bias-buf)] (make-buffer (* M N) :half))]
      (testing "the fused kernel gains the operand param and the helper"
        (is (str/includes? (:source fused-k) "restrict bias"))
        (is (str/includes? (:source fused-k) "silu_f(float x)"))
        (is (str/includes? (:source fused-k) "silu_f(((acc00.s0) + bias[col]))")
            "the axis-map's j must bind to the store slot's col"))
      (testing "fused result == unfused GEMM followed by a separate bias+silu pass"
        ;; the two-pass reference: read C back, then apply the epilogue on the host
        (let [ref (vec (map-indexed (fn [idx v] (silu (+ (double v) (aget biasd (mod idx N)))))
                                    unfused))]
          (is (= (count ref) (count fused)))
          ;; f16 store on both sides; compare at f16 tolerance
          (is (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 3.0e-2) fused ref))
              (str "fused " (take 4 fused) " vs two-pass " (take 4 ref))))))))

;; ── Q2b: epilogue LEGALITY + the honest cost model (device-free) ───────────────────────
(deftest epilogue-legality-refuses-the-cases-that-would-miscompile
  (let [am (requiring-resolve 'raster.compiler.ir.axis-map/of-axes)
        bias {:acc 'acc :expr (list 'raster.numeric/+ 'acc (list 'aget 'bias 'j))
              :operands [{:sym 'bias :map (am [['j 512]]) :dtype :float}]}]
    (testing "a bias/activation epilogue is legal"
      (is (:ok (sco/epilogue-legal? bias))))
    (testing "the accumulator must appear exactly once (>1 duplicates the reduction result)"
      (is (= :accumulator-used-more-than-once
             (:reason (sco/epilogue-legal? {:acc 'acc :expr (list 'raster.numeric/* 'acc 'acc)}))))
      (is (= :epilogue-ignores-accumulator
             (:reason (sco/epilogue-legal? {:acc 'acc :expr (list 'aget 'bias 'j)})))))
    (testing "a reduction inside the epilogue is refused with its own reason (re-tiling, not yet done)"
      (is (= :reduction-in-epilogue
             (:reason (sco/epilogue-legal? {:acc 'acc :expr (list 'raster.par/reduce 'a 0.0 'i 8 'acc)})))))
    (testing "layout/distribution-changing ops are refused"
      (is (= :layout-changing-op-in-epilogue
             (:reason (sco/epilogue-legal? {:acc 'acc :expr (list 'raster.par/scan 'o 'a 0.0 'i 8 nil 'acc)})))))
    (testing "epilogue-splice itself enforces legality (fails loud, never silently miscompiles)"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"illegal epilogue"
            (sco/epilogue-splice {:acc 'acc :expr (list 'raster.numeric/* 'acc 'acc)} '[i j] :float))))))

(deftest epilogue-cost-reflects-the-eliminated-round-trip
  (let [hw (requiring-resolve 'raster.compiler.core.hardware/derive-gemm-tile)
        am (requiring-resolve 'raster.compiler.ir.axis-map/of-axes)
        bias {:acc 'acc :expr (list 'raster.numeric/+ 'acc (list 'aget 'bias 'j))
              :operands [{:sym 'bias :map (am [['j 512]]) :dtype :float}]}
        c (sco/epilogue-cost bias :half [256 512] (hw {}))]
    (testing "fusing an epilogue REMOVES traffic (it does not duplicate work)"
      (is (:profitable c))
      (is (< (:fused-bytes c) (:unfused-bytes c)))
      ;; unfused ≈ write C + read C + write C = 3·M·N·2 bytes; fused ≈ just the operand read
      (is (= (* 3 256 512 2) (- (:unfused-bytes c) (* 512 4))))
      (is (= (* 512 4) (:fused-bytes c)))
      (is (> (:saved-bytes c) 700000)))
    (testing "accumulator register estimate follows the derived tile (sg-m·sg-n/subgroup)"
      (is (= 64 (:acc-regs c))))))

;; ── Q3: the fusion pass DISCOVERS the epilogue (contract → elementwise map ⇒ one kernel) ──
(deftest fusion-pass-discovers-contract-epilogue
  (let [pf (requiring-resolve 'raster.compiler.passes.parallel.par-fusion/fuse-contract-map)
        route (requiring-resolve 'raster.compiler.passes.parallel.contract-route/route-contraction)
        M 128 N 128 K 64
        mm (list 'raster.par/contract 'C [['i M] ['j N]] [['l K]]
                 (list '* (list 'aget 'A (list '+ (list '* 'i K) 'l))
                       (list 'aget 'B (list '+ (list '* 'l N) 'j))))
        bindings ['C mm
                  'out (list 'raster.par/map! 'out 't (* M N) nil
                             (list 'silu_f (list 'aget 'C 't)))]]
    (testing "contract + elementwise map collapse into ONE contract carrying an :epilogue"
      (let [r (pf bindings [])
            fused (second (:bindings r))
            opts (apply hash-map (drop 5 fused))]
        (is (= 1 (:fused r)))
        (is (= 2 (count (:bindings r))) "4 bindings → 2: the intermediate is gone")
        (is (= 'raster.par/contract (first fused)))
        (is (= 'out (second fused)) "the fused contraction writes the map's target directly")
        (is (some? (:epilogue opts)))
        (is (= (list 'silu_f (:acc (:epilogue opts))) (:expr (:epilogue opts)))
            "the map body becomes the epilogue, with (aget C t) → the accumulator")))
    (testing "the routed descriptor reports the fusion and splices it into the store"
      (let [fused (second (:bindings (pf bindings [])))
            r (route fused :dtype :half)]
        (is (:fused-epilogue r))
        (is (str/includes? (:source r) "silu_f((acc00.s0))")
            "the epilogue must appear in the STORE slot, not a second kernel")))
    (testing "REFUSALS: an already-fused contract, and a map reading other arrays"
      (let [bias-map ['C mm
                      'out (list 'raster.par/map! 'out 't (* M N) nil
                                 (list 'raster.numeric/+ (list 'aget 'C 't) (list 'aget 'bias 't)))]]
        (is (nil? (pf bias-map []))
            "a body reading OTHER arrays needs flat-index→free-axis decomposition; must not fuse yet")))))

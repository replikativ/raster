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

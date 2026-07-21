(ns raster.compiler.backend.gpu.dp4a-contract-device-test
  "W1 quant peak leaf: the DP4A-tensorized int8 contraction (int8×4 dot → int32) — where int8
   quant reaches peak, analogous to f16's DPAS leaf. Exercises the per-leaf layout requirement:
   dp4a needs both operands K-contiguous, so B is [N,K] TRANSPOSED (B[j,l]=j·K+l, the :nt
   orientation). Device output == reference int8 matmul. Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

;; B is [N,K] TRANSPOSED: element = A[i,l]·B[j,l] (both operands K-contiguous for dp4a packing)
(defn- dp4a-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'j k) 'l)))))

(defn- ref-dp4a [^bytes A ^bytes B m k n scale]
  (let [scale (double scale) C (float-array (* m n))]
    (dotimes [i m] (dotimes [j n]
      (let [acc (loop [l 0 a 0]
                  (if (< l k)
                    (recur (inc l) (+ a (* (int (aget A (+ (* i k) l))) (int (aget B (+ (* j k) l))))))
                    a))]
        (aset C (+ (* i n) j) (float (* scale (double acc)))))))
    C))

(defn- run-dp4a [m k n scale]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        make-buffer (ns-resolve ze 'make-buffer)
        arr->buf! (ns-resolve ze 'array->buffer!)
        buf->floats (ns-resolve ze 'buffer->float-array)
        launch-2d! (ns-resolve ze 'launch-2d!)
        A (byte-array (map #(byte (- (mod % 255) 127)) (range (* m k))))       ; [M,K]
        B (byte-array (map #(byte (- (mod (* 3 %) 255) 127)) (range (* n k)))) ; [N,K]
        sr (cl/contract-form->segred (dp4a-form m n k) :dtype :byte)
        {:keys [kernel-name source array-params packed]} (sco/generate-dp4a-contraction-kernel sr 'C :scheme {:scale scale})
        _ (do (assert (= '[A B] array-params)) (assert (= :int8x4 packed)))
        _ (register! kernel-name {:source source :dtype :byte})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        abuf (arr->buf! (make-buffer (* m k) :byte) A)   ; int8 bytes; kernel reads as packed int[]
        bbuf (arr->buf! (make-buffer (* n k) :byte) B)
        obuf (make-buffer (* m n) :float)
        nseg (* m n)
        gx (long (Math/ceil (/ (double nseg) 256.0)))
        args [(:segment abuf) (:segment bbuf) (:segment obuf)
              {:type :float :value (float scale)} {:type :int :value (int nseg)}]]
    (launch-2d! kernel-handle [256 1] [gx 1] args)
    {:gpu (vec (buf->floats obuf)) :cpu (vec (ref-dp4a A B m k n scale))}))

(defn- rel-close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true? (map (fn [a b] (< (/ (Math/abs (- (double a) (double b))) (max 1.0 (Math/abs (double b)))) tol)) xs ys))))

(deftest dp4a-emits-packed-int8x4-tensorize
  (testing "dp4a kernel packs the contract axis into int8×4 words + uses rstr_dp4a (device-free)"
    (let [sr (cl/contract-form->segred (dp4a-form 64 64 64) :dtype :byte)
          {:keys [source packed dtype acc-dtype]} (sco/generate-dp4a-contraction-kernel sr 'C :scheme {:scale 0.01})]
      (is (= :int8x4 packed)) (is (= :byte dtype)) (is (= :int acc-dtype))
      (is (str/includes? source "inline int rstr_dp4a"))       ; the int8×4 dot helper
      (is (str/includes? source "acc = rstr_dp4a("))           ; tensorized MAC
      (is (str/includes? source "const int* restrict A"))      ; operands reinterpreted as packed int
      (is (str/includes? source "p < 16"))))                   ; KP = K/4 = 16
  (testing "gate: K must be a multiple of 4, and B must be [N,K] transposed"
    (is (thrown? AssertionError
                 (sco/generate-dp4a-contraction-kernel (cl/contract-form->segred (dp4a-form 8 8 6) :dtype :byte) 'C)))
    ;; B in [K,N] (non-transposed, B[l,j]=l·N+j) → not K-contiguous → rejected
    (let [bad (list 'raster.par/contract 'C [['i 8] ['j 8]] [['l 8]]
                    (list '* (list 'aget 'A (list '+ (list '* 'i 8) 'l))
                          (list 'aget 'B (list '+ (list '* 'l 8) 'j))))]
      (is (thrown? AssertionError
                   (sco/generate-dp4a-contraction-kernel (cl/contract-form->segred bad :dtype :byte) 'C))))))

(deftest dp4a-contraction-matches-ref-on-device
  (if-not @gpu?
    (println "[skip] dp4a-contract-device: no GPU device available")
    (do
      (testing "int8 dp4a matmul (128×64×256) == reference"
        (let [{:keys [gpu cpu]} (run-dp4a 128 64 256 0.01)]
          (is (rel-close? gpu cpu 1.0e-4) (str "gpu " (take 3 gpu) " vs cpu " (take 3 cpu)))))
      (testing "non-square, K multiple of 4 (100×32×70)"
        (let [{:keys [gpu cpu]} (run-dp4a 100 32 70 0.005)]
          (is (rel-close? gpu cpu 1.0e-4) "non-square"))))))

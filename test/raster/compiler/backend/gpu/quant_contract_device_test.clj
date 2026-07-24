(ns raster.compiler.backend.gpu.quant-contract-device-test
  "W1 quant spike: the SAME contraction abstraction spans int8. A par/contract over int8
   operands lowers (via the shared analyze-contraction + orientation gate) to a WIDENING
   kernel — int8 operands, int32 accumulate — with a dequant epilogue (out = scale·acc), and
   its f32 output matches a reference int8 matmul on device. Proves quantized matmul is the
   same SOAC contraction with different (operand/accumulate/output dtype, epilogue) facets;
   the dp4a / int8-DPAS packed dot is the tensorize leaf (perf layer). Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco])
  (:import [java.lang.foreign MemorySegment]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- ref-quant-matmul [^bytes A ^bytes B m k n scale a-zp b-zp]
  (let [scale (double scale) a-zp (int a-zp) b-zp (int b-zp)
        C (float-array (* m n))]
    (dotimes [i m] (dotimes [j n]
      (let [acc (loop [l 0 a 0]
                  (if (< l k)
                    (recur (inc l) (+ a (* (- (int (aget A (+ (* i k) l))) a-zp)
                                           (- (int (aget B (+ (* l n) j))) b-zp))))
                    a))]
        (aset C (+ (* i n) j) (float (* scale (double acc)))))))
    C))

(defn- run-quant [m k n scheme]
  (let [{:keys [scale a-zp b-zp] :or {scale 1.0 a-zp 0 b-zp 0}} scheme
        ze (find-ns 'raster.gpu.ze-runtime)
        register! (ns-resolve ze 'register-kernel!)
        ensure-loaded! (ns-resolve ze 'ensure-kernel-loaded!)
        make-buffer (ns-resolve ze 'make-buffer)
        arr->buf! (ns-resolve ze 'array->buffer!)
        buf->floats (ns-resolve ze 'buffer->float-array)
        launch-2d! (ns-resolve ze 'launch-2d!)
        A (byte-array (map #(byte (- (mod % 255) 127)) (range (* m k))))
        B (byte-array (map #(byte (- (mod (* 3 %) 255) 127)) (range (* k n))))
        sr (cl/contract-form->segred (matmul-form m n k) :dtype :byte)
        {:keys [kernel-name source array-params]} (sco/generate-quant-contraction-kernel sr 'C :scheme scheme)
        _ (assert (= '[A B] array-params))
        _ (register! kernel-name {:source source :dtype :byte})
        {:keys [kernel-handle]} (ensure-loaded! kernel-name)
        abuf (arr->buf! (make-buffer (* m k) :byte) A)
        bbuf (arr->buf! (make-buffer (* k n) :byte) B)
        obuf (make-buffer (* m n) :float)
        nseg (* m n)
        gx (long (Math/ceil (/ (double nseg) 256.0)))
        args [(:segment abuf) (:segment bbuf) (:segment obuf)
              {:type :float :value (float scale)} {:type :int :value (int nseg)}]]
    (launch-2d! kernel-handle [256 1] [gx 1] args)
    {:gpu (vec (buf->floats obuf)) :cpu (vec (ref-quant-matmul A B m k n scale a-zp b-zp))}))

(defn- rel-close? [xs ys tol]
  (and (= (count xs) (count ys))
       (every? true? (map (fn [a b] (< (/ (Math/abs (- (double a) (double b))) (max 1.0 (Math/abs (double b)))) tol)) xs ys))))

(deftest quant-contraction-emits-generic-types-and-scheme
  (testing "types come from the GENERIC dtype facet map (not quant-specific); scheme carries semantics"
    (let [sr (cl/contract-form->segred (matmul-form 64 64 64) :dtype :byte)
          {:keys [source dtype acc-dtype out-dtype array-params scheme]}
          (sco/generate-quant-contraction-kernel sr 'C :scheme {:scale 0.02})]
      (is (= :byte dtype))     ; int8 operand = the generic :byte dtype (int8_t)
      (is (= :int acc-dtype))  ; int32 accumulate = the generic :int dtype
      (is (= :float out-dtype))
      (is (= '[A B] array-params))
      (is (= 0.02 (:scale scheme)))
      (is (str/includes? source "const char* restrict A"))   ; :byte → opencl "char"
      (is (str/includes? source "int acc = 0"))              ; :int widening accumulator
      (is (str/includes? source "(int)A["))                  ; widen operand to acc dtype
      (is (str/includes? source "out[seg] = scale * (float)acc"))))  ; dequant epilogue
  (testing "zero-point extends the decode scheme (asymmetric quant) — subtract in the load"
    (let [sr (cl/contract-form->segred (matmul-form 32 32 32) :dtype :byte)
          src (:source (sco/generate-quant-contraction-kernel sr 'C :scheme {:scale 0.1 :a-zp 5 :b-zp -3}))]
      (is (str/includes? src "((int)A[i * 32 + l] - 5)"))
      (is (str/includes? src "((int)B[l * 32 + j] - -3)")))))

(deftest quant-contraction-matches-ref-on-device
  (if-not @gpu?
    (println "[skip] quant-contract-device: no GPU device available")
    (do
      (testing "symmetric int8 matmul (128×96×128) dequant == reference"
        (let [{:keys [gpu cpu]} (run-quant 128 96 128 {:scale 0.01})]
          (is (rel-close? gpu cpu 1.0e-6) (str "gpu " (take 3 gpu) " vs cpu " (take 3 cpu)))))
      (testing "non-square int8 (100×70×64)"
        (let [{:keys [gpu cpu]} (run-quant 100 70 64 {:scale 0.005})]
          (is (rel-close? gpu cpu 1.0e-6) "non-square")))
      (testing "ASYMMETRIC int8 with zero-points (96×80×64) == reference"
        (let [{:keys [gpu cpu]} (run-quant 96 80 64 {:scale 0.008 :a-zp 7 :b-zp -4})]
          (is (rel-close? gpu cpu 1.0e-6) "asymmetric zero-point"))))))

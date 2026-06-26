(ns raster.dl.quant
  "Weight quantization for fast/low-memory inference.

  Q4 block quantization (llama.cpp Q4_0-equivalent quant level): symmetric 4-bit,
  one fp32 scale per block of 32 weights, dequant w = (q - 8) * d. Packed two
  nibbles per byte (adjacent layout: byte k of a block holds weights 2k (low) and
  2k+1 (high)). This is raster's own packing — same quant LEVEL as GGUF Q4_0 for
  fair benchmarking; exact GGUF bit-layout (interleaved halves) is a later concern.

  The hot op is `dequant-matmul-q4`: y = x @ dequant(W)^T, accumulating q*x per
  block and applying the block scale once (a dequant-dot — the stepping stone to a
  fully int8×int4 quantized dot). Activations stay f32 here; an int8-activation
  variant (llama.cpp-style) is the perf follow-up."
  (:refer-clojure :exclude [aget aset alength aclone + - * /])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength alloc-like]]
            [raster.numeric :as n :refer [+ - * /]]))

(def ^:const QK4 32)

(defn quantize-q4
  "Quantize a flat f32 weight array (length divisible by 32) to Q4 block format.
  Returns {:qs byte[len/2], :ds float[len/32]}. For a [out,in] weight matrix in
  row-major order with `in` divisible by 32, this quantizes each row in blocks of
  32 along `in` (the matmul reduction axis)."
  [^floats w]
  (let [len (clojure.core/alength w)
        nb (quot len QK4)
        ds (float-array nb)
        qs (byte-array (quot len 2))]
    (dotimes [b nb]
      (let [base (clojure.core/* b QK4)
            ;; signed max-abs value over the block
            mxv (loop [j 0 amax 0.0 mv 0.0]
                  (if (clojure.core/< j QK4)
                    (let [v (clojure.core/aget w (clojure.core/+ base j))
                          a (Math/abs v)]
                      (if (clojure.core/> a amax) (recur (clojure.core/inc j) a v)
                          (recur (clojure.core/inc j) amax mv)))
                    mv))
            d (clojure.core// mxv -8.0)
            id (if (clojure.core/zero? d) 0.0 (clojure.core// 1.0 d))]
        (clojure.core/aset ds b (float d))
        (dotimes [k 16]
          (let [w0 (clojure.core/aget w (clojure.core/+ base (clojure.core/* 2 k)))
                w1 (clojure.core/aget w (clojure.core/+ base (clojure.core/+ (clojure.core/* 2 k) 1)))
                q0 (clojure.core/max 0 (clojure.core/min 15 (clojure.core/+ (Math/round (clojure.core/* w0 id)) 8)))
                q1 (clojure.core/max 0 (clojure.core/min 15 (clojure.core/+ (Math/round (clojure.core/* w1 id)) 8)))]
            (clojure.core/aset qs (clojure.core/+ (quot base 2) k)
                               (unchecked-byte (clojure.core/bit-or q0 (clojure.core/bit-shift-left q1 4))))))))
    {:qs qs :ds ds}))

(defn dequant-q4
  "Dequantize a Q4 block-quantized array back to f32 (debug/validation)."
  [^bytes qs ^floats ds len]
  (let [out (float-array len)]
    (dotimes [b (quot len QK4)]
      (let [base (clojure.core/* b QK4)
            d (clojure.core/aget ds b)]
        (dotimes [k 16]
          (let [byte-val (clojure.core/bit-and (clojure.core/aget qs (clojure.core/+ (quot base 2) k)) 0xFF)
                q0 (clojure.core/- (clojure.core/bit-and byte-val 0xF) 8)
                q1 (clojure.core/- (clojure.core/bit-shift-right byte-val 4) 8)]
            (clojure.core/aset out (clojure.core/+ base (clojure.core/* 2 k)) (float (clojure.core/* q0 d)))
            (clojure.core/aset out (clojure.core/+ base (clojure.core/+ (clojure.core/* 2 k) 1)) (float (clojure.core/* q1 d)))))))
    out))

;; y[M,out] = x[M,in] @ dequant(W[out,in])^T, with W Q4 block-32 quantized.
;; Per output block: accumulate (q-8)*x over 32 weights, then scale by the block d
;; (dequant-dot). x is f32; qs/ds are the quantized weight. Index/bit math is
;; clojure.core; the float accumulation goes through raster.numeric.
(deftm dequant-matmul-q4
  [x :- (Array float) qs :- (Array byte) ds :- (Array float)
   M :- Long in-dim :- Long out-dim :- Long] :- (Array float)
  (let [y (alloc-like x (* M out-dim))
        nblk (quot in-dim QK4)]
    (dotimes [m M]
      (dotimes [o out-dim]
        (let [row-blk0 (quot (clojure.core/* o in-dim) QK4)
              xrow (clojure.core/* m in-dim)
              acc (loop [blk 0 a 0.0]
                    (if (clojure.core/< blk nblk)
                      (let [d (aget ds (clojure.core/+ row-blk0 blk))
                            i0 (clojure.core/* blk QK4)
                            byte0 (clojure.core/+ (quot (clojure.core/* o in-dim) 2) (quot i0 2))
                            xa (clojure.core/+ xrow i0)
                            blk-acc (loop [k 0 ba 0.0]
                                      (if (clojure.core/< k 16)
                                        (let [bv (clojure.core/bit-and (aget qs (clojure.core/+ byte0 k)) 0xFF)
                                              q0 (clojure.core/- (clojure.core/bit-and bv 0xF) 8)
                                              q1 (clojure.core/- (clojure.core/bit-shift-right bv 4) 8)
                                              x0 (aget x (clojure.core/+ xa (clojure.core/* 2 k)))
                                              x1 (aget x (clojure.core/+ xa (clojure.core/+ (clojure.core/* 2 k) 1)))]
                                          (recur (clojure.core/inc k)
                                                 (+ ba (+ (* x0 (double q0)) (* x1 (double q1))))))
                                        ba))]
                        (recur (clojure.core/inc blk) (+ a (* d blk-acc))))
                      a))]
          (aset y (clojure.core/+ (clojure.core/* m out-dim) o) acc))))
    y))

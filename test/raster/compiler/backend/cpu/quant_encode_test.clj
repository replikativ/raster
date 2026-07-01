(ns raster.compiler.backend.cpu.quant-encode-test
  "The parametric weight quantizer: Q4_0 is one data row (the wrapper must be identical
   to the parametric call), and the SAME encoder produces a valid 8-bit format that
   round-trips much tighter — proving quantization is parametric over {bits, pack} for
   the legacy family (the int-MAC kernel core is shared)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.quant :as q]))

(defn- dequant
  "Reconstruct w ≈ d*(q - zp) from {:wq :ws} per the format (the inverse of the
   unsigned-offset encoding the dpbusd kernel assumes)."
  [{:keys [wq ws]} {:keys [block zero-point pack]} len]
  (let [out (float-array len) blk (long block) zp (long zero-point) half (quot blk 2)]
    (dotimes [b (quot len blk)]
      (let [d (aget ws b) base (* b blk)]
        (if (= pack :nibble-interleaved)
          (dotimes [k half]
            (let [bv (bit-and (long (aget wq (+ (* b half) k))) 0xFF)]
              (aset out (+ base k)        (float (* d (- (bit-and bv 0xF) zp))))
              (aset out (+ base k half)   (float (* d (- (bit-shift-right bv 4) zp))))))
          (dotimes [k blk]
            (aset out (+ base k)
                  (float (* d (- (bit-and (long (aget wq (+ base k))) 0xFF) zp))))))))
    out))

(deftest parametric-encode
  (let [len 256
        w (let [a (float-array len) r (java.util.Random. 42)]
            (dotimes [i len] (aset a i (float (- (.nextDouble r) 0.5)))) a)
        maxerr (fn [fmt]
                 (let [dq (dequant (q/quantize-weight w fmt) fmt len)]
                   (reduce max 0.0 (map (fn [i] (Math/abs (- (aget w i) (aget dq i))))
                                        (range len)))))]
    (testing "the Q4 wrapper is exactly the parametric q4-0 call"
      (let [a (q/quantize-weight-q4 w) b (q/quantize-weight w q/q4-0)]
        (is (java.util.Arrays/equals ^bytes (:wq a) ^bytes (:wq b)))
        (is (java.util.Arrays/equals ^floats (:ws a) ^floats (:ws b)))))
    (testing "byte sizes follow the pack"
      (is (= (quot len 2) (alength ^bytes (:wq (q/quantize-weight w q/q4-0)))))
      (is (= len          (alength ^bytes (:wq (q/quantize-weight w q/q8-0))))))
    (testing "8-bit round-trips far tighter than 4-bit (256 vs 16 levels)"
      (let [e4 (maxerr q/q4-0) e8 (maxerr q/q8-0)]
        (is (pos? e4))
        (is (< e8 (* 0.2 e4)) (str "Q8 err " e8 " should be << Q4 err " e4))))))

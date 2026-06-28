(ns raster.compiler.backend.cpu.quant-k-test
  "Q4_K super-block format — the registry case the flat {bits,pack} params don't cover.
   Proves the pipeline generalizes to a structurally different format: encode/decode
   round-trip soundly, and the asymmetric per-sub-block min beats symmetric Q4_0 on
   offset (non-zero-mean) data — the reason K-quants exist."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.quant :as q]))

(defn- maxerr [^floats w ^floats dq]
  (areduce w i m 0.0 (max m (Math/abs (- (aget w i) (aget dq i))))))

(defn- gen [len lo hi seed]
  (let [a (float-array len) r (java.util.Random. seed)]
    (dotimes [i len] (aset a i (float (+ lo (* (- hi lo) (.nextDouble r)))))) a))

(deftest q4k-roundtrip
  (testing "encode→decode reconstructs within bounded error"
    (let [len 1024 w (gen len -0.5 0.5 7)
          enc (q/quantize-weight-q4k w q/q4-K)
          dq  (q/dequant-q4k enc q/q4-K len)]
      (is (= (quot len 2) (alength ^bytes (:wq enc))))
      (is (= (quot len 256) (alength ^floats (:da enc))))   ; super-blocks
      (is (= (quot len 32) (alength ^bytes (:aq enc))))      ; sub-blocks
      (is (< (maxerr w dq) 0.05) "Q4_K should reconstruct ~symmetric data well")))
  (testing "on OFFSET data the asymmetric min makes Q4_K beat symmetric Q4_0"
    (let [len 1024 w (gen len 1.0 2.0 11)               ; mean ~1.5, no values near 0
          q4k (q/dequant-q4k (q/quantize-weight-q4k w q/q4-K) q/q4-K len)
          ;; Q4_0 reference dequant
          {:keys [wq ws]} (q/quantize-weight w q/q4-0)
          q40 (let [out (float-array len)]
                (dotimes [b (quot len 32)]
                  (let [d (aget ws b) base (* b 32)]
                    (dotimes [k 16]
                      (let [bv (bit-and (long (aget wq (+ (* b 16) k))) 0xFF)]
                        (aset out (+ base k)    (float (* d (- (bit-and bv 0xF) 8))))
                        (aset out (+ base k 16) (float (* d (- (bit-shift-right bv 4) 8))))))))
                out)
          ek (maxerr w q4k) e0 (maxerr w q40)]
      (is (< ek e0) (str "Q4_K err " ek " should beat Q4_0 err " e0 " on offset data")))))

(deftest q4k-q8k-dot-structure
  (testing "the structured K-dot equals dot(dequant-weight, dequant-act) and tracks f32"
    (let [in 512
          w (gen in -0.6 0.6 3) x (gen in -0.8 0.8 9)
          ew (q/quantize-weight-q4k w q/q4-K)
          ea (q/quantize-act-q8k x in q/q4-K)
          dotk (q/q4k-q8k-dot ew ea in q/q4-K)
          ;; reference 1: dot of the dequantized weight & dequantized activation —
          ;; the structured dot must equal this EXACTLY (it's the same reconstruction)
          dqw (q/dequant-q4k ew q/q4-K in)
          dqa (let [out (float-array in) sub 32]
                (dotimes [sb (quot in 256)]
                  (let [d (aget ^floats (:xs ea) sb)]
                    (dotimes [i 256] (let [idx (+ (* sb 256) i)]
                                       (aset out idx (float (* d (aget ^bytes (:xq ea) idx)))))))) out)
          ref-dq (areduce dqw i s 0.0 (+ s (* (aget dqw i) (aget dqa i))))
          ;; reference 2: the true f32 dot — the K-dot tracks it within quant error,
          ;; measured against the L1 product scale (a random dot near-cancels, so error
          ;; vs the cancelling sum is meaningless; vs Σ|w·x| it is the real quant error)
          ref-f32 (areduce w i s 0.0 (+ s (* (aget w i) (aget x i))))
          l1 (areduce w i s 0.0 (+ s (Math/abs (* (aget w i) (aget x i)))))]
      (is (< (Math/abs (- dotk ref-dq)) 1e-2)
          (str "structured K-dot " dotk " must equal dequant·dequant dot " ref-dq))
      (is (< (Math/abs (- dotk ref-f32)) (* 0.02 l1))
          (str "K-dot " dotk " vs f32 " ref-f32 " — err " (Math/abs (- dotk ref-f32))
               " should be < 2% of L1 scale " l1)))))

(deftest q6k-second-variant
  (testing "Q6_K (symmetric, 16×16, int8-scale, 6-bit) round-trips — registry generalizes"
    (let [len 1024 w (gen len -0.7 0.7 5)
          enc (q/quantize-weight-q6k w q/q6-K)
          dq  (q/dequant-q6k enc q/q6-K len)]
      (is (= len (alength ^bytes (:wq enc))))
      (is (= (quot len 16) (alength ^bytes (:sc enc))))   ; 16-elem sub-blocks
      (is (= (quot len 256) (alength ^floats (:ds enc))))
      ;; 6-bit ≈ 64 levels (step ≈ max/32) → tighter than Q4_K's 4-bit (0.05 bound)
      (is (< (maxerr w dq) 0.03) "Q6_K should reconstruct tightly (6-bit)")))
  (testing "Q6_K·q8_K dot equals dequant·dequant exactly and tracks f32"
    (let [in 512 w (gen in -0.6 0.6 13) x (gen in -0.8 0.8 17)
          ew (q/quantize-weight-q6k w q/q6-K)
          ea (q/quantize-act-q8k x in q/q6-K)             ; same activation quantizer, 16-sub
          dotk (q/q6k-q8k-dot ew ea in q/q6-K)
          dqw (q/dequant-q6k ew q/q6-K in)
          dqa (let [out (float-array in)]
                (dotimes [sb (quot in 256)]
                  (let [d (aget ^floats (:xs ea) sb)]
                    (dotimes [i 256] (let [idx (+ (* sb 256) i)]
                                       (aset out idx (float (* d (aget ^bytes (:xq ea) idx)))))))) out)
          ref-dq (areduce dqw i s 0.0 (+ s (* (aget dqw i) (aget dqa i))))
          ref-f32 (areduce w i s 0.0 (+ s (* (aget w i) (aget x i))))
          l1 (areduce w i s 0.0 (+ s (Math/abs (* (aget w i) (aget x i)))))]
      (is (< (Math/abs (- dotk ref-dq)) 1e-2) "Q6_K structured dot == dequant·dequant")
      (is (< (Math/abs (- dotk ref-f32)) (* 0.015 l1)) "Q6_K dot tracks f32 (tighter than Q4_K)"))))

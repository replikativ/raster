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

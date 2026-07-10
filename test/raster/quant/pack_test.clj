(ns raster.quant.pack-test
  "Host-side quant packers (raster.quant.pack) — layout faithfulness of the
  packed buffers against the kernel read conventions and the CPU reference
  packer. Model-free, CPU-only."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.quant.pack :as pack]
            [raster.compiler.backend.cpu.quant :as cq]))

(defn- gen ^floats [n seed]
  (let [a (float-array n) r (java.util.Random. (long seed))]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5))))
    a))

(deftest nextpad-test
  (is (= 256 (pack/nextpad 1)))
  (is (= 256 (pack/nextpad 256)))
  (is (= 512 (pack/nextpad 257)))
  (is (= 768 (pack/nextpad 640))))

(deftest quantize-one-q8-roundtrip
  (testing "signed q8_0 pack: dequant(wp,ws) reconstructs W within per-block step/2"
    (let [out 4 in 64 nb (quot in 32)
          W (gen (* out in) 7)
          {:keys [wp ws]} (pack/quantize-one-q8 W in out "t")]
      (is (= (* out (quot in 4)) (alength ^ints wp)))
      (is (= (* out nb) (alength ^floats ws)))
      (dotimes [o out]
        (dotimes [b nb]
          (let [d (double (aget ^floats ws (+ (* o nb) b)))]
            (dotimes [k 32]
              (let [i (+ (* b 32) k)
                    word (aget ^ints wp (+ (* o (quot in 4)) (* b 8) (quot k 4)))
                    q (unchecked-byte (bit-and (bit-shift-right word (* 8 (rem k 4))) 0xFF))
                    wv (double (aget W (+ (* o in) i)))]
                ;; |round(w/d)*d - w| <= d/2 (+ eps); q must sit in [-127,127]
                (is (<= -127 (long q) 127))
                (is (< (Math/abs (- (* (long q) d) wv)) (+ (* 0.5 d) 1e-7))
                    (str "row " o " elem " i)))))))))
  (testing "in % 32 enforced"
    (is (thrown? AssertionError (pack/quantize-one-q8 (float-array 30) 30 1 "bad")))))

(deftest q4k-of-test
  (testing "no-pad case matches the CPU reference packer byte-for-byte"
    (let [out 2 in 256
          W (gen (* out in) 11)
          {:keys [wp da db aq bq] :as z} (pack/q4k-of W out in)
          ref (cq/quantize-weight-q4k W cq/q4-K)]
      (is (= in (:in z)))
      (is (= out (:out z)))
      (is (java.util.Arrays/equals ^floats da ^floats (:da ref)))
      (is (java.util.Arrays/equals ^floats db ^floats (:db ref)))
      (is (java.util.Arrays/equals ^bytes aq ^bytes (:aq ref)))
      (is (java.util.Arrays/equals ^bytes bq ^bytes (:bq ref)))
      ;; wp is the little-endian int reinterpretation of ref's wq bytes
      (let [^bytes wq (:wq ref)
            bb (.order (java.nio.ByteBuffer/wrap wq) java.nio.ByteOrder/LITTLE_ENDIAN)]
        (is (= (quot (alength wq) 4) (alength ^ints wp)))
        (dotimes [i (alength ^ints wp)]
          (is (= (.getInt bb (* i 4)) (aget ^ints wp i)))))))
  (testing "padding: in=640 pads rows to 768; padded packing == packing the padded W"
    (let [out 3 in 640 inp 768
          W (gen (* out in) 13)
          Wp (float-array (* out inp))
          _ (dotimes [o out] (System/arraycopy W (* o in) Wp (* o inp) in))
          z (pack/q4k-of W out in)
          ref (cq/quantize-weight-q4k Wp cq/q4-K)]
      (is (= inp (:in z)))
      (is (java.util.Arrays/equals ^floats (:da z) ^floats (:da ref)))
      (is (java.util.Arrays/equals ^bytes (:aq z) ^bytes (:aq ref))))))

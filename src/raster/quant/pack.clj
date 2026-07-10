(ns raster.quant.pack
  "Host-side weight packers for the quantized GEMV/GEMM kernel layouts.

  These pack row-major f32 weight matrices into the buffer sets the
  raster.quant.kernels-k device kernels read — the layouts are OWNED by those
  kernels, so the packers live next to them on the substrate (downstream model
  runners like pretrained-rstr call them at bind time):

    - `quantize-one-q8`  → signed q8_0 int32-packed rows {:wp :ws} read by
      qmatmul-i8-gemm! / i8gemv-dp4a!.
    - `q4k-of`           → the Q4_K dp4a buffer set {:wp :da :db :aq :bq}
      read by qmatmul-q4k-dp4a!, delegating to the validated CPU packer
      (raster.compiler.backend.cpu.quant/quantize-weight-q4k), zero-padding
      rows to in % 256 == 0 and reinterpreting the nibble bytes as int[].

  Plain host-side load-time code (defn, not deftm): runs once per weight at
  model-bind time, never inside a compiled program."
  (:require [raster.compiler.backend.cpu.quant :as cq]))

(defn- bytes->ints
  "Reinterpret a little-endian byte[] as int[] (the dp4a kernels read packed
  quants as int[])."
  ^ints [^bytes b]
  (let [n (quot (alength b) 4)
        o (int-array n)
        bb (.order (java.nio.ByteBuffer/wrap b) java.nio.ByteOrder/LITTLE_ENDIAN)]
    (dotimes [i n] (aset o i (.getInt bb (* i 4))))
    o))

(defn nextpad
  "Round `in` up to the next multiple of 256 (the Q4_K super-block size)."
  ^long [^long in]
  (long (* 256 (Math/ceil (/ (double in) 256.0)))))

(defn- padw
  "Zero-pad each row of a row-major [out,in] weight up to inp (Q4_K needs
  in % 256 == 0)."
  ^floats [^floats W ^long out ^long in ^long inp]
  (if (= in inp)
    W
    (let [Wp (float-array (* out inp))]
      (dotimes [o out] (System/arraycopy W (* o in) Wp (* o inp) in))
      Wp)))

(defn q4k-of
  "Quantize a row-major [out,in] f32 weight to the Q4_K dp4a buffer set
  {:wp int[] :da :db floats :aq :bq bytes :in <padded in> :out} — the layout
  qmatmul-q4k-dp4a! reads. Rows are zero-padded to in % 256 == 0."
  [^floats W ^long out ^long in]
  (let [inp (nextpad in)
        z (cq/quantize-weight-q4k (padw W out in inp) cq/q4-K)]
    {:wp (bytes->ints (:wq z)) :da (:da z) :db (:db z) :aq (:aq z) :bq (:bq z)
     :in inp :out out}))

(defn quantize-one-q8
  "Signed q8_0 quantize+pack ONE row-major [out,in] weight: per-32 block
  d = max|w|/127, q in [-127,127], 4 bytes/int32 (little-endian lanes) →
  {:wp int[out*in/4] :ws float[out*in/32] :in :out} — the layout
  qmatmul-i8-gemm! reads. `in` must be %32. `nm` names the tensor in errors."
  [^floats W in out nm]
  (assert (zero? (mod (long in) 32)) (str nm " in=" in " not %32"))
  (let [in (long in) out (long out) nb (quot in 32)
        wp (int-array (* out (quot in 4)))
        ws (float-array (* out nb))]
    (dotimes [o out]
      (dotimes [b nb]
        (let [base (+ (* o in) (* b 32))
              mx (loop [k 0 mm 0.0]
                   (if (< k 32)
                     (recur (inc k) (max mm (Math/abs (double (aget W (+ base k))))))
                     mm))
              d (/ mx 127.0) id (if (pos? d) (/ 1.0 d) 0.0)]
          (aset ws (+ (* o nb) b) (float d))
          (dotimes [w 8]
            (let [e (+ base (* w 4))
                  q (fn [i] (bit-and (long (Math/round (* (aget W (+ e i)) id))) 0xFF))]
              (aset wp (+ (* o (quot in 4)) (* b 8) w)
                    (unchecked-int (bit-or (q 0) (bit-shift-left (q 1) 8)
                                           (bit-shift-left (q 2) 16)
                                           (bit-shift-left (q 3) 24)))))))))
    {:wp wp :ws ws :in in :out out}))

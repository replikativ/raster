(ns raster.perf.dpas-fragment-probe
  "Opt-in physical-layout oracle, not a production kernel or performance benchmark.
   Compare the existing Intel 8r16 A block load with lane-local FP32→FP16 construction."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.ocl-runtime :as ocl]))

(defn probe-artifact []
  (artifact/make
   {:kernel-name "dpas_a_fragment_mapping_oracle"
    :target :opencl-c
    :source
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable
     #pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
     __attribute__((intel_reqd_sub_group_size(16)))
     __kernel void dpas_a_fragment_mapping_oracle(
       __global const float* A, __global const half* H, __global int* bits) {
       int lane = get_sub_group_local_id();
       int row = 8 * (get_group_id(0) / 2);
       int col = 16 * (get_group_id(0) % 2);
       ushort8 loaded;
       intel_sub_group_2d_block_read_16b_8r16x1c(
         (__global void*)H, 64, 16, 64, (int2)(col, row), &loaded);
       short8 converted;
       for (int r = 0; r < 8; ++r) {
         int i = (row + r) * 32 + col + lane;
         converted[r] = as_short(convert_half_rte(A[i]));
         bits[2*i] = (int)loaded[r];
         bits[2*i+1] = (int)as_ushort(converted[r]);
       }
     }"
    :abi [(abi/slot 'A :input :float) (abi/slot 'H :input :half)
          (abi/slot 'bits :output :int)]
    :arguments '[A H bits]
    :launch (launch/spec {:workgroup-size [16] :group-count [4]})
    :effects {:kind :fragment-layout-oracle}
    :provenance {:test-only true :instruction :intel-8r16-a-load}
    :attributes {:compilation {:language-standard "CL2.0"
                               :extensions #{"cl_khr_fp16" "cl_intel_subgroup_2d_block_io"}}}}))

(defn inputs []
  {:coordinates (float-array (map #(/ (- % 256) 8.0) (range 512)))
   :rounding (float-array
              (take 512 (cycle [0.0 -0.0 1.0 -1.0
                                (+ 1.0 (/ 1.0 2048)) (+ 1.0 (/ 3.0 2048))
                                (- (+ 1.0 (/ 1.0 2048)))
                                (/ 1.0 16777216) (/ 1.0 33554432)
                                (/ 3.0 33554432) (/ 1.0 16384)
                                (- (/ 1.0 33554432)) (- (/ 3.0 33554432)) Float/MIN_VALUE
                                65504.0 65520.0 -65520.0
                                Float/POSITIVE_INFINITY Float/NEGATIVE_INFINITY Float/NaN])))})

(defn half-bits-match?
  "NaN class is preserved; NaN payload equality is not part of this oracle's contract."
  [expected actual]
  (or (= expected actual)
      (and (= 0x7c00 (bit-and expected 0x7c00)) (pos? (bit-and expected 0x3ff))
           (<= 0 actual 65535)
           (= 0x7c00 (bit-and actual 0x7c00)) (pos? (bit-and actual 0x3ff)))))

(defn run!
  "Execute four distinct 8×16 tiles in a 16×32 allocation, including nonzero row/K offsets.
   Unsupported devices/compiler failures throw, never count as successful validation."
  []
  (ocl/init!)
  (let [compiled (probe-artifact)
        _ (ocl/register-kernel! (:kernel-name compiled) compiled)]
    {:kind :dpas-a-fragment-layout-oracle
     :device (select-keys (ocl/selected-device-info) [:name :vendor :version :extensions])
     :scope {:production-lowering? false :performance-evidence? false
             :subgroup 16 :tile [8 16] :shape [16 32]
             :rounding :nearest-even :overflow :ieee :nan :class-only}
     :results
     (mapv
      (fn [[id values]]
        (let [halves (short-array (map #(Float/floatToFloat16 (float %)) values))
              expected (mapv #(bit-and 0xffff (int %)) halves)
              live (atom [])
              buffer! (fn [values dtype]
                        (let [b (ocl/make-buffer (count values) dtype)]
                          (swap! live conj b)
                          (ocl/array->buffer! b values)
                          b))]
          (try
            (let [a (buffer! values :float)
                  h (buffer! halves :half)
                  bits (buffer! (int-array (repeat 1024 -1)) :int)]
              (ocl/launch-registered-bound! (ocl/bind-kernel-call (call/make compiled [a h bits])))
              (let [actual (vec (ocl/buffer->array bits))
                    mismatches (vec (for [i (range 512) leg (range 2)
                                          :let [got (nth actual (+ (* 2 i) leg))]
                                          :when (not (half-bits-match? (nth expected i) got))]
                                      {:index i :leg (if (zero? leg) :block-load :lane-convert)
                                       :expected (nth expected i) :actual got}))]
                (when (seq mismatches)
                  (throw (ex-info "DPAS A fragment layout/rounding mismatch"
                                  {:case id :count (count mismatches) :head (take 16 mismatches)})))
                {:case id :checked-components 1024 :passed? true}))
            (finally (doseq [b (reverse @live)] (ocl/free-buffer! b))))))
      (inputs))}))

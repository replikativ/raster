(ns raster.compiler.backend.gpu.tile-input-conversion-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.matrix-target :as target]
            [raster.compiler.backend.gpu.matrix-fragment-source :as fragment]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]
            [raster.gpu.ocl-runtime :as ocl]))

(defn conversion-region []
  (body/->ScalarSSARegion
   ['element] [] [] :float
   [(body/->ScalarCompute (body/value 'half-value :half)
                          (body/cast-expression 'element :half :nearest-even :ieee))]
   'half-value :half))

(defn converted-body []
  (schedule/matrix-body
   {:id :converted-tile-input :row 'A :col 'B :out 'C
    :dimensions [13 32 32] :result-dtype :float
    :input-value-regions {'A (conversion-region)}
    :tile {:block-m 16 :block-n 32 :sg-m 8 :sg-n 16 :block-k 32 :num-stages 1
           :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}}))

(deftest intel-input-conversion-is-an-explicit-typed-load-and-prefetch
  (let [kernel (converted-body)
        emitted (target/emit-matrix-kernel "converted_input" kernel :opencl-intel)
        source (:source emitted)]
    (is (= kernel (body/validate! kernel)))
    (is (= :float (get-in kernel [:parameters 0 :dtype])))
    (is (re-find #"__global const float\* restrict A" source))
    (is (re-find #"convert_half_rte" source))
    (is (re-find #"block_prefetch_32b_8r16x1c" source))
    (is (not (re-find #"block_prefetch_16b_8r16x1c" source)))
    (is (re-find #"a_wb = K \* 4, a_pb = K \* 4" source))
    (is (re-find #"< M \? .* : \(half\)0" source))
    (is (not (re-find #"block_read_16b_8r16x1c" source)))
    (is (some #{ {:expression 'K :op :<= :value 4194304}} (:preconditions emitted)))))

(deftest unsupported-input-regions-never-become-implicit-target-casts
  (let [kernel (converted-body)]
    (doseq [bad [(walk/postwalk #(if (instance? raster.compiler.ir.kernel_body.ScalarExpr %)
                                  (assoc-in % [:options :rounding] :toward-zero) %) kernel)
                 (walk/postwalk #(if (instance? raster.compiler.ir.kernel_body.TilePrefetch %)
                                  (assoc-in % [:layout :dtype] :half) %) kernel)]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (target/emit-matrix-kernel "bad_input" bad :opencl-intel))))
    (doseq [dialect [:cuda :hip]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"transformed tile input"
                            (target/emit-matrix-kernel "bad_target" kernel dialect)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (fragment/emit-matrix-kernel "bad_direct_target" kernel dialect))))))

(defn run-device!
  "Opt-in generated-kernel oracle; tiny allocations, no timing or schedule promotion.
   Includes partial M tiles and compares FP32 output against independently half-rounded inputs."
  []
  (ocl/init!)
  (let [kernel (converted-body)
        emitted (target/emit-matrix-kernel "converted_input_device" kernel :opencl-intel)
        slots (mapv (fn [{:keys [id kind dtype role]}]
                      (abi/slot id kind dtype :role role :c-name (name id)
                                :alignment (get-in emitted [:parameter-alignments id])))
                    (:parameters kernel))
        compiled (artifact/make
                  {:kernel-name "converted_input_device" :target :opencl-c :source (:source emitted)
                   :abi (body-abi/project-contracts slots kernel) :arguments '[A B C M N K]
                   :launch (:launch kernel) :preconditions (:preconditions emitted)
                   :effects {:kind :matrix-oracle} :provenance {:kernel-body (:id kernel)}
                   :attributes {:compilation {:language-standard "CL2.0"
                                              :extensions #{"cl_khr_fp16" "cl_intel_subgroup_2d_block_io"
                                                            "cl_intel_subgroup_matrix_multiply_accumulate"}}}})
        a (float-array (map #(/ (- (mod % 13) 6) 8.0) (range (* 13 32))))
        b (float-array (map #(/ (- (mod % 11) 5) 8.0) (range (* 32 32))))
        half #(Float/floatToFloat16 (float %))
        rounded #(double (Float/float16ToFloat (half %)))
        expected (vec (for [i (range 13) j (range 32)]
                        (float (reduce + (for [k (range 32)]
                                           (* (rounded (aget a (+ (* i 32) k)))
                                              (rounded (aget b (+ (* k 32) j)))))))))
        live (atom [])
        buffer! (fn [values dtype]
                  (let [buf (ocl/make-buffer (count values) dtype)]
                    (swap! live conj buf)
                    (ocl/array->buffer! buf values)
                    buf))]
    (try
      (ocl/register-kernel! (:kernel-name compiled) compiled)
      (let [ab (buffer! a :float) bb (buffer! (short-array (map half b)) :half)
            cb (buffer! (float-array (repeat (* 13 32) Float/NaN)) :float)]
        (ocl/launch-registered-bound!
         (ocl/bind-kernel-call
          (call/make compiled [ab bb cb {:type :int :value 13}
                               {:type :int :value 32} {:type :int :value 32}])))
        (let [actual (vec (ocl/buffer->array cb))]
          (when-not (= expected actual)
            (throw (ex-info "generated converted-input GEMM differs from rounded host oracle"
                            {:expected (take 16 expected) :actual (take 16 actual)})))
          {:passed? true :elements (count actual) :shape [13 32 32] :comparison :exact
           :generated-kernel? true :partial-m-tile? true :temporary-buffers 0}))
      (finally (doseq [buf (reverse @live)] (ocl/free-buffer! buf))))))

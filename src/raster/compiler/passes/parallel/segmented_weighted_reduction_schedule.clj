(ns raster.compiler.passes.parallel.segmented-weighted-reduction-schedule
  "Apply cooperative hardware schedules to schedule-neutral segmented weighted reductions."
  (:require [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as schedule]))

(defn- decline
  [reason & [data]]
  (merge {:ok false :reason reason} data))

(defn plan-subgroup-online
  "Plan one subgroup per segment with one shared score and lane-strided value accumulators.

   The first executable storage rows are dense or CSR paged FP16 KV with either contiguous
   interval or explicitly indexed CSR membership. Those are legality constraints of this
   schedule, not new semantic operation kinds."
  ([plan] (plan-subgroup-online plan nil))
  ([plan desc]
   (let [{:keys [membership storage score value operands output accumulator-dtype] :as plan}
         (swr/validate! plan)
         subgroup-size (hardware/preferred-subgroup-size desc)
         max-workgroup-size (hardware/maximum-workgroup-size desc)
         supported-widths (hardware/supported-subgroup-sizes desc)
         components (:components value)
         operand-dtypes (mapv :dtype operands)]
     (cond
       (and desc (not= :gpu (:device-type desc)))
       (decline :score-reuse-requires-gpu {:device-type (:device-type desc)})

       (or (not (integer? subgroup-size))
           (not (integer? max-workgroup-size)))
       (decline :score-reuse-missing-execution-capability
                {:subgroup-size subgroup-size
                 :max-workgroup-size max-workgroup-size})

       (and (seq supported-widths)
            (not (contains? (set supported-widths) subgroup-size)))
       (decline :score-reuse-subgroup-width-unsupported
                {:subgroup-size subgroup-size
                 :supported-subgroup-sizes (set supported-widths)})

       (not (swr/online-softmax-algebra? plan))
       (decline :score-reuse-requires-online-softmax-algebra)

       (not= :logical-attention-visibility (:kind membership))
       (decline :score-reuse-membership-unsupported {:membership (:kind membership)})

       (not (contains? #{:interval :csr} (:visibility-kind membership)))
       (decline :score-reuse-visibility-unsupported
                {:visibility-kind (:visibility-kind membership)})

       (not= :routed-paged-kv (:kind storage))
       (decline :score-reuse-storage-unsupported {:storage (:kind storage)})

       (not (contains? #{:dense-paged :csr-paged} (:route-kind storage)))
       (decline :score-reuse-route-unsupported {:route-kind (:route-kind storage)})

       (or (not= :none (get-in storage [:k-format :quantization]))
           (not= :none (get-in storage [:v-format :quantization])))
       (decline :score-reuse-quantized-kv-unimplemented
                {:k-format (:k-format storage) :v-format (:v-format storage)})

       (not= :float accumulator-dtype)
       (decline :score-reuse-accumulator-unsupported {:actual accumulator-dtype})

       (let [route-operands (case (:route-kind storage)
                              :dense-paged 3
                              :csr-paged 4)
             visibility-operands (if (= :csr (:visibility-kind membership)) 2 0)
             expected-count (+ 5 route-operands visibility-operands)]
         (not (and (= expected-count (count operand-dtypes))
                   (contains? #{:half :float} (first operand-dtypes))
                   (= [:int :int :half :half]
                      (subvec operand-dtypes 1 5))
                   (every? #(= :int %) (subvec operand-dtypes 5))
                   (contains? #{:half :float} (:dtype output)))))
       (decline :score-reuse-storage-dtypes-unsupported
                {:operand-dtypes operand-dtypes :output-dtype (:dtype output)})

       (or (not (integer? components)) (not (pos? components)))
       (decline :score-reuse-static-value-width-required {:components components})

       (or (not (pos? (long subgroup-size)))
           (> (long subgroup-size) (long max-workgroup-size)))
       (decline :score-reuse-invalid-subgroup-geometry
                {:subgroup-size subgroup-size :max-workgroup-size max-workgroup-size})

       ;; A static cap makes register-state feasibility explicit. Gemma D=256/SG16 uses 16.
       (> (quot (+ components (dec (long subgroup-size))) (long subgroup-size)) 32)
       (decline :score-reuse-register-state-too-wide
                {:components components :subgroup-size subgroup-size
                 :components-per-lane
                 (quot (+ components (dec (long subgroup-size))) (long subgroup-size))})

       :else
       {:ok true
        :schedule
        (schedule/make
         {:strategy :subgroup-online-score-reuse
          :workgroup-size (long subgroup-size)
          :segment-mapping :one-workgroup-per-segment
          :membership-traversal (case (:visibility-kind membership)
                                  :interval :contiguous-interval
                                  :csr :csr-row)
          :score-reduction {:kind :subgroup :width (long subgroup-size)
                            :axis (get-in score [:axis :name])}
          :value-mapping {:kind :lane-strided
                          :components components
                          :components-per-lane
                          (quot (+ components (dec (long subgroup-size)))
                                (long subgroup-size))}
          :state {:kind :online-normalized-weighted-sum
                  :components [:maximum :denominator :weighted-values]}
          :numerical-mode {:score-accumulate accumulator-dtype
                           :state-accumulate accumulator-dtype
                           ;; The OpenCL subgroup builtin fixes neither its association tree nor
                           ;; bitwise result. A target with an explicit shuffle tree may choose a
                           ;; stricter schedule later; this row records exactly what it emits.
                           :dot-order :implementation-defined
                           :online-rescale? true}
          :attributes {:storage-kind (:kind storage)
                       :route-kind (:route-kind storage)
                       :visibility-kind (:visibility-kind membership)}})}))))

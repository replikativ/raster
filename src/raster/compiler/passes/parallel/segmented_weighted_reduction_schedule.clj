(ns raster.compiler.passes.parallel.segmented-weighted-reduction-schedule
  "Apply cooperative hardware schedules to schedule-neutral segmented weighted reductions."
  (:require [clojure.string :as str]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as schedule]))

(defn- decline
  [reason & [data]]
  (merge {:ok false :reason reason} data))

(defn plan-subgroup-online
  "Plan one subgroup per segment with one shared score and lane-strided value accumulators.

   The first executable storage row is dense paged FP16 KV with interval visibility.  Those are
   legality constraints of this schedule, not new semantic operation kinds."
  ([plan] (plan-subgroup-online plan nil))
  ([plan desc]
   (let [{:keys [membership storage score value operands output accumulator-dtype] :as plan}
         (swr/validate! plan)
         subgroup-size (long (or (:subgroup-size desc) 16))
         max-workgroup-size (long (or (:max-workgroup-size desc) 256))
         vendor (some-> (:vendor desc) str str/lower-case)
         matrix-family (get-in desc [:matrix :family])
         known-non-intel? (and (or vendor matrix-family)
                               (not (or (= :dpas matrix-family)
                                        (and vendor (str/includes? vendor "intel")))))
         components (:components value)
         operand-dtypes (mapv :dtype operands)]
     (cond
       (and desc (not= :gpu (:device-type desc)))
       (decline :score-reuse-requires-gpu {:device-type (:device-type desc)})

       known-non-intel?
       (decline :score-reuse-requires-intel-subgroup-dialect
                {:vendor (:vendor desc) :matrix-family matrix-family})

       (not (swr/online-softmax-algebra? plan))
       (decline :score-reuse-requires-online-softmax-algebra)

       (not= :logical-attention-visibility (:kind membership))
       (decline :score-reuse-membership-unsupported {:membership (:kind membership)})

       (not= :interval (:visibility-kind membership))
       (decline :score-reuse-visibility-unsupported
                {:visibility-kind (:visibility-kind membership)})

       (not= :routed-paged-kv (:kind storage))
       (decline :score-reuse-storage-unsupported {:storage (:kind storage)})

       (not= :dense-paged (:route-kind storage))
       (decline :score-reuse-route-unsupported {:route-kind (:route-kind storage)})

       (or (not= :none (get-in storage [:k-format :quantization]))
           (not= :none (get-in storage [:v-format :quantization])))
       (decline :score-reuse-quantized-kv-unimplemented
                {:k-format (:k-format storage) :v-format (:v-format storage)})

       (not= :float accumulator-dtype)
       (decline :score-reuse-accumulator-unsupported {:actual accumulator-dtype})

       (not (and (= 8 (count operand-dtypes))
                 (contains? #{:half :float} (first operand-dtypes))
                 (= [:int :int :half :half :int :int :int]
                    (subvec operand-dtypes 1 8))
                 (contains? #{:half :float} (:dtype output))))
       (decline :score-reuse-storage-dtypes-unsupported
                {:operand-dtypes operand-dtypes :output-dtype (:dtype output)})

       (or (not (integer? components)) (not (pos? components)))
       (decline :score-reuse-static-value-width-required {:components components})

       (or (not (pos? subgroup-size)) (> subgroup-size max-workgroup-size))
       (decline :score-reuse-invalid-subgroup-geometry
                {:subgroup-size subgroup-size :max-workgroup-size max-workgroup-size})

       ;; A static cap makes register-state feasibility explicit. Gemma D=256/SG16 uses 16.
       (> (quot (+ components (dec subgroup-size)) subgroup-size) 32)
       (decline :score-reuse-register-state-too-wide
                {:components components :subgroup-size subgroup-size
                 :components-per-lane (quot (+ components (dec subgroup-size)) subgroup-size)})

       :else
       {:ok true
        :schedule
        (schedule/make
         {:strategy :subgroup-online-score-reuse
          :workgroup-size subgroup-size
          :segment-mapping :one-workgroup-per-segment
          :membership-traversal :sequential
          :score-reduction {:kind :subgroup :width subgroup-size
                            :axis (get-in score [:axis :name])}
          :value-mapping {:kind :lane-strided
                          :components components
                          :components-per-lane
                          (quot (+ components (dec subgroup-size)) subgroup-size)}
          :state {:kind :online-normalized-weighted-sum
                  :components [:maximum :denominator :weighted-values]}
          :numerical-mode {:score-accumulate accumulator-dtype
                           :state-accumulate accumulator-dtype
                           :dot-order :subgroup-tree
                           :online-rescale? true}
          :attributes {:storage-kind (:kind storage)
                       :route-kind (:route-kind storage)
                       :visibility-kind (:visibility-kind membership)}})}))))

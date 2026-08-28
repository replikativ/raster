(ns raster.compiler.passes.parallel.segmented-weighted-reduction-schedule
  "Apply cooperative hardware schedules to schedule-neutral segmented weighted reductions."
  (:require [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.segmented-weighted-reduction-schedule :as schedule]))

(defn- decline
  [reason & [data]]
  (merge {:ok false :reason reason} data))

(defn- membership-capacity
  [{:keys [membership storage source-operation]}]
  (if (= :csr (:visibility-kind membership))
    (get-in source-operation [:visibility :key-index-capacity])
    (let [page-size (:page-size storage)]
      (* page-size
         (case (:route-kind storage)
           :dense-paged (get-in storage [:route-shape :pages-per-sequence])
           :csr-paged (get-in storage [:route-shape :page-index-capacity]))))))

(defn- schedule-value
  [plan subgroup-size membership-tiling strategy]
  (let [{:keys [membership storage score value accumulator-dtype]} plan
        components (:components value)
        pipelined? (= :subgroup-online-pipelined-history strategy)
        key-elements (get-in score [:axis :extent])
        shared-memory-bytes (when pipelined?
                              (* 2 (+ key-elements components) 2))]
    (schedule/make
     {:strategy strategy
      :workgroup-size (long subgroup-size)
      :segment-mapping (if (= :subgroup-online-tiled-history strategy)
                         {:partial :one-workgroup-per-segment-tile
                          :merge :one-workgroup-per-segment}
                         :one-workgroup-per-segment)
      :membership-traversal (case (:visibility-kind membership)
                              :interval :contiguous-interval
                              :csr :csr-row)
      :score-reduction {:kind :subgroup :width (long subgroup-size)
                        :axis (get-in score [:axis :name])}
      :membership-tiling membership-tiling
      :value-mapping {:kind :lane-strided
                      :components components
                      :components-per-lane
                      (quot (+ components (dec (long subgroup-size)))
                            (long subgroup-size))}
      :state (schedule/online-state)
      :staging (if pipelined?
                 {:kind :double-buffered-membership-rows
                  :stages 2
                  :members-per-iteration 2
                  :element-dtype :half
                  :key-elements key-elements
                  :value-elements components
                  :transfer-bytes 16
                  :overlap :preferred
                  ;; The finite layout member is explicit even before autotuning selects a
                  ;; non-identity member. Current row stages are 1-D, so identity is the only
                  ;; meaningful/copy-contiguous choice; swizzle search remains a later axis.
                  :layout-swizzle :identity
                  :tail-policy :separate-epilogue
                  :shared-memory-bytes shared-memory-bytes}
                 {:kind :none})
      :numerical-mode {:score-accumulate accumulator-dtype
                       :state-accumulate accumulator-dtype
                       ;; The OpenCL subgroup builtin fixes neither its association tree nor
                       ;; bitwise result. A target with an explicit shuffle tree may choose a
                       ;; stricter schedule later; this row records exactly what it emits.
                       :dot-order :implementation-defined
                       ;; Tiling is a legal floating reduction reassociation, but not bitwise the
                       ;; same association as the sequential online update. Record the exact order
                       ;; so differential tests and future strict modes can distinguish them.
                       :online-state-order
                       (if (= :subgroup-online-tiled-history strategy)
                         :increasing-members-within-tile-then-increasing-tile-left-fold
                         :increasing-membership)
                       :online-rescale? true}
      :attributes {:storage-kind (:kind storage)
                   :route-kind (:route-kind storage)
                   :visibility-kind (:visibility-kind membership)}})))

(defn- plan-subgroup-online*
  "Plan one subgroup per segment with one shared score and lane-strided value accumulators.

   The first executable storage rows are dense or CSR paged FP16 KV with either contiguous
   interval or explicitly indexed CSR membership. Those are legality constraints of this
   schedule, not new semantic operation kinds."
  [plan desc strategy tile-size]
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

      (and (= :subgroup-online-pipelined-history strategy)
           (not= :dense-paged (:route-kind storage)))
      (decline :pipelined-history-requires-dense-paged-route
               {:route-kind (:route-kind storage)})

      (and (= :subgroup-online-pipelined-history strategy)
           (not= :interval (:visibility-kind membership)))
      (decline :pipelined-history-requires-interval-membership
               {:visibility-kind (:visibility-kind membership)})

      (and (= :subgroup-online-pipelined-history strategy)
           (or (not (zero? (mod (* 2 (get-in score [:axis :extent])) 16)))
               (not (zero? (mod (* 2 components) 16)))))
      (decline :pipelined-history-row-transfer-width-unsupported
               {:transfer-bytes 16
                :key-elements (get-in score [:axis :extent])
                :value-elements components})

      (and (= :subgroup-online-pipelined-history strategy)
           (> (* 2 (+ (get-in score [:axis :extent]) components) 2)
              (long (or (get-in desc [:execution :scratchpad-bytes])
                        (get-in desc [:cache :slm])
                        (:shared-local-memory desc)
                        65536))))
      (decline :pipelined-history-shared-memory-exceeded
               {:required (* 2 (+ (get-in score [:axis :extent]) components) 2)
                :available (long (or (get-in desc [:execution :scratchpad-bytes])
                                     (get-in desc [:cache :slm])
                                     (:shared-local-memory desc)
                                     65536))})

      (and (= :subgroup-online-tiled-history strategy)
           (not (and (pos-int? tile-size) (<= tile-size Integer/MAX_VALUE))))
      (decline :score-reuse-invalid-history-tile-size {:tile-size tile-size})

      :else
      (let [capacity (membership-capacity plan)
            membership-tiling
            (if (= :subgroup-online-tiled-history strategy)
              {:kind :static-contiguous-tiles
               :tile-size tile-size
               :tile-count (quot (+ capacity (dec tile-size)) tile-size)
               :membership-capacity capacity
               :merge-order :increasing-membership-tile}
              {:kind :sequential})]
        {:ok true
         :schedule (schedule-value plan subgroup-size membership-tiling strategy)}))))

(defn plan-subgroup-online
  "Plan one subgroup per segment with one shared score and lane-strided value accumulators."
  ([plan] (plan-subgroup-online plan nil))
  ([plan desc]
   (plan-subgroup-online* plan desc :subgroup-online-score-reuse nil)))

(defn plan-subgroup-online-tiled
  "Plan independent, statically bounded history tiles followed by an ordered online-state merge.

  The capacity is a compile-time storage/membership bound. Runtime visibility still clips every
  tile, while the tile count makes graph-owned partial storage and launch geometry explicit."
  ([plan] (plan-subgroup-online-tiled plan nil))
  ([plan desc]
   (plan-subgroup-online*
    plan desc :subgroup-online-tiled-history
    (or (:segmented-weighted-reduction-history-tile-size desc) 256))))

(defn plan-subgroup-online-pipelined
  "Plan two rotating workgroup K/V rows while preserving increasing membership order.

  The first lowering row deliberately requires dense physical pages and interval membership. The
  schedule remains a segmented weighted reduction; routed storage is merely the proven way this
  implementation obtains two contiguous rows for cooperative staging."
  ([plan] (plan-subgroup-online-pipelined plan nil))
  ([plan desc]
   (plan-subgroup-online*
    plan desc :subgroup-online-pipelined-history nil)))

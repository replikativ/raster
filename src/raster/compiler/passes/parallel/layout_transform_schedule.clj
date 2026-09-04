(ns raster.compiler.passes.parallel.layout-transform-schedule
  "Target-neutral schedules for affine, elementwise layout transformations.

   These bodies are deliberately independent of GEMM. A mixed-precision contraction may use
   them to materialize a representation, while fusion and buffer planning may later eliminate
   the materialization or reuse it. Target emitters only spell the already scheduled body."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(def ^:private workgroup-size 256)

(defn- global-index
  [id]
  [(body/->IndexBinding :layout-group :group 0)
   (body/->IndexBinding :layout-lane :local 0)
   (body/->IndexCompute
    id
    (body/index-cast
     (body/expression :add
                      (body/expression :mul :layout-group workgroup-size)
                      :layout-lane)
     :long :exact))])

(defn cast-body
  "Schedule a dense one-dimensional representation conversion.

   `vector-width` is expressed as statically unrolled lane-owned elements, not a target vector
   builtin. This preserves the same coalesced access geometry on OpenCL, CUDA and HIP while
   leaving physical vector formation to target lowering. Narrowing behavior is an explicit part
   of the body and can therefore participate in dispatch equivalence and certification."
  [{:keys [id input output extent-id source-dtype destination-dtype vector-width rounding overflow]
    :or {extent-id :layout-elements vector-width 1}}]
  (let [source-dtype (dtype/canon source-dtype)
        destination-dtype (dtype/canon destination-dtype)]
    (when-not (and rounding overflow)
      (throw (ex-info "layout conversion requires explicit rounding and overflow policies"
                      {:reason :layout-transform-numerical-policy
                       :rounding rounding :overflow overflow})))
    (when-not (contains? #{1 2 4} vector-width)
      (throw (ex-info "layout conversion vector width must be 1, 2, or 4"
                      {:reason :layout-transform-vector-width
                       :vector-width vector-width})))
    (let [work-item :layout-work-item
          base :layout-base
          extent extent-id
          operations
          (vec
           (mapcat
            (fn [offset]
              (let [coordinate (body/expression :add base
                                                (body/index-cast offset :long :exact))
                    active (keyword (str "layout-active-" offset))
                    loaded (keyword (str "layout-loaded-" offset))
                    converted (keyword (str "layout-converted-" offset))]
                [(body/->ScalarLoad
                  (body/value loaded source-dtype) input [coordinate] active
                  (body/literal 0 source-dtype) :streaming)
                 (body/->ScalarCompute
                  (body/value converted destination-dtype)
                  (body/cast-expression loaded destination-dtype rounding overflow))
                 (body/->ScalarStore output [coordinate] converted active)]))
            (range vector-width)))
          masks
          (mapv (fn [offset]
                  (body/->Mask
                   (keyword (str "layout-active-" offset))
                   [(body/predicate :lt (body/expression
                                         :add base (body/index-cast offset :long :exact))
                                    (body/index-cast extent :long :exact))]))
                (range vector-width))]
      (body/make
       {:id id
        :parameters [(body/->KernelParameter
                      input :input source-dtype [extent] :global
                      (layout/row-major [extent] source-dtype) :source)
                     (body/->KernelParameter
                      output :output destination-dtype [extent] :global
                      (layout/row-major [extent] destination-dtype) :destination)
                     (body/->KernelParameter extent :scalar :int [] nil nil :extent)]
        :stable-reads [(body/stable-read input)]
        :indices (conj (global-index work-item)
                       (body/->IndexCompute
                        base (body/expression
                              :mul work-item
                              (body/index-cast vector-width :long :exact))))
        :masks masks
        :operations operations
        :schedule {:strategy :affine-elementwise-cast
                   :elements-per-work-item vector-width
                   :rounding rounding :overflow overflow}
        :launch (launch/spec
                 {:workgroup-size [workgroup-size]
                  :group-count [(launch/ceil-div
                                 (launch/ceil-div (launch/runtime-value extent) vector-width)
                                 workgroup-size)]})
        :provenance {:dialect :kernel-body :source-dialect :layout-transform}
        :attributes {:kind :layout-transform :operation :cast
                     :source-dtype source-dtype :destination-dtype destination-dtype}}))))

(defn transpose-body
  "Schedule a dense row-major [rows, cols] -> [cols, rows] transpose."
  [{:keys [id input output row-extent-id column-extent-id element-dtype]
    :or {row-extent-id :layout-rows column-extent-id :layout-cols}}]
  (let [element-dtype (dtype/canon element-dtype)
        linear :layout-linear
        row :layout-row
        column :layout-column
        rows-id row-extent-id
        cols-id column-extent-id
        elements (body/expression :mul
                                  (body/index-cast rows-id :long :exact)
                                  (body/index-cast cols-id :long :exact))]
    (body/make
     {:id id
      :parameters [(body/->KernelParameter
                    input :input element-dtype [rows-id cols-id] :global
                    (layout/row-major [rows-id cols-id] element-dtype) :source)
                   (body/->KernelParameter
                    output :output element-dtype [cols-id rows-id] :global
                    (layout/row-major [cols-id rows-id] element-dtype) :destination)
                   (body/->KernelParameter rows-id :scalar :int [] nil nil :extent)
                   (body/->KernelParameter cols-id :scalar :int [] nil nil :extent)]
      :stable-reads [(body/stable-read input)]
      :indices (into (global-index linear)
                     [(body/->IndexCompute
                       row (body/expression :floor-div linear
                                            (body/index-cast cols-id :long :exact)))
                      (body/->IndexCompute
                       column (body/expression :mod linear
                                               (body/index-cast cols-id :long :exact)))])
      :masks [(body/->Mask :layout-active [(body/predicate :lt linear elements)])]
      :operations [(body/->ScalarLoad
                    (body/value :layout-value element-dtype)
                    input [row column] :layout-active
                    (body/literal 0 element-dtype) :streaming)
                   (body/->ScalarStore
                    output [column row] :layout-value :layout-active)]
      :schedule {:strategy :affine-layout-transform :permutation [1 0]}
      :launch (launch/spec
               {:workgroup-size [workgroup-size]
                :group-count [(launch/ceil-div
                               (launch/product (launch/runtime-value rows-id)
                                               (launch/runtime-value cols-id))
                               workgroup-size)]})
      :provenance {:dialect :kernel-body :source-dialect :layout-transform}
      :attributes {:kind :layout-transform :operation :transpose
                   :permutation [1 0] :element-dtype element-dtype}})))

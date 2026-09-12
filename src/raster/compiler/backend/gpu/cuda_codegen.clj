(ns raster.compiler.backend.gpu.cuda-codegen
  "CUDA-C target lowering for verified matrix KernelBody values.

  The first row deliberately covers direct, aligned f16×f16→f32 WMMA bodies. Unsupported body
   structure fails before source emission. Coordinate-free FP32 store regions use the shared scalar
   emitter over fragment elements; indexed epilogues, masks, views and slices are never silently
   discarded. Shared-memory staging and WGMMA are later schedules over the same
  typed contraction and KernelBody vocabulary."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as scalar-emitter]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.matrix-target-names :as target-names]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]
            [raster.compiler.backend.gpu.matrix-fragment-source :as fragment-source]))

(defn- decline!
  [condition reason data]
  (when-not condition
    (throw (ex-info "CUDA matrix lowering does not implement this verified body"
                    (assoc data :reason reason :target :cuda)))))

(defn- parameter-declarations
  [parameters dimension-parameters]
  (let [dimension-name (into {} (map (fn [[axis id]] [id (str/upper-case (name axis))]))
                             dimension-parameters)]
    (mapv
     (fn [{:keys [id kind dtype role] :as parameter}]
       (cond
         (and (= :lhs role) (= :input kind) (= :half dtype))
         "const half* __restrict__ A"

         (and (= :rhs role) (= :input kind) (= :half dtype))
         "const half* __restrict__ B"

         (and (= :result role) (= :output kind) (= :float dtype))
         "float* __restrict__ C"

         (and (= :dimension role) (= :scalar kind) (= :int dtype)
              (contains? dimension-name id))
         (str "int " (get dimension-name id))

         (and (= :epilogue role) (= :scalar kind) (= :float dtype))
         (str "float " (c-emit/c-symbol id))

         :else
         (throw (ex-info "CUDA matrix lowering cannot render this ordered ABI parameter"
                         {:reason :cuda-mma-extra-abi-unsupported
                          :target :cuda :parameter parameter}))))
     parameters)))

(defn- emit-plan
  [kernel-name {:keys [instruction dimensions dimension-values parameters
                       block-m block-n block-k result-dtype
                       dimension-parameters schedule-parameters group-z k-lower k-upper
                       buffer-offsets] :as plan} epilogue]
  (let [[M N K] dimensions]
    (decline! (= {:family :mma :m 16 :n 16 :k 16 :subgroup 32} instruction)
              :cuda-mma-instruction-unsupported {:instruction instruction})
    (decline! (= :float result-dtype)
              :cuda-mma-result-dtype-unsupported {:result-dtype result-dtype})
    (decline! (and (every? #(and (integer? %) (pos? %)) dimensions)
                   (zero? (mod M block-m))
                   (zero? (mod N block-n))
                   (zero? (mod K block-k)))
              :cuda-mma-requires-aligned-static-dimensions
              {:dimensions dimensions :block [block-m block-n block-k]})
    (decline! (= [0 (:k dimension-parameters)] [k-lower k-upper])
              :cuda-mma-k-slice-unsupported {:k-range [k-lower k-upper]})
    (decline! (and (nil? group-z) (empty? schedule-parameters)
                   (every? nil? (vals buffer-offsets)))
              :cuda-mma-views-or-schedule-parameters-unsupported
              {:group-z group-z :schedule-parameters schedule-parameters
               :buffer-offsets buffer-offsets})
    (let [declarations (parameter-declarations parameters dimension-parameters)
          _ (decline! (= 6 (count (remove #(= :epilogue (:role %)) parameters)))
                      :cuda-mma-extra-abi-unsupported {:parameters parameters})
          specialized-dimensions
          (mapv dimension-values [(:m dimension-parameters)
                                  (:n dimension-parameters)
                                  (:k dimension-parameters)])
          _ (decline! (= dimensions specialized-dimensions)
                      :cuda-mma-dimension-specialization-invalid
                      {:dimensions dimensions :dimension-values dimension-values})]
      (fragment-source/emit-direct kernel-name plan declarations epilogue :cuda))))

(defn emit-matrix-kernel
  "Lower the currently supported CUDA WMMA subset of a verified matrix KernelBody.

  This boundary takes no tile, dimension, launch or ABI side channel. The returned source is a
  target spelling of the analyzed body; unsupported verified bodies fail with a structured reason."
  [kernel-name kernel-body]
  (let [plan (matrix-plan/analyze kernel-body)
        names (target-names/validate! kernel-name kernel-body
                                      (target-names/parameter-names kernel-body nil))
        region (scalar-emitter/lower-uniform-store-region kernel-body names :cuda)
        source (emit-plan kernel-name plan (:epilogue region))
        helpers (c-emit/intrinsic-helper-module source :cuda {})]
    (decline! (empty? (:compilation helpers)) :cuda-mma-helper-compilation-unsupported
              {:compilation (:compilation helpers)})
    (str "#include <mma.h>\n#include <math.h>\n"
         (c-dialect/helper-source (c-dialect/resolve! :cuda) (:source helpers))
         "\n" source)))

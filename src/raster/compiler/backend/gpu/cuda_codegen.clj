(ns raster.compiler.backend.gpu.cuda-codegen
  "CUDA-C target lowering for verified matrix KernelBody values.

  The first row deliberately covers direct, aligned f16×f16→f32 WMMA bodies. Unsupported body
  structure fails before source emission; masks, views, slices, epilogues and extra ABI values are
  never silently discarded. Shared-memory staging and WGMMA are later schedules over the same
  typed contraction and KernelBody vocabulary."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]))

(defn- decline!
  [condition reason data]
  (when-not condition
    (throw (ex-info "CUDA matrix lowering does not implement this verified body"
                    (assoc data :reason reason :target :cuda)))))

(defn- identity-store?
  [store]
  (nil? (:value-region store)))

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

         :else
         (throw (ex-info "CUDA matrix lowering cannot render this ordered ABI parameter"
                         {:reason :cuda-mma-extra-abi-unsupported
                          :target :cuda :parameter parameter}))))
     parameters)))

(defn- emit-plan
  [kernel-name {:keys [instruction dimensions dimension-values parameters
                       mi ni ki subgroup block-m block-n sg-m sg-n
                       block-k lhs-ids rhs-ids stores prefetch result-dtype
                       dimension-parameters schedule-parameters group-z k-lower k-upper
                       buffer-offsets]}]
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
    (decline! (every? identity-store? stores)
              :cuda-mma-store-region-unsupported {:stores stores})
    (let [declarations (parameter-declarations parameters dimension-parameters)
          _ (decline! (= 6 (count declarations))
                      :cuda-mma-extra-abi-unsupported {:parameters parameters})
          specialized-dimensions
          (mapv dimension-values [(:m dimension-parameters)
                                  (:n dimension-parameters)
                                  (:k dimension-parameters)])
          _ (decline! (= dimensions specialized-dimensions)
                      :cuda-mma-dimension-specialization-invalid
                      {:dimensions dimensions :dimension-values dimension-values})
          nms (count lhs-ids) nns (count rhs-ids)      ;; fragments per warp (M, N)
          ncols (quot block-n sg-n)                   ;; warp columns
          ksteps (quot block-k ki)
          ms (range nms) ns (range nns)
          warps-per-block (* (quot block-m sg-m) (quot block-n sg-n))
          frag (fn [role & [layout]] (str "wmma::fragment<wmma::" role ", " mi ", " ni ", " ki ", "
                                          (if (= role "accumulator") "float" (str "half, wmma::" layout))
                                          ">"))]
      (str
       "#include <mma.h>\n"
       "using namespace nvcuda;\n\n"
       "// Tiled WMMA GEMM (parametric): block " block-m "x" block-n ", warp-tile " sg-m "x" sg-n
       ", K " block-k ", frag " mi "x" ni "x" ki ", warp " subgroup
       ", warps/block " warps-per-block "\n"
       "extern \"C\" __global__ void " kernel-name "(\n    "
       (str/join ",\n    " declarations) ") {\n"
       "  if (M != " M " || N != " N " || K != " K ") { asm volatile(\"trap;\"); return; }\n"
       "  int warpId = threadIdx.x / " subgroup ";\n"
       "  int warp_row = warpId / " ncols ";\n"
       "  int warp_col = warpId % " ncols ";\n"
       "  int m_base = blockIdx.y * " block-m " + warp_row * " sg-m ";\n"
       "  int n_base = blockIdx.x * " block-n " + warp_col * " sg-n ";\n"
       ;; accumulator fragments
       (apply str (for [m ms n ns] (str "  " (frag "accumulator") " acc" m "_" n ";\n")))
       (apply str (for [m ms n ns] (str "  wmma::fill_fragment(acc" m "_" n ", 0.0f);\n")))
       "  " (frag "matrix_a" "row_major") " " (str/join ", " (for [m ms] (str "a" m))) ";\n"
       "  " (frag "matrix_b" "row_major") " " (str/join ", " (for [n ns] (str "b" n))) ";\n"
       "  for (int k = 0; k < K; k += " block-k ") {\n"
       (apply str
              (for [ks (range ksteps)]
                (let [koff (* ks ki)]
                  (str
                   "    { int pk = k + " (+ koff (* prefetch ki)) ";\n"
                   "      if (pk < K && ((int)threadIdx.x % " subgroup ") == 0) {\n"
                   (apply str
                          (for [m ms]
                            (str "        #pragma unroll\n"
                                 "        for (int pr = 0; pr < " mi "; ++pr) {\n"
                                 "          const half* pp = A + (m_base + " (* m mi)
                                 " + pr) * K + pk;\n"
                                 "          asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pp));\n"
                                 "        }\n")))
                   "      } }\n"
                   (apply str (for [m ms] (str "    wmma::load_matrix_sync(a" m ", A + (m_base + " (* m mi) ") * K + k + " koff ", K);\n")))
                   (apply str (for [n ns] (str "    wmma::load_matrix_sync(b" n ", B + (k + " koff ") * N + n_base + " (* n ni) ", N);\n")))
                   (apply str (for [m ms n ns] (str "    wmma::mma_sync(acc" m "_" n ", a" m ", b" n ", acc" m "_" n ");\n")))))))
       "  }\n"
       (apply str (for [m ms n ns]
                    (str "  wmma::store_matrix_sync(C + (m_base + " (* m mi) ") * N + n_base + " (* n ni)
                         ", acc" m "_" n ", N, wmma::mem_row_major);\n")))
       "}\n"))))

(defn emit-matrix-kernel
  "Lower the currently supported CUDA WMMA subset of a verified matrix KernelBody.

  This boundary takes no tile, dimension, launch or ABI side channel. The returned source is a
  target spelling of the analyzed body; unsupported verified bodies fail with a structured reason."
  [kernel-name kernel-body]
  (emit-plan kernel-name (matrix-plan/analyze kernel-body)))

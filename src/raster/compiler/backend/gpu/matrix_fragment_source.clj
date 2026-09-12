(ns raster.compiler.backend.gpu.matrix-fragment-source
  "Shared direct-fragment mainloop spelling for analyzed matrix bodies.
   Target admission stays in the target emitter; this internal renderer introduces no schedules,
   layouts, storage, or precision decisions. The dialect selects instruction spellings only."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-opencl :as scalar-emitter]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.compiler.backend.gpu.matrix-target-names :as target-names]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]))

(defn- fragment-dialect! [target]
  (case target
    :cuda {:diagnostic-prefix "cuda-mma-"
           :instruction {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
           :header "#include <mma.h>\n#include <math.h>\n"
           :namespace "wmma" :operand-type "half"
           :namespace-declaration "using namespace nvcuda;\n\n"
           :dimension-failure "asm volatile(\"trap;\");"
           :prefetch "asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pp));"}
    :hip {:diagnostic-prefix "hip-mfma-"
          :instruction {:family :mfma :m 16 :n 16 :k 16 :subgroup 64}
          :header (str "#include <hip/hip_runtime.h>\n#include <rocwmma/rocwmma.hpp>\n#include <math.h>\n"
                       "#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)\n"
                       "#error Raster MFMA candidate requires gfx90a (wave64)\n#endif\n")
          :namespace "rocwmma" :operand-type "rocwmma::float16_t"
          :namespace-declaration ""
          :dimension-failure "__builtin_trap();"
          :prefetch "__builtin_prefetch(pp, 0, 3);"}
    (throw (ex-info "matrix fragment dialect is not implemented"
                    {:reason :matrix-fragment-dialect-not-lowered :target target}))))

(defn emit-direct
  "Render an admitted matrix plan. Declarations preserve the caller's checked ordered ABI;
   epilogue is the existing typed scalar-region lowering, never a user kernel template."
  [kernel-name {:keys [dimensions mi ni ki subgroup block-m block-n sg-m sg-n
                       block-k lhs-ids rhs-ids prefetch index-dtype]} declarations epilogue target]
  (let [dialect (fragment-dialect! target)
        fragment-namespace (:namespace dialect)
        operand-type (:operand-type dialect)
        [M N K] dimensions
        index-type (c-dialect/type-name (c-dialect/resolve! target) index-dtype)
        nms (count lhs-ids) nns (count rhs-ids)      ;; fragments per warp (M, N)
          ncols (quot block-n sg-n)                   ;; warp columns
          ksteps (quot block-k ki)
          ms (range nms) ns (range nns)
          warps-per-block (* (quot block-m sg-m) (quot block-n sg-n))
          frag (fn [role & [layout]] (str fragment-namespace "::fragment<" fragment-namespace "::" role ", " mi ", " ni ", " ki ", "
                                          (if (= role "accumulator") "float" (str operand-type ", " fragment-namespace "::" layout))
                                          ">"))]
      (str
       (:namespace-declaration dialect)
       "// Tiled WMMA GEMM (parametric): block " block-m "x" block-n ", warp-tile " sg-m "x" sg-n
       ", K " block-k ", frag " mi "x" ni "x" ki ", warp " subgroup
       ", warps/block " warps-per-block "\n"
       "extern \"C\" __global__ void " kernel-name "(\n    "
       (str/join ",\n    " declarations) ") {\n"
       "  if (M != " M " || N != " N " || K != " K ") { " (:dimension-failure dialect) " return; }\n"
       "  int warpId = threadIdx.x / " subgroup ";\n"
       "  int warp_row = warpId / " ncols ";\n"
       "  int warp_col = warpId % " ncols ";\n"
       "  int m_base = blockIdx.y * " block-m " + warp_row * " sg-m ";\n"
       "  int n_base = blockIdx.x * " block-n " + warp_col * " sg-n ";\n"
       ;; accumulator fragments
       (apply str (for [m ms n ns] (str "  " (frag "accumulator") " acc" m "_" n ";\n")))
       (apply str (for [m ms n ns] (str "  " fragment-namespace "::fill_fragment(acc" m "_" n ", 0.0f);\n")))
       "  " (frag "matrix_a" "row_major") " " (str/join ", " (for [m ms] (str "a" m))) ";\n"
       "  " (frag "matrix_b" "row_major") " " (str/join ", " (for [n ns] (str "b" n))) ";\n"
       "  for (" index-type " k = 0; k < K; k += " block-k ") {\n"
       (apply str
              (for [ks (range ksteps)]
                (let [koff (* ks ki)]
                  (str
                   "    { " index-type " pk = k + " (+ koff (* prefetch ki)) ";\n"
                   "      if (pk < K && ((int)threadIdx.x % " subgroup ") == 0) {\n"
                   (apply str
                          (for [m ms]
                            (str "        #pragma unroll\n"
                                 "        for (int pr = 0; pr < " mi "; ++pr) {\n"
                                 "          const " operand-type "* pp = A + (m_base + " (* m mi)
                                 " + pr) * K + pk;\n"
                                 "          " (:prefetch dialect) "\n"
                                 "        }\n")))
                   "      } }\n"
                   (apply str (for [m ms] (str "    " fragment-namespace "::load_matrix_sync(a" m ", A + (m_base + " (* m mi) ") * K + k + " koff ", K);\n")))
                   (apply str (for [n ns] (str "    " fragment-namespace "::load_matrix_sync(b" n ", B + (k + " koff ") * N + n_base + " (* n ni) ", N);\n")))
                   (apply str (for [m ms n ns] (str "    " fragment-namespace "::mma_sync(acc" m "_" n ", a" m ", b" n ", acc" m "_" n ");\n")))))))
       "  }\n"
       (apply str (for [m ms n ns]
                    (str (when epilogue
                           (let [acc (str "acc" m "_" n)
                                 value (str acc ".x[rstr_epilogue_element]")]
                             (str "  #pragma unroll\n"
                                  "  for (int rstr_epilogue_element = 0; rstr_epilogue_element < " acc ".num_elements; ++rstr_epilogue_element) {\n"
                                  "    " value " = " (epilogue value "" "") ";\n  }\n")))
                         "  " fragment-namespace "::store_matrix_sync(C + (m_base + " (* m mi) ") * N + n_base + " (* n ni)
                         ", acc" m "_" n ", N, " fragment-namespace "::mem_row_major);\n")))
       "}\n")))

(defn- decline!
  [target condition reason data]
  (when-not condition
    (throw (ex-info "matrix fragment lowering does not implement this verified body"
                    (assoc data :reason (keyword (str (:diagnostic-prefix (fragment-dialect! target))
                                                     (name reason)))
                           :target target)))))

(defn- parameter-declarations
  [parameters dimension-parameters target]
  (let [dimension-name (into {} (map (fn [[axis id]] [id (str/upper-case (name axis))]))
                             dimension-parameters)]
    (mapv
     (fn [{:keys [id kind dtype role] :as parameter}]
       (cond
         (and (= :lhs role) (= :input kind) (= :half dtype))
         (str "const " (:operand-type (fragment-dialect! target)) "* __restrict__ A")

         (and (= :rhs role) (= :input kind) (= :half dtype))
         (str "const " (:operand-type (fragment-dialect! target)) "* __restrict__ B")

         (and (= :result role) (= :output kind) (= :float dtype))
         "float* __restrict__ C"

         (and (= :dimension role) (= :scalar kind) (= :int dtype)
              (contains? dimension-name id))
         (str "int " (get dimension-name id))

         (and (= :epilogue role) (= :scalar kind) (= :float dtype))
         (str "float " (c-emit/c-symbol id))

         :else
         (decline! target false :extra-abi-unsupported {:parameter parameter})))
     parameters)))

(defn- emit-admitted-plan
  [kernel-name {:keys [instruction dimensions dimension-values parameters
                       block-m block-n block-k result-dtype
                       dimension-parameters schedule-parameters group-z k-lower k-upper
                       buffer-offsets] :as plan} epilogue target]
  (let [[M N K] dimensions]
    (decline! target (= (:instruction (fragment-dialect! target)) instruction)
              :instruction-unsupported {:instruction instruction})
    (decline! target (= :float result-dtype)
              :result-dtype-unsupported {:result-dtype result-dtype})
    (decline! target (and (every? #(and (integer? %) (pos? %)) dimensions)
                   (zero? (mod M block-m))
                   (zero? (mod N block-n))
                   (zero? (mod K block-k)))
              :requires-aligned-static-dimensions
              {:dimensions dimensions :block [block-m block-n block-k]})
    (decline! target (= [0 (:k dimension-parameters)] [k-lower k-upper])
              :k-slice-unsupported {:k-range [k-lower k-upper]})
    (decline! target (and (nil? group-z) (empty? schedule-parameters)
                   (every? nil? (vals buffer-offsets)))
              :views-or-schedule-parameters-unsupported
              {:group-z group-z :schedule-parameters schedule-parameters
               :buffer-offsets buffer-offsets})
    (let [declarations (parameter-declarations parameters dimension-parameters target)
          _ (decline! target (= 6 (count (remove #(= :epilogue (:role %)) parameters)))
                      :extra-abi-unsupported {:parameters parameters})
          specialized-dimensions
          (mapv dimension-values [(:m dimension-parameters)
                                  (:n dimension-parameters)
                                  (:k dimension-parameters)])
          _ (decline! target (= dimensions specialized-dimensions)
                      :dimension-specialization-invalid
                      {:dimensions dimensions :dimension-values dimension-values})]
      (emit-direct kernel-name plan declarations epilogue target))))

(defn emit-matrix-kernel
  "Emit the direct fragment subset for a verified body.
   HIP is a source-only candidate requiring the separately pinned rocWMMA compile environment;
   it is not admitted by matrix-target or advertised as a runtime-supported artifact."
  [kernel-name kernel-body target]
  (let [dialect (fragment-dialect! target)
        plan (matrix-plan/analyze kernel-body)
        names (target-names/validate! kernel-name kernel-body
                                      (target-names/parameter-names kernel-body nil))
        region (scalar-emitter/lower-uniform-store-region kernel-body names target)
        source (emit-admitted-plan kernel-name plan (:epilogue region) target)
        helpers (c-emit/intrinsic-helper-module source target {})]
    (decline! target (empty? (:compilation helpers)) :helper-compilation-unsupported
              {:compilation (:compilation helpers)})
    (str (:header dialect)
         (c-dialect/helper-source (c-dialect/resolve! target) (:source helpers))
         "\n" source)))

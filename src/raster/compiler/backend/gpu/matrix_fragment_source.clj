(ns raster.compiler.backend.gpu.matrix-fragment-source
  "Shared direct-fragment mainloop spelling for analyzed matrix bodies.
   Target admission stays in the target emitter; this internal renderer introduces no schedules,
   layouts, storage, or precision decisions. The dialect selects instruction spellings only."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]))

(defn- fragment-dialect! [target]
  (case target
    :cuda {:namespace "wmma" :operand-type "half"
           :namespace-declaration "using namespace nvcuda;\n\n"
           :dimension-failure "asm volatile(\"trap;\");"
           :prefetch "asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(pp));"}
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

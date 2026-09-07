(ns raster.compiler.reference.gemm-opencl
  "Historical GEMM source oracles for differential tests and explicitly labeled benchmarks.
   Not a production scheduling or emission API."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.ir.kernel-abi :as kabi]))

(defn emit-gemm-tiled
  "Independent tile-parametric XMX GEMM source oracle.

   Ordinary direct/tiled, split-K, and batched GEMM lower from a verified KernelBody. This
   independent generator stays executable for source-equivalence checks. Production compiler and
   resident routes lower verified KernelBody values instead.

   Computes C = A·B with FP16 inputs and FP32 accumulation. The tile geometry
   (workgroup BLOCK_M×BLOCK_N, per-subgroup SG_M×SG_N, K-step BLOCK_K, prefetch depth) is COMPUTED,
   not hardcoded; the DPAS instruction shape (M_i×N_i×K_i) + subgroup size come from the :matrix
   descriptor. The Arc default (128×128, 32×32, K 32, DPAS 8×16×16, subgroup 16) reproduces the
   original hand-unrolled kernel bit-for-bit (validated over plain/split-k/batched paths); varying
   the tile is bit-invariant since each output element's K-reduction is the same DPAS sequence. The
   Intel DPAS builtins are fixed here; a different vendor's matrix instruction (WGMMA/MFMA) is a
   separate emitter keyed by :matrix :family (the one genuine fork). Handles arbitrary M,N,K with
   full boundary checking; supports alpha/beta, c-dtype, split-k? (grid-z partials) and batched?
   (grid-z slabs)."
  [kernel-name & {:keys [block-m block-n sg-m sg-n block-k matrix alpha beta c-dtype split-k? schedule-splits-arg? batched? prefetch
                         epilogue epilogue-params]
                  :or {block-m 128 block-n 128 sg-m 32 sg-n 32 block-k 32
                       matrix {:m 8 :n 16 :k 16 :subgroup 16}
                       alpha 1.0 beta 0.0 c-dtype :half split-k? false batched? false prefetch 3}}]
  ;; EPILOGUE (feature 4 store-splice): `epilogue` is (fn [acc-c row-c col-c] -> C-expr) that
  ;; transforms the accumulator value into the stored value — a bias/activation/residual/dequant
  ;; folded into the store slot (which already has row/col in scope), eliminating a separate
  ;; elementwise kernel + a DRAM round-trip of C. `epilogue-params` is a C param-decl string for the
  ;; epilogue's operand arrays (e.g. ", __global const float* restrict bias"). When `epilogue` is nil
  ;; the emitted string is byte-identical to the plain GEMM. Mutually exclusive with beta≠0.
  (let [{mi :m ni :n ki :k sg :subgroup} matrix]
    (doseq [[nm a b] [["sg-m/M_i" sg-m mi] ["sg-n/N_i" sg-n ni] ["block-m/sg-m" block-m sg-m]
                      ["block-n/sg-n" block-n sg-n] ["block-k/K_i" block-k ki]]]
      (when-not (zero? (rem a b))
        (throw (ex-info (str "emit-gemm-tiled: " nm " not divisible (" a "/" b ")") {:tile [block-m block-n sg-m sg-n block-k] :matrix matrix}))))
    (when (and split-k? (not= beta 0.0))
      (throw (ex-info "split-k GEMM requires beta = 0" {:beta beta})))
    (when (and epilogue (not= beta 0.0))
      (throw (ex-info "epilogue and beta≠0 are mutually exclusive (both write the store slot)" {:beta beta})))
    (when (and epilogue split-k?)
      (throw (ex-info "epilogue fuses into the final store; not valid on split-k partials" {})))
    (when (and split-k? batched?)
      (throw (ex-info "batched and split-k both claim grid-z — mutually exclusive" {})))
    (let [nms (quot sg-m mi) nns (quot sg-n ni)      ;; M/N subtiles per subgroup
          ncols (quot block-n sg-n)                   ;; subgroup columns
          ksteps (quot block-k ki)
          c-dtype (if split-k? :float c-dtype)
          c-type (if (= c-dtype :float) "float" "half")
          store-cast (if (= c-dtype :float) "" "(half)")
          kend (if split-k? "k_hi" "K")
          accT (str "float" mi)
          ms (range nms) ns (range nns)
          amul (fn [m] (if (zero? m) "m_base" (str "m_base+" (* m mi))))
          kstep (fn [kpos]
                  (str "        { int pk = " kpos " + " (* prefetch ki) ";\n"
                       "          if (pk < " kend ") {\n"
                       (apply str (for [m ms] (str "            intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(pk, " (amul m) "));\n")))
                       "          } }\n"
                       (apply str (for [n ns] (str "        bp" n " = as_int8(intel_subgroup_block_read_transform_u16_k16((__global void*)B, b_wb, K, b_pb, (int2)(n_base" n ", " kpos ")));\n")))
                       (apply str (for [m ms] (str "        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" kpos ", " (amul m) "), &a" m ");\n")))
                       (apply str (for [m ms] (str "        sa" m " = as_short8(a" m ");\n")))
                       (apply str (for [m ms n ns] (str "        acc" m n " = intel_sub_group_f16_f16_matrix_mad_k16(sa" m ", bp" n ", acc" m n ");\n")))))]
      (str
       "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
       "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n\n"
       "// Tiled GEMM (parametric): WG " block-m "x" block-n ", SG " sg-m "x" sg-n ", K " block-k
       ", DPAS " mi "x" ni "x" ki ", sg " sg "\n"
       "__attribute__((intel_reqd_sub_group_size(" sg ")))\n"
       "__kernel void " kernel-name "(\n"
       "    __global const half* restrict A,\n    __global const half* restrict B,\n"
       "    __global " c-type "* restrict C,\n    int M, int N, int K"
       (when split-k? ", int KC")
       (when (and split-k? schedule-splits-arg?) ", int splits")
       (when (not= alpha 1.0) ", float alpha") (when (not= beta 0.0) ", float beta")
       epilogue-params
       ") {\n"
       "    int sg_id = get_sub_group_id();\n    int sg_lid = get_sub_group_local_id();\n"
       "    int sg_row = sg_id / " ncols ";\n    int sg_col = sg_id % " ncols ";\n"
       "    int m_base = get_group_id(1) * " block-m " + sg_row * " sg-m ";\n"
       (apply str (for [n ns] (str "    int n_base" n " = get_group_id(0) * " block-n " + sg_col * " sg-n (when (pos? n) (str " + " (* n ni))) ";\n")))
       "    if (m_base >= M || n_base0 >= N) return;\n"
       (when batched?
         "    long slab = get_group_id(2);\n    A += slab*(long)M*(long)K;\n    B += slab*(long)K*(long)N;\n    C += slab*(long)M*(long)N;\n")
       (when split-k?
         "    int k_lo = get_group_id(2) * KC;\n    if (k_lo >= K) return;\n    int k_hi = k_lo + KC; if (k_hi > K) k_hi = K;\n    C += (long)get_group_id(2) * (long)M * (long)N;\n")
       (apply str (for [m ms] (str "    " accT " " (str/join ", " (for [n ns] (str "acc" m n "=0.0f"))) ";\n")))
       "    int a_wb = K * 2, a_pb = K * 2;\n    int b_wb = N * 2, b_pb = N * 2;\n"
       "    ushort8 " (str/join ", " (for [m ms] (str "a" m))) ";\n"
       "    short8 " (str/join ", " (for [m ms] (str "sa" m))) ";\n"
       "    int8 " (str/join ", " (for [n ns] (str "bp" n))) ";\n"
       (apply str (for [p (range prefetch)]
                    (str "    if (" (when split-k? "k_lo + ") (* p ki) " < " kend ") {\n"
                         (apply str (for [m ms] (str "        intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" (when split-k? "k_lo + ") (* p ki) ", " (amul m) "));\n")))
                         "    }\n")))
       "    int k = " (if split-k? "k_lo" "0") ";\n"
       "    for (; k + " (dec block-k) " < " kend "; k += " block-k ") {\n"
       (apply str (for [ks (range ksteps)] (kstep (if (zero? ks) "k" (str "k + " (* ks ki))))))
       "    }\n"
       "    for (; k < " kend "; k += " ki ") {\n"
       (kstep "k")
       "    }\n"
       (apply str
              (for [m ms i (range mi)]
                (str "    { int row = m_base + " (+ (* m mi) i) ";\n      if (row < M) {\n"
                     (apply str (for [n ns]
                                  (let [acc-expr (str (when (not= alpha 1.0) "alpha * ") "acc" m n ".s" i)]
                                    (str "        { int col = n_base" n " + sg_lid;\n          if (col < N) "
                                         (cond
                                           (not= beta 0.0)
                                           (str "{ float old=(float)C[row*N+col]; C[row*N+col] = " store-cast "(" acc-expr " + beta*old); }\n")
                                           epilogue
                                           (str "C[row*N+col] = " store-cast "(" (epilogue acc-expr "row" "col") ");\n")
                                           :else
                                           (str "C[row*N+col] = " store-cast "(" acc-expr ");\n"))
                                         "        }\n"))))
                     "      }\n    }\n")))
       "}\n"))))

(defn- emit-dpas-plan
  "Retained source-oracle descriptor; never used for production selection."
  [kernel-name row-arr col-arr out-sym [M N L] [i-sym j-sym] tile epilogue]
  (let [effective-tile tile
        result-dtype :half
        sg (long (get-in effective-tile [:matrix :subgroup] 16))
        ep (when epilogue
             (sco/epilogue-splice epilogue [i-sym j-sym] (get epilogue :dtype :float)))
        effective-epilogue epilogue
        source (apply emit-gemm-tiled kernel-name
                        (concat [:c-dtype :half
                                 :block-m (:block-m effective-tile)
                                 :block-n (:block-n effective-tile)
                                 :sg-m (:sg-m effective-tile) :sg-n (:sg-n effective-tile)
                                 :block-k (:block-k effective-tile) :matrix (:matrix effective-tile)
                                 :prefetch (:num-stages effective-tile 3)]
                                (when ep [:epilogue (:epilogue ep)
                                          :epilogue-params (:epilogue-params ep)])))]
    {:kernel-name kernel-name
      :source source
      :array-params [row-arr col-arr]
      :abi (kabi/validate!
                (vec (concat
                      [(kabi/slot row-arr :input :half :c-name "A" :role :operand)
                       (kabi/slot col-arr :input :half :c-name "B" :role :operand)
                       (kabi/slot out-sym :output result-dtype :c-name "C" :role :result)
                       (kabi/slot 'M :scalar :int :role :dimension)
                       (kabi/slot 'N :scalar :int :role :dimension)
                       (kabi/slot 'K :scalar :int :role :dimension)]
                      (for [{:keys [sym dtype] :or {dtype :float}} (:operands effective-epilogue)]
                        (kabi/slot sym :input dtype :c-name (ce/c-symbol sym) :role :epilogue))
                      (for [{:keys [sym dtype] :or {dtype :float}} (:scalars effective-epilogue)]
                        (kabi/slot sym :scalar dtype :c-name (ce/c-symbol sym) :role :epilogue)))))
      :dims [M N L]
      :dtype result-dtype
      :tile effective-tile
      :epilogue-params (when ep (:epilogue-params ep))
      :epilogue-operands (when ep (mapv :sym (:operands effective-epilogue)))
      :epilogue-scalars (when ep (mapv :sym (:scalars effective-epilogue)))
      :workgroup [(* (quot (:block-m effective-tile) (:sg-m effective-tile))
                       (quot (:block-n effective-tile) (:sg-n effective-tile))
                       sg) 1]
      :tensorized true}))

(defn generate-dpas-contraction-kernel
  "Legacy direct entry to the proven DPAS/XMX OpenCL emitter.

   Production canonical f16 contractions now travel through ContractionFacts and a scheduled
   KernelBody before reaching `generate-dpas-kernel-body`. This entry remains only as the
   independent source oracle for the direct operation lowerer.
   Its legality analysis determines operand orientation and launch dimensions; the emitted body is
   f16 input, f32 accumulation and f16 output with a tile-parametric K16 matrix instruction.

   Returns {:kernel-name :source :array-params [row-arr col-arr] :dims [M N L] :dtype :half
            :tensorized true}  — NB: :array-params is in [row col] BINDING order (row's
   buffer → A slot, col's → B slot, out → C), NOT sorted-by-name. Returns
   {:tensorized false :reason …} when the gate rejects (caller falls back to regtiled)."
  [segred out-sym & {:keys [dtype desc tile epilogue] :or {dtype :half}}]
  (let [gate (sco/dpas-contraction-legal? segred dtype)]
    (if-not (:ok gate)
      {:tensorized false :reason (:reason gate) :detail gate}
      (let [{:keys [M N L row-arr col-arr]} gate
            kernel-name (str "dpas_contract_" (gensym ""))
            ;; TILE GEOMETRY IS DERIVED FROM THE HARDWARE DESCRIPTOR, never hardcoded: the
            ;; per-subgroup accumulator tile is GRF-bound and rounded to the matrix (DPAS)
            ;; fragment granularity, so a part with a different GRF budget / subgroup size /
            ;; matrix shape gets a correctly rescaled tile from the same rule. An explicit
            ;; `tile` (e.g. an autotune result via hw/gemm-tile-candidates) overrides.
            ;; hw/derive-gemm-tile's own defaults reproduce the Arc 140V config, so passing no
            ;; descriptor is equivalent to the previous literal — with zero magic numbers here.
            tile (or tile (hw/derive-gemm-tile (or desc {})))]
        (emit-dpas-plan kernel-name row-arr col-arr out-sym [M N L]
                        [(:i-sym gate) (:j-sym gate)] tile epilogue)))))

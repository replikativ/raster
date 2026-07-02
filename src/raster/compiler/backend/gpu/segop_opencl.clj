(ns raster.compiler.backend.gpu.segop-opencl
  "OpenCL kernel generation from SegOp IR.

   Translates SegMap and SegRed records into OpenCL C source strings.
   Uses the pre-computed inputs/outputs/scalars from SegOp lowering
   instead of re-analyzing par forms.

   This is the GPU counterpart to segop_simd.clj — both consume the
   same SegOp IR but produce different target code."
  (:require [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.segop :as segop]
            [clojure.string :as str]))

;; ================================================================
;; SegOp field accessors
;; ================================================================

;; Innermost (reduced/mapped) dim — `first` for the 1-D case (unchanged).
;; The N-D segmented GPU kernel (deferred, Step 5) iterates outer segment dims.
(defn- seg-idx [segop] (:name (segop/seg-space-reduced-dim (:space segop))))
(defn- seg-bound [segop] (:bound (segop/seg-space-reduced-dim (:space segop))))

;; ================================================================
;; SegMap → OpenCL kernel
;; ================================================================

(defn generate-segmap-kernel
  "Generate an OpenCL C kernel from a SegMap record.

   Returns {:kernel-name str :source str :array-params [syms]
            :scalar-params [syms] :dtype kw}."
  [segmap out-sym & {:keys [dtype kernel-name-prefix scalar-types]
                     :or {dtype :double kernel-name-prefix "par_map"
                          scalar-types {}}}]
  (let [idx (seg-idx segmap)
        body (:lambda segmap)
        cast-fn (:cast-fn segmap)
        dtype (or (:dtype segmap) dtype)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        ;; Use pre-computed inputs/scalars from SegOp
        arr-params (vec (sort-by name (:inputs segmap)))
        scl-params (vec (sort-by name (:scalars segmap)))
        ;; Build parameter strings
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        scl-type (fn [s] (ce/scalar-native-type s scalar-types ctype))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param (str "__global " ctype "* restrict out")
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param scl-param-str "int _n_bound"]))
        ;; Emit body as C expression
        adapted-body (ce/adapt-casts-for-dtype body dtype)
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype]
                   (ce/emit-expr adapted-body idx (set (map #(symbol (name %)) arr-params))))
        cast-str (if cast-fn (str "(" (name cast-fn) ")(" body-str ")") body-str)
        source (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                    "        out[idx] = " cast-str ";\n"
                    "    }\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :out-param out-param
     :dtype dtype}))

;; ================================================================
;; SegRed → OpenCL kernel (two-phase reduction)
;; ================================================================

(defn generate-segred-kernel
  "Generate OpenCL C reduction kernels from a SegRed record.

   For two-phase reduction (default for large arrays):
   Phase 1: block-local shared-memory tree reduction
   Phase 2: single-block reduction of partial results

   Returns {:kernel-name str :source str :array-params [syms]
            :scalar-params [syms] :dtype kw :n-phases int}."
  [segred out-sym & {:keys [dtype kernel-name-prefix]
                     :or {dtype :double kernel-name-prefix "par_reduce"}}]
  (let [idx (seg-idx segred)
        bound (seg-bound segred)
        {:keys [acc init lambda]} (:reduce-op segred)
        dtype (or (:dtype segred) dtype)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        use-fp64? (= dtype :double)
        arr-params (vec (sort-by name (:inputs segred)))
        scl-params (vec (sort-by name (:scalars segred)))
        ;; Detect reduction op from lambda — unwrap let to find op, keep let for elem
        [let-bindings inner-body]
        (if (and (seq? lambda) (contains? #{'let* 'let} (first lambda)))
          (let [[_ binds & bdy] lambda]
            [(vec (partition 2 binds)) (last bdy)])
          [nil lambda])
        op-sym (when (seq? inner-body) (first inner-body))
        normalized-op (get {'clojure.core/+ '+, 'clojure.core/* '*, 'clojure.core/- '-} op-sym op-sym)
        c-op (condp = normalized-op '+ "+" '* "*" 'Math/max "fmax" 'Math/min "fmin" "+")
        c-identity-val ({"+" "0.0" "*" "1.0" "fmax" "-INFINITY" "fmin" "INFINITY"} c-op "0.0")
        identity-val ({"+" 0.0 "*" 1.0 "fmax" Double/NEGATIVE_INFINITY "fmin" Double/POSITIVE_INFINITY} c-op 0.0)
        ;; Extract element expression (the non-acc part), re-wrap in let if needed
        [_acc-pos elem-expr-raw]
        (when (and (seq? inner-body) (>= (count inner-body) 3))
          (cond
            (= (nth inner-body 1) acc) [:left (nth inner-body 2)]
            (= (nth inner-body 2) acc) [:right (nth inner-body 1)]
            (and (seq? (nth inner-body 1)) (= 'double (first (nth inner-body 1)))
                 (= acc (second (nth inner-body 1))))
            [:left (nth inner-body 2)]
            :else [nil nil]))
        ;; Re-wrap in let if there were bindings (preserves local variable scope)
        elem-expr (if (and elem-expr-raw (seq let-bindings))
                    (list 'let* (vec (mapcat identity let-bindings)) elem-expr-raw)
                    elem-expr-raw)
        adapted-elem (when elem-expr (ce/adapt-casts-for-dtype elem-expr dtype))
        idx-c-name (ce/c-symbol idx)
        elem-str (when adapted-elem
                   (binding [ce/*emit-config* ce/opencl-config
                             ce/*scalar-type* ctype]
                     (ce/emit-expr adapted-elem idx (set (map #(symbol (name %)) arr-params)) idx-c-name)))
        ;; Build kernel source
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " ctype "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        scl-param-str (str/join ", "
                                (map (fn [s] (str ctype " " (ce/c-symbol s))) scl-params))
        ;; Use output param name matching invoke-reduction-kernel expectations
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str
                                      (str "__global " ctype "* restrict output")
                                      scl-param-str
                                      "int _n_bound"]))
        ;; Static shared memory — matches invoke-reduction-kernel (no __local arg)
        source (when elem-str
                 (str (when use-fp64? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
                      "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n"
                      "__kernel void " kernel-name
                      "(" all-params ") {\n"
                      "    __local " ctype " sdata[256];\n"
                      "    int tid = get_local_id(0);\n"
                      "    " ctype " val = " c-identity-val ";\n"
                      "    int stride = get_global_size(0);\n"
                      "    int " (ce/c-symbol idx) " = get_global_id(0);\n"
                      "    for (; " (ce/c-symbol idx) " < _n_bound; " (ce/c-symbol idx) " += stride) {\n"
                      "        val = (val " c-op " " elem-str ");\n"
                      "    }\n"
                      "    sdata[tid] = val;\n"
                      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                      "    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
                      "        if (tid < s) {\n"
                      "            sdata[tid] = (sdata[tid] " c-op " sdata[tid + s]);\n"
                      "        }\n"
                      "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                      "    }\n"
                      "    if (tid == 0) output[get_group_id(0)] = sdata[0];\n"
                      "}\n"))]
    (when source
      {:kernel-name kernel-name
       :source source
       :array-params arr-params
       :scalar-params scl-params
       :dtype dtype
       :n-phases 2
       :identity-val identity-val
       :c-op c-op})))

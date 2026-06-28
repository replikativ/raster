(ns raster.compiler.backend.gpu.segop-opencl
  "OpenCL kernel generation from SegOp IR.

   Translates SegMap and SegRed records into OpenCL C source strings.
   Uses the pre-computed inputs/outputs/scalars from SegOp lowering
   instead of re-analyzing par forms.

   This is the GPU counterpart to segop_simd.clj — both consume the
   same SegOp IR but produce different target code."
  (:require [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [clojure.string :as str]))

;; ================================================================
;; SegOp field accessors
;; ================================================================

(defn- seg-idx [segop] (-> segop :space :dims first :name))
(defn- seg-bound [segop] (-> segop :space :dims first :bound))

;; ================================================================
;; SegMap → OpenCL kernel
;; ================================================================

(defn generate-segmap-kernel
  "Generate an OpenCL C kernel from a SegMap record.

   Mirrors the legacy par-map-void generator's array handling so a fused pure
   par/map (possibly composed of several maps the SOAC fuser collapsed) emits
   correct C: (1) normalize the devirtualized array prims (.invk aget_m_T-impl …)
   back to aget heads so array detection + per-element typing + emit recognize
   them (else the array is mis-classified scalar and aget becomes a broken helper
   call); (2) type each INPUT array by its declared element type (array-types,
   merged with the body's :tag metadata) — a float input read through a double*
   param silently miscompiles; (3) type the OUTPUT by the map's computed element
   dtype (:dtype segmap), which may differ from the inputs (e.g. a float input
   promoted to double by a double literal).

   Returns {:kernel-name str :source str :array-params [syms]
            :scalar-params [syms] :dtype kw}."
  [segmap out-sym & {:keys [dtype kernel-name-prefix scalar-types array-types]
                     :or {dtype :double kernel-name-prefix "par_map"
                          scalar-types {} array-types {}}}]
  (let [idx (seg-idx segmap)
        ;; (1) normalize .invk array prims → aget/aset heads
        body (ce/normalize-array-prims (:lambda segmap))
        cast-fn (:cast-fn segmap)
        out-dtype (or (:dtype segmap) dtype)
        default-ctype (get codegen/opencl-type-map dtype "double")
        out-ctype (get codegen/opencl-type-map out-dtype "double")
        ;; (2) per-array element types: declared (array-types) ∪ body :tag metadata
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ;; Use pre-computed inputs/scalars from SegOp
        arr-params (vec (sort-by name (:inputs segmap)))
        scl-params (vec (sort-by name (:scalars segmap)))
        arr-type (fn [s]
                   (let [t (get array-types s (get array-types (symbol (name s)) dtype))]
                     (get codegen/opencl-type-map t default-ctype)))
        ;; fp64 needed when the output OR any input array is double
        use-fp64? (or (= out-dtype :double)
                      (some #(= "double" (arr-type %)) arr-params))
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global const " (arr-type s) "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        ;; Integer scalar params seed *int-vars* so index math stays integer
        int-scalar-syms (set (keep (fn [[k v]] (when (= v :int) (symbol (name k)))) scalar-types))
        scl-type (fn [s]
                   (let [sname (name s)
                         explicit (get scalar-types s (get scalar-types (symbol sname)))]
                     (cond
                       (= explicit :int)    "int"
                       (= explicit :long)   "long"
                       (= explicit :double) "double"
                       (= explicit :float)  default-ctype
                       (or (re-find #"(?i)n[-_]|size|count|len|idx|offset" sname)
                           (contains? #{'long 'int} (:tag (meta s))))
                       "int"
                       :else default-ctype)))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param (str "__global " out-ctype "* restrict out")
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param scl-param-str "int _n_bound"]))
        ;; Emit body as C expression
        adapted-body (ce/adapt-casts-for-dtype body out-dtype)
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* out-ctype
                           ce/*idx-sym* idx
                           ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
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
     :dtype out-dtype}))

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

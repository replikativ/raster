(ns raster.compiler.backend.gpu.kernel-body-opencl
  "C-family lowering of verified scheduled KernelBody values.

  Matrix fragments retain their Intel OpenCL leaf. The general scalar/control path is shared by
  OpenCL, CUDA and HIP through thin target dialect descriptors; geometry is recovered from explicit
  index/operation IR, never from the source algorithm or a second schedule registry."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.matrix-body-plan :as matrix-plan]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]))

(defn- record-kind? [simple-name value]
  (= (str "raster.compiler.ir.kernel_body." simple-name)
     (some-> value class .getName)))

(defn- value-id? [value]
  (or (symbol? value) (keyword? value)))

(defn- target-name [value]
  (ce/c-symbol value))

(declare emit-index-expression target-type)

(defn- index-expression-dtype
  [expression types]
  (cond
    (integer? expression) :int
    (value-id? expression)
    (or (get types expression)
        (throw (ex-info "index expression references a value with no declared dtype"
                        {:reason :index-value-dtype-unknown :value expression})))
    (record-kind? "IndexCast" expression) (dtype/canon (:dtype expression))
    (record-kind? "IndexExpr" expression)
    (index-expression-dtype (first (:arguments expression)) types)
    :else (throw (ex-info "index expression has an unsupported operand"
                          {:reason :index-expression-operand :expression expression}))))

(defn- emit-infix [operator arguments env]
  (str "(" (str/join (str " " operator " ")
                     (map #(emit-index-expression % env) arguments)) ")"))

(defn- emit-index-expression
  [expression env]
  (cond
    (number? expression) (str expression)
    (value-id? expression) (or (get env expression) (target-name expression))
    (record-kind? "IndexCast" expression)
    (str "("
         (target-type (:dtype expression))
         ")(" (emit-index-expression (:argument expression) env) ")")
    (record-kind? "IndexExpr" expression)
    (let [arguments (:arguments expression)]
      (case (:op expression)
        :add (emit-infix "+" arguments env)
        :sub (emit-infix "-" arguments env)
        :mul (emit-infix "*" arguments env)
        :floor-div (emit-infix "/" arguments env)
        :mod (emit-infix "%" arguments env)
        :min (str "min(" (str/join ", " (map #(emit-index-expression % env) arguments)) ")")
        :max (str "max(" (str/join ", " (map #(emit-index-expression % env) arguments)) ")")
        :ceil-div (let [[numerator denominator] arguments]
                    (str "((" (emit-index-expression numerator env) " + "
                         (emit-index-expression denominator env) " - 1) / "
                         (emit-index-expression denominator env) ")"))))
    :else (throw (ex-info "OpenCL lowering cannot spell kernel index expression"
                          {:reason :kernel-body-opencl-unimplemented
                           :expression expression}))))

(defn- emit-wide-expression
  [expression env]
  (if (and (record-kind? "IndexExpr" expression) (= :mul (:op expression)))
    (str/join " * " (map #(str "(long)" (emit-index-expression % env))
                         (:arguments expression)))
    (str "(long)" (emit-index-expression expression env))))

(defn- nested-operations [operations]
  (mapcat (fn [operation]
            (cons operation
                  (when-let [nested (:operations operation)]
                    (nested-operations nested))))
          operations))

(def ^:private scalar-builtins
  (into #{"float" "double" "half" "int" "long" "short" "char"
          "uint" "ulong" "ushort" "uchar" "if"}
        (comp (filter string?) (filter #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" %)))
        (vals ce/op-map)))

(defn- validate-emitted-scalar-calls!
  [region emitted]
  (let [calls (into #{} (map second) (re-seq #"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(" emitted))
        ;; rstr_dp4a is accompanied by a helper in general kernels, but matrix store regions do not
        ;; inject helper source. Keep the scalar-region target vocabulary self-contained.
        unsupported (set/difference calls (disj scalar-builtins "rstr_dp4a"))]
    (when (seq unsupported)
      (throw (ex-info "OpenCL scalar region contains calls without a typed target lowering"
                      {:reason :scalar-region-opencl-call-unsupported
                       :calls (vec (sort unsupported)) :region region})))
    emitted))

(defn- scalar-tag [dtype]
  (symbol (name dtype)))

(defn- array-tag [dtype]
  (symbol (str (name dtype) "s")))

(defn- lower-scalar-region
  [kernel-body region parameter-names]
  (let [parameters (into {} (map (juxt :id identity)) (:parameters kernel-body))
        operand-ids (set (map :sym (:operands region)))
        scalar-ids (remove operand-ids (rest (:parameters region)))
        free-syms (subvec (vec (get-in kernel-body [:attributes :axis-symbols])) 0 2)
        [i-sym j-sym] free-syms
        ctype (dtype/ctype :opencl (:result-dtype region))
        array-syms (set (map (comp #(symbol (name %)) :sym) (:operands region)))
        int-vars (into #{} (map #(symbol (name %))) free-syms)
        indices (into {} (map (juxt :sym (comp axis-map/index-expr :map)))
                      (:operands region))
        expression (descriptor/rewrite-aget-indices (:expression region) indices)
        accumulator (first (:parameters region))
        accumulator-token (str "__acc_" (name (gensym "")))
        typed-parameters
        (into {}
              (for [id (rest (:parameters region))
                    :let [parameter (get parameters id)]]
                [id (with-meta (symbol (get parameter-names id))
                      {:raster.type/tag (if (= :scalar (:kind parameter))
                                          (scalar-tag (:dtype parameter))
                                          (array-tag (:dtype parameter)))})]))
        replacements (assoc typed-parameters accumulator
                            (with-meta (symbol accumulator-token)
                              {:raster.type/tag (scalar-tag (:result-dtype region))}))
        emitted (validate-emitted-scalar-calls!
                 region
                 (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr
                    (walk/postwalk-replace replacements expression)
                    (gensym "z__") array-syms)))
        params (apply str
                      (concat
                       (for [{:keys [sym dtype] :or {dtype :float}} (:operands region)]
                         (str ", __global const " (dtype/ctype :opencl dtype)
                              "* restrict " (get parameter-names sym)))
                       (for [id scalar-ids
                             :let [parameter (get parameters id)]]
                         (str ", " (dtype/ctype :opencl (:dtype parameter))
                              " " (get parameter-names id)))))]
    {:epilogue (fn [accumulator-expression row col]
                 (-> emitted
                     (str/replace accumulator-token (str "(" accumulator-expression ")"))
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol i-sym) "\\b")) row)
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol j-sym) "\\b")) col)))
     :epilogue-params params
     :region region}))

(declare lower-scalar-ssa-region)

(defn lower-store-region
  "Lower the one verified ScalarRegion shared by all matrix stores to OpenCL scalar syntax.

  Returns nil for identity stores. This is a target lowering of typed KernelBody data: callers
  never supply source callbacks, parameter declaration strings, or helper source."
  ([kernel-body]
   (lower-store-region
    kernel-body
    (into {} (map (fn [parameter] [(:id parameter) (ce/c-symbol (:id parameter))]))
          (:parameters kernel-body))))
  ([kernel-body parameter-names]
  (let [kernel-body (body/validate! kernel-body)
        stores (filter #(record-kind? "TileStore" %)
                       (nested-operations (:operations kernel-body)))
        regions (distinct (map :value-region stores))]
    (when-not (= 1 (count regions))
      (throw (ex-info "all matrix tile stores must carry the same scalar region"
                      {:reason :raster/bug :regions (vec regions)})))
    (when-let [region (first regions)]
      (if (record-kind? "ScalarSSARegion" region)
        (lower-scalar-ssa-region kernel-body region parameter-names)
        (lower-scalar-region kernel-body region parameter-names))))))

(defn- require!
  [condition message data]
  (when-not condition
    (throw (ex-info message (assoc data :reason :kernel-body-opencl-unimplemented)))))

(defn- intel-matrix-plan
  [kernel-body]
  (let [{:keys [instruction] :as plan} (matrix-plan/analyze kernel-body)
        {mi :m ni :n ki :k subgroup :subgroup} instruction]
    (require! (and (= :dpas (:family instruction))
                   (= 8 mi) (= 16 ki)
                   (contains? #{8 16} subgroup)
                   (= ni subgroup))
              "Intel OpenCL lowering has no builtin for this matrix instruction"
              {:instruction instruction})
    plan))

(defn- emit-plan
  [kernel-name {:keys [mi ni ki subgroup block-m block-n block-k sg-m sg-n
                       ncols lhs-ids rhs-ids mad-by-operands stores prefetch result-dtype
                       dimension-parameters schedule-parameters group-z k-lower k-upper
                       buffer-offsets]}
   {:keys [epilogue epilogue-params parameter-names]}]
  (let [nms (count lhs-ids)
        nns (count rhs-ids)
        ksteps (quot block-k ki)
        acc-id (fn [m n] (:accumulator (get mad-by-operands
                                            [(nth lhs-ids m) (nth rhs-ids n)])))
        store-by-fragment (into {} (map (juxt :fragment identity)) stores)
        ms (range nms)
        ns (range nns)
        amul (fn [m] (if (zero? m) "m_base" (str "m_base+" (* m mi))))
        c-type (case result-dtype :float "float" :half "half")
        store-cast (if (= :half result-dtype) "(half)" "")
        parameter-names (merge (zipmap [(:m dimension-parameters)
                                        (:n dimension-parameters)
                                        (:k dimension-parameters)]
                                       ["M" "N" "K"])
                               (into {} (map (fn [parameter]
                                               [(:id parameter) (target-name (:id parameter))])
                                             schedule-parameters))
                               parameter-names)
        index-names (cond-> parameter-names group-z (assoc group-z (target-name group-z)))
        sliced-k? (not= [0 (:k dimension-parameters)] [k-lower k-upper])
        k-begin (if sliced-k? "k_begin" "0")
        k-end (if sliced-k? "k_end" "K")
        scalar-params (apply str
                             (for [parameter schedule-parameters]
                               (str ", int " (get parameter-names (:id parameter)))))
        grid-z-line (when group-z
                      (str "    int " (get index-names group-z)
                           " = (int)get_group_id(2);\n"))
        offset-line (fn [pointer role]
                      (when-let [offset (get buffer-offsets role)]
                        (when-not (and (number? offset) (zero? offset))
                          (str "    " pointer " += "
                               (emit-wide-expression offset index-names) ";\n"))))
        k-range-lines (when sliced-k?
                        (str "    int k_begin = "
                             (emit-index-expression k-lower index-names) ";\n"
                             "    int k_end = "
                             (emit-index-expression k-upper index-names) ";\n"
                             "    if (k_begin >= k_end) return;\n"))
        kstep (fn [kpos]
                (str "        { int pk = " kpos " + " (* prefetch ki) ";\n"
                     "          if (pk < " k-end ") {\n"
                     (apply str
                            (for [m ms]
                              (str "            intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(pk, " (amul m) "));\n")))
                     "          } }\n"
                     (apply str
                            (for [n ns]
                              (str "        bp" n " = as_int8(intel_subgroup_block_read_transform_u16_k16((__global void*)B, b_wb, K, b_pb, (int2)(n_base" n ", " kpos ")));\n")))
                     (apply str
                            (for [m ms]
                              (str "        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" kpos ", " (amul m) "), &a" m ");\n")))
                     (apply str (for [m ms] (str "        sa" m " = as_short8(a" m ");\n")))
                     (apply str
                            (for [m ms n ns]
                              (do
                                (require! (contains? store-by-fragment (acc-id m n))
                                          "matrix accumulator has no store operation"
                                          {:accumulator (acc-id m n)})
                                (str "        acc" m n " = intel_sub_group_f16_f16_matrix_mad_k16(sa" m ", bp" n ", acc" m n ");\n"))))))]
    (str
     "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
     "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n\n"
     "// Tiled GEMM (parametric): WG " block-m "x" block-n ", SG " sg-m "x" sg-n
     ", K " block-k ", DPAS " mi "x" ni "x" ki ", sg " subgroup "\n"
     "__attribute__((intel_reqd_sub_group_size(" subgroup ")))\n"
     "__kernel void " kernel-name "(\n"
     "    __global const half* restrict A,\n    __global const half* restrict B,\n"
     "    __global " c-type "* restrict C,\n    int M, int N, int K"
     scalar-params
     epilogue-params
     ") {\n"
     "    int sg_id = get_sub_group_id();\n    int sg_lid = get_sub_group_local_id();\n"
     "    int sg_row = sg_id / " ncols ";\n    int sg_col = sg_id % " ncols ";\n"
     "    int m_base = get_group_id(1) * " block-m " + sg_row * " sg-m ";\n"
     (apply str
            (for [n ns]
              (str "    int n_base" n " = get_group_id(0) * " block-n " + sg_col * " sg-n
                   (when (pos? n) (str " + " (* n ni))) ";\n")))
     "    if (m_base >= M || n_base0 >= N) return;\n"
     grid-z-line
     k-range-lines
     (offset-line "A" :lhs)
     (offset-line "B" :rhs)
     (offset-line "C" :result)
     (apply str
            (for [m ms]
              (str "    float" mi " "
                   (str/join ", " (for [n ns] (str "acc" m n "=0.0f"))) ";\n")))
     "    int a_wb = K * 2, a_pb = K * 2;\n    int b_wb = N * 2, b_pb = N * 2;\n"
     "    ushort8 " (str/join ", " (for [m ms] (str "a" m))) ";\n"
     "    short8 " (str/join ", " (for [m ms] (str "sa" m))) ";\n"
     "    int8 " (str/join ", " (for [n ns] (str "bp" n))) ";\n"
     (apply str
            (for [p (range prefetch)
                  :let [position (if sliced-k?
                                   (if (zero? p)
                                     k-begin
                                     (str k-begin " + " (* p ki)))
                                   (str (* p ki)))]]
              (str "    if (" position " < " k-end ") {\n"
                   (apply str
                          (for [m ms]
                            (str "        intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" position ", " (amul m) "));\n")))
                   "    }\n")))
     "    int k = " k-begin ";\n"
     "    for (; k + " (dec block-k) " < " k-end "; k += " block-k ") {\n"
     (apply str
            (for [ks (range ksteps)]
              (kstep (if (zero? ks) "k" (str "k + " (* ks ki))))))
     "    }\n"
     "    for (; k < " k-end "; k += " ki ") {\n"
     (kstep "k")
     "    }\n"
     (apply str
            (for [m ms i (range mi)]
              (str "    { int row = m_base + " (+ (* m mi) i) ";\n      if (row < M) {\n"
                   (apply str
                          (for [n ns]
                            (let [acc-expr (str "acc" m n ".s" i)]
                              (str "        { int col = n_base" n " + sg_lid;\n          if (col < N) "
                                   (if epilogue
                                     (str "C[row*N+col] = " store-cast "("
                                          (epilogue acc-expr "row" "col") ");\n")
                                     (str "C[row*N+col] = " store-cast "(" acc-expr ");\n"))
                                   "        }\n"))))
                   "      }\n    }\n")))
     "}\n")))

(defn- matrix-parameter-names
  [kernel-body overrides]
  (let [{:keys [m n k]} (get-in kernel-body [:attributes :dimension-parameters])]
    (merge
     (into {}
           (map (fn [{:keys [id role]}]
                  [id (case role :lhs "A" :rhs "B" :result "C" (ce/c-symbol id))]))
           (:parameters kernel-body))
     {m "M" n "N" k "K"}
     overrides)))

(defn emit-matrix-kernel
  "Lower a verified f16 DPAS KernelBody directly to OpenCL C.

  ScalarRegion stores are lowered as part of this boundary. The optional target map contains only
  target naming policy; it cannot inject source or replace the store expression."
  ([kernel-name kernel-body]
   (emit-matrix-kernel kernel-name kernel-body {}))
  ([kernel-name kernel-body {:keys [parameter-names]}]
   (let [parameter-names (matrix-parameter-names kernel-body parameter-names)]
     (emit-plan kernel-name (intel-matrix-plan kernel-body)
                (assoc (or (lower-store-region kernel-body parameter-names) {})
                       :parameter-names parameter-names)))))

;; ---------------------------------------------------------------------------
;; General scalar/control KernelBody lowering
;; ---------------------------------------------------------------------------

(def ^:private scalar-operation-kinds
  #{"ScalarCompute" "ScalarLoad" "ScalarStore" "AtomicRMW" "Yield" "IfRegion" "ForLoop"
    "PipelineYield" "PipelinedFor"
    "Collective" "WorkgroupBarrier" "AsyncWorkgroupCopy" "AsyncCommit" "AsyncWait"})

(defn- scalar-body-operations
  [operations]
  (mapcat
   (fn [operation]
     (concat [operation]
             (when (record-kind? "IfRegion" operation)
               (concat (scalar-body-operations (:then-operations operation))
                       (scalar-body-operations (:else-operations operation))))
             (when (or (record-kind? "ForLoop" operation)
                       (record-kind? "PipelinedFor" operation))
               (scalar-body-operations (:operations operation)))))
   operations))

(defn- scalar-expressions
  [value]
  ;; ScalarExpr is legal in any scalar-valued operation field, not only ScalarCompute: masked
  ;; load fallbacks, stores/atomics, loop arguments and yields, and collective inputs all carry
  ;; values.  Traverse the complete verified operation tree so helper discovery is not coupled to
  ;; one convenient placement of an expression.
  (cond
    (record-kind? "ScalarExpr" value)
    (cons value (mapcat scalar-expressions (:arguments value)))

    (map? value) (mapcat scalar-expressions (vals value))
    ;; Keep this deliberately broader than vectors/lists: the verifier may add a scalar-valued
    ;; collection field to a legal operation without making helper discovery placement-sensitive.
    (coll? value) (mapcat scalar-expressions value)
    :else []))

(defn- trapping-arithmetic-requirements
  [operations]
  (->> operations
       (mapcat scalar-expressions)
       (keep (fn [expression]
               (when (= :trap (get-in expression [:options :overflow]))
                 [(intrinsics/canonical (:op expression))
                  (dtype/canon (:result-type expression))])))
       distinct
       (sort-by pr-str)))

(defn- scalar-defined-ids
  [operations]
  (mapcat
   (fn [operation]
     (cond
       (contains? #{"ScalarCompute" "ScalarLoad" "Collective"}
                  (some-> operation class .getSimpleName))
       [(:id (:result operation))]

       (record-kind? "IfRegion" operation)
       (concat (map :id (:results operation))
               (scalar-defined-ids (:then-operations operation))
               (scalar-defined-ids (:else-operations operation)))

       (record-kind? "ForLoop" operation)
       (concat [(:id (:index operation))]
               (map (comp :id :binding) (:iter-args operation))
               (map :id (:results operation))
               (scalar-defined-ids (:operations operation)))

       (record-kind? "PipelinedFor" operation)
       (concat [(:id (:index operation))]
               (map (comp :id :binding) (:iter-args operation))
               (map :id (:results operation))
               (scalar-defined-ids (:operations operation)))

       :else []))
   operations))

(defn- scalar-local-name [id]
  (str "rstr_" (target-name id)))

(def ^:dynamic *scalar-dialect*
  "Validated target spelling used by the shared scalar/control emitter."
  (c-dialect/resolve! :opencl-intel))

(defn- target-type [type]
  (c-dialect/type-name *scalar-dialect* type))

(defn- emit-floating-literal [value suffix]
  (cond
    (Double/isNaN (double value)) "NAN"
    (= Double/POSITIVE_INFINITY (double value)) "INFINITY"
    (= Double/NEGATIVE_INFINITY (double value)) "(-INFINITY)"
    :else (str (Double/toString (double value)) suffix)))

(defn- emit-literal
  [{:keys [value type]}]
  (case (dtype/canon type)
    :byte (str "(" (target-type :byte) ")" value)
    :int (str value)
    :long (str value (if (c-dialect/opencl? *scalar-dialect*) "L" "LL"))
    :half (if (c-dialect/opencl? *scalar-dialect*)
            (str "(half)(" (emit-floating-literal value "f") ")")
            (str "__float2half_rn(" (emit-floating-literal value "f") ")"))
    :float (emit-floating-literal value "f")
    :double (emit-floating-literal value "")
    (throw (ex-info "C-family scalar literal has no target spelling"
                    {:reason :kernel-body-c-unimplemented
                     :dialect (:id *scalar-dialect*) :literal value :type type}))))

(defn- scalar-value-type
  [value types]
  (cond
    (record-kind? "Literal" value) (if (= :predicate (:type value))
                                     :predicate (dtype/canon (:type value)))
    (record-kind? "ScalarExpr" value) (if (= :predicate (:result-type value))
                                        :predicate (dtype/canon (:result-type value)))
    (value-id? value) (get types value)
    :else nil))

(declare emit-scalar-value)

(defn- cast-suffix
  [rounding overflow]
  (str (case overflow
         :saturate "_sat"
         (:wrap :trap :exact :ieee) "")
       (case rounding
         :toward-zero "_rtz"
         :nearest-even "_rte"
         :up "_rtp"
         :down "_rtn"
         :exact "")))

(defn- emit-cast
  [expression context]
  (let [argument (first (:arguments expression))
        source-type (scalar-value-type argument (:types context))
        result-type (dtype/canon (:result-type expression))
        {:keys [rounding overflow]} (:options expression)
        argument-source (emit-scalar-value argument context)
        source-fp? (dtype/fp-dtype? source-type)
        result-fp? (dtype/fp-dtype? result-type)
        narrowing-fp? (and source-fp? result-fp?
                           (> (dtype/bytes-of source-type) (dtype/bytes-of result-type)))]
    (cond
      ;; Same-width and widening FP conversions are exact in the KernelBody contract.
      (and source-fp? result-fp? (not narrowing-fp?)
           (= [:exact :exact] [rounding overflow]))
      (cond
        (and (not (c-dialect/opencl? *scalar-dialect*))
             (= source-type :half) (= result-type :float))
        (str "__half2float(" argument-source ")")

        (and (not (c-dialect/opencl? *scalar-dialect*))
             (= source-type :half) (= result-type :double))
        (str "(double)(__half2float(" argument-source "))")

        :else (str "(" (target-type result-type) ")(" argument-source ")"))

      ;; OpenCL conversion suffixes state the requested rounding of IEEE narrowing.
      (and narrowing-fp? (= :ieee overflow))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (case [source-type result-type rounding]
          [:float :half :nearest-even] (str "__float2half_rn(" argument-source ")")
          [:double :float :nearest-even] (str "__double2float_rn(" argument-source ")")
          (throw (ex-info "CUDA/HIP cannot preserve this floating narrowing policy"
                          {:reason :kernel-body-c-cast-policy
                           :dialect (:id *scalar-dialect*)
                           :source-type source-type :result-type result-type
                           :rounding rounding :overflow overflow}))))

      ;; Wrapping integral conversion: OpenCL spells it `convert_T`; on CUDA and HIP the plain C
      ;; cast between two's-complement integer types is the same modular truncation.
      (and (not source-fp?) (not result-fp?) (= :wrap overflow))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (str "(" (target-type result-type) ")(" argument-source ")"))

      ;; Saturating FP->integer conversion has a direct OpenCL spelling only.
      (and source-fp? (not result-fp?) (= :saturate overflow))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (throw (ex-info "CUDA/HIP cannot preserve this saturating cast policy"
                        {:reason :kernel-body-c-cast-policy
                         :dialect (:id *scalar-dialect*)
                         :source-type source-type :result-type result-type
                         :rounding rounding :overflow overflow})))

      ;; Integral widening is exact. Narrowing/exact and trapping conversions need a proof or
      ;; runtime check that this target layer does not currently carry.
      (and (not source-fp?) (not result-fp?) (= [:exact :exact] [rounding overflow])
           (<= (dtype/bytes-of source-type) (dtype/bytes-of result-type)))
      (str "(" (target-type result-type) ")(" argument-source ")")

      ;; Every signed 32-bit integer is exactly representable as IEEE f64.
      (and (not source-fp?) result-fp? (= :double result-type)
           (<= (dtype/bytes-of source-type) 4)
           (= [:exact :exact] [rounding overflow]))
      (str "(" (target-type result-type) ")(" argument-source ")")

      ;; Integral-to-floating precision loss is explicit. OpenCL conversion builtins preserve the
      ;; requested rounding; CUDA/HIP expose the corresponding signed integer RN intrinsics.
      (and (not source-fp?) result-fp? (= :nearest-even rounding)
           (= (if (= :half result-type) :ieee :exact) overflow))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (case [source-type result-type]
          [:byte :float] (str "(float)(" argument-source ")")
          [:byte :double] (str "(double)(" argument-source ")")
          [:int :float] (str "__int2float_rn(" argument-source ")")
          [:int :double] (str "(double)(" argument-source ")")
          [:long :float] (str "__ll2float_rn(" argument-source ")")
          [:long :double] (str "__ll2double_rn(" argument-source ")")
          (throw (ex-info "CUDA/HIP cannot preserve this integral-to-floating policy"
                          {:reason :kernel-body-c-cast-policy
                           :dialect (:id *scalar-dialect*)
                           :source-type source-type :result-type result-type
                           :rounding rounding :overflow overflow}))))

      :else
      (throw (ex-info "C-family target cannot preserve this KernelBody cast policy"
                      {:reason :kernel-body-c-cast-policy
                       :dialect (:id *scalar-dialect*)
                       :source-type source-type :result-type result-type
                       :rounding rounding :overflow overflow})))))

(defn- emit-intrinsic-expression
  [expression context]
  (let [op (intrinsics/canonical (:op expression))
        lowering (intrinsics/op->c-lowering op false)
        arguments (mapv #(emit-scalar-value % context) (:arguments expression))
        operand-type (scalar-value-type (first (:arguments expression)) (:types context))
        integral? (contains? #{:byte :int :long} operand-type)
        overflow-policy (get-in expression [:options :overflow])]
    (cond
      ;; C-family signed overflow is undefined. Preserve the verified modulo-2^N contract by
      ;; doing the arithmetic in the same-width unsigned representation and converting the bits
      ;; back to the expression's declared signed storage type.
      (= :wrap overflow-policy)
      (let [unsigned-type (c-dialect/unsigned-type-name *scalar-dialect* operand-type)]
        (str "(" (target-type operand-type) ")((" unsigned-type ")(" (first arguments)
             ") " (:op lowering) " (" unsigned-type ")(" (second arguments) "))"))

      ;; A checked helper first proves representability in unsigned arithmetic.  It only evaluates
      ;; the signed operation on the safe branch, so the check itself cannot introduce C signed UB.
      (= :trap overflow-policy)
      (str (c-dialect/trapping-arithmetic-name *scalar-dialect* op operand-type)
           "(" (str/join ", " arguments) ")")

      ;; The shared C descriptor uses the floating spelling. OpenCL integer min/max are distinct
      ;; overloads, so target spelling must retain the verified operand dtype.
      (and integral? (contains? #{:min :max} op))
      (str (name op) "(" (str/join ", " arguments) ")")

      ;; Signed right shift is not the target-neutral unsigned-shift contract.
      (= :ushr op)
      (let [[value amount] arguments
            unsigned-type (c-dialect/unsigned-type-name *scalar-dialect* operand-type)]
        (str "(" (target-type operand-type) ")((" unsigned-type ")(" value ") >> "
             amount ")"))

      ;; OpenCL's integral abs returns an unsigned type, while KernelBody currently declares a
      ;; same-signed-type result. Refuse the mismatch until the IR states that representation step.
      (and (c-dialect/opencl? *scalar-dialect*) integral? (= :abs op))
      (throw (ex-info "OpenCL integral abs disagrees with the KernelBody result type"
                      {:reason :kernel-body-opencl-intrinsic
                       :operation op :operand-type operand-type}))

      :else
      (case (:kind lowering)
        :infix (str "(" (str/join (str " " (:op lowering) " ") arguments) ")")
        :fn (str (:op lowering) "(" (str/join ", " arguments) ")")
        :floored-mod (let [[a b] arguments]
                       (str "((" a " % " b " + " b ") % " b ")"))
        (throw (ex-info "OpenCL has no scalar spelling for a verified KernelBody intrinsic"
                        {:reason :kernel-body-opencl-intrinsic
                         :operation op :expression expression}))))))

(defn- emit-scalar-value
  [value {:keys [names] :as context}]
  (cond
    (record-kind? "Literal" value)
    (if (= :predicate (:type value)) (if (:value value) "true" "false") (emit-literal value))

    (value-id? value)
    (or (get names value)
        (throw (ex-info "OpenCL scalar lowering lost an SSA name"
                        {:reason :raster/bug :value value})))

    (record-kind? "ScalarExpr" value)
    (case (:op value)
      :cast (emit-cast value context)
      :select (let [[condition if-true if-false] (:arguments value)]
                (str "(" (emit-scalar-value condition context) " ? "
                     (emit-scalar-value if-true context) " : "
                     (emit-scalar-value if-false context) ")"))
      :isnan (str "isnan(" (emit-scalar-value (first (:arguments value)) context) ")")
      (emit-intrinsic-expression value context))

    :else
    (throw (ex-info "OpenCL scalar lowering received an untyped value"
                    {:reason :raster/bug :value value}))))

(defn- emit-layout-expression
  [expression names]
  (cond
    (number? expression) (str expression)
    (value-id? expression) (or (get names expression) (scalar-local-name expression))
    (seq? expression)
    (let [[op & arguments] expression
          operator ({'+ "+" '* "*" '- "-" 'quot "/"} op)]
      (when-not operator
        (throw (ex-info "OpenCL cannot lower a symbolic layout stride"
                        {:reason :kernel-body-opencl-layout :expression expression})))
      (str "(" (str/join (str " " operator " ")
                         (map #(emit-layout-expression % names) arguments)) ")"))
    :else
    (throw (ex-info "OpenCL cannot lower a layout expression"
                    {:reason :kernel-body-opencl-layout :expression expression}))))

(defn- emit-storage-index
  [storage coordinates names]
  (let [layout (:layout storage)
        dense-offset
        (fn [dense-layout]
          (let [strides (layout/resolve-strides dense-layout)
                terms (mapv (fn [coordinate stride]
                              (str "((long)(" (emit-index-expression coordinate names)
                                   ") * (long)(" (emit-layout-expression stride names) "))"))
                            coordinates strides)]
            (if (seq terms) (str/join " + " terms) "0")))
        local-offset
        (case (:kind layout)
          (:row-major :col-major) (dense-offset layout)
          :shared-memory
          (do
            (layout/validate-shared-memory! layout)
            (if (= :identity (:swizzle layout))
              (dense-offset layout)
              (let [[row column] coordinates
                    row-source (emit-index-expression row names)
                    column-source (emit-index-expression column names)
                    columns (second (:shape layout))
                    mask (dec (layout/swizzle-width (:swizzle layout)))]
                (str "((long)(" row-source ") * (long)(" columns ") + "
                     "((long)(" column-source ") ^ ((long)(" row-source ") & " mask ")))"))))
          (throw (ex-info "C-family scalar memory lowering requires a dense strided layout or verified shared layout"
                          {:reason :kernel-body-opencl-layout :storage (:id storage)
                           :layout layout})))
        view-offset (some-> storage :view :element-offset)]
    (if view-offset
      (str "((long)(" (emit-index-expression view-offset names) ") + " local-offset ")")
      (str "(" local-offset ")"))))

(defn- lower-scalar-ssa-region
  [kernel-body region parameter-names]
  (let [storage (into {} (map (juxt :id identity)) (:parameters kernel-body))
        accumulator (first (:parameters region))
        [row-id col-id] (:indices region)
        accumulator-token (str "__acc_" (name (gensym "")))
        row-token (str "__row_" (name (gensym "")))
        col-token (str "__col_" (name (gensym "")))
        initial-context
        {:names (merge parameter-names
                       {accumulator accumulator-token row-id row-token col-id col-token})
         :types (merge
                 (into {} (map (fn [id] [id (dtype/canon (:dtype (get storage id)))]))
                       (rest (:parameters region)))
                 {accumulator (dtype/canon (:accumulator-dtype region))
                  row-id :int col-id :int})}
        final-context
        (reduce
         (fn [context operation]
           (cond
             (record-kind? "ScalarCompute" operation)
             (let [result (:result operation)
                   expression (emit-scalar-value (:expression operation) context)]
               (-> context
                   (assoc-in [:names (:id result)] (str "(" expression ")"))
                   (assoc-in [:types (:id result)] (dtype/canon (:type result)))))

             (record-kind? "ScalarLoad" operation)
             (let [result (:result operation)
                   parameter (get storage (:buffer operation))
                   index (emit-storage-index parameter (:coordinates operation) (:names context))
                   load (str (get parameter-names (:buffer operation)) "[" index "]")]
               (when (:predicate operation)
                 (throw (ex-info "matrix scalar SSA store regions do not admit masked loads"
                                 {:reason :kernel-body-matrix-scalar-region-mask
                                  :operation operation})))
               (-> context
                   (assoc-in [:names (:id result)] load)
                   (assoc-in [:types (:id result)] (dtype/canon (:type result)))))

             :else
             (throw (ex-info "matrix store region contains unsupported scalar SSA"
                             {:reason :kernel-body-matrix-scalar-region-operation
                              :operation operation}))))
         initial-context (:operations region))
        emitted (get-in final-context [:names (:result region)])
        external-parameters (rest (:parameters region))
        epilogue-parameters
        (filterv #(= :epilogue (:role (get storage %))) external-parameters)
        params
        (apply str
               (for [id epilogue-parameters
                     :let [parameter (get storage id)]]
                 (if (= :scalar (:kind parameter))
                   (str ", " (dtype/ctype :opencl (:dtype parameter)) " "
                        (get parameter-names id))
                   (str ", __global const " (dtype/ctype :opencl (:dtype parameter))
                        "* restrict " (get parameter-names id)))))]
    (when-not emitted
      (throw (ex-info "matrix scalar SSA region has no emitted result"
                      {:reason :raster/bug :region region})))
    {:epilogue (fn [accumulator-expression row col]
                 (-> emitted
                     (str/replace accumulator-token (str "(" accumulator-expression ")"))
                     (str/replace row-token row)
                     (str/replace col-token col)))
     :epilogue-params params
     :region region}))

(defn- emit-mask-predicate
  [predicate names]
  (let [arguments (:arguments predicate)]
    (case (:op predicate)
      :lt (emit-infix "<" arguments names)
      :lte (emit-infix "<=" arguments names)
      :eq (emit-infix "==" arguments names)
      :and (str "(" (str/join " && " (map #(emit-mask-predicate % names) arguments)) ")")
      :or (str "(" (str/join " || " (map #(emit-mask-predicate % names) arguments)) ")")
      :not (str "(!" (emit-mask-predicate (first arguments) names) ")")
      (throw (ex-info "OpenCL cannot lower a KernelBody mask predicate"
                      {:reason :raster/bug :predicate predicate})))))

(defn- emit-mask
  [mask-id {:keys [masks names]}]
  (when mask-id
    (let [predicates (:predicates (get masks mask-id))]
      (str "(" (str/join " && " (map #(emit-mask-predicate % names) predicates)) ")"))))

(defn- indent-lines [depth source]
  (let [prefix (apply str (repeat depth "  "))]
    (apply str (map #(str prefix % "\n") (str/split-lines source)))))

(declare emit-scalar-operations)

(defn- add-value
  [context spec]
  (-> context
      (assoc-in [:names (:id spec)] (scalar-local-name (:id spec)))
      (assoc-in [:types (:id spec)] (if (= :predicate (:type spec))
                                      :predicate (dtype/canon (:type spec))))))

(defn- emit-yield-assignments
  [results values context depth]
  (apply str
         (map (fn [result value]
                (indent-lines depth
                              (str (get-in context [:names (:id result)]) " = "
                                   (emit-scalar-value value context) ";")))
              results values)))

(defn- async-group-events
  [context group]
  (or (:events group)
      (into [] (keep #(get-in context [:async-copies %])) (:copies group))))

(defn- emit-layout-aware-cooperative-copy
  [operation context source-storage destination-storage source-base destination-base copy-name]
  (let [destination-layout (:layout destination-storage)
        [row column] (:destination-coordinates operation)
        row-source (emit-index-expression row (:names context))
        column-source (emit-index-expression column (:names context))
        source-index (emit-storage-index source-storage (:source-coordinates operation)
                                         (:names context))
        columns (second (:shape destination-layout))
        mask (dec (layout/swizzle-width (:swizzle destination-layout)))
        offset (str copy-name "_element_offset")
        thread-id (c-dialect/workgroup-linear-thread-id
                   *scalar-dialect* (:workgroup-shape context))]
    (str "for (int " offset " = " thread-id "; " offset " < " (:elements operation)
         "; " offset " += " (:workgroup-width context) ") {\n"
         "  " destination-base "[((long)(" row-source ") * (long)(" columns ") + "
         "((long)(" column-source ") + " offset ") ^ ((long)(" row-source ") & " mask
         "))] = " source-base "[((long)(" source-index ") + " offset ")];\n"
         "}")))

(defn- emit-scalar-operation
  [operation context depth]
  (cond
    (record-kind? "ScalarCompute" operation)
    (let [result (:result operation)
          next-context (add-value context result)]
      [(indent-lines depth
                     (str (target-type (get-in next-context [:types (:id result)])) " "
                          (get-in next-context [:names (:id result)]) " = "
                          (emit-scalar-value (:expression operation) context) ";"))
       next-context])

    (record-kind? "ScalarLoad" operation)
    (let [result (:result operation)
          storage (get-in context [:storage (:buffer operation)])
          next-context (add-value context result)
          base-name (get-in context [:names (or (some-> storage :view :buffer)
                                                (:id storage))])
          index (emit-storage-index storage (:coordinates operation) (:names context))
          load (str base-name "[" index "]")
          source (if-let [predicate (emit-mask (:predicate operation) context)]
                   (str "(" predicate " ? " load " : "
                        (emit-scalar-value (:other operation) context) ")")
                   load)]
      [(indent-lines depth
                     (str (target-type (get-in next-context [:types (:id result)])) " "
                          (get-in next-context [:names (:id result)]) " = " source ";"))
       next-context])

    (record-kind? "ScalarStore" operation)
    (let [storage (get-in context [:storage (:buffer operation)])
          base-name (get-in context [:names (or (some-> storage :view :buffer)
                                                (:id storage))])
          assignment (str base-name "["
                          (emit-storage-index storage (:coordinates operation) (:names context))
                          "] = " (emit-scalar-value (:value operation) context) ";")]
      [(if-let [predicate (emit-mask (:predicate operation) context)]
         (str (indent-lines depth (str "if (" predicate ") {"))
              (indent-lines (inc depth) assignment)
              (indent-lines depth "}"))
         (indent-lines depth assignment))
       context])

    (record-kind? "AtomicRMW" operation)
    (let [storage (get-in context [:storage (:buffer operation)])
          base-name (get-in context [:names (or (some-> storage :view :buffer)
                                                (:id storage))])
          index (emit-storage-index storage (:coordinates operation) (:names context))
          atomic-name (c-dialect/atomic-add-name *scalar-dialect* (:dtype storage))
          statement (str atomic-name "(" base-name " + " index ", "
                         (emit-scalar-value (:value operation) context) ");")]
      [(if-let [predicate (emit-mask (:predicate operation) context)]
         (str (indent-lines depth (str "if (" predicate ") {"))
              (indent-lines (inc depth) statement)
              (indent-lines depth "}"))
         (indent-lines depth statement))
       context])

    (record-kind? "IfRegion" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          declarations (apply str
                              (for [result results]
                                (indent-lines depth
                                              (str (target-type
                                                    (get-in result-context
                                                            [:types (:id result)]))
                                                   " "
                                                   (get-in result-context [:names (:id result)])
                                                   ";"))))
          then-ops (pop (:then-operations operation))
          else-ops (pop (:else-operations operation))
          then-yield (peek (:then-operations operation))
          else-yield (peek (:else-operations operation))
          [then-source then-context] (emit-scalar-operations then-ops context (inc depth))
          [else-source else-context] (emit-scalar-operations else-ops context (inc depth))]
      [(str declarations
            (indent-lines depth (str "if (" (get-in context [:names (:condition operation)]) ") {"))
            then-source
            (emit-yield-assignments results (:values then-yield)
                                    (update then-context :names merge (:names result-context))
                                    (inc depth))
            (indent-lines depth "} else {")
            else-source
            (emit-yield-assignments results (:values else-yield)
                                    (update else-context :names merge (:names result-context))
                                    (inc depth))
            (indent-lines depth "}"))
       result-context])

    (record-kind? "ForLoop" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          index (:index operation)
          loop-context (add-value result-context index)
          loop-context (reduce (fn [ctx arg] (add-value ctx (:binding arg)))
                               loop-context (:iter-args operation))
          initializers (apply str
                              (map (fn [result arg]
                                     (indent-lines depth
                                                   (str (target-type
                                                         (get-in result-context
                                                                 [:types (:id result)]))
                                                        " "
                                                        (get-in result-context
                                                                [:names (:id result)])
                                                        " = "
                                                        (emit-scalar-value (:initial arg) context)
                                                        ";")))
                                   results (:iter-args operation)))
          index-name (get-in loop-context [:names (:id index)])
          bindings (apply str
                          (map (fn [arg result]
                                 (let [binding (:binding arg)]
                                   (indent-lines (inc depth)
                                                 (str (target-type
                                                       (get-in loop-context
                                                               [:types (:id binding)]))
                                                      " "
                                                      (get-in loop-context
                                                              [:names (:id binding)])
                                                      " = "
                                                      (get-in result-context
                                                              [:names (:id result)])
                                                      ";"))))
                               (:iter-args operation) results))
          body-operations (pop (:operations operation))
          yield-op (peek (:operations operation))
          [body-source body-context] (emit-scalar-operations body-operations loop-context
                                                             (inc depth))
          index-type (get-in loop-context [:types (:id index)])
          unsigned-index-type (c-dialect/unsigned-type-name *scalar-dialect* index-type)
          upper-source (emit-index-expression (:upper operation) (:names context))
          step-source (if (integer? (:step operation))
                        (str (:step operation))
                        (emit-index-expression (:step operation) (:names context)))
          loop-header (str "for (" (target-type index-type)
                           " " index-name " = "
                           (emit-index-expression (:lower operation) (:names context)) "; "
                           index-name " < "
                           upper-source ";) {")
          checked-advance
          (str (indent-lines
                (inc depth)
                (str "if ((" unsigned-index-type ")(" upper-source ") - ("
                     unsigned-index-type ")(" index-name ") <= ("
                     unsigned-index-type ")(" step-source ")) break;"))
               (indent-lines (inc depth) (str index-name " += " step-source ";")))]
      [(str initializers
            (when (get-in operation [:attributes :unroll])
              (indent-lines depth "#pragma unroll"))
            (indent-lines depth loop-header)
            bindings body-source
            (emit-yield-assignments results (:values yield-op)
                                    (update body-context :names merge (:names result-context))
                                    (inc depth))
            checked-advance
            (indent-lines depth "}"))
       result-context])

    (record-kind? "PipelinedFor" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          index (:index operation)
          loop-context (add-value result-context index)
          loop-context (reduce (fn [ctx arg] (add-value ctx (:binding arg)))
                               loop-context (:iter-args operation))
          initializers
          (apply str
                 (map (fn [result arg]
                        (indent-lines depth
                                      (str (target-type
                                            (get-in result-context [:types (:id result)]))
                                           " " (get-in result-context [:names (:id result)])
                                           " = " (emit-scalar-value (:initial arg) context) ";")))
                      results (:iter-args operation)))
          initial-groups (:async-groups context)
          native-events? (= :native-events (c-dialect/async-copy-mode *scalar-dialect*))
          binding-groups
          (mapv
           (fn [arg group]
             (let [binding (:binding arg)
                   event-count (count (:copies group))
                   events (mapv #(str (scalar-local-name binding) "_event_" %)
                                (range event-count))]
               {:id binding :copies (:copies group) :events events}))
           (:async-iter-args operation) initial-groups)
          event-initializers
          (when native-events?
            (apply str
                   (mapcat
                    (fn [binding-group initial-group]
                      (map (fn [binding-event initial-event]
                             (indent-lines depth
                                           (str "event_t " binding-event " = " initial-event ";")))
                           (:events binding-group)
                           (async-group-events context initial-group)))
                    binding-groups initial-groups)))
          loop-context (assoc loop-context :async-groups binding-groups)
          index-name (get-in loop-context [:names (:id index)])
          scalar-bindings
          (apply str
                 (map (fn [arg result]
                        (let [binding (:binding arg)]
                          (indent-lines (inc depth)
                                        (str (target-type
                                              (get-in loop-context [:types (:id binding)]))
                                             " " (get-in loop-context [:names (:id binding)])
                                             " = " (get-in result-context [:names (:id result)])
                                             ";"))))
                      (:iter-args operation) results))
          body-operations (pop (:operations operation))
          yield-op (peek (:operations operation))
          [body-source body-context] (emit-scalar-operations body-operations loop-context
                                                             (inc depth))
          yielded-by-id (into {} (map (juxt :id identity)) (:async-groups body-context))
          yielded-groups (mapv yielded-by-id (:groups yield-op))
          next-event-names
          (when native-events?
            (mapv (fn [binding-group group]
                    (mapv (fn [event-index event]
                            [(str (scalar-local-name (:id binding-group))
                                  "_next_event_" event-index)
                             event])
                          (range) (async-group-events body-context group)))
                  binding-groups yielded-groups))
          event-backedge
          (when native-events?
            (str
             (apply str
                    (for [[temporary source-event] (mapcat identity next-event-names)]
                      (indent-lines (inc depth)
                                    (str "event_t " temporary " = " source-event ";"))))
             (apply str
                    (mapcat
                     (fn [binding-group temporaries]
                       (map (fn [binding-event [temporary _]]
                              (indent-lines (inc depth)
                                            (str binding-event " = " temporary ";")))
                            (:events binding-group) temporaries))
                     binding-groups next-event-names))))
          index-type (get-in loop-context [:types (:id index)])
          unsigned-index-type (c-dialect/unsigned-type-name *scalar-dialect* index-type)
          upper-source (emit-index-expression (:upper operation) (:names context))
          step-source (str (:step operation))
          loop-header (str "for (" (target-type index-type)
                           " " index-name " = "
                           (emit-index-expression (:lower operation) (:names context)) "; "
                           index-name " < "
                           upper-source ";) {")
          checked-advance
          (str (indent-lines
                (inc depth)
                (str "if ((" unsigned-index-type ")(" upper-source ") - ("
                     unsigned-index-type ")(" index-name ") <= ("
                     unsigned-index-type ")(" step-source ")) break;"))
               (indent-lines (inc depth) (str index-name " += " step-source ";")))
          output-groups
          (mapv (fn [result binding-group]
                  (assoc binding-group :id result))
                (:async-results operation) binding-groups)
          next-context (assoc result-context :async-groups output-groups
                              :async-issued [])]
      [(str initializers event-initializers
            (when (get-in operation [:attributes :unroll])
              (indent-lines depth "#pragma unroll"))
            (indent-lines depth loop-header)
            scalar-bindings body-source
            (emit-yield-assignments results (:values yield-op)
                                    (update body-context :names merge (:names result-context))
                                    (inc depth))
            event-backedge
            checked-advance
            (indent-lines depth "}"))
       next-context])

    (record-kind? "Collective" operation)
    (let [result (:result operation)
          next-context (add-value context result)
          input (emit-scalar-value (:input operation) context)
          result-name (get-in next-context [:names (:id result)])
          result-type (get-in next-context [:types (:id result)])
          width (:width operation)
          source
          (case (:kind operation)
            :broadcast
            (indent-lines depth
                          (str (target-type result-type) " " result-name " = "
                               (c-dialect/broadcast-expression
                                *scalar-dialect* input (:source-lane operation) width) ";"))

            :reduce
            (let [operator (intrinsics/canonical (:operator operation))
                  association (:association operation)]
              (if (c-dialect/opencl? *scalar-dialect*)
                (let [_ (when-not (= :implementation-defined association)
                          (throw
                           (ex-info
                            "OpenCL subgroup builtin cannot preserve an explicit reduction tree"
                            {:reason :kernel-body-opencl-collective-association
                             :association association})))
                      builtin (c-dialect/opencl-reduction-builtin operator)]
                  (when-not builtin
                    (throw (ex-info "OpenCL has no matching subgroup reduction builtin"
                                    {:reason :kernel-body-opencl-collective
                                     :operator (:operator operation)})))
                  (indent-lines depth
                                (str (target-type result-type) " " result-name " = "
                                     builtin "(" input ");")))
                (let [distances (if (= :implementation-defined association)
                                  (when (zero? (bit-and width (dec width)))
                                    (vec (take-while pos?
                                                     (iterate #(quot % 2) (quot width 2)))))
                                  (:distances association))
                      _ (when-not (seq distances)
                          (throw (ex-info "CUDA/HIP subgroup reduction requires a power-of-two tree"
                                          {:reason :kernel-body-c-collective-association
                                           :dialect (:id *scalar-dialect*)
                                           :width width :association association})))
                      combine (fn [rhs]
                                (case operator
                                  :+ (str result-name " += " rhs ";")
                                  :* (str result-name " *= " rhs ";")
                                  :bit-and (str result-name " &= " rhs ";")
                                  :bit-or (str result-name " |= " rhs ";")
                                  :bit-xor (str result-name " ^= " rhs ";")
                                  (:min :max)
                                  (let [fn-name (case [operator result-type]
                                                  [:min :float] "fminf"
                                                  [:max :float] "fmaxf"
                                                  [:min :double] "fmin"
                                                  [:max :double] "fmax"
                                                  nil)]
                                    (when-not fn-name
                                      (throw (ex-info "CUDA/HIP min/max collective dtype is unsupported"
                                                      {:reason :kernel-body-c-collective
                                                       :dialect (:id *scalar-dialect*)
                                                       :operator operator :dtype result-type})))
                                    (str result-name " = " fn-name "(" result-name ", " rhs ");"))
                                  (throw (ex-info "CUDA/HIP has no matching subgroup reduction"
                                                  {:reason :kernel-body-c-collective
                                                   :dialect (:id *scalar-dialect*)
                                                   :operator operator}))))]
                  (str (indent-lines depth
                                     (str (target-type result-type) " " result-name " = " input ";"))
                       (apply str
                              (for [distance distances]
                                (indent-lines depth
                                              (combine
                                               (c-dialect/shuffle-down-expression
                                                *scalar-dialect* result-name distance width)))))
                       (indent-lines depth
                                     (str result-name " = "
                                          (c-dialect/broadcast-expression
                                           *scalar-dialect* result-name 0 width) ";")))))))]
      [source next-context])

    (record-kind? "WorkgroupBarrier" operation)
    [(indent-lines depth (c-dialect/workgroup-barrier *scalar-dialect*)) context]

    (record-kind? "AsyncWorkgroupCopy" operation)
    (let [destination-storage (get-in context [:storage (:destination operation)])
          layout-aware? (and (layout/shared-memory-layout? (:layout destination-storage))
                             (not= :identity (get-in destination-storage [:layout :swizzle])))
          mode (if layout-aware?
                 :synchronous-layout-aware
                 (c-dialect/async-copy-mode *scalar-dialect*))
          _ (when (and (= :required (:overlap operation))
                       (contains? #{:synchronous-cooperative :synchronous-layout-aware} mode))
              (throw (ex-info "target cannot preserve required async-copy overlap"
                              {:reason :kernel-body-c-async-overlap
                               :dialect (:id *scalar-dialect*)
                               :operation operation :lowering mode})))
          source-storage (get-in context [:storage (:source operation)])
          source-base (get-in context [:names (or (some-> source-storage :view :buffer)
                                                  (:id source-storage))])
          destination-base (get-in context [:names (:id destination-storage)])
          source-index (emit-storage-index source-storage (:source-coordinates operation)
                                           (:names context))
          destination-index (emit-storage-index destination-storage
                                                (:destination-coordinates operation)
                                                (:names context))
          copy-name (scalar-local-name (:id operation))
          source (str "(&" source-base "[" source-index "])")
          destination (str "(&" destination-base "[" destination-index "])")
          source-code (if layout-aware?
                        (emit-layout-aware-cooperative-copy
                         operation context source-storage destination-storage
                         source-base destination-base copy-name)
                        (c-dialect/async-copy
                         *scalar-dialect*
                         {:name copy-name :source source :destination destination
                          :elements (:elements operation)
                          :element-bytes (dtype/bytes-of (:dtype source-storage))
                          :transfer-bytes (:transfer-bytes operation)
                          :overlap (:overlap operation)
                          :workgroup-width (:workgroup-width context)}))]
      [(indent-lines depth source-code)
       (-> context
           (assoc-in [:async-copies (:id operation)] (when-not layout-aware? copy-name))
           (update :async-issued conj (:id operation)))])

    (record-kind? "AsyncCommit" operation)
    [(indent-lines depth (c-dialect/async-commit *scalar-dialect* (:id operation)))
     (-> context
         (assoc :async-issued [])
         (update :async-groups conj {:id (:id operation) :copies (:copies operation)}))]

    (record-kind? "AsyncWait" operation)
    (let [group-count (count (:groups operation))
          waited (subvec (:async-groups context) 0 group-count)
          event-names (mapv identity (mapcat #(async-group-events context %) waited))]
      [(indent-lines depth
                     (c-dialect/async-wait *scalar-dialect* event-names
                                           (:pending-groups operation)))
       (assoc context :async-groups (subvec (:async-groups context) group-count))])

    (record-kind? "Yield" operation)
    (throw (ex-info "OpenCL scalar lowering found a misplaced Yield"
                    {:reason :raster/bug :operation operation}))

    (record-kind? "PipelineYield" operation)
    (throw (ex-info "OpenCL scalar lowering found a misplaced PipelineYield"
                    {:reason :raster/bug :operation operation}))

    :else
    (throw (ex-info "OpenCL scalar lowering found an unsupported KernelBody operation"
                    {:reason :kernel-body-opencl-unimplemented :operation operation}))))

(defn- emit-scalar-operations
  [operations context depth]
  (reduce (fn [[source ctx] operation]
            (let [[next-source next-context] (emit-scalar-operation operation ctx depth)]
              [(str source next-source) next-context]))
          ["" context] operations))

(defn- scalar-storage-map
  [kernel-body]
  (let [parameters (into {} (map (juxt :id identity)) (:parameters kernel-body))
        allocations (into {}
                          (map (fn [allocation]
                                 [(:id allocation)
                                  (assoc allocation :kind :allocation
                                         :memory-space :workgroup)]))
                          (:allocations kernel-body))]
    (reduce (fn [storage view]
              (let [parent (get parameters (:buffer view))]
                (assoc storage (:id view)
                       (assoc parent :id (:id view) :shape (:shape view)
                              :layout (:layout view) :view view))))
            (merge parameters allocations) (:views kernel-body))))

(defn- emit-scalar-kernel*
  [kernel-name kernel-body {:keys [parameter-names]}]
  (let [kernel-body (body/validate! kernel-body)
        operations (vec (scalar-body-operations (:operations kernel-body)))
        unsupported (remove #(contains? scalar-operation-kinds
                                        (some-> % class .getSimpleName))
                            operations)
        _ (when (seq (:fragments kernel-body))
            (throw (ex-info "scalar OpenCL lowering cannot consume matrix fragments"
                            {:reason :kernel-body-opencl-unimplemented
                             :fragments (:fragments kernel-body)})))
        _ (when (seq unsupported)
            (throw (ex-info "scalar OpenCL lowering cannot consume legacy tile operations"
                            {:reason :kernel-body-opencl-unimplemented
                             :operations (vec unsupported)})))
        parameters (:parameters kernel-body)
        parameter-names (into {}
                              (map (fn [parameter]
                                     [(:id parameter)
                                      (or (get parameter-names (:id parameter))
                                          (scalar-local-name (:id parameter)))])
                                   parameters))
        allocation-names (into {}
                               (map (fn [allocation]
                                      [(:id allocation) (scalar-local-name (:id allocation))]))
                               (:allocations kernel-body))
        _ (when-not (every? #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" %)
                            (vals parameter-names))
            (throw (ex-info "C-family kernel parameter names must be C identifiers"
                            {:reason :kernel-body-c-parameter-name
                             :dialect (:id *scalar-dialect*)
                             :parameter-names parameter-names})))
        indices (:indices kernel-body)
        names (reduce (fn [env index]
                        (assoc env (:id index) (scalar-local-name (:id index))))
                      (merge parameter-names allocation-names) indices)
        types (reduce (fn [env index]
                        (assoc env (:id index)
                               (if (record-kind? "IndexBinding" index)
                                 :int
                                 (index-expression-dtype (:expression index) env))))
                      (into {}
                            (map (fn [parameter]
                                   [(:id parameter) (dtype/canon (:dtype parameter))])
                                 (filter #(= :scalar (:kind %)) parameters)))
                      indices)
        all-names (vals names)
        _ (when-not (= (count all-names) (count (set all-names)))
            (throw (ex-info "C-family parameter and index names collide after target spelling"
                            {:reason :kernel-body-c-name-collision
                             :dialect (:id *scalar-dialect*) :names names})))
        local-ids (concat (map :id (:allocations kernel-body))
                          (map :id indices) (scalar-defined-ids (:operations kernel-body))
                          (keep #(when (contains? #{"AsyncWorkgroupCopy" "AsyncCommit"}
                                                  (some-> % class .getSimpleName))
                                   (:id %))
                                operations))
        emitted-local-names (map scalar-local-name local-ids)
        all-emitted-names (concat (vals parameter-names) emitted-local-names)
        _ (when-not (= (count all-emitted-names) (count (set all-emitted-names)))
            (throw (ex-info "C-family parameter or SSA names collide after target spelling"
                            {:reason :kernel-body-c-name-collision
                             :dialect (:id *scalar-dialect*)
                             :ids (vec local-ids)})))
        storage (scalar-storage-map kernel-body)
        masks (into {} (map (juxt :id identity)) (:masks kernel-body))
        context {:names names :types types :storage storage :masks masks
                 :workgroup-shape (get-in kernel-body [:launch :workgroup-size])
                 :workgroup-width (reduce * (get-in kernel-body [:launch :workgroup-size]))
                 :async-copies {} :async-issued [] :async-groups []}
        allocation-plan (body/workgroup-memory-plan (:allocations kernel-body))
        arena-name "rstr_workgroup_memory"
        allocation-source
        (when (seq (:allocations kernel-body))
          (str (indent-lines
                1 (c-dialect/workgroup-arena-declaration
                   *scalar-dialect* arena-name (:bytes allocation-plan)
                   (:alignment allocation-plan)))
               (apply str
                      (for [{:keys [allocation byte-offset]} (:allocations allocation-plan)]
                        (indent-lines
                         1 (c-dialect/workgroup-pointer-declaration
                            *scalar-dialect* (:dtype allocation)
                            (get allocation-names (:id allocation)) arena-name byte-offset))))))
        index-source
        (apply str
               (for [index indices]
                 (let [name (get names (:id index))]
                   (if (record-kind? "IndexBinding" index)
                     (indent-lines
                      1
                      (str "int " name " = (int)"
                           (c-dialect/index-binding
                            *scalar-dialect* (:source index) (:axis index)
                            (get-in kernel-body [:schedule :subgroup-size]))
                           ";"))
                     (indent-lines
                      1 (str (target-type (get types (:id index))) " " name " = "
                             (emit-index-expression (:expression index) names) ";"))))))
        [operation-source _] (emit-scalar-operations (:operations kernel-body) context 1)
        helper-source
        (c-dialect/helper-source
         *scalar-dialect*
         (str (apply str
                     (map (fn [[operation type]]
                            (c-dialect/trapping-arithmetic-helper-source
                             *scalar-dialect* operation type))
                          (trapping-arithmetic-requirements operations)))
              (when (and (c-dialect/opencl? *scalar-dialect*)
                         (str/includes? operation-source "atomic_add_float("))
                ce/opencl-atomic-add-float-helper)
              (ce/intrinsic-helper-sources operation-source
                                           (:id *scalar-dialect*))))
        storage-declarations (concat parameters (:allocations kernel-body))
        stable-reads (set (map :buffer (:stable-reads kernel-body)))
        uses-half? (some #(= :half (dtype/canon (:dtype %))) storage-declarations)
        uses-double? (some #(= :double (dtype/canon (:dtype %))) storage-declarations)
        collective (first (filter #(record-kind? "Collective" %) operations))
        uses-subgroups? (or collective
                            (some #(and (record-kind? "IndexBinding" %)
                                        (contains? #{:subgroup :lane} (:source %)))
                                  indices))
        subgroup-size (or (some-> collective :width)
                          (get-in kernel-body [:schedule :subgroup-size]))
        attribute (c-dialect/subgroup-attribute
                   *scalar-dialect* subgroup-size uses-subgroups?)]
    (str (c-dialect/preamble *scalar-dialect*
                             {:uses-half? uses-half? :uses-double? uses-double?
                              :uses-subgroups? uses-subgroups?})
         helper-source
         (when (seq helper-source) "\n")
         attribute
         (c-dialect/entry-prefix *scalar-dialect*) (target-name kernel-name) "(\n    "
         (str/join ",\n    "
                   (map #(c-dialect/parameter-declaration
                          *scalar-dialect*
                          (cond-> % (contains? stable-reads (:id %))
                                  (assoc :restrict? true))
                          (get parameter-names (:id %)))
                        parameters))
         ") {\n"
         allocation-source index-source operation-source
         "}\n")))

(defn emit-scalar-kernel
  "Lower a verified scalar/control KernelBody directly to a C-family target dialect.

  The body already fixes schedules, types, layouts, masks, convergence and numerical conversion
  policies. `parameter-names` may only select ABI spelling; `target-dialect` defaults to the
  production Intel OpenCL row and may select `:opencl-portable`, `:cuda`, or `:hip`. Target
  selection cannot alter the scheduled body. `target-features` may only select a capability-gated
  physical implementation (for example Ampere `cp.async`) with the same verified semantics."
  ([kernel-name kernel-body]
   (emit-scalar-kernel kernel-name kernel-body {}))
  ([kernel-name kernel-body {:keys [target-dialect subgroup-attribute target-features] :as options
                             :or {target-dialect :opencl-intel subgroup-attribute :intel}}]
   (let [target-dialect (if (and (= :opencl-intel target-dialect)
                                 (= :none subgroup-attribute))
                          :opencl-portable
                          target-dialect)]
     (binding [*scalar-dialect*
               (merge (c-dialect/resolve! target-dialect)
                      (select-keys target-features [:compute-capability :architecture]))]
       (emit-scalar-kernel* kernel-name kernel-body options)))))

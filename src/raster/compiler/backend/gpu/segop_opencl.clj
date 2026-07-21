(ns raster.compiler.backend.gpu.segop-opencl
  "OpenCL kernel generation from SegOp IR.

   Translates SegMap and SegRed records into OpenCL C source strings.
   Uses the pre-computed inputs/outputs/scalars from SegOp lowering
   instead of re-analyzing par forms.

   This is the GPU counterpart to segop_simd.clj — both consume the
   same SegOp IR but produce different target code."
  (:require [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.backend.intrinsics :as intrinsics]
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
        ;; Use pre-computed inputs/outputs/scalars from SegOp.
        ;; :outputs may carry SECONDARY outputs beyond `out-sym` — the side-effect
        ;; aset targets of a horizontally-fused multi-output map. Those are array
        ;; params too (declared NON-const, appended after the inputs so the invoke's
        ;; positional arg order matches the C signature); an input that is also
        ;; written (read+write buffer) likewise loses const.
        written (set (map #(symbol (name %)) (:outputs segmap)))
        input-params (vec (sort-by name (:inputs segmap)))
        input-name-set (set (map #(symbol (name %)) input-params))
        out-name (when out-sym (symbol (name out-sym)))
        extra-outs (vec (sort-by name
                                 (remove #(or (= % out-name)
                                              (contains? input-name-set %))
                                         written)))
        arr-params (into input-params extra-outs)
        scl-params (vec (sort-by name (:scalars segmap)))
        arr-dtype (fn [s] (get array-types s (get array-types (symbol (name s)) dtype)))
        arr-type (fn [s] (get codegen/opencl-type-map (arr-dtype s) default-ctype))
        written-params (filterv #(contains? written (symbol (name %))) arr-params)
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global "
                                                  (when-not (contains? written (symbol (name s)))
                                                    "const ")
                                                  (arr-type s) "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        ;; Integer scalar params seed *int-vars* so index math stays integer
        int-scalar-syms (set (keep (fn [[k v]] (when (= v :int) (symbol (name k)))) scalar-types))
        scl-type (fn [s] (ce/scalar-native-type s scalar-types default-ctype))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param (str "__global " out-ctype "* restrict out")
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param scl-param-str "int _n_bound"]))
        ;; Emit body as C expression
        adapted-body (ce/adapt-casts-for-dtype body out-dtype)
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* out-ctype
                           ce/*idx-sym* idx
                           ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                   (ce/emit-expr adapted-body idx arr-sym-set))
        cast-str (if cast-fn (str "(" (name cast-fn) ")(" body-str ")") body-str)
        scalar-body-str (str "out[idx] = " cast-str ";")
        ;; Affine-index vectorization (shared c_emit): a SegMap store is `out[idx] = f(..)`,
        ;; expressed here as the synthetic aset the vectorizer analyzes. The store target
        ;; is the literal `out` param (not c-symbol-mangled), so pass :store-name. nil ⇒
        ;; scalar loop.
        loop-region (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* out-ctype
                              ce/*idx-sym* idx
                              ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                      (ce/emit-vectorized-elementwise-loop
                       (list 'aset 'out idx (if cast-fn (list cast-fn adapted-body) adapted-body))
                       idx (conj arr-sym-set 'out) "idx" scalar-body-str
                       {:n-bound "_n_bound" :store-name "out"}))
        ;; pragmas cover the output dtype AND every input array's dtype
        source (str (apply codegen/extension-pragmas out-dtype (map arr-dtype arr-params))
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    "
                    (or loop-region
                        (str "for (int idx = get_global_id(0); idx < _n_bound; idx += get_global_size(0)) {\n"
                             "        " scalar-body-str "\n"
                             "    }"))
                    "\n}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     ;; array params (by sig name) the kernel WRITES — secondary fused outputs and
     ;; read+write inputs. The staging invoke copies these back to their JVM arrays
     ;; after launch; the resident role-derivation marks written PARAMS :output.
     :written-arrays written-params
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
        ;; #55 fix: normalize devirtualized array prims ((.invk aget-impl arr i)
        ;; → canonical aget head) BEFORE any rewrapping, exactly as SegMap does.
        ;; Without it, a parametric-array kernel (qlinear-k) emitted broken
        ;; gpufn_aget helper calls while a typed-array kernel (decoder-gpu)
        ;; emitted x[i] — same op, ns-sensitive lowering.
        lambda (ce/normalize-array-prims lambda)
        dtype (or (:dtype segred) dtype)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        ctype (get codegen/opencl-type-map dtype "double")
        arr-params (vec (sort-by name (:inputs segred)))
        scl-params (vec (sort-by name (:scalars segred)))
        ;; Detect reduction op from lambda — unwrap let to find op, keep let for elem
        [let-bindings inner-body]
        (if (and (seq? lambda) (contains? #{'let* 'let} (first lambda)))
          (let [[_ binds & bdy] lambda]
            ;; A reduce combine's let body must be ONE expression `(op acc elem)`. Taking
            ;; `(last bdy)` of a multi-statement body would SILENTLY DROP the earlier forms
            ;; — the same shape as the single-aset-void store-drop. If earlier statements
            ;; exist they carry computation/effects the combine depends on; reject loudly.
            (when (> (count bdy) 1)
              (throw (ex-info (str "SegRed: reduce combine lambda has a multi-statement body ("
                                   (count bdy) " forms) — only a single combine expression is"
                                   " modeled; earlier forms would be dropped")
                              {:lambda lambda :body (vec bdy)})))
            [(vec (partition 2 binds)) (last bdy)])
          [nil lambda])
        ;; .invk-aware: the walker devirtualizes (raster.numeric/+ acc x) into
        ;; (.invk _plus_impl acc x) with :raster.op/original metadata. semantic-op recovers the
        ;; original op and call-args the real operands — never parse the mangled impl name (which
        ;; would mis-detect the op and capture the impl symbol as the element). Same fix #37 made
        ;; for SegMap; here it keeps SegRed combine-op detection sound for both bare and .invk forms.
        op-sym (when (seq? inner-body) (descriptor/semantic-op inner-body))
        normalized-op (get {'clojure.core/+ '+, 'clojure.core/* '*,
                            'raster.numeric/+ '+, 'raster.numeric/* '*,
                            'clojure.core/max 'max, 'raster.numeric/max 'max, 'Math/max 'max,
                            'clojure.core/min 'min, 'raster.numeric/min 'min, 'Math/min 'min}
                           op-sym op-sym)
        ;; Unknown combine ops must FAIL LOUD — the old default silently combined with "+"
        ;; (a max reduce summed the per-lane maxima). Only associative ops are legal here.
        c-op (condp = normalized-op '+ "+" '* "*" 'max "fmax" 'min "fmin"
                    (throw (ex-info (str "SegRed: unsupported reduce combine op " op-sym
                                         " — GPU reduction needs an associative op (+ * max min)")
                                    {:op op-sym :lambda lambda})))
        c-identity-val ({"+" "0.0" "*" "1.0" "fmax" "-INFINITY" "fmin" "INFINITY"} c-op "0.0")
        identity-val ({"+" 0.0 "*" 1.0 "fmax" Double/NEGATIVE_INFINITY "fmin" Double/POSITIVE_INFINITY} c-op 0.0)
        ;; fmax/fmin are functions, not infix operators
        c-combine (fn [a b] (if (#{"fmax" "fmin"} c-op)
                              (str c-op "(" a ", " b ")")
                              (str "(" a " " c-op " " b ")")))
        ;; Extract the element expression (the non-acc operand) from the SEMANTIC args.
        op-args (vec (when (seq? inner-body) (descriptor/call-args inner-body)))
        ;; A segmented reduce combine is BINARY: (op acc elem). A variadic combine like
        ;; (+ acc x y) has 3 operands — extracting only ONE non-acc operand would SILENTLY
        ;; emit a kernel that sums just `x`, dropping `y` (the store-drop family). A legit
        ;; fused map→reduce nests the map body as a single elem operand, so >2 is unmodeled.
        _ (when (> (count op-args) 2)
            (throw (ex-info (str "SegRed: reduce combine op has " (count op-args)
                                 " operands — only a binary (op acc elem) combine is modeled;"
                                 " extra operands would be dropped")
                            {:op op-sym :op-args op-args :lambda lambda})))
        acc-at? (fn [a] (or (= a acc)
                            (and (seq? a) (= 'double (first a)) (= acc (second a)))))
        [_acc-pos elem-expr-raw]
        (when (>= (count op-args) 2)
          (let [a0 (nth op-args 0) a1 (nth op-args 1)]
            (cond
              (acc-at? a0) [:left a1]
              (acc-at? a1) [:right a0]
              :else [nil nil])))
        ;; Re-wrap in let if there were bindings (preserves local variable scope)
        ;; preserve the raw expr's metadata across the rewrap — dropping it
        ;; severed :raster.op/original on .invk forms (part of #55)
        elem-expr (if (and elem-expr-raw (seq let-bindings))
                    (with-meta (list 'let* (vec (mapcat identity let-bindings)) elem-expr-raw)
                      (meta elem-expr-raw))
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
                 (str (codegen/extension-pragmas dtype)
                      "#if defined(cl_khr_subgroups)\n#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n#elif defined(cl_intel_subgroups)\n#pragma OPENCL EXTENSION cl_intel_subgroups : enable\n#endif\n"
                      "__kernel void " kernel-name
                      "(" all-params ") {\n"
                      "    __local " ctype " sdata[256];\n"
                      "    int tid = get_local_id(0);\n"
                      "    " ctype " val = " c-identity-val ";\n"
                      "    int stride = get_global_size(0);\n"
                      "    int " (ce/c-symbol idx) " = get_global_id(0);\n"
                      "    for (; " (ce/c-symbol idx) " < _n_bound; " (ce/c-symbol idx) " += stride) {\n"
                      "        val = " (c-combine "val" elem-str) ";\n"
                      "    }\n"
                      "    sdata[tid] = val;\n"
                      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                      "    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
                      "        if (tid < s) {\n"
                      "            sdata[tid] = " (c-combine "sdata[tid]" "sdata[tid + s]") ";\n"
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

;; ================================================================
;; Segmented reduction (contraction) → OpenCL — the multi-axis SegSpace path
;; ================================================================

(defn generate-segmented-reduce-kernel
  "NAIVE segmented reduction (a contraction) → OpenCL. One work-item per SEGMENT (free-axis
   tuple); each sequentially folds over the reduced (innermost) axis. segment-dims = the
   free/parallel axes, reduced-dim = the contracted axis (Futhark innermost-reduced
   convention). Emits: decompose the flat segment id into the free indices (row-major via
   suffix products), loop over the reduced axis accumulating the combine's element (the
   product), store out[seg]. This is the multi-axis SegSpace emit path the 1-D emitters
   never handled; NAIVE (one thread/output) — the substrate the BlkRegTiling-style tiling
   pass optimizes. Combine op/element detection mirrors generate-segred-kernel (a shared
   helper is a later dedup). Bounds must be symbols or int literals in this prototype."
  [segred out-sym & {:keys [dtype kernel-name-prefix]
                     :or {dtype :double kernel-name-prefix "contract"}}]
  (let [space    (:space segred)
        seg-dims (segop/seg-space-segment-dims space)   ; free (parallel) axes, outer→inner
        red-dim  (segop/seg-space-reduced-dim space)    ; contracted axis (innermost)
        _ (when (empty? seg-dims)
            (throw (ex-info "segmented-reduce: no segment dims — use generate-segred-kernel for a full reduction"
                            {:space space})))
        dtype (or (:dtype segred) dtype)
        ctype (get codegen/opencl-type-map dtype "double")
        {:keys [acc init lambda]} (:reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        ;; combine op + element detection (mirrors generate-segred-kernel)
        op-sym (when (seq? lambda) (descriptor/semantic-op lambda))
        normalized-op (get {'+ '+ 'clojure.core/+ '+ 'raster.numeric/+ '+
                            '* '* 'clojure.core/* '* 'raster.numeric/* '*
                            'max 'max 'clojure.core/max 'max 'Math/max 'max 'raster.numeric/max 'max
                            'min 'min 'clojure.core/min 'min 'Math/min 'min 'raster.numeric/min 'min}
                           op-sym op-sym)
        c-op (condp = normalized-op '+ "+" '* "*" 'max "fmax" 'min "fmin"
                    (throw (ex-info (str "segmented-reduce: unsupported combine op " op-sym
                                         " — need an associative op (+ * max min)")
                                    {:op op-sym :lambda lambda})))
        c-combine (fn [a b] (if (#{"fmax" "fmin"} c-op)
                              (str c-op "(" a ", " b ")")
                              (str "(" a " " c-op " " b ")")))
        op-args (vec (when (seq? lambda) (descriptor/call-args lambda)))
        acc-at? (fn [a] (or (= a acc) (and (seq? a) (= 'double (first a)) (= acc (second a)))))
        elem-expr (when (>= (count op-args) 2)
                    (let [a0 (nth op-args 0) a1 (nth op-args 1)]
                      (cond (acc-at? a0) a1 (acc-at? a1) a0 :else nil)))
        _ (when (nil? elem-expr)
            (throw (ex-info "segmented-reduce: could not isolate the element (non-acc) operand"
                            {:lambda lambda :acc acc})))
        ;; params
        arr-params (vec (sort-by name (:inputs segred)))
        scl-params (vec (sort-by name (:scalars segred)))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        bound-c (fn [b] (cond (symbol? b) (ce/c-symbol b)
                              (number? b) (str b)
                              :else (throw (ex-info "segmented-reduce: bound must be a symbol or int in this prototype"
                                                    {:bound b}))))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        scl-param-str (str/join ", " (map (fn [s] (str "int " (ce/c-symbol s))) scl-params))
        ;; Trailing `int _nseg` (= number of segments = launch count) matches the generic
        ;; emitter convention (generate-segmap-kernel's `int _n_bound`) and the arg order
        ;; invoke-registered-kernel builds (inputs, output, scalars, count).
        all-params (str/join ", " (remove empty?
                                          [arr-param-str (str "__global " ctype "* restrict out")
                                           scl-param-str "int _nseg"]))
        seg-bound-cs (mapv (fn [d] (bound-c (:bound d))) seg-dims)
        ;; row-major decompose: idx_p = (seg / product(bounds after p)) % bound_p
        decomp (str/join "\n"
                         (map-indexed
                          (fn [p d]
                            (let [after (drop (inc p) seg-bound-cs)
                                  div (if (seq after) (str "(seg / (" (str/join " * " after) "))") "seg")
                                  rhs (if (seq after) (str div " % " (bound-c (:bound d))) (str "seg % " (bound-c (:bound d))))]
                              (str "    int " (ce/c-symbol (:name d)) " = " rhs ";")))
                          seg-dims))
        ;; ce/emit-expr renders the primary index sym as the C var "idx" (its convention);
        ;; the reduced-axis loop var MUST therefore be named "idx" so the body's references
        ;; to the reduced index resolve. Free indices render by their own c-symbol names.
        red-bound-c (bound-c (:bound red-dim))
        int-vars (into #{} (map #(symbol (name %)))
                       (concat (map :name seg-dims) [(:name red-dim)] scl-params))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        elem-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr (ce/adapt-casts-for-dtype elem-expr dtype) (:name red-dim) arr-sym-set))
        source (str (codegen/extension-pragmas dtype)
                    "__kernel void " kernel-name "(" all-params ") {\n"
                    "    int seg = get_global_id(0);\n"
                    "    if (seg >= _nseg) return;\n"
                    decomp "\n"
                    "    " ctype " acc = " (str init) ";\n"
                    "    for (int idx = 0; idx < " red-bound-c "; idx++) {\n"
                    "        acc = " (c-combine "acc" elem-str) ";\n"
                    "    }\n"
                    "    out[seg] = acc;\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :dtype dtype
     :c-op c-op}))

(defn generate-segmap-nd-kernel
  "N-D pure map → OpenCL: an OUTER PRODUCT / broadcast / elementwise contraction with ZERO
   contract axes. One work-item per output element; decompose the flat index into the free
   indices (row-major, outer→inner) and write out[seg] = body. This is the empty-reduce
   projection of a contraction — the SegMap counterpart of generate-segmented-reduce-kernel.
   The SegMap's space dims are ALL free/output axes (no reduced dim). Trailing `int _nseg`
   count param (matches the generic emitter convention)."
  [segmap out-sym & {:keys [dtype] :or {dtype :double}}]
  (let [dims (segop/seg-space-dims (:space segmap))   ; all free (no reduced dim)
        dtype (or (:dtype segmap) dtype)
        ctype (get codegen/opencl-type-map dtype "double")
        body (ce/normalize-array-prims (:lambda segmap))
        arr-params (vec (sort-by name (:inputs segmap)))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        kernel-name (str "segmap_nd_" (gensym ""))
        bound-c (fn [b] (cond (symbol? b) (ce/c-symbol b)
                              (number? b) (str b)
                              :else (throw (ex-info "segmap-nd: bound must be symbol or int" {:bound b}))))
        dim-cs (mapv (fn [d] (bound-c (:bound d))) dims)
        decomp (str/join "\n"
                         (map-indexed
                          (fn [p d]
                            (let [after (drop (inc p) dim-cs)
                                  rhs (if (seq after)
                                        (str "(seg / (" (str/join " * " after) ")) % " (bound-c (:bound d)))
                                        (str "seg % " (bound-c (:bound d))))]
                              (str "    int " (ce/c-symbol (:name d)) " = " rhs ";")))
                          dims))
        int-vars (into #{} (map #(symbol (name (:name %)))) dims)
        dummy (gensym "z__")
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr (ce/adapt-casts-for-dtype body dtype) dummy arr-sym-set))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        src (str (codegen/extension-pragmas dtype)
                 "__kernel void " kernel-name "(" arr-param-str ", __global " ctype "* restrict out, int _nseg) {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 decomp "\n"
                 "    out[seg] = " body-str ";\n"
                 "}\n")]
    {:kernel-name kernel-name :source src :array-params arr-params :dtype dtype}))

;; ================================================================
;; Block-tiled + __local-staged contraction (BlkRegTiling, block-tile level)
;; ================================================================

(defn- syms-in [expr] (set (filter symbol? (tree-seq coll? seq expr))))

(defn generate-tiled-contraction-kernel
  "BLOCK-TILED + __local-staged contraction → OpenCL (Futhark BlkRegTiling, block-tile
   level — no register/DPAS tiling yet). Square T×T tiles; workgroup = T×T threads (one
   output/thread); cooperatively stage A/B tiles into __local, loop the contracted axis in
   T-chunks. Zero-padded loads + a guarded store handle non-tile-divisible dims.

   Prototype scope: EXACTLY 2 free axes (the tiled M×N output) + 1 contracted axis, LITERAL
   dims, a sum-of-two-agets element (GEMM-shape redomap). The operands are assigned row/col
   by which DECLARED axis their index depends on (row: free0+contract, not free1; col:
   contract+free1, not free0) — the variance test made trivial by the declared axes, no
   recognition. This is the perf substrate under DPAS tensorize (step 4).

   Requires a 2-D launch: workgroup [T T], grid [ceil(N/T) ceil(M/T)] (group-id(0)=col/N,
   group-id(1)=row/M). Returns {:kernel-name :source :array-params :dtype :tile :dims [M N L]}."
  [segred out-sym & {:keys [dtype tile] :or {dtype :double tile 16}}]
  (let [space    (:space segred)
        seg-dims (segop/seg-space-segment-dims space)
        red-dim  (segop/seg-space-reduced-dim space)
        _ (assert (= 2 (count seg-dims)) "tiled-contraction: exactly 2 free axes (prototype)")
        [fi fj] seg-dims
        M (:bound fi) N (:bound fj) L (:bound red-dim)
        _ (assert (every? number? [M N L]) "tiled-contraction: literal dims (prototype)")
        i-sym (:name fi) j-sym (:name fj) l-sym (:name red-dim)
        dtype (or (:dtype segred) dtype)
        ctype (get codegen/opencl-type-map dtype "double")
        {:keys [init lambda]} (:reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        op-sym (when (seq? lambda) (descriptor/semantic-op lambda))
        _ (assert (#{'+ 'clojure.core/+ 'raster.numeric/+} op-sym)
                  "tiled-contraction: combine must be + / sum-of-products (prototype)")
        acc-sym (:acc (:reduce-op segred))
        acc-at? (fn [a] (or (= a acc-sym) (and (seq? a) (= 'double (first a)) (= acc-sym (second a)))))
        op-args (vec (descriptor/call-args lambda))
        elem (let [a0 (nth op-args 0) a1 (nth op-args 1)] (if (acc-at? a0) a1 a0))
        _ (assert (and (seq? elem) (#{'* 'clojure.core/* 'raster.numeric/*} (descriptor/semantic-op elem)))
                  "tiled-contraction: element must be a product of two agets (prototype)")
        parts (fn [e] (let [e (ce/normalize-array-prims e)]
                        (assert (and (seq? e) (= 'aget (first e))) "tiled-contraction: product operand must be an aget")
                        {:arr (nth e 1) :idx (nth e 2)}))
        [pa pb] (mapv parts (descriptor/call-args elem))
        dep? (fn [idx s] (contains? (syms-in idx) s))
        rc  (fn [x y] (when (and (dep? (:idx x) i-sym) (dep? (:idx x) l-sym) (not (dep? (:idx x) j-sym))
                                 (dep? (:idx y) l-sym) (dep? (:idx y) j-sym) (not (dep? (:idx y) i-sym)))
                        [x y]))
        [rowop colop] (or (rc pa pb) (rc pb pa)
                          (throw (ex-info "tiled-contraction: operands don't match A(i,l)·B(l,j) variance"
                                          {:pa pa :pb pb})))
        T (long tile)
        kernel-name (str "tiled_contract_" (gensym ""))
        arr-params (vec (sort-by name (:inputs segred)))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        dummy (gensym "z__")
        emit-load (fn [{:keys [arr idx]}]
                    (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* ctype
                              ce/*int-vars* (into ce/*int-vars* (map #(symbol (name %)) [i-sym j-sym l-sym]))]
                      (ce/emit-expr (list 'aget arr idx) dummy arr-sym-set)))
        row-load (emit-load rowop)   ; uses C vars i, l
        col-load (emit-load colop)   ; uses C vars l, j
        src (str (codegen/extension-pragmas dtype)
                 "__kernel void " kernel-name "(" arr-param-str
                 ", __global " ctype "* restrict out) {\n"
                 "    __local " ctype " As[" T "][" T "];\n"
                 "    __local " ctype " Bs[" T "][" T "];\n"
                 "    int li = get_local_id(1);\n"
                 "    int lj = get_local_id(0);\n"
                 "    int " (ce/c-symbol i-sym) " = get_group_id(1) * " T " + li;\n"
                 "    int " (ce/c-symbol j-sym) " = get_group_id(0) * " T " + lj;\n"
                 "    " ctype " acc = " (str init) ";\n"
                 "    for (int l0 = 0; l0 < " L "; l0 += " T ") {\n"
                 "        { int " (ce/c-symbol l-sym) " = l0 + lj; As[li][lj] = ((" (ce/c-symbol i-sym) " < " M ") && (" (ce/c-symbol l-sym) " < " L ")) ? " row-load " : 0.0; }\n"
                 "        { int " (ce/c-symbol l-sym) " = l0 + li; Bs[li][lj] = ((" (ce/c-symbol l-sym) " < " L ") && (" (ce/c-symbol j-sym) " < " N ")) ? " col-load " : 0.0; }\n"
                 "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                 "        for (int t = 0; t < " T "; t++) { acc = acc + As[li][t] * Bs[t][lj]; }\n"
                 "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                 "    }\n"
                 "    if ((" (ce/c-symbol i-sym) " < " M ") && (" (ce/c-symbol j-sym) " < " N ")) { out[" (ce/c-symbol i-sym) " * " N " + " (ce/c-symbol j-sym) "] = acc; }\n"
                 "}\n")]
    {:kernel-name kernel-name
     :source src
     :array-params arr-params
     :dtype dtype
     :tile T
     :dims [M N L]}))

;; ================================================================
;; Register-tiled + __local-staged contraction (BlkRegTiling, register level)
;; ================================================================

(defn- analyze-contraction
  "Shared structural analysis for the tiled contraction emitters. Prototype scope: 2 free
   axes + 1 contract, LITERAL dims, sum-of-two-agets element. Returns dims (M N L), axis
   syms, dtype/ctype, init, array params, and the row/col operand LOAD strings (row uses C
   vars i-sym,l-sym; col uses l-sym,j-sym — the caller declares them with the right values).
   Operands are assigned by DECLARED-axis dependence (no recognition). (generate-tiled-
   contraction-kernel predates this and still inlines the same logic — dedup TODO.)"
  [segred dtype]
  (let [space (:space segred)
        seg-dims (segop/seg-space-segment-dims space)
        red-dim  (segop/seg-space-reduced-dim space)
        _ (assert (= 2 (count seg-dims)) "tiled: exactly 2 free axes (prototype)")
        [fi fj] seg-dims
        M (:bound fi) N (:bound fj) L (:bound red-dim)
        _ (assert (every? number? [M N L]) "tiled: literal dims (prototype)")
        i-sym (:name fi) j-sym (:name fj) l-sym (:name red-dim)
        dtype (or (:dtype segred) dtype)
        ctype (get codegen/opencl-type-map dtype "double")
        {:keys [init lambda]} (:reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        _ (assert (#{'+ 'clojure.core/+ 'raster.numeric/+} (descriptor/semantic-op lambda))
                  "tiled: combine must be + (prototype)")
        acc-sym (:acc (:reduce-op segred))
        acc-at? (fn [a] (or (= a acc-sym) (and (seq? a) (= 'double (first a)) (= acc-sym (second a)))))
        op-args (vec (descriptor/call-args lambda))
        elem (let [a0 (nth op-args 0) a1 (nth op-args 1)] (if (acc-at? a0) a1 a0))
        _ (assert (and (seq? elem) (#{'* 'clojure.core/* 'raster.numeric/*} (descriptor/semantic-op elem)))
                  "tiled: element must be a product of two agets (prototype)")
        parts (fn [e] (let [e (ce/normalize-array-prims e)]
                        (assert (and (seq? e) (= 'aget (first e))) "tiled: operand must be an aget")
                        {:arr (nth e 1) :idx (nth e 2)}))
        [pa pb] (mapv parts (descriptor/call-args elem))
        dep? (fn [idx s] (contains? (syms-in idx) s))
        rc (fn [x y] (when (and (dep? (:idx x) i-sym) (dep? (:idx x) l-sym) (not (dep? (:idx x) j-sym))
                                (dep? (:idx y) l-sym) (dep? (:idx y) j-sym) (not (dep? (:idx y) i-sym)))
                       [x y]))
        [rowop colop] (or (rc pa pb) (rc pb pa)
                          (throw (ex-info "tiled: operands don't match A(i,l)·B(l,j) variance" {:pa pa :pb pb})))
        arr-params (vec (sort-by name (:inputs segred)))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        dummy (gensym "z__")
        emit-load (fn [{:keys [arr idx]}]
                    (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* ctype
                              ce/*int-vars* (into ce/*int-vars* (map #(symbol (name %)) [i-sym j-sym l-sym]))]
                      (ce/emit-expr (list 'aget arr idx) dummy arr-sym-set)))]
    {:M M :N N :L L :i-sym i-sym :j-sym j-sym :l-sym l-sym
     :dtype dtype :ctype ctype :init init :arr-params arr-params
     :row-load (emit-load rowop) :col-load (emit-load colop)
     ;; operand arrays + index exprs (for orientation analysis, e.g. DPAS tensorize):
     ;; rowop is the A(i,l) operand (→ A slot), colop the B(l,j) operand (→ B slot).
     :row-arr (:arr rowop) :row-idx (:idx rowop)
     :col-arr (:arr colop) :col-idx (:idx colop)}))

(defn generate-regtiled-contraction-kernel
  "REGISTER-TILED + __local-staged contraction → OpenCL (Futhark BlkRegTiling, register
   level). Block tile BM×BN over the output, BK contraction chunk; each thread owns a
   TM×TN register micro-tile of outputs (workgroup (BM/TM)×(BN/TN) threads). Cooperative
   flattened staging of A/B into __local, then a register-blocked inner MAC (acc[TM][TN] +=
   a[TM]·b[TN]). Zero-padded loads + guarded store handle non-divisible dims. The register
   tile is precisely the fragment a DPAS/tensorize step (step 4) will consume.

   Prototype scope as analyze-contraction. Requires a 2-D launch: workgroup [BN/TN BM/TM],
   grid [ceil(N/BN) ceil(M/BM)]. Returns {:kernel-name :source :array-params :dtype :block
   [BM BN BK] :micro [TM TN] :workgroup [x y] :dims [M N L]}."
  [segred out-sym & {:keys [dtype bm bn bk tm tn]
                     :or {dtype :double bm 64 bn 64 bk 16 tm 4 tn 4}}]
  (let [{:keys [M N L i-sym j-sym l-sym ctype init arr-params row-load col-load]}
        (analyze-contraction segred dtype)
        _ (assert (and (zero? (rem bm tm)) (zero? (rem bn tn)))
                  "regtiled: BM%TM and BN%TN must be 0")
        nt-row (quot bm tm) nt-col (quot bn tn) NT (* nt-row nt-col)
        i-c (ce/c-symbol i-sym) j-c (ce/c-symbol j-sym) l-c (ce/c-symbol l-sym)
        kernel-name (str "regtiled_contract_" (gensym ""))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        src (str (codegen/extension-pragmas (or (:dtype segred) dtype))
                 "__kernel void " kernel-name "(" arr-param-str ", __global " ctype "* restrict out) {\n"
                 "    __local " ctype " As[" bm "][" bk "];\n"
                 "    __local " ctype " Bs[" bk "][" bn "];\n"
                 "    int tr = get_local_id(1);\n"
                 "    int tc = get_local_id(0);\n"
                 "    int tid = tr * " nt-col " + tc;\n"
                 "    int block_i = get_group_id(1) * " bm ";\n"
                 "    int block_j = get_group_id(0) * " bn ";\n"
                 "    " ctype " acc[" tm "][" tn "];\n"
                 "    for (int m = 0; m < " tm "; m++) for (int n = 0; n < " tn "; n++) acc[m][n] = " (str init) ";\n"
                 "    for (int l0 = 0; l0 < " L "; l0 += " bk ") {\n"
                 "        for (int idx = tid; idx < " (* bm bk) "; idx += " NT ") {\n"
                 "            int r = idx / " bk "; int c = idx % " bk ";\n"
                 "            int " i-c " = block_i + r; int " l-c " = l0 + c;\n"
                 "            As[r][c] = ((" i-c " < " M ") && (" l-c " < " L ")) ? " row-load " : 0.0;\n"
                 "        }\n"
                 "        for (int idx = tid; idx < " (* bk bn) "; idx += " NT ") {\n"
                 "            int r = idx / " bn "; int c = idx % " bn ";\n"
                 "            int " l-c " = l0 + r; int " j-c " = block_j + c;\n"
                 "            Bs[r][c] = ((" l-c " < " L ") && (" j-c " < " N ")) ? " col-load " : 0.0;\n"
                 "        }\n"
                 "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                 "        for (int t = 0; t < " bk "; t++) {\n"
                 "            " ctype " a[" tm "]; " ctype " b[" tn "];\n"
                 "            for (int m = 0; m < " tm "; m++) a[m] = As[tr*" tm "+m][t];\n"
                 "            for (int n = 0; n < " tn "; n++) b[n] = Bs[t][tc*" tn "+n];\n"
                 "            for (int m = 0; m < " tm "; m++) for (int n = 0; n < " tn "; n++) acc[m][n] = acc[m][n] + a[m] * b[n];\n"
                 "        }\n"
                 "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                 "    }\n"
                 "    for (int m = 0; m < " tm "; m++) for (int n = 0; n < " tn "; n++) {\n"
                 "        int gr = block_i + tr*" tm " + m; int gc = block_j + tc*" tn " + n;\n"
                 "        if ((gr < " M ") && (gc < " N ")) out[gr * " N " + gc] = acc[m][n];\n"
                 "    }\n"
                 "}\n")]
    {:kernel-name kernel-name
     :source src
     :array-params arr-params
     :dtype (or (:dtype segred) dtype)
     :block [bm bn bk]
     :micro [tm tn]
     :workgroup [nt-col nt-row]
     :dims [M N L]}))

;; ================================================================
;; DPAS/XMX-tensorized contraction (PEAK) — raster's edge over Futhark
;; ================================================================

(defn- canonical-rowmajor?
  "Does affine index `idx` equal `(+ (* outer stride) inner)` (row-major, leading dim
   `stride`)? outer/inner are axis syms, stride the expected leading extent (a number).
   Order-agnostic on both the + and the *. Returns true/false."
  [idx outer stride inner]
  (let [idx (ce/normalize-array-prims idx)
        add? #(and (seq? %) (#{'+ 'clojure.core/+ 'raster.numeric/+} (first %)))
        mul? #(and (seq? %) (#{'* 'clojure.core/* 'raster.numeric/*} (first %)))]
    (boolean
     (when (and (add? idx) (= 3 (count idx)))
       (let [[_ a b] idx
             [prod other] (cond (mul? a) [a b] (mul? b) [b a] :else [nil nil])]
         (when (and prod (= other inner) (= 3 (count prod)))
           (let [[_ p q] prod]
             (or (and (= p outer) (= q stride))
                 (and (= q outer) (= p stride))))))))))

(defn dpas-contraction-legal?
  "The tensorize LEGALITY GATE: is `segred` a contraction that lowers to the DPAS/XMX
   matmul body? This is where the recognizer's affine-invariance core rehomes — but stated
   as a *legality check on already-declared axes*, not a recognition. Returns {:ok true …}
   with the extracted operands/dims, or {:ok false :reason kw} so the caller falls back to
   the portable register-tiled kernel.

   Conditions (Arc DPAS): exactly 2 free + 1 contract axis, element = product of two agets,
   BOTH operands in canonical row-major orientation (A[i,l]=i·L+l ⇒ [M,L]; B[l,j]=l·N+j ⇒
   [L,N]) so the golden 2D-block reads address them correctly, dtype ∈ DPAS types (Arc:
   half; bf16/int8 are future variants), and the operand PITCHES are 16-byte-aligned —
   N·2 and K·2 multiples of 16 ⇔ N%8==0 and K%8==0 for f16. The pitch condition is a HARD
   hardware constraint of intel_sub_group_2d_block_read: a mis-aligned pitch (e.g. N=70,
   K=124) SILENTLY MISCOMPILES ~80% of outputs (device-verified) — production GEMM shapes
   (N∈{640,1024,2048}) are all N%8==0 so never hit it, but a general contraction can, and
   the gate must reject it so the caller falls back to the register-tiled kernel (which
   handles arbitrary dims). M (the block-read HEIGHT) is unconstrained. Transposed operands
   (:tn/:nt) and a batch axis are legal extensions the golden body already has flags for —
   rejected here as :non-canonical-orientation / :not-a-contraction until wired."
  [segred dtype]
  (let [dtype (or (:dtype segred) dtype)
        pitch-ok? (fn [d] (and (number? d) (zero? (mod (long d) 8))))]
    (if-not (#{:half :float16} dtype)
      {:ok false :reason :dtype-not-dpas :dtype dtype}
      (try
        (let [{:keys [M N L i-sym j-sym l-sym row-idx col-idx row-arr col-arr arr-params]}
              (analyze-contraction segred dtype)]
          (cond
            (not (canonical-rowmajor? row-idx i-sym L l-sym))
            {:ok false :reason :non-canonical-orientation :operand :row :idx row-idx}
            (not (canonical-rowmajor? col-idx l-sym N j-sym))
            {:ok false :reason :non-canonical-orientation :operand :col :idx col-idx}
            (not (pitch-ok? N))   ; B pitch = N·2 bytes must be 16-byte aligned
            {:ok false :reason :n-pitch-unaligned :N N}
            (not (pitch-ok? L))   ; A pitch = K·2 bytes must be 16-byte aligned
            {:ok false :reason :k-pitch-unaligned :L L}
            :else
            {:ok true :M M :N N :L L :row-arr row-arr :col-arr col-arr :arr-params arr-params}))
        (catch clojure.lang.ExceptionInfo e
          {:ok false :reason :not-a-contraction :msg (.getMessage e)})))))

(defn generate-dpas-contraction-kernel
  "DPAS/XMX-tensorized contraction → OpenCL (PEAK; raster's edge over Futhark's portable
   ~50-70%-peak tiling). The general, IR-driven contribution is the LEGALITY GATE +
   operand-orientation analysis (dpas-contraction-legal?); the DPAS BODY is the validated
   emit-gemm-nonsquare-kernel reused verbatim (f16-in / f32-acc / f16-out, 128×128 tiles,
   16 subgroups, K16 mad). The SOAC IR decides WHICH input is the row operand (→ A slot) vs
   col operand (→ B slot) and the dims to launch with — so a batched or transposed
   contraction re-tensorizes through the golden's :batched?/:split-k? variants rather than a
   separate hand kernel. NON-gemm-specific at the IR boundary; peak at the hardware boundary.

   Returns {:kernel-name :source :array-params [row-arr col-arr] :dims [M N L] :dtype :half
            :tensorized true}  — NB: :array-params is in [row col] BINDING order (row's
   buffer → A slot, col's → B slot, out → C), NOT sorted-by-name. Returns
   {:tensorized false :reason …} when the gate rejects (caller falls back to regtiled)."
  [segred out-sym & {:keys [dtype] :or {dtype :half}}]
  (let [gate (dpas-contraction-legal? segred dtype)]
    (if-not (:ok gate)
      {:tensorized false :reason (:reason gate) :detail gate}
      (let [{:keys [M N L row-arr col-arr]} gate
            kernel-name (str "dpas_contract_" (gensym ""))
            source (codegen/emit-gemm-nonsquare-kernel kernel-name :c-dtype :half)]
        {:kernel-name kernel-name
         :source source
         :array-params [row-arr col-arr]      ;; [A-slot B-slot] binding order
         :dims [M N L]
         :dtype :half
         :tensorized true}))))

;; ================================================================
;; QUANT (int8) contraction — the SAME skeleton, WIDENING facet
;; ================================================================

(defn generate-quant-contraction-kernel
  "QUANT contraction → OpenCL. The SAME SegRed skeleton as the f16/f64 ladder — nothing
   structurally new — carrying the facets a quantized matmul adds:
     • WIDENING accumulate — int8 operands, int32 accumulator (int8's key op is widening, not
       same-width: char×char must not accumulate in char);
     • a DECODE SCHEME on the operands + a dequant EPILOGUE.

   TYPES ARE GENERIC, not quant-specific: the operand dtype is :byte (=int8, `int8_t`), the
   accumulate dtype :int (=int32), the output :float — all pulled from the ONE dtype facet
   map (dtype-info), so widening is just a dtype PAIR, not a special type. What IS quant-
   specific is the `scheme` — a DECODE descriptor (scale, per-operand zero-point; extensible
   to block-size / q4 packing) that is NOT a native dtype (q8_0/q4_k have no C type; they
   decode to int + scale). It generates the operand decode + the dequant epilogue:
       out = scale · Σ_l (A[i,l] − a-zp)·(B[l,j] − b-zp)
   (symmetric per-tensor int8 ⇒ zero-points 0). This decode scheme IS the operand load-lambda
   — the same slot the register-tiled path can fuse (the fusion frontier). The dp4a / int8-
   DPAS packed dot is the tensorize LEAF (perf layer, analogous to f16's DPAS); this is the
   NAIVE correctness substrate. Canonical row-major orientation only (reuses analyze-
   contraction + the DPAS gate's orientation check).

   scheme: {:scale s (default 1.0), :a-zp z (0), :b-zp z (0)}."
  [segred out-sym & {:keys [scheme] :or {scheme {}}}]
  (let [{:keys [scale a-zp b-zp] :or {scale 1.0 a-zp 0 b-zp 0}} scheme
        {:keys [M N L i-sym j-sym l-sym row-arr col-arr row-idx col-idx]}
        (analyze-contraction segred :byte)
        _ (assert (canonical-rowmajor? row-idx i-sym L l-sym)
                  "quant: row operand must be [M,L] row-major (A[i,l]=i·L+l)")
        _ (assert (canonical-rowmajor? col-idx l-sym N j-sym)
                  "quant: col operand must be [L,N] row-major (B[l,j]=l·N+j)")
        ;; C types from the ONE dtype facet map — generic int8/int32/float, not quant types.
        op-ctype  (get codegen/opencl-type-map :byte)   ; int8 storage (int8_t; signed)
        acc-ctype (get codegen/opencl-type-map :int)    ; int32 widening accumulate
        out-ctype (get codegen/opencl-type-map :float)  ; dequant output
        A (ce/c-symbol row-arr) B (ce/c-symbol col-arr)
        ;; per-operand decode = widen to acc dtype, then subtract zero-point (the load-lambda)
        decode (fn [arr idx zp]
                 (let [w (str "(" acc-ctype ")" arr "[" idx "]")]
                   (if (zero? zp) w (str "(" w " - " zp ")"))))
        a-term (decode A (str "i * " L " + l") a-zp)
        b-term (decode B (str "l * " N " + j") b-zp)
        kernel-name (str "quant_contract_" (gensym ""))
        src (str "__kernel void " kernel-name "(__global const " op-ctype "* restrict " A
                 ", __global const " op-ctype "* restrict " B
                 ", __global " out-ctype "* restrict out, " out-ctype " scale, int _nseg) {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 "    int i = (seg / " N ") % " M ";\n"
                 "    int j = seg % " N ";\n"
                 "    " acc-ctype " acc = 0;\n"              ; WIDENING: int32 accumulator
                 "    for (int l = 0; l < " L "; l++) {\n"
                 "        acc += " a-term " * " b-term ";\n" ; decode + widening MAC
                 "    }\n"
                 "    out[seg] = scale * (" out-ctype ")acc;\n"   ; dequant epilogue
                 "}\n")]
    {:kernel-name kernel-name :source src
     :array-params [row-arr col-arr]
     :dtype :byte :acc-dtype :int :out-dtype :float
     :scheme (merge {:scale scale :a-zp a-zp :b-zp b-zp} scheme)
     :dims [M N L]}))

(defn generate-dp4a-contraction-kernel
  "DP4A-tensorized int8 contraction → OpenCL — the int8 PEAK LEAF (analogous to f16's DPAS
   leaf). Uses the dp4a int8×4 dot-accumulate primitive (rstr_dp4a: 4 int8 MACs into an int32
   in one op; native dp4a is a drop-in). This is where int8 quant reaches peak, the same way
   generate-dpas-contraction-kernel is f16's peak.

   PER-LEAF LAYOUT REQUIREMENT: dp4a packs 4 int8 along the contract axis into a 32-bit word,
   so BOTH operands must be K-contiguous. A[M,K] row-major is (A[i,l]=i·K+l, K contiguous);
   B must therefore be [N,K] TRANSPOSED (B[j,l]=j·K+l) — the :nt orientation, DIFFERENT from
   f16-DPAS's :nn. The required operand layout is a property of the TENSORIZE LEAF, not the
   contraction — the gate dispatches (dtype, leaf) → layout. K must be a multiple of 4.

   Operands are reinterpreted at launch as int[] (4 packed int8 per int; the USM bytes are
   identical to the int8 buffer). Symmetric quant only for now (zero-points fold into a
   correction term — future). scheme: {:scale s}."
  [segred out-sym & {:keys [scheme] :or {scheme {}}}]
  (let [{:keys [scale a-zp b-zp] :or {scale 1.0 a-zp 0 b-zp 0}} scheme
        {:keys [M N L i-sym j-sym l-sym row-arr col-arr row-idx col-idx]}
        (analyze-contraction segred :byte)
        _ (assert (zero? (mod L 4)) "dp4a: contract axis K must be a multiple of 4 (int8×4 packing)")
        _ (assert (canonical-rowmajor? row-idx i-sym L l-sym)
                  "dp4a: row operand must be [M,K] row-major (A[i,l]=i·K+l)")
        _ (assert (canonical-rowmajor? col-idx j-sym L l-sym)
                  "dp4a: col operand must be [N,K] TRANSPOSED (B[j,l]=j·K+l) — dp4a needs both operands K-contiguous")
        _ (assert (and (zero? a-zp) (zero? b-zp))
                  "dp4a leaf: symmetric quant only (zero-points fold into a correction term — future)")
        A (ce/c-symbol row-arr) B (ce/c-symbol col-arr)
        KP (quot L 4)                            ; packed contract length (int8×4 words)
        acc-ctype (get codegen/opencl-type-map :int)
        out-ctype (get codegen/opencl-type-map :float)
        helper (:c-helper-src (intrinsics/descriptor 'dp4a))
        kernel-name (str "dp4a_contract_" (gensym ""))
        src (str helper
                 "__kernel void " kernel-name "(__global const int* restrict " A
                 ", __global const int* restrict " B
                 ", __global " out-ctype "* restrict out, " out-ctype " scale, int _nseg) {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 "    int i = (seg / " N ") % " M ";\n"
                 "    int j = seg % " N ";\n"
                 "    " acc-ctype " acc = 0;\n"
                 "    for (int p = 0; p < " KP "; p++) {\n"
                 "        acc = rstr_dp4a(" A "[i * " KP " + p], " B "[j * " KP " + p], acc);\n"
                 "    }\n"
                 "    out[seg] = scale * (" out-ctype ")acc;\n"
                 "}\n")]
    {:kernel-name kernel-name :source src
     :array-params [row-arr col-arr]
     :dtype :byte :acc-dtype :int :out-dtype :float :packed :int8x4
     :scheme (merge {:scale scale :a-zp a-zp :b-zp b-zp} scheme)
     :dims [M N L]}))

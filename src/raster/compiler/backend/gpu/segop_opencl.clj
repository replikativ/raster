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
            [raster.compiler.core.dtype :as dt]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as cstage]
            [raster.compiler.ir.contraction-facts :as cf]
            [clojure.walk :as walk]
            [clojure.set]
            [raster.compiler.ir.segop :as segop]
            [clojure.string :as str]))

;; ================================================================
;; SegOp field accessors
;; ================================================================

;; Innermost (reduced/mapped) dim — `first` for the 1-D case (unchanged).
;; The N-D segmented GPU kernel (generate-segmented-reduce-kernel) iterates outer segment dims.
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
        default-ctype (dt/ctype :opencl dtype)
        out-ctype (dt/ctype :opencl out-dtype)
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
        ctype (dt/ctype :opencl dtype)
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
        ctype (dt/ctype :opencl dtype)
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
        ctype (dt/ctype :opencl dtype)
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
        ;; SYMBOLIC axis bounds must be DECLARED as int params, exactly as the segmented-reduce
        ;; sibling does — the decompose above emits their names, so without this the kernel
        ;; references undeclared identifiers and fails to compile.
        scl-params (vec (sort-by name (:scalars segmap)))
        scl-param-str (str/join "" (map (fn [s] (str ", int " (ce/c-symbol s))) scl-params))
        src (str (codegen/extension-pragmas dtype)
                 "__kernel void " kernel-name "(" arr-param-str ", __global " ctype "* restrict out"
                 scl-param-str ", int _nseg) {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 decomp "\n"
                 "    out[seg] = " body-str ";\n"
                 "}\n")]
    {:kernel-name kernel-name :source src :array-params arr-params
     :scalar-params scl-params :dtype dtype}))

;; ================================================================
;; Block-tiled + __local-staged contraction (BlkRegTiling, block-tile level)
;; ================================================================

(defn- syms-in [expr] (set (filter symbol? (tree-seq coll? seq expr))))

;; analyze-contraction is defined below (next to the register-tiled emitter that also uses it);
;; declared here because the block-tiled emitter above it shares the same analysis.
(declare analyze-contraction)

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
        ;; Structural preconditions of the TENSORIZE leaves. These are ex-info (not assert) so
        ;; the legality gate catches them and the router FALLS BACK to the general naive leaf,
        ;; instead of an AssertionError escaping and hard-failing a legal contraction.
        _ (when-not (= 2 (count seg-dims))
            (throw (ex-info "tensorize: needs exactly 2 free axes" {:reason :not-2-free :n-free (count seg-dims)})))
        [fi fj] seg-dims
        M (:bound fi) N (:bound fj) L (:bound red-dim)
        _ (when-not (every? number? [M N L])
            (throw (ex-info "tensorize: needs literal dims" {:reason :symbolic-dims :dims [M N L]})))
        i-sym (:name fi) j-sym (:name fj) l-sym (:name red-dim)
        dtype (or (:dtype segred) dtype)
        ctype (dt/ctype :opencl dtype)
        {:keys [init lambda]} (:reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        _ (when-not (#{'+ 'clojure.core/+ 'raster.numeric/+} (descriptor/semantic-op lambda))
            (throw (ex-info "tensorize: combine must be +" {:reason :non-plus-combine :op (descriptor/semantic-op lambda)})))
        acc-sym (:acc (:reduce-op segred))
        acc-at? (fn [a] (or (= a acc-sym) (and (seq? a) (= 'double (first a)) (= acc-sym (second a)))))
        op-args (vec (descriptor/call-args lambda))
        elem (let [a0 (nth op-args 0) a1 (nth op-args 1)] (if (acc-at? a0) a1 a0))
        _ (when-not (and (seq? elem) (#{'* 'clojure.core/* 'raster.numeric/*} (descriptor/semantic-op elem)))
            (throw (ex-info "tensorize: element must be a product of two agets" {:reason :non-product-element})))
        parts (fn [e] (let [e (ce/normalize-array-prims e)]
                        (when-not (and (seq? e) (= 'aget (first e)))
                          (throw (ex-info "tensorize: operand must be an aget" {:reason :non-aget-operand})))
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
  "Does affine index `idx` equal `(+ (* outer stride) inner)` (row-major, leading dim `stride`)?

   ONE RELATION. This used to be a hand-rolled pattern match that was +/*-order-agnostic but could
   only see a literal 3-element sum, while `am/index-matches?` flattened nested sums but compared
   terms positionally. The two accepted DIFFERENT languages, so a legal :nt operand written
   `(+ l (* j K))` passed one gate and was rejected by the other — reaching no leaf at all. Both
   now normalize to the same affine form, which accepts every arithmetically-identical spelling."
  [idx outer stride inner]
  (am/index-matches? (am/of-axes [[outer 1] [inner stride]])
                     (ce/normalize-array-prims idx)))

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
    (if-not (and (dt/known? dtype) (= :half (dt/canon dtype)))
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
            ;; i-sym/j-sym are the FREE axes; an epilogue binds them to the store slot's row/col
            {:ok true :M M :N N :L L :i-sym i-sym :j-sym j-sym
             :row-arr row-arr :col-arr col-arr :arr-params arr-params}))
        (catch clojure.lang.ExceptionInfo e
          {:ok false :reason :not-a-contraction :msg (.getMessage e)})))))

(def ^:private epilogue-forbidden-ops
  "Ops that cannot appear in a store-spliced epilogue because they force a distribution/layout
   change of the accumulator (Triton's blocked-only / anchor set): a scan, a NON-associative
   reduction, a permuting reshape, or an atomic. An associative reduction is not forbidden outright
   — it is a RE-TILING decision (distribute the accumulator so the reduction is warp-local) which
   we do not implement yet, so it is refused here with its own reason."
  '#{raster.par/scan raster.par/scan-exclusive raster.par/scatter! raster.par/reduce-by-key
     raster.par/reduce raster.par/reduce-into raster.par/contract})

(defn epilogue-legal?
  "Legality of splicing `expr` into the contraction's STORE slot. Returns {:ok true} or
   {:ok false :reason kw}. Distilled from Halide / XLA / Triton / MLIR-linalg:

     • the epilogue is emitted strictly AFTER the reduction loop closes. Placing it inside forces
       accumulator multi-buffering under software pipelining — a silent 2x on the most expensive
       resource. Our splice is in the store slot, so this holds by construction; it is asserted
       here so a future change cannot quietly break it.
     • no op that forces a layout/distribution change: scan, non-associative reduce, permuting
       reshape, atomic. A reduction in the epilogue is a re-tiling decision (redistribute the
       accumulator so the reduce is warp-local) — legal in principle, unimplemented, so refused
       with :reduction-in-epilogue rather than silently miscompiled.
     • the accumulator must appear exactly ONCE: more than one use duplicates the reduction result
       through the epilogue expression, which is the recompute case the ceilings guard against."
  [{:keys [acc expr]}]
  (let [nodes (tree-seq coll? seq expr)
        heads (into #{} (keep #(when (seq? %) (first %))) nodes)
        acc-uses (count (filter #(= acc %) nodes))]
    (cond
      (zero? acc-uses)                     {:ok false :reason :epilogue-ignores-accumulator}
      (> acc-uses 1)                       {:ok false :reason :accumulator-used-more-than-once}
      (some #{'raster.par/reduce 'raster.par/reduce-into} heads)
      {:ok false :reason :reduction-in-epilogue}
      (seq (clojure.set/intersection heads epilogue-forbidden-ops))
      {:ok false :reason :layout-changing-op-in-epilogue
       :ops (clojure.set/intersection heads epilogue-forbidden-ops)}
      :else {:ok true})))

(defn epilogue-cost
  "Byte-traffic model for splicing an epilogue into the contraction's store, and the register
   pressure it adds. Returns {:fused-bytes :unfused-bytes :saved-bytes :profitable :acc-regs}.

   NOTE ON A TEST THAT DOES NOT TRANSFER. Halide's `is_func_trivial_to_inline`
   (1 + sizeof(out) >= arith + bytes) is a PRODUCER-INLINING test: should I duplicate a producer's
   work at each consumer, paying recompute? Applied to an epilogue it gives the wrong answer — it
   scores a bias-add as unprofitable (call 3 vs inline 6 at f16) because it charges the bias LOAD
   without crediting the eliminated round-trip. Epilogue (output) fusion is the opposite direction:
   it REMOVES traffic rather than duplicating work.

       unfused:  write C  +  read C  +  read operands  +  write C   ~ 3·M·N + operands
       fused:    read operands                                     ~ operands

   So an epilogue is profitable whenever it is legal and fits in registers — which is why Triton
   and XLA fuse epilogues unconditionally and gate on legality + register pressure instead. The
   recompute ceilings (8x CPU / 10x GPU) and the multi-consumer refusal belong to the PRODUCER
   direction; the analogue here is `epilogue-legal?`'s accumulator-used-once rule.

   Register estimate follows Triton's closed form: elems-per-thread × threads-per-warp ×
   warps-per-CTA × elem-bytes / 4 registers for the accumulator; the epilogue's live values must
   fit in the remaining budget (Triton clamps accumulator traffic at maxnreg/2 to avoid spilling)."
  [{:keys [operands]} out-dtype [M N] tile]
  (let [bytes-of (fn [d] (long (dt/bytes-of d)))   ; one registry, not a private table
        cb (bytes-of out-dtype)
        c-elems (* (long M) (long N))
        operand-bytes (reduce + 0 (for [{:keys [dtype] :or {dtype :float}} operands]
                                    (* (long N) (bytes-of dtype))))
        unfused (+ (* 3 c-elems cb) operand-bytes)   ; write + read + write, plus operand reads
        fused   operand-bytes
        ;; accumulator registers per lane: (block-m·block-n / (subgroups·subgroup)) f32 values
        {:keys [block-m block-n sg-m sg-n]} tile
        sg (long (get-in tile [:matrix :subgroup] 16))
        acc-regs (when (and block-m block-n sg-m sg-n)
                   (quot (* (long sg-m) (long sg-n)) sg))]
    {:fused-bytes fused :unfused-bytes unfused
     :saved-bytes (- unfused fused)
     :profitable (< fused unfused)
     :acc-regs acc-regs}))

(defn epilogue-splice
  "Build the (fn [acc-expr row col] -> C-expr) + param-decls + helpers that emit-gemm-tiled's
   store-splice expects, from a DOMAIN-AGNOSTIC spec. The compiler learns no op names: the spec is
   just an EXPRESSION over the accumulator and some operands, each operand carrying its own
   axis-map, so bias / activation / residual / dequant-scale are all the same mechanism and
   COMPOSE by nesting (one bigger expression), per CLAUDE.md's domain-agnostic-passes rule.

     {:acc  acc            ;; symbol standing for the contraction's accumulator
      :expr <s-expr>       ;; e.g. (raster.numeric/* (raster.numeric/+ acc (aget bias j)) s)
      :operands [{:sym bias :map <axis-map over the FREE axes> :dtype :float}]
      :scalars  [{:sym s :dtype :float}]      ;; optional uniform scalars
      :helpers  <C source string>}  ;; optional, prepended (e.g. a silu_f definition)

   `free-syms` are the contraction's two free-axis symbols, bound to the store slot's `row`/`col`
   C variables — so an operand's axis-map generates its index exactly as in the kernel body.
   Returns {:epilogue fn :epilogue-params str :epilogue-helpers str|nil}."
  [{:keys [acc expr operands scalars helpers] :as spec} [i-sym j-sym] dtype]
  (let [legal (epilogue-legal? spec)
        _ (when-not (:ok legal)
            (throw (ex-info (str "epilogue-splice: illegal epilogue (" (:reason legal) ")")
                            (assoc legal :expr expr))))
        ctype (dt/ctype :opencl dtype)
        arr-syms (set (map (comp #(symbol (name %)) :sym) operands))
        int-vars (into #{} (map #(symbol (name %))) [i-sym j-sym])
        ;; substitute each operand's aget index from its declared map (maps, not pattern-matching)
        idx-of (into {} (for [{:keys [sym map]} operands] [sym (am/index-expr map)]))
        expr' (walk/postwalk
               (fn [f] (if (and (seq? f) (= 'aget (first f)) (contains? idx-of (second f)))
                         (list 'aget (second f) (get idx-of (second f)))
                         f))
               expr)
        acc-token (str "__acc_" (name (gensym "")))
        ;; emit once with a distinctive token standing in for the accumulator, then splice the
        ;; real acc C-expression in at call time (the hook supplies it per store slot)
        emitted (binding [ce/*emit-config* ce/opencl-config
                          ce/*scalar-type* ctype
                          ce/*int-vars* (into ce/*int-vars* int-vars)]
                  (ce/emit-expr (walk/postwalk-replace {acc (symbol acc-token)} expr')
                                (gensym "z__") arr-syms))
        params (apply str
                      (concat
                       (for [{:keys [sym dtype] :or {dtype :float}} operands]
                         (str ", __global const " (dt/ctype :opencl dtype)
                              "* restrict " (ce/c-symbol sym)))
                       (for [{:keys [sym dtype] :or {dtype :float}} scalars]
                         (str ", " (dt/ctype :opencl dtype) " " (ce/c-symbol sym)))))]
    {:epilogue (fn [acc-expr row col]
                 (-> emitted
                     (str/replace acc-token (str "(" acc-expr ")"))
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol i-sym) "\\b")) row)
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol j-sym) "\\b")) col)))
     :epilogue-params params
     :epilogue-helpers helpers}))

(defn generate-dpas-contraction-kernel
  "DPAS/XMX-tensorized contraction → OpenCL (PEAK; raster's edge over Futhark's portable
   ~50-70%-peak tiling). The general, IR-driven contribution is the LEGALITY GATE +
   operand-orientation analysis (dpas-contraction-legal?); the DPAS BODY is the validated
   emit-gemm-tiled reused verbatim (f16-in / f32-acc / f16-out, tile-parametric,
   16 subgroups, K16 mad). The SOAC IR decides WHICH input is the row operand (→ A slot) vs
   col operand (→ B slot) and the dims to launch with — so a batched or transposed
   contraction re-tensorizes through the golden's :batched?/:split-k? variants rather than a
   separate hand kernel. NON-gemm-specific at the IR boundary; peak at the hardware boundary.

   Returns {:kernel-name :source :array-params [row-arr col-arr] :dims [M N L] :dtype :half
            :tensorized true}  — NB: :array-params is in [row col] BINDING order (row's
   buffer → A slot, col's → B slot, out → C), NOT sorted-by-name. Returns
   {:tensorized false :reason …} when the gate rejects (caller falls back to regtiled)."
  [segred out-sym & {:keys [dtype desc tile epilogue] :or {dtype :half}}]
  (let [gate (dpas-contraction-legal? segred dtype)]
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
            tile (or tile (hw/derive-gemm-tile (or desc {})))
            sg (long (get-in tile [:matrix :subgroup] 16))
            ;; EPILOGUE FUSION: fold the consumer expression into the store slot, so a
            ;; bias/activation/residual/dequant costs no extra kernel and no DRAM round-trip of C.
            ep (when epilogue
                 (epilogue-splice epilogue
                                  [(:i-sym gate) (:j-sym gate)]
                                  (get epilogue :dtype :float)))
            source (apply codegen/emit-gemm-tiled kernel-name
                          (concat [:c-dtype :half
                                   :block-m (:block-m tile) :block-n (:block-n tile)
                                   :sg-m (:sg-m tile) :sg-n (:sg-n tile)
                                   :block-k (:block-k tile) :matrix (:matrix tile)]
                                  (when ep [:epilogue (:epilogue ep)
                                            :epilogue-params (:epilogue-params ep)
                                            :epilogue-helpers (:epilogue-helpers ep)])))]
        {:kernel-name kernel-name
         :source source
         :array-params [row-arr col-arr]      ;; [A-slot B-slot] binding order
         :dims [M N L]
         :dtype :half
         :tile tile
         ;; The epilogue's operand arrays are EXTRA kernel params, appended AFTER the dims.
         ;; Surfaced so a caller binds them — the signature has them either way, so omitting
         ;; them from the descriptor would be a launch-arity bug (6 args bound to a 7-arg kernel).
         :epilogue-params (when ep (:epilogue-params ep))
         :epilogue-operands (when ep (mapv :sym (:operands epilogue)))
         ;; …and its SCALARS. epilogue-splice has always emitted these into the signature, but
         ;; nothing surfaced them, so any epilogue carrying a `:scalars` entry tripped
         ;; validate-descriptor's scalar count and threw. That unusable capability is exactly why
         ;; `:scheme` had to invent a private per-tensor scale channel instead of being an
         ;; epilogue. Surfacing them makes the scale expressible where it belongs.
         :epilogue-scalars (when ep (mapv :sym (:scalars epilogue)))
         ;; workgroup = (block-m/sg-m)·(block-n/sg-n) subgroups × the matrix subgroup size
         :workgroup [(* (quot (:block-m tile) (:sg-m tile))
                        (quot (:block-n tile) (:sg-n tile))
                        sg) 1]
         :tensorized true}))))

;; ================================================================
;; QUANT (int8) contraction — the SAME skeleton, WIDENING facet
;; ================================================================

(defn- flat-decompose-c
  "C declarations recovering each free index from the flat segment id `seg`, row-major:
   idx_p = (seg / Π bounds-after-p) % bound_p, with the innermost simplifying to `seg % bound`.
   ONE source for the row-major decompose. (Debt: three older copies of this arithmetic remain
   inline — generate-segmented-reduce-kernel, generate-segmap-nd-kernel, and the two quant
   kernels — and should collapse onto this helper; not done here to keep their emitted text
   provably byte-identical.)"
  [free-axes]
  (let [v (vec free-axes) n (count v)]
    (str/join "\n"
              (map-indexed
               (fn [p [sym bound]]
                 ;; bounds go through c-symbol too: a symbolic bound named `n-cols` was emitted
                 ;; raw as `(seg / (n-cols))` — valid C that computes `n - cols`.
                 (let [bc (fn [b] (if (symbol? b) (ce/c-symbol b) b))
                       after (map (comp bc second) (subvec v (inc p) n))
                       div (if (seq after) (str "(seg / (" (str/join " * " after) "))") "seg")]
                   (str "    int " (ce/c-symbol sym) " = " div " % " (bc bound) ";")))
               v))))

(defn staged-inner-dp4a-legal?
  "Can the INNERMOST stage of a staged contraction be tensorized with dp4a (4 int8 MACs into an
   int32 in one op)? Returns {:ok true :packed-maps {sym amap} :packed-extent n} or
   {:ok false :reason kw}.

   This is the int8 PEAK leaf for block-quant, and it is the same structure llama.cpp hand-writes:
   the inner stage is already an exact int32 accumulation over a short K-contiguous run, which is
   precisely dp4a's shape. Because the stage list says which axis the inner accumulation runs over,
   there is nothing to recognize — the gate only has to CHECK.

   Required, and the failure each check prevents:
     • declared operand axis-maps. Tensorizing needs to know each operand's innermost axis; per the
       compiler's declare-don't-pattern-match rule that is data, not inference.
     • each map must VERIFIABLY equal the operand's actual index in the body (am/index-matches?).
       Assuming a layout while having checked only the axis symbols is how a transpose rewrite
       silently miscompiled before; a leaf may only assume what it has proved.
     • the inner stage's axis must be the INNERMOST axis of both maps — dp4a packs 4 consecutive
       elements along the contraction, so both operands must be contiguous in it (the :nt layout).
     • int8 operands, integral inner accumulator (widening as a dtype pair).
     • the inner extent must be a literal multiple of 4, else the packed load is mis-strided."
  [{:keys [stages body operands dtype]}]
  (let [inner-stage (peek (vec stages))
        int-acc? (contains? #{:int :long :int32} (:dtype inner-stage))
        agets (into {} (keep (fn [f] (when (and (seq? f) (= 'aget (first f)) (symbol? (second f)))
                                      [(second f) (nth f 2)]))
                             (tree-seq coll? seq body)))
        ;; THE BODY IS DISCARDED when this leaf fires — the whole summand is replaced by one
        ;; rstr_dp4a call — so the gate must account for EVERY term first. The requirement lives in
        ;; ir/contraction-facts as `body-product-of`, shared with every other body-replacing leaf,
        ;; rather than re-derived per gate.
        exact-product? (some? (cf/body-product-of body (map :sym operands)))]
    (cond
      (not= 2 (count operands)) {:ok false :reason :dp4a-needs-two-declared-operands}
      (not (every? :map operands)) {:ok false :reason :operand-without-a-declared-map}
      (not (and (dt/known? dtype) (= :byte (dt/canon dtype))))
      {:ok false :reason :dp4a-needs-int8-operands :dtype dtype}
      (not exact-product?)
      {:ok false :reason :body-has-unmodeled-terms
       :detail "this leaf replaces the body with a single hardware op; any term beyond the two declared operands would be silently dropped"
       :body body :declared (mapv :sym operands)}
      (not int-acc?) {:ok false :reason :inner-stage-accumulator-not-integral
                      :dtype (:dtype inner-stage)}
      :else
      (or
       ;; the declared map must PROVABLY be the operand's actual index expression
       (first (keep (fn [{:keys [sym map]}]
                      (let [idx (get agets sym)]
                        (cond
                          (nil? idx) {:ok false :reason :declared-operand-not-read-by-the-body :sym sym}
                          (not (am/index-matches? map idx))
                          {:ok false :reason :declared-map-does-not-match-the-body-index
                           :sym sym :declared (am/index-expr map) :actual idx}
                          (not= (:axis inner-stage) (am/innermost-axis map))
                          {:ok false :reason :inner-stage-axis-is-not-contiguous
                           :sym sym :innermost (am/innermost-axis map) :axis (:axis inner-stage)}
                          :else nil)))
                    operands))
       (let [p-sym (gensym "p__")
             packed (into {} (for [{:keys [sym map]} operands]
                               [sym (am/pack-innermost map 4 p-sym)]))]
         (if (some nil? (vals packed))
           {:ok false :reason :inner-extent-not-a-multiple-of-4 :extent (:extent inner-stage)}
           {:ok true :packed-maps packed :p-sym p-sym
            :packed-extent (quot (long (:extent inner-stage)) 4)}))))))

(defn generate-staged-contraction-kernel
  "STAGED contraction → OpenCL: a reduction accumulating in N levels, each with its own
   accumulator dtype, with a `lift` splicing each level's partial sum into the level above.
   This is the shape every block-quantized format needs and the flat leaves cannot express —
   an int32 MAC inside the block, a float accumulate across blocks (see ir/contract-stages).
   2 stages = q8_0/q4_0, 3 = k-quants, 1 = the flat contraction (so this emitter subsumes it).

   The emitted nest is exactly the schedule `backend/cpu/quant.clj` already uses on CPU and
   llama.cpp hand-writes per format — here it comes from the stage list, not from a kernel per
   format. Domain-agnostic: no scale/zero-point/format concept appears below; a stage is an
   accumulator dtype plus a lift expression, and the lift's operand arrays are indexed by their
   DECLARED axis-maps (never by inferred strides).

   Spec:
     {:free-axes [[i M] [j N]]      ;; output axes, outer→inner (any rank ≥ 1)
      :stages    [outer … inner]    ;; see ir/contract-stages for the stage shape
      :body      <expr>             ;; the summand, over the free + stage axes
      :inputs    [a b]              ;; the body's operand arrays
      :dtype     :byte              ;; the body operands' element dtype
      :out-dtype :float}

   Returns {:kernel-name :source :array-params :dtype :out-dtype :dims :stages :out-elems
            :lift-operands}. :array-params is the body's inputs; :lift-operands are the EXTRA
   scale arrays, bound after them — surfaced because omitting them from a launch descriptor is
   an arity bug (the signature has them either way)."
  [{:keys [free-axes stages body inputs dtype out-dtype operands tensorize-inner? contract-axes
           epilogue]
    :or {dtype :float out-dtype :float} :as spec} out-sym]
  (let [;; The stage list is checked against the axes the FORM DECLARED, never against axes derived
        ;; from the stages themselves. Deriving them made the span rule unfireable, and a stage list
        ;; that under-covers the contract space emitted a kernel summing a FRACTION of the terms
        ;; while par/contract's CPU path summed all of them — a silent divergence between two
        ;; consumers of one form. A gate must not validate its own arguments.
        _ (when (nil? contract-axes)
            (throw (ex-info "staged contraction: :contract-axes is required — the stage list can only be validated against the axes the form declared"
                            {:reason :missing-declared-contract-axes :stages stages})))
        ;; This emitter interpolates extents into the loop bounds and the decompose but declares no
        ;; scalar params for them, so a symbolic bound emits an UNDECLARED identifier — and
        ;; validate-descriptor passes it (the counts agree), so it fails only at device build, i.e.
        ;; never in CI. Its segmented-reduce sibling declares them; until this one does, refuse.
        _ (let [syms (remove number? (concat (map second free-axes) (map :extent stages)))]
            (when (seq syms)
              (throw (ex-info (str "staged contraction: symbolic bounds are not supported by this "
                                   "emitter (it declares no scalar params for them) — "
                                   (pr-str (vec syms)))
                              {:reason :symbolic-bounds-unsupported :bounds (vec syms)}))))
        legal (cstage/stages-legal? stages contract-axes)
        _ (when-not (:ok legal)
            (throw (ex-info (str "staged contraction: illegal stages (" (:reason legal) ")")
                            (assoc legal :stages stages :contract-axes contract-axes))))
        stages (vec stages)
        ;; TENSORIZE THE INNER STAGE: the innermost accumulation is already an exact int32 sum over
        ;; a short K-contiguous run, which is exactly dp4a's shape. Requested explicitly (a schedule
        ;; choice), then GATED — a rejection falls back to the scalar nest, never to a wrong kernel.
        tz (when tensorize-inner?
             (let [g (staged-inner-dp4a-legal? spec)]
               (when-not (:ok g)
                 (throw (ex-info (str "staged contraction: cannot tensorize inner stage ("
                                      (:reason g) ")") g)))
               g))
        ;; packed operands are READ as int32 words (4 int8 each); the buffer bytes are unchanged
        op-ctype  (get codegen/opencl-type-map (if tz :int dtype) "float")
        out-ctype (dt/ctype :opencl out-dtype)
        lift-ops (cstage/lift-operands stages)
        idx-of (cstage/stage-index-exprs stages)
        ;; every axis in scope is an int loop/decompose variable
        int-vars (into #{} (map (comp symbol name))
                       (concat (map first free-axes) (map :axis stages)
                               (when tz [(:p-sym tz)])))
        arr-syms (into #{} (map (comp symbol name)) (concat inputs (map :sym lift-ops)))
        emit (fn [expr]
               (binding [ce/*emit-config* ce/opencl-config
                         ce/*scalar-type* out-ctype
                         ce/*int-vars* (into ce/*int-vars* int-vars)]
                 (ce/emit-expr expr (gensym "z__") arr-syms)))
        acc-name (fn [d] (str "acc_" d))
        ;; PER-OPERAND DECODE — the load-lambda. `:decode` is an expression in `x`, standing for the
        ;; raw load, applied to that operand's every read. This is where a zero-point subtraction
        ;; belongs: `Σ(a-za)(b-zb)` is exact on the load path and needs no correction reductions,
        ;; whereas a per-tensor SCALE factors out of the sum entirely and belongs in the epilogue.
        ;; WIDENING is deliberately NOT expressed here — it is the dtype PAIR (operand dtype +
        ;; accumulator dtype), and hiding it in a lambda would blind the tensorize gate to the very
        ;; pair it dispatches on.
        decodes (into {} (keep (fn [{:keys [sym decode]}] (when decode [sym decode]))) operands)
        body (if (empty? decodes)
               body
               (walk/postwalk
                (fn [f] (if (and (seq? f) (= 'aget (first f)) (contains? decodes (second f)))
                          (walk/postwalk-replace {'x f} (get decodes (second f)))
                          f))
                body))
        ;; EPILOGUE — the store splice. An epilogue is a lift on a virtual outermost level of
        ;; extent 1, which is why it needs no linearity (nothing to distribute over one iteration)
        ;; while a real stage lift does. Gives this emitter the dequant scale that the two quant
        ;; leaves hardwired into their store lines.
        ep (when epilogue
             (when-not (= 2 (count free-axes))
               (throw (ex-info "staged contraction: an epilogue needs exactly 2 free axes (the store slot binds row/col)"
                               {:reason :epilogue-needs-2-free :n-free (count free-axes)})))
             (epilogue-splice epilogue (mapv first free-axes) (or (:dtype epilogue) out-dtype)))
        ;; innermost accumulates the body; each outer accumulates its lift with `inner` bound to
        ;; the accumulator one level down. Built inside-out.
        n (count stages)
        ;; ONE description of the innermost loop, used at both nesting depths.
        inner-loop
        (fn [indent acc]
          (let [{:keys [dtype init axis extent]} (peek stages)
                t (dt/ctype :opencl dtype)
                [loop-var bound step]
                (if tz [(:p-sym tz) (:packed-extent tz) "dp4a"] [axis extent "scalar"])
                pad (apply str (repeat indent " "))]
            (str pad t " " acc " = " (or init 0) ";\n"
                 pad "for (int " (ce/c-symbol loop-var) " = 0; "
                 (ce/c-symbol loop-var) " < " bound "; " (ce/c-symbol loop-var) "++) {\n"
                 pad "    " acc
                 (if (= "dp4a" step)
                   ;; 4 int8 MACs into the int32 accumulator per op; indices come from the PACKED
                   ;; maps, so the stride rescaling is the axis-map algebra's, not hand-written.
                   (str " = rstr_dp4a("
                        (str/join ", " (for [{:keys [sym]} operands]
                                         (str (ce/c-symbol sym) "["
                                              (emit (am/index-expr (get (:packed-maps tz) sym)))
                                              "]")))
                        ", " acc ")")
                   (str " += " (emit body)))
                 ";\n"
                 pad "}\n")))
        inner-most (inner-loop 8 (acc-name (dec n)))
        nest (reduce
              (fn [inner-src d]
                (let [{:keys [dtype init axis extent lift]} (nth stages d)
                      t (dt/ctype :opencl dtype)
                      lift' (cstage/substitute-operand-indices lift idx-of)
                      ;; splice the level-below accumulator in for `inner`
                      lift'' (walk/postwalk-replace {'inner (symbol (acc-name (inc d)))} lift')]
                  (str "    " t " " (acc-name d) " = " (or init 0) ";\n"
                       "    for (int " (ce/c-symbol axis) " = 0; "
                       (ce/c-symbol axis) " < " extent "; " (ce/c-symbol axis) "++) {\n"
                       inner-src
                       "        " (acc-name d) " += " (emit lift'') ";\n"
                       "    }\n")))
              inner-most
              (reverse (range (dec n))))
        ;; a single stage has no lift, so its accumulator is the whole nest, unindented
        nest (if (= 1 n) (inner-loop 4 (acc-name 0)) nest)
        n-out (am/n-elements (am/of-axes (vec free-axes)))
        kernel-name (str "staged_contract_" (gensym ""))
        params (str (str/join ", " (for [a inputs]
                                     (str "__global const " op-ctype "* restrict " (ce/c-symbol a))))
                    (apply str (for [{:keys [sym dtype] :or {dtype :float}} lift-ops]
                                 (str ", __global const " (dt/ctype :opencl dtype)
                                      "* restrict " (ce/c-symbol sym))))
                    ", __global " out-ctype "* restrict out"
                    (when ep (:epilogue-params ep))
                    ", int _nseg")
        src (str
                 ;; fp64/fp16 need their OpenCL extension enabled. All three sibling emitters emit
                 ;; this; the staged one did not, so a :double staged contraction — reachable from
                 ;; opencl_pass, whose default dtype IS :double — emitted `double` with no pragma.
                 (codegen/extension-pragmas out-dtype dtype)
                 (when tz (:c-helper-src (intrinsics/descriptor 'dp4a)))
                 (when ep (:epilogue-helpers ep))
                 "__kernel void " kernel-name "(" params ") {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 (flat-decompose-c free-axes) "\n"
                 nest
                 "    out[seg] = "
                 (let [acc (str "(" out-ctype ")" (acc-name 0))]
                   (if ep
                     ((:epilogue ep) acc
                      (ce/c-symbol (first (first free-axes)))
                      (ce/c-symbol (first (second free-axes))))
                     acc))
                 ";\n"
                 "}\n")]
    {:kernel-name kernel-name :source src
     :array-params (vec inputs)
     :lift-operands (mapv :sym lift-ops)
     :epilogue-operands (when ep (mapv :sym (:operands epilogue)))
     :epilogue-scalars (when ep (mapv :sym (:scalars epilogue)))
     :dtype dtype :out-dtype out-dtype
     :stages stages
     ;; the operand buffers are BOUND unchanged; only the kernel's view of them widens
     :tensorized (boolean tz)
     :packed (when tz :int8x4)
     :out-elems n-out
     :dims (mapv second free-axes)}))

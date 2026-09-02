(ns raster.compiler.backend.jvm.par-simd
  "SIMD codegen for declarative parallel primitives.

  Compiles raster.par/map! → Java Vector API (DoubleVector) loop + scalar tail.
  Compiles raster.par/reduce → SIMD horizontal reduction with 4 accumulators.

  Scheduled programs consume their checked SegOps and treat the retained dtype as authoritative.
  Direct source callers use an explicit counted compatibility lowering; fusion belongs exclusively
  to the typed functional middle end.

  Supports:
    - Arithmetic: +, -, *, /
    - Math unary: Math/abs, Math/sqrt, Math/exp, Math/log, Math/sin, Math/cos
    - Math binary: Math/max, Math/min, Math/fma, Math/pow
    - Nested compositions of the above
    - Scalar broadcasts (constants and scalar vars)

  Size guard: skip SIMD for statically-known n < 8 (overhead > benefit).
  For dynamic sizes, emit both paths with runtime branch on n."
  (:require [raster.compiler.core.macroexpand :as macroexpand]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.backend.jvm.segop-simd :as segop-simd]
            [raster.compiler.backend.jvm.bytecode :as bc]
            [raster.compiler.ir.soac]
            [raster.compiler.passes.parallel.soac-lower]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.segop :as segop]
            [raster.runtime.hardware :as hw]
            [clojure.set :as set]))

(defn- desugar-par-fallback
  "Desugar .invk forms in par/map!/reduce fallback expansions.
   The bytecode compiler's closure splitter would otherwise collect .invk
   as a free variable."
  [form]
  (bc/desugar-invk form))

;; ================================================================
;; Shared SIMD infrastructure — delegates to segop_simd
;; ================================================================

(def ^:private simd-unary-ops descriptor/simd-unary-ops)
(def ^:private simd-lanewise-unary-ops descriptor/simd-lanewise-unary-ops)
(def ^:private simd-binary-ops descriptor/simd-binary-ops)
(def ^:private simd-ternary-ops descriptor/simd-ternary-ops)
(def ^:private simd-type-info segop-simd/simd-type-info)
(def ^:private aget-form? segop-simd/aget-form?)
(defn- simd-able-expr? [expr idx] (segop-simd/simd-able? expr idx))

;; Integer index/offset/bound arithmetic must emit as clojure.core primitives —
;; bare symbols resolve to raster.numeric in the kernel ns and box per iteration.
;; See segop-simd's ix+ rationale / CLAUDE.md (integer index arith stays clojure.core).
(defn- ix+ [a b] (list 'clojure.core/+ a b))
(defn- ix- [a b] (list 'clojure.core/- a b))
(defn- ix<= [a b] (list 'clojure.core/<= a b))
(defn- ix< [a b] (list 'clojure.core/< a b))

(defn- collect-scalars
  "Collect scalar symbols from body expression (delegates to segop-simd analysis)."
  [body-expr idx-sym]
  (let [arrays (set (keys (segop-simd/collect-arrays body-expr idx-sym)))]
    (clojure.set/difference
     (segop-simd/collect-free-syms body-expr idx-sym)
     arrays #{idx-sym})))

(defn- collect-array-loads
  "Collect array symbols from aget patterns. Returns set of array symbols."
  [body-expr idx-sym]
  (set (keys (segop-simd/collect-arrays body-expr idx-sym))))

(defn- collect-free-symbols
  "Collect free symbols respecting bindings."
  ([form] (segop-simd/collect-free-syms-scoped form))
  ([form bound] (segop-simd/collect-free-syms-scoped form bound)))

(defn- wrap-in-fn
  "Wrap SIMD code in an immediately-invoked fn for JIT isolation."
  [simd-code free-syms]
  (let [params (vec (sort-by str free-syms))]
    (list* (list 'fn* (list params simd-code)) params)))

(defn- emit-simd-body
  "Compile body to SIMD vector operations (delegates to segop-simd)."
  [expr idx-sym vec-env species-sym elem-type]
  (segop-simd/emit-simd expr idx-sym vec-env species-sym elem-type))

(defn- cast->elem-type
  "Derive element type keyword from cast symbol. Used as fallback when
  :elem-type metadata is not available on the par form."
  [cast]
  (case cast
    float :float
    double :double
    long :long
    int :int
    nil))

(defn- out-sym->elem-type
  "Infer elem-type from output array symbol's type tag.
  Last resort when neither form metadata nor cast provides the type."
  [out-sym]
  (when (symbol? out-sym)
    (case (or (:raster.type/tag (meta out-sym)) (:tag (meta out-sym)))
      floats :float
      doubles :double
      longs :long
      ints :int
      nil)))

;; ================================================================
;; JIT isolation: lift nested SIMD let* for separate compilation
;; ================================================================

(defn lift-nested-lets
  "Lift direct sub-expressions that are let* forms to top level so each
  SIMD loop gets its own JIT compilation unit. This prevents
  deoptimization when SIMD code is nested inside wrapper expressions
  like Math/sqrt.

  Transforms:
    (Math/sqrt (let* [...SIMD...] (loop* [...] ...)))
  →
    (let* [__tmp (let* [...SIMD...] (loop* [...] ...))]
      (Math/sqrt __tmp))

  Only lifts direct sub-expressions (not deeply nested ones) to avoid
  breaking variable scope. Only lifts when the form is NOT already a
  top-level let*."
  [form]
  (if (and (seq? form) (contains? #{'let 'let*} (first form)))
    form  ;; Already a let* — no lifting needed
    ;; Lift any DIRECT sub-expressions that are let* with loop*,
    ;; wrapping each in an immediately-invoked fn for JIT isolation
    (let [bindings (atom [])
          result
          (mapv (fn [f]
                  (if (and (seq? f)
                           (contains? #{'let* 'let} (first f))
                           (some #(and (seq? %) (contains? #{'loop* 'loop} (first %)))
                                 (tree-seq seq? seq f)))
                    (let [tmp (gensym "__simd_result__")
                          free-syms (collect-free-symbols f)
                          wrapped (wrap-in-fn f free-syms)]
                      (swap! bindings conj [tmp wrapped])
                      tmp)
                    f))
                form)]
      (if (seq @bindings)
        (list* 'let* (vec (mapcat identity @bindings)) [(apply list result)])
        form))))

;; ================================================================
;; SIMD code generation for parallel stencil
;; ================================================================

(defn- compile-stencil-info
  "Compile verified stencil fields to a SIMD S-expression.
  For 1D radius-1 stencils: load 3 overlapping vectors at offsets
  (i-1), i, (i+1), compute body using vector ops, store result.
  Scalar tail for remaining elements.
  Returns SIMD S-expression, or nil if unsupported."
  [{:keys [out in-arrays radius boundary cast idx bound body] :as info}
   & {:keys [elem-type]}]
  (let [
        elem-type (or elem-type (:elem-type info) (cast->elem-type cast) (out-sym->elem-type out))
        {:keys [species-expr from-array broadcast cast-fn]} (get simd-type-info elem-type)]
    (when (and out idx bound body (= radius 1) (= :dirichlet boundary))
      ;; Collect arrays and scalars from body
      (when (simd-able-expr? body idx)
        (let [arr-syms (collect-array-loads body idx)
              scalar-syms (collect-scalars body idx)
              species-sym (gensym "species__")
              lanes-sym (gensym "lanes__")
              n-sym (gensym "n__")
              j-sym (gensym "j__")
              vec-tag (case elem-type
                        :double 'jdk.incubator.vector.DoubleVector
                        :float 'jdk.incubator.vector.FloatVector)
              tag-vec (fn [value]
                        (if (instance? clojure.lang.IObj value)
                          (vary-meta value assoc :tag vec-tag)
                          value))
              ;; For each array, generate 3 vector symbols: left, center, right
              arr-shifted-syms
              (into {} (map (fn [s]
                              [s {:left (tag-vec (gensym (str "v_" (name s) "_left_")))
                                  :center (tag-vec (gensym (str "v_" (name s) "_center_")))
                                  :right (tag-vec (gensym (str "v_" (name s) "_right_")))}])
                            arr-syms))
              ;; Scalar broadcast symbols
              scalar-vec-syms (into {}
                                    (map (fn [s] [(symbol (name s))
                                                  (tag-vec (gensym (str "sv_" (name s) "_")))])
                                         scalar-syms))
              scalar-broadcast-bindings (vec (mapcat (fn [[s sv]]
                                                       [sv (list broadcast species-sym (list cast-fn s))])
                                                     scalar-vec-syms))
              ;; Build vec-env: aget references need to resolve based on offset
              ;; We rewrite the body to replace (aget arr (- i 1)) with left vector,
              ;; (aget arr i) with center, (aget arr (+ i 1)) with right
              ;; Build a custom emit that handles offset-indexed agets
              simd-result-sym (tag-vec (gensym "vresult__"))
              ;; Load bindings for shifted arrays
              load-bindings (vec (mapcat
                                  (fn [s]
                                    (let [{:keys [left center right]} (get arr-shifted-syms s)]
                                      [left (list from-array species-sym s (ix- j-sym 1))
                                       center (list from-array species-sym s j-sym)
                                       right (list from-array species-sym s (ix+ j-sym 1))]))
                                  arr-syms))
              ;; Build vec-env for body emission: map each aget pattern to its shifted vector
              ;; We need a custom approach: rewrite body replacing agets with vector refs
              rewrite-body
              (fn rewrite-body [expr]
                (cond
                  ;; (aget arr i) → center vector
                  (and (seq? expr) (descriptor/aget-op? (first expr))
                       (= 3 (count expr))
                       (= idx (descriptor/unwrap-int-cast (nth expr 2))))
                  (let [arr (second expr)]
                    (:center (get arr-shifted-syms (symbol (name arr)))))

                  ;; (aget arr (- i 1)) → left vector
                  (and (seq? expr) (descriptor/aget-op? (first expr))
                       (= 3 (count expr))
                       (let [idx-expr (nth expr 2)]
                         (and (seq? idx-expr)
                              (contains? #{'- 'clojure.core/-} (first idx-expr))
                              (= idx (descriptor/unwrap-int-cast (second idx-expr)))
                              (= 1 (descriptor/unwrap-int-cast (nth idx-expr 2))))))
                  (let [arr (second expr)]
                    (:left (get arr-shifted-syms (symbol (name arr)))))

                  ;; (aget arr (+ i 1)) → right vector
                  (and (seq? expr) (descriptor/aget-op? (first expr))
                       (= 3 (count expr))
                       (let [idx-expr (nth expr 2)]
                         (and (seq? idx-expr)
                              (contains? #{'+ 'clojure.core/+} (first idx-expr))
                              (= idx (descriptor/unwrap-int-cast (second idx-expr)))
                              (= 1 (descriptor/unwrap-int-cast (nth idx-expr 2))))))
                  (let [arr (second expr)]
                    (:right (get arr-shifted-syms (symbol (name arr)))))

                  ;; Number → broadcast
                  (number? expr)
                  (tag-vec (list broadcast species-sym (list cast-fn expr)))

                  ;; Symbol → look up in scalar env
                  (symbol? expr)
                  (or (get scalar-vec-syms (symbol (name expr)))
                      (tag-vec (list broadcast species-sym (list cast-fn expr))))

                  ;; Primitive cast — transparent
                  (and (seq? expr) (contains? #{'double 'float} (first expr)) (= 2 (count expr)))
                  (rewrite-body (second expr))

                  ;; Binary op
                  (and (seq? expr) (= 3 (count expr)) (contains? simd-binary-ops (first expr)))
                  (let [method (simd-binary-ops (first expr))]
                    (tag-vec
                     (list method (rewrite-body (nth expr 1)) (rewrite-body (nth expr 2)))))

                  ;; Unary direct method
                  (and (seq? expr) (= 2 (count expr)) (contains? simd-unary-ops (first expr)))
                  (let [method (simd-unary-ops (first expr))]
                    (tag-vec (list method (rewrite-body (second expr)))))

                  ;; Unary lanewise
                  (and (seq? expr) (= 2 (count expr)) (contains? simd-lanewise-unary-ops (first expr)))
                  (let [op-const (simd-lanewise-unary-ops (first expr))]
                    (tag-vec (list '.lanewise (rewrite-body (second expr)) op-const)))

                  ;; Ternary (fma)
                  (and (seq? expr) (= 4 (count expr)) (contains? simd-ternary-ops (first expr)))
                  (let [method (simd-ternary-ops (first expr))]
                    (tag-vec
                     (list method (rewrite-body (nth expr 1))
                           (rewrite-body (nth expr 2))
                           (rewrite-body (nth expr 3)))))

                  ;; N-ary + or *
                  (and (seq? expr) (> (count expr) 3)
                       (contains? #{'+ '* 'clojure.core/+ 'clojure.core/*} (first expr)))
                  (let [method (simd-binary-ops (first expr))]
                    (clojure.core/reduce (fn [a b] (tag-vec (list method a b)))
                                         (map rewrite-body (rest expr))))

                  :else nil))
              simd-body-expr (rewrite-body body)
              ;; Scalar tail body with j instead of idx
              scalar-body (clojure.walk/postwalk
                           (fn [f] (if (= f idx) j-sym f))
                           body)
              scalar-aset (if cast
                            (list 'clojure.core/aset out j-sym (list cast scalar-body))
                            (list 'clojure.core/aset out j-sym scalar-body))]
          (when simd-body-expr
            (list 'let* [species-sym species-expr
                         lanes-sym (list '.length species-sym)
                         n-sym (list 'int bound)]
                  (list 'do
                        (par/stencil-alias-guard-form out in-arrays)
                        (list 'if (ix<= n-sym 0)
                              out
                              (list 'do
                ;; Fill boundary elements with boundary value
                        (list 'clojure.core/aset out 0 0.0)
                        (list 'clojure.core/aset out (ix- n-sym 1) 0.0)
                ;; SIMD interior loop: [radius, n-radius) with vector width steps
                        (if (seq scalar-broadcast-bindings)
                          (list 'let* (vec scalar-broadcast-bindings)
                                (list 'loop* [j-sym (list 'int radius)]
                                      (list 'if (ix<= (ix+ j-sym lanes-sym) (ix- n-sym radius))
                                            (list 'let* (vec (concat load-bindings [simd-result-sym simd-body-expr]))
                                                  (list '.intoArray simd-result-sym out j-sym)
                                                  (list 'recur (ix+ j-sym lanes-sym)))
                        ;; Scalar tail
                                            (list 'loop* [j-sym j-sym]
                                                  (list 'if (ix< j-sym (ix- n-sym radius))
                                                        (list 'do scalar-aset
                                                              (list 'recur (list 'inc j-sym)))
                                                        out)))))
                          (list 'loop* [j-sym (list 'int radius)]
                                (list 'if (list '<= (list '+ j-sym lanes-sym) (list '- n-sym radius))
                                      (list 'let* (vec (concat load-bindings [simd-result-sym simd-body-expr]))
                                            (list '.intoArray simd-result-sym out j-sym)
                                            (list 'recur (list '+ j-sym lanes-sym)))
                      ;; Scalar tail
                                      (list 'loop* [j-sym j-sym]
                                            (list 'if (list '< j-sym (list '- n-sym radius))
                                                  (list 'do scalar-aset
                                                        (list 'recur (list 'inc j-sym)))
                                                  out)))))))))))))))

(defn compile-segstencil
  "Compile a scheduled SegStencil without recovering fields from source syntax."
  [segstencil]
  (compile-stencil-info
   {:out (:out-sym segstencil)
    :in-arrays (vec (sort-by name (:inputs segstencil)))
    :radius (:radius segstencil)
    :boundary (:boundary segstencil)
    :cast (:cast-fn segstencil)
    :idx (:name (segop/seg-space-reduced-dim (:space segstencil)))
    :bound (:bound (segop/seg-space-reduced-dim (:space segstencil)))
    :body (:lambda segstencil)
    :elem-type (:dtype segstencil)}))

;; ================================================================
;; Pipeline pass: SIMD optimization
;; ================================================================

(defn- effective-min-elements
  "Compute minimum elements for SIMD from hardware registry.
  AVX2 (4 lanes) → 8, AVX-512 (8 lanes) → 16."
  []
  (try
    (hw/init!)
    (max 8 (* 2 (int (hw/simd-lanes :cpu:0 :double))))
    (catch Exception _ 8)))

(def ^:private segop-id-counter (atom 0))

(def ^:dynamic *bound-segops*
  "SegOps owned by the ParallelProgram equation currently being transformed, or nil for direct
   S-expression callers. Consumption is dtype-guarded."
  nil)

(defn simd-pass
  "Pipeline pass: walk an S-expression, replace raster.par/* forms with SIMD code.

  Stats tracked: {:simd-maps N :simd-reduces N :fallback N :fused 0 :skipped-small N}.
  The retained zero-valued :fused field keeps reporting stable after deletion of backend fusion.

  Options:
    :simd? — enable/disable SIMD (default true)
    :min-elements — minimum elements for SIMD (default 8)"
  [form & {:keys [simd? min-elements] :or {simd? true min-elements nil}}]
  (if-not simd?
    (let [p (when (parallel-program/parallel-program? form)
              (parallel-program/validate! form segop/segop-node?))
          source (if p (parallel-program/source-form p) form)]
      {:form (par/expand-par-forms source) :stats {:simd? false}})
    (let [parallel-program (when (parallel-program/parallel-program? form)
                             (parallel-program/validate! form segop/segop-node?))
          source-form (if parallel-program (parallel-program/source-form parallel-program) form)
          min-elements (or min-elements (effective-min-elements))
          stats (atom {:simd-maps 0 :simd-reduces 0 :fallback 0 :fused 0 :skipped-small 0})
            ;; Scheduled SegOps own their checked dtype. Only the raw compatibility door derives a
            ;; dtype from source syntax, because it has no typed equation to consume.
          take-bound (fn [pred]
                       (when-let [so (first (filter pred *bound-segops*))]
                         (swap! stats update :segop-reused (fnil inc 0))
                         so))
          par->segmap (fn [form]
                        (or (take-bound #(instance? raster.compiler.ir.segop.SegMap %))
                            (try
                              (let [par-info (par/extract-par-map-info form)
                                    dtype (or (:elem-type par-info)
                                              (cast->elem-type (:cast par-info))
                                              (out-sym->elem-type (:out par-info))
                                              :double)]
                                (when (System/getProperty "raster.debug.simd-dtype")
                                  (binding [*out* *err*]
                                    (println "[par-simd] compatibility dtype=" dtype
                                             " out=" (:out par-info)
                                             " cast=" (:cast par-info)
                                             " from-meta=" (:elem-type par-info)
                                             " out-tag=" (or (:raster.type/tag (meta (:out par-info)))
                                                             (:tag (meta (:out par-info)))))))
                                (swap! stats update :segop-relowered (fnil inc 0))
                                (let [soac (raster.compiler.ir.soac/par-form->soac
                                            (:out par-info) form (swap! segop-id-counter inc))
                                      segops (raster.compiler.passes.parallel.soac-lower/lower-soac
                                              soac :cpu:0 :dtype dtype)]
                                  (first segops)))
                              ;; A structured unsupported-form refusal may fall back to scalar code.
                              ;; Raw implementation exceptions must escape, as on the GPU boundary.
                              (catch clojure.lang.ExceptionInfo _ nil))))
          par->segred (fn [form]
                        (or (take-bound #(instance? raster.compiler.ir.segop.SegRed %))
                            (try
                              (let [par-info (par/extract-par-reduce-info form)
                                    dtype (or (:elem-type par-info) :double)
                                    sym (gensym "red_")]
                                (swap! stats update :segop-relowered (fnil inc 0))
                                (let [soac (raster.compiler.ir.soac/par-form->soac
                                            sym form (swap! segop-id-counter inc) :dtype dtype)
                                      segops (raster.compiler.passes.parallel.soac-lower/lower-soac
                                              soac :cpu:0 :dtype dtype)]
                                  (first segops)))
                              (catch clojure.lang.ExceptionInfo _ nil))))
          transform
          (fn transform [form]
            (cond
              ;; raster.par/map! — always go through SegOp
              (par/par-map-form? form)
              (let [{:keys [bound out cast offset]} (par/extract-par-map-info form)]
                (if (and (number? bound) (< bound min-elements))
                  (do (swap! stats update :skipped-small inc)
                      ;; Recurse into expanded form to process nested par primitives
                      (transform (par/expand-par-map! form)))
                  (if-let [segmap (par->segmap form)]
                    (if-let [simd-form (segop-simd/compile-segmap segmap out cast
                                                                  :store-offset offset)]
                      (do (swap! stats update :simd-maps inc)
                          simd-form)
                      (do (swap! stats update :fallback inc)
                          ;; Recurse into expanded form to process nested par primitives
                          (transform (par/expand-par-map! form))))
                    ;; SegOp conversion failed — fallback
                    (do (swap! stats update :fallback inc)
                        ;; Recurse into expanded form to process nested par primitives
                        (transform (par/expand-par-map! form))))))

              ;; raster.par/reduce — always go through SegOp
              (par/par-reduce-form? form)
              (let [{:keys [bound]} (par/extract-par-reduce-info form)]
                (if (and (number? bound) (< bound min-elements))
                  (do (swap! stats update :skipped-small inc)
                      (transform (par/expand-par-reduce form)))
                  (if-let [segred (par->segred form)]
                    (if-let [simd-form (segop-simd/compile-segred segred)]
                      (do (swap! stats update :simd-reduces inc)
                          simd-form)
                      (do (swap! stats update :fallback inc)
                          (transform (par/expand-par-reduce form))))
                    (do (swap! stats update :fallback inc)
                        (transform (par/expand-par-reduce form))))))

              ;; raster.par/stencil!
              (par/par-stencil-form? form)
              (let [{:keys [radius]} (par/extract-par-stencil-info form)
                    scheduled (take-bound
                               #(instance? raster.compiler.ir.segop.SegStencil %))
                    simd-form (when scheduled (compile-segstencil scheduled))]
                (cond
                  simd-form
                  (do (swap! stats update :simd-stencils (fnil inc 0))
                      simd-form)

                  scheduled
                  (do (swap! stats update :fallback inc)
                      (par/expand-par-stencil! form))

                  (= 1 radius)
                  (throw (ex-info "radius-one stencil source has no verified TypedSOAC schedule"
                                  {:reason :unscheduled-stencil
                                   :target-dialect :jvm-simd
                                   :form form}))

                  :else
                  (do (swap! stats update :fallback inc)
                      (par/expand-par-stencil! form))))

              ;; raster.par/gather → hardware vgather (flat 4-arg form only)
              (par/par-gather-form? form)
              (let [args (vec (rest form))]
                (if (= 4 (count args))
                  (let [[out src index n] args
                        elem-type (or (out-sym->elem-type out) :double)]
                    (if-let [simd-form (segop-simd/compile-par-gather out src index n elem-type)]
                      (do (swap! stats update :simd-gathers (fnil inc 0))
                          simd-form)
                      (do (swap! stats update :fallback inc)
                          (par/expand-par-gather! form))))
                  (do (swap! stats update :fallback inc)
                      (par/expand-par-gather! form))))

              ;; Explicit-store maps consume their scheduled scalar region even when there is no
              ;; legal SIMD schedule. This keeps reducing-scatter algebra attached to the same
              ;; SegMap while executing an exact sequential read/modify/write loop on the JVM.
              (par/par-map-void-form? form)
              (let [scheduled (take-bound
                               #(and (instance? raster.compiler.ir.segop.SegMap %)
                                     (nil? (:out-sym %))))]
                (if-let [scalar-form (some-> scheduled segop-simd/compile-effect-segmap)]
                  (do (swap! stats update :scalar-effect-maps (fnil inc 0))
                      scalar-form)
                  (do (swap! stats update :fallback inc)
                      (par/expand-par-map-void! form))))

              ;; Correctness boundary for a parallel primitive without a JVM SIMD schedule.
              ;; In particular, a typed segmented contraction projects back to par/contract for
              ;; host execution. Generic sequence recursion would leave its free/contract axis
              ;; binders as unresolved bytecode symbols. Expand through raster.par's canonical
              ;; sequential semantics and retain the counted schedule debt instead. Ordinary
              ;; macroexpand is deliberately wrong here: it rewrites the walker's `.invk` nodes
              ;; into interop forms before the bytecode backend consumes them.
              (par/par-form? form)
              (do (swap! stats update :fallback inc)
                  (macroexpand/macroexpand-all-preserving
                   form
                   ;; Preserve the walker's devirtualized typed calls. Unlike macroexpand-core,
                   ;; this intentionally expands raster.par macros at the correctness boundary.
                   #(and (symbol? %) (.startsWith (name %) "."))))

              ;; let/let* — typed fusion has already made every legal fusion decision before
              ;; scheduling. A backend-local source rewrite would invalidate that certificate and
              ;; force the JVM emitter to invent a second, less-checked schedule.
              (and (seq? form) (contains? #{'let 'let*} (first form)))
              (let [[let-sym bindings & body] form
                    pairs (partition 2 bindings)
                    new-bindings (vec (mapcat (fn [[s e]]
                                                [s (binding [*bound-segops*
                                                             (when parallel-program
                                                               (parallel-program/operations-for-binding
                                                                parallel-program s e))]
                                                     (transform e))])
                                              pairs))
                    new-body (doall
                              (map (fn [expr]
                                     (binding [*bound-segops*
                                               (when parallel-program
                                                 (parallel-program/operations-for-source
                                                  parallel-program expr))]
                                       (transform expr)))
                                   body))]
                (list* let-sym new-bindings new-body))

              ;; do block — recurse explicitly (not via generic seq, for clarity)
              (and (seq? form) (= 'do (first form)))
              (apply list 'do (doall (map transform (rest form))))

              ;; Generic seq — recurse
              (seq? form) (apply list (doall (map transform form)))
              (vector? form) (mapv transform form)
              :else form))]
      {:form (transform source-form)
       :stats @stats})))

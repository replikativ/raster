(ns raster.compiler.ir.par
  "Parallel form predicates, expansion, and analysis for the compiler.

  Pure IR module — no dependency on raster.core or raster.par macros.
  Compiler passes (bytecode, SIMD, GPU, pipeline) require this module
  instead of raster.par to avoid the core → bytecode → par → core cycle.

  Functions fall into three categories:
  - Predicates: par-map-form?, par-reduce-form?, etc.
  - Expanders: expand-par-map!, expand-par-reduce, etc.
  - Info extractors: extract-par-map-info, extract-par-reduce-info, etc."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]))

;; splitmix64 mixing constants — written via parseUnsignedLong because
;; the hex values exceed Long/MAX_VALUE and Clojure would read them as BigInt.
(def SM-GAMMA (Long/parseUnsignedLong "9e3779b97f4a7c15" 16))
(def SM-MIX1  (Long/parseUnsignedLong "bf58476d1ce4e5b9" 16))
(def SM-MIX2  (Long/parseUnsignedLong "94d049bb133111eb" 16))

;; ================================================================
;; Expansion functions (par forms → sequential loop S-expressions)
;; ================================================================

(defn expand-par-active-ids
  "Expand a (raster.par/active-ids! ids n-active n-agents base-seed) form to sequential loop S-expression."
  [form]
  (let [[_ ids-arr n-active n-agents base-seed] form
        i-sym (gensym "i__")]
    (list 'let ['n-active__ (list 'int n-active) 'n-agents__ (list 'long n-agents) 'base__ (list 'long base-seed)]
          (list 'dotimes [i-sym 'n-active__]
                (list 'let ['state__ (list 'unchecked-add 'base__
                                           (list 'unchecked-multiply (list 'long i-sym) SM-GAMMA))
                            's1__ (list 'bit-xor 'state__ (list 'unsigned-bit-shift-right 'state__ 30))
                            's2__ (list 'unchecked-multiply 's1__ SM-MIX1)
                            's3__ (list 'bit-xor 's2__ (list 'unsigned-bit-shift-right 's2__ 27))
                            's4__ (list 'unchecked-multiply 's3__ SM-MIX2)
                            's5__ (list 'bit-xor 's4__ (list 'unsigned-bit-shift-right 's4__ 31))
                            'idx__ (list 'int (list 'mod (list 'bit-and 's5__ Long/MAX_VALUE) 'n-agents__))]
                      (list 'clojure.core/aset ids-arr i-sym 'idx__)))
          ids-arr)))

(defn expand-par-collect!
  "Expand a (raster.par/collect! count-arr arr1 val1 ...) form to sequential S-expr."
  [form]
  (let [[_ count-arr & pairs] form
        slot-sym (gensym "slot__")]
    (list 'let [slot-sym (list 'int (list 'raster.par/atomic-add! count-arr 0 (list 'int 1)))]
          (cons 'do (map (fn [[arr val]] (list 'clojure.core/aset arr slot-sym val))
                         (partition 2 pairs))))))

(defn expand-par-rng-fill
  "Expand a (raster.par/rng-fill! seeds n base-seed) form to sequential loop S-expression."
  [form]
  (let [[_ seeds-arr n base-seed] form
        i-sym (gensym "i__")]
    (list 'let ['n__ (list 'int n) 'base__ (list 'long base-seed)]
          (list 'dotimes [i-sym 'n__]
                (list 'let ['state__ (list 'unchecked-add 'base__
                                           (list 'unchecked-multiply (list 'long i-sym) SM-GAMMA))
                            's1__ (list 'bit-xor 'state__ (list 'unsigned-bit-shift-right 'state__ 30))
                            's2__ (list 'unchecked-multiply 's1__ SM-MIX1)
                            's3__ (list 'bit-xor 's2__ (list 'unsigned-bit-shift-right 's2__ 27))
                            's4__ (list 'unchecked-multiply 's3__ SM-MIX2)
                            's5__ (list 'bit-xor 's4__ (list 'unsigned-bit-shift-right 's4__ 31))]
                      (list 'clojure.core/aset seeds-arr i-sym 's5__)))
          seeds-arr)))

(defn expand-par-stencil!
  "Expand a (raster.par/stencil! out [in-arrays] radius boundary cast idx bound body)
  form to a plain sequential loop S-expression. Used by pipeline passes."
  [form]
  (let [[_ out-sym in-arrays radius boundary cast-fn idx-sym bound-expr body-expr] form
        n-sym (gensym "n__")
        j-sym (gensym "j__")
        aset-expr (if cast-fn
                    (list cast-fn body-expr)
                    body-expr)]
    (list 'let [n-sym (list 'int bound-expr)]
          ;; Fill boundary elements with boundary value
          (list 'do
                (list 'dotimes [j-sym radius]
                      (list 'clojure.core/aset out-sym j-sym boundary)
                      (list 'clojure.core/aset out-sym (list '- n-sym 1 j-sym) boundary))
            ;; Interior loop
                (list 'dotimes [j-sym (list '- n-sym (list '* 2 radius))]
                      (list 'let [idx-sym (list 'int (list '+ j-sym radius))]
                            (list 'clojure.core/aset out-sym idx-sym aset-expr)))
                out-sym))))

(defn expand-par-map!
  "Expand a (raster.par/map! out idx bound cast body) form to an int-counted
  loop* S-expression. Uses loop* directly instead of Clojure's dotimes because
  dotimes promotes the bound to long via (long n), preventing C2 from
  recognizing the loop as a counted loop for auto-vectorization (SuperWord).
  With int counter + int bound, the bytecode compiler emits IF_ICMPGE which
  C2 recognizes as a counted loop pattern.

  Also handles offset variant: (raster.par/map! out idx bound :offset base cast body)"
  [form]
  (if (and (>= (count form) 8) (= :offset (nth form 4)))
    ;; Offset variant
    (let [[_ out-sym i-sym bound-expr _kw base-expr cast-fn body-expr] form
          n-sym (gensym "n__")
          base-sym (gensym "base__")
          aset-expr (if cast-fn
                      (list cast-fn body-expr)
                      body-expr)]
      (list 'let* [n-sym (list 'int bound-expr)
                   base-sym (list 'int base-expr)]
            (list 'loop* [i-sym '(int 0)]
                  (list 'if (list '< i-sym n-sym)
                        (list 'do
                              (list 'clojure.core/aset out-sym (list '+ base-sym i-sym) aset-expr)
                              (list 'recur (list 'clojure.core/unchecked-inc-int i-sym)))
                        out-sym))))
    ;; Standard variant
    (let [[_ out-sym i-sym bound-expr cast-fn body-expr] form
          n-sym (gensym "n__")
          aset-expr (if cast-fn
                      (list cast-fn body-expr)
                      body-expr)]
      (list 'let* [n-sym (list 'int bound-expr)]
            (list 'loop* [i-sym '(int 0)]
                  (list 'if (list '< i-sym n-sym)
                        (list 'do
                              (list 'clojure.core/aset out-sym i-sym aset-expr)
                              (list 'recur (list 'clojure.core/unchecked-inc-int i-sym)))
                        out-sym))))))

(defn expand-par-map2!
  "Expand a (raster.par/map2! out1 out2 idx bound cast body1 body2) form to
  an int-counted loop* S-expression."
  [form]
  (let [[_ out1-sym out2-sym i-sym bound-expr cast-fn body1 body2] form
        n-sym (gensym "n__")
        aset-expr1 (if cast-fn (list cast-fn body1) body1)
        aset-expr2 (if cast-fn (list cast-fn body2) body2)]
    (list 'let* [n-sym (list 'int bound-expr)]
          (list 'loop* [i-sym '(int 0)]
                (list 'if (list '< i-sym n-sym)
                      (list 'do
                            (list 'clojure.core/aset out1-sym i-sym aset-expr1)
                            (list 'clojure.core/aset out2-sym i-sym aset-expr2)
                            (list 'recur (list 'clojure.core/unchecked-inc-int i-sym)))
                      nil)))))

(defn expand-par-butterfly!
  "Expand a (raster.par/butterfly! re im idx half wr wi base) form to an
  int-counted loop* S-expression for paired stride butterfly operations."
  [form]
  (let [[_ re im i-sym half-expr wr wi base-expr] form
        n-sym (gensym "n__")
        base-sym (gensym "base__")
        lo-sym (gensym "lo__")
        hi-sym (gensym "hi__")
        ur-sym (gensym "ur__")
        ui-sym (gensym "ui__")
        rhi-sym (gensym "rhi__")
        ihi-sym (gensym "ihi__")
        wr-sym (gensym "wr__")
        wi-sym (gensym "wi__")
        vr-sym (gensym "vr__")
        vi-sym (gensym "vi__")]
    (list 'let* [n-sym (list 'int half-expr)
                 base-sym (list 'int base-expr)]
          (list 'loop* [i-sym '(int 0)]
                (list 'if (list '< i-sym n-sym)
                      (list 'let* [lo-sym (list '+ base-sym i-sym)
                                   hi-sym (list '+ lo-sym n-sym)
                                   ur-sym (list 'clojure.core/aget re lo-sym)
                                   ui-sym (list 'clojure.core/aget im lo-sym)
                                   rhi-sym (list 'clojure.core/aget re hi-sym)
                                   ihi-sym (list 'clojure.core/aget im hi-sym)
                                   wr-sym (list 'clojure.core/aget wr i-sym)
                                   wi-sym (list 'clojure.core/aget wi i-sym)
                                   vr-sym (list '- (list '* wr-sym rhi-sym) (list '* wi-sym ihi-sym))
                                   vi-sym (list '+ (list '* wr-sym ihi-sym) (list '* wi-sym rhi-sym))]
                            (list 'clojure.core/aset re lo-sym (list '+ ur-sym vr-sym))
                            (list 'clojure.core/aset re hi-sym (list '- ur-sym vr-sym))
                            (list 'clojure.core/aset im lo-sym (list '+ ui-sym vi-sym))
                            (list 'clojure.core/aset im hi-sym (list '- ui-sym vi-sym))
                            (list 'recur (list 'clojure.core/unchecked-inc-int i-sym)))
                      nil)))))

(defn expand-par-reduce
  "Expand a (raster.par/reduce acc init idx bound body) form to an int-counted
  loop* S-expression. See expand-par-map! for rationale."
  [form]
  (let [[_ acc-sym init-expr i-sym bound-expr body-expr] form
        n-sym (gensym "n__")]
    (list 'let* [n-sym (list 'int bound-expr)]
          (list 'loop* [i-sym '(int 0) acc-sym init-expr]
                (list 'if (list '< i-sym n-sym)
                      (list 'recur (list 'clojure.core/unchecked-inc-int i-sym) body-expr)
                      acc-sym)))))

(defn expand-par-scan
  "Expand a (raster.par/scan out acc init idx bound cast body) form to a
  plain sequential loop S-expression. Used by pipeline passes."
  [form]
  (let [[_ out-sym acc-sym init-expr i-sym bound-expr cast-fn body-expr] form
        n-sym (gensym "n__")
        store-expr (if cast-fn
                     (list cast-fn acc-sym)
                     acc-sym)]
    (list 'let [n-sym (list 'int bound-expr)]
          (list 'loop [i-sym 0 acc-sym init-expr]
                (list 'if (list '< i-sym n-sym)
                      (list 'let [acc-sym body-expr]
                            (list 'clojure.core/aset out-sym i-sym store-expr)
                            (list 'recur (list 'inc i-sym) acc-sym))
                      out-sym)))))

(defn expand-par-scan-exclusive
  "Expand a (raster.par/scan-exclusive out acc init idx bound cast body) form
  to a plain sequential loop S-expression. Used by pipeline passes."
  [form]
  (let [[_ out-sym acc-sym init-expr i-sym bound-expr cast-fn body-expr] form
        n-sym (gensym "n__")
        init-store (if cast-fn
                     (list cast-fn init-expr)
                     init-expr)
        store-expr (if cast-fn
                     (list cast-fn acc-sym)
                     acc-sym)]
    (list 'let [n-sym (list 'int bound-expr)]
          (list 'do
                (list 'clojure.core/aset out-sym 0 init-store)
                (list 'loop [i-sym 0 acc-sym init-expr]
                      (list 'if (list '< i-sym n-sym)
                            (list 'let [acc-sym body-expr]
                                  (list 'clojure.core/aset out-sym (list 'inc i-sym) store-expr)
                                  (list 'recur (list 'inc i-sym) acc-sym))
                            out-sym))))))

(defn expand-par-map-void!
  "Expand a (raster.par/map-void! idx bound body) form to a plain
  dotimes loop S-expression. Used by pipeline passes."
  [form]
  (let [[_ i-sym bound-expr body-expr] form
        n-sym (gensym "n__")]
    (list 'let [n-sym (list 'int bound-expr)]
          (list 'dotimes [i-sym n-sym]
                body-expr)
          nil)))

(defn expand-par-scatter!
  "Expand a (raster.par/scatter! ...) form to sequential loop S-expression."
  [form]
  (let [args (rest form)]
    (case (count args)
      4 (let [[out src index n] args
              i-sym (gensym "i__")
              idx-sym (gensym "idx__")]
          (list 'let ['n__ (list 'int n)]
                (list 'dotimes [i-sym 'n__]
                      (list 'let [idx-sym (list 'clojure.core/aget index i-sym)]
                            (list 'clojure.core/aset out idx-sym
                                  (list '+ (list 'clojure.core/aget out idx-sym)
                                        (list 'clojure.core/aget src i-sym)))))
                out))
      5 (let [[out src index n stride] args
              i-sym (gensym "i__")
              d-sym (gensym "d__")
              idx-sym (gensym "idx__")]
          (list 'let ['n__ (list 'int n) 'stride__ (list 'int stride)]
                (list 'dotimes [i-sym 'n__]
                      (list 'let [idx-sym (list 'clojure.core/aget index i-sym)]
                            (list 'dotimes [d-sym 'stride__]
                                  (list 'let ['src-pos__ (list '+ (list '* i-sym 'stride__) d-sym)
                                              'dst-pos__ (list '+ (list '* idx-sym 'stride__) d-sym)]
                                        (list 'clojure.core/aset out 'dst-pos__
                                              (list '+ (list 'clojure.core/aget out 'dst-pos__)
                                                    (list 'clojure.core/aget src 'src-pos__)))))))
                out)))))

(defn expand-par-gather!
  "Expand a (raster.par/gather out src index n [stride]) form to sequential loop."
  [form]
  (let [args (rest form)]
    (case (count args)
      4 (let [[out src index n] args
              i-sym (gensym "i__")]
          (list 'let ['n__ (list 'int n)]
                (list 'dotimes [i-sym 'n__]
                      (list 'clojure.core/aset out i-sym
                            (list 'clojure.core/aget src
                                  (list 'clojure.core/aget index i-sym))))
                out))
      5 (let [[out src index n stride] args
              i-sym (gensym "i__")
              d-sym (gensym "d__")]
          (list 'let ['n__ (list 'int n) 'stride__ (list 'int stride)]
                (list 'dotimes [i-sym 'n__]
                      (list 'let ['sbase__ (list '* (list 'clojure.core/aget index i-sym) 'stride__)
                                  'obase__ (list '* i-sym 'stride__)]
                            (list 'dotimes [d-sym 'stride__]
                                  (list 'clojure.core/aset out (list '+ 'obase__ d-sym)
                                        (list 'clojure.core/aget src (list '+ 'sbase__ d-sym))))))
                out)))))

(defn expand-par-reduce-by-key
  "Expand a (raster.par/reduce-by-key output keys vals n op) form to sequential loop."
  [form]
  (let [[_ output keys vals n op] form
        i-sym (gensym "i__")]
    (list 'let ['n__ (list 'int n)]
          (list 'dotimes [i-sym 'n__]
                (list 'let ['k__ (list 'clojure.core/aget keys i-sym)
                            'v__ (list 'clojure.core/aget vals i-sym)]
                      (list 'clojure.core/aset output 'k__
                            (list (or op '+) (list 'clojure.core/aget output 'k__) 'v__))))
          output)))

(defn expand-par-forms
  "Walk an S-expression tree, expanding all parallel forms to sequential
  loops. Used as a late pass when SIMD/CUDA backends don't handle
  specific forms, ensuring correctness."
  [form]
  (cond
    (and (seq? form) (= 'raster.par/map! (first form)))
    (expand-par-forms (expand-par-map! form))

    (and (seq? form) (= 'raster.par/map2! (first form)))
    (expand-par-forms (expand-par-map2! form))

    (and (seq? form) (= 'raster.par/reduce (first form)))
    (expand-par-forms (expand-par-reduce form))

    (and (seq? form) (= 'raster.par/scan (first form)))
    (expand-par-forms (expand-par-scan form))

    (and (seq? form) (= 'raster.par/scan-exclusive (first form)))
    (expand-par-forms (expand-par-scan-exclusive form))

    (and (seq? form) (= 'raster.par/stencil! (first form)))
    (expand-par-forms (expand-par-stencil! form))

    (and (seq? form) (= 'raster.par/scatter! (first form)))
    (expand-par-forms (expand-par-scatter! form))

    (and (seq? form) (= 'raster.par/gather (first form)))
    (expand-par-forms (expand-par-gather! form))

    (and (seq? form) (= 'raster.par/reduce-by-key (first form)))
    (expand-par-forms (expand-par-reduce-by-key form))

    (and (seq? form) (= 'raster.par/map-void! (first form)))
    (expand-par-forms (expand-par-map-void! form))

    (and (seq? form) (= 'raster.par/rng-fill! (first form)))
    (expand-par-forms (expand-par-rng-fill form))

    (and (seq? form) (= 'raster.par/active-ids! (first form)))
    (expand-par-forms (expand-par-active-ids form))

    (and (seq? form) (= 'raster.par/collect! (first form)))
    (expand-par-forms (expand-par-collect! form))

    (and (seq? form) (= 'raster.par/butterfly! (first form)))
    (expand-par-forms (expand-par-butterfly! form))

    (seq? form) (let [r (apply list (map expand-par-forms form))]
                  (if-let [m (meta form)] (with-meta r m) r))
    (vector? form) (mapv expand-par-forms form)
    :else form))

;; ================================================================
;; Parallel form predicates
;; ================================================================

(defn par-form?
  "Check if a form is a parallel primitive.
  Accepts both fully-qualified and alias forms."
  [form]
  (and (seq? form)
       (contains? #{'raster.par/pmap 'par/pmap
                    'raster.par/map! 'par/map!
                    'raster.par/map2! 'par/map2!
                    'raster.par/reduce 'par/reduce
                    'raster.par/reduce-into 'par/reduce-into
                    'raster.par/scan 'par/scan
                    'raster.par/scan-exclusive 'par/scan-exclusive
                    'raster.par/stencil! 'par/stencil!
                    'raster.par/scatter! 'par/scatter!
                    'raster.par/gather 'par/gather
                    'raster.par/reduce-by-key 'par/reduce-by-key
                    'raster.par/map-void! 'par/map-void!
                    'raster.par/rng-fill! 'par/rng-fill!
                    'raster.par/active-ids! 'par/active-ids!
                    'raster.par/collect! 'par/collect!
                    'raster.par/butterfly! 'par/butterfly!} (first form))))

(defn par-map-form?
  "Check if a form is a raster.par/map! parallel map (imperative, with output buffer)."
  [form]
  (and (seq? form) (contains? #{'raster.par/map! 'par/map!} (first form))))

(defn par-map-pure-form?
  "Check if a form is a raster.par/pmap (pure, no output buffer).
  IR form: (raster.par/pmap idx bound cast body)"
  [form]
  (and (seq? form) (contains? #{'raster.par/pmap 'par/pmap} (first form))))

(defn extract-par-map-pure-info
  "Extract structured info from pure par/map form.
  (raster.par/pmap idx bound cast body)
  Returns {:idx :bound :cast :body} + optional :elem-type from metadata."
  [form]
  ;; Key convention (NOT drift): the element type is carried on AST METADATA under
  ;; the namespaced :raster.type/elem-type, and translated here into the bare
  ;; :elem-type field of the plain info maps handed to the backend. Namespaced on
  ;; metadata, bare on plain data maps — this is the single translation boundary.
  (let [[_ idx bound cast body] form
        elem-type (:raster.type/elem-type (meta form))]
    (cond-> {:idx idx :bound bound :cast cast :body body}
      elem-type (assoc :elem-type elem-type))))

(defn par-reduce-form?
  "Check if a form is a raster.par/reduce parallel reduction."
  [form]
  (and (seq? form) (contains? #{'raster.par/reduce 'par/reduce} (first form))))

(defn par-stencil-form?
  "Check if a form is a raster.par/stencil! parallel stencil."
  [form]
  (and (seq? form) (contains? #{'raster.par/stencil! 'par/stencil!} (first form))))

(defn par-scatter-form?
  "Check if a form is a raster.par/scatter! parallel scatter."
  [form]
  (and (seq? form) (contains? #{'raster.par/scatter! 'par/scatter!} (first form))))

(defn par-gather-form?
  "Check if a form is a raster.par/gather parallel gather."
  [form]
  (and (seq? form) (contains? #{'raster.par/gather 'par/gather} (first form))))

(defn par-reduce-by-key-form?
  "Check if a form is a raster.par/reduce-by-key parallel reduction."
  [form]
  (and (seq? form) (contains? #{'raster.par/reduce-by-key 'par/reduce-by-key} (first form))))

(defn par-reduce-into-form?
  "Check if a form is a raster.par/reduce-into — an internal IR form (like pmap) emitted by the
   resident reduce-fusion pass: a parallel reduction that writes its scalar result into element 0
   of a caller-supplied 1-element output buffer (so it stays device-resident) instead of
   returning a host scalar. Form: (raster.par/reduce-into out-buf acc init idx bound body)."
  [form]
  (and (seq? form) (contains? #{'raster.par/reduce-into 'par/reduce-into} (first form))))

(defn extract-par-reduce-into-info
  "Extract {:out-buf, :reduce-form, :bound} from a reduce-into form. :reduce-form is the
   equivalent plain par/reduce (out-buf dropped) so existing SegRed codegen is reused verbatim."
  [form]
  (when (par-reduce-into-form? form)
    (let [[_ out-buf acc init idx bound body] form]
      {:out-buf out-buf :bound bound
       :reduce-form (with-meta (list 'raster.par/reduce acc init idx bound body) (meta form))})))

(defn par-rng-fill-form?
  "Check if a form is a raster.par/rng-fill! parallel RNG fill."
  [form]
  (and (seq? form) (contains? #{'raster.par/rng-fill! 'par/rng-fill!} (first form))))

(defn par-scan-exclusive-form?
  "Check if a form is a raster.par/scan-exclusive parallel exclusive scan."
  [form]
  (and (seq? form) (contains? #{'raster.par/scan-exclusive 'par/scan-exclusive} (first form))))

(defn par-map-void-form?
  "Check if a form is a raster.par/map-void! side-effect-only parallel map."
  [form]
  (and (seq? form) (contains? #{'raster.par/map-void! 'par/map-void!} (first form))))

(defn par-active-ids-form?
  "Check if a form is a raster.par/active-ids! parallel active-id generation."
  [form]
  (and (seq? form) (contains? #{'raster.par/active-ids! 'par/active-ids!} (first form))))

(defn par-collect-form?
  "Check if a form is a raster.par/collect! scatter-write form."
  [form]
  (and (seq? form) (contains? #{'raster.par/collect! 'par/collect!} (first form))))

(defn par-butterfly-form?
  "Check if a form is a raster.par/butterfly! paired stride transform."
  [form]
  (and (seq? form) (contains? #{'raster.par/butterfly! 'par/butterfly!} (first form))))

(defn par-map2-form?
  "Check if a form is a raster.par/map2! parallel paired map."
  [form]
  (and (seq? form) (contains? #{'raster.par/map2! 'par/map2!} (first form))))

;; ================================================================
;; Scope structure (unified for free-var analysis & substitution)
;; ================================================================

(defn- pl
  "Rebuild a par list form preserving the original form's metadata."
  [form & children]
  (with-meta (apply list children) (meta form)))

(defn par-scope-info
  "Decompose a par form into scope structure for free-variable analysis
   and substitution. Returns:
     {:scoped-syms [sym ...]     ;; symbols bound by this form (loop vars, accumulators)
      :inner-exprs [expr ...]    ;; body expressions that see scoped-syms
      :outer-exprs [expr ...]    ;; expressions in the enclosing scope
      :rebuild (fn [scoped' inner' outer'] -> form)}
   `:rebuild` reconstructs the form from (possibly transformed) scoped-syms,
   inner-exprs, and outer-exprs vectors of the SAME arity as the originals —
   co-located with the decomposition so a new par variant updates ONE place.
   Returns nil for non-par forms or forms without explicit scope structure
   (scatter!, rng-fill!, active-ids!, collect!, reduce-by-key)."
  [form]
  (when (and (seq? form) (symbol? (first form)))
    (let [head (first form)
          n (name head)]
      (cond
        ;; pmap (pure): (head idx bound cast body)
        (par-map-pure-form? form)
        (let [[_ idx bound cast body] form]
          {:scoped-syms [idx] :inner-exprs [body] :outer-exprs [bound cast]
           :rebuild (fn [[idx'] [body'] [bound' cast']]
                      (pl form head idx' bound' cast' body'))})

        ;; map!: standard (head out idx bound cast body)
        ;;    or offset  (head out idx bound :offset base cast body)
        (par-map-form? form)
        (if (and (>= (count form) 8) (= :offset (nth form 4)))
          (let [[_ out idx bound _kw base cast body] form]
            {:scoped-syms [idx] :inner-exprs [body] :outer-exprs [out bound base cast]
             :rebuild (fn [[idx'] [body'] [out' bound' base' cast']]
                        (pl form head out' idx' bound' :offset base' cast' body'))})
          (let [[_ out idx bound cast body] form]
            {:scoped-syms [idx] :inner-exprs [body] :outer-exprs [out bound cast]
             :rebuild (fn [[idx'] [body'] [out' bound' cast']]
                        (pl form head out' idx' bound' cast' body'))}))

        ;; map2!: (head out1 out2 idx bound cast body1 body2)
        (par-map2-form? form)
        (let [[_ out1 out2 idx bound cast body1 body2] form]
          {:scoped-syms [idx] :inner-exprs [body1 body2] :outer-exprs [out1 out2 bound cast]
           :rebuild (fn [[idx'] [body1' body2'] [out1' out2' bound' cast']]
                      (pl form head out1' out2' idx' bound' cast' body1' body2'))})

        ;; reduce: (head acc init idx bound body)
        (par-reduce-form? form)
        (let [[_ acc init idx bound body] form]
          {:scoped-syms [acc idx] :inner-exprs [body] :outer-exprs [init bound]
           :rebuild (fn [[acc' idx'] [body'] [init' bound']]
                      (pl form head acc' init' idx' bound' body'))})

        ;; scan / scan-exclusive: (head out acc init idx bound cast body)
        (or (= n "scan") (= n "scan-exclusive"))
        (let [[_ out acc init idx bound cast body] form]
          {:scoped-syms [acc idx] :inner-exprs [body] :outer-exprs [out init bound cast]
           :rebuild (fn [[acc' idx'] [body'] [out' init' bound' cast']]
                      (pl form head out' acc' init' idx' bound' cast' body'))})

        ;; map-void!: (head idx bound body)
        (par-map-void-form? form)
        (let [[_ idx bound body] form]
          {:scoped-syms [idx] :inner-exprs [body] :outer-exprs [bound]
           :rebuild (fn [[idx'] [body'] [bound']]
                      (pl form head idx' bound' body'))})

        ;; stencil!: (head out [ins] radius boundary cast idx bound body)
        (par-stencil-form? form)
        (let [[_ out ins radius boundary cast idx bound body] form]
          {:scoped-syms [idx] :inner-exprs [body] :outer-exprs [out ins radius boundary cast bound]
           :rebuild (fn [[idx'] [body'] [out' ins' radius' boundary' cast' bound']]
                      (pl form head out' ins' radius' boundary' cast' idx' bound' body'))})

        ;; butterfly!: (head re im idx half wr wi base) — fixed body, no user exprs inside
        (par-butterfly-form? form)
        (let [[_ re im idx half wr wi base] form]
          {:scoped-syms [idx] :inner-exprs [] :outer-exprs [re im half wr wi base]
           :rebuild (fn [[idx'] _ [re' im' half' wr' wi' base']]
                      (pl form head re' im' idx' half' wr' wi' base'))})

        ;; Forms without explicit scope: scatter!, rng-fill!, active-ids!, collect!, reduce-by-key
        :else nil))))

;; ================================================================
;; Parallel form info extractors
;; ================================================================

(defn extract-par-map-info
  "Extract structured info from a par/map! form.
  Standard form:  (par/map! out idx bound cast body)
    Returns {:out, :idx, :bound, :cast, :body}
  Offset form:    (par/map! out idx bound :offset base cast body)
    Returns {:out, :idx, :bound, :cast, :body, :offset base}"
  [form]
  (when (par-map-form? form)
    (let [elem-type (:raster.type/elem-type (meta form))]
      (if (and (>= (count form) 8) (= :offset (nth form 4)))
        ;; Offset variant: (par/map! out idx bound :offset base cast body)
        (let [[_ out idx bound _kw base cast body] form]
          (cond-> {:out out :idx idx :bound bound :cast cast :body body :offset base}
            elem-type (assoc :elem-type elem-type)))
        ;; Standard variant: (par/map! out idx bound cast body)
        (let [[_ out idx bound cast body] form]
          (cond-> {:out out :idx idx :bound bound :cast cast :body body}
            elem-type (assoc :elem-type elem-type)))))))

(defn extract-par-map2-info
  "Extract structured info from a par/map2! form.
  Returns {:out1, :out2, :idx, :bound, :cast, :body1, :body2} or nil."
  [form]
  (when (par-map2-form? form)
    (let [[_ out1 out2 idx bound cast body1 body2] form
          elem-type (:raster.type/elem-type (meta form))]
      (cond-> {:out1 out1 :out2 out2 :idx idx :bound bound :cast cast
               :body1 body1 :body2 body2}
        elem-type (assoc :elem-type elem-type)))))

(defn extract-par-reduce-info
  "Extract structured info from a par/reduce form.
  Returns {:acc, :init, :idx, :bound, :body} or nil."
  [form]
  (when (par-reduce-form? form)
    (let [[_ acc init idx bound body] form
          elem-type (:raster.type/elem-type (meta form))]
      (cond-> {:acc acc :init init :idx idx :bound bound :body body}
        elem-type (assoc :elem-type elem-type)))))

(defn- unwrap-array-sym
  "Unwrap cast wrappers around an array argument: (double arr) → arr, (float arr) → arr;
  then strip metadata to a bare symbol."
  [arr-sym]
  (let [unwrapped (if (and (seq? arr-sym)
                           (contains? #{'double 'float 'int 'long
                                        'clojure.core/double 'clojure.core/float} (first arr-sym))
                           (= 2 (count arr-sym)))
                    (second arr-sym)
                    arr-sym)]
    (if (symbol? unwrapped) (symbol (name unwrapped)) unwrapped)))

(defn collect-aget-arrays
  "Collect array symbols referenced via aget in a body expression.
  Returns a set of array symbols (stripped of metadata).

  Recognizes BOTH surface aget forms — (clojure.core/aget arr idx),
  (raster.arrays/aget arr idx) — AND devirtualized interface calls,
  (.invk <impl> arr idx), via the :raster.op/original op the walker stamps on
  the .invk form. Reading that carried semantic identity (rather than the surface
  op alone) is what keeps INTERMEDIATE arrays — which the walker devirtualizes to
  .invk while it emits primitive clojure.core/aget for typed params — classified
  as array inputs rather than scalars. (Semantic identity is carried as metadata,
  never recovered from the mangled impl name — see CLAUDE.md compiler principles.)"
  [body-expr]
  (cond
    ;; Recognize both bare (aget arr idx) and devirtualized (.invk aget-impl arr idx);
    ;; descriptor/aget-array-sym unwraps casts + strips metadata. Recurse into the
    ;; args too so nested aget reads (and the index expr) are still collected.
    (descriptor/aget-call? body-expr)
    (apply set/union #{(descriptor/aget-array-sym body-expr)}
           (map collect-aget-arrays (descriptor/call-args body-expr)))

    (seq? body-expr)
    (apply set/union #{} (map collect-aget-arrays (rest body-expr)))

    (vector? body-expr)
    (apply set/union #{} (map collect-aget-arrays body-expr))

    :else #{}))

(defn extract-par-stencil-info
  "Extract structured info from a par/stencil! form.
  Returns {:out, :inputs, :radius, :boundary, :cast, :idx, :bound, :body} or nil."
  [form]
  (when (par-stencil-form? form)
    (let [[_ out-sym in-arrays radius boundary cast-fn idx-sym bound-expr body-expr] form
          elem-type (:raster.type/elem-type (meta form))]
      (cond-> {:out out-sym :in-arrays in-arrays :radius radius :boundary boundary
               :cast cast-fn :idx idx-sym :bound bound-expr :body body-expr}
        elem-type (assoc :elem-type elem-type)))))

(defn extract-par-scatter-info
  "Extract structured info from a par/scatter! form.
  Returns {:out, :src, :index, :n, :stride} or nil."
  [form]
  (when (par-scatter-form? form)
    (let [args (rest form)]
      (case (count args)
        4 (let [[out src index n] args]
            {:out out :src src :index index :n n :stride nil})
        5 (let [[out src index n stride] args]
            {:out out :src src :index index :n n :stride stride})
        nil))))

(defn extract-par-reduce-by-key-info
  "Extract structured info from a par/reduce-by-key form."
  [form]
  (when (par-reduce-by-key-form? form)
    (let [[_ output keys vals n op] form]
      {:out output :keys keys :vals vals :n n :op (or op '+)})))

(defn extract-par-rng-fill-info
  "Extract structured info from a par/rng-fill! form."
  [form]
  (when (par-rng-fill-form? form)
    (let [[_ seeds-arr n base-seed] form]
      {:seeds seeds-arr :n n :base-seed base-seed})))

(defn extract-par-scan-exclusive-info
  "Extract structured info from a par/scan-exclusive form."
  [form]
  (when (par-scan-exclusive-form? form)
    (let [[_ out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr] form]
      {:out out-sym :acc acc-sym :init init-expr
       :idx idx-sym :bound bound-expr :cast cast-fn :body body-expr})))

(defn extract-par-map-void-info
  "Extract structured info from a par/map-void! form.
  Returns {:idx, :bound, :body} or nil."
  [form]
  (when (par-map-void-form? form)
    (let [[_ idx-sym bound-expr body-expr] form]
      {:idx idx-sym
       :bound bound-expr
       :body body-expr})))

(defn extract-par-active-ids-info
  "Extract structured info from a raster.par/active-ids! form.
  Returns {:ids, :n-active, :n-agents, :base-seed} or nil."
  [form]
  (when (par-active-ids-form? form)
    (let [[_ ids-arr n-active n-agents base-seed] form]
      {:ids ids-arr :n-active n-active :n-agents n-agents :base-seed base-seed})))

(defn extract-par-collect-info
  "Extract structured info from a raster.par/collect! form.
  Returns {:count-arr sym :pairs [[arr val] ...]} or nil."
  [form]
  (when (par-collect-form? form)
    (let [[_ count-arr & pairs] form]
      {:count-arr count-arr
       :pairs (partition 2 pairs)})))

(defn extract-par-butterfly-info
  "Extract structured info from a raster.par/butterfly! form.
  Returns {:re, :im, :idx, :half, :wr, :wi, :base} or nil."
  [form]
  (when (par-butterfly-form? form)
    (let [[_ re im idx half wr wi base] form]
      {:re re :im im :idx idx :half half :wr wr :wi wi :base base})))

(ns raster.compiler.passes.parallel.gemm-recognize
  "Structural recognizer for a GEMM redomap — a matmul written in the raster
   language as a nested map over C[i,j] with an inner k-reduction of the product
   of two array reads. It extracts the SAME descriptor the BLAS-name-match
   (gpu-plan/match-gemm-call) emits — {:variant :A :B :C :m :k :n :alpha :beta} —
   so both front doors converge on ONE emission path (emit-gemm-tiled) instead of
   the redomap falling through to the naive per-element SegOp kernel.

   This is Futhark's `isTileableRedomap`: the register-tiling precondition is that
   the reduction inputs are INVARIANT along one tile dimension (A invariant in j,
   B invariant in i). That invariance IS the affine-index shape checked here — A
   indexed by (i,p), B by (p,j) — so the variance test falls out of
   `row-major-linear-index?` rather than needing a separate analysis.

   PURE + device-free. CONSERVATIVE by mandate: a false positive would dispatch a
   tiled XMX kernel for a non-GEMM redomap — a silent miscompile — so every
   structural condition is PROVEN or the matcher returns nil. Stage-2 gates any
   match numerically (bit-identical to the naive redomap) as a backstop.

   Stage 1 = the loop-nest (Design A) matcher: a hand-written / lowered nested
   `dotimes` matmul. A Screma-level lift (Design B, catches par/map+par/reduce
   dense) reuses this same index/variance/extraction core."
  (:require [raster.compiler.passes.parallel.patterns :as pat]
            [raster.compiler.core.op-descriptor :as descriptor]))

(def ^:private canonical-add #{'+ 'clojure.core/+ 'raster.numeric/+})
(def ^:private canonical-mul #{'* 'clojure.core/* 'raster.numeric/*})
(def ^:private aget-syms descriptor/aget-ops)
(def ^:private aset-syms descriptor/aset-ops)
(def ^:private value-casts #{'double 'float 'int 'long})

(defn- zero-init?
  "A dot-product accumulator must start at 0 (0, 0.0, (double 0.0), (float 0))."
  [expr]
  (or (and (number? expr) (zero? expr))
      (and (seq? expr) (= 2 (count expr))
           (contains? value-casts (first expr))
           (number? (second expr)) (zero? (second expr)))))

(defn- peel-dotimes
  "(dotimes [sym bound] single-body) → {:sym :bound :body}, else nil."
  [form]
  (when (and (seq? form) (= 'dotimes (first form))
             (vector? (second form)) (= 2 (count (second form))))
    (let [[_ [sym bound] & body] form]
      (when (and (symbol? sym) (= 1 (count body)))
        {:sym sym :bound bound :body (first body)}))))

(defn- aget-affine
  "An array read `(aget arr idx-expr)` → {:arr :idx}, else nil. idx-expr may be
   affine (not bare). Handles BOTH the literal head and a devirtualized
   `(.invk impl arr idx)` whose :raster.op/original is an aget op — so the matcher
   fires whether or not the walker has lowered array reads by this pipeline stage.
   A missed match is only a missed offload (safe-fail), never a miscompile."
  [form]
  (when (seq? form)
    (cond
      ;; literal (aget arr idx)
      (and (contains? aget-syms (first form)) (= 3 (count form)) (symbol? (second form)))
      {:arr (second form) :idx (nth form 2)}
      ;; devirtualized (.invk impl arr idx) with :raster.op/original an aget op
      (contains? aget-syms (descriptor/semantic-op form))
      (let [args (descriptor/call-args form)]
        (when (and (= 2 (count args)) (symbol? (first args)))
          {:arr (first args) :idx (second args)})))))

(defn- a-variant
  "Transpose of the A operand from its index shape, or nil.
     :n → A is m×k, index = (+ (* i k) p)  (row-major (i,p), cols k)
     :t → A is k×m, index = (+ (* p m) i)  (row-major (p,i), cols m)"
  [idx i p m k]
  (cond (pat/row-major-linear-index? idx i p k) :n
        (pat/row-major-linear-index? idx p i m) :t))

(defn- b-variant
  "Transpose of the B operand from its index shape, or nil.
     :n → B is k×n, index = (+ (* p n) j)  (row-major (p,j), cols n)
     :t → B is n×k, index = (+ (* j k) p)  (row-major (j,p), cols k)"
  [idx p j n k]
  (cond (pat/row-major-linear-index? idx p j n) :n
        (pat/row-major-linear-index? idx j p k) :t))

(defn- match-inner-dot
  "The inner k-reduction `(loop [acc 0 p 0] (if (< p k) (recur (+ acc (* (aget A idxA)
   (aget B idxB))) (inc p)) acc))`. Returns {:p :k :ga :gb} (the reduce index/bound and
   the two affine aget operands) or nil. Operand ORDER is as written; the caller
   resolves which is A vs B by index shape."
  [loop-form]
  (when-let [{:keys [acc-init index-sym bound-expr update-expr acc-sym]}
             (pat/match-reduce-loop loop-form)]
    (when (and (zero-init? acc-init)
               (seq? update-expr)
               (contains? canonical-add (descriptor/semantic-op update-expr)))
      (let [[ua ub] (descriptor/call-args update-expr)
            prod (cond (= ua acc-sym) ub (= ub acc-sym) ua)]
        (when (and prod (seq? prod)
                   (contains? canonical-mul (descriptor/semantic-op prod)))
          (let [args (descriptor/call-args prod)]
            (when (= 2 (count args))
              (let [ga (aget-affine (first args))
                    gb (aget-affine (second args))]
                (when (and ga gb)
                  {:p index-sym :k bound-expr :ga ga :gb gb})))))))))

(defn match-gemm-loop-nest
  "Match a nested-`dotimes` GEMM redomap. Returns
     {:variant :A :B :C :m :k :n :alpha :beta}
   (identical to gpu-plan/match-gemm-call — the shared emission contract) or nil.

   Recognized shape (α=1; β∈{0 fresh-store, 1 accumulating-store}):
     (dotimes [i m]
       (dotimes [j n]
         (aset C (+ (* i n) j)
           [(+ (aget C (+ (* i n) j))]                 ; present ⇒ β=1
             (loop [acc 0 p 0]
               (if (< p k)
                 (recur (+ acc (* (aget A idxA) (aget B idxB))) (inc p))
                 acc)))))
   idxA/idxB may carry either transpose (row/col swapped), yielding :nn/:nt/:tn/:tt."
  [form]
  (when-let [{i :sym m :bound inner :body} (peel-dotimes form)]
    (when-let [{j :sym n :bound body :body} (peel-dotimes inner)]
      (when (and (seq? body) (contains? aset-syms (first body)) (= 4 (count body)))
        (let [[_ c-sym idx-c stored] body]
          (when (and (symbol? c-sym)
                     (pat/row-major-linear-index? idx-c i j n))    ;; C[i,j], m×n
            ;; strip an optional store cast, then split off an accumulating (β=1) add.
            (let [uncast (if (and (seq? stored) (= 2 (count stored))
                                  (contains? value-casts (first stored)))
                           (second stored) stored)
                  c-read? (fn [e] (and (seq? e) (contains? aget-syms (first e))
                                       (= 3 (count e)) (= c-sym (second e)) (= idx-c (nth e 2))))
                  [beta inner-loop]
                  (if (and (seq? uncast)
                           (contains? canonical-add (descriptor/semantic-op uncast))
                           (= 2 (count (descriptor/call-args uncast))))
                    (let [[a1 a2] (descriptor/call-args uncast)]
                      (cond (c-read? a1) [1.0 a2]
                            (c-read? a2) [1.0 a1]
                            :else [0.0 uncast]))
                    [0.0 uncast])]
              (when-let [{:keys [p k ga gb]} (match-inner-dot inner-loop)]
                ;; resolve A/B by index shape (product is commutative): the operand whose
                ;; index is (i,p)/(p,i) is A; the one that is (p,j)/(j,p) is B.
                (letfn [(assign [gA gB]
                          (when-let [av (a-variant (:idx gA) i p m k)]
                            (when-let [bv (b-variant (:idx gB) p j n k)]
                              {:variant (keyword (str (name av) (name bv)))
                               :A (:arr gA) :B (:arr gB) :C c-sym
                               :m m :n n :k k :alpha 1.0 :beta beta})))]
                  (or (assign ga gb) (assign gb ga)))))))))))

(defn match-gemm-redomap
  "Match a GEMM redomap as it appears in a let* BINDING VALUE — either a bare
   nested `dotimes`, or a `(do … (dotimes …) result)` wrapper (the matmul writes C
   as a side effect, then yields the result buffer). Returns the match-gemm-loop-nest
   descriptor augmented with :result-sym (the do's tail expr, or C when the store is
   the tail) — the exact shape gpu-plan/rewrite-gemm consumes, so the redomap front
   door reuses the BLAS emission path verbatim. nil if not a GEMM redomap."
  [expr]
  (let [[nest tail] (if (and (seq? expr) (= 'do (first expr)))
                      (let [body (rest expr)]
                        [(some #(when (and (seq? %) (= 'dotimes (first %))) %) body)
                         (last body)])
                      [expr nil])]
    (when nest
      (when-let [d (match-gemm-loop-nest nest)]
        ;; result-sym = the do's tail buffer, unless the tail IS the store dotimes
        ;; itself (no explicit result) — then the output C is the result.
        (assoc d :result-sym (if (and (some? tail) (not= tail nest)) tail (:C d)))))))

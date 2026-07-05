(ns raster.compiler.passes.parallel.loop-lift
  "Lift plain sequential loop forms into explicit raster.par operations.

  This pass is the raw-syntax recovery stage for parallel structure. It scans
  flattened let* bindings, recognizes loop patterns expressed with dotimes or
  loop/recur, and rewrites them into explicit raster.par nodes for downstream
  SIMD and GPU planning.

  Patterns detected:
    1. Map pattern: (dotimes [i n] (aset out i (cast body))) => raster.par/map!
    2. Reduce pattern: (loop [i 0 acc init] (if (< i n) (recur (inc i) body) acc))
                       => raster.par/reduce
    3. Scan pattern: loop/recur accumulation with per-step aset => raster.par/scan

  Safety invariants:
    - Map body must be element-wise (no loop-carried deps)
    - Map output array must not be read across indices (no RAW hazard)
    - All aget calls must use the loop variable as index
    - Reduce/scan loops must have the expected single index/accumulator shape"
  (:require [clojure.walk]
            [raster.ad.purity :as purity]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.patterns :as patterns]))

;; ================================================================
;; Symbol collection
;; ================================================================

;; ================================================================
;; Pattern: dotimes + aset => raster.par/map!
;; ================================================================

(defn- body-element-wise?
  "Check if body is element-wise w.r.t. the loop index.
  Element-wise means:
    - No writes to arrays (no aset except at top level)
    - All aget calls use idx-sym as the index
    - No references to the accumulator (if any)
  Returns true if body is safe for parallel map."
  [body idx-sym acc-sym]
  (letfn [(check [e]
            (cond
              ;; aget: index must match idx-sym
              (and (seq? e)
                   (descriptor/aget-op? (first e))
                   (>= (count e) 3))
              (and (patterns/idx-matches? (nth e 2) idx-sym)
                   (every? #(check %) (rest e)))

              ;; aset inside body = not element-wise (side effects)
              (and (seq? e)
                   (descriptor/aset-op? (first e)))
              false

              ;; Accumulator reference = loop-carried dependency (not for reduces)
              (and acc-sym (symbol? e) (= e acc-sym))
              false

              ;; let bindings: check both bindings and body, adding bound syms as locals
              (and (seq? e) (contains? #{'let 'let*} (first e)))
              (let [[_ binds & body-forms] e]
                (and (every? check (map second (partition 2 binds)))
                     (every? check body-forms)))

              ;; Recurse into sub-expressions
              (seq? e)
              (every? check (rest e))

              (vector? e)
              (every? check e)

              :else true))]
    (check body)))

(defn- no-raw-hazard?
  "Check for RAW (read-after-write) hazard on the output array.
  A hazard exists when the output is read at a different index than the write.
  Reading at the same element index (same idx-sym) is safe for parallel map —
  each thread reads and writes only its own element (e.g. SGD: A[i] += lr * dA[i])."
  [out-sym body idx-sym]
  (letfn [(unsafe-read? [e sym]
            (cond
              ;; aget of out-sym: safe only if index is exactly idx-sym
              (and (seq? e)
                   (descriptor/aget-op? (first e))
                   (>= (count e) 3)
                   (symbol? (second e))
                   (= (name sym) (name (second e))))
              (not (patterns/idx-matches? (nth e 2) idx-sym))

              (seq? e) (some #(unsafe-read? % sym) (rest e))
              (vector? e) (some #(unsafe-read? % sym) e)
              :else false))]
    (not (unsafe-read? body out-sym))))

(defn- dotimes-map-pattern?
  "Check if a form matches the dotimes->map! pattern.
  Returns {:out sym :idx sym :bound expr :cast fn-or-nil :body expr} or nil.

  Matches:
    (dotimes [i n] (aset out i (cast body)))
    (dotimes [i n] (let [...] (aset out i body)))
  And nested let wrappers:
    (let [n_ (int bound)] (dotimes [i n_] (aset out i body)))"
  [form]
  (when-let [{:keys [out-sym index-sym bound-expr cast-fn value-expr] :as _info}
             (patterns/match-dotimes-map-loop form)]
    (when (and (body-element-wise? value-expr index-sym nil)
               (no-raw-hazard? out-sym value-expr index-sym))
      {:out out-sym
       :idx index-sym
       :bound bound-expr
       :cast cast-fn
       :body value-expr})))

;; ================================================================
;; Pattern: nested dotimes => raster.par/map! (flattened 1D)
;; ================================================================

(defn- nested-dotimes-map-pattern?
  "Detect nested dotimes pattern for 2D array operations:
    (dotimes [i R] (dotimes [j C] (aset buf (+ (* i C) j) body)))
    (dotimes [i R] (let [...] (dotimes [j C] (let [...] (aset buf (+ (* i C) j) body)))))
  Only matches when the aset index is row-major linearization (i*C + j),
  ensuring the flattened 1D iteration produces identical results.
  Returns {:out sym :idx sym :bound expr :cast fn-or-nil :body expr} or nil."
  [form]
  (when-let [{:keys [out-sym index-sym bound-expr cast-fn value-expr] :as _info}
             (patterns/match-nested-dotimes-row-major-map form)]
    {:out out-sym
     :idx index-sym
     :bound bound-expr
     :cast cast-fn
     :body value-expr}))

(def ^:private contains-sym?
  "True if `sym` occurs anywhere in `form` (shared helper in patterns)."
  patterns/contains-sym?)

(defn- reduce-init-elem-type
  "Element type of a reduce accumulator init, or nil when it cannot be
  proven: a (float x)/(double x) cast, a typed numeric literal, or a
  symbol carrying a :raster.type/tag (the walked/AD-prep dialect stamps
  binding symbols). nil = the lift DECLINES (conservative bail — a missed
  optimization, never a throw mid-pass)."
  [init]
  (cond
    (and (seq? init) (contains? #{'float 'clojure.core/float} (first init))) :float
    (and (seq? init) (contains? #{'double 'clojure.core/double} (first init))) :double
    (number? init) (condp instance? init
                     Double :double
                     Float :float
                     Long :long
                     Integer :int
                     nil)
    (symbol? init) (case (or (:raster.type/tag (meta init)) (:tag (meta init)))
                     float :float
                     double :double
                     long :long
                     int :int
                     nil)
    :else nil))

(defn- form-tag-elem-type
  "Element type from a form's :raster.type/tag stamp (walked dialect), or
  nil when unstamped. Emitters/passes READ stamps, never re-infer."
  [form]
  (case (:raster.type/tag (meta form))
    float :float
    double :double
    long :long
    int :int
    nil))

(defn- acc-cast-elem-type
  "Element type evidenced by a cast on the accumulator reference inside the
  update body — recur-LUB widening spells the accumulator as (double acc) /
  (float acc) even when the init literal is integer. nil when the acc is
  referenced bare (or not found)."
  [update-expr acc-sym]
  (let [found (atom nil)
        casts {'double :double 'clojure.core/double :double
               'float :float 'clojure.core/float :float}
        walk (fn walk [e]
               (when (seq? e)
                 (if (and (contains? casts (first e))
                          (= 2 (count e))
                          (= acc-sym (second e)))
                   (reset! found (get casts (first e)))
                   (run! walk (descriptor/call-args e)))))]
    (walk update-expr)
    @found))

(def ^:private read-array-elem-types
  "Element types of the arrays READ inside a body (shared helper in patterns)."
  patterns/read-array-elem-types)

(defn- reduce-elem-type
  "The provable accumulator element type of a reduce loop, or nil to DECLINE.
  Independent evidence sources — the body's :raster.type/tag stamp, an
  accumulator cast in the update ((double acc), recur-LUB spelling), the
  init expression, and the element types of the arrays the body READS —
  must AGREE wherever known:
    - under recur-LUB widening (init 0, double accumulation) or parametric
      :dtype narrowing (init 0.0 in a T=float kernel) the init literal alone
      is NOT authoritative — lifting with a contradicted type truncates or
      miscasts (integer-init-accumulator / jit-aot-differential regressions);
    - a MIXED-precision reduce (double accumulator over float[] reads) must
      decline outright: the SIMD segred loads arrays at the accumulator
      dtype ([F→[D ClassCastException)."
  [update-expr acc-sym acc-init]
  (let [known (remove nil? (concat [(form-tag-elem-type update-expr)
                                    (acc-cast-elem-type update-expr acc-sym)
                                    (reduce-init-elem-type acc-init)]
                                   (read-array-elem-types update-expr)))]
    (when (and (seq known) (apply = known))
      (first known))))

(defn- reduction-update-ok?
  "The update body must be (⊕ acc e) / (⊕ e acc) where ⊕ is a registered
   commutative monoid (associative AND commutative — descriptor/commutative-
   monoid-op?) and the non-acc operand e is acc-free. This is the soundness gate
   that prevents lifting a non-associative/non-commutative recurrence (e.g.
   (- acc e)) or a serially-dependent body into a reordered multi-accumulator
   SIMD reduce."
  [full-body acc-sym]
  (let [op   (descriptor/semantic-op full-body)
        args (vec (descriptor/call-args full-body))]
    (and op
         (= 2 (count args))
         (descriptor/commutative-monoid-op? op)
         (let [[a b] args]
           (or (and (patterns/acc-ref? a acc-sym) (not (contains-sym? b acc-sym)))
               (and (patterns/acc-ref? b acc-sym) (not (contains-sym? a acc-sym))))))))

(defn- pure-op?
  "True only if `op` is *proven* pure by the beichte effect classifier
   (raster.ad.purity), applied to the SEMANTIC operator. This queries the
   shared effect registry — no ad-hoc symbol whitelist in this pass — so it is
   namespace-precise (resolves the actual var / honors the !-convention) and
   complete-by-construction (every raster.numeric/raster.math public is
   registered pure; new ops are picked up automatically). :impure and :unknown
   both fail closed → conservative bail. nil op (unrecognized call shape) fails."
  [op]
  (and op (= :pure (purity/pure-op? op))))

(defn- pure-elem?
  "True if expr is a pure function of the index and loop-invariants. Each call's
   purity is decided by beichte on the op RECOVERED from .invk metadata via
   descriptor/semantic-op — never by pure-sexp? on the .invk form itself (which
   blanket-trusts .invk and would hide an impure devirtualized op). Unknown or
   impure ops bail, so a reordered/parallel reduction never changes behavior."
  [expr]
  (cond
    (seq? expr)    (and (pure-op? (descriptor/semantic-op expr))
                        (every? pure-elem? (descriptor/call-args expr)))
    (vector? expr) (every? pure-elem? expr)
    :else          true))

(defn- loop-reduce-pattern?
  "Check if a form matches the loop->reduce pattern.
  Returns {:acc sym :init expr :idx sym :bound expr :body expr} or nil.

  A loop/recur reduction is lifted to a (reordered, multi-accumulator) SIMD
  par/reduce ONLY when every precondition below is *proven*; otherwise it is
  left as a scalar loop. The gates are conservative (bail when unsure), so a
  false negative costs only a missed optimization — never a wrong result."
  [form]
  (when (seq? form)
    (cond
      (contains? #{'loop 'loop*} (first form))
      (when-let [{:keys [acc-sym acc-init index-sym then-branch else-expr update-expr bound-expr]} (patterns/match-reduce-loop form)]
        (when (and
               ;; (1) no dropped effects: the consequent is a DIRECT recur, not a
               ;;     do/let that could carry side effects or hidden acc-deps.
               (seq? then-branch) (= 'recur (first then-branch))
               ;; (2) bound is loop-invariant — must not reference index or acc.
               (not (contains-sym? bound-expr index-sym))
               (not (contains-sym? bound-expr acc-sym))
               ;; (3) shape: update is (⊕ acc e), ⊕ reassoc-tolerant, e acc-free.
               (reduction-update-ok? update-expr acc-sym)
               ;; (4) e is pure, so reordering the reduction is observationally safe.
               (pure-elem? update-expr)
               ;; (5) all array reads are at the index; no aset; no cross-index reads.
               (body-element-wise? update-expr index-sym nil)
               ;; (6) the exit value depends only on acc + invariants, never the
               ;;     index (which would otherwise escape its loop scope).
               (not (contains-sym? else-expr index-sym))
               ;; (7) the accumulator element type is provable AND consistent
               ;;     across body tag / acc cast / init (reduce-elem-type) —
               ;;     emission needs it for the par/reduce elem-type metadata,
               ;;     and a contradicted init (recur-LUB widening, parametric
               ;;     narrowing) must decline, not truncate.
               (some? (reduce-elem-type update-expr acc-sym acc-init)))
          {:acc acc-sym
           :init acc-init
           :idx index-sym
           :bound bound-expr
           :body update-expr
           :elem-type (reduce-elem-type update-expr acc-sym acc-init)
           :post-fn (when (not= else-expr acc-sym) else-expr)}))

      (contains? #{'let 'let*} (first form))
      (let [[_ bindings-vec & body-exprs] form]
        (when (and (= 2 (count bindings-vec)) (= 1 (count body-exprs)))
          (let [[n-sym bound-wrapper] (take 2 bindings-vec)
                inner (first body-exprs)
                actual-bound (if (and (seq? bound-wrapper)
                                      (= 'int (first bound-wrapper))
                                      (= 2 (count bound-wrapper)))
                               (second bound-wrapper)
                               bound-wrapper)]
            (when-let [info (loop-reduce-pattern? inner)]
              (if (= (:bound info) n-sym)
                (assoc info :bound actual-bound)
                info))))))))

(defn- loop-scan-pattern?
  "Check if a form matches the loop->scan pattern.
  Returns {:out sym :acc sym :init expr :idx sym :bound expr :cast fn-or-nil :body expr}
  or nil."
  [form]
  (when (seq? form)
    (cond
      (contains? #{'loop 'loop*} (first form))
      (when-let [{:keys [out-sym else-expr acc-sym acc-init index-sym bound-expr cast-fn acc-next-expr]} (patterns/match-scan-loop form)]
        (when (= out-sym else-expr)
          {:out else-expr
           :acc acc-sym
           :init acc-init
           :idx index-sym
           :bound bound-expr
           :cast cast-fn
           :body acc-next-expr}))

      (contains? #{'let 'let*} (first form))
      (let [[_ bindings-vec & body-exprs] form]
        (when (and (= 2 (count bindings-vec)) (= 1 (count body-exprs)))
          (let [[n-sym bound-wrapper] (take 2 bindings-vec)
                inner (first body-exprs)
                actual-bound (if (and (seq? bound-wrapper)
                                      (= 'int (first bound-wrapper))
                                      (= 2 (count bound-wrapper)))
                               (second bound-wrapper)
                               bound-wrapper)]
            (when-let [info (loop-scan-pattern? inner)]
              (if (= (:bound info) n-sym)
                (assoc info :bound actual-bound)
                info)))))

      :else nil)))

;; ================================================================
;; Main pass: lift loop forms in let* bindings
;; ================================================================

(defn- empty-stats []
  {:maps-detected 0 :reduces-detected 0 :scans-detected 0})

(defn- add-stats
  [left right]
  (merge-with + left right))

(defn- cast->elem-type-kw [cast]
  (case cast
    float :float
    double :double
    long :long
    int :int
    (nil) nil
    nil))

(defn- out-sym->elem-type
  "Infer elem-type from output symbol's type tag metadata.
  Used when cast is nil (e.g., in-place mutation like SGD updates)."
  [out-sym]
  (when (symbol? out-sym)
    (case (or (:raster.type/tag (meta out-sym)) (:tag (meta out-sym)))
      floats :float
      doubles :double
      longs :long
      ints :int
      nil)))

(defn- map-lift-form
  [{:keys [out idx bound cast body]}]
  (let [form (list 'raster.par/map! out idx bound cast body)
        et (or (cast->elem-type-kw cast) (out-sym->elem-type out))]
    (if et (with-meta form {:raster.type/elem-type et}) form)))

(defn- scan-lift-form
  [{:keys [out acc init idx bound cast body]}]
  (let [form (list 'raster.par/scan out acc init idx bound cast body)
        et (or (cast->elem-type-kw cast) (out-sym->elem-type out))]
    (if et (with-meta form {:raster.type/elem-type et}) form)))

(defn- reduce-lift-form
  [{:keys [acc init idx bound body elem-type post-fn]}]
  (let [reduce-form (list 'raster.par/reduce acc init idx bound body)
        ;; Elem-type is provable by construction: loop-reduce-pattern? gate (7)
        ;; declined any loop reduce-elem-type cannot classify consistently.
        et (or elem-type
               (throw (ex-info (str "loop-lift: unreachable — reduce init `"
                                    (pr-str init) "` passed the pattern gate "
                                    "but has no inferable element type.")
                               {:init init})))
        reduce-form (with-meta reduce-form {:raster.type/elem-type et})]
    (if post-fn
      (clojure.walk/postwalk-replace {acc reduce-form} post-fn)
      reduce-form)))

(defn- let-form?
  [form]
  (and (seq? form) (contains? #{'let 'let*} (first form))))

(defn- classify-loop-form
  [expr]
  (cond
    (nested-dotimes-map-pattern? expr) :nested-map
    (dotimes-map-pattern? expr) :map
    (loop-scan-pattern? expr) :scan
    (loop-reduce-pattern? expr) :reduce
    (let-form? expr) :nested-let
    :else :plain))

(defmulti rewrite-binding-expr
  "Rewrite a binding expression according to the loop shape it encodes."
  (fn [_expr] (classify-loop-form _expr)))

(defmethod rewrite-binding-expr :nested-map
  [expr]
  {:expr (map-lift-form (nested-dotimes-map-pattern? expr))
   :stats {:maps-detected 1 :reduces-detected 0 :scans-detected 0}})

(defmethod rewrite-binding-expr :map
  [expr]
  {:expr (map-lift-form (dotimes-map-pattern? expr))
   :stats {:maps-detected 1 :reduces-detected 0 :scans-detected 0}})

(defmethod rewrite-binding-expr :scan
  [expr]
  {:expr (scan-lift-form (loop-scan-pattern? expr))
   :stats {:maps-detected 0 :reduces-detected 0 :scans-detected 1}})

(defmethod rewrite-binding-expr :reduce
  [expr]
  {:expr (reduce-lift-form (loop-reduce-pattern? expr))
   :stats {:maps-detected 0 :reduces-detected 1 :scans-detected 0}})

(declare lift-parallel-forms)

(defmethod rewrite-binding-expr :nested-let
  [expr]
  (let [inner (lift-parallel-forms expr)]
    {:expr (:form inner)
     :stats (:stats inner)}))

(defmethod rewrite-binding-expr :plain
  [expr]
  {:expr expr :stats (empty-stats)})

(declare lift-parallel-forms)

(defn- lift-expr
  "Try to lift a single expression that may contain walker-devirtualized .invk
  ops. The pattern matchers are metadata-aware (they recover op identity from
  :raster.op/original via descriptor/semantic-op), so matching runs DIRECTLY on
  the original form — no rewriting. If it classifies as a liftable loop, emit the
  par form whose body is the ORIGINAL .invk subforms, preserving the typed
  invokedynamic dispatch the backend relies on. Otherwise return orig unchanged,
  recursing into nested let/loop.

  This is why we never globally denorm: rewriting .invk → bare ops would both
  de-devirtualize surrounding calls (backend type miscompiles, e.g. [D as [S)
  and drop the invokedynamic path for lifted bodies (boxed IFn dispatch)."
  [orig]
  (let [k (classify-loop-form orig)]
    (cond
      (contains? #{:nested-map :map :scan :reduce} k)
      (rewrite-binding-expr orig)

      (let-form? orig)
      (let [inner (lift-parallel-forms orig)]
        {:expr (:form inner) :stats (:stats inner)})

      :else
      {:expr orig :stats (empty-stats)})))

(defn lift-parallel-forms
  "Scan a flat let* form for loop patterns that can be lifted into raster.par nodes.

  Detects:
    - nested dotimes + row-major aset => raster.par/map! (flattened 1D)
    - dotimes + aset => raster.par/map! (including in-place updates like SGD)
    - loop/recur accumulation => raster.par/reduce
    - loop/recur with aset + accumulator => raster.par/scan

  Returns {:form new-form :stats {:maps-detected N :reduces-detected N :scans-detected N}}"
  [let-form]
  (if-not (let-form? let-form)
    ;; Not a let* — still try to lift a bare loop in expression/body position
    ;; (e.g. a deftm whose whole body is a reduction loop).
    (let [{new-expr :expr new-stats :stats} (lift-expr let-form)]
      {:form new-expr :stats new-stats})
    (let [[let-sym bindings-vec & body-exprs] let-form
          pairs (vec (partition 2 bindings-vec))
          {:keys [pairs stats]}
          (reduce (fn [{:keys [pairs stats]} [sym expr]]
                    (let [{new-expr :expr new-stats :stats} (lift-expr expr)]
                      {:pairs (conj pairs [sym new-expr])
                       :stats (add-stats stats new-stats)}))
                  {:pairs [] :stats (empty-stats)}
                  pairs)
          {:keys [body stats]}
          (reduce (fn [{:keys [body stats]} expr]
                    (let [{new-expr :expr new-stats :stats} (lift-expr expr)]
                      {:body (conj body new-expr)
                       :stats (add-stats stats new-stats)}))
                  {:body [] :stats stats}
                  body-exprs)]
      {:form (list* let-sym (vec (mapcat identity pairs)) body)
       :stats stats})))

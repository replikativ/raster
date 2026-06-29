(ns raster.compiler.passes.parallel.par-fusion
  "Fusion of parallel forms across function boundaries and form types.

  Three fusion strategies:
    1. Map→reduce fusion: eliminate intermediate arrays
    2. Horizontal fusion: merge independent maps with same bound
    3. Binding reorder: schedule same-bound par forms adjacently

  Runs BEFORE backend-specific passes (SIMD/CUDA) to maximize
  optimization opportunities. Uses liveness analysis to verify
  intermediate elimination is safe.

  Design informed by Futhark's fusion via dependency graphs."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.descriptors :as desc]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]
            [raster.compiler.passes.parallel.schedule-support :as schedule-support]
            [clojure.set :as set]))

;; ================================================================
;; Symbol collection utilities
;; ================================================================

(def ^:private free-syms
  "Collect unqualified free symbols from an S-expression."
  util/free-syms-flat)

(defn- all-symbol-uses
  "Collect ALL symbol uses in an S-expression (including qualified)."
  [e]
  (cond (symbol? e) #{e}
        (seq? e)    (apply set/union #{} (map all-symbol-uses e))
        (vector? e) (apply set/union #{} (map all-symbol-uses e))
        :else       #{}))

;; ================================================================
;; 1. Map→Reduce Fusion
;; ================================================================

(defn- aget-reads-sym?
  "Check if body reads from sym via aget."
  [body sym]
  (cond
    (and (seq? body)
         (descriptor/aget-op? (first body))
         (>= (count body) 3)
         (symbol? (second body))
         (= (name sym) (name (second body))))
    true

    (seq? body) (some #(aget-reads-sym? % sym) (rest body))
    (vector? body) (some #(aget-reads-sym? % sym) body)
    :else false))

(defn- sym-used-outside?
  "Check if sym is used in any of the remaining bindings or body expressions,
  beyond the map that produces it and the reduce that reads it."
  [sym remaining-pairs body-exprs map-idx reduce-idx]
  (let [exclude-set #{map-idx reduce-idx}
        all-uses (apply set/union #{}
                        (concat
                         (map-indexed (fn [i [_s e]]
                                        (if (contains? exclude-set i)
                                          #{}
                                          (all-symbol-uses e)))
                                      remaining-pairs)
                         (map all-symbol-uses body-exprs)))]
    (contains? all-uses sym)))

(defn fuse-map-reduce
  "Fuse consecutive map! → reduce pairs in a let* binding vector.

  Before:
    tmp (raster.par/map! tmp i n double (* (aget a i) (aget a i)))
    result (raster.par/reduce acc 0.0 j n (+ acc (aget tmp j)))

  After:
    result (raster.par/reduce acc 0.0 j n (+ acc (* (aget a j) (aget a j))))

  Returns {:bindings new-bindings :fused count} or nil if no fusion."
  [bindings-vec body-exprs]
  (let [pairs (vec (partition 2 bindings-vec))
        n (count pairs)]
    (loop [i 0 result [] fused 0 skip-set #{}]
      (if (>= i n)
        (when (pos? fused)
          {:bindings (vec (mapcat identity result))
           :fused fused})
        (if (contains? skip-set i)
          (recur (inc i) result fused skip-set)
          (let [[sym expr] (nth pairs i)]
            (if-let [map-info (desc/map-form-info expr)]
              ;; Found a map! — look for a reduce that reads from sym
              (let [tmp-sym (:out map-info)
                    ;; Search remaining pairs for a reduce reading tmp
                    reduce-idx
                    (first
                     (keep-indexed
                      (fn [j [_s e]]
                        (when (> j i)
                          (when-let [reduce-info (desc/reduce-form-info e)]
                            (when (aget-reads-sym? (:body reduce-info) tmp-sym)
                              j))))
                      pairs))
                    ;; Check liveness: tmp must not be used elsewhere
                    safe? (and reduce-idx
                               (not (sym-used-outside? tmp-sym pairs body-exprs i reduce-idx)))]
                (if safe?
                  ;; Fuse: inline map body into reduce body
                  (let [[rsym rexpr] (nth pairs reduce-idx)
                        reduce-info (desc/reduce-form-info rexpr)
                        fused-body (fusion-support/substitute-aget
                                    (:body reduce-info) tmp-sym
                                    (:idx map-info) (:idx reduce-info)
                                    (:body map-info))
                        fused-reduce (list 'raster.par/reduce
                                           (:acc reduce-info) (:init reduce-info)
                                           (:idx reduce-info) (:bound reduce-info)
                                           fused-body)]
                    ;; Skip the map binding, replace the reduce binding
                    (recur (inc i)
                           (conj result [rsym fused-reduce])
                           (inc fused)
                           (conj skip-set reduce-idx)))
                  ;; Can't fuse, keep as-is
                  (recur (inc i) (conj result [sym expr]) fused skip-set)))
              ;; Not a map — keep as-is
              (recur (inc i) (conj result [sym expr]) fused skip-set))))))))

;; ================================================================
;; 2. Horizontal Fusion
;; ================================================================

(defn fuse-horizontal-maps
  "Fuse independent raster.par/map! forms with the same bound into a
  single map that writes multiple outputs.

  Before:
    out1 (raster.par/map! out1 i n double (Math/sin (aget a i)))
    out2 (raster.par/map! out2 i n double (Math/cos (aget a i)))

  After (as do block):
    _fused (raster.par/map! out1 i n double
             (do (aset out2 i (double (Math/cos (aget a i))))
                 (Math/sin (aget a i))))

  Returns {:bindings new-bindings :fused count} or nil if no fusion."
  [bindings-vec]
  (let [pairs (vec (partition 2 bindings-vec))
        n (count pairs)]
    (when (>= n 2)
      (loop [i 0 result [] fused 0 skip-set #{}]
        (if (>= i n)
          (when (pos? fused)
            {:bindings (vec (mapcat identity result))
             :fused fused})
          (if (contains? skip-set i)
            (recur (inc i) result fused skip-set)
            (let [[sym expr] (nth pairs i)]
              (if-let [info-i (desc/map-form-info expr)]
                ;; Find all subsequent independent maps with same bound
                (let [;; Collect fusable partners
                      partners
                      (vec (keep-indexed
                            (fn [j [sym-j expr-j]]
                              (when (and (> j i)
                                         (not (contains? skip-set j)))
                                (when-let [info-j (desc/map-form-info expr-j)]
                                  (when (schedule-support/same-bound? (:bound info-i) (:bound info-j))
                                     ;; Check independence: j doesn't read from i's output
                                    (let [free-j (free-syms expr-j)]
                                      (when (not (contains? free-j (:out info-i)))
                                        {:index j :sym sym-j :info info-j}))))))
                            pairs))]
                  (if (seq partners)
                    ;; Fuse: primary map body becomes a do block that also writes partner outputs
                    (let [idx (:idx info-i)
                          ;; Build side-effect writes for each partner
                          side-effects
                          (mapv (fn [{:keys [info]}]
                                  (fusion-support/emit-side-effect-aset
                                   (:out info) idx (:idx info) (:cast info) (:body info)))
                                partners)
                          ;; Primary body is last in do block
                          fused-body (list* 'do (concat side-effects [(:body info-i)]))
                          ;; Preserve :raster.type/elem-type from the primary form so
                          ;; downstream SIMD codegen picks the right Vector species
                          ;; (else falls back to :double — breaks f32 with [F → [D cast).
                          orig-meta (meta expr)
                          fused-expr (cond-> (list 'raster.par/map! (:out info-i) idx
                                                   (:bound info-i) (:cast info-i) fused-body)
                                       orig-meta (with-meta orig-meta))
                          new-skip (set (map :index partners))]
                      (recur (inc i)
                             (conj result [sym fused-expr])
                             (+ fused (count partners))
                             (set/union skip-set new-skip)))
                    ;; No partners found
                    (recur (inc i) (conj result [sym expr]) fused skip-set)))
                ;; Not a map
                (recur (inc i) (conj result [sym expr]) fused skip-set)))))))))

(defn- collect-aset-targets
  "Collect symbols used as the array argument to aset in an S-expression body."
  [body]
  (cond
    (and (seq? body)
         (descriptor/aset-op? (first body))
         (>= (count body) 3)
         (symbol? (second body)))
    (set/union #{(second body)} (apply set/union #{} (map collect-aset-targets (rest body))))

    (seq? body)    (apply set/union #{} (map collect-aset-targets body))
    (vector? body) (apply set/union #{} (map collect-aset-targets body))
    :else #{}))

(defn fuse-horizontal-map-voids
  "Fuse independent raster.par/map-void! forms with the same bound into a
  single map-void! with a do body.

  Two map-void! forms are independent when the arrays written by form A
  (aset targets) do not appear as free symbols in form B's body, and vice
  versa. This is a conservative sound check — it may miss some fusions
  but will never create incorrect ones.

  Before:
    _ (raster.par/map-void! i n (aset output i (compute-output ...)))
    _ (raster.par/map-void! i n (aset other  i (compute-other ...)))

  After:
    _ (raster.par/map-void! i n
        (do (aset output i (compute-output ...))
            (aset other  i (compute-other ...))))

  Returns {:bindings new-bindings :fused count} or nil if no fusion."
  [bindings-vec]
  (let [pairs (vec (partition 2 bindings-vec))
        n     (count pairs)]
    (when (>= n 2)
      (loop [i 0 result [] fused 0 skip-set #{}]
        (if (>= i n)
          (when (pos? fused)
            {:bindings (vec (mapcat identity result))
             :fused    fused})
          (if (contains? skip-set i)
            (recur (inc i) result fused skip-set)
            (let [[sym expr] (nth pairs i)]
              (if-let [info-i (desc/map-void-form-info expr)]
                (let [targets-i (collect-aset-targets (:body info-i))
                      free-i    (free-syms (:body info-i))
                      partners
                      (vec (keep-indexed
                            (fn [j [_sym-j expr-j]]
                              (when (and (> j i)
                                         (not (contains? skip-set j)))
                                (when-let [info-j (desc/map-void-form-info expr-j)]
                                  (let [targets-j (collect-aset-targets (:body info-j))
                                        free-j    (free-syms (:body info-j))]
                                    (when (and (schedule-support/same-bound? (:bound info-i) (:bound info-j))
                                                ;; A's writes don't conflict with B's reads and vice versa
                                               (empty? (set/intersection targets-i free-j))
                                               (empty? (set/intersection targets-j free-i)))
                                      {:index j :info info-j})))))
                            pairs))]
                  (if (seq partners)
                    (let [idx      (:idx info-i)
                          ;; Adjust partner index vars to use primary idx
                          partner-bodies
                          (mapv (fn [{:keys [info]}]
                                  (fusion-support/rewrite-index-sym
                                   (:body info) (:idx info) idx))
                                partners)
                          fused-body (list* 'do (concat [(:body info-i)] partner-bodies))
                          fused-expr (list 'raster.par/map-void! idx (:bound info-i) fused-body)
                          new-skip   (set (map :index partners))]
                      (recur (inc i)
                             (conj result [sym fused-expr])
                             (+ fused (count partners))
                             (set/union skip-set new-skip)))
                    (recur (inc i) (conj result [sym expr]) fused skip-set)))
                (recur (inc i) (conj result [sym expr]) fused skip-set)))))))))

;; ================================================================
;; 3. Binding Reorder for Cross-Function Fusion
;; ================================================================

(defn reorder-for-fusion
  "Reorder let* bindings to group par forms with same bound together,
  while respecting data dependencies. Enables fusion on forms from
  different inlined functions.

  Returns reordered bindings vector, or nil if no reorder needed."
  [bindings-vec]
  (let [pairs (vec (partition 2 bindings-vec))
        n (count pairs)
        dep-graph (schedule-support/build-symbol-dependency-graph pairs free-syms)
        ;; Classify: par forms vs non-par
        par-indices (filterv (fn [i]
                               (let [[_sym expr] (nth pairs i)]
                                 (desc/par-form? expr)))
                             (range n))
        par-groups (schedule-support/group-by-bound
                    par-indices
                    (fn [i]
                      (let [[_sym expr] (nth pairs i)]
                        (desc/bound-expr expr))))
        ;; Check if any group has >1 member (worth reordering)
        groups par-groups
        multi-groups (filter (fn [[_ idxs]] (> (count idxs) 1)) groups)]
    (when (seq multi-groups)
      (let [;; Process: non-par first (in order), then par groups
            non-par-indices (filterv (fn [i]
                                       (let [[_ expr] (nth pairs i)]
                                         (not (desc/par-form? expr))))
                                     (range n))
            grouped-par-indices (vec (mapcat second (sort-by first groups)))
            new-order (schedule-support/topo-order dep-graph (concat non-par-indices grouped-par-indices))]
        (when (not= new-order (vec (range n)))
          (vec (mapcat (fn [i] (nth pairs i)) new-order)))))))

;; ================================================================
;; Combined fusion pass
;; ================================================================

(defn par-fusion-pass
  "Apply all par fusion strategies to a form.
  Runs: map→reduce fusion, horizontal map! fusion, horizontal map-void! fusion, binding reorder.
  Recurses into do blocks and generic seq forms to reach inner let* blocks.

  Returns {:form new-form :stats {:map-reduce N :horizontal N :horizontal-void N :reorder bool}}
  or {:form original-form :stats {:map-reduce 0 :horizontal 0 :horizontal-void 0 :reorder false}}."
  [form]
  (cond
    ;; let/let* — main fusion target
    (and (seq? form) (contains? #{'let 'let*} (first form)))
    (let [[let-sym bindings-vec & body-exprs] form
          stats (atom {:map-reduce 0 :horizontal 0 :horizontal-void 0 :reorder false})
          ;; 1. Reorder bindings for better fusion opportunities
          bindings-1 (or (when-let [reordered (reorder-for-fusion bindings-vec)]
                           (swap! stats assoc :reorder true)
                           reordered)
                         bindings-vec)
          ;; 2. Map→reduce fusion
          bindings-2 (if-let [{:keys [bindings fused]} (fuse-map-reduce bindings-1 body-exprs)]
                       (do (swap! stats assoc :map-reduce fused)
                           bindings)
                       bindings-1)
          ;; 3. Horizontal map! fusion
          bindings-3 (if-let [{:keys [bindings fused]} (fuse-horizontal-maps bindings-2)]
                       (do (swap! stats assoc :horizontal fused)
                           bindings)
                       bindings-2)
          ;; 3b. Horizontal map-void! fusion
          bindings-3 (if-let [{:keys [bindings fused]} (fuse-horizontal-map-voids bindings-3)]
                       (do (swap! stats update :horizontal-void + fused)
                           bindings)
                       bindings-3)
          ;; 4. Recurse into sub-expressions
          final-pairs (partition 2 bindings-3)
          recursed-bindings (vec (mapcat
                                  (fn [[sym expr]]
                                    (let [r (par-fusion-pass expr)
                                          rs (:stats r)]
                                      (when (pos? (+ (:map-reduce rs) (:horizontal rs)
                                                     (:horizontal-void rs 0)))
                                        (swap! stats (fn [s]
                                                       (-> s
                                                           (update :map-reduce + (:map-reduce rs))
                                                           (update :horizontal + (:horizontal rs))
                                                           (update :horizontal-void + (:horizontal-void rs 0))))))
                                      [sym (:form r)]))
                                  final-pairs))
          recursed-body (map (fn [b]
                               (let [r (par-fusion-pass b)]
                                 (:form r)))
                             body-exprs)]
      {:form (let [r (list* let-sym (vec recursed-bindings) recursed-body)]
               (if-let [m (meta form)] (with-meta r m) r))
       :stats @stats})

    ;; do block — recurse into sub-forms
    (and (seq? form) (= 'do (first form)))
    (let [results (map par-fusion-pass (rest form))
          merged-stats (clojure.core/reduce
                        (fn [s r] (-> s
                                      (update :map-reduce + (:map-reduce (:stats r)))
                                      (update :horizontal + (:horizontal (:stats r)))
                                      (update :horizontal-void + (:horizontal-void (:stats r) 0))
                                      (update :reorder #(or %1 %2) (:reorder (:stats r)))))
                        {:map-reduce 0 :horizontal 0 :horizontal-void 0 :reorder false}
                        results)]
      {:form (let [r (apply list 'do (map :form results))]
               (if-let [m (meta form)] (with-meta r m) r))
       :stats merged-stats})

    ;; Generic seq — recurse into sub-forms
    (seq? form)
    (let [results (map par-fusion-pass form)
          merged-stats (clojure.core/reduce
                        (fn [s r] (-> s
                                      (update :map-reduce + (:map-reduce (:stats r)))
                                      (update :horizontal + (:horizontal (:stats r)))
                                      (update :horizontal-void + (:horizontal-void (:stats r) 0))
                                      (update :reorder #(or %1 %2) (:reorder (:stats r)))))
                        {:map-reduce 0 :horizontal 0 :horizontal-void 0 :reorder false}
                        results)]
      {:form (let [r (apply list (map :form results))]
               (if-let [m (meta form)] (with-meta r m) r))
       :stats merged-stats})

    ;; Not a seq — return as-is
    :else
    {:form form :stats {:map-reduce 0 :horizontal 0 :horizontal-void 0 :reorder false}}))

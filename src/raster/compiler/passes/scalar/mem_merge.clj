(ns raster.compiler.passes.scalar.mem-merge
  "Futhark-style memory block merging via interference graph coloring.

  After buffer fusion and backend passes, multiple temporary buffers may
  have non-overlapping lifetimes. This pass merges them into shared
  max-size buffers, reducing total memory footprint.

  Algorithm:
    1. Collect allocations from let* bindings (double-array, float-array, etc.)
    2. Build interference graph via liveness analysis (two allocs interfere
       if their live ranges overlap)
    3. Greedy graph coloring respecting memory space constraints
    4. Rewrite: replace individual allocs with shared max-size buffer references

  Constraints:
    - Different device types never share buffers
    - Symbolic sizes only merge when statically equal
    - Only buffers tagged ^:hoistable are candidates (these are allocated once)"
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(def ^:dynamic *disable-mem-merge*
  "When true, skip buffer sharing (for debugging)."
  false)

;; ================================================================
;; Allocation detection
;; ================================================================

(defn- alloc-info
  "Extract allocation info from an expression.
  Returns {:type :double-array|:float-array|:long-array|:int-array :size expr :device :cpu}
  or nil."
  [expr]
  (let [op (form/effective-op expr)]
    (when (and op (descriptor/alloc-op? op))
      (let [head-name (name op)
            args (form/effective-args expr)]
        (when (or (.contains ^String head-name "double-array")
                  (.contains ^String head-name "float-array")
                  (.contains ^String head-name "long-array")
                  (.contains ^String head-name "int-array"))
          {:type (keyword head-name)
           :size (first args)
           :device :cpu})))))

;; ================================================================
;; Liveness analysis (simplified for let* forms)
;; ================================================================

(def ^:private free-syms
  "Collect unqualified free symbols from an S-expression."
  util/free-syms-flat)

(defn- compute-liveness
  "Compute live ranges for each binding in a let* form.
  Returns {sym -> {:def idx :last-use idx}} where idx is the binding index.
  Body expression uses are assigned index (count pairs).

  Handles buffer aliasing: when binding `result = (op ... buf)` references
  a hoistable buffer sym, the buffer's liveness extends through all uses of
  `result` (since result points to the same underlying memory)."
  [pairs body-exprs]
  (let [n (count pairs)
        ;; Build def points
        defs (into {} (map-indexed (fn [i [sym _]] [sym i]) pairs))
        ;; Compute last-use for each symbol
        last-use (atom {})
        _ (doseq [i (range n)]
            (let [[_sym expr] (nth pairs i)
                  used (free-syms expr)]
              (doseq [s used]
                (when (contains? defs s)
                  (swap! last-use assoc s (max (get @last-use s 0) i))))))
        ;; Body uses count as n (past last binding)
        body-used (apply set/union #{} (map free-syms body-exprs))
        _ (doseq [s body-used]
            (when (contains? defs s)
              (swap! last-use assoc s n)))
        ;; Alias propagation: if `result = buf` (direct alias) or `result = (do ... buf)`,
        ;; then buf's liveness must extend to last-use of result, since they share memory.
        ;; Also: `result = (f ... buf ...)` where buf is a hoistable alloc — the
        ;; into-variant function writes into buf and returns it, so result aliases buf.
        ;; Build direct alias map: sym -> aliased-sym (one level)
        hoistable-alloc-syms (set (keep (fn [[sym expr]]
                                          (when (and (:raster.buffer/hoistable (meta sym))
                                                     (alloc-info expr))
                                            sym))
                                        pairs))
        ;; Extract the "return value" of an expression — the symbol that this
        ;; expression ultimately evaluates to (following let*/do/loop*/if tails).
        ;; Returns a symbol or nil if the return value isn't a simple symbol reference.
        return-sym
        (fn return-sym [expr]
          (cond
            (symbol? expr) (when (contains? defs expr) expr)
            (not (seq? expr)) nil
            :else
            (let [head (first expr)]
              (cond
                ;; (do ... last-expr)
                (= head 'do) (return-sym (last expr))
                ;; (let* [...] body) or (loop* [...] body)
                (#{'let* 'loop*} head) (return-sym (last (nnext expr)))
                ;; (if test then else) — check else branch (common: loop returns buffer)
                (= head 'if)
                (let [[_ _ then-branch else-branch] expr]
                  (or (return-sym else-branch) (return-sym then-branch)))
                :else nil))))
        direct-alias
        (reduce
         (fn [m [sym expr]]
           (let [aliased (return-sym expr)]
             (if aliased
               (assoc m sym aliased)
               m)))
         {} pairs)
        ;; Buffer-through-call aliases: when result = (f ... buf ...) and buf is a
        ;; hoistable alloc, the into-variant writes into buf and returns it.
        ;; Extend buf's liveness through result's last-use.
        buf-call-aliases
        (reduce
         (fn [m [sym expr]]
           (if (form/call-form? expr)
             (let [buf-args (filter (fn [a]
                                      (and (symbol? a)
                                           (or (contains? hoistable-alloc-syms a)
                                               ;; Also follow one level of alias
                                               (contains? hoistable-alloc-syms
                                                          (get direct-alias a)))))
                                    (rest expr))]
               (reduce (fn [m2 buf-arg]
                         (let [root (if (contains? hoistable-alloc-syms buf-arg)
                                      buf-arg
                                      (get direct-alias buf-arg))]
                           (update m2 root (fnil conj #{}) sym)))
                       m buf-args))
             m))
         {} pairs)
        ;; Compute transitive alias roots: follow alias chains to the ultimate source
        alias-root
        (fn alias-root
          ([sym] (alias-root sym #{}))
          ([sym seen]
           (if-let [target (get direct-alias sym)]
             (if (contains? seen target)
               sym ;; cycle — stop
               (alias-root target (conj seen sym)))
             sym)))
        ;; Invert: for each root, collect all syms that transitively alias it
        alias-map
        (reduce
         (fn [m [sym _]]
           (let [root (alias-root sym)]
             (if (not= root sym)
               (update m root (fnil conj #{}) sym)
               m)))
         {} pairs)
        ;; Extend liveness of aliased roots through their transitive aliases' live ranges
        _ (doseq [[src-sym alias-syms] alias-map]
            (doseq [a alias-syms]
              (let [alias-lu (get @last-use a (get defs a 0))]
                (swap! last-use update src-sym (fnil max 0) alias-lu))))
        ;; Extend liveness of buffer allocs through call results that alias them
        _ (doseq [[buf-sym call-result-syms] buf-call-aliases]
            (doseq [r call-result-syms]
              (let [result-lu (get @last-use r (get defs r 0))]
                (swap! last-use update buf-sym (fnil max 0) result-lu))))]
    (into {}
          (map (fn [[sym def-idx]]
                 [sym {:def def-idx
                       :last-use (get @last-use sym def-idx)}])
               defs))))

(defn- ranges-overlap?
  "Check if two live ranges overlap.
  Range [def1, last-use1] overlaps [def2, last-use2] if they share any point."
  [{def1 :def lu1 :last-use} {def2 :def lu2 :last-use}]
  (and (<= def1 lu2) (<= def2 lu1)))

;; ================================================================
;; Interference graph
;; ================================================================

(defn- max-size-expr
  "Compute the max of two size expressions.
  For numeric sizes, returns the larger. For symbolic, returns a max expression."
  [s1 s2]
  (cond
    (and (number? s1) (number? s2)) (max s1 s2)
    (= s1 s2) s1
    :else (list 'Math/max (list 'long s1) (list 'long s2))))

(defn- build-interference-graph
  "Build interference graph for allocation symbols.
  Returns {sym -> set-of-interfering-syms}."
  [alloc-syms liveness alloc-infos]
  (let [syms (vec alloc-syms)]
    (reduce
     (fn [graph [i j]]
       (let [si (nth syms i)
             sj (nth syms j)
             ri (get liveness si)
             rj (get liveness sj)]
         (if (ranges-overlap? ri rj)
           (-> graph
               (update si (fnil conj #{}) sj)
               (update sj (fnil conj #{}) si))
           graph)))
     (into {} (map (fn [s] [s #{}]) syms))
     (for [i (range (count syms))
           j (range (inc i) (count syms))]
       [i j]))))

;; ================================================================
;; Greedy graph coloring
;; ================================================================

(defn- greedy-color
  "Greedy graph coloring. Returns {sym -> color-int}.
  Colors are assigned in order of decreasing interference degree."
  [interference-graph]
  (let [;; Sort by degree descending (most constrained first)
        ordered (sort-by (fn [s] (- (count (get interference-graph s #{}))))
                         (keys interference-graph))
        coloring (atom {})]
    (doseq [sym ordered]
      (let [neighbor-colors (set (keep (fn [n] (get @coloring n))
                                       (get interference-graph sym #{})))
            ;; Find smallest color not used by neighbors
            color (loop [c 0]
                    (if (contains? neighbor-colors c)
                      (recur (inc c))
                      c))]
        (swap! coloring assoc sym color)))
    @coloring))

;; ================================================================
;; Rewriting
;; ================================================================

(defn- compute-color-sizes
  "For each color, compute the max size across all allocations with that color."
  [coloring alloc-infos]
  (reduce
   (fn [sizes [sym color]]
     (let [info (get alloc-infos sym)]
       (update sizes color
               (fn [prev-size]
                 (if prev-size
                   (max-size-expr prev-size (:size info))
                   (:size info))))))
   {} coloring))

(defn- compute-color-types
  "For each color, determine the allocation type."
  [coloring alloc-infos]
  (reduce
   (fn [types [sym color]]
     (let [info (get alloc-infos sym)]
       (assoc types color (:type info))))
   {} coloring))

(defn- alloc-expr-for-type
  "Generate an allocation expression for a given type and size."
  [alloc-type size-expr]
  (case alloc-type
    :double-array (list 'double-array size-expr)
    :float-array  (list 'float-array size-expr)
    :long-array   (list 'long-array size-expr)
    :int-array    (list 'int-array size-expr)
    ;; Fallback
    (list 'double-array size-expr)))

(defn- zero-fill-expr
  "Generate a zero-fill expression for a shared buffer."
  [shared-sym alloc-type]
  (list 'java.util.Arrays/fill shared-sym
        (case alloc-type
          :double-array 0.0
          :float-array  (list 'float 0.0)
          :long-array   (long 0)
          :int-array    (int 0))))

;; ================================================================
;; Main pass
;; ================================================================

(defn merge-memory-blocks
  "Merge temporary buffers with non-overlapping lifetimes into shared blocks.

  Analyzes a flat let* form for allocations, builds an interference graph,
  and uses greedy coloring to assign buffers to shared memory blocks.

  Options:
    :device-env  - map of {sym -> :cpu|:cuda} for device-aware merging

  Returns {:form new-form :stats {:blocks N :colors M :bytes-saved K}}"
  [let-form & {:keys [device-env]}]
  (if (or *disable-mem-merge*
          (not (form/binding-form? let-form)))
    {:form let-form :stats {:blocks 0 :colors 0 :bytes-saved 0}}
    (let [[let-sym bindings-vec & body-exprs] let-form
          pairs (vec (partition 2 bindings-vec))
          ;; 1. Collect allocations (only ^:hoistable ones)
          alloc-infos
          (into {}
                (keep (fn [[sym expr]]
                        (when (:raster.buffer/hoistable (meta sym))
                          (when-let [info (alloc-info expr)]
                            (let [info (if (and device-env (contains? device-env sym))
                                         (assoc info :device (get device-env sym))
                                         info)]
                              [sym info]))))
                      pairs))
          alloc-syms-all (set (keys alloc-infos))
          ;; Exclude buffers that are returned (referenced in body expression).
          ;; Returned buffers must keep their exact size — merging with a larger
          ;; buffer would expose extra elements to the caller.
          ;; Follow alias chains: if body references `dA` which aliases `buf_dA`,
          ;; then `buf_dA` is returned.
          body-syms (apply set/union #{} (map free-syms body-exprs))
          alias-map (into {}
                          (keep (fn [[sym expr]]
                                  (cond
                                    (and (symbol? expr) (contains? alloc-syms-all expr))
                                    [sym expr]
                                ;; (do ... buf-sym) pattern
                                    (and (seq? expr) (= 'do (first expr))
                                         (symbol? (last expr))
                                         (contains? alloc-syms-all (last expr)))
                                    [sym (last expr)]
                                    :else nil))
                                pairs))
          returned-bufs (set (keep (fn [s]
                                     (or (when (contains? alloc-syms-all s) s)
                                         (get alias-map s)))
                                   body-syms))
          ;; Exclude buffers passed to opaque function calls.
          ;; Opaque calls may use (alength buf) internally, which would return
          ;; the wrong size if the buffer is shared with a different-sized one.
          ;; A call is opaque if it's to a qualified symbol (not a local binding).
          opaque-used-bufs
          (let [all-buf-syms (into alloc-syms-all (keys alias-map))
                resolve-buf (fn [s] (or (when (contains? alloc-syms-all s) s)
                                        (get alias-map s)))]
            (set (for [[_sym init] pairs
                       :when (and (seq? init)
                                  (let [h (first init)]
                                    (or ;; qualified call: (ns/fn args...)
                                     (and (symbol? h) (namespace h))
                                        ;; .invk call: (.invk ns/impl-sym args...)
                                     (= '.invk h))))
                       arg (rest init)
                       :when (symbol? arg)
                       :let [root (resolve-buf arg)]
                       :when root]
                   root)))
          alloc-syms (set/difference alloc-syms-all returned-bufs opaque-used-bufs)]
      (if (< (count alloc-syms) 2)
        {:form let-form :stats {:blocks (count alloc-syms-all) :colors (count alloc-syms-all) :bytes-saved 0}}
        ;; 2. Liveness analysis
        (let [liveness (compute-liveness pairs body-exprs)
              ;; 3. Group compatible allocations
              compat-groups
              (vals (group-by (fn [sym]
                                (let [info (get alloc-infos sym)]
                                  [(:type info) (:device info)]))
                              alloc-syms))
              ;; 4. For each compatibility group, build interference + color
              all-coloring (atom {})
              color-offset (atom 0)
              _ (doseq [group compat-groups]
                  (when (> (count group) 1)
                    (let [group-graph (build-interference-graph
                                       group liveness alloc-infos)
                          group-coloring (greedy-color group-graph)
                          offset @color-offset
                          shifted (into {} (map (fn [[s c]] [s (+ c offset)])
                                                group-coloring))]
                      (swap! all-coloring merge shifted)
                      (swap! color-offset + (inc (apply max (vals group-coloring))))))
                  (when (= 1 (count group))
                    (let [sym (first group)]
                      (swap! all-coloring assoc sym @color-offset)
                      (swap! color-offset inc))))
              coloring @all-coloring
              n-colors (if (empty? coloring) 0 (inc (apply max (vals coloring))))
              n-blocks (count alloc-syms)
              ;; Check if any merging actually happened
              merged? (< n-colors n-blocks)]
          (if-not merged?
            {:form let-form
             :stats {:blocks n-blocks :colors n-colors :bytes-saved 0}}
            ;; 5. Rewrite: replace individual allocs with shared buffer references.
            ;;
            ;; A shared buffer's size expression may reference symbols defined
            ;; inside the let* (e.g. `n_123` from expand-par).  The shared
            ;; buffer allocation must therefore be placed AFTER all its
            ;; size-expression deps are defined.
            ;;
            ;; For each color we compute the earliest legal position for its
            ;; shared buffer.  Allocs in the color that appear before that
            ;; position keep their original allocation (can't alias something
            ;; that doesn't exist yet).  The shared buffer is emitted at the
            ;; first alloc position >= the legal position, and all subsequent
            ;; allocs in the color alias to it.
            (let [color-sizes (compute-color-sizes coloring alloc-infos)
                  color-types (compute-color-types coloring alloc-infos)
                  ;; Compute most conservative write-mode per color:
                  ;; :overwrite only if ALL constituents are :overwrite
                  color-write-modes
                  (reduce (fn [m [sym color]]
                            (let [wm (or (:raster.buffer/write-mode (meta sym)) :accumulate)]
                              (update m color
                                      (fn [prev]
                                        (if (nil? prev) wm
                                            (if (or (= prev :accumulate) (= wm :accumulate))
                                              :accumulate
                                              :overwrite))))))
                          {} coloring)
                  ;; Generate shared buffer symbols with write-mode and type tag
                  color-buf-syms (into {}
                                       (map (fn [c]
                                              (let [arr-tag (case (get color-types c)
                                                              :float-array 'floats
                                                              :long-array 'longs
                                                              :int-array 'ints
                                                              'doubles)]
                                                [c (with-meta (gensym (str "shared_buf_" c "_"))
                                                     {:raster.buffer/hoistable true
                                                      :raster.buffer/write-mode (get color-write-modes c :accumulate)
                                                      :raster.type/tag arr-tag})]))
                                            (range n-colors)))
                  ;; Collect free symbols in each color's size expression
                  collect-syms (fn collect-syms [e]
                                 (cond (symbol? e) #{e}
                                       (seq? e) (reduce into #{} (map collect-syms (rest e)))
                                       :else #{}))
                  color-size-deps (into {}
                                        (map (fn [[c size-expr]]
                                               [c (collect-syms size-expr)])
                                             color-sizes))
                  ;; Build index: binding position of each sym
                  sym-positions (into {} (map-indexed (fn [i [sym _]] [sym i]) pairs))
                  ;; For each color, the shared buffer can be placed at or after
                  ;; the latest position of any dep symbol (or -1 if no deps).
                  color-min-pos
                  (into {}
                        (map (fn [[c deps]]
                               [c (if (empty? deps) -1
                                      (reduce max -1 (map #(get sym-positions % -1) deps)))])
                             color-size-deps))
                  ;; Track which colors we've emitted their shared buffer for
                  emitted-colors (atom #{})
                  ;; Rewrite bindings: single forward pass
                  new-pairs
                  (vec
                   (mapcat
                    (fn [[idx [sym expr]]]
                      (if-not (contains? alloc-syms sym)
                          ;; Non-alloc binding: pass through unchanged
                        [[sym expr]]
                        (let [color (get coloring sym)
                              shared-sym (get color-buf-syms color)
                              alloc-type (get color-types color)
                              ;; Overwrite buffers fully write before reading — no
                              ;; zeroing needed.  Only accumulating/unknown buffers
                              ;; need zeroing to avoid stale data across calls.
                              needs-zero? (not= :overwrite (:raster.buffer/write-mode (meta sym)))]
                          (if (contains? @emitted-colors color)
                              ;; Shared buffer already defined: zero-fill before reuse
                              ;; (unless this use fully overwrites the buffer)
                            (if needs-zero?
                              [[(with-meta (gensym "zfill_") {:raster.effect/effectful true})
                                (zero-fill-expr shared-sym alloc-type)]
                               [sym shared-sym]]
                              [[sym shared-sym]])
                              ;; First occurrence in this color
                            (if (> (get color-min-pos color -1) idx)
                                ;; Deps not yet satisfied at this position:
                                ;; keep original alloc expression (no merging for this occurrence)
                              [[sym expr]]
                                ;; Deps satisfied: emit shared buffer + alias
                              (do (swap! emitted-colors conj color)
                                  [[(get color-buf-syms color)
                                    (alloc-expr-for-type (get color-types color) (get color-sizes color))]
                                   [sym shared-sym]]))))))
                    (map-indexed vector pairs)))
                  new-bindings (vec (mapcat identity new-pairs))]
              {:form (list* let-sym new-bindings body-exprs)
               :stats {:blocks n-blocks
                       :colors n-colors
                       :bytes-saved (- n-blocks n-colors)}})))))))

(ns raster.compiler.passes.scalar.resolve-alength
  "Pre-mem-merge normalization: resolve (alength hoistable-buf) to the
  original allocation size expression.

  After buffer-fuse, hoistable buffers have known allocation sizes:
    buf = (double-array size-expr)
  Into-variant calls return the buffer they write into:
    result = (dense-into! W x b buf)  ;; result aliases buf

  Downstream code may use (alength result) to determine iteration counts.
  If mem-merge later resizes the physical buffer, (alength result) would
  return the wrong value. This pass resolves such alength calls to the
  original size-expr, decoupling iteration counts from physical buffer
  sizes.

  This is semantics-preserving: (alength (double-array k)) == k by
  definition. We're just making this explicit via partial evaluation."
  (:require [clojure.string]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

;; ================================================================
;; Allocation detection
;; ================================================================

(defn- clone-op?
  "True if sym is a clone-like allocation: (aclone source-array).
  Clone ops take an array (not a size) as argument."
  [sym]
  (and (symbol? sym)
       (let [base (let [n (name sym)
                        idx (clojure.string/index-of n "_m_")]
                    (if idx (subs n 0 (int idx)) n))]
         (= base "aclone"))))

(defn- alloc-size
  "Extract the size expression from an array allocation.
   (double-array size) → size
   (zeros-like ref size) → size (last arg is the size)
   (.invk zeros-like_m_...-impl ref size) → size (devirtualized form)
   (aclone source) → (clojure.core/alength source) — clone takes array, not size
   Returns nil for non-alloc expressions."
  [expr]
  (let [op-sym (form/effective-op expr)
        args (form/effective-args expr)
        argc (count args)]
    (when (and (symbol? op-sym)
               (descriptor/alloc-op? op-sym)
               (pos? argc))
      (cond
        ;; Clone-like: (aclone source-array) — size is (alength source)
        (and (= 1 argc) (clone-op? op-sym))
        (list 'clojure.core/alength (first args))

        ;; Standard 1-arg: (double-array size) → size
        (= 1 argc)
        (first args)

        ;; Multi-arg: (zeros-like ref size) — size is the last arg
        :else
        (last args)))))

;; ================================================================
;; Alias tracking
;; ================================================================

(defn- alength-call?
  "Is expr an alength call? Recognizes:
   - (alength x), (clojure.core/alength x), (raster.arrays/alength x)
   - (.invk raster.arrays/alength_m_*-impl x) — walker devirtualized form
   - (long (alength x)) — walker-wrapped form
   Returns the target symbol or nil."
  [expr]
  (when (seq? expr)
    (cond
      ;; Direct: (alength x) / (clojure.core/alength x) / (raster.arrays/alength x)
      (and (descriptor/alength-op? (first expr))
           (= 2 (count expr))
           (symbol? (second expr)))
      (second expr)

      ;; Devirtualized: (.invk raster.arrays/alength_m_*-impl x)
      (and (= '.invk (first expr))
           (= 3 (count expr))
           (symbol? (second expr))
           (let [s (str (second expr))]
             (and (.contains s "alength_m_")
                  (.endsWith s "-impl")))
           (symbol? (nth expr 2)))
      (nth expr 2)

      ;; Wrapped: (long (alength x)) — recurse
      (and (= 'long (first expr))
           (= 2 (count expr))
           (seq? (second expr)))
      (alength-call? (second expr))

      :else nil)))

(defn- resolve-alength-in-expr
  "Replace (alength sym) with resolved size in a single expression.
  size-map: {sym → size-expr} for all syms with known logical sizes."
  [expr size-map]
  (cond
    ;; Direct alength call on a known sym
    (alength-call? expr)
    (let [target (alength-call? expr)]
      (if-let [size (get size-map target)]
        size
        expr))

    ;; quote is opaque — never resolve inside quoted data
    (and (seq? expr) (= 'quote (first expr))) expr

    (seq? expr)
    ;; A nested binder (let*/loop*/dotimes/fn*/par/...) that SHADOWS a size-map
    ;; name must have that name removed from the map for its inits+body — else
    ;; `(alength shadowed-buf)` inside wrongly resolves to the OUTER alloc size.
    ;; Outer exprs (e.g. a dotimes bound) still see the full map.
    (if-let [{:keys [scopes outer rebuild]} (form/scope-info expr)]
      (rebuild (mapv (fn [{:keys [binders inits body]}]
                       (let [m (apply dissoc size-map (filter symbol? binders))]
                         {:binders binders
                          :inits (mapv #(resolve-alength-in-expr % m) inits)
                          :body  (mapv #(resolve-alength-in-expr % m) body)}))
                     scopes)
               (mapv #(resolve-alength-in-expr % size-map) outer))
      ;; non-binder seq — recurse into all children
      (util/remake-from expr (map #(resolve-alength-in-expr % size-map) expr)))

    :else expr))

;; ================================================================
;; Main pass
;; ================================================================

(defn resolve-alength-pass
  "Resolve (alength hoistable-buf) to the original allocation size.

  Walks let* bindings forward:
  1. For each hoistable alloc `buf = (double-array size-expr)`, record {buf → size-expr}
  2. For each binding `result = (f ... buf ...)` where buf is a hoistable alloc,
     record {result → size-expr} (result aliases buf via into-variant pattern)
  3. Replace all (alength sym) where sym is in the size-map with the size-expr

  Returns {:form new-form :stats {:resolved N}}"
  [form]
  (if-not (and (seq? form) (form/binding-form? form))
    {:form form :stats {:resolved 0}}
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (vec (partition 2 bindings-vec))
          ;; Forward pass: build size map and rewrite simultaneously
          size-map (atom {})
          resolved-count (atom 0)

          new-pairs
          (mapv
           (fn [[sym init]]
              ;; First, resolve any alength calls in this binding's init expr
             (let [init' (resolve-alength-in-expr init @size-map)
                   resolved? (not= init init')]
               (when resolved?
                 (swap! resolved-count inc))

                ;; Track ALL allocations — not just hoistable ones.
                ;; mem-merge can resize any buffer, so all alength calls on
                ;; locally-allocated arrays must be resolved to the original
                ;; size expression before mem-merge runs.
               (when-let [size (alloc-size init')]
                 (swap! size-map assoc sym size))

                ;; Track aliases: sym = known-size-sym (direct alias).
                ;; Check size-map to handle transitive alias chains like
                ;; d_out = dx__5 = dx (alloc).
               (when (and (symbol? init')
                          (contains? @size-map init'))
                 (when-let [size (get @size-map init')]
                   (swap! size-map assoc sym size)))

                ;; Track call-through aliases: sym = (f ... out-buf ...) where an
                ;; into-variant writes into and RETURNS one of its argument
                ;; buffers, so (alength sym) must resolve to that buffer's logical
                ;; size before mem-merge resizes it.
                ;;
                ;; Identify the output buffer AUTHORITATIVELY from the op-descriptor
                ;; buffer-semantics (:in-place-arg) — never by guessing "the last
                ;; hoistable argument". That heuristic mis-identified the DY *input*
                ;; as the output for into-variants like conv2d-backward-db-into!
                ;; [dy db ...] (whose output db is arg 1, and is often allocated
                ;; out-of-scope so it is the only NON-tracked buffer). It then
                ;; aliased (alength result) to the wrong, larger input buffer,
                ;; overrunning every later reduce/scale loop bound — a runtime OOB
                ;; when the mis-aliased buffer is larger, and a SILENT wrong-count
                ;; miscompile when it is smaller or equal.
                ;;
                ;; SAFE FALLBACK: only pure into-variants (:allocates? false with a
                ;; concrete :in-place-arg) alias, and only when that arg's size is
                ;; already known. Otherwise DO NOT alias — leaving (alength result)
                ;; intact returns the true runtime length, which is always correct
                ;; (the aliased buffer, if any, is then not a mem-merge target).
               (when (and (not (contains? @size-map sym))
                          (form/call-form? init'))
                 (when-let [[bs _] (descriptor/resolve-buffer-semantics
                                    (form/effective-op init'))]
                   (when (and (not (:allocates? bs)) (:in-place-arg bs))
                     (let [args    (vec (form/effective-args init'))
                           out-idx (:in-place-arg bs)
                           out-arg (when (< out-idx (count args)) (nth args out-idx))]
                       (when (and (symbol? out-arg) (contains? @size-map out-arg))
                         (swap! size-map assoc sym (get @size-map out-arg)))))))

                ;; Track loop/let return aliases: sym = (let* [...] (loop* [...]
                ;; (if ... (recur ...) buf))). When the expanded par/map! loop
                ;; returns the output buffer, sym aliases that buffer.
                ;; Skip if sym already has a known size to avoid overwriting.
               (when (and (not (contains? @size-map sym))
                          (seq? init'))
                 (letfn [(extract-return [e]
                           (cond
                             (symbol? e) e
                             (not (seq? e)) nil
                             ;; let*/let: return is the last body expr
                             (form/binding-form? e) (extract-return (last e))
                             ;; loop*: return is the last body expr (the if)
                             (= 'loop* (first e)) (extract-return (last e))
                             ;; if: the else branch is the non-recur return
                             (= 'if (first e))
                             (let [branches (rest (rest e)) ;; then, else
                                   else-branch (second branches)]
                               (extract-return else-branch))
                             :else nil))]
                   (when-let [return-sym (extract-return init')]
                     (when (contains? @size-map return-sym)
                       (swap! size-map assoc sym (get @size-map return-sym))))))

               [sym init']))
           pairs)

          ;; Resolve alength in body expressions
          new-body (mapv #(resolve-alength-in-expr % @size-map) body-exprs)
          body-resolved? (not= (vec body-exprs) new-body)
          _ (when body-resolved? (swap! resolved-count inc))

          new-bindings (vec (mapcat identity new-pairs))]
      {:form (list* let-sym new-bindings new-body)
       :stats {:resolved @resolved-count}})))

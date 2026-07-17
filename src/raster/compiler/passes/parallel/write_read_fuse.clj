(ns raster.compiler.passes.parallel.write-read-fuse
  "Producer-consumer fusion for 2D write → 1D read patterns.

  Detects the pattern produced by AD templates for dense backward:
    1. Buffer allocation:   buf = (double-array ...)
    2. 2D producer:         (dotimes [i rows] (par/map! buf j cols :offset (* i cols) body))
    3. 1D consumer:         (par/map! target k total ... (f (aget buf k)))

  When buf is zero-initialized (double-array) and used only by producer→consumer,
  fuses the consumer into the producer's 2D loop structure:
    (dotimes [i rows] (par/map! target j cols :offset (* i cols) (f body_simplified)))

  This eliminates the intermediate buffer allocation and one full memory pass.
  Mirrors XLA's fused_computation pattern for dW+SGD fusion.

  Runs between loop-lift and soac-fuse in the pipeline."
  (:require [clojure.walk :as walk]
            [raster.compiler.ir.par :as par]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.passes.parallel.descriptors :as desc]))

;; ================================================================
;; Pattern detection
;; ================================================================

(defn- zero-init-alloc?
  "Check if an expression is a zero-initializing allocation:
  (clojure.core/double-array ...) or (raster.arrays/zeros-like ...)"
  [expr]
  (and (seq? expr)
       (let [head (first expr)]
         (or (= head 'clojure.core/double-array)
             (and (symbol? head)
                  (.startsWith (name head) "zeros-like"))))))

(defn- dotimes-par-map-producer?
  "Check if an expression matches:
    (dotimes [i rows] (par/map! buf j cols :offset offset-expr type body))
  Returns {:outer-idx, :outer-bound, :buf, :inner-idx, :inner-bound,
           :offset, :cast, :body} or nil."
  [expr]
  (when (and (seq? expr)
             (= 'dotimes (first expr))
             (vector? (second expr))
             (= 2 (count (second expr))))
    (let [[_dotimes [outer-idx outer-bound] inner-form] expr]
      (when (par/par-map-form? inner-form)
        (when-let [info (par/extract-par-map-info inner-form)]
          (when (:offset info)
            {:outer-idx    outer-idx
             :outer-bound  outer-bound
             :buf          (:out info)
             :inner-idx    (:idx info)
             :inner-bound  (:bound info)
             :offset       (:offset info)
             :cast         (:cast info)
             :body         (:body info)}))))))

(defn- substitute-buf-reads
  "Eliminate self-reads of a zero-initialized buffer from the producer body.
  Replaces (aget buf idx) → 0.0, then folds (+ 0.0 x) → x for additions
  (postwalk processes children first, so aget→0.0 happens before the parent
  addition is visited)."
  [body buf-sym]
  (walk/postwalk
   (fn [form]
     (cond
       ;; Replace (aget buf idx) → 0.0
       (and (seq? form)
            (descriptor/aget-op? (first form))
            (>= (count form) 3)
            (symbol? (second form))
            (= (name buf-sym) (name (second form))))
       0.0
       ;; Fold (+ 0.0 x) → x via :raster.op/original metadata on the + form
       (and (seq? form)
            (= 'raster.numeric/+ (:raster.op/original (meta form))))
       (let [args (vec (if (= '.invk (first form))
                         (drop 2 form)
                         (rest form)))]
         (if (= 2 (count args))
           (cond
             (and (number? (args 0)) (zero? (args 0))) (args 1)
             (and (number? (args 1)) (zero? (args 1))) (args 0)
             :else form)
           form))
       :else form))
   body))

(defn- collect-aget-syms
  "Collect all array symbols read via aget in an expression."
  [expr]
  (cond
    (and (seq? expr)
         (descriptor/aget-op? (first expr))
         (>= (count expr) 3)
         (symbol? (second expr)))
    #{(second expr)}

    (seq? expr) (apply clojure.set/union #{} (map collect-aget-syms (rest expr)))
    (vector? expr) (apply clojure.set/union #{} (map collect-aget-syms expr))
    ;; Map literals (e.g. a `case*` clause map) can hide an aget read; missing them makes the
    ;; sole-consumer/liveness check map-blind — a buffer still read inside a case* would look
    ;; dead and be wrongly fused away. Recurse into keys+vals like util/free-syms-flat does.
    (map? expr) (apply clojure.set/union #{} (map collect-aget-syms (apply concat expr)))
    :else #{}))

(defn- reads-sym?
  "Check if body reads from sym via aget."
  [body sym]
  (some #(= (name sym) (name %)) (collect-aget-syms body)))

(defn- substitute-aget-with-expr
  "Replace (aget sym idx-expr) in body with replacement-expr,
  rewriting the producer's index variables to match the consumer context."
  [body target-sym producer-body outer-idx inner-idx offset-expr]
  (walk/postwalk
   (fn [form]
     (if (and (seq? form)
              (descriptor/aget-op? (first form))
              (>= (count form) 3)
              (symbol? (second form))
              (= (name target-sym) (name (second form))))
       ;; Replace the aget with the producer's body expression.
       ;; The aget index is (+ offset j) or just k — the producer body
       ;; already uses the correct outer-idx/inner-idx scope.
       producer-body
       form))
   body))

;; ================================================================
;; Alias resolution
;; ================================================================

(defn- build-alias-map
  "Build a map from symbol names to their roots, following simple
  sym = sym bindings in the let* pairs."
  [pairs]
  (let [direct (reduce (fn [m [sym expr]]
                         (if (symbol? expr)
                           (assoc m (name sym) (name expr))
                           m))
                       {} pairs)]
    ;; Transitive resolution
    (loop [result direct, fuel 10]
      (if (zero? fuel)
        result
        (let [next (reduce-kv
                    (fn [m k v]
                      (if-let [root (get m v)]
                        (assoc m k root)
                        m))
                    result result)]
          (if (= next result) result (recur next (dec fuel))))))))

(defn- resolve-alias
  "Resolve a symbol name through the alias map to its root."
  [alias-map sym-name]
  (get alias-map sym-name sym-name))

;; ================================================================
;; Liveness check
;; ================================================================

(defn- sym-used-in?
  "Check if sym-name appears in any aget reads within expression."
  [sym-name expr]
  (some #(= sym-name (name %)) (collect-aget-syms expr)))

(defn- sym-referenced?
  "Check if sym-name is referenced (as any symbol) in expression."
  [sym-name expr]
  (cond
    (symbol? expr) (= sym-name (name expr))
    (seq? expr) (some (partial sym-referenced? sym-name) expr)
    (vector? expr) (some (partial sym-referenced? sym-name) expr)
    :else false))

(defn- sole-consumer?
  "Check that buf-name is only used by the producer and consumer bindings,
  plus alias bindings. No other binding reads from it."
  [buf-name pairs producer-idx consumer-idx alias-map body-exprs]
  (let [alias-names (into #{buf-name}
                          (keep (fn [[k v]] (when (= v buf-name) k)))
                          alias-map)]
    (and
     (not (some (fn [[idx [_sym expr]]]
                  (when (and (not= idx producer-idx)
                             (not= idx consumer-idx))
                    ;; Check if this binding references any alias of buf
                    ;; (but allow simple alias bindings like dW = buf)
                    (when-not (and (symbol? expr)
                                   (contains? alias-names (name expr)))
                      (some #(sym-referenced? % expr) alias-names))))
                (map-indexed vector pairs)))
     ;; Also check body expressions
     (not (some (fn [body-expr]
                  (some #(sym-referenced? % body-expr) alias-names))
                body-exprs)))))

;; ================================================================
;; Fusion transform
;; ================================================================

(defn- find-fusion-pairs
  "Scan let* bindings for producer-consumer pairs that can be fused.
  Returns a list of {:producer-idx, :consumer-idx, :alloc-idx,
                     :alias-idxs, :producer-info, :consumer-info}."
  [pairs body-exprs]
  (let [alias-map (build-alias-map pairs)
        indexed-pairs (vec (map-indexed vector pairs))
        ;; Track which syms are zero-init allocations
        alloc-syms (into {}
                         (keep (fn [[idx [sym expr]]]
                                 (when (zero-init-alloc? expr)
                                   [(name sym) idx])))
                         indexed-pairs)]
    (reduce
     (fn [fusions [prod-idx [prod-sym prod-expr]]]
       (if-let [prod-info (dotimes-par-map-producer? prod-expr)]
         (let [buf-name (name (:buf prod-info))
               buf-root (resolve-alias alias-map buf-name)
               ;; Check if root is zero-initialized
               alloc-idx (get alloc-syms buf-root)]
           (if (nil? alloc-idx)
             fusions
             ;; Find a consumer par/map! that reads from buf (via alias chain)
             (let [;; All names that alias to this buffer
                   buf-aliases (into (hash-set buf-root buf-name)
                                     (keep (fn [[k v]] (when (= v buf-root) k)))
                                     alias-map)
                   consumer
                   (first
                    (keep (fn [[cons-idx [_cons-sym cons-expr]]]
                            (when (> cons-idx prod-idx)
                              (when-let [cons-info (desc/map-form-info cons-expr)]
                                ;; Consumer reads from a buf alias?
                                (let [cons-inputs (desc/map-form-info cons-expr)
                                      cons-body (:body cons-info)]
                                  (when (some #(contains? buf-aliases (name %))
                                              (collect-aget-syms cons-body))
                                    [cons-idx cons-info])))))
                          indexed-pairs))
                   ;; Alias bindings to delete
                   alias-idxs (keep (fn [[idx [_sym expr]]]
                                      (when (and (symbol? expr)
                                                 (contains? buf-aliases (name expr))
                                                 (not= idx alloc-idx)
                                                 (not= idx prod-idx))
                                        idx))
                                    indexed-pairs)]
               (if (nil? consumer)
                 fusions
                 (let [[cons-idx cons-info] consumer]
                   ;; Verify sole-consumer: buf not used elsewhere
                   (if (sole-consumer? buf-root pairs prod-idx cons-idx
                                       alias-map body-exprs)
                     (conj fusions {:producer-idx  prod-idx
                                    :consumer-idx  cons-idx
                                    :alloc-idx     alloc-idx
                                    :alias-idxs    (vec alias-idxs)
                                    :producer-info prod-info
                                    :consumer-info cons-info
                                    :buf-aliases   buf-aliases})
                     fusions))))))
         fusions))
     [] indexed-pairs)))

(defn- remap-consumer-index
  "Replace all occurrences of cons-idx symbol with flat-idx-expr in body.
  cons-idx was the consumer's flat loop variable; after fusion it maps to
  (+ offset inner-idx) in the producer's 2D structure."
  [body cons-idx flat-idx-expr]
  (walk/postwalk
   (fn [form]
     (if (and (symbol? form) (= (name cons-idx) (name form)))
       flat-idx-expr
       form))
   body))

(defn- apply-fusion
  "Apply a single producer-consumer fusion.
  Replaces the consumer's 1D par/map! with the producer's 2D dotimes+par/map!
  structure, inlining the consumer body.

  Index mapping: consumer used flat index k over [0, rows*cols).
  After fusion, k = offset + j = (* i cols) + j in the 2D structure.
  All consumer array accesses (aget arr k) become (aget arr (+ offset j)).
  The dW buffer read (aget dW k) is replaced with the producer body directly."
  [{:keys [producer-info consumer-info buf-aliases]}]
  (let [{:keys [outer-idx outer-bound buf inner-idx inner-bound offset cast body]}
        producer-info
        cons-body (:body consumer-info)
        cons-idx  (:idx consumer-info)
        cons-out  (:out consumer-info)
        cons-cast (:cast consumer-info)
        ;; The flat index expression in the fused 2D structure
        flat-idx-expr (list '.invk 'raster.numeric/_plus__m_long_long-impl
                            offset inner-idx)
        ;; Step 1: Replace self-reads with 0.0 (buffer is zero-initialized)
        simplified-body (substitute-buf-reads body buf)
        ;; Step 2: Remap consumer's flat index to 2D (offset + j) FIRST.
        ;; This must happen before dW substitution to avoid replacing
        ;; the producer body's outer-idx (which may have the same name
        ;; as the consumer's flat index).
        remapped-body (remap-consumer-index cons-body cons-idx flat-idx-expr)
        ;; Step 3: In remapped body, replace (aget dW_alias ...) → simplified producer body
        fused-body
        (walk/postwalk
         (fn [form]
           (if (and (seq? form)
                    (descriptor/aget-op? (first form))
                    (>= (count form) 3)
                    (symbol? (second form))
                    (contains? buf-aliases (name (second form))))
             ;; Replace the dW read with the simplified producer body
             simplified-body
             form))
         remapped-body)]
    ;; Build the fused 2D structure writing to consumer's output.
    ;; Stamp :raster.type/elem-type on the par/map! so downstream SIMD codegen
    ;; picks the right Vector species (else falls back to :double, breaking
    ;; f32 with [F → [D cast at runtime).
    (let [elem-type (case cons-cast
                      (float clojure.core/float) :float
                      (double clojure.core/double) :double
                      (long clojure.core/long) :long
                      (int clojure.core/int) :int
                      (case (or (:raster.type/tag (meta cons-out)) (:tag (meta cons-out)))
                        floats :float
                        doubles :double
                        longs :long
                        ints :int
                        nil))
          map-form (cond-> (list* 'raster.par/map!
                                  cons-out inner-idx inner-bound
                                  :offset offset
                                  cons-cast
                                  [fused-body])
                     elem-type (with-meta {:raster.type/elem-type elem-type}))]
      (list 'dotimes [outer-idx outer-bound] map-form))))

(defn fuse-write-read
  "Main entry point: fuse 2D producer → 1D consumer patterns in let* bindings.
  Returns {:form new-form :stats {:write-read-fused count}}."
  [form]
  (if-not (and (seq? form) (contains? #{'let* 'let} (first form)))
    {:form form :stats {:write-read-fused 0}}
    (let [[let-sym bindings-vec & body-exprs] form
          pairs (vec (partition 2 bindings-vec))
          fusions (find-fusion-pairs pairs body-exprs)]
      (if (empty? fusions)
        {:form form :stats {:write-read-fused 0}}
        ;; Apply fusions: remove alloc+producer+aliases, replace consumer
        (let [;; Collect all indices to delete
              delete-idxs (into #{}
                                (mapcat (fn [{:keys [producer-idx alloc-idx alias-idxs]}]
                                          (cons alloc-idx (cons producer-idx alias-idxs))))
                                fusions)
              ;; Build replacement map: consumer-idx → fused form
              replacements (into {}
                                 (map (fn [fusion]
                                        [(:consumer-idx fusion)
                                         (apply-fusion fusion)]))
                                 fusions)
              ;; Rebuild bindings
              new-pairs (keep-indexed
                         (fn [idx [sym expr]]
                           (cond
                             ;; Delete allocation, producer, and alias bindings
                             (contains? delete-idxs idx) nil
                             ;; Replace consumer with fused form
                             (contains? replacements idx) [sym (get replacements idx)]
                             ;; Keep as-is
                             :else [sym expr]))
                         pairs)
              new-bindings (vec (mapcat identity new-pairs))]
          {:form (list* let-sym new-bindings body-exprs)
           :stats {:write-read-fused (count fusions)}})))))

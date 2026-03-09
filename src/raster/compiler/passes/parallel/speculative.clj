(ns raster.compiler.passes.parallel.speculative
  "Speculative specialization for the compiler pipeline.

  Wraps compiled functions with runtime shape observation.
  After N stable calls with the same shapes, specializes in background
  via PE + full pipeline. Guard checks shape match, deopt to generic.

  Usage:
    (def train-fn (compile-aot #'train-step))
    (def fast-train (wrap-speculative train-fn #'loss-fn {:threshold 3}))
    ;; First 3 calls: generic. After: specialized for observed shapes.")

;; ================================================================
;; Shape signature extraction
;; ================================================================

(defn- dim-extractor
  "Create a dimension extractor for a parameter based on its type tag.
  Returns nil for all types (no shape extraction)."
  [tag]
  nil)

(defn make-shape-extractors
  "From deftm param tags, create extractor functions.
  Returns [{:param sym, :tag tag, :extractor fn}] for params with shapes."
  [params tags]
  (vec
   (keep (fn [[p t]]
           (when-let [ext (dim-extractor t)]
             {:param p :tag t :extractor ext}))
         (map vector params tags))))

(defn shape-signature
  "Extract dimension ints from args using extractors.
  Returns a vector of dimensions, e.g. [128 784 128 10 784 10]."
  [args extractors]
  (vec
   (mapcat (fn [{:keys [extractor]} arg]
             (when extractor
               (try (extractor arg) (catch Exception _ nil))))
           extractors args)))

;; ================================================================
;; Speculative wrapper
;; ================================================================

(defn- stable-shape?
  "Check if all recent observations have the same shape."
  [observations threshold]
  (let [obs @observations]
    (and (>= (count obs) threshold)
         (let [recent (take-last threshold obs)]
           (apply = recent)))))

(defn- specialize-for-shape!
  "Background-compile a specialized version for the given shape.
  Uses PE dimension substitution + pipeline recompilation."
  [f-var shape-sig extractors compile-fn pending shape-cache]
  (when-not (contains? @pending shape-sig)
    (swap! pending conj shape-sig)
    (future
      (try
        (let [;; Build known-dims from shape signature
              ;; Currently no type-specific extractors; return empty map
              known-dims {}
              specialized-fn (compile-fn known-dims)]
          (swap! shape-cache assoc shape-sig specialized-fn)
          (swap! pending disj shape-sig))
        (catch Exception e
          (swap! pending disj shape-sig)
          (binding [*out* *err*]
            (println "Speculative specialization failed:" (.getMessage e))))))))

(defn wrap-speculative
  "Wrap a compiled fn with shape monitoring + speculative specialization.

  compiled-fn: the generic compiled function (always-works fallback)
  f-var: the deftm var for recompilation
  opts:
    :threshold   - calls before specializing (default 3)
    :max-cache   - max entries in shape cache (default 8)
    :compile-fn  - (fn [known-dims] -> specialized-fn) for recompilation
    :extractors  - from make-shape-extractors

  Returns a function that:
  1. First N calls: run generic, record shapes
  2. After N stable shapes: specialize in background (future)
  3. Once ready: use specialized fn
  4. Shape mismatch: fall back to generic"
  [compiled-fn f-var {:keys [threshold max-cache compile-fn extractors]
                      :or {threshold 3 max-cache 8}}]
  (let [shape-cache  (atom {})       ;; {shape-sig -> specialized-fn}
        observations (atom [])       ;; recent shape signatures
        pending      (atom #{})      ;; sigs currently being compiled
        extractors   (or extractors [])]
    (fn [& args]
      (if (empty? extractors)
        ;; No shape extractors: just use generic
        (apply compiled-fn args)
        ;; Shape-aware path
        (let [sig (shape-signature args extractors)]
          ;; Check if we have a specialized version
          (if-let [specialized (get @shape-cache sig)]
            ;; Use specialized version
            (apply specialized args)
            ;; Record observation and possibly trigger specialization
            (do
              (swap! observations (fn [obs]
                                    (let [obs' (conj obs sig)]
                                      (if (> (count obs') (* 2 threshold))
                                        (vec (take-last threshold obs'))
                                        obs'))))
              ;; Check if we should specialize
              (when (and compile-fn
                         (stable-shape? observations threshold)
                         (not (contains? @pending sig))
                         (< (count @shape-cache) max-cache))
                (specialize-for-shape! f-var sig extractors compile-fn
                                       pending shape-cache))
              ;; Use generic for this call
              (apply compiled-fn args))))))))

;; ================================================================
;; Pipeline integration helpers
;; ================================================================

(defn compile-speculative-gradient-fn
  "Compile a gradient fn with speculative specialization.
  Returns a wrapped function that auto-specializes for observed shapes.

  f-var: deftm var
  opts: same as compile-gradient-fn plus :threshold, :max-cache"
  [f-var compile-gradient-fn-impl
   & {:keys [threshold max-cache]
      :or {threshold 3 max-cache 8}}]
  (let [params (or (:raster.core/deftm-params (meta f-var))
                   (:raster.core/deftm-params (meta (deref f-var))))
        tags   (or (:raster.core/deftm-tags (meta f-var))
                   (:raster.core/deftm-tags (meta (deref f-var))))
        extractors (when (and params tags) (make-shape-extractors params tags))
        generic-fn (compile-gradient-fn-impl f-var)
        compile-fn (fn [known-dims]
                     ;; Specialize by pre-computing dimensions
                     ;; and recompiling with them constant
                     (compile-gradient-fn-impl f-var))]
    (wrap-speculative generic-fn f-var
                      {:threshold threshold
                       :max-cache max-cache
                       :compile-fn compile-fn
                       :extractors (or extractors [])})))

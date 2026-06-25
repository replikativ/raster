(ns raster.compiler.passes.parallel.compound-detect
  "Compound kernel detection pass.

  Detects `dotimes` loops containing multiple par forms (map!, reduce!,
  stencil!) and wraps them in a `raster.compiler/compound-kernel` marker
  for downstream codegen.

  Strategy selection:
    :local  — single OpenCL kernel with __local scratch (n ≤ max_workgroup_size)
    :global — persistent __global DeviceBuffers with per-phase kernel launches

  Usage in pipeline:
    (compound-detect-pass form opts)
    ;; Returns {:form new-form :stats {:compound-kernels N}}"
  (:require [raster.compiler.passes.parallel.descriptors :as desc]
            [raster.compiler.passes.parallel.execution-plan :as execution-plan]
            [raster.runtime.hardware :as hw]
            [raster.compiler.core.util :as util]
            [clojure.walk :as walk]
            [clojure.set :as set]))

;; ================================================================
;; Phase extraction — classify par forms inside a dotimes body
;; ================================================================

(defn- extract-phases
  "Walk the body of a dotimes loop and extract par/stencil phases.
  Returns a vector of phase descriptors:
    {:type :map/:reduce/:stencil, :form <original-form>, :out <sym>, :inputs #{sym}, ...}"
  [body]
  (let [phases (atom [])]
    (walk/prewalk
     (fn [form]
       (if-let [info (desc/par-form-info form)]
         (do
           (case (:type info)
             :map
             (swap! phases conj
                    {:type :map
                     :form form
                     :out (:out info)
                     :inputs (:inputs info)
                     :body (:body info)
                     :idx (:idx info)
                     :bound (:bound info)})

             :stencil
             (swap! phases conj
                    {:type :stencil
                     :form form
                     :out (:out info)
                     :inputs (:inputs info)
                     :body (:body info)
                     :idx (:idx info)
                     :bound (:bound info)
                     :radius (:radius info)
                     :boundary (:boundary info)})

             :reduce
             (swap! phases conj
                    {:type :reduce
                     :form form
                     :out (:acc info)
                     :inputs (:inputs info)
                     :body (:body info)
                     :idx (:idx info)
                     :bound (:bound info)
                     :init (:init info)
                     :acc (:acc info)})

             nil)
           form)
         form))
     body)
    @phases))

;; ================================================================
;; Validation — compound kernel body must be fully expanded
;; ================================================================

(defn- contains-opaque-calls?
  "Check if a dotimes body contains opaque function calls that can't
  be compiled to a GPU kernel (.invk, unresolved var calls, etc.).
  Returns true if the body is NOT suitable for compound kernel fusion."
  [body]
  (let [opaque? (atom false)]
    (walk/prewalk
     (fn [form]
       (when (and (seq? form)
                  (symbol? (first form))
                  (let [head (first form)]
                    (or
                       ;; .invk = walker-devirtualized call, not yet fully inlined
                     (= '.invk head)
                       ;; Any dot-method call
                     (.startsWith (name head) "."))))
         (reset! opaque? true))
       form)
     body)
    @opaque?))

;; ================================================================
;; Array classification
;; ================================================================

(defn- classify-arrays
  "Classify arrays as :input (read-only), :output (escapes loop), or :scratch.
  - inputs: read by phases but never written
  - outputs: written by the last phase(s) and used after the loop
  - scratch: written and read only within the loop body"
  [phases outer-live-syms]
  (let [all-written (set (map :out phases))
        all-read (apply set/union (map :inputs phases))
        inputs (set/difference all-read all-written)
        outputs (set/intersection all-written outer-live-syms)
        scratch (set/difference all-written outputs)]
    {:inputs inputs
     :outputs outputs
     :scratch scratch}))

;; ================================================================
;; Strategy selection
;; ================================================================

(defn select-strategy
  "Choose :local or :global compound kernel strategy.
  :local — single kernel with __local scratch arrays (fast, n ≤ max_workgroup_size)
  :global — persistent __global buffers with per-phase kernel launches (any n)"
  [n-estimate scratch-count device-id]
  (let [slm (if device-id
              (try (hw/shared-local-memory device-id) (catch Exception _ 65536))
              65536)
        max-wg (if device-id
                 (try
                   (or (get-in (hw/device device-id) [:capabilities :max-workgroup-size])
                       1024)
                   (catch Exception _ 1024))
                 1024)
        ;; Estimate scratch memory: each scratch array uses n * 8 bytes (double)
        scratch-bytes (* scratch-count n-estimate 8)]
    (if (and (<= n-estimate max-wg)
             (<= scratch-bytes (long (* 0.75 slm))))
      :local
      :global)))

;; ================================================================
;; Live symbol analysis
;; ================================================================

(defn- scalar-sym?
  "Is this symbol a scalar parameter (not an operator, var ref, or method call)?
  Principled filter: scalars are simple unqualified symbols that aren't
  special forms, primitive casts, or operators."
  [sym]
  (and (symbol? sym)
       ;; Not qualified (no namespace/ prefix) — excludes var refs and namespaced ops
       (nil? (namespace sym))
       ;; Not a method call (.invk, .foo)
       (not (.startsWith (name sym) "."))
       ;; Not a special form or common operator
       (not (contains? #{'let 'let* 'do 'if 'when 'dotimes 'loop 'loop* 'recur
                         'fn 'fn* 'case 'case* 'quote 'new 'throw 'try
                         'double 'float 'long 'int 'byte 'short 'char 'boolean
                         '+ '- '* '/ 'nil 'true 'false} sym))))

(defn- collect-live-after-dotimes
  "Determine which symbols defined before/in a dotimes are live after it.
  Examines the forms following the dotimes in the enclosing let*/do."
  [rest-forms]
  (if (seq rest-forms)
    (apply set/union (map util/free-syms-flat rest-forms))
    #{}))

;; ================================================================
;; Detection pass
;; ================================================================

(defn- try-detect-compound
  "Check if a dotimes form contains ≥2 par forms → compound kernel candidate.
  Returns the compound-kernel marker form, or nil if not a candidate."
  [dotimes-form rest-forms opts]
  (when (and (seq? dotimes-form)
             (= 'dotimes (first dotimes-form))
             (vector? (second dotimes-form))
             (= 2 (count (second dotimes-form))))
    (let [[_ [trip-var trip-bound] & body] dotimes-form
          phases (extract-phases (cons 'do body))]
      (when (and (>= (count phases) 2)
                 ;; Validate: no opaque calls in the body
                 (not (contains-opaque-calls? (cons 'do body))))
        (let [live-after (collect-live-after-dotimes rest-forms)
              {:keys [inputs outputs scratch]} (classify-arrays phases live-after)
              ;; Try to determine par size from first phase's bound
              par-size (some :bound phases)
              ;; Try to get numeric estimate for strategy selection.
              ;; Default to large value so unknown sizes get :global (safe choice).
              n-estimate (if (number? par-size) par-size 65536)
              device-id (:target-device opts)
              strategy (select-strategy n-estimate (count scratch) device-id)
              execution (execution-plan/compound-execution strategy (count phases) par-size)
              ;; Collect scalar params: actual numeric scalars from the body
              all-arrays (set/union inputs outputs scratch)
              phase-idx-acc (set (mapcat (fn [p] (remove nil? [(:idx p) (:acc p)])) phases))
              scalar-syms (->> (util/free-syms-flat (cons 'do body))
                               (remove nil?)
                               (remove all-arrays)
                               (remove #{trip-var})
                               (remove phase-idx-acc)
                               (filter scalar-sym?)
                               set)]
          (list 'raster.compiler/compound-kernel
                {:execution execution
                 :trip-count-sym trip-var
                 :trip-count-bound trip-bound
                 :inputs (vec (sort-by name inputs))
                 :outputs (vec (sort-by name outputs))
                 :scratch (vec (sort-by name scratch))
                 :scalars (vec (sort-by name (disj scalar-syms nil)))
                 :phases (mapv (fn [p]
                                 (-> (select-keys p [:type :out :inputs :body
                                                     :idx :bound :radius
                                                     :boundary :init :acc])
                                     (update :inputs vec)))
                               phases)}
                dotimes-form))))))

(defn- detect-in-form
  "Recursively walk a form looking for dotimes candidates.
  Replaces dotimes with compound-kernel markers where appropriate."
  [form opts]
  (cond
    ;; let*/let — scan bindings and body for compound opportunities
    (and (seq? form) (contains? #{'let 'let*} (first form)))
    (let [[let-sym bindings & body-exprs] form
          pairs (partition 2 bindings)
          new-bindings (vec (mapcat
                             (fn [[sym expr]]
                               [sym (detect-in-form expr opts)])
                             pairs))
          ;; Process body: check each form, passing rest-forms for liveness
          body-vec (vec body-exprs)
          new-body (mapv (fn [i]
                           (let [f (nth body-vec i)
                                 rest-forms (subvec body-vec (inc i))]
                             (if (and (seq? f) (= 'dotimes (first f)))
                               (or (try-detect-compound f rest-forms opts)
                                   (detect-in-form f opts))
                               (detect-in-form f opts))))
                         (range (count body-vec)))]
      (let [r (apply list let-sym new-bindings new-body)]
        (if-let [m (meta form)] (with-meta r m) r)))

    ;; do — scan statements for compound opportunities
    (and (seq? form) (= 'do (first form)))
    (let [stmts (vec (rest form))
          new-stmts (mapv (fn [i]
                            (let [f (nth stmts i)
                                  rest-forms (subvec stmts (inc i))]
                              (if (and (seq? f) (= 'dotimes (first f)))
                                (or (try-detect-compound f rest-forms opts)
                                    (detect-in-form f opts))
                                (detect-in-form f opts))))
                          (range (count stmts)))
          r (apply list 'do new-stmts)]
      (if-let [m (meta form)] (with-meta r m) r))

    ;; dotimes at top level (no enclosing let/do context)
    (and (seq? form) (= 'dotimes (first form)))
    (or (try-detect-compound form [] opts)
        ;; Recurse into body
        (let [[_ binding & body] form
              r (apply list 'dotimes binding (map #(detect-in-form % opts) body))]
          (if-let [m (meta form)] (with-meta r m) r)))

    ;; Other seq forms — recurse
    (seq? form)
    (let [[head & args] form
          r (if (symbol? head)
              (apply list head (map #(detect-in-form % opts) args))
              (apply list (map #(detect-in-form % opts) form)))]
      (if-let [m (meta form)] (with-meta r m) r))

    :else form))

;; ================================================================
;; Pipeline pass entry point
;; ================================================================

(defn compound-detect-pass
  "Pipeline pass: detect compound kernel opportunities in dotimes loops.

  Walks the form looking for dotimes containing ≥2 par forms and wraps
  them in raster.compiler/compound-kernel markers with metadata for codegen.

  No-op when :target-device is not set (CPU/SIMD path) — compound markers
  only make sense for GPU backends that can fuse them into kernels.

  Validates that the dotimes body is fully expanded (no opaque .invk calls).
  If the body contains un-inlined function calls, it's skipped — the expand
  pass must run before compound-detect for full fusion.

  Returns {:form new-form :stats {:compound-kernels N}}."
  [form opts]
  (let [result (detect-in-form form opts)
        count-compounds (atom 0)]
    (walk/postwalk
     (fn [f]
       (when (and (seq? f) (= 'raster.compiler/compound-kernel (first f)))
         (swap! count-compounds inc))
       f)
     result)
    {:form result
     :stats {:compound-kernels @count-compounds}}))

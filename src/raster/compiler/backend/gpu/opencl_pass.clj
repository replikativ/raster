(ns raster.compiler.backend.gpu.opencl-pass
  "Unified OpenCL pipeline pass.

   Walks S-expressions, replaces par forms with Level Zero kernel
   invocation markers. Uses SegOp-based codegen for generic map/reduce,
   delegates to par_opencl generators for specialized forms.

   This is the GPU counterpart of simd-pass — both consume the same
   par form vocabulary but produce different target code."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac]
            [raster.compiler.passes.parallel.soac-lower]
            [raster.compiler.backend.gpu.segop-opencl :as segop-cl]
            [raster.compiler.backend.gpu.par-opencl :as legacy]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [raster.runtime.hardware :as hw]))

;; ================================================================
;; Single source of declared GPU param types (shared by BOTH compile entries:
;; raster.gpu.core/compile-deftm-internal! and the pipeline's pass-backend).
;; ================================================================

(def array-tag->dtype
  "Deftm array tag (the Java array-class symbol) → element dtype keyword."
  {'doubles :double 'floats :float 'longs :long 'ints :int 'bytes :byte})

(defn derive-param-types
  "Declared scalar + array element types for the GPU emitter, read from a deftm's params + tags
  (the typed-dispatch system already knows these — we read them instead of letting the emitter
  guess from parameter names). Scalar params: long/int→:int, double/float→:float. Array params:
  the tag's element dtype, with float-family (float/double) mapped to the KERNEL dtype (a single-
  precision kernel reads float buffers regardless of a parametric (All [T]) param's default).
  Returns {:scalar-types {sym kw} :array-types {sym kw}}, attached as form metadata that
  opencl-pass reads. ONE derivation, both compile paths."
  [params tags dtype]
  (when (and params tags)
    {:scalar-types (into {} (keep (fn [[p t]]
                                    (case t
                                      (long longs int ints) [p :int]
                                      (double doubles float floats) [p :float]
                                      nil))
                                  (map vector params tags)))
     :array-types (into {} (keep (fn [[p t]]
                                   (when-let [dt (array-tag->dtype t)]
                                     [p (if (#{:float :double} dt) dtype dt)]))
                                 (map vector params tags)))}))

(def ^:private segop-id-counter (atom 0))

(defn- par->segmap
  "Compute SegMap on-the-fly from a par/map! form."
  [form dtype]
  (try
    (let [par-info (par/extract-par-map-info form)
          soac (raster.compiler.ir.soac/par-form->soac
                (:out par-info) form (swap! segop-id-counter inc))
          segops (raster.compiler.passes.parallel.soac-lower/lower-soac
                  soac :ze:0 :dtype (or dtype :double))]
      (first segops))
    (catch Exception _ nil)))

(defn- par->segred
  "Compute SegRed on-the-fly from a par/reduce form."
  [form dtype]
  (try
    (let [sym (gensym "red_")
          soac (raster.compiler.ir.soac/par-form->soac
                sym form (swap! segop-id-counter inc))
          segops (raster.compiler.passes.parallel.soac-lower/lower-soac
                  soac :ze:0 :dtype (or dtype :double))]
      (first segops))
    (catch Exception _ nil)))

(defn- maybe-compile-spirv
  "Optionally compile OpenCL C to SPIR-V."
  [kernel compile-spirv? device-id]
  (if compile-spirv?
    (assoc kernel :spv-bytes
           (legacy/compile-kernel-to-spirv (:source kernel) :device-id device-id))
    kernel))

(defn opencl-pass
  "Pipeline pass: walk S-expression, replace par forms with GPU kernel invocations.

   Uses SegOp-based codegen for par/map! and par/reduce.
   Delegates to legacy generators for specialized forms (stencil, scatter,
   scan, rng-fill, active-ids, reduce-by-key, compound-kernel, map-void).

   Returns {:form new-form :stats {:ze-maps N :ze-reduces N :fallback N}
            :kernels [{:kernel-name :source ...} ...]}

   Options:
     :device-id     — target Level Zero device (default :ze:0)
     :dtype         — :double or :float (default :double)
     :min-elements  — minimum elements for GPU (default 4096)
     :compile-spirv? — compile to SPIR-V now (default false)"
  [form & {:keys [device-id dtype min-elements compile-spirv? scalar-types array-types]
           :or {device-id :ze:0 dtype :double min-elements 4096
                compile-spirv? false}}]
  ;; DECLARED types from derive-param-types (opts) override the name-heuristic fallback in the
  ;; kernel generators — e.g. `features` (Long→int) and `gain-offset` (Double→float, whose name
  ;; would otherwise misfire the "offset"→int heuristic). Form-meta types are the base.
  (let [top-scalar-types (merge (or (:scalar-types (meta form)) {}) scalar-types)
        top-array-types (merge (or (:array-types (meta form)) {}) array-types)
        stats (atom {:ze-maps 0 :ze-reduces 0 :ze-compounds 0 :fallback 0})
        kernels (atom [])

        register-kernel!
        (fn [kernel stat-key]
          (let [k (maybe-compile-spirv kernel compile-spirv? device-id)]
            (swap! stats update stat-key inc)
            (swap! kernels conj k)
            k))

        transform
        (fn transform [form]
          (cond
            ;; === Compound kernel ===
            (and (seq? form) (= 'raster.compiler/compound-kernel (first form)))
            (let [[_ metadata original-dotimes] form
                  strategy (get-in metadata [:execution :strategy])]
              (case strategy
                :local
                (let [k (register-kernel!
                         (legacy/generate-compound-local-kernel metadata
                                                                :dtype dtype :device-id device-id)
                         :ze-compounds)]
                  (legacy/emit-compound-kernel-invocation metadata [k]))
                :global
                (transform original-dotimes)))

            ;; === par/map! — SegOp path ===
            (par/par-map-form? form)
            (let [{:keys [bound out]} (par/extract-par-map-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map! form))
                (if-let [segmap (par->segmap form dtype)]
                  (let [kernel (segop-cl/generate-segmap-kernel segmap out
                                                                :dtype dtype :scalar-types top-scalar-types
                                                                :array-types (merge top-array-types (:array-types (meta form))))]
                    (let [k (register-kernel! kernel :ze-maps)]
                      (list 'raster.gpu.ze-runtime/invoke-registered-kernel
                            (:kernel-name k)
                            (vec (:array-params k))
                            out
                            (vec (:scalar-params k))
                            bound)))
                  ;; SegOp failed — use legacy
                  (let [kernel (legacy/generate-par-map-kernel form
                                                               :dtype dtype :device-id device-id
                                                               :scalar-types top-scalar-types)
                        k (register-kernel! kernel :ze-maps)]
                    (list 'raster.gpu.ze-runtime/invoke-registered-kernel
                          (:kernel-name k)
                          (vec (:array-params k))
                          out
                          (vec (:scalar-params k))
                          bound)))))

            ;; === par/reduce-into — resident SegRed writing a caller-supplied 1-elem buffer ===
            ;; Same SegRed kernel as par/reduce (it already has an `output` param), but the
            ;; resident invoke carries the output buffer (4-arg) so it stays device-resident
            ;; instead of round-tripping a host scalar. Emitted by the reduce-fusion pass.
            (par/par-reduce-into-form? form)
            (let [{:keys [out-buf reduce-form bound]} (par/extract-par-reduce-into-info form)]
              (if-let [segred (par->segred reduce-form dtype)]
                (let [kernel (segop-cl/generate-segred-kernel segred nil :dtype dtype)]
                  (if kernel
                    (let [k (register-kernel! kernel :ze-reduces)]
                      (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                            (:kernel-name k) (vec (:array-params k)) out-buf bound))
                    (let [k (register-kernel!
                             (legacy/generate-par-reduce-kernel reduce-form
                                                                :dtype dtype :device-id device-id)
                             :ze-reduces)]
                      (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                            (:kernel-name k) (vec (:array-params k)) out-buf bound))))
                (let [k (register-kernel!
                         (legacy/generate-par-reduce-kernel reduce-form
                                                            :dtype dtype :device-id device-id)
                         :ze-reduces)]
                  (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                        (:kernel-name k) (vec (:array-params k)) out-buf bound))))

            ;; === par/reduce — SegOp path ===
            (par/par-reduce-form? form)
            (let [{:keys [bound]} (par/extract-par-reduce-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-reduce form))
                (if-let [segred (par->segred form dtype)]
                  (let [kernel (segop-cl/generate-segred-kernel segred nil :dtype dtype)]
                    (if kernel
                      (let [k (register-kernel! kernel :ze-reduces)]
                        (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                              (:kernel-name k)
                              (vec (:array-params k))
                              bound))
                      ;; SegOp codegen failed — use legacy
                      (let [k (register-kernel!
                               (legacy/generate-par-reduce-kernel form
                                                                  :dtype dtype :device-id device-id)
                               :ze-reduces)]
                        (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                              (:kernel-name k)
                              (vec (:array-params k))
                              bound))))
                  ;; SegOp lowering failed — use legacy
                  (let [k (register-kernel!
                           (legacy/generate-par-reduce-kernel form
                                                              :dtype dtype :device-id device-id)
                           :ze-reduces)]
                    (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                          (:kernel-name k)
                          (vec (:array-params k))
                          bound)))))

            ;; === Specialized forms — delegate to legacy generators ===

            ;; par/map-void!
            (par/par-map-void-form? form)
            (let [{:keys [bound]} (par/extract-par-map-void-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-map-void! form))
                (let [kernel (legacy/generate-par-map-void-kernel form
                                                                  :dtype dtype :device-id device-id
                                                                  :array-types top-array-types
                                                                  :scalar-types top-scalar-types)
                      k (register-kernel! kernel :ze-maps)
                      soa-exp (or (:soa-expansions k) {})
                      all-params (:array-params k)
                      plain-params (filterv #(not (contains? soa-exp (symbol (name %)))) all-params)
                      soa-params (filterv #(contains? soa-exp (symbol (name %))) all-params)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel
                        (:kernel-name k)
                        (vec (concat plain-params soa-params))
                        (vec (:scalar-params k))
                        bound))))

            ;; par/scan-exclusive
            (par/par-scan-exclusive-form? form)
            (let [{:keys [bound]} (par/extract-par-scan-exclusive-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-scan-exclusive form))
                (let [kernel (legacy/generate-par-scan-exclusive-kernel form
                                                                        :dtype dtype :device-id device-id)
                      k (maybe-compile-spirv kernel compile-spirv? device-id)
                      block-entry (assoc k :kernel-name (:block-kernel-name k))
                      prop-entry (assoc k :kernel-name (:prop-kernel-name k))]
                  (swap! stats update :ze-maps inc)
                  (swap! kernels conj block-entry)
                  (swap! kernels conj prop-entry)
                  (let [{:keys [out]} (par/extract-par-scan-exclusive-info form)]
                    (list 'raster.gpu.ze-runtime/invoke-registered-scan-exclusive-kernel
                          (:block-kernel-name k) (:prop-kernel-name k)
                          (vec (:input-arrays k)) out bound)))))

            ;; par/rng-fill!
            (par/par-rng-fill-form? form)
            (let [{:keys [seeds n base-seed]} (par/extract-par-rng-fill-info form)
                  k (register-kernel!
                     (legacy/generate-par-rng-fill-kernel :device-id device-id)
                     :ze-maps)]
              (list 'raster.gpu.ze-runtime/invoke-registered-rng-fill-kernel
                    (:kernel-name k) seeds n base-seed))

            ;; par/active-ids!
            (par/par-active-ids-form? form)
            (let [{:keys [ids n-active n-agents base-seed]} (par/extract-par-active-ids-info form)
                  k (register-kernel!
                     (legacy/generate-par-active-ids-kernel :device-id device-id)
                     :ze-maps)]
              (list 'raster.gpu.ze-runtime/invoke-registered-active-ids-kernel
                    (:kernel-name k) ids n-active n-agents base-seed))

            ;; par/stencil!
            (par/par-stencil-form? form)
            (let [{:keys [bound]} (par/extract-par-stencil-info form)]
              (if (and (number? bound) (< bound min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-stencil! form))
                (let [kernel (legacy/generate-par-stencil-kernel form
                                                                 :dtype dtype :device-id device-id
                                                                 :scalar-types top-scalar-types)
                      k (register-kernel! kernel :ze-maps)
                      {:keys [out]} (par/extract-par-stencil-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-kernel
                        (:kernel-name k) (vec (:array-params k))
                        out (vec (:scalar-params k)) bound))))

            ;; par/scatter!
            (par/par-scatter-form? form)
            (let [{:keys [n]} (par/extract-par-scatter-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-scatter! form))
                (let [k (register-kernel!
                         (legacy/generate-par-scatter-kernel form
                                                             :dtype dtype :device-id device-id)
                         :ze-maps)
                      {:keys [out src index stride]} (par/extract-par-scatter-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-scatter-kernel
                        (:kernel-name k) out src index n (when stride stride)))))

            ;; par/reduce-by-key
            (par/par-reduce-by-key-form? form)
            (let [{:keys [n]} (par/extract-par-reduce-by-key-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-reduce-by-key form))
                (let [k (register-kernel!
                         (legacy/generate-par-reduce-by-key-kernel form
                                                                   :dtype dtype :device-id device-id)
                         :ze-reduces)
                      {:keys [out keys vals]} (par/extract-par-reduce-by-key-info form)]
                  (list 'raster.gpu.ze-runtime/invoke-registered-reduce-by-key-kernel
                        (:kernel-name k) out keys vals n))))

            ;; par/butterfly!
            (par/par-butterfly-form? form)
            (let [{:keys [half]} (par/extract-par-butterfly-info form)]
              (if (and (number? half) (< half min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-butterfly! form))
                ;; No dedicated GPU kernel yet — fall back to scalar expansion
                (do (swap! stats update :fallback inc)
                    (par/expand-par-butterfly! form))))

            ;; === Structural recursion (PRESERVE metadata) ===
            ;; A bare (apply list head ...) rebuild drops the form's metadata — in particular
            ;; :raster.op/original on a devirtualized .invk (e.g. blas/dgemm!), which downstream
            ;; GEMM recognition (parse-gpu-step, gpu_plan) reads. Re-attach the original meta.

            (and (seq? form) (contains? #{'let 'let*} (first form)))
            (let [[let-sym bindings & body-exprs] form
                  pairs (partition 2 bindings)
                  new-bindings (vec (mapcat (fn [[sym expr]] [sym (transform expr)]) pairs))
                  new-body (mapv transform body-exprs)]
              (with-meta (apply list let-sym new-bindings new-body) (meta form)))

            (and (seq? form) (= 'do (first form)))
            (with-meta (apply list 'do (mapv transform (rest form))) (meta form))

            (seq? form)
            (let [head (first form)
                  rebuilt (if (symbol? head)
                            (apply list head (mapv transform (rest form)))
                            (mapv transform form))]
              (if (and (instance? clojure.lang.IObj rebuilt) (meta form))
                (with-meta rebuilt (meta form))
                rebuilt))

            :else form))]
    {:form (transform form)
     :stats @stats
     :kernels @kernels}))

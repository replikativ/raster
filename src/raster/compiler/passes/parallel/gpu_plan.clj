(ns raster.compiler.passes.parallel.gpu-plan
  "GPU execution plan pass: rewrite pipeline forms for persistent GPU execution.

  Runs after loop-lift, before backend. Transforms the let* form so that:
  1. All heavy ops (GEMM, elementwise, reduction) run on GPU
  2. Data stays in persistent DeviceBuffers between kernel launches
  3. Only scalar results (loss) are read back to CPU
  4. Weight arrays are converted CPU↔GPU at function boundaries

  Backend-agnostic: generates symbolic kernel invocations that map to
  Level Zero today but can target CUDA/ROCm later via runtime dispatch.

  The pass produces:
    {:form new-let*-form
     :kernels [{:kernel-name :source :dtype ...} ...]
     :gpu-buffers {:sym {:size-expr :dtype :role ...} ...}
     :stats {:gemm-rewrites :map-rewrites :reduce-rewrites ...}}"
  (:require
   [clojure.set :as set]
   [raster.compiler.core.op-descriptor :as op]
   [raster.compiler.passes.parallel.device :as device]
   [raster.compiler.passes.parallel.patterns :as patterns]
   [raster.compiler.passes.parallel.gemm-recognize :as gemm-recognize]
   [raster.compiler.backend.gpu.opencl-codegen :as codegen]))

;; ================================================================
;; Pattern matching helpers
;; ================================================================

(defn- invk-original-op
  "Extract the :raster.op/original symbol from a .invk form.
  Returns the original qualified op symbol, or nil."
  [form]
  (when (and (seq? form) (= '.invk (first form)))
    (:raster.op/original (meta form))))

(def ^:private blas-gemm-ops
  "Set of original op symbols that are BLAS GEMM variants. Keys of the shared
   op-descriptor registry (also used by the RESIDENT extractor in pipeline)."
  (set (keys op/blas-gemm-ops)))

(defn- blas-gemm-call?
  "Match .invk calls to raster.linalg.blas/dgemm variants.
  Uses :raster.op/original metadata instead of substring matching."
  [form]
  (let [inner (or (when-let [[inner _] (patterns/unwrap-do-result form)] inner)
                  form)]
    (contains? blas-gemm-ops (invk-original-op inner))))

(defn- array-alloc?
  "Check if form is an array allocation (double-array, float-array, etc.)."
  [form]
  (and (seq? form)
       (= 2 (count form))
       (contains? #{'double-array 'float-array 'long-array 'int-array
                    'clojure.core/double-array 'clojure.core/float-array
                    'clojure.core/long-array 'clojure.core/int-array}
                  (first form))))

;; ================================================================
;; GEMM variant detection
;; ================================================================

(defn- extract-gemm-invk
  "Extract the .invk form from either direct or do-wrapped BLAS call."
  [form]
  (let [inner (or (first (patterns/unwrap-do-result form)) form)]
    (when (contains? blas-gemm-ops (invk-original-op inner))
      inner)))

(defn- classify-gemm
  "Classify a BLAS .invk call into GEMM variant.
  Handles both direct and do-wrapped (.invk blas/dgemm! ...) patterns.
  Returns {:variant :nn|:tn|:nt, :A sym :B sym :C sym :m expr :k expr :n expr
           :alpha expr :beta expr :result-sym sym-or-nil} or nil.

  Returns nil — 'not a GPU-representable GEMM', so gpu-plan-pass KEEPS the original CPU
  BLAS call — when alpha≠1.0 or beta≠0.0. invoke-registered-gemm! takes only
  (A B C m n k) and OVERWRITES C, so rewriting an accumulating GEMM (beta=1, e.g.
  nn/linear's bias fold) to it silently turned C += A·B into C = A·B. Declining the
  offload costs performance; dropping the scalars cost CORRECTNESS. See
  op/gemm-default-alpha-beta?."
  [form]
  (when (blas-gemm-call? form)
    (let [invk (extract-gemm-invk form)
          result-sym (patterns/unwrap-do-result form)
          original-op (invk-original-op invk)
          args (drop 2 invk)
          variant (get op/blas-gemm-ops original-op :nn)]
      (when (and (>= (count args) 8)
                 (op/gemm-default-alpha-beta? (nth args 6) (nth args 7)))
        {:variant variant
         :A (nth args 0) :B (nth args 1) :C (nth args 2)
         :m (nth args 3) :k (nth args 4) :n (nth args 5)
         :alpha (nth args 6) :beta (nth args 7)
         :result-sym result-sym}))))

;; ================================================================
;; Buffer planning (Phase A)
;; ================================================================

(defn- collect-array-allocs
  "Scan let* bindings for double-array allocations.
  Returns map of {sym {:size-expr expr :binding-idx int}}."
  [bindings-vec]
  (let [pairs (partition 2 bindings-vec)]
    (into {}
          (keep-indexed
           (fn [idx [sym expr]]
             (when (array-alloc? expr)
               [sym {:size-expr (second expr)
                     :binding-idx idx}]))
           pairs))))

(defn- classify-buffers
  "Classify array buffers by role.
  :weight  — from function params (passed in, modified by SGD)
  :intermediate — allocated in body, used between ops
  :input   — from function params, read-only"
  [alloc-map param-set weight-set]
  (merge
    ;; Allocated arrays are intermediates
   (into {} (map (fn [[sym info]] [sym (assoc info :role :intermediate)]) alloc-map))
    ;; Params that are weights
   (into {} (map (fn [p] [p {:role :weight :size-expr nil}]) weight-set))
    ;; Params that are read-only inputs
   (into {} (map (fn [p] [p {:role :input :size-expr nil}])
                 (set/difference param-set weight-set)))))

;; ================================================================
;; Kernel generation and registration
;; ================================================================

(defn- gen-gemm-kernel
  "Generate a GEMM kernel for a classified GEMM op.
  Returns {:kernel-name :source :launch-config-fn :dtype}."
  [gemm-info dtype kernel-counter]
  (let [kname (str "gemm_" (name (:variant gemm-info)) "_" (swap! kernel-counter inc))
        ;; Tile-parametric XMX GEMM (handles arbitrary M,N,K)
        ;; XMX GEMM takes FP16 in, FP32 accum. Output dtype matches storage.
        source (codegen/emit-gemm-tiled kname :c-dtype :float)]
    {:kernel-name kname
     :source source
     :dtype dtype
     :type :gemm
     :variant (:variant gemm-info)}))

(defn- gen-transpose-kernel
  "Generate a transpose kernel. Returns {:kernel-name :source :dtype}."
  [dtype kernel-counter]
  (let [kname (str "transpose_" (swap! kernel-counter inc))]
    {:kernel-name kname
     :source (codegen/emit-transpose-kernel kname :dtype (or dtype :float))
     :dtype (or dtype :float)
     :type :transpose}))

;; ================================================================
;; Operation rewriting (Phase B)
;; ================================================================

(defn- rewrite-gemm
  "Rewrite a BLAS GEMM call to GPU GEMM kernel invocation.
  For :tn and :nt variants, emits transpose + nn-GEMM.
  Returns {:bindings [[sym expr] ...] :kernels [...] :result-sym sym}."
  [gemm-info _buf-plan dtype kernel-counter]
  (let [{:keys [A B C m k n result-sym]} gemm-info]
    (case (:variant gemm-info)
      :nn
      (let [gemm-kernel (gen-gemm-kernel gemm-info dtype kernel-counter)]
        {:bindings
         [[(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-gemm!
                                  (:kernel-name gemm-kernel)
                                  A B C m n k)]]
         :kernels [gemm-kernel]
         :result-sym (or result-sym C)})

      :tn
      ;; A^T @ B: transpose A first, then nn-GEMM
      (let [t-kernel (gen-transpose-kernel dtype kernel-counter)
            gemm-kernel (gen-gemm-kernel (assoc gemm-info :variant :nn) dtype kernel-counter)
            At-sym (with-meta (gensym "At_buf__") {:raster.buffer/hoistable true})]
        {:bindings
         [[At-sym (device/alloc-expr dtype (list '* m k))]
          [(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-transpose!
                                  (:kernel-name t-kernel) A At-sym k m)]
          [(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-gemm!
                                  (:kernel-name gemm-kernel)
                                  At-sym B C m n k)]]
         :kernels [t-kernel gemm-kernel]
         :result-sym (or result-sym C)})

      :nt
      ;; A @ B^T: transpose B first, then nn-GEMM
      (let [t-kernel (gen-transpose-kernel dtype kernel-counter)
            gemm-kernel (gen-gemm-kernel (assoc gemm-info :variant :nn) dtype kernel-counter)
            Bt-sym (with-meta (gensym "Bt_buf__") {:raster.buffer/hoistable true})]
        {:bindings
         [[Bt-sym (device/alloc-expr dtype (list '* k n))]
          [(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-transpose!
                                  (:kernel-name t-kernel) B Bt-sym n k)]
          [(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-gemm!
                                  (:kernel-name gemm-kernel)
                                  A Bt-sym C m n k)]]
         :kernels [t-kernel gemm-kernel]
         :result-sym (or result-sym C)}))))

;; ================================================================
;; Main GPU plan pass
;; ================================================================

(defn gpu-plan-pass
  "Walk the let* form and rewrite for persistent GPU execution.

  Phase A: Scan bindings for array allocations, classify buffers.
  Phase B: Rewrite BLAS calls to GPU GEMM, transpose, axpy kernels.

  Handles the buffer-fuse wrapping pattern:
    (do (.invk blas/dgemm! ...) buf)  — GEMM in do-wrapper
    (do (dotimes [...] ...) buf)       — transpose in do-wrapper
    (do \"string\" (let [...] (dotimes [...] ...)))  — SGD update

  Options:
    :dtype         — :float or :double (default :float)
    :weight-params — set of weight param symbols
    :active-params — all function params

  Returns {:form new-let*-form
           :kernels [{:kernel-name :source :dtype ...} ...]
           :gpu-buffers buffer-plan
           :stats {:gemm-rewrites N :map-rewrites N :reduce-rewrites N}}"
  [let-form & {:keys [dtype weight-params active-params]
               :or {dtype :float}}]
  (if-not (and (seq? let-form) (contains? #{'let 'let*} (first let-form)))
    {:form let-form :kernels [] :gpu-buffers {} :stats {}}
    (let [[let-sym bindings-vec & body-exprs] let-form
          ;; Phase A: buffer planning
          alloc-map (collect-array-allocs bindings-vec)
          param-set (set (or active-params []))
          weight-set (set (or weight-params []))
          buf-plan (classify-buffers alloc-map param-set weight-set)

          ;; Phase B: operation rewriting
          kernel-counter (atom 0)
          all-kernels (atom [])
          gemm-rewrites (atom 0)
          transpose-rewrites (atom 0)
          map-rewrites (atom 0)
          reduce-rewrites (atom 0)
          pairs (partition 2 bindings-vec)

          new-pairs
          (reduce
           (fn [acc [sym expr]]
             (cond
                ;; BLAS GEMM call (direct or do-wrapped) → GPU GEMM kernel(s)
               (blas-gemm-call? expr)
               (if-let [gemm-info (classify-gemm expr)]
                 (let [{:keys [bindings kernels result-sym]}
                       (rewrite-gemm gemm-info buf-plan dtype kernel-counter)]
                   (swap! gemm-rewrites inc)
                   (swap! all-kernels into kernels)
                    ;; Replace original binding: emit rewrite bindings,
                    ;; then assign result-sym to the original binding name
                   (let [bs (vec bindings)]
                     (into acc (conj bs [sym result-sym]))))
                  ;; Can't classify — keep original
                 (conj acc [sym expr]))

                ;; GEMM REDOMAP (matmul written in the raster language, no BLAS call) →
                ;; the SAME rewrite-gemm/emit-gemm-tiled path. #38: the redomap front door
                ;; converges with the BLAS-name-match instead of falling through to the
                ;; naive per-element SegOp kernel. gemm-info carries the identical
                ;; {:variant :A :B :C :m :k :n :alpha :beta :result-sym} contract.
               (gemm-recognize/match-gemm-redomap expr)
               (let [gemm-info (gemm-recognize/match-gemm-redomap expr)
                     {:keys [bindings kernels result-sym]}
                     (rewrite-gemm gemm-info buf-plan dtype kernel-counter)]
                 (swap! gemm-rewrites inc)
                 (swap! all-kernels into kernels)
                 (into acc (conj (vec bindings) [sym result-sym])))

                ;; Nested dotimes transpose → GPU transpose kernel
               (patterns/match-do-wrapped-transpose expr)
               (let [{:keys [src out out-buf rows cols]} (patterns/match-do-wrapped-transpose expr)
                     t-kernel (gen-transpose-kernel dtype kernel-counter)]
                 (swap! transpose-rewrites inc)
                 (swap! all-kernels conj t-kernel)
                 (conj acc
                       [(gensym "_gpu_") (list 'raster.gpu.ze-runtime/invoke-registered-transpose!
                                               (:kernel-name t-kernel) src out rows cols)]
                       [sym out-buf]))

                ;; Everything else: keep as-is
                ;; (par/map!, par/reduce go to backend pass which handles them)
               :else
               (conj acc [sym expr])))
           []
           pairs)

          new-bindings (vec (mapcat identity new-pairs))
          new-form (list* let-sym new-bindings body-exprs)]

      {:form new-form
       :kernels @all-kernels
       :gpu-buffers buf-plan
       :stats {:gemm-rewrites @gemm-rewrites
               :transpose-rewrites @transpose-rewrites
               :map-rewrites @map-rewrites
               :reduce-rewrites @reduce-rewrites}})))

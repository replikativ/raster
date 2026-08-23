(ns raster.compiler.backend.gpu.opencl-pass
  "Unified OpenCL pipeline pass.

   Walks S-expressions, replaces par forms with Level Zero kernel
   invocation markers. Uses SegOp-based codegen for generic map/reduce,
   delegates to par_opencl generators for specialized forms.

   This is the GPU counterpart of simd-pass — both consume the same
   par form vocabulary but produce different target code."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.soac-lower]
            [raster.compiler.backend.gpu.segop-opencl :as segop-cl]
            [raster.compiler.passes.parallel.contract-route :as croute]
            [raster.compiler.backend.gpu.par-opencl :as legacy]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [raster.runtime.hardware :as hw]))

;; Buffer semantics of the emitted kernel-invoke marker: invoke-registered-kernel
;; WRITES its `out` arg in place (arg 2 of [kname inputs out scalars n]) and
;; RETURNS it, so the binding sym is a pure alias of the out buffer. Declared in
;; the registry (not pattern-matched in passes) so resolve-alength's call-through
;; alias tracking resolves `(alength <invoke-binding>)` to the out buffer's alloc
;; size — a shape horizontal fusion produces when a later map's bound reads the
;; length of a fused map's result.
(descriptor/register-op-descriptor!
 'raster.gpu.ze-runtime/invoke-registered-kernel
 {:buffer {:allocates? false :in-place-arg 2}})

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

(def ^:private fatal-reasons
  "A violated invariant is not \"the SegOp path does not cover this form\". Letting one fall through
   to the legacy generator would run the pipeline on with the bug intact."
  #{:raster/fatal :raster/bug})

(defn- segop-attempt
  "Run a SegOp lowering `thunk`, returning the SegOp or nil — and RECORDING why on `stats` when it
   declines, so the choice between the modern SegOp path and `legacy/generate-par-*` stops being
   invisible.

   Both outcomes increment the same `:ze-maps`/`:ze-reduces` counter, so nothing downstream — not
   the kernel record, not explain-pipeline — could say which of the two code generators produced a
   kernel, or why the modern one declined. That is the same warn-and-vanish shape north-star §3.5
   rejects at the SegOp boundary, one door along."
  [stats kind form dtype thunk]
  (let [r (try {:ok (thunk)} (catch Exception e {:err e}))]
    (cond
      (:err r)
      (let [e (:err r)]
        (when (contains? fatal-reasons (:reason (ex-data e))) (throw e))
        (swap! stats update :segop-declined (fnil conj [])
               {:kind kind :op (when (seq? form) (first form))
                :dtype (or dtype :double)
                :reason (or (:reason (ex-data e)) :no-lowering-rule)
                :message (.getMessage e)
                :fallback :legacy-codegen})
        nil)

      (nil? (:ok r))
      (do (swap! stats update :segop-declined (fnil conj [])
                 {:kind kind :op (when (seq? form) (first form))
                  :dtype (or dtype :double)
                  :reason :lowering-produced-nothing
                  :fallback :legacy-codegen})
          nil)

      :else (:ok r))))

(def ^:dynamic *bound-segops*
  "The SegOp records `segop-lower` attached to the binding CURRENTLY being transformed (its
   `::segops` metadata), or nil in body position / when the pass did not run.

   This is the seam that makes the SegOp boundary REAL instead of nominal. `segop-lower` lowered
   every par form with the real target device and stored the result on the binder symbol — and
   nothing read it: this pass re-lowered each form from scratch with `:ze:0` hardcoded, ignoring
   both the stored result and its own `device-id` (north-star §2.1, ledger #6/#11). Now the
   stored SegOp is consumed; re-lowering is the fallback and is COUNTED, so a form that bypasses
   the pass's output shows up in the stats instead of silently taking a second path."
  nil)

(defn- take-bound-segop
  "The precomputed SegOp of `kind` for the current binding, if segop-lower produced one."
  [stats kind pred]
  (when-let [so (first (filter pred *bound-segops*))]
    (swap! stats update :segop-reused (fnil inc 0))
    so))

(defn- par->segmap
  "The SegMap for a par/map! form: the one `segop-lower` already computed for this binding when
   available; otherwise re-lowered here with the REAL `device-id` (it was `:ze:0` hardcoded) and
   counted as `:segop-relowered`. nil (with a recorded decline) ⇒ legacy codegen."
  [stats form dtype device-id]
  (or (take-bound-segop stats :segmap #(instance? raster.compiler.ir.segop.SegMap %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (segop-attempt stats :segmap form dtype
                         #(let [par-info (par/extract-par-map-info form)
                                soac (raster.compiler.ir.soac/par-form->soac
                                      (:out par-info) form (swap! segop-id-counter inc))]
                            (first (raster.compiler.passes.parallel.soac-lower/lower-soac
                                    soac (or device-id :ze:0) :dtype (or dtype :double))))))))

(defn- par->segred
  "The SegRed for a par/reduce form; same consume-then-relower contract as `par->segmap`."
  [stats form dtype device-id]
  (or (take-bound-segop stats :segred #(instance? raster.compiler.ir.segop.SegRed %))
      (do (swap! stats update :segop-relowered (fnil inc 0))
          (segop-attempt stats :segred form dtype
                         #(let [sym (gensym "red_")
                                soac (raster.compiler.ir.soac/par-form->soac
                                      sym form (swap! segop-id-counter inc))]
                            (first (raster.compiler.passes.parallel.soac-lower/lower-soac
                                    soac (or device-id :ze:0) :dtype (or dtype :double))))))))

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
        stats (atom {:ze-maps 0 :ze-reduces 0 :ze-compounds 0 :ze-contracts 0 :fallback 0})
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
                (if-let [segmap (par->segmap stats form dtype device-id)]
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

            ;; === par/contract — tensor contraction, routed through the DPAS legality gate ===
            ;; The routing brain (contract-route) chooses the hardware-optimal kernel: DPAS/XMX
            ;; tensorize when legal (canonical f16 matmul, pitch-aligned) → peak (byte-identical
            ;; to the hand GEMM front door); otherwise the portable register-tiled kernel (any
            ;; dtype, arbitrary dims). Emits a 2D invoke-registered-contraction! marker.
            (croute/par-contract-form? form)
            (let [out-sym (second form)
                  ;; pass the REAL descriptor: route-contraction fed `(or desc {})` into
                  ;; derive-gemm-tile, so the "hardware-derived" tile was derived from an empty map
                  ;; — Arc constants on every device. device-id is already in scope here.
                  ;; CONSUME the SegContract segop-lower recorded (its facts were derived and
                  ;; verified once, at the boundary) — the same take-bound-segop seam map/reduce
                  ;; use. Without one (door C, no pass run) route from the form and count it.
                  bound-sc (take-bound-segop stats :segcontract
                                             #(instance? raster.compiler.ir.segop.SegContract %))
                  _ (when-not bound-sc (swap! stats update :segop-relowered (fnil inc 0)))
                  r (croute/route-contraction
                     form :dtype dtype
                     :facts (:facts bound-sc)
                     :desc (try ((requiring-resolve 'raster.compiler.core.hardware/descriptor-for)
                                 device-id)
                                (catch Throwable _ nil)))
                  ;; :pre-steps (an inserted layout transpose) is PRODUCED by route-contraction
                  ;; under :prefer-peak? but executed by NOTHING — the kernel would index a
                  ;; transposed layout against an untransposed buffer, a wrong answer with no
                  ;; diagnostic. Unreachable today (:prefer-peak? is never set here), so refusing
                  ;; costs nothing now and converts a future silent miscompile into a loud one.
                  _ (when (seq (:pre-steps r))
                      (throw (ex-info (str "contraction: descriptor carries :pre-steps, which no "
                                           "pass executes — the operand would be read untransposed")
                                      {:reason :raster/fatal :pre-steps (:pre-steps r)})))
                  ;; Register the ROUTED KERNEL MAP, not three hand-picked keys. Dropping
                  ;; :c-op/:identity-val made invoke-reduction-kernel fall back to its
                  ;; `:or {c-op "+" identity-val 0.0}` for the HOST-SIDE FINAL COMBINE — so a
                  ;; full reduction with :combine max or :combine * spanning more than one
                  ;; workgroup silently SUMMED the per-group partials. Reachable today: a
                  ;; 0-free-axis contraction at :float/:double is exactly what this pass routes.
                  ;; :strategy/:fallback-reason/:declines/:tile are the ROUTING DECISION. They
                  ;; were dropped here, so no diagnostic could say which leaf a kernel came from
                  ;; or why a faster one was refused. (:scalar-params is kept for the existing
                  ;; consumers; route-contraction actually returns :scalar-args — ledger.)
                  k (register-kernel! (merge (select-keys r [:c-op :identity-val :array-params
                                                             :scalar-params :out-dtype
                                                             :strategy :fallback-reason :declines :tile])
                                             {:kernel-name (:kernel-name r) :source (:source r)
                                              :dtype (:dtype r)})
                                      :ze-contracts)]
              ;; A full reduction (0 free axes) has its own two-phase launch protocol + a
              ;; host-side final combine — emit the reduction invoke, not the 2-D contraction one.
              (if (= :reduction (:invoke r))
                (list 'raster.gpu.ze-runtime/invoke-reduction-kernel
                      (:kernel-name k)
                      (vec (:array-params r))
                      (:reduce-bound r))
              ;; Pass the descriptor through INTACT — scalar-args as data (the exact shape
              ;; launch-2d! wants), explicit out-dtype/out-elems. Any :strategy works; the
              ;; old (count scalar-args) + [m n k] reconstruction only covered :dpas/:regtiled.
              (list 'raster.gpu.ze-runtime/invoke-registered-contraction!
                    (:kernel-name k)
                    (vec (:array-params r))       ; operand symbols (vector literal evaluates them)
                    out-sym
                    (:dtype r)
                    (:out-dtype r)
                    (:out-elems r)                ; may be a symbolic expr (symbolic axis bounds)
                    (vec (:wg r))
                    (vec (:grid r))
                    (vec (:scalar-args r)))))

            ;; === par/reduce-into — resident SegRed writing a caller-supplied 1-elem buffer ===
            ;; Same SegRed kernel as par/reduce (it already has an `output` param), but the
            ;; resident invoke carries the output buffer (4-arg) so it stays device-resident
            ;; instead of round-tripping a host scalar. Emitted by the reduce-fusion pass.
            (par/par-reduce-into-form? form)
            (let [{:keys [out-buf reduce-form bound]} (par/extract-par-reduce-into-info form)]
              (if-let [segred (par->segred stats reduce-form dtype device-id)]
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
                (if-let [segred (par->segred stats form dtype device-id)]
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

            ;; par/gather — out[e*stride+d] = src[index[e]*stride+d]. Writes every
            ;; output element once (no atomics/accumulation), so it lowers to a map-void
            ;; kernel and binds through the existing resident map path.
            (par/par-gather-form? form)
            (let [{:keys [n stride]} (par/extract-par-gather-info form)]
              (if (and (number? n) (< n min-elements))
                (do (swap! stats update :fallback inc)
                    (par/expand-par-gather! form))
                (let [k (register-kernel!
                         (legacy/generate-par-gather-kernel form
                                                            :dtype dtype :device-id device-id)
                         :ze-maps)
                      {:keys [out src index]} (par/extract-par-gather-info form)
                      strip-cast (fn [x] (if (and (seq? x)
                                                  (#{'long 'int 'clojure.core/long 'clojure.core/int} (first x)))
                                           (second x) x))]
                  (list 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel
                        (:kernel-name k) (vec [out src index])
                        (if stride [(strip-cast stride)] []) n))))

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
                  ;; the binder symbol carries what segop-lower computed for this init — make it
                  ;; visible to the par branches so they consume it instead of re-lowering
                  new-bindings (vec (mapcat (fn [[sym expr]]
                                              [sym (binding [*bound-segops*
                                                             (:raster.compiler.passes.parallel.segop-lower-pass/segops (meta sym))]
                                                     (transform expr))])
                                            pairs))
                  new-body (mapv transform body-exprs)]
              (with-meta (apply list let-sym new-bindings new-body) (meta form)))

            ;; a body-position par form: segop-lower wraps it in (do …) carrying ::body-segops
            ;; — consume that exactly as a binding's ::segops, so body forms stop re-lowering
            (and (seq? form) (= 'do (first form))
                 (:raster.compiler.passes.parallel.segop-lower-pass/body-segops (meta form)))
            (binding [*bound-segops* (:raster.compiler.passes.parallel.segop-lower-pass/body-segops (meta form))]
              (with-meta (apply list 'do (mapv transform (rest form))) (meta form)))

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

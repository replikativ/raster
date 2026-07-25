(ns raster.compiler.passes.parallel.contract-route
  "Routing brain for tensor contractions: a `(raster.par/contract …)` SOAC form → the
   hardware-optimal kernel choice, via the tensorize LEGALITY GATE.

   This is the pipeline INTEGRATION seam for the SOAC contraction ladder — the piece that
   makes the emitters load-bearing instead of proven-but-bypassed. `route-contraction`
   decides: if the DPAS/XMX gate accepts (canonical matmul, f16, pitch-aligned) → the peak
   tensorized kernel (byte-identical to the hand-wired GEMM front door); otherwise → the
   portable register-tiled kernel (any dtype, arbitrary dims). Same decision the walker/
   opencl-pass makes when it meets a contraction; kept in ONE place so the gate's hardware
   knowledge lives with the emitters, not scattered across passes.

   Returns a launch-ready descriptor. Strategies: :segmap (0 contract axes), :naive-segred
   (general), :regtiled (portable tiled), :dpas (f16 peak), :dp4a (int8 peak), :quant-naive
   (int8 widening). Keys: :kernel-name :source :array-params (binding order) :dtype :out-dtype
   :out-elems :wg [x y] :grid [gx gy] :scalar-args [{:type :value}…] :dims, plus optional
   :fallback-reason, :scheme (quant decode) and :pre-steps (inserted layout rearranges)."
  (:require [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.axis-map :as am]))

(defn par-contract-form?
  "Is `form` a (raster.par/contract out free-axes contract-axes body & opts) form?"
  [form]
  (and (seq? form) (= 'raster.par/contract (first form))))

(defn- ceil-div
  "⌈a/b⌉. `a` may be a SYMBOLIC expression (symbolic axis bounds) — then the quotient is built
   as a form the call site evaluates at runtime, mirroring how par/map!'s `bound` is handled."
  [a b]
  (if (number? a)
    (long (Math/ceil (/ (double a) (double b))))
    (list 'quot (list 'clojure.core/+ a (dec (long b))) (long b))))

(declare route-2free-1contract route-quant)

(defn route-contraction
  "Route a contraction form to the hardware-optimal kernel via the DPAS legality gate.
   dtype selects the element type of the intended kernel (:half tries DPAS; :byte/:int8 tries
   the int8 quant leaves — dp4a for the :nt operand layout, quant naive-widening for :nn;
   anything else, or a gate rejection, falls back to the register-tiled portable kernel).
   scheme = the quant decode descriptor {:scale :a-zp :b-zp} for int8 (default {:scale 1.0})."
  [contract-form & {:keys [dtype scheme prefer-peak? desc tile epilogue]
                    :or {dtype :half scheme {:scale 1.0} prefer-peak? false}}]
  (let [out-sym (second contract-form)
        free-axes (nth contract-form 2)
        contract-axes (nth contract-form 3)
        n-free (count free-axes)
        n-contract (count contract-axes)
        ;; Number of output elements = product of the free-axis bounds. Bounds may be SYMBOLS
        ;; (contract-lower supports them and puts them in :scalars), so build the product
        ;; symbolically and only fold to an int when every bound is a literal.
        free-bounds (map second free-axes)
        nseg (if (every? number? free-bounds)
               (reduce * 1 free-bounds)
               (reduce (fn [a b] (list 'clojure.core/* a b)) free-bounds))
        ;; memoized so the cond's test arm doesn't regenerate the kernel
        ;; a fused contraction carries its epilogue in the form's trailing opts (par-fusion's
        ;; fuse-contract-map puts it there); an explicit :epilogue kwarg overrides.
        form-opts (apply hash-map (drop 5 contract-form))
        epilogue (or epilogue (:epilogue form-opts))
        tensorize-plan (memoize #(route-2free-1contract contract-form out-sym dtype desc tile epilogue))]
    (cond
      ;; int8 → the quant leaves (dp4a for :nt, quant naive-widening for :nn)
      (#{:byte :int8} dtype)
      (route-quant contract-form out-sym scheme n-free n-contract nseg prefer-peak?)

      ;; 0 FREE axes → a full reduction to a scalar. This is the last cell of contract's
      ;; algebra: (n free, 0 contract) = map, (n, n) = contraction, (0, n) = REDUCTION. The
      ;; SegSpace then has only the reduced dim — exactly the 1-D shape (seg-space-1d?) that
      ;; generate-segred-kernel's two-phase tree reduction already consumes, so no new emitter.
      ;; Its launch protocol differs (two phases + a host-side final combine), so the descriptor
      ;; says so with :invoke :reduction rather than pretending it is a 2-D kernel launch.
      (zero? n-free)
      (let [sr (cl/contract-form->segred contract-form :dtype dtype)
            k (sco/generate-segred-kernel sr out-sym :dtype dtype)
            red-bound (second (first contract-axes))]
        {:strategy :full-reduce
         :invoke :reduction
         :kernel-name (:kernel-name k) :source (:source k)
         :array-params (:array-params k)
         :dtype dtype :out-dtype dtype :out-elems 1
         :n-phases (:n-phases k)
         :reduce-bound red-bound          ; element count the reduction spans
         :scalar-args [] :dims [1]})

      ;; 0 contract axes → outer product / broadcast → pure N-D SegMap (1-D launch)
      (zero? n-contract)
      (let [sm (cl/contract-form->segmap contract-form :dtype dtype)
            {:keys [kernel-name source array-params]} (sco/generate-segmap-nd-kernel sm out-sym :dtype dtype)]
        {:strategy :segmap
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :out-dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         :scalar-args [{:type :int :value nseg}] :out-elems nseg :dims [nseg]})

      ;; 2 free + 1 contract → the tensorize fast path (DPAS if legal, else regtiled).
      ;; Returns nil when the form fails a TENSORIZE structural precondition (symbolic dims,
      ;; non-+ combine, non-product element …) — then we fall through to the general naive
      ;; leaf rather than hard-failing a perfectly legal contraction.
      (and (= 2 n-free) (= 1 n-contract) (tensorize-plan))
      (tensorize-plan)

      ;; everything else (n-free≠2, n≥2 contract axes, or a tensorize-ineligible 2-free form)
      ;; → naive segmented reduce (general: any dtype, symbolic dims, any assoc combine).
      ;; contract-form->segred flattens n≥2 contract axes into one innermost dim.
      :else
      (let [sr (cl/contract-form->segred contract-form :dtype dtype)
            {:keys [kernel-name source array-params]} (sco/generate-segmented-reduce-kernel sr out-sym :dtype dtype)]
        {:strategy :naive-segred
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :out-dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         :scalar-args [{:type :int :value nseg}] :out-elems nseg :dims [nseg]}))))

(defn- route-2free-1contract
  "The tensorize fast path: DPAS if the gate accepts, else the register-tiled portable kernel.
   Returns nil if the form fails a structural precondition of BOTH (the emitters signal that
   with ex-info) — the caller then routes to the general naive leaf."
  [contract-form out-sym dtype desc tile epilogue]
  (try
   (let [sr (cl/contract-form->segred contract-form :dtype dtype)
         dpas (sco/generate-dpas-contraction-kernel sr out-sym :dtype dtype :desc desc :tile tile
                                                    :epilogue epilogue)]
    (if (:tensorized dpas)
      (let [[M N _L] (:dims dpas)
            {:keys [block-m block-n]} (:tile dpas)]
        {:strategy :dpas
         :kernel-name (:kernel-name dpas)
         :source (:source dpas)
         :array-params (:array-params dpas)          ; [row col] = [A-slot B-slot]
         :dtype :half :out-dtype :half :out-elems (* M N)
         :tile (:tile dpas)                          ; the DERIVED tile actually emitted
         :fused-epilogue (boolean epilogue)
         :wg (:workgroup dpas)                       ; derived from that tile
         :grid [(ceil-div N block-n) (ceil-div M block-m)]  ; [gc-n gc-m] (id0=N, id1=M)
         :scalar-args (mapv (fn [v] {:type :int :value (int v)}) (:dims dpas))  ; [m n k] params
         :dims (:dims dpas)})
      ;; gate rejected (dtype/orientation/pitch) → portable register-tiled kernel
      (let [rt (sco/generate-regtiled-contraction-kernel sr out-sym :dtype dtype)
            [bm bn _bk] (:block rt)
            [M N _L] (:dims rt)]
        {:strategy :regtiled
         :fallback-reason (:reason dpas)
         :kernel-name (:kernel-name rt)
         :source (:source rt)
         :array-params (:array-params rt)            ; sorted-by-name (dims baked in source)
         :dtype (:dtype rt) :out-dtype (:dtype rt) :out-elems (* M N)
         :wg (:workgroup rt)
         :grid [(ceil-div N bn) (ceil-div M bm)]
         :scalar-args []                             ; regtiled bakes dims → no scalar params
         :dims (:dims rt)})))
   (catch clojure.lang.ExceptionInfo _ nil)))

(defn- quant-descriptor
  "A launch descriptor for an int8 quant leaf (dp4a or quant-naive). Both are 1-D kernels with
   signature (…arrays…, out(f32), float scale, int _nseg): int8 operands in, dequantized f32 out."
  [strategy k out-dtype scale nseg]
  {:strategy strategy
   :kernel-name (:kernel-name k) :source (:source k)
   :array-params (:array-params k)          ; [row col] binding order (dp4a) / sorted (quant)
   :dtype :byte :out-dtype out-dtype
   :scheme (:scheme k)
   :wg [256 1] :grid [(ceil-div nseg 256) 1]
   :scalar-args [{:type :float :value (float scale)} {:type :int :value nseg}]
   :out-elems nseg
   :dims (:dims k)})

(defn- operand-map
  "The declared axis-map for `arr` if the form carries :maps, else DERIVE one by checking the
   aget index against the canonical row-major layout for `expected-axes`. Returns nil when the
   index is NOT that layout — which is the point: the old rewrite ASSUMED canonical strides and
   silently miscompiled a non-canonical operand."
  [contract-form arr idx expected-axes extents]
  (let [opts (apply hash-map (drop 5 contract-form))
        declared (get-in opts [:maps arr])]
    (cond
      declared (when (am/canonical? declared expected-axes) declared)
      :else (let [cand (am/of-axes (mapv vector expected-axes extents))]
              (when (am/index-matches? cand idx) cand)))))

(defn- retarget-to-layout
  "Rewrite a 2-operand contraction so `arr`'s operand has the layout the LEAF requires, inserting
   a physical transpose pre-step for the difference. The general form of the old :nn→:nt special
   case: the required layout is declared by the leaf, the actual layout comes from the operand's
   map, and the mismatch is `am/permutation` — not a hand-written index rewrite.

   VERIFIES the operand's actual layout before rewriting (the previous version substituted
   assumed canonical strides having only checked which axis SYMBOLS appeared, so a [K,M]-strided
   or batch-offset operand silently computed the wrong thing). Returns {:form :pre-step} or nil.

   Scope: a 2-free/1-contract product of two agets whose col operand needs group transposition."
  [contract-form required-col-axes dtype]
  (let [[_ out free-axes contract-axes body] contract-form]
    (when (and (= 2 (count free-axes)) (= 1 (count contract-axes))
               (seq? body) (#{'* 'clojure.core/* 'raster.numeric/*} (first body)) (= 3 (count body)))
      (let [[i-sym M] (first free-axes) [j-sym N] (second free-axes) [l-sym K] (first contract-axes)
            ext {i-sym M j-sym N l-sym K}
            agets (mapv ce/normalize-array-prims (rest body))
            agets (filterv #(and (seq? %) (= 'aget (first %))) agets)]
        (when (= 2 (count agets))
          (let [syms-of (fn [g] (set (filter symbol? (tree-seq coll? seq (nth g 2)))))
                col (first (filter #(and (contains? (syms-of %) j-sym)
                                         (not (contains? (syms-of %) i-sym))) agets))
                row (first (filter #(and (contains? (syms-of %) i-sym)
                                         (not (contains? (syms-of %) j-sym))) agets))]
            (when (and col row (not= col row))
              (let [col-arr (nth col 1)
                    actual-axes [l-sym j-sym]          ; the :nn storage we can retarget from
                    cmap (operand-map contract-form col-arr (nth col 2) actual-axes
                                      (mapv ext actual-axes))
                    want (am/of-axes (mapv (fn [a] [a (ext a)]) required-col-axes))]
                ;; only proceed when the ACTUAL layout is verified and the difference is a
                ;; genuine 2-D group transposition
                (when (and cmap (am/transposed-2d? cmap want))
                  (let [col-t (gensym (str (name col-arr) "__t"))
                        new-body (list '* row (list 'aget col-t (am/index-expr want)))]
                    {:form (list 'raster.par/contract out free-axes contract-axes new-body
                                 :maps (assoc (get (apply hash-map (drop 5 contract-form)) :maps {})
                                              col-t want))
                     :pre-step {:op :transpose :src col-arr :dst col-t
                                :rows (ext l-sym) :cols (ext j-sym) :dtype dtype}}))))))))))

(defn- quant-naive!
  "The int8 naive-widening leaf, with a CLEAR error when the operand layout isn't one it can
   index. There is no correct generic fallback for int8: the generic naive segred accumulates in
   the element type, and int8×int8 must widen to int32 — so we fail loudly rather than emit a
   silently-wrong kernel."
  [sr out-sym scheme]
  (try (sco/generate-quant-contraction-kernel sr out-sym :scheme scheme)
       (catch clojure.lang.ExceptionInfo e
         (throw (ex-info (str "int8 contraction: no quant leaf handles this operand layout ("
                              (:reason (ex-data e)) "). int8 requires canonical row-major operands"
                              " (A[i,l]=i·K+l, B[l,j]=l·N+j) or, for dp4a, B[j,l]=j·K+l.")
                         (assoc (ex-data e) :dtype :byte) e)))))

(defn- route-quant
  "Route an int8 (:byte) contraction. 2-free/1-contract: try the dp4a peak leaf (requires the
   :nt operand layout — B stored [N,K], K-contiguous, K%4==0); if that orientation isn't met,
   either INSERT a byte-transpose pre-step so :nn reaches dp4a (B3-insert, `prefer-peak?`), or
   fall to the quant naive-widening kernel (:nn, default). The emitters ASSERT their own layout
   requirements, so a thrown AssertionError is the (clean) 'not this leaf' signal — dp4a and
   quant-naive are complementary (:nt vs :nn). Non-2-free / multi-contract int8 is deferred."
  [contract-form out-sym scheme n-free n-contract nseg prefer-peak?]
  (let [scale (get scheme :scale 1.0)]
    (if (and (= 2 n-free) (= 1 n-contract))
      (let [sr (cl/contract-form->segred contract-form :dtype :byte)
            dp4a (try (sco/generate-dp4a-contraction-kernel sr out-sym :scheme scheme)
                      (catch clojure.lang.ExceptionInfo _ nil))]
        (cond
          dp4a  (quant-descriptor :dp4a dp4a :float scale nseg)     ; :nt → peak int8 leaf
          ;; :nn + prefer-peak? → transpose col operand, then dp4a (B3-insert)
          prefer-peak?
          (if-let [{form* :form pre-step :pre-step}
                   (retarget-to-layout contract-form '[j l] :byte)]   ; dp4a needs B as [N,K]
            (let [srt (cl/contract-form->segred form* :dtype :byte)
                  k (sco/generate-dp4a-contraction-kernel srt out-sym :scheme scheme)]
              (assoc (quant-descriptor :dp4a k :float scale nseg) :pre-steps [pre-step]))
            (quant-descriptor :quant-naive (quant-naive! sr out-sym scheme) :float scale nseg))
          :else (quant-descriptor :quant-naive (quant-naive! sr out-sym scheme) :float scale nseg)))
      (throw (ex-info "route-quant: int8 supported for 2 free + 1 contract axes (C1 first cut)"
                      {:n-free n-free :n-contract n-contract})))))

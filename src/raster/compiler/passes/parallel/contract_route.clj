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

   Returns a launch-ready descriptor: {:strategy :dpas|:regtiled, :kernel-name, :source,
   :array-params (binding order), :dtype, :wg [x y], :grid [gx gy], :scalar-args [{:type :int
   :value n}…], :dims [M N L]}."
  (:require [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]))

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
  [contract-form & {:keys [dtype scheme prefer-peak?] :or {dtype :half scheme {:scale 1.0} prefer-peak? false}}]
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
        tensorize-plan (memoize #(route-2free-1contract contract-form out-sym dtype))]
    (cond
      ;; int8 → the quant leaves (dp4a for :nt, quant naive-widening for :nn)
      (#{:byte :int8} dtype)
      (route-quant contract-form out-sym scheme n-free n-contract nseg prefer-peak?)

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
  [contract-form out-sym dtype]
  (try
   (let [sr (cl/contract-form->segred contract-form :dtype dtype)
         dpas (sco/generate-dpas-contraction-kernel sr out-sym :dtype dtype)]
    (if (:tensorized dpas)
      (let [[M N _L] (:dims dpas)
            {:keys [block-m block-n]} (:tile dpas)]
        {:strategy :dpas
         :kernel-name (:kernel-name dpas)
         :source (:source dpas)
         :array-params (:array-params dpas)          ; [row col] = [A-slot B-slot]
         :dtype :half :out-dtype :half :out-elems (* M N)
         :wg (:workgroup dpas)                       ; derived from the emitted tile
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

(defn- nn-matmul->nt-plan
  "B3-insert: turn a canonical :nn int8 matmul form C[i,j]=Σ_l A[i,l]·B[l,j] (B stored [K,N])
   into the dp4a-ready :nt form + a byte-transpose PRE-STEP that produces Bᵀ[N,K]. Returns
   {:nt-form :pre-step} or nil if the form isn't a canonical 2-operand matmul. The col operand
   (its index contains the free1 axis j) is the one transposed; row operand A stays [M,K]."
  [contract-form]
  (let [[_ _out free-axes contract-axes body] contract-form]
    (when (and (= 2 (count free-axes)) (= 1 (count contract-axes))
               (seq? body) (= '* (first body)) (= 3 (count body)))
      (let [[i-sym M] (first free-axes) [j-sym N] (second free-axes) [l-sym K] (first contract-axes)
            agets (filter #(and (seq? %) (= 'aget (first %))) (rest body))]
        (when (= 2 (count agets))
          (let [syms-of (fn [g] (set (filter symbol? (tree-seq coll? seq (nth g 2)))))
                col (first (filter #(contains? (syms-of %) j-sym) agets))
                row (first (filter #(contains? (syms-of %) i-sym) agets))]
            (when (and col row (not= col row))
              (let [row-arr (nth row 1) col-arr (nth col 1)
                    col-t (symbol (str (name col-arr) "__t"))   ; the transposed col operand
                    nt-body (list '* (list 'aget row-arr (list '+ (list '* i-sym K) l-sym))
                                  (list 'aget col-t (list '+ (list '* j-sym K) l-sym)))]
                {:nt-form (list 'raster.par/contract (second contract-form)
                                [[i-sym M] [j-sym N]] [[l-sym K]] nt-body)
                 ;; transpose the ORIGINAL col operand [K,N] → [N,K] at BYTE granularity
                 :pre-step {:op :transpose :src col-arr :dst col-t :rows K :cols N :dtype :byte}}))))))))

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
                      (catch AssertionError _ nil))]
        (cond
          dp4a  (quant-descriptor :dp4a dp4a :float scale nseg)     ; :nt → peak int8 leaf
          ;; :nn + prefer-peak? → transpose col operand, then dp4a (B3-insert)
          prefer-peak?
          (if-let [{:keys [nt-form pre-step]} (nn-matmul->nt-plan contract-form)]
            (let [srt (cl/contract-form->segred nt-form :dtype :byte)
                  k (sco/generate-dp4a-contraction-kernel srt out-sym :scheme scheme)]
              (assoc (quant-descriptor :dp4a k :float scale nseg) :pre-steps [pre-step]))
            (let [q (sco/generate-quant-contraction-kernel sr out-sym :scheme scheme)]
              (quant-descriptor :quant-naive q :float scale nseg)))
          :else (let [q (sco/generate-quant-contraction-kernel sr out-sym :scheme scheme)]
                  (quant-descriptor :quant-naive q :float scale nseg))))  ; :nn → naive widening
      (throw (ex-info "route-quant: int8 supported for 2 free + 1 contract axes (C1 first cut)"
                      {:n-free n-free :n-contract n-contract})))))

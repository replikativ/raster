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
  (:require [raster.compiler.core.op-descriptor :as od]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as cf]))

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

(defn- contract-operand-arrays
  "The array symbols the contraction BODY reads, i.e. every `(aget arr …)` target. These are the
   contraction's own operands; a staged contraction's per-stage scale arrays are NOT among them —
   those are declared on the stages and surfaced separately as :lift-operands, so the two groups
   cannot be confused at bind time."
  [body]
  (distinct (map :sym (od/aget-reads body))))

(defn kernel-signature-params
  "The parameter list of the single __kernel in `src`, split at top-level commas. Used to check a
   launch descriptor against the kernel it actually describes."
  [src]
  (when-let [i (str/index-of src "__kernel void")]
    (let [open (str/index-of src "(" i)
          ;; scan to the matching close paren
          close (loop [k (inc open) depth 1]
                  (cond (>= k (count src)) nil
                        (= \( (.charAt ^String src k)) (recur (inc k) (inc depth))
                        (= \) (.charAt ^String src k)) (if (= 1 depth) k (recur (inc k) (dec depth)))
                        :else (recur (inc k) depth)))]
      (when close
        (->> (str/split (subs src (inc open) close) #",")
             (map str/trim)
             (remove str/blank?)
             vec)))))

(defn validate-descriptor
  "Check a launch descriptor against the kernel source it describes, and THROW on a mismatch.

   The bug class this exists for has now bitten twice: a descriptor that under-describes the kernel
   fails at LAUNCH, not at compile — the kernel is valid C, the caller simply binds the wrong number
   of arguments. First occurrence: invoke-registered-contraction! reconstructed scalar-args from a
   `case` with no default, so four of six strategies crashed. Second: an epilogue's operand arrays
   were declared in the signature but absent from the descriptor, so a caller bound 6 args to a
   7-arg kernel. Both are mechanically detectable by comparing the emitted signature with what the
   descriptor says to bind, which is what this does.

   Pointer params must equal (array-params + 1 output + epilogue-operands). The rest depends on the
   INVOKE PROTOCOL, of which there are two — writing this validator is what forced them to be stated
   explicitly instead of living implicitly in two call sites:

     default (:invoke nil)  invoke-registered-contraction! binds :scalar-args positionally and
                            launches with :wg/:grid — so all three must be present and match.
     :invoke :reduction     invoke-reduction-kernel supplies the kernel's single trailing count
                            param from :reduce-bound and computes its own two-phase launch geometry
                            — so :scalar-args must be EMPTY and :wg/:grid ABSENT.

   Returns the descriptor unchanged when consistent."
  [{:keys [strategy kernel-name source array-params scalar-args epilogue-operands lift-operands
           epilogue-scalars out-elems wg grid invoke reduce-bound] :as d}]
  (let [params (kernel-signature-params source)
        _ (when (nil? params)
            (throw (ex-info "contract descriptor: no __kernel signature found in source"
                            {:strategy strategy :kernel-name kernel-name})))
        ptr? (fn [p] (str/includes? p "*"))
        n-ptr (count (filter ptr? params))
        n-scalar (count (remove ptr? params))
        ;; EXTRA operand arrays are pointer params too, whichever seam declared them: an
        ;; epilogue's (bias/residual/scale, appended after the dims) or a staged contraction's
        ;; lift operands (the per-block scales, bound between the operands and the output).
        expect-ptr (+ (count array-params) 1 (count epilogue-operands) (count lift-operands))
        ;; TWO invoke protocols, made explicit here because the validator forced the question:
        ;;   :invoke nil (default) — invoke-registered-contraction! binds :scalar-args positionally,
        ;;                           so they must match the kernel's scalar params exactly.
        ;;   :invoke :reduction    — invoke-reduction-kernel supplies the kernel's single trailing
        ;;                           count param itself, from :reduce-bound; :scalar-args stays empty.
        ;; an epilogue's SCALARS are kernel scalar params too — they are emitted into the
        ;; signature by epilogue-splice, so a descriptor that omits them under-counts and the
        ;; capability becomes unusable (which is what pushed `:scheme` into a private channel)
        expect-scalar (if (= :reduction invoke)
                        1
                        (+ (count scalar-args) (count epilogue-scalars)))]
    (cond
      (not= n-ptr expect-ptr)
      (throw (ex-info (str "contract descriptor: kernel takes " n-ptr " pointer params but the "
                           "descriptor binds " expect-ptr " (" (count array-params) " operands + out"
                           (when (seq epilogue-operands)
                             (str " + " (count epilogue-operands) " epilogue"))
                           (when (seq lift-operands)
                             (str " + " (count lift-operands) " lift")) ")")
                      {:strategy strategy :kernel-name kernel-name :params params
                       :array-params array-params :epilogue-operands epilogue-operands
                       :lift-operands lift-operands}))
      (not= n-scalar expect-scalar)
      (throw (ex-info (str "contract descriptor: kernel takes " n-scalar " scalar params but the "
                           (if (= :reduction invoke)
                             "reduction invoke supplies 1 (the count, from :reduce-bound)"
                             (str "descriptor supplies " expect-scalar " scalar-args")))
                      {:strategy strategy :kernel-name kernel-name :params params
                       :invoke invoke :scalar-args scalar-args}))
      (and (= :reduction invoke) (seq scalar-args))
      (throw (ex-info "contract descriptor: :invoke :reduction must leave :scalar-args empty (the count comes from :reduce-bound)"
                      {:strategy strategy :scalar-args scalar-args}))
      (nil? out-elems)
      (throw (ex-info "contract descriptor: :out-elems is required (the invoke sizes the output with it)"
                      {:strategy strategy}))
      ;; wg/grid belong to the 2-D launch contract only. The reduction invoke computes its own
      ;; two-phase geometry, so a :reduction descriptor legitimately carries neither — a third
      ;; protocol difference this validator forced into the open rather than leaving implicit.
      (and (not= :reduction invoke)
           (not (and (vector? wg) (= 2 (count wg)) (vector? grid) (= 2 (count grid)))))
      (throw (ex-info "contract descriptor: :wg and :grid must both be 2-element vectors"
                      {:strategy strategy :wg wg :grid grid}))
      (and (= :reduction invoke) (or (some? wg) (some? grid)))
      (throw (ex-info "contract descriptor: :invoke :reduction must not carry :wg/:grid (the invoke owns the two-phase launch)"
                      {:strategy strategy :wg wg :grid grid}))
      (and (= :reduction invoke) (nil? reduce-bound))
      (throw (ex-info "contract descriptor: :invoke :reduction requires :reduce-bound"
                      {:strategy strategy}))
      :else d)))

(declare route-2free-1contract route-quant)

(def ^:private splice-capable-strategies
  "Strategies whose emitter has a store splice, and can therefore honour an :epilogue.
   A WHITELIST on purpose: refuse by ABSENCE of support, never by a blacklist of shapes — a
   blacklist got this wrong twice, most recently by exempting the only shape production produces.
   `:dpas` splices through emit-gemm-tiled; the three staged-emitter strategies splice at the store
   (which is how a quant dequant scale is expressed since :scheme was deleted)."
  #{:dpas :dp4a :quant-naive :staged-segred})

(defn- epilogue-honoured-or-refused
  "An :epilogue is a STORE SPLICE, and only the DPAS leaf has one. Every other leaf drops it
   silently — the consumer's computation vanishes with no error.

   Decided AFTER routing, on the strategy actually chosen, because a pre-routing shape test got this
   wrong twice: a shape blacklist exempted (2 free, 1 contract) — the only shape production produces
   — and a version keyed on the tensorize plan forced that plan early and refused shapes DPAS would
   have accepted. The chosen strategy is the only reliable witness of whether a splice exists."
  [epilogue d]
  (when (and (seq epilogue) (not (contains? splice-capable-strategies (:strategy d))))
    (throw (ex-info (str "contraction: strategy " (:strategy d) " has no store splice, so its "
                         ":epilogue would be silently dropped")
                    {:reason :epilogue-unsupported-by-this-leaf :strategy (:strategy d)})))
  d)

(defn route-contraction
  "Route a contraction form to the hardware-optimal kernel via the DPAS legality gate.
   dtype selects the element type of the intended kernel (:half tries DPAS; :byte/:int8 tries
   the int8 quant leaves — dp4a for the :nt operand layout, quant naive-widening for :nn;
   anything else, or a gate rejection, falls back to the register-tiled portable kernel).
   int8 needs no decode descriptor: a scale is an ordinary :epilogue and a zero-point an ordinary
   per-operand :decode, both carried on the form like any other contraction data."
  [contract-form & {:keys [dtype prefer-peak? desc tile epilogue stages operands]
                    :or {dtype :half prefer-peak? false}}]
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
        stages (or stages (:stages (apply hash-map (drop 5 contract-form))))
        ;; declared operand axis-maps, needed to tensorize a staged inner stage (the gate VERIFIES
        ;; them against the body rather than trusting them)
        operands (or operands (:operands (apply hash-map (drop 5 contract-form))))
        tensorize-plan (memoize #(route-2free-1contract contract-form out-sym dtype desc tile epilogue))]
    ;; Every descriptor is validated against the kernel it describes before it leaves this fn. The
    ;; failure mode it guards is a LAUNCH-time arity mismatch (valid C, wrong number of bound args),
    ;; which has bitten twice; validating at generation makes it a loud compile-time error instead.
    (validate-descriptor
     (epilogue-honoured-or-refused
      epilogue
      (cond
      ;; STAGED contract axis → the multi-level accumulate leaf. Checked FIRST because staging is
      ;; a property of the reduction itself, not of the dtype: it is what lets a block-quantized
      ;; format (int32 MAC inside the block, float accumulate across blocks) be expressed in the
      ;; contraction algebra at all. The flat leaves below cannot represent it — they have one
      ;; accumulator. 1 stage is the flat case and is left to them.
      (and (seq stages) (> (count stages) 1))
      (let [;; The staged emitter hardwires `+=` at every level, and a lift's linearity argument
            ;; assumes `+`. A form carrying :combine max routed here and was SILENTLY SUMMED —
            ;; contraction-facts surfaces :combine and nothing read it. Refuse rather than ignore.
            _ (let [cmb (:combine (cf/contraction-facts contract-form :dtype dtype))]
                (when-not (contains? '#{+ clojure.core/+ raster.numeric/+} cmb)
                  (throw (ex-info (str "staged contraction: only `+` combine is supported (got "
                                       cmb ") — every accumulator level uses += and a lift's "
                                       "linearity argument assumes addition")
                                  {:reason :non-plus-combine-on-staged :combine cmb}))))
            ;; PEAK: with declared+verified operand maps, the inner stage tensorizes to dp4a (4 int8
            ;; MACs per int32 op) — the int8 peak leaf, and the same shape llama.cpp hand-writes.
            ;; Only attempted when the caller asked for peak AND the gate passes; a gate rejection
            ;; falls back to the scalar nest, so requesting peak can never yield a wrong kernel.
            spec {:free-axes free-axes :stages stages
                  ;; the axes the FORM declared, read off FACTS — a single derivation whose only
                  ;; input is the form, so there is no separately-computed value to pass
                  ;; inconsistently (which is what made the span rule unfireable)
                  :contract-axes (:contract-axes (cf/contraction-facts contract-form :dtype dtype))
                  :body (nth contract-form 4)
                  :inputs (vec (sort-by name (contract-operand-arrays (nth contract-form 4))))
                  :operands operands
                  :dtype dtype :out-dtype (or (:out-dtype (apply hash-map (drop 5 contract-form)))
                                              :float)}
            tz? (and prefer-peak? (seq operands)
                     (:ok (sco/staged-inner-dp4a-legal? spec)))
            k (sco/generate-staged-contraction-kernel
               (assoc spec :tensorize-inner? (boolean tz?)) out-sym)]
        {:strategy :staged-segred
         :kernel-name (:kernel-name k) :source (:source k)
         :array-params (:array-params k)
         ;; the per-stage scale arrays are EXTRA pointer params — surfaced so a caller binds them
         :lift-operands (:lift-operands k)
         :dtype (:dtype k) :out-dtype (:out-dtype k)
         :out-elems (:out-elems k)
         :stages (:stages k)
         ;; the operand BUFFERS are bound unchanged; only the kernel's view of them widens to int32
         :tensorized (:tensorized k) :packed (:packed k)
         :scalar-args [{:type :int :value nseg}]
         :wg [256 1]
         :grid [(ceil-div nseg 256) 1]})

      ;; int8 → the quant leaves (dp4a for :nt, quant naive-widening for :nn)
      (#{:byte :int8} dtype)
      (route-quant contract-form out-sym n-free n-contract nseg prefer-peak?)

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
         ;; CARRY THE COMBINE. invoke-reduction-kernel reads :c-op/:identity-val from the kernel
         ;; REGISTRY for its host-side final combine, defaulting to `+`/0.0. Widening the
         ;; registration to pass the whole descriptor through (as a previous commit did) is a no-op
         ;; unless the descriptor actually contains them — generate-segred-kernel returns them into
         ;; a local that was discarded here. Without this, a multi-workgroup `:combine max` or `*`
         ;; silently SUMS its per-group partials.
         :c-op (:c-op k) :identity-val (:identity-val k)
         :reduce-bound red-bound          ; element count the reduction spans
         :scalar-args [] :dims [1]})

      (zero? n-contract)
      (let [sm (cl/contract-form->segmap contract-form :dtype dtype)
            {:keys [kernel-name source array-params scalar-params]}
            (sco/generate-segmap-nd-kernel sm out-sym :dtype dtype)]
        {:strategy :segmap
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :out-dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         :scalar-args (conj (mapv (fn [p] {:type :int :value p}) scalar-params)
                            {:type :int :value nseg})
         :out-elems nseg :dims [nseg]})

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
            {:keys [kernel-name source array-params scalar-params]}
            (sco/generate-segmented-reduce-kernel sr out-sym :dtype dtype)]
        {:strategy :naive-segred
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :out-dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         ;; SYMBOLIC axis bounds become int kernel params (the emitter declares them, sorted by
         ;; name); they must be bound BEFORE the trailing count or the launch arity is wrong.
         :scalar-args (conj (mapv (fn [p] {:type :int :value p}) scalar-params)
                            {:type :int :value nseg})
         :out-elems nseg :dims [nseg]}))))))

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
         ;; extra kernel args the epilogue needs, in signature order (after out + the dims)
         :epilogue-operands (:epilogue-operands dpas)
         :epilogue-params (:epilogue-params dpas)
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
               (seq? body) (od/multiplication-op? (od/semantic-op body))
               ;; TWO arguments, not three: `call-args` excludes the head (and the `.invk` receiver),
               ;; whereas the literal check this replaced counted the whole form. Getting this wrong
               ;; made `retarget-to-layout` bail unconditionally, so :nn int8 + :prefer-peak? stopped
               ;; producing its transpose pre-step and silently fell back to :quant-naive.
               (= 2 (count (od/call-args body))))
      (let [[i-sym M] (first free-axes) [j-sym N] (second free-axes) [l-sym K] (first contract-axes)
            ext {i-sym M j-sym N l-sym K}
            agets (mapv ce/normalize-array-prims (od/call-args body))
            agets (filterv od/aget-call? agets)]
        (when (= 2 (count agets))
          (let [syms-of (fn [g] (set (filter symbol? (tree-seq coll? seq (od/aget-index g)))))
                col (first (filter #(and (contains? (syms-of %) j-sym)
                                         (not (contains? (syms-of %) i-sym))) agets))
                row (first (filter #(and (contains? (syms-of %) i-sym)
                                         (not (contains? (syms-of %) j-sym))) agets))]
            (when (and col row (not= col row))
              (let [col-arr (od/aget-array-sym col)
                    actual-axes [l-sym j-sym]          ; the :nn storage we can retarget from
                    cmap (operand-map contract-form col-arr (od/aget-index col) actual-axes
                                      (mapv ext actual-axes))
                    want (am/of-axes (mapv (fn [a] [a (ext a)]) required-col-axes))]
                ;; only proceed when the ACTUAL layout is verified and the difference is a
                ;; genuine 2-D group transposition
                (when (and cmap (am/transposed-2d? cmap want))
                  (let [col-t (gensym (str (name col-arr) "__t"))
                        ;; QUALIFIED: this form RE-ENTERS route-contraction, so emitting bare ops
                        ;; here would hand the router the one spelling its matchers used to be the
                        ;; only ones that worked — and violates CLAUDE.md's emit-qualified rule.
                        new-body (list 'raster.numeric/*
                                       row (list 'raster.arrays/aget col-t (am/index-expr want)))
                        ;; PRESERVE THE FORM'S OPTS. Rebuilding with only :maps silently dropped
                        ;; :epilogue, :decode, :init, :combine and :out-dtype — so a retargeted
                        ;; contraction lost its scale (wrong by a constant factor), and a
                        ;; `:combine max` became a sum. Carry everything, and re-key the col
                        ;; operand's entries onto the transposed array.
                        opts (apply hash-map (drop 5 contract-form))
                        opts (cond-> (assoc opts :maps (assoc (get opts :maps {}) col-t want))
                               (get-in opts [:decode col-arr])
                               (update :decode #(-> % (dissoc col-arr)
                                                    (assoc col-t (get % col-arr)))))]
                    {:form (apply list 'raster.par/contract out free-axes contract-axes new-body
                                  (apply concat opts))
                     :pre-step {:op :transpose :src col-arr :dst col-t
                                :rows (ext l-sym) :cols (ext j-sym) :dtype dtype}}))))))))))

(defn- route-quant
  "Route an int8 (:byte) contraction — with no int8-specific emitter left.

   THE QUANT LEAVES ARE GONE. `generate-quant-contraction-kernel` and
   `generate-dp4a-contraction-kernel` were the staged emitter with ONE contract level: an int32
   accumulator, a zero-point on the load path, a scale at the store. Expressed that way they are the
   same kernel, so both are deleted and int8 routes through the one emitter.

   `:scheme {:scale :a-zp :b-zp}` is gone too. It was a closed three-field record with a private
   scale channel, and it existed only because an epilogue's `:scalars` were emitted into the kernel
   signature but never surfaced in the descriptor. Now the scale is an ordinary `:epilogue` and the
   zero-point an ordinary per-operand `:decode` — which buys per-ROW and per-COLUMN scales for free
   (an epilogue operand carries an axis-map) and lets scale compose with bias and activation by
   nesting one expression. Neither was expressible before.

   Peak-vs-portable is a LAYOUT question answered from the leaf-layouts table rather than a
   hand-written orientation check: dp4a packs 4 int8 along the contraction, so it needs both operands
   K-contiguous. If the form does not satisfy that, `prefer-peak?` may insert a byte-transpose to
   reach it; otherwise the scalar nest runs, which has no layout requirement."
  [contract-form out-sym n-free n-contract nseg prefer-peak?]
  (when-not (and (= 2 n-free) (= 1 n-contract))
    ;; :raster/fatal — falling through to the generic segred would accumulate int8 in int8 and
    ;; silently overflow, so there is no correct fallback here
    (throw (ex-info "route-quant: int8 supported for 2 free + 1 contract axes"
                    {:reason :raster/fatal :n-free n-free :n-contract n-contract})))
  (let [opts (apply hash-map (drop 5 contract-form))
        k-extent (second (first (nth contract-form 3)))
        spec (fn [form tz?]
               (let [f (cf/contraction-facts form :dtype :byte)
                     ;; when tensorizing, each operand's map comes from the layout check-layout
                     ;; PROVED — verified by construction, so no `:maps` declaration is needed to
                     ;; reach the peak leaf
                     lmaps (when tz?
                             (cf/layout-maps f (:dp4a cf/leaf-layouts)
                                             (cf/check-layout f (:dp4a cf/leaf-layouts))))]
                 {:free-axes (:free-axes f) :contract-axes (:contract-axes f)
                  :body (:body f)
                  ;; ONE contract level, int32 accumulate — the widening is the dtype PAIR
                  :stages [{:axis (ffirst (:contract-axes f))
                            :extent (second (first (:contract-axes f)))
                            :dtype :int :init 0}]
                  :operands (mapv (fn [o] (cond-> o (get lmaps (:sym o))
                                                  (assoc :map (get lmaps (:sym o)))))
                                  (:operands f))
                  :inputs (vec (sort-by name (contract-operand-arrays (:body f))))
                  :dtype :byte :out-dtype (get opts :out-dtype :float)
                  :epilogue (:epilogue f)
                  :tensorize-inner? tz?}))
        dp4a-ok? (fn [form]
                   ;; The router must consult the SAME gate the emitter will apply. It previously
                   ;; checked only k%4 and the operand layout, so a `:decode` (or a body the leaf
                   ;; would discard) passed here and then THREW inside
                   ;; generate-staged-contraction-kernel — turning a correct :quant-naive kernel
                   ;; into a hard compile error. Keep the emitter's throw: it is the right contract
                   ;; for an EXPLICIT :tensorize-inner? true. The router just has to stop asking
                   ;; when the answer is no. Same pattern the :staged-segred branch already uses.
                   (and (number? k-extent) (zero? (mod (long k-extent) 4))
                        (:ok (cf/check-layout (cf/contraction-facts form :dtype :byte)
                                              (:dp4a cf/leaf-layouts)))
                        (:ok (sco/staged-inner-dp4a-legal? (spec form true)))))
        emit (fn [form tz? pre-steps]
               (let [k (sco/generate-staged-contraction-kernel (spec form tz?) out-sym)]
                 (cond-> {:strategy (if tz? :dp4a :quant-naive)
                          :kernel-name (:kernel-name k) :source (:source k)
                          :array-params (:array-params k)
                          :lift-operands (:lift-operands k)
                          :epilogue-operands (:epilogue-operands k)
                          :epilogue-scalars (:epilogue-scalars k)
                          :dtype :byte :out-dtype (:out-dtype k)
                          :out-elems (:out-elems k)
                          :tensorized (:tensorized k) :packed (:packed k)
                          :scalar-args [{:type :int :value nseg}]
                          :wg [256 1] :grid [(ceil-div nseg 256) 1]}
                   (seq pre-steps) (assoc :pre-steps pre-steps))))]
    (cond
      (dp4a-ok? contract-form) (emit contract-form true nil)
      prefer-peak?
      (if-let [{form* :form pre-step :pre-step} (retarget-to-layout contract-form '[j l] :byte)]
        (if (dp4a-ok? form*) (emit form* true [pre-step]) (emit contract-form false nil))
        (emit contract-form false nil))
      :else (emit contract-form false nil))))

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

   Returns a launch-ready descriptor. Strategies: :segmap (0 contract axes), :portable-segred
   (scheduled general), :naive-segred (explicit migration fallback), :regtiled (portable tiled),
   :dpas (f16 peak), :dp4a (int8 peak), :quant-naive
   (int8 widening). Keys: :kernel-name :source :array-params (binding order) :dtype :out-dtype
   :out-elems :wg/:grid (uniform 1-3D geometry) :scalar-args [{:type :value}…] :dims, plus optional
   :fallback-reason, :scheme (quant decode) and :pre-steps (inserted layout rearranges)."
  (:require [raster.compiler.core.op-descriptor :as od]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.typed-soac-projection :as typed-projection]))

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

(defn- launch-dimension
  "Translate the router's legacy arithmetic dimension into canonical launch IR. The router emits
  symbolic ceil-div as `(quot (+ value (dec divisor)) divisor)`; preserve that operation as a
  CeilDiv so KernelCall resolves the underlying compiler value directly from the ABI arguments."
  [dimension]
  (cond
    (integer? dimension)
    (if (pos? dimension)
      dimension
      (throw (ex-info "contract launch dimension must be positive"
                      {:dimension dimension})))

    :else
    (let [[q numerator divisor] (when (seq? dimension) dimension)
          [plus value offset] (when (seq? numerator) numerator)]
      (if (and (contains? '#{quot clojure.core/quot} q)
               (contains? '#{+ clojure.core/+} plus)
               (integer? divisor) (pos? divisor)
               (= (dec divisor) offset))
        (klaunch/ceil-div value divisor)
        (klaunch/runtime-value dimension)))))

(defn- descriptor-artifact
  "Close a validated single-launch contraction descriptor into one executable compiler value."
  [{:keys [strategy kernel-name source abi arguments wg grid out-elems dtype out-dtype]
    :as descriptor}]
  (when (and (not (number? out-elems)) (not (some #(= out-elems %) arguments)))
    (throw (ex-info "contract descriptor: symbolic :out-elems is absent from artifact arguments"
                    {:strategy strategy :out-elems out-elems :arguments arguments})))
  (kart/make
   {:kernel-name kernel-name
    :source source
    :abi abi
    :arguments arguments
    :launch (if (= :portable-segred strategy)
              (get-in descriptor [:kernel-body :launch])
              (klaunch/spec
               {:workgroup-size (mapv launch-dimension wg)
                :group-count (mapv launch-dimension grid)}))
    :temporaries []
    :effects {:kind :tensor-contraction}
    :provenance {:dialect :segcontract :strategy strategy}
    :attributes
    (merge {:strategy strategy
            :out-elems out-elems
            :dtype dtype
            :out-dtype out-dtype}
           (select-keys descriptor [:fallback-reason :declines :tile :tensorized :packed
                                    :fused-epilogue :dims :stages :kernel-body]))}))

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

   Pointer params must equal the distinct bound inputs plus one output; a result transform may
   reuse a contraction input by compiler identity without adding a second ABI slot. Default single-launch
   leaves must also supply matching 1-3D workgroup/grid data; validation closes those fields and the
   ordered compiler values into a KernelArtifact. Full reductions already arrive as a verified
   artifact and retain their distinct two-phase invoke protocol, whose scheduler owns geometry.

   Returns the descriptor with `:arguments`, the compiler values in exact ABI order, and an
   executable `:artifact` for every route."
  [{:keys [strategy kernel-name source array-params scalar-args epilogue-operands lift-operands abi
           epilogue-scalars out-elems wg grid invoke reduce-bound] :as d}]
  (let [params (kernel-signature-params source)
        _ (when (nil? params)
            (throw (ex-info "contract descriptor: no __kernel signature found in source"
                            {:strategy strategy :kernel-name kernel-name})))
        ptr? (fn [p] (str/includes? p "*"))
        param-shape (mapv (fn [p]
                            {:c-name (second (re-find #"([A-Za-z_][A-Za-z0-9_]*)\s*$" p))
                             :pointer? (ptr? p)})
                          params)
        n-ptr (count (filter ptr? params))
        n-scalar (count (remove ptr? params))
        ;; EXTRA operand arrays are pointer params too, whichever seam declared them: an
        ;; epilogue's (bias/residual/scale, appended after the dims) or a staged contraction's
        ;; lift operands (the per-block scales, bound between the operands and the output).
        distinct-pointer-inputs
        (distinct (concat array-params epilogue-operands lift-operands))
        expect-ptr (inc (count distinct-pointer-inputs))
        ;; Full reduction carries a complete ordered ABI. Other leaves still construct their
        ;; compiler argument values from descriptor scalars here, exactly once, before artifact
        ;; validation eliminates the positional convention from all runtime paths.
        ;; an epilogue's SCALARS are kernel scalar params too — they are emitted into the
        ;; signature by epilogue-splice, so a descriptor that omits them under-counts and the
        ;; capability becomes unusable (which is what pushed `:scheme` into a private channel)
        abi-epilogue-scalars
        (count (filter #(and (= :scalar (:kind %)) (= :epilogue (:role %))) abi))
        expect-scalar (if (= :reduction invoke)
                        (count (when abi (kabi/scalar-slots abi)))
                        (+ (count scalar-args) abi-epilogue-scalars))]
    (cond
      (nil? abi)
      (throw (ex-info "contract descriptor: the ordered :abi is required"
                      {:strategy strategy :kernel-name kernel-name}))
      (not= param-shape (kabi/signature-shape abi))
      (throw (ex-info "contract descriptor: ordered ABI does not match the emitted kernel signature"
                      {:strategy strategy :kernel-name kernel-name
                       :signature param-shape :abi (kabi/signature-shape abi)}))
      (not= n-ptr expect-ptr)
      (throw (ex-info (str "contract descriptor: kernel takes " n-ptr " pointer params but the "
                           "descriptor binds " expect-ptr " (" (count array-params) " operands + out"
                           (when (seq epilogue-operands)
                             (str " + distinct transform inputs from "
                                  (count epilogue-operands) " references"))
                           (when (seq lift-operands)
                             (str " + distinct lift inputs from "
                                  (count lift-operands) " references")) ")")
                      {:strategy strategy :kernel-name kernel-name :params params
                       :array-params array-params :epilogue-operands epilogue-operands
                       :lift-operands lift-operands}))
      (not= n-scalar expect-scalar)
      (throw (ex-info (str "contract descriptor: kernel takes " n-scalar " scalar params but the "
                           (if (= :reduction invoke)
                             (str "ordered reduction ABI supplies " expect-scalar)
                             (str "descriptor supplies " expect-scalar " scalar-args")))
                      {:strategy strategy :kernel-name kernel-name :params params
                       :invoke invoke :scalar-args scalar-args}))
      (nil? out-elems)
      (throw (ex-info "contract descriptor: :out-elems is required (the artifact sizes the output with it)"
                      {:strategy strategy}))
      ;; A single-launch descriptor preserves the schedule's actual dimensionality. The reduction
      ;; invoke computes its own two-phase geometry, so it legitimately carries neither.
      (and (not= :reduction invoke)
           (not (and (vector? wg) (vector? grid)
                     (<= 1 (count wg) 3)
                     (= (count wg) (count grid)))))
      (throw (ex-info "contract descriptor: :wg and :grid must have matching 1-3D geometry"
                      {:strategy strategy :wg wg :grid grid}))
      (and (= :reduction invoke) (or (some? wg) (some? grid)))
      (throw (ex-info "contract descriptor: :invoke :reduction must not carry :wg/:grid (the invoke owns the two-phase launch)"
                      {:strategy strategy :wg wg :grid grid}))
      (and (= :reduction invoke) (nil? reduce-bound))
      (throw (ex-info "contract descriptor: :invoke :reduction requires :reduce-bound"
                      {:strategy strategy}))
      :else
      (if (= :reduction invoke)
        (do (kabi/validate-reduction-arguments! abi (:arguments d)) d)
        (let [remaining-scalars (volatile! (seq scalar-args))
              arguments
              (mapv (fn [{:keys [name kind role] :as slot}]
                      (case kind
                        (:input :output) name
                        :scalar
                        (if (= :epilogue role)
                          name
                          (if-let [arg (first @remaining-scalars)]
                            (do (vswap! remaining-scalars next) (:value arg))
                            (throw (ex-info "contract descriptor: no value for ABI scalar"
                                            {:strategy strategy :slot slot
                                             :scalar-args scalar-args}))))))
                    abi)
              descriptor (assoc d :arguments arguments)]
          (assoc descriptor :artifact (descriptor-artifact descriptor)))))))

(declare route-2free-1contract route-quant)

(defn- compatibility-form!
  "Require an explicit source/leaf spelling at a compatibility-only boundary.

   Ordinary typed contraction families consume verified facts plus their scheduled SegRed and
   must never pass through this gate. A leaf that still operates on surface syntax may project a
   form explicitly before calling here; the common router does not manufacture one implicitly."
  [contract-form leaf]
  (or contract-form
      (throw (ex-info "contraction compatibility leaf requires a surface form"
                      {:reason :contraction-compatibility-form-required
                       :leaf leaf
                       :fallback :none}))))

(def ^:private splice-capable-strategies
  "Strategies whose scheduled body or emitter has a typed store region and can honour an :epilogue.
   A WHITELIST on purpose: refuse by ABSENCE of support, never by a blacklist of shapes — a
   blacklist got this wrong twice, most recently by exempting the only shape production produces.
   `:dpas`, `:regtiled` and `:portable-segred` lower through typed KernelBody storage; the three staged-emitter
   strategies still splice at the store (which is how a quant dequant scale is expressed since
   :scheme was deleted)."
  #{:dpas :dp4a :regtiled :portable-segred :quant-naive :staged-segred})

(def ^:private contraction-families
  "Target schedule families for ordinary scalar typed contractions."
  #{:matrix :register-tiled :portable})

(def ^:private strategy-family
  {:dpas :matrix
   :dp4a :matrix
   :regtiled :register-tiled
   :portable-segred :portable
   :naive-segred :portable
   :full-reduce :portable
   :segmap :portable
   :quant-naive :portable
   :staged-segred :portable})

(defn- epilogue-honoured-or-refused
  "An :epilogue is a typed result-store region. A leaf without one would drop the consumer's
   computation, so refusal is mandatory.

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
  "Route a contraction form through verified facts, an applied hardware schedule and a target leaf.

   `contract-form` may be nil when verified `:facts` and a scheduled operation are supplied.
   Ordinary typed matrix, register-tiled, portable and full-reduction routes need no source
   spelling. Staged, SegMap and source fallbacks require one explicitly; the int8 compatibility
   leaf performs its temporary projection locally until it consumes the typed scalar vocabulary.

   Canonical f16 products first become a target-neutral KernelBody and are then lowered by the
   OpenCL DPAS backend.  Unsupported shapes retain the portable register-tiled route.  Byte/int8
   products use the quant leaves (dp4a for :nt, widening for :nn) until those instruction families
   consume the same scheduled body vocabulary.  Quantization remains an operand/decode concern,
   not a buffer-ownership or graph-composition concern."
  [contract-form & {:keys [dtype prefer-peak? desc tile epilogue stages operands facts operation-id
                           candidate-families scheduled-operation]
                    :or {dtype :half prefer-peak? false}}]
  (let [contract-facts (or facts (cf/contraction-facts contract-form :dtype dtype))
        contract-form (or contract-form (:form contract-facts))
        _ (when-not (cf/facts? contract-facts)
            (throw (ex-info "contraction routing requires verified facts"
                            {:reason :contraction-route-input
                             :facts contract-facts :form contract-form})))
        out-sym (:out contract-facts)
        free-axes (:free-axes contract-facts)
        contract-axes (:contract-axes contract-facts)
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
        ;; A TypedSOAC result transform projects to an epilogue in the form's trailing opts;
        ;; an explicit :epilogue kwarg overrides it at this temporary leaf boundary.
        form-opts (:opts contract-facts)
        epilogue (or epilogue (:epilogue contract-facts))
        stages (or stages (:stages contract-facts))
        ;; declared operand axis-maps, needed to tensorize a staged inner stage (the gate VERIFIES
        ;; them against the body rather than trusting them)
        operands (or operands (:operands form-opts))
        candidate-families (set (or candidate-families contraction-families))
        _ (when-not (and (seq candidate-families)
                         (every? contraction-families candidate-families))
            (throw (ex-info "contraction route requires known non-empty candidate families"
                            {:reason :contraction-candidate-families
                             :families candidate-families
                             :allowed contraction-families})))
        tensorize-plan (memoize #(route-2free-1contract out-sym dtype desc tile
                                                        epilogue contract-facts operation-id
                                                        candidate-families))]
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
        (let [contract-form (compatibility-form! contract-form :staged-segred)
              ;; The staged emitter hardwires `+=` at every level, and a lift's linearity argument
            ;; assumes `+`. A form carrying :combine max routed here and was SILENTLY SUMMED —
            ;; contraction-facts surfaces :combine and nothing read it. Refuse rather than ignore.
              _ (let [cmb (:combine (cf/scalar-reduction-view contract-facts))]
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
                    :contract-axes (:contract-axes contract-facts)
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
           :abi (:abi k)
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
        (route-quant
         ;; Quant leaves still rewrite surface layout for the optional transpose pre-step. Keep
         ;; that temporary projection local to the leaf instead of making syntax a common router
         ;; invariant; ordinary typed contraction families never observe it.
         (compatibility-form! (or contract-form (cf/surface-form contract-facts)) :quant)
         out-sym n-free n-contract nseg prefer-peak?)

      ;; 0 FREE axes → a full reduction to a scalar. This is the last cell of contract's
      ;; algebra: (n free, 0 contract) = map, (n, n) = contraction, (0, n) = REDUCTION. The
      ;; SegSpace then has only the reduced dim — exactly the 1-D shape (seg-space-1d?) that
      ;; generate-segred-kernel's two-phase tree reduction already consumes, so no new emitter.
      ;; Its launch protocol differs (two phases + a host-side final combine), so the descriptor
      ;; says so with :invoke :reduction rather than pretending it is a 2-D kernel launch.
        (zero? n-free)
        (let [sr (or scheduled-operation
                     (cl/contract-form->segred
                      (compatibility-form! contract-form :full-reduce)
                      :dtype dtype :facts contract-facts))
              k (sco/generate-segred-kernel sr out-sym :dtype dtype)
              attrs (:attributes k)
              red-bound (second (first contract-axes))]
          {:strategy :full-reduce
           :invoke :reduction
           :artifact k
           :kernel-name (:kernel-name k) :source (:source k)
           :array-params (:array-params attrs)
           :abi (:abi k) :arguments (:arguments k)
           :dtype dtype :out-dtype dtype :out-elems 1
           :n-phases (:n-phases attrs)
         ;; CARRY THE COMBINE for descriptor inspection as well as in the artifact attributes.
         ;; The staging runtime reads the artifact attributes for its host-side final combine.
           :c-op (:c-op attrs) :identity-val (:identity-val attrs)
           :reduce-bound red-bound          ; element count the reduction spans
           :scalar-args [] :dims [1]})

        (zero? n-contract)
        (let [sm (cl/contract-form->segmap
                  (compatibility-form! contract-form :segmap) :dtype dtype)
              {:keys [kernel-name source array-params scalar-params abi]}
              (sco/generate-segmap-nd-kernel sm out-sym :dtype dtype)]
          {:strategy :segmap
           :kernel-name kernel-name :source source :array-params array-params :abi abi
           :dtype dtype :out-dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
           :scalar-args (conj (mapv (fn [p] {:type :int :value p}) scalar-params)
                              {:type :int :value nseg})
           :out-elems nseg :dims [nseg]})

      ;; 2 free + 1 contract → the tensorize fast path (DPAS if legal, else regtiled).
      ;; A `{::declines …}` result means BOTH tensorize leaves refused for a documented reason —
      ;; we fall through to the general naive leaf rather than hard-failing a perfectly legal
      ;; contraction, and carry the reasons onto that leaf's descriptor.
        (and (= 2 n-free) (= 1 n-contract) (:strategy (tensorize-plan)))
        (tensorize-plan)

      ;; everything else (n-free≠2, n≥2 contract axes, or a tensorize-ineligible 2-free form)
      ;; → the portable scheduled KernelBody when its operand layouts are provable. During the
      ;; migration, unsupported gathers retain the verified source emitter as an explicit decline.
        :else
        (do
          (when-not (contains? candidate-families :portable)
            (throw (ex-info "no enabled contraction schedule family can lower this equation"
                            {:reason :no-legal-contraction-family
                             :operation operation-id
                             :families candidate-families
                             :declines (when (and (= 2 n-free) (= 1 n-contract))
                                         (::declines (tensorize-plan)))})))
          (let [sr (or scheduled-operation
                       (cl/contract-form->segred
                        (compatibility-form! contract-form :portable-segred)
                        :dtype dtype :facts contract-facts))
                portable (contraction-schedule/plan-portable-body contract-facts sr desc)
                emitted (when (:ok portable)
                          (sco/generate-contraction-kernel-body (:body portable)))
                {:keys [kernel-name source array-params scalar-params abi
                        epilogue-operands epilogue-scalars]}
                (or emitted (sco/generate-segmented-reduce-kernel sr out-sym :dtype dtype))
            ;; WHY THIS LEAF AND NOT A FASTER ONE. Only meaningful for the 2-free/1-contract shape,
            ;; where a tensorize leaf was actually attempted; for every other shape no faster leaf
            ;; exists to decline, so an empty vector is the honest answer rather than a fabricated
            ;; "not eligible".
                declines (when (and (= 2 n-free) (= 1 n-contract))
                           (::declines (tensorize-plan)))
                declines (cond-> (vec declines)
                           (not (:ok portable))
                           (conj {:leaf :portable-kernel-body
                                  :reason (:reason portable)
                                  :data (:detail portable)}))
                workgroup-size (or (:workgroup-size portable) 256)]
            (cond->
             {:strategy (if (:ok portable) :portable-segred :naive-segred)
              :declines declines
              :kernel-name kernel-name :source source :array-params array-params :abi abi
              :epilogue-operands epilogue-operands
              :epilogue-scalars epilogue-scalars
              :fused-epilogue (boolean epilogue)
              :kernel-body (:body portable)
              :dtype dtype :out-dtype dtype
         ;; SYMBOLIC axis bounds become int kernel params (the emitter declares them, sorted by
         ;; name); they must be bound BEFORE the trailing count or the launch arity is wrong.
              :scalar-args (conj (mapv (fn [p] {:type :int :value p}) scalar-params)
                                 {:type :int :value nseg})
              :out-elems nseg :dims [nseg]
              :wg [workgroup-size]
              :grid [(ceil-div nseg workgroup-size)]}
          ;; the LAST decline is the decisive one — the leaf that would otherwise have taken the
          ;; work. Using the first reported DPAS's generic :not-a-contraction where regtiled's
          ;; specific :symbolic-dims / :non-plus-combine is the actual answer.
              (seq declines) (assoc :fallback-reason (:reason (last declines)))))))))))

(defn- validated-typed-contraction-families
  [schedule operation-id]
  (let [families (get-in schedule [:tuning-space :families])]
    (when-not (and (vector? families) (seq families)
                   (= (count families) (count (distinct families)))
                   (every? contraction-families families))
      (throw (ex-info "typed contraction schedule requires known distinct candidate families"
                      {:reason :typed-contraction-families
                       :operation operation-id
                       :families families
                       :allowed contraction-families})))
    families))

(defn- scheduled-typed-contraction-context
  "Validate that one scheduled SegRed is the physical schedule for its TypedSOAC equation.

   This is the typed semantic/schedule seam. The immutable equation id joins the two dialects;
   dtype, iteration space, physical operands and result storage are checked here so target routing
   cannot accept a SegRed borrowed from another equation."
  [program operation]
  (let [program (soac-dialect/validate! program)
        _ (when-not (instance? raster.compiler.ir.segop.SegRed operation)
            (throw (ex-info "typed contraction route requires a scheduled SegRed"
                            {:reason :typed-contraction-operation :operation operation})))
        operation-id (:id operation)
        equation (or (some #(when (= operation-id (second %)) %)
                           (soac-dialect/equations program))
                     (throw (ex-info "typed contraction program lacks its scheduled equation"
                                     {:reason :typed-contraction-equation
                                      :operation operation-id})))
        components (typed-projection/segmented-reduce-contract-components program equation)
        facts (cf/from-components components)
        equation-dtype (:dtype components)
        expected-space (conj (:free-axes facts) (:flat-contract-axis facts))
        actual-space (mapv (juxt :name :bound) (get-in operation [:space :dims]))
        transform-inputs (set (map :sym (get-in facts [:epilogue :operands])))
        expected-inputs (into transform-inputs (map :sym) (:operands facts))
        expected-outputs #{(:out facts)}
        schedule (reduction/validate-schedule! (:schedule operation))]
    (when-not (= :contraction (:phase operation))
      (throw (ex-info "typed contraction SegRed has the wrong phase"
                      {:reason :typed-contraction-phase
                       :operation operation-id :phase (:phase operation)})))
    (when-not (= equation-dtype (:dtype operation))
      (throw (ex-info "typed contraction SegRed dtype disagrees with its equation"
                      {:reason :typed-contraction-operation-dtype
                       :operation operation-id :equation-dtype equation-dtype
                       :operation-dtype (:dtype operation)})))
    (when-not (= expected-space actual-space)
      (throw (ex-info "typed contraction SegRed iteration space disagrees with its equation"
                      {:reason :typed-contraction-space
                       :operation operation-id :expected expected-space :actual actual-space})))
    (when-not (and (= expected-inputs (segop/operation-inputs operation))
                   (= expected-outputs (segop/operation-outputs operation)))
      (throw (ex-info "typed contraction SegRed storage boundary disagrees with its equation"
                      {:reason :typed-contraction-storage
                       :operation operation-id
                       :expected-inputs expected-inputs
                       :actual-inputs (segop/operation-inputs operation)
                       :expected-outputs expected-outputs
                       :actual-outputs (segop/operation-outputs operation)})))
    {:program program
     :operation operation
     :operation-id operation-id
     :equation equation
     :components components
     :facts facts
     :dtype equation-dtype
     :schedule schedule}))

(defn route-typed-contraction
  "Route one scheduled scalar TypedSOAC contraction from its algorithm and exact SegRed.

   The equation is the semantic input and the supplied SegRed is the already-applied schedule.
   Target routing selects and verifies the concrete matrix, register-tiled or portable leaf without
   re-lowering a generated host form into a second SegRed."
  [program operation & options]
  (let [{:keys [program operation-id facts dtype schedule]}
        (scheduled-typed-contraction-context program operation)
        _ (when-not (= :hardware-contraction-candidates (:strategy schedule))
            (throw (ex-info "typed contraction requires a contraction candidate schedule"
                            {:reason :typed-contraction-schedule
                             :operation operation-id :schedule schedule})))
        families (validated-typed-contraction-families schedule operation-id)
        options (apply hash-map options)
        _ (when (and (contains? options :dtype)
                     (not= dtype (:dtype options)))
            (throw (ex-info "typed contraction route dtype disagrees with its equation"
                            {:reason :typed-contraction-dtype
                             :operation operation-id
                             :equation-dtype dtype
                             :route-dtype (:dtype options)})))
        routed (try
                 (apply route-contraction nil
                        (mapcat identity
                                (assoc options
                                       :dtype dtype
                                       :facts facts
                                       :scheduled-operation operation
                                       :candidate-families families
                                       :operation-id operation-id)))
                 (catch clojure.lang.ExceptionInfo exception
                   (let [{:keys [reason strategy] :as data} (ex-data exception)]
                     (if (= :epilogue-unsupported-by-this-leaf reason)
                       (throw (ex-info "no enabled contraction family can lower the result transform"
                                       {:reason :no-legal-contraction-family
                                        :operation operation-id
                                        :families families
                                        :declines [{:leaf strategy :reason reason
                                                    :data (dissoc data :reason :strategy)}]}
                                       exception))
                       (throw exception)))))
        selected-family (get strategy-family (:strategy routed))]
    (when-not (contains? (set families) selected-family)
      (throw (ex-info "selected contraction leaf is outside the enabled schedule families"
                      {:reason :no-legal-contraction-family
                       :operation operation-id
                       :families families
                       :declines [{:leaf (:strategy routed)
                                   :reason :strategy-outside-schedule-families
                                   :data {:selected-family selected-family
                                          :enabled-families families}}]})))
    routed))

(defn- enabled-family-decline?
  [decline]
  (not= :schedule-family-disabled (:reason decline)))

(defn- candidate-declines
  [family declines]
  (into []
        (comp (filter enabled-family-decline?)
              (map #(assoc % :candidate-family family)))
        declines))

(defn route-typed-contraction-candidates
  "Emit every representable family enabled by one typed contraction schedule.

   Compilation remains pure: this function neither benchmarks nor chooses. Each candidate is a
   complete validated artifact with its concrete leaf strategy; real legality failures are retained
   with their candidate family, while leaves disabled deliberately by a single-family probe are not
   reported as failures. Different physical ABIs are permitted here because offline compile-time
   selection measures each candidate independently; a runtime KernelDispatch requires a later
   common-interface normalization."
  [program operation & options]
  (let [operation-id (:id operation)
        schedule (reduction/validate-schedule! (:schedule operation))
        families (validated-typed-contraction-families schedule operation-id)
        results
        (mapv
         (fn [family]
           (let [pinned (assoc-in schedule [:tuning-space :families] [family])]
             (try
               (-> (apply route-typed-contraction program (assoc operation :schedule pinned) options)
                   (assoc :family family)
                   (assoc :candidate-schedule pinned)
                   (update :declines #(candidate-declines family %)))
               (catch clojure.lang.ExceptionInfo exception
                 (let [{:keys [reason declines] :as data} (ex-data exception)]
                   (if (= :no-legal-contraction-family reason)
                     {:family family
                      :strategy nil
                      :candidate-schedule pinned
                      :declines (candidate-declines family declines)
                      :reason reason
                      :detail (dissoc data :declines)}
                     (throw exception)))))))
         families)
        candidates (filterv :strategy results)]
    {:operation-id operation-id
     :schedule schedule
     :candidates candidates
     :declines (into [] (mapcat :declines) results)}))

(defn route-typed-contraction-candidates!
  "Emit typed contraction candidates or fail when no enabled family has a legal target leaf."
  [program operation & options]
  (let [result (apply route-typed-contraction-candidates
                      program operation options)]
    (if (seq (:candidates result))
      result
      (throw (ex-info "no executable typed contraction candidates"
                      {:reason :typed-contraction-no-candidates
                       :route result})))))

(def ^:private logical-interface-fields
  [:kind :dtype :kernel-dtype :role :binding :field])

(defn- public-interface-slot?
  [slot]
  (or (not= :scalar (:kind slot))
      (= :epilogue (:role slot))))

(defn- artifact-logical-interface
  [artifact]
  (->> (map vector (:abi artifact) (:arguments artifact))
       (filter (comp public-interface-slot? first))
       vec))

(defn- common-logical-interface
  [candidates operation-id]
  (let [interfaces (mapv (comp artifact-logical-interface :artifact) candidates)
        arguments (mapv second (first interfaces))]
    (doseq [[candidate interface] (map vector candidates interfaces)]
      (when-not (= arguments (mapv second interface))
        (throw (ex-info "typed contraction candidates have different logical ABI arguments"
                        {:reason :typed-contraction-candidate-interface-arguments
                         :operation operation-id
                         :family (:family candidate)
                         :expected arguments
                         :actual (mapv second interface)}))))
    (let [abi
          (mapv
           (fn [index argument]
             (let [slots (mapv #(first (nth % index)) interfaces)
                   semantic-views (mapv #(select-keys % logical-interface-fields) slots)
                   _ (when-not (apply = semantic-views)
                       (throw (ex-info
                               "typed contraction candidates have incompatible logical ABI semantics"
                               {:reason :typed-contraction-candidate-interface-semantics
                                :operation operation-id :argument argument
                                :slots slots})))
                   aliasing (when (some #(= :no-write-alias (:aliasing %)) slots)
                              :no-write-alias)
                   alignments (keep :alignment slots)
                   alignment (when (seq alignments) (apply max alignments))]
               (cond-> (assoc (first slots) :name argument)
                 aliasing (assoc :aliasing aliasing)
                 (nil? aliasing) (dissoc :aliasing)
                 alignment (assoc :alignment alignment)
                 (nil? alignment) (dissoc :alignment))))
           (range (count arguments)) arguments)]
      (kabi/validate! abi)
      {:abi abi :arguments arguments})))

(defn- static-private-scalars!
  [candidate operation-id]
  (when (:invoke candidate)
    (throw (ex-info "typed contraction candidate uses a non-graph leaf invocation protocol"
                    {:reason :typed-contraction-dispatch-invoke-protocol
                     :operation operation-id
                     :family (:family candidate)
                     :strategy (:strategy candidate)
                     :invoke (:invoke candidate)})))
  (doseq [[slot compiler-value] (map vector (get-in candidate [:artifact :abi])
                                     (get-in candidate [:artifact :arguments]))
          :when (and (= :scalar (:kind slot))
                     (not (public-interface-slot? slot)))]
    (when-not (and (contains? #{:int :long} (:kernel-dtype slot))
                   (or (integer? compiler-value)
                       (and (klaunch/expression? compiler-value)
                            (empty? (klaunch/expression-references compiler-value)))))
      (throw (ex-info
              "static typed contraction dispatch has a runtime-dependent private scalar"
              {:reason :typed-contraction-dispatch-dynamic-scalar
               :operation operation-id
               :family (:family candidate)
               :slot slot
               :compiler-value compiler-value}))))
  candidate)

(defn- graph-buffer-role
  [kind]
  (case kind
    :input :input
    :output :output
    :inout :inout))

(defn- candidate-graph
  [candidate operation-id common-abi common-arguments]
  (let [{:keys [family strategy artifact candidate-schedule]} candidate
        artifact (kart/validate! artifact)
        interface (mapv vector common-abi common-arguments)
        pointer-interface (filterv #(not= :scalar (:kind (first %))) interface)
        buffers (mapv (fn [[slot argument]]
                        [argument
                         (kgraph/buffer argument (:dtype slot)
                                        (when (contains? #{:output :inout} (:kind slot))
                                          (get-in artifact [:attributes :out-elems]))
                                        :device (graph-buffer-role (:kind slot)))])
                      pointer-interface)
        buffer-map (into {} buffers)
        inputs (into [] (comp (filter #(contains? #{:input :inout} (:kind (first %))))
                              (map #(get buffer-map (second %))))
                     pointer-interface)
        outputs (into [] (comp (filter #(contains? #{:output :inout} (:kind (first %))))
                               (map #(get buffer-map (second %))))
                      pointer-interface)
        uses (mapv (fn [slot argument]
                     (kgraph/->ValueUse argument (kabi/slot-access slot)))
                   (mapv first pointer-interface) (mapv second pointer-interface))]
    (kgraph/make
     {:inputs inputs
      :outputs outputs
      :abi common-abi
      :arguments common-arguments
      :nodes [(kgraph/->ScheduledKernel
               [:typed-contraction operation-id family strategy] artifact uses [])]
      :effects (:effects artifact)
      :provenance {:operation-id operation-id
                   :semantic-op :contraction
                   :lowering :typed-contraction-candidate}
      :attributes {:strategy strategy
                   :candidate-family family
                   :candidate-schedule candidate-schedule}})))

(defn- candidate-dispatch-id
  [operation-id candidates]
  (let [identity
        [operation-id
         (mapv (fn [{:keys [family strategy artifact]}]
                 (let [kernel-name (:kernel-name artifact)]
                   {:family family
                    :strategy strategy
                    :artifact (-> (select-keys artifact
                                               [:target :source :abi :arguments :launch :effects])
                                  (update :source str/replace kernel-name "<entry-point>"))}))
               candidates)]]
    (format "raster_typed_contraction_dispatch_%08x"
            (bit-and 0xffffffff (long (hash identity))))))

(defn- deterministic-candidate-entry-point
  [candidate dispatch-id]
  (let [artifact (kart/validate! (:artifact candidate))
        old-name (:kernel-name artifact)
        strategy-name (str/replace (name (:strategy candidate)) #"[^A-Za-z0-9_]" "_")
        new-name (str dispatch-id "_" strategy-name)]
    (assoc candidate :artifact
           (kart/validate!
            (-> artifact
                (assoc :kernel-name new-name)
                (update :source str/replace old-name new-name))))))

(defn- typed-contraction-tuning-contract
  [schedule dispatch-id abi]
  (let [interface (mapv #(select-keys % [:kind :dtype :kernel-dtype :role
                                         :aliasing :alignment])
                        abi)]
    {:schedule-path [:typed-contraction :measured-selectors]
     :schedule-key dispatch-id
     :numerical-mode {:reduction (:numerical-mode schedule)
                      :interface interface}
     :layout {:external-interface interface}}))

(defn route-static-typed-contraction-dispatch
  "Normalize legal static typed contraction leaves behind one logical ABI-compatible dispatch.

   Leaf-only dimensions remain graph-private derived scalars, while typed result-transform scalar
   captures join operand/result pointers in the shared public ABI. Matrix, register-tiled and
   portable kernels can therefore compete through the existing KernelDispatch tuning machinery.
   Runtime-dependent private dimensions are refused until they have a common public ABI rather
   than silently baking one sample."
  [program operation & options]
  (let [operation-id (:id operation)
        schedule (reduction/validate-schedule! (:schedule operation))
        {:keys [candidates] :as routed}
        (apply route-typed-contraction-candidates!
               program operation options)
        candidates (mapv #(static-private-scalars! % operation-id) candidates)
        {:keys [abi arguments]} (common-logical-interface candidates operation-id)
        default-strategy (:strategy (first candidates))
        dispatch-id (candidate-dispatch-id operation-id candidates)
        candidates (mapv #(deterministic-candidate-entry-point % dispatch-id) candidates)
        alternatives (mapv #(candidate-graph % operation-id abi arguments) candidates)]
    (kdispatch/make
     {:id dispatch-id
      :alternatives alternatives
      :default-strategy default-strategy
      :selector {:kind :fixed-strategy :strategy default-strategy}
      :provenance {:operation-id operation-id
                   :semantic-op :contraction
                   :source-dialect :typed-soac}
      :attributes {:operation-family :typed-contraction
                   :candidate-schedules
                   (into {} (map (juxt :strategy :candidate-schedule)) candidates)
                   :declines (:declines routed)
                   :tuning (typed-contraction-tuning-contract schedule dispatch-id abi)
                   :selection :analytic-fixed}})))

(def ^:private decline-reasons
  "The reasons a tensorize leaf may legitimately REFUSE a shape — a WHITELIST.

   A denylist was written here first and was wrong for the usual reason: it assumed every unknown
   reason is safe to treat as a decline. `emit-gemm-tiled` throws VALIDATION errors carrying no
   `:reason` at all (`opencl_codegen.clj:649-660`: `{:tile …}`, `{:beta beta}`, `{}`), so a genuine
   emitter bug — a non-divisible tile, split-k with beta≠0 — would have been filed as \"this shape
   is not tiled-leaf shaped\" and silently demoted to `:naive-segred`. That is exactly the
   loud-to-silent trade this whole change exists to prevent.

   So: only these reasons are declines. Anything else — a different reason, or none — propagates.
   Adding a gate reason means adding it here; forgetting to is a LOUD failure, which is the safe
   direction. These are the reasons the legality gates themselves produce; grep `:reason :` in
   `segop_opencl.clj` to see the full set."
  #{;; shape / structure — the contraction is fine, it just is not tiled-leaf shaped
    :symbolic-dims :symbolic-bounds-unsupported :non-plus-combine :non-product-element
    :non-aget-operand :not-a-contraction :not-2-free :body-has-unmodeled-terms
    ;; dtype / orientation / alignment
    :dtype-not-dpas :non-canonical-orientation :n-pitch-unaligned :k-pitch-unaligned
    :non-zero-matrix-init :partial-matrix-k-fragment :matrix-family-not-lowered
    :matrix-instruction-not-lowered
    ;; declared-operand and quant-leaf legality
    :operand-without-a-declared-map :missing-declared-contract-axes
    :declared-operand-not-read-by-the-body :declared-map-does-not-match-the-body-index
    :dp4a-needs-int8-operands :dp4a-needs-two-declared-operands
    :decode-on-a-body-replacing-leaf :inner-extent-not-a-multiple-of-4
    :inner-stage-accumulator-not-integral :inner-stage-axis-is-not-contiguous
    ;; epilogue legality
    :epilogue-needs-2-free :epilogue-ignores-accumulator :accumulator-used-more-than-once
    :reduction-in-epilogue :layout-changing-op-in-epilogue
    :epilogue-unsupported-by-this-leaf
    :register-tiled-kernel-body-declined :scalar-region-kernel-body-declined})

(defn- decline
  "A structured record of ONE leaf declining. `:data` is the gate's own ex-data minus its `:reason`
   (already lifted) — deliberately small: a descriptor is compared and logged, and stuffing whole
   forms in here makes diffs unreadable and would poison any future descriptor-derived cache key."
  [leaf reason message data]
  (cond-> {:leaf leaf :reason reason}
    message (assoc :message message)
    (seq data) (assoc :data data)))

(defn- decline-of
  "Classify an ExceptionInfo from a leaf gate. Returns a decline record for a WHITELISTED legality
   reason; RETHROWS anything else — an unrecognized or absent reason is the compiler saying
   something is wrong, and a violated invariant is not a routing decision."
  [leaf ^clojure.lang.ExceptionInfo e]
  (let [data (ex-data e)
        reason (:reason data)
        reported-reason (if (contains? #{:register-tiled-kernel-body-declined
                                         :scalar-region-kernel-body-declined}
                                       reason)
                          (:missing-rule data)
                          reason)]
    (when-not (contains? decline-reasons reason) (throw e))
    (decline leaf reported-reason (.getMessage e) (dissoc data :reason :missing-rule))))

(defn- route-2free-1contract
  "The tensorize fast path: DPAS if the gate accepts, else the register-tiled portable kernel.

   Returns a descriptor, or `{::declines [...]}` when BOTH tensorize leaves refuse — the caller then
   routes to the general naive leaf and carries the declines onto ITS descriptor.

   This used to be `(catch ExceptionInfo _ nil)`. The emitters throw messages as specific as
   \"tensorize: operand must be an aget\" and \"tensorize: needs literal dims\"; the catch discarded
   every one of them, so a canonical matmul demoted to `:naive-segred` with NOTHING recorded, and the
   only way to discover why was to read the emitter source. Two things are true at once and the old
   shape could express neither: a decline is usually LEGITIMATE (symbolic dims, a non-`+` combine and
   a non-product body are all perfectly good contractions that merely are not tiled-leaf shaped), and
   it is always worth REPORTING."
  [out-sym dtype desc tile epilogue contract-facts operation-id candidate-families]
  (let [acc (volatile! [])
        note! (fn [d] (vswap! acc conj d) nil)
        matrix? (contains? candidate-families :matrix)
        register-tiled? (contains? candidate-families :register-tiled)]
    (try
      (let [scheduled (when matrix?
                        (contraction-schedule/plan-matrix-body
                         contract-facts desc tile {:operation-id operation-id}))
            dpas (cond
                   (not matrix?)
                   {:tensorized false :reason :schedule-family-disabled :family :matrix}

                   (:ok scheduled)
                   (sco/generate-dpas-kernel-body (:body scheduled) out-sym)

                   :else
                   {:tensorized false :reason (:reason scheduled) :detail scheduled})]
        (when-not (:tensorized dpas)
          (note! (decline :dpas (:reason dpas) nil (dissoc dpas :tensorized :reason))))
        (if (:tensorized dpas)
          (let [[M N _L] (:dims dpas)
                {:keys [block-m block-n]} (:tile dpas)]
            {:strategy :dpas
             :kernel-name (:kernel-name dpas)
             :source (:source dpas)
             :array-params (:array-params dpas)          ; [row col] = [A-slot B-slot]
             :abi (:abi dpas)
             :dtype :half :out-dtype :half :out-elems (* M N)
             :tile (:tile dpas)                          ; the DERIVED tile actually emitted
             :kernel-body (:kernel-body dpas)             ; scheduled target-neutral body, when present
             :fused-epilogue (boolean epilogue)
         ;; extra kernel args the epilogue needs, in signature order (after out + the dims)
             :epilogue-operands (:epilogue-operands dpas)
             :epilogue-scalars (:epilogue-scalars dpas)
             :epilogue-params (:epilogue-params dpas)
             :wg (:workgroup dpas)                       ; derived from that tile
             :grid [(ceil-div N block-n) (ceil-div M block-m)]  ; [gc-n gc-m] (id0=N, id1=M)
             :scalar-args (mapv (fn [v] {:type :int :value (int v)}) (:dims dpas))  ; [m n k] params
             :dims (:dims dpas)})
      ;; gate rejected (dtype/orientation/pitch) → portable register-tiled kernel when enabled
          (if register-tiled?
            (let [rt (sco/generate-register-tiled-kernel-body
                      contract-facts out-sym :operation-id operation-id :descriptor desc)
                  [bm bn _bk] (:block rt)
                  [M N _L] (:dims rt)]
              {:strategy :regtiled
               :fallback-reason (:reason dpas)             ; kept: existing consumers read it
               :declines @acc
               :kernel-name (:kernel-name rt)
               :source (:source rt)
               :array-params (:array-params rt)
               :abi (:abi rt)
               :dtype (:dtype rt) :out-dtype (:dtype rt) :out-elems (* M N)
               :kernel-body (:kernel-body rt)
               :fused-epilogue (boolean epilogue)
               :epilogue-operands (:epilogue-operands rt)
               :epilogue-scalars (:epilogue-scalars rt)
               :wg (:workgroup rt)
               :grid [(ceil-div N bn) (ceil-div M bm)]
               :scalar-args []                             ; regtiled bakes dims → no scalar params
               :dims (:dims rt)})
            (do
              (note! (decline :regtiled :schedule-family-disabled nil
                              {:family :register-tiled}))
              {::declines @acc}))))
      (catch clojure.lang.ExceptionInfo e
     ;; whichever tensorize leaf threw, record WHY and let the caller fall through. `decline-of`
     ;; rethrows :raster/fatal and :raster/bug — those are not routing decisions.
        (note! (decline-of (if (seq @acc) :regtiled :dpas) e))
        {::declines @acc}))))

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
                          :abi (:abi k)
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

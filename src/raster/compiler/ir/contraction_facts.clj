(ns raster.compiler.ir.contraction-facts
  "FACTS: the one derivation of everything checkable about a `par/contract` form.

   WHY THIS EXISTS — a gate must not validate its own arguments. Two silent miscompiles shipped
   because a gate was handed a value derived from the very thing it was meant to check:

     • `stages-legal?` was called with contract axes derived FROM THE STAGES, so its span rule
       could never fire; a stage under-covering its axis emitted a kernel summing a fraction of
       the terms while the interpreted path summed all of them.
     • the dp4a gate verified operand maps but never the body SHAPE, and that leaf DISCARDS the
       body — so an extra factor was silently dropped.

   Patching each instance leaves the shape intact. This closes it structurally: there is exactly
   ONE producer of facts, its only input is the FORM, and a checker takes facts — so there is no
   argument to pass inconsistently. A checker that wants the contract axes cannot be handed a
   different set; it reads `:contract-axes`, which came from the form's own declaration slot.

   DECLARATIONS ARE EVIDENCE, NOT FACTS. A form may declare operand axis-maps (`:maps`), which is
   what lets a merged/batched operand be understood at all — a derived map cannot guess grouping.
   But a declaration is admitted only after `am/index-matches?` proves it generates the operand's
   actual index. A declaration that fails verification does not fall back to derivation; it makes
   the whole extraction fail. Trusting a declared layout while having checked only its axis
   symbols is how a transpose rewrite risked a silent miscompile."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as od]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as contract-stages]
            [raster.compiler.ir.reduction :as reduction]))

(def ^:private facts-tag ::facts)

(defn facts?
  "Is `x` the output of `contraction-facts`? Checkers assert this so a hand-built map — the shape
   that made the original defects writable — cannot be substituted."
  [x]
  (boolean (and (map? x) (get x facts-tag))))

(defn- form-opts
  "The trailing option map of a contract form, parsed ONCE. (It was previously re-parsed with
   `(apply hash-map (drop 5 form))` at six separate call sites.)"
  [form]
  (let [tail (drop 5 form)]
    (when (odd? (count tail))
      (throw (ex-info "contract form: trailing options must be key/value pairs"
                      {:reason :malformed-opts :opts (vec tail)})))
    (apply hash-map tail)))

(defn flatten-contract-axes
  "Normalize one or more contraction axes to the single reduced coordinate consumed by SegRed.
   Returns `[index extent body]`, with original axis references replaced by row-major coordinate
   decomposition. This is part of semantic fact construction, not an emitter-specific rewrite."
  [contract-axes body]
  (if (= 1 (count contract-axes))
    (let [[[index extent]] contract-axes]
      [index extent body])
    (let [flat-index (gensym "contract_index__")
          extents (mapv second contract-axes)
          multiply (fn [values]
                     (if (every? number? values)
                       (reduce * values)
                       (reduce (fn [left right]
                                 (list 'clojure.core/* left right))
                               values)))
          flat-extent (multiply extents)
          suffix (fn [position]
                   (let [remaining (subvec extents (inc position))]
                     (if (empty? remaining) 1 (multiply remaining))))
          substitutions
          (into {}
                (map-indexed
                 (fn [position [index _]]
                   [index (list 'clojure.core/rem
                                (list 'clojure.core/quot flat-index (suffix position))
                                (nth extents position))])
                 contract-axes))]
      [flat-index flat-extent (walk/postwalk-replace substitutions body)])))

(defn- canonical-reduction-facts
  [out contract-axes body opts dtype]
  (when (seq contract-axes)
    (let [stages (:stages opts)
          semantic-body (if (seq stages)
                          (contract-stages/flat-equivalent stages body)
                          body)
          [index extent flat-body] (flatten-contract-axes contract-axes semantic-body)
          accumulator (gensym "contract_acc__")
          operator
          (reduction/scalar
           {:accumulator accumulator
            :neutral (get opts :init 0.0)
            :dtype (or (:acc-dtype opts) dtype)
            :result out
            :index index
            :step-result (list (get opts :combine '+) accumulator flat-body)
            :algebra (or (:algebra opts) {})
            :attributes {:source :raster.par/contract
                         :contract-axes (mapv vec contract-axes)
                         :stages stages}})]
      {:reduction operator
       :flat-contract-axis [index extent]})))

(defn- aget-terms
  "Every array read the expression makes, as {:sym :idx}, in encounter order.

   Delegates to `od/aget-reads`, which classifies via the registry — so bare `aget`,
   `clojure.core/aget` (WHAT THE WALKER ACTUALLY EMITS), `raster.arrays/aget`, and a devirtualized
   `.invk` are all recognized. This docstring previously CLAIMED that while the code matched
   `(= 'aget (first f))`; the result was `:operands []` for every compiled deftm, so no tensorize
   gate could ever see an operand and a canonical f16 matmul routed to `:naive-segred`."
  [expr]
  (mapv #(select-keys % [:sym :idx]) (od/aget-reads expr)))

(defn- verify-declared-map
  "A declared map is admitted only if it provably generates the operand's ACTUAL index."
  [sym declared idx]
  (when declared
    (when-not (am/index-matches? declared idx)
      (throw (ex-info (str "contract form: declared :maps entry for `" sym
                           "` does not generate that operand's actual index")
                      {:reason :declared-map-does-not-match-the-body-index
                       :sym sym :declared (am/index-expr declared) :actual idx})))
    declared))

(defn contraction-facts
  "Derive the checkable facts of `(raster.par/contract out free-axes contract-axes body & opts)`.
   The ONLY input is the form (plus the intended element dtype, which is a compilation choice
   rather than a property of the form). Throws on a malformed form or an unverifiable declaration."
  [form & {:keys [dtype] :or {dtype :double}}]
  (when-not (and (seq? form) (= 'raster.par/contract (first form)))
    (throw (ex-info "contraction-facts: not a par/contract form" {:reason :not-a-contract-form
                                                                  :form form})))
  (let [[_ out free-axes contract-axes body] form
        opts (form-opts form)
        _ (when-not (vector? free-axes)
            (throw (ex-info "contract form: free-axes must be a vector" {:reason :malformed-free-axes})))
        _ (when-not (vector? contract-axes)
            (throw (ex-info "contract form: contract-axes must be a vector" {:reason :malformed-contract-axes})))
        declared (:maps opts)
        ;; `:decode` is the per-operand LOAD-LAMBDA, an expression in `x` (the raw load). This is
        ;; where a zero-point subtraction belongs — exact on the load path, needing no correction
        ;; reductions — whereas a scale that factors out of the sum belongs in the epilogue.
        decode (:decode opts)
        terms (mapv (fn [{:keys [sym idx]}]
                      (cond-> {:sym sym :idx idx
                               :map (verify-declared-map sym (get declared sym) idx)}
                        (get decode sym) (assoc :decode (get decode sym))))
                    (aget-terms body))
        ;; role → axis symbol, so a leaf contract can say [:free0 :contract0] instead of naming
        ;; the user's own symbols
        roles (merge (into {} (map-indexed (fn [n [a _]] [(keyword (str "free" n)) a])) free-axes)
                     (into {} (map-indexed (fn [n [a _]] [(keyword (str "contract" n)) a])) contract-axes))
        dims (merge (into {} (map-indexed (fn [n [_ e]] [(keyword (str "free" n)) e])) free-axes)
                    (into {} (map-indexed (fn [n [_ e]] [(keyword (str "contract" n)) e])) contract-axes))
        normalized (canonical-reduction-facts out contract-axes body opts dtype)]
    (merge
     {facts-tag true
      :form form
      :out out
      :free-axes (mapv vec free-axes)
      :contract-axes (mapv vec contract-axes)
      :n-free (count free-axes)
      :n-contract (count contract-axes)
      :body body
      :operands terms
      ;; Compatibility projections. New semantic and schedule passes consume :reduction; leaf
      ;; gates still read these until the contraction KernelBody vertical is complete.
      :combine (get opts :combine '+)
      :init (get opts :init 0.0)
      :dtype dtype
      :out-dtype (get opts :out-dtype)
      :stages (:stages opts)
      :epilogue (:epilogue opts)
      :roles roles
      :dims dims
      :opts opts}
     normalized)))

(defn scalar-reduction-view
  "Project the canonical one-component ProductReduction into the contraction facts needed by
   legality and schedule passes. This is a checked view, not a second stored representation."
  [facts]
  (when-not (facts? facts)
    (throw (ex-info "contraction reduction view requires verified facts"
                    {:reason :raster/bug :facts facts})))
  (let [{:keys [acc init lambda]} (reduction/scalar-op (:reduction facts))]
    (when-not (and (seq? lambda) (= 3 (count lambda)) (= acc (second lambda)))
      (throw (ex-info "canonical contraction reduction is not an accumulator/element fold"
                      {:reason :raster/bug :accumulator acc :lambda lambda})))
    {:accumulator acc
     :neutral init
     :combine (first lambda)
     :element (nth lambda 2)
     :dtype (first (reduction/dtypes (:reduction facts)))}))

;; ── body shape: what a leaf that DISCARDS the body must first account for ────────────
(defn body-product-of
  "If `body` is exactly the product of agets on the operands named by `syms` (each once, no other
   factors), return those terms; otherwise nil.

   This is the requirement a body-REPLACING leaf must carry. dp4a and DPAS both emit a hardware op
   in place of the whole summand, so any term the gate has not accounted for is silently dropped —
   with its array still declared as an unread kernel parameter. `nil` here means `refuse`, and the
   router falls back to a leaf that actually evaluates the body."
  [body syms]
  (let [syms (set syms)]
    (when (and (seq? body) (od/multiplication-op? (od/semantic-op body)))
      (let [args (vec (od/call-args body))]
        (when (= (count syms) (count args))
          (let [terms (keep (fn [t] (let [arr (od/aget-array-sym t)]
                                      (when (contains? syms arr)
                                        {:sym arr :idx (od/aget-index t)})))
                            args)]
            (when (and (= (count terms) (count args))
                       (= syms (set (map :sym terms))))
              (vec terms))))))))

(defn- permutations
  "All orderings of `xs`. Operand counts here are 2 (occasionally 3), so the factorial cost is
   irrelevant and an explicit search is clearer than a variance heuristic."
  [xs]
  (if (<= (count xs) 1)
    (list (vec xs))
    (for [i (range (count xs))
          rest-perm (permutations (concat (take i xs) (drop (inc i) xs)))]
      (into [(nth xs i)] rest-perm))))

(defn- subsets
  [values]
  (if-let [value (first values)]
    (let [tail (subsets (next values))]
      (concat tail (map #(cons value %) tail)))
    (list '())))

(defn operand-axis-map
  "Return a verified physical AxisMap for one contraction operand.

   Explicit maps were verified while facts were constructed. For ordinary dense operands, infer
   only a plain permutation/broadcast map whose generated flat index is provably equal to the
   actual load index. Non-affine gathers and ambiguous layouts return nil and therefore cannot
   enter a schedule that needs a physical buffer shape."
  [facts operand]
  (when-not (facts? facts)
    (throw (ex-info "operand map inference requires verified contraction facts"
                    {:reason :raster/bug :facts facts})))
  (let [operand (if (symbol? operand)
                  (some #(when (= operand (:sym %)) %) (:operands facts))
                  operand)]
    (when operand
      (or (:map operand)
          (let [axes (vec (concat (:free-axes facts) (:contract-axes facts)))
                candidates (for [selection (rest (sort-by count (subsets axes)))
                                 ordering (permutations selection)]
                             (am/of-axes ordering))]
            (first (filter #(am/index-matches? % (:idx operand)) candidates)))))))

;; ── leaf layout requirements as DATA ────────────────────────────────────────────────
(def leaf-layouts
  "Each tensorize leaf's required operand layout, as ROLE → the axis roles that index it,
   outer→inner. `:nn` and `:nt` are not concepts here — they are two different data rows:

     :dpas / :quant  row A[i,l]  col B[l,j]   → col is [:contract0 :free1]
     :dp4a           row A[i,l]  col B[j,l]   → col is [:free1 :contract0]  (K-contiguous)

   The required layout is a property of the LEAF (dp4a packs 4 int8 along the contraction, so both
   operands must be contiguous in it), never of the contraction. Adding a layout — a batch axis, a
   merged group, a `:tn` orientation — is a row edit, not a new predicate. This replaces three
   hand-written checks that called one pattern-matcher with six different argument orders."
  {:dpas        {:row [:free0 :contract0] :col [:contract0 :free1]}
   :quant-naive {:row [:free0 :contract0] :col [:contract0 :free1]}
   :dp4a        {:row [:free0 :contract0] :col [:free1 :contract0]}})

(defn- role-map
  "The axis-map a role-spec denotes, e.g. [:free0 :contract0] → of-axes [[i M] [l L]] (index i·L+l)."
  [facts axis-roles]
  (am/of-axes (mapv (fn [r] [(get-in facts [:roles r]) (get-in facts [:dims r])]) axis-roles)))

(defn check-layout
  "Assign the body's operands to a leaf's declared roles BY VERIFICATION, and return
   {:ok true :bindings {role sym}} or {:ok false :reason :layout …}.

   This replaces the variance test (\"row is the operand depending on free0+contract but not
   free1\"). An operand is the row operand because its index PROVABLY IS the row layout — not
   because of which axes it happens to mention. Every candidate assignment is checked, so a leaf
   cannot be handed an operand whose layout was assumed rather than proved.

   `required` is a `leaf-layouts` entry."
  [facts required]
  (when-not (facts? facts)
    (throw (ex-info "check-layout: expected contraction-facts" {:reason :raster/bug})))
  (let [ops (:operands facts)
        roles (vec (keys required))]
    (if (not= (count ops) (count roles))
      {:ok false :reason :operand-count
       :required (count roles) :actual (count ops) :operands (mapv :sym ops)}
      (let [fits? (fn [assignment]
                    (every? (fn [[role op]]
                              (am/index-matches? (role-map facts (get required role)) (:idx op)))
                            assignment))
            candidates (map #(zipmap roles %) (permutations ops))]
        (if-let [hit (first (filter fits? candidates))]
          {:ok true :bindings (into {} (map (fn [[r o]] [r (:sym o)])) hit)}
          {:ok false :reason :layout
           :required (into {} (map (fn [r] [r (am/index-expr (role-map facts (get required r)))])) roles)
           :actual (mapv (juxt :sym :idx) ops)})))))

(defn layout-maps
  "For a PASSING `check-layout` verdict, the axis-map of each operand — derived from the layout the
   verdict proved, keyed by operand symbol.

   This is how a tensorize leaf gets its operand maps without a caller declaring them. The map is
   verified by construction: `check-layout` only returns bindings after `am/index-matches?` proves
   each operand's actual index equals the required layout's. So the leaf gets the strongest form of
   what it needs — a layout that was checked, not asserted — and a form need not carry `:maps` just
   to reach a peak leaf."
  [facts required verdict]
  (when (:ok verdict)
    (into {} (map (fn [[role sym]] [sym (role-map facts (get required role))]))
          (:bindings verdict))))

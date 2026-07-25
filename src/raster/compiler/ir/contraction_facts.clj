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
  (:require [raster.compiler.ir.axis-map :as am]
            [raster.compiler.core.op-descriptor :as od]))

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

(defn- aget-terms
  "Every `(aget arr idx)` the expression reads, as {:sym :idx}, in encounter order. Uses the
   op-descriptor's aget classification rather than a local literal set, so a devirtualized
   `.invk` read is recognized the same way a bare `aget` is."
  [expr]
  (->> (tree-seq coll? seq expr)
       (keep (fn [f]
               (when (and (seq? f) (= 'aget (first f)) (symbol? (second f)))
                 {:sym (second f) :idx (nth f 2)})))
       vec))

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
        terms (mapv (fn [{:keys [sym idx]}]
                      {:sym sym :idx idx
                       :map (verify-declared-map sym (get declared sym) idx)})
                    (aget-terms body))
        ;; role → axis symbol, so a leaf contract can say [:free0 :contract0] instead of naming
        ;; the user's own symbols
        roles (merge (into {} (map-indexed (fn [n [a _]] [(keyword (str "free" n)) a])) free-axes)
                     (into {} (map-indexed (fn [n [a _]] [(keyword (str "contract" n)) a])) contract-axes))
        dims (merge (into {} (map-indexed (fn [n [_ e]] [(keyword (str "free" n)) e])) free-axes)
                    (into {} (map-indexed (fn [n [_ e]] [(keyword (str "contract" n)) e])) contract-axes))]
    {facts-tag true
     :form form
     :out out
     :free-axes (mapv vec free-axes)
     :contract-axes (mapv vec contract-axes)
     :n-free (count free-axes)
     :n-contract (count contract-axes)
     :body body
     :operands terms
     :combine (get opts :combine '+)
     :init (get opts :init 0.0)
     :dtype dtype
     :out-dtype (get opts :out-dtype)
     :stages (:stages opts)
     :epilogue (:epilogue opts)
     :roles roles
     :dims dims
     :opts opts}))

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
          (let [terms (keep (fn [t] (when (and (seq? t) (= 'aget (first t))
                                               (contains? syms (second t)))
                                      {:sym (second t) :idx (nth t 2)}))
                            args)]
            (when (and (= (count terms) (count args))
                       (= syms (set (map :sym terms))))
              (vec terms))))))))

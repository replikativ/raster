(ns raster.compiler.passes.parallel.staged-contraction-schedule
  "Schedule admission for staged contractions, independent of target source emission.
   Operand maps, body equivalence and numerical bounds are checked here; emitters consume the
   resulting packed index maps. No quantization-format registry or alternate scalar IR."
  (:require [raster.compiler.core.dtype :as dt]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.scalar-range :as scalar-range]))

(defn inner-dp4a-plan
  "Can the INNERMOST stage of a staged contraction be tensorized with dp4a (4 int8 MACs into an
   int32 in one op)? Returns {:ok true :packed-maps {sym amap} :packed-extent n} or
   {:ok false :reason kw}.

   This is the int8 PEAK leaf for block-quant, and it is the same structure llama.cpp hand-writes:
   the inner stage is already an exact int32 accumulation over a short K-contiguous run, which is
   precisely dp4a's shape. Because the stage list says which axis the inner accumulation runs over,
   there is nothing to recognize — the gate only has to CHECK.

   Required, and the failure each check prevents:
     • declared operand axis-maps. Tensorizing needs to know each operand's innermost axis; per the
       compiler's declare-don't-pattern-match rule that is data, not inference.
     • each map must VERIFIABLY equal the operand's actual index in the body (am/index-matches?).
       Assuming a layout while having checked only the axis symbols is how a transpose rewrite
       silently miscompiled before; a leaf may only assume what it has proved.
     • the inner stage's axis must be the INNERMOST axis of both maps — dp4a packs 4 consecutive
       elements along the contraction, so both operands must be contiguous in it (the :nt layout).
     • int8 operands, integral inner accumulator (widening as a dtype pair).
     • the inner extent must be a literal multiple of 4, else the packed load is mis-strided."
  [{:keys [stages body operands dtype]}]
  (let [inner-stage (peek (vec stages))
        int-acc? (contains? #{:int :long :int32} (:dtype inner-stage))
        agets (into {} (map (juxt :sym :idx)) (descriptor/aget-reads body))
        ;; THE BODY IS DISCARDED when this leaf fires — the whole summand is replaced by one
        ;; rstr_dp4a call — so the gate must account for EVERY term first. The requirement lives in
        ;; ir/contraction-facts as `body-product-of`, shared with every other body-replacing leaf,
        ;; rather than re-derived per gate.
        exact-product? (some? (cf/body-product-of body (map :sym operands)))]
    (cond
      (not= 2 (count operands)) {:ok false :reason :dp4a-needs-two-declared-operands}
      (not (every? :map operands)) {:ok false :reason :operand-without-a-declared-map}
      (not (and (dt/known? dtype) (= :byte (dt/canon dtype))))
      {:ok false :reason :dp4a-needs-int8-operands :dtype dtype}
      ;; A :decode is a LOAD-LAMBDA applied by substituting into the body — and this leaf discards
      ;; the body, replacing the whole summand with one rstr_dp4a call. So a zero-point would be
      ;; silently dropped: Σ a·b instead of Σ(a−za)(b−zb), wrong by a constant-plus-linear term with
      ;; no diagnostic. Load-bearing, since q4_0/q8_0 carry zero-points 8 and 128. Same underlying
      ;; reason as :body-has-unmodeled-terms below — the body is not evaluated here.
      (some :decode operands)
      {:ok false :reason :decode-on-a-body-replacing-leaf
       :detail "this leaf replaces the body with a single hardware op, so a per-operand :decode (e.g. a zero-point) would be silently dropped"
       :decoded (mapv :sym (filter :decode operands))}

      (not exact-product?)
      {:ok false :reason :body-has-unmodeled-terms
       :detail "this leaf replaces the body with a single hardware op; any term beyond the two declared operands would be silently dropped"
       :body body :declared (mapv :sym operands)}
      (not int-acc?) {:ok false :reason :inner-stage-accumulator-not-integral
                      :dtype (:dtype inner-stage)}
      ;; The hardware operation consumes AND returns int32, even when the enclosing carry is
      ;; Long. Prove every prefix, not merely the final sum of a sampled input. This also keeps
      ;; scalar and packed accumulation exact for all admitted signed-byte operands.
      (not (scalar-range/contained-in-dtype?
            (scalar-range/accumulation-prefixes
             (when (or (nil? (:init inner-stage))
                       (constant/zero-value? (:init inner-stage)))
               (scalar-range/literal 0 :int))
             (scalar-range/arithmetic :* (repeat 2 (scalar-range/for-dtype :byte)))
             (:extent inner-stage))
            :int))
      {:ok false :reason :unproved-dp4a-accumulator-range
       :dtype (:dtype inner-stage) :extent (:extent inner-stage)}
      :else
      (or
       ;; the declared map must PROVABLY be the operand's actual index expression
       (first (keep (fn [{:keys [sym map]}]
                      (let [idx (get agets sym)]
                        (cond
                          (nil? idx) {:ok false :reason :declared-operand-not-read-by-the-body :sym sym}
                          (not (am/index-matches? map idx))
                          {:ok false :reason :declared-map-does-not-match-the-body-index
                           :sym sym :declared (am/index-expr map) :actual idx}
                          (not= (:axis inner-stage) (am/innermost-axis map))
                          {:ok false :reason :inner-stage-axis-is-not-contiguous
                           :sym sym :innermost (am/innermost-axis map) :axis (:axis inner-stage)}
                          :else nil)))
                    operands))
       (let [p-sym (gensym "p__")
             packed (into {} (for [{:keys [sym map]} operands]
                               [sym (am/pack-innermost map 4 p-sym)]))]
         (if (some nil? (vals packed))
           {:ok false :reason :inner-extent-not-a-multiple-of-4 :extent (:extent inner-stage)}
           {:ok true :packed-maps packed :p-sym p-sym
            :packed-extent (quot (long (:extent inner-stage)) 4)}))))))

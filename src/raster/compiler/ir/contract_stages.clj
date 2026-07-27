(ns raster.compiler.ir.contract-stages
  "STAGED contract axes: a contraction whose reduction accumulates in more than one stage, each
   with its own accumulator dtype.

   WHY THIS EXISTS. `flatten-contract-axes` collapses n≥2 contract axes into one flat reduced
   dim (the A0 convention, Futhark's single-width Screma). That is right for dense contraction —
   and it is exactly what block-quantized arithmetic cannot do. Every real int8/int4 format
   (GGUF q8_0/q4_0, k-quants, AWQ, and raster's own `backend/cpu/quant.clj`) computes

       Σ_blk  d[blk] · ( Σ_{t<B} a[blk,t]·b[blk,t] )
              ^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
              float acc      int32 acc, exact

   The per-block scale multiplies a PARTIAL sum, so the reduction is two-level: an int32 MAC
   inside the block, a float accumulate across blocks. Flatten it and the int accumulator is
   gone — you are back to per-element float multiplies, which is the entire cost the format
   exists to avoid. Our CPU quant backend already schedules it this way (\"deferred per-block
   scale, float-vector accumulation, one reduce/row\"); this namespace is what lets the SAME
   structure be expressed in the contraction algebra instead of beside it.

   Number of stages is not special-cased — it is the format's nesting depth:
     1 stage  → today's flat contraction (nothing changes)
     2 stages → q8_0 / q4_0            (per-block scale)
     3 stages → k-quants               (super-block scale-of-scales)

   STAGES ARE A SCHEDULE, NOT NEW SEMANTICS. Because a lift must be LINEAR in the inner
   accumulator (see `stages-legal?`), a staged contraction is *equal in exact arithmetic* to the
   flat contraction whose body absorbed the lift factors — `flat-equivalent` derives it. Staging
   changes only (a) the accumulator dtype/rounding and (b) performance. That is what makes it a
   schedule annotation, and it is why an interpreter/CPU path needs no staged implementation: it
   runs the flat equivalent. It also hands us a free ORACLE — the flat form is the reference a
   staged device kernel is validated against.

   REPRESENTATION — stages are OUTER→INNER, positionally aligned with the contract axes:

     [{:axis blk :extent NB :dtype :float :init 0.0
       :lift (raster.numeric/* inner (aget da blk) (aget db blk))
       :operands [{:sym da :map <axis-map>} {:sym db :map <axis-map>}]}
      {:axis t   :extent 32 :dtype :int   :init 0}]

   The innermost stage carries no lift — it accumulates the contraction body itself. Each outer
   stage's `:lift` is an expression in the symbol `inner` (the stage-below's accumulated value)
   and may reference its own axis plus extra operand arrays, each with a declared axis-map (so a
   scale is indexed by ITS map, not by pattern-matching — same rule as the epilogue seam)."
  (:require [raster.compiler.core.op-descriptor :as od]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.core.dtype :as dt]
            [clojure.set :as set]
            [clojure.walk :as walk]))

(def ^:private forbidden-in-lift
  "Ops that force a layout/distribution change and so cannot appear in a lift: the lift runs
   between two accumulator levels, where a re-tiling decision is not available to us."
  '#{raster.par/scan raster.par/scan-exclusive raster.par/scatter! raster.par/reduce-by-key
     raster.par/reduce raster.par/reduce-into raster.par/contract})

(defn- mul-head? [h]
  (contains? '#{* clojure.core/* raster.numeric/*} h))

(defn linear-in-inner
  "If `lift` is LINEAR in `inner` — i.e. `inner` itself, or a product one of whose factors is
   `inner` — return the vector of the REMAINING factors (possibly empty). Otherwise nil.

   Linearity is what makes staging a schedule rather than a different computation: it is exactly
   the condition under which the accumulate distributes,
       Σ_blk F(blk)·(Σ_t x)  =  Σ_blk Σ_t F(blk)·x
   so the lift factors can be pushed into the body to obtain the flat equivalent. A non-linear
   lift (say `(sqrt inner)`) has no flat form — it is a genuinely different reduction, and is
   refused rather than silently mis-scheduled."
  [lift inner]
  (cond
    (= lift inner) []
    (and (seq? lift) (mul-head? (first lift)))
    (let [args (vec (rest lift))
          n (count (filter #(= inner %) args))]
      (when (= 1 n) (vec (remove #(= inner %) args))))
    :else nil))

(declare contract-extent)

(defn stages-legal?
  "Legality of a staged contract axis. Returns {:ok true} or {:ok false :reason kw …}.

   Rules, and the failure each one prevents:

     • stages must cover the contract axes exactly once, OUTER→INNER. A stage naming an axis
       that is not a contract axis (or omitting one) means the reduction does not span the
       contraction — a wrong answer, not a slow one.
     • the INNERMOST stage takes no lift: it accumulates the contraction body.
     • every OUTER stage must have a lift, and that lift must use `inner` EXACTLY ONCE. Zero uses
       DISCARDS the partial sum below it (silently wrong); more than one use duplicates the whole
       inner reduction. Direct analogue of the epilogue's accumulator-used-once rule.
     • a lift must be LINEAR in `inner` (see `linear-in-inner`) — otherwise there is no flat
       equivalent and staging would change the computation rather than schedule it.
     • accumulator dtypes must not NARROW going outward: a float inner accumulator feeding an
       integral outer accumulator truncates every partial sum. (int → float, the quant case, is
       the whole point and is fine.)
     • no layout/distribution-changing op in a lift."
  [stages contract-axes & {:keys [inner] :or {inner 'inner}}]
  (let [stages (vec stages)
        ;; dtype classification comes from the ONE registry. The ad-hoc sets this replaces listed
        ;; :short/:uint/:int8/:int32/:float16 — four of which no emitter could spell, so the gate
        ;; blessed accumulator dtypes that silently became "float" C types downstream.
        n (count stages)]
    (cond
      (zero? n) {:ok false :reason :no-stages}

      (not= (mapv :axis stages) (mapv first contract-axes))
      {:ok false :reason :stages-do-not-match-contract-axes
       :stage-axes (mapv :axis stages) :contract-axes (mapv first contract-axes)}

      ;; EXTENTS too, not just axis names. Comparing symbols alone let a stage under-cover its
      ;; axis — a stage of extent 4 against a declared extent of 32 emitted a loop summing an
      ;; EIGHTH of the terms, while the interpreted path summed all of them. The stage list must
      ;; span exactly the space the form declared.
      (not= (mapv :extent stages) (mapv second contract-axes))
      {:ok false :reason :stage-extents-do-not-span-the-contract-axes
       :stage-extents (mapv :extent stages) :declared-extents (mapv second contract-axes)
       :spans (contract-extent stages)
       :declared (am/group-extent (mapv vec contract-axes))}

      (some? (:lift (peek stages)))
      {:ok false :reason :innermost-stage-has-a-lift :axis (:axis (peek stages))}

      :else
      (or
       ;; every outer stage: lift present, uses `inner` once, linear, no forbidden ops
       (first
        (keep
         (fn [{:keys [axis lift]}]
           (let [nodes (tree-seq coll? seq lift)
                 uses (count (filter #(= inner %) nodes))
                 heads (into #{} (keep #(when (seq? %) (first %))) nodes)]
             (cond
               (nil? lift) {:ok false :reason :outer-stage-without-a-lift :axis axis}
               (zero? uses) {:ok false :reason :lift-discards-inner-accumulator :axis axis}
               (> uses 1) {:ok false :reason :inner-accumulator-used-more-than-once
                           :axis axis :uses uses}
               (seq (set/intersection heads forbidden-in-lift))
               {:ok false :reason :layout-changing-op-in-lift :axis axis
                :ops (set/intersection heads forbidden-in-lift)}
               (nil? (linear-in-inner lift inner))
               {:ok false :reason :lift-not-linear-in-inner :axis axis :lift lift}
               :else nil)))
         (butlast stages)))
       ;; dtypes must not narrow outward (inner → outer)
       ;; an accumulator dtype the compiler cannot spell is a REFUSAL, not a silent substitution
       (first (keep (fn [{:keys [axis dtype]}]
                      (when-not (dt/known? dtype)
                        {:ok false :reason :unknown-stage-accumulator-dtype :axis axis :dtype dtype}))
                    stages))
       (first
        (keep (fn [[outer inner-st]]
                (when (and (dt/integral? (:dtype outer))
                           (not (dt/integral? (:dtype inner-st))))
                  {:ok false :reason :narrowing-stage-accumulator
                   :outer (:dtype outer) :inner (:dtype inner-st)}))
              (partition 2 1 stages)))
       {:ok true}))))

(defn contract-extent
  "Total extent of the staged contract axis = product of the stage extents. Folds literals and
   stays symbolic otherwise (reuses the axis-map product so one rule covers both)."
  [stages]
  (am/group-extent (mapv (fn [{:keys [axis extent]}] [axis extent]) stages)))

(defn lift-operands
  "All operand specs across all stages, in outer→inner stage order. These become EXTRA kernel
   params (the scale arrays), so a caller must bind them — surfaced explicitly for the same
   reason the epilogue's operands are: omitting them from a launch descriptor is an arity bug."
  [stages]
  (vec (mapcat :operands stages)))

(defn stage-index-exprs
  "Map of operand symbol → its flat index expression, taken from each operand's DECLARED axis-map.
   The emitter substitutes these into `(aget scale …)` rather than inferring strides."
  [stages]
  (into {} (for [{:keys [sym map]} (lift-operands stages)] [sym (am/index-expr map)])))

(defn substitute-operand-indices
  "Rewrite `(aget s _)` → `(aget s <index from s's declared map>)` for every lift operand,
   preserving the read's spelling and metadata.

   Registry-classified: a lift is AUTHOR-WRITTEN data, and CLAUDE.md's emit-qualified rule tells
   authors to spell the read `raster.arrays/aget` — which the old literal `(= 'aget (first f))`
   silently ignored, leaving the operand's index as its PLACEHOLDER. That corrupts the emitted
   kernel AND `flat-equivalent`, which is the interpreter-side semantic reference the staging
   linearity law is checked against — so the oracle would have agreed with the wrong kernel."
  [expr idx-of]
  (od/rewrite-aget-indices expr idx-of))

(defn flat-equivalent
  "The FLAT contraction body equal (in exact arithmetic) to this staged contraction: every stage's
   lift factors multiplied into the body. Returns `body` unchanged for a single stage.

   This is the semantics of a staged contraction — staging is a schedule over it. Two uses:
   an interpreter/CPU path evaluates this instead of implementing stages at all, and a staged
   device kernel is validated against it (they agree up to the accumulator's rounding, which is
   the only thing staging actually changes).

   The returned body is SELF-CONTAINED: each lift operand's `aget` index is substituted from its
   declared axis-map, so the placeholder index in a lift expression never escapes to a consumer.
   The stage axes remain free variables — the caller supplies the flat contract axis and its
   decomposition (`flatten-contract-axes` does exactly that), so `da[i,blk]` becomes
   `da[i*NB + l/B]` once the axes are flattened — no extra machinery."
  [stages body & {:keys [inner] :or {inner 'inner}}]
  (let [stages (vec stages)
        idx-of (stage-index-exprs stages)
        factors (map #(substitute-operand-indices % idx-of)
                     (mapcat (fn [{:keys [lift]}]
                               (when lift (linear-in-inner lift inner)))
                             (butlast stages)))]
    (if (empty? factors)
      body
      (cons 'raster.numeric/* (cons body (vec factors))))))


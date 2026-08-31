(ns raster.compiler.ir.axis-map
  "Indexing maps: how an operand's PHYSICAL index is built from the ITERATION axes.

   This is the one abstraction MLIR's `linalg.generic` has that our `par/contract` lacked.
   `par/contract` already carries the other two thirds — `iterator_types` (free axes = parallel,
   contract axes = reduction) and the body region — but the *indexing map* was implicit inside
   the body's `aget` index arithmetic and had to be recovered by pattern-matching. With the map
   declared, orientation is DATA: `:nn` vs `:nt` is `[[i][l]] · [[l][j]]` vs `[[i][l]] · [[j][l]]`,
   not a special case. The map is literally the einsum subscript, kept instead of discarded.

   REPRESENTATION — `{:groups [[[i M]] [[l K]]]}`:
     • one GROUP per physical dimension, outer→inner (row-major across groups);
     • each group is a vector of `[axis-sym extent]` pairs, row-major WITHIN the group.
   A multi-axis group is a MERGE (einops `\"b (c h) w\"` → `[[[b B]] [[c C] [h H]] [[w W]]]`);
   reading a merged dim back as separate iteration axes is a SPLIT. So split/merge need no extra
   machinery — they are one nesting level in the same structure. Also covers, for free:
     transpose  — reorder groups            broadcast — omit an axis
     batch      — extra leading group       diagonal  — repeat an axis

   Extents may be numbers or symbols; index expressions fold to constants when everything is
   literal and stay symbolic otherwise."
  (:refer-clojure :exclude [shape])
  (:require [raster.compiler.core.op-descriptor :as od]))

;; ── construction ────────────────────────────────────────────────────────────────────
(defn of-axes
  "Map for a plain (non-merged) operand: one axis per physical dim.
   (of-axes '[[i M] [l K]]) => {:groups [[[i M]] [[l K]]]}"
  [axis-extent-pairs]
  {:groups (mapv vector axis-extent-pairs)})

(defn of-groups
  "Map from explicit groups (each a vector of [axis extent] pairs)."
  [groups]
  {:groups (mapv vec groups)})

;; ── queries ─────────────────────────────────────────────────────────────────────────
(defn- mul
  "Product that folds literals and stays symbolic otherwise."
  [xs]
  (let [xs (remove #(= 1 %) xs)]
    (cond (empty? xs) 1
          (every? number? xs) (reduce * xs)
          (= 1 (count xs)) (first xs)
          :else (let [nums (filter number? xs) syms (remove number? xs)
                      k (reduce * 1 nums)]
                  (if (= 1 k) (cons 'clojure.core/* syms)
                      (cons 'clojure.core/* (cons k syms)))))))

(defn group-extent [group] (mul (map second group)))

(defn shape
  "Physical shape: one extent per group."
  [amap]
  (mapv group-extent (:groups amap)))

(defn n-elements [amap] (mul (shape amap)))

(defn axes
  "Flat list of iteration-axis symbols in map order (atomic order)."
  [amap]
  (vec (for [g (:groups amap) [a _] g] a)))

(defn rank [amap] (count (:groups amap)))

;; ── index generation (the single source of row-major decomposition) ──────────────────
(defn- rowmajor
  "Row-major flat index over `[[expr extent] …]` outer→inner: Σ_p expr_p × Π_{q>p} extent_q.
   Folds literals; emits `clojure.core` index arithmetic (integer index math, per CLAUDE.md)."
  [pairs]
  (let [v (vec pairs)
        n (count v)
        terms (keep-indexed
               (fn [p [e _]]
                 (let [suf (mul (map second (subvec v (inc p) n)))]
                   (cond (= 0 e) nil
                         (= 1 suf) e
                         :else (list 'clojure.core/* e suf))))
               v)
        ;; Flatten nested sums so that two maps with the same ATOMIC order produce a
        ;; STRUCTURALLY identical index expression — that syntactic equality is what makes
        ;; "a regroup is a pure reinterpretation, no data movement" checkable by `=`.
        terms (mapcat (fn [t] (if (and (seq? t) (= 'clojure.core/+ (first t))) (rest t) [t]))
                      (remove nil? terms))]
    (case (count terms)
      0 0
      1 (first terms)
      (cons 'clojure.core/+ terms))))

(defn index-expr
  "The operand's flat physical index as an S-expression of the iteration-axis symbols.
   Row-major across groups; row-major within each group (so a merged group `[c h]` contributes
   `c*H + h`). This is THE row-major decomposition — emitters call it instead of re-deriving."
  [amap]
  (rowmajor (for [g (:groups amap)]
              [(rowmajor (for [[a e] g] [a e])) (group-extent g)])))

(defn coordinate-exprs
  "Return one row-major coordinate expression per physical axis-map group.

  Unlike `index-expr`, this preserves physical rank for scheduled storage whose KernelParameter
  is not flattened. A merged group still becomes one coordinate."
  [amap]
  (mapv (fn [group]
          (rowmajor (for [[axis extent] group] [axis extent])))
        (:groups amap)))

;; ── layout relations (what the gate and the rearrange-inserter ask) ──────────────────
(defn canonical?
  "Is this map the plain row-major layout over `expected-axes` (in that order, unmerged)?
   Replaces the ad-hoc `(+ (* outer stride) inner)` pattern match: a comparison, not a regex."
  [amap expected-axes]
  (= (mapv (fn [g] (mapv first g)) (:groups amap))
     (mapv vector expected-axes)))

(defn same-atomic-order?
  "True when two maps visit the SAME atomic axes in the same order and differ only in grouping.
   Then one is a pure RESHAPE of the other — a reinterpretation with NO data movement, which is
   exactly the case a rearrange-elision pass may drop."
  [a b]
  (= (axes a) (axes b)))

(defn permutation
  "The permutation of `from`'s groups that yields `to`'s group order, or nil when `to` is not a
   reordering of `from` (different axis sets, or a regroup rather than a reorder). Used to turn a
   layout MISMATCH into a concrete transpose — the general form of the old :nn→:nt special case."
  [from to]
  (let [fk (mapv (fn [g] (mapv first g)) (:groups from))
        tk (mapv (fn [g] (mapv first g)) (:groups to))]
    (when (and (= (count fk) (count tk)) (= (set fk) (set tk)) (apply distinct? fk))
      (mapv #(.indexOf ^java.util.List fk %) tk))))

(defn- op-head
  "Unify +/* heads (bare, clojure.core, raster.numeric)."
  [h] (get '{+ + clojure.core/+ + raster.numeric/+ +
             * * clojure.core/* * raster.numeric/* *} h h))

;; ── affine normal form: ONE canonicalization for index expressions ───────────────────
;; An index expression is normalized to {atom → polynomial-coefficient}, where an atom is an
;; ITERATION AXIS (or an opaque subterm such as quot/mod/rem/aget) and a coefficient is a map
;; from a sorted vector of symbolic factors to a number — so `(* i k)` and `(* k i)`, and
;; `(+ (* i K) (+ (* b B) t))` and `(+ (* i K) (* b B) t)`, all normalize to one value.
;;
;; WHY NORMALIZE RATHER THAN COMPARE STRUCTURALLY. This file previously carried TWO relations
;; that accepted DIFFERENT languages: a hand-rolled pattern match that was +/*-order-agnostic but
;; could not see merged groups, and a structural comparison that flattened nested sums but
;; rejected commuted ones. A legal :nt operand written `(+ l (* j K))` therefore passed one gate
;; and was rejected by the other — reaching no leaf at all. And an over-strict relation is worse
;; than a lax one here: it rejects CORRECT declarations, which teaches callers to skip
;; verification, and a skipped gate is a silent one.
;;
;; `axes` is required because `(* i k)` is ambiguous without it — axis × extent, or axis × axis.
;; Symbols in `axes` are atoms; every other symbol is a coefficient factor.
(defn- p* [a b]
  (reduce (fn [acc [sa ca]]
            (reduce (fn [acc [sb cb]]
                      (update acc (vec (sort-by str (concat sa sb))) (fnil + 0) (* ca cb)))
                    acc b))
          {} a))
(defn- p+ [a b] (into {} (remove (comp zero? val)) (merge-with + a b)))
(def ^:private p-one {[] 1})

(defn affine
  "Index expression → `{atom → coeff-polynomial}` (the constant term keyed `::one`), or nil when
   the expression is not affine in `axes` (e.g. an axis multiplied by an axis). quot/mod/rem and
   nested agets are treated as OPAQUE atoms and never distributed through — distributing them
   would make two different gathers compare equal, which is a silent wrong-operand bug."
  [e axes]
  (let [axes (set axes)
        opaque (fn [x] [::opaque (pr-str x)])
        aff (fn aff [e]
              (cond
                (number? e) {::one {[] e}}
                (symbol? e) (if (axes e) {e p-one} {::one {[e] 1}})
                ;; The walked dialect wraps axes AND extents in integer casts —
                ;; `(clojure.core/* i (long k))`. Without unwrapping, `(long k)` is an OPAQUE atom,
                ;; the index is not the row-major layout, and the DPAS orientation gate declines
                ;; with :non-canonical-orientation — so a compiled deftm cannot reach the tensorized
                ;; leaf even once its operands are recognized. `od/unwrap-int-cast` is the same
                ;; unwrapping `affine-step` and `idx-matches?` already apply; using it here makes
                ;; the affine normal form agree with them instead of being stricter by accident.
                (and (seq? e) (not= e (od/unwrap-int-cast e))) (aff (od/unwrap-int-cast e))
                (seq? e)
                (let [h (op-head (first e)) args (rest e)]
                  (cond
                    (= '+ h) (reduce (fn [acc a] (when-let [x (aff a)] (when acc (merge-with p+ acc x))))
                                     {} args)
                    (= '* h) (reduce (fn [acc a]
                                       (when-let [x (aff a)]
                                         (when acc
                                           ;; affine × affine is affine only if at most one side
                                           ;; carries an axis
                                           (let [ax? (fn [m] (some #(not= ::one %) (keys m)))]
                                             (when-not (and (ax? acc) (ax? x))
                                               (let [[base other] (if (ax? acc) [acc x] [x acc])
                                                     k (get other ::one)]
                                                 (when k
                                                   (into {} (map (fn [[a c]] [a (p* c k)])) base))))))))
                                     {::one p-one} args)
                    :else {(opaque e) p-one}))
                :else {(opaque e) p-one}))]
    (some->> (aff e) (into {} (remove (comp empty? val))))))

(defn index=
  "Do two index expressions denote the same element for all values of `axes`? The single index
   relation — replaces the two divergent ones. Returns false when either side is non-affine."
  [a b axes]
  (let [pa (affine a axes) pb (affine b axes)]
    (boolean (and pa pb (= pa pb)))))

(defn index-matches?
  "Does `idx-expr` equal this map's generated index? The VERIFICATION step: a leaf may only assume
   an operand's layout if the operand's actual index provably IS that layout."
  [amap idx-expr]
  (index= idx-expr (index-expr amap) (axes amap)))

(defn transposed-2d?
  "True when `to` is `from` with its two groups swapped (the 2-D transpose case a physical
   transpose kernel realizes)."
  [from to]
  (= [1 0] (permutation from to)))

;; ── flat index → free-axis map (what an epilogue's operands need) ────────────────────
;; A contraction's consumer is usually a 1-D map over the FLAT output: `(par/map! out t (* M N) …)`
;; with `t = i·N + j` for free axes [[i M] [j N]]. To fuse that map as an epilogue we must express
;; each operand's index in terms of the free axes instead of `t`. These are the broadcast shapes
;; that appear in practice; anything else returns nil so the caller refuses to fuse rather than
;; guessing (an operand indexed wrongly is a silent miscompile).

(defn- core-op? [h & names] (contains? (set (mapcat (fn [n] [n (symbol "clojure.core" (name n))
                                                             (symbol "raster.numeric" (name n))]) names)) h))

(defn flat-index->map
  "Interpret `idx-expr`, an index expression in the flat map variable `t`, as an axis-map over the
   contraction's `free-axes` (a vector of [sym extent], outer→inner). Recognized:

     t              → the full output layout           (elementwise operand, e.g. a residual)
     (mod  t Ninner)→ the innermost axis only          (per-column broadcast, e.g. a bias)
     (rem  t Ninner)→ same
     (quot t Ninner)→ the outer axes only              (per-row broadcast, e.g. a row scale)
     <constant/none>→ nil

   Returns an axis-map, or nil when the shape is not recognized."
  [idx-expr t free-axes]
  (let [fa (vec free-axes)
        n-inner (second (peek fa))
        outer (vec (butlast fa))]
    (cond
      (= idx-expr t) (of-axes fa)
      (and (seq? idx-expr) (= 3 (count idx-expr))
           (= t (second idx-expr))
           (= n-inner (nth idx-expr 2)))
      (let [h (first idx-expr)]
        (cond (core-op? h 'mod 'rem) (of-axes [(peek fa)])
              (core-op? h 'quot)     (when (seq outer) (of-axes outer))
              :else nil))
      :else nil)))

;; ── packing: reinterpreting a buffer as wider words ─────────────────────────────────
(defn pack-innermost
  "Reinterpret the buffer this map indexes as words holding `factor` consecutive elements, e.g. a
   `char*` read as `int*` (4 int8 per int32, what dp4a consumes). The bytes are unchanged; only the
   INDEXING changes, and the change is entirely local to the map: the innermost axis's extent is
   divided by `factor` and the axis renamed to `new-sym` (which now counts words, not elements).

   Everything else follows from row-major arithmetic and needs no special case — each outer group's
   stride is expressed as a product that already contains the innermost extent, so dividing that one
   extent rescales every stride correctly. `(index-expr (pack-innermost m 4 'p))` is therefore the
   packed index, with no separate derivation.

   Returns nil when the innermost extent is not a literal multiple of `factor` — the caller must
   then refuse to tensorize rather than emit a mis-strided load."
  [amap factor new-sym]
  (let [groups (:groups amap)
        gi (dec (count groups))
        g (nth groups gi)
        ai (dec (count g))
        [_ e] (nth g ai)]
    (when (and (number? e) (zero? (mod (long e) (long factor))))
      {:groups (assoc groups gi (assoc g ai [new-sym (quot (long e) (long factor))]))})))

(defn innermost-axis
  "The last atomic axis in map order — the one with stride 1. A tensorize leaf that packs along the
   contraction must check that THIS is the axis it means to pack."
  [amap]
  (peek (axes amap)))

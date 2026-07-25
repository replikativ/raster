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
  (:refer-clojure :exclude [shape]))

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

(defn- canonicalize-ops
  "Normalize +/* heads (bare, clojure.core, raster.numeric) so index expressions from different
   producers compare structurally."
  [e]
  (cond (seq? e) (let [h (first e)
                       h' (get '{+ + clojure.core/+ + raster.numeric/+ +
                                 * * clojure.core/* * raster.numeric/* *} h h)]
                   (cons h' (map canonicalize-ops (rest e))))
        :else e))

(defn index-matches?
  "Does `idx-expr` equal this map's generated index, modulo operator qualification? This is the
   VERIFICATION step: a leaf may only assume an operand's layout if the operand's actual index
   expression provably is that layout."
  [amap idx-expr]
  (= (canonicalize-ops idx-expr) (canonicalize-ops (index-expr amap))))

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

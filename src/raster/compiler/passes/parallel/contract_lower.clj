(ns raster.compiler.passes.parallel.contract-lower
  "W1 step 2a: lower a `(raster.par/contract out free-axes contract-axes body …)` FORM
   to a segmented `SegRed`.

   segment-dims = the free (parallel/output) axes; reduced-dim = the (flattened) contracted axis
   (Futhark convention: the INNERMOST SegSpace dim is the reduced one). Because the
   free/contract split is DECLARED in the surface form, this lowering is deterministic —
   no index-expression recognition, no variance test. The naive segmented emitter and,
   later, the BlkRegTiling-style tiling pass consume this SegRed. A contraction is
   represented like a fused map→reduce: the product (map body) lives in the reduce
   combine's element slot `(+ acc <product>)`, so `map-lambda` is nil — matching how a
   `SoacReduce` fuses its map into the canonical reduction operator."
  (:require [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.par :as ir-par]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [clojure.set :as set]))

(defn flatten-contract-axes
  "Flatten n≥1 contract axes into ONE innermost reduced dim (the A0 convention, endorsed by
   Futhark's single-width Screma). Returns [k-sym k-bound body'] where k-sym is the single
   flat reduced index, k-bound its extent (product of the contract bounds), and body' the
   `body` with each original contract index li substituted by its row-major decomposition of
   the flat index: li = (kflat / prod(bounds after i)) mod bound_i. For a single contract axis
   this is the identity. The naive segmented emitter then loops the single k-sym unchanged."
  [contract-axes body]
  (contraction-facts/flatten-contract-axes contract-axes body))

(defn contract-form->segred
  "Parse `(raster.par/contract out [[i mi] …] [[k mk]] body & opts)` → a segmented SegRed.
   opts: :init (combine init, default 0.0), :combine (default +). :id/:dtype/:grid via
   kwargs. Supports one or more contracted axes. Pure + device-free (grid stays the passed value,
   default nil — the device-aware pass fills it)."
  [form & {:keys [id dtype grid facts] :or {id 0 dtype :double grid nil}}]
  (let [[_ out free-axes contract-axes] form
        facts (or facts (contraction-facts/contraction-facts form :dtype dtype))
        _ (when-not (and (contraction-facts/facts? facts)
                         (= form (:form facts))
                         (= dtype (:dtype facts)))
            (throw (ex-info "contract lowering requires facts derived from the same form"
                            {:reason :contraction-facts-mismatch
                             :form form :dtype dtype :facts facts})))
        ;; ZERO free axes is legal: the SegSpace then has only the reduced dim, i.e. exactly the
        ;; 1-D shape (segop/seg-space-1d?) that the full-reduction emitter consumes.
        _ (assert (vector? free-axes) "contract-lower: free-axes must be a vector")
        _ (assert (and (vector? contract-axes) (pos? (count contract-axes)))
                  "contract-lower: contract-form->segred needs ≥1 contract axis (0 → contract-form->segmap)")
        _ (assert (symbol? out) "contract-lower: out must be a symbol")
        ;; The canonical ProductReduction and flattened axis/body were derived together once.
        [k-sym k-bound] (:flat-contract-axis facts)
        body (:element (contraction-facts/scalar-reduction-view facts))
        free-dims (mapv (fn [[s b]] {:name s :bound b}) free-axes)
        red-dim   {:name k-sym :bound k-bound}
        ;; N-D space: free (segment) dims OUTER, contracted (reduced) dim INNERMOST.
        space (segop/make-seg-space-nd (conj free-dims red-dim))
        reduction (:reduction facts)
        arrays  (set (ir-par/collect-aget-arrays body))
        inputs  (disj arrays out)
        ;; scalars = the symbol-valued axis bounds (the contraction dims). Body-level
        ;; extra scalars (e.g. a scale) are a later refinement.
        bound-syms (into #{} (filter symbol?)
                         (conj (mapv second free-axes) k-bound))]
    (segop/->SegRed id space
                    (segop/->SegLevel :thread :virtual)
                    reduction
                    nil                 ; map-lambda: nil (product is in the combine)
                    inputs
                    #{out}
                    (set/difference bound-syms arrays #{out})
                    grid
                    :segmented
                    nil
                    dtype)))

(defn contract-form->segmap
  "Parse a 0-CONTRACT `(raster.par/contract out [[i mi] …] [] body)` form → a SegMap (a pure
   N-D map = outer product / broadcast / elementwise). The free axes ARE the map/output space
   (row-major, outer→inner); `body` is the map element. This is the empty-reduce projection of
   the same contraction node (Futhark: a Map is a Screma with an empty reduce list; Dex: an
   outer product is a `for` with no accumulator)."
  [form & {:keys [id dtype grid] :or {id 0 dtype :double grid nil}}]
  (let [[_ out free-axes contract-axes body] form
        _ (assert (and (vector? free-axes) (pos? (count free-axes)))
                  "contract-lower: free-axes must be a non-empty vector")
        _ (assert (empty? contract-axes)
                  "contract-form->segmap: expects ZERO contract axes (use contract-form->segred otherwise)")
        _ (assert (symbol? out) "contract-lower: out must be a symbol")
        free-dims (mapv (fn [[s b]] {:name s :bound b}) free-axes)
        space (segop/make-seg-space-nd free-dims)
        arrays (set (ir-par/collect-aget-arrays body))
        inputs (disj arrays out)
        bound-syms (into #{} (filter symbol?) (mapv second free-axes))]
    (segop/->SegMap id space
                    (segop/->SegLevel :thread :virtual)
                    body                ; map lambda = the element expression
                    inputs
                    #{out}
                    (set/difference bound-syms arrays #{out})
                    grid
                    dtype
                    out
                    nil)))

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
   `SoacReduce` fuses its map into the combine (see soac-lower/reduce-op-info)."
  (:require [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.par :as ir-par]
            [clojure.set :as set]
            [clojure.walk :as walk]))

(defn flatten-contract-axes
  "Flatten n≥1 contract axes into ONE innermost reduced dim (the A0 convention, endorsed by
   Futhark's single-width Screma). Returns [k-sym k-bound body'] where k-sym is the single
   flat reduced index, k-bound its extent (product of the contract bounds), and body' the
   `body` with each original contract index li substituted by its row-major decomposition of
   the flat index: li = (kflat / prod(bounds after i)) mod bound_i. For a single contract axis
   this is the identity. The naive segmented emitter then loops the single k-sym unchanged."
  [contract-axes body]
  (if (= 1 (count contract-axes))
    (let [[[k-sym k-bound]] contract-axes] [k-sym k-bound body])
    (let [flat-sym (gensym "kflat__")
          bounds   (mapv second contract-axes)
          k-bound  (if (every? number? bounds) (reduce * bounds)
                       (reduce (fn [a b] (list 'clojure.core/* a b)) bounds))
          suffix   (fn [p] (let [after (subvec bounds (inc p))]
                             (cond (empty? after) 1
                                   (every? number? after) (reduce * after)
                                   :else (reduce (fn [a b] (list 'clojure.core/* a b)) after))))
          subst    (into {} (map-indexed
                             (fn [p [s _]]
                               [s (list 'clojure.core/rem
                                        (list 'clojure.core/quot flat-sym (suffix p))
                                        (nth bounds p))])
                             contract-axes))]
      [flat-sym k-bound (walk/postwalk-replace subst body)])))

(defn contract-form->segred
  "Parse `(raster.par/contract out [[i mi] …] [[k mk]] body & opts)` → a segmented SegRed.
   opts: :init (combine init, default 0.0), :combine (default +). :id/:dtype/:grid via
   kwargs. Prototype: exactly one contracted axis. Pure + device-free (grid stays the
   passed value, default nil — the device-aware pass fills it)."
  [form & {:keys [id dtype grid] :or {id 0 dtype :double grid nil}}]
  (let [[_ out free-axes contract-axes body & opts] form
        opts-map (apply hash-map opts)
        init (get opts-map :init 0.0)
        combine (get opts-map :combine '+)
        _ (assert (and (vector? free-axes) (pos? (count free-axes)))
                  "contract-lower: free-axes must be a non-empty vector")
        _ (assert (and (vector? contract-axes) (pos? (count contract-axes)))
                  "contract-lower: contract-form->segred needs ≥1 contract axis (0 → contract-form->segmap)")
        _ (assert (symbol? out) "contract-lower: out must be a symbol")
        ;; n≥2 contract axes → flatten into one innermost reduced dim + substitute indices.
        [k-sym k-bound body] (flatten-contract-axes contract-axes body)
        free-dims (mapv (fn [[s b]] {:name s :bound b}) free-axes)
        red-dim   {:name k-sym :bound k-bound}
        ;; N-D space: free (segment) dims OUTER, contracted (reduced) dim INNERMOST.
        space (segop/make-seg-space-nd (conj free-dims red-dim))
        acc-sym (gensym "acc__")
        ;; fused map→reduce: product sits in the combine's element slot.
        reduce-op {:acc acc-sym :init init :lambda (list combine acc-sym body)}
        arrays  (set (ir-par/collect-aget-arrays body))
        inputs  (disj arrays out)
        ;; scalars = the symbol-valued axis bounds (the contraction dims). Body-level
        ;; extra scalars (e.g. a scale) are a later refinement.
        bound-syms (into #{} (filter symbol?)
                         (conj (mapv second free-axes) k-bound))]
    (segop/->SegRed id space
                    (segop/->SegLevel :thread :virtual)
                    reduce-op
                    nil                 ; map-lambda: nil (product is in the combine)
                    inputs
                    #{out}
                    (set/difference bound-syms arrays #{out})
                    grid
                    :segmented
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

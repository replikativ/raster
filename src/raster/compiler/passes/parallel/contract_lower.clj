(ns raster.compiler.passes.parallel.contract-lower
  "W1 step 2a: lower a `(raster.par/contract out free-axes contract-axes body …)` FORM
   to a segmented `SegRed`.

   segment-dims = the free (parallel/output) axes; reduced-dim = the contracted axis
   (Futhark convention: the INNERMOST SegSpace dim is the reduced one). Because the
   free/contract split is DECLARED in the surface form, this lowering is deterministic —
   no index-expression recognition, no variance test. The naive segmented emitter and,
   later, the BlkRegTiling-style tiling pass consume this SegRed. A contraction is
   represented like a fused map→reduce: the product (map body) lives in the reduce
   combine's element slot `(+ acc <product>)`, so `map-lambda` is nil — matching how a
   `SoacReduce` fuses its map into the combine (see soac-lower/reduce-op-info)."
  (:require [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.par :as ir-par]
            [clojure.set :as set]))

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
        _ (assert (and (vector? contract-axes) (= 1 (count contract-axes)))
                  "contract-lower: prototype supports exactly one contracted axis")
        _ (assert (symbol? out) "contract-lower: out must be a symbol")
        [k-sym k-bound] (first contract-axes)
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

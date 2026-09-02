(ns raster.par
  "Declarative parallel primitives for Raster.

  Provides parallel forms (map!, reduce) that the compiler pipeline
  can optimize to SIMD (CPU) or CUDA (GPU) code. Forms are preserved
  through the walker and pipeline, then either:
    1. Compiled to SIMD/CUDA by backend passes
    2. Expanded to sequential loops as fallback

  The parallel forms are structural compiler primitives recognized
  by the walker. They are also macros so that deftm bodies compile
  correctly at runtime — the macro expansion provides the sequential
  fallback, while the pipeline sees the pre-expansion S-expression.

  Common array operations are provided as deftm wrappers that use
  broadcast/reduce! internally and participate in typed dispatch."
  (:refer-clojure :exclude [aget aset alength aclone reduce map pmap])
  (:require [clojure.walk :as walk]
            [raster.core :refer [deftm broadcast reduce!]] ;; broadcast/reduce! are typed-macro stubs; scan defined locally
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.numeric :as n]
            [raster.compiler.ir.par :as ir.par]))

;; Splitmix64 constants — canonical definitions in ir.par, aliased here for macros
(def SM-GAMMA ir.par/SM-GAMMA)
(def SM-MIX1  ir.par/SM-MIX1)
(def SM-MIX2  ir.par/SM-MIX2)

;; ================================================================
;; Runtime macros (fallback expansion when eval'd)
;; ================================================================

(defmacro ^:no-doc map!
  "Internal: imperative parallel map into pre-allocated buffer.
  Use par/map (the pure functional form) instead in user code.

  Form: (raster.par/map! out idx bound cast body-value-expr)
  Semantics: for idx in 0..bound: out[idx] = cast(body-value-expr)

  Offset form: (raster.par/map! out idx bound :offset base cast body-value-expr)
  Semantics: for idx in 0..bound: out[base+idx] = cast(body-value-expr)"
  ([out-sym idx-sym bound-expr cast-fn body-expr]
   (let [n-sym (gensym "n__")
         aset-expr (if cast-fn
                     `(~cast-fn ~body-expr)
                     body-expr)]
     `(let [~n-sym (int ~bound-expr)]
        (dotimes [~idx-sym ~n-sym]
          (clojure.core/aset ~out-sym ~idx-sym ~aset-expr))
        ~out-sym)))
  ([out-sym idx-sym bound-expr _offset base-expr cast-fn body-expr]
   (let [n-sym (gensym "n__")
         base-sym (gensym "base__")
         aset-expr (if cast-fn
                     `(~cast-fn ~body-expr)
                     body-expr)]
     `(let [~n-sym (int ~bound-expr)
            ~base-sym (int ~base-expr)]
        (dotimes [~idx-sym ~n-sym]
          (clojure.core/aset ~out-sym (clojure.core/unchecked-add-int ~base-sym ~idx-sym) ~aset-expr))
        ~out-sym))))

(defmacro map
  "Functional index-range map. Returns a NEW array of length n.
  Form: (raster.par/map [i n] body)
  Semantics: allocate out[n]; for i in 0..n: out[i] = (double body); return out

  Inside deftm, the output element type is inferred from array-typed
  symbols in the body (like Futhark's type inference). Falls back to double[].
  Optional :like ref overrides inference with a specific reference array."
  [[i-sym bound-expr] & args]
  (let [;; Parse optional :like keyword
        [like-ref body] (if (= :like (first args))
                          [(second args) (nth args 2)]
                          [nil (first args)])
        n-sym (gensym "n__")
        out-sym (gensym "out__")]
    ;; Runtime fallback: allocate double[] (or same type as :like ref)
    (if like-ref
      `(let [~n-sym (int ~bound-expr)
             ~out-sym (raster.arrays/alloc-like ~like-ref ~n-sym)]
         (dotimes [~i-sym ~n-sym]
           (clojure.core/aset ~out-sym ~i-sym (double ~body)))
         ~out-sym)
      `(let [~n-sym (int ~bound-expr)
             ~out-sym (clojure.core/double-array ~n-sym)]
         (dotimes [~i-sym ~n-sym]
           (clojure.core/aset ~out-sym ~i-sym (double ~body)))
         ~out-sym))))

(defmacro pmap
  "Pure map IR form (compiler internal). Runtime fallback: allocate + fill.
  IR form: (raster.par/pmap idx bound cast body)
  Used by the walker as the pure equivalent of par/map!. The materialize pass
  converts this to alloc + par/map! before backend compilation."
  [idx-sym bound-expr cast-fn body-expr]
  (let [n-sym (gensym "n__")
        out-sym (gensym "out__")
        alloc-fn (case cast-fn
                   float  'clojure.core/float-array
                   double 'clojure.core/double-array
                   long   'clojure.core/long-array
                   int    'clojure.core/int-array
                   'clojure.core/double-array)
        aset-expr (if cast-fn
                    `(~cast-fn ~body-expr)
                    `(double ~body-expr))]
    `(let [~n-sym (int ~bound-expr)
           ~out-sym (~alloc-fn ~n-sym)]
       (dotimes [~idx-sym ~n-sym]
         (clojure.core/aset ~out-sym ~idx-sym ~aset-expr))
       ~out-sym)))

(defmacro map2!
  "Parallel paired map — writes to two output arrays per index.
  Form: (par/map2! out1 out2 idx bound cast body1 body2)
  Semantics: for idx in 0..bound: out1[idx]=cast(body1), out2[idx]=cast(body2)"
  [out1-sym out2-sym idx-sym bound-expr cast-fn body1 body2]
  (let [n-sym (gensym "n__")
        aset-expr1 (if cast-fn `(~cast-fn ~body1) body1)
        aset-expr2 (if cast-fn `(~cast-fn ~body2) body2)]
    `(let [~n-sym (int ~bound-expr)]
       (dotimes [~idx-sym ~n-sym]
         (clojure.core/aset ~out1-sym ~idx-sym ~aset-expr1)
         (clojure.core/aset ~out2-sym ~idx-sym ~aset-expr2)))))

(defmacro reduce
  "Parallel reduction. When compiled (eval'd), expands to a plain
  sequential loop. The compiler pipeline's backend pass uses
  int-counted loop* for optimization.

  Form: (raster.par/reduce acc init idx bound body-expr)
  Semantics: acc = init; for idx in 0..bound: acc = body; return acc"
  [acc-sym init-expr idx-sym bound-expr body-expr]
  (let [n-sym (gensym "n__")]
    `(let [~n-sym (int ~bound-expr)]
       (loop [~idx-sym 0 ~acc-sym ~init-expr]
         (if (< ~idx-sym ~n-sym)
           (recur (inc ~idx-sym) ~body-expr)
           ~acc-sym)))))

(defmacro product-reduce!
  "Typed segmented product reduction.

  Form:
    (product-reduce! outputs
      [[acc neutral dtype] ...]
      [[segment-index segment-bound] ...]
      reduce-index reduce-bound
      [element-local value ...]
      [element-value ...]
      [[left right] ...]
      [combine-local value ...]
      [combined-value ...]
      algebra)

  `outputs` is ordered with the components; nil discards a final component.  `algebra` records
  semantic facts for scheduling and is not a runtime argument.  The interpreted fallback is a
  deterministic sequential fold per segment; accelerator lowering may choose a parallel tree.

  Example: argmax reduces `[value,index]`, discards the winning value, and writes the index."
  [outputs components segment-axes idx-sym bound-expr
   element-bindings element-results combine-parameters combine-bindings combine-results algebra]
  (assert (vector? outputs) "product-reduce!: outputs must be a vector")
  (assert (and (vector? components) (seq components)
               (every? #(and (vector? %) (= 3 (count %))) components))
          "product-reduce!: components must be [[acc neutral dtype] ...]")
  (assert (= (count outputs) (count components) (count element-results)
             (count combine-parameters) (count combine-results))
          "product-reduce!: outputs, components, element and combine results must have equal arity")
  (assert (and (vector? segment-axes)
               (every? #(and (vector? %) (= 2 (count %))) segment-axes))
          "product-reduce!: segment axes must be [[index bound] ...]")
  (assert (and (vector? element-bindings) (even? (count element-bindings)))
          "product-reduce!: element bindings must be a flat binding vector")
  (assert (and (vector? combine-bindings) (even? (count combine-bindings)))
          "product-reduce!: combine bindings must be a flat binding vector")
  (assert (every? #(and (vector? %) (= 2 (count %))) combine-parameters)
          "product-reduce!: combine parameters must be [[left right] ...]")
  (assert (map? algebra) "product-reduce!: algebra must be a map")
  (let [accs (mapv first components)
        neutrals (mapv second components)
        n-sym (gensym "n__")
        flat-index (clojure.core/reduce (fn [flat [segment bound]]
                                          `(+ (* ~flat (int ~bound)) ~segment))
                                        0 segment-axes)
        stores (keep (fn [[out acc]]
                       (when out `(clojure.core/aset ~out ~flat-index ~acc)))
                     (clojure.core/map vector outputs accs))
        fold `(let [~n-sym (long ~bound-expr)]
                (loop [~idx-sym (long 0)
                       ~@(mapcat vector accs neutrals)]
                  (if (< ~idx-sym ~n-sym)
                    (let [~@element-bindings
                          ~@(mapcat (fn [[[left right] acc element]]
                                      [left acc right element])
                                    (clojure.core/map vector combine-parameters accs element-results))
                          ~@combine-bindings]
                      (recur (unchecked-inc ~idx-sym) ~@combine-results))
                    (do ~@stores nil))))
        segmented (clojure.core/reduce (fn [body [segment bound]]
                                         `(dotimes [~segment (int ~bound)] ~body))
                                       fold (reverse segment-axes))]
    `(do ~segmented nil)))

(defmacro segmented-fold-map!
  "Ordered folds followed by a dense map, independently for every segment.

  Form:
    (segmented-fold-map! outputs
      [[segment-index segment-bound] ...]
      index map-extent
      [[accumulator identity dtype fold-extent step] ...]
      [map-result ...])

  Fold `n` may reference the completed results of folds `0..n-1`. `:element` denotes the
  surrounding tensor element dtype; the typed frontend resolves it before scheduling.  The
  interpreted fallback preserves the declared fold order exactly. Accelerator schedules may
  parallelize segments, but must not reassociate an ordered fold.

  The result buffers are dense row-major tensors with shape
  `[segment-bound ... map-extent]`. This is a general row-statistics primitive: softmax,
  normalization and ragged scientific reductions are applications, not compiler operations."
  [outputs segment-axes idx-sym map-extent folds map-results]
  (assert (and (vector? outputs) (seq outputs) (every? symbol? outputs))
          "segmented-fold-map!: outputs must be a non-empty symbol vector")
  (assert (and (vector? segment-axes) (seq segment-axes)
               (every? #(and (vector? %) (= 2 (count %)) (symbol? (first %)))
                       segment-axes))
          "segmented-fold-map!: segment axes must be [[index bound] ...]")
  (assert (symbol? idx-sym) "segmented-fold-map!: map index must be a symbol")
  (assert (and (vector? folds) (seq folds)
               (every? #(and (vector? %) (= 5 (count %)) (symbol? (first %))) folds))
          "segmented-fold-map!: folds must be [[acc identity dtype extent step] ...]")
  (assert (= (count outputs) (count map-results))
          "segmented-fold-map!: outputs and map results must have equal arity")
  (let [flat-index
        (clojure.core/reduce
         (fn [flat [index bound]] `(+ (* ~flat (long ~bound)) ~index))
         0 (conj (vec segment-axes) [idx-sym map-extent]))
        cast-result
        (fn [dtype value]
          (if-let [cast (get {:float `float :double `double :int `int :long `long
                              :byte `byte}
                             dtype)]
            `(~cast ~value)
            value))
        stores (mapv (fn [output result]
                       `(clojure.core/aset ~output ~flat-index ~result))
                     outputs map-results)
        mapped `(dotimes [~idx-sym (int ~map-extent)] ~@stores)
        folded
        (clojure.core/reduce
         (fn [body [acc identity dtype fold-extent step]]
           (let [fold-index (gensym "fold_index__")
                 folded-value (gensym "fold_value__")]
             `(let [~folded-value
                    (loop [~fold-index (long 0) ~acc ~identity]
                      (if (< ~fold-index (long ~fold-extent))
                        (recur (unchecked-inc ~fold-index)
                               ~(cast-result dtype
                                             (walk/postwalk-replace {idx-sym fold-index} step)))
                        ~acc))]
                (let [~acc ~folded-value] ~body))))
         mapped (reverse folds))
        segmented (clojure.core/reduce
                   (fn [body [segment bound]]
                     `(dotimes [~segment (int ~bound)] ~body))
                   folded (reverse segment-axes))]
    `(do ~segmented nil)))

(defmacro contract
  "Explicit tensor contraction (SOAC). Declares FREE (parallel/output) axes and
  CONTRACTED (reduced) axes separately — the reliable, unambiguous scheduling signal
  that a typed segmented reduction carries as parallel-vs-reduction structure (vs a nested
  map+reduce where the reduce hides inside the map-lambda and must be recognized).

  Form: (raster.par/contract out [[i mi] [j mj] ...] [[k mk]] body)
    out          — output array; length = product of free-axis bounds, ROW-MAJOR over
                   the free axes in declared order (outer→inner)
    free-axes    — [[idx-sym bound] ...] the parallel output axes (≥1)
    free-axes    — [[idx-sym bound] ...] the parallel/output axes. ZERO free axes is legal and
                   means a FULL REDUCTION to a scalar, written to out[0].
    contract-axes— [[idx-sym bound] ...] the reduced axes: 0 (outer product / pure map),
                   1, or n (n≥2 are flattened into one innermost reduced dim)
    body         — the summand expression; may reference every free and contract idx
    opts         — :init (accumulator init, default 0.0), :combine (default +)

  Semantics: for each free index tuple f, out[flat(f)] = combine-fold over the
  contracted axis of body. The COMPILER recognizes the `raster.par/contract` form
  directly (free/contract axes are explicit → no index-expr matching), lifting it to a
  segmented reduction the tiling pass consumes. This macro is the interpreted runtime
  fallback; it is fully compositional because it enters a segmented TypedSOAC reduction, so its
  output fuses with downstream SOACs (epilogue) like any reduction.

  Example (C[m,n] = A[m,k]·B[k,n], row-major):
    (raster.par/contract C [[i m] [j n]] [[l k]]
      (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))"
  [out free-axes contract-axes body & {:keys [init combine stages] :or {init 0.0 combine '+}}]
  (assert (vector? free-axes) "par/contract: free-axes must be a vector of [idx bound]")
  (assert (vector? contract-axes)
          "par/contract: contract-axes must be a vector (0 → outer product/map; n≥1 → contraction, n≥2 flattened)")
  (let [;; A STAGED contract axis (:stages) is a SCHEDULE: the multi-level accumulate is equal, in
        ;; exact arithmetic, to the flat contraction whose body absorbed the lift factors, so the
        ;; interpreted expansion runs THAT and needs no staged implementation. Ignoring :stages
        ;; here would silently drop the lift factors (e.g. every per-block quant scale) — a wrong
        ;; answer, not a slower one. See compiler/ir/contract_stages.clj; the GPU leaf that
        ;; actually schedules the levels is segop-opencl/generate-staged-contraction-kernel.
        _ (when (seq stages)
            (let [legal ((requiring-resolve 'raster.compiler.ir.contract-stages/stages-legal?)
                         stages contract-axes)]
              (assert (:ok legal) (str "par/contract: illegal :stages (" (:reason legal) ")"))))
        body (if (seq stages)
               ((requiring-resolve 'raster.compiler.ir.contract-stages/flat-equivalent) stages body)
               body)
        free-syms   (mapv first free-axes)
        int-bounds  (mapv (fn [[_ b]] `(int ~b)) free-axes)
        f-sym   (gensym "f__")
        out-sym (gensym "out__")
        F-sym   (gensym "F__")
        acc-sym (gensym "acc__")
        nfree   (count free-axes)
        ;; Product over the free axes. ZERO free axes is the empty product = 1: the output is a
        ;; single element and the generic expansion degenerates to one iteration writing out[0] —
        ;; i.e. a FULL REDUCTION, the (0 free, n contract) cell of contract's algebra.
        F-expr  (if (empty? int-bounds)
                  1
                  (clojure.core/reduce (fn [a b] `(clojure.core/* ~a ~b)) int-bounds))
        ;; row-major decompose flat f into each free index:
        ;;   idx_p = (f / (product of bounds after p)) mod bound_p
        suffix  (fn [p] (let [after (subvec int-bounds (inc p))]
                          (if (empty? after) 1
                              (clojure.core/reduce (fn [a b] `(clojure.core/* ~a ~b)) after))))
        decomp  (vec (mapcat (fn [p sym]
                               [sym `(clojure.core/rem
                                      (clojure.core/quot ~f-sym ~(suffix p))
                                      ~(nth int-bounds p))])
                             (range nfree) free-syms))]
    (if (empty? contract-axes)
      ;; 0 contract axes → pure map (outer product / broadcast): out[f] = body.
      `(let [~out-sym ~out ~F-sym ~F-expr]
         (dotimes [~f-sym ~F-sym]
           (let [~@decomp]
             (clojure.core/aset ~out-sym ~f-sym ~body)))
         ~out-sym)
      ;; n≥1 contract axes → contraction: reduce `body` over the (flattened) contracted axis.
      ;; flatten-contract-axes collapses n≥2 axes into one flat index k-sym and substitutes
      ;; each original contract index in the body (one source of truth with the compiler path).
      (let [[k-sym k-bound sbody]
            ((requiring-resolve 'raster.compiler.ir.contraction-facts/flatten-contract-axes)
             contract-axes body)]
        `(let [~out-sym ~out ~F-sym ~F-expr]
           (dotimes [~f-sym ~F-sym]
             (let [~@decomp]
               (clojure.core/aset ~out-sym ~f-sym
                                  (loop [~k-sym 0 ~acc-sym ~init]
                                    (if (clojure.core/< ~k-sym (int ~k-bound))
                                      (recur (clojure.core/inc ~k-sym) (~combine ~acc-sym ~sbody))
                                      ~acc-sym)))))
           ~out-sym)))))

(defmacro scan
  "Parallel prefix scan (inclusive). When compiled (eval'd), expands to
  a plain sequential loop. The compiler pipeline can optimize the
  pre-expansion S-expression form to SIMD/CUDA.

  Form: (raster.par/scan out acc init idx bound cast body-expr)
  Semantics: acc = init; for idx in 0..bound: acc = body; out[idx] = acc; return out"
  [out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr]
  (let [n-sym (gensym "n__")
        store-expr (if cast-fn
                     `(~cast-fn ~acc-sym)
                     acc-sym)]
    `(let [~n-sym (int ~bound-expr)]
       (loop [~idx-sym 0 ~acc-sym ~init-expr]
         (if (< ~idx-sym ~n-sym)
           (let [~acc-sym ~body-expr]
             (clojure.core/aset ~out-sym ~idx-sym ~store-expr)
             (recur (inc ~idx-sym) ~acc-sym))
           ~out-sym)))))

(defmacro scan-exclusive
  "Parallel exclusive prefix scan. When compiled (eval'd), expands to
  a plain sequential loop. The compiler pipeline can optimize the
  pre-expansion S-expression form to SIMD/CUDA/OpenCL.

  Form: (raster.par/scan-exclusive out acc init idx bound cast body-expr)
  Semantics:
    out[0] = init
    acc = init
    for idx in 0..bound-1:
      acc = body
      out[idx+1] = cast(acc)
    return out

  Output array has bound+1 elements. Maps to GPU exclusive_scan."
  [out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr]
  (let [n-sym (gensym "n__")
        store-expr (if cast-fn
                     `(~cast-fn ~acc-sym)
                     acc-sym)]
    `(let [~n-sym (int ~bound-expr)]
       (clojure.core/aset ~out-sym 0 ~(if cast-fn `(~cast-fn ~init-expr) init-expr))
       (loop [~idx-sym 0 ~acc-sym ~init-expr]
         (if (< ~idx-sym ~n-sym)
           (let [~acc-sym ~body-expr]
             (clojure.core/aset ~out-sym (inc ~idx-sym) ~store-expr)
             (recur (inc ~idx-sym) ~acc-sym))
           ~out-sym)))))

;; ================================================================
;; Scatter macro (runtime fallback)
;; ================================================================

(defmacro scatter!
  "Parallel scatter-add. When compiled (eval'd), expands to a sequential
  loop. The compiler pipeline can optimize the pre-expansion S-expression
  form to CUDA (using atomicAdd).

  Form: (raster.par/scatter! output src index n)
  Semantics: for i in 0..n: output[index[i]] += src[i]

  Strided form: (raster.par/scatter! output src index n stride)
  Semantics: for i in 0..n, d in 0..stride:
    output[index[i]*stride + d] += src[i*stride + d]"
  ([output src index n]
   `(let [n# (int ~n)]
      (dotimes [i# n#]
        (let [idx# (clojure.core/aget ~index i#)]
          (clojure.core/aset ~output idx#
                             (+ (clojure.core/aget ~output idx#)
                                (clojure.core/aget ~src i#)))))
      ~output))
  ([output src index n stride]
   `(let [n# (int ~n)
          stride# (int ~stride)]
      (dotimes [i# n#]
        (let [idx# (clojure.core/aget ~index i#)]
          (dotimes [d# stride#]
            (let [src-pos# (+ (* i# stride#) d#)
                  dst-pos# (+ (* idx# stride#) d#)]
              (clojure.core/aset ~output dst-pos#
                                 (+ (clojure.core/aget ~output dst-pos#)
                                    (clojure.core/aget ~src src-pos#)))))))
      ~output)))

(defmacro gather
  "Parallel gather: out[i] = src[index[i]]. The SIMD backend emits a hardware
  vector gather (DoubleVector.fromArray with an index map); otherwise expands
  to a sequential loop.

  Form: (raster.par/gather out src index n)
  Semantics: for i in 0..n: output[i] = src[index[i]]

  Strided form: (raster.par/gather out src index n stride)
  Semantics: for i in 0..n, d in 0..stride: out[i*stride+d] = src[index[i]*stride+d]"
  ([output src index n]
   `(let [n# (int ~n)]
      (dotimes [i# n#]
        (clojure.core/aset ~output i#
                           (clojure.core/aget ~src (clojure.core/aget ~index i#))))
      ~output))
  ([output src index n stride]
   `(let [n# (int ~n)
          stride# (int ~stride)]
      (dotimes [i# n#]
        (let [sbase# (* (clojure.core/aget ~index i#) stride#)
              obase# (* i# stride#)]
          (dotimes [d# stride#]
            (clojure.core/aset ~output (+ obase# d#)
                               (clojure.core/aget ~src (+ sbase# d#))))))
      ~output)))

;; ================================================================
;; Reduce-by-key macro (runtime fallback)
;; ================================================================

(defmacro reduce-by-key
  "Parallel segmented reduction by key. When compiled (eval'd), expands to
  a sequential loop. The compiler pipeline compiles to GPU atomics.

  Form: (raster.par/reduce-by-key output keys vals n op)
  Semantics: for i in 0..n: output[keys[i]] op= vals[i]

  op is one of: + (default). Only additive reduction supported on GPU."
  [output keys vals n op]
  `(let [n# (int ~n)]
     (dotimes [i# n#]
       (let [k# (clojure.core/aget ~keys i#)
             v# (clojure.core/aget ~vals i#)]
         (clojure.core/aset ~output k#
                            (+ (clojure.core/aget ~output k#) v#))))
     ~output))

;; ================================================================
;; Stencil macro (runtime fallback)
;; ================================================================

(defmacro map-void!
  "Parallel side-effect-only map. No output array — body is executed for
  side effects (aset, atomic-add!, etc.). Expands to dotimes on CPU.

  Form: (raster.par/map-void! idx bound body)
  Semantics: for idx in 0..bound: body (for side effects)"
  [idx-sym bound-expr body-expr]
  (let [n-sym (gensym "n__")]
    `(let [~n-sym (int ~bound-expr)]
       (dotimes [~idx-sym ~n-sym]
         ~body-expr)
       nil)))

(deftm ^:no-inline unique-index
  "Assert that an indexed-write destination is unique over its enclosing parallel iteration.

   Runtime semantics are identity. The typed compiler consumes this marker as a conflict contract;
   callers must establish uniqueness before dispatch (for example with validate-block-indices!)."
  [index :- Long] :- Long
  index)

(defmacro collect!
  "Atomically claim a slot in count-arr and write values to SoA arrays.
  Useful for building output queues in parallel kernels.

  Form: (raster.par/collect! count-arr arr1 val1 arr2 val2 ...)
  Semantics:
    slot = atomic_add(count-arr, 0, 1)
    arr1[slot] = val1
    arr2[slot] = val2
    ..."
  [count-arr & pairs]
  (assert (even? (count pairs)) "par/collect! requires even number of array/value pairs")
  (let [slot-sym (gensym "slot__")]
    `(let [~slot-sym (int (atomic-add! ~count-arr 0 (int 1)))]
       ~@(clojure.core/map (fn [[arr val]] `(clojure.core/aset ~arr ~slot-sym ~val))
                           (partition 2 pairs)))))

(defmacro atomic-add!
  "Atomic add to array element. Returns old value.
  Sequential fallback: plain read-modify-write (correct for single-threaded).
  On GPU: emits OpenCL atomic_add (int) or CAS loop (float).

  Form: (raster.par/atomic-add! arr idx val)
  Semantics: old = arr[idx]; arr[idx] += val; return old"
  [arr idx val]
  (let [tag (:tag (meta arr))]
    (if (contains? #{'floats 'float} tag)
      ;; Float array path
      `(let [a# ~arr
             i# (int ~idx)
             old# (clojure.core/aget a# i#)]
         (clojure.core/aset a# i# (float (+ (float old#) (float ~val))))
         old#)
      ;; Int array path (default)
      `(let [a# ~arr
             i# (int ~idx)
             old# (clojure.core/aget a# i#)]
         (clojure.core/aset a# i# (unchecked-add-int old# (int ~val)))
         old#))))

(deftm ^:no-inline dp4a
  "4-way int8 dot-accumulate. `a` and `b` each pack four signed int8 lanes into an int32,
  little-endian (lane 0 = low byte). Returns `acc + Σ_i lane_i(a)·lane_i(b)`.

  This is the JVM/reference impl of the `rstr_dp4a` primitive: in a deftm/par body the
  GPU and CPU-C backends lower a `(par/dp4a a b acc)` call to `rstr_dp4a(a, b, acc)` (a
  portable scalar helper the OpenCL/C compiler pattern-matches to a hardware dp4a:
  Intel/CUDA `__dp4a`, AMD `sdot4`). The masked extraction below is sign-correct
  regardless of how the int sign-extends into the JVM long.

  A `deftm`, not a `defn`, so the walker stamps its Integer result type. The target primitive and
  packed operands are int32; the JVM reference explicitly wraps to the same representation instead
  of exposing an accidental 64-bit result. As a plain defn it had no
  declared return type, every let-bound `(par/dp4a …)` intermediate reached the GPU fixpoint
  edge untagged, and the typedness census (correctly) refused the program — `bind-decode!` on
  gemma-3-270m failed with `14 non-exempt untagged binding(s) {raster.par/dp4a 14}` from the
  d1/d2 pairs in `quant/kernels-k`'s Q4_K kernel. A loop-accumulator use slipped past only
  because loop BINDERS are census-exempt. The backends match this op by its symbol through the
  intrinsics registry (`:dp4a` → `rstr_dp4a`), which the `:raster.op/original` stamp preserves,
  so emission is unchanged. Do not exempt dp4a from the census instead: the census exists to
  catch exactly the narrowing class an untyped integer accumulate can hide."
  [a :- Long, b :- Long, acc :- Long] :- Integer
  (let [a (long (unchecked-int a)) b (long (unchecked-int b))
        sx (fn ^long [^long x ^long sh]
             (let [v (bit-and (unsigned-bit-shift-right x sh) 0xFF)]
               (if (>= v 128) (- v 256) v)))]
    (unchecked-int
     (+ (long acc)
        (* (sx a 0) (sx b 0))
        (* (sx a 8) (sx b 8))
        (* (sx a 16) (sx b 16))
        (* (sx a 24) (sx b 24))))))

(defmacro rng-fill!
  "Fill a long array with splitmix64 pseudo-random seeds.
  Each element i gets: splitmix64(base-seed + i * 0x9e3779b97f4a7c15).
  CPU fallback: sequential loop. GPU: compiled to parallel per-element kernel.

  Form: (raster.par/rng-fill! seeds-arr n base-seed)
  Semantics: for i in 0..n: seeds[i] = splitmix64(base_seed + i * golden_ratio)"
  [seeds-arr n base-seed]
  `(let [n# (int ~n)
         base# (long ~base-seed)]
     (dotimes [i# n#]
       (let [state# (unchecked-add base# (unchecked-multiply (long i#) SM-GAMMA))
             s1# (bit-xor state# (unsigned-bit-shift-right state# 30))
             s2# (unchecked-multiply s1# SM-MIX1)
             s3# (bit-xor s2# (unsigned-bit-shift-right s2# 27))
             s4# (unchecked-multiply s3# SM-MIX2)
             s5# (bit-xor s4# (unsigned-bit-shift-right s4# 31))]
         (clojure.core/aset ~seeds-arr i# s5#)))
     ~seeds-arr))

(defmacro active-ids!
  "Fill an int array with random agent indices in [0, n-agents).
  Each element i gets: int(splitmix64(base-seed + i * golden_ratio) mod n-agents).
  CPU fallback: sequential loop. GPU: compiled to parallel per-element kernel.

  Form: (raster.par/active-ids! ids-arr n-active n-agents base-seed)
  Semantics: for i in 0..n-active: ids[i] = splitmix64_mod(base_seed + i * golden_ratio, n-agents)"
  [ids-arr n-active n-agents base-seed]
  `(let [n-active# (int ~n-active)
         n-agents# (long ~n-agents)
         base# (long ~base-seed)]
     (dotimes [i# n-active#]
       (let [state# (unchecked-add base# (unchecked-multiply (long i#) SM-GAMMA))
             s1# (bit-xor state# (unsigned-bit-shift-right state# 30))
             s2# (unchecked-multiply s1# SM-MIX1)
             s3# (bit-xor s2# (unsigned-bit-shift-right s2# 27))
             s4# (unchecked-multiply s3# SM-MIX2)
             s5# (bit-xor s4# (unsigned-bit-shift-right s4# 31))
             idx# (int (mod (bit-and s5# (long 0x7FFFFFFFFFFFFFFF)) n-agents#))]
         (clojure.core/aset ~ids-arr i# idx#)))
     ~ids-arr))

(defmacro butterfly!
  "Paired stride transform — reads from (base+k) and (base+k+half) in two arrays,
   applies transform, writes results back to both positions.
   Covers FFT butterfly, Hadamard, bitonic sort, NTT.

   Form: (par/butterfly! re im idx half wr wi base)
   Semantics: for idx in 0..half:
     ur = re[base+idx], ui = im[base+idx]
     vr = wr[idx]*re[base+idx+half] - wi[idx]*im[base+idx+half]
     vi = wr[idx]*im[base+idx+half] + wi[idx]*re[base+idx+half]
     re[base+idx] = ur + vr, re[base+idx+half] = ur - vr
     im[base+idx] = ui + vi, im[base+idx+half] = ui - vi"
  [re im idx-sym half-expr wr wi base-expr]
  (let [n-sym (gensym "n__")
        base-sym (gensym "base__")]
    `(let [~n-sym (int ~half-expr)
           ~base-sym (int ~base-expr)]
       (dotimes [~idx-sym ~n-sym]
         (let [lo# (int (+ ~base-sym ~idx-sym))
               hi# (int (+ lo# ~n-sym))
               ur# (clojure.core/aget ~re lo#)
               ui# (clojure.core/aget ~im lo#)
               rhi# (clojure.core/aget ~re hi#)
               ihi# (clojure.core/aget ~im hi#)
               wr# (clojure.core/aget ~wr ~idx-sym)
               wi# (clojure.core/aget ~wi ~idx-sym)
               vr# (- (* wr# rhi#) (* wi# ihi#))
               vi# (+ (* wr# ihi#) (* wi# rhi#))]
           (clojure.core/aset ~re lo# (+ ur# vr#))
           (clojure.core/aset ~re hi# (- ur# vr#))
           (clojure.core/aset ~im lo# (+ ui# vi#))
           (clojure.core/aset ~im hi# (- ui# vi#))))
       nil)))

(defmacro stencil!
  "Parallel stencil operation. When compiled (eval'd), expands to a
  plain dotimes loop. The compiler pipeline can optimize the
  pre-expansion S-expression form to SIMD/CUDA.

  Form: (raster.par/stencil! out [in-arrays] radius boundary cast idx-sym bound body)
  Semantics: zero boundary elements, loop i in [radius, bound-radius): out[i] = body"
  [out-sym in-arrays radius boundary cast-fn idx-sym bound-expr body-expr]
  (let [n-sym (gensym "n__")
        j-sym (gensym "j__")]
    `(let [~n-sym (int ~bound-expr)]
       ;; Zero boundary elements (Dirichlet)
       ~@(when (= boundary :dirichlet)
           [`(dotimes [~j-sym ~radius]
               (clojure.core/aset ~out-sym ~j-sym 0.0)
               (clojure.core/aset ~out-sym (- ~n-sym 1 ~j-sym) 0.0))])
       ;; Interior loop
       (dotimes [~j-sym (- ~n-sym (* 2 ~radius))]
         (let [~idx-sym (int (+ ~j-sym ~radius))]
           (clojure.core/aset ~out-sym ~idx-sym
                              ~(if cast-fn
                                 `(~cast-fn ~body-expr)
                                 body-expr))))
       ~out-sym)))

;; ================================================================
;; Common array operations as deftm — parametric over element type.
;; Element-wise ops preserve the array type.
;; Reductions accumulate in Double for numerical stability.
;; ================================================================

;; Element-wise operations: (All [T]) — body uses broadcast which is type-polymorphic
(deftm axpy
  "Compute y + alpha*x element-wise (BLAS axpy)."
  (All [T] [alpha :- T,
            x :- (Array T), y :- (Array T)] :- (Array T)
       (broadcast [x y] (+ y (* alpha x)))))

(deftm scale
  "Scale array x by scalar alpha element-wise."
  (All [T] [alpha :- T,
            x :- (Array T)] :- (Array T)
       (broadcast [x] (* alpha x))))

(deftm fill
  "Fill every element of out with val."
  (All [T] [out :- (Array T), val :- T] :- (Array T)
       (dotimes [i (alength out)]
         (aset out i val))
       out))

;; Scalar reductions: accumulate in Double for stability
(deftm dot-product
  "Compute the dot product of arrays a and b."
  (All [T] [a :- (Array T), b :- (Array T)] :- Double
       (reduce! [acc 0.0] [a b] (+ acc (* a b)))))

(deftm sum
  "Sum all elements of array a."
  (All [T] [a :- (Array T)] :- Double
       (reduce! [acc 0.0] [a] (+ acc a))))

(deftm amax
  "Return the maximum element of array a."
  (All [T] [a :- (Array T)] :- Double
       (reduce! [acc ##-Inf] [a] (Math/max acc (double a)))))

(deftm amin
  "Return the minimum element of array a."
  (All [T] [a :- (Array T)] :- Double
       (reduce! [acc ##Inf] [a] (Math/min acc (double a)))))

(deftm norm
  "Compute the L2 norm of array a."
  (All [T] [a :- (Array T)] :- Double
       (Math/sqrt (reduce! [acc 0.0] [a] (+ acc (* a a))))))

;; ================================================================
;; Scan-based operations (prefix sum, cumulative product)
;; These accumulate in element type (scan output = same type as input).
;; ================================================================

(deftm cumsum
  "Inclusive prefix sum (cumulative sum) of array a."
  [a :- (Array double)] :- (Array double)
  (scan [acc 0.0] [a] (+ acc a)))

(deftm cumprod
  "Inclusive prefix product (cumulative product) of array a."
  [a :- (Array double)] :- (Array double)
  (scan [acc 1.0] [a] (* acc a)))

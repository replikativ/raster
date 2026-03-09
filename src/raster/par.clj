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
  (:require [raster.core :refer [deftm broadcast reduce!]] ;; broadcast/reduce! are typed-macro stubs; scan defined locally
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
  (All [T] [alpha :- Double,
            x :- (Array T), y :- (Array T)] :- (Array T)
       (broadcast [x y] (+ y (* alpha x)))))

(deftm scale
  "Scale array x by scalar alpha element-wise."
  (All [T] [alpha :- Double,
            x :- (Array T)] :- (Array T)
       (broadcast [x] (* alpha x))))

(deftm fill
  "Fill every element of out with val."
  (All [T] [out :- (Array T), val :- Double] :- (Array T)
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

(ns raster.ga.core
  "Geometric Algebra (Clifford Algebras) for Raster.

   Implements Cl(p,q,r) for arbitrary metric signature:
     p = positive-squaring dimensions
     q = negative-squaring dimensions
     r = zero-squaring (degenerate) dimensions

   Multivectors are represented as dense double[2^n] coefficient arrays,
   indexed by blade bit-patterns ordered by grade then lexicographic.

   Common signatures:
     (vga 3)       ;; Cl(3,0,0) — Euclidean 3D
     (pga 3)       ;; Cl(3,0,1) — Projective 3D
     (cga 3)       ;; Cl(4,1,0) — Conformal 3D
     (sta)         ;; Cl(1,3,0) — Spacetime
     (exterior 3)  ;; Cl(0,0,3) — Exterior algebra Λ(R³)

   Usage:
     (ns my.app
       (:refer-clojure :exclude [+ - * / zero? ...])
       (:require [raster.numeric :refer [+ - *]]
                 [raster.ga.core :refer [vga basis geometric-product wedge]]))"
  (:refer-clojure :exclude [+ - * /
                            aget aset alength aclone
                            reverse])
  (:require [raster.core :refer [deftm defvalue defval defabstract]]
            [raster.compiler.core.types :as types]
            [raster.types.algebraic-types :as alg]
            [raster.numeric :as num]
            [raster.arrays :refer [aget aset alength aclone acopy!]])
  (:import [raster.types.algebraic_types GradedAlgebra]))

;; ================================================================
;; Clifford — abstract type for all Clifford algebra elements
;; ================================================================

(defabstract Clifford
  "Abstract type for all Clifford algebra elements (multivectors).
  Complex is Cl(0,1), quaternions are Cl(0,2), etc."
  :extends alg/Algebra)

;; ================================================================
;; Signature — metric signature of the algebra
;; ================================================================

(defvalue Signature [p :- Long, q :- Long, r :- Long])

(defn sig-dim
  "Total vector space dimension n = p + q + r."
  ^long [^Signature sig]
  (clojure.core/+ (.p sig) (.q sig) (.r sig)))

(defn algebra-dim
  "Algebra dimension 2^n."
  ^long [^Signature sig]
  (bit-shift-left 1 (sig-dim sig)))

;; Common signature constructors
(defn vga
  "Euclidean (Vector GA): Cl(n,0,0)."
  [n] (->Signature n 0 0))

(defn pga
  "Projective GA: Cl(n,0,1)."
  [n] (->Signature n 0 1))

(defn cga
  "Conformal GA: Cl(n+1,1,0)."
  [n] (->Signature (clojure.core/+ n 1) 1 0))

(defn sta
  "Spacetime Algebra: Cl(1,3,0)."
  [] (->Signature 1 3 0))

(defn exterior
  "Exterior Algebra: Cl(0,0,n)."
  [n] (->Signature 0 0 n))

;; ================================================================
;; Blade indexing — bit-pattern based
;; ================================================================

(defn blade-grade
  "Grade of a blade (number of basis vectors present)."
  ^long [^long bits]
  (Long/bitCount bits))

(defn- count-swaps
  "Count the number of adjacent transpositions needed to sort
   basis vectors when multiplying blade a by blade b.
   This determines the sign from reordering."
  ^long [^long a ^long b]
  (loop [a a
         count 0]
    (if (clojure.core/== a 0)
      count
      (let [a (unsigned-bit-shift-right a 1)]
        (recur a (clojure.core/+ count (Long/bitCount (clojure.core/bit-and a b))))))))

(defn blade-product-sign
  "Sign (+1, -1, or 0) of the geometric product of two basis blades.
   Returns 0 if shared basis vectors include any degenerate (r) dimension."
  ^long [^long a ^long b ^Signature sig]
  (let [shared (clojure.core/bit-and a b)
        p (.p sig)
        q (.q sig)
        ;; Check degenerate dimensions: bits >= p+q are degenerate
        ;; If any shared bit is in the degenerate range, product is 0
        pq (clojure.core/+ p q)]
    (if (clojure.core/> (clojure.core/bit-and shared
                                              (bit-shift-left
                                               (clojure.core/- (bit-shift-left 1 (.r sig)) 1)
                                               pq))
                        0)
      0
      ;; Count metric sign flips from negative-squaring dimensions
      (let [;; Transposition sign
            swap-sign (if (clojure.core/even? (count-swaps a b)) 1 -1)
            ;; Metric sign: each shared negative-squaring dim contributes -1
            neg-mask (clojure.core/bit-and shared
                                           (bit-shift-left
                                            (clojure.core/- (bit-shift-left 1 q) 1)
                                            p))
            metric-flips (Long/bitCount neg-mask)
            metric-sign (if (clojure.core/even? metric-flips) 1 -1)]
        (clojure.core/* swap-sign metric-sign)))))

(defn blade-product-result
  "Result blade (XOR) of geometric product of two basis blades."
  ^long [^long a ^long b]
  (clojure.core/bit-xor a b))

;; Blade ordering: sorted by grade, then by bit-pattern within grade.
;; We precompute lookup tables per dimension.

(defn make-blade-order
  "Returns [bits->index, index->bits] for dimension n.
   Blades ordered by grade, then lexicographic bit-pattern within grade."
  [^long n]
  (let [dim (bit-shift-left 1 n)
        ;; Group blade bit-patterns by grade, sort within grade
        sorted-blades (->> (range dim)
                           (sort-by (fn [bits]
                                      [(Long/bitCount bits) bits]))
                           vec)
        bits->index (long-array dim)]
    (dotimes [i dim]
      (clojure.core/aset bits->index (long (nth sorted-blades i)) i))
    {:bits->index bits->index
     :index->bits sorted-blades}))

(def ^:private blade-order-cache (atom {}))

(defn get-blade-order [^long n]
  (or (clojure.core/get @blade-order-cache n)
      (let [order (make-blade-order n)]
        (swap! blade-order-cache assoc n order)
        order)))

(defn blade-index
  "Index of blade bit-pattern in the dense coefficient array."
  ^long [^long bits ^long n]
  (clojure.core/aget ^longs (:bits->index (get-blade-order n)) bits))

(defn index-blade
  "Blade bit-pattern at a given index in the dense coefficient array."
  ^long [^long index ^long n]
  (long (nth (:index->bits (get-blade-order n)) index)))

;; ================================================================
;; Grade ranges — start/end indices for each grade in dense array
;; ================================================================

(defn- binomial
  "Binomial coefficient C(n,k)."
  ^long [^long n ^long k]
  (if (or (clojure.core/< k 0) (clojure.core/> k n))
    0
    (let [k (clojure.core/min k (clojure.core/- n k))]
      (loop [i 0 result 1]
        (if (clojure.core/== i k)
          result
          (recur (inc i)
                 (clojure.core/quot (clojure.core/* result (clojure.core/- n i))
                                    (inc i))))))))

(defn grade-start
  "Start index of grade k in the dense array."
  ^long [^long n ^long k]
  (loop [g 0 start 0]
    (if (clojure.core/== g k)
      start
      (recur (inc g) (clojure.core/+ start (binomial n g))))))

(defn grade-size
  "Number of blades of grade k in dimension n."
  ^long [^long n ^long k]
  (binomial n k))

;; ================================================================
;; Multivector — general dense representation
;; ================================================================

(defvalue Multivector [sig :- Signature, data :- (Array double)]
  :implements GradedAlgebra)

(defn multivector
  "Create a zero multivector for the given signature."
  [sig]
  (->Multivector sig (double-array (algebra-dim sig))))

(defn scalar-mv
  "Create a scalar multivector (grade 0 only)."
  [sig ^double s]
  (let [dim (algebra-dim sig)
        data (double-array dim)]
    (clojure.core/aset data 0 s)
    (->Multivector sig data)))

(defn basis
  "Create the i-th basis vector (0-indexed) as a multivector.
   basis(0) = e₁, basis(1) = e₂, etc."
  [sig ^long i]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        bits (bit-shift-left 1 i)
        data (double-array dim)
        idx (blade-index bits n)]
    (clojure.core/aset data idx 1.0)
    (->Multivector sig data)))

(defn pseudoscalar
  "The pseudoscalar e₁₂₃...ₙ."
  [sig]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        bits (clojure.core/- dim 1) ;; all bits set
        data (double-array dim)
        idx (blade-index bits n)]
    (clojure.core/aset data idx 1.0)
    (->Multivector sig data)))

(defn from-grade
  "Create a multivector with only grade-k components.
   coeffs is a seq of C(n,k) doubles in lexicographic blade order."
  [sig ^long k coeffs]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        data (double-array dim)
        start (grade-start n k)
        coeffs (vec coeffs)]
    (dotimes [i (count coeffs)]
      (clojure.core/aset data (clojure.core/+ start i) (double (nth coeffs i))))
    (->Multivector sig data)))

;; ================================================================
;; Precomputed multiplication table
;; ================================================================
;; For a given signature, precompute for every (i,j) pair:
;;   - result index k  (int[])
;;   - sign as double  (double[] — avoids int→double cast in hot loop)
;; Both flat arrays indexed by i*dim+j.
;; This eliminates all per-iteration blade-index/index-blade/blade-product-sign calls.

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  mul-table-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- make-mul-table
  "Precompute multiplication table for a signature.
   Returns [int[] result-idx, double[] signs]."
  [sig]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        total (clojure.core/* dim dim)
        result-idx (int-array total)
        signs (double-array total)]
    (dotimes [i dim]
      (let [a-bits (index-blade i n)]
        (dotimes [j dim]
          (let [b-bits (index-blade j n)
                flat (clojure.core/+ (clojure.core/* i dim) j)
                sign (blade-product-sign a-bits b-bits sig)]
            (clojure.core/aset signs flat (double sign))
            (if (clojure.core/== sign 0)
              (clojure.core/aset result-idx flat (int 0))
              (let [r-bits (blade-product-result a-bits b-bits)]
                (clojure.core/aset result-idx flat (int (blade-index r-bits n)))))))))
    ;; Store as Object[2] for fast field-style access (no vector destructuring)
    (let [arr (object-array 2)]
      (clojure.core/aset arr 0 result-idx)
      (clojure.core/aset arr 1 signs)
      arr)))

(defn- sig-key ^long [^Signature sig]
  ;; Pack p,q,r into a single long (each < 2^20)
  (clojure.core/+ (.p sig)
                  (bit-shift-left (.q sig) 20)
                  (bit-shift-left (.r sig) 40)))

;; ================================================================
;; Precomputed grade tables for grade-filtered products
;; ================================================================

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  grade-table-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- make-grade-table
  "Precompute grade of each dense array index for dimension n.
   Returns int[dim] where grade-table[i] = grade of blade at index i."
  [^long n]
  (let [dim (bit-shift-left 1 n)
        grades (int-array dim)]
    (dotimes [i dim]
      (clojure.core/aset grades i (int (blade-grade (index-blade i n)))))
    grades))

(defn get-grade-table ^ints [^long n]
  (let [k (Long/valueOf n)]
    (or (.get grade-table-cache k)
        (let [table (make-grade-table n)]
          (.putIfAbsent grade-table-cache k table)
          (or (.get grade-table-cache k) table)))))

;; ================================================================
;; Adaptive compilation — emit unrolled product fn per signature
;; ================================================================
;; On first use of a signature, compiles a specialized function with
;; all loop iterations unrolled and signs/indices as literal constants.
;; This eliminates loop overhead, table lookups, and accumulator reads.

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  compiled-product-cache (java.util.concurrent.ConcurrentHashMap.))

(defn term-expr
  "Generate code for sign * a[i] * b[j], staying primitive."
  [a-sym b-sym [i j sign]]
  (let [prod `(clojure.core/* (clojure.core/aget ~a-sym ~i)
                              (clojure.core/aget ~b-sym ~j))]
    (case (long sign)
      1  prod
      -1 `(clojure.core/- 0.0 ~prod)
      `(clojure.core/* ~(double sign) ~prod))))

(defn sum-expr
  "Reduce terms with nested binary + to stay primitive (no variadic boxing)."
  [terms]
  (reduce (fn [acc t] `(clojure.core/+ ~acc ~t)) terms))

(defn- compile-unrolled-fn
  "Generate and eval an unrolled product fn from a contribution map.
   contribs: {result-index k -> [[i j sign] ...]}
   Returns (fn [^doubles a ^doubles b] -> double[dim])."
  [dim contribs]
  (let [a-sym (gensym "a")
        b-sym (gensym "b")
        r-sym (gensym "r")
        aset-forms
        (for [k (range dim)
              :let [terms (get contribs k)]
              :when (seq terms)]
          `(clojure.core/aset ~r-sym ~k
                              ~(sum-expr (map #(term-expr a-sym b-sym %) terms))))
        fn-form `(fn [~(with-meta a-sym {:tag 'doubles})
                      ~(with-meta b-sym {:tag 'doubles})]
                   (let [~r-sym (double-array ~dim)]
                     ~@aset-forms
                     ~r-sym))]
    (eval fn-form)))

(defn collect-contribs
  "Collect contributions for all (i,j) pairs, optionally filtered by grade."
  [sig grade-filter]
  (let [n (sig-dim sig)
        dim (algebra-dim sig)
        grades (when grade-filter (get-grade-table n))]
    (reduce
     (fn [m i]
       (reduce
        (fn [m j]
          (let [a-bits (index-blade i n)
                b-bits (index-blade j n)
                sign (blade-product-sign a-bits b-bits sig)]
            (if (clojure.core/== sign 0)
              m
              (let [r-bits (blade-product-result a-bits b-bits)
                    k (blade-index r-bits n)]
                (if (and grade-filter
                         (not (grade-filter
                               (clojure.core/aget ^ints grades i)
                               (clojure.core/aget ^ints grades j)
                               (clojure.core/aget ^ints grades k))))
                  m
                  (update m k (fnil conj []) [i j sign]))))))
        m (range dim)))
     {} (range dim))))

(defn- compile-product-fn [sig]
  (compile-unrolled-fn (algebra-dim sig) (collect-contribs sig nil)))

(defn- get-compiled-product [sig]
  (let [k (Long/valueOf (sig-key sig))]
    (or (.get compiled-product-cache k)
        (let [f (compile-product-fn sig)]
          (.putIfAbsent compiled-product-cache k f)
          (or (.get compiled-product-cache k) f)))))

;; ================================================================
;; Geometric Product — adaptively compiled
;; ================================================================

(deftm geometric-product [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        f (get-compiled-product sig)]
    (->Multivector sig (f (.data a) (.data b)))))

;; ================================================================
;; Grade-filtered products — adaptively compiled
;; ================================================================
(def ^:private ^java.util.concurrent.ConcurrentHashMap
  compiled-filtered-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- compile-filtered-product-fn
  "Generate an unrolled grade-filtered product fn for a signature + filter.
   grade-filter: (fn [grade-a grade-b grade-result] -> bool) applied at compile time."
  [sig grade-filter]
  (compile-unrolled-fn (algebra-dim sig) (collect-contribs sig grade-filter)))

(def ^:private ^:const filter-key-offsets
  {:wedge 1, :inner 2, :left-contract 3, :scalar 4})

(defn- get-compiled-filtered [sig filter-key grade-filter]
  ;; Pack sig + filter into a single Long key for fast ConcurrentHashMap lookup
  (let [offset (long (get filter-key-offsets filter-key 0))
        k (Long/valueOf (clojure.core/+ (sig-key sig)
                                        (bit-shift-left offset 60)))]
    (or (.get compiled-filtered-cache k)
        (let [f (compile-filtered-product-fn sig grade-filter)]
          (.putIfAbsent compiled-filtered-cache k f)
          (or (.get compiled-filtered-cache k) f)))))

(deftm wedge [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        f (get-compiled-filtered sig :wedge
                                 (fn [ga gb rg] (clojure.core/== rg (clojure.core/+ ga gb))))]
    (->Multivector sig (f (.data a) (.data b)))))

(deftm inner [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        f (get-compiled-filtered sig :inner
                                 (fn [ga gb rg]
                                   (and (clojure.core/> ga 0) (clojure.core/> gb 0)
                                        (clojure.core/== rg (Math/abs (clojure.core/- ga gb))))))]
    (->Multivector sig (f (.data a) (.data b)))))

(deftm left-contract [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        f (get-compiled-filtered sig :left-contract
                                 (fn [ga gb rg]
                                   (clojure.core/== rg (clojure.core/- gb ga))))]
    (->Multivector sig (f (.data a) (.data b)))))

(deftm scalar-product [a :- Multivector, b :- Multivector] :- Double
  (let [sig (.sig a)
        f (get-compiled-filtered sig :scalar
                                 (fn [_ga _gb rg] (clojure.core/== rg 0)))
        result (f (.data a) (.data b))]
    (clojure.core/aget ^doubles result 0)))

;; ================================================================
;; Precomputed sign tables for involutions
;; ================================================================

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  involution-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- make-involution-signs
  "Precompute sign arrays for reverse, grade-involution, conjugate.
   Returns {:reverse double[], :grade-inv double[], :conjugate double[]}."
  [^long n]
  (let [dim (bit-shift-left 1 n)
        grades (get-grade-table n)
        rev-signs (double-array dim)
        ginv-signs (double-array dim)
        conj-signs (double-array dim)]
    (dotimes [i dim]
      (let [g (clojure.core/aget ^ints grades i)]
        (clojure.core/aset rev-signs i
                           (if (clojure.core/even? (clojure.core/quot (clojure.core/* g (clojure.core/- g 1)) 2))
                             1.0 -1.0))
        (clojure.core/aset ginv-signs i
                           (if (clojure.core/even? g) 1.0 -1.0))
        (clojure.core/aset conj-signs i
                           (if (clojure.core/even? (clojure.core/quot (clojure.core/* g (clojure.core/+ g 1)) 2))
                             1.0 -1.0))))
    {:reverse rev-signs :grade-inv ginv-signs :conjugate conj-signs}))

(defn- get-involution-signs [^long n]
  (let [k (Long/valueOf n)]
    (or (.get involution-cache k)
        (let [signs (make-involution-signs n)]
          (.putIfAbsent involution-cache k signs)
          (or (.get involution-cache k) signs)))))

;; ================================================================
;; Adaptively compiled involutions
;; ================================================================

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  compiled-involution-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- compile-involution-fn
  "Generate an unrolled involution fn from precomputed sign array.
   Returns (fn [^doubles x] -> double[dim])."
  [^long dim ^doubles signs]
  (let [x-sym (gensym "x")
        r-sym (gensym "r")
        aset-forms
        (for [i (range dim)
              :let [s (clojure.core/aget signs i)]]
          (cond
            (clojure.core/== s 1.0)
            `(clojure.core/aset ~r-sym ~i (clojure.core/aget ~x-sym ~i))
            (clojure.core/== s -1.0)
            `(clojure.core/aset ~r-sym ~i (clojure.core/- 0.0 (clojure.core/aget ~x-sym ~i)))
            :else
            `(clojure.core/aset ~r-sym ~i (clojure.core/* ~s (clojure.core/aget ~x-sym ~i)))))
        fn-form `(fn [~(with-meta x-sym {:tag 'doubles})]
                   (let [~r-sym (double-array ~dim)]
                     ~@aset-forms
                     ~r-sym))]
    (eval fn-form)))

(defn- get-compiled-involution [^long n involution-key]
  ;; Pack n + involution-key into single Long
  (let [inv-offset (case involution-key :reverse 0 :grade-inv 1 :conjugate 2)
        k (Long/valueOf (clojure.core/+ n (bit-shift-left inv-offset 20)))]
    (or (.get compiled-involution-cache k)
        (let [signs (involution-key (get-involution-signs n))
              dim (bit-shift-left 1 n)
              f (compile-involution-fn dim signs)]
          (.putIfAbsent compiled-involution-cache k f)
          (or (.get compiled-involution-cache k) f)))))

(deftm reverse-mv [x :- Multivector] :- Multivector
  (let [sig (.sig x)
        n (sig-dim sig)
        f (get-compiled-involution n :reverse)]
    (->Multivector sig (f (.data x)))))

(deftm grade-involution [x :- Multivector] :- Multivector
  (let [sig (.sig x)
        n (sig-dim sig)
        f (get-compiled-involution n :grade-inv)]
    (->Multivector sig (f (.data x)))))

(deftm conjugate [x :- Multivector] :- Multivector
  (let [sig (.sig x)
        n (sig-dim sig)
        f (get-compiled-involution n :conjugate)]
    (->Multivector sig (f (.data x)))))

;; ================================================================
;; Grade selection
;; ================================================================

(def ^:private ^java.util.concurrent.ConcurrentHashMap
  compiled-grade-select-cache (java.util.concurrent.ConcurrentHashMap.))

(defn- compile-grade-select-fn
  "Generate an unrolled grade-select fn for dimension n and grade k.
   Returns (fn [^doubles x] -> double[dim])."
  [^long n ^long k]
  (let [dim (bit-shift-left 1 n)
        grades (get-grade-table n)
        x-sym (gensym "x")
        r-sym (gensym "r")
        aset-forms
        (for [i (range dim)
              :when (clojure.core/== (clojure.core/aget ^ints grades i) k)]
          `(clojure.core/aset ~r-sym ~i (clojure.core/aget ~x-sym ~i)))
        fn-form `(fn [~(with-meta x-sym {:tag 'doubles})]
                   (let [~r-sym (double-array ~dim)]
                     ~@aset-forms
                     ~r-sym))]
    (eval fn-form)))

(defn- get-compiled-grade-select [^long n ^long k]
  (let [cache-key (Long/valueOf (clojure.core/+ n (bit-shift-left k 20)))]
    (or (.get compiled-grade-select-cache cache-key)
        (let [f (compile-grade-select-fn n k)]
          (.putIfAbsent compiled-grade-select-cache cache-key f)
          (or (.get compiled-grade-select-cache cache-key) f)))))

(deftm grade-select [x :- Multivector, k :- Long] :- Multivector
  (let [sig (.sig x)
        n (sig-dim sig)
        f (get-compiled-grade-select n k)]
    (->Multivector sig (f (.data x)))))

(deftm mv-scalar [x :- Multivector] :- Double
  (clojure.core/aget ^doubles (.data x) 0))

;; ================================================================
;; Norm — inlined to avoid deftm dispatch overhead
;; ================================================================

(deftm norm-squared [x :- Multivector] :- Double
  (scalar-product x (reverse-mv x)))

(deftm norm [x :- Multivector] :- Double
  (Math/sqrt (Math/abs (norm-squared x))))

;; ================================================================
;; Hodge star (dual)
;; ================================================================

(deftm hodge-star [x :- Multivector] :- Multivector
  (let [sig (.sig x)
        n (sig-dim sig)
        I (pseudoscalar sig)]
    ;; ⋆x = x ⌋ I⁻¹
    ;; For non-degenerate algebras, I⁻¹ = ±I depending on n and signature
    ;; x · I⁻¹ via left contraction
    (left-contract x I)))

;; ================================================================
;; Regressive product: a ∨ b = ⋆⁻¹(⋆a ∧ ⋆b)
;; ================================================================

(deftm regressive [a :- Multivector, b :- Multivector] :- Multivector
  ;; Undual(Dual(a) ∧ Dual(b))
  ;; For simplicity, use the identity: a ∨ b = J⁻¹((J a) ∧ (J b))
  ;; where J is right complement
  (let [sig (.sig a)
        n (sig-dim sig)
        dim (algebra-dim sig)
        all-bits (clojure.core/- dim 1)
        ;; Right complement: map blade e_S to sign * e_{complement(S)}
        complement-mv (fn [mv]
                        (let [mv-data (.data mv)
                              result (double-array dim)]
                          (dotimes [i dim]
                            (let [bits (index-blade i n)
                                  comp-bits (clojure.core/bit-xor bits all-bits)
                                  sign (blade-product-sign bits comp-bits sig)
                                  k (blade-index comp-bits n)]
                              (clojure.core/aset ^doubles result k
                                                 (clojure.core/* (double sign)
                                                                 (clojure.core/aget ^doubles mv-data i)))))
                          (->Multivector sig result)))]
    (complement-mv (wedge (complement-mv a) (complement-mv b)))))

;; ================================================================
;; Multivector arithmetic — raster.numeric integration
;; ================================================================

(deftm raster.numeric/+ [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        dim (algebra-dim sig)
        a-data (.data a)
        b-data (.data b)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i
                         (clojure.core/+ (clojure.core/aget ^doubles a-data i)
                                         (clojure.core/aget ^doubles b-data i))))
    (->Multivector sig result)))

(deftm raster.numeric/- [a :- Multivector, b :- Multivector] :- Multivector
  (let [sig (.sig a)
        dim (algebra-dim sig)
        a-data (.data a)
        b-data (.data b)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i
                         (clojure.core/- (clojure.core/aget ^doubles a-data i)
                                         (clojure.core/aget ^doubles b-data i))))
    (->Multivector sig result)))

(deftm raster.numeric/* [a :- Multivector, b :- Multivector] :- Multivector
  (geometric-product a b))

;; Scalar-multivector operations
(deftm raster.numeric/* [a :- Double, b :- Multivector] :- Multivector
  (let [sig (.sig b)
        dim (algebra-dim sig)
        b-data (.data b)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i
                         (clojure.core/* a (clojure.core/aget ^doubles b-data i))))
    (->Multivector sig result)))

(deftm raster.numeric/* [a :- Multivector, b :- Double] :- Multivector
  (let [sig (.sig a)
        dim (algebra-dim sig)
        a-data (.data a)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i
                         (clojure.core/* (clojure.core/aget ^doubles a-data i) b)))
    (->Multivector sig result)))

(deftm raster.numeric/+ [a :- Double, b :- Multivector] :- Multivector
  (let [sig (.sig b)
        dim (algebra-dim sig)
        b-data (.data b)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i (clojure.core/aget ^doubles b-data i)))
    (clojure.core/aset ^doubles result 0
                       (clojure.core/+ a (clojure.core/aget ^doubles result 0)))
    (->Multivector sig result)))

(deftm raster.numeric/+ [a :- Multivector, b :- Double] :- Multivector
  (let [sig (.sig a)
        dim (algebra-dim sig)
        a-data (.data a)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i (clojure.core/aget ^doubles a-data i)))
    (clojure.core/aset ^doubles result 0
                       (clojure.core/+ (clojure.core/aget ^doubles result 0) b))
    (->Multivector sig result)))

(deftm raster.numeric/- [a :- Multivector] :- Multivector
  (let [sig (.sig a)
        dim (algebra-dim sig)
        a-data (.data a)
        result (double-array dim)]
    (dotimes [i dim]
      (clojure.core/aset ^doubles result i
                         (clojure.core/- (clojure.core/aget ^doubles a-data i))))
    (->Multivector sig result)))

;; ================================================================
;; Display
;; ================================================================

(defn blade-name
  "Human-readable name for a blade bit-pattern."
  [^long bits ^long n]
  (if (clojure.core/== bits 0)
    "1"
    (let [sb (StringBuilder.)]
      (dotimes [i n]
        (when-not (clojure.core/== (clojure.core/bit-and bits (bit-shift-left 1 i)) 0)
          (when (clojure.core/> (.length sb) 0)
            (.append sb "∧"))
          (.append sb "e")
          (.append sb (inc i))))
      (.toString sb))))

(defn mv->str
  "String representation of a multivector."
  [mv]
  (let [sig (.sig mv)
        n (sig-dim sig)
        dim (algebra-dim sig)
        data (.data mv)
        terms (for [i (range dim)
                    :let [bits (index-blade i n)
                          coeff (clojure.core/aget ^doubles data i)]
                    :when (not (clojure.core/== coeff 0.0))]
                (if (clojure.core/== bits 0)
                  (str coeff)
                  (str coeff "·" (blade-name bits n))))]
    (if (empty? terms)
      "0"
      (clojure.core/apply str (interpose " + " terms)))))

(defmethod print-method Multivector [mv ^java.io.Writer w]
  (.write w (str "#ga/mv[" (mv->str mv) "]")))

;; ================================================================
;; Array protocol: make Multivector usable as ODE state
;; and recognizable by broadcast!/reduce!/scan! in deftm bodies
;; ================================================================

;; Register Multivector as array-like: elements are double, stored as double[]
(raster.compiler.core.types/register-array-like! 'Multivector 'double 'double)
;; These overloads allow raster.ode solvers to operate directly on
;; multivectors without manual double-array conversion.

(deftm raster.numeric/similar [mv :- Multivector] :- Multivector
  (->Multivector (.-sig mv) (double-array (clojure.core/alength ^doubles (.-data mv)))))

(deftm raster.arrays/alength [mv :- Multivector] :- Long
  (long (clojure.core/alength ^doubles (.-data mv))))

(deftm raster.arrays/aclone [mv :- Multivector] :- Multivector
  (->Multivector (.-sig mv) (clojure.core/aclone ^doubles (.-data mv))))

(deftm raster.arrays/aget [mv :- Multivector, i :- Long] :- Double
  (clojure.core/aget ^doubles (.-data mv) (int i)))

(deftm raster.arrays/aset [mv :- Multivector, i :- Long, v :- Double]
  (clojure.core/aset ^doubles (.-data mv) (int i) (double v)))

(deftm raster.arrays/acopy! [src :- Multivector, src-off :- Long,
                             dst :- Multivector, dst-off :- Long, len :- Long]
  (System/arraycopy ^doubles (.-data src) (int src-off)
                    ^doubles (.-data dst) (int dst-off) (int len)))

(ns raster.arrays
  "Polymorphic array operations extensible to value type arrays.
   API-compatible with clojure.core/aget, aset, alength, aclone.

   When the walker devirtualizes these deftm calls, the resolved method
   body calls clojure.core/aget ^doubles etc., emitting identical
   bytecode (daload, etc.) to hardcoded type hints. Zero overhead.

   Covers double[], long[], int[], float[] primitives plus Object[]
   fallback. Uncommon types (byte[], short[], char[], boolean[]) go
   through the Object fallback via runtime dispatch.

   To add support for new array types (e.g. Float16[]):
     (deftm aget [arr :- (Array Float16), i :- Long] :- Float16
       (->Float16 (clojure.core/aget ^shorts arr (int i))))"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm]]))

;; ================================================================
;; aget — element access
;; ================================================================

(deftm aget "Read element at index i from a typed array."
  [arr :- (Array double), i :- Long] :- Double
  (clojure.core/aget ^doubles arr (int i)))
(deftm aget [arr :- (Array long), i :- Long] :- Long
  (clojure.core/aget ^longs arr (int i)))
(deftm aget [arr :- (Array int), i :- Long] :- Integer
  (clojure.core/aget ^ints arr (int i)))
(deftm aget [arr :- (Array float), i :- Long] :- Float
  (clojure.core/aget ^floats arr (int i)))
;; Object[] fallback (also covers byte[], short[], char[], boolean[])
(deftm aget [arr :- Object, i :- Long]
  (clojure.core/aget ^objects arr (int i)))

;; ================================================================
;; aset — element mutation
;; ================================================================

;; aset is void — the JVM store ops (dastore etc.) don't return values.
;; No return annotation → return-tag nil → sibling method returns Object (null).
(deftm aset "Write value v at index i in a typed array."
  [arr :- (Array double), i :- Long, v :- Double]
  (clojure.core/aset ^doubles arr (int i) (double v)))
(deftm aset [arr :- (Array long), i :- Long, v :- Long]
  (clojure.core/aset ^longs arr (int i) (long v)))
(deftm aset [arr :- (Array int), i :- Long, v :- Integer]
  (clojure.core/aset ^ints arr (int i) (int v)))
(deftm aset [arr :- (Array float), i :- Long, v :- Float]
  (clojure.core/aset ^floats arr (int i) (float v)))
;; Narrowing: double → float (Julia-style automatic conversion)
(deftm aset [arr :- (Array float), i :- Long, v :- Double]
  (clojure.core/aset ^floats arr (int i) (float v)))
;; Object[] fallback — call RT.aset directly because clojure.core/aset
;; with ^objects emits AASTORE (void) causing stack underflow in deftype methods
(deftm aset [arr :- Object, i :- Long, v :- Object]
  (clojure.lang.RT/aset ^objects arr (int i) v))

;; ================================================================
;; alength — array length
;; ================================================================

;; JVM arraylength returns int. We cast to long to ensure consistent
;; Long type even when the walker inlines the body.
(deftm alength "Return the length of a typed array as Long."
  [arr :- (Array double)] :- Long
  (long (clojure.core/alength ^doubles arr)))
(deftm alength [arr :- (Array long)] :- Long
  (long (clojure.core/alength ^longs arr)))
(deftm alength [arr :- (Array int)] :- Long
  (long (clojure.core/alength ^ints arr)))
(deftm alength [arr :- (Array float)] :- Long
  (long (clojure.core/alength ^floats arr)))
;; Object[] fallback
(deftm alength [arr :- Object] :- Long
  (long (clojure.core/alength ^objects arr)))

;; ================================================================
;; aclone — array copy
;; ================================================================

(deftm aclone "Return a shallow copy of the array."
  [arr :- (Array double)] :- (Array double)
  (clojure.core/aclone ^doubles arr))
(deftm aclone [arr :- (Array long)] :- (Array long)
  (clojure.core/aclone ^longs arr))
(deftm aclone [arr :- (Array int)] :- (Array int)
  (clojure.core/aclone ^ints arr))
(deftm aclone [arr :- (Array float)] :- (Array float)
  (clojure.core/aclone ^floats arr))
;; Object[] fallback
(deftm aclone [arr :- Object]
  (clojure.core/aclone ^objects arr))

;; ================================================================
;; acopy! — region copy (src → dst with offsets)
;; GPU-compilable replacement for System/arraycopy.
;; ================================================================

(deftm acopy! "Copy len elements from src[src-off..] into dst[dst-off..]. Returns dst."
  [src :- (Array double) src-off :- Long
   dst :- (Array double) dst-off :- Long
   len :- Long] :- (Array double)
  (System/arraycopy src (int src-off) dst (int dst-off) (int len))
  dst)

(deftm acopy! [src :- (Array long) src-off :- Long
               dst :- (Array long) dst-off :- Long
               len :- Long] :- (Array long)
  (System/arraycopy src (int src-off) dst (int dst-off) (int len))
  dst)

(deftm acopy! [src :- (Array int) src-off :- Long
               dst :- (Array int) dst-off :- Long
               len :- Long] :- (Array int)
  (System/arraycopy src (int src-off) dst (int dst-off) (int len))
  dst)

(deftm acopy! [src :- (Array float) src-off :- Long
               dst :- (Array float) dst-off :- Long
               len :- Long] :- (Array float)
  (System/arraycopy src (int src-off) dst (int dst-off) (int len))
  dst)

;; ================================================================
;; zeros-like — allocate a zero array of the same element type
;; Used by AD templates for type-generic gradient buffer allocation.
;; ================================================================

(deftm zeros-like "Allocate a zero-filled array of length n with the same element type as ref."
  [ref :- (Array double) n :- Long] :- (Array double)
  (double-array n))
(deftm zeros-like [ref :- (Array float) n :- Long] :- (Array float)
  (float-array n))
(deftm zeros-like [ref :- (Array long) n :- Long] :- (Array long)
  (long-array n))
(deftm zeros-like [ref :- (Array int) n :- Long] :- (Array int)
  (int-array n))

;; ================================================================
;; fill-zero! — zero a buffer (for hoisted buffer reuse)
;; ================================================================

(deftm fill-zero! "Zero all elements of arr in-place. Returns arr."
  [arr :- (Array double)] :- (Array double)
  (java.util.Arrays/fill ^doubles arr 0.0)
  arr)
(deftm fill-zero! [arr :- (Array float)] :- (Array float)
  (java.util.Arrays/fill ^floats arr (float 0.0))
  arr)
(deftm fill-zero! [arr :- (Array long)] :- (Array long)
  (java.util.Arrays/fill ^longs arr (long 0))
  arr)
(deftm fill-zero! [arr :- (Array int)] :- (Array int)
  (java.util.Arrays/fill ^ints arr (int 0))
  arr)

;; ================================================================
;; alloc-like — allocate array of same type (Julia's `similar`)
;; The reference array determines the element type; n is the size.
;; The allocated array is zero-initialized (JVM guarantee).
;; ================================================================

(deftm alloc-like "Allocate a zero-initialized array of length n matching ref's element type."
  (All [T] [ref :- (Array T), n :- Long] :- (Array T)
  ;; Body uses T which gets substituted to concrete type at specialization.
  ;; double → (double-array n), float → (float-array n), etc.
  ;; We need concrete impls since array constructors aren't polymorphic in JVM.
  ;; Workaround: register concrete overloads directly.
       (zeros-like ref n)))

;; ================================================================
;; fill! — fill array with arbitrary value
;; ================================================================

(deftm fill! "Fill all elements of arr with value v in-place. Returns arr."
  [arr :- (Array double), v :- Double] :- (Array double)
  (java.util.Arrays/fill ^doubles arr (double v))
  arr)
(deftm fill! [arr :- (Array float), v :- Float] :- (Array float)
  (java.util.Arrays/fill ^floats arr (float v))
  arr)
(deftm fill! [arr :- (Array long), v :- Long] :- (Array long)
  (java.util.Arrays/fill ^longs arr (long v))
  arr)
(deftm fill! [arr :- (Array int), v :- Integer] :- (Array int)
  (java.util.Arrays/fill ^ints arr (int v))
  arr)

;; ================================================================
;; argmax — index of maximum element
;; ================================================================

(deftm argmax "Return the index of the maximum element in the array."
  [arr :- (Array double)] :- Long
  (let [n (clojure.core/alength ^doubles arr)]
    (loop [i 1 best 0 best-val (clojure.core/aget ^doubles arr 0)]
      (if (< i n)
        (let [v (clojure.core/aget ^doubles arr (int i))]
          (if (> v best-val)
            (recur (inc i) i v)
            (recur (inc i) best best-val)))
        best))))

(deftm argmax [arr :- (Array float)] :- Long
  (let [n (clojure.core/alength ^floats arr)]
    (loop [i 1 best 0 best-val (double (clojure.core/aget ^floats arr 0))]
      (if (< i n)
        (let [v (double (clojure.core/aget ^floats arr (int i)))]
          (if (> v best-val)
            (recur (inc i) i v)
            (recur (inc i) best best-val)))
        best))))

;; ================================================================
;; argmin — index of minimum element
;; ================================================================

(deftm argmin
  "Return the index of the minimum element in the array."
  [arr :- (Array double)] :- Long
  (let [n (clojure.core/alength ^doubles arr)]
    (loop [i 1 best-idx 0 best-val (clojure.core/aget ^doubles arr 0)]
      (if (< i n)
        (let [v (clojure.core/aget ^doubles arr (int i))]
          (if (< v best-val)
            (recur (unchecked-inc i) i v)
            (recur (unchecked-inc i) best-idx best-val)))
        best-idx))))

(deftm argmin [arr :- (Array float)] :- Long
  (let [n (clojure.core/alength ^floats arr)]
    (loop [i 1 best-idx 0 best-val (clojure.core/aget ^floats arr 0)]
      (if (< i n)
        (let [v (clojure.core/aget ^floats arr (int i))]
          (if (< v best-val)
            (recur (unchecked-inc i) i v)
            (recur (unchecked-inc i) best-idx best-val)))
        best-idx))))

(ns raster.dl.tensor
  "Tensor utilities for the Raster deep learning framework.

  Tensors are flat typed arrays (double[], float[], etc.) with shape tracked
  in a global registry (keyed by identity). All operations are pure utility
  functions for user code and tests. The compiler operates on flat arrays
  and explicit index arithmetic within deftm bodies.

  Convention: shapes are row-major (C order). A [B, C, H, W] tensor has
  element [b,c,h,w] at index b*C*H*W + c*H*W + h*W + w."
  (:require [raster.arrays :as ra])
  (:import [java.util Random WeakHashMap Collections]))

;; ================================================================
;; Shape registry (WeakHashMap for auto-cleanup)
;; ================================================================

(defonce ^:private shape-registry
  (Collections/synchronizedMap (WeakHashMap.)))

(defn register-tensor
  "Register shape for any typed array. Returns the array."
  [arr shape]
  (.put shape-registry arr (vec shape))
  arr)

(defn- register-shape!
  "Internal: register shape. Accepts any array type."
  [arr shape]
  (.put shape-registry arr (vec shape))
  arr)

;; ================================================================
;; Shape metadata
;; ================================================================

(defn tensor
  "Create a tensor from data with shape.
  data: seqable of numbers or existing double[]
  shape: vector of dimension sizes, e.g. [2 3 4]
  Returns the double[] with shape registered."
  ^doubles [data shape]
  (let [arr (if (instance? (Class/forName "[D") data)
              ^doubles data
              (double-array data))
        expected (reduce * 1 shape)]
    (assert (= (alength arr) expected)
            (str "Shape " shape " expects " expected " elements, got " (alength arr)))
    (register-shape! arr shape)))

(defn tshape
  "Get the shape vector for a tensor."
  [t]
  (.get shape-registry t))

(defn trank
  "Get the rank (number of dimensions) of a tensor."
  ^long [t]
  (count (tshape t)))

(defn tsize
  "Get the total number of elements in a tensor."
  ^long [t]
  (ra/alength t))

;; ================================================================
;; Index arithmetic
;; ================================================================

(defn tidx
  "Compute flat index from multi-dimensional indices and shape.
  Shape is row-major: [d0, d1, ..., dn] with strides computed right-to-left."
  [shape & indices]
  (assert (= (count shape) (count indices))
          (str "Shape " shape " has " (count shape) " dims but got " (count indices) " indices"))
  (long (loop [dims (seq shape)
               idxs (seq indices)
               flat 0]
          (if dims
            (let [d (long (first dims))
                  i (long (first idxs))]
              (assert (and (>= i 0) (< i d))
                      (str "Index " i " out of bounds for dim " d))
              (recur (next dims) (next idxs) (+ (* flat d) i)))
            flat))))

(defn strides
  "Compute strides for a row-major shape. Returns a vector of strides."
  [shape]
  (let [n (count shape)]
    (loop [i (dec n) s 1 acc (transient (vec (repeat n 0)))]
      (if (>= i 0)
        (recur (dec i) (* s (long (nth shape i))) (assoc! acc i s))
        (persistent! acc)))))

;; ================================================================
;; Creation helpers
;; ================================================================

(defn zeros
  "Create a zero tensor with given shape."
  ^doubles [shape]
  (tensor (double-array (reduce * 1 shape)) shape))

(defn ones
  "Create a tensor of ones with given shape."
  ^doubles [shape]
  (let [n (reduce * 1 shape)
        arr (double-array n)]
    (java.util.Arrays/fill arr 1.0)
    (tensor arr shape)))

(defn full
  "Create a tensor filled with a constant value."
  ^doubles [shape value]
  (let [n (reduce * 1 shape)
        arr (double-array n)]
    (java.util.Arrays/fill arr (double value))
    (tensor arr shape)))

(defn randn
  "Create a tensor with standard normal random values.
  Optional: provide a Random instance for reproducibility."
  (^doubles [shape]
   (randn shape (Random.)))
  (^doubles [shape ^Random rng]
   (let [n (reduce * 1 shape)
         arr (double-array n)]
     (dotimes [i n]
       (aset arr i (.nextGaussian rng)))
     (tensor arr shape))))

(defn rand-uniform
  "Create a tensor with uniform random values in [low, high)."
  (^doubles [shape]
   (rand-uniform shape 0.0 1.0))
  (^doubles [shape low high]
   (rand-uniform shape low high (Random.)))
  (^doubles [shape low high ^Random rng]
   (let [n (reduce * 1 shape)
         arr (double-array n)
         range-val (- (double high) (double low))
         low-val (double low)]
     (dotimes [i n]
       (aset arr i (+ low-val (* range-val (.nextDouble rng)))))
     (tensor arr shape))))

(defn arange
  "Create a 1D tensor [0, 1, ..., n-1]."
  ^doubles [n]
  (let [arr (double-array n)]
    (dotimes [i n]
      (aset arr i (double i)))
    (tensor arr [n])))

(defn linspace
  "Create a 1D tensor of n evenly spaced values from start to end (inclusive)."
  ^doubles [start end n]
  (let [arr (double-array n)
        step (if (> n 1) (/ (- (double end) (double start)) (double (dec n))) 0.0)]
    (dotimes [i n]
      (aset arr i (+ (double start) (* step (double i)))))
    (tensor arr [n])))

;; ================================================================
;; Reshape / Transpose
;; ================================================================

(defn reshape
  "Reshape tensor to new shape (zero-copy, shares underlying array).
  Total element count must match."
  [t new-shape]
  (let [old-n (ra/alength t)
        new-n (reduce * 1 new-shape)]
    (assert (= old-n new-n)
            (str "Cannot reshape " old-n " elements to shape " new-shape
                 " (" new-n " elements)"))
    (register-shape! t new-shape)))

(defn transpose-2d
  "Transpose a 2D tensor [rows, cols] -> [cols, rows].
  Returns a new array. Works with any numeric array type."
  [t rows cols]
  (let [out (ra/alloc-like t (* rows cols))]
    (dotimes [i rows]
      (dotimes [j cols]
        (ra/aset out (+ (* j rows) i) (ra/aget t (+ (* i cols) j)))))
    (register-tensor out [cols rows])))

(defn flatten-tensor
  "Flatten a tensor to 1D (zero-copy)."
  [t]
  (reshape t [(ra/alength t)]))

;; ================================================================
;; Basic operations (for tests and user code, not deftm)
;; ================================================================

(defn tensor-add
  "Element-wise addition of two tensors (must have same size).
  Works with any numeric array type."
  [a b]
  (let [n (ra/alength a)
        out (ra/alloc-like a n)]
    (assert (= n (ra/alength b)) "Tensors must have same size")
    (dotimes [i n]
      (ra/aset out i (+ (double (ra/aget a i)) (double (ra/aget b i)))))
    (let [shape (tshape a)]
      (if shape (register-tensor out shape) out))))

(defn tensor-scale
  "Scale a tensor by a scalar. Works with any numeric array type."
  [t alpha]
  (let [n (ra/alength t)
        out (ra/alloc-like t n)
        a (double alpha)]
    (dotimes [i n]
      (ra/aset out i (* a (double (ra/aget t i)))))
    (let [shape (tshape t)]
      (if shape (register-tensor out shape) out))))

(defn tensor-eq?
  "Check if two tensors are approximately equal element-wise."
  ([a b] (tensor-eq? a b 1e-6))
  ([a b eps]
   (let [n (ra/alength a)]
     (and (= n (ra/alength b))
          (loop [i 0]
            (if (< i n)
              (if (< (Math/abs (- (double (ra/aget a i)) (double (ra/aget b i)))) (double eps))
                (recur (inc i))
                false)
              true))))))

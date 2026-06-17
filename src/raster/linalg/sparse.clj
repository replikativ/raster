(ns raster.linalg.sparse
  "Sparse vector type with polymorphic arithmetic dispatch.

   Uses sorted compressed format (sorted index + value arrays)
   for efficient element-wise operations that skip zeros.

   Extends raster.numeric/+, -, * for SparseVector types,
   demonstrating how raster's dispatch handles storage format
   specialization — like Julia's SparseArrays.jl.

   Usage:
     (require '[raster.linalg.sparse :refer [sparse-vector svec to-dense nnz]])
     (+ (svec 10 {0 1.0, 5 2.0}) (svec 10 {0 3.0, 7 4.0}))
     ;=> SparseVector[len=10, nnz=3]"
  (:refer-clojure :exclude [+ - * / aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.numeric]))

;; ================================================================
;; SparseVector type (sorted compressed: parallel index + value arrays)
;; ================================================================

(defvalue SparseVector (All [T]) [indices :- (Array int), values :- (Array T), length :- Long])

(defn sparse-vector
  "Create a sparse vector from index and value arrays.
   Indices must be sorted and unique."
  [^ints indices ^doubles values ^long length]
  (->SparseVector indices values length))

(defn svec
  "Create a sparse vector from a length and a map of {index value}.
   (svec 10 {0 1.0, 5 2.0, 9 3.0})"
  [^long length entries]
  (let [sorted (sort-by key entries)
        n (count sorted)
        idx (int-array (map key sorted))
        vals (double-array (map val sorted))]
    (->SparseVector idx vals length)))

(defn nnz
  "Number of non-zero entries."
  ^long [^SparseVector v]
  (alength (.-indices v)))

(defn to-dense
  "Convert sparse vector to dense double-array."
  ^doubles [^SparseVector v]
  (let [out (double-array (.-length v))
        idx (.-indices v)
        vals (.-values v)
        n (alength idx)]
    (dotimes [i n]
      (aset out (aget idx i) (aget vals i)))
    out))

(defmethod print-method SparseVector [^SparseVector v ^java.io.Writer w]
  (.write w (str "SparseVector[len=" (.-length v) ", nnz=" (alength (.-indices v)) "]")))

;; ================================================================
;; Merge helper for sparse + sparse
;; ================================================================

(defn- merge-sparse
  "Merge two sorted index arrays with a binary operation on values.
   Returns [new-indices new-values] with zero entries removed."
  [^ints ai ^doubles av ^ints bi ^doubles bv op]
  (let [an (alength ai) bn (alength bi)
        ;; Worst case: all indices are distinct
        max-n (clojure.core/+ an bn)
        ri (int-array max-n)
        rv (double-array max-n)]
    (loop [ia 0 ib 0 k 0]
      (cond
        (and (>= ia an) (>= ib bn))
        [(java.util.Arrays/copyOf ri (int k))
         (java.util.Arrays/copyOf rv (int k))]

        (>= ia an)
        (let [val (double (op 0.0 (aget bv ib)))]
          (if (not= val 0.0)
            (do (aset ri k (aget bi ib))
                (aset rv k val)
                (recur ia (inc ib) (inc k)))
            (recur ia (inc ib) k)))

        (>= ib bn)
        (let [val (double (op (aget av ia) 0.0))]
          (if (not= val 0.0)
            (do (aset ri k (aget ai ia))
                (aset rv k val)
                (recur (inc ia) ib (inc k)))
            (recur (inc ia) ib k)))

        :else
        (let [idx-a (aget ai ia) idx-b (aget bi ib)]
          (cond
            (< idx-a idx-b)
            (let [val (double (op (aget av ia) 0.0))]
              (if (not= val 0.0)
                (do (aset ri k idx-a)
                    (aset rv k val)
                    (recur (inc ia) ib (inc k)))
                (recur (inc ia) ib k)))

            (> idx-a idx-b)
            (let [val (double (op 0.0 (aget bv ib)))]
              (if (not= val 0.0)
                (do (aset ri k idx-b)
                    (aset rv k val)
                    (recur ia (inc ib) (inc k)))
                (recur ia (inc ib) k)))

            :else ;; equal indices
            (let [val (double (op (aget av ia) (aget bv ib)))]
              (if (not= val 0.0)
                (do (aset ri k idx-a)
                    (aset rv k val)
                    (recur (inc ia) (inc ib) (inc k)))
                (recur (inc ia) (inc ib) k)))))))))

;; ================================================================
;; Arithmetic: SparseVector × SparseVector
;; ================================================================

(deftm raster.numeric/+ [a :- SparseVector, b :- SparseVector] :- SparseVector
  (let [[idx vals] (merge-sparse (.-indices a) (.-values a)
                                 (.-indices b) (.-values b)
                                 clojure.core/+)]
    (->SparseVector idx vals (.-length a))))

(deftm raster.numeric/- [a :- SparseVector] :- SparseVector
  (let [vals (.-values a)
        n (alength vals)
        nv (double-array n)]
    (dotimes [i n] (aset nv i (clojure.core/- (aget vals i))))
    (->SparseVector (aclone (.-indices a)) nv (.-length a))))

(deftm raster.numeric/- [a :- SparseVector, b :- SparseVector] :- SparseVector
  (let [[idx vals] (merge-sparse (.-indices a) (.-values a)
                                 (.-indices b) (.-values b)
                                 clojure.core/-)]
    (->SparseVector idx vals (.-length a))))

;; ================================================================
;; Arithmetic: scalar × SparseVector
;; ================================================================

(deftm raster.numeric/* [c :- Number, a :- SparseVector] :- SparseVector
  (let [c (double c)
        vals (.-values a)
        n (alength vals)
        nv (double-array n)]
    (dotimes [i n] (aset nv i (clojure.core/* c (aget vals i))))
    (->SparseVector (aclone (.-indices a)) nv (.-length a))))

(deftm raster.numeric/* [a :- SparseVector, c :- Number] :- SparseVector
  (let [c (double c)
        vals (.-values a)
        n (alength vals)
        nv (double-array n)]
    (dotimes [i n] (aset nv i (clojure.core/* (aget vals i) c)))
    (->SparseVector (aclone (.-indices a)) nv (.-length a))))

;; ================================================================
;; Dot product and norm
;; ================================================================

(deftm dot [a :- SparseVector, b :- SparseVector] :- Double
  (let [ai (.-indices a) av (.-values a)
        bi (.-indices b) bv (.-values b)
        an (alength ai) bn (alength bi)]
    (loop [ia 0 ib 0 sum 0.0]
      (if (or (>= ia an) (>= ib bn))
        sum
        (let [idx-a (aget ai ia) idx-b (aget bi ib)]
          (cond
            (< idx-a idx-b) (recur (inc ia) ib sum)
            (> idx-a idx-b) (recur ia (inc ib) sum)
            :else (recur (inc ia) (inc ib)
                         (clojure.core/+ sum (clojure.core/* (aget av ia) (aget bv ib))))))))))

(deftm norm [a :- SparseVector] :- Double
  (let [vals (.-values a)
        n (alength vals)]
    (n/sqrt (loop [i 0 sum 0.0]
              (if (>= i n) sum
                  (let [v (aget vals i)]
                    (recur (inc i) (clojure.core/+ sum (clojure.core/* v v)))))))))

;; ================================================================
;; CSR Sparse Matrix (Compressed Sparse Row)
;; ================================================================

(defvalue CSRMatrix (All [T]) [rowptr :- (Array int), colidx :- (Array int), values :- (Array T),
                               nrows :- Long, ncols :- Long, nnz :- Long])

(defvalue COOMatrix (All [T]) [rowidx :- (Array int), colidx :- (Array int), values :- (Array T),
                               nrows :- Long, ncols :- Long, nnz :- Long])

(defn csr-matrix
  "Create a CSR matrix from row-ptr, col-idx, values arrays."
  [row-ptr col-idx values nrows ncols]
  (->CSRMatrix row-ptr col-idx values nrows ncols (alength values)))

(defmethod print-method CSRMatrix [^CSRMatrix m ^java.io.Writer w]
  (.write w (str "CSRMatrix[" (.-nrows m) "×" (.-ncols m) ", nnz=" (.-nnz m) "]")))

(defmethod print-method COOMatrix [^COOMatrix m ^java.io.Writer w]
  (.write w (str "COOMatrix[" (.-nrows m) "×" (.-ncols m) ", nnz=" (.-nnz m) "]")))

;; ================================================================
;; COO <-> CSR conversion
;; ================================================================

(deftm coo-to-csr [coo :- COOMatrix] :- CSRMatrix
  (let [nr (.-nrows coo)
        nc (.-ncols coo)
        nz (.-nnz coo)
        ri (.-rowidx coo)
        ci (.-colidx coo)
        vs (.-values coo)
        ;; Count entries per row
        row-counts (int-array nr)
        _ (dotimes [k nz]
            (let [r (aget ri k)]
              (aset row-counts r (unchecked-add-int (aget row-counts r) 1))))
        ;; Build row-ptr (cumulative sum)
        row-ptr (int-array (inc nr))
        _ (dotimes [i nr]
            (aset row-ptr (inc i) (unchecked-add-int (aget row-ptr i) (aget row-counts i))))
        ;; Fill col-idx and values in row order
        out-ci (int-array nz)
        out-vs (double-array nz)
        pos (int-array nr)
        _ (dotimes [k nz]
            (let [r (aget ri k)
                  p (unchecked-add-int (aget row-ptr r) (aget pos r))]
              (aset out-ci p (aget ci k))
              (aset out-vs p (aget vs k))
              (aset pos r (unchecked-add-int (aget pos r) 1))))]
    (->CSRMatrix row-ptr out-ci out-vs nr nc nz)))

(deftm csr-from-dense
  "Convert dense row-major matrix A[m,n] to CSR."
  [A :- (Array double) m :- Long n :- Long] :- CSRMatrix
  ;; First pass: count nonzeros
  (let [total-nnz (loop [i 0 cnt 0]
                    (if (>= i (clojure.core/* m n)) cnt
                        (if (not= (aget A i) 0.0)
                          (recur (inc i) (inc cnt))
                          (recur (inc i) cnt))))
        row-ptr (int-array (inc m))
        col-idx (int-array total-nnz)
        values (double-array total-nnz)]
    (loop [i 0 k 0]
      (when (< i m)
        (aset row-ptr i (int k))
        (let [k2 (loop [j 0 kk k]
                   (if (>= j n) kk
                       (let [v (aget A (clojure.core/+ (clojure.core/* i n) j))]
                         (if (not= v 0.0)
                           (do (aset col-idx kk (int j))
                               (aset values kk v)
                               (recur (inc j) (inc kk)))
                           (recur (inc j) kk)))))]
          (recur (inc i) k2))))
    (aset row-ptr m (int total-nnz))
    (->CSRMatrix row-ptr col-idx values m n total-nnz)))

(deftm csr-to-dense
  "Convert CSR matrix to dense row-major double-array."
  [A :- CSRMatrix] :- (Array double)
  (let [m (.-nrows A)
        n (.-ncols A)
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)
        out (double-array (clojure.core/* m n))]
    (dotimes [i m]
      (loop [k (aget rp i)]
        (when (< k (aget rp (inc i)))
          (let [j (aget ci k)]
            (aset out (clojure.core/+ (clojure.core/* i n) j) (aget vs k)))
          (recur (inc k)))))
    out))

;; ================================================================
;; Sparse matrix-vector multiply
;; ================================================================

(deftm spmv
  "Sparse matrix-vector multiply: y = alpha*A*x + beta*y.
  A is CSRMatrix, x[ncols], y[nrows]."
  [A :- CSRMatrix x :- (Array double) y :- (Array double)
   alpha :- Double beta :- Double] :- (Array double)
  (let [m (.-nrows A)
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)]
    (dotimes [i m]
      (let [sum (loop [k (aget rp i) s 0.0]
                  (if (>= k (aget rp (inc i))) s
                      (recur (inc k)
                             (clojure.core/+ s (clojure.core/* (aget vs k) (aget x (aget ci k)))))))]
        (aset y i (clojure.core/+ (clojure.core/* beta (aget y i)) (clojure.core/* alpha sum)))))
    y))

(deftm spmv-t
  "Sparse matrix-vector multiply (transpose): y = alpha*A^T*x + beta*y.
  A is CSRMatrix, x[nrows], y[ncols]."
  [A :- CSRMatrix x :- (Array double) y :- (Array double)
   alpha :- Double beta :- Double] :- (Array double)
  (let [m (.-nrows A)
        n (.-ncols A)
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)]
    ;; Scale y by beta first
    (dotimes [j n] (aset y j (clojure.core/* beta (aget y j))))
    ;; Accumulate A^T * x
    (dotimes [i m]
      (let [xi (clojure.core/* alpha (aget x i))]
        (loop [k (aget rp i)]
          (when (< k (aget rp (inc i)))
            (let [j (aget ci k)]
              (aset y j (clojure.core/+ (aget y j) (clojure.core/* (aget vs k) xi))))
            (recur (inc k))))))
    y))

(deftm spmm
  "Sparse-dense matrix multiply: C = A*B.
  A is CSRMatrix[m,k], B is dense[k,n] row-major, C is dense[m,n]."
  [A :- CSRMatrix B :- (Array double) C :- (Array double)
   n :- Long] :- (Array double)
  (let [m (.-nrows A)
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)]
    ;; Zero C
    (dotimes [i (clojure.core/* m n)] (aset C i 0.0))
    ;; C[i,j] = sum_k A[i,k] * B[k,j]
    (dotimes [i m]
      (loop [k (aget rp i)]
        (when (< k (aget rp (inc i)))
          (let [col (aget ci k)
                a-val (aget vs k)]
            (dotimes [j n]
              (aset C (clojure.core/+ (clojure.core/* i n) j)
                    (clojure.core/+ (aget C (clojure.core/+ (clojure.core/* i n) j))
                                    (clojure.core/* a-val (aget B (clojure.core/+ (clojure.core/* col n) j)))))))
          (recur (inc k)))))
    C))

;; ================================================================
;; CSR utilities
;; ================================================================

(deftm csr-scale
  "Scale all values in CSR matrix by scalar."
  [A :- CSRMatrix c :- Double] :- CSRMatrix
  (let [nz (.-nnz A)
        vs (.-values A)
        nv (double-array nz)]
    (dotimes [i nz] (aset nv i (clojure.core/* c (aget vs i))))
    (->CSRMatrix (.-rowptr A) (.-colidx A) nv (.-nrows A) (.-ncols A) nz)))

(deftm csr-diag
  "Extract diagonal of CSR matrix as double-array."
  [A :- CSRMatrix] :- (Array double)
  (let [n (min (.-nrows A) (.-ncols A))
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)
        d (double-array n)]
    (dotimes [i n]
      (loop [k (aget rp i)]
        (when (< k (aget rp (inc i)))
          (if (== (aget ci k) i)
            (aset d i (aget vs k))
            (recur (inc k))))))
    d))

(deftm csr-eye
  "Create CSR identity matrix of size n."
  [n :- Long] :- CSRMatrix
  (let [rp (int-array (inc n))
        ci (int-array n)
        vs (double-array n)]
    (dotimes [i n]
      (aset rp i (int i))
      (aset ci i (int i))
      (aset vs i 1.0))
    (aset rp n (int n))
    (->CSRMatrix rp ci vs n n n)))

(deftm csr-transpose
  "Transpose CSR matrix (CSR of A^T)."
  [A :- CSRMatrix] :- CSRMatrix
  (let [m (.-nrows A)
        n (.-ncols A)
        nz (.-nnz A)
        rp (.-rowptr A)
        ci (.-colidx A)
        vs (.-values A)
        ;; Count entries per column (= row of transpose)
        col-counts (int-array n)
        _ (dotimes [k nz]
            (let [j (aget ci k)]
              (aset col-counts j (unchecked-add-int (aget col-counts j) 1))))
        ;; Build new row-ptr
        new-rp (int-array (inc n))
        _ (dotimes [j n]
            (aset new-rp (inc j) (unchecked-add-int (aget new-rp j) (aget col-counts j))))
        new-ci (int-array nz)
        new-vs (double-array nz)
        pos (int-array n)
        _ (dotimes [i m]
            (loop [k (aget rp i)]
              (when (< k (aget rp (inc i)))
                (let [j (aget ci k)
                      p (unchecked-add-int (aget new-rp j) (aget pos j))]
                  (aset new-ci p (int i))
                  (aset new-vs p (aget vs k))
                  (aset pos j (unchecked-add-int (aget pos j) 1)))
                (recur (inc k)))))]
    (->CSRMatrix new-rp new-ci new-vs n m nz)))

;; Per-row merge using a marker array (tag = i+1) so no per-row clearing is
;; needed and column order within a row is irrelevant. A and B must share shape.

(deftm csr-add
  "Linear combination alpha*A + beta*B of two CSR matrices (union of patterns)."
  [A :- CSRMatrix alpha :- Double B :- CSRMatrix beta :- Double] :- CSRMatrix
  (let [n   (.-nrows A)
        nc  (.-ncols A)
        arp (.-rowptr A) aci (.-colidx A) avs (.-values A)
        brp (.-rowptr B) bci (.-colidx B) bvs (.-values B)
        mark (int-array nc)
        rowcnt (int-array n)
        ;; pass 1: union size per row
        _ (dotimes [i n]
            (let [tag (inc i)]
              (loop [k (aget arp i)]
                (when (< k (aget arp (inc i)))
                  (let [j (aget aci k)]
                    (when (not (n/== (aget mark j) tag))
                      (aset mark j tag) (aset rowcnt i (n/+ (aget rowcnt i) 1))))
                  (recur (inc k))))
              (loop [k (aget brp i)]
                (when (< k (aget brp (inc i)))
                  (let [j (aget bci k)]
                    (when (not (n/== (aget mark j) tag))
                      (aset mark j tag) (aset rowcnt i (n/+ (aget rowcnt i) 1))))
                  (recur (inc k))))))
        rowptr (int-array (inc n))
        _ (dotimes [i n] (aset rowptr (inc i) (n/+ (aget rowptr i) (aget rowcnt i))))
        total (aget rowptr n)
        out-ci (int-array total)
        out-vs (double-array total)
        acc (double-array nc)
        mark2 (int-array nc)]                ; separate marker for pass 2
    ;; pass 2: accumulate values, list touched cols in appearance order
    (dotimes [i n]
      (let [tag (inc i)
            start (aget rowptr i)
            wa (loop [k (aget arp i) w start]
                 (if (< k (aget arp (inc i)))
                   (let [j (aget aci k)]
                     (if (n/== (aget mark2 j) tag)
                       (do (aset acc j (n/+ (aget acc j) (n/* alpha (aget avs k)))) (recur (inc k) w))
                       (do (aset mark2 j tag) (aset acc j (n/* alpha (aget avs k)))
                           (aset out-ci w (int j)) (recur (inc k) (inc w)))))
                   w))
            wb (loop [k (aget brp i) w wa]
                 (if (< k (aget brp (inc i)))
                   (let [j (aget bci k)]
                     (if (n/== (aget mark2 j) tag)
                       (do (aset acc j (n/+ (aget acc j) (n/* beta (aget bvs k)))) (recur (inc k) w))
                       (do (aset mark2 j tag) (aset acc j (n/* beta (aget bvs k)))
                           (aset out-ci w (int j)) (recur (inc k) (inc w)))))
                   w))]
        (loop [p start] (when (< p wb) (aset out-vs p (aget acc (aget out-ci p))) (recur (inc p))))))
    (->CSRMatrix rowptr out-ci out-vs n nc total)))

(deftm csr-hadamard
  "Elementwise (Hadamard) product A∘B of two CSR matrices (intersection of patterns)."
  [A :- CSRMatrix B :- CSRMatrix] :- CSRMatrix
  (let [n   (.-nrows A)
        nc  (.-ncols A)
        arp (.-rowptr A) aci (.-colidx A) avs (.-values A)
        brp (.-rowptr B) bci (.-colidx B) bvs (.-values B)
        mark (int-array nc)
        accb (double-array nc)
        rowcnt (int-array n)
        ;; pass 1: intersection size per row
        _ (dotimes [i n]
            (let [tag (inc i)]
              (loop [k (aget brp i)]
                (when (< k (aget brp (inc i))) (aset mark (aget bci k) tag) (recur (inc k))))
              (loop [k (aget arp i)]
                (when (< k (aget arp (inc i)))
                  (when (n/== (aget mark (aget aci k)) tag)
                    (aset rowcnt i (n/+ (aget rowcnt i) 1)))
                  (recur (inc k))))))
        rowptr (int-array (inc n))
        _ (dotimes [i n] (aset rowptr (inc i) (n/+ (aget rowptr i) (aget rowcnt i))))
        total (aget rowptr n)
        out-ci (int-array total)
        out-vs (double-array total)]
    (dotimes [i n]
      (let [tag (inc i)]
        (loop [k (aget brp i)]
          (when (< k (aget brp (inc i)))
            (let [j (aget bci k)] (aset mark j tag) (aset accb j (aget bvs k)))
            (recur (inc k))))
        (loop [k (aget arp i) w (aget rowptr i)]
          (if (< k (aget arp (inc i)))
            (let [j (aget aci k)]
              (if (n/== (aget mark j) tag)
                (do (aset out-ci w (int j)) (aset out-vs w (n/* (aget avs k) (aget accb j)))
                    (recur (inc k) (inc w)))
                (recur (inc k) w)))
            nil))))
    (->CSRMatrix rowptr out-ci out-vs n nc total)))

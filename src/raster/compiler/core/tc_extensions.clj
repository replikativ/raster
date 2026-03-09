(ns raster.compiler.core.tc-extensions
  "TypedClojure extensions for Raster type checking.

  1. Value type propagation: (* 32 64) → (t/Val 2048) for shape inference.
  2. Numeric promotion: (+ Long Integer) → Long instead of Number.
     Mirrors Julia/Raster's runtime promotion rules at type-check time.

  Must be loaded before type-checking code that relies on shape inference.
  Safe to load multiple times (defmethod is idempotent for same dispatch val)."
  (:require [clojure.core.typed :as t]
            [typed.clj.checker.check :as chk]
            [typed.clj.checker.subtype :as sub]
            [typed.cljc.analyzer :as ana2]
            [typed.cljc.checker.type-rep :as r]
            [typed.cljc.checker.type-ctors :as c]
            [typed.cljc.checker.utils :as u]
            [typed.cljc.checker.check-below :as below]
            [typed.cljc.checker.check :as cljc-chk]
            [typed.cljc.checker.check.unanalyzed :refer [defuspecial]]
            [typed.clj.checker.check.unanalyzed :as unan]))

;; ================================================================
;; Annotations for unannotated vars used in deftm bodies
;; ================================================================

;; Constants
(t/ann raster.numeric/pi Double)
(t/ann raster.numeric/e Double)
(t/ann raster.numeric/pos-inf Double)
(t/ann raster.numeric/neg-inf Double)
(t/ann raster.numeric/nan-val Double)

;; Clojure core missing annotations
(t/ann ^:no-check clojure.core/object-array [(t/U Long (t/Seqable t/Any)) :-> (Array Object)])
(t/ann ^:no-check clojure.core/long-array [(t/U Long (t/Seqable Long)) :-> (Array long)])
(t/ann ^:no-check clojure.core/float-array [(t/U Long (t/Seqable t/Num)) :-> (Array float)])
(t/ann ^:no-check clojure.core/conj! (t/All [x] [(t/Transient x) x :-> (t/Transient x)]))
(t/ann ^:no-check clojure.core/transient (t/All [x] [(t/Coll x) :-> (t/Transient x)]))
(t/ann ^:no-check clojure.core/persistent! (t/All [x] [(t/Transient x) :-> (t/Coll x)]))

;; Raster symbolic computation
(t/ann ^:no-check raster.sym.core/unwrap [t/Any :-> t/Any])
;; ->Sym is a record constructor, checked separately
(t/ann raster.sym.core/->Sym [t/Any t/Any t/Any :-> t/Any])

;; Raster AD
(t/ann ^:no-check raster.ad.reverse/value+grad [[t/Any :-> t/Any] :-> [t/Any :-> t/Any]])
(t/ann raster.ad.jet/->Jet [t/Any :-> t/Any])
(t/ann ^:no-check raster.ad.jet/primal [t/Any :-> t/Any])
(t/ann ^:no-check raster.ad.jet/taylor-derivative [t/Any Long :-> t/Any])

;; Raster numeric internals
(t/ann ^:no-check raster.numeric/redispatch [t/Any t/Any :* :-> t/Any])

;; Raster LAPACK/BLAS helpers (int-seg takes 1 arg: value; byte-seg takes 2: alloc + capacity)
(t/ann ^:no-check raster.linalg.lapack/int-seg [Long :-> t/Any])
(t/ann ^:no-check raster.linalg.lapack/byte-seg [Long Long :-> t/Any])

;; ODE solver constants (SDIRK4 Butcher tableau coefficients)
(t/ann ^:no-check raster.ode.core/sdirk4-a31 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a32 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a41 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a42 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a43 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a51 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a52 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a53 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-a54 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-b1 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-b2 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-b3 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-b4 Double)
(t/ann ^:no-check raster.ode.core/sdirk4-b5 Double)

;; Scientific computing constants
(t/ann ^:no-check raster.sci.distributions/LOG_2PI Double)
(t/ann ^:no-check raster.sci.distributions/SQRT_2 Double)
(t/ann ^:no-check raster.sci.distributions/INV_SQRT_2 Double)
(t/ann ^:no-check raster.sci.distributions/LOG_2 Double)
(t/ann ^:no-check raster.sci.distributions/INV_SQRT_PI Double)
(t/ann raster.sci.distributions/lgamma [Double :-> Double])
(t/ann raster.sci.distributions/regularized-beta-cf [Double Double Double :-> Double])
(t/ann ^:no-check raster.sci.special/INV_SQRT_PI Double)
(t/ann ^:no-check raster.sci.special/SQ2OPI Double)
(t/ann raster.sci.special/lgamma [Double :-> Double])
(t/ann raster.sci.special/digamma [Double :-> Double])
(t/ann raster.sci.special/trigamma [Double :-> Double])
(t/ann raster.sci.special/betainc [Double Double Double :-> Double])

;; Type promotion
(t/ann ^:no-check raster.types.promote/promote [t/Any t/Any :-> t/Any])

;; Symbolic computation
(t/ann ^:no-check raster.sym.fn-algebra/D t/Any)
(t/ann ^:no-check raster.sym.fn-algebra/partial-d [t/Any Long :-> t/Any])

;; Geometric algebra helpers
(t/ann ^:no-check raster.ga.core/blade-index [t/Any t/Any :-> Long])
(t/ann ^:no-check raster.ga.core/pseudoscalar [t/Any :-> t/Any])
(t/ann ^:no-check raster.ga.core/get-compiled-product [t/Any :-> t/Any])
(t/ann ^:no-check raster.ga.core/get-compiled-filtered [t/Any t/Any :-> t/Any])
(t/ann ^:no-check raster.ga.core/get-compiled-involution [t/Any :-> t/Any])
(t/ann ^:no-check raster.ga.core/get-compiled-grade-select [t/Any :-> t/Any])

;; AD internals
(t/ann ^:no-check raster.ad.jet/jet-const [t/Any :-> t/Any])
(t/ann ^:no-check raster.ad.forward/zero-partials [t/Any :-> t/Any])

;; Par constants
(t/ann ^:no-check raster.par/SM-MIX2 Long)

;; BLAS/Panama FFI internals
(t/ann ^:no-check raster.linalg.blas/CBLAS_ROW_MAJOR Integer)
(t/ann ^:no-check raster.linalg.blas/CBLAS_NO_TRANS Integer)
(t/ann ^:no-check raster.linalg.blas/CBLAS_TRANS Integer)
(t/ann ^:no-check raster.linalg.blas/dgemm-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/sgemm-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/dgemv-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/sgemv-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/daxpy-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/saxpy-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/dger-mh t/Any)
(t/ann ^:no-check raster.linalg.blas/sger-mh t/Any)

;; DL optimizer (used in deftm bodies)
(t/ann raster.dl.optim/sgd-step! [(Array double) (Array double) Double :-> t/Any])

(defn- check-expr-from-opts
  "Extract the check-expr function from TC opts map."
  [opts]
  (get opts (keyword "typed.cljc.checker.check" "check-expr")))

(defn- make-value-propagating-handler
  "Create an invoke-special handler that propagates Value types through
  a numeric operation. When all args are (t/Val <number>), computes the
  result and returns (t/Val result). Otherwise returns nil to fall through
  to normal type checking."
  [op-fn]
  (fn [{:keys [args] :as expr} expected opts]
    (let [check-expr (check-expr-from-opts opts)]
      (when (and check-expr (seq args))
        (let [cargs (mapv #(check-expr % nil opts) args)
              types (mapv (comp r/ret-t u/expr-type) cargs)]
          (when (every? r/Value? types)
            (let [vals (map :val types)]
              (when (every? number? vals)
                (let [result (apply op-fn vals)]
                  (-> expr
                      (assoc :args cargs
                             u/expr-type (below/maybe-check-below
                                          (r/ret (r/Value-maker result))
                                          expected
                                          opts))))))))))))

;; Register Value-propagating handlers for core arithmetic ops.
;; Each returns nil when args aren't all Value types, falling through
;; to TC's normal annotation-based checking.

(def ^:private arithmetic-ops
  {'clojure.core/*    *
   'clojure.core/+    +
   'clojure.core/-    -
   'clojure.core/quot quot
   'clojure.core/rem  rem
   'clojure.core/mod  mod
   'clojure.core/max  max
   'clojure.core/min  min})

(doseq [[sym op-fn] arithmetic-ops
        :let [handler (make-value-propagating-handler op-fn)]]
  (defmethod chk/-invoke-special sym
    [expr expected opts]
    (handler expr expected opts)))

;; Unary ops: inc, dec
(defmethod chk/-invoke-special 'clojure.core/inc
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 1 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)
            t (r/ret-t (u/expr-type (first cargs)))]
        (when (and (r/Value? t) (number? (:val t)))
          (-> expr
              (assoc :args cargs
                     u/expr-type (below/maybe-check-below
                                  (r/ret (r/Value-maker (inc (:val t))))
                                  expected
                                  opts))))))))

(defmethod chk/-invoke-special 'clojure.core/dec
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 1 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)
            t (r/ret-t (u/expr-type (first cargs)))]
        (when (and (r/Value? t) (number? (:val t)))
          (-> expr
              (assoc :args cargs
                     u/expr-type (below/maybe-check-below
                                  (r/ret (r/Value-maker (dec (:val t))))
                                  expected
                                  opts))))))))

;; ================================================================
;; Numeric promotion for raster.numeric ops
;;
;; When raster.numeric/+ sees (+ Long Integer), the deftm dispatch
;; table has [Long Long → Long] and [Integer Integer → Integer] but
;; no cross-type entry. TC falls to [Number Number → Number].
;;
;; These invoke-special handlers apply Julia/Raster promotion rules
;; at type-check time: mixed integer types → Long, any float type
;; present → Double (or Float if all are Float).
;; ================================================================

(def ^:private numeric-type-classes
  "Map of TC type class symbols to their numeric category and rank.
  Includes both boxed (java.lang.Long) and primitive (long) forms,
  plus BigInt/BigInteger treated as rank 3 (Long)."
  {'byte                  {:category :integer :rank 0}
   'java.lang.Byte        {:category :integer :rank 0}
   'short                 {:category :integer :rank 1}
   'java.lang.Short       {:category :integer :rank 1}
   'int                   {:category :integer :rank 2}
   'java.lang.Integer     {:category :integer :rank 2}
   'long                  {:category :integer :rank 3}
   'java.lang.Long        {:category :integer :rank 3}
   'clojure.lang.BigInt   {:category :integer :rank 3}
   'java.math.BigInteger  {:category :integer :rank 3}
   'float                 {:category :float   :rank 4}
   'java.lang.Float       {:category :float   :rank 4}
   'double                {:category :float   :rank 5}
   'java.lang.Double      {:category :float   :rank 5}
   ;; Number is the abstract base — conservatively treat as double
   'java.lang.Number      {:category :float   :rank 5}})

(defn- resolve-tc-type
  "Resolve TC Name types (type aliases like AnyInteger) to their definitions."
  [t opts]
  (if (instance? (resolve 'typed.cljc.checker.type_rep.Name) t)
    (try (c/resolve-Name t opts)
         (catch Exception _ t))
    t))

(defn- tc-type->numeric-info
  "Extract numeric category and rank from a TC type.
  Returns {:category :integer/:float, :rank int} or nil for non-numeric.
  Handles Value, RClass, Name (aliases), and Union types (like AnyInteger)."
  [t opts]
  (let [t (resolve-tc-type t opts)]
    (cond
      ;; Value types: (t/Val 42) → check class of the value
      (r/Value? t)
      (let [cls (class (:val t))]
        (get numeric-type-classes (symbol (.getName ^Class cls))))

      ;; RClass types: direct class reference (Long, Integer, Double, etc.)
      (instance? (resolve 'typed.cljc.checker.type_rep.RClass) t)
      (get numeric-type-classes (:the-class t))

      ;; Union types: AnyInteger = (U Integer Long BigInt ...).
      ;; If all members are numeric, use the max rank.
      (r/Union? t)
      (let [member-infos (keep #(tc-type->numeric-info % opts) (:types t))]
        (when (seq member-infos)
          (apply max-key :rank member-infos)))

      ;; Subtype check as fallback: if t <: Number, treat as generic Number
      :else
      (try
        (let [long-t (c/RClass-of 'java.lang.Long opts)
              number-t (c/RClass-of 'java.lang.Number opts)]
          (cond
            (sub/subtype? t long-t opts)
            {:category :integer :rank 3}

            (sub/subtype? t number-t opts)
            {:category :float :rank 5}  ;; conservative: any Number → Double

            :else nil))
        (catch Exception _ nil)))))

(defn- promote-numeric-types
  "Given checked arg types, compute the promoted return type using
  Julia-style numeric widening. Returns a TC type or nil."
  [types opts]
  (let [infos (mapv #(tc-type->numeric-info % opts) types)]
    (when (every? some? infos)
      (let [max-rank (apply max (map :rank infos))
            result-class (case (int max-rank)
                           0 'java.lang.Long    ;; all Byte → promote to Long
                           1 'java.lang.Long    ;; any Short → promote to Long
                           2 'java.lang.Long    ;; any Integer → promote to Long
                           3 'java.lang.Long    ;; all Long
                           4 'java.lang.Float   ;; any Float (no Double)
                           5 'java.lang.Double  ;; any Double
                           nil)]
        (when result-class
          (c/RClass-of result-class opts))))))

(defn- make-promoting-handler
  "Create an invoke-special handler that applies numeric promotion.
  First tries Value propagation (constant folding), then falls back
  to promotion-based type inference."
  [op-fn]
  (let [value-handler (make-value-propagating-handler op-fn)]
    (fn [{:keys [args] :as expr} expected opts]
      ;; Try constant propagation first
      (or (value-handler expr expected opts)
          ;; Fall back to numeric promotion
          (let [check-expr (check-expr-from-opts opts)]
            (when (and check-expr (seq args))
              (let [cargs (mapv #(check-expr % nil opts) args)
                    types (mapv (comp r/ret-t u/expr-type) cargs)
                    promoted (promote-numeric-types types opts)]
                (when promoted
                  (-> expr
                      (assoc :args cargs
                             u/expr-type (below/maybe-check-below
                                          (r/ret promoted)
                                          expected
                                          opts)))))))))))

;; Register promoting handlers for raster.numeric arithmetic ops.
;; These override TC's normal overload resolution with promotion-aware
;; type inference, matching the runtime behavior.
(def ^:private raster-numeric-ops
  {'raster.numeric/*    *
   'raster.numeric/+    +
   'raster.numeric/-    -
   'raster.numeric//    /})

(doseq [[sym op-fn] raster-numeric-ops
        :let [handler (make-promoting-handler op-fn)]]
  (defmethod chk/-invoke-special sym
    [expr expected opts]
    (handler expr expected opts)))

;; ================================================================
;; Array access with integer-widening for index arguments
;;
;; raster.arrays/aget and aset are annotated with Long index params,
;; but dotimes loop variables are AnyInteger (U Integer Long BigInt...).
;; These handlers accept any integer type for the index and infer the
;; element type from the array argument.
;; ================================================================

(defn- array-type->element-type
  "Extract the element TC type from an array type.
  Handles PrimitiveArray (Array3) types — returns the :output-type."
  [t]
  (when (instance? (resolve 'typed.cljc.checker.type_rep.PrimitiveArray) t)
    (:output-type t)))

(defmethod chk/-invoke-special 'raster.arrays/aget
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 2 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)
            [arr-type idx-type] (mapv (comp r/ret-t u/expr-type) cargs)
            elem-type (array-type->element-type arr-type)
            idx-info (tc-type->numeric-info idx-type opts)]
        ;; Accept if index is any integer type and array is a known array type
        (when (and elem-type idx-info (= :integer (:category idx-info)))
          (-> expr
              (assoc :args cargs
                     u/expr-type (below/maybe-check-below
                                  (r/ret elem-type)
                                  expected
                                  opts))))))))

(defmethod chk/-invoke-special 'raster.arrays/aset
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 3 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)
            [arr-type idx-type _val-type] (mapv (comp r/ret-t u/expr-type) cargs)
            elem-type (array-type->element-type arr-type)
            idx-info (tc-type->numeric-info idx-type opts)]
        ;; Accept if index is any integer type and array is a known array type
        (when (and elem-type idx-info (= :integer (:category idx-info)))
          (-> expr
              (assoc :args cargs
                     u/expr-type (below/maybe-check-below
                                  (r/ret elem-type)
                                  expected
                                  opts))))))))

;; ================================================================
;; Math static method widening
;;
;; Math/sqrt etc. take primitive double but TC sometimes infers Number
;; (e.g. from promoted arithmetic). Accept any numeric type.
;; ================================================================

(defn- make-math-unary-handler
  "Create a host-call-special handler for a Math static method that
  takes one numeric arg and returns double."
  []
  (fn [{:keys [args] :as expr} expected opts]
    (let [check-expr (check-expr-from-opts opts)]
      (when (and check-expr (= 1 (count args)))
        (let [cargs (mapv #(check-expr % nil opts) args)
              arg-type (r/ret-t (u/expr-type (first cargs)))
              info (tc-type->numeric-info arg-type opts)]
          (when info
            (-> expr
                (assoc :args cargs
                       u/expr-type (below/maybe-check-below
                                    (r/ret (c/RClass-of 'double opts))
                                    expected
                                    opts)))))))))

(defn- make-math-binary-handler
  "Create a host-call-special handler for a Math static method that
  takes two numeric args and returns double."
  []
  (fn [{:keys [args] :as expr} expected opts]
    (let [check-expr (check-expr-from-opts opts)]
      (when (and check-expr (= 2 (count args)))
        (let [cargs (mapv #(check-expr % nil opts) args)
              types (mapv (comp r/ret-t u/expr-type) cargs)
              infos (mapv #(tc-type->numeric-info % opts) types)]
          (when (every? some? infos)
            (-> expr
                (assoc :args cargs
                       u/expr-type (below/maybe-check-below
                                    (r/ret (c/RClass-of 'double opts))
                                    expected
                                    opts)))))))))

;; Unary Math methods: accept any numeric type, return double
(let [handler (make-math-unary-handler)]
  (doseq [method '[java.lang.Math/sqrt java.lang.Math/exp java.lang.Math/log
                   java.lang.Math/sin java.lang.Math/cos java.lang.Math/tan
                   java.lang.Math/asin java.lang.Math/acos java.lang.Math/atan
                   java.lang.Math/sinh java.lang.Math/cosh java.lang.Math/tanh
                   java.lang.Math/abs java.lang.Math/ceil java.lang.Math/floor
                   java.lang.Math/round java.lang.Math/signum
                   java.lang.Math/log10 java.lang.Math/log1p
                   java.lang.Math/expm1 java.lang.Math/cbrt]]
    (defmethod chk/-host-call-special [:static-call method]
      [expr expected opts]
      (handler expr expected opts))))

;; Binary Math methods: accept any numeric types, return double
(let [handler (make-math-binary-handler)]
  (doseq [method '[java.lang.Math/max java.lang.Math/min java.lang.Math/pow
                   java.lang.Math/atan2 java.lang.Math/hypot]]
    (defmethod chk/-host-call-special [:static-call method]
      [expr expected opts]
      (handler expr expected opts))))

;; Void-returning static methods: java.util.Arrays/fill returns void,
;; which causes TC's Type->Class assertion to fail. Handle it explicitly.
(defmethod chk/-host-call-special [:static-call 'java.util.Arrays/fill]
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 2 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)]
        (-> expr
            (assoc :args cargs
                   u/expr-type (below/maybe-check-below
                                (r/ret (r/Bottom))
                                expected
                                opts)))))))

(defmethod chk/-host-call-special [:static-call 'java.lang.System/arraycopy]
  [{:keys [args] :as expr} expected opts]
  (let [check-expr (check-expr-from-opts opts)]
    (when (and check-expr (= 5 (count args)))
      (let [cargs (mapv #(check-expr % nil opts) args)]
        (-> expr
            (assoc :args cargs
                   u/expr-type (below/maybe-check-below
                                (r/ret (r/Bottom))
                                expected
                                opts)))))))

;; ================================================================
;; Par combinator type rules (defuspecial)
;;
;; broadcast, reduce!, scan are compiler directives processed by the
;; walker before macroexpansion. Their stub macros throw, which causes
;; TC to produce TCErrors. These defuspecial handlers rewrite the forms
;; into TC-checkable equivalents so TC can verify types through them.
;; ================================================================

(defn- broadcast-tc-expansion
  "Rewrite (broadcast [x y ...] body) into a TC-checkable let+dotimes form.
  Semantics: element-wise map over input arrays, return new array."
  [bindings body]
  (let [first-arr (first bindings)
        idx-sym   '__tc_i
        out-sym   '__tc_out]
    `(let [~out-sym (raster.arrays/alloc-like ~first-arr
                                              (raster.arrays/alength ~first-arr))]
       (dotimes [~idx-sym (raster.arrays/alength ~first-arr)]
         (let [~@(mapcat (fn [sym] [sym `(raster.arrays/aget ~sym ~idx-sym)])
                         bindings)]
           (raster.arrays/aset ~out-sym ~idx-sym ~body)))
       ~out-sym)))

(defuspecial defuspecial__broadcast
  "Type rule for raster.core/broadcast"
  [{[_ bindings body] :form :keys [env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        rewritten  (broadcast-tc-expansion bindings body)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.core/broadcast
 'raster.compiler.core.tc-extensions/defuspecial__broadcast)

(defn- reduce!-tc-expansion
  "Rewrite (reduce! [acc init] [x y ...] body) into TC-checkable form.
  Semantics: fold over arrays, accumulating scalar result."
  [[acc-sym init] bindings body]
  (let [first-arr (first bindings)
        idx-sym   '__tc_i]
    `(let [~acc-sym ~init]
       (dotimes [~idx-sym (raster.arrays/alength ~first-arr)]
         (let [~@(mapcat (fn [sym] [sym `(raster.arrays/aget ~sym ~idx-sym)])
                         bindings)
               ~acc-sym ~body]
           nil))
       ~acc-sym)))

(defuspecial defuspecial__reduce!
  "Type rule for raster.core/reduce!"
  [{[_ acc-binding bindings body] :form :keys [env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        rewritten  (reduce!-tc-expansion acc-binding bindings body)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.core/reduce!
 'raster.compiler.core.tc-extensions/defuspecial__reduce!)

(defn- scan-tc-expansion
  "Rewrite (scan [acc init] [x ...] body) into TC-checkable form.
  Semantics: prefix scan, returns array of accumulated values."
  [[acc-sym init] bindings body]
  (let [first-arr (first bindings)
        idx-sym   '__tc_i
        out-sym   '__tc_out]
    `(let [~out-sym (raster.arrays/alloc-like ~first-arr
                                              (raster.arrays/alength ~first-arr))
           ~acc-sym ~init]
       (dotimes [~idx-sym (raster.arrays/alength ~first-arr)]
         (let [~@(mapcat (fn [sym] [sym `(raster.arrays/aget ~sym ~idx-sym)])
                         bindings)
               ~acc-sym ~body]
           (raster.arrays/aset ~out-sym ~idx-sym ~acc-sym)))
       ~out-sym)))

(defuspecial defuspecial__scan
  "Type rule for raster.core/scan"
  [{[_ acc-binding bindings body] :form :keys [env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        rewritten  (scan-tc-expansion acc-binding bindings body)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.core/scan
 'raster.compiler.core.tc-extensions/defuspecial__scan)

;; ================================================================
;; Par combinator IR forms (raster.par/pmap, map!, reduce)
;;
;; These IR forms appear in post-walker code (fixpoint stage).
;; TC needs to type-check through them for binding-tag inference.
;; Each is rewritten to a TC-checkable sequential equivalent.
;; ================================================================

(defn- cast->alloc-fn
  "Map a cast symbol to the corresponding array allocation function."
  [cast-sym]
  (case cast-sym
    float  'clojure.core/float-array
    double 'clojure.core/double-array
    long   'clojure.core/long-array
    int    'clojure.core/int-array
    'clojure.core/double-array))

(defn- pmap-tc-expansion
  "Rewrite (raster.par/pmap idx bound cast body) into TC-checkable form.
  Semantics: allocate array, fill with body result, return array."
  [idx-sym bound-expr cast-fn body-expr]
  (let [out-sym   '__tc_pmap_out
        n-sym     '__tc_pmap_n
        alloc-fn  (cast->alloc-fn cast-fn)
        store-expr (if cast-fn `(~cast-fn ~body-expr) `(double ~body-expr))]
    `(let [~n-sym (int ~bound-expr)
           ~out-sym (~alloc-fn ~n-sym)]
       (dotimes [~idx-sym ~n-sym]
         (clojure.core/aset ~out-sym ~idx-sym ~store-expr))
       ~out-sym)))

(defuspecial defuspecial__pmap
  "Type rule for raster.par/pmap (pure parallel map IR form)"
  [{[_ idx-sym bound-expr cast-fn body-expr] :form :keys [env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        rewritten  (pmap-tc-expansion idx-sym bound-expr cast-fn body-expr)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.par/pmap
 'raster.compiler.core.tc-extensions/defuspecial__pmap)

(defn- reduce-tc-expansion
  "Rewrite (raster.par/reduce acc init idx bound body) into TC-checkable loop."
  [acc-sym init-expr idx-sym bound-expr body-expr]
  (let [n-sym '__tc_reduce_n]
    `(let [~n-sym (int ~bound-expr)]
       (loop [~idx-sym (int 0) ~acc-sym ~init-expr]
         (if (clojure.core/< ~idx-sym ~n-sym)
           (recur (clojure.core/unchecked-inc-int ~idx-sym) ~body-expr)
           ~acc-sym)))))

(defuspecial defuspecial__reduce
  "Type rule for raster.par/reduce (parallel reduction)"
  [{[_ acc-sym init-expr idx-sym bound-expr body-expr] :form :keys [env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        rewritten  (reduce-tc-expansion acc-sym init-expr idx-sym bound-expr body-expr)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.par/reduce
 'raster.compiler.core.tc-extensions/defuspecial__reduce)

(defn- map!-tc-expansion
  "Rewrite (raster.par/map! out idx bound cast body) into TC-checkable form.
  Also handles offset variant: (raster.par/map! out idx bound :offset base cast body)"
  [out-sym idx-sym bound-expr offset-base cast-fn body-expr]
  (let [n-sym      '__tc_map_n
        store-expr (if cast-fn `(~cast-fn ~body-expr) body-expr)]
    (if offset-base
      (let [base-sym '__tc_map_base]
        `(let [~n-sym (int ~bound-expr)
               ~base-sym (int ~offset-base)]
           (dotimes [~idx-sym ~n-sym]
             (clojure.core/aset ~out-sym
                                (clojure.core/unchecked-add-int ~base-sym ~idx-sym)
                                ~store-expr))
           ~out-sym))
      `(let [~n-sym (int ~bound-expr)]
         (dotimes [~idx-sym ~n-sym]
           (clojure.core/aset ~out-sym ~idx-sym ~store-expr))
         ~out-sym))))

(defuspecial defuspecial__map!
  "Type rule for raster.par/map! (imperative parallel map)"
  [{:keys [form env] :as expr} expected opts]
  (let [check-expr (::cljc-chk/check-expr opts)
        args (rest form)
        ;; Parse standard vs offset variant
        [out-sym idx-sym bound-expr offset-base cast-fn body-expr]
        (if (= :offset (nth args 3 nil))
          ;; Offset: (map! out idx bound :offset base cast body)
          [(nth args 0) (nth args 1) (nth args 2) (nth args 4) (nth args 5) (nth args 6)]
          ;; Standard: (map! out idx bound cast body)
          [(nth args 0) (nth args 1) (nth args 2) nil (nth args 3) (nth args 4)])
        rewritten (map!-tc-expansion out-sym idx-sym bound-expr offset-base cast-fn body-expr)]
    (-> rewritten
        (ana2/unanalyzed env opts)
        (ana2/inherit-top-level expr)
        (check-expr expected opts))))

(unan/install-defuspecial
 'raster.par/map!
 'raster.compiler.core.tc-extensions/defuspecial__map!)

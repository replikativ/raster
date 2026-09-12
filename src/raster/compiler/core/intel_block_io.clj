(ns raster.compiler.core.intel-block-io
  "Physical constraints for Intel's row-major FP16 matrix block-I/O lowering.

  These are target requirements, not tensor semantics. All scalar conditions use the existing
  checked executable expression algebra. See cl_intel_subgroup_2d_block_io §6.13.X.6."
  (:require [clojure.walk :as walk]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-precondition :as precondition]))

(defn- partition-preconditions
  "Accept the canonical full-K or uniform grid-Z partition. Fragment alignment must hold at
  every partition boundary, not merely at the full surface width. Matrix-plan analysis also
  verifies that these declared bounds are the actual loop bounds."
  [kernel-body k]
  (let [[lower upper] (get-in kernel-body [:attributes :iteration-range :k])]
    (if (= [0 k] [lower upper])
      []
      (let [[group chunk] (:arguments lower)
            limit (second (:arguments upper))
            group-binding (some #(when (= group (:id %)) %) (:indices kernel-body))
            chunk-parameter (some #(when (= chunk (:id %)) %) (:parameters kernel-body))]
        (when-not (and (= lower (body/expression :mul group chunk))
                       (= :group (:source group-binding)) (= 2 (:axis group-binding))
                       (= :scalar (:kind chunk-parameter))
                       (or (= limit k) (integer? limit))
                       (= upper (body/expression :min (body/expression :add lower chunk) limit)))
          (throw (ex-info "Intel matrix K partition requires a canonical uniform grid-Z range"
                          {:reason :matrix-partition-contract :lower lower :upper upper})))
        (cond-> [{:expression chunk :op :> :value 0}
                 {:expression (body/expression :mod chunk 16) :op := :value 0}]
          (integer? limit) (conj {:expression k :op := :value limit}))))))

(defn matrix-preconditions
  "Conditions on the two dense half surfaces A[M,K] and B[K,N]. Optional slice strides are in
  elements and describe rebasing an input pointer between workgroups, not the scalar C stores."
  ([m n k] (matrix-preconditions m n k []))
  ([m n k input-slice-strides]
   (into [{:expression m :op :>= :value 1}
          {:expression m :op :<= :value 16777216}
          {:expression n :op :>= :value 32}
          {:expression n :op :<= :value 8388608}
          {:expression k :op :>= :value 32}
          {:expression k :op :<= :value 8388608}
          {:expression (body/expression :mod n 8) :op := :value 0}
          ;; The K16 instruction consumes whole fragments, beyond the pitch requirement.
          {:expression (body/expression :mod k 16) :op := :value 0}]
         (map (fn [stride]
                {:expression (body/expression :mod stride 32) :op := :value 0}))
         input-slice-strides)))

(defn fallback-cases
  "Turn the conjunction into ordered failure cases without duplicating its bounds."
  [conditions strategy]
  (mapv (fn [condition]
          (-> condition
              (update :op {:< :>= :<= :> := :!= :!= := :>= :< :> :<=})
              (assoc :strategy strategy)))
        conditions))

(defn body-requirements
  "Project the physical contract from a validated canonical matrix body. Matrix-plan analysis
  separately verifies its row-major operands and leading-slice views before target emission."
  [kernel-body]
  (let [kernel-body (body/validate! kernel-body)
        {:keys [m n k]} (get-in kernel-body [:attributes :dimension-parameters])
        inputs (into #{} (comp (filter #(contains? #{:lhs :rhs} (:role %))) (map :id))
                     (:parameters kernel-body))
        groups (into {} (comp (filter #(= :group (:source %))) (map #(vector (:id %) 1)))
                     (:indices kernel-body))
        strides (for [view (:views kernel-body) :when (contains? inputs (:buffer view))]
                  (walk/postwalk-replace groups (:element-offset view)))]
    {:preconditions (cond-> (into (matrix-preconditions m n k strides)
                                  (partition-preconditions kernel-body k))
                      ;; A transformed FP32 TileLoad still prefetches the physical FP32 surface.
                      ;; Preserve the existing 16 MiB byte-width limit with four-byte elements.
                      (some #(and (= :lhs (:role %)) (= :float (:dtype %))) (:parameters kernel-body))
                      (conj {:expression k :op :<= :value 4194304}))
     :parameter-alignments (zipmap inputs (repeat 64))}))

(defn static-failure
  "Return the failed condition for concrete dimensions, otherwise leave runtime checks intact."
  [m n k]
  (when (every? integer? [m n k])
    (try
      (precondition/check! (matrix-preconditions m n k) {})
      nil
      (catch clojure.lang.ExceptionInfo failure
        (if (= :kernel-precondition-failed (:reason (ex-data failure)))
          (:condition (ex-data failure))
          (throw failure))))))

(ns raster.compiler.ir.kernel-precondition
  "Checked scalar-expression preconditions on an executable, independent of algorithm or target.
  Expressions use the existing launch/storage algebra; this is not another numerical IR."
  (:require [raster.compiler.ir.kernel-launch :as launch]))

(def comparison-ops #{:< :<= := :!= :>= :>})

(defn compare-value? [op actual expected]
  (case op
    :< (< actual expected)
    :<= (<= actual expected)
    := (= actual expected)
    :!= (not= actual expected)
    :>= (>= actual expected)
    :> (> actual expected)))

(defn validate!
  "Validate ordered constraints against integral ABI slot names, not caller argument expressions.
  This keeps a derived or privately specialized scalar tied to its actual physical value."
  [conditions scalar-identities]
  (when-not (vector? conditions)
    (throw (ex-info "kernel preconditions must be an ordered vector"
                    {:reason :kernel-preconditions :conditions conditions})))
  (doseq [{:keys [expression op value] :as condition} conditions]
    (when-not (and (map? condition) (= #{:expression :op :value} (set (keys condition)))
                   (launch/expression? expression) (contains? comparison-ops op)
                   (integer? value) (<= Long/MIN_VALUE value Long/MAX_VALUE))
      (throw (ex-info "kernel precondition requires a checked integer comparison"
                      {:reason :kernel-precondition-invalid :condition condition})))
    (when-not (every? scalar-identities (launch/expression-references expression))
      (throw (ex-info "kernel precondition references values outside the integral scalar ABI"
                      {:reason :kernel-precondition-scope :condition condition
                       :scalar-identities scalar-identities}))))
  conditions)

(defn check!
  "Check in order, before allocation or launch. Arithmetic retains checked overflow semantics."
  [conditions resolve-value]
  (doseq [{:keys [expression op value] :as condition} conditions]
    (let [actual (launch/resolve-expression resolve-value expression)]
      (when-not (compare-value? op actual value)
        (throw (ex-info "kernel scalar precondition failed"
                        {:reason :kernel-precondition-failed
                         :condition condition :actual actual})))))
  true)

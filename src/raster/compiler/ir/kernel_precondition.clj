(ns raster.compiler.ir.kernel-precondition
  "Checked executable preconditions. Integer expressions use the launch/storage
   algebra. Original host object relations require an explicit, separate scope;
   they never imply physical pointer range or device storage independence."
  (:require [raster.compiler.ir.kernel-launch :as launch]))

(def comparison-ops #{:< :<= := :!= :>= :>})

(defn- validate-host-object-condition!
  [condition identities]
  (when-not (and (= #{:op :scope :left :right} (set (keys condition)))
                 (= :host-object-distinct (:op condition))
                 (= :original-host-arguments (:scope condition))
                 (symbol? (:left condition)) (symbol? (:right condition))
                 (not= (:left condition) (:right condition)))
    (throw (ex-info "Invalid original-host-object precondition"
                    {:reason :kernel-precondition-invalid :condition condition})))
  (when-not (and (set? identities) (contains? identities (:left condition))
                 (contains? identities (:right condition)))
    (throw (ex-info "Object precondition requires an explicit original-host-argument scope"
                    {:reason :kernel-precondition-scope :condition condition}))))

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
  Original host object conditions additionally require :host-object-identities;
  scalar-only consumers reject them rather than treating identity as integer inequality."
  ([conditions scalar-identities] (validate! conditions scalar-identities {}))
  ([conditions scalar-identities {:keys [host-object-identities]}]
   (when-not (vector? conditions)
     (throw (ex-info "kernel preconditions must be an ordered vector"
                     {:reason :kernel-preconditions :conditions conditions})))
   (doseq [{:keys [expression op value] :as condition} conditions]
     (if (= :host-object-distinct op)
       (validate-host-object-condition! condition host-object-identities)
       (do
         (when-not (and (map? condition) (= #{:expression :op :value} (set (keys condition)))
                        (launch/expression? expression) (contains? comparison-ops op)
                        (launch/expression? value))
           (throw (ex-info "kernel precondition requires a checked integer comparison"
                           {:reason :kernel-precondition-invalid :condition condition})))
         (when-not (every? scalar-identities
                           (into (launch/expression-references expression)
                                 (launch/expression-references value)))
           (throw (ex-info "kernel precondition references values outside the integral scalar ABI"
                           {:reason :kernel-precondition-scope :condition condition
                            :scalar-identities scalar-identities}))))))
   conditions))

(defn check!
  "Check in order, before allocation or launch. Arithmetic retains checked overflow
   semantics. Host relations require :host-objects mapping identities to ORIGINAL
   argument objects, never staged pointers or reconstructed device handles."
  ([conditions resolve-value] (check! conditions resolve-value {}))
  ([conditions resolve-value {:keys [host-objects]}]
   (doseq [{:keys [expression op value] :as condition} conditions]
     (if (= :host-object-distinct op)
       (do
         (validate-host-object-condition! condition (when (map? host-objects)
                                                     (set (keys host-objects))))
         (when (identical? (get host-objects (:left condition))
                           (get host-objects (:right condition)))
           (throw (ex-info "Original host argument objects must differ"
                           {:reason :kernel-precondition-failed :condition condition}))))
       (let [actual (launch/resolve-expression resolve-value expression)
             expected (launch/resolve-expression resolve-value value)]
         (when-not (compare-value? op actual expected)
           (throw (ex-info "kernel scalar precondition failed"
                           {:reason :kernel-precondition-failed
                            :condition condition :actual actual :expected expected}))))))
   true))

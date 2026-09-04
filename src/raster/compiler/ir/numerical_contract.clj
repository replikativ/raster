(ns raster.compiler.ir.numerical-contract
  "Shared numerical attestation for verified schedule refinements.

   This value is producer evidence, not an inferred floating-point proof.  Scheduling passes state
   whether they preserve evaluation exactly, reassociate a known accumulator, or accept a named
   error model.  IR boundaries consume the same validator so numerical policy cannot drift between
   graph and single-kernel refinements."
  (:require [raster.compiler.core.dtype :as dtype]))

(def modes #{:exact :reassociated :bounded-error})
(def rounding-policies
  #{:nearest-even :toward-zero :up :down :implementation-defined})

(defn- canonical-dtype?
  [value]
  (and (dtype/known? value) (= value (dtype/canon value))))

(defn- accumulator-component?
  [component]
  (and (map? component)
       (some? (:value component))
       (canonical-dtype? (:dtype component))
       (contains? rounding-policies (:rounding component))
       (keyword? (:policy component))))

(defn- component-accumulators?
  [contract]
  (let [components (:accumulators contract)]
    (and (vector? components)
         (seq components)
         (every? accumulator-component? components)
         (= (count components) (count (set (map :value components)))))))

(defn validate!
  "Validate and return a numerical contract.

   `context` lets an owning IR preserve its public diagnostic identity while sharing this one
   contract."
  ([contract] (validate! contract {}))
  ([contract {:keys [reason ir]
              :or {reason :numerical-contract ir :numerical-contract}}]
   (letfn [(fail! [message data]
             (throw (ex-info message (assoc data :reason reason :ir ir))))]
     (when-not (and (map? contract)
                    (contains? modes (:mode contract))
                    (keyword? (:policy contract)))
       (fail! "numerical contract requires a supported mode and named policy"
              {:value contract :supported modes}))
     (case (:mode contract)
       :exact nil
       :reassociated
       (when-not (or (and (contains? rounding-policies (:rounding contract))
                          (canonical-dtype? (:accumulator-dtype contract)))
                     (component-accumulators? contract))
         (fail! "reassociated numerical contract requires one accumulator or checked components"
                {:value contract}))
       :bounded-error
       (when-not (and (contains? rounding-policies (:rounding contract))
                      (canonical-dtype? (:accumulator-dtype contract))
                      (map? (:error-model contract))
                      (keyword? (get-in contract [:error-model :kind])))
         (fail! "bounded-error numerical contract requires rounding, accumulator dtype, and an error model"
                {:value contract})))
     contract)))

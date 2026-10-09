(ns raster.ode.amr-cycle-contract
  "Structural temporal evidence for the bounded ratio-2 diffusion producer.
   Stage meanings and time fractions are producer attestations, not inferred PDE proofs.
   No device/runtime, workload certificate, timestep stability or storage release is granted."
  (:require [raster.compiler.ir.emitted-parallel-program :as emitted]
            [raster.compiler.ir.validate :refer [fail!]]))

(def ^:private stage-contracts
  [{:kind :coarse-prediction :time-fractions [[1 1]] :register-transition :none}
   {:kind :boundary-endpoints :time-fractions [[0 1] [1 1]] :register-transition :none}
   {:kind :fine-substep-0 :time-fractions [[0 1] [1 2]] :register-transition :none}
   {:kind :fine-substep-1 :time-fractions [[1 2] [1 2]] :register-transition :none}
   {:kind :register-reset :time-fractions [] :register-transition :reset}
   {:kind :coarse-transport :time-fractions [[-1 1]] :register-transition :accumulate}
   {:kind :fine-transport :time-fractions [[1 2] [1 2]] :register-transition :accumulate}
   {:kind :reflux :time-fractions [[1 1]] :register-transition :consume}
   {:kind :average-down :time-fractions [] :register-transition :none}])

(defn- register-access [equation register]
  (let [accesses (mapv :access (filter #(= register (:destination %))
                                      (get-in equation [:attributes :result-storage])))]
    {:equation (:id equation) :storage-accesses accesses
     :read? (boolean (or (some #{register} (:operands equation))
                         (some #{:read :read-write} accesses)))
     :write? (boolean (some #{:write :read-write} accesses))}))

(defn- register-transition! [stage accesses]
  (let [events (filterv #(or (:read? %) (:write? %)) accesses)
        valid? (case (:register-transition stage)
                 :none (empty? events)
                 :reset (and (seq events) (every? #(and (:write? %) (not (:read? %))) events))
                 :accumulate (and (seq events) (every? #(and (:read? %) (:write? %)) events))
                 :consume (and (seq events) (every? #(and (:read? %) (not (:write? %))) events)))]
    (when-not valid?
      (fail! "temporal stage disagrees with retained register dataflow"
             :amr-temporal-register-state {:stage (:kind stage) :accesses events}))
    events))

(defn stage-projection
  "Check a producer's ordered stage partition against retained semantic equations.

   Every numerical top-level equation must appear exactly once, in source order. Physical
   fusion does not change coverage; nested control stays inside its semantic equation. Register
   reads/result destinations come from retained typed dataflow. These structural facts do not
   establish full-write reset coverage, numerical coefficients, CFL, completion or release.
   Fractions are exact rational pairs in the bounded producer contract, not measured times:
   fine substeps give [boundary theta, dt fraction], endpoints give [old, predicted theta],
   and transport stages give signed time weights. Stage meanings remain producer-attested."
  [program register stages]
  (let [equations (emitted/retained-numerical-equations program)
        by-id (into {} (map (juxt :id identity)) equations)
        abstract (get (:values program) register)]
    (when-not (and (some #{register} (:inputs program))
                   (= :tensor (:kind abstract)) (= :double (:dtype abstract))
                   (vector? (:shape abstract)) (seq (:shape abstract))
                   (= {:kind :plain} (:representation abstract)))
      (fail! "temporal register requires an explicit FP64 program buffer"
             :amr-temporal-register {:register register}))
    (when-not (and (vector? stages) (= (count stage-contracts) (count stages)))
      (fail! "ratio-2 temporal evidence requires all ordered stages"
             :amr-temporal-stages {}))
    (doseq [[i stage expected] (map vector (range) stages stage-contracts)]
      (when-not (and (map? stage)
                     (= #{:kind :equations :dependencies :time-fractions :register-transition}
                        (set (keys stage)))
                     (= expected (select-keys stage (keys expected)))
                     (vector? (:equations stage)) (seq (:equations stage))
                     (vector? (:dependencies stage))
                     (= (if (zero? i) [] [(:kind (nth stages (dec i)))])
                        (:dependencies stage)))
        (fail! "temporal stage must retain its exact ratio-2 contract and predecessor"
               :amr-temporal-stage {:stage (:kind stage) :position i})))
    (when-not (= (mapv :id equations) (vec (mapcat :equations stages)))
      (fail! "temporal stages must partition every retained numerical equation in source order"
             :amr-temporal-equation-coverage {}))
    (let [projected
          (mapv (fn [stage]
                  (assoc stage :register-accesses
                         (register-transition! stage
                                               (mapv #(register-access (by-id %) register)
                                                     (:equations stage)))))
                stages)]
      {:equations equations :register register :stages projected
       :register-last-stage (:kind (last (filter #(seq (:register-accesses %)) projected)))
       :full-write-reset :unproven :completion :unproven :release :unproven
       :mathematical-meaning :producer-attested})))

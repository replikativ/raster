(ns raster.ode.amr-cycle-contract-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.emitted-parallel-program :as emitted]
            [raster.ode.amr-cycle-contract :as contract]
            [raster.ode.amr-cycle-lowering :as lowering]
            [raster.ode.amr-subcycle :as subcycle]
            [raster.ode.amr-subcycle-test :as oracle]))

(def ^:private retained-cycle
  (delay
    (let [target :ocl:temporal-contract
          descriptor {:device-id target :device-type :gpu :backend :ocl :fp64? true
                      :subgroup-dialect :intel-opencl :max-workgroup-size 256}
          p (subcycle/project (#'oracle/hierarchy) {:domain-lengths [1.0 1.0] :diffusivity 0.2})
          {:keys [coarse fine]} (#'oracle/initial-state)]
      (with-redefs [hardware/descriptor-for (constantly descriptor)]
        (get-in (lowering/lower p (double-array coarse) (double-array fine) 0.001 {:target target})
                [:lowering :plan :instances 0 :call :program])))))

(defn- stages [program]
  ;; Independent producer partition of this authored numerical cycle. The validator
  ;; accepts variable stage sizes, not a fixed kernel count or name heuristic.
  (let [ids (mapv :id (emitted/retained-numerical-equations program))
        sizes [2 5 3 4 1 1 4 1 2]
        offsets (reductions + 0 sizes)
        specs [{:kind :coarse-prediction :time-fractions [[1 1]] :register-transition :none}
               {:kind :boundary-endpoints :time-fractions [[0 1] [1 1]] :register-transition :none}
               {:kind :fine-substep-0 :time-fractions [[0 1] [1 2]] :register-transition :none}
               {:kind :fine-substep-1 :time-fractions [[1 2] [1 2]] :register-transition :none}
               {:kind :register-reset :time-fractions [] :register-transition :reset}
               {:kind :coarse-transport :time-fractions [[-1 1]] :register-transition :accumulate}
               {:kind :fine-transport :time-fractions [[1 2] [1 2]] :register-transition :accumulate}
               {:kind :reflux :time-fractions [[1 1]] :register-transition :consume}
               {:kind :average-down :time-fractions [] :register-transition :none}]]
    (mapv (fn [i spec start size]
            (assoc spec :equations (subvec ids start (+ start size))
                   :dependencies (if (zero? i) [] [(:kind (nth specs (dec i)))])))
          (range) specs offsets sizes)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest stages-retain-semantic-coverage-and-register-transitions
  (let [p @retained-cycle s (stages p) evidence (contract/stage-projection p 'register s)]
    (is (= (mapv :id (:equations evidence)) (vec (mapcat :equations s))))
    (is (= 9 (count (:stages evidence))))
    (is (= :reflux (:register-last-stage evidence)))
    (is (= :producer-attested (:mathematical-meaning evidence)))
    (is (= [:unproven :unproven :unproven]
           (mapv evidence [:full-write-reset :completion :release])))
    (is (= {:equation (first (:equations (nth s 4))) :storage-accesses [:write]
            :read? false :write? true}
           (first (get-in evidence [:stages 4 :register-accesses]))))
    (is (= 2 (count (get-in evidence [:stages 6 :register-accesses]))))
    (is (every? :read? (get-in evidence [:stages 7 :register-accesses])))
    (is (not-any? :write? (get-in evidence [:stages 7 :register-accesses])))))

(deftest missing-reordered-or-duplicated-semantic-equations-fail
  (let [p @retained-cycle s (stages p)]
    (doseq [bad [(update-in s [0 :equations] pop)
                 (update-in s [0 :equations] #(vec (reverse %)))
                 (update-in s [0 :equations] conj (first (:equations (nth s 1))))]]
      (is (= :amr-temporal-equation-coverage
             (reason #(contract/stage-projection p 'register bad)))))))

(deftest temporal-contract-and-register-claims-fail-closed
  (let [p @retained-cycle s (stages p)]
    (doseq [bad [(assoc-in s [1 :dependencies] [])
                 (assoc-in s [3 :time-fractions] [[0 1] [1 2]])
                 (assoc-in s [5 :register-transition] :reset)
                 (assoc-in s [0 :physical-kernel-count] 2)
                 (assoc-in s [4 :kind] :reflux)]]
      (is (= :amr-temporal-stage (reason #(contract/stage-projection p 'register bad)))))
    (is (= :amr-temporal-stages (reason #(contract/stage-projection p 'register (pop s)))))
    (is (= :amr-temporal-register (reason #(contract/stage-projection p 'interface s))))
    (is (= :amr-temporal-register (reason #(contract/stage-projection p 'dt s))))
    (is (= :amr-temporal-register-state (reason #(contract/stage-projection p 'flux-0 s))))))

(deftest storage-access-roles-cannot-be-discarded-by-register-projection
  (let [p @retained-cycle s (stages p)
        equations (emitted/retained-numerical-equations p)
        reset-id (first (:equations (nth s 4)))
        consume-id (first (:equations (nth s 7)))]
    ;; Isolate access interpretation after ordinary emitted-program validation.
    ;; The unchanged actual program is independently validated in the tests above.
    (doseq [[id access] [[reset-id :read-write] [consume-id :write]]]
      (let [changed (mapv (fn [equation]
                            (if (= id (:id equation))
                              (update-in equation [:attributes :result-storage]
                                         (fn [entries]
                                           (conj (filterv #(not= 'register (:destination %)) entries)
                                                 {:destination 'register :access access})))
                              equation)) equations)]
        (with-redefs [emitted/retained-numerical-equations (constantly changed)]
          (is (= :amr-temporal-register-state
                 (reason #(contract/stage-projection p 'register s)))))))))

(deftest register-storage-facets-are-required
  (let [p @retained-cycle s (stages p)
        equations (emitted/retained-numerical-equations p)]
    ;; These controls isolate the stage layer's extra storage obligations.
    (with-redefs [emitted/retained-numerical-equations (constantly equations)]
      (doseq [bad [(assoc-in p [:values 'register :shape] [])
                   (assoc-in p [:values 'register :representation] {:kind :packed})]]
        (is (= :amr-temporal-register
               (reason #(contract/stage-projection bad 'register s))))))))

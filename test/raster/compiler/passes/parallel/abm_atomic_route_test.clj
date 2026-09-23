(ns raster.compiler.passes.parallel.abm-atomic-route-test
  "Production-route coverage for value-returning atomics in the firms workload."
  (:require [clojure.test :refer [deftest is]]
            [raster.abm.firms.phases :as phases]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.pipeline :as pipeline]))

(defn- body-operations [artifact]
  (letfn [(walk [operations]
            (mapcat (fn [operation]
                      (cons operation
                            (concat (walk (or (:operations operation) []))
                                    (walk (or (:then-operations operation) []))
                                    (walk (or (:else-operations operation) [])))))
                    operations))]
    (walk (get-in artifact [:attributes :kernel-body :operations]))))

(deftest fill-members-retains-the-atomic-old-value-type
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/fill-members-par! :ze:debug
                    :dtype :float :on-non-resident :nil)]
    (is (some? descriptor))
    (is (= 2 (count (:steps descriptor))))
    (is (kernel-artifact/kernel-artifact? (get-in descriptor [:steps 0 :artifact])))
    (let [executable (get-in descriptor [:steps 1 :artifact])
          artifacts (if (kernel-graph/kernel-graph? executable)
                      (mapv :operation (:nodes executable))
                      [executable])
          atomics (for [artifact artifacts
                        operation (body-operations artifact)
                        :when (= "AtomicRMW" (some-> operation class .getSimpleName))]
                    operation)]
      (is (every? kernel-artifact/kernel-artifact? artifacts))
      (is (= 1 (count atomics)))
      (is (= :int (get-in (first atomics) [:result :type]))))))

(deftest collect-preserves-its-fetch-add-ticket-as-a-resident-unique-scatter
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/count-startups-par! :ze:debug
                    :dtype :float :on-non-resident :nil)
        artifact (get-in descriptor [:steps 0 :artifact])
        operations (body-operations artifact)
        atomics (filter #(= "AtomicRMW" (some-> % class .getSimpleName)) operations)
        conditionals (filter #(= "IfRegion" (some-> % class .getSimpleName)) operations)]
    (is (some? descriptor))
    (is (= 1 (count (:steps descriptor))))
    (is (kernel-artifact/kernel-artifact? artifact))
    (is (= 1 (count atomics)))
    (is (= :int (get-in (first atomics) [:result :type])))
    (is (some #(= :int (get-in % [:results 0 :type])) conditionals)
        "the guarded atomic exports its old value through typed SSA control")
    (is (some #(= "ScalarStore" (some-> % class .getSimpleName)) operations))))

(deftest csr-income-distribution-is-one-resident-nested-effect-map
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/distribute-income-par! :ze:debug
                    :dtype :float :on-non-resident :nil)
        artifact (get-in descriptor [:steps 0 :artifact])
        operations (body-operations artifact)]
    (is (some? descriptor))
    (is (= 1 (count (:steps descriptor))))
    (is (kernel-artifact/kernel-artifact? artifact))
    (is (some #(= "ForLoop" (some-> % class .getSimpleName)) operations))
    (is (some #(= "ScalarStore" (some-> % class .getSimpleName)) operations))))

(deftest startup-execution-separates-unique-firm-inits-from-ordered-agent-updates
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/execute-startups-par! :ze:debug
                    :dtype :float :on-non-resident :nil)
        artifacts (mapv :artifact (:steps descriptor))]
    (is (some? descriptor))
    (is (= 2 (count artifacts)))
    (is (every? kernel-artifact/kernel-artifact? artifacts))
    (is (= [:one-work-item-per-element :one-work-item-ordered-loop]
           (mapv #(get-in % [:attributes :kernel-body :schedule :strategy]) artifacts)))
    (is (some #(= "AtomicRMW" (some-> % class .getSimpleName))
              (body-operations (second artifacts))))))

(deftest agent-decisions-keep-bundle-projections-and-ordered-scalar-solvers-in-one-kernel
  (let [descriptor (pipeline/compile-gpu-program
                    #'phases/agent-decide-par! :ze:debug
                    :dtype :float :on-non-resident :nil)
        artifact (get-in descriptor [:steps 0 :artifact])
        operations (body-operations artifact)]
    (is (some? descriptor))
    (is (= 1 (count (:steps descriptor))))
    (is (kernel-artifact/kernel-artifact? artifact))
    (is (<= 2 (count (filter #(= "ForLoop" (some-> % class .getSimpleName)) operations)))
        "the bounded Newton solve and friend selection remain typed ordered loops")
    (is (= ["effort" "income" "theta" "endowment" "firm" "friends" "cache"]
           (get-in descriptor [:value-specs 'agents :physical-layout :field-order])))
    (is (= ["a" "b" "beta" "te" "output" "n-workers" "alive" "members" "offsets"]
           (get-in descriptor [:value-specs 'firms :physical-layout :field-order])))))

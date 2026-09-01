(ns raster.dl.optim-resident-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.optim :as optim]))

(deftest adaptive-optimizers-use-one-certified-resident-effect-map
  (doseq [[label optimizer expected-scalars]
          [["Adam" #'optim/adam-step!
            '#{bc1 bc2 beta1 beta2 eps lr}]
           ["AdamW" #'optim/adamw-step!
            '#{bc1 bc2 beta1 beta2 eps lr weight-decay}]]]
    (testing label
      (let [program (pipeline/compile-gpu-program
                     optimizer :ze:0 :dtype :float
                     :on-non-resident :nil :compiler-report? true)
            report (:compiler-report program)
            step (first (:steps program))
            artifact (:artifact step)
            abi (:abi step)
            scalar-names (->> abi
                              (filter #(= :parameter (:role %)))
                              (map :name)
                              set)]
        (is (some? program))
        (is (= 1 (count (:steps program))))
        (is (= :typed-soac (get-in report [:route :source-dialect])))
        (is (true? (get-in report [:route :typed-validated])))
        (is (= 1 (get-in report [:lowering :typed-reused])))
        (is (zero? (get-in report [:lowering :fallback])))
        (is (true? (get-in report [:residency :resident?])))
        (is (= :map-void (:convention step)))
        (is (= :elementwise-map (get-in artifact [:effects :kind])))
        (is (= expected-scalars scalar-names))
        (is (= #{'param 'm 'v}
               (->> abi
                    (filter #(= :inout (:kind %)))
                    (map :name)
                    set)))
        (is (= ['grad] (->> abi
                            (filter #(= :input (:kind %)))
                            (mapv :name))))
        (is (not (str/includes? (:source artifact) "vstore4"))
            "lexical moment snapshots must not be inlined past their state writes")))))

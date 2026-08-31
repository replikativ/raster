(ns raster.compiler.passes.parallel.fusion-placement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.fusion-placement :as placement]))

(deftest roofline-placement-witness-is-target-neutral-and-complete
  (let [expressions '[(exp (exp (exp x)))]
        poor (placement/placement-decision
              {:abstract-machine {:ridge {:float 2.0}}
               :dtype :float :expressions expressions :consumer-count 2})
        rich (placement/placement-decision
              {:abstract-machine {:ridge {:float 100.0}}
               :dtype :float :expressions expressions :consumer-count 2})]
    (testing "an expensive fan-out remains materialized when recomputation costs more than a read"
      (is (= :materialize (:decision poor)))
      (is (false? (:fuse? poor)))
      (is (= 30 (:producer-flops-per-element poor)))
      (is (= 4 (:element-bytes poor)))
      (is (= 8.0 (:recompute-threshold-flops poor)))
      (is (= :recompute-more-expensive-than-read (:reason poor))))
    (testing "the same algorithm recomputes on a compute-rich abstract machine"
      (is (= :recompute (:decision rich)))
      (is (:fuse? rich))
      (is (= 400.0 (:recompute-threshold-flops rich)))
      (is (= :recompute-cheaper-than-read (:reason rich))))
    (is (not-any? #(contains? poor %) [:device :vendor :backend]))))

(deftest placement-abstains-in-the-compatibility-preserving-direction
  (let [arguments {:dtype :float :expressions '[(exp x)] :consumer-count 3}]
    (is (= {:decision :materialize :reason :no-abstract-machine}
           (select-keys (placement/placement-decision arguments) [:decision :reason])))
    (is (= {:decision :recompute :reason :unknown-roofline-ridge}
           (select-keys (placement/placement-decision
                         (assoc arguments :abstract-machine {:ridge {:half 4.0}}))
                        [:decision :reason])))
    (is (= {:decision :eliminate :reason :sole-consumer}
           (select-keys (placement/placement-decision
                         (assoc arguments :consumer-count 1))
                        [:decision :reason])))))

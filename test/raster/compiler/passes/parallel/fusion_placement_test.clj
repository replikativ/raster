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
      (is (:producer-cost-complete? poor))
      (is (empty? (:unknown-cost-ops poor)))
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
    (is (= {:decision :materialize :reason :unknown-roofline-ridge}
           (select-keys (placement/placement-decision
                         (assoc arguments :abstract-machine {:ridge {:half 4.0}}))
                        [:decision :reason])))
    (is (= {:decision :eliminate :reason :sole-consumer}
           (select-keys (placement/placement-decision
                         (assoc arguments :consumer-count 1))
                        [:decision :reason])))))

(deftest incomplete-placement-prices-never-enable-duplicated-work
  (let [machine {:ridge {:float 100.0}}
        unknown-op (placement/placement-decision
                    {:abstract-machine machine :dtype :float
                     :expressions '[(arbitrary-user-call x)] :consumer-count 2})
        unknown-width (placement/placement-decision
                       {:abstract-machine {:ridge {:mystery 100.0}} :dtype :mystery
                        :expressions '[(+ x 1)] :consumer-count 2})
        unknown-ridge (placement/placement-decision
                       {:abstract-machine {:ridge {:half 100.0}} :dtype :float
                        :expressions '[(+ x 1)] :consumer-count 2})]
    (is (= {:decision :materialize :reason :unknown-producer-cost
            :producer-cost-complete? false
            :unknown-cost-ops '[arbitrary-user-call]}
           (select-keys unknown-op [:decision :reason :producer-cost-complete?
                                    :unknown-cost-ops])))
    (is (= {:decision :materialize :reason :unknown-element-width}
           (select-keys unknown-width [:decision :reason])))
    (is (= {:decision :materialize :reason :unknown-roofline-ridge}
           (select-keys unknown-ridge [:decision :reason])))))

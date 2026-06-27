(ns raster.compiler.passes.placement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as opd]
            [raster.compiler.passes.placement :as pl]))

;; Two test ops with explicit placement tags, so the test does not depend on any
;; library op's registration.
(opd/register-placement! 'test.ops/gmm :gpu)     ;; a GPU op
(opd/register-placement! 'test.ops/cnorm :jvm)   ;; a CPU op
;; test.ops/free has NO placement tag -> defaults to the CPU.

(def ^:private policy {:gpu :ze:0 :cpu-quant :ze:0 :jvm :cpu:0 :default :cpu:0})

(deftest op-device-test
  (testing "device-type tag maps through the policy to a concrete device"
    (is (= :ze:0 (pl/op-device '(test.ops/gmm x) policy)))
    (is (= :cpu:0 (pl/op-device '(test.ops/cnorm x) policy))))
  (testing "untagged op and non-call default to the CPU"
    (is (= :cpu:0 (pl/op-device '(test.ops/free x) policy)))
    (is (= :cpu:0 (pl/op-device 'x policy)))))

(deftest place-inserts-transfers-at-device-boundaries
  (let [form '(let* [a (test.ops/gmm x)        ;; GPU
                     b (test.ops/cnorm a)      ;; CPU  <- needs a transfer of a (gpu->cpu)
                     c (test.ops/gmm b)]       ;; GPU  <- needs a transfer of b (cpu->gpu)
                c)
        {:keys [devices transfers form] :as _r} (pl/place form policy)]
    (testing "each binding is assigned its op's concrete device"
      (is (= {'a :ze:0 'b :cpu:0 'c :ze:0} devices)))
    (testing "a transfer is inserted at each cross-device data-flow edge"
      (is (= 2 (count transfers)))
      (is (= #{{:sym 'a :from :ze:0 :to :cpu:0 :at 'b}
               {:sym 'b :from :cpu:0 :to :ze:0 :at 'c}}
             (set transfers))))
    (testing "cross-device input syms are wrapped in xfer markers in the placed form"
      (let [s (pr-str form)]
        (is (.contains s "raster.compiler.passes.placement/xfer :ze:0 :cpu:0 a"))
        (is (.contains s "raster.compiler.passes.placement/xfer :cpu:0 :ze:0 b"))))
    (testing "each placed binding value carries its :device in metadata"
      (let [binds (->> form second (partition 2))]
        (is (= :ze:0 (:device (meta (second (nth binds 0))))))
        (is (= :cpu:0 (:device (meta (second (nth binds 1))))))))))

(deftest same-device-chain-needs-no-transfers
  (let [form '(let* [a (test.ops/gmm x)
                     b (test.ops/gmm a)] b)]   ;; both GPU
    (is (empty? (:transfers (pl/place form policy))))))

(deftest non-let-form-returned-unplaced
  (is (= '(test.ops/gmm x) (:form (pl/place '(test.ops/gmm x) policy)))))

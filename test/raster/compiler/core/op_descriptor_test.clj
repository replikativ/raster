(ns raster.compiler.core.op-descriptor-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]))

(deftest descriptor-facets-merge-directly-test
  (testing "buffer, device, and shape registrations converge on one descriptor"
    (descriptor/register-buffer-semantics! 'test.descriptor/op
                                           {:allocates? true
                                            :in-place-arg 0
                                            :mutating? true
                                            :alloc-form (fn [_ _] '(double-array 16))
                                            :rewrite-fn (fn [_ buf] buf)})
    (descriptor/register-device-rule! 'test.descriptor/op
                                      (fn [_args _env _params-set] :ze:0))
    (descriptor/register-dim-rule! 'test.descriptor/op
                                   (fn [_args _env _params-set] 'n))
    (let [[descriptor-map base-op] (descriptor/resolve-op-descriptor 'test.descriptor/op)]
      (is (= 'test.descriptor/op base-op))
      (is (= true (get-in descriptor-map [:buffer :allocates?])))
      (is (= 0 (get-in descriptor-map [:buffer :in-place-arg])))
      (is (= true (get-in descriptor-map [:effects :mutating?])))
      (is (fn? (get-in descriptor-map [:device :rule])))
      (is (fn? (get-in descriptor-map [:shape :dim-rule]))))
    (is (= true (:allocates? (descriptor/get-buffer-semantics 'test.descriptor/op))))
    (is (= true (descriptor/mutating-op? 'test.descriptor/op)))
    (is (fn? (descriptor/get-device-rule 'test.descriptor/op)))))
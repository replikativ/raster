(ns raster.compiler.core.op-descriptor-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]))

(deftest allocation-initialization-contracts
  (is (= :zero (descriptor/allocation-initialization 'double-array)))
  (is (= :zero (descriptor/allocation-initialization 'raster.arrays/zeros-like_m_double)))
  (is (= :unspecified (descriptor/allocation-initialization 'similar_m_double)))
  (is (= :copy (descriptor/allocation-initialization 'clojure.core/aclone)))
  (is (descriptor/copy-allocation-op? 'aclone_m_double))
  (is (nil? (descriptor/allocation-initialization 'example/not-an-allocation))))

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

(deftest relational-comparisons-own-their-boolean-result-test
  (testing "all surface spellings share comparison semantics and a fixed Boolean result"
    (doseq [[op kind] [['< :lt] ['clojure.core/< :lt] ['raster.numeric/< :lt]
                       ['<= :le] ['clojure.core/<= :le] ['raster.numeric/<= :le]
                       ['> :gt] ['clojure.core/> :gt] ['raster.numeric/> :gt]
                       ['>= :ge] ['clojure.core/>= :ge] ['raster.numeric/>= :ge]
                       ['== :eq] ['clojure.core/== :eq] ['raster.numeric/== :eq]
                       ['not= :ne] ['clojure.core/not= :ne] ['raster.numeric/not= :ne]]]
      (is (= kind (descriptor/comparison-kind op)) (str op " comparison kind"))
      (is (= 'boolean (descriptor/result-tag op ['long 'long]))
          (str op " result tag")))))

(deftest affine-steps-share-the-typed-wrapping-operator-registry
  (doseq [op '[unchecked-add unchecked-add-int
               clojure.core/unchecked-add clojure.core/unchecked-add-int]]
    (is (descriptor/wrapping-addition-op? op))
    (is (= 1 (descriptor/affine-step (list op '(clojure.core/long i)
                                               '(clojure.core/long 1))
                                     'i))))
  (is (not (descriptor/wrapping-addition-op? 'clojure.core/+)))
  (is (= 1 (descriptor/affine-step '(clojure.core/+ i 1) 'i))))

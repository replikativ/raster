(ns raster.compiler.ir.abstract-value-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]))

(deftest logical-values-do-not-prescribe-physical-buffer-count
  (let [value (av/tensor {:dtype :float
                          :shape ['batch 4096]
                          :logical-layout {:order [0 1]}
                          :representation {:kind :quantized :scheme :q4-k}
                          :memory-space :device
                          :placement {:device :gpu}
                          :ownership :borrowed
                          :effects #{:read}})]
    (is (av/abstract-value? value))
    (is (= ['batch 4096] (:shape value)))
    (is (= :q4-k (get-in value [:representation :scheme])))
    (is (= :borrowed (:ownership value)))
    (is (= #{:read} (:effects value)))
    (is (not (contains? value :view)))
    (is (not (contains? value :buffer)))))

(deftest abstract-value-validation-is-fail-loud
  (testing "tensor semantics require a dtype"
    (is (= :abstract-value-dtype
           (:reason (ex-data
                     (try (av/make {:kind :tensor :shape [8]})
                          (catch clojure.lang.ExceptionInfo error error)))))))
  (testing "negative realized extents are never legal"
    (is (= :abstract-value-shape
           (:reason (ex-data
                     (try (av/tensor {:dtype :float :shape [-1]})
                          (catch clojure.lang.ExceptionInfo error error))))))))

(deftest scalar-domains-are-validated-semantic-preconditions
  (doseq [[type lower upper] [[:byte -3 4] [:int 0 Integer/MAX_VALUE]
                              [:long Long/MIN_VALUE Long/MAX_VALUE]]]
    (let [domain {:lower lower :upper upper}
          value (av/tensor {:dtype type :shape [] :attributes {:scalar-domain domain}})]
      (is (= domain (get-in value [:attributes :scalar-domain])))))
  (doseq [[type shape domain] [[:float [] {:lower 0 :upper 1}]
                               [:half [] {:lower 0 :upper 1}]
                               [:unknown [] {:lower 0 :upper 1}]
                               [:long [1] {:lower 0 :upper 1}]
                               [:long [] {:lower 2 :upper 1}]
                               [:long [] {:lower 0.0 :upper 1}]
                               [:byte [] {:lower 0 :upper 128}]
                               [:long [] {:lower 0 :upper 9223372036854775808N}]
                               [:long [] {:lower 0 :upper 1 :proof :asserted}]
                               [:long [] nil]]]
    (is (= :abstract-value-scalar-domain
           (try (av/tensor {:dtype type :shape shape :attributes {:scalar-domain domain}})
                nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest scalar-domain-refinement-composes-as-constraint-intersection
  (let [value (fn [lower upper]
                (av/tensor {:dtype :long :shape []
                            :attributes {:scalar-domain {:lower lower :upper upper}}}))
        base (av/tensor {:dtype :long :shape []})
        a (value 0 10) b (value 3 8) c (value 5 12)
        refine av/merge-refinement]
    (is (= a (refine a a)))
    (is (= a (refine a base) (refine base a)))
    (is (= (refine a b) (refine b a)))
    (is (= (refine (refine a b) c) (refine a (refine b c))))
    (is (= {:lower 5 :upper 8}
           (get-in (refine (refine a b) c) [:attributes :scalar-domain])))
    (is (nil? (refine a (value 11 12))))
    (is (nil? (refine a (assoc a :memory-space :device))))
    (is (nil? (refine a (assoc-in a [:attributes :unrelated] true))))
    (is (av/storage-contract-compatible? base a)
        "logical domains do not alter the physical element representation")
    (is (not (av/scalar-domain-implies? base a)))
    (is (av/scalar-domain-implies? (refine a b) a))))

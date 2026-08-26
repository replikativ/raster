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

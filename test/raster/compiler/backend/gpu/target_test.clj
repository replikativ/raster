(ns raster.compiler.backend.gpu.target-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.target :as target]))

(deftest descriptors-select-source-dialects-without-claiming-runtime-support
  (is (= :cuda (target/kernel-body-c-dialect
                {:device-type :gpu :backend :cuda :vendor "NVIDIA"})))
  (is (= :hip (target/kernel-body-c-dialect
               {:device-type :gpu :backend :hip :vendor "AMD"})))
  (is (= :opencl-intel (target/kernel-body-c-dialect
                        {:device-type :gpu :backend :ze :vendor "Intel"})))
  (is (= :opencl-portable (target/kernel-body-c-dialect
                           {:device-type :gpu :backend :opencl :vendor "AMD"
                            :subgroup-dialect :opencl-portable})))
  (testing "vendor names do not override an explicitly different runtime backend"
    (is (= :cuda (target/kernel-body-c-dialect
                  {:device-type :gpu :backend :cuda :vendor "Intel"
                   :matrix {:family :dpas}}))))
  (testing "generic non-Intel OpenCL must prove portable subgroup support explicitly"
    (is (nil? (target/kernel-body-c-dialect
               {:device-type :gpu :backend :opencl :vendor "AMD"}))))
  (is (nil? (target/kernel-body-c-dialect {:device-type :cpu :backend :cuda}))))

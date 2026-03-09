(ns raster.compiler.passes.parallel.device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.device :as device]))

;; ================================================================
;; device-type
;; ================================================================

(deftest device-type-test
  (testing "device ids normalize to backend families"
    (is (= :cpu (device/device-type :cpu:0)))
    (is (= :cuda (device/device-type :cuda:1)))
    (is (= :ze (device/device-type :ze:0)))
    (is (= :ocl (device/device-type :ocl:3))))

  (testing "bare family keywords still parse when encountered"
    (is (= :cpu (device/device-type :cpu)))
    (is (= :cuda (device/device-type :cuda)))))

;; ================================================================
;; allocation helpers
;; ================================================================

(deftest alloc-expr-test
  (testing "dtype-specific array constructors"
    (is (= '(double-array 128) (device/alloc-expr :double 128)))
    (is (= '(float-array n) (device/alloc-expr :float 'n)))))

;; ================================================================
;; backend selection
;; ================================================================

(deftest select-backend-test
  (testing "explicit device targets win"
    (is (= :cuda (device/select-backend :cuda:0 nil)))
    (is (= :opencl (device/select-backend :ze:0 nil)))
    (is (= :opencl (device/select-backend :ocl:0 nil))))

  (testing "CPU falls back to simd or scalar by size"
    (is (= :simd (device/select-backend :cpu:0 nil)))
    (is (= :simd (device/select-backend :cpu:0 16)))
    (is (= :scalar (device/select-backend :cpu:0 2)))))

(deftest runtime-policy-helpers-test
  (testing "GPU target and dtype helpers centralize target policy"
    (is (true? (device/gpu-target? :ze:0)))
    (is (true? (device/gpu-target? :cuda:0)))
    (is (false? (device/gpu-target? :cpu:0)))
    (is (true? (device/level-zero-target? :ze:0)))
    (is (nil? (device/preferred-dtype :cpu:0)))
    (is (= :float (device/preferred-dtype :ze:0))))

  (testing "runtime backend selection handles device and simd policy"
    (is (= :opencl (device/select-runtime-backend :ze:0 true nil)))
    (is (= :cuda (device/select-runtime-backend :cuda:0 true nil)))
    (is (= :simd (device/select-runtime-backend nil true nil)))
    (is (= :scalar (device/select-runtime-backend nil false nil)))))

;; ================================================================
;; Device descriptors
;; ================================================================

(deftest device-descriptor-test
  (testing "register and retrieve device rule"
    (descriptor/register-device-rule! 'test.ns/test-op
                                      (fn [_args _env _ps] :cpu))
    (is (fn? (descriptor/get-device-rule 'test.ns/test-op)))
    (is (contains? (descriptor/registered-op-descriptors) 'test.ns/test-op))))

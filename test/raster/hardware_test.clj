(ns raster.hardware-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.runtime.hardware :as hw]
            [raster.compiler.core.hardware :as chw]
            [raster.runtime.hardware-catalogue :as catalogue]))

;; Reset hardware state between tests
(use-fixtures :each (fn [f] (hw/reset-hardware!) (f)))

;; ================================================================
;; CPU detection
;; ================================================================

(deftest cpu-detection-test
  (testing "CPU auto-detected on init"
    (hw/init!)
    (let [cpu (hw/device :cpu:0)]
      (is (some? cpu))
      (is (= :cpu (:type cpu)))
      (is (= 0 (:index cpu)))
      (is (string? (:name cpu)))
      (is (pos? (get-in cpu [:capabilities :cores])))
      (is (pos? (get-in cpu [:capabilities :simd-width]))))))

(deftest simd-lanes-test
  (testing "SIMD lanes for double"
    (hw/init!)
    (let [lanes (hw/simd-lanes :cpu:0 :double)]
      (is (pos? lanes))
      (is (contains? #{2 4 8} lanes))))  ;; NEON=2, AVX2=4, AVX-512=8

  (testing "SIMD lanes for float"
    (hw/init!)
    (let [lanes (hw/simd-lanes :cpu:0 :float)]
      (is (pos? lanes))
      (is (>= lanes (hw/simd-lanes :cpu:0 :double))))))

;; ================================================================
;; Device type extraction
;; ================================================================

(deftest device-type-test
  (testing "device-type from device-id"
    (is (= :cpu (hw/device-type :cpu:0)))
    (is (= :cuda (hw/device-type :cuda:0)))
    (is (= :cuda (hw/device-type :cuda:1)))
    (is (= :cpu (hw/device-type :cpu:0)))))

;; ================================================================
;; Target device registration
;; ================================================================

(deftest register-target-device-test
  (testing "Register a GPU for cross-compilation"
    (hw/init!)
    (let [dev (hw/register-target-device! :cuda:0
                                          {:name "NVIDIA H100"
                                           :capabilities {:sm-count 132}})]
      (is (= :cuda:0 (:id dev)))
      (is (= :cuda (:type dev)))
      (is (= 0 (:index dev)))
      ;; sm-count from user spec
      (is (= 132 (get-in dev [:capabilities :sm-count])))
      ;; warp-size from catalogue
      (is (= 32 (get-in dev [:capabilities :warp-size])))
      ;; Should be retrievable
      (is (some? (hw/device :cuda:0)))))

  (testing "Catalogue fills gaps"
    (hw/init!)
    (hw/register-target-device! :cuda:5
                                {:name "NVIDIA A100"})
    (let [dev (hw/device :cuda:5)]
      ;; All from catalogue
      (is (= 108 (get-in dev [:capabilities :sm-count])))
      (is (= [8 0] (get-in dev [:capabilities :compute-capability]))))))

;; ================================================================
;; Hardware queries
;; ================================================================

(deftest hardware-queries-test
  (testing "warp-size defaults to 32"
    (hw/init!)
    (hw/register-target-device! :cuda:5 {:name "NVIDIA RTX 4090"})
    (is (= 32 (hw/warp-size :cuda:5))))

  (testing "sm-count from catalogue"
    (hw/init!)
    (hw/register-target-device! :cuda:5 {:name "NVIDIA RTX 4090"})
    (is (= 128 (hw/sm-count :cuda:5))))

  (testing "compute capability"
    (hw/init!)
    (hw/register-target-device! :cuda:6 {:name "NVIDIA A100"})
    (is (= [8 0] (hw/compute-capability :cuda:6)))
    (is (= "sm80" (hw/compute-capability-str :cuda:6)))))

;; ================================================================
;; Launch config
;; ================================================================

(deftest launch-config-test
  ;; launch geometry now lives in core.hardware (descriptor-taking) — one implementation, no
  ;; runtime.hardware duplicate. Tested against a hand-built A100-shaped descriptor (device-free).
  (let [desc {:device-type :gpu :subgroup-size 32 :max-workgroup-size 1024
              :sm-count 108 :max-warps-per-sm 64 :max-blocks-per-sm 32}]
    (testing "nil for small n (below the min-elements launch floor)"
      (is (nil? (chw/launch-config desc 100))))
    (testing "config for large n"
      (let [cfg (chw/launch-config desc 100000)]
        (is (some? cfg))
        (is (pos? (:block-size cfg)))
        (is (zero? (rem (:block-size cfg) 32)) "block-size is a multiple of the warp/subgroup")
        (is (pos? (:grid-size cfg)))
        (is (= 0 (:shared-mem cfg)))))
    (testing "reduction config carries shared memory"
      (is (pos? (:shared-mem (chw/launch-config desc 100000 :reduction? true)))))))

;; ================================================================
;; Catalogue
;; ================================================================

(deftest catalogue-test
  (testing "GPU lookup by name"
    (is (some? (catalogue/find-gpu-spec "NVIDIA H100")))
    (is (some? (catalogue/find-gpu-spec "nvidia a100")))  ;; case-insensitive
    (is (nil? (catalogue/find-gpu-spec "AMD Instinct"))))

  (testing "CPU defaults for arch"
    (let [defaults (catalogue/cpu-defaults-for-arch "amd64")]
      (is (= 4 (:simd-width defaults)))
      (is (= 8 (:simd-width-float defaults))))))

;; ================================================================
;; Default device
;; ================================================================

(deftest default-device-test
  (testing "Default device is CPU"
    (hw/init!)
    (let [dev (hw/default-device)]
      (is (some? dev))
      (is (= :cpu (:type dev))))))

;; ================================================================
;; Multiple init calls
;; ================================================================

(deftest idempotent-init-test
  (testing "Multiple init! calls are safe"
    (hw/init!)
    (hw/init!)
    (hw/init!)
    (is (some? (hw/device :cpu:0)))))

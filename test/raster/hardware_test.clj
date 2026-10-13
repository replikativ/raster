(ns raster.hardware-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.runtime.hardware :as hw]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.cuda-driver-probe :as cuda-probe]
            [raster.gpu.runtime-backend :as runtime-backend]
            [raster.compiler.core.hardware :as chw]
            [raster.runtime.hardware-catalogue :as catalogue]))

;; Reset hardware state between tests
(use-fixtures :each hardware-fixture/isolated)

(deftest cuda-observed-limits-reach-the-shared-compiler-descriptor
  (let [attributes {:sm-count 40 :warp-size 32 :max-threads-per-block 1024
                    :max-threads-per-sm 1536 :max-blocks-per-sm 16
                    :registers-per-block 65536 :registers-per-sm 65536
                    :shared-memory-per-block 49152 :shared-memory-per-sm 102400
                    :global-memory-bytes 8222998528
                    :max-workgroup-dims [1024 1024 64]}
        device (with-redefs [cuda-probe/probe
                             (constantly {:status :ready :driver-api-version 13020
                                          :devices [{:ordinal 0 :name "NVIDIA A100"
                                                     :compute-capability [8 6]
                                                     :attributes attributes}]})]
                 (first (#'hw/detect-cuda-devices)))
        descriptor (with-redefs [hw/device (constantly device)]
                     (chw/descriptor-for :cuda:0))]
    (doseq [[key value] attributes]
      (is (= value (get-in device [:capabilities key])) (str key))
      (is (= :detected (get-in device [:source key])) (str key)))
    (is (= 48 (get-in device [:capabilities :max-warps-per-sm])))
    (is (= :derived (get-in device [:source :max-warps-per-sm])))
    (is (= {:sm-count 40 :max-warps-per-sm 48 :max-blocks-per-sm 16
            :registers-per-block 65536 :shared-memory-per-block 49152
            :subgroup-size 32 :max-workgroup-size 1024}
           (select-keys descriptor [:sm-count :max-warps-per-sm :max-blocks-per-sm
                                    :registers-per-block :shared-memory-per-block
                                    :subgroup-size :max-workgroup-size])))
    (is (= [1024 1024 64] (get-in descriptor [:execution :max-workgroup-dims])))
    (is (= :detected (get-in descriptor [:execution :provenance :scratchpad-bytes])))
    (is (= 8222998528 (:global-memory-bytes descriptor)))
    (is (= 1280 (chw/grid-size descriptor 1000000 32))
        "resident-block bound limits this direct launch helper")
    (is (= 80 (chw/grid-size descriptor 1000000 1024))
        "resident-warp bound limits this direct launch helper")))

(deftest cuda-discovery-retains-driver-facts-and-field-provenance
  (let [detected (with-redefs [cuda-probe/probe
                              (constantly {:status :ready :driver-api-version 13020
                                           :devices [{:ordinal 0 :name "NVIDIA A100"
                                                      :compute-capability [8 6]}
                                                     {:ordinal 1 :name "other GPU"
                                                      :compute-capability [9 0]}]})]
                   (#'hw/detect-cuda-devices))
        device (first detected)]
    (is (= [:cuda:0] (mapv :id detected))
        "discovery must not promise ordinals the session layer cannot select")
    (is (= :cuda (:type device)))
    (is (= 0 (:index device)))
    (is (= 13020 (:driver-api-version device)))
    (is (nil? (:driver-version device))
        "CUDA driver API compatibility is not the NVIDIA driver build")
    (is (= [8 6] (get-in device [:capabilities :compute-capability]))
        "driver observation wins over the A100 catalogue")
    (is (= :detected (get-in device [:source :compute-capability])))
    (is (= 108 (get-in device [:capabilities :sm-count])))
    (is (= :catalogued (get-in device [:source :sm-count])))
    (is (= :detected (get-in device [:source :driver-api-version])))))

(deftest cuda-unavailable-attributes-retain-diagnostics-and-labelled-fallbacks
  (doseq [[name expected-value expected-source]
          [["unknown fixture GPU" nil :unavailable] ["NVIDIA A100" 108 :catalogued]]]
    (let [diagnostics {:sm-count [{:reason :cuda-driver-attribute-unavailable
                                  :attribute 16 :error-code 1}]}
          device (with-redefs [cuda-probe/probe
                               (constantly {:status :ready :driver-api-version 13020
                                            :devices [{:ordinal 0 :name name
                                                       :compute-capability [8 6]
                                                       :attribute-errors diagnostics}]})]
                   (first (#'hw/detect-cuda-devices)))]
      (is (= expected-value (get-in device [:capabilities :sm-count])))
      (is (= expected-source (get-in device [:source :sm-count])))
      (is (= diagnostics (:probe-diagnostics device)))
      (is (nil? (get-in device [:capabilities :max-warps-per-sm]))))))

(deftest cuda-discovery-never-registers-unavailable-or-failed-probes
  (doseq [status [:unavailable :no-device :driver-error]]
    (with-redefs [cuda-probe/probe
                  (constantly {:status status
                               :devices [{:ordinal 0 :name "invalid observation"}]})]
      (is (empty? (#'hw/detect-cuda-devices)) (str status))))
  (with-redefs [cuda-probe/probe
                (fn [] (throw (ex-info "probe failure" {:reason :cuda-driver-error})))]
    (is (empty? (#'hw/detect-cuda-devices)))))

(deftest hardware-init-admits-observed-cuda-device-without-runtime-initialization
  (with-redefs-fn {#'hw/detect-level-zero-devices (constantly [])
                  #'hw/detect-opencl-devices (constantly [])
                  #'cuda-probe/probe
                  (constantly {:status :ready :driver-api-version 13020
                               :devices [{:ordinal 0 :name "fixture GPU"
                                          :compute-capability [8 6]}]})}
    (fn []
      (hw/init!)
      (let [device (hw/device :cuda:0)]
        (is (= "fixture GPU" (:name device)))
        (is (= [8 6] (get-in device [:capabilities :compute-capability])))
        (is (= 13020 (:driver-api-version device)))))))

(deftest observed-cuda-device-remains-compile-only-until-runtime-is-qualified
  (with-redefs-fn {#'hw/detect-level-zero-devices (constantly [])
                  #'hw/detect-opencl-devices (constantly [])
                  #'cuda-probe/probe
                  (constantly {:status :ready :driver-api-version 13020
                               :devices [{:ordinal 0 :name "fixture GPU"
                                          :compute-capability [8 6]}]})}
    (fn []
      (hw/init!)
      (is (= :cuda (:type (hw/device :cuda:0))))
      (let [error (try (runtime-backend/backend-type :cuda:0)
                       nil (catch clojure.lang.ExceptionInfo e e))]
        (is (some? error))
        (is (true? (:compile-only? (ex-data error))))))))

(deftest cuda-driver-api-version-participates-in-calibration-identity
  (let [signature (fn [api-version]
                    (with-redefs [cuda-probe/probe
                                  (constantly {:status :ready :driver-api-version api-version
                                               :devices [{:ordinal 0 :name "fixture GPU"
                                                          :compute-capability [8 6]}]})]
                      (let [device (first (#'hw/detect-cuda-devices))]
                        (with-redefs [hw/device (constantly device)]
                          (hw/device-signature :cuda:0)))))]
    (is (not= (signature 13010) (signature 13020)))
    (is (= (signature 13020) (signature 13020)))))

(deftest observed-resource-limits-participate-in-calibration-identity
  (let [signature (fn [attributes]
                    (with-redefs [cuda-probe/probe
                                  (constantly {:status :ready :driver-api-version 13020
                                               :devices [{:ordinal 0 :name "fixture GPU"
                                                          :compute-capability [8 6]
                                                          :attributes attributes}]})]
                      (let [device (first (#'hw/detect-cuda-devices))]
                        (with-redefs [hw/device (constantly device)]
                          (hw/device-signature :cuda:0)))))]
    (doseq [[key a b] [[:warp-size 32 64]
                      [:max-workgroup-dims [1024 1024 64] [512 512 64]]
                      [:max-threads-per-block 1024 512]
                      [:max-threads-per-sm 1536 2048]
                      [:max-blocks-per-sm 16 32]
                      [:registers-per-block 65536 32768]
                      [:registers-per-sm 65536 32768]
                      [:shared-memory-per-block 49152 32768]
                      [:shared-memory-per-sm 102400 65536]]]
      (is (not= (signature {key a}) (signature {key b})) (str key)))
    (is (= (signature {:sm-count 40 :warp-size 32})
           (signature (array-map :warp-size 32 :sm-count 40)))
        "map insertion order is not a new calibration identity")))

;; ================================================================
;; CPU detection
;; ================================================================

(deftest opencl-driver-version-participates-in-calibration-identity
  (let [detected (fn [driver]
                   (with-redefs [ocl/query-devices
                                 (constantly [{:name "fixture GPU" :vendor "fixture"
                                               :version "OpenCL 3.0" :driver-version driver}])]
                     (first (#'hw/detect-opencl-devices))))
        first-device (detected "driver-A")
        next-device (detected "driver-B")
        signature (fn [device]
                    (with-redefs [hw/device (constantly device)]
                      (hw/device-signature :ocl:0)))]
    (is (= "driver-A" (:driver-version first-device)))
    (is (= "driver-A" (get-in first-device [:capabilities :driver-version])))
    (is (= :detected (get-in first-device [:source :driver-version])))
    (is (not= (signature first-device) (signature next-device)))
    (is (= (signature first-device) (signature (detected "driver-A"))))))

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

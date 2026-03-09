(ns raster.gpu.ze-runtime-test
  "Level Zero runtime integration tests.

  These tests require an Intel GPU with Level Zero driver installed.
  Tests are skipped gracefully when no GPU is available."
  (:require [clojure.test :refer [deftest is testing]]))

;; ================================================================
;; GPU availability check
;; ================================================================

(defn- ze-available?
  "Check if Level Zero is available on this machine."
  []
  (try
    (require 'raster.gpu.ze-runtime)
    (let [query-fn (resolve 'raster.gpu.ze-runtime/query-devices)]
      (and query-fn (seq (query-fn))))
    (catch Exception _ false)))

(defmacro when-ze
  "Run test body only when Level Zero is available."
  [& body]
  `(if (ze-available?)
     (do ~@body)
     (println "  [SKIP] No Level Zero GPU available")))

;; ================================================================
;; Device query tests
;; ================================================================

(deftest query-devices-test
  (when-ze
   (testing "Query returns at least one device"
     (require 'raster.gpu.ze-runtime)
     (let [devices ((resolve 'raster.gpu.ze-runtime/query-devices))]
       (is (seq devices))
       (let [dev (first devices)]
         (is (string? (:name dev)))
         (is (pos? (:total-eus dev)))
         (is (pos? (:simd-width dev)))
         (is (pos? (:threads-per-eu dev)))
         (println "  Found device:" (:name dev)
                  "EUs:" (:total-eus dev)
                  "SIMD:" (:simd-width dev)))))))

;; ================================================================
;; Init / context / device
;; ================================================================

(deftest init-test
  (when-ze
   (testing "Init is idempotent"
     (require 'raster.gpu.ze-runtime)
     (let [init! (resolve 'raster.gpu.ze-runtime/init!)]
       (init!)
       (init!)  ;; second call should be safe
       (is (some? ((resolve 'raster.gpu.ze-runtime/context))))
       (is (some? ((resolve 'raster.gpu.ze-runtime/device))))))))

;; ================================================================
;; Memory allocation
;; ================================================================

(deftest alloc-shared-test
  (when-ze
   (testing "Shared memory allocation and free"
     (require 'raster.gpu.ze-runtime)
     (let [init! (resolve 'raster.gpu.ze-runtime/init!)
           alloc-shared (resolve 'raster.gpu.ze-runtime/alloc-shared)
           free! (resolve 'raster.gpu.ze-runtime/free!)]
       (init!)
       (let [seg (alloc-shared (* 1024 8))]  ;; 1024 doubles
         (is (some? seg))
         (is (= (* 1024 8) (.byteSize seg)))
         (free! seg))))))

;; ================================================================
;; Data transfer
;; ================================================================

(deftest copy-roundtrip-test
  (when-ze
   (testing "Copy double array to device and back"
     (require 'raster.gpu.ze-runtime)
     (let [init! (resolve 'raster.gpu.ze-runtime/init!)
           alloc-shared (resolve 'raster.gpu.ze-runtime/alloc-shared)
           copy-to (resolve 'raster.gpu.ze-runtime/copy-to-device!)
           copy-from (resolve 'raster.gpu.ze-runtime/copy-from-device!)
           free! (resolve 'raster.gpu.ze-runtime/free!)]
       (init!)
       (let [n 1000
             src (double-array (range n))
             dst (double-array n)
             seg (alloc-shared (* n 8))]
         (copy-to seg src)
         (copy-from dst seg)
          ;; Verify roundtrip
         (is (= (seq src) (seq dst)))
         (free! seg))))))

;; ================================================================
;; Hardware integration
;; ================================================================

(deftest hardware-ze-device-test
  (when-ze
   (testing "Hardware registry detects Level Zero device"
     (require '[raster.runtime.hardware :as hw])
     (let [reset! (resolve 'raster.runtime.hardware/reset-hardware!)
           init! (resolve 'raster.runtime.hardware/init!)
           device (resolve 'raster.runtime.hardware/device)
           devices (resolve 'raster.runtime.hardware/devices)
           device-type (resolve 'raster.runtime.hardware/device-type)]
       (reset!)
       (init!)
       (let [all-devs (devices)
             ze-devs (filter (fn [[k _]] (= :ze (device-type k)))
                             all-devs)]
         (is (seq ze-devs) "Should detect at least one Level Zero device")
         (when (seq ze-devs)
           (let [[dev-id dev-map] (first ze-devs)]
             (is (= :ze (:type dev-map)))
             (is (string? (:name dev-map)))
             (println "  HW registry:" dev-id (:name dev-map)))))))))

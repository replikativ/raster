(ns raster.gpu.runtime-backend-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gpu]
            [raster.gpu.runtime-backend :as backend]))

(deftest resident-backend-facts-are-shared
  (is (= :ze (gpu/backend-type :ze:0)))
  (is (= :ocl (gpu/backend-type :ocl:0)))
  (is (= :ze (gpu/backend-type :ze-contracts))
      "synthetic compiler target IDs retain the established prefix behavior")
  (is (= :ocl (gpu/backend-type :ocl-equation-first-source-test)))
  (is (= {:namespace 'raster.gpu.ze-runtime
          :kernel-body-c-dialect :opencl-portable
          :memory-space :shared
          :coherence :host-coherent
          :owned-slice? false}
         (backend/descriptor :ze:0)))
  (is (= {:namespace 'raster.gpu.ocl-runtime
          :kernel-body-c-dialect :opencl-portable
          :memory-space :device
          :coherence :explicit-transfer
          :owned-slice? true}
         (backend/descriptor :ocl:0)))
  (is (= :opencl-portable (gpu/kernel-body-c-dialect (atom {:device-id :ocl:0}))))
  (is (= :opencl-portable (gpu/kernel-body-c-dialect (atom {:device-id :ze:0})))))

(deftest compile-only-targets-do-not-promise-resident-execution
  (doseq [device-id [:cuda:0 :hip:0]]
    (testing (str device-id)
      (let [error (try (gpu/backend-type device-id)
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (some? error))
        (is (= device-id (:device-id (ex-data error))))
        (is (true? (:compile-only? (ex-data error)))))))
  (doseq [device-id [:unknown:0 nil]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (backend/backend-type device-id)))))

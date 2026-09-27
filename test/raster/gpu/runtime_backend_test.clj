(ns raster.gpu.runtime-backend-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gpu]
            [raster.gpu.runtime-backend :as backend]))

(deftest resident-backend-facts-are-shared
  (is (= :ze (gpu/backend-type :ze:0)))
  (is (= :ocl (gpu/backend-type :ocl:0)))
  (is (= {:namespace 'raster.gpu.ze-runtime
          :memory-space :shared
          :coherence :host-coherent
          :owned-slice? false}
         (backend/descriptor :ze:0)))
  (is (= {:namespace 'raster.gpu.ocl-runtime
          :memory-space :device
          :coherence :explicit-transfer
          :owned-slice? true}
         (backend/descriptor :ocl:0))))

(deftest compile-only-targets-do-not-promise-resident-execution
  (doseq [device-id [:cuda:0 :hip:0]]
    (testing (str device-id)
      (let [error (try (gpu/backend-type device-id)
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (some? error))
        (is (= device-id (:device-id (ex-data error))))
        (is (true? (:compile-only? (ex-data error)))))))
  (doseq [device-id [:ze-bogus :ocl-bogus nil]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (backend/backend-type device-id)))))
